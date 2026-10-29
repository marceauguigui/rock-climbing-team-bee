"""Stance-transition task: one limb moves to a target hold, the three others stay hooked.

Same task as climb_env.py (ClimbingSim), vectorised on GPU. World frame as in the simulator:
x right along the wall, y into the wall, z up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from climbing_sim.routes import generate_wall
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .scene_cfg import N_SLOTS, PARKED, START_STATES

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
PARTNER = (1, 0, 3, 2)


@dataclass(kw_only=True)
class StanceCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  resampling_time_range: tuple[float, float] = (1e9, 1e9)  # only on reset

  levels: tuple[int, ...] = (1, 2, 3)
  walls_per_level: int = 64

  single_transition: bool = False
  fixed_limb: str | None = None
  fixed_target: bool = False

  reach: tuple[float, float] = (0.03, 0.12)
  max_lateral: float = 0.08
  hand_above_feet: float = 0.10
  no_crossing: float = 0.03

  approach_height: float = 0.03
  approach_radius: float = 0.015
  hook_tol: float = 0.006
  success_hold_steps: int = 10
  support_grace_steps: int = 5
  transition_timeout_s: float = 4.0

  def build(self, env: ManagerBasedRlEnv) -> StanceCommand:
    return StanceCommand(self, env)


class StanceCommand(CommandTerm):
  cfg: StanceCommandCfg

  def __init__(self, cfg: StanceCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.holds: list[Entity] = [env.scene[f"hold_{i}"] for i in range(N_SLOTS)]
    self.ball_ids, _ = self.robot.find_sites(
      [f"{l}_ball" for l in LIMBS], preserve_order=True
    )
    self.timeout_steps = int(round(cfg.transition_timeout_s / env.step_dt))
    self._build_wall_bank()
    self._build_start_states()

    n, dev = self.num_envs, self.device
    self.wall = torch.zeros(n, dtype=torch.long, device=dev)
    self.slot_pos = torch.zeros(n, N_SLOTS, 3, device=dev)
    self.slot_valid = torch.zeros(n, N_SLOTS, dtype=torch.bool, device=dev)
    self.stance = torch.zeros(n, 4, dtype=torch.long, device=dev)
    self.moving = torch.zeros(n, dtype=torch.long, device=dev)
    self.target = torch.zeros(n, dtype=torch.long, device=dev)
    self.start_slot = torch.zeros(n, dtype=torch.long, device=dev)
    self.approaching = torch.ones(n, dtype=torch.bool, device=dev)
    self.left_start = torch.zeros(n, dtype=torch.bool, device=dev)
    self.needs_init = torch.ones(n, dtype=torch.bool, device=dev)
    self.finished = torch.zeros(n, dtype=torch.bool, device=dev)
    self.transition_steps = torch.zeros(n, dtype=torch.long, device=dev)
    self.hooked_steps = torch.zeros(n, dtype=torch.long, device=dev)
    self.support_lost = torch.zeros(n, 4, dtype=torch.long, device=dev)
    self.successes = torch.zeros(n, dtype=torch.long, device=dev)
    self.prev_dist = torch.zeros(n, device=dev)
    self.prev_trunk_dist = torch.zeros(n, device=dev)

    self.hooked = torch.full((n, 4), -1, dtype=torch.long, device=dev)
    self.progress = torch.zeros(n, device=dev)
    self.trunk_progress = torch.zeros(n, device=dev)
    self.hooked_target = torch.zeros(n, dtype=torch.bool, device=dev)
    self.back_to_start = torch.zeros(n, dtype=torch.bool, device=dev)
    self.success = torch.zeros(n, dtype=torch.bool, device=dev)
    self.failed = torch.zeros(n, dtype=torch.bool, device=dev)

    self.metrics["success_rate"] = torch.zeros(n, device=dev)
    self.metrics["transitions"] = torch.zeros(n, device=dev)

  # ---- setup -------------------------------------------------------------------------
  def _build_wall_bank(self):
    walls = []
    for level in self.cfg.levels:
      seeds = [0] if level == 1 else range(self.cfg.walls_per_level)
      walls += [(level, generate_wall(level, s)) for s in seeds]
    h_max = max(len(w.holds) for _, w in walls)
    sockets = np.full((len(walls), h_max, 3), np.inf, dtype=np.float32)
    start = np.zeros((len(walls), 4), dtype=np.int64)
    for k, (_, w) in enumerate(walls):
      sockets[k, : len(w.holds)] = w.sockets()
      start[k] = [w.holds.index(w.start[l]) for l in LIMBS]
    self.bank_sockets = torch.tensor(sockets, device=self.device)
    self.bank_start = torch.tensor(start, device=self.device)
    self.bank_level = torch.tensor([lvl for lvl, _ in walls], device=self.device)

  def _build_start_states(self):
    names = self.robot.joint_names
    root, joints = [], []
    for level in (1, 2, 3):
      s = START_STATES[str(level)]
      root.append(s["root_pos"] + s["root_quat"] + [0.0] * 6)
      joints.append([s["joints"][j] for j in names])
    self.start_root = torch.tensor(root, device=self.device)
    self.start_joints = torch.tensor(joints, device=self.device)

  # ---- helpers -----------------------------------------------------------------------
  def _stance_pos(self, ids=slice(None)) -> torch.Tensor:
    pos = self.slot_pos[ids]
    return torch.gather(pos, 1, self.stance[ids, :, None].expand(-1, -1, 3))

  def _candidates(self, ids: torch.Tensor, limb: torch.Tensor) -> torch.Tensor:
    cfg = self.cfg
    ar = torch.arange(len(ids), device=self.device)
    stance = self._stance_pos(ids)
    pos = self.slot_pos[ids]
    cur = stance[ar, limb]
    d = pos - cur[:, None]
    dist = torch.linalg.norm(d[..., [0, 2]], dim=-1)
    ok = self.slot_valid[ids] & (dist >= cfg.reach[0]) & (dist <= cfg.reach[1])
    ok &= (d[..., 0].abs() <= cfg.max_lateral) & (d[..., 2] > -0.02)
    ok &= ~(torch.arange(N_SLOTS, device=self.device)[None, None] == self.stance[ids, :, None]).any(1)

    partner_x = stance[ar, torch.tensor(PARTNER, device=self.device)[limb], 0]
    is_left = (limb % 2) == 0
    ok &= torch.where(
      is_left[:, None],
      pos[..., 0] < partner_x[:, None] - cfg.no_crossing,
      pos[..., 0] > partner_x[:, None] + cfg.no_crossing,
    )
    feet_top = stance[:, 2:, 2].max(-1).values
    hands_low = stance[:, :2, 2].min(-1).values
    is_hand = limb < 2
    ok &= torch.where(
      is_hand[:, None],
      pos[..., 2] > feet_top[:, None] + cfg.hand_above_feet,
      pos[..., 2] < hands_low[:, None] - cfg.hand_above_feet,
    )
    return ok

  def _select_transition(self, ids: torch.Tensor):
    if len(ids) == 0:
      return
    cfg, n = self.cfg, len(ids)
    masks = torch.stack(
      [self._candidates(ids, torch.full((n,), l, device=self.device)) for l in range(4)], 1
    )
    has = masks.any(-1)
    if cfg.fixed_limb is not None:
      limb_score = torch.zeros(n, 4, device=self.device)
      limb_score[:, LIMBS.index(cfg.fixed_limb)] = 1.0
      limb_score = torch.where(has, limb_score, -1.0)
    else:
      limb_score = torch.where(has, torch.rand(n, 4, device=self.device), -1.0)
    limb = limb_score.argmax(-1)
    ar = torch.arange(n, device=self.device)
    mask = masks[ar, limb]
    if cfg.fixed_target:
      cur = self._stance_pos(ids)[ar, limb]
      d = self.slot_pos[ids] - cur[:, None]
      mid = sum(cfg.reach) / 2
      score = -(d[..., 0].abs() + (torch.linalg.norm(d[..., [0, 2]], dim=-1) - mid).abs())
    else:
      score = torch.rand(n, N_SLOTS, device=self.device)
    target = torch.where(mask, score, -torch.inf).argmax(-1)

    ok = has[ar, limb]
    self.finished[ids] = ~ok
    self.moving[ids] = limb
    self.target[ids] = target
    self.start_slot[ids] = self.stance[ids, limb]
    self.approaching[ids] = True
    self.left_start[ids] = False
    self.needs_init[ids] = True
    self.transition_steps[ids] = 0
    self.hooked_steps[ids] = 0
    self.support_lost[ids] = 0

  # ---- CommandTerm -------------------------------------------------------------------
  def _resample_command(self, env_ids: torch.Tensor):
    n = len(env_ids)
    levels = torch.tensor(self.cfg.levels, device=self.device)
    allowed = (self.bank_level[None] == levels[:, None]).any(0).nonzero().squeeze(-1)
    wall = allowed[torch.randint(len(allowed), (n,), device=self.device)]
    self.wall[env_ids] = wall

    sockets, start = self.bank_sockets[wall], self.bank_start[wall]
    # slots: the holds nearest a point just above the hands, so the way up is covered
    hands = torch.gather(sockets, 1, start[:, :2, None].expand(-1, -1, 3)).mean(1)
    centre = hands + torch.tensor([0.0, 0.0, 0.05], device=self.device)
    dist = torch.linalg.norm(sockets - centre[:, None], dim=-1)
    dist = torch.nan_to_num(dist, nan=torch.inf)
    dist.scatter_(1, start, -1.0)
    near_d, near = torch.topk(dist, N_SLOTS, dim=1, largest=False)
    valid = torch.isfinite(near_d)
    pos = torch.gather(sockets, 1, near[..., None].expand(-1, -1, 3))
    pos = torch.where(valid[..., None], pos, torch.tensor(PARKED, device=self.device))
    self.slot_pos[env_ids] = pos
    self.slot_valid[env_ids] = valid
    self.stance[env_ids] = (near[:, None, :] == start[..., None]).float().argmax(-1)

    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(n, 4)
    for i, hold in enumerate(self.holds):
      hold.write_mocap_pose_to_sim(torch.cat([pos[:, i], quat], -1), env_ids=env_ids)

    level = self.bank_level[wall] - 1
    self.robot.write_root_state_to_sim(self.start_root[level], env_ids=env_ids)
    self.robot.write_joint_state_to_sim(
      self.start_joints[level], torch.zeros_like(self.start_joints[level]), env_ids=env_ids
    )

    self.successes[env_ids] = 0
    self.success[env_ids] = False
    self.failed[env_ids] = False
    self._select_transition(env_ids)

  def _update_command(self):
    ids = self.success.nonzero().squeeze(-1)
    if len(ids):
      self.stance[ids, self.moving[ids]] = self.target[ids]
      self.successes[ids] += 1
      if not self.cfg.single_transition:
        self._select_transition(ids)
    self.success[:] = False

  def _update_metrics(self):
    self.metrics["success_rate"] = (self.successes > 0).float()
    self.metrics["transitions"] = self.successes.float()

  @property
  def command(self) -> torch.Tensor:
    return self.stance_observation()

  # ---- per-step state, called once per step by the termination term -----------------
  def step_update(self) -> torch.Tensor:
    cfg, ar = self.cfg, torch.arange(self.num_envs, device=self.device)
    balls = self.robot.data.site_pos_w[:, self.ball_ids]
    d = torch.cdist(balls, self.slot_pos)
    near_d, near = d.min(-1)
    self.hooked = torch.where(near_d < cfg.hook_tol, near, -1)

    target_pos = self.slot_pos[ar, self.target]
    up = torch.tensor([0.0, 0.0, cfg.approach_height], device=self.device)
    point = target_pos + up * self.approaching[:, None]
    ball = balls[ar, self.moving]
    dist = torch.linalg.norm(ball - point, dim=-1)
    self.progress = torch.where(self.needs_init, 0.0, self.prev_dist - dist)
    switch = self.approaching & (dist < cfg.approach_radius)
    self.approaching &= ~switch
    self.prev_dist = torch.where(switch, torch.linalg.norm(ball - target_pos, dim=-1), dist)

    trunk_dist = torch.linalg.norm(self.robot.data.root_link_pos_w - target_pos, dim=-1)
    is_hand = self.moving < 2
    self.trunk_progress = torch.where(
      self.needs_init | ~is_hand, 0.0, self.prev_trunk_dist - trunk_dist
    )
    self.prev_trunk_dist = trunk_dist
    self.needs_init[:] = False

    hooked_m = self.hooked[ar, self.moving]
    self.left_start |= hooked_m != self.start_slot
    self.back_to_start = self.left_start & (hooked_m == self.start_slot)

    on_hold = self.hooked == self.stance
    self.support_lost = torch.where(on_hold, 0, self.support_lost + 1)
    self.support_lost[ar, self.moving] = 0
    self.transition_steps += 1

    self.failed = (self.support_lost.max(-1).values > cfg.support_grace_steps) | (
      self.transition_steps > self.timeout_steps
    )
    self.hooked_target = hooked_m == self.target
    self.hooked_steps = torch.where(self.hooked_target, self.hooked_steps + 1, 0)
    self.success = (self.hooked_steps >= cfg.success_hold_steps) & ~self.failed

    done = self.failed | self.finished
    if cfg.single_transition:
      done |= self.success
    return done

  # ---- observations ------------------------------------------------------------------
  def _goal_pos(self) -> torch.Tensor:
    ar = torch.arange(self.num_envs, device=self.device)
    goal = self._stance_pos().clone()
    goal[ar, self.moving] = self.slot_pos[ar, self.target]
    return goal

  def _support_centre(self, goal: torch.Tensor) -> torch.Tensor:
    support = torch.ones(self.num_envs, 4, 1, device=self.device)
    support[torch.arange(self.num_envs, device=self.device), self.moving] = 0.0
    return (goal * support).sum(1) / 3.0

  @staticmethod
  def to_wall(p: torch.Tensor) -> torch.Tensor:
    return torch.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)

  def stance_observation(self) -> torch.Tensor:
    goal = self._goal_pos()
    rel = self.to_wall(goal - self._support_centre(goal)[:, None])
    one_hot = torch.nn.functional.one_hot(self.moving, 4).float()
    return torch.cat([rel.flatten(1), one_hot], -1)

  def privileged_observation(self) -> torch.Tensor:
    goal = self._goal_pos()
    centre = self._support_centre(goal)
    trunk = self.to_wall(self.robot.data.root_link_pos_w - centre)
    balls = self.robot.data.site_pos_w[:, self.ball_ids]
    on_goal = (self.hooked == torch.cat([self.stance], 1)).float()
    ar = torch.arange(self.num_envs, device=self.device)
    on_goal[ar, self.moving] = self.hooked_target.float()
    return torch.cat([trunk, self.to_wall(balls - goal).flatten(1), on_goal], -1)
