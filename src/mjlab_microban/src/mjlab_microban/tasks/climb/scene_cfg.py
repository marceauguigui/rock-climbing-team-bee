"""Climbing robot, wall and holds as mjlab entities, built from the hackathon simulator's own models."""

import json
import os
from pathlib import Path

import microban_model
import mujoco
from bam.mjlab import BamActuatorCfg
from climbing_sim.scene import HOLD_PARTS, HOLD_SPHERES, HOLD_VISUAL
from climbing_sim.wall import WALL_HEIGHT, WALL_THICKNESS, WALL_WIDTH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

START_STATES = json.loads((Path(__file__).parent / "start_states.json").read_text())
N_SLOTS = 32
HOLD_FRICTION = 0.5

# Holds, wall and floor only collide with the robot, never with each other.
STATIC_CONTYPE, STATIC_CONAFFINITY = 2, 1
# The simulator keeps the robot's self-collisions; turning them off is much cheaper on GPU.
SELF_COLLISIONS = os.environ.get("CLIMB_SELF_COLLISIONS", "1") == "1"


def robot_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(microban_model.mjcf(climb=True)))
  if not SELF_COLLISIONS:
    for geom in spec.geoms:
      if geom.contype or geom.conaffinity:
        geom.contype, geom.conaffinity = 1, 2
  # qpos0 = the hanging pose, so the robot never starts inside the wall
  trunk = spec.body("trunk")
  trunk.pos = START_STATES["1"]["root_pos"]
  trunk.quat = START_STATES["1"]["root_quat"]
  return spec


def wall_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="board")
  body.add_geom(
    name="board",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[WALL_WIDTH / 2, WALL_THICKNESS / 2, WALL_HEIGHT / 2],
    pos=[WALL_WIDTH / 2, WALL_THICKNESS / 2, WALL_HEIGHT / 2],
    rgba=[0.86, 0.75, 0.56, 1],
    contype=STATIC_CONTYPE,
    conaffinity=STATIC_CONAFFINITY,
  )
  body.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[2.0, 2.0, 0.05],
    pos=[WALL_WIDTH / 2, 0.0, -0.05],
    rgba=[0.5, 0.5, 0.5, 1],
    contype=STATIC_CONTYPE,
    conaffinity=STATIC_CONAFFINITY,
  )
  return spec


def hold_spec() -> mujoco.MjSpec:
  """One hold, its origin at the socket centre (as in the simulator)."""
  spec = mujoco.MjSpec()
  spec.add_mesh(name="hold_visual", file=str(HOLD_VISUAL))
  for f in HOLD_PARTS:
    spec.add_mesh(name="hold_" + os.path.basename(f)[:-4], file=f)
  body = spec.worldbody.add_body(name="hold")
  body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="hold_visual",
    contype=0,
    conaffinity=0,
    group=1,
    rgba=[0.85, 0.35, 0.15, 1],
  )
  collision = {
    "group": 3,
    "friction": [HOLD_FRICTION, 0.005, 0.0001],
    "condim": 3,
    "contype": STATIC_CONTYPE,
    "conaffinity": STATIC_CONAFFINITY,
  }
  for f in HOLD_PARTS:
    name = "hold_" + os.path.basename(f)[:-4]
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, **collision)
  for sp in HOLD_SPHERES:
    body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[sp["r"], 0, 0], pos=sp["pos"], **collision
    )
  return spec


HANG = START_STATES["1"]

ACTUATORS = BamActuatorCfg(
  motor_name="xl330",
  model="m6",
  target_names_expr=(r"head", r".*_(shoulder|elbow|hip|knee|ankle).*"),
  kp_fw=125,
  vin_range=(7.0, 8.0),
  vin_drop_resistance_range=(0.0, 0.2),
  vin_min=6.0,
  delay_min_lag=4,  # physics steps of 2 ms: 8-12 ms, the robot's 10 ms latency
  delay_max_lag=6,
)

ROBOT_CFG = EntityCfg(
  spec_fn=robot_spec,
  init_state=EntityCfg.InitialStateCfg(
    pos=tuple(HANG["root_pos"]),
    rot=tuple(HANG["root_quat"]),
    joint_pos=dict(HANG["joints"]),
    joint_vel={r".*": 0.0},
  ),
  articulation=EntityArticulationInfoCfg(
    actuators=(ACTUATORS,), soft_joint_pos_limit_factor=0.95
  ),
)

PARKED = (5.0, 5.0, -5.0)  # unused holds, far from everything

WALL_CFG = EntityCfg(spec_fn=wall_spec)
HOLD_CFG = EntityCfg(spec_fn=hold_spec, init_state=EntityCfg.InitialStateCfg(pos=PARKED))


def entities() -> dict[str, EntityCfg]:
  return {
    "robot": ROBOT_CFG,
    "wall": WALL_CFG,
    **{f"hold_{i}": HOLD_CFG for i in range(N_SLOTS)},
  }
