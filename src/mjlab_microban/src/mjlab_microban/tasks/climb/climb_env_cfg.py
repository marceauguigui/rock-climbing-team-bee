"""Climbing task for mjlab: stance transitions on the hackathon wall, Microban with BAM servos."""

from bam.mjlab import bam_init
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

STAGES = {
  1: {"levels": (1,), "single_transition": True, "fixed_limb": "right_hand",
      "fixed_target": True},
  2: {"levels": (1,), "single_transition": True},
  3: {"levels": (1, 2, 3)},
}

from . import mdp
from .scene_cfg import entities
from .stance_command import StanceCommandCfg

ACTUATED = (r".*(?<!head)$",)
CMD = {"command_name": "stance"}


def make_climb_env_cfg(stage: int = 1, play: bool = False) -> ManagerBasedRlEnvCfg:
  robot = SceneEntityCfg("robot", joint_names=ACTUATED)

  actor = {
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": robot},
      noise=Unoise(n_min=-0.002, n_max=0.002),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": robot},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      delay_min_lag=1,
      delay_max_lag=1,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      delay_min_lag=4,
      delay_max_lag=4,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.03, n_max=0.03),
      delay_min_lag=3,
      delay_max_lag=3,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "stance": ObservationTermCfg(func=mdp.stance_target, params=CMD),
  }
  critic = {
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel, params={"asset_cfg": robot}),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, params={"asset_cfg": robot}),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "stance": ObservationTermCfg(func=mdp.stance_target, params=CMD),
    "privileged": ObservationTermCfg(func=mdp.climb_privileged, params=CMD),
  }

  rewards = {
    "progress": RewardTermCfg(func=mdp.progress, weight=50.0, params=CMD),
    "trunk_progress": RewardTermCfg(func=mdp.trunk_progress, weight=20.0, params=CMD),
    "hooked": RewardTermCfg(func=mdp.hooked_target, weight=2.0, params=CMD),
    "back_to_start": RewardTermCfg(func=mdp.back_to_start, weight=-0.05, params=CMD),
    "success": RewardTermCfg(func=mdp.success, weight=20.0, params=CMD),
    "failure": RewardTermCfg(func=mdp.failure, weight=-5.0, params=CMD),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.0005),
    "joint_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits, weight=-0.1, params={"asset_cfg": robot}
    ),
  }

  events = {
    "bam_init": EventTermCfg(mode="startup", func=bam_init),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
    "trunk_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
        "operation": "add",
        "ranges": {0: (-0.005, 0.005), 1: (-0.005, 0.005), 2: (-0.005, 0.005)},
      },
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=None, entities=entities(), num_envs=1, extent=2.0),
    observations={
      "actor": ObservationGroupCfg(terms=actor, concatenate_terms=True,
                                   enable_corruption=not play),
      "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True,
                                    enable_corruption=False),
    },
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot", actuator_names=ACTUATED, scale=0.5, use_default_offset=True
      )
    },
    commands={"stance": StanceCommandCfg(**STAGES[stage])},
    events=events,
    rewards=rewards,
    terminations={
      "climb": TerminationTermCfg(func=mdp.climb_done, params=CMD),
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="trunk",
      distance=1.2,
      elevation=-10.0,
      azimuth=-90.0,
    ),
    sim=SimulationCfg(
      nconmax=512,
      njmax=2048,
      mujoco=MujocoCfg(
        timestep=0.002,
        iterations=5,
        ls_iterations=10,
        ccd_iterations=6,
        # multi-point mesh contacts need GBs on GPU; MuJoCo on CPU has them off too
        disableflags=("multiccd",),
      ),
    ),
    decimation=10,  # 20 ms control step, as on the robot
    episode_length_s=30.0,
    scale_rewards_by_dt=False,
  )
  return cfg


MicrobanClimbRlCfg = RslRlOnPolicyRunnerCfg(
  actor=RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0,
                      "std_type": "scalar"},
  ),
  critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu",
                       obs_normalization=True),
  algorithm=RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
  ),
  logger="tensorboard",
  experiment_name="microban_climb",
  save_interval=25,
  num_steps_per_env=24,
  max_iterations=10_000,
)
