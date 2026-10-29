from mjlab.tasks.registry import register_mjlab_task

from .climb_env_cfg import STAGES, MicrobanClimbRlCfg, make_climb_env_cfg

for _stage in STAGES:
  register_mjlab_task(
    task_id=f"Mjlab-Climb-Microban-Stage{_stage}",
    env_cfg=make_climb_env_cfg(stage=_stage),
    play_env_cfg=make_climb_env_cfg(stage=_stage, play=True),
    rl_cfg=MicrobanClimbRlCfg,
  )
