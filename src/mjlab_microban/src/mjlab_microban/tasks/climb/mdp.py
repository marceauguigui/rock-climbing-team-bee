"""Observation, reward and termination terms reading the stance command's state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

  from .stance_command import StanceCommand


def _term(env: ManagerBasedRlEnv, command_name: str) -> StanceCommand:
  return env.command_manager.get_term(command_name)  # type: ignore[return-value]


def stance_target(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).stance_observation()


def climb_privileged(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).privileged_observation()


def climb_done(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Updates the task state once per step (runs before the rewards), returns failures."""
  return _term(env, command_name).step_update()


def progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).progress


def trunk_progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).trunk_progress


def hooked_target(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).hooked_target.float()


def back_to_start(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).back_to_start.float()


def success(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).success.float()


def failure(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).failed.float()
