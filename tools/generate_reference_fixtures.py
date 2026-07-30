#!/usr/bin/env python3
"""Capture a deterministic Python-native fixture for browser parity tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np


def values(array) -> list:
  return np.asarray(array).tolist()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--repo-root', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--policy', required=True, type=Path)
  parser.add_argument('--demo-settings-from', type=Path, help='Stage-3 policy.json or browser manifest that supplies the environment settings.')
  args = parser.parse_args()
  repo_root = args.repo_root.resolve()
  sys.path.insert(0, str(repo_root))
  from run_native_policy import resolve_evaluation_stage  # pylint: disable=import-outside-toplevel
  from rl.checkpoints import PolicyBundle  # pylint: disable=import-outside-toplevel
  from rl.conveyor_spec import EnvConfig, OBSERVATION_SIZE  # pylint: disable=import-outside-toplevel
  from rl.native_checkpoints import load_native_policy  # pylint: disable=import-outside-toplevel
  from rl.native_environment_registry import create_environment  # pylint: disable=import-outside-toplevel
  from rl.ppo import ActorCritic, ObservationStats, deterministic_action  # pylint: disable=import-outside-toplevel

  policy_path = args.policy.resolve()
  metadata = json.loads(policy_path.with_name('policy.json').read_text(encoding='utf-8'))
  settings_metadata = metadata
  if args.demo_settings_from:
    settings_path = args.demo_settings_from.resolve()
    if not settings_path.is_file():
      raise FileNotFoundError(f'Missing demo settings source: {settings_path}')
    settings_metadata = json.loads(settings_path.read_text(encoding='utf-8')).get('policy_metadata', {})
  if settings_metadata.get('stage_id') != 3 or settings_metadata.get('environment_id') != 'multi_box':
    raise ValueError('Reference generation requires stage-3 multi_box demo settings.')
  saved = settings_metadata['environment']
  resolved = settings_metadata['resolved_stage']
  stage = resolve_evaluation_stage(settings_metadata, stage_override=3, belt_speed_override=float(resolved['belt_speed']), spawn_interval_override=float(resolved['spawn_interval']))
  config = EnvConfig(
      model_path=saved['model_path'], num_envs=1, model_seed=0, randomize_joint_poses=False,
      joint_pose_noise=tuple(saved['joint_pose_noise']), joint_limit_margin=float(saved['joint_limit_margin']),
      max_joint_pose_reset_attempts=int(saved['max_joint_pose_reset_attempts']), box_surface_friction=float(saved['box_surface_friction']),
      belt_speed=float(resolved['belt_speed']), belt_speed_noise=0.0, spawn_interval=float(resolved['spawn_interval']), spawn_interval_noise=0.0,
      action_repeat=int(saved['action_repeat']), max_failures=int(saved['max_failures']), max_episode_seconds=float(saved['max_episode_seconds']),
  )
  environment = create_environment(config, stage, env_index=0)
  network = ActorCritic(hidden_size=256)
  params_template = network.init(jax.random.PRNGKey(0), jnp.zeros((OBSERVATION_SIZE,), dtype=jnp.float32))
  policy = load_native_policy(policy_path, PolicyBundle(params=params_template, observation_stats=ObservationStats(jnp.zeros((129,), dtype=jnp.float32), jnp.ones((129,), dtype=jnp.float32), jnp.zeros((129,), dtype=jnp.float32))))
  observation = environment.reset()
  shared = environment.shared
  initial = {
      'qpos': values(environment.data.qpos), 'qvel': values(environment.data.qvel), 'ctrl': values(environment.data.ctrl),
      'observation': values(observation), 'active': values(environment.active), 'target_bins': values(shared.target_bins),
      'control_targets': values(environment.control_targets), 'previous_action': values(environment.previous_action),
  }
  transitions = []
  reset_count = 0
  for _ in range(120):
    raw_observation = observation.copy()
    action = np.asarray(jax.device_get(deterministic_action(network, policy.params, policy.observation_stats, raw_observation)), dtype=np.float32)
    result = environment.step(action)
    info = result.info
    transitions.append({
        'raw_observation': values(raw_observation), 'action': values(action), 'target_controls': values(environment.control_targets),
        'qpos': values(environment.data.qpos), 'qvel': values(environment.data.qvel), 'observation': values(result.observation),
        'reward': float(result.reward), 'physics_steps': int(info.physics_steps),
        'info': {'correct_count': int(info.correct_count), 'wrong_count': int(info.wrong_count), 'missed_count': int(info.missed_count), 'first_contact_count': int(info.first_contact_count), 'returned_episode': bool(info.returned_episode)},
    })
    observation = result.observation
    reset_count += int(info.returned_episode)
    if len(transitions) >= 100 and reset_count:
      break
  document = {
      'format_version': 1, 'settings': {'platform': 'cpu', 'physics_kernel': 'python', 'seed': 0, 'belt_speed': float(resolved['belt_speed']), 'spawn_interval': float(resolved['spawn_interval']), 'noise': 0.0, 'randomize_joint_poses': False},
      'model': {'timestep': float(environment.model.opt.timestep), 'nq': int(environment.model.nq), 'nv': int(environment.model.nv), 'nu': int(environment.model.nu), 'joint_ranges': values(np.stack([shared.joint_low, shared.joint_high], axis=-1)), 'box_half_extents': values(shared.box_half_extents)},
      'initial': initial, 'transitions': transitions,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(document) + '\n', encoding='utf-8')
  print(f'Wrote {len(transitions)} Python reference transitions')


if __name__ == '__main__':
  main()
