#!/usr/bin/env python3
"""Convert a schema-4 native actor to a compact browser artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--repo-root', required=True, type=Path)
  parser.add_argument('--policy', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--stage', type=int, default=3, help='Stage used by the browser environment.')
  parser.add_argument('--demo-settings-from', type=Path, help='Existing policy.json or browser manifest whose environment settings stay active when replacing only the actor.')
  return parser.parse_args()


def write_arrays(path: Path, arrays: list[tuple[str, np.ndarray]]) -> list[dict[str, object]]:
  offset = 0
  blocks: list[bytes] = []
  manifest: list[dict[str, object]] = []
  for name, value in arrays:
    value32 = np.asarray(value, dtype='<f4', order='C')
    raw = value32.tobytes(order='C')
    manifest.append({'name': name, 'shape': list(value32.shape), 'dtype': 'float32-le', 'byte_offset': offset, 'byte_length': len(raw)})
    blocks.append(raw)
    offset += len(raw)
  path.write_bytes(b''.join(blocks))
  return manifest


def main() -> None:
  args = parse_args()
  repo_root = args.repo_root.resolve()
  policy_path = args.policy.resolve()
  output = args.output.resolve()
  metadata_path = policy_path.with_name('policy.json')
  if not policy_path.is_file() or not metadata_path.is_file():
    raise FileNotFoundError('Expected policy.msgpack and adjacent policy.json.')
  metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
  expected = {'schema_version': 4, 'environment_backend': 'native', 'observation_version': 2, 'observation_size': 129, 'action_size': 6, 'hidden_size': 256}
  for key, value in expected.items():
    if metadata.get(key) != value:
      raise ValueError(f'Unsupported policy: expected {key}={value!r}, got {metadata.get(key)!r}.')
  if metadata.get('environment_id') != 'multi_box':
    raise ValueError(
        f'Policy environment is {metadata.get("environment_id")!r}; this static demo only accepts multi_box artifacts.'
    )
  demo_environment = None
  if args.demo_settings_from:
    settings_path = args.demo_settings_from.resolve()
    if not settings_path.is_file():
      raise FileNotFoundError(f'Missing demo settings source: {settings_path}')
    settings_data = json.loads(settings_path.read_text(encoding='utf-8'))
    settings_metadata = settings_data.get('policy_metadata', settings_data)
    if settings_metadata.get('stage_id') != args.stage or settings_metadata.get('environment_id') != 'multi_box':
      raise ValueError(f'Demo settings must be a stage-{args.stage} multi_box policy or manifest.')
    demo_environment = {
        'stage_id': settings_metadata['stage_id'], 'stage_name': settings_metadata['stage_name'],
        'environment_id': settings_metadata['environment_id'], 'reward_id': settings_metadata['reward_id'],
        'resolved_stage': settings_metadata['resolved_stage'], 'environment': settings_metadata['environment'],
        'source': str(settings_path),
    }
  elif metadata.get('stage_id') != args.stage:
    raise ValueError(
        f'Policy is stage {metadata.get("stage_id")!r}; pass --demo-settings-from with an existing stage-{args.stage} '
        'policy or browser manifest to replace only its actor weights.'
    )
  sys.path.insert(0, str(repo_root))
  from rl.checkpoints import PolicyBundle  # pylint: disable=import-outside-toplevel
  from rl.native_checkpoints import load_native_policy  # pylint: disable=import-outside-toplevel
  from rl.ppo import ActorCritic, ObservationStats, deterministic_action  # pylint: disable=import-outside-toplevel

  network = ActorCritic(hidden_size=256)
  params_template = network.init(jax.random.PRNGKey(0), jnp.zeros((129,), dtype=jnp.float32))
  template = PolicyBundle(
      params=params_template,
      observation_stats=ObservationStats(jnp.zeros((129,), dtype=jnp.float32), jnp.ones((129,), dtype=jnp.float32), jnp.zeros((129,), dtype=jnp.float32)),
  )
  policy = load_native_policy(policy_path, template)
  params = policy.params['params']
  arrays: list[tuple[str, np.ndarray]] = [
      ('observation_mean', np.asarray(jax.device_get(policy.observation_stats.mean))),
      ('observation_variance', np.asarray(jax.device_get(policy.observation_stats.variance))),
      ('observation_count', np.asarray(jax.device_get(policy.observation_stats.count))),
  ]
  for layer in ('Dense_0', 'Dense_1', 'Dense_2'):
    arrays.append((f'{layer}.kernel', np.asarray(jax.device_get(params[layer]['kernel']))))
    arrays.append((f'{layer}.bias', np.asarray(jax.device_get(params[layer]['bias']))))
  arrays.append(('actor_log_std', np.asarray(jax.device_get(params['log_std']))))
  output.mkdir(parents=True, exist_ok=True)
  array_manifest = write_arrays(output / 'weights.bin', arrays)
  fixture_observations = np.stack([
      np.zeros((129,), dtype=np.float32),
      np.linspace(-0.75, 0.75, 129, dtype=np.float32),
      np.where(np.arange(129) % 3 == 0, 0.25, -0.5).astype(np.float32),
  ])
  fixture_actions = [
      np.asarray(jax.device_get(deterministic_action(network, policy.params, policy.observation_stats, observation)), dtype=np.float32).tolist()
      for observation in fixture_observations
  ]
  (output / 'action-fixture.json').write_text(json.dumps({'format_version': 1, 'observations': fixture_observations.tolist(), 'expected_actions': fixture_actions}, indent=2) + '\n', encoding='utf-8')
  policy_id = output.name
  manifest = {
      'format_version': 1, 'policy_id': policy_id, 'weights_file': 'weights.bin', 'arrays': array_manifest,
      'actor': {'layers': [{'kernel': 'Dense_0.kernel', 'bias': 'Dense_0.bias', 'activation': 'tanh'}, {'kernel': 'Dense_1.kernel', 'bias': 'Dense_1.bias', 'activation': 'tanh'}, {'kernel': 'Dense_2.kernel', 'bias': 'Dense_2.bias', 'activation': 'tanh'}], 'output_activation': 'tanh'},
      'normalization': {'epsilon': 1e-6, 'clip': [-10, 10], 'validity': 'version-2 box validity mask'},
      'policy_metadata': metadata,
      'source_policy': str(policy_path.relative_to(repo_root)),
      'training_state_provenance': 'run-level policy.msgpack (authoritative); no Orbax training-state checkpoint is deployed',
  }
  if demo_environment:
    manifest['demo_environment'] = demo_environment
  (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
  active = output.parent / 'active.json'
  active.write_text(json.dumps({'format_version': 1, 'policy_id': policy_id, 'manifest': f'{policy_id}/manifest.json'}, indent=2) + '\n', encoding='utf-8')
  print(f'Exported {policy_id} from {policy_path}')


if __name__ == '__main__':
  main()
