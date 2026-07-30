#!/usr/bin/env python3
"""Prepare the native Panda scene and emit browser-readable parity metadata."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--repo-root', required=True, type=Path)
  parser.add_argument('--source', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--model-seed', required=True, type=int)
  parser.add_argument('--box-surface-friction', required=True, type=float)
  return parser.parse_args()


def named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
  result = mujoco.mj_name2id(model, object_type, name)
  if result < 0:
    raise ValueError(f'Missing required MuJoCo object: {name}')
  return int(result)


def as_list(value: object) -> list[object]:
  return np.asarray(value).tolist()


def extend_sorting_bins(specification: mujoco.MjSpec) -> list[dict[str, float | int]]:
  """Extend open sorting bins to the conveyor's far-end clearance.

  Their inner walls remain flush with the conveyor edges and retain clearance
  from the Panda base.  The bins are visual/recycling regions, so their
  collision masks intentionally stay disabled.
  """
  configurations = {
      'left_collection_bin': (0, 1.085, 0.404, 0.575),
      'right_collection_bin': (1, 0.835, -0.404, 0.825),
  }
  bodies = {body.name: body for body in specification.bodies}
  geoms = {geom.name: geom for geom in specification.geoms}
  regions: list[dict[str, float | int]] = []
  for body_name, (target_bin, x, y, half_length) in configurations.items():
    body = bodies.get(body_name)
    if body is None:
      raise ValueError(f'Missing required sorting-bin body: {body_name}')
    body.pos = [x, y, 0]
    prefix = 'left' if body_name.startswith('left') else 'right'
    for suffix in ('bin_base', 'bin_outer_wall', 'bin_inner_wall'):
      geom = geoms.get(f'{prefix}_{suffix}')
      if geom is None:
        raise ValueError(f'Missing required sorting-bin geometry: {prefix}_{suffix}')
      geom.size[0] = half_length
    far_wall = geoms.get(f'{prefix}_bin_far_wall')
    if far_wall is None:
      raise ValueError(f'Missing required sorting-bin geometry: {prefix}_bin_far_wall')
    far_wall.pos[0] = half_length
    base = geoms[f'{prefix}_bin_base']
    half_width = float(base.size[1])
    # These are the bin-base extents in the MuJoCo world frame.  They are
    # deliberately exported separately from the trained policy's historic
    # goal constants so display geometry and sort classification cannot drift.
    regions.append({
        'target_bin': target_bin,
        'x_min': x - half_length,
        'x_max': x + half_length,
        'y_min': y - half_width,
        'y_max': y + half_width,
    })
  return regions


def main() -> None:
  args = parse_args()
  repo_root = args.repo_root.resolve()
  source = args.source.resolve()
  output = args.output.resolve()
  if not source.is_file() or not (repo_root / 'rl' / 'conveyor_model.py').is_file():
    raise FileNotFoundError('Expected a robotics repository and MJCF source.')
  sys.path.insert(0, str(repo_root))
  from rl.conveyor_model import (  # pylint: disable=import-outside-toplevel
      NATIVE_BOX_COLLISION_BIT,
      NATIVE_LINESEARCH_ITERATIONS,
      NATIVE_SOLVER_ITERATIONS,
      NATIVE_WORKSPACE_COLLISION_BIT,
      ROBOT_COLLISION_GROUP,
      apply_box_randomization,
      discover_box_addresses,
      set_box_surface_friction,
  )
  from rl.conveyor_spec import (  # pylint: disable=import-outside-toplevel
      ACTION_SIZE,
      BOX_COUNT,
      BOX_NAMES,
      GREEN_RGBA,
      BLUE_RGBA,
      PANDA_HOME_CTRL,
      PANDA_HOME_QPOS,
    )

  specification = mujoco.MjSpec.from_file(str(source))
  robot_geoms = [geom for geom in specification.geoms if int(geom.group) == ROBOT_COLLISION_GROUP]
  table_geom = next((geom for geom in specification.geoms if geom.name == 'table_block'), None)
  surface_geom = next((geom for geom in specification.geoms if geom.name == 'conveyor_surface'), None)
  box_geoms = [geom for geom in specification.geoms if geom.name in {f'{name}_geom' for name in BOX_NAMES}]
  hand_geom = next((geom for geom in specification.geoms if geom.name == 'hand_capsule'), None)
  push_site = next((site for site in specification.sites if site.name == 'push_point'), None)
  if not robot_geoms or table_geom is None or surface_geom is None or len(box_geoms) != BOX_COUNT or hand_geom is None or push_site is None:
    raise ValueError('Source scene does not satisfy the native model contract.')

  specification.option.iterations = NATIVE_SOLVER_ITERATIONS
  specification.option.ls_iterations = NATIVE_LINESEARCH_ITERATIONS
  sorting_bin_regions = extend_sorting_bins(specification)
  hand_geom.pos = push_site.pos
  for geom in robot_geoms:
    geom.contype = int(geom.contype) | NATIVE_WORKSPACE_COLLISION_BIT
  for geom in (table_geom, surface_geom):
    geom.conaffinity = int(geom.conaffinity) | NATIVE_WORKSPACE_COLLISION_BIT
  for geom in box_geoms:
    geom.contype = int(geom.contype) | NATIVE_BOX_COLLISION_BIT
    geom.conaffinity = int(geom.conaffinity) | NATIVE_BOX_COLLISION_BIT

  output.mkdir(parents=True, exist_ok=True)
  xml_path = output / 'native_scene.xml'
  xml_path.write_text(specification.to_xml(), encoding='utf-8')
  assets_source = source.parent / 'assets'
  shutil.copytree(assets_source, output / 'assets', dirs_exist_ok=True)
  for license_name in ('LICENSE', 'README.md'):
    candidate = source.parent / license_name
    if candidate.is_file():
      shutil.copy2(candidate, output / license_name)
  (output / 'LICENSES.md').write_text(
      'Prepared browser assets are derived from the bundled Franka Emika Panda MuJoCo Menagerie scene.\n'
      'See LICENSE in this directory for the source license text.\n',
      encoding='utf-8',
  )

  model = specification.compile()
  scratch = mujoco.MjData(model)
  colors = np.where((np.arange(BOX_COUNT) % 2)[:, None] == 0, GREEN_RGBA, BLUE_RGBA)
  apply_box_randomization(model, args.model_seed, colors)
  set_box_surface_friction(model, args.box_surface_friction)
  mujoco.mj_setConst(model, scratch)
  boxes = discover_box_addresses(model)

  required = [*(f'joint{index}' for index in range(1, 8)), 'finger_joint1', 'finger_joint2', *(f'actuator{index}' for index in range(1, 9)), 'push_point', 'hand_capsule', 'conveyor_surface', 'table_block']
  required.extend(BOX_NAMES)
  required.extend(f'{name}_geom' for name in BOX_NAMES)
  required.extend(f'{name}_freejoint' for name in BOX_NAMES)
  for name in required:
    object_type = mujoco.mjtObj.mjOBJ_BODY if name in BOX_NAMES else (
        mujoco.mjtObj.mjOBJ_GEOM if name.endswith('_geom') or name in {'hand_capsule', 'conveyor_surface', 'table_block'} else (
          mujoco.mjtObj.mjOBJ_SITE if name == 'push_point' else (
            mujoco.mjtObj.mjOBJ_ACTUATOR if name.startswith('actuator') else mujoco.mjtObj.mjOBJ_JOINT
          )
        )
      )
    named_id(model, object_type, name)

  controlled_joint_ids = np.asarray([named_id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{index}') for index in range(1, ACTION_SIZE + 1)])
  locked_joint_ids = np.asarray([named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ('joint7', 'finger_joint1', 'finger_joint2')])
  box_patch = []
  body_patch = []
  for index, name in enumerate(BOX_NAMES):
    geom_id = int(boxes.geom_ids[index])
    body_id = int(boxes.body_ids[index])
    box_patch.append({
        'id': geom_id, 'name': f'{name}_geom', 'size': as_list(model.geom_size[geom_id]),
        'friction': as_list(model.geom_friction[geom_id]), 'rgba': as_list(model.geom_rgba[geom_id]),
    })
    body_patch.append({'id': body_id, 'name': name, 'mass': float(model.body_mass[body_id]), 'inertia': as_list(model.body_inertia[body_id])})
  surface_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'conveyor_surface')
  pairs = []
  box_geom_ids = {int(value) for value in boxes.geom_ids}
  for pair_id in range(model.npair):
    if int(model.pair_geom1[pair_id]) == surface_id or int(model.pair_geom2[pair_id]) == surface_id:
      other = int(model.pair_geom2[pair_id]) if int(model.pair_geom1[pair_id]) == surface_id else int(model.pair_geom1[pair_id])
      if other in box_geom_ids:
        pairs.append({'id': pair_id, 'friction': as_list(model.pair_friction[pair_id])})
  patch = {
      'format_version': 1, 'requires_mj_setConst': True,
      'geoms': box_patch + [{'id': surface_id, 'name': 'conveyor_surface', 'friction': as_list(model.geom_friction[surface_id])}],
      'bodies': body_patch, 'pairs': pairs,
  }
  (output / 'model_patch.json').write_text(json.dumps(patch, indent=2) + '\n', encoding='utf-8')
  manifest = {
      'format_version': 1, 'mujoco_version': mujoco.__version__, 'model_file': 'native_scene.xml',
      'nq': int(model.nq), 'nv': int(model.nv), 'nu': int(model.nu), 'timestep': float(model.opt.timestep),
      'controlled_joint_qpos_addresses': as_list(model.jnt_qposadr[controlled_joint_ids]),
      'controlled_joint_qvel_addresses': as_list(model.jnt_dofadr[controlled_joint_ids]),
      'locked_joint_qpos_addresses': as_list(model.jnt_qposadr[locked_joint_ids]),
      'box_body_ids': as_list(boxes.body_ids), 'box_geom_ids': as_list(boxes.geom_ids),
      'box_qpos_addresses': as_list(boxes.qpos_addresses), 'box_qvel_addresses': as_list(boxes.qvel_addresses),
      'push_site_id': named_id(model, mujoco.mjtObj.mjOBJ_SITE, 'push_point'),
      'hand_geom_id': named_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'hand_capsule'),
      'conveyor_surface_geom_id': surface_id,
      'wrist_body_id': int(model.jnt_bodyid[named_id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint7')]),
      'joint_low': as_list(model.jnt_range[controlled_joint_ids, 0]), 'joint_high': as_list(model.jnt_range[controlled_joint_ids, 1]),
      'box_half_extents': as_list(model.geom_size[boxes.geom_ids]), 'target_bins': list(range(BOX_COUNT)),
      'sorting_bin_regions': sorting_bin_regions,
      'panda_home_qpos': as_list(PANDA_HOME_QPOS), 'panda_home_ctrl': as_list(PANDA_HOME_CTRL),
      'native_collision': {'robot_group': ROBOT_COLLISION_GROUP, 'workspace_bit': NATIVE_WORKSPACE_COLLISION_BIT, 'box_bit': NATIVE_BOX_COLLISION_BIT},
      'asset_files': [str(path.relative_to(output)) for path in sorted((output / 'assets').rglob('*')) if path.is_file()],
  }
  manifest['target_bins'] = [index % 2 for index in range(BOX_COUNT)]
  (output / 'model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
  # Python MuJoCo owns these wrappers through normal reference counting.
  del scratch
  del model
  print(f'Prepared {xml_path}')


if __name__ == '__main__':
  main()
