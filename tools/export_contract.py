#!/usr/bin/env python3
"""Export the native stage contract as data consumed by browser code."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


def json_value(value):
  if isinstance(value, np.generic): return value.item()
  if isinstance(value, np.ndarray): return value.tolist()
  if isinstance(value, slice): return [value.start, value.stop]
  if is_dataclass(value): return {key: json_value(item) for key, item in asdict(value).items()}
  if isinstance(value, dict): return {str(key): json_value(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)): return [json_value(item) for item in value]
  return value


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--repo-root', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--stage', required=True, type=int)
  args = parser.parse_args()
  sys.path.insert(0, str(args.repo_root.resolve()))
  from rl import conveyor_spec as spec  # pylint: disable=import-outside-toplevel
  from rl.native_curriculum import resolve_curriculum_stage  # pylint: disable=import-outside-toplevel
  from rl import native_curriculum_rewards as native_rewards  # pylint: disable=import-outside-toplevel

  stage = resolve_curriculum_stage(args.stage)
  reward_weights = {
      'engagement': native_rewards.ENGAGEMENT_WEIGHTS,
      'normal_transfer': native_rewards.NORMAL_TRANSFER_WEIGHTS,
      'slow_coordination': native_rewards.SLOW_COORDINATION_WEIGHTS,
      'fast_throughput': native_rewards.FAST_THROUGHPUT_WEIGHTS,
  }[stage.reward_id]
  constants = {
      name: getattr(spec, name) for name in (
          'ACTION_SIZE', 'ACTION_LOW', 'ACTION_HIGH', 'ACTION_DELTA_RADIANS', 'BOX_COUNT', 'MAX_ACTIVE_BOXES',
          'TOP_BOX_COUNT', 'BOX_FEATURE_SIZE', 'OBSERVATION_SIZE', 'OBSERVATION_VERSION', 'BELT_TOP_Z', 'INLET_X',
          'EXIT_X', 'LATERAL_LIMIT', 'LATERAL_VELOCITY_SETTLE_THRESHOLD', 'ANGULAR_VELOCITY_SETTLE_THRESHOLD',
          'PARK_X', 'PARK_Y0', 'PARK_SPACING', 'PARK_Z', 'SAFETY_Z', 'FALLEN_Z', 'INLET_CLEARANCE', 'MAX_BOX_AGE',
          'MAX_LINEAR_SPEED', 'MAX_ANGULAR_SPEED', 'PANDA_TOOL_REFERENCE_X', 'GREEN_CLASS', 'BLUE_CLASS',
          'GREEN_RGBA', 'BLUE_RGBA', 'BLUE_BIN_X_CENTER', 'BLUE_BIN_X_MIN', 'BLUE_BIN_X_MAX', 'GREEN_BIN_X_CENTER',
          'GREEN_BIN_X_MIN', 'GREEN_BIN_X_MAX', 'GREEN_Y_CENTER', 'GREEN_Y_MIN', 'GREEN_Y_MAX', 'BLUE_Y_CENTER',
          'BLUE_Y_MIN', 'BLUE_Y_MAX', 'BIN_X_MAX', 'INTERCEPTION_X_MIN', 'PUSH_CLEARANCE', 'PUSH_TARGET_Z',
          'SAFE_HEIGHT_TOLERANCE', 'SAFE_UPRIGHT_Z', 'SAFE_LINEAR_SPEED', 'SAFE_ANGULAR_SPEED',
          'HAND_WRIST_HORIZONTAL_TOLERANCE', 'HAND_WRIST_VERTICAL_TOLERANCE', 'HAND_WRIST_ALIGNMENT_SCALE',
          'HOVER_HEIGHT_TOLERANCE', 'HOVER_HEIGHT_SCALE', 'DEFAULT_JOINT_POSE_NOISE', 'PANDA_HOME_QPOS', 'PANDA_HOME_CTRL',
      )
  }
  data = {
      'format_version': 1, 'constants': json_value(constants), 'observation_layout': {
          name: json_value(getattr(spec.ObservationLayout, name)) for name in ('JOINT_POSITION', 'JOINT_VELOCITY', 'TARGET_ERROR', 'PREVIOUS_ACTION', 'TOOL_POSITION', 'TOOL_ROTATION_6D', 'BOXES', 'GLOBAL')
      },
      'box_feature_offsets': json_value(spec.BOX_FEATURE_OFFSETS),
      'observation_normalize_mask': json_value(spec.OBSERVATION_NORMALIZE_MASK),
      'stage': json_value(stage), 'stage_reward_weights': json_value(reward_weights),
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
  main()
