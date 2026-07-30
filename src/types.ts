import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';

export type Slice = [number, number];

export interface ModelManifest {
  format_version: number;
  mujoco_version: string;
  model_file: string;
  nq: number;
  nv: number;
  nu: number;
  timestep: number;
  controlled_joint_qpos_addresses: number[];
  controlled_joint_qvel_addresses: number[];
  locked_joint_qpos_addresses: number[];
  box_body_ids: number[];
  box_geom_ids: number[];
  box_qpos_addresses: number[];
  box_qvel_addresses: number[];
  push_site_id: number;
  hand_geom_id: number;
  conveyor_surface_geom_id: number;
  wrist_body_id: number;
  joint_low: number[];
  joint_high: number[];
  box_half_extents: number[][];
  target_bins: number[];
  /** Acceptance footprints derived from the prepared, enlarged sorting bins. */
  sorting_bin_regions: Array<{ target_bin: 0 | 1; x_min: number; x_max: number; y_min: number; y_max: number }>;
  panda_home_qpos: number[];
  panda_home_ctrl: number[];
  asset_files: string[];
}

export interface ModelPatch {
  requires_mj_setConst: boolean;
  geoms: Array<{ id: number; name: string; size?: number[]; friction: number[]; rgba?: number[] }>;
  bodies: Array<{ id: number; name: string; mass: number; inertia: number[] }>;
  pairs: Array<{ id: number; friction: number[] }>;
}

export interface Contract {
  constants: Record<string, number | number[]>;
  observation_layout: Record<string, Slice>;
  box_feature_offsets: Record<string, Slice>;
  observation_normalize_mask: boolean[];
  stage: { stage_id: number; name: string; environment_id: string; reward_id: string; belt_speed: number; spawn_interval: number; max_active_boxes: number };
  stage_reward_weights: Record<string, number>;
}

export interface PolicyArray { name: string; shape: number[]; dtype: string; byte_offset: number; byte_length: number }
export interface PolicyManifest {
  policy_id: string;
  weights_file: string;
  arrays: PolicyArray[];
  policy_metadata: {
    schema_version: number; observation_version: number; observation_size: number; action_size: number; hidden_size: number;
    environment_steps: number; stage_id: number; stage_name: string; environment_id: string; reward_id: string;
    resolved_stage: { belt_speed: number; spawn_interval: number; max_active_boxes: number };
    environment: Record<string, unknown>;
  };
  demo_environment?: {
    stage_id: number;
    stage_name: string;
    environment_id: string;
    reward_id: string;
    resolved_stage: { belt_speed: number; spawn_interval: number; max_active_boxes: number };
    environment: Record<string, unknown>;
    source: string;
  };
}

export interface ActivePolicy { policy_id: string; manifest: string }
export interface MujocoHandle { mujoco: MainModule; model: MjModel; data: MjData; dispose(): void }

export interface LoadedDemo extends MujocoHandle { modelManifest: ModelManifest; modelPatch: ModelPatch; contract: Contract; activePolicy: ActivePolicy }

export interface BrowserEnvironmentSettings {
  beltSpeed: number;
  spawnInterval: number;
  beltSpeedNoise: number;
  spawnIntervalNoise: number;
  seed: number;
  playbackSpeed: number;
  randomizeJointPoses: boolean;
  jointPoseNoise: number[];
  jointLimitMargin: number;
  maxJointPoseResetAttempts: number;
  maxFailures: number;
  maxEpisodeSeconds: number;
  actionRepeat: number;
  maxActiveBoxes: number;
}

export interface SortDrop {
  boxIndex: number;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  targetBin: 0 | 1;
  correct: boolean;
}

export interface BrowserStepResult {
  observation: Float32Array;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  physicsSteps: number;
  simulationTime: number;
  correctCount: number;
  wrongCount: number;
  missedCount: number;
  firstContactCount: number;
  correctByClass: [number, number];
  wrongByClass: [number, number];
  missedByClass: [number, number];
  dropEvents: SortDrop[];
  missedReasons: Record<string, number>;
  unsafeReasons: Record<string, number>;
  activeBoxCount: number;
  episodeReturn: number;
  returnedEpisode: boolean;
}

export interface TelemetrySnapshot extends BrowserStepResult {
  policyTransitions: number;
  totalPhysicsSteps: number;
  checkpointId: string;
  environmentSteps: number;
  readyState: string;
}

export interface VisualMeshAsset {
  geomId: number;
  positions: Float32Array;
  normals?: Float32Array;
  indices: Uint32Array;
  rgba: [number, number, number, number];
}

export interface RendererState {
  qpos: Float64Array;
  siteXpos: Float64Array;
  geomXpos: Float64Array;
  geomXmat: Float64Array;
}

export type SimulationWorkerRequest =
  | { type: 'initialize' }
  | { type: 'step' }
  | { type: 'reset'; settings: BrowserEnvironmentSettings };

export type SimulationWorkerResponse =
  | { type: 'status'; message: string }
  | { type: 'ready'; settings: BrowserEnvironmentSettings; initial: BrowserStepResult; state: RendererState; visualMeshes: VisualMeshAsset[]; modelManifest: ModelManifest; policyId: string; environmentSteps: number; timestep: number }
  | { type: 'step'; result: BrowserStepResult; state: RendererState }
  | { type: 'reset'; result: BrowserStepResult; state: RendererState }
  | { type: 'error'; message: string };
