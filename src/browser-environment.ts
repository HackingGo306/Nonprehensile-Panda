import type { MjData, MjModel } from '@mujoco/mujoco';
import type { Policy } from './policy';
import type { BrowserEnvironmentSettings, BrowserStepResult, Contract, LoadedDemo, ModelManifest, PolicyManifest, SortDrop } from './types';

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const finite = (value: number) => Number.isFinite(value) ? value : 0;

class Mulberry32 {
  constructor(private state: number) {}
  next(): number { let value = this.state += 0x6D2B79F5; value = Math.imul(value ^ value >>> 15, value | 1); value ^= value + Math.imul(value ^ value >>> 7, value | 61); return ((value ^ value >>> 14) >>> 0) / 4294967296; }
  uniform(low: number, high: number): number { return low + this.next() * (high - low); }
  choice(values: number[]): number { return values[Math.floor(this.next() * values.length)]; }
}

type Arrays = { qpos: Float64Array; qvel: Float64Array; ctrl: Float64Array; xpos: Float64Array; xmat: Float64Array; siteXpos: Float64Array; siteXmat: Float64Array };
type Substep = { correct: boolean[]; wrong: boolean[]; missed: boolean[]; firstContact: boolean[]; missedReasons: Record<string, boolean[]>; unsafeReasons: Record<string, boolean[]>; robotConveyorCollision: boolean; dropEvents: SortDrop[] };

export function defaultSettings(metadata: PolicyManifest['policy_metadata'], demoEnvironment?: PolicyManifest['demo_environment']): BrowserEnvironmentSettings {
  const source = demoEnvironment ?? metadata;
  const environment = source.environment as Record<string, unknown>;
  const resolved = source.resolved_stage as Record<string, unknown>;
  return {
    beltSpeed: Number(resolved.belt_speed), spawnInterval: Number(resolved.spawn_interval), beltSpeedNoise: Number(environment.belt_speed_noise), spawnIntervalNoise: Number(environment.spawn_interval_noise), seed: Number(environment.model_seed ?? 0), playbackSpeed: 1,
    randomizeJointPoses: Boolean(environment.randomize_joint_poses), jointPoseNoise: Array.from(environment.joint_pose_noise as number[]), jointLimitMargin: Number(environment.joint_limit_margin),
    maxJointPoseResetAttempts: Number(environment.max_joint_pose_reset_attempts), maxFailures: Number(environment.max_failures), maxEpisodeSeconds: Number(environment.max_episode_seconds), actionRepeat: Number(environment.action_repeat), maxActiveBoxes: Number(resolved.max_active_boxes),
  };
}

export class BrowserEnvironment {
  private readonly model: MjModel;
  private readonly data: MjData;
  private readonly manifest: ModelManifest;
  private readonly contract: Contract;
  private readonly mujoco: LoadedDemo['mujoco'];
  private readonly active = new Array<boolean>(12).fill(false);
  private readonly contacted = new Array<boolean>(12).fill(false);
  private readonly spawnedAt = new Float64Array(12);
  private readonly controlTargets = new Float32Array(6);
  private readonly previousAction = new Float32Array(6);
  private readonly targetBins: number[];
  private settings: BrowserEnvironmentSettings;
  private taskBeltSpeed = 0;
  private taskSpawnInterval = 0;
  private nextSpawnTime = 0;
  private nextClass = 0;
  private episodeSteps = 0;
  private failureCount = 0;
  private successCount = 0;
  private wrongCount = 0;
  private missedCount = 0;
  private episodeReturn = 0;
  private generalRng = new Mulberry32(0);
  private poseRng = new Mulberry32(1);
  private beltRng = new Mulberry32(2);
  private spawnRng = new Mulberry32(3);
  private lastMissedReasons: Record<string, number> = {};
  private lastUnsafeReasons: Record<string, number> = {};
  private destroyed = false;

  constructor(private readonly demo: LoadedDemo, initialSettings: BrowserEnvironmentSettings) {
    this.model = demo.model;
    this.data = demo.data;
    this.manifest = demo.modelManifest;
    this.contract = demo.contract;
    this.mujoco = demo.mujoco;
    this.targetBins = [...this.manifest.target_bins];
    this.settings = initialSettings;
  }

  private arrays(): Arrays {
    return { qpos: this.data.qpos as Float64Array, qvel: this.data.qvel as Float64Array, ctrl: this.data.ctrl as Float64Array, xpos: this.data.xpos as Float64Array, xmat: this.data.xmat as Float64Array, siteXpos: this.data.site_xpos as Float64Array, siteXmat: this.data.site_xmat as Float64Array };
  }
  private c(name: string): number { return Number(this.contract.constants[name]); }
  private ca(name: string): number[] { return this.contract.constants[name] as number[]; }
  private position(bodyId: number): [number, number, number] { const x = this.arrays().xpos; return [x[bodyId * 3], x[bodyId * 3 + 1], x[bodyId * 3 + 2]]; }
  private tool(): [number, number, number] { const x = this.arrays().siteXpos; const base = this.manifest.push_site_id * 3; return [x[base], x[base + 1], x[base + 2]]; }
  private maxEpisodeSteps(): number { return Math.max(1, Math.round(this.settings.maxEpisodeSeconds / (this.manifest.timestep * this.settings.actionRepeat))); }
  private activeCount(): number { return this.active.filter(Boolean).length; }
  private boxPosition(index: number): [number, number, number] { const address = this.manifest.box_qpos_addresses[index]; const qpos = this.arrays().qpos; return [qpos[address], qpos[address + 1], qpos[address + 2]]; }
  private binRegion(targetBin: number): ModelManifest['sorting_bin_regions'][number] {
    const region = this.manifest.sorting_bin_regions.find((candidate) => candidate.target_bin === targetBin);
    if (!region) throw new Error(`Model manifest has no sorting-bin region for target bin ${targetBin}.`);
    return region;
  }
  private inBinRegion(targetBin: number, x: number, y: number): boolean {
    const region = this.binRegion(targetBin);
    return x >= region.x_min && x <= region.x_max && y >= region.y_min && y <= region.y_max;
  }
  private binXMax(targetBin: number): number { return this.binRegion(targetBin).x_max; }
  private furthestBinX(): number { return Math.max(...this.manifest.sorting_bin_regions.map((region) => region.x_max)); }
  private setBoxVelocity(index: number, values: number[]): void { const address = this.manifest.box_qvel_addresses[index]; this.arrays().qvel.set(values, address); }
  private zeroReasons(): void { this.lastMissedReasons = { fallen: 0, unsafe_entry: 0, deadline: 0, exited: 0, unclassified: 0 }; this.lastUnsafeReasons = { height: 0, upright: 0, linear_speed: 0, angular_speed: 0, unclassified: 0 }; }

  reset(settings = this.settings): Float32Array {
    if (this.destroyed) throw new Error('Environment has been destroyed.');
    if (![settings.beltSpeed, settings.spawnInterval, settings.playbackSpeed].every((value) => Number.isFinite(value) && value > 0)) throw new Error('Belt speed, spawn interval, and playback speed must be positive finite values.');
    if (![settings.beltSpeedNoise, settings.spawnIntervalNoise].every((value) => Number.isFinite(value) && value >= 0 && value < 1)) throw new Error('Noise values must be finite and within [0, 1).');
    if (!Number.isInteger(settings.seed)) throw new Error('Seed must be an integer.');
    this.settings = { ...settings, jointPoseNoise: [...settings.jointPoseNoise] };
    this.generalRng = new Mulberry32(settings.seed >>> 0);
    this.poseRng = new Mulberry32((settings.seed + 0x9e3779b9) >>> 0);
    this.beltRng = new Mulberry32((settings.seed + 0x243f6a88) >>> 0);
    this.spawnRng = new Mulberry32((settings.seed + 0xb7e15162) >>> 0);
    this.mujoco.mj_resetData(this.model, this.data);
    const arrays = this.arrays();
    for (let index = 0; index < 6; index += 1) {
      arrays.qpos[this.manifest.controlled_joint_qpos_addresses[index]] = this.manifest.panda_home_qpos[index];
      arrays.qvel[this.manifest.controlled_joint_qvel_addresses[index]] = 0;
      this.controlTargets[index] = this.manifest.panda_home_ctrl[index];
    }
    for (let index = 0; index < 3; index += 1) arrays.qpos[this.manifest.locked_joint_qpos_addresses[index]] = this.manifest.panda_home_qpos[index + 6];
    arrays.ctrl.set(this.manifest.panda_home_ctrl);
    for (let index = 0; index < 12; index += 1) this.parkBox(index);
    this.active.fill(false); this.contacted.fill(false); this.spawnedAt.fill(0); this.previousAction.fill(0);
    this.episodeSteps = 0; this.failureCount = 0; this.successCount = 0; this.wrongCount = 0; this.missedCount = 0; this.episodeReturn = 0; this.nextSpawnTime = 0; this.nextClass = settings.seed % 2;
    this.taskBeltSpeed = settings.beltSpeed * (1 + this.beltRng.uniform(-settings.beltSpeedNoise, settings.beltSpeedNoise));
    this.taskSpawnInterval = settings.spawnInterval;
    this.zeroReasons();
    this.mujoco.mj_forward(this.model, this.data);
    if (settings.randomizeJointPoses) this.resetJointPose();
    else this.mujoco.mj_forward(this.model, this.data);
    return this.observe();
  }

  private resetJointPose(): void {
    const arrays = this.arrays();
    const low = this.manifest.joint_low.map((value) => value + this.settings.jointLimitMargin);
    const high = this.manifest.joint_high.map((value) => value - this.settings.jointLimitMargin);
    for (let attempt = 0; attempt < this.settings.maxJointPoseResetAttempts; attempt += 1) {
      for (let index = 0; index < 6; index += 1) {
        const noise = this.settings.jointPoseNoise[index] ?? 0;
        const candidate = clamp(this.manifest.panda_home_qpos[index] + this.poseRng.uniform(-noise, noise), low[index], high[index]);
        arrays.qpos[this.manifest.controlled_joint_qpos_addresses[index]] = candidate;
        arrays.qvel[this.manifest.controlled_joint_qvel_addresses[index]] = 0;
        this.controlTargets[index] = candidate;
        arrays.ctrl[index] = candidate;
      }
      this.mujoco.mj_forward(this.model, this.data);
      if (!this.robotConveyorCollision()) return;
    }
    throw new Error(`Unable to sample a valid joint pose after ${this.settings.maxJointPoseResetAttempts} attempts.`);
  }

  private parkBox(index: number): void {
    const address = this.manifest.box_qpos_addresses[index];
    const arrays = this.arrays();
    arrays.qpos.set([this.c('PARK_X'), this.c('PARK_Y0') + this.c('PARK_SPACING') * index, this.c('PARK_Z'), 1, 0, 0, 0], address);
    arrays.qvel.fill(0, this.manifest.box_qvel_addresses[index], this.manifest.box_qvel_addresses[index] + 6);
  }

  private sampleSpawnInterval(): number { return this.settings.spawnInterval * (1 + this.spawnRng.uniform(-this.settings.spawnIntervalNoise, this.settings.spawnIntervalNoise)); }
  private trySpawn(): void {
    if (this.data.time < this.nextSpawnTime || this.activeCount() >= Math.min(this.settings.maxActiveBoxes, this.c('MAX_ACTIVE_BOXES'))) return;
    const inlet = this.active.some((active, index) => active && this.position(this.manifest.box_body_ids[index])[0] >= this.c('INLET_X') - 0.08 && this.position(this.manifest.box_body_ids[index])[0] <= this.c('INLET_X') + this.c('INLET_CLEARANCE') && Math.abs(this.position(this.manifest.box_body_ids[index])[1]) <= 0.22);
    const overlapDistance = 2 * Math.max(...this.manifest.box_half_extents.map((extent) => extent[0]));
    if (inlet && this.taskBeltSpeed * this.taskSpawnInterval > overlapDistance) { this.nextSpawnTime += Math.max(0.002, Math.min(0.02, this.taskSpawnInterval)); return; }
    const eligible = this.active.map((active, index) => !active && this.targetBins[index] === this.nextClass ? index : -1).filter((index) => index >= 0);
    if (!eligible.length) return;
    const index = this.generalRng.choice(eligible);
    const half = this.manifest.box_half_extents[index];
    const yaw = this.generalRng.uniform(-0.25, 0.25);
    this.arrays().qpos.set([this.c('INLET_X'), this.generalRng.uniform(-this.c('LATERAL_LIMIT'), this.c('LATERAL_LIMIT')), this.c('BELT_TOP_Z') + half[2] + 0.01, Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], this.manifest.box_qpos_addresses[index]);
    this.setBoxVelocity(index, [0, 0, 0, 0, 0, 0]);
    this.active[index] = true; this.contacted[index] = false; this.spawnedAt[index] = this.data.time;
    this.taskSpawnInterval = this.sampleSpawnInterval(); this.nextSpawnTime = this.data.time + this.taskSpawnInterval; this.nextClass = 1 - this.nextClass;
  }

  private driveAndStep(control: Float32Array): void {
    this.trySpawn();
    for (let index = 0; index < 12; index += 1) {
      if (!this.active[index]) this.parkBox(index);
      else {
        const position = this.boxPosition(index);
        if (position[0] >= this.c('INLET_X') - 0.05 && position[0] <= this.c('EXIT_X') && Math.abs(position[1]) <= 0.22) this.arrays().qvel[this.manifest.box_qvel_addresses[index]] = this.taskBeltSpeed;
      }
    }
    this.arrays().ctrl.set(control);
    this.mujoco.mj_step(this.model, this.data);
    for (let index = 0; index < 12; index += 1) if (this.active[index]) {
      const base = this.manifest.box_qvel_addresses[index]; const qvel = this.arrays().qvel;
      if (Math.abs(qvel[base + 1]) <= this.c('LATERAL_VELOCITY_SETTLE_THRESHOLD')) qvel[base + 1] = 0;
      if (Math.hypot(qvel[base + 3], qvel[base + 4], qvel[base + 5]) <= this.c('ANGULAR_VELOCITY_SETTLE_THRESHOLD')) qvel.fill(0, base + 3, base + 6);
    }
  }

  private contactBoxes(): boolean[] {
    const touching = new Array<boolean>(12).fill(false);
    const contact = this.data.contact;
    try {
      for (let index = 0; index < contact.size(); index += 1) {
        const item = contact.get(index);
        if (!item) continue;
        const other = item.geom1 === this.manifest.hand_geom_id ? item.geom2 : item.geom2 === this.manifest.hand_geom_id ? item.geom1 : -1;
        const box = this.manifest.box_geom_ids.indexOf(other);
        if (box >= 0) touching[box] = true;
        item.delete();
      }
    } finally { contact.delete(); }
    return touching;
  }

  private robotConveyorCollision(): boolean {
    const contact = this.data.contact;
    try {
      for (let index = 0; index < contact.size(); index += 1) {
        const item = contact.get(index); if (!item) continue;
        const other = item.geom1 === this.manifest.conveyor_surface_geom_id ? item.geom2 : item.geom2 === this.manifest.conveyor_surface_geom_id ? item.geom1 : -1;
        const robotGroup = other >= 0 ? (this.model.geom_group as Int32Array)[other] : -1;
        item.delete();
        if (robotGroup === 3) return true;
      }
      return false;
    } finally { contact.delete(); }
  }

  private recycle(captured: boolean[], deadline: boolean[]): { exited: boolean[]; fallen: boolean[]; expired: boolean[]; excessive: boolean[]; nonfinite: boolean[] } {
    const exited = new Array<boolean>(12).fill(false), fallen = new Array<boolean>(12).fill(false), expired = new Array<boolean>(12).fill(false), excessive = new Array<boolean>(12).fill(false), nonfinite = new Array<boolean>(12).fill(false);
    const qpos = this.arrays().qpos, qvel = this.arrays().qvel;
    for (let index = 0; index < 12; index += 1) if (this.active[index]) {
      const position = this.position(this.manifest.box_body_ids[index]); const velocityBase = this.manifest.box_qvel_addresses[index];
      exited[index] = position[0] > this.c('EXIT_X'); fallen[index] = position[2] < this.c('SAFETY_Z') || position[2] < this.c('FALLEN_Z'); expired[index] = this.data.time - this.spawnedAt[index] > this.c('MAX_BOX_AGE');
      excessive[index] = Math.hypot(qvel[velocityBase], qvel[velocityBase + 1], qvel[velocityBase + 2]) > this.c('MAX_LINEAR_SPEED') || Math.hypot(qvel[velocityBase + 3], qvel[velocityBase + 4], qvel[velocityBase + 5]) > this.c('MAX_ANGULAR_SPEED');
      const positionBase = this.manifest.box_qpos_addresses[index]; nonfinite[index] = !Array.from(qpos.slice(positionBase, positionBase + 7)).every(Number.isFinite) || !Array.from(qvel.slice(velocityBase, velocityBase + 6)).every(Number.isFinite);
      if (captured[index] || deadline[index] || exited[index] || fallen[index] || expired[index] || excessive[index] || nonfinite[index]) { this.active[index] = false; this.contacted[index] = false; this.parkBox(index); }
    }
    return { exited, fallen, expired, excessive, nonfinite };
  }

  private substep(control: Float32Array): Substep {
    this.driveAndStep(control);
    const touching = this.contactBoxes(); const firstContact = touching.map((value, index) => this.active[index] && value && !this.contacted[index]);
    for (let index = 0; index < 12; index += 1) this.contacted[index] ||= this.active[index] && touching[index];
    const correct = new Array<boolean>(12).fill(false), wrong = new Array<boolean>(12).fill(false), unsafeEntry = new Array<boolean>(12).fill(false), deadline = new Array<boolean>(12).fill(false);
    const unsafe = { height: new Array<boolean>(12).fill(false), upright: new Array<boolean>(12).fill(false), linear_speed: new Array<boolean>(12).fill(false), angular_speed: new Array<boolean>(12).fill(false) };
    const qvel = this.arrays().qvel, xmat = this.arrays().xmat;
    for (let index = 0; index < 12; index += 1) if (this.active[index]) {
      const [x, y, z] = this.position(this.manifest.box_body_ids[index]); const target = this.targetBins[index];
      const green = this.inBinRegion(this.c('GREEN_CLASS'), x, y);
      const blue = this.inBinRegion(this.c('BLUE_CLASS'), x, y);
      const base = this.manifest.box_qvel_addresses[index]; const rotationBase = this.manifest.box_body_ids[index] * 9;
      unsafe.height[index] = Math.abs(z - (this.c('BELT_TOP_Z') + this.manifest.box_half_extents[index][2])) > this.c('SAFE_HEIGHT_TOLERANCE'); unsafe.upright[index] = xmat[rotationBase + 8] < this.c('SAFE_UPRIGHT_Z');
      unsafe.linear_speed[index] = Math.hypot(qvel[base], qvel[base + 1], qvel[base + 2]) > this.c('SAFE_LINEAR_SPEED'); unsafe.angular_speed[index] = Math.hypot(qvel[base + 3], qvel[base + 4], qvel[base + 5]) > this.c('SAFE_ANGULAR_SPEED');
      const stable = !unsafe.height[index] && !unsafe.upright[index] && !unsafe.linear_speed[index] && !unsafe.angular_speed[index];
      const correctRegion = target === this.c('GREEN_CLASS') ? green : blue; const wrongRegion = target === this.c('GREEN_CLASS') ? blue : green;
      correct[index] = correctRegion && stable; wrong[index] = wrongRegion; unsafeEntry[index] = correctRegion && !stable; deadline[index] = x > this.furthestBinX() && !correct[index] && !wrong[index] && !unsafeEntry[index];
    }
    const captured = correct.map((value, index) => value || wrong[index] || unsafeEntry[index]);
    // Outcome boxes are still parked immediately for identical policy inputs.
    // The renderer receives a snapshot and carries the visual box to its
    // landing point independently, so the demonstration shows the drop
    // without changing MuJoCo state, spawn timing, or PPO inference.
    const dropEvents = captured.flatMap((capturedBox, index): SortDrop[] => {
      if (!capturedBox) return [];
      const qposAddress = this.manifest.box_qpos_addresses[index]; const qpos = this.arrays().qpos;
      return [{ boxIndex: index, position: this.boxPosition(index), quaternion: [qpos[qposAddress + 3], qpos[qposAddress + 4], qpos[qposAddress + 5], qpos[qposAddress + 6]], targetBin: this.targetBins[index] as 0 | 1, correct: correct[index] }];
    });
    const recycle = this.recycle(captured, deadline);
    const missed = unsafeEntry.map((value, index) => value || deadline[index] || recycle.exited[index] || recycle.fallen[index] || recycle.expired[index] || recycle.excessive[index]);
    return { correct, wrong, missed, firstContact, missedReasons: { fallen: recycle.fallen.map((value, index) => value && missed[index]), unsafe_entry: unsafeEntry.map((value, index) => value && missed[index]), deadline: deadline.map((value, index) => value && missed[index]), exited: recycle.exited.map((value, index) => value && missed[index]) }, unsafeReasons: unsafe, robotConveyorCollision: this.robotConveyorCollision(), dropEvents };
  }

  private count(mask: boolean[]): number { return mask.reduce((sum, value) => sum + Number(value), 0); }
  private byClass(mask: boolean[]): [number, number] { return [this.count(mask.map((value, index) => value && this.targetBins[index] === 0)), this.count(mask.map((value, index) => value && this.targetBins[index] === 1))]; }

  private shaping(beforeActive: boolean[], beforePositions: number[][], beforeTool: number[], terminal: boolean[]): { directed: number; approach: number; unsafeMotion: number; jointSpeed: number; handVertical: number; hover: number } {
    const afterTool = this.tool(); const directions = this.targetBins.map((target) => target === this.c('GREEN_CLASS') ? 1 : -1); let directed = 0;
    const candidates: number[] = [];
    for (let index = 0; index < 12; index += 1) {
      const after = this.position(this.manifest.box_body_ids[index]); const common = beforeActive[index] && this.active[index] && !terminal[index]; const targetMax = this.binXMax(this.targetBins[index]);
      if (common && beforePositions[index][0] >= this.c('INTERCEPTION_X_MIN') && beforePositions[index][0] <= targetMax) directed += clamp(directions[index] * (after[1] - beforePositions[index][1]) / this.c('GREEN_Y_MIN'), -0.25, 0.25);
      candidates.push(common && beforePositions[index][0] <= targetMax ? beforePositions[index][0] : -1e6);
    }
    const focus = candidates.indexOf(Math.max(...candidates)); let approach = 0;
    if (candidates[focus] > -1e5) {
      const beforeTarget = [...beforePositions[focus]], afterTarget = [...this.position(this.manifest.box_body_ids[focus])]; const offset = -directions[focus] * (this.manifest.box_half_extents[focus][1] + this.c('PUSH_CLEARANCE'));
      beforeTarget[1] += offset; afterTarget[1] += offset; beforeTarget[2] = this.c('PUSH_TARGET_Z'); afterTarget[2] = this.c('PUSH_TARGET_Z');
      const potential = (target: number[], tool: number[]) => Math.max(0, 1 - Math.hypot(target[0] - tool[0], target[1] - tool[1], target[2] - tool[2]) / 0.75);
      approach = clamp(potential(afterTarget, afterTool) - potential(beforeTarget, beforeTool), -1, 1);
    }
    const unsafeCount = Array.from({ length: 12 }, (_, index) => this.active[index] && (Math.abs(this.position(this.manifest.box_body_ids[index])[2] - (this.c('BELT_TOP_Z') + this.manifest.box_half_extents[index][2])) > this.c('SAFE_HEIGHT_TOLERANCE'))).filter(Boolean).length;
    const qvel = this.arrays().qvel; let jointSpeed = 0; for (let index = 0; index < 6; index += 1) jointSpeed += clamp(qvel[this.manifest.controlled_joint_qvel_addresses[index]] / 3, -2, 2) ** 2; jointSpeed /= 6;
    const wrist = this.position(this.manifest.wrist_body_id); const horizontal = Math.max(Math.hypot(afterTool[0] - wrist[0], afterTool[1] - wrist[1]) - this.c('HAND_WRIST_HORIZONTAL_TOLERANCE'), 0); const above = Math.max(afterTool[2] - wrist[2] - this.c('HAND_WRIST_VERTICAL_TOLERANCE'), 0);
    return { directed: clamp(directed, -1, 1), approach, unsafeMotion: unsafeCount / Math.max(this.activeCount(), 1), jointSpeed, handVertical: clamp(Math.max(horizontal, above) / this.c('HAND_WRIST_ALIGNMENT_SCALE'), 0, 1), hover: clamp(Math.max(Math.abs(afterTool[2] - this.c('PUSH_TARGET_Z')) - this.c('HOVER_HEIGHT_TOLERANCE'), 0) / this.c('HOVER_HEIGHT_SCALE'), 0, 1) };
  }

  step(action: Float32Array): BrowserStepResult {
    if (action.length !== 6 || !Array.from(action).every(Number.isFinite)) throw new Error('Action must contain six finite values.');
    const clipped = Float32Array.from(action, (value) => clamp(value, -1, 1)); const targets = new Float32Array(6);
    for (let index = 0; index < 6; index += 1) targets[index] = clamp(this.controlTargets[index] + clipped[index] * this.c('ACTION_DELTA_RADIANS'), this.manifest.joint_low[index], this.manifest.joint_high[index]);
    const control = new Float32Array([...targets, this.manifest.panda_home_ctrl[6], 0.04]); const beforeActive = [...this.active]; const beforePositions = this.manifest.box_body_ids.map((body) => [...this.position(body)]); const beforeTool = [...this.tool()];
    const combined: Substep = { correct: new Array<boolean>(12).fill(false), wrong: new Array<boolean>(12).fill(false), missed: new Array<boolean>(12).fill(false), firstContact: new Array(12).fill(false), missedReasons: { fallen: new Array(12).fill(false), unsafe_entry: new Array(12).fill(false), deadline: new Array(12).fill(false), exited: new Array(12).fill(false) }, unsafeReasons: { height: new Array(12).fill(false), upright: new Array(12).fill(false), linear_speed: new Array(12).fill(false), angular_speed: new Array(12).fill(false) }, robotConveyorCollision: false, dropEvents: [] };
    let physicsSteps = 0;
    for (; physicsSteps < this.settings.actionRepeat; physicsSteps += 1) {
      const result = this.substep(control); for (const field of ['correct', 'wrong', 'missed', 'firstContact'] as const) combined[field] = combined[field].map((value, index) => value || result[field][index]);
      for (const group of ['missedReasons', 'unsafeReasons'] as const) for (const name of Object.keys(combined[group])) combined[group][name] = combined[group][name].map((value, index) => value || result[group][name][index]);
      combined.robotConveyorCollision ||= result.robotConveyorCollision;
      combined.dropEvents.push(...result.dropEvents);
    }
    const correctCount = this.count(combined.correct), wrongCount = this.count(combined.wrong), missedCount = this.count(combined.missed), firstContactCount = this.count(combined.firstContact); const terminal = combined.correct.map((value, index) => value || combined.wrong[index] || combined.missed[index]);
    const signals = this.shaping(beforeActive, beforePositions, beforeTool, terminal); const weights = this.contract.stage_reward_weights;
    const meanSquare = (values: Float32Array) => Array.from(values).reduce((total, value) => total + value * value, 0) / values.length;
    const actionChange = Array.from(clipped).reduce((total, value, index) => total + (value - this.previousAction[index]) ** 2, 0) / 6;
    const reward = correctCount * weights.correct_sort + wrongCount * weights.wrong_sort + missedCount * weights.missed_box + signals.approach * weights.approach_progress + firstContactCount * weights.first_contact + signals.directed * weights.directed_progress + signals.unsafeMotion * weights.unsafe_box_motion + signals.jointSpeed * weights.joint_speed + meanSquare(clipped) * weights.action_magnitude + actionChange * weights.action_change + Number(combined.robotConveyorCollision) * weights.robot_conveyor_collision + signals.handVertical * weights.hand_vertical_error + signals.hover * weights.hover_height_error;
    this.controlTargets.set(targets); this.previousAction.set(clipped); this.episodeSteps += 1; this.failureCount += wrongCount + missedCount; this.successCount += correctCount; this.wrongCount += wrongCount; this.missedCount += missedCount; this.episodeReturn += reward;
    this.lastMissedReasons = Object.fromEntries(Object.entries(combined.missedReasons).map(([name, mask]) => [name, this.count(mask)])); this.lastMissedReasons.unclassified = this.count(combined.missed.map((value, index) => value && !Object.values(combined.missedReasons).some((mask) => mask[index])));
    this.lastUnsafeReasons = Object.fromEntries(Object.entries(combined.unsafeReasons).map(([name, mask]) => [name, this.count(mask)])); this.lastUnsafeReasons.unclassified = 0;
    // This is a playback experience, not a training rollout.  Failed and timed
    // episodes are reset as normal cycle boundaries; no physics condition can
    // surface as an "unstable environment" or halt the player.
    const terminated = false; const truncated = this.failureCount >= this.settings.maxFailures || this.episodeSteps >= this.maxEpisodeSteps(); const returnedEpisode = truncated; const simulationTime = this.data.time; const activeBoxCount = this.activeCount(); const episodeReturn = this.episodeReturn;
    const result: BrowserStepResult = { observation: this.observe(), reward: finite(reward), terminated, truncated, physicsSteps, simulationTime, correctCount, wrongCount, missedCount, firstContactCount, correctByClass: this.byClass(combined.correct), wrongByClass: this.byClass(combined.wrong), missedByClass: this.byClass(combined.missed), dropEvents: combined.dropEvents, missedReasons: { ...this.lastMissedReasons }, unsafeReasons: { ...this.lastUnsafeReasons }, activeBoxCount, episodeReturn, returnedEpisode };
    if (returnedEpisode) result.observation = this.reset(this.settings);
    return result;
  }

  runPolicyStep(policy: Policy): BrowserStepResult { return this.step(policy.act(this.observe())); }

  observe(): Float32Array {
    const output = new Float32Array(129); const layout = this.contract.observation_layout; const arrays = this.arrays();
    for (let index = 0; index < 6; index += 1) { const qpos = arrays.qpos[this.manifest.controlled_joint_qpos_addresses[index]], qvel = arrays.qvel[this.manifest.controlled_joint_qvel_addresses[index]], range = this.manifest.joint_high[index] - this.manifest.joint_low[index]; output[layout.JOINT_POSITION[0] + index] = finite(2 * (qpos - this.manifest.joint_low[index]) / range - 1); output[layout.JOINT_VELOCITY[0] + index] = clamp(qvel / 3, -1, 1); output[layout.TARGET_ERROR[0] + index] = clamp(2 * (this.controlTargets[index] - qpos) / range, -1, 1); output[layout.PREVIOUS_ACTION[0] + index] = this.previousAction[index]; }
    const tool = this.tool(); const reference = [this.c('PANDA_TOOL_REFERENCE_X'), 0, 0.42], scale = [0.75, 0.75, 0.5]; for (let index = 0; index < 3; index += 1) output[layout.TOOL_POSITION[0] + index] = clamp((tool[index] - reference[index]) / scale[index], -2, 2);
    const toolMatrix = arrays.siteXmat; const toolBase = this.manifest.push_site_id * 9; [0, 1, 3, 4, 6, 7].forEach((offset, index) => { output[layout.TOOL_ROTATION_6D[0] + index] = toolMatrix[toolBase + offset]; });
    const targetMax = this.targetBins.map((target) => this.binXMax(target)); const selected = Array.from({ length: 12 }, (_, index) => index).sort((left, right) => { const lu = this.active[left] && this.position(this.manifest.box_body_ids[left])[0] <= targetMax[left] ? this.position(this.manifest.box_body_ids[left])[0] : -1e6; const ru = this.active[right] && this.position(this.manifest.box_body_ids[right])[0] <= targetMax[right] ? this.position(this.manifest.box_body_ids[right])[0] : -1e6; return ru - lu || left - right; }).slice(0, 4);
    for (let slot = 0; slot < selected.length; slot += 1) { const index = selected[slot], offset = layout.BOXES[0] + slot * this.c('BOX_FEATURE_SIZE'); const position = this.position(this.manifest.box_body_ids[index]); const valid = this.active[index] && position[0] <= targetMax[index]; if (!valid) continue; const half = this.manifest.box_half_extents[index]; const qposBase = this.manifest.box_qpos_addresses[index], qvelBase = this.manifest.box_qvel_addresses[index]; const target = this.targetBins[index]; const goal = [target === 0 ? this.c('GREEN_BIN_X_CENTER') : this.c('BLUE_BIN_X_CENTER'), target === 0 ? this.c('GREEN_Y_CENTER') : this.c('BLUE_Y_CENTER'), this.c('BELT_TOP_Z') + half[2]]; const quat = [arrays.qpos[qposBase + 3], arrays.qpos[qposBase + 4], arrays.qpos[qposBase + 5], arrays.qpos[qposBase + 6]]; if (quat[0] < 0) for (let item = 0; item < 4; item += 1) quat[item] *= -1; output.set([(position[0] - tool[0]), (position[1] - tool[1]) / 0.75, (position[2] - tool[2]) / 0.5, (goal[0] - position[0]), (goal[1] - position[1]) / 0.75, (goal[2] - position[2]) / 0.5, ...quat, clamp(arrays.qvel[qvelBase] / this.c('MAX_LINEAR_SPEED'), -1, 1), clamp(arrays.qvel[qvelBase + 1] / this.c('MAX_LINEAR_SPEED'), -1, 1), clamp(arrays.qvel[qvelBase + 2] / this.c('MAX_LINEAR_SPEED'), -1, 1), clamp(arrays.qvel[qvelBase + 3] / this.c('MAX_ANGULAR_SPEED'), -1, 1), clamp(arrays.qvel[qvelBase + 4] / this.c('MAX_ANGULAR_SPEED'), -1, 1), clamp(arrays.qvel[qvelBase + 5] / this.c('MAX_ANGULAR_SPEED'), -1, 1), half[0] / 0.065, half[1] / 0.055, half[2] / 0.06, target === 0 ? 1 : 0, target === 1 ? 1 : 0, clamp(2 * (position[0] - this.c('INLET_X')) / (this.c('BIN_X_MAX') - this.c('INLET_X')) - 1, -1, 1), 1], offset); }
    output.set([clamp(this.taskBeltSpeed / 0.5, 0, 2), clamp(this.taskSpawnInterval / 6, 0, 1), this.failureCount / this.settings.maxFailures, this.episodeSteps / this.maxEpisodeSteps()], layout.GLOBAL[0]);
    for (let index = 0; index < output.length; index += 1) output[index] = finite(output[index]); return output;
  }

  destroy(): void { this.destroyed = true; }
}
