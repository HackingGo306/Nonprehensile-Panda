# Agent-Ready Implementation Plan: Static MuJoCo Checkpoint Demo

## 0. Instructions for the implementing agent

Your working directory is the separate web-project directory:

```text
/Users/cameron/robotics-demo
```

The current source workspace is a sibling directory:

```text
/Users/cameron/robotics
```

This handoff document is `/Users/cameron/robotics/plan.md`, which is reachable from the project directory as `../robotics/plan.md`.

Access the source workspace whenever needed. Read its Python code, MuJoCo XML, assets, requirements, tests, and checkpoints through absolute paths or sibling paths such as `../robotics/...`; do not assume those files are inside the web-project directory. Build the browser project at `/Users/cameron/robotics-demo/`, outside the source repository. The production result must be a static site: HTML, CSS, JavaScript, WebAssembly, MuJoCo model assets, and a browser-readable policy artifact. There must be no Flask/FastAPI server, Python runtime, API, database, or server-side inference at deployment time.

The raw Python/JAX checkpoint is only an offline build input. The user accepts that the converted policy weights will be visible to the browser.

Before changing files:

1. Read this plan completely.
2. Read the source files listed in Section 2 from `/Users/cameron/robotics`.
3. Preserve all existing user changes. Do not run `git reset`, `git checkout`, or broad cleanup commands.
4. Keep all website code, npm files, generated assets, fixtures, and production output under `/Users/cameron/robotics-demo/`. Do not create the website under `/Users/cameron/robotics/`. The plan is stored at `/Users/cameron/robotics/plan.md`; the website implementation belongs in `/Users/cameron/robotics-demo/`.
5. Do not begin by porting PPO training. The website only needs deterministic policy inference.

When a step is complete, run its specified validation before proceeding. If an API in the selected MuJoCo JavaScript package differs from the examples in this plan, inspect the installed package typings and official WASM examples, then update the implementation to the equivalent API while preserving the stated behavior.

## 1. Deliverable

Create a static website that:

- Loads the official MuJoCo JavaScript/WebAssembly engine in the browser.
- Loads the Panda/conveyor MuJoCo model and its local assets.
- Runs the stage-3 native conveyor environment in the browser.
- Loads a manually selected converted PPO checkpoint.
- Selects deterministic six-dimensional policy actions in the browser.
- Allows the user to change belt speed, spawn interval, their noise values, and the random seed.
- Steps MuJoCo physics and the task logic without network calls to an application API.
- Renders the robot, conveyor, boxes, bins, and success/failure indicators.
- Displays checkpoint metadata and live evaluation telemetry.
- Builds to `/Users/cameron/robotics-demo/dist/`, which can be deployed to a static host.

The website is an evaluation/demo player, not a training UI. The curriculum stage and reward profile remain fixed to the selected checkpoint's stage unless a later plan explicitly adds support for other policies.

## 2. Existing code that defines the behavior

Read these files before implementing the browser counterpart:

- [`README.md`](/Users/cameron/robotics/README.md): installation, native evaluation, stages, and checkpoint conventions.
- [`run_native_policy.py`](/Users/cameron/robotics/run_native_policy.py): policy loading, stage resolution, environment creation, deterministic evaluation, and metrics.
- [`rl/native_checkpoints.py`](/Users/cameron/robotics/rl/native_checkpoints.py): native schema-4 policy validation and Flax serialization.
- [`rl/native_environment_registry.py`](/Users/cameron/robotics/rl/native_environment_registry.py): native environment factory and stable environment IDs.
- [`rl/native_env.py`](/Users/cameron/robotics/rl/native_env.py): native MuJoCo model preparation, reset, spawn scheduling, action stepping, outcomes, rewards, and observation construction.
- [`rl/native_single_box_env.py`](/Users/cameron/robotics/rl/native_single_box_env.py) and [`rl/native_multi_box_env.py`](/Users/cameron/robotics/rl/native_multi_box_env.py): environment behavior selected by curriculum stage.
- [`rl/native_curriculum.py`](/Users/cameron/robotics/rl/native_curriculum.py): stage definitions.
- [`rl/native_curriculum_rewards.py`](/Users/cameron/robotics/rl/native_curriculum_rewards.py): stage-3 reward function and reward component names.
- [`rl/conveyor_spec.py`](/Users/cameron/robotics/rl/conveyor_spec.py): constants, model path, action contract, observation layout, geometry, task limits, and reward configuration.
- [`rl/conveyor_model.py`](/Users/cameron/robotics/rl/conveyor_model.py): native model compilation, collision-filter mutations, box randomization, friction, robot initialization, and parked-box state.
- [`rl/ppo.py`](/Users/cameron/robotics/rl/ppo.py): `ActorCritic`, `ObservationStats`, `normalize_observation`, and `deterministic_action`.
- [`rl/native_conveyor_kernel.cpp`](/Users/cameron/robotics/rl/native_conveyor_kernel.cpp): read only to understand the optional optimization; do not port or use it in the browser.
- [`assets/mujoco_menagerie/franka_emika_panda/mjx_scene.xml`](/Users/cameron/robotics/assets/mujoco_menagerie/franka_emika_panda/mjx_scene.xml): source MJCF scene.

The official browser engine documentation is:

- [MuJoCo JavaScript/WebAssembly bindings README](https://github.com/google-deepmind/mujoco/blob/main/wasm/README.md)
- [MuJoCo Python API documentation](https://github.com/google-deepmind/mujoco/blob/main/doc/python.rst)

## 3. Exact initial policy and environment

Use this existing policy for the first demo:

```text
Run directory:
/Users/cameron/robotics/checkpoints/conveyor_native_ppo/20260723_111300_seed0

Latest training-state checkpoint:
/Users/cameron/robotics/checkpoints/conveyor_native_ppo/20260723_111300_seed0/step_000214040576

Run-level inference policy:
/Users/cameron/robotics/checkpoints/conveyor_native_ppo/20260723_111300_seed0/policy.msgpack

Adjacent metadata:
/Users/cameron/robotics/checkpoints/conveyor_native_ppo/20260723_111300_seed0/policy.json
```

Important: the existing evaluator loads `policy.msgpack` from the run directory. The `step_*` directory is an Orbax training-state checkpoint and does not replace the run-level policy file for browser inference. Keep both paths in the metadata, but use the run-level policy as the conversion input.

The selected policy metadata currently specifies:

```text
schema_version             4
environment_backend        native
observation_version        2
observation_size           129
action_size                6
hidden_size                256
environment_id             multi_box
stage_id                   3
stage_name                 multi_box_slow
reward_id                  slow_coordination
max_active_boxes           8
resolved belt speed        0.15 m/s
stage default spawn interval 1.50 s
saved/evaluation spawn interval 0.75 s
model_seed                 0
action_repeat              25
max_failures               5
max_episode_seconds        20.0
box_surface_friction       0.9
belt_speed_noise            0.2
spawn_interval_noise        0.1
randomize_joint_poses       true
joint_pose_noise            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
joint_limit_margin          0.05
max_joint_pose_reset_attempts 1000
```

The web demo must use these as the initial defaults. The UI may change belt speed, spawn interval, both noise values, and seed. It should display a warning when the current settings differ from the saved defaults.

## 4. Non-negotiable physics/model parity details

### 4.1 Do not load the raw `mjx_scene.xml` as the final native model without preparation

The source XML is primarily the MJX scene. The native Python loader changes the model before compiling it:

- Solver iterations become 10.
- Line-search iterations become 10.
- Panda collision geoms in collision group 3 receive the native workspace collision bit.
- `table_block` and `conveyor_surface` receive the native workspace collision affinity bit.
- All 12 `sorting_box_*_geom` geoms receive the native box collision bit on `contype` and `conaffinity`.
- The `hand_capsule` geom is required.
- The `push_point` site is required.
- Box geometry sizes, masses, inertias, colors, and friction are randomized/prepared for `model_seed=0`.
- Conveyor/box pair friction is set from `box_surface_friction`.

These transformations are implemented in [`rl/conveyor_model.py`](/Users/cameron/robotics/rl/conveyor_model.py), especially `compile_native_model`, `apply_box_randomization`, and `set_box_surface_friction`.

Create `/Users/cameron/robotics-demo/tools/prepare_model.py` that:

1. Resolves the source repository from the explicit `--repo-root` argument, then loads the source XML with `mujoco.MjSpec.from_file`.
2. Applies the same `MjSpec` mutations as `compile_native_model`.
3. Compiles the spec.
4. Applies the same fixed model-seed box randomization and friction settings as the selected policy.
5. Writes a browser-ready `/Users/cameron/robotics-demo/public/model/native_scene.xml` using the supported `MjSpec.to_xml()`/`encode()` API. The emitted XML must be self-contained or must preserve a relative `assets/` layout copied into the output directory.
6. Writes `/Users/cameron/robotics-demo/public/model/model_patch.json` if any compiled-model values cannot be represented in the emitted XML.
7. Copies every referenced asset while preserving the relative paths expected by the generated XML.
8. Writes `/Users/cameron/robotics-demo/public/model/model_manifest.json` containing MuJoCo version, model dimensions, named object IDs, joint addresses, box addresses, box half-extents, and all native collision settings needed by the browser.

If the installed Python API's XML serialization does not preserve a modified compiled value, do not silently ignore it. Put that value into `model_patch.json`, apply the patch to the browser `MjModel` before simulation, and call the equivalent constant-recompute/forward operation. Validate the resulting browser model against the Python prepared model.

### 4.2 Native Python reference path

Use the Python environment with `--physics-kernel python` as the reference for parity. Do not compare browser results to the optional C++ kernel initially because the browser executes ordinary MuJoCo steps, not the repository's native C++ batching extension.

### 4.3 Timing and action semantics

The MuJoCo model timestep is 0.002 seconds and the selected policy uses 25 physics steps per policy transition. One browser policy transition must do:

1. Read the current raw observation.
2. Normalize it exactly like `rl/ppo.py`.
3. Run the deterministic actor.
4. Clip the six policy values to `[-1, 1]`.
5. Update joint targets using:

   ```text
   target = clip(previous_target + action * 0.05, joint_low, joint_high)
   ```

6. Set MuJoCo controls to `[target_0..target_5, PANDA_HOME_CTRL[6], 0.04]`.
7. Execute up to 25 task/physics substeps.
8. Return the next observation and transition telemetry.

The native environment may stop the repeated substeps early when the simulation becomes unstable or when a terminal box outcome occurs. Port that behavior.

## 5. Project setup

### 5.1 Create the project

From `/Users/cameron/robotics-demo`, create a Vite TypeScript project in the current directory. If `/Users/cameron/robotics-demo/` already exists, preserve it and reconcile files rather than overwriting it. If the directory does not exist, create it. If filesystem permissions prevent writing outside `/Users/cameron/robotics`, request the required permission; do not silently place the project inside the source repository.

Expected setup commands for a new project:

```sh
mkdir -p /Users/cameron/robotics-demo
cd /Users/cameron/robotics-demo
npm create vite@latest . -- --template vanilla-ts
npm install @mujoco/mujoco@3.10.0 three
npm install -D vitest jsdom @types/three
```

Use the single-threaded import:

```ts
import loadMujoco from '@mujoco/mujoco';
```

Do not use `@mujoco/mujoco/mt` in the first implementation. The multi-threaded build requires cross-origin isolation headers; the single-threaded build is sufficient for a static demo.

### 5.2 Required project structure

The final source layout should be approximately:

```text
robotics-demo/
  package.json
  package-lock.json
  tsconfig.json
  vite.config.ts
  index.html
  public/
    model/
      native_scene.xml
      model_patch.json
      model_manifest.json
      assets/...
      LICENSES.md
    contract.json
    policy/
      active.json
      <policy-id>/manifest.json
      <policy-id>/weights.bin
  src/
    main.ts
    types.ts
    mujoco-runtime.ts
    model-loader.ts
    browser-environment.ts
    observation.ts
    policy.ts
    renderer.ts
    controls.ts
    telemetry.ts
    styles.css
  tools/
    prepare_model.py
    export_policy.py
    export_contract.py
    generate_reference_fixtures.py
  fixtures/
    python-reference/
  README.md
```

Vite is only a bundler and development preview tool. The production artifact is `/Users/cameron/robotics-demo/dist/`; there is no application server.

## 6. Offline build tools

### 6.1 `prepare_model.py`

Implement this before the browser environment so model preparation is reproducible. It must accept:

```text
--repo-root /Users/cameron/robotics
--source /Users/cameron/robotics/assets/mujoco_menagerie/franka_emika_panda/mjx_scene.xml
--output /Users/cameron/robotics-demo/public/model
--model-seed 0
--box-surface-friction 0.9
```

It must fail if any of these named objects are missing:

```text
joint1 ... joint7
finger_joint1
finger_joint2
actuator1 ... actuator8
push_point
hand_capsule
conveyor_surface
table_block
sorting_box_00 ... sorting_box_11
sorting_box_00_geom ... sorting_box_11_geom
sorting_box_00_freejoint ... sorting_box_11_freejoint
```

The manifest must expose enough information for TypeScript to avoid hard-coded numeric IDs where possible:

```json
{
  "format_version": 1,
  "mujoco_version": "...",
  "model_file": "native_scene.xml",
  "nq": 0,
  "nv": 0,
  "nu": 0,
  "timestep": 0.002,
  "controlled_joint_qpos_addresses": [],
  "controlled_joint_qvel_addresses": [],
  "locked_joint_qpos_addresses": [],
  "box_body_ids": [],
  "box_geom_ids": [],
  "box_qpos_addresses": [],
  "box_qvel_addresses": [],
  "push_site_id": 0,
  "hand_geom_id": 0,
  "conveyor_surface_geom_id": 0,
  "joint_low": [],
  "joint_high": [],
  "box_half_extents": [],
  "target_bins": []
}
```

### 6.2 `export_contract.py`

Generate `/Users/cameron/robotics-demo/public/contract.json` from Python constants in `rl/conveyor_spec.py` and the selected stage. Include:

- `OBSERVATION_SIZE = 129`
- `ACTION_SIZE = 6`
- `ACTION_DELTA_RADIANS = 0.05`
- `BOX_COUNT = 12`
- `TOP_BOX_COUNT = 4`
- `BOX_FEATURE_SIZE = 23`
- All `ObservationLayout` slices and `BOX_FEATURE_OFFSETS`
- `OBSERVATION_NORMALIZE_MASK`
- Geometry/bin/task constants
- Stage-3 reward weights
- Default environment settings

This generated contract is the source of truth for browser constants. Do not manually duplicate numeric values in TypeScript when they can be exported.

### 6.3 `export_policy.py`

The exporter must accept:

```sh
python /Users/cameron/robotics-demo/tools/export_policy.py \
  --repo-root /Users/cameron/robotics \
  --policy /Users/cameron/robotics/checkpoints/conveyor_native_ppo/20260723_111300_seed0/policy.msgpack \
  --output /Users/cameron/robotics-demo/public/policy/20260723_111300_seed0_step_214040576
```

The script must:

1. Resolve the policy file and adjacent `policy.json`.
2. Validate native schema 4 and observation version 2.
3. Validate observation size 129, action size 6, and hidden size 256.
4. Construct the same `ActorCritic(hidden_size=256)` template used by `run_native_policy.py`.
5. Load the policy bundle with `load_native_policy`.
6. Extract actor-only parameters in explicit, documented order.
7. Extract `ObservationStats.mean`, `variance`, and `count`.
8. Export the normalization mask and observation-validity behavior.
9. Write a little-endian `Float32Array`-compatible `weights.bin`.
10. Write `manifest.json` with array names, shapes, byte offsets, dtypes, policy metadata, and source checkpoint path relative to the repository root.
11. Write or update `/Users/cameron/robotics-demo/public/policy/active.json` to point to the selected policy directory.
12. Write a small `action-fixture.json` containing known observations and expected deterministic actions for browser tests.

The tool must accept an explicit `--repo-root` argument. It must not infer the source repository from `Path(__file__).resolve().parents[2]`, because the web project is a sibling directory. The source repository is `/Users/cameron/robotics`; all commands in this plan pass that path explicitly or use `../robotics` from the web project.

Do not export critic parameters, optimizer state, JAX RNG state, or the full training checkpoint.

The actor parameter order must be explicit, not inferred from dictionary iteration. Use the actual Flax parameter tree to identify the three actor dense kernels/biases and `actor_log_std`; deterministic inference does not need `actor_log_std`, but exporting it is acceptable for diagnostics.

## 7. Browser runtime modules

### 7.1 `mujoco-runtime.ts`

Responsibilities:

- Load the WASM module once.
- Expose initialized MuJoCo bindings.
- Load `native_scene.xml` and model assets using the binding's virtual filesystem/VFS mechanism.
- Create and destroy `MjModel`/`MjData`.
- Own the model/data lifecycle.
- Surface initialization errors in the UI.

Do not create a second model/data instance per animation frame. Explicitly call `.delete()` on Embind objects when the simulation is destroyed, following the official WASM README.

### 7.2 `model-loader.ts`

Responsibilities:

- Load `model_manifest.json`, `contract.json`, and `policy/active.json`.
- Load the prepared XML and assets.
- Apply `model_patch.json` before `mj_forward` if present.
- Validate `nq`, `nv`, `nu`, timestep, required named IDs, and box count.
- Return a model/data handle and the validated manifests.

If model validation fails, show a blocking error with the missing object or mismatched dimension. Never start the policy loop with a partially compatible model.

### 7.3 `policy.ts`

Implement a typed-array deterministic actor:

```text
raw observation (Float32Array[129])
  -> validity mask
  -> masked standardization using mean / sqrt(variance + 1e-6)
  -> clip standardized features to [-10, 10]
  -> leave non-normalized features unchanged
  -> zero invalid box features
  -> Dense(129, 256) + tanh
  -> Dense(256, 256) + tanh
  -> Dense(256, 6)
  -> tanh
  -> action Float32Array[6]
```

This must match `normalize_observation()` and `deterministic_action()` in [`rl/ppo.py`](/Users/cameron/robotics/rl/ppo.py). The action output is the normalized incremental joint action, not an absolute joint pose.

Add unit tests that run the exported `action-fixture.json` through the browser implementation and compare each action element to the Python expected value using an agreed tolerance such as `1e-5`.

### 7.4 `browser-environment.ts`

Implement stage 3 (`multi_box`) only for the first release.

Required internal state:

- `active[12]`
- `contacted[12]`
- `spawnedAt[12]`
- `targetBins[12]`, alternating class 0/1 as in Python
- `nextClass`
- `nextSpawnTime`
- `controlTargets[6]`
- `previousAction[6]`
- `episodeSteps`
- `failureCount`
- `episodeReturn`
- `successCount`, `wrongCount`, `missedCount`
- RNG streams for model/pose/belt/spawn behavior
- Current resolved task belt speed and spawn interval

Implement these methods:

```ts
reset(settings: BrowserEnvironmentSettings): Float32Array
observe(): Float32Array
step(action: Float32Array): BrowserStepResult
runPolicyStep(policy: Policy): BrowserStepResult
destroy(): void
```

The `BrowserStepResult` must include:

```ts
type BrowserStepResult = {
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
  missedReasons: Record<string, number>;
  unsafeReasons: Record<string, number>;
  activeBoxCount: number;
  episodeReturn: number;
};
```

Port the native step order exactly:

1. Validate the six finite action values.
2. Clip to `[-1, 1]`.
3. Update/clamp target joint positions by `action * 0.05` radians.
4. Capture pre-step active boxes, positions, and tool position.
5. Repeat up to 25 times:
   - Attempt a scheduled spawn.
   - Park inactive boxes and zero their velocity.
   - Force belt velocity for active boxes on the belt.
   - Write all eight actuator controls.
   - Call `mujoco.mj_step(model, data)`.
   - Zero settled lateral/angular box velocities.
   - Detect hand-box contacts.
   - Detect correct/wrong bin entries.
   - Detect unsafe motion and robot-conveyor collision.
   - Recycle exited, fallen, expired, nonfinite, excessive-speed, or externally removed boxes.
   - Stop early on instability or terminal box outcome where the native environment does so.
6. Compute reward signals and reward components.
7. Update counters and `previousAction`.
8. Build the next observation.
9. Auto-reset after terminal/truncated episode boundaries, matching native semantics.

The browser may keep the full event detail for the UI even if only aggregate reward is shown.

For reset randomization, preserve the selected policy's default `randomize_joint_poses=true` behavior. The browser must sample each controlled joint around the home pose using the configured ±0.2-radian noise, clamp to the model joint limits minus the 0.05-radian margin, reject invalid/contacting poses, and retry up to 1000 times. A browser-specific seeded PRNG is acceptable for the first release; document that browser seeds are deterministic within the browser implementation but are not required to reproduce NumPy's `SeedSequence` bit-for-bit.

### 7.5 `observation.ts`

Port `NativeConveyorEnv.observe()` without changing feature order. The output must be exactly 129 `float32` values:

```text
0:6       controlled joint positions
6:12      controlled joint velocities
12:18     target error
18:24     previous action
24:27     normalized tool position
27:33     tool rotation first two matrix columns
33:125    four selected box feature blocks, 23 values each
125:129  belt speed, spawn interval, failure ratio, episode progress
```

Use the generated `contract.json` for slices and constants. Match the Python formulas exactly, including:

- Joint position normalization from model joint limits.
- Joint velocity clipping after division by 3.
- Tool position normalization relative to `[0.6, 0.0, 0.42]` and scale `[0.75, 0.75, 0.50]`.
- Box selection by sortable urgency and stable descending sort.
- Quaternion sign normalization.
- Box velocity normalization.
- Goal positions from target class and box half-extents.
- Invalid box blocks zeroed after the `valid` mask.
- Global feature clipping.
- NaN/positive-infinity/negative-infinity replacement with zero.

Assert on every reset and policy step that the observation shape is 129.

### 7.6 `controls.ts` and `telemetry.ts`

`controls.ts` must validate UI inputs before applying them:

- Belt speed: finite and `> 0`.
- Spawn interval: finite and `> 0`.
- Noise values: finite and in `[0, 1)`.
- Seed: integer.
- Playback speed: finite and positive.

Apply setting changes on reset. Keep an `isDirty` indicator when form values differ from the active environment.

`telemetry.ts` must display:

- Active checkpoint ID and training environment steps.
- Stage 3 / `multi_box_slow`.
- Current belt speed and spawn interval.
- Simulation time, policy transitions, and physics steps.
- Reward and episode return.
- Correct, wrong, missed, first-contact, and active-box counts.
- Last missed/unsafe reason.
- Current model and policy compatibility status.

### 7.7 `renderer.ts`

Use the official WASM scene-update APIs and the Three.js approach shown by the MuJoCo WASM project.

Required visuals:

- Panda model and paddle.
- Conveyor/table.
- Green and blue target bins.
- Active boxes with class colors.
- Optional collision-geometry overlay.
- Temporary success indicators in the target bins.

The render loop must not control physics correctness. It should consume the latest simulation state while a separate accumulator schedules policy/physics work.

If complete mesh rendering blocks the first functional milestone, temporarily render a 2D schematic from MuJoCo state and keep the 3D renderer behind a separate milestone. Do not block policy, physics, controls, or parity work on cosmetic rendering.

## 8. Reference fixtures and parity testing

### 8.1 Generate fixtures from Python

Implement `/Users/cameron/robotics-demo/tools/generate_reference_fixtures.py`. It must use:

```text
platform: cpu
physics kernel: python
policy: selected run-level policy.msgpack
stage: 3
belt speed: 0.15
spawn interval: 0.75
spawn interval noise: 0.0 for deterministic fixture generation
belt speed noise: 0.0 for deterministic fixture generation
randomize joint poses: false for deterministic fixture generation
model seed: 0
environment index/seed: 0
```

Use the same policy/environment construction as `run_native_policy.py`, but record structured data instead of only printing totals.

Write `/Users/cameron/robotics-demo/fixtures/python-reference/stage3_seed0.json` containing:

```json
{
  "format_version": 1,
  "settings": {},
  "model": {
    "timestep": 0.002,
    "nq": 0,
    "nv": 0,
    "nu": 0,
    "joint_ranges": [],
    "box_half_extents": []
  },
  "initial": {
    "qpos": [],
    "qvel": [],
    "ctrl": [],
    "observation": [],
    "active": [],
    "target_bins": [],
    "control_targets": [],
    "previous_action": []
  },
  "transitions": [
    {
      "raw_observation": [],
      "action": [],
      "target_controls": [],
      "qpos": [],
      "qvel": [],
      "observation": [],
      "reward": 0,
      "physics_steps": 25,
      "info": {}
    }
  ]
}
```

Record at least 100 policy transitions, or stop after a terminal/reset event and record at least 20 transitions after the reset. Use deterministic noise-free settings in the fixture so differences are attributable to implementation, not random scheduling.

### 8.2 Required parity checks

Add tests that compare:

1. Model dimensions and timestep.
2. Joint ranges and controlled/locked addresses.
3. Initial `qpos`, `qvel`, and `ctrl` after reset.
4. Browser observation against Python observation.
5. Browser deterministic action against Python action.
6. Browser target controls against Python target controls.
7. Browser `qpos`/`qvel` after one policy transition against Python.
8. Physics event counts and reward for the first 100 transitions.

Use tight tolerances for pure policy math (`1e-5` absolute). Use explicit documented tolerances for physics (`1e-5` to `1e-4` for state values initially, adjusted only after identifying the source of drift). Require matching event classes/counts over the fixture window.

If physics diverges immediately, stop and diagnose model preparation, solver settings, friction, box mass/inertia, collision filters, XML assets, or control ordering before tuning tolerances.

### 8.3 Browser smoke test

Run the app in a supported desktop browser, initially Chrome on macOS/Linux:

1. Load the static build.
2. Confirm MuJoCo WASM initialization.
3. Confirm model manifest validation.
4. Confirm policy manifest validation.
5. Reset and render the stage-3 environment.
6. Run 100 policy transitions.
7. Change belt speed and spawn interval.
8. Reset and verify the new values appear in telemetry and global observation features.
9. Use pause, resume, single-step, and playback-speed controls.
10. Verify the browser Network panel shows only static asset requests and no API calls.
11. Destroy/reinitialize the environment and check for no repeated WASM object leaks.

## 9. Package scripts and commands

Add these scripts to `/Users/cameron/robotics-demo/package.json`:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "prepare:model": "python tools/prepare_model.py --repo-root ../robotics --source ../robotics/assets/mujoco_menagerie/franka_emika_panda/mjx_scene.xml --output public/model --model-seed 0 --box-surface-friction 0.9",
    "export:contract": "python tools/export_contract.py --repo-root ../robotics --output public/contract.json --stage 3",
    "test:reference": "python tools/generate_reference_fixtures.py --repo-root ../robotics --output fixtures/python-reference/stage3_seed0.json --policy ../robotics/checkpoints/conveyor_native_ppo/20260723_111300_seed0/policy.msgpack"
  }
}
```

The exact npm script syntax may be adjusted for the final project, but the named operations must remain available.

Run the following from the agent's working directory `/Users/cameron/robotics-demo`:

```sh
cd /Users/cameron/robotics-demo

# Install the existing Python runtime dependencies from the sibling source workspace if needed.
python3 -m pip install -r ../robotics/requirements-rl.txt

npm install
npm run prepare:model
npm run export:contract
python tools/export_policy.py \
  --repo-root ../robotics \
  --policy ../robotics/checkpoints/conveyor_native_ppo/20260723_111300_seed0/policy.msgpack \
  --output public/policy/20260723_111300_seed0_step_214040576
npm run test:reference
npm test
npm run build
npm run preview
```

The commands may require a Python virtual environment with the repository's MuJoCo/JAX/Flax dependencies. Do not add a web server dependency just to execute the model; `vite`/`vite preview` are only for development and static artifact preview.

## 10. Checkpoint update workflow

Document this exact workflow in `/Users/cameron/robotics-demo/README.md`:

1. Select a native schema-4 run-level `policy.msgpack`.
2. Confirm adjacent `policy.json` has the intended environment, stage, observation version, and action size.
3. Run `tools/export_policy.py` with an explicit output directory.
4. Run the policy math fixtures.
5. If model/stage settings changed, regenerate the model and contract artifacts.
6. Regenerate reference fixtures.
7. Run browser tests and production build.
8. Deploy only `/Users/cameron/robotics-demo/dist/`.

Version artifacts by checkpoint:

```text
robotics-demo/public/policy/
  active.json
  20260723_111300_seed0_step_214040576/
    manifest.json
    weights.bin
```

The raw checkpoint directory must not be copied into `/Users/cameron/robotics-demo/public/`.

## 11. Static hosting requirements

The production `dist/` directory must include:

- `index.html`
- Bundled JavaScript and CSS
- MuJoCo `.wasm` and loader assets
- Prepared `native_scene.xml`
- Every referenced mesh/texture asset
- `model_manifest.json`, `model_patch.json`, and `contract.json`
- Active policy manifest and `weights.bin`

The site must use relative URLs and work when hosted below a path prefix if configured through Vite's `base` option. Do not use absolute filesystem paths.

For local development, use `npm run dev` or `npm run preview`. Opening `index.html` directly with `file://` is not a required deployment mode because browser module/WASM loading commonly requires an HTTP origin. This does not mean the product has an application backend.

Start with the single-threaded MuJoCo WASM build. Only consider the multi-threaded build after measuring a real performance problem; it requires cross-origin isolation headers and may complicate static hosting.

## 12. Failure handling and UX requirements

The page must show a readable blocking error for:

- WASM initialization failure.
- Missing XML or mesh asset.
- Invalid model manifest.
- Missing or invalid policy artifact.
- Observation/action dimension mismatch.
- Unsupported browser API.
- Nonfinite simulation state.
- Environment reset failure.

While loading, disable run controls and show progress stages:

```text
Loading MuJoCo WASM
Loading model assets
Validating model
Loading policy
Validating policy
Ready
```

If the simulation becomes unstable, pause automatically, preserve the last telemetry, and offer Reset. Do not continue stepping a nonfinite MuJoCo state.

## 13. Acceptance criteria

The implementation is complete only when all of the following are true:

- `/Users/cameron/robotics-demo/` is a separate npm project outside the source repository.
- Running `cd /Users/cameron/robotics-demo && npm run build` succeeds and creates `/Users/cameron/robotics-demo/dist/`.
- The production site contains no Flask, FastAPI, Python process, application API, or server-side inference requirement.
- MuJoCo advances in the browser through the official WASM bindings.
- The prepared browser model includes the native collision, solver, friction, and box-randomization settings.
- The selected stage-3 policy loads from the converted artifact.
- Browser policy outputs match the Python fixture outputs within the documented tolerance.
- The browser constructs exactly 129 observation features in the documented order.
- Browser action application matches the native incremental six-joint target update.
- Belt speed, spawn interval, noise values, and seed are user-configurable and validated.
- Reset applies the selected settings and resets all task counters/state.
- Correct, wrong, missed, first-contact, reward, and active-box telemetry are displayed.
- At least 100 reference transitions pass model/action/environment parity checks.
- The browser smoke test makes no application API requests.
- MuJoCo/Embind objects are explicitly destroyed when the environment is disposed.
- The static `dist/` directory contains every runtime asset and can be deployed independently.

## 14. Out of scope for the first release

Do not add these until the stage-3 demo passes the acceptance criteria:

- PPO training in the browser.
- Browser-side checkpoint training-state restore.
- Automatic checkpoint discovery.
- Runtime checkpoint uploads.
- Multiple curriculum stages in one page.
- Multi-threaded MuJoCo WASM.
- Server-side inference.
- User-authored arbitrary MJCF models.
- Bit-for-bit NumPy RNG compatibility if fixed-state parity already passes.
