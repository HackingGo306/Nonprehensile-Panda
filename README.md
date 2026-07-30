# Static MuJoCo Conveyor Policy Demo

This project is a browser-only evaluation player for the stage-3 Panda conveyor environment. The deployed output is `dist/`: static HTML, JavaScript, CSS, MuJoCo WASM, prepared model assets, and converted policy weights. It does not need a Flask/FastAPI service, Python process, database, or inference API.

## Run locally

The sibling robotics source tree is used only to generate offline artifacts. Create a local build environment once, then run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r ../robotics/requirements-rl.txt
npm install

.venv/bin/python tools/prepare_model.py --repo-root ../robotics --source ../robotics/assets/mujoco_menagerie/franka_emika_panda/mjx_scene.xml --output public/model --model-seed 0 --box-surface-friction 0.9
.venv/bin/python tools/export_contract.py --repo-root ../robotics --output public/contract.json --stage 3
.venv/bin/python tools/export_policy.py --repo-root ../robotics --policy ../robotics/checkpoints/conveyor_native_ppo/20260723_111300_seed0/policy.msgpack --output public/policy/20260723_111300_seed0_policy_375062528 --stage 3 --demo-settings-from public/policy/20260723_111300_seed0_policy_320012288/manifest.json
.venv/bin/python tools/generate_reference_fixtures.py --repo-root ../robotics --output fixtures/python-reference/stage3_seed0.json --policy ../robotics/checkpoints/conveyor_native_ppo/20260723_111300_seed0/policy.msgpack --demo-settings-from public/policy/20260723_111300_seed0_policy_320012288/manifest.json

npm test
npm run build
npm run preview
```

Use `npm run dev` while developing. Open the site through an HTTP origin; opening `index.html` via `file://` is not supported because browser WASM/module loading requires HTTP.

## Policy and model contract

The selected run-level `policy.msgpack` and its adjacent `policy.json` are the source of truth. The active actor is the schema-4 native PPO policy at 375,062,528 environment steps, with observation version 2, 129 observation values, and six actions. Its source checkpoint is stage 4 `multi_box_fast`, but this demo deliberately runs it in the existing stage-3 `multi_box_slow` environment.

The browser starts with the saved resolved settings: 0.15 m/s belt speed, 0.75 s spawn interval, 0.20 belt noise, 0.15 spawn noise, model seed 0, 25 physics substeps, and randomized reset joint poses. The run-level `policy.msgpack` is the sole policy input; no Orbax training-state checkpoint is loaded or deployed.

The browser PRNG is repeatable within this implementation but intentionally does not reproduce NumPy `SeedSequence` bit-for-bit. The deterministic reference fixture disables both task noise and randomized reset poses, so it isolates model, observation, control, and policy parity.

The sibling run may continue training and overwrite its run-level `policy.msgpack`/`policy.json`. The deployed conversion is pinned to 375,062,528 steps. Its manifest records both the checkpoint’s source metadata and a separate `demo_environment` override, making the deliberate stage-4-actor/stage-3-environment pairing explicit. The browser applies the override only to environment defaults; the actor weights and normalization are from the selected run-level policy.

## Refreshing a checkpoint

1. Select a native schema-4 run-level `policy.msgpack` and verify its `policy.json` environment, stage, observation version, and action size.
2. Export it to a new `public/policy/<checkpoint-id>/` directory with `tools/export_policy.py`; the script updates `public/policy/active.json`. When replacing only the actor, pass `--demo-settings-from` with the current browser manifest to preserve its environment values.
3. Regenerate the model and contract whenever the selected stage or model settings change.
4. Regenerate the deterministic Python fixture, run `npm test`, then run `npm run build`.
5. Deploy only `dist/`. Do not copy a raw `policy.msgpack`, Orbax training-state checkpoint, optimizer state, or Python dependencies to `public/`.

## Available scripts

- `npm run prepare:model` prepares the browser model (with a Python environment that has MuJoCo available).
- `npm run export:contract` exports native constants and stage reward weights.
- `npm run export:policy` exports the active browser policy.
- `npm run test:reference` creates the deterministic Python-native fixture.
- `npm test` validates browser actor output against Python action fixtures.
- `npm run build` creates the deployable static site.

The app uses the single-threaded official `@mujoco/mujoco` build. It therefore works on normal static hosting without the cross-origin-isolation headers required by the multithreaded build.
# Nonprehensile-Panda
