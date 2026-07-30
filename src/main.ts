import './styles.css';
import { createControls } from './controls';
import { ConveyorRenderer } from './renderer';
import { updateTelemetry } from './telemetry';
import type { BrowserEnvironmentSettings, BrowserStepResult, RendererState, SimulationWorkerResponse, TelemetrySnapshot } from './types';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <main class="shell">
    <section class="workspace">
      <section class="viewport-card"><div class="viewport-toolbar"><div class="scene-identity"><span class="status"><span class="status-dot"></span><span id="ready-state">Loading MuJoCo WASM</span></span></div><aside class="controls" aria-label="Evaluation settings">
        <label>Belt speed <output>%</output><input id="belt-speed" type="number" min="50" max="250" step="1" /></label>
        <label>Spawn speed <output>%</output><input id="spawn-speed" type="number" min="50" max="250" step="1" /></label>
        <label>Spawn noise <output>relative</output><input id="spawn-noise" type="number" min="0" max="0.5" step="0.01" /></label>
        <label>Random seed<input id="seed" type="number" step="1" /></label>
        <label>Playback speed <output>×</output><input id="playback-speed" type="number" min="0.1" step="0.1" /></label>
        <span id="settings-warning" class="warning" hidden>Click "Apply and Reset" to apply changes</span>
      </aside><div class="scene-actions"><aside class="playback-controls" aria-label="Playback controls"><button id="run">Run</button><button id="pause" class="quiet">Pause</button><button id="step" class="quiet">Single step</button><button id="reset" class="quiet">Apply &amp; reset</button></aside><button id="toggle-telemetry" class="toolbar-button" type="button" aria-controls="telemetry-panel" aria-expanded="false">Info</button></div></div><canvas id="scene" aria-label="Drag to orbit, right-drag to pan, and scroll to zoom."></canvas><div id="scene-loading" class="scene-loading"><div><span class="scene-spinner"></span><strong id="scene-loading-state">Preparing the MuJoCo scene</strong><img class="scene-loading-robot" src="./images/Roboarm.gif" alt="" aria-hidden="true" /><small>Loading native physics and Panda meshes</small></div></div><div id="blocking-error" class="blocking-error" hidden></div>
      <aside id="telemetry-panel" class="panel telemetry" hidden><div class="panel-heading"><h2>Info</h2><span class="badge">Stage 3</span></div><dl>
        <div><dt>Checkpoint</dt><dd id="checkpoint">—</dd></div><div><dt>Training steps</dt><dd id="training-steps">—</dd></div><div><dt>Simulation time</dt><dd id="simulation-time">—</dd></div><div><dt>Transitions</dt><dd id="policy-transitions">0</dd></div><div><dt>Physics steps</dt><dd id="physics-steps">0</dd></div><div><dt>Reward</dt><dd id="reward">—</dd></div><div><dt>Episode return</dt><dd id="episode-return">—</dd></div><div><dt>Correct / wrong / missed</dt><dd><span id="correct">0</span> / <span id="wrong">0</span> / <span id="missed">0</span></dd></div><div><dt>First contacts</dt><dd id="contacts">0</dd></div><div><dt>Active boxes</dt><dd id="active-boxes">0</dd></div><div><dt>Last missed reason</dt><dd id="missed-reason">none</dd></div><div><dt>Last unsafe reason</dt><dd id="unsafe-reason">none</dd></div>
      </dl></aside></section>
    </section>
  </main>`;

const status = app.querySelector<HTMLElement>('#ready-state')!;
const errorBox = app.querySelector<HTMLElement>('#blocking-error')!;
const loadingBox = app.querySelector<HTMLElement>('#scene-loading')!;
const loadingState = app.querySelector<HTMLElement>('#scene-loading-state')!;
const telemetryPanel = app.querySelector<HTMLElement>('#telemetry-panel')!;
const telemetryToggle = app.querySelector<HTMLButtonElement>('#toggle-telemetry')!;
const canvas = app.querySelector<HTMLCanvasElement>('#scene')!;
const setStatus = (value: string) => { status.textContent = value; };
const setSceneLoading = (value: string) => { loadingState.textContent = value; loadingBox.hidden = false; };
const hideSceneLoading = () => { loadingBox.hidden = true; };
const nextFrame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));
const showError = (error: unknown) => { hideSceneLoading(); errorBox.hidden = false; errorBox.textContent = error instanceof Error ? error.message : String(error); setStatus('Blocked by an initialization error'); };

telemetryToggle.addEventListener('click', () => {
  const hidden = telemetryPanel.toggleAttribute('hidden');
  telemetryToggle.textContent = hidden ? 'Info' : 'Hide Info';
  telemetryToggle.setAttribute('aria-expanded', String(!hidden));
});

function start(): void {
  const worker = new Worker(new URL('./simulation-worker.ts', import.meta.url), { type: 'module' });
  let renderer: ConveyorRenderer | undefined;
  let settings: BrowserEnvironmentSettings | undefined;
  let policyId = '';
  let environmentSteps = 0;
  let timestep = 0;
  let controls: ReturnType<typeof createControls> | undefined;
  let running = false;
  let busy = false;
  let transitions = 0;
  let physicsSteps = 0;
  let carry = 0;
  let lastTimestamp = performance.now();

  const publish = (result: BrowserStepResult, state: RendererState, readyState = 'Ready'): void => {
    renderer?.update(state, result.dropEvents);
    const snapshot: TelemetrySnapshot = { ...result, policyTransitions: transitions, totalPhysicsSteps: physicsSteps, checkpointId: policyId, environmentSteps, readyState };
    updateTelemetry(app, snapshot);
  };

  const execute = (): void => {
    if (busy) return;
    busy = true;
    worker.postMessage({ type: 'step' });
  };

  const tick = (timestamp: number): void => {
    const currentSettings = settings;
    const delta = Math.min(0.25, (timestamp - lastTimestamp) / 1000); lastTimestamp = timestamp;
    if (controls && currentSettings) controls.setDirty(currentSettings);
    if (running && currentSettings && !busy) {
      const period = timestep * currentSettings.actionRepeat;
      carry = Math.min(carry + delta * currentSettings.playbackSpeed, period * 4);
      if (carry >= period) { carry -= period; execute(); }
    }
    requestAnimationFrame(tick);
  };

  worker.onmessage = (event: MessageEvent<SimulationWorkerResponse>) => {
    const message = event.data;
    if (message.type === 'status') { setStatus(message.message); setSceneLoading(message.message); return; }
    if (message.type === 'error') { worker.terminate(); renderer?.dispose(); showError(new Error(message.message)); return; }
    if (message.type === 'ready') {
      settings = message.settings; policyId = message.policyId; environmentSteps = message.environmentSteps; timestep = message.timestep;
      renderer = new ConveyorRenderer(canvas, message.modelManifest, message.visualMeshes);
      controls = createControls(app, settings);
      controls.onRun(() => { running = true; controls?.setRunning(true); });
      controls.onPause(() => { running = false; controls?.setRunning(false); });
      controls.onStep(() => { running = false; controls?.setRunning(false); execute(); });
      controls.onReset((next) => { settings = next; running = false; busy = true; transitions = 0; physicsSteps = 0; carry = 0; controls?.setRunning(false); setSceneLoading('Resetting native physics state'); worker.postMessage({ type: 'reset', settings: next }); });
      publish(message.initial, message.state);
      void renderer.whenMeshSceneReady().then(() => { hideSceneLoading(); setStatus('Ready'); controls?.setRunning(false); });
      requestAnimationFrame(tick);
      return;
    }
    busy = false;
    if (message.type === 'step') {
      transitions += 1; physicsSteps += message.result.physicsSteps;
      publish(message.result, message.state);
    } else {
      publish(message.result, message.state); hideSceneLoading(); setStatus('Ready');
    }
  };

  setStatus('Starting MuJoCo WASM'); setSceneLoading('Starting MuJoCo WASM');
  void nextFrame().then(() => worker.postMessage({ type: 'initialize' }));
}

try { start(); } catch (error) { showError(error); }
