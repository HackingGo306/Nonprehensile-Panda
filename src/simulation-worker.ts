import { BrowserEnvironment, defaultSettings } from './browser-environment';
import { loadDemo } from './model-loader';
import { loadPolicy, type Policy } from './policy';
import type { BrowserEnvironmentSettings, BrowserStepResult, LoadedDemo, RendererState, SimulationWorkerRequest, SimulationWorkerResponse, VisualMeshAsset } from './types';

let demo: LoadedDemo | undefined;
let environment: BrowserEnvironment | undefined;
let policy: Policy | undefined;
const workerScope = globalThis as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<SimulationWorkerRequest>) => void) | null;
};

const post = (message: SimulationWorkerResponse, transfer: Transferable[] = []): void => workerScope.postMessage(message, transfer);
const status = (message: string): void => post({ type: 'status', message });

function initialResult(): BrowserStepResult {
  if (!environment || !demo) throw new Error('Simulation has not been initialized.');
  return {
    observation: environment.observe(), reward: 0, terminated: false, truncated: false, physicsSteps: 0, simulationTime: demo.data.time,
    correctCount: 0, wrongCount: 0, missedCount: 0, firstContactCount: 0, correctByClass: [0, 0], wrongByClass: [0, 0], missedByClass: [0, 0], dropEvents: [],
    missedReasons: {}, unsafeReasons: {}, activeBoxCount: 0, episodeReturn: 0, returnedEpisode: false,
  };
}

function rendererState(): RendererState {
  if (!demo) throw new Error('Simulation has not been initialized.');
  return {
    qpos: Float64Array.from(demo.data.qpos as Float64Array),
    siteXpos: Float64Array.from(demo.data.site_xpos as Float64Array),
    geomXpos: Float64Array.from(demo.data.geom_xpos as Float64Array),
    geomXmat: Float64Array.from(demo.data.geom_xmat as Float64Array),
  };
}

function visualMeshes(): VisualMeshAsset[] {
  if (!demo) throw new Error('Simulation has not been initialized.');
  const { model, mujoco } = demo;
  const meshType = mujoco.mjtGeom.mjGEOM_MESH.value;
  const assets: VisualMeshAsset[] = [];
  for (let geomId = 0; geomId < model.ngeom; geomId += 1) {
    if ((model.geom_type as Int32Array)[geomId] !== meshType || (model.geom_group as Int32Array)[geomId] !== 2) continue;
    const meshId = (model.geom_dataid as Int32Array)[geomId];
    const vertexStart = (model.mesh_vertadr as Int32Array)[meshId] * 3;
    const vertexCount = (model.mesh_vertnum as Int32Array)[meshId] * 3;
    const normalStart = (model.mesh_normaladr as Int32Array)[meshId] * 3;
    const normalCount = (model.mesh_normalnum as Int32Array)[meshId] * 3;
    const faceStart = (model.mesh_faceadr as Int32Array)[meshId] * 3;
    const faceCount = (model.mesh_facenum as Int32Array)[meshId] * 3;
    const materialId = (model.geom_matid as Int32Array)[geomId];
    const rgba = materialId >= 0 ? (model.mat_rgba as Float32Array).subarray(materialId * 4, materialId * 4 + 4) : [0.72, 0.75, 0.78, 1];
    assets.push({
      geomId,
      positions: Float32Array.from((model.mesh_vert as Float32Array).subarray(vertexStart, vertexStart + vertexCount)),
      normals: normalCount === vertexCount ? Float32Array.from((model.mesh_normal as Float32Array).subarray(normalStart, normalStart + normalCount)) : undefined,
      indices: Uint32Array.from((model.mesh_face as Int32Array).subarray(faceStart, faceStart + faceCount)),
      rgba: [rgba[0], rgba[1], rgba[2], rgba[3]],
    });
  }
  return assets;
}

function transferState(state: RendererState): Transferable[] {
  return [state.qpos.buffer, state.siteXpos.buffer, state.geomXpos.buffer, state.geomXmat.buffer];
}

function transferResult(result: BrowserStepResult, state: RendererState): Transferable[] {
  return [result.observation.buffer, ...transferState(state)];
}

async function initialize(): Promise<void> {
  status('Loading MuJoCo Scene');
  demo = await loadDemo();
  status('Loading deterministic PPO policy');
  policy = await loadPolicy(demo.activePolicy.manifest, demo.contract);
  status('Validating native physics state');
  const settings = defaultSettings(policy.manifest.policy_metadata, policy.manifest.demo_environment);
  environment = new BrowserEnvironment(demo, settings);
  environment.reset(settings);
  status('Preparing native Panda meshes');
  const state = rendererState();
  const meshes = visualMeshes();
  const initial = initialResult();
  const transfer = [...transferResult(initial, state), ...meshes.flatMap((mesh) => [mesh.positions.buffer, ...(mesh.normals ? [mesh.normals.buffer] : []), mesh.indices.buffer])];
  post({ type: 'ready', settings, initial, state, visualMeshes: meshes, modelManifest: demo.modelManifest, policyId: policy.manifest.policy_id, environmentSteps: policy.manifest.policy_metadata.environment_steps, timestep: demo.modelManifest.timestep }, transfer);
}

function step(): void {
  if (!environment || !policy) throw new Error('Simulation has not been initialized.');
  const result = environment.runPolicyStep(policy);
  const state = rendererState();
  post({ type: 'step', result, state }, transferResult(result, state));
}

function reset(settings: BrowserEnvironmentSettings): void {
  if (!environment) throw new Error('Simulation has not been initialized.');
  environment.reset(settings);
  const result = initialResult();
  const state = rendererState();
  post({ type: 'reset', result, state }, transferResult(result, state));
}

workerScope.onmessage = (event: MessageEvent<SimulationWorkerRequest>) => {
  void (async () => {
    try {
      if (event.data.type === 'initialize') await initialize();
      else if (event.data.type === 'step') step();
      else reset(event.data.settings);
    } catch (error) {
      post({ type: 'error', message: error instanceof Error ? error.message : String(error) });
    }
  })();
};
