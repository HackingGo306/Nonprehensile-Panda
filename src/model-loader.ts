import type { MainModule } from '@mujoco/mujoco';
import { createHandle, getMujoco } from './mujoco-runtime';
import type { ActivePolicy, Contract, LoadedDemo, ModelManifest, ModelPatch } from './types';

const asset = (path: string) => `${import.meta.env.BASE_URL}${path}`;
const yieldToBrowser = (): Promise<void> => typeof window === 'undefined' ? Promise.resolve() : new Promise((resolve) => window.requestAnimationFrame(() => resolve()));

async function json<T>(path: string): Promise<T> {
  const response = await fetch(asset(path));
  if (!response.ok) throw new Error(`Unable to load ${path}: ${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

async function bytes(path: string): Promise<Uint8Array> {
  const response = await fetch(asset(path));
  if (!response.ok) throw new Error(`Unable to load ${path}: ${response.status} ${response.statusText}`);
  return new Uint8Array(await response.arrayBuffer());
}

function setValues(target: unknown, offset: number, values: number[]): void {
  const typed = target as { set(values: number[], offset?: number): void };
  typed.set(values, offset);
}

function applyPatch(mujoco: MainModule, model: LoadedDemo['model'], data: LoadedDemo['data'], patch: ModelPatch): void {
  for (const geom of patch.geoms) {
    if (geom.size) setValues(model.geom_size, geom.id * 3, geom.size);
    setValues(model.geom_friction, geom.id * 3, geom.friction);
    if (geom.rgba) setValues(model.geom_rgba, geom.id * 4, geom.rgba);
  }
  for (const body of patch.bodies) {
    (model.body_mass as Float64Array)[body.id] = body.mass;
    setValues(model.body_inertia, body.id * 3, body.inertia);
  }
  for (const pair of patch.pairs) setValues(model.pair_friction, pair.id * 5, pair.friction);
  if (patch.requires_mj_setConst) mujoco.mj_setConst(model, data);
  mujoco.mj_forward(model, data);
}

function validate(mujoco: MainModule, model: LoadedDemo['model'], manifest: ModelManifest): void {
  for (const [name, actual, expected] of [['nq', model.nq, manifest.nq], ['nv', model.nv, manifest.nv], ['nu', model.nu, manifest.nu]] as const) {
    if (actual !== expected) throw new Error(`Model ${name} is ${actual}; expected ${expected}.`);
  }
  if (Math.abs(model.opt.timestep - manifest.timestep) > 1e-12) throw new Error(`Model timestep is ${model.opt.timestep}; expected ${manifest.timestep}.`);
  const checks: Array<[number, string]> = [
    [mujoco.mjtObj.mjOBJ_SITE.value, 'push_point'], [mujoco.mjtObj.mjOBJ_GEOM.value, 'hand_capsule'],
    [mujoco.mjtObj.mjOBJ_GEOM.value, 'conveyor_surface'], [mujoco.mjtObj.mjOBJ_GEOM.value, 'table_block'],
    ...Array.from({ length: 6 }, (_, index) => [mujoco.mjtObj.mjOBJ_JOINT.value, `joint${index + 1}`] as [number, string]),
  ];
  for (const [objectType, name] of checks) if (mujoco.mj_name2id(model, objectType, name) < 0) throw new Error(`Required model object is missing: ${name}.`);
  if (manifest.box_body_ids.length !== 12 || manifest.box_qpos_addresses.length !== 12) throw new Error('The model manifest does not define the required twelve boxes.');
  const regions = manifest.sorting_bin_regions;
  if (regions.length !== 2 || new Set(regions.map((region) => region.target_bin)).size !== 2 || !regions.every((region) => [region.x_min, region.x_max, region.y_min, region.y_max].every(Number.isFinite) && region.x_min < region.x_max && region.y_min < region.y_max)) {
    throw new Error('The model manifest has invalid sorting-bin acceptance regions.');
  }
}

export async function loadDemo(): Promise<LoadedDemo> {
  const [mujoco, manifest, patch, contract, active] = await Promise.all([
    getMujoco(), json<ModelManifest>('model/model_manifest.json'), json<ModelPatch>('model/model_patch.json'), json<Contract>('contract.json'), json<ActivePolicy>('policy/active.json'),
  ]);
  const [modelBytes, assetContents] = await Promise.all([bytes(`model/${manifest.model_file}`), Promise.all(manifest.asset_files.map(async (file) => [file, await bytes(`model/${file}`)] as const))]);
  // Ensure the canvas loading state paints before synchronous WASM allocation.
  await yieldToBrowser();
  const vfs = new mujoco.MjVFS();
  let handle: LoadedDemo | undefined;
  try {
    vfs.addBuffer(manifest.model_file, modelBytes);
    for (let index = 0; index < assetContents.length; index += 1) {
      const [file, content] = assetContents[index];
      vfs.addBuffer(file, content);
      if ((index + 1) % 8 === 0) await yieldToBrowser();
    }
    await yieldToBrowser();
    const model = manifest.model_file.endsWith('.mjb') ? mujoco.MjModel.from_binary_path(manifest.model_file, vfs) : mujoco.MjModel.from_xml_string(modelBytes, vfs);
    const data = new mujoco.MjData(model);
    const base = createHandle(mujoco, model, data);
    applyPatch(mujoco, model, data, patch);
    validate(mujoco, model, manifest);
    handle = { ...base, modelManifest: manifest, modelPatch: patch, contract, activePolicy: active };
    return handle;
  } catch (error) {
    handle?.dispose();
    throw error;
  } finally {
    vfs.delete();
  }
}

export { asset };
