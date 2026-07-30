import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import loadMujoco from '@mujoco/mujoco';
import { describe, expect, it } from 'vitest';
import { BrowserEnvironment, defaultSettings } from '../src/browser-environment';
import type { ActivePolicy, Contract, LoadedDemo, ModelManifest, ModelPatch, PolicyManifest } from '../src/types';

const root = new URL('../', import.meta.url);
const loadJson = async <T>(path: string) => JSON.parse(await readFile(new URL(path, root), 'utf8')) as T;

describe('prepared MuJoCo browser artifact', () => {
  it('loads through the official WASM VFS and applies its model patch', async () => {
    const [manifest, patch, contract, active, mujoco] = await Promise.all([
      loadJson<ModelManifest>('public/model/model_manifest.json'),
      loadJson<ModelPatch>('public/model/model_patch.json'),
      loadJson<Contract>('public/contract.json'),
      loadJson<ActivePolicy>('public/policy/active.json'),
      loadMujoco(),
    ]);
    const policyManifest = await loadJson<PolicyManifest>(`public/policy/${active.manifest}`);
    const modelBytes = new Uint8Array(await readFile(new URL(`public/model/${manifest.model_file}`, root)));
    const vfs = new mujoco.MjVFS();
    vfs.addBuffer(manifest.model_file, modelBytes);
    for (const file of manifest.asset_files) vfs.addBuffer(file, new Uint8Array(await readFile(join(new URL('public/model/', root).pathname, file))));
    const model = manifest.model_file.endsWith('.mjb') ? mujoco.MjModel.from_binary_path(manifest.model_file, vfs) : mujoco.MjModel.from_xml_string(modelBytes, vfs);
    const data = new mujoco.MjData(model);
    try {
      expect([model.nq, model.nv, model.nu]).toEqual([manifest.nq, manifest.nv, manifest.nu]);
      expect(model.opt.timestep).toBeCloseTo(manifest.timestep, 12);
      expect(manifest.sorting_bin_regions).toEqual([
        { target_bin: 0, x_min: 0.51, x_max: 1.66, y_min: 0.264, y_max: 0.544 },
        { target_bin: 1, x_min: 0.010000000000000009, x_max: 1.66, y_min: -0.544, y_max: -0.264 },
      ]);
      for (const geom of patch.geoms) {
        if (geom.size) (model.geom_size as Float64Array).set(geom.size, geom.id * 3);
        (model.geom_friction as Float64Array).set(geom.friction, geom.id * 3);
        if (geom.rgba) (model.geom_rgba as Float32Array).set(geom.rgba, geom.id * 4);
      }
      for (const body of patch.bodies) { (model.body_mass as Float64Array)[body.id] = body.mass; (model.body_inertia as Float64Array).set(body.inertia, body.id * 3); }
      for (const pair of patch.pairs) (model.pair_friction as Float64Array).set(pair.friction, pair.id * 5);
      mujoco.mj_setConst(model, data); mujoco.mj_forward(model, data);
      const first = patch.geoms[0];
      expect(Array.from((model.geom_size as Float64Array).slice(first.id * 3, first.id * 3 + 3))).toEqual(first.size);
      expect(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, 'push_point')).toBe(manifest.push_site_id);
      expect(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, 'hand_capsule')).toBe(manifest.hand_geom_id);
      const visualMeshes = Array.from({ length: model.ngeom }, (_, geomId) => geomId).filter((geomId) => (model.geom_type as Int32Array)[geomId] === mujoco.mjtGeom.mjGEOM_MESH.value && (model.geom_group as Int32Array)[geomId] === 2);
      expect(visualMeshes.length).toBeGreaterThan(50);
      expect(visualMeshes.every((geomId) => (model.mesh_vertnum as Int32Array)[(model.geom_dataid as Int32Array)[geomId]] > 0)).toBe(true);
      const demo = { mujoco, model, data, modelManifest: manifest, modelPatch: patch, contract, activePolicy: active } as LoadedDemo;
      const environment = new BrowserEnvironment(demo, defaultSettings(policyManifest.policy_metadata, policyManifest.demo_environment));
      expect(environment.reset()).toHaveLength(129);
      expect(environment.step(new Float32Array(6)).terminated).toBe(false);
      environment.destroy();
    } finally { data.delete(); model.delete(); vfs.delete(); }
  }, 30_000);
});
