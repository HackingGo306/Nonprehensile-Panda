import { readFile } from 'node:fs/promises';
import { describe, expect, it } from 'vitest';
import { createPolicy } from '../src/policy';
import type { ActivePolicy, Contract, PolicyManifest } from '../src/types';

const root = new URL('../', import.meta.url);
const loadJson = async <T>(path: string) => JSON.parse(await readFile(new URL(path, root), 'utf8')) as T;

describe('exported deterministic PPO actor', () => {
  it('matches Python action fixtures within 1e-5', async () => {
    const contract = await loadJson<Contract>('public/contract.json');
    const active = await loadJson<ActivePolicy>('public/policy/active.json');
    const directory = active.manifest.replace(/\/manifest\.json$/, '');
    const manifest = await loadJson<PolicyManifest>(`public/policy/${active.manifest}`);
    const fixture = await loadJson<{ observations: number[][]; expected_actions: number[][] }>(`public/policy/${directory}/action-fixture.json`);
    const file = await readFile(new URL(`public/policy/${directory}/weights.bin`, root));
    const policy = createPolicy(manifest, contract, file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
    fixture.observations.forEach((observation, index) => {
      const actual = policy.act(Float32Array.from(observation));
      actual.forEach((value, actionIndex) => expect(value).toBeCloseTo(fixture.expected_actions[index][actionIndex], 5));
    });
  });
});
