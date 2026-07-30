import type { Contract, PolicyArray, PolicyManifest } from './types';
import { asset } from './model-loader';

export interface Policy {
  readonly manifest: PolicyManifest;
  act(rawObservation: Float32Array): Float32Array;
}

function arrayView(buffer: ArrayBuffer, descriptor: PolicyArray): Float32Array {
  if (descriptor.dtype !== 'float32-le' || descriptor.byte_offset % 4 !== 0 || descriptor.byte_length % 4 !== 0) throw new Error(`Unsupported weight array ${descriptor.name}.`);
  return new Float32Array(buffer, descriptor.byte_offset, descriptor.byte_length / 4);
}

function dense(input: Float32Array, kernel: Float32Array, bias: Float32Array, outputSize: number, applyTanh: boolean): Float32Array {
  const result = new Float32Array(outputSize);
  for (let column = 0; column < outputSize; column += 1) {
    let total = bias[column];
    for (let row = 0; row < input.length; row += 1) total += input[row] * kernel[row * outputSize + column];
    result[column] = Math.fround(applyTanh ? Math.tanh(total) : total);
  }
  return result;
}

class BrowserPolicy implements Policy {
  constructor(readonly manifest: PolicyManifest, private readonly contract: Contract, private readonly arrays: Map<string, Float32Array>) {}

  act(rawObservation: Float32Array): Float32Array {
    const expectedSize = Number(this.manifest.policy_metadata.observation_size);
    if (rawObservation.length !== expectedSize) throw new Error(`Policy expected ${expectedSize} observation values, got ${rawObservation.length}.`);
    const mean = this.arrays.get('observation_mean')!;
    const variance = this.arrays.get('observation_variance')!;
    const normalized = new Float32Array(rawObservation.length);
    const boxes = this.contract.observation_layout.BOXES;
    const boxSize = Number(this.contract.constants.BOX_FEATURE_SIZE);
    const validOffset = this.contract.box_feature_offsets.valid[0];
    for (let index = 0; index < rawObservation.length; index += 1) {
      let valid = true;
      if (index >= boxes[0] && index < boxes[1]) {
        const local = index - boxes[0];
        const block = Math.floor(local / boxSize);
        valid = rawObservation[boxes[0] + block * boxSize + validOffset] > 0.5;
      }
      if (!valid) normalized[index] = 0;
      else if (this.contract.observation_normalize_mask[index]) normalized[index] = Math.fround(Math.max(-10, Math.min(10, (rawObservation[index] - mean[index]) / Math.sqrt(variance[index] + 1e-6))));
      else normalized[index] = rawObservation[index];
    }
    const hidden1 = dense(normalized, this.arrays.get('Dense_0.kernel')!, this.arrays.get('Dense_0.bias')!, 256, true);
    const hidden2 = dense(hidden1, this.arrays.get('Dense_1.kernel')!, this.arrays.get('Dense_1.bias')!, 256, true);
    const meanAction = dense(hidden2, this.arrays.get('Dense_2.kernel')!, this.arrays.get('Dense_2.bias')!, 6, false);
    for (let index = 0; index < meanAction.length; index += 1) meanAction[index] = Math.fround(Math.tanh(meanAction[index]));
    return meanAction;
  }
}

export function createPolicy(manifest: PolicyManifest, contract: Contract, buffer: ArrayBuffer): Policy {
  if (manifest.policy_metadata.observation_size !== 129 || manifest.policy_metadata.action_size !== 6 || manifest.policy_metadata.hidden_size !== 256) throw new Error('Policy dimensions are incompatible with the browser demo.');
  const arrays = new Map(manifest.arrays.map((descriptor) => [descriptor.name, arrayView(buffer, descriptor)]));
  for (const name of ['observation_mean', 'observation_variance', 'Dense_0.kernel', 'Dense_0.bias', 'Dense_1.kernel', 'Dense_1.bias', 'Dense_2.kernel', 'Dense_2.bias']) if (!arrays.has(name)) throw new Error(`Policy artifact is missing ${name}.`);
  return new BrowserPolicy(manifest, contract, arrays);
}

export async function loadPolicy(policyManifestPath: string, contract: Contract): Promise<Policy> {
  const response = await fetch(asset(`policy/${policyManifestPath}`));
  if (!response.ok) throw new Error(`Unable to load active policy manifest: ${response.statusText}.`);
  const manifest = await response.json() as PolicyManifest;
  const weightResponse = await fetch(asset(`policy/${manifest.policy_id}/${manifest.weights_file}`));
  if (!weightResponse.ok) throw new Error(`Unable to load policy weights: ${weightResponse.statusText}.`);
  return createPolicy(manifest, contract, await weightResponse.arrayBuffer());
}
