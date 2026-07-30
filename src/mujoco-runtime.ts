import loadMujoco, { type MainModule, type MjData, type MjModel } from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import type { MujocoHandle } from './types';

let modulePromise: Promise<MainModule> | undefined;

export function getMujoco(): Promise<MainModule> {
  modulePromise ??= loadMujoco({ locateFile: (file: string) => (file.endsWith('.wasm') ? wasmUrl : file) }) as Promise<MainModule>;
  return modulePromise;
}

export function createHandle(mujoco: MainModule, model: MjModel, data: MjData): MujocoHandle {
  let disposed = false;
  return {
    mujoco, model, data,
    dispose() {
      if (disposed) return;
      disposed = true;
      data.delete();
      model.delete();
    },
  };
}
