import type { BrowserEnvironmentSettings } from './types';

export interface DemoControls {
  values(): BrowserEnvironmentSettings;
  setValues(settings: BrowserEnvironmentSettings): void;
  setRunning(running: boolean): void;
  onRun(listener: () => void): void;
  onPause(listener: () => void): void;
  onStep(listener: () => void): void;
  onReset(listener: (settings: BrowserEnvironmentSettings) => void): void;
  setDirty(active: BrowserEnvironmentSettings): void;
}

const input = (root: ParentNode, id: string) => root.querySelector<HTMLInputElement>(`#${id}`)!;
const number = (root: ParentNode, id: string) => Number(input(root, id).value);
const beltSpeedReference = 0.15;
const spawnIntervalReference = 1.5;
const percentage = (value: number, reference: number) => Number((100 * value / reference).toFixed(2));

type NumberConstraint = { min?: number; max?: number; step: number };
const numericConstraints: Record<string, NumberConstraint> = {
  'belt-speed': { min: 50, max: 250, step: 1 },
  'spawn-speed': { min: 50, max: 250, step: 1 },
  'spawn-noise': { min: 0, max: 0.99, step: 0.01 },
  seed: { step: 1 },
  'playback-speed': { min: 0.1, step: 0.1 },
};

const decimals = (value: number) => (String(value).split('.')[1] ?? '').length;

function snap(element: HTMLInputElement, constraint: NumberConstraint): number {
  const value = Number(element.value);
  if (!Number.isFinite(value)) throw new Error(`${element.labels?.[0]?.textContent?.trim() || 'Value'} must be a finite number.`);
  const nearest = Math.round(value / constraint.step) * constraint.step;
  const clamped = Math.min(constraint.max ?? Infinity, Math.max(constraint.min ?? -Infinity, nearest));
  const precision = Math.max(decimals(constraint.step), decimals(constraint.min ?? 0), decimals(constraint.max ?? 0));
  const result = Number(clamped.toFixed(precision));
  element.value = String(result);
  return result;
}

export function createControls(root: ParentNode, initial: BrowserEnvironmentSettings): DemoControls {
  const read = (): BrowserEnvironmentSettings => ({
    beltSpeed: beltSpeedReference * number(root, 'belt-speed') / 100, spawnInterval: spawnIntervalReference * 100 / number(root, 'spawn-speed'), beltSpeedNoise: initial.beltSpeedNoise, spawnIntervalNoise: number(root, 'spawn-noise'), seed: number(root, 'seed'), playbackSpeed: number(root, 'playback-speed'), randomizeJointPoses: initial.randomizeJointPoses,
    jointPoseNoise: initial.jointPoseNoise, jointLimitMargin: initial.jointLimitMargin, maxJointPoseResetAttempts: initial.maxJointPoseResetAttempts, maxFailures: initial.maxFailures, maxEpisodeSeconds: initial.maxEpisodeSeconds, actionRepeat: initial.actionRepeat, maxActiveBoxes: initial.maxActiveBoxes,
  });
  const validate = (): BrowserEnvironmentSettings => {
    for (const [id, constraint] of Object.entries(numericConstraints)) snap(input(root, id), constraint);
    const settings = read();
    if (![settings.beltSpeed, settings.spawnInterval, settings.playbackSpeed].every((value) => Number.isFinite(value) && value > 0)) throw new Error('Belt speed, spawn speed, and playback speed must be positive.');
    if (![settings.beltSpeedNoise, settings.spawnIntervalNoise].every((value) => Number.isFinite(value) && value >= 0 && value < 1)) throw new Error('Noise values must be in [0, 1).');
    if (!Number.isInteger(settings.seed)) throw new Error('Seed must be an integer.');
    return settings;
  };
  const setValues = (settings: BrowserEnvironmentSettings) => {
    input(root, 'belt-speed').value = String(percentage(settings.beltSpeed, beltSpeedReference)); input(root, 'spawn-speed').value = String(percentage(spawnIntervalReference, settings.spawnInterval)); input(root, 'spawn-noise').value = String(settings.spawnIntervalNoise); input(root, 'seed').value = String(settings.seed); input(root, 'playback-speed').value = String(settings.playbackSpeed);
  };
  setValues(initial);
  for (const [id, constraint] of Object.entries(numericConstraints)) {
    const element = input(root, id);
    const normalize = () => { if (element.value !== '' && Number.isFinite(Number(element.value))) snap(element, constraint); };
    element.addEventListener('change', normalize);
    element.addEventListener('blur', normalize);
  }
  return {
    values: validate, setValues,
    setRunning(running) { input(root, 'run').disabled = running; input(root, 'pause').disabled = !running; },
    onRun(listener) { input(root, 'run').addEventListener('click', listener); }, onPause(listener) { input(root, 'pause').addEventListener('click', listener); }, onStep(listener) { input(root, 'step').addEventListener('click', listener); },
    onReset(listener) { input(root, 'reset').addEventListener('click', () => listener(validate())); },
    setDirty(active) { const dirty = JSON.stringify(read()) !== JSON.stringify(active); root.querySelector('#settings-warning')!.toggleAttribute('hidden', !dirty); },
  };
}
