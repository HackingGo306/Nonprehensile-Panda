import type { TelemetrySnapshot } from './types';

const format = (value: number, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';

export function updateTelemetry(root: ParentNode, snapshot: TelemetrySnapshot): void {
  const set = (id: string, value: string) => { const node = root.querySelector(`#${id}`); if (node) node.textContent = value; };
  set('checkpoint', snapshot.checkpointId); set('training-steps', snapshot.environmentSteps.toLocaleString()); set('ready-state', snapshot.readyState); set('simulation-time', `${format(snapshot.simulationTime)} s`); set('policy-transitions', String(snapshot.policyTransitions)); set('physics-steps', String(snapshot.totalPhysicsSteps)); set('reward', format(snapshot.reward)); set('episode-return', format(snapshot.episodeReturn)); set('correct', String(snapshot.correctCount)); set('wrong', String(snapshot.wrongCount)); set('missed', String(snapshot.missedCount)); set('contacts', String(snapshot.firstContactCount)); set('active-boxes', String(snapshot.activeBoxCount));
  const reason = Object.entries(snapshot.missedReasons).find(([, value]) => value > 0)?.[0] ?? 'none'; const unsafe = Object.entries(snapshot.unsafeReasons).find(([, value]) => value > 0)?.[0] ?? 'none'; set('missed-reason', reason); set('unsafe-reason', unsafe);
}
