'use client';

/**
 * Training telemetry view (R16): live loss sparkline + step/epoch/lr readouts
 * + phase banner + anomaly log — all derived from train.* events at zero
 * token cost (the watcher streams them; the LLM stays suspended).
 */
import { useMemo } from 'react';
import { cn } from '@/lib/cn';
import type { AgentEvent } from '@/lib/events';
import { Activity, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';

interface Metric {
  step: number;
  epoch?: number;
  loss: number;
  lr?: number;
}

function deriveTraining(events: AgentEvent[]) {
  const metrics: Metric[] = [];
  const anomalies: { kind: string; evidence: string }[] = [];
  let phase: string | null = null;
  for (const e of events) {
    const p = e.payload as Record<string, unknown>;
    if (e.kind === 'train.metric') {
      const loss = Number(p.loss);
      if (Number.isFinite(loss)) {
        metrics.push({ step: Number(p.step ?? metrics.length), epoch: p.epoch as number | undefined, loss, lr: p.lr as number | undefined });
      }
    } else if (e.kind === 'train.anomaly') {
      anomalies.push({ kind: String(p.anomalyKind ?? '?'), evidence: String(p.evidence ?? '').slice(0, 300) });
    } else if (e.kind === 'train.phase') {
      phase = String(p.phase ?? null);
    }
  }
  return { metrics, anomalies, phase };
}

function Sparkline({ metrics }: { metrics: Metric[] }) {
  // Downsample to ≤240 points for a crisp polyline.
  const points = useMemo(() => {
    if (metrics.length < 2) return '';
    const stride = Math.max(1, Math.floor(metrics.length / 240));
    const sampled = metrics.filter((_, i) => i % stride === 0 || i === metrics.length - 1);
    const losses = sampled.map((m) => m.loss);
    const min = Math.min(...losses);
    const max = Math.max(...losses);
    const range = max - min || 1;
    const W = 640;
    const H = 180;
    return sampled
      .map((m, i) => `${((i / (sampled.length - 1)) * W).toFixed(1)},${(H - ((m.loss - min) / range) * (H - 12) - 6).toFixed(1)}`)
      .join(' ');
  }, [metrics]);

  if (!points) return null;
  return (
    <svg viewBox="0 0 640 180" className="w-full h-44">
      <polyline points={points} fill="none" stroke="#EAB308" strokeWidth="1.75" strokeLinejoin="round" />
    </svg>
  );
}

const PHASE_TONE: Record<string, string> = {
  launch: 'text-info',
  running: 'text-amber-400',
  finished: 'text-success',
  failed: 'text-danger',
};

export function TrainingCanvas({ events }: { events: AgentEvent[] }) {
  const { metrics, anomalies, phase } = useMemo(() => deriveTraining(events), [events]);
  const last = metrics[metrics.length - 1];
  const best = metrics.length ? Math.min(...metrics.map((m) => m.loss)) : null;

  if (!phase && metrics.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-[11px] uppercase tracking-widest text-white/30 font-black">
        Training telemetry appears here once a run starts
      </div>
    );
  }

  const PhaseIcon = phase === 'finished' ? CheckCircle2 : phase === 'failed' ? XCircle : Activity;

  return (
    <div className="h-full overflow-y-auto p-6 space-y-5" data-lenis-prevent="true">
      <div className="flex items-center gap-3">
        <PhaseIcon className={cn('w-4 h-4', PHASE_TONE[phase ?? 'running'] ?? 'text-white/40')} />
        <span className={cn('text-[11px] uppercase tracking-[0.25em] font-black', PHASE_TONE[phase ?? 'running'] ?? 'text-white/40')}>
          {phase ?? 'running'}
        </span>
        <span className="ml-auto text-[10px] font-mono text-white/40">{metrics.length} metric points</span>
      </div>

      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'step', value: last ? String(last.step) : '—' },
          { label: 'loss', value: last ? last.loss.toFixed(4) : '—' },
          { label: 'best loss', value: best !== null ? best.toFixed(4) : '—' },
          { label: last?.epoch !== undefined ? 'epoch' : 'lr', value: last?.epoch !== undefined ? last.epoch.toFixed(2) : last?.lr ? last.lr.toExponential(1) : '—' },
        ].map((s) => (
          <div key={s.label} className="rounded-lg border border-white/10 bg-white/[0.02] px-4 py-3">
            <div className="text-[9px] uppercase tracking-widest font-black text-white/35">{s.label}</div>
            <div className="text-[18px] font-mono text-white mt-0.5">{s.value}</div>
          </div>
        ))}
      </div>

      {metrics.length > 1 && (
        <div className="rounded-lg border border-white/10 bg-white/[0.02] p-4">
          <div className="text-[9px] uppercase tracking-widest font-black text-white/35 mb-2">loss</div>
          <Sparkline metrics={metrics} />
        </div>
      )}

      {anomalies.length > 0 && (
        <div className="space-y-2">
          {anomalies.map((a, i) => (
            <div key={i} className="rounded-lg border border-danger/30 bg-danger/5 px-4 py-3 flex gap-3">
              <AlertTriangle className="w-4 h-4 text-danger shrink-0 mt-0.5" />
              <div className="min-w-0">
                <div className="text-[11px] font-bold text-danger uppercase tracking-widest">{a.kind}</div>
                <pre className="text-[10px] text-white/50 whitespace-pre-wrap break-words font-mono mt-1">{a.evidence}</pre>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
