'use client';

/**
 * SocraticStream — renders the AgentThinking / AgentPlanning / AgentAsking /
 * AgentGarnishing / AgentExecuting event kinds as colour-coded "live thought"
 * cards.
 *
 * Blueprint §3.1 dictates the visual taxonomy:
 *
 *   thinking   purple — "I am currently…" reasoning trace
 *   planning   blue   — step-by-step plan card
 *   asking     yellow — Socratic question pause
 *   garnishing cyan   — UI scaffolding being applied
 *   executing  green  — live execution log tail
 *
 * Each card pops in with a spring animation and the freshest entry pulses
 * once to draw the eye.
 */
import { useMemo } from 'react';
import type { AgentEvent } from '@/lib/types';
import { cn } from '@/lib/cn';
import { Brain, ListChecks, HelpCircle, Sparkles, Play, AlertOctagon, Eye, ShieldCheck } from 'lucide-react';

const STREAM_KINDS = new Set<string>([
  'AgentThinking', 'AgentPlanning', 'AgentAsking', 'AgentGarnishing', 'AgentExecuting',
  'AuditCritique', 'AuditOverride',
  'CircuitBreakerTripped',
  'SandboxBenchmarkStarted', 'SandboxBenchmarkCompleted',
  'DataHealthReport',
]);

type Tone = 'thinking' | 'planning' | 'asking' | 'garnishing' | 'executing' | 'critic' | 'sandbox' | 'breaker' | 'health';

const TONE_STYLE: Record<Tone, { ring: string; bar: string; text: string; chip: string; icon: React.ComponentType<{ className?: string }>; label: string }> = {
  thinking:   { ring: 'border-thinking/30',   bar: 'bg-thinking',   text: 'text-thinking',   chip: 'bg-thinking/15 text-thinking',     icon: Brain,        label: 'thinking' },
  planning:   { ring: 'border-planning/30',   bar: 'bg-planning',   text: 'text-planning',   chip: 'bg-planning/15 text-planning',     icon: ListChecks,   label: 'planning' },
  asking:     { ring: 'border-asking/30',     bar: 'bg-asking',     text: 'text-asking',     chip: 'bg-asking/15 text-asking',         icon: HelpCircle,   label: 'asking' },
  garnishing: { ring: 'border-garnishing/30', bar: 'bg-garnishing', text: 'text-garnishing', chip: 'bg-garnishing/15 text-garnishing', icon: Sparkles,     label: 'garnishing' },
  executing:  { ring: 'border-executing/30',  bar: 'bg-executing',  text: 'text-executing',  chip: 'bg-executing/15 text-executing',   icon: Play,         label: 'executing' },
  critic:     { ring: 'border-warn/40',       bar: 'bg-warn',       text: 'text-warn',       chip: 'bg-warn/15 text-warn',             icon: ShieldCheck,  label: 'audit critic' },
  sandbox:    { ring: 'border-success/40',    bar: 'bg-success',    text: 'text-success',    chip: 'bg-success/15 text-success',       icon: Eye,          label: 'sandbox' },
  breaker:    { ring: 'border-danger/40',     bar: 'bg-danger',     text: 'text-danger',     chip: 'bg-danger/15 text-danger',         icon: AlertOctagon, label: 'circuit breaker' },
  health:     { ring: 'border-info/40',       bar: 'bg-info',       text: 'text-info',       chip: 'bg-info/15 text-info',             icon: ShieldCheck,  label: 'data health' },
};

function classify(e: AgentEvent): Tone | null {
  switch (e.kind) {
    case 'AgentThinking':   return 'thinking';
    case 'AgentPlanning':   return 'planning';
    case 'AgentAsking':     return 'asking';
    case 'AgentGarnishing': return 'garnishing';
    case 'AgentExecuting':  return 'executing';
    case 'AuditCritique':
    case 'AuditOverride':   return 'critic';
    case 'SandboxBenchmarkStarted':
    case 'SandboxBenchmarkCompleted': return 'sandbox';
    case 'CircuitBreakerTripped': return 'breaker';
    case 'DataHealthReport': return 'health';
    default: return null;
  }
}

export function SocraticStream({ events }: { events: AgentEvent[] }) {
  const stream = useMemo(() => events.filter((e) => STREAM_KINDS.has(e.kind)), [events]);
  if (stream.length === 0) return null;

  // Most recent at the top of the stream — feels alive.
  const latestId = stream[stream.length - 1]?.id;

  return (
    <div className="space-y-2">
      {stream.map((e) => {
        const tone = classify(e);
        if (!tone) return null;
        const style = TONE_STYLE[tone];
        const Icon = style.icon;
        const isLatest = e.id === latestId;
        return (
          <div
            key={e.id}
            className={cn(
              'relative rounded-md border bg-white/[0.025] backdrop-blur px-3.5 py-2.5 animate-pop-in',
              style.ring,
              isLatest && 'animate-glow',
            )}
          >
            <div className={cn('absolute left-0 top-0 bottom-0 w-[2px] rounded-l', style.bar)} />
            <div className="flex items-start gap-3 pl-2">
              <Icon className={cn('w-3.5 h-3.5 mt-0.5 shrink-0', style.text)} />
              <div className="min-w-0 flex-1 space-y-1.5">
                <div className="flex items-center gap-2">
                  <span className={cn('px-1.5 py-0.5 rounded text-[8.5px] font-black uppercase tracking-[0.18em]', style.chip)}>
                    {style.label}
                  </span>
                  <span className="text-[9px] uppercase tracking-widest text-white/30 font-black">
                    {String(e.actor)}
                  </span>
                  <span className="ml-auto text-[9px] font-mono text-white/30">
                    {new Date(e.created_at).toLocaleTimeString()}
                  </span>
                </div>
                <Body event={e} tone={tone} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Body({ event, tone }: { event: AgentEvent; tone: Tone }) {
  const p = event.payload as Record<string, unknown>;
  if (event.kind === 'AgentPlanning') {
    const steps = (p.steps as string[]) || [];
    return (
      <div className="space-y-1">
        {p.title ? (
          <p className="text-[12px] text-white/85 font-bold">{String(p.title)}</p>
        ) : null}
        <ol className="space-y-1 pl-4 list-decimal text-[11.5px] text-white/65 marker:text-white/30">
          {steps.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ol>
      </div>
    );
  }
  if (event.kind === 'AgentGarnishing') {
    const node = (p.node as Record<string, unknown>) || {};
    return (
      <p className="text-[12px] text-white/80 leading-relaxed">
        Materializing node{' '}
        <code className="px-1 py-[1px] rounded bg-white/5 text-[11px] font-mono text-white/90">
          {String(node.id ?? node.type ?? 'node')}
        </code>{' '}
        on the canvas.
      </p>
    );
  }
  if (event.kind === 'AgentAsking') {
    return (
      <p className="text-[12.5px] text-white/90 leading-relaxed font-medium">
        {String(p.question ?? '')}
      </p>
    );
  }
  if (event.kind === 'DataHealthReport') {
    const r = (p.report as Record<string, unknown>) ?? {};
    return (
      <div className="space-y-1">
        <p className="text-[12px] text-white/85 font-bold uppercase tracking-widest">
          Verdict: {String(r.verdict ?? 'unknown')}
        </p>
        <p className="text-[11.5px] text-white/65">{String(r.summary ?? '')}</p>
      </div>
    );
  }
  if (event.kind === 'AuditCritique' || event.kind === 'AuditOverride') {
    return (
      <div className="space-y-1">
        <p className="text-[12px] text-white/90">{String(p.summary ?? '')}</p>
        {p.advice ? (
          <p className="text-[11px] text-white/55 italic leading-relaxed">{String(p.advice)}</p>
        ) : null}
      </div>
    );
  }
  if (event.kind === 'CircuitBreakerTripped') {
    return (
      <p className="text-[12px] text-white/90">
        Loop guard tripped on <code className="font-mono text-[11px] text-white/70">{String(p.kind ?? '')}</code> —{' '}
        {String(p.advice ?? 'cooling off downstream handlers')}.
      </p>
    );
  }
  if (event.kind === 'SandboxBenchmarkCompleted') {
    const benchmarks = (p.benchmarks as Array<{ name: string; score: number }>) || [];
    return (
      <div className="grid grid-cols-2 gap-1.5">
        {benchmarks.map((b) => (
          <div key={b.name} className="flex items-center justify-between text-[11px] text-white/75 font-mono">
            <span className="text-white/45 uppercase tracking-widest text-[9px] font-black">{b.name}</span>
            <span>{(b.score * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    );
  }
  // thinking + executing fall through
  return (
    <p className={cn('text-[12px] leading-relaxed', tone === 'thinking' ? 'text-white/70 italic' : 'text-white/85')}>
      {String(p.text ?? '')}
    </p>
  );
}
