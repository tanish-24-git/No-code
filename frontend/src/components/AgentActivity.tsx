'use client';

/**
 * Live agent console. Subscribes to /api/sessions/{id}/events and renders
 * each event as a typed card. Designed to feel like a Stitch / n8n run-log,
 * not a chat transcript: most events are structured.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import useSWR from 'swr';
import { api, fetcher } from '@/lib/api';
import { useSessionEvents } from '@/lib/sse';
import { cn } from '@/lib/cn';
import type { AgentEvent, AgentSession, EventKind } from '@/lib/types';
import {
  Activity,
  AlertTriangle,
  Bot,
  Brain,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Database,
  FileText,
  GitBranch,
  HelpCircle,
  Loader2,
  PackageCheck,
  Play,
  Send,
  Sparkles,
  User,
  XCircle,
  Zap,
} from 'lucide-react';

type Props = { sessionId: string | null };

// ── Event grouping ────────────────────────────────────────────────────────
//
// Some events are "story beats" (rendered as cards). Others are noise we
// fold into a collapsible run log (tool calls, decision records, intake
// started, etc.). The grouping is purely cosmetic.

const STORY_KINDS: ReadonlySet<EventKind> = new Set([
  'AssistantMessage',
  'UserMessage',
  'UserClarificationRequested',
  'UserClarificationReceived',
  'PipelineDraftCreated',
  'PipelineApprovalRequested',
  'PipelineApproved',
  'PipelineRejected',
  'PipelineExecutionStarted',
  'TrainingMetricUpdated',
  'TrainingAnomalyDetected',
  'RecoveryPlanGenerated',
  'RetryRequested',
  'RetryApproved',
  'RetryDenied',
  'TrainingCompleted',
  'EvaluationCompleted',
  'ExportChoiceRequested',
  'ExportCompleted',
  'SessionClosed',
  'Error',
]);

const STAGE_LABEL: Record<string, string> = {
  init: 'Initializing',
  profiling: 'Profiling Dataset',
  clarifying: 'Awaiting Clarification',
  planning: 'Drafting Pipeline',
  awaiting_approval: 'Awaiting Your Approval',
  executing: 'Starting Training',
  monitoring: 'Training Live',
  recovering: 'Recovering',
  evaluating: 'Evaluating',
  awaiting_export_choice: 'Awaiting Export Decision',
  finalizing: 'Finalizing',
  done: 'Complete',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

export function AgentActivity({ sessionId }: Props) {
  const { events, connected, closed } = useSessionEvents(sessionId);
  const { data: session, mutate: mutateSession } = useSWR<AgentSession>(
    sessionId ? `/api/sessions/${sessionId}` : null,
    fetcher,
    { refreshInterval: 1500 },
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  const [showRunLog, setShowRunLog] = useState(false);
  const [busy, setBusy] = useState(false);

  // Refresh session detail when meaningful events arrive.
  useEffect(() => {
    if (events.length === 0) return;
    const last = events[events.length - 1];
    if ([
      'TaskInferred',
      'UserClarificationRequested',
      'UserClarificationReceived',
      'PipelineDraftCreated',
      'PipelineApprovalRequested',
      'PipelineApproved',
      'TrainingCompleted',
      'EvaluationCompleted',
      'ExportChoiceRequested',
      'ExportCompleted',
      'SessionClosed',
      'Error',
    ].includes(last.kind)) {
      mutateSession();
    }
  }, [events, mutateSession]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [events.length]);

  const story = useMemo(() => events.filter((e) => STORY_KINDS.has(e.kind)), [events]);
  const noise = useMemo(() => events.filter((e) => !STORY_KINDS.has(e.kind)), [events]);

  if (!sessionId) {
    return (
      <div className="flex-1 flex items-center justify-center p-12">
        <div className="text-center space-y-3">
          <Bot className="w-10 h-10 text-white/20 mx-auto" />
          <p className="text-[10px] uppercase tracking-[0.3em] text-white/30 font-black">
            Upload a dataset to start a session
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-black">
      {/* Stage header */}
      <StageHeader session={session} connected={connected} closed={closed} />

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-4 space-y-3 scrollbar-thin">
        {story.length === 0 && (
          <div className="flex items-center gap-3 text-[11px] text-white/40">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Booting agents…
          </div>
        )}
        {story.map((e) => (
          <EventCard
            key={e.id}
            event={e}
            session={session}
            sessionId={sessionId}
            onAction={() => mutateSession()}
            busy={busy}
            setBusy={setBusy}
          />
        ))}
      </div>

      {/* Collapsible run log of internal events */}
      {noise.length > 0 && (
        <div className="border-t border-white/5">
          <button
            onClick={() => setShowRunLog((s) => !s)}
            className="w-full px-5 py-2.5 flex items-center gap-2 text-[9px] font-black uppercase tracking-[0.25em] text-white/30 hover:text-white/60 transition-colors"
          >
            <ChevronRight className={cn('w-3 h-3 transition-transform', showRunLog && 'rotate-90')} />
            Run log · {noise.length} internal events
          </button>
          {showRunLog && (
            <div className="max-h-48 overflow-y-auto px-5 pb-4 space-y-1 font-mono text-[10px] text-white/40">
              {noise.map((e) => (
                <div key={e.id} className="flex items-center gap-2">
                  <span className="text-white/20">{new Date(e.created_at).toLocaleTimeString()}</span>
                  <span className="text-white/60">{e.kind}</span>
                  <span className="text-white/30">‹{e.actor}›</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Bottom message bar (free-form messages → /messages endpoint) */}
      <MessageBar sessionId={sessionId} disabled={closed} />
    </div>
  );
}

function StageHeader({
  session,
  connected,
  closed,
}: {
  session?: AgentSession;
  connected: boolean;
  closed: boolean;
}) {
  const stage = session ? STAGE_LABEL[session.state] ?? session.state : 'Connecting…';
  const conf = session ? Math.round(session.confidence * 100) : 0;
  const live = !closed && connected;

  return (
    <div className="px-5 py-3 border-b border-white/5 bg-[#070707] flex items-center gap-4">
      <div className="flex items-center gap-2">
        <span
          className={cn(
            'w-1.5 h-1.5 rounded-full',
            live ? 'bg-success shadow-[0_0_10px_rgba(16,185,129,0.6)] animate-pulse' :
            closed ? 'bg-white/40' : 'bg-warn shadow-[0_0_10px_rgba(245,158,11,0.6)]'
          )}
        />
        <span className="text-[9px] uppercase tracking-[0.25em] font-black text-white/40">
          {live ? 'Live' : closed ? 'Closed' : 'Reconnecting'}
        </span>
      </div>
      <div className="flex-1 flex items-center gap-3">
        <Sparkles className="w-3.5 h-3.5 text-white/50" />
        <span className="text-[11px] font-bold uppercase tracking-[0.18em] text-white">
          {stage}
        </span>
      </div>
      {session && (
        <div className="flex items-center gap-1.5 text-[9px] uppercase tracking-widest text-white/40 font-black">
          <Brain className="w-3 h-3" />
          <span>conf {conf}%</span>
        </div>
      )}
    </div>
  );
}

// ── Cards ─────────────────────────────────────────────────────────────────

type CardProps = {
  event: AgentEvent;
  session?: AgentSession;
  sessionId: string;
  onAction: () => void;
  busy: boolean;
  setBusy: (v: boolean) => void;
};

function EventCard(props: CardProps) {
  const { event } = props;
  switch (event.kind) {
    case 'AssistantMessage':
      return <BubbleCard text={String(event.payload.text ?? '')} actor="agent" />;
    case 'UserMessage':
      return <BubbleCard text={String(event.payload.text ?? '')} actor="user" />;
    case 'UserClarificationRequested':
      return <ClarificationCard {...props} />;
    case 'UserClarificationReceived':
      return (
        <BubbleCard
          text={`Answer received: ${stringifyValue((event.payload.answer as { value?: unknown })?.value)}`}
          actor="user"
        />
      );
    case 'PipelineDraftCreated':
      return <PipelineDraftCard {...props} />;
    case 'PipelineApprovalRequested':
      return <ApprovalRequiredCard {...props} />;
    case 'PipelineApproved':
      return <StatusCard icon={<CheckCircle2 className="w-4 h-4" />} tone="success" title="Pipeline approved" subtitle={event.payload.auto ? 'auto-approved' : (event.rationale ?? 'user approved')} />;
    case 'PipelineRejected':
      return <StatusCard icon={<XCircle className="w-4 h-4" />} tone="danger" title="Pipeline rejected" subtitle={String(event.payload.reason ?? '')} />;
    case 'PipelineExecutionStarted':
      return <StatusCard icon={<Play className="w-4 h-4" />} tone="info" title="Training started" subtitle={`job ${(event.payload.job_id as string ?? '').slice(0, 8)}`} />;
    case 'TrainingMetricUpdated':
      return <MetricRow event={event} />;
    case 'TrainingAnomalyDetected':
      return <StatusCard icon={<AlertTriangle className="w-4 h-4" />} tone="warn" title="Anomaly detected" subtitle={String(event.payload.reason ?? event.payload.anomaly ?? '')} />;
    case 'RecoveryPlanGenerated':
      return <RecoveryCard event={event} />;
    case 'RetryApproved':
      return <StatusCard icon={<CheckCircle2 className="w-4 h-4" />} tone="success" title="Recovery approved" />;
    case 'RetryDenied':
      return <StatusCard icon={<XCircle className="w-4 h-4" />} tone="danger" title="Recovery denied" />;
    case 'TrainingCompleted':
      return <StatusCard icon={<CheckCircle2 className="w-4 h-4" />} tone="success" title="Training finished" />;
    case 'EvaluationCompleted':
      return <EvaluationCard event={event} />;
    case 'ExportChoiceRequested':
      return <ExportChoiceCard {...props} />;
    case 'ExportCompleted':
      return <StatusCard icon={<PackageCheck className="w-4 h-4" />} tone="success" title="Model exported" subtitle={summariseExport(event.payload.export)} />;
    case 'SessionClosed':
      return <StatusCard icon={<CheckCircle2 className="w-4 h-4" />} tone="success" title="Session closed" subtitle={String(event.payload.reason ?? '')} />;
    case 'Error':
      return <StatusCard icon={<XCircle className="w-4 h-4" />} tone="danger" title="Error" subtitle={String(event.payload.error ?? '')} />;
    default:
      return null;
  }
}

function BubbleCard({ text, actor }: { text: string; actor: 'agent' | 'user' }) {
  const isUser = actor === 'user';
  return (
    <div className={cn('flex gap-3', isUser && 'flex-row-reverse')}>
      <div className={cn('w-7 h-7 shrink-0 rounded flex items-center justify-center border', isUser ? 'bg-white text-black border-white' : 'bg-white/5 text-white/70 border-white/10')}>
        {isUser ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
      </div>
      <div className={cn('max-w-[85%] px-3.5 py-2.5 rounded-lg text-[12.5px] leading-relaxed border whitespace-pre-wrap',
        isUser ? 'bg-white text-black border-white/20' : 'bg-white/[0.03] text-white/90 border-white/10')}>
        {text}
      </div>
    </div>
  );
}

function StatusCard({ icon, tone, title, subtitle }: { icon: React.ReactNode; tone: 'success' | 'warn' | 'danger' | 'info'; title: string; subtitle?: string }) {
  const colorMap = {
    success: 'border-success/30 bg-success/5 text-success',
    warn: 'border-warn/30 bg-warn/5 text-warn',
    danger: 'border-danger/30 bg-danger/5 text-danger',
    info: 'border-info/30 bg-info/5 text-info',
  } as const;
  return (
    <div className={cn('flex items-center gap-3 px-3.5 py-2.5 rounded-lg border', colorMap[tone])}>
      <div className="opacity-80">{icon}</div>
      <div className="flex-1 min-w-0">
        <p className="text-[11px] font-black uppercase tracking-widest leading-tight">{title}</p>
        {subtitle && <p className="text-[10px] text-white/50 truncate font-medium normal-case mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}

function ClarificationCard({ event, sessionId, onAction, busy, setBusy, session }: CardProps) {
  const q = event.payload as {
    question_id: string;
    question: string;
    kind: 'single_choice' | 'multi_choice' | 'text' | 'yes_no';
    options?: string[];
    context?: string;
  };
  const [single, setSingle] = useState<string>('');
  const [multi, setMulti] = useState<string[]>([]);
  const [text, setText] = useState<string>('');
  const answered = session?.clarifications?.some((c) => c.question_id === q.question_id);

  const submit = async () => {
    setBusy(true);
    try {
      const value = q.kind === 'multi_choice' ? multi : q.kind === 'text' ? text.trim() : single;
      if (q.kind === 'multi_choice' ? multi.length === 0 : !value) return;
      await api(`/api/sessions/${sessionId}/clarifications/${q.question_id}`, {
        method: 'POST',
        body: JSON.stringify({ value }),
      });
      onAction();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="border border-white/15 bg-white/[0.03] rounded-lg p-4 space-y-3">
      <div className="flex items-start gap-3">
        <HelpCircle className="w-4 h-4 text-white/60 mt-0.5 shrink-0" />
        <div className="space-y-1">
          <p className="text-[10px] uppercase tracking-[0.25em] text-white/40 font-black">Agent needs clarification</p>
          <p className="text-[13px] text-white font-medium">{q.question}</p>
          {q.context && <p className="text-[11px] text-white/50 leading-relaxed">{q.context}</p>}
        </div>
      </div>

      {!answered && (
        <div className="space-y-2.5 pl-7">
          {q.kind === 'single_choice' && (
            <div className="flex flex-wrap gap-1.5">
              {(q.options ?? []).map((o) => (
                <button
                  key={o}
                  onClick={() => setSingle(o)}
                  className={cn(
                    'px-3 py-1.5 rounded text-[11px] font-bold uppercase tracking-widest border transition-all',
                    single === o
                      ? 'bg-white text-black border-white'
                      : 'bg-white/5 text-white/70 border-white/10 hover:border-white/30 hover:text-white',
                  )}
                >
                  {o}
                </button>
              ))}
            </div>
          )}
          {q.kind === 'multi_choice' && (
            <div className="flex flex-wrap gap-1.5">
              {(q.options ?? []).map((o) => {
                const on = multi.includes(o);
                return (
                  <button
                    key={o}
                    onClick={() =>
                      setMulti((arr) => (on ? arr.filter((x) => x !== o) : [...arr, o]))
                    }
                    className={cn(
                      'px-3 py-1.5 rounded text-[11px] font-bold uppercase tracking-widest border transition-all',
                      on
                        ? 'bg-white text-black border-white'
                        : 'bg-white/5 text-white/70 border-white/10 hover:border-white/30 hover:text-white',
                    )}
                  >
                    {o}
                  </button>
                );
              })}
            </div>
          )}
          {q.kind === 'text' && (
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={3}
              placeholder="type your answer…"
              className="w-full bg-black border border-white/10 rounded px-3 py-2 text-[12px] text-white placeholder:text-white/30 focus:border-white/40 focus:outline-none"
            />
          )}
          {q.kind === 'yes_no' && (
            <div className="flex gap-1.5">
              {['yes', 'no'].map((o) => (
                <button
                  key={o}
                  onClick={() => setSingle(o)}
                  className={cn(
                    'px-4 py-1.5 rounded text-[11px] font-bold uppercase tracking-widest border transition-all',
                    single === o
                      ? 'bg-white text-black border-white'
                      : 'bg-white/5 text-white/70 border-white/10 hover:border-white/30',
                  )}
                >
                  {o}
                </button>
              ))}
            </div>
          )}

          <button
            onClick={submit}
            disabled={busy || (q.kind === 'multi_choice' ? multi.length === 0 : q.kind === 'text' ? !text.trim() : !single)}
            className="px-3.5 py-1.5 bg-white text-black rounded text-[10px] font-black uppercase tracking-widest disabled:opacity-30 disabled:cursor-not-allowed hover:bg-white/90"
          >
            Submit answer
          </button>
        </div>
      )}
      {answered && (
        <p className="pl-7 text-[10px] uppercase tracking-widest text-white/40 font-black">Answered.</p>
      )}
    </div>
  );
}

function PipelineDraftCard({ event }: CardProps) {
  const summary = (event.payload.summary as { title?: string; summary?: string }) ?? {};
  const cfg = (event.payload.config as Record<string, unknown>) ?? {};
  const minutes = Number(event.payload.estimated_minutes ?? 0);
  return (
    <div className="border border-white/10 bg-white/[0.03] rounded-lg p-4 space-y-3">
      <div className="flex items-center gap-3">
        <GitBranch className="w-4 h-4 text-white/60" />
        <p className="text-[10px] uppercase tracking-[0.25em] text-white/50 font-black">Pipeline draft</p>
        {minutes > 0 && (
          <span className="ml-auto text-[10px] uppercase tracking-widest text-white/40 font-black">
            ~{minutes.toFixed(0)} min
          </span>
        )}
      </div>
      <div className="text-[13px] text-white font-medium">{summary.title ?? 'Pipeline draft'}</div>
      {summary.summary && (
        <div className="text-[12px] text-white/70 whitespace-pre-wrap leading-relaxed">{summary.summary}</div>
      )}
      <details className="text-[11px]">
        <summary className="cursor-pointer text-white/40 hover:text-white/70 uppercase tracking-widest font-black">
          full config
        </summary>
        <pre className="mt-2 bg-black border border-white/10 rounded p-3 text-[10px] text-white/60 overflow-x-auto font-mono">
{JSON.stringify(cfg, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function ApprovalRequiredCard({ sessionId, onAction, busy, setBusy, session, event }: CardProps) {
  const decided = session?.state !== 'awaiting_approval';
  const summary = (event.payload.summary as { summary?: string }) ?? {};
  const minutes = Number(event.payload.estimated_minutes ?? 0);

  const decide = async (approve: boolean) => {
    setBusy(true);
    try {
      await api(`/api/sessions/${sessionId}/approve`, {
        method: 'POST',
        body: JSON.stringify({ approve, reason: approve ? 'user approved' : 'user rejected' }),
      });
      onAction();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="border border-warn/30 bg-warn/[0.04] rounded-lg p-4 space-y-3">
      <div className="flex items-center gap-2">
        <AlertTriangle className="w-4 h-4 text-warn" />
        <p className="text-[10px] uppercase tracking-[0.25em] font-black text-warn">Awaiting your approval</p>
        {minutes > 0 && (
          <span className="ml-auto text-[10px] uppercase tracking-widest text-white/40 font-black">~{minutes.toFixed(0)} min</span>
        )}
      </div>
      {summary.summary && (
        <div className="text-[12px] text-white/80 whitespace-pre-wrap leading-relaxed">{summary.summary}</div>
      )}
      {!decided && (
        <div className="flex gap-2 pt-1">
          <button
            disabled={busy}
            onClick={() => decide(true)}
            className="px-4 py-1.5 bg-white text-black rounded text-[10px] font-black uppercase tracking-widest disabled:opacity-30 hover:bg-white/90"
          >
            Approve & run
          </button>
          <button
            disabled={busy}
            onClick={() => decide(false)}
            className="px-4 py-1.5 bg-transparent text-white border border-white/20 rounded text-[10px] font-black uppercase tracking-widest hover:bg-white/5 disabled:opacity-30"
          >
            Reject
          </button>
        </div>
      )}
    </div>
  );
}

function MetricRow({ event }: { event: AgentEvent }) {
  const p = event.payload as { step?: number; epoch?: number; loss?: number | null; status?: string };
  return (
    <div className="flex items-center gap-3 px-3.5 py-2 rounded border border-white/5 bg-white/[0.02] text-[10.5px]">
      <Activity className="w-3.5 h-3.5 text-info" />
      <span className="text-white/40 font-black uppercase tracking-widest text-[9px]">metric</span>
      <span className="font-mono text-white/70">step {p.step ?? '-'}</span>
      <span className="font-mono text-white/40">epoch {p.epoch ?? '-'}</span>
      {p.loss != null && (
        <span className="font-mono text-info">loss {Number(p.loss).toFixed(4)}</span>
      )}
      {p.status && (
        <span className="ml-auto text-[9px] uppercase tracking-widest text-white/30 font-black">{p.status}</span>
      )}
    </div>
  );
}

function RecoveryCard({ event }: { event: AgentEvent }) {
  const p = event.payload as { rationale?: string; operations?: { op: string; path: string; old: unknown; new: unknown }[]; confidence?: number };
  return (
    <div className="border border-warn/30 bg-warn/[0.04] rounded-lg p-3.5 space-y-2">
      <div className="flex items-center gap-2">
        <Zap className="w-4 h-4 text-warn" />
        <p className="text-[10px] uppercase tracking-[0.25em] text-warn font-black">Recovery plan</p>
        {p.confidence != null && (
          <span className="ml-auto text-[10px] uppercase tracking-widest text-white/40 font-black">conf {Math.round(p.confidence * 100)}%</span>
        )}
      </div>
      <p className="text-[12px] text-white/80">{p.rationale}</p>
      {p.operations && p.operations.length > 0 && (
        <ul className="text-[11px] text-white/60 font-mono space-y-0.5">
          {p.operations.map((op, i) => (
            <li key={i}>
              <span className="text-warn">{op.op}</span> {op.path}: {String(op.old)} → {String(op.new)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function EvaluationCard({ event }: { event: AgentEvent }) {
  const e = (event.payload.evaluation as Record<string, unknown>) ?? {};
  const score = typeof e.score === 'number' ? (e.score * 100).toFixed(1) : null;
  return (
    <div className="border border-success/30 bg-success/[0.04] rounded-lg p-3.5 space-y-1.5">
      <div className="flex items-center gap-2">
        <CheckCircle2 className="w-4 h-4 text-success" />
        <p className="text-[10px] uppercase tracking-[0.25em] text-success font-black">Evaluation complete</p>
      </div>
      <div className="grid grid-cols-3 gap-3 text-[10px] uppercase tracking-widest text-white/60 font-black">
        <Stat label="Loss" value={typeof e.final_loss === 'number' ? e.final_loss.toFixed(4) : '-'} />
        <Stat label="Score" value={score ? `${score}%` : '-'} />
        <Stat label="Steps" value={String(e.training_steps ?? '-')} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-white/30">{label}</div>
      <div className="text-white text-[14px] font-mono normal-case">{value}</div>
    </div>
  );
}

function ExportChoiceCard({ sessionId, onAction, busy, setBusy, session }: CardProps) {
  const decided = session?.state !== 'awaiting_export_choice';
  const [choice, setChoice] = useState<'local' | 'hf' | 'both'>('local');
  const [repoId, setRepoId] = useState<string>('');

  const submit = async () => {
    if ((choice === 'hf' || choice === 'both') && !repoId.trim()) return;
    setBusy(true);
    try {
      await api(`/api/sessions/${sessionId}/export`, {
        method: 'POST',
        body: JSON.stringify({ choice, hf_repo_id: repoId.trim() || undefined }),
      });
      onAction();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="border border-white/15 bg-white/[0.03] rounded-lg p-4 space-y-3">
      <div className="flex items-center gap-2">
        <PackageCheck className="w-4 h-4 text-white/60" />
        <p className="text-[10px] uppercase tracking-[0.25em] text-white/50 font-black">Where should the model go?</p>
      </div>

      {!decided && (
        <>
          <div className="grid grid-cols-3 gap-2">
            {(['local', 'hf', 'both'] as const).map((c) => (
              <button
                key={c}
                onClick={() => setChoice(c)}
                className={cn(
                  'px-3 py-3 rounded text-[10px] font-black uppercase tracking-widest border transition-all',
                  choice === c
                    ? 'bg-white text-black border-white'
                    : 'bg-white/5 text-white/70 border-white/10 hover:border-white/30 hover:text-white',
                )}
              >
                {c === 'hf' ? 'Hugging Face' : c}
              </button>
            ))}
          </div>
          {(choice === 'hf' || choice === 'both') && (
            <input
              value={repoId}
              onChange={(e) => setRepoId(e.target.value)}
              placeholder="user/my-trained-model"
              className="w-full bg-black border border-white/10 rounded px-3 py-2 text-[12px] text-white placeholder:text-white/30 focus:border-white/40 focus:outline-none"
            />
          )}
          <button
            onClick={submit}
            disabled={busy || ((choice === 'hf' || choice === 'both') && !repoId.trim())}
            className="px-3.5 py-1.5 bg-white text-black rounded text-[10px] font-black uppercase tracking-widest disabled:opacity-30 hover:bg-white/90"
          >
            Confirm export
          </button>
        </>
      )}
      {decided && <p className="text-[10px] uppercase tracking-widest text-white/40 font-black">Confirmed.</p>}
    </div>
  );
}

function MessageBar({ sessionId, disabled }: { sessionId: string; disabled: boolean }) {
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const send = async () => {
    if (!text.trim() || disabled) return;
    setSending(true);
    try {
      await api(`/api/sessions/${sessionId}/messages`, {
        method: 'POST',
        body: JSON.stringify({ text: text.trim() }),
      });
      setText('');
    } finally {
      setSending(false);
    }
  };
  return (
    <div className="border-t border-white/5 p-3 flex items-center gap-2 bg-[#070707]">
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
          }
        }}
        disabled={disabled || sending}
        placeholder={disabled ? 'session closed' : 'send a message to the agent…'}
        className="flex-1 bg-black border border-white/10 rounded px-3 py-2 text-[12px] text-white placeholder:text-white/30 focus:border-white/40 focus:outline-none disabled:opacity-40"
      />
      <button
        onClick={send}
        disabled={!text.trim() || disabled || sending}
        className="w-9 h-9 rounded bg-white text-black flex items-center justify-center disabled:opacity-30 hover:bg-white/90"
      >
        <Send className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────

function stringifyValue(v: unknown): string {
  if (Array.isArray(v)) return v.join(', ');
  if (typeof v === 'object' && v !== null) return JSON.stringify(v);
  return String(v ?? '');
}

function summariseExport(v: unknown): string {
  if (!v || typeof v !== 'object') return '';
  const e = v as { local?: { local_path?: string }; hf?: { repo_id?: string } };
  const parts: string[] = [];
  if (e.local?.local_path) parts.push(`local: ${e.local.local_path}`);
  if (e.hf?.repo_id) parts.push(`hf: ${e.hf.repo_id}`);
  return parts.join(' · ');
}
