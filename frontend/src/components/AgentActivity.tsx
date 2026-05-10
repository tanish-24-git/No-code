'use client';

/**
 * Agent chat surface (ChatGPT-style).
 *
 * Plain neutral theme. AI bubbles align left, user bubbles align right. No
 * coloured cards, no logos, no status pills - the goal is for the user to
 * read this like a normal conversation.
 *
 * Inline action affordances replace the old card-based UI:
 *   - clarification questions render as a chat bubble with chips below
 *   - phase plans render as a chat bubble with [approve] / [comment] under it
 *   - pipeline draft + approval, recovery, export - same idea, all inline
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import useSWR from 'swr';
import { api, fetcher } from '@/lib/api';
import { useSessionEvents } from '@/lib/sse';
import { cn } from '@/lib/cn';
import type { AgentEvent, AgentSession, EventKind } from '@/lib/types';
import { Loader2, Send } from 'lucide-react';


// ── Folding rules ─────────────────────────────────────────────────────────
//
// Anything that maps to a chat bubble lives in CHAT_KINDS. Internal events
// (tool calls, decision records, intake progress, etc.) get folded into a
// single "still working..." indicator near the bottom of the transcript.

const CHAT_KINDS: ReadonlySet<EventKind> = new Set([
  'AssistantMessage',
  'UserMessage',
  'PhasePlanProposed',
  'UserClarificationRequested',
  'UserClarificationReceived',
  'PipelineApprovalRequested',
  'TrainingCompleted',
  'EvaluationCompleted',
  'ExportChoiceRequested',
  'ExportCompleted',
  'TrainingAnomalyDetected',
  'RecoveryPlanGenerated',
  'SessionClosed',
  'Error',
]);

const STAGE_LABEL: Record<string, string> = {
  init: 'Initializing',
  profiling: 'Profiling dataset',
  clarifying: 'Awaiting clarification',
  planning: 'Drafting pipeline',
  awaiting_approval: 'Awaiting approval',
  executing: 'Starting training',
  monitoring: 'Training',
  recovering: 'Recovering',
  evaluating: 'Evaluating',
  awaiting_export_choice: 'Awaiting export choice',
  finalizing: 'Finalizing',
  done: 'Complete',
  failed: 'Failed',
  cancelled: 'Cancelled',
};


type Props = { sessionId: string | null };

export function AgentActivity({ sessionId }: Props) {
  const { events, connected, closed } = useSessionEvents(sessionId);
  const { data: session, mutate: mutateSession } = useSWR<AgentSession>(
    sessionId ? `/api/sessions/${sessionId}` : null,
    fetcher,
    { refreshInterval: 1500 },
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (events.length === 0) return;
    const last = events[events.length - 1];
    if ([
      'TaskInferred',
      'UserClarificationRequested',
      'UserClarificationReceived',
      'PhasePlanProposed',
      'PhaseApproved',
      'PhaseStarted',
      'IntakeCompleted',
      'DatasetProfileCompleted',
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

  const transcript = useMemo(() => {
    const unique = new Map<string, AgentEvent>();
    
    // Sort events by time first so we process them in order.
    const sorted = [...events].sort((a, b) => 
      new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    );

    sorted.forEach((e) => {
      // Logic for deduplication:
      // 1. If it's a PhasePlanProposed, use the phase as the key.
      //    We want the LATEST one for that phase.
      // 2. Otherwise use the event.id.
      let key = e.id;
      if (e.kind === 'PhasePlanProposed') {
        key = `phase-${e.payload.phase}`;
        // For phases, we actually want to OVERWRITE with the latest one 
        // in case the agent re-emitted it with updated steps/summary.
        unique.set(key, e);
        return;
      }
      
      // For general events (messages, status), we keep the first occurrence 
      // of an ID to avoid SSE-reconnect duplicates.
      if (!unique.has(key)) {
        unique.set(key, e);
      }
    });

    const raw = Array.from(unique.values())
      .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
      .filter((e) => CHAT_KINDS.has(e.kind));

    const result: AgentEvent[] = [];

    for (const e of raw) {
      const payload = { ...e.payload };
      const last = result[result.length - 1];

      // Delta grouping logic for AssistantMessage and AgentThinking
      const currentDelta = payload.delta as string | undefined;

      if (
        (e.kind === 'AssistantMessage' || e.kind === 'AgentThinking') &&
        currentDelta !== undefined
      ) {
        if (
          last &&
          last.kind === e.kind &&
          last.parent_event_id === e.parent_event_id &&
          last.actor === e.actor
        ) {
          const prevText = (last.payload.text as string) || '';
          last.payload.text = prevText + currentDelta;
          continue;
        }
        payload.text = currentDelta;
      }

      result.push({ ...e, payload });
    }
    return result;
  }, [events]);
  const hasInternalChatter = useMemo(
    () => events.some((e) => !CHAT_KINDS.has(e.kind) && e.kind !== 'NodeMaterialized'),
    [events],
  );
  const stillWorking = !closed && hasInternalChatter && transcript.length > 0;

  const isAtBottom = useRef(true);

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const threshold = 150;
    isAtBottom.current = scrollHeight - scrollTop - clientHeight < threshold;
  };

  const lastMessageText = transcript[transcript.length - 1]?.payload?.text;
  useEffect(() => {
    if (isAtBottom.current) {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
    }
  }, [transcript.length, lastMessageText]);

  if (!sessionId) {
    return (
      <div className="flex-1 flex items-center justify-center p-12">
        <div className="text-center text-[12px] text-fg-3">
          Upload a dataset to start a session.
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-bg">
      <Header session={session} connected={connected} closed={closed} />

      <div 
        ref={scrollRef} 
        onScroll={handleScroll}
        onWheel={(e) => { e.stopPropagation(); e.nativeEvent.stopImmediatePropagation(); }}
        onTouchMove={(e) => { e.stopPropagation(); e.nativeEvent.stopImmediatePropagation(); }}
        className="flex-1 overflow-y-auto overscroll-y-contain px-5 py-5 space-y-4"
        data-lenis-prevent="true"
      >
        {transcript.length === 0 && (
          <div className="flex items-center gap-2 text-[12px] text-fg-3">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            <span>Booting agents...</span>
          </div>
        )}
        {transcript.map((e) => (
          <ChatRow
            key={e.id}
            event={e}
            session={session}
            sessionId={sessionId}
            onAction={() => mutateSession()}
            busy={busy}
            setBusy={setBusy}
          />
        ))}
        {stillWorking && <Working />}
      </div>

      <MessageBar sessionId={sessionId} disabled={closed} />
    </div>
  );
}


function Header({
  session,
  connected,
  closed,
}: {
  session?: AgentSession;
  connected: boolean;
  closed: boolean;
}) {
  const stage = session ? STAGE_LABEL[session.state] ?? session.state : 'Connecting...';
  const live = !closed && connected;
  return (
    <div className="px-5 py-3 border-b border-border-2 bg-bg flex items-center gap-3">
      <span
        className={cn(
          'w-1.5 h-1.5 rounded-full',
          live ? 'bg-fg-2' : closed ? 'bg-fg-3' : 'bg-fg-3',
        )}
      />
      <span className="text-[11px] text-fg-2">{stage}</span>
      {session && session.confidence > 0 && (
        <span className="ml-auto text-[10px] text-fg-3 font-mono">
          conf {Math.round(session.confidence * 100)}%
        </span>
      )}
    </div>
  );
}


function Working() {
  return (
    <div className="flex justify-start">
      <div className="px-3 py-1.5 text-[11px] text-fg-3 flex items-center gap-2">
        <Loader2 className="w-3 h-3 animate-spin" />
        <span>working...</span>
      </div>
    </div>
  );
}


// ── Chat row dispatch ─────────────────────────────────────────────────────

type RowProps = {
  event: AgentEvent;
  session?: AgentSession;
  sessionId: string;
  onAction: () => void;
  busy: boolean;
  setBusy: (v: boolean) => void;
};

function ChatRow(props: RowProps) {
  const { event } = props;
  switch (event.kind) {
    case 'UserMessage':
      return <Bubble side="right" text={String(event.payload.text ?? '')} />;
    case 'AssistantMessage':
      return <Bubble side="left" text={String(event.payload.text ?? '')} />;
    case 'PhasePlanProposed':
      return <PhasePlanRow {...props} />;
    case 'UserClarificationRequested':
      return <ClarificationRow {...props} />;
    case 'UserClarificationReceived':
      return (
        <Bubble
          side="right"
          text={`> ${stringifyValue((event.payload.answer as { value?: unknown })?.value)}`}
        />
      );
    case 'PipelineApprovalRequested':
      return <ApprovalRow {...props} />;
    case 'ExportChoiceRequested':
      return <ExportRow {...props} />;
    case 'TrainingCompleted':
      return <Bubble side="left" text="Training finished." />;
    case 'EvaluationCompleted':
      return <EvaluationRow event={event} />;
    case 'ExportCompleted':
      return <Bubble side="left" text={`Exported. ${summariseExport(event.payload.export)}`} />;
    case 'TrainingAnomalyDetected':
      return <Bubble side="left" text={`Anomaly detected: ${event.payload.reason ?? event.payload.anomaly}`} />;
    case 'RecoveryPlanGenerated':
      return <RecoveryRow event={event} />;
    case 'SessionClosed':
      return <Bubble side="left" text="Session closed." />;
    case 'Error':
      return <Bubble side="left" text={`Error: ${event.payload.error ?? ''}`} />;
    default:
      return null;
  }
}


// ── Bubble ────────────────────────────────────────────────────────────────

function Bubble({ side, text, children }: { side: 'left' | 'right'; text?: string; children?: React.ReactNode }) {
  return (
    <div className={cn('flex', side === 'right' && 'justify-end')}>
      <div
        className={cn(
          'max-w-[88%] px-4 py-2.5 rounded-2xl text-[13px] leading-relaxed whitespace-pre-wrap break-words',
          side === 'right'
            ? 'bg-fg text-bg rounded-br-sm shadow-sm'
            : 'bg-bg-2 text-fg rounded-bl-sm border border-border shadow-sm',
        )}
      >
        {text}
        {children}
      </div>
    </div>
  );
}


// ── Phase plan (chat-bubble with approve / comment under it) ──────────────

function PhasePlanRow({ event, session, sessionId, onAction, busy, setBusy }: RowProps) {
  const p = event.payload as {
    phase: string;
    title?: string;
    summary?: string;
    plan_markdown?: string;
    requires_approval?: boolean;
    steps?: string[];
  };
  const [comment, setComment] = useState('');
  const [showComment, setShowComment] = useState(false);
  const [acted, setActed] = useState(false);

  const approve = async () => {
    setBusy(true);
    setActed(true);
    try {
      await api(`/api/sessions/${sessionId}/phase/${p.phase}/approve`, { method: 'POST' });
      onAction();
    } catch (e) {
      setActed(false);
    } finally { setBusy(false); }
  };

  const submitComment = async () => {
    if (!comment.trim()) return;
    setBusy(true);
    setActed(true);
    try {
      await api(`/api/sessions/${sessionId}/phase/${p.phase}/comment`, {
        method: 'POST',
        body: JSON.stringify({ text: comment.trim() }),
      });
      setComment('');
      setShowComment(false);
      onAction();
    } catch (e) {
      setActed(false);
    } finally { setBusy(false); }
  };

  return (
    <div className="flex justify-start">
      <div className="max-w-[88%] space-y-2">
        <Bubble side="left">
          <div className="space-y-1.5">
            {p.title && <div className="text-[12px] font-bold">{p.title}</div>}
            {p.summary && <div className="text-[12.5px]">{p.summary}</div>}
            {p.steps && p.steps.length > 0 && (
              <ol className="list-decimal pl-4 space-y-0.5 text-[12px] text-fg-2">
                {p.steps.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ol>
            )}
          </div>
        </Bubble>
        {p.requires_approval && !acted && session?.state === 'awaiting_approval' && (
          <div className="flex items-center gap-2 pl-1">
            <button
              disabled={busy}
              onClick={approve}
              className="px-3 py-1 rounded-full bg-fg text-bg text-[11px] font-medium disabled:opacity-30 hover:opacity-90"
            >
              Approve
            </button>
            <button
              disabled={busy}
              onClick={() => setShowComment((s) => !s)}
              className="px-3 py-1 rounded-full bg-bg-2 text-fg-2 border border-border text-[11px] font-medium hover:text-fg disabled:opacity-30"
            >
              Comment
            </button>
          </div>
        )}
        {showComment && (
          <div className="flex items-center gap-2 pl-1">
            <input
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submitComment();
              }}
              placeholder="Tell the agent what to change..."
              className="flex-1 bg-bg-2 border border-border rounded-full px-3 py-1.5 text-[12px] text-fg placeholder:text-fg-3 outline-none focus:border-fg-3"
            />
            <button
              disabled={busy || !comment.trim()}
              onClick={submitComment}
              className="px-3 py-1.5 rounded-full bg-fg text-bg text-[11px] font-medium disabled:opacity-30 hover:opacity-90"
            >
              Send
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


// ── Clarification (chat bubble + chip choices) ────────────────────────────

function ClarificationRow({ event, session, sessionId, onAction, busy, setBusy }: RowProps) {
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
  const [acted, setActed] = useState(false);
  const answered = session?.clarifications?.some((c) => c.question_id === q.question_id);

  const submit = async () => {
    setBusy(true);
    setActed(true);
    try {
      const value = q.kind === 'multi_choice' ? multi : q.kind === 'text' ? text.trim() : single;
      if (q.kind === 'multi_choice' ? multi.length === 0 : !value) {
        setActed(false);
        return;
      }
      await api(`/api/sessions/${sessionId}/clarifications/${q.question_id}`, {
        method: 'POST',
        body: JSON.stringify({ value }),
      });
      onAction();
    } catch (e) {
      setActed(false);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex justify-start">
      <div className="max-w-[88%] space-y-2">
        <Bubble side="left">
          <div className="space-y-1">
            <div className="text-[12.5px]">{q.question}</div>
            {q.context && <div className="text-[11px] text-fg-2">{q.context}</div>}
          </div>
        </Bubble>
        {!answered && !acted && (
          <div className="space-y-2 pl-1">
            {q.kind === 'single_choice' && (
              <div className="flex flex-wrap gap-1.5">
                {(q.options ?? []).map((o) => (
                  <button
                    key={o}
                    onClick={() => setSingle(o)}
                    className={cn(
                      'px-3 py-1 rounded-full text-[11px] border transition-all',
                      single === o
                        ? 'bg-fg text-bg border-fg'
                        : 'bg-bg-2 text-fg-2 border-border hover:text-fg',
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
                        'px-3 py-1 rounded-full text-[11px] border transition-all',
                        on
                          ? 'bg-fg text-bg border-fg'
                          : 'bg-bg-2 text-fg-2 border-border hover:text-fg',
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
                rows={2}
                placeholder="Type your answer..."
                className="w-full bg-bg-2 border border-border rounded-lg px-3 py-2 text-[12px] text-fg placeholder:text-fg-3 outline-none focus:border-fg-3 resize-none"
              />
            )}
            {q.kind === 'yes_no' && (
              <div className="flex gap-1.5">
                {['yes', 'no'].map((o) => (
                  <button
                    key={o}
                    onClick={() => setSingle(o)}
                    className={cn(
                      'px-4 py-1 rounded-full text-[11px] border transition-all',
                      single === o
                        ? 'bg-fg text-bg border-fg'
                        : 'bg-bg-2 text-fg-2 border-border hover:text-fg',
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
              className="px-3 py-1 rounded-full bg-fg text-bg text-[11px] font-medium disabled:opacity-30 hover:opacity-90"
            >
              Send
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


// ── Pipeline approval (chat bubble + approve / reject) ────────────────────

function ApprovalRow({ event, session, sessionId, onAction, busy, setBusy }: RowProps) {
  const [acted, setActed] = useState(false);
  const decided = session?.state !== 'awaiting_approval' || acted;
  const summary = (event.payload.summary as { summary?: string }) ?? {};
  const minutes = Number(event.payload.estimated_minutes ?? 0);

  const decide = async (approve: boolean) => {
    setBusy(true);
    setActed(true);
    try {
      await api(`/api/sessions/${sessionId}/approve`, {
        method: 'POST',
        body: JSON.stringify({ approve, reason: approve ? 'user approved' : 'user rejected' }),
      });
      onAction();
    } catch (e) {
      setActed(false);
    } finally { setBusy(false); }
  };

  return (
    <div className="flex justify-start">
      <div className="max-w-[88%] space-y-2">
        <Bubble side="left">
          <div className="space-y-1">
            <div className="text-[12px] font-bold">Pipeline ready - approve to start training</div>
            {summary.summary && <div className="text-[12px] text-fg-2 whitespace-pre-wrap">{summary.summary}</div>}
            {minutes > 0 && <div className="text-[11px] text-fg-3">~{minutes.toFixed(0)} minutes estimated</div>}
          </div>
        </Bubble>
        {!decided && !acted && (
          <div className="flex items-center gap-2 pl-1">
            <button
              disabled={busy}
              onClick={() => decide(true)}
              className="px-3 py-1 rounded-full bg-fg text-bg text-[11px] font-medium disabled:opacity-30 hover:opacity-90"
            >
              Approve & run
            </button>
            <button
              disabled={busy}
              onClick={() => decide(false)}
              className="px-3 py-1 rounded-full bg-bg-2 text-fg-2 border border-border text-[11px] font-medium hover:text-fg disabled:opacity-30"
            >
              Reject
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


// ── Recovery + evaluation + export rows ──────────────────────────────────

function RecoveryRow({ event }: { event: AgentEvent }) {
  const p = event.payload as { rationale?: string };
  return <Bubble side="left" text={`Recovery plan: ${p.rationale ?? 'see audit log'}`} />;
}

function EvaluationRow({ event }: { event: AgentEvent }) {
  const e = (event.payload.evaluation as Record<string, unknown>) ?? {};
  const score = typeof e.score === 'number' ? `${(e.score * 100).toFixed(1)}%` : '-';
  const loss = typeof e.final_loss === 'number' ? e.final_loss.toFixed(4) : '-';
  return <Bubble side="left" text={`Evaluation: score ${score}, final loss ${loss}.`} />;
}

function ExportRow({ event, sessionId, onAction, busy, setBusy, session }: RowProps) {
  const decided = session?.state !== 'awaiting_export_choice';
  const [choice, setChoice] = useState<'local' | 'hf' | 'both'>('local');
  const [repoId, setRepoId] = useState<string>('');

  const [acted, setActed] = useState(false);

  const submit = async () => {
    if ((choice === 'hf' || choice === 'both') && !repoId.trim()) return;
    setBusy(true);
    setActed(true);
    try {
      await api(`/api/sessions/${sessionId}/export`, {
        method: 'POST',
        body: JSON.stringify({ choice, hf_repo_id: repoId.trim() || undefined }),
      });
      onAction();
    } catch (e) {
      setActed(false);
    } finally { setBusy(false); }
  };

  return (
    <div className="flex justify-start">
      <div className="max-w-[88%] space-y-2">
        <Bubble side="left" text="Where should the trained model go?" />
        {!decided && !acted && (
          <div className="space-y-2 pl-1">
            <div className="flex gap-1.5">
              {(['local', 'hf', 'both'] as const).map((c) => (
                <button
                  key={c}
                  onClick={() => setChoice(c)}
                  className={cn(
                    'px-3 py-1 rounded-full text-[11px] border transition-all',
                    choice === c ? 'bg-fg text-bg border-fg' : 'bg-bg-2 text-fg-2 border-border hover:text-fg',
                  )}
                >
                  {c === 'hf' ? 'HuggingFace' : c}
                </button>
              ))}
            </div>
            {(choice === 'hf' || choice === 'both') && (
              <input
                value={repoId}
                onChange={(e) => setRepoId(e.target.value)}
                placeholder="user/my-trained-model"
                className="w-full bg-bg-2 border border-border rounded-full px-3 py-1.5 text-[12px] text-fg placeholder:text-fg-3 outline-none focus:border-fg-3"
              />
            )}
            <button
              onClick={submit}
              disabled={busy || ((choice === 'hf' || choice === 'both') && !repoId.trim())}
              className="px-3 py-1 rounded-full bg-fg text-bg text-[11px] font-medium disabled:opacity-30 hover:opacity-90"
            >
              Confirm
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


// ── Bottom message bar ────────────────────────────────────────────────────

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
    } finally { setSending(false); }
  };
  return (
    <div className="border-t border-border-2 p-3 flex items-end gap-2 bg-bg">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
          }
        }}
        rows={1}
        disabled={disabled || sending}
        placeholder={disabled ? 'session closed' : 'Message the agent...'}
        className="flex-1 bg-bg-2 border border-border rounded-2xl px-4 py-2 text-[13px] text-fg placeholder:text-fg-3 outline-none focus:border-fg-3 resize-none max-h-32"
      />
      <button
        onClick={send}
        disabled={!text.trim() || disabled || sending}
        className="w-9 h-9 rounded-full bg-fg text-bg flex items-center justify-center disabled:opacity-30 hover:opacity-90"
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
  return parts.join(' - ');
}
