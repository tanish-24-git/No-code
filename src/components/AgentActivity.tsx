'use client';

/**
 * Agent chat surface (ChatGPT-style), driven by event taxonomy v2.
 *
 * The reducer turns the append-only event stream into a transcript:
 *   - chat.delta frames coalesce into a live streaming bubble per agent run;
 *     the final chat.message replaces the accumulated deltas.
 *   - thinking deltas render as a dimmed live bubble, dropped once the
 *     final message lands.
 *   - user.ask / approval.requested / budget.exceeded render as inline
 *     actionable rows (chips / buttons / top-up), gated on the session's
 *     pending state so stale cards never re-arm after a reload.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import useSWR from 'swr';
import { api, fetcher } from '@/lib/api';
import { useSessionEvents } from '@/lib/sse';
import { cn } from '@/lib/cn';
import {
  STATUS_LABEL,
  spentUsd,
  type AgentEvent,
  type SessionRecord,
} from '@/lib/events';
import { Loader2, Send, Wrench } from 'lucide-react';

// ── Transcript reducer ────────────────────────────────────────────────────

type Item =
  | { key: string; type: 'user'; text: string }
  | { key: string; type: 'assistant'; text: string; streaming: boolean; runId: string }
  | { key: string; type: 'thinking'; text: string; runId: string }
  | { key: string; type: 'system'; text: string }
  | { key: string; type: 'error'; text: string }
  | { key: string; type: 'tool'; label: string }
  | { key: string; type: 'ask'; questionId: string; question: string; kind: string; options?: string[]; answered: boolean }
  | { key: string; type: 'approval'; approvalId: string; tool: string; summary: string; body?: string; decided: boolean; approved?: boolean }
  | { key: string; type: 'budget'; spentUsd: number; budgetUsd: number; deltaNeeded?: number; resolved: boolean };

const THINKING_TAIL = 600;

function reduceTranscript(events: AgentEvent[]): Item[] {
  const items: Item[] = [];
  const sorted = [...events].sort((a, b) => (a.id < b.id ? -1 : 1));

  const last = () => items[items.length - 1];
  const dropTrailingThinking = (runId: string) => {
    const l = last();
    if (l && l.type === 'thinking' && l.runId === runId) items.pop();
  };

  for (const e of sorted) {
    const runId = e.agentRunId ?? 'system';
    const p = e.payload as Record<string, unknown>;
    switch (e.kind) {
      case 'chat.message': {
        const role = String(p.role);
        const text = String(p.text ?? '');
        if (role === 'user') {
          items.push({ key: e.id, type: 'user', text });
        } else if (role === 'system') {
          items.push({ key: e.id, type: 'system', text });
        } else {
          dropTrailingThinking(runId);
          // Replace the streaming bubble this message finalizes.
          for (let i = items.length - 1; i >= 0; i--) {
            const it = items[i];
            if (it.type === 'assistant' && it.runId === runId) {
              if (it.streaming) {
                it.text = text;
                it.streaming = false;
              } else {
                items.push({ key: e.id, type: 'assistant', text, streaming: false, runId });
              }
              break;
            }
            if (i === 0) items.push({ key: e.id, type: 'assistant', text, streaming: false, runId });
          }
          if (items.length === 0) items.push({ key: e.id, type: 'assistant', text, streaming: false, runId });
        }
        break;
      }
      case 'chat.delta': {
        const delta = String(p.delta ?? '');
        if (!delta) break;
        if (p.channel === 'thinking') {
          const l = last();
          if (l && l.type === 'thinking' && l.runId === runId) {
            l.text = (l.text + delta).slice(-THINKING_TAIL);
          } else {
            items.push({ key: e.id, type: 'thinking', text: delta, runId });
          }
        } else {
          const l = last();
          if (l && l.type === 'assistant' && l.streaming && l.runId === runId) {
            l.text += delta;
          } else {
            dropTrailingThinking(runId);
            items.push({ key: e.id, type: 'assistant', text: delta, streaming: true, runId });
          }
        }
        break;
      }
      case 'tool.called':
        items.push({ key: e.id, type: 'tool', label: String(p.tool ?? 'tool') });
        break;
      case 'user.ask':
        items.push({
          key: e.id,
          type: 'ask',
          questionId: String(p.questionId ?? ''),
          question: String(p.question ?? ''),
          kind: String(p.kind ?? 'text'),
          options: Array.isArray(p.options) ? (p.options as string[]) : undefined,
          answered: false,
        });
        break;
      case 'user.answer': {
        const qid = String(p.questionId ?? '');
        for (const it of items) {
          if (it.type === 'ask' && it.questionId === qid) it.answered = true;
        }
        items.push({ key: e.id, type: 'user', text: String(p.value ?? '') });
        break;
      }
      case 'approval.requested':
        items.push({
          key: e.id,
          type: 'approval',
          approvalId: String(p.approvalId ?? ''),
          tool: String(p.tool ?? ''),
          summary: String(p.summary ?? ''),
          body: typeof p.body === 'string' ? p.body : undefined,
          decided: false,
        });
        break;
      case 'approval.decided': {
        const aid = String(p.approvalId ?? '');
        for (const it of items) {
          if (it.type === 'approval' && it.approvalId === aid) {
            it.decided = true;
            it.approved = Boolean(p.approved);
          }
        }
        break;
      }
      case 'budget.exceeded':
        items.push({
          key: e.id,
          type: 'budget',
          spentUsd: Number(p.spentUsd ?? 0),
          budgetUsd: Number(p.budgetUsd ?? 0),
          deltaNeeded: typeof p.deltaNeeded === 'number' ? p.deltaNeeded : undefined,
          resolved: false,
        });
        break;
      case 'budget.topup': {
        for (const it of items) {
          if (it.type === 'budget') it.resolved = true;
        }
        items.push({ key: e.id, type: 'system', text: `Budget increased to $${Number(p.newBudgetUsd ?? 0).toFixed(2)}.` });
        break;
      }
      case 'error':
        items.push({ key: e.id, type: 'error', text: String(p.message ?? '') });
        break;
      default:
        break;
    }
  }
  return items;
}

// ── Component ─────────────────────────────────────────────────────────────

type Props = { sessionId: string | null };

export function AgentActivity({ sessionId }: Props) {
  const { events, connected } = useSessionEvents(sessionId);
  const { data: session, mutate: mutateSession } = useSWR<SessionRecord>(
    sessionId ? `/api/sessions/${sessionId}` : null,
    fetcher,
    { refreshInterval: 2000 },
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const lastEvent = events[events.length - 1];
    if (!lastEvent) return;
    if (['session.status', 'user.ask', 'user.answer', 'approval.requested', 'approval.decided', 'budget.exceeded', 'budget.topup'].includes(lastEvent.kind)) {
      mutateSession();
    }
  }, [events, mutateSession]);

  const transcript = useMemo(() => reduceTranscript(events), [events]);

  const lastItem = transcript[transcript.length - 1];
  const streamingNow = !!lastItem && ((lastItem.type === 'assistant' && lastItem.streaming) || lastItem.type === 'thinking');
  const working = session?.status === 'running' && !streamingNow;

  const isAtBottom = useRef(true);
  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    isAtBottom.current = scrollHeight - scrollTop - clientHeight < 150;
  };
  const lastText = lastItem && 'text' in lastItem ? lastItem.text : '';
  useEffect(() => {
    if (isAtBottom.current) {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
    }
  }, [transcript.length, lastText]);

  if (!sessionId) {
    return (
      <div className="flex-1 flex items-center justify-center p-12">
        <div className="text-center text-[12px] text-fg-3">Create a session to talk to the agent.</div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-bg">
      <Header session={session} connected={connected} />

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        onWheel={(e) => { e.stopPropagation(); e.nativeEvent.stopImmediatePropagation(); }}
        onTouchMove={(e) => { e.stopPropagation(); e.nativeEvent.stopImmediatePropagation(); }}
        className="flex-1 overflow-y-auto overscroll-y-contain px-5 py-5 space-y-4"
        data-lenis-prevent="true"
      >
        {transcript.length === 0 && (
          <div className="text-[12px] text-fg-3">
            Tell the agent what you want to fine-tune — or upload a dataset to get started.
          </div>
        )}
        {transcript.map((item) => (
          <Row
            key={item.key}
            item={item}
            session={session}
            sessionId={sessionId}
            busy={busy}
            setBusy={setBusy}
            onAction={() => mutateSession()}
          />
        ))}
        {working && <Working />}
      </div>

      <MessageBar sessionId={sessionId} />
    </div>
  );
}

function Header({ session, connected }: { session?: SessionRecord; connected: boolean }) {
  const label = session ? STATUS_LABEL[session.status] ?? session.status : 'Connecting…';
  const spent = spentUsd(session?.ledger);
  return (
    <div className="px-5 py-3 border-b border-border-2 bg-bg flex items-center gap-3">
      <span className={cn('w-1.5 h-1.5 rounded-full', connected ? 'bg-success' : 'bg-fg-3')} />
      <span className="text-[11px] text-fg-2">{label}</span>
      {session && (
        <span className="ml-auto text-[10px] text-fg-3 font-mono">
          ${spent.toFixed(4)} / ${session.budgetUsd.toFixed(2)}
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
        <span>working…</span>
      </div>
    </div>
  );
}

// ── Rows ──────────────────────────────────────────────────────────────────

type RowProps = {
  item: Item;
  session?: SessionRecord;
  sessionId: string;
  busy: boolean;
  setBusy: (v: boolean) => void;
  onAction: () => void;
};

function Row(props: RowProps) {
  const { item } = props;
  switch (item.type) {
    case 'user':
      return <Bubble side="right" text={item.text} />;
    case 'assistant':
      return <Bubble side="left" text={item.text} streaming={item.streaming} />;
    case 'thinking':
      return (
        <div className="flex justify-start">
          <div className="max-w-[88%] px-4 py-2 rounded-2xl text-[11px] leading-relaxed whitespace-pre-wrap break-words bg-bg-2/50 text-fg-3 italic border border-border/50">
            {item.text}
          </div>
        </div>
      );
    case 'system':
      return <div className="text-center text-[10px] text-fg-3 py-1">{item.text}</div>;
    case 'error':
      return (
        <div className="flex justify-start">
          <div className="max-w-[88%] px-4 py-2.5 rounded-2xl text-[12px] leading-relaxed whitespace-pre-wrap break-words bg-danger/10 text-danger border border-danger/20">
            {item.text}
          </div>
        </div>
      );
    case 'tool':
      return (
        <div className="flex items-center gap-1.5 text-[10px] text-fg-3 pl-1">
          <Wrench className="w-3 h-3" />
          <span className="font-mono">{item.label}</span>
        </div>
      );
    case 'ask':
      return <AskRow {...props} item={item} />;
    case 'approval':
      return <ApprovalRow {...props} item={item} />;
    case 'budget':
      return <BudgetRow {...props} item={item} />;
    default:
      return null;
  }
}

function Bubble({ side, text, streaming, children }: { side: 'left' | 'right'; text?: string; streaming?: boolean; children?: React.ReactNode }) {
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
        {streaming && <span className="inline-block w-1.5 h-3.5 ml-0.5 bg-fg-2 animate-pulse align-text-bottom" />}
        {children}
      </div>
    </div>
  );
}

// ── ask_user (chips / text) ───────────────────────────────────────────────

function AskRow({ item, session, sessionId, busy, setBusy, onAction }: RowProps & { item: Extract<Item, { type: 'ask' }> }) {
  const [single, setSingle] = useState('');
  const [multi, setMulti] = useState<string[]>([]);
  const [text, setText] = useState('');
  const [acted, setActed] = useState(false);
  const active = !item.answered && !acted && session?.pendingQuestion?.questionId === item.questionId;

  const submit = async () => {
    const value = item.kind === 'multi' ? multi.join(', ') : item.kind === 'text' ? text.trim() : single;
    if (!value) return;
    setBusy(true);
    setActed(true);
    try {
      await api(`/api/sessions/${sessionId}/messages`, {
        method: 'POST',
        body: JSON.stringify({ text: value, questionId: item.questionId }),
      });
      onAction();
    } catch {
      setActed(false);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex justify-start">
      <div className="max-w-[88%] space-y-2">
        <Bubble side="left" text={item.question} />
        {active && (
          <div className="space-y-2 pl-1">
            {(item.kind === 'single' || item.kind === 'yes_no') && (
              <div className="flex flex-wrap gap-1.5">
                {(item.options ?? (item.kind === 'yes_no' ? ['yes', 'no'] : [])).map((o) => (
                  <button
                    key={o}
                    onClick={() => setSingle(o)}
                    className={cn(
                      'px-3 py-1 rounded-full text-[11px] border transition-all',
                      single === o ? 'bg-fg text-bg border-fg' : 'bg-bg-2 text-fg-2 border-border hover:text-fg',
                    )}
                  >
                    {o}
                  </button>
                ))}
              </div>
            )}
            {item.kind === 'multi' && (
              <div className="flex flex-wrap gap-1.5">
                {(item.options ?? []).map((o) => {
                  const on = multi.includes(o);
                  return (
                    <button
                      key={o}
                      onClick={() => setMulti((arr) => (on ? arr.filter((x) => x !== o) : [...arr, o]))}
                      className={cn(
                        'px-3 py-1 rounded-full text-[11px] border transition-all',
                        on ? 'bg-fg text-bg border-fg' : 'bg-bg-2 text-fg-2 border-border hover:text-fg',
                      )}
                    >
                      {o}
                    </button>
                  );
                })}
              </div>
            )}
            {item.kind === 'text' && (
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={2}
                placeholder="Type your answer…"
                className="w-full bg-bg-2 border border-border rounded-lg px-3 py-2 text-[12px] text-fg placeholder:text-fg-3 outline-none focus:border-fg-3 resize-none"
              />
            )}
            <button
              onClick={submit}
              disabled={busy || (item.kind === 'multi' ? multi.length === 0 : item.kind === 'text' ? !text.trim() : !single)}
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

// ── approval.requested (plan / command / hf upload) ───────────────────────

function ApprovalRow({ item, session, sessionId, busy, setBusy, onAction }: RowProps & { item: Extract<Item, { type: 'approval' }> }) {
  const [acted, setActed] = useState(false);
  const active = !item.decided && !acted && session?.pendingApproval?.approvalId === item.approvalId;

  const decide = async (approved: boolean) => {
    setBusy(true);
    setActed(true);
    try {
      await api(`/api/sessions/${sessionId}/approvals/${item.approvalId}`, {
        method: 'POST',
        body: JSON.stringify({ approved }),
      });
      onAction();
    } catch {
      setActed(false);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex justify-start">
      <div className="max-w-[88%] space-y-2">
        <Bubble side="left">
          <div className="space-y-1.5">
            <div className="text-[12px] font-bold">{item.summary || `Approval needed: ${item.tool}`}</div>
            {item.body && (
              <pre className="text-[11px] text-fg-2 whitespace-pre-wrap break-words font-mono bg-bg rounded-lg p-2 border border-border max-h-64 overflow-y-auto">
                {item.body}
              </pre>
            )}
          </div>
        </Bubble>
        {item.decided && (
          <div className="text-[10px] text-fg-3 pl-1">{item.approved ? 'Approved' : 'Denied'}</div>
        )}
        {active && (
          <div className="flex items-center gap-2 pl-1">
            <button
              disabled={busy}
              onClick={() => decide(true)}
              className="px-3 py-1 rounded-full bg-fg text-bg text-[11px] font-medium disabled:opacity-30 hover:opacity-90"
            >
              Approve
            </button>
            <button
              disabled={busy}
              onClick={() => decide(false)}
              className="px-3 py-1 rounded-full bg-bg-2 text-fg-2 border border-border text-[11px] font-medium hover:text-fg disabled:opacity-30"
            >
              Deny
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ── budget.exceeded (top-up) ──────────────────────────────────────────────

function BudgetRow({ item, session, sessionId, busy, setBusy, onAction }: RowProps & { item: Extract<Item, { type: 'budget' }> }) {
  const [amount, setAmount] = useState(item.deltaNeeded ? String(Math.ceil(item.deltaNeeded * 100) / 100) : '1');
  const [acted, setActed] = useState(false);
  const active = !item.resolved && !acted && session?.status === 'paused_budget';

  const topUp = async () => {
    const add = Number.parseFloat(amount);
    if (!Number.isFinite(add) || add <= 0) return;
    setBusy(true);
    setActed(true);
    try {
      await api(`/api/sessions/${sessionId}/budget`, {
        method: 'POST',
        body: JSON.stringify({ addUsd: add }),
      });
      onAction();
    } catch {
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
            <div className="text-[12px] font-bold">Budget limit reached</div>
            <div className="text-[12px] text-fg-2">
              ${item.spentUsd.toFixed(2)} spent of ${item.budgetUsd.toFixed(2)}.
              {item.deltaNeeded !== undefined && ` Roughly $${item.deltaNeeded.toFixed(2)} more needed to finish.`}
            </div>
          </div>
        </Bubble>
        {active && (
          <div className="flex items-center gap-2 pl-1">
            <span className="text-[11px] text-fg-3">Add $</span>
            <input
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="w-20 bg-bg-2 border border-border rounded-full px-3 py-1.5 text-[12px] text-fg outline-none focus:border-fg-3"
            />
            <button
              disabled={busy}
              onClick={topUp}
              className="px-3 py-1 rounded-full bg-fg text-bg text-[11px] font-medium disabled:opacity-30 hover:opacity-90"
            >
              Add budget & continue
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Bottom message bar ────────────────────────────────────────────────────

function MessageBar({ sessionId }: { sessionId: string }) {
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const send = async () => {
    if (!text.trim()) return;
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
        disabled={sending}
        placeholder="Message the agent…"
        className="flex-1 bg-bg-2 border border-border rounded-2xl px-4 py-2 text-[13px] text-fg placeholder:text-fg-3 outline-none focus:border-fg-3 resize-none max-h-32"
      />
      <button
        onClick={send}
        disabled={!text.trim() || sending}
        className="w-9 h-9 rounded-full bg-fg text-bg flex items-center justify-center disabled:opacity-30 hover:opacity-90"
      >
        <Send className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}
