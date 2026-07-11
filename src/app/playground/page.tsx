'use client';

/**
 * Session-driven playground.
 *
 *   left   — session list + new session (dataset upload returns in M2)
 *   center — canvas area (Training | Agents views land in M4/M5)
 *   right  — AgentActivity: live SSE-driven chat
 */
import { useEffect, useState } from 'react';
import useSWR from 'swr';
import { api, fetcher, uploadFile } from '@/lib/api';
import { STATUS_TONE, type SessionListItem, type SessionRecord } from '@/lib/events';
import { useSessionEvents } from '@/lib/sse';
import { AgentActivity } from '@/components/AgentActivity';
import { AgentGraph } from '@/components/AgentGraph';
import { BudgetBar } from '@/components/BudgetBar';
import { TrainingCanvas } from '@/components/TrainingCanvas';
import { cn } from '@/lib/cn';
import { Bot, Database, GitBranch, LineChart, Loader2, Plus, Upload, XCircle } from 'lucide-react';

const ACCEPTED_DATASET_TYPES = '.csv,.tsv,.json,.jsonl,.txt,.md,.pdf,.docx,.html,.parquet,.xml,.yaml,.yml,.zip';

export default function PlaygroundPage() {
  const { data, mutate: mutateSessions } = useSWR<{ sessions: SessionListItem[] }>(
    '/api/sessions',
    fetcher,
    { refreshInterval: 3000 },
  );
  const sessions = data?.sessions;
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  // Auto-select most recent session if none selected.
  useEffect(() => {
    if (!activeSessionId && sessions && sessions.length > 0) {
      setActiveSessionId(sessions[0].id);
    }
    // Clear selection if it no longer exists (e.g. data/ wiped).
    if (activeSessionId && sessions && !sessions.some((s) => s.id === activeSessionId)) {
      setActiveSessionId(sessions[0]?.id ?? null);
    }
  }, [sessions, activeSessionId]);

  const createSession = async (): Promise<string> => {
    setCreating(true);
    try {
      const s = await api<SessionListItem>('/api/sessions', { method: 'POST', body: JSON.stringify({}) });
      await mutateSessions();
      setActiveSessionId(s.id);
      return s.id;
    } finally {
      setCreating(false);
    }
  };

  const [uploading, setUploading] = useState(false);
  const handleFile = async (file: File) => {
    setUploading(true);
    try {
      const sid = activeSessionId ?? (await createSession());
      await uploadFile(`/api/sessions/${sid}/upload`, file);
      mutateSessions();
    } finally {
      setUploading(false);
    }
  };

  const cancelSession = async (id: string) => {
    await api(`/api/sessions/${id}/cancel`, { method: 'POST' }).catch(() => {});
    mutateSessions();
  };

  const active = sessions?.find((s) => s.id === activeSessionId);
  const { data: sessionDetail } = useSWR<SessionRecord>(
    activeSessionId ? `/api/sessions/${activeSessionId}` : null,
    fetcher,
    { refreshInterval: 2500 },
  );

  // Center view: live agent graph vs training telemetry (M5).
  const [view, setView] = useState<'agents' | 'training'>('agents');
  const { events: graphEvents } = useSessionEvents(activeSessionId);
  // Auto-switch to training when metrics start flowing.
  useEffect(() => {
    const last = graphEvents[graphEvents.length - 1];
    if (last?.kind === 'train.metric') setView('training');
  }, [graphEvents]);

  return (
    <div className="h-[calc(100vh-112px)] bg-black flex overflow-hidden">
      {/* ── Left rail ─────────────────────────────────────────────────── */}
      <aside className="w-[280px] border-r border-white/5 flex flex-col bg-[#050505]">
        <div className="p-4 border-b border-white/5 space-y-2">
          <label
            className={cn(
              'block w-full cursor-pointer rounded-lg border-2 border-dashed transition-all',
              uploading
                ? 'border-white/10 bg-white/[0.02] opacity-60 cursor-wait'
                : 'border-white/15 hover:border-white/40 hover:bg-white/[0.03]',
            )}
          >
            <input
              type="file"
              accept={ACCEPTED_DATASET_TYPES}
              className="hidden"
              disabled={uploading}
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            <div className="px-4 py-6 text-center space-y-2">
              {uploading ? (
                <Loader2 className="w-5 h-5 text-white/60 mx-auto animate-spin" />
              ) : (
                <Upload className="w-5 h-5 text-white/60 mx-auto" />
              )}
              <p className="text-[10px] uppercase tracking-[0.25em] font-black text-white/60">
                {uploading ? 'Uploading…' : 'Drop dataset to start'}
              </p>
              <p className="text-[9px] uppercase tracking-widest text-white/30 font-medium">
                csv json jsonl txt md pdf docx html …
              </p>
            </div>
          </label>
          <button
            onClick={() => void createSession()}
            disabled={creating}
            className="w-full rounded-md border border-white/10 hover:border-white/30 hover:bg-white/[0.03] transition-all px-3 py-2 flex items-center justify-center gap-2 disabled:opacity-60"
          >
            <Plus className="w-3.5 h-3.5 text-white/50" />
            <span className="text-[9px] uppercase tracking-[0.25em] font-black text-white/50">
              {creating ? 'Creating…' : 'Empty session'}
            </span>
          </button>
        </div>

        {/* Session list */}
        <div
          className="flex-1 overflow-y-auto overscroll-y-contain"
          onWheel={(e) => { e.stopPropagation(); e.nativeEvent.stopImmediatePropagation(); }}
          onTouchMove={(e) => { e.stopPropagation(); e.nativeEvent.stopImmediatePropagation(); }}
          data-lenis-prevent="true"
        >
          <div className="px-4 py-3 flex items-center gap-2">
            <Database className="w-3 h-3 text-white/30" />
            <span className="text-[9px] uppercase tracking-[0.25em] font-black text-white/40">Sessions</span>
            <span className="ml-auto text-[10px] font-mono text-white/30">{sessions?.length ?? 0}</span>
          </div>
          <div className="px-2 space-y-1 pb-4">
            {(sessions ?? []).map((s) => {
              const isActive = s.id === activeSessionId;
              return (
                <div
                  key={s.id}
                  className={cn(
                    'group rounded-md border transition-all',
                    isActive
                      ? 'border-white/20 bg-white/[0.04]'
                      : 'border-transparent hover:border-white/10 hover:bg-white/[0.02]',
                  )}
                >
                  <button onClick={() => setActiveSessionId(s.id)} className="w-full text-left px-3 py-2.5">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-[10px] font-mono text-white/40 truncate max-w-[120px]">{s.title}</span>
                      <span
                        className={cn(
                          'px-1.5 py-0.5 rounded text-[8px] uppercase tracking-widest font-black',
                          STATUS_TONE[s.status] ?? 'bg-white/5 text-white/40',
                        )}
                      >
                        {s.status.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div className="text-[10px] text-white/40 font-mono truncate">
                      {new Date(s.updatedAt).toLocaleTimeString()}
                    </div>
                  </button>
                  {isActive && ['running', 'training'].includes(s.status) && (
                    <button
                      onClick={() => cancelSession(s.id)}
                      className="w-full flex items-center justify-center gap-1.5 py-1.5 text-[9px] uppercase tracking-widest font-black text-white/40 hover:text-danger border-t border-white/5"
                    >
                      <XCircle className="w-3 h-3" /> cancel
                    </button>
                  )}
                </div>
              );
            })}
            {sessions?.length === 0 && (
              <div className="px-3 py-6 text-center">
                <p className="text-[10px] uppercase tracking-widest text-white/30 font-black">No sessions yet</p>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* ── Center: canvas area (Agent Graph + Training views land in M4/M5) ── */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        <header className="h-14 px-5 border-b border-white/5 bg-black flex items-center gap-4">
          {active ? (
            <>
              <Bot className="w-4 h-4 text-white/40" />
              <div className="min-w-0">
                <div className="text-[12px] font-bold text-white truncate">{active.title}</div>
                <div className="text-[10px] uppercase tracking-widest text-white/40 font-black">
                  {active.datasetFiles.length > 0
                    ? `${active.datasetFiles.length} dataset file(s)`
                    : 'no dataset yet'}
                </div>
              </div>
            </>
          ) : (
            <div className="text-[10px] uppercase tracking-[0.3em] text-white/30 font-black">No active session</div>
          )}
          <div className="ml-auto flex items-center gap-4">
            <div className="flex rounded-md border border-white/10 overflow-hidden">
              {(
                [
                  { key: 'agents', label: 'Agents', Icon: GitBranch },
                  { key: 'training', label: 'Training', Icon: LineChart },
                ] as const
              ).map(({ key, label, Icon }) => (
                <button
                  key={key}
                  onClick={() => setView(key)}
                  className={cn(
                    'px-3 py-1.5 flex items-center gap-1.5 text-[9px] uppercase tracking-widest font-black transition-colors',
                    view === key ? 'bg-white/10 text-white' : 'text-white/40 hover:text-white/70',
                  )}
                >
                  <Icon className="w-3 h-3" /> {label}
                </button>
              ))}
            </div>
            <BudgetBar session={sessionDetail} />
            {active && (
              <span
                className={cn(
                  'px-2.5 py-1 rounded text-[9px] uppercase tracking-widest font-black',
                  STATUS_TONE[active.status] ?? 'bg-white/5 text-white/40',
                )}
              >
                {active.status.replace(/_/g, ' ')}
              </span>
            )}
          </div>
        </header>

        <div className="flex-1 relative">
          {active ? (
            view === 'agents' ? (
              <AgentGraph events={graphEvents} />
            ) : (
              <TrainingCanvas events={graphEvents} />
            )
          ) : (
            <EmptyState hasSession={false} />
          )}
        </div>
      </main>

      {/* ── Right: live agent activity ────────────────────────────────── */}
      <aside className="w-[440px] border-l border-white/5 flex flex-col bg-[#050505] min-w-0">
        <AgentActivity sessionId={activeSessionId} />
      </aside>
    </div>
  );
}

function EmptyState({ hasSession }: { hasSession: boolean }) {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="max-w-md text-center space-y-6 px-12">
        <div className="w-16 h-16 mx-auto rounded-2xl border border-white/10 bg-white/[0.02] flex items-center justify-center">
          <Bot className="w-7 h-7 text-white/40" />
        </div>
        <div className="space-y-2">
          <h2 className="text-2xl font-bold text-white">{hasSession ? 'Talk to the agent' : 'Create a session'}</h2>
          <p className="text-[12px] text-white/50 leading-relaxed">
            {hasSession
              ? 'The live agent graph and training view will appear here as the agents work. Use the chat on the right.'
              : 'Start a session from the panel on the left. The agent profiles your data, proposes a plan, trains, evaluates and publishes — all live in this view.'}
          </p>
        </div>
      </div>
    </div>
  );
}
