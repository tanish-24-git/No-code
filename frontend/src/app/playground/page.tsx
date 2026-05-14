'use client';

/**
 * Session-driven playground.
 *
 *   left  — session list, dataset upload (auto-starts a session on success)
 *   center — pipeline canvas (read-only preview of the auto-built graph)
 *   right  — AgentActivity: live SSE-driven console (cards + chat)
 */
import { useEffect, useState } from 'react';
import useSWR from 'swr';
import { api, fetcher, uploadFile } from '@/lib/api';
import type {
  AgentSession,
  Dataset,
  DatasetUploadResponse,
  Pipeline,
  SessionListItem,
} from '@/lib/types';
import { ACCEPTED_DATASET_TYPES } from '@/lib/types';
import { AgentActivity } from '@/components/AgentActivity';
import { PipelineCanvas } from '@/components/PipelineCanvas';
import { useSessionEvents } from '@/lib/sse';
import { cn } from '@/lib/cn';
import {
  Database,
  Loader2,
  Play,
  Plus,
  Upload,
  XCircle,
} from 'lucide-react';

const STAGE_TONE: Record<string, string> = {
  init: 'bg-white/5 text-white/40',
  profiling: 'bg-info/10 text-info',
  clarifying: 'bg-warn/10 text-warn',
  planning: 'bg-info/10 text-info',
  awaiting_approval: 'bg-warn/10 text-warn',
  executing: 'bg-info/10 text-info',
  monitoring: 'bg-info/10 text-info',
  recovering: 'bg-warn/10 text-warn',
  evaluating: 'bg-info/10 text-info',
  awaiting_export_choice: 'bg-warn/10 text-warn',
  finalizing: 'bg-info/10 text-info',
  done: 'bg-success/10 text-success',
  failed: 'bg-danger/10 text-danger',
  cancelled: 'bg-white/5 text-white/40',
};

export default function PlaygroundPage() {
  const { data: sessions, mutate: mutateSessions } = useSWR<SessionListItem[]>(
    '/api/sessions',
    fetcher,
    { refreshInterval: 3000 },
  );
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  // Auto-select most recent session if none selected.
  useEffect(() => {
    if (!activeSessionId && sessions && sessions.length > 0) {
      setActiveSessionId(sessions[0].id);
    }
  }, [sessions, activeSessionId]);

  const { data: session, error: sessionError } = useSWR<AgentSession>(
    activeSessionId ? `/api/sessions/${activeSessionId}` : null,
    fetcher,
    { refreshInterval: 1500, shouldRetryOnError: false },
  );

  // If the persisted session ID points to a session that no longer exists
  // (common after `docker compose down -v`), clear it so the auto-select
  // effect can pick the next most recent session or fall back to empty.
  useEffect(() => {
    if (sessionError && activeSessionId) {
      const msg = String((sessionError as Error)?.message ?? '');
      if (msg.startsWith('404')) {
        setActiveSessionId(null);
        mutateSessions();
      }
    }
  }, [sessionError, activeSessionId, mutateSessions]);

  const pipelineId = session?.pipeline_id;
  const { data: pipeline } = useSWR<Pipeline>(
    pipelineId ? `/api/pipelines/${pipelineId}` : null,
    fetcher,
    { refreshInterval: 2500 },
  );

  // Live event stream feeds the canvas so nodes pop in phase-by-phase.
  const { events: sessionEvents } = useSessionEvents(activeSessionId);

  const { data: dataset } = useSWR<Dataset>(
    session?.dataset_id ? `/api/datasets/${session.dataset_id}` : null,
    fetcher,
  );

  const handleFile = async (file: File) => {
    setUploading(true);
    try {
      const r = (await uploadFile('/api/datasets/upload', file)) as DatasetUploadResponse;
      setActiveSessionId(r.session_id);
      mutateSessions();
    } finally {
      setUploading(false);
    }
  };

  const cancelSession = async (id: string) => {
    await api(`/api/sessions/${id}/cancel`, { method: 'POST' }).catch(() => {});
    mutateSessions();
  };

  return (
    <div className="h-[calc(100vh-112px)] bg-black flex overflow-hidden">
      {/* ── Left rail ─────────────────────────────────────────────────── */}
      <aside className="w-[280px] border-r border-white/5 flex flex-col bg-[#050505]">
        {/* Upload hero */}
        <div className="p-4 border-b border-white/5">
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
                {uploading ? 'Processing...' : 'Drop dataset to start'}
              </p>
              <p className="text-[9px] uppercase tracking-widest text-white/30 font-medium">
                csv json jsonl txt md pdf docx html
              </p>
            </div>
          </label>
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
            <span className="text-[9px] uppercase tracking-[0.25em] font-black text-white/40">
              Sessions
            </span>
            <span className="ml-auto text-[10px] font-mono text-white/30">{sessions?.length ?? 0}</span>
          </div>
          <div className="px-2 space-y-1 pb-4">
            {(sessions ?? []).map((s) => {
              const active = s.id === activeSessionId;
              return (
                <div
                  key={s.id}
                  className={cn(
                    'group rounded-md border transition-all',
                    active
                      ? 'border-white/20 bg-white/[0.04]'
                      : 'border-transparent hover:border-white/10 hover:bg-white/[0.02]',
                  )}
                >
                  <button
                    onClick={() => setActiveSessionId(s.id)}
                    className="w-full text-left px-3 py-2.5"
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-[10px] font-mono text-white/40">
                        {s.id.slice(0, 8)}
                      </span>
                      <span
                        className={cn(
                          'px-1.5 py-0.5 rounded text-[8px] uppercase tracking-widest font-black',
                          STAGE_TONE[s.state] ?? 'bg-white/5 text-white/40',
                        )}
                      >
                        {s.state.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div className="text-[10px] text-white/40 font-mono truncate">
                      {new Date(s.updated_at).toLocaleTimeString()}
                    </div>
                  </button>
                  {active && !['done', 'failed', 'cancelled'].includes(s.state) && (
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
                <p className="text-[10px] uppercase tracking-widest text-white/30 font-black">
                  No sessions yet
                </p>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* ── Center: pipeline canvas ───────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        <header className="h-14 px-5 border-b border-white/5 bg-black flex items-center gap-4">
          {dataset ? (
            <>
              <Database className="w-4 h-4 text-white/40" />
              <div className="min-w-0">
                <div className="text-[12px] font-bold text-white truncate">{dataset.name}</div>
                <div className="text-[10px] uppercase tracking-widest text-white/40 font-black">
                  {dataset.row_count} rows · {dataset.column_names.length} cols · {dataset.file_type}
                </div>
              </div>
            </>
          ) : (
            <div className="text-[10px] uppercase tracking-[0.3em] text-white/30 font-black">
              No active session
            </div>
          )}
          {pipeline && (
            <>
              <div className="h-6 w-px bg-white/10" />
              <div className="text-[11px] text-white/70 font-medium">{pipeline.name}</div>
              {pipeline.is_agent_configured && (
                <span className="px-2 py-0.5 bg-success/10 text-success rounded text-[9px] font-black uppercase tracking-widest">
                  Agent built
                </span>
              )}
            </>
          )}
          <div className="ml-auto flex items-center gap-2">
            {session && (
              <span
                className={cn(
                  'px-2.5 py-1 rounded text-[9px] uppercase tracking-widest font-black',
                  STAGE_TONE[session.state] ?? 'bg-white/5 text-white/40',
                )}
              >
                {session.state.replace(/_/g, ' ')}
              </span>
            )}
          </div>
        </header>

        <div className="flex-1 relative">
          {session ? (
            <PipelineCanvas
              pipeline={pipeline}
              events={sessionEvents}
              onGraphChange={() => {}}
              onSelectNode={() => {}}
            />
          ) : (
            <EmptyState />
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

function EmptyState() {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="max-w-md text-center space-y-6 px-12">
        <div className="w-16 h-16 mx-auto rounded-2xl border border-white/10 bg-white/[0.02] flex items-center justify-center">
          <Plus className="w-7 h-7 text-white/40" />
        </div>
        <div className="space-y-2">
          <h2 className="text-2xl font-bold text-white">Drop a dataset</h2>
          <p className="text-[12px] text-white/50 leading-relaxed">
            Upload a CSV / JSON / JSONL file from the panel on the left. Agents will profile it,
            ask any clarifying questions, draft a pipeline, and walk you through training and
            export — all live in this view.
          </p>
        </div>
        <div className="flex items-center justify-center gap-2 text-[10px] uppercase tracking-[0.25em] text-white/40 font-black">
          <Play className="w-3 h-3" />
          One upload starts the agent
        </div>
      </div>
    </div>
  );
}
