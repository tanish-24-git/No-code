'use client';

/**
 * Model testing playground.
 *
 * Pick a trained model, send a prompt, see the response stream back. The
 * backend route `/api/models/{id}/test` proxies the prompt through the
 * configured LLM provider with a role-play system prompt that reflects
 * the original training context — gives the user a realistic preview
 * before wiring up real inference infra.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import useSWR from 'swr';
import { fetcher, sseUrl } from '@/lib/api';
import type { ModelRecord, Settings } from '@/lib/types';
import { cn } from '@/lib/cn';
import {
  Bot,
  Box,
  CheckCircle2,
  Cpu,
  ExternalLink,
  Loader2,
  Send,
  Settings as SettingsIcon,
  Sliders,
  StopCircle,
  User,
} from 'lucide-react';

type Turn = { role: 'user' | 'assistant'; content: string };

export default function TestPage() {
  const { data: models } = useSWR<ModelRecord[]>('/api/models', fetcher, { refreshInterval: 5000 });
  const { data: settings } = useSWR<Settings>('/api/settings', fetcher);

  const trained = useMemo(() => (models ?? []).filter((m) => m.kind === 'trained'), [models]);
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    if (!activeId && trained.length > 0) setActiveId(trained[0].id);
  }, [trained, activeId]);

  const active = trained.find((m) => m.id === activeId) ?? null;

  return (
    <div className="h-[calc(100vh-52px)] bg-black flex overflow-hidden">
      <aside className="w-[280px] border-r border-white/5 bg-[#050505] flex flex-col">
        <div className="p-4 border-b border-white/5">
          <h2 className="text-[10px] uppercase tracking-[0.3em] font-black text-white/60">
            Trained Models
          </h2>
          <p className="text-[10px] text-white/30 font-medium mt-1">
            Pick one and send it a prompt.
          </p>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {trained.length === 0 && (
            <div className="px-4 py-8 text-center space-y-2">
              <Box className="w-5 h-5 mx-auto text-white/20" />
              <p className="text-[10px] uppercase tracking-widest text-white/30 font-black">
                No trained models yet
              </p>
              <p className="text-[10px] text-white/30 font-medium">
                Train one in the playground first.
              </p>
            </div>
          )}
          {trained.map((m) => (
            <button
              key={m.id}
              onClick={() => setActiveId(m.id)}
              className={cn(
                'w-full text-left px-3 py-2.5 rounded border transition-all',
                m.id === activeId
                  ? 'border-white/20 bg-white/[0.05]'
                  : 'border-transparent hover:border-white/10 hover:bg-white/[0.02]',
              )}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-[11px] font-bold text-white truncate">
                  {m.name ?? m.id.slice(0, 12)}
                </span>
                {m.is_pushed_to_hub && (
                  <CheckCircle2 className="w-3 h-3 text-success" />
                )}
              </div>
              <div className="text-[9px] uppercase tracking-widest text-white/40 font-black">
                {m.hf_repo_id ?? m.local_path?.split(/[/\\]/).pop() ?? 'local'}
              </div>
            </button>
          ))}
        </div>
        <div className="p-3 border-t border-white/5 text-[10px] text-white/40 leading-relaxed font-medium">
          <div className="flex items-center gap-1.5 mb-1">
            <SettingsIcon className="w-3 h-3" />
            <span className="uppercase tracking-widest font-black text-white/60">Test setup</span>
          </div>
          {settings?.is_configured ? (
            <p>
              Backed by <span className="text-white">{settings.llm_provider}</span> ·{' '}
              <span className="text-white">{settings.llm_model}</span>
            </p>
          ) : (
            <p>
              <a href="/settings" className="text-warn hover:underline">
                Configure your LLM provider
              </a>{' '}
              to enable testing.
            </p>
          )}
        </div>
      </aside>

      <main className="flex-1 flex flex-col min-w-0">
        {active ? (
          <Chat key={active.id} model={active} disabled={!settings?.is_configured} />
        ) : (
          <EmptyState />
        )}
      </main>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center px-12">
      <div className="text-center space-y-4 max-w-sm">
        <Bot className="w-10 h-10 text-white/20 mx-auto" />
        <h2 className="text-xl font-bold text-white">No model selected</h2>
        <p className="text-[12px] text-white/50">
          Train a model in the playground, then come back here to chat with it.
        </p>
      </div>
    </div>
  );
}

function Chat({ model, disabled }: { model: ModelRecord; disabled: boolean }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState('');
  const [system, setSystem] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [showSettings, setShowSettings] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [turns]);

  const send = async () => {
    if (!draft.trim() || streaming || disabled) return;
    const promptText = draft.trim();
    setTurns((t) => [...t, { role: 'user', content: promptText }, { role: 'assistant', content: '' }]);
    setDraft('');
    setStreaming(true);

    const ctl = new AbortController();
    abortRef.current = ctl;
    try {
      const res = await fetch(sseUrl(`/api/models/${model.id}/test`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: ctl.signal,
        body: JSON.stringify({
          prompt: promptText,
          system_prompt: system.trim() || undefined,
          temperature,
          max_tokens: maxTokens,
        }),
      });
      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => '');
        setTurns((t) => {
          const c = [...t];
          c[c.length - 1] = { role: 'assistant', content: `[error] ${res.status}: ${text}` };
          return c;
        });
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const dataLines = frame
            .split('\n')
            .filter((l) => l.startsWith('data: '))
            .map((l) => l.slice(6));
          if (dataLines.length === 0) continue;
          const payload = dataLines.join('\n');
          if (payload === '[DONE]') {
            await reader.cancel().catch(() => {});
            return;
          }
          setTurns((t) => {
            const c = [...t];
            const last = c[c.length - 1];
            c[c.length - 1] = { ...last, content: last.content + payload + '\n' };
            return c;
          });
        }
      }
    } catch (e) {
      const name = (e as { name?: string })?.name;
      if (name !== 'AbortError') {
        setTurns((t) => {
          const c = [...t];
          c[c.length - 1] = {
            role: 'assistant',
            content: `[error] ${e instanceof Error ? e.message : String(e)}`,
          };
          return c;
        });
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  };

  const stop = () => {
    abortRef.current?.abort();
    setStreaming(false);
  };

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-black">
      {/* Header */}
      <header className="h-14 border-b border-white/5 px-5 flex items-center gap-4 bg-[#070707]">
        <div className="w-8 h-8 rounded border border-white/10 bg-white/5 flex items-center justify-center">
          <Cpu className="w-4 h-4 text-white/60" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-[12px] font-bold text-white truncate">
            {model.name ?? model.id.slice(0, 12)}
          </div>
          <div className="text-[9px] uppercase tracking-widest text-white/40 font-black">
            {model.hf_repo_id ?? model.local_path ?? 'local'}
          </div>
        </div>
        {model.hf_repo_id && (
          <a
            href={`https://huggingface.co/${model.hf_repo_id}`}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest font-black text-white/40 hover:text-white"
          >
            <ExternalLink className="w-3 h-3" /> on HF
          </a>
        )}
        <button
          onClick={() => setShowSettings((s) => !s)}
          className={cn(
            'flex items-center gap-1.5 px-2.5 py-1.5 rounded border text-[10px] uppercase tracking-widest font-black transition-all',
            showSettings
              ? 'bg-white text-black border-white'
              : 'bg-white/5 text-white/60 border-white/10 hover:border-white/30 hover:text-white',
          )}
        >
          <Sliders className="w-3 h-3" /> tune
        </button>
      </header>

      {showSettings && (
        <div className="border-b border-white/5 bg-[#050505] px-5 py-4 grid grid-cols-1 md:grid-cols-3 gap-4">
          <Field label="Temperature">
            <input
              type="number"
              step="0.05"
              min="0"
              max="2"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-full bg-black border border-white/10 rounded px-3 py-2 text-[11px] text-white font-mono focus:border-white/40 outline-none"
            />
          </Field>
          <Field label="Max tokens">
            <input
              type="number"
              step="64"
              min="32"
              max="4096"
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              className="w-full bg-black border border-white/10 rounded px-3 py-2 text-[11px] text-white font-mono focus:border-white/40 outline-none"
            />
          </Field>
          <Field label="System prompt (optional)">
            <textarea
              rows={2}
              value={system}
              onChange={(e) => setSystem(e.target.value)}
              placeholder="e.g. You are a JSON-only extraction assistant…"
              className="w-full bg-black border border-white/10 rounded px-3 py-2 text-[11px] text-white placeholder:text-white/30 focus:border-white/40 outline-none resize-none"
            />
          </Field>
        </div>
      )}

      {/* Conversation */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-5 space-y-3">
        {turns.length === 0 && <Suggestions onPick={(s) => setDraft(s)} />}
        {turns.map((t, i) => (
          <Bubble key={i} role={t.role} text={t.content} streaming={streaming && i === turns.length - 1 && t.role === 'assistant'} />
        ))}
      </div>

      {/* Input bar */}
      <div className="border-t border-white/5 p-3 flex items-end gap-2 bg-[#070707]">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              send();
            }
          }}
          rows={2}
          placeholder={disabled ? 'Configure your LLM provider in Settings to test' : 'Send a prompt to the model…'}
          disabled={disabled || streaming}
          className="flex-1 bg-black border border-white/10 rounded px-3 py-2 text-[12px] text-white placeholder:text-white/30 focus:border-white/40 outline-none resize-none disabled:opacity-40"
        />
        {streaming ? (
          <button
            onClick={stop}
            className="h-9 px-3 rounded border border-white/20 text-white/70 text-[10px] uppercase tracking-widest font-black hover:bg-white/5 flex items-center gap-1.5"
          >
            <StopCircle className="w-3.5 h-3.5" /> stop
          </button>
        ) : (
          <button
            onClick={send}
            disabled={!draft.trim() || disabled}
            className="h-9 px-4 rounded bg-white text-black text-[10px] uppercase tracking-widest font-black flex items-center gap-1.5 disabled:opacity-30 hover:bg-white/90"
          >
            <Send className="w-3.5 h-3.5" /> send
          </button>
        )}
      </div>
      <div className="px-5 py-1.5 text-[9px] uppercase tracking-widest text-white/30 font-black bg-[#050505] border-t border-white/5">
        Ctrl + Enter to send. Replies come from the trained adapter when available; otherwise the configured LLM provider answers.
      </div>
    </div>
  );
}

function Bubble({ role, text, streaming }: { role: 'user' | 'assistant'; text: string; streaming: boolean }) {
  const isUser = role === 'user';
  return (
    <div className={cn('flex gap-3', isUser && 'flex-row-reverse')}>
      <div
        className={cn(
          'w-8 h-8 shrink-0 rounded flex items-center justify-center border',
          isUser ? 'bg-white text-black border-white' : 'bg-white/5 text-white/70 border-white/10',
        )}
      >
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>
      <div
        className={cn(
          'max-w-[80%] px-4 py-3 rounded-lg text-[13px] leading-relaxed border whitespace-pre-wrap',
          isUser ? 'bg-white text-black border-white/30' : 'bg-white/[0.03] text-white/90 border-white/10',
        )}
      >
        {text || (streaming ? <Loader2 className="w-3.5 h-3.5 animate-spin inline opacity-50" /> : ' ')}
      </div>
    </div>
  );
}

function Suggestions({ onPick }: { onPick: (s: string) => void }) {
  const ideas = [
    'Summarize this in one sentence: …',
    'Translate to Spanish: …',
    'Classify the sentiment of this text: …',
    'Extract entities as JSON: …',
  ];
  return (
    <div className="space-y-3 max-w-2xl mx-auto">
      <p className="text-[10px] uppercase tracking-[0.25em] text-white/30 font-black">Try a prompt</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {ideas.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="text-left px-3 py-3 bg-white/[0.02] border border-white/5 rounded-lg hover:border-white/15 hover:bg-white/[0.04] transition-all text-[12px] text-white/70 leading-relaxed"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label className="text-[9px] uppercase tracking-[0.25em] font-black text-white/40">{label}</label>
      {children}
    </div>
  );
}
