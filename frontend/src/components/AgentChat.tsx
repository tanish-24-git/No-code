'use client';

import { useEffect, useRef, useState } from 'react';
import useSWR from 'swr';
import { api, fetcher } from '@/lib/api';
import type { ChatMessage, Inference } from '@/lib/types';

type Props = {
  pipelineId?: string;
  inferenceId?: string;
  datasetId?: string;
  onPipelineChanged?: () => void;
};

const KICKSTART = `Hi — I can read your inference endpoints, hardware, and the active pipeline.\n\nTry: "look at my inference endpoints and suggest generation metrics for the local Ollama one" or "configure this pipeline for my dataset".`;

export function AgentChat({ pipelineId, inferenceId, datasetId, onPipelineChanged }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [focusInfId, setFocusInfId] = useState<string | undefined>(inferenceId);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const { data: inferences } = useSWR<Inference[]>('/api/inferences', fetcher);
  const settings = useSWR<{ is_configured: boolean }>('/api/settings', fetcher).data;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, streaming]);

  const send = async () => {
    if (!draft.trim() || streaming) return;
    if (!settings?.is_configured) {
      setMessages((m) => [...m, { role: 'assistant', content: 'LLM is not configured. Open Settings or run /setup first.' }]);
      return;
    }
    const next: ChatMessage[] = [...messages, { role: 'user', content: draft.trim() }];
    setMessages(next);
    setDraft('');
    setStreaming(true);

    const ctl = new AbortController();
    abortRef.current = ctl;

    try {
      const res = await fetch('/api/agent/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: ctl.signal,
        body: JSON.stringify({
          messages: next,
          pipeline_id: pipelineId,
          inference_id: focusInfId,
          dataset_id: datasetId,
        }),
      });
      if (!res.ok || !res.body) {
        const text = await res.text();
        setMessages((m) => [...m, { role: 'assistant', content: `[error] ${res.status}: ${text}` }]);
        setStreaming(false);
        return;
      }
      // Append empty assistant message and stream into it.
      setMessages((m) => [...m, { role: 'assistant', content: '' }]);

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        // Parse SSE frames separated by blank lines.
        let idx;
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const lines = frame.split('\n');
          const dataLines = lines
            .filter((l) => l.startsWith('data: '))
            .map((l) => l.slice(6));
          if (dataLines.length === 0) continue;
          const payload = dataLines.join('\n');
          if (payload === '[DONE]') {
            await reader.cancel().catch(() => {});
            setStreaming(false);
            onPipelineChanged?.();
            return;
          }
          setMessages((m) => {
            const copy = [...m];
            const last = copy[copy.length - 1];
            copy[copy.length - 1] = { ...last, content: last.content + payload + '\n' };
            return copy;
          });
        }
      }
    } catch (e) {
      if ((e as { name?: string })?.name !== 'AbortError') {
        setMessages((m) => [...m, { role: 'assistant', content: `[error] ${e instanceof Error ? e.message : String(e)}` }]);
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
      onPipelineChanged?.();
    }
  };

  const stop = () => {
    abortRef.current?.abort();
    setStreaming(false);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border">
        <span className="dot dot-success" />
        <span className="text-[11px] uppercase tracking-wider text-fg-2">agent</span>
        <select
          className="ml-auto bg-bg-3 border border-border rounded px-2 py-1 text-[11px] text-fg max-w-[180px]"
          value={focusInfId ?? ''}
          onChange={(e) => setFocusInfId(e.target.value || undefined)}
        >
          <option value="">no focused endpoint</option>
          {inferences?.map((it) => (
            <option key={it.id} value={it.id}>
              focus · {it.name}
            </option>
          ))}
        </select>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.length === 0 && (
          <div className="text-fg-2 text-xs whitespace-pre-line">{KICKSTART}</div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={m.role === 'user' ? 'text-fg' : 'text-fg-2'}>
            <div className="text-[10px] uppercase tracking-wider mb-1">
              {m.role === 'user' ? 'you' : 'agent'}
            </div>
            <div className="text-xs whitespace-pre-wrap leading-relaxed">{m.content}</div>
          </div>
        ))}
      </div>

      <div className="border-t border-border p-2 space-y-2">
        <textarea
          className="textarea h-[68px] resize-none"
          placeholder="ask about your inference, dataset, or pipeline…"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              send();
            }
          }}
          disabled={streaming}
        />
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-fg-3">⌘/ctrl + enter</span>
          <button className="btn ml-auto" onClick={stop} disabled={!streaming}>stop</button>
          <button className="btn btn-primary" onClick={send} disabled={!draft.trim() || streaming}>
            {streaming ? 'streaming…' : 'send'}
          </button>
        </div>
      </div>
    </div>
  );
}
