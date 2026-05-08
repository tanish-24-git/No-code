'use client';

import { useEffect, useRef, useState } from 'react';

type Props = { jobId: string | null };

const COLOR = (line: string) => {
  if (line.startsWith('[ERROR]') || line.startsWith('[FATAL]')) return 'text-danger';
  if (line.startsWith('[WARN]')) return 'text-warn';
  if (line.startsWith('[METRIC]')) return 'text-info';
  if (line.startsWith('[STEP]')) return 'text-success';
  if (line.startsWith('[DONE]')) return 'text-success';
  return 'text-fg-2';
};

export function LogPanel({ jobId }: Props) {
  const [lines, setLines] = useState<string[]>([]);
  const [done, setDone] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLines([]);
    setDone(false);
    if (!jobId) return;
    const es = new EventSource(`/api/jobs/${jobId}/logs`);
    es.onmessage = (e) => {
      if (e.data === '[DONE]') {
        setDone(true);
        es.close();
        return;
      }
      setLines((prev) => [...prev, e.data]);
    };
    es.onerror = () => {
      es.close();
    };
    return () => es.close();
  }, [jobId]);

  const isAtBottom = useRef(true);

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const threshold = 100;
    isAtBottom.current = scrollHeight - scrollTop - clientHeight < threshold;
  };

  useEffect(() => {
    if (isAtBottom.current) {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
    }
  }, [lines.length]);

  return (
    <div className="flex flex-col h-full bg-bg-2/60 border border-border rounded">
      <div className="px-3 py-2 border-b border-border flex items-center gap-2 text-[11px]">
        <span className={`dot ${done ? 'dot-success' : jobId ? 'dot-warn' : 'dot-info'}`} />
        <span className="uppercase tracking-wider text-fg-2">job logs</span>
        {jobId && <span className="ml-auto text-fg-3 font-mono text-[10px]">{jobId.slice(0, 8)}</span>}
      </div>
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-3 font-mono text-[11px] space-y-0.5"
      >
        {!jobId && <div className="text-fg-3">click run to start a job — logs stream here.</div>}
        {lines.map((l, i) => (
          <div key={i} className={COLOR(l)}>
            {l}
          </div>
        ))}
      </div>
    </div>
  );
}
