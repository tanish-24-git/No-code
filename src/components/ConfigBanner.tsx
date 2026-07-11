'use client';

import useSWR from 'swr';
import { fetcher } from '@/lib/api';

interface HealthInfo {
  status: string;
  llm: {
    configured: boolean;
    missing: string[];
    model: string | null;
    freeTier: boolean;
  };
  uv: { found: boolean; version?: string };
  gpu: { present: boolean; name?: string };
}

export function ConfigBanner() {
  const { data, error } = useSWR<HealthInfo>('/api/health', fetcher, { refreshInterval: 30_000 });

  if (error || !data) return null;

  if (!data.llm.configured) {
    return (
      <div className="bg-warn-dim border-b border-warn/30 text-warn px-6 py-2 text-[12px] flex items-center gap-3 z-40 sticky top-[52px]">
        <span className="dot dot-warn" />
        <span className="text-fg">
          Agent is not configured. Set{' '}
          {data.llm.missing.map((k, i) => (
            <span key={k}>
              {i > 0 && ', '}
              <code className="text-warn">{k}</code>
            </span>
          ))}{' '}
          in <code className="text-warn">.env</code> at the project root, then restart{' '}
          <code className="text-warn">npm run dev</code>. Free-tier key? Set both prices to{' '}
          <code className="text-warn">0</code>.
        </span>
      </div>
    );
  }

  if (!data.uv.found) {
    return (
      <div className="bg-warn-dim border-b border-warn/30 text-warn px-6 py-2 text-[12px] flex items-center gap-3 z-40 sticky top-[52px]">
        <span className="dot dot-warn" />
        <span className="text-fg">
          <code className="text-warn">uv</code> was not found on PATH — the agent needs it to run
          generated Python. Install it from{' '}
          <a
            href="https://docs.astral.sh/uv/getting-started/installation/"
            target="_blank"
            rel="noreferrer"
            className="underline text-warn"
          >
            astral.sh/uv
          </a>{' '}
          and restart.
        </span>
      </div>
    );
  }

  return null;
}
