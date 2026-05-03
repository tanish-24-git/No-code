'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import useSWR from 'swr';
import { fetcher } from '@/lib/api';
import type { Health } from '@/lib/types';

const HIDDEN_ON = new Set(['/setup', '/settings']);

export function ConfigBanner() {
  const pathname = usePathname();
  const { data, error } = useSWR<Health>('/health', fetcher, { refreshInterval: 15_000 });

  if (HIDDEN_ON.has(pathname)) return null;
  if (error) return null;            // Nav already shows backend-offline state.
  if (!data) return null;
  if (data.llm.configured) return null;

  return (
    <div className="bg-warn-dim border-b border-warn/30 text-warn px-6 py-2 text-[12px] flex items-center gap-3 z-40 sticky top-[52px]">
      <span className="dot dot-warn" />
      <span className="text-fg">
        Agent is not configured. Set <code className="text-warn">LLM_PROVIDER</code>,{' '}
        <code className="text-warn">LLM_API_KEY</code>, and{' '}
        <code className="text-warn">LLM_MODEL</code> in <code className="text-warn">backend/.env</code>, or run setup.
      </span>
      <Link href="/setup" className="ml-auto btn btn-primary">
        run setup
      </Link>
    </div>
  );
}
