'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import useSWR from 'swr';
import { fetcher } from '@/lib/api';
import type { Health } from '@/lib/types';
import { cn } from '@/lib/cn';

const links = [
  { href: '/', label: 'home' },
  { href: '/playground', label: 'playground' },
  { href: '/inference', label: 'inference' },
  { href: '/models', label: 'models' },
  { href: '/settings', label: 'settings' },
];

export function Nav() {
  const pathname = usePathname();
  const { data, error } = useSWR<Health>('/health', fetcher, { refreshInterval: 30_000 });
  const dot = error ? 'dot-danger' : 'dot-success';
  const label = error
    ? 'backend offline'
    : data
    ? data.hardware.device === 'cuda'
      ? `local · cuda · ${data.hardware.gpu_name ?? 'gpu'}`
      : data.hardware.device === 'mps'
      ? 'local · mps'
      : 'local · cpu'
    : 'connecting…';

  return (
    <nav className="fixed top-0 inset-x-0 h-[52px] bg-bg/90 border-b border-border backdrop-blur z-50 flex items-center px-7">
      <Link href="/" className="font-sans font-extrabold text-[15px] tracking-tight flex items-center gap-2">
        <span className={cn('dot animate-pulse', error ? 'dot-danger' : 'dot-success')} />
        FineTune Studio
      </Link>
      <div className="ml-auto flex items-center gap-0.5">
        {links.map((l) => {
          const active = pathname === l.href || (l.href !== '/' && pathname.startsWith(l.href));
          return (
            <Link
              key={l.href}
              href={l.href}
              className={cn(
                'px-3.5 py-1.5 rounded text-[12px] tracking-wider border transition-colors',
                active
                  ? 'text-fg bg-bg-3 border-border-2'
                  : 'text-fg-2 hover:text-fg hover:bg-bg-3 hover:border-border border-transparent',
              )}
            >
              {l.label}
            </Link>
          );
        })}
      </div>
      <div className="ml-5 pl-5 border-l border-border flex items-center gap-2 text-[11px] text-fg-3">
        <span className={cn('dot', dot)} />
        <span>{label}</span>
      </div>
    </nav>
  );
}
