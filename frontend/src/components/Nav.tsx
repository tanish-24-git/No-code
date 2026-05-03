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
    ? data.hardware?.device === 'cuda'
      ? `cuda · ${data.hardware?.gpu_name ?? 'gpu'}`
      : data.hardware?.device === 'mps'
      ? 'mps'
      : 'cpu'
    : 'connecting…';

  return (
    <nav className="fixed top-6 inset-x-8 z-50 flex items-center justify-center pointer-events-none">
      {/* Center Nav */}
      <div className="flex items-center gap-1 p-1 bg-white/5 border border-white/10 rounded-lg backdrop-blur-md pointer-events-auto">
        {links.map((l) => {
          const active = pathname === l.href || (l.href !== '/' && pathname.startsWith(l.href));
          return (
            <Link
              key={l.href}
              href={l.href}
              className={cn(
                'px-4 py-1 rounded text-[10px] uppercase tracking-widest transition-all',
                active
                  ? 'bg-white text-black font-bold'
                  : 'text-fg-3 hover:text-fg hover:bg-white/5',
              )}
            >
              {l.label}
            </Link>
          );
        })}
      </div>

      {/* Right Status (Floating absolute to keep nav centered) */}
      <div className="absolute right-0 flex items-center gap-4 pointer-events-auto">
        <div className="flex items-center gap-2 px-3 py-1 bg-white/5 border border-white/10 rounded text-[10px] uppercase tracking-widest text-fg-2">
          <span className={cn('w-1.5 h-1.5 rounded-full bg-white', dot)} />
          <span>{label}</span>
        </div>
      </div>
    </nav>
  );
}
