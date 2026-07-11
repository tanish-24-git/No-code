'use client';

import { cn } from '@/lib/cn';
import { spentUsd, type SessionRecord } from '@/lib/events';

/** Compact spend meter for the workspace header. */
export function BudgetBar({ session }: { session?: SessionRecord | null }) {
  if (!session) return null;
  const spent = spentUsd(session.ledger);
  const budget = session.budgetUsd || 1;
  const frac = Math.min(1, spent / budget);
  const tone = frac >= 1 ? 'bg-danger' : frac >= 0.85 ? 'bg-warn' : 'bg-success';
  return (
    <div className="flex items-center gap-2" title={`$${spent.toFixed(4)} spent of $${budget.toFixed(2)}`}>
      <div className="w-24 h-1.5 rounded-full bg-white/10 overflow-hidden">
        <div className={cn('h-full rounded-full transition-all', tone)} style={{ width: `${frac * 100}%` }} />
      </div>
      <span className="text-[10px] font-mono text-white/50">
        ${spent < 0.01 && spent > 0 ? spent.toFixed(4) : spent.toFixed(2)} / ${budget.toFixed(2)}
      </span>
    </div>
  );
}
