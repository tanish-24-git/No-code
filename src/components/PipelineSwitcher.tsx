'use client';

import { useState } from 'react';
import { api } from '@/lib/api';
import type { Pipeline } from '@/lib/types';

type Props = {
  pipelines: Pipeline[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onCreated: (pipeline: Pipeline) => void;
};

export function PipelineSwitcher({ pipelines, activeId, onSelect, onCreated }: Props) {
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState('untitled pipeline');

  const create = async () => {
    setCreating(true);
    try {
      const p = await api<Pipeline>('/api/pipelines', {
        method: 'POST',
        body: JSON.stringify({ name }),
      });
      onCreated(p);
      setName('untitled pipeline');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="flex items-center gap-2 px-4 py-2 border-b border-border bg-bg-2/40">
      <select
        className="bg-bg-3 border border-border rounded px-2 py-1 text-xs text-fg max-w-[260px]"
        value={activeId ?? ''}
        onChange={(e) => onSelect(e.target.value)}
      >
        {pipelines.length === 0 && <option value="">no pipelines yet</option>}
        {pipelines.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name} {p.is_agent_configured ? '[built]' : ''}
          </option>
        ))}
      </select>
      <input
        className="bg-bg-3 border border-border rounded px-2 py-1 text-xs text-fg w-[200px]"
        placeholder="new pipeline name"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <button className="btn btn-primary" onClick={create} disabled={!name || creating}>
        + new
      </button>
    </div>
  );
}
