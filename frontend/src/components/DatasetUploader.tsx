'use client';

import { useState } from 'react';
import useSWR from 'swr';
import { fetcher, uploadFile, api } from '@/lib/api';
import type { Dataset } from '@/lib/types';

type Props = {
  pipelineId: string | null;
  attachedId: string | null;
  onAttach: (datasetId: string) => void;
};

export function DatasetUploader({ pipelineId, attachedId, onAttach }: Props) {
  const { data: list, mutate } = useSWR<Dataset[]>('/api/datasets', fetcher);
  const [uploading, setUploading] = useState(false);

  const handleFile = async (file: File) => {
    setUploading(true);
    try {
      const r = (await uploadFile('/api/datasets/upload', file)) as Dataset;
      await mutate();
      if (pipelineId) {
        await api(`/api/pipelines/${pipelineId}`, {
          method: 'PUT',
          body: JSON.stringify({ config: { dataset_id: r.id } }),
        });
        onAttach(r.id);
      }
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="px-4 py-3 border-b border-border bg-bg-2/40">
      <div className="flex items-center gap-3">
        <label className="btn cursor-pointer">
          <input
            type="file"
            accept=".csv,.json,.jsonl"
            className="hidden"
            onChange={(e) => e.target.files && handleFile(e.target.files[0])}
            disabled={uploading}
          />
          {uploading ? 'uploading…' : 'upload dataset'}
        </label>
        <select
          className="bg-bg-3 border border-border rounded px-2 py-1 text-xs text-fg max-w-[260px]"
          value={attachedId ?? ''}
          onChange={(e) => onAttach(e.target.value)}
        >
          <option value="">no dataset attached</option>
          {list?.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name} · {d.row_count} rows
            </option>
          ))}
        </select>
        <span className="text-[11px] text-fg-3">CSV · JSON · JSONL</span>
      </div>
    </div>
  );
}
