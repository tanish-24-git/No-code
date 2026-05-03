'use client';

import { useState } from 'react';
import useSWR from 'swr';
import { fetcher, uploadFile, api } from '@/lib/api';
import type { Dataset } from '@/lib/types';
import { Upload, Database, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/cn';

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
    <div className="bg-black/80 backdrop-blur-xl border border-white/10 rounded-lg p-1 flex items-center gap-1 shadow-2xl">
      <label className={cn(
        "flex items-center gap-2 px-3 py-1.5 rounded cursor-pointer transition-all",
        uploading ? "bg-white/5 opacity-50" : "hover:bg-white/5 active:bg-white/10"
      )}>
        <input
          type="file"
          accept=".csv,.json,.jsonl"
          className="hidden"
          onChange={(e) => e.target.files && handleFile(e.target.files[0])}
          disabled={uploading}
        />
        <Upload className="w-3.5 h-3.5 text-white/60" />
        <span className="text-[10px] font-black uppercase tracking-widest text-white/80">
          {uploading ? 'Processing...' : 'Upload'}
        </span>
      </label>

      <div className="w-px h-4 bg-white/10 mx-1" />

      <div className="relative group">
        <Database className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-white/30" />
        <select
          className="bg-transparent border-none pl-9 pr-8 py-1.5 text-[10px] font-black uppercase tracking-widest text-white/60 hover:text-white transition-all appearance-none cursor-pointer outline-none min-w-[180px]"
          value={attachedId ?? ''}
          onChange={(e) => onAttach(e.target.value)}
        >
          <option value="" className="bg-black text-white/40">Select Source</option>
          {list?.map((d) => (
            <option key={d.id} value={d.id} className="bg-black text-white">
              {d.name.toUpperCase()} ({d.row_count})
            </option>
          ))}
        </select>
        <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-white/20 pointer-events-none" />
      </div>
    </div>
  );
}
