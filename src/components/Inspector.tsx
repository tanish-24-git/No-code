'use client';

import type { Pipeline, PipelineConfig } from '@/lib/types';
import { cn } from '@/lib/cn';
import { Info } from 'lucide-react';

type Props = {
  pipeline: Pipeline;
  onPatch: (patch: Partial<PipelineConfig>) => void;
};

const NUMERIC: (keyof PipelineConfig)[] = [
  'epochs',
  'batch_size',
  'learning_rate',
  'max_seq_len',
  'gradient_accumulation',
  'split_ratio',
];

const STRING: (keyof PipelineConfig)[] = ['project_name', 'base_model', 'language'];

const ENUMS: Partial<Record<keyof PipelineConfig, readonly string[]>> = {
  task_type: ['Classification', 'Chat', 'QA', 'Extraction'],
  output_type: ['JSON', 'text', 'label', 'multi-label'],
  domain: ['General', 'Finance', 'Medical', 'Legal', 'Code'],
  training_mode: ['fast', 'balanced', 'high_quality'],
  training_method: ['lora', 'qlora', 'full'],
  precision: ['bf16', 'fp16', 'float32'],
  lora_rank: ['8', '16', '32', '64'],
};

const BOOLS: (keyof PipelineConfig)[] = ['early_stopping', 'class_balancing', 'data_augmentation'];

export function Inspector({ pipeline, onPatch }: Props) {
  const c = pipeline.config;
  const reasoning = pipeline.reasoning ?? {};

  return (
    <div className="space-y-8 pb-12">
      <header className="space-y-2">
        <div className="flex items-center justify-between">
          <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40">Project Identifier</h4>
          {pipeline.is_agent_configured && (
            <span className="px-2 py-0.5 bg-white/10 border border-white/20 rounded text-[8px] font-black uppercase tracking-widest text-white/60">
              Agent Validated
            </span>
          )}
        </div>
        <p className="text-sm font-bold text-white uppercase tracking-widest">{pipeline.name}</p>
      </header>

      <Section title="Deployment Identity">
        {STRING.map((k) => (
          <Field key={k} label={k} reason={reasoning[k]}>
            <input
              className="w-full bg-white/[0.03] border border-white/10 px-3 py-2 text-[11px] text-white focus:border-white/40 outline-none transition-all font-mono"
              value={String(c[k] ?? '')}
              onChange={(e) => onPatch({ [k]: e.target.value } as Partial<PipelineConfig>)}
            />
          </Field>
        ))}
      </Section>

      <Section title="Computation Parameters">
        <div className="grid grid-cols-2 gap-4">
          {NUMERIC.map((k) => (
            <Field key={k} label={k} reason={reasoning[k]}>
              <input
                className="w-full bg-white/[0.03] border border-white/10 px-3 py-2 text-[11px] text-white focus:border-white/40 outline-none transition-all font-mono"
                type="number"
                step={k === 'learning_rate' || k === 'split_ratio' ? '0.01' : '1'}
                value={String(c[k] ?? '')}
                onChange={(e) => onPatch({ [k]: Number(e.target.value) } as unknown as Partial<PipelineConfig>)}
              />
            </Field>
          ))}
        </div>
      </Section>

      <Section title="Model Architecture">
        {((Object.keys(ENUMS) as (keyof PipelineConfig)[]).map((k) => (
          <Field key={k} label={k} reason={reasoning[k]}>
            <select
              className="w-full bg-white/[0.03] border border-white/10 px-3 py-2 text-[11px] text-white focus:border-white/40 outline-none transition-all appearance-none cursor-pointer uppercase tracking-widest font-black"
              value={String(c[k] ?? '')}
              onChange={(e) => {
                const v = e.target.value;
                const cast = k === 'lora_rank' ? Number(v) : v;
                onPatch({ [k]: cast } as unknown as Partial<PipelineConfig>);
              }}
            >
              {ENUMS[k]!.map((opt) => (
                <option key={opt} value={opt} className="bg-black text-white">
                  {opt}
                </option>
              ))}
            </select>
          </Field>
        )))}
      </Section>

      <Section title="System Optimization">
        <div className="space-y-3">
          {BOOLS.map((k) => (
            <div key={k} className="flex items-center justify-between p-3 bg-white/[0.02] border border-white/5 rounded">
               <div className="space-y-0.5">
                  <p className="text-[9px] text-white/30 font-black uppercase tracking-widest">{k.replace(/_/g, ' ')}</p>
                  {reasoning[k] && <p className="text-[8px] text-white/20 italic">{reasoning[k]}</p>}
               </div>
               <button
                  onClick={() => onPatch({ [k]: !c[k] } as Partial<PipelineConfig>)}
                  className={cn(
                    "w-12 h-6 rounded-full border transition-all relative",
                    c[k] ? "bg-white border-white" : "bg-white/5 border-white/10"
                  )}
               >
                  <div className={cn(
                    "absolute top-1 w-4 h-4 rounded-full transition-all",
                    c[k] ? "left-7 bg-black" : "left-1 bg-white/20"
                  )} />
               </button>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <h5 className="text-[9px] font-black uppercase tracking-[0.3em] text-white/20 whitespace-nowrap">{title}</h5>
        <div className="h-px w-full bg-white/5" />
      </div>
      <div className="space-y-4">{children}</div>
    </div>
  );
}

function Field({
  label,
  reason,
  children,
}: {
  label: string;
  reason?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="group space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-[9px] font-black uppercase tracking-widest text-white/40 group-hover:text-white/60 transition-colors">
          {label.replace(/_/g, ' ')}
        </label>
        {reason && (
          <div className="relative group/tip">
            <Info className="w-3 h-3 text-white/20 cursor-help" />
            <div className="absolute right-0 top-full mt-2 w-48 p-2 bg-white text-black text-[9px] font-bold uppercase tracking-tighter opacity-0 group-hover/tip:opacity-100 transition-opacity z-50 pointer-events-none">
              {reason}
            </div>
          </div>
        )}
      </div>
      {children}
    </div>
  );
}
