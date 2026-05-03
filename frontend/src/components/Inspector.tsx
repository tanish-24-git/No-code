'use client';

import type { Pipeline, PipelineConfig } from '@/lib/types';

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
    <div className="p-3 space-y-3 text-xs">
      <div>
        <div className="text-[10px] uppercase tracking-wider text-fg-2 mb-1">pipeline</div>
        <div className="text-fg">{pipeline.name}</div>
        {pipeline.is_agent_configured && (
          <div className="pill mt-1">
            <span className="dot dot-success" />
            agent-configured
          </div>
        )}
      </div>

      <Section title="strings">
        {STRING.map((k) => (
          <Field key={k} label={k} value={String(c[k] ?? '')} reason={reasoning[k]}>
            <input
              className="input"
              value={String(c[k] ?? '')}
              onChange={(e) => onPatch({ [k]: e.target.value } as Partial<PipelineConfig>)}
            />
          </Field>
        ))}
      </Section>

      <Section title="numbers">
        {NUMERIC.map((k) => (
          <Field key={k} label={k} value={String(c[k] ?? '')} reason={reasoning[k]}>
            <input
              className="input"
              type="number"
              step={k === 'learning_rate' || k === 'split_ratio' ? '0.01' : '1'}
              value={String(c[k] ?? '')}
              onChange={(e) => onPatch({ [k]: Number(e.target.value) } as unknown as Partial<PipelineConfig>)}
            />
          </Field>
        ))}
      </Section>

      <Section title="enums">
        {(Object.keys(ENUMS) as (keyof PipelineConfig)[]).map((k) => (
          <Field key={k} label={k} value={String(c[k] ?? '')} reason={reasoning[k]}>
            <select
              className="select"
              value={String(c[k] ?? '')}
              onChange={(e) => {
                const v = e.target.value;
                const cast = k === 'lora_rank' ? Number(v) : v;
                onPatch({ [k]: cast } as unknown as Partial<PipelineConfig>);
              }}
            >
              {ENUMS[k]!.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </Field>
        ))}
      </Section>

      <Section title="flags">
        {BOOLS.map((k) => (
          <Field key={k} label={k} value={c[k] ? 'on' : 'off'} reason={reasoning[k]}>
            <button
              className={`btn w-full justify-center ${c[k] ? 'btn-primary' : ''}`}
              onClick={() => onPatch({ [k]: !c[k] } as Partial<PipelineConfig>)}
            >
              {c[k] ? 'enabled' : 'disabled'}
            </button>
          </Field>
        ))}
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="border border-border rounded bg-bg-2/40 overflow-hidden" open>
      <summary className="px-2 py-1 text-[10px] uppercase tracking-wider text-fg-2 cursor-pointer select-none">
        {title}
      </summary>
      <div className="p-2 space-y-2">{children}</div>
    </details>
  );
}

function Field({
  label,
  value: _v,
  reason,
  children,
}: {
  label: string;
  value: string;
  reason?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center justify-between">
        <span className="text-fg-2">{label}</span>
        {reason && (
          <span className="text-[10px] text-fg-3" title={reason}>
            ⓘ
          </span>
        )}
      </div>
      <div className="mt-1">{children}</div>
      {reason && <div className="mt-1 text-[10px] text-fg-3">{reason}</div>}
    </div>
  );
}
