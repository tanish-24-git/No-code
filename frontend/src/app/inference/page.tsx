'use client';

import { useState } from 'react';
import useSWR from 'swr';
import { api, fetcher } from '@/lib/api';
import type { Inference, InferenceKind } from '@/lib/types';

const KIND_LABELS: Record<InferenceKind, string> = {
  ollama: 'Ollama',
  openai_compat: 'OpenAI-compatible',
  huggingface_inference: 'Hugging Face Inference',
  anthropic: 'Anthropic',
};

const PRESET_BASE_URL: Record<InferenceKind, string> = {
  ollama: 'http://localhost:11434',
  openai_compat: 'http://localhost:8080',
  huggingface_inference: 'https://api-inference.huggingface.co',
  anthropic: 'https://api.anthropic.com',
};

export default function InferencePage() {
  const { data: list, mutate } = useSWR<Inference[]>('/api/inferences', fetcher);
  const [showAdd, setShowAdd] = useState(false);

  return (
    <div className="max-w-[1100px] mx-auto px-8 py-10 space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="font-sans font-bold text-2xl">Inference endpoints</h1>
          <p className="text-fg-2 text-sm mt-1">
            Register the inference servers you actually run. The agent reads this list as a tool and
            recommends generation metrics tuned to each one.
          </p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowAdd(true)}>
          + add endpoint
        </button>
      </header>

      {showAdd && <AddForm onClose={() => setShowAdd(false)} onSaved={() => { setShowAdd(false); mutate(); }} />}

      {!list ? (
        <div className="text-fg-2">loading…</div>
      ) : list.length === 0 ? (
        <div className="card text-fg-2 text-sm">No endpoints yet. Click <em>+ add endpoint</em> to register your first.</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {list.map((it) => (
            <InferenceCard key={it.id} item={it} onChanged={mutate} />
          ))}
        </div>
      )}
    </div>
  );
}

function InferenceCard({ item, onChanged }: { item: Inference; onChanged: () => void }) {
  const [probing, setProbing] = useState(false);
  const probe = async () => {
    setProbing(true);
    try {
      await api(`/api/inferences/${item.id}/probe`, { method: 'POST' });
      onChanged();
    } finally {
      setProbing(false);
    }
  };
  const remove = async () => {
    if (!confirm(`Delete endpoint "${item.name}"?`)) return;
    await api(`/api/inferences/${item.id}`, { method: 'DELETE' });
    onChanged();
  };

  const reach = item.last_probe;
  const reachDot = !reach ? 'dot-warn' : reach.reachable ? 'dot-success' : 'dot-danger';
  const metrics = Object.entries(item.suggested_metrics ?? {});

  return (
    <div className="card space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-medium">{item.name}</div>
          <div className="text-[11px] text-fg-2">{KIND_LABELS[item.kind]} · {item.base_url}</div>
        </div>
        <span className="pill">
          <span className={`dot ${reachDot}`} />
          {!reach ? 'unprobed' : reach.reachable ? `${Math.round(reach.latency_ms ?? 0)}ms` : 'unreachable'}
        </span>
      </div>

      {item.default_model && (
        <div className="text-[11px] text-fg-2">default · {item.default_model}</div>
      )}

      {reach?.models?.length ? (
        <div className="text-[11px] text-fg-2">
          models · {reach.models.slice(0, 6).join(', ')}{reach.models.length > 6 ? ` +${reach.models.length - 6}` : ''}
        </div>
      ) : null}

      {metrics.length > 0 && (
        <div className="border-t border-border pt-3">
          <div className="label mb-2">agent-suggested metrics</div>
          <div className="grid grid-cols-2 gap-2 text-[11px]">
            {metrics.map(([k, v]) => (
              <div key={k} className="flex justify-between gap-2 bg-bg-3 px-2 py-1 rounded">
                <span className="text-fg-2">{k}</span>
                <span className="text-fg">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-2 pt-1">
        <button className="btn" onClick={probe} disabled={probing}>{probing ? 'probing…' : 'probe'}</button>
        <button className="btn btn-danger ml-auto" onClick={remove}>delete</button>
      </div>
    </div>
  );
}

function AddForm({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const [kind, setKind] = useState<InferenceKind>('ollama');
  const [name, setName] = useState('');
  const [baseUrl, setBaseUrl] = useState(PRESET_BASE_URL.ollama);
  const [apiKey, setApiKey] = useState('');
  const [defaultModel, setDefaultModel] = useState('');
  const [notes, setNotes] = useState('');
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const save = async () => {
    setSaving(true);
    setErr(null);
    try {
      await api('/api/inferences', {
        method: 'POST',
        body: JSON.stringify({
          name,
          kind,
          base_url: baseUrl,
          api_key: apiKey || null,
          default_model: defaultModel || null,
          notes: notes || null,
        }),
      });
      onSaved();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm uppercase tracking-wider text-fg">add endpoint</h3>
        <button className="text-fg-2 text-xs hover:text-fg" onClick={onClose}>close</button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label className="label">kind</label>
          <select
            className="select"
            value={kind}
            onChange={(e) => {
              const k = e.target.value as InferenceKind;
              setKind(k);
              setBaseUrl(PRESET_BASE_URL[k]);
            }}
          >
            {(Object.keys(KIND_LABELS) as InferenceKind[]).map((k) => (
              <option key={k} value={k}>
                {KIND_LABELS[k]}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">name</label>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="local llama 3.1" />
        </div>
        <div className="md:col-span-2">
          <label className="label">base URL</label>
          <input className="input" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
        </div>
        <div>
          <label className="label">api key (optional)</label>
          <input className="input" type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
        </div>
        <div>
          <label className="label">default model (optional)</label>
          <input className="input" value={defaultModel} onChange={(e) => setDefaultModel(e.target.value)} placeholder="llama3.1:8b" />
        </div>
        <div className="md:col-span-2">
          <label className="label">notes (optional)</label>
          <input className="input" value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="purpose, hardware notes…" />
        </div>
      </div>
      {err && <div className="text-danger text-xs">✗ {err}</div>}
      <div className="flex gap-2">
        <button className="btn btn-primary" onClick={save} disabled={!name || !baseUrl || saving}>
          {saving ? 'saving…' : 'save endpoint'}
        </button>
        <button className="btn" onClick={onClose}>cancel</button>
      </div>
    </div>
  );
}
