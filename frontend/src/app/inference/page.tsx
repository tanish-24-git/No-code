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
    <div className="pt-24 px-8 pb-12 min-h-screen bg-black">
      <div className="max-w-[1400px] mx-auto space-y-12">
        <header className="flex flex-col md:flex-row md:items-end justify-between gap-8">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="px-3 py-1 bg-white/5 border border-white/10 rounded flex items-center gap-2">
                <span className="text-[10px] font-black tracking-[0.2em] uppercase text-white/40">Network</span>
              </div>
            </div>
            <h1 className="text-5xl font-bold tracking-tight text-white uppercase">
              Inference <br/> Endpoints
            </h1>
            <p className="text-white/40 text-lg max-w-xl font-light">
              Register and monitor your generation infrastructure. Connect local Ollama instances or remote API providers.
            </p>
          </div>
          <button 
            className="flex items-center gap-3 px-6 py-3 bg-white text-black rounded hover:bg-white/90 transition-all font-bold"
            onClick={() => setShowAdd(true)}
          >
            <span className="text-[10px] uppercase font-black tracking-[0.2em]">Add Endpoint</span>
          </button>
        </header>

        {showAdd && <AddForm onClose={() => setShowAdd(false)} onSaved={() => { setShowAdd(false); mutate(); }} />}

        {!list ? (
          <div className="text-white/20 uppercase text-[10px] tracking-widest font-black">Scanning network…</div>
        ) : list.length === 0 ? (
          <div className="p-12 border border-white/5 rounded-xl bg-white/[0.01] text-center">
            <p className="text-white/20 text-sm font-medium uppercase tracking-widest">No endpoints registered.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {list.map((it) => (
              <InferenceCard key={it.id} item={it} onChanged={mutate} />
            ))}
          </div>
        )}
      </div>
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
  const reachColor = !reach ? "text-white/20" : reach.reachable ? "text-white" : "text-white/10";
  const metrics = Object.entries(item.suggested_metrics ?? {});

  return (
    <div className="group p-6 bg-white/[0.02] border border-white/5 rounded-xl hover:bg-white/[0.04] hover:border-white/20 transition-all duration-500">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h3 className="text-lg font-bold text-white mb-1">{item.name}</h3>
          <div className="text-[9px] text-white/30 font-black uppercase tracking-widest">
            {KIND_LABELS[item.kind]} · {item.base_url}
          </div>
        </div>
        <div className={cn(
          "px-3 py-1 rounded text-[9px] font-black uppercase tracking-widest border border-white/10",
          reachColor
        )}>
          {!reach ? 'unprobed' : reach.reachable ? `${Math.round(reach.latency_ms ?? 0)}ms` : 'offline'}
        </div>
      </div>

      {item.default_model && (
        <div className="mb-6">
          <p className="text-[9px] text-white/20 uppercase tracking-widest font-black mb-1">Default Model</p>
          <p className="text-xs text-white/60 font-bold">{item.default_model}</p>
        </div>
      )}

      {metrics.length > 0 && (
        <div className="space-y-3 mb-8">
          <p className="text-[9px] text-white/20 uppercase tracking-widest font-black">Optimization Parameters</p>
          <div className="grid grid-cols-2 gap-2">
            {metrics.map(([k, v]) => (
              <div key={k} className="flex justify-between items-center bg-white/5 px-2 py-1.5 rounded border border-white/5">
                <span className="text-[9px] text-white/30 uppercase tracking-tight font-bold">{k}</span>
                <span className="text-[10px] text-white font-mono">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-3 pt-6 border-t border-white/5">
        <button 
          className="flex-1 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded text-[9px] font-black uppercase tracking-widest text-white transition-all"
          onClick={probe} 
          disabled={probing}
        >
          {probing ? 'probing…' : 'probe network'}
        </button>
        <button 
          className="px-4 py-2 hover:bg-white/5 text-white/20 hover:text-white transition-all rounded"
          onClick={remove}
        >
          <span className="text-[9px] font-black uppercase tracking-widest">Delete</span>
        </button>
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
