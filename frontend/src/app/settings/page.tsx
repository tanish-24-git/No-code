'use client';

import { useEffect, useMemo, useState } from 'react';
import useSWR from 'swr';
import { api, fetcher } from '@/lib/api';
import type { Provider, ProviderInfo, Settings } from '@/lib/types';

export default function SettingsPage() {
  const { data, mutate } = useSWR<Settings>('/api/settings', fetcher);
  const { data: providers } = useSWR<ProviderInfo[]>('/api/settings/providers', fetcher);

  const [provider, setProvider] = useState<Provider>('anthropic');
  const [model, setModel] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [hfToken, setHfToken] = useState('');
  const [verifying, setVerifying] = useState<'llm' | 'hf' | null>(null);
  const [verifyMsg, setVerifyMsg] = useState<string | null>(null);

  const activeSpec = useMemo(
    () => providers?.find((p) => p.name === provider),
    [providers, provider],
  );

  // Hydrate from server state once on first load.
  useEffect(() => {
    if (!data) return;
    if (data.llm_provider) setProvider(data.llm_provider);
    if (data.llm_model) setModel(data.llm_model);
    if (data.llm_base_url) setBaseUrl(data.llm_base_url);
  }, [data]);

  // When the provider changes and the user hasn't typed a model yet, suggest one.
  useEffect(() => {
    if (!activeSpec) return;
    if (!model || (data?.llm_model !== model && !activeSpec.sample_models.includes(model))) {
      // keep user's freeform value if they typed it
    }
    if (!model && activeSpec.sample_models.length > 0) {
      setModel(activeSpec.sample_models[0]);
    }
  }, [activeSpec]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!data || !providers) return <div className="p-8 text-fg-2">loading...</div>;

  const saveLLM = async () => {
    await api('/api/settings/llm', {
      method: 'POST',
      body: JSON.stringify({
        provider,
        model,
        base_url: baseUrl || null,
        api_key: apiKey || undefined,
      }),
    });
    setApiKey('');
    mutate();
  };
  const clearLLM = async () => {
    await api('/api/settings/llm', { method: 'DELETE' });
    mutate();
  };
  const saveHF = async () => {
    await api('/api/settings/hf-token', { method: 'POST', body: JSON.stringify({ token: hfToken }) });
    setHfToken('');
    mutate();
  };
  const clearHF = async () => {
    await api('/api/settings/hf-token', { method: 'DELETE' });
    mutate();
  };

  const verifyLLM = async () => {
    setVerifying('llm');
    setVerifyMsg(null);
    try {
      const r = await api<{ valid: boolean; detail: string | null; models?: string[] }>(
        '/api/settings/verify-llm',
        { method: 'POST' },
      );
      setVerifyMsg(
        r.valid
          ? `OK - ${r.detail}${r.models && r.models.length ? ` - ${r.models.length} models reachable` : ''}`
          : `Failed - ${r.detail || 'invalid'}`,
      );
    } catch (e) {
      setVerifyMsg(`Failed - ${e instanceof Error ? e.message : 'error'}`);
    } finally {
      setVerifying(null);
    }
  };

  const verifyHF = async () => {
    setVerifying('hf');
    setVerifyMsg(null);
    try {
      const r = await api<{ valid: boolean; username: string | null; detail: string | null }>(
        '/api/settings/verify-hf',
        { method: 'POST' },
      );
      setVerifyMsg(r.valid ? `OK - HF token valid - ${r.username}` : `Failed - ${r.detail || 'invalid'}`);
      mutate();
    } catch (e) {
      setVerifyMsg(`Failed - ${e instanceof Error ? e.message : 'error'}`);
    } finally {
      setVerifying(null);
    }
  };

  const updateFlag = async (patch: Partial<Settings>) => {
    await api('/api/settings', { method: 'PUT', body: JSON.stringify(patch) });
    mutate();
  };

  const sourceBadge = (s: 'env' | 'ui' | 'unset') => {
    if (s === 'env') return <span className="pill"><span className="dot dot-info" /> from .env</span>;
    if (s === 'ui') return <span className="pill"><span className="dot dot-success" /> set in UI</span>;
    return <span className="pill"><span className="dot dot-warn" /> not set</span>;
  };

  return (
    <div className="max-w-[820px] mx-auto px-8 py-10 space-y-8">
      <header>
        <h1 className="font-sans font-bold text-2xl mb-1">Settings</h1>
        <p className="text-fg-2 text-sm">
          Values from the UI override values from <code className="text-fg">backend/.env</code>. Sensitive
          values are encrypted at rest under <code className="text-fg">data/.encryption_key</code>.
        </p>
      </header>

      <section className="card space-y-3">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <h2 className="text-fg text-sm tracking-wider uppercase">LLM provider (agent brain)</h2>
          {sourceBadge(data.llm_source)}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="label">provider</label>
            <select
              className="select"
              value={provider}
              onChange={(e) => {
                const next = e.target.value;
                setProvider(next);
                const spec = providers.find((p) => p.name === next);
                // Reset base URL to the new provider's default unless the user
                // already has a custom one for this provider.
                setBaseUrl('');
                if (spec && spec.sample_models.length > 0) {
                  setModel(spec.sample_models[0]);
                }
              }}
            >
              {providers.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.label}
                </option>
              ))}
            </select>
            {activeSpec?.notes && (
              <div className="text-[10px] text-fg-3 mt-1">{activeSpec.notes}</div>
            )}
          </div>
          <div>
            <label className="label">model</label>
            <input
              className="input"
              list="model-suggestions"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder={activeSpec?.sample_models[0] ?? 'model-id'}
            />
            <datalist id="model-suggestions">
              {(activeSpec?.sample_models ?? []).map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>

          <div className="md:col-span-2">
            <label className="label">
              base URL (optional, default: {activeSpec?.base_url || 'SDK default'})
            </label>
            <input
              className="input"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder={activeSpec?.base_url || 'http://localhost:11434/v1'}
            />
          </div>

          <div className="md:col-span-2">
            <label className="label">
              api key {data.llm_api_key_set ? `(currently ${data.llm_api_key_masked} - leave blank to keep)` : ''}
              {activeSpec && !activeSpec.needs_key && ' - optional for this provider'}
            </label>
            <input
              className="input"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={activeSpec?.needs_key ? 'sk-... or hf_...' : 'optional - blank for local servers'}
            />
          </div>
        </div>

        <div className="flex gap-2 flex-wrap">
          <button className="btn btn-primary" onClick={saveLLM} disabled={!model}>
            save
          </button>
          <button className="btn" onClick={verifyLLM} disabled={!data.is_configured || verifying === 'llm'}>
            {verifying === 'llm' ? 'verifying...' : 'verify'}
          </button>
          {data.llm_source === 'ui' && (
            <button className="btn btn-danger" onClick={clearLLM}>
              clear UI override
            </button>
          )}
        </div>
      </section>

      <section className="card space-y-3">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <h2 className="text-fg text-sm tracking-wider uppercase">Hugging Face token</h2>
          {sourceBadge(data.hf_source)}
        </div>
        <p className="text-fg-2 text-xs">
          Used for pulling base models and pushing trained adapters. Read-write tokens recommended for push.
          {data.hf_username ? ` Currently signed in as ${data.hf_username}.` : ''}
        </p>
        <div>
          <label className="label">
            access token {data.hf_token_set ? `(currently ${data.hf_token_masked})` : ''}
          </label>
          <input
            className="input"
            type="password"
            placeholder="hf_..."
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
          />
        </div>
        <div className="flex gap-2 flex-wrap">
          <button className="btn btn-primary" onClick={saveHF} disabled={!hfToken}>
            save
          </button>
          <button className="btn" onClick={verifyHF} disabled={!data.hf_token_set || verifying === 'hf'}>
            {verifying === 'hf' ? 'verifying...' : 'verify'}
          </button>
          {data.hf_source === 'ui' && (
            <button className="btn btn-danger" onClick={clearHF}>
              clear UI override
            </button>
          )}
        </div>
      </section>

      {verifyMsg && <div className="text-xs text-fg">{verifyMsg}</div>}

      <section className="card space-y-3">
        <h2 className="text-fg text-sm tracking-wider uppercase">Behaviour</h2>
        <Toggle
          label="auto-configure pipeline on dataset upload"
          checked={data.auto_config_on_upload}
          onChange={(v) => updateFlag({ auto_config_on_upload: v })}
        />
        <Toggle
          label="show agent reasoning in chat and inspector"
          checked={data.show_agent_reasoning}
          onChange={(v) => updateFlag({ show_agent_reasoning: v })}
        />
      </section>
    </div>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center justify-between gap-4 text-xs">
      <span>{label}</span>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={`relative w-9 h-5 rounded-full border ${
          checked ? 'bg-success/20 border-success/40' : 'bg-bg-3 border-border-2'
        }`}
      >
        <span
          className={`absolute top-0.5 w-4 h-4 rounded-full transition-all ${
            checked ? 'left-[18px] bg-success' : 'left-0.5 bg-fg-2'
          }`}
        />
      </button>
    </label>
  );
}
