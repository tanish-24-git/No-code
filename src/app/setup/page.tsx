'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import useSWR from 'swr';
import { api, fetcher } from '@/lib/api';
import type { Health, Provider, ProviderInfo, Settings } from '@/lib/types';

const STEPS = ['provider', 'model', 'apikey', 'verify', 'done'] as const;
type Step = (typeof STEPS)[number];

export default function SetupPage() {
  const router = useRouter();
  const { data: health } = useSWR<Health>('/health', fetcher, { refreshInterval: 5000 });
  const { data: settings, mutate } = useSWR<Settings>('/api/settings', fetcher);
  const { data: providers } = useSWR<ProviderInfo[]>('/api/settings/providers', fetcher);

  const [step, setStep] = useState<Step>('provider');
  const [provider, setProvider] = useState<Provider>('anthropic');
  const [model, setModel] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [verifyResult, setVerifyResult] = useState<string | null>(null);

  const activeSpec = useMemo(
    () => providers?.find((p) => p.name === provider),
    [providers, provider],
  );

  useEffect(() => {
    if (settings?.is_configured) setStep('done');
  }, [settings]);

  // Default the model field when provider changes (only if user hasn't typed one).
  useEffect(() => {
    if (activeSpec && !model && activeSpec.sample_models.length > 0) {
      setModel(activeSpec.sample_models[0]);
    }
  }, [activeSpec]); // eslint-disable-line react-hooks/exhaustive-deps

  const saveAndVerify = async () => {
    setBusy(true);
    setError(null);
    try {
      await api('/api/settings/llm', {
        method: 'POST',
        body: JSON.stringify({
          provider,
          model,
          base_url: baseUrl || null,
          api_key: apiKey || undefined,
        }),
      });
      const r = await api<{ valid: boolean; detail: string | null; models?: string[] }>(
        '/api/settings/verify-llm',
        { method: 'POST' },
      );
      if (!r.valid) {
        setError(`Verification failed: ${r.detail || 'unknown'}`);
        setStep('verify');
        await mutate();
        return;
      }
      setVerifyResult(
        r.models && r.models.length
          ? `Reachable. ${r.models.length} models available.`
          : 'Reachable.',
      );
      setStep('done');
      await mutate();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="max-w-[860px] mx-auto px-8 py-12 space-y-8">
      <header>
        <div className="text-[11px] uppercase tracking-wider text-fg-2 mb-2">First-run setup</div>
        <h1 className="font-sans font-bold text-3xl mb-2">Configure the agent</h1>
        <p className="text-fg-2 text-sm">
          The agent needs an LLM to think. Pick a provider, paste a key (or point at a local server),
          and verify. You can change all of this later in Settings.
        </p>
      </header>

      <Stepper current={step} />

      {step === 'provider' && (
        <section className="card space-y-4">
          <h2 className="text-sm uppercase tracking-wider">1. Choose a provider</h2>
          {!providers ? (
            <div className="text-fg-2 text-xs">loading providers...</div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {providers.map((p) => (
                <ProviderCard
                  key={p.name}
                  active={provider === p.name}
                  spec={p}
                  onClick={() => {
                    setProvider(p.name);
                    setBaseUrl('');
                    setModel(p.sample_models[0] ?? '');
                  }}
                />
              ))}
            </div>
          )}
          <div className="flex justify-end">
            <button className="btn btn-primary" onClick={() => setStep('model')}>
              next
            </button>
          </div>
        </section>
      )}

      {step === 'model' && (
        <section className="card space-y-4">
          <h2 className="text-sm uppercase tracking-wider">2. Pick a model</h2>
          <div>
            <label className="label">model id</label>
            <input
              className="input"
              list="setup-model-suggestions"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder={activeSpec?.sample_models[0] ?? 'model-id'}
            />
            <datalist id="setup-model-suggestions">
              {(activeSpec?.sample_models ?? []).map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
            {activeSpec?.sample_models && activeSpec.sample_models.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {activeSpec.sample_models.map((m) => (
                  <button
                    key={m}
                    type="button"
                    className="pill hover:bg-bg-3 cursor-pointer"
                    onClick={() => setModel(m)}
                  >
                    {m}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div>
            <label className="label">
              base URL (optional, default: {activeSpec?.base_url || 'SDK default'})
            </label>
            <input
              className="input"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder={activeSpec?.base_url || 'leave blank to use the default'}
            />
          </div>
          <div className="flex justify-between">
            <button className="btn" onClick={() => setStep('provider')}>back</button>
            <button className="btn btn-primary" onClick={() => setStep('apikey')} disabled={!model}>
              next
            </button>
          </div>
        </section>
      )}

      {step === 'apikey' && (
        <section className="card space-y-4">
          <h2 className="text-sm uppercase tracking-wider">3. Add API key</h2>
          <p className="text-fg-2 text-xs">
            Stored encrypted under <code>data/.encryption_key</code>.
            {activeSpec && !activeSpec.needs_key
              ? ' This provider is local and does not need an API key.'
              : ''}
          </p>
          <div>
            <label className="label">api key</label>
            <input
              className="input"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={activeSpec?.needs_key ? 'sk-... or hf_...' : 'optional'}
            />
          </div>
          <div className="flex justify-between">
            <button className="btn" onClick={() => setStep('model')}>back</button>
            <button
              className="btn btn-primary"
              onClick={saveAndVerify}
              disabled={busy || (Boolean(activeSpec?.needs_key) && !apiKey)}
            >
              {busy ? 'saving and verifying...' : 'save and verify'}
            </button>
          </div>
        </section>
      )}

      {step === 'verify' && (
        <section className="card space-y-4">
          <h2 className="text-sm uppercase tracking-wider">Verification failed</h2>
          {error && <div className="text-danger text-xs">{error}</div>}
          <p className="text-fg-2 text-sm">
            Double-check the model id, the base URL, and the API key. For local servers, make sure the
            server is running on the listed port.
          </p>
          <div className="flex justify-between">
            <button className="btn" onClick={() => setStep('apikey')}>back</button>
            <button className="btn btn-primary" onClick={saveAndVerify} disabled={busy}>
              try again
            </button>
          </div>
        </section>
      )}

      {step === 'done' && (
        <section className="card space-y-4">
          <h2 className="text-sm uppercase tracking-wider">Setup complete</h2>
          <p className="text-fg text-sm">
            {verifyResult || 'Your LLM is configured. The agent is ready.'}
          </p>
          <div className="text-fg-2 text-xs space-y-1">
            <div>Provider: <span className="text-fg">{settings?.llm_provider ?? provider}</span></div>
            <div>Model: <span className="text-fg">{settings?.llm_model || model}</span></div>
            <div>Source: <span className="text-fg">{settings?.llm_source ?? 'ui'}</span></div>
            {health?.hardware && (
              <div>
                Hardware: <span className="text-fg">{health.hardware.device}</span>
                {health.hardware.gpu_name ? ` - ${health.hardware.gpu_name}` : ''}
              </div>
            )}
          </div>
          <div className="flex flex-wrap gap-2 pt-2">
            <button className="btn btn-primary" onClick={() => router.push('/playground')}>
              open playground
            </button>
            <Link href="/settings" className="btn">
              fine-tune settings
            </Link>
            <Link href="/inference" className="btn">
              register an inference endpoint
            </Link>
          </div>
        </section>
      )}
    </div>
  );
}

function Stepper({ current }: { current: Step }) {
  const labels: Record<Step, string> = {
    provider: 'provider',
    model: 'model',
    apikey: 'api key',
    verify: 'fix errors',
    done: 'done',
  };
  const order: Step[] = ['provider', 'model', 'apikey', 'done'];
  const idx = order.indexOf(current === 'verify' ? 'apikey' : current);
  return (
    <div className="flex items-center gap-2 text-[11px] text-fg-3 flex-wrap">
      {order.map((s, i) => (
        <div key={s} className="flex items-center gap-2">
          <span
            className={`px-2 py-1 rounded border ${
              i < idx
                ? 'bg-success/15 border-success/40 text-success'
                : i === idx
                ? 'bg-bg-3 border-border-2 text-fg'
                : 'bg-bg-3/30 border-border text-fg-3'
            }`}
          >
            {i + 1}. {labels[s]}
          </span>
          {i < order.length - 1 && <span>-</span>}
        </div>
      ))}
    </div>
  );
}

function ProviderCard({
  active,
  spec,
  onClick,
}: {
  active: boolean;
  spec: ProviderInfo;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`text-left card transition-colors ${
        active ? 'border-success/40 bg-success/5' : 'hover:bg-bg-2'
      }`}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className="text-sm">{spec.label}</span>
        {!spec.needs_key && (
          <span className="pill"><span className="dot dot-info" /> no key</span>
        )}
      </div>
      <div className="text-[10px] text-fg-3 mb-1 break-all">{spec.base_url || '(custom URL)'}</div>
      {spec.notes && <div className="text-[10px] text-fg-2 leading-relaxed">{spec.notes}</div>}
    </button>
  );
}
