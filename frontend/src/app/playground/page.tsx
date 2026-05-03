'use client';

import { useEffect, useMemo, useState } from 'react';
import useSWR from 'swr';
import { api, fetcher } from '@/lib/api';
import type { NodeGraph, Pipeline, PipelineConfig } from '@/lib/types';
import { PipelineSwitcher } from '@/components/PipelineSwitcher';
import { PipelineCanvas } from '@/components/PipelineCanvas';
import { Inspector } from '@/components/Inspector';
import { AgentChat } from '@/components/AgentChat';
import { LogPanel } from '@/components/LogPanel';
import { DatasetUploader } from '@/components/DatasetUploader';

export default function PlaygroundPage() {
  const { data: pipelines, mutate: mutateList } = useSWR<Pipeline[]>('/api/pipelines', fetcher);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [_selectedNode, setSelectedNode] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [showChat, setShowChat] = useState(true);

  // Auto-select first pipeline once we know what's there.
  useEffect(() => {
    if (!activeId && pipelines && pipelines.length > 0) {
      setActiveId(pipelines[0].id);
    }
  }, [pipelines, activeId]);

  const { data: pipeline, mutate: mutatePipeline } = useSWR<Pipeline>(
    activeId ? `/api/pipelines/${activeId}` : null,
    fetcher,
  );

  const startJob = async () => {
    if (!activeId) return;
    const r = await api<{ job_id: string }>('/api/jobs/start', {
      method: 'POST',
      body: JSON.stringify({ pipeline_id: activeId }),
    });
    setJobId(r.job_id);
  };

  const updateConfig = async (patch: Partial<PipelineConfig>) => {
    if (!pipeline) return;
    const merged = { ...pipeline.config, ...patch };
    const updated = await api<Pipeline>(`/api/pipelines/${pipeline.id}`, {
      method: 'PUT',
      body: JSON.stringify({ config: merged }),
    });
    mutatePipeline(updated, false);
  };

  const updateGraph = async (graph: NodeGraph) => {
    if (!pipeline) return;
    // Fire and forget; we already render the latest graph locally.
    api(`/api/pipelines/${pipeline.id}`, { method: 'PUT', body: JSON.stringify({ node_graph: graph }) }).catch(() => {});
  };

  const attachDataset = async (datasetId: string) => {
    if (!pipeline) return;
    await updateConfig({ dataset_id: datasetId || null });
  };

  const headerCfg = useMemo(() => pipeline?.config, [pipeline]);

  return (
    <div className="h-[calc(100vh-52px)] flex flex-col">
      <PipelineSwitcher
        pipelines={pipelines ?? []}
        activeId={activeId}
        onSelect={(id) => setActiveId(id)}
        onCreated={(p) => {
          mutateList();
          setActiveId(p.id);
        }}
      />

      {pipeline ? (
        <DatasetUploader pipelineId={pipeline.id} attachedId={pipeline.config.dataset_id} onAttach={attachDataset} />
      ) : null}

      <div className="flex-1 grid grid-cols-[1fr_320px_360px] min-h-0">
        {/* Canvas + run/log */}
        <div className="flex flex-col min-w-0 border-r border-border">
          <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-bg-2/40">
            <span className="text-[11px] text-fg-2 uppercase tracking-wider">canvas</span>
            <span className="text-[11px] text-fg-3">
              {headerCfg
                ? `${headerCfg.training_method} · ${headerCfg.base_model} · ${headerCfg.epochs}ep · bs=${headerCfg.batch_size}`
                : 'no pipeline'}
            </span>
            <button
              className="btn ml-auto"
              onClick={() => setShowChat((v) => !v)}
            >
              {showChat ? 'hide agent' : 'show agent'}
            </button>
            <button className="btn btn-primary" onClick={startJob} disabled={!activeId}>
              ▶ run pipeline
            </button>
          </div>
          <div className="flex-1 min-h-0">
            {pipeline ? (
              <PipelineCanvas
                pipeline={pipeline}
                onGraphChange={updateGraph}
                onSelectNode={setSelectedNode}
              />
            ) : (
              <div className="h-full flex items-center justify-center text-fg-2 text-sm">
                Create a pipeline to start.
              </div>
            )}
          </div>
          <div className="h-[28%] min-h-[180px] p-3">
            <LogPanel jobId={jobId} />
          </div>
        </div>

        {/* Inspector */}
        <div className="overflow-y-auto border-r border-border bg-bg/40">
          {pipeline ? (
            <Inspector pipeline={pipeline} onPatch={updateConfig} />
          ) : (
            <div className="p-4 text-fg-2 text-xs">No pipeline selected.</div>
          )}
        </div>

        {/* Agent chat */}
        {showChat && (
          <div className="bg-bg-2/40">
            <AgentChat
              pipelineId={activeId ?? undefined}
              datasetId={pipeline?.config.dataset_id ?? undefined}
              onPipelineChanged={() => mutatePipeline()}
            />
          </div>
        )}
      </div>
    </div>
  );
}
