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
import { Play, PanelRight, MessageSquare, Terminal, Settings } from 'lucide-react';
import { cn } from '@/lib/cn';

export default function PlaygroundPage() {
  const { data: pipelines, mutate: mutateList } = useSWR<Pipeline[]>('/api/pipelines', fetcher);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [_selectedNode, setSelectedNode] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'inspector' | 'chat' | 'logs'>('inspector');

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
    api(`/api/pipelines/${pipeline.id}`, { method: 'PUT', body: JSON.stringify({ node_graph: graph }) }).catch(() => {});
  };

  const attachDataset = async (datasetId: string) => {
    if (!pipeline) return;
    await updateConfig({ dataset_id: datasetId || null });
  };

  return (
    <div className="h-screen bg-black flex flex-col overflow-hidden">
      {/* Streamlined Header */}
      <header className="h-16 border-b border-white/10 px-6 flex items-center justify-between bg-black z-30">
        <div className="flex items-center gap-6">
          <PipelineSwitcher
            pipelines={pipelines ?? []}
            activeId={activeId}
            onSelect={(id) => setActiveId(id)}
            onCreated={(p) => {
              mutateList();
              setActiveId(p.id);
            }}
          />
          {pipeline && (
            <div className="h-4 w-px bg-white/10" />
          )}
          {pipeline && (
            <div className="flex items-center gap-4">
               <span className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40">Active Project:</span>
               <span className="text-[11px] font-bold text-white uppercase tracking-widest">{pipeline.name}</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-4">
          <button 
            className="flex items-center gap-2 px-4 py-2 bg-white text-black rounded text-[10px] font-black uppercase tracking-widest hover:bg-white/90 transition-all shadow-[0_0_20px_rgba(255,255,255,0.1)]"
            onClick={startJob} 
            disabled={!activeId}
          >
            <Play className="w-3 h-3 fill-current" />
            Run Execution
          </button>
        </div>
      </header>

      <main className="flex-1 flex min-h-0 relative">
        {/* Massive Canvas Area */}
        <div className="flex-1 relative flex flex-col min-w-0">
          {pipeline ? (
            <div className="flex-1 relative">
              <PipelineCanvas
                pipeline={pipeline}
                onGraphChange={updateGraph}
                onSelectNode={setSelectedNode}
              />
              
              {/* Floating Overlay Controls */}
              <div className="absolute top-6 left-6 z-10 flex flex-col gap-4">
                <DatasetUploader 
                  pipelineId={pipeline.id} 
                  attachedId={pipeline.config.dataset_id ?? null} 
                  onAttach={attachDataset} 
                />
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-white/20 uppercase text-[10px] font-black tracking-[0.3em] animate-pulse">Initializing Studio...</p>
            </div>
          )}

          {/* Collapsible Bottom Logs */}
          <div className="h-40 border-t border-white/10 bg-black/50 backdrop-blur-xl overflow-hidden flex flex-col">
            <div className="px-4 py-2 border-b border-white/5 flex items-center gap-2">
              <Terminal className="w-3 h-3 text-white/40" />
              <span className="text-[9px] font-black uppercase tracking-widest text-white/40">Telemetry Output</span>
            </div>
            <div className="flex-1 overflow-y-auto p-4 font-mono text-[10px]">
              <LogPanel jobId={jobId} />
            </div>
          </div>
        </div>

        {/* Streamlined Right Inspector Panel */}
        <aside className="w-[400px] border-l border-white/10 flex flex-col bg-[#050505] z-20">
          {/* Tab Navigation */}
          <div className="flex border-b border-white/5">
            <TabButton 
              active={activeTab === 'inspector'} 
              onClick={() => setActiveTab('inspector')}
              icon={<Settings className="w-3.5 h-3.5" />}
              label="Config"
            />
            <TabButton 
              active={activeTab === 'chat'} 
              onClick={() => setActiveTab('chat')}
              icon={<MessageSquare className="w-3.5 h-3.5" />}
              label="Agent"
            />
          </div>

          <div className="flex-1 overflow-y-auto">
            {activeTab === 'inspector' && pipeline && (
              <div className="p-6">
                <h3 className="text-[10px] font-black uppercase tracking-[0.3em] text-white/40 mb-8">Pipeline Parameters</h3>
                <Inspector pipeline={pipeline} onPatch={updateConfig} />
              </div>
            )}
            
            {activeTab === 'chat' && (
              <div className="h-full flex flex-col">
                <AgentChat
                  pipelineId={activeId ?? undefined}
                  datasetId={pipeline?.config.dataset_id ?? undefined}
                  onPipelineChanged={() => mutatePipeline()}
                />
              </div>
            )}

            {!pipeline && activeTab === 'inspector' && (
              <div className="p-12 text-center">
                 <p className="text-white/20 uppercase text-[10px] font-black tracking-widest">Select project to view config</p>
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

function TabButton({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button 
      onClick={onClick}
      className={cn(
        "flex-1 flex items-center justify-center gap-2 py-4 border-b-2 transition-all",
        active ? "border-white bg-white/5 text-white" : "border-transparent text-white/30 hover:text-white/60"
      )}
    >
      {icon}
      <span className="text-[10px] font-black uppercase tracking-widest">{label}</span>
    </button>
  );
}
