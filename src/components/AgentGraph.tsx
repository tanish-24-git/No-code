'use client';

/**
 * The live agent graph (R17): agent nodes appear as they're spawned, yellow
 * animated edges pulse when agents hand off work, artifacts hang off their
 * producers, and every node carries live status + per-run spend. Derived
 * ENTIRELY from the session event stream, so it replays perfectly.
 */
import { useEffect, useMemo, useState } from 'react';
import ReactFlow, {
  Background,
  BackgroundVariant,
  Handle,
  Position,
  ReactFlowProvider,
  type Edge,
  type Node,
  type NodeProps,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { cn } from '@/lib/cn';
import { layoutGraph } from '@/lib/graphLayout';
import type { AgentEvent } from '@/lib/events';
import { Bot, Cpu, Database, FileCode, FileText, Package, Sparkles } from 'lucide-react';

const MESSAGE_EDGE_TTL_MS = 4_000;

// ── Graph derivation ──────────────────────────────────────────────────────

type AgentStatus = 'pending' | 'working' | 'done' | 'failed' | 'canceled' | 'input-required';

interface AgentNodeData {
  name: string;
  definitionId: string;
  status: AgentStatus;
  summary?: string;
  ephemeral: boolean;
  agentKind: 'llm' | 'code';
  spentUsd: number;
}

interface ArtifactNodeData {
  label: string;
  artifactKind: string;
}

interface DerivedGraph {
  nodes: Node[];
  edges: Edge[];
  agentCount: number;
}

function deriveGraph(events: AgentEvent[], now: number): DerivedGraph {
  const agents = new Map<string, AgentNodeData>();
  const spawnEdges: Edge[] = [];
  const artifactNodes = new Map<string, Node<ArtifactNodeData>>();
  const artifactEdges: Edge[] = [];
  const messageEdges = new Map<string, Edge & { ts: number }>();

  const sorted = [...events].sort((a, b) => (a.id < b.id ? -1 : 1));
  for (const e of sorted) {
    const p = e.payload as Record<string, unknown>;
    switch (e.kind) {
      case 'agent.spawned': {
        const id = String(p.agentRunId);
        agents.set(id, {
          name: String(p.name ?? id),
          definitionId: String(p.definitionId ?? id),
          status: id === 'orchestrator' ? 'pending' : 'working',
          ephemeral: Boolean(p.ephemeral),
          agentKind: (p.agentKind as 'llm' | 'code') ?? 'llm',
          spentUsd: 0,
        });
        if (p.parentRunId) {
          spawnEdges.push({
            id: `spawn-${p.parentRunId}-${id}`,
            source: String(p.parentRunId),
            target: id,
            style: { stroke: 'rgba(255,255,255,0.25)', strokeWidth: 1.5 },
            ...(p.ephemeral ? { label: 'created', labelStyle: { fill: '#EAB308', fontSize: 9 } } : {}),
          });
        }
        break;
      }
      case 'agent.status': {
        const id = e.agentRunId ?? String(p.agentRunId ?? '');
        const agent = agents.get(id);
        if (agent) {
          agent.status = (p.status as AgentStatus) ?? agent.status;
          if (typeof p.summary === 'string') agent.summary = p.summary;
        }
        break;
      }
      case 'agent.message': {
        const from = String(p.fromRunId ?? '');
        const to = String(p.toRunId ?? '');
        if (!from || !to) break;
        const key = `${from}->${to}`;
        messageEdges.set(key, {
          id: `msg-${key}`,
          source: from,
          target: to,
          ts: Date.parse(e.ts),
          animated: true,
          style: { stroke: '#EAB308', strokeWidth: 2 },
        });
        break;
      }
      case 'agent.artifact': {
        const producer = e.agentRunId ?? 'orchestrator';
        const label = String(p.label ?? 'artifact');
        const id = `artifact-${label}`;
        artifactNodes.set(id, {
          id,
          type: 'artifact',
          position: { x: 0, y: 0 },
          data: { label, artifactKind: String(p.artifactKind ?? 'file') },
        });
        artifactEdges.push({
          id: `art-${producer}-${id}`,
          source: producer,
          target: id,
          style: { stroke: 'rgba(255,255,255,0.15)', strokeDasharray: '4 3' },
        });
        break;
      }
      case 'budget.usage': {
        const id = e.agentRunId ?? '';
        const agent = agents.get(id);
        if (agent) agent.spentUsd += Number(p.lastCallUsd ?? 0);
        break;
      }
      default:
        break;
    }
  }

  const nodes: Node[] = [
    ...[...agents.entries()].map(([id, data]) => ({
      id,
      type: 'agent',
      position: { x: 0, y: 0 },
      data,
    })),
    ...artifactNodes.values(),
  ];

  // Recent messages animate yellow; older ones stay as faint traces.
  const edges: Edge[] = [
    ...spawnEdges,
    ...artifactEdges,
    ...[...messageEdges.values()].map(({ ts, ...edge }) => {
      const fresh = now - ts < MESSAGE_EDGE_TTL_MS;
      return fresh
        ? edge
        : { ...edge, animated: false, style: { stroke: 'rgba(234,179,8,0.25)', strokeWidth: 1 } };
    }),
  ];

  // Dedup edges by id (spawn edges can repeat on replay).
  const seen = new Set<string>();
  const uniqueEdges = edges.filter((e) => (seen.has(e.id) ? false : (seen.add(e.id), true)));

  return { nodes: layoutGraph(nodes, uniqueEdges), edges: uniqueEdges, agentCount: agents.size };
}

// ── Node renderers ────────────────────────────────────────────────────────

const STATUS_STYLE: Record<AgentStatus, string> = {
  pending: 'border-dashed border-white/20 text-white/50',
  working: 'border-amber-400/70 text-white shadow-[0_0_18px_rgba(234,179,8,0.25)] animate-pulse',
  'input-required': 'border-warn/70 text-white',
  done: 'border-success/50 text-white bg-success/5',
  failed: 'border-danger/60 text-white bg-danger/5',
  canceled: 'border-white/15 text-white/40',
};

function AgentNode({ data }: NodeProps<AgentNodeData>) {
  const Icon = data.agentKind === 'code' ? Cpu : data.ephemeral ? Sparkles : Bot;
  return (
    <div
      className={cn(
        'w-[210px] rounded-lg border bg-[#0a0a0a] px-3 py-2.5 transition-all',
        STATUS_STYLE[data.status] ?? STATUS_STYLE.pending,
      )}
      title={data.summary}
    >
      <Handle type="target" position={Position.Top} className="!bg-white/30 !border-0 !w-2 !h-2" />
      <div className="flex items-center gap-2">
        <Icon className="w-3.5 h-3.5 shrink-0 opacity-70" />
        <span className="text-[11px] font-bold truncate">{data.name}</span>
        {data.spentUsd > 0 && (
          <span className="ml-auto text-[9px] font-mono text-white/40">${data.spentUsd.toFixed(3)}</span>
        )}
      </div>
      <div className="mt-1 flex items-center gap-1.5">
        <span className="text-[8px] uppercase tracking-widest font-black text-white/35">{data.status}</span>
        {data.ephemeral && (
          <span className="text-[8px] uppercase tracking-widest font-black text-amber-400/70">created</span>
        )}
      </div>
      {data.summary && <div className="mt-1 text-[9px] text-white/40 truncate">{data.summary}</div>}
      <Handle type="source" position={Position.Bottom} className="!bg-white/30 !border-0 !w-2 !h-2" />
    </div>
  );
}

const ARTIFACT_ICON: Record<string, typeof FileText> = {
  dataset: Database,
  script: FileCode,
  config: FileText,
  checkpoint: Package,
  metrics: FileText,
  'eval-report': FileText,
  model: Package,
  'model-card': FileText,
};

function ArtifactNode({ data }: NodeProps<ArtifactNodeData>) {
  const Icon = ARTIFACT_ICON[data.artifactKind] ?? FileText;
  return (
    <div className="w-[160px] rounded-md border border-white/10 bg-white/[0.03] px-2.5 py-1.5">
      <Handle type="target" position={Position.Top} className="!bg-white/20 !border-0 !w-1.5 !h-1.5" />
      <div className="flex items-center gap-1.5">
        <Icon className="w-3 h-3 shrink-0 text-white/40" />
        <span className="text-[10px] text-white/70 truncate">{data.label}</span>
      </div>
      <div className="text-[8px] uppercase tracking-widest font-black text-white/30">{data.artifactKind}</div>
      <Handle type="source" position={Position.Bottom} className="!bg-white/20 !border-0 !w-1.5 !h-1.5" />
    </div>
  );
}

const nodeTypes = { agent: AgentNode, artifact: ArtifactNode };

// ── Component ─────────────────────────────────────────────────────────────

export function AgentGraph({ events }: { events: AgentEvent[] }) {
  // Ticker so fresh yellow edges fade back to traces after the TTL.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1_500);
    return () => clearInterval(t);
  }, []);

  const { nodes, edges, agentCount } = useMemo(() => deriveGraph(events, now), [events, now]);

  if (agentCount === 0) {
    return (
      <div className="h-full flex items-center justify-center text-[11px] uppercase tracking-widest text-white/30 font-black">
        Agents appear here as they spawn
      </div>
    );
  }

  return (
    // React Flow needs an explicitly-sized parent (warning #004) — the
    // playground slots this into a flex-1 container, so fill it.
    <div className="h-full w-full">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={FIT_VIEW_OPTIONS}
          proOptions={PRO_OPTIONS}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          zoomOnScroll
          panOnDrag
        >
          <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="rgba(255,255,255,0.06)" />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
}

// Stable identities — new objects per render trip React Flow warning #002.
const FIT_VIEW_OPTIONS = { padding: 0.25, maxZoom: 1.1 };
const PRO_OPTIONS = { hideAttribution: true };
