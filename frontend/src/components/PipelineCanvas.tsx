'use client';

/**
 * Phase-driven pipeline canvas.
 *
 * The canvas starts empty and grows as `NodeMaterialized` events arrive
 * from the agent runtime. Each new node fades in. Edges are auto-derived
 * by chaining nodes in the order they appear (left-to-right) so the user
 * sees the agent's plan unfold visually as the backend works through its
 * phases.
 *
 * If the backend has already finalized a pipeline (we received a Pipeline
 * record), its node_graph takes precedence - this preserves the existing
 * "view a pipeline" use case.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, {
  addEdge,
  Background,
  Connection,
  Controls,
  Edge,
  Node,
  Position,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  type NodeTypes,
  Handle,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Cpu, Database, Box } from 'lucide-react';
import type { AgentEvent, Pipeline } from '@/lib/types';
import { cn } from '@/lib/cn';

type Props = {
  pipeline?: Pipeline | null;
  events?: AgentEvent[];
  onGraphChange?: (graph: Pipeline['node_graph']) => void;
  onSelectNode?: (nodeId: string | null) => void;
};

function PipelineNode({ data }: { data: { label?: string; type?: string; status?: string } }) {
  const Icon = data.type === 'train' ? Cpu : data.type === 'dataset' ? Database : Box;

  return (
    <div className="group relative animate-pop-in">
      <div className="relative w-[200px] bg-bg border border-border rounded-md overflow-hidden hover:border-fg-3 transition-colors">
        <div className="px-3 py-2 bg-bg-2 border-b border-border flex items-center gap-2">
          <Icon className="w-3 h-3 text-fg-3" />
          <span className="text-[9px] font-bold uppercase tracking-[0.15em] text-fg-2">
            {data.type ?? 'node'}
          </span>
        </div>
        <div className="px-3 py-2.5">
          <p className="text-[10px] text-fg font-medium truncate">
            {data.label ?? 'untitled'}
          </p>
          {data.status && (
            <p className="text-[9px] text-fg-3 mt-1">{data.status}</p>
          )}
        </div>

        <Handle type="target" position={Position.Left} className="!w-1.5 !h-1.5 !bg-fg-3 !border-none !left-[-3px]" />
        <Handle type="source" position={Position.Right} className="!w-1.5 !h-1.5 !bg-fg-3 !border-none !right-[-3px]" />
      </div>
    </div>
  );
}

const nodeTypes: NodeTypes = {
  dataset: PipelineNode,
  preprocess: PipelineNode,
  balance: PipelineNode,
  split: PipelineNode,
  augment: PipelineNode,
  template_prompts: PipelineNode,
  convert_format: PipelineNode,
  train: PipelineNode,
  evaluate: PipelineNode,
  sandbox: PipelineNode,
  export: PipelineNode,
};

const NODE_X_STEP = 240;
const NODE_Y = 80;

function CanvasInner({ pipeline, events, onGraphChange, onSelectNode }: Props) {
  // Source of truth #1: a finalized pipeline takes precedence.
  const fromPipeline = useMemo<Node[] | null>(() => {
    if (!pipeline?.node_graph?.nodes?.length) return null;
    return pipeline.node_graph.nodes.map((n) => ({
      id: n.id,
      type: n.type,
      position: n.position,
      data: { label: (n.data?.label as string) || n.id, type: n.type, ...n.data },
    }));
  }, [pipeline?.id, pipeline?.node_graph?.nodes]);

  // Source of truth #2: agent-driven node materialization. Used when no
  // pipeline exists yet, or before its nodes have been published.
  const fromEvents = useMemo<Node[]>(() => {
    if (!events?.length) return [];
    const seen = new Map<string, Node>();
    let i = 0;
    for (const e of events) {
      if (e.kind !== 'NodeMaterialized') continue;
      const n = (e.payload as { node?: { id?: string; type?: string; data?: Record<string, unknown> } }).node;
      if (!n?.id) continue;
      seen.set(n.id, {
        id: n.id,
        type: n.type ?? 'preprocess',
        position: { x: 40 + i * NODE_X_STEP, y: NODE_Y },
        data: { label: (n.data?.label as string) || n.type || n.id, type: n.type, ...(n.data ?? {}) },
      });
      i++;
    }
    return Array.from(seen.values());
  }, [events]);

  const computedNodes = fromPipeline ?? fromEvents;
  const computedEdges: Edge[] = useMemo(() => {
    if (fromPipeline && pipeline?.node_graph?.edges?.length) {
      return pipeline.node_graph.edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        animated: false,
        style: { stroke: '#fff', strokeWidth: 1.2, opacity: 0.25 },
      }));
    }
    // Auto-chain nodes in declaration order.
    const out: Edge[] = [];
    for (let i = 0; i < computedNodes.length - 1; i++) {
      const a = computedNodes[i];
      const b = computedNodes[i + 1];
      out.push({
        id: `auto_${a.id}_${b.id}`,
        source: a.id,
        target: b.id,
        animated: true,
        style: { stroke: '#fff', strokeWidth: 1.2, opacity: 0.25 },
      });
    }
    return out;
  }, [fromPipeline, pipeline?.node_graph?.edges, computedNodes]);

  const [nodes, setNodes, onNodesChange] = useNodesState(computedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(computedEdges);
  const [hasFitInitial, setHasFitInitial] = useState(false);

  // Keep canvas in sync with computed sources.
  useEffect(() => {
    setNodes(computedNodes);
    setEdges(computedEdges);
    if (computedNodes.length > 0) setHasFitInitial(true);
  }, [computedNodes, computedEdges, setNodes, setEdges]);

  useEffect(() => {
    if (!onGraphChange) return;
    onGraphChange({
      nodes: nodes.map((n) => ({ id: n.id, type: n.type ?? 'dataset', position: n.position, data: n.data ?? {} })),
      edges: edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
      viewport: pipeline?.node_graph?.viewport ?? { x: 0, y: 0, zoom: 1 },
    });
  }, [nodes, edges]); // eslint-disable-line react-hooks/exhaustive-deps

  const onConnect = useCallback(
    (params: Connection) =>
      setEdges((eds) =>
        addEdge({ ...params, animated: false, style: { stroke: '#fff', strokeWidth: 1.2, opacity: 0.25 } }, eds),
      ),
    [setEdges],
  );

  if (computedNodes.length === 0) {
    return <Empty />;
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onNodeClick={(_, node) => onSelectNode?.(node.id)}
      onPaneClick={() => onSelectNode?.(null)}
      nodeTypes={nodeTypes}
      fitView={hasFitInitial}
      fitViewOptions={{ padding: 0.5 }}
      proOptions={{ hideAttribution: true }}
    >
      <Background color="#222" gap={36} size={1} />
      <Controls
        showZoom={false}
        showFitView={false}
        showInteractive={false}
        className="!bg-bg-2 !border-border !rounded-none !p-1 scale-75 origin-bottom-left"
      />
    </ReactFlow>
  );
}

function Empty() {
  return (
    <div className="absolute inset-0 flex items-center justify-center">
      <p className="text-[12px] text-fg-3">Canvas will fill in as agents complete each phase.</p>
    </div>
  );
}

export function PipelineCanvas(props: Props) {
  return (
    <div className={cn('relative w-full h-full bg-bg')}>
      <ReactFlowProvider>
        <CanvasInner {...props} />
      </ReactFlowProvider>
    </div>
  );
}
