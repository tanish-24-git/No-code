'use client';

import { useCallback, useEffect } from 'react';
import ReactFlow, {
  addEdge,
  Background,
  Connection,
  Controls,
  Edge,
  MiniMap,
  Node,
  Position,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  type NodeTypes,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Handle } from 'reactflow';
import type { Pipeline } from '@/lib/types';

type Props = {
  pipeline: Pipeline;
  onGraphChange: (graph: Pipeline['node_graph']) => void;
  onSelectNode: (nodeId: string | null) => void;
};

const NODE_COLOR: Record<string, string> = {
  dataset: '#60a5fa',
  preprocess: '#fbbf24',
  train: '#4ade80',
  evaluate: '#c084fc',
  export: '#f472b6',
};

function PipelineNode({ id, data }: { id: string; data: { label: string; type: string; status?: string } }) {
  const color = NODE_COLOR[data.type] ?? '#888';
  return (
    <div className="bg-node-bg border border-node-border rounded shadow-md min-w-[180px]">
      <div className="bg-node-header px-3 py-2 border-b border-node-border flex items-center gap-2 text-[11px]">
        <span className="w-2 h-2 rounded-full" style={{ background: color }} />
        <span className="text-fg uppercase tracking-wider">{data.type}</span>
        {data.status && <span className="ml-auto text-[10px] text-fg-2">{data.status}</span>}
      </div>
      <div className="px-3 py-2 text-[11px] text-fg-2">{data.label || id}</div>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes: NodeTypes = {
  dataset: PipelineNode,
  preprocess: PipelineNode,
  train: PipelineNode,
  evaluate: PipelineNode,
  export: PipelineNode,
};

function CanvasInner({ pipeline, onGraphChange, onSelectNode }: Props) {
  // The wrapper passes key={pipeline.id} so this component remounts when the
  // user switches pipelines. That makes these initial values one-shot, which
  // is exactly what useNodesState / useEdgesState want.
  const initialNodes: Node[] = pipeline.node_graph.nodes.map((n) => ({
    id: n.id,
    type: n.type,
    position: n.position,
    data: { label: (n.data?.label as string) || n.id, type: n.type, ...n.data },
  }));
  const initialEdges: Edge[] = pipeline.node_graph.edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    animated: false,
  }));

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Keep parent in sync (debounced via animation frame to avoid spamming PUTs).
  useEffect(() => {
    let raf: number | null = null;
    const handle = () => {
      onGraphChange({
        nodes: nodes.map((n) => ({ id: n.id, type: n.type ?? 'dataset', position: n.position, data: n.data ?? {} })),
        edges: edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
        viewport: pipeline.node_graph.viewport,
      });
    };
    raf = requestAnimationFrame(handle);
    return () => {
      if (raf) cancelAnimationFrame(raf);
    };
  }, [nodes, edges]); // eslint-disable-line react-hooks/exhaustive-deps

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge({ ...params, id: `e_${Date.now()}` }, eds)),
    [setEdges],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onNodeClick={(_, node) => onSelectNode(node.id)}
      onPaneClick={() => onSelectNode(null)}
      nodeTypes={nodeTypes}
      fitView
      proOptions={{ hideAttribution: true }}
    >
      <Background gap={32} size={1} color="rgba(255,255,255,0.04)" />
      <MiniMap pannable zoomable />
      <Controls />
    </ReactFlow>
  );
}

export function PipelineCanvas(props: Props) {
  // key={pipeline.id} forces a remount when the active pipeline changes,
  // which lets CanvasInner safely treat the initial nodes/edges as one-shot
  // values for useNodesState / useEdgesState.
  return (
    <ReactFlowProvider>
      <CanvasInner key={props.pipeline.id} {...props} />
    </ReactFlowProvider>
  );
}
