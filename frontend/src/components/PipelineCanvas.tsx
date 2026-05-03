'use client';

import { useCallback, useEffect, useMemo } from 'react';
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
import { Cpu, Database, Activity, Box } from 'lucide-react';
import type { Pipeline } from '@/lib/types';
import { cn } from '@/lib/cn';

type Props = {
  pipeline: Pipeline;
  onGraphChange: (graph: Pipeline['node_graph']) => void;
  onSelectNode: (nodeId: string | null) => void;
};

function PipelineNode({ data }: { data: any }) {
  const Icon = data.type === 'train' ? Cpu : data.type === 'dataset' ? Database : Box;
  
  return (
    <div className="group relative">
      <div className="absolute inset-0 bg-white/[0.01] blur-md rounded-lg opacity-0 group-hover:opacity-100 transition-opacity" />
      
      <div className="relative w-[200px] bg-black border border-white/10 rounded-md overflow-hidden hover:border-white/40 transition-all duration-300">
        <div className="px-3 py-2 bg-white/5 border-b border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Icon className="w-3 h-3 text-white/40" />
            <span className="text-[9px] font-black uppercase tracking-[0.15em] text-white/80">
              {data.type}
            </span>
          </div>
          <div className="w-1.5 h-1.5 rounded-full bg-white/20" />
        </div>

        <div className="px-3 py-2.5">
          <p className="text-[10px] text-white font-bold truncate uppercase tracking-tight">
            {data.label || 'Untitled Node'}
          </p>
          {data.status && (
            <p className="text-[8px] text-white/30 uppercase font-black tracking-widest mt-1">
              {data.status}
            </p>
          )}
        </div>

        <Handle type="target" position={Position.Left} className="!w-1.5 !h-1.5 !bg-white/20 !border-none !left-[-3px]" />
        <Handle type="source" position={Position.Right} className="!w-1.5 !h-1.5 !bg-white/20 !border-none !right-[-3px]" />
      </div>
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
  const initialNodes: Node[] = useMemo(() => pipeline.node_graph.nodes.map((n) => ({
    id: n.id,
    type: n.type,
    position: n.position,
    data: { 
      label: (n.data?.label as string) || n.id, 
      type: n.type, 
      ...n.data 
    },
  })), [pipeline.node_graph.nodes]);

  const initialEdges: Edge[] = useMemo(() => pipeline.node_graph.edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    animated: false,
    style: { stroke: '#fff', strokeWidth: 1.5, opacity: 0.2 }
  })), [pipeline.node_graph.edges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  useEffect(() => {
    onGraphChange({
      nodes: nodes.map((n) => ({ id: n.id, type: n.type ?? 'dataset', position: n.position, data: n.data ?? {} })),
      edges: edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
      viewport: pipeline.node_graph.viewport,
    });
  }, [nodes, edges]); // eslint-disable-line react-hooks/exhaustive-deps

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge({ ...params, animated: false, style: { stroke: '#fff', strokeWidth: 1.5, opacity: 0.2 } }, eds)),
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
      fitViewOptions={{ padding: 0.5 }}
      proOptions={{ hideAttribution: true }}
    >
      <Background color="#111" gap={40} size={1} />
      <Controls 
        showZoom={false} 
        showFitView={false} 
        showInteractive={false}
        className="!bg-white/5 !border-white/10 !rounded-none !p-1 scale-75 origin-bottom-left" 
      />
    </ReactFlow>
  );
}

export function PipelineCanvas(props: Props) {
  return (
    <div className="w-full h-full bg-black">
      <ReactFlowProvider>
        <CanvasInner key={props.pipeline.id} {...props} />
      </ReactFlowProvider>
    </div>
  );
}
