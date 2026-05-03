'use client';

import React, { useMemo, useCallback } from 'react';
import ReactFlow, { 
  Background, 
  Controls, 
  Node, 
  Edge,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Database, Wand2, Cpu, BarChart3, Plus, Play, MoreHorizontal } from 'lucide-react';
import { cn } from '@/lib/cn';

const initialNodes: Node[] = [
  {
    id: '1',
    type: 'custom',
    data: { label: 'Dataset', icon: <Database className="w-5 h-5" />, desc: 'Raw source data' },
    position: { x: 50, y: 150 },
  },
  {
    id: '2',
    type: 'custom',
    data: { label: 'Preprocess', icon: <Wand2 className="w-5 h-5" />, desc: 'Token optimization' },
    position: { x: 300, y: 150 },
  },
  {
    id: '3',
    type: 'custom',
    data: { label: 'Fine-tune', icon: <Cpu className="w-5 h-5" />, desc: 'Weight adjustment' },
    position: { x: 550, y: 150 },
  },
];

// Start with no connections
const initialEdges: Edge[] = [];

function CustomNode({ data }: { data: any }) {
  return (
    <div className="group relative">
      <div className="absolute inset-0 bg-white/[0.02] blur-xl rounded-xl opacity-0 group-hover:opacity-100 transition-opacity" />
      <div className="relative px-6 py-5 bg-[#121212] border border-white/20 rounded-lg min-w-[200px] hover:border-white/40 transition-all duration-500 select-none cursor-grab active:cursor-grabbing shadow-2xl">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="text-white opacity-80 group-hover:opacity-100 transition-opacity">
              {data.icon}
            </div>
            <span className="text-[10px] font-black uppercase tracking-[0.3em] text-white/90">
              {data.label}
            </span>
          </div>
          <MoreHorizontal className="w-3 h-3 text-white/20" />
        </div>
        
        <div className="h-px w-full bg-white/5 mb-4" />
        
        <p className="text-[9px] text-fg-3 leading-relaxed font-medium uppercase tracking-wider">
          {data.desc}
        </p>
        
        <Handle type="target" position={Position.Left} className="!w-2 !h-2 !bg-white !border-none !left-[-4px]" />
        <Handle type="source" position={Position.Right} className="!w-2 !h-2 !bg-white !border-none !right-[-4px]" />
      </div>
    </div>
  );
}

export default function HeroFlow() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge({ ...params, animated: false, style: { stroke: '#fff', strokeWidth: 1.5, opacity: 0.5 } }, eds)),
    [setEdges]
  );

  const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);

  const addNode = (type: string) => {
    const id = (nodes.length + 1).toString();
    const newNode: Node = {
      id,
      type: 'custom',
      data: { 
        label: type, 
        icon: type === 'Model' ? <Cpu className="w-5 h-5" /> : <BarChart3 className="w-5 h-5" />,
        desc: 'Pipeline module' 
      },
      position: { x: 100, y: 100 },
    };
    setNodes((nds) => nds.concat(newNode));
  };

  return (
    <div className="w-full h-[600px] border border-white/5 rounded-2xl bg-[#080808] overflow-hidden relative group">
      {/* Industrial Controls */}
      <div className="absolute top-8 left-8 right-8 z-20 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-4 pointer-events-auto">
          <div className="px-4 py-2 bg-white/5 border border-white/10 rounded backdrop-blur-sm flex items-center gap-3">
            <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
            <span className="text-[9px] uppercase font-black tracking-[0.3em] text-white/50">Node Designer Mode</span>
          </div>
          
          <div className="flex items-center gap-1 p-1 bg-white/5 border border-white/10 rounded">
            <button 
              onClick={() => addNode('Model')} 
              className="p-2 hover:bg-white/10 rounded text-white/40 hover:text-white transition-all flex items-center gap-2 px-3"
            >
              <Plus className="w-3 h-3" />
              <span className="text-[9px] uppercase font-bold tracking-widest">Add Node</span>
            </button>
          </div>
        </div>

        <div className="pointer-events-auto">
          <button className="flex items-center gap-3 px-6 py-2.5 bg-white text-black rounded hover:bg-white/90 transition-all font-bold shadow-[0_0_20px_rgba(255,255,255,0.1)]">
            <Play className="w-3 h-3 fill-current" />
            <span className="text-[10px] uppercase font-black tracking-[0.2em]">Demo</span>
          </button>
        </div>
      </div>
      
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.4 }}
        translateExtent={[[0, 0], [1400, 600]]}
        nodeExtent={[[0, 0], [1400, 600]]}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        panOnDrag={false}
        panOnScroll={false}
        preventScrolling={true}
        nodesDraggable={true}
        elementsSelectable={true}
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

      {/* Edge Vignette */}
      <div className="absolute inset-0 pointer-events-none border border-white/5" />
    </div>
  );
}
