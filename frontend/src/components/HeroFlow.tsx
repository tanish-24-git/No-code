'use client';

import React, { useMemo, useCallback, useState, useEffect, useRef } from 'react';
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
  Connection,
  MarkerType,
  ReactFlowProvider,
  useReactFlow
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Database, Wand2, Cpu, BarChart3, Plus, MoreHorizontal, CheckCircle2, X } from 'lucide-react';
import { cn } from '@/lib/cn';

const initialNodes: Node[] = [
  {
    id: '1',
    type: 'custom',
    data: { label: 'Dataset', icon: <Database />, desc: 'Raw source data' },
    position: { x: 0, y: 220 },
  },
  {
    id: '2',
    type: 'custom',
    data: { label: 'Preprocess', icon: <Wand2 />, desc: 'Token optimization' },
    position: { x: 300, y: 220 },
  },
  {
    id: '3',
    type: 'custom',
    data: { label: 'Fine-tune', icon: <Cpu />, desc: 'Weight adjustment' },
    position: { x: 600, y: 220 },
  },
];

const initialEdges: Edge[] = [];

function CustomNode({ data }: { data: any }) {
  if (data.isHidden) return null;
  return (
    <div className={cn(
      "group relative animate-pop",
      (data.isSuccess || data.isActive) && "z-10"
    )}>
      {/* Background Glow */}
      <div className={cn(
        "absolute inset-[-1px] blur-2xl rounded-xl opacity-0 transition-all duration-700",
        data.isSuccess ? "bg-white/20 opacity-100" : (data.isActive ? "bg-white/10 opacity-50" : "bg-white/[0.01] group-hover:opacity-100")
      )} />
      
      <div className={cn(
        "relative px-6 py-4 bg-[#0a0a0a] border-2 rounded-xl min-w-[200px] transition-all duration-700 select-none",
        data.isSuccess ? "border-white shadow-[0_0_40px_rgba(255,255,255,0.25)] scale-[1.05]" : 
        (data.isActive ? "border-white/50 shadow-[0_0_20px_rgba(255,255,255,0.1)]" : "border-white/10 hover:border-white/30")
      )}>
        {/* Shine Overlay */}
        {(data.isSuccess || data.isActive) && (
          <div className="absolute inset-0 rounded-xl overflow-hidden pointer-events-none">
            <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/[0.05] to-transparent animate-pulse" />
          </div>
        )}

        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={cn(
              "transition-all duration-500",
              data.isSuccess ? "text-white drop-shadow-[0_0_8px_white]" : 
              (data.isActive ? "text-white opacity-100" : "text-white opacity-60 group-hover:opacity-100")
            )}>
              {React.cloneElement(data.icon as React.ReactElement, { className: "w-5 h-5" })}
            </div>
            <div className="flex flex-col">
              <span className="text-[11px] font-black uppercase tracking-[0.25em] text-white/90">
                {data.label}
              </span>
              {data.isSuccess && (
                <span className="text-[8px] text-white/80 font-bold uppercase tracking-widest mt-1 animate-pulse">
                  System Active
                </span>
              )}
            </div>
          </div>
        </div>
        
        <div className="h-[1px] w-full bg-white/5 mb-4" />
        
        <p className="text-[9px] text-white/30 leading-relaxed font-medium uppercase tracking-wider">
          {data.desc}
        </p>
        
        <Handle type="target" position={Position.Left} className="!w-2 !h-2 !bg-white !border-none !left-[-4px]" />
        <Handle type="source" position={Position.Right} className="!w-2 !h-2 !bg-white !border-none !right-[-4px]" />
      </div>
    </div>
  );
}

function FlowContent() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [isDemoRunning, setIsDemoRunning] = useState(false);
  const [showNotification, setShowNotification] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const hasStartedRef = useRef(false);
  const { fitView } = useReactFlow();

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge({ 
      ...params, 
      animated: true, 
      style: { stroke: '#fff', strokeWidth: 2, opacity: 0.9, filter: 'drop-shadow(0 0 4px white)' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#fff' }
    }, eds)),
    [setEdges]
  );

  const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);

  const runDemo = useCallback(async () => {
    if (isDemoRunning) return;
    
    setIsDemoRunning(true);
    setShowNotification(false);
    setNodes([]);
    setEdges([]);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    const demoNodesData = [
      { id: '1', label: 'Dataset', icon: <Database />, desc: 'Raw source data', x: 0, y: 220 },
      { id: '2', label: 'Preprocess', icon: <Wand2 />, desc: 'Token optimization', x: 280, y: 220 },
      { id: '3', label: 'Fine-tune', icon: <Cpu />, desc: 'Weight adjustment', x: 560, y: 220 },
      { id: '4', label: 'Model', icon: <BarChart3 />, desc: 'Production Ready', x: 840, y: 220 }
    ];

    // Pop each node
    for (const nodeData of demoNodesData) {
      await sleep(600);
      const newNode: Node = {
        id: nodeData.id,
        type: 'custom',
        data: { label: nodeData.label, icon: nodeData.icon, desc: nodeData.desc, isActive: true },
        position: { x: nodeData.x, y: nodeData.y },
      };
      setNodes((nds) => [...nds, newNode]);
      fitView({ duration: 800, padding: 1.2 });
      
      // Briefly deactivate isActive to stop the shine after pop
      setTimeout(() => {
        setNodes((nds) => nds.map(n => n.id === nodeData.id ? { ...n, data: { ...n.data, isActive: false } } : n));
      }, 1000);
    }

    // Connect nodes one by one
    await sleep(800);
    for (let i = 0; i < demoNodesData.length - 1; i++) {
      await sleep(600);
      
      // Light up the nodes involved in connection
      setNodes((nds) => nds.map(n => 
        (n.id === demoNodesData[i].id || n.id === demoNodesData[i+1].id) 
          ? { ...n, data: { ...n.data, isActive: true } } 
          : n
      ));

        const edge: Edge = {
          id: `e${demoNodesData[i].id}-${demoNodesData[i+1].id}`,
          source: demoNodesData[i].id,
          target: demoNodesData[i+1].id,
          type: 'straight',
          animated: true,
          style: { stroke: '#ffffff', strokeWidth: 1, opacity: 0.6 },
          markerEnd: { type: MarkerType.ArrowClosed, color: '#ffffff', width: 10, height: 10 }
        };
      setEdges((eds) => [...eds, edge]);
      
      await sleep(400);
      setNodes((nds) => nds.map(n => ({ ...n, data: { ...n.data, isActive: false } })));
    }

    // Final finish sequence
    await sleep(1500);
    
    // Mark the final model node as successful with maximum shine
    setNodes((nds) => nds.map(node => 
      node.id === '4' ? { ...node, data: { ...node.data, isSuccess: true, isActive: false } } : node
    ));

    setShowNotification(true);
    setIsDemoRunning(false);

    // Auto hide notification
    setTimeout(() => setShowNotification(false), 5000);
  }, [isDemoRunning, setNodes, setEdges, fitView]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasStartedRef.current) {
          hasStartedRef.current = true;
          runDemo();
        }
      },
      { threshold: 0.5 }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, [runDemo]);

  return (
    <div ref={containerRef} className="w-full h-[600px] border border-white/5 rounded-2xl bg-[#080808] overflow-hidden relative group">
      {/* Notification Overlay */}
      {showNotification && (
        <div className="absolute top-24 left-1/2 -translate-x-1/2 z-50 animate-pop">
          <div className="bg-white text-black px-6 py-4 rounded-lg flex items-center gap-4 shadow-[0_0_50px_rgba(255,255,255,0.2)] border border-white/20">
            <div className="bg-black/5 p-2 rounded-full">
              <CheckCircle2 className="w-5 h-5 text-black" />
            </div>
            <div>
              <p className="text-xs font-black uppercase tracking-widest">Pipeline Complete</p>
              <p className="text-[10px] opacity-60 font-medium">Model successfully deployed to edge node</p>
            </div>
            <button 
              onClick={() => setShowNotification(false)}
              className="ml-4 hover:opacity-50 transition-opacity"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Industrial Controls */}
      <div className="absolute top-8 left-8 right-8 z-20 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-4 pointer-events-auto">
          <div className="px-4 py-2 bg-white/5 border border-white/10 rounded backdrop-blur-sm flex items-center gap-3">
            <div className={cn("w-1.5 h-1.5 rounded-full animate-pulse", isDemoRunning ? "bg-green-500 shadow-[0_0_10px_#22c55e]" : "bg-white")} />
            <span className="text-[9px] uppercase font-black tracking-[0.3em] text-white/50">
              {isDemoRunning ? 'Processing Pipeline...' : 'Node Designer Mode'}
            </span>
          </div>
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
        fitViewOptions={{ padding: 1.2 }}
        translateExtent={[[-500, -500], [1500, 1500]]}
        nodeExtent={[[-500, -500], [1500, 1500]]}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        panOnDrag={false}
        panOnScroll={false}
        minZoom={0.1}
        preventScrolling={true}
        nodesDraggable={!isDemoRunning}
        elementsSelectable={!isDemoRunning}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#ffffff" variant="dots" gap={40} size={1} style={{ backgroundColor: '#1a1a1a' }} />
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

export default function HeroFlow() {
  return (
    <ReactFlowProvider>
      <FlowContent />
    </ReactFlowProvider>
  );
}
