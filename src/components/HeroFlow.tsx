'use client';

import React from 'react';
import ReactFlow, {
  Background,
  BackgroundVariant,
  Handle,
  Position,
  Node,
  Edge,
  ReactFlowProvider,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { 
  Database, 
  Brain, 
  Cpu, 
  Zap,
  ShieldCheck,
  Activity,
  Layers,
  CloudUpload,
  Box,
  GitMerge
} from 'lucide-react';
import { cn } from '@/lib/cn';

// --- CUSTOM NODE COMPONENTS (Balanced Scale) ---

const TriggerNode = ({ data }: any) => (
  <div className="group relative">
    <div className="flex items-center">
      <div className="w-12 h-20 bg-[#1a1a1a] border-y border-l border-white/20 rounded-l-[40px] flex items-center justify-center">
        <Zap className="w-5 h-5 text-orange-500 fill-orange-500/20" />
      </div>
      <div className="px-5 py-3 bg-[#1a1a1a] border border-white/20 border-l-0 rounded-r-xl min-w-[160px] h-20 flex flex-col justify-center">
        <div className="text-[12px] font-black text-white/90 uppercase tracking-widest">{data.title}</div>
        <div className="text-[9px] text-white/40 mt-1 uppercase font-bold">{data.label}</div>
      </div>
    </div>
    <Handle type="source" position={Position.Right} style={{ background: '#fff', border: '2px solid #333', width: 8, height: 8 }} />
  </div>
);

const AgentNode = ({ data }: any) => (
  <div className="relative bg-[#1a1a1a] border border-white/20 rounded-[24px] min-w-[260px] shadow-2xl">
    <div className="p-6 flex items-center gap-4">
      <div className="w-12 h-12 bg-white/5 rounded-xl flex items-center justify-center border border-white/10">
        <data.icon className="w-6 h-6 text-white/80" />
      </div>
      <div>
        <div className="text-base font-bold text-white tracking-tight">{data.title}</div>
        <div className="text-[10px] text-white/40 font-mono">{data.status}</div>
      </div>
    </div>
    
    <div className="border-t border-white/10 px-6 py-3 flex justify-between bg-white/[0.01] rounded-b-[24px]">
      <div className="flex flex-col items-center gap-1">
        <Handle type="target" id="input-1" position={Position.Bottom} style={{ background: '#fff', border: '2px solid #333', width: 8, height: 8 }} />
        <span className="text-[7px] text-white/30 uppercase font-black">HW</span>
      </div>
      <div className="flex flex-col items-center gap-1">
        <Handle type="target" id="input-2" position={Position.Bottom} style={{ background: '#fff', border: '2px solid #333', width: 8, height: 8 }} />
        <span className="text-[7px] text-white/30 uppercase font-black">Logic</span>
      </div>
      <div className="flex flex-col items-center gap-1">
        <Handle type="target" id="input-3" position={Position.Bottom} style={{ background: '#fff', border: '2px solid #333', width: 8, height: 8 }} />
        <span className="text-[7px] text-white/30 uppercase font-black">Tools</span>
      </div>
    </div>

    <Handle type="target" position={Position.Left} style={{ background: '#fff', border: '2px solid #333', width: 8, height: 8 }} />
    <Handle type="source" position={Position.Right} style={{ background: '#fff', border: '2px solid #333', width: 8, height: 8 }} />
  </div>
);

const LogicNode = ({ data }: any) => (
  <div className="relative bg-[#1a1a1a] border border-white/20 rounded-xl p-5 min-w-[160px] flex items-center gap-3">
    <div className="w-10 h-10 bg-green-500/10 rounded-lg flex items-center justify-center border border-green-500/20">
      <GitMerge className="w-5 h-5 text-green-500" />
    </div>
    <div>
      <div className="text-[11px] font-bold text-white/90 uppercase tracking-tight">{data.title}</div>
      <div className="text-[8px] text-white/20 uppercase font-black">Verification</div>
    </div>

    <Handle type="target" position={Position.Left} style={{ background: '#fff', border: '2px solid #333', width: 8, height: 8 }} />
    <Handle type="source" id="true" position={Position.Right} style={{ top: '30%', background: '#fff', border: '2px solid #333', width: 8, height: 8 }} />
    <Handle type="source" id="false" position={Position.Right} style={{ top: '70%', background: '#fff', border: '2px solid #333', width: 8, height: 8 }} />
  </div>
);

const CircleNode = ({ data }: any) => (
  <div className="flex flex-col items-center gap-3">
    <div className="w-16 h-16 rounded-full bg-[#1a1a1a] border border-white/20 flex items-center justify-center shadow-xl group hover:border-orange-500 transition-all">
      <data.icon className="w-7 h-7 text-white/60 group-hover:text-orange-500 transition-all" />
      <Handle type="source" position={Position.Top} style={{ background: '#fff', border: '2px solid #333', width: 8, height: 8 }} />
      <Handle type="target" position={Position.Left} style={{ background: '#fff', border: '2px solid #333', width: 8, height: 8 }} />
    </div>
    <div className="text-[9px] font-bold text-white/30 uppercase tracking-[0.2em] text-center max-w-[100px]">{data.label}</div>
  </div>
);

// --- DATASET ---


// --- DYNAMIC DATA MAP ---
const FLOW_DATA = {
  data: {
    nodes: [
      { id: 't1', type: 'trigger', position: { x: 0, y: 150 }, data: { title: 'Intake', label: 'Dataset Entry' } },
      { id: 'a1', type: 'agent', position: { x: 300, y: 125 }, data: { title: 'Cleaning Swarm', status: 'PII Scrubbing', icon: Database } },
      { id: 'l1', type: 'logic', position: { x: 650, y: 145 }, data: { title: 'Audit Gate' } },
      { id: 'c1', type: 'circle', position: { x: 920, y: 40 }, data: { label: 'Secure Vault', icon: ShieldCheck } },
      { id: 'c2', type: 'circle', position: { x: 920, y: 260 }, data: { label: 'Clean Dataset', icon: Box } },
      { id: 'in-1', type: 'circle', position: { x: 260, y: 380 }, data: { label: 'Deduplicator', icon: Activity } },
      { id: 'in-2', type: 'circle', position: { x: 400, y: 380 }, data: { label: 'RegEx Engine', icon: GitMerge } },
    ],
    edges: [
      { id: 'e1', source: 't1', target: 'a1', style: { stroke: '#ff4d00', strokeWidth: 2 } },
      { id: 'e2', source: 'a1', target: 'l1', style: { stroke: '#fff', strokeWidth: 2 } },
      { id: 'e3', source: 'l1', sourceHandle: 'true', target: 'c1', animated: true, style: { stroke: '#22c55e', strokeWidth: 2 } },
      { id: 'e4', source: 'l1', sourceHandle: 'false', target: 'c2', style: { stroke: '#ef4444', strokeWidth: 2 } },
      { id: 'ei1', source: 'in-1', target: 'a1', targetHandle: 'input-1', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
      { id: 'ei2', source: 'in-2', target: 'a1', targetHandle: 'input-2', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
    ]
  },
  optimize: {
    nodes: [
      { id: 't1', type: 'trigger', position: { x: 0, y: 150 }, data: { title: 'Weights', label: 'Base Model Llama-3' } },
      { id: 'a1', type: 'agent', position: { x: 300, y: 125 }, data: { title: 'PEFT Architect', status: 'LoRA / QLoRA', icon: Cpu } },
      { id: 'l1', type: 'logic', position: { x: 650, y: 145 }, data: { title: 'Rank Check' } },
      { id: 'c1', type: 'circle', position: { x: 920, y: 40 }, data: { label: 'Adapter Hub', icon: CloudUpload } },
      { id: 'c2', type: 'circle', position: { x: 920, y: 260 }, data: { label: 'Model Registry', icon: Box } },
      { id: 'in-1', type: 'circle', position: { x: 260, y: 380 }, data: { label: 'Quantizer', icon: Layers } },
      { id: 'in-2', type: 'circle', position: { x: 400, y: 380 }, data: { label: 'Alpha Scaler', icon: Activity } },
    ],
    edges: [
      { id: 'e1', source: 't1', target: 'a1', style: { stroke: '#ff4d00', strokeWidth: 2 } },
      { id: 'e2', source: 'a1', target: 'l1', style: { stroke: '#fff', strokeWidth: 2 } },
      { id: 'e3', source: 'l1', sourceHandle: 'true', target: 'c1', animated: true, style: { stroke: '#22c55e', strokeWidth: 2 } },
      { id: 'e4', source: 'l1', sourceHandle: 'false', target: 'c2', style: { stroke: '#ef4444', strokeWidth: 2 } },
      { id: 'ei1', source: 'in-1', target: 'a1', targetHandle: 'input-1', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
      { id: 'ei2', source: 'in-2', target: 'a1', targetHandle: 'input-2', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
    ]
  },
  compute: {
    nodes: [
      { id: 't1', type: 'trigger', position: { x: 0, y: 150 }, data: { title: 'Job Queue', label: 'Batch Training' } },
      { id: 'a1', type: 'agent', position: { x: 300, y: 125 }, data: { title: 'VRAM Balancer', status: 'H100 Allocation', icon: Zap } },
      { id: 'l1', type: 'logic', position: { x: 650, y: 145 }, data: { title: 'Memory Check' } },
      { id: 'c1', type: 'circle', position: { x: 920, y: 40 }, data: { label: 'Cloud Compute', icon: CloudUpload } },
      { id: 'c2', type: 'circle', position: { x: 920, y: 260 }, data: { label: 'Device Target', icon: Cpu } },
      { id: 'in-1', type: 'circle', position: { x: 260, y: 380 }, data: { label: 'OOM Protect', icon: ShieldCheck } },
      { id: 'in-2', type: 'circle', position: { x: 400, y: 380 }, data: { label: 'Node Scaler', icon: Layers } },
    ],
    edges: [
      { id: 'e1', source: 't1', target: 'a1', style: { stroke: '#ff4d00', strokeWidth: 2 } },
      { id: 'e2', source: 'a1', target: 'l1', style: { stroke: '#fff', strokeWidth: 2 } },
      { id: 'e3', source: 'l1', sourceHandle: 'true', target: 'c1', animated: true, style: { stroke: '#22c55e', strokeWidth: 2 } },
      { id: 'e4', source: 'l1', sourceHandle: 'false', target: 'c2', style: { stroke: '#ef4444', strokeWidth: 2 } },
      { id: 'ei1', source: 'in-1', target: 'a1', targetHandle: 'input-1', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
      { id: 'ei2', source: 'in-2', target: 'a1', targetHandle: 'input-2', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
    ]
  },
  monitor: {
    nodes: [
      { id: 't1', type: 'trigger', position: { x: 0, y: 150 }, data: { title: 'Training Stream', label: 'Epoch 12/50' } },
      { id: 'a1', type: 'agent', position: { x: 300, y: 125 }, data: { title: 'Loss Observer', status: 'Tracking Metrics', icon: Activity } },
      { id: 'l1', type: 'logic', position: { x: 650, y: 145 }, data: { title: 'Eval Gate' } },
      { id: 'c1', type: 'circle', position: { x: 920, y: 40 }, data: { label: 'W&B Log', icon: GitMerge } },
      { id: 'c2', type: 'circle', position: { x: 920, y: 260 }, data: { label: 'Checkpoint', icon: Box } },
      { id: 'in-1', type: 'circle', position: { x: 260, y: 380 }, data: { label: 'Gradient Norm', icon: Zap } },
      { id: 'in-2', type: 'circle', position: { x: 400, y: 380 }, data: { label: 'Learning Rate', icon: Activity } },
    ],
    edges: [
      { id: 'e1', source: 't1', target: 'a1', style: { stroke: '#ff4d00', strokeWidth: 2 } },
      { id: 'e2', source: 'a1', target: 'l1', style: { stroke: '#fff', strokeWidth: 2 } },
      { id: 'e3', source: 'l1', sourceHandle: 'true', target: 'c1', animated: true, style: { stroke: '#22c55e', strokeWidth: 2 } },
      { id: 'e4', source: 'l1', sourceHandle: 'false', target: 'c2', style: { stroke: '#ef4444', strokeWidth: 2 } },
      { id: 'ei1', source: 'in-1', target: 'a1', targetHandle: 'input-1', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
      { id: 'ei2', source: 'in-2', target: 'a1', targetHandle: 'input-2', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
    ]
  },
  security: {
    nodes: [
      { id: 't1', type: 'trigger', position: { x: 0, y: 150 }, data: { title: 'User Access', label: 'Auth Handshake' } },
      { id: 'a1', type: 'agent', position: { x: 300, y: 125 }, data: { title: 'Security Vault', status: 'AES-256 Encryption', icon: ShieldCheck } },
      { id: 'l1', type: 'logic', position: { x: 650, y: 145 }, data: { title: 'Policy Gate' } },
      { id: 'c1', type: 'circle', position: { x: 920, y: 40 }, data: { label: 'Audit Logs', icon: Database } },
      { id: 'c2', type: 'circle', position: { x: 920, y: 260 }, data: { label: 'Secure Storage', icon: Box } },
      { id: 'in-1', type: 'circle', position: { x: 260, y: 380 }, data: { label: 'VPN Tunnel', icon: Activity } },
      { id: 'in-2', type: 'circle', position: { x: 400, y: 380 }, data: { label: 'Token Manager', icon: Layers } },
    ],
    edges: [
      { id: 'e1', source: 't1', target: 'a1', style: { stroke: '#ff4d00', strokeWidth: 2 } },
      { id: 'e2', source: 'a1', target: 'l1', style: { stroke: '#fff', strokeWidth: 2 } },
      { id: 'e3', source: 'l1', sourceHandle: 'true', target: 'c1', animated: true, style: { stroke: '#22c55e', strokeWidth: 2 } },
      { id: 'e4', source: 'l1', sourceHandle: 'false', target: 'c2', style: { stroke: '#ef4444', strokeWidth: 2 } },
      { id: 'ei1', source: 'in-1', target: 'a1', targetHandle: 'input-1', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
      { id: 'ei2', source: 'in-2', target: 'a1', targetHandle: 'input-2', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
    ]
  },
  distill: {
    nodes: [
      { id: 't1', type: 'trigger', position: { x: 0, y: 150 }, data: { title: 'Teacher Model', label: '70B Param LLM' } },
      { id: 'a1', type: 'agent', position: { x: 300, y: 125 }, data: { title: 'Model Distiller', status: 'Knowledge Transfer', icon: Layers } },
      { id: 'l1', type: 'logic', position: { x: 650, y: 145 }, data: { title: 'Loss Match' } },
      { id: 'c1', type: 'circle', position: { x: 920, y: 40 }, data: { label: 'Edge Target', icon: Cpu } },
      { id: 'c2', type: 'circle', position: { x: 920, y: 260 }, data: { label: 'Student Model', icon: Box } },
      { id: 'in-1', type: 'circle', position: { x: 260, y: 380 }, data: { label: 'Layer Pruning', icon: Activity } },
      { id: 'in-2', type: 'circle', position: { x: 400, y: 380 }, data: { label: 'Sparsity Mask', icon: Zap } },
    ],
    edges: [
      { id: 'e1', source: 't1', target: 'a1', style: { stroke: '#ff4d00', strokeWidth: 2 } },
      { id: 'e2', source: 'a1', target: 'l1', style: { stroke: '#fff', strokeWidth: 2 } },
      { id: 'e3', source: 'l1', sourceHandle: 'true', target: 'c1', animated: true, style: { stroke: '#22c55e', strokeWidth: 2 } },
      { id: 'e4', source: 'l1', sourceHandle: 'false', target: 'c2', style: { stroke: '#ef4444', strokeWidth: 2 } },
      { id: 'ei1', source: 'in-1', target: 'a1', targetHandle: 'input-1', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
      { id: 'ei2', source: 'in-2', target: 'a1', targetHandle: 'input-2', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
    ]
  },
  prompt: {
    nodes: [
      { id: 't1', type: 'trigger', position: { x: 0, y: 150 }, data: { title: 'System Prompt', label: 'V0 Initial' } },
      { id: 'a1', type: 'agent', position: { x: 300, y: 125 }, data: { title: 'Prompt Tuner', status: 'RLHF Optimization', icon: Brain } },
      { id: 'l1', type: 'logic', position: { x: 650, y: 145 }, data: { title: 'Coherence Check' } },
      { id: 'c1', type: 'circle', position: { x: 920, y: 40 }, data: { label: 'Prompt Library', icon: Database } },
      { id: 'c2', type: 'circle', position: { x: 920, y: 260 }, data: { label: 'Final Template', icon: GitMerge } },
      { id: 'in-1', type: 'circle', position: { x: 260, y: 380 }, data: { label: 'Thought Trace', icon: Activity } },
      { id: 'in-2', type: 'circle', position: { x: 400, y: 380 }, data: { label: 'Reward Model', icon: ShieldCheck } },
    ],
    edges: [
      { id: 'e1', source: 't1', target: 'a1', style: { stroke: '#ff4d00', strokeWidth: 2 } },
      { id: 'e2', source: 'a1', target: 'l1', style: { stroke: '#fff', strokeWidth: 2 } },
      { id: 'e3', source: 'l1', sourceHandle: 'true', target: 'c1', animated: true, style: { stroke: '#22c55e', strokeWidth: 2 } },
      { id: 'e4', source: 'l1', sourceHandle: 'false', target: 'c2', style: { stroke: '#ef4444', strokeWidth: 2 } },
      { id: 'ei1', source: 'in-1', target: 'a1', targetHandle: 'input-1', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
      { id: 'ei2', source: 'in-2', target: 'a1', targetHandle: 'input-2', style: { stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1.5, strokeDasharray: '4,4' } },
    ]
  },
};

// Module scope: stable identity across renders AND strict-mode double-mounts
// (useMemo inside the component still trips React Flow warning #002 in dev).
const HERO_NODE_TYPES = {
  trigger: TriggerNode,
  agent: AgentNode,
  logic: LogicNode,
  circle: CircleNode,
};
const HERO_FIT_VIEW = { padding: 0.25 };

function FlowContent({ activeTab }: { activeTab: string }) {
  const { nodes, edges } = (FLOW_DATA as any)[activeTab] || FLOW_DATA.data;

  return (
    <div className="w-full h-full bg-[#0a0a0c] relative">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={HERO_NODE_TYPES}
        fitView
        fitViewOptions={HERO_FIT_VIEW}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        panOnScroll={false}
        preventScrolling={true}
        className="pointer-events-none"
      >
        <Background 
          variant={BackgroundVariant.Dots} 
          color="#ffffff" 
          size={1} 
          gap={24} 
          className="opacity-[0.15]"
        />
      </ReactFlow>
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_center,transparent_0%,rgba(10,10,12,0.8)_100%)]" />
    </div>
  );
}

export default function HeroFlow({ activeTab = 'data' }: { activeTab?: string }) {
  return (
    <div className="w-full h-full">
      <ReactFlowProvider>
        <FlowContent activeTab={activeTab} />
      </ReactFlowProvider>
    </div>
  );
}
