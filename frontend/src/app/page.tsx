import Link from 'next/link';
import { cn } from '@/lib/cn';
import {
  Database,
  Wand2,
  Cpu,
  BarChart3,
  Github,
  Zap,
  Shield,
  Cpu as Chip,
  Layout,
  Box,
  GitBranch,
  Brain,
  ListChecks,
  HelpCircle,
  Sparkles,
  Play,
  Eye,
  ShieldCheck,
} from 'lucide-react';
import { Nav } from '@/components/Nav';
import HeroFlow from '@/components/HeroFlow';

export default function HomePage() {
  return (
    <div className="relative min-h-screen bg-black overflow-y-auto selection:bg-white/10 antialiased font-sans text-fg">
      {/* Primary Navigation */}
      <Nav />

      <section className="relative flex flex-col pt-52 px-6 md:px-24 pb-32 max-w-[1400px] mx-auto w-full items-center text-center">
        {/* Hero Section */}
        <div className="mb-24 animate-fade-in max-w-4xl flex flex-col items-center">
          <div className="flex items-center gap-3 mb-8">
            <div className="px-3 py-1 bg-white/5 border border-white/10 rounded flex items-center gap-2">
              <span className="text-[10px] font-black tracking-[0.2em] uppercase text-white/40">Build · v3 · Autonomous Hive</span>
            </div>
          </div>
          <h1 className="text-6xl md:text-9xl font-bold tracking-tight mb-10 leading-[1.0] text-white">
            Crap data → fortune weights.
          </h1>
          <p className="text-white/50 text-xl max-w-3xl leading-relaxed font-light">
            FineTune Studio is an open-source agentic studio for LLM fine-tuning. 19 specialized
            agents profile your data, probe your hardware, ask Socratic questions, draft a
            DoRA · GaLore · Unsloth pipeline, train it, recover from failure, and benchmark in a
            sandbox — all live-streamed, all in one prompt.
          </p>

          {/* Live "thought stream" sample */}
          <div className="mt-10 max-w-2xl w-full text-left rounded-lg border border-white/10 bg-white/[0.02] backdrop-blur-md p-5 space-y-2 font-mono text-[12px]">
            <ThoughtLine tone="thinking"   icon={Brain}      text="Probing hardware — 12GB VRAM, no MPS, CUDA 12.4." />
            <ThoughtLine tone="planning"   icon={ListChecks} text="1. Profile  2. Health-check  3. Rank models  4. Train" />
            <ThoughtLine tone="asking"     icon={HelpCircle} text="I see 30% duplicates. Merge or treat as separate domains?" />
            <ThoughtLine tone="garnishing" icon={Sparkles}   text="popping `train` node onto the canvas…" />
            <ThoughtLine tone="executing"  icon={Play}       text="step 240/720 · loss 1.7841 · 4.2 tok/ms" />
          </div>

          <div className="mt-8 flex items-center gap-3">
            <Link
              href="/playground"
              className="px-6 py-3 bg-white text-black rounded text-[11px] font-black uppercase tracking-[0.2em] hover:bg-white/90 transition"
            >
              Launch playground →
            </Link>
            <a
              href="https://github.com/tanish-24-git/finetune-studio"
              target="_blank"
              rel="noopener noreferrer"
              className="px-6 py-3 bg-transparent text-white border border-white/20 rounded text-[11px] font-black uppercase tracking-[0.2em] hover:bg-white/5 transition flex items-center gap-2"
            >
              <Github className="w-3.5 h-3.5" /> Star on GitHub
            </a>
          </div>
        </div>

        {/* Node-Based Workflow Visual (HeroFlow) */}
        <div className="w-full mb-40 animate-fade-in" style={{ animationDelay: '0.2s' }}>
          <HeroFlow />
        </div>

        {/* Project Context & Features */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-48">
          <FeatureCard
            icon={<Brain className="w-10 h-10 text-thinking/80" />}
            title="19-agent swarm"
            body="Hierarchical hive — Orchestrator, Data Alchemist, Architect, Audit Critic, Sandbox. Federated blackboard. Event bus. Loop-safe."
          />
          <FeatureCard
            icon={<Sparkles className="w-10 h-10 text-garnishing/80" />}
            title="Socratic streaming"
            body="Five colour-coded tones — thinking · planning · asking · garnishing · executing — make every decision auditable in real time."
          />
          <FeatureCard
            icon={<Cpu className="w-10 h-10 text-info/80" />}
            title="SOTA-2026 stack"
            body="DoRA, GaLore, Unsloth fused kernels, QLoRA-int4, DPO/ORPO alignment. Picked dynamically, with rationale."
          />
          <FeatureCard
            icon={<ShieldCheck className="w-10 h-10 text-warn/80" />}
            title="Audit Critic"
            body="An independent reviewer vetoes risky plans. Veto surfaces in the UI for explicit user resolution before any compute is spent."
          />
          <FeatureCard
            icon={<Zap className="w-10 h-10 text-success/80" />}
            title="TAO recovery"
            body="L1 retry, L2 adapt (OOM / NaN / divergence), L3 escalate. Crashes don't end the run — they trigger a plan diff with confidence scores."
          />
          <FeatureCard
            icon={<Eye className="w-10 h-10 text-success/80" />}
            title="Clean-room sandbox"
            body="Post-training MMLU / GSM8K / HumanEval lite benchmarks in an isolated process — no leaked context."
          />
          <FeatureCard
            icon={<Database className="w-10 h-10 text-white/60" />}
            title="Universal data alchemy"
            body="Recursive folder scan, schema induction, semantic dedup, sensitive-info redaction (emails / keys / PII), low-entropy filter."
          />
          <FeatureCard
            icon={<Shield className="w-10 h-10 text-white/60" />}
            title="Local-first, no telemetry"
            body="Pure FastAPI + Next.js. JSON on disk. 17 LLM providers — pick one, paste a key. Encrypted at rest."
          />
        </div>

        {/* Detailed Workflow (Text-based Deep Dive) */}
        <div className="space-y-32">
          <div className="flex items-center gap-8 mb-24">
            <h2 className="text-4xl font-light uppercase tracking-tighter text-white">Engineered for <span className="font-bold">Precision</span></h2>
            <div className="flex-1 h-px bg-white/10" />
          </div>
          
          <div className="space-y-48 px-4 md:px-0">
            <WorkflowStep 
              num="01"
              icon={<Database className="w-8 h-8" />}
              label="Data Management"
              title="Clean. Segment. Synthesize."
              desc="Ingest raw text or structured formats. Our engine automatically performs semantic deduplication and quality scoring to ensure only the best data reaches your model."
              side="left"
            />
            <WorkflowStep 
              num="02"
              icon={<Wand2 className="w-8 h-8" />}
              label="Optimization"
              title="Token-level Intelligence."
              desc="Transform data with hardware-aware tokenization. We optimize context packing to maximize GPU utilization and minimize training costs."
              side="right"
            />
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 py-12 px-12 bg-black">
        <div className="max-w-[1400px] mx-auto flex flex-col md:flex-row items-center justify-between gap-8">
          <div className="flex items-center gap-8 text-[10px] uppercase tracking-[0.2em] text-fg-3">
            <span>&copy; 2024 NO-CODE STUDIO</span>
            <span className="w-1 h-1 bg-white/20 rounded-full" />
            <span>Advanced LLM Fine-tuning</span>
          </div>
          
          <a 
            href="https://github.com/tanish-24-git" 
            target="_blank" 
            rel="noopener noreferrer"
            className="group flex items-center gap-3 px-6 py-3 bg-white/5 border border-white/10 rounded-full hover:bg-white hover:border-white transition-all"
          >
            <Github className="w-4 h-4 group-hover:text-black" />
            <span className="text-[11px] font-bold tracking-widest uppercase group-hover:text-black">
              Built by Tanish Jagtap
            </span>
          </a>
        </div>
      </footer>
    </div>
  );
}

function ThoughtLine({
  tone,
  icon: Icon,
  text,
}: {
  tone: 'thinking' | 'planning' | 'asking' | 'garnishing' | 'executing';
  icon: React.ComponentType<{ className?: string }>;
  text: string;
}) {
  const colorMap: Record<string, { dot: string; chip: string; text: string }> = {
    thinking:   { dot: 'bg-thinking',   chip: 'bg-thinking/15 text-thinking',     text: 'text-white/80 italic' },
    planning:   { dot: 'bg-planning',   chip: 'bg-planning/15 text-planning',     text: 'text-white/80' },
    asking:     { dot: 'bg-asking',     chip: 'bg-asking/15 text-asking',         text: 'text-white' },
    garnishing: { dot: 'bg-garnishing', chip: 'bg-garnishing/15 text-garnishing', text: 'text-white/85' },
    executing:  { dot: 'bg-executing',  chip: 'bg-executing/15 text-executing',   text: 'text-white/80' },
  };
  const s = colorMap[tone];
  return (
    <div className="flex items-center gap-3">
      <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', s.dot)} />
      <span className={cn('px-1.5 py-0.5 rounded text-[8.5px] font-black uppercase tracking-[0.18em]', s.chip)}>
        [{tone}]
      </span>
      <Icon className="w-3 h-3 text-white/40 shrink-0" />
      <span className={cn('text-[12px]', s.text)}>{text}</span>
    </div>
  );
}

function FeatureCard({ icon, title, body }: { icon: React.ReactNode; title: string; body: string }) {
  return (
    <div className="group p-8 bg-white/[0.02] border border-white/5 rounded-xl hover:bg-white/[0.04] hover:border-white/20 transition-all duration-500">
      <div className="mb-8 group-hover:scale-110 transition-transform">{icon}</div>
      <h3 className="text-lg font-bold mb-4 uppercase tracking-widest text-white">{title}</h3>
      <p className="text-sm text-white/50 leading-relaxed">{body}</p>
    </div>
  );
}

function WorkflowStep({ num, icon, label, title, desc, side }: { num: string; icon: React.ReactNode; label: string; title: string; desc: string; side: 'left' | 'right' }) {
  return (
    <div className={cn(
      "flex flex-col md:flex-row items-center gap-12 md:gap-24",
      side === 'right' ? "md:flex-row-reverse" : ""
    )}>
      <div className="flex-1 w-full flex flex-col gap-6 animate-fade-in">
        <div className="flex items-center gap-4">
          <span className="text-6xl font-black text-white/5 tracking-tighter">{num}</span>
          <div className="h-px flex-1 bg-white/10" />
        </div>
        <div className="space-y-4">
          <h4 className="text-white/40 text-xs font-black uppercase tracking-[0.4em]">{label}</h4>
          <h3 className="text-3xl font-bold text-white">{title}</h3>
          <p className="text-lg text-white/40 leading-relaxed max-w-xl">
            {desc}
          </p>
        </div>
      </div>
      
      <div className="flex-1 flex justify-center items-center relative animate-fade-in">
        <div className="absolute inset-0 bg-white/[0.02] blur-[100px] rounded-full" />
        <div className="w-48 h-48 md:w-64 md:h-64 rounded-xl border border-white/10 bg-white/5 flex items-center justify-center relative z-10 group hover:border-white/40 transition-all duration-500">
          <div className="text-white/60 transition-transform duration-500 group-hover:scale-110 group-hover:text-white">
            {icon}
          </div>
          {/* Subtle orbital ring */}
          <div className="absolute inset-[-20px] border border-white/[0.03] rounded-full animate-[spin_10s_linear_infinite]" />
        </div>
      </div>
    </div>
  );
}
