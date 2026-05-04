import Link from 'next/link';
import { cn } from '@/lib/cn';
import { Database, Wand2, Cpu, BarChart3, Github, Zap, Shield, Cpu as Chip, Layout, Box, GitBranch } from 'lucide-react';
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
              <span className="text-[10px] font-black tracking-[0.2em] uppercase text-white/40">Build Version 2.0</span>
            </div>
          </div>
          <h1 className="text-6xl md:text-9xl font-bold tracking-tight mb-10 leading-[1.0] text-white">
            Fine-tuning Studio.
          </h1>
          <p className="text-white/40 text-xl max-w-2xl leading-relaxed font-light">
            The ultimate no-code studio for specialized intelligence. 
            Automate datasets, optimize tokenization, and deploy high-performance models in minutes.
          </p>
        </div>

        {/* Node-Based Workflow Visual (HeroFlow) */}
        <div className="w-full mb-40 animate-fade-in" style={{ animationDelay: '0.2s' }}>
          <HeroFlow />
        </div>

        {/* Project Context & Features */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-48">
          <div className="group p-8 bg-white/[0.02] border border-white/5 rounded-xl hover:bg-white/[0.04] hover:border-white/20 transition-all duration-500">
            <Layout className="w-10 h-10 text-white/60 mb-8 group-hover:scale-110 transition-transform" />
            <h3 className="text-lg font-bold mb-4 uppercase tracking-widest text-white">Interactive Canvas</h3>
            <p className="text-sm text-white/40 leading-relaxed">
              Design complex training pipelines with a drag-and-drop interface. Connect datasets to preprocessors and models effortlessly.
            </p>
          </div>
          
          <div className="group p-8 bg-white/[0.02] border border-white/5 rounded-xl hover:bg-white/[0.04] hover:border-white/20 transition-all duration-500">
            <Box className="w-10 h-10 text-white/60 mb-8 group-hover:scale-110 transition-transform" />
            <h3 className="text-lg font-bold mb-4 uppercase tracking-widest text-white">Model Zoo</h3>
            <p className="text-sm text-white/40 leading-relaxed">
              Direct access to Llama-3, Mistral, and Gemma base weights. Pre-configured for LoRA, QLoRA, and full fine-tuning.
            </p>
          </div>

          <div className="group p-8 bg-white/[0.02] border border-white/5 rounded-xl hover:bg-white/[0.04] hover:border-white/20 transition-all duration-500">
            <GitBranch className="w-10 h-10 text-white/60 mb-8 group-hover:scale-110 transition-transform" />
            <h3 className="text-lg font-bold mb-4 uppercase tracking-widest text-white">Versioning</h3>
            <p className="text-sm text-white/40 leading-relaxed">
              Automatically track every training run, dataset version, and hyperparameter configuration for full reproducibility.
            </p>
          </div>

          <div className="group p-8 bg-white/[0.02] border border-white/5 rounded-xl hover:bg-white/[0.04] hover:border-white/20 transition-all duration-500">
            <Shield className="w-10 h-10 text-white/60 mb-8 group-hover:scale-110 transition-transform" />
            <h3 className="text-lg font-bold mb-4 uppercase tracking-widest text-white">Compliance</h3>
            <p className="text-sm text-white/40 leading-relaxed">
              Built-in safety filters and alignment monitoring to ensure your specialized models remain helpful and harmless.
            </p>
          </div>
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
