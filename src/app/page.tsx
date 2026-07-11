'use client';

import Link from 'next/link';
import Image from 'next/image';
import { cn } from '@/lib/cn';
import { motion } from 'framer-motion';
import {
  Cpu,
  Github,
  Box,
  Brain,
  ShieldCheck,
  Check,
  Activity,
  ArrowRight,
} from 'lucide-react';
import HeroFlow from '@/components/HeroFlow';
import { IntegrationGrid } from '@/components/IntegrationGrid';
import { useState } from 'react';

export default function HomePage() {
  const [activeTab, setActiveTab] = useState('data');

  return (
    <div className="relative min-h-screen bg-black overflow-x-hidden selection:bg-orange-500/30 antialiased font-sans text-white">
      {/* Hero Section */}
      <section className="relative pt-20 pb-32 overflow-hidden px-6 bg-gradient-to-b from-black via-[#0d0705] to-black">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-0 right-0 w-full h-full bg-[radial-gradient(circle_at_70%_50%,rgba(255,77,0,0.03),transparent_70%)]" />
        </div>

        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center min-h-[70vh]">
            <motion.div
              initial={{ opacity: 0, x: -40 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
              className="relative z-10"
            >
              <h1 className="text-6xl md:text-8xl font-black tracking-tighter mb-8 leading-[0.9] text-white uppercase">
                FINETUNE <br />
                <span className="text-white/20">STUDIO</span>
              </h1>

              <p className="max-w-xl text-white/40 text-lg md:text-xl font-medium leading-relaxed mb-12">
                Drop a dataset, state a goal. An agent swarm writes the training code,
                runs it on your machine, and hands you a fine-tuned model — within the
                dollar budget you set.
              </p>

              <div className="flex flex-col sm:flex-row items-center gap-6">
                <Link href="/playground" className="w-full sm:w-auto px-10 py-5 bg-[#ff4d00] text-white rounded-xl text-sm font-bold hover:bg-[#ff4d00]/90 transition shadow-[0_0_40px_rgba(255,77,0,0.2)] flex items-center justify-center gap-2 group">
                  Open Studio
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </Link>
                <a href="https://github.com/tanish-24-git/No-code" target="_blank" rel="noreferrer" className="w-full sm:w-auto px-10 py-5 bg-white/5 border border-white/10 text-white rounded-xl text-sm font-bold hover:bg-white/10 transition flex items-center justify-center gap-2">
                  <Github className="w-5 h-5" />
                  GitHub
                </a>
              </div>

              <div className="flex flex-wrap gap-x-8 gap-y-3 mt-12 text-[11px] uppercase tracking-widest font-black text-white/30">
                <span>No hardcoded pipelines</span>
                <span>Any OpenAI-compatible LLM</span>
                <span>Budget-capped, not rate-limited</span>
                <span>Local-first</span>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1.5, ease: [0.22, 1, 0.36, 1] }}
              className="relative hidden lg:flex items-center justify-center"
            >
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,77,0,0.06),transparent_70%)] pointer-events-none" />
              <div className="relative w-full h-[650px] flex items-center justify-center">
                <div
                  className="relative w-full h-full flex items-center justify-center"
                  style={{
                    maskImage: 'radial-gradient(circle, black 40%, transparent 80%)',
                    WebkitMaskImage: 'radial-gradient(circle, black 40%, transparent 80%)',
                  }}
                >
                  <Image
                    src="/hero_lightning_bolt_removed_bg.png"
                    alt=""
                    width={700}
                    height={700}
                    priority
                    className="relative z-10 w-full h-auto max-w-[650px] object-contain grayscale brightness-[1.4] contrast-[1.3] opacity-80 hover:opacity-100 transition-all duration-1000 mix-blend-screen"
                  />
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* --- MAIN WORKFLOW CANVAS SECTION --- */}
      <section className="py-40 px-6 md:px-24 max-w-[1400px] mx-auto">
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
          className="grid grid-cols-1 lg:grid-cols-12 gap-16 items-stretch"
        >
          <div className="lg:col-span-4 lg:sticky lg:top-40 h-fit self-start space-y-4">
            <div className="space-y-2 mb-12">
              <h3 className="text-orange-500 text-xs font-black uppercase tracking-[0.4em]">How it works</h3>
              <p className="text-white/40 text-[11px] font-medium uppercase tracking-widest">Select a stage to visualize its flow</p>
            </div>
            <div className="space-y-3">
              <TabItem
                active={activeTab === 'data'}
                label="Dataset Analysis"
                desc="Agents write probes for YOUR data — schema, dupes, quality."
                onClick={() => setActiveTab('data')}
              />
              <TabItem
                active={activeTab === 'optimize'}
                label="Training Strategy"
                desc="LoRA / QLoRA config generated per dataset and hardware."
                onClick={() => setActiveTab('optimize')}
              />
              <TabItem
                active={activeTab === 'compute'}
                label="Hardware Probe"
                desc="GPU/VRAM detection drives model size and precision."
                onClick={() => setActiveTab('compute')}
              />
              <TabItem
                active={activeTab === 'monitor'}
                label="Detached Training"
                desc="Zero-token watcher streams loss, wakes agents on anomalies."
                onClick={() => setActiveTab('monitor')}
              />
              <TabItem
                active={activeTab === 'security'}
                label="Guardrails"
                desc="Plan approval, workspace isolation, budget ceiling."
                onClick={() => setActiveTab('security')}
              />
            </div>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 1 }}
            className="lg:col-span-8 relative group min-h-[600px] lg:h-[800px]"
          >
            <div className="absolute top-10 left-10 z-20 flex items-center gap-4 bg-black/60 backdrop-blur-2xl border border-white/10 px-6 py-3 rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)]">
              <div className="relative flex h-3 w-3">
                <div className="animate-ping absolute inline-flex h-full w-full rounded-full bg-orange-400 opacity-75"></div>
                <div className="relative inline-flex rounded-full h-3 w-3 bg-orange-500 shadow-[0_0_15px_#ff4d00]"></div>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] font-black text-white uppercase tracking-[0.2em] leading-none mb-1">Agent Graph</span>
                <span className="text-[9px] font-bold text-white/40 uppercase tracking-widest leading-none">Illustrative flow</span>
              </div>
            </div>

            <div className="absolute inset-0 bg-gradient-to-tr from-orange-500/5 via-transparent to-white/[0.02] rounded-[3rem] blur-3xl opacity-50 group-hover:opacity-70 transition-opacity duration-1000" />
            <div className="relative w-full h-full overflow-hidden rounded-[3rem] border border-white/10 bg-[#0d0d0f]/80 backdrop-blur-3xl shadow-[0_0_100px_rgba(0,0,0,0.5)]">
              <HeroFlow activeTab={activeTab} />
            </div>
          </motion.div>
        </motion.div>
      </section>

      {/* --- INTEGRATIONS SECTION --- */}
      <IntegrationGrid />

      {/* --- BENTO FEATURES SECTION --- */}
      <section className="py-40 px-6 md:px-24 max-w-[1440px] mx-auto">
        <div className="text-center mb-32">
          <h2 className="text-5xl md:text-7xl font-black mb-8 tracking-tighter leading-none text-white uppercase">
            Fine-tune models <br />
            <span className="text-white/20">you can actually follow</span>
          </h2>
          <p className="text-white/40 text-xl max-w-3xl mx-auto font-medium leading-relaxed">
            Drop any dataset. Watch every agent. Keep humans in the loop.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          <div className="lg:col-span-8 grid gap-6">
            <div className="h-[400px] group relative overflow-hidden bg-[#0d0d0f] border border-white/5 rounded-[32px] p-10 hover:border-orange-500/20 transition-all duration-500 flex flex-col">
              <div className="absolute top-0 right-0 w-[400px] h-[400px] bg-orange-500/5 blur-[100px] -mr-32 -mt-32" />
              <div className="space-y-4 relative z-10">
                <h3 className="text-3xl font-bold">Generated code, <br />not canned pipelines</h3>
                <p className="text-white/40 font-light max-w-sm text-lg">
                  CSV, JSONL, PDF, docs — agents write the preprocessing and training
                  scripts for your exact data, then run them in an isolated workspace.
                </p>
              </div>

              <div className="mt-auto pt-12 flex justify-center relative z-10">
                <div className="relative">
                  <div className="absolute inset-0 bg-white/5 blur-3xl rounded-full scale-150" />
                  <div className="relative flex items-center gap-8 opacity-40">
                    <Box className="w-16 h-16 text-white" />
                    <div className="w-24 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                    <Cpu className="w-20 h-20 text-white" />
                    <div className="w-24 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                    <Brain className="w-16 h-16 text-white" />
                  </div>
                </div>
              </div>
            </div>

            <div className="h-[300px] group relative overflow-hidden bg-[#0d0d0f] border border-white/5 rounded-[32px] p-10 hover:border-blue-500/20 transition-all duration-500 flex items-center gap-12">
              <div className="absolute bottom-0 left-0 w-[300px] h-[300px] bg-blue-500/5 blur-[80px] -ml-32 -mb-32" />
              <div className="flex-1 space-y-4 relative z-10">
                <h3 className="text-2xl font-bold">Runs on your machine</h3>
                <p className="text-white/40 font-light max-w-sm">Your data and weights never leave your disk. Bring any LLM endpoint for the agents.</p>
                <div className="space-y-2 pt-2">
                  {['Node + uv is the whole install', 'Any OpenAI-compatible or Anthropic-style API', 'Full source code, Apache-2.0'].map((item) => (
                    <div key={item} className="flex items-center gap-3 text-sm text-white/60">
                      <Check className="w-4 h-4 text-green-500" /> {item}
                    </div>
                  ))}
                </div>
              </div>
              <div className="hidden md:block w-48 h-48 relative z-10">
                <div className="absolute inset-0 bg-white/5 rounded-3xl rotate-12" />
                <div className="absolute inset-0 bg-[#161618] border border-white/10 rounded-3xl flex items-center justify-center">
                  <div className="text-center">
                    <ShieldCheck className="w-12 h-12 text-blue-500 mx-auto mb-2" />
                    <div className="text-[10px] font-bold text-white/40 uppercase tracking-widest">Local First</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-4 group relative overflow-hidden bg-[#0d0d0f] border border-white/5 rounded-[32px] p-10 hover:border-purple-500/20 transition-all duration-500">
            <div className="absolute bottom-0 right-0 w-[400px] h-[400px] bg-purple-500/5 blur-[100px] -mr-32 -mb-32" />
            <div className="h-full flex flex-col relative z-10">
              <div className="space-y-6 mb-12">
                <h3 className="text-3xl font-bold leading-tight">A budget, <br />not a meter</h3>
                <p className="text-white/40 text-lg leading-relaxed font-light">
                  Set a dollar ceiling in .env. The run pauses BEFORE it would overshoot
                  and asks you to top up — never a surprise bill.
                </p>
              </div>

              <div className="flex-1 space-y-6">
                <div className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] font-bold text-white/40 uppercase tracking-widest">Training run</span>
                    <div className="flex gap-1">
                      <div className="w-1 h-1 rounded-full bg-green-500" />
                      <div className="w-1 h-1 rounded-full bg-green-500" />
                      <div className="w-1 h-1 rounded-full bg-orange-500 animate-pulse" />
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="p-3 bg-white/5 border border-white/5 rounded-xl text-xs flex justify-between items-center">
                      <span className="text-white/60">step 420/720 · loss</span>
                      <span className="text-green-500">1.78 ↓</span>
                    </div>
                    <div className="p-3 bg-white/5 border border-white/5 rounded-xl text-xs flex justify-between items-center">
                      <span className="text-white/60">LLM spend</span>
                      <span className="text-green-500">$0.83 / $2.00</span>
                    </div>
                    <div className="p-3 bg-orange-500/10 border border-orange-500/20 rounded-xl text-xs">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-orange-500 font-bold">Human-in-the-loop</span>
                        <span className="text-[9px] px-1.5 py-0.5 bg-orange-500 text-white rounded">Plan approval</span>
                      </div>
                      <div className="text-white/40 italic text-[10px]">Approve once — the swarm executes the rest…</div>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-4 bg-white/5 rounded-2xl border border-white/5">
                  <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center">
                    <Activity className="w-4 h-4 text-purple-500" />
                  </div>
                  <div className="flex-1">
                    <div className="text-[10px] font-bold text-white/40">ANOMALY WATCHER</div>
                    <div className="text-xs text-white/60">NaN · OOM · divergence · stall — auto-recovery</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* --- FOOTER --- */}
      <footer className="mt-24 px-6 md:px-24">
        <div className="max-w-[1400px] mx-auto bg-gradient-to-br from-[#1a0f0a] to-[#0a0a0c] border-x border-t border-white/5 rounded-t-[40px] p-12 md:p-20 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-orange-500/5 blur-[120px] rounded-full -mr-64 -mt-64" />

          <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-16 mb-24">
            <div className="lg:col-span-6 space-y-8">
              <Link href="/" className="flex items-center gap-3 group">
                <div className="w-10 h-10 bg-orange-500 rounded-lg flex items-center justify-center shadow-[0_0_20px_rgba(255,77,0,0.3)] group-hover:scale-110 transition-transform">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <span className="text-2xl font-bold tracking-tighter text-white">
                  FineTune<span className="text-white/40 font-light">Studio</span>
                </span>
              </Link>
              <p className="text-white/40 text-lg font-medium leading-tight">
                Any dataset in, fine-tuned model out.
              </p>
              <div className="flex gap-6 pt-4">
                <a href="https://github.com/tanish-24-git/No-code" target="_blank" rel="noreferrer" className="text-white/40 hover:text-white transition-colors"><Github className="w-5 h-5" /></a>
              </div>
            </div>

            <div className="lg:col-span-6 flex lg:justify-end">
              <div className="grid grid-cols-2 gap-12 sm:gap-24">
                <div className="space-y-6">
                  <h4 className="text-xs font-bold text-white uppercase tracking-widest">Studio</h4>
                  <ul className="space-y-4 text-sm font-medium text-white/50">
                    <li><Link href="/playground" className="hover:text-orange-500 transition-colors">Playground</Link></li>
                    <li><Link href="/models" className="hover:text-orange-500 transition-colors">Models</Link></li>
                  </ul>
                </div>
                <div className="space-y-6">
                  <h4 className="text-xs font-bold text-white uppercase tracking-widest">Resources</h4>
                  <ul className="space-y-4 text-sm font-medium text-white/50">
                    <li><a href="https://github.com/tanish-24-git/No-code" target="_blank" rel="noreferrer" className="hover:text-orange-500 transition-colors">GitHub</a></li>
                    <li><a href="https://github.com/tanish-24-git/No-code#readme" target="_blank" rel="noreferrer" className="hover:text-orange-500 transition-colors">Docs</a></li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="relative z-10 py-12 border-t border-white/5 flex flex-col md:flex-row items-center justify-between gap-6 text-[11px] font-medium text-white/20 uppercase tracking-[0.2em]">
            <span>&copy; 2026 FINETUNE STUDIO</span>
            <a href="https://github.com/tanish-24-git" target="_blank" rel="noreferrer" className="hover:text-white transition-colors">Built by Tanish Jagtap</a>
          </div>
        </div>
      </footer>
    </div>
  );
}

function TabItem({ active, label, desc, onClick }: { active: boolean; label: string; desc: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full text-left p-4 rounded-xl border transition-all duration-500 relative overflow-hidden group',
        active
          ? 'bg-gradient-to-br from-white/10 to-white/5 border-white/20 ring-1 ring-white/10'
          : 'bg-transparent border-white/5 hover:border-white/10',
      )}
    >
      <div
        className={cn(
          'absolute left-0 top-0 w-1 h-full bg-orange-500 transition-transform duration-500',
          active ? 'scale-y-100' : 'scale-y-0',
        )}
      />
      <div className="relative z-10">
        <h4
          className={cn(
            'text-sm font-bold tracking-tight transition-colors duration-300',
            active ? 'text-white' : 'text-white/40 group-hover:text-white/70',
          )}
        >
          {label}
        </h4>
        <p
          className={cn(
            'text-[10px] leading-relaxed transition-colors duration-300 mt-1 line-clamp-1',
            active ? 'text-white/60' : 'text-white/20 group-hover:text-white/40',
          )}
        >
          {desc}
        </p>
      </div>
    </button>
  );
}
