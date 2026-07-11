'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/cn';
import { Brain, Github, Package, PlayCircle } from 'lucide-react';

const links = [
  { href: '/playground', label: 'Playground', icon: <PlayCircle className="w-3.5 h-3.5" /> },
  { href: '/models', label: 'Models', icon: <Package className="w-3.5 h-3.5" /> },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <div className="fixed top-6 inset-x-0 z-50 flex justify-center px-6">
      <nav className="w-full max-w-[1200px] h-16 bg-black/60 backdrop-blur-2xl border border-white/10 rounded-2xl px-6 md:px-8 flex items-center justify-between shadow-[0_20px_50px_rgba(0,0,0,0.5)]">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-8 h-8 bg-orange-500 rounded-lg flex items-center justify-center shadow-[0_0_15px_rgba(255,77,0,0.3)] group-hover:scale-110 transition-transform">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <span className="text-lg font-bold tracking-tighter text-white hidden sm:inline-block">
            FineTune<span className="text-white/40 font-light">Studio</span>
          </span>
        </Link>

        {/* Desktop Links */}
        <div className="hidden md:flex items-center gap-6">
          {links.map((l) => (
            <Link
              key={l.label}
              href={l.href}
              className={cn(
                "flex items-center gap-2 text-[13px] font-medium transition-colors",
                pathname === l.href ? "text-orange-500" : "text-white/60 hover:text-white"
              )}
            >
              {l.icon}
              {l.label}
            </Link>
          ))}
        </div>

        {/* Right Actions */}
        <div className="flex items-center gap-4">
          <a 
            href="https://github.com/tanish-24-git/No-code" 
            target="_blank" 
            rel="noopener noreferrer"
            className="p-2 bg-white/5 border border-white/10 rounded-lg text-white/60 hover:text-white hover:bg-white/10 transition-all"
            title="GitHub Repository"
          >
            <Github className="w-5 h-5" />
          </a>
          
          <div className="h-4 w-[1px] bg-white/10 mx-2 hidden sm:block" />
          
          <Link
            href="/playground"
            className="px-4 py-2 bg-[#ff4d00] text-white rounded-lg text-sm font-bold hover:bg-[#ff4d00]/90 transition shadow-[0_0_20px_rgba(255,77,0,0.2)]"
          >
            Get Started
          </Link>
        </div>
      </nav>
    </div>
  );
}
