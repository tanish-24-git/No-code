'use client';

import React from 'react';
import Image from 'next/image';
import { cn } from '@/lib/cn';

const integrationsRow1 = [
  { id: 'huggingface', src: '/huggingface-color.png' },
  { id: 'openai', src: '/images.png' },
  { id: 'gemini', src: '/Google_ai_studio_logo.png' },
  { id: 'nvidia', src: '/download.png' },
  { id: 'mistral', src: '/images (3).png' },
  { id: 'meta', src: '/images (2).png' },
];

const integrationsRow2 = [
  { id: 'groq', src: '/images (1).png' },
  { id: 'huggingface2', src: '/huggingface-color.png' },
  { id: 'openai2', src: '/images.png' },
  { id: 'nvidia2', src: '/download.png' },
  { id: 'mistral2', src: '/images (3).png' },
  { id: 'meta2', src: '/images (2).png' },
];

export function IntegrationGrid() {
  return (
    <div className="relative py-32 overflow-hidden bg-black">
      <div className="max-w-[1400px] mx-auto px-6 mb-20 text-center">
        <h2 className="text-4xl md:text-6xl font-bold tracking-tight mb-6">
          Industrial Ecosystem. <br />
          <span className="text-orange-500">Native Integration.</span>
        </h2>
        <p className="text-white/40 text-lg font-light max-w-2xl mx-auto">
          Built-in support for the world&apos;s leading AI models and infrastructure providers.
        </p>
      </div>

      <div className="relative space-y-12">
        {/* Row 1: Left to Right */}
        <div className="flex overflow-hidden [mask-image:linear-gradient(to_right,transparent,black_15%,black_85%,transparent)]">
          <div className="flex animate-scroll-right whitespace-nowrap gap-8 py-4">
            {[...integrationsRow1, ...integrationsRow1, ...integrationsRow1].map((item, i) => (
              <IntegrationCard key={i} item={item} />
            ))}
          </div>
        </div>

        {/* Row 2: Right to Left */}
        <div className="flex overflow-hidden [mask-image:linear-gradient(to_right,transparent,black_15%,black_85%,transparent)]">
          <div className="flex animate-scroll-left whitespace-nowrap gap-8 py-4">
            {[...integrationsRow2, ...integrationsRow2, ...integrationsRow2].map((item, i) => (
              <IntegrationCard key={i} item={item} />
            ))}
          </div>
        </div>
      </div>

      <style jsx global>{`
        @keyframes scroll-left {
          0% { transform: translateX(0); }
          100% { transform: translateX(-33.33%); }
        }
        @keyframes scroll-right {
          0% { transform: translateX(-33.33%); }
          100% { transform: translateX(0); }
        }
        .animate-scroll-left {
          animation: scroll-left 40s linear infinite;
        }
        .animate-scroll-right {
          animation: scroll-right 40s linear infinite;
        }
      `}</style>
    </div>
  );
}

function IntegrationCard({ item }: { item: any }) {
  return (
    <div className="group flex items-center justify-center px-12 py-8 bg-white/[0.02] border border-white/5 rounded-3xl hover:bg-white/[0.05] hover:border-white/10 transition-all cursor-default min-w-[200px]">
      <Image 
        src={item.src} 
        alt={item.id}
        width={120}
        height={48}
        className="h-12 w-auto object-contain opacity-40 group-hover:opacity-100 group-hover:scale-110 transition-all duration-300"
      />
    </div>
  );
}



