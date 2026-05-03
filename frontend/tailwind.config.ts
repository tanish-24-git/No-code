import type { Config } from 'tailwindcss';

// Tokens are lifted directly from finetune-studio.html so the Next.js port
// renders pixel-identically.
const config: Config = {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: { DEFAULT: '#0a0a0a', 2: '#111111', 3: '#161616' },
        border: { DEFAULT: '#1f1f1f', 2: '#2a2a2a' },
        fg: { DEFAULT: '#e8e8e8', 2: '#888888', 3: '#444444' },
        accent: { DEFAULT: '#e8e8e8', 2: '#b0b0b0' },
        success: { DEFAULT: '#4ade80', dim: '#1a3a27' },
        warn: { DEFAULT: '#fbbf24', dim: '#2d2510' },
        danger: { DEFAULT: '#f87171', dim: '#2d1a1a' },
        info: { DEFAULT: '#60a5fa', dim: '#1a2540' },
        node: { bg: '#0e0e0e', border: '#252525', header: '#141414' },
      },
      fontFamily: {
        // Backed by next/font/google in src/app/layout.tsx, which exposes
        // the actual font as a CSS variable.
        sans: ['var(--font-sans)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'monospace'],
      },
      animation: { pulse: 'pulse 2.5s ease-in-out infinite' },
    },
  },
  plugins: [],
};

export default config;
