import type { Config } from 'tailwindcss';

// Tokens are lifted directly from finetune-studio.html so the Next.js port
// renders pixel-identically.
const config: Config = {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: { DEFAULT: '#000000', 2: '#09090b', 3: '#121212' },
        border: { DEFAULT: '#1a1a1a', 2: '#262626' },
        fg: { DEFAULT: '#ffffff', 2: '#a1a1aa', 3: '#52525b' },
        accent: { DEFAULT: '#ffffff', 2: '#e5e5e5' },
        success: { DEFAULT: '#10b981', dim: '#064e3b' },
        warn: { DEFAULT: '#f59e0b', dim: '#451a03' },
        danger: { DEFAULT: '#ef4444', dim: '#450a0a' },
        info: { DEFAULT: '#3b82f6', dim: '#172554' },
        node: { bg: '#09090b', border: '#1a1a1a', header: '#121212' },
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
