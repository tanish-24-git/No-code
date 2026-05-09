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
        // Socratic stream tones (blueprint §11).
        thinking:   { DEFAULT: '#a855f7', dim: '#3b0764' }, // purple
        planning:   { DEFAULT: '#3b82f6', dim: '#172554' }, // blue (= info)
        asking:     { DEFAULT: '#f59e0b', dim: '#451a03' }, // yellow (= warn)
        garnishing: { DEFAULT: '#06b6d4', dim: '#083344' }, // cyan
        executing:  { DEFAULT: '#10b981', dim: '#064e3b' }, // green (= success)
      },
      fontFamily: {
        // Backed by next/font/google in src/app/layout.tsx, which exposes
        // the actual font as a CSS variable.
        sans: ['var(--font-sans)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'monospace'],
      },
      animation: {
        pulse: 'pulse 2.5s ease-in-out infinite',
        'pop-in': 'pop-in 0.35s cubic-bezier(0.34, 1.56, 0.64, 1)',
        'fade-in': 'fade-in 0.3s ease-out',
        glow: 'glow 1.6s ease-out',
      },
      keyframes: {
        'pop-in': {
          '0%': { transform: 'scale(0.85) translateY(8px)', opacity: '0' },
          '100%': { transform: 'scale(1) translateY(0)', opacity: '1' },
        },
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 0 rgba(168,85,247,0)' },
          '50%': { boxShadow: '0 0 24px rgba(168,85,247,0.55)' },
          '100%': { boxShadow: '0 0 0 rgba(168,85,247,0)' },
        },
      },
    },
  },
  plugins: [],
};

export default config;
