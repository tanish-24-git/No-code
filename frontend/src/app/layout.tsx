import type { Metadata } from 'next';
import { DM_Mono, Syne } from 'next/font/google';
import './globals.css';
import { Nav } from '@/components/Nav';
import { ConfigBanner } from '@/components/ConfigBanner';


// next/font self-hosts these at build time, so there is no runtime fetch and
// no FOUT. The CSS variables are exposed to Tailwind via the body className.
const dmMono = DM_Mono({
  subsets: ['latin'],
  weight: ['300', '400', '500'],
  variable: '--font-mono',
  display: 'swap',
});

const syne = Syne({
  subsets: ['latin'],
  weight: ['400', '600', '700', '800'],
  variable: '--font-sans',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'FineTune Studio',
  description: 'Local-first, open-source LLM fine-tuning + inference copilot.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${dmMono.variable} ${syne.variable}`}>
      <body>
        <Nav />
        <ConfigBanner />
        <main className="pt-[52px] min-h-screen">{children}</main>
      </body>
    </html>
  );
}
