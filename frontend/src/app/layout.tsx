import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { Nav } from '@/components/Nav';
import { ConfigBanner } from '@/components/ConfigBanner';


// next/font self-hosts these at build time.
const jetbrains = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'No-code',
  description: 'AI Agent Registry & Fine-Tuning Studio.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${jetbrains.variable} ${inter.variable} scroll-smooth`}>
      <body className="antialiased">
        <Nav />
        <ConfigBanner />
        <main className="pt-[52px] min-h-screen">{children}</main>
      </body>
    </html>
  );
}
