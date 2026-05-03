/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Standalone output produces a self-contained server.js plus a minimal
  // node_modules tree, which is what the Dockerfile copies into the runtime
  // stage. Locally, `npm run build` still works as usual.
  output: 'standalone',
  async rewrites() {
    // Resolution order:
    //   1. BACKEND_URL  (server-only, used inside Docker: e.g. http://backend:8000)
    //   2. NEXT_PUBLIC_API_BASE  (legacy fallback for local dev)
    //   3. http://localhost:8000  (default)
    const target =
      process.env.BACKEND_URL ||
      process.env.NEXT_PUBLIC_API_BASE ||
      'http://localhost:8000';
    return [
      { source: '/api/:path*', destination: `${target}/api/:path*` },
      { source: '/health', destination: `${target}/health` },
    ];
  },
};

export default nextConfig;
