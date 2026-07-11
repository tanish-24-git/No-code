/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    // instrumentation.ts runs register() once at server boot — the harness
    // uses it to resume sessions and re-attach training watchers.
    instrumentationHook: true,
  },
};

export default nextConfig;
