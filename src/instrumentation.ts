/**
 * Next.js instrumentation hook — runs once at server boot (nodejs runtime).
 *
 * The agent harness lives inside the Next.js process; this is where we
 * restore long-lived state after a restart: reload session checkpoints,
 * re-attach training watchers to still-running processes, and synthesize
 * wake notifications for runs that finished while the server was down.
 */
export async function register() {
  if (process.env.NEXT_RUNTIME === 'nodejs') {
    // Dynamic import so the server-only harness never leaks into edge/client bundles.
    const { getRuntime } = await import('./server/runtime');
    await getRuntime().resumeAll();
  }
}
