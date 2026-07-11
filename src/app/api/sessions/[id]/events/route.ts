import { isValidSessionId } from '@/server/session';
import { getRuntime } from '@/server/runtime';
import type { AgentEvent } from '@/server/events';

export const dynamic = 'force-dynamic';

const KEEPALIVE_MS = 15_000;

/**
 * The per-session SSE stream: replays the persisted event log (past
 * `?since=<ulid>` / Last-Event-ID), then tails the live bus. All frames are
 * unnamed `message` events — the kind lives in the JSON payload — and carry
 * `id:` so EventSource's native Last-Event-ID resume works.
 */
export async function GET(req: Request, { params }: { params: { id: string } }) {
  if (!isValidSessionId(params.id)) {
    return Response.json({ error: 'invalid session id' }, { status: 400 });
  }
  const rt = getRuntime();
  if (!rt.getSession(params.id)) {
    return Response.json({ error: 'session not found' }, { status: 404 });
  }

  const url = new URL(req.url);
  const since = url.searchParams.get('since') ?? req.headers.get('last-event-id') ?? undefined;
  const sessionId = params.id;
  const encoder = new TextEncoder();

  let unsubscribe: (() => void) | null = null;
  let keepalive: ReturnType<typeof setInterval> | null = null;

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const send = (event: AgentEvent) => {
        try {
          controller.enqueue(encoder.encode(`id: ${event.id}\ndata: ${JSON.stringify(event)}\n\n`));
        } catch {
          // controller already closed
        }
      };

      // 1. Replay backlog.
      for (const event of rt.bus.replay(sessionId, since ?? undefined)) send(event);

      // 2. Live tail.
      unsubscribe = rt.bus.subscribe(sessionId, send);

      // 3. Keepalive comments so proxies don't drop the connection.
      keepalive = setInterval(() => {
        try {
          controller.enqueue(encoder.encode(`: keepalive\n\n`));
        } catch {
          // closed
        }
      }, KEEPALIVE_MS);
    },
    cancel() {
      unsubscribe?.();
      if (keepalive) clearInterval(keepalive);
    },
  });

  // Abort (tab closed) → release subscriber.
  req.signal.addEventListener('abort', () => {
    unsubscribe?.();
    if (keepalive) clearInterval(keepalive);
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  });
}
