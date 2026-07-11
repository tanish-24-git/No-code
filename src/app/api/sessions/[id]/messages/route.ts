import { isValidSessionId } from '@/server/session';
import { getRuntime } from '@/server/runtime';

export const dynamic = 'force-dynamic';

/**
 * User message entry point: starts the orchestrator when idle, steers it when
 * running (drained between tool calls), answers pending questions (M2+ passes
 * `questionId`).
 */
export async function POST(req: Request, { params }: { params: { id: string } }) {
  if (!isValidSessionId(params.id)) {
    return Response.json({ error: 'invalid session id' }, { status: 400 });
  }
  const rt = getRuntime();
  if (!rt.getSession(params.id)) {
    return Response.json({ error: 'session not found' }, { status: 404 });
  }

  const body = await req.json().catch(() => null);
  const text = typeof body?.text === 'string' ? body.text.trim() : '';
  if (!text) return Response.json({ error: 'text is required' }, { status: 400 });

  const result = rt.handleUserMessage(params.id, text);
  return Response.json({ ok: true, ...result }, { status: 202 });
}
