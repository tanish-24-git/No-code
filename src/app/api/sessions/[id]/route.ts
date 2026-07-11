import { isValidSessionId } from '@/server/session';
import { getRuntime } from '@/server/runtime';

export const dynamic = 'force-dynamic';

export async function GET(_req: Request, { params }: { params: { id: string } }) {
  if (!isValidSessionId(params.id)) {
    return Response.json({ error: 'invalid session id' }, { status: 400 });
  }
  const session = getRuntime().getSession(params.id);
  if (!session) return Response.json({ error: 'session not found' }, { status: 404 });
  return Response.json(session);
}

export async function DELETE(_req: Request, { params }: { params: { id: string } }) {
  if (!isValidSessionId(params.id)) {
    return Response.json({ error: 'invalid session id' }, { status: 400 });
  }
  const ok = getRuntime().deleteSession(params.id);
  if (!ok) return Response.json({ error: 'session not found' }, { status: 404 });
  return new Response(null, { status: 204 });
}
