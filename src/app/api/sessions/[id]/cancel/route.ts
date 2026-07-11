import { isValidSessionId } from '@/server/session';
import { getRuntime } from '@/server/runtime';

export const dynamic = 'force-dynamic';

/** Hard-cancel the active loop (M5 also kills session processes). */
export async function POST(_req: Request, { params }: { params: { id: string } }) {
  if (!isValidSessionId(params.id)) {
    return Response.json({ error: 'invalid session id' }, { status: 400 });
  }
  const rt = getRuntime();
  if (!rt.getSession(params.id)) {
    return Response.json({ error: 'session not found' }, { status: 404 });
  }
  const canceled = rt.cancelSession(params.id);
  rt.setStatus(params.id, 'idle');
  rt.bus.emit(params.id, 'chat.message', { role: 'system', text: 'Canceled by user.' });
  return Response.json({ ok: true, canceled });
}
