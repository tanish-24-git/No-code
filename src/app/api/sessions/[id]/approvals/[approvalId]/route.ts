import { isValidSessionId } from '@/server/session';
import { getRuntime } from '@/server/runtime';
import { commandPrefix } from '@/server/agent/tools/run-terminal';

export const dynamic = 'force-dynamic';

/**
 * Resolve a pending approval card.
 * Body: { approved: boolean, scope?: 'once' | 'always-similar' }
 * 'always-similar' (every-command mode) allowlists the command's two-token
 * prefix for the rest of the session.
 */
export async function POST(req: Request, { params }: { params: { id: string; approvalId: string } }) {
  if (!isValidSessionId(params.id)) {
    return Response.json({ error: 'invalid session id' }, { status: 400 });
  }
  const rt = getRuntime();
  const session = rt.getSession(params.id);
  if (!session) return Response.json({ error: 'session not found' }, { status: 404 });

  const body = await req.json().catch(() => null);
  const approved = Boolean(body?.approved);
  const scope = body?.scope === 'always-similar' ? 'always-similar' : 'once';

  if (approved && scope === 'always-similar') {
    const command = session.pendingApproval?.payload?.command;
    if (typeof command === 'string' && command.trim()) {
      const prefix = commandPrefix(command);
      rt.store.update(params.id, (r) => {
        if (!r.commandAllowPrefixes.includes(prefix)) r.commandAllowPrefixes.push(prefix);
      });
    }
  }

  const ok = rt.decideApproval(params.id, params.approvalId, approved);
  if (!ok) return Response.json({ error: 'no matching pending approval' }, { status: 409 });
  return Response.json({ ok: true, approved });
}
