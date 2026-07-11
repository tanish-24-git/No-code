import { isValidSessionId } from '@/server/session';
import { getRuntime } from '@/server/runtime';
import { spentUsd } from '@/server/agent/budget';

export const dynamic = 'force-dynamic';

export async function GET(_req: Request, { params }: { params: { id: string } }) {
  if (!isValidSessionId(params.id)) return Response.json({ error: 'invalid session id' }, { status: 400 });
  const session = getRuntime().getSession(params.id);
  if (!session) return Response.json({ error: 'session not found' }, { status: 404 });
  return Response.json({
    budgetUsd: session.budgetUsd,
    spentUsd: spentUsd(session.ledger),
    entries: session.ledger,
  });
}

/** Body: { addUsd: number } — raises the budget and resumes a paused loop. */
export async function POST(req: Request, { params }: { params: { id: string } }) {
  if (!isValidSessionId(params.id)) return Response.json({ error: 'invalid session id' }, { status: 400 });
  const rt = getRuntime();
  if (!rt.getSession(params.id)) return Response.json({ error: 'session not found' }, { status: 404 });

  const body = await req.json().catch(() => null);
  const addUsd = Number(body?.addUsd);
  if (!Number.isFinite(addUsd) || addUsd <= 0 || addUsd > 10_000) {
    return Response.json({ error: 'addUsd must be a positive number' }, { status: 400 });
  }
  const result = rt.topUpBudget(params.id, addUsd);
  return Response.json({ ok: true, ...result });
}
