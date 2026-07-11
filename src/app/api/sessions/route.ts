import { getRuntime } from '@/server/runtime';

export const dynamic = 'force-dynamic';

export async function GET() {
  const rt = getRuntime();
  const sessions = rt.listSessions().map((s) => ({
    id: s.id,
    title: s.title,
    status: s.status,
    createdAt: s.createdAt,
    updatedAt: s.updatedAt,
    datasetFiles: s.datasetFiles,
  }));
  return Response.json({ sessions });
}

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  const rt = getRuntime();
  const session = rt.createSession(typeof body?.title === 'string' ? body.title : undefined);
  return Response.json(session, { status: 201 });
}
