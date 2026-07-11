import { createWriteStream, mkdirSync } from 'node:fs';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { isValidSessionId } from '@/server/session';
import { getRuntime } from '@/server/runtime';

export const dynamic = 'force-dynamic';

const MAX_BYTES = 512 * 1024 * 1024; // 512MB
// Executables and pickle are pointless-to-dangerous as datasets.
const DENY_EXT = new Set(['.exe', '.dll', '.msi', '.scr', '.bat', '.cmd', '.ps1', '.sh', '.pkl', '.pickle']);

function sanitizeName(name: string): string {
  const base = path.basename(name).replace(/[^\w.\- ()]/g, '_');
  return base.length > 128 ? base.slice(-128) : base;
}

export async function POST(req: Request, { params }: { params: { id: string } }) {
  if (!isValidSessionId(params.id)) {
    return Response.json({ error: 'invalid session id' }, { status: 400 });
  }
  const rt = getRuntime();
  if (!rt.getSession(params.id)) return Response.json({ error: 'session not found' }, { status: 404 });

  const form = await req.formData().catch(() => null);
  const file = form?.get('file');
  if (!(file instanceof File)) return Response.json({ error: 'multipart field "file" is required' }, { status: 400 });
  if (file.size === 0) return Response.json({ error: 'empty file' }, { status: 400 });
  if (file.size > MAX_BYTES) return Response.json({ error: `file exceeds ${MAX_BYTES / 1024 / 1024}MB cap` }, { status: 413 });

  const name = sanitizeName(file.name || 'dataset');
  if (DENY_EXT.has(path.extname(name).toLowerCase())) {
    return Response.json({ error: `file type ${path.extname(name)} is not accepted as a dataset` }, { status: 415 });
  }

  const datasetDir = path.join(rt.workspaceDir(params.id), 'dataset');
  mkdirSync(datasetDir, { recursive: true });
  const dest = path.join(datasetDir, name);
  await pipeline(Readable.fromWeb(file.stream() as import('node:stream/web').ReadableStream), createWriteStream(dest));

  rt.notifyDatasetUpload(params.id, name, file.size);
  return Response.json({ ok: true, file: `dataset/${name}`, sizeBytes: file.size }, { status: 201 });
}
