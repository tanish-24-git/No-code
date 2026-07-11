import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';

/**
 * FINETUNE.md — the session's persistent memory file.
 * Re-injected into EVERY model request for EVERY agent, so standing user
 * directives and key facts survive context compaction. Mutated only via the
 * update_memory tool (never free-form writes).
 */

const SECTIONS = [
  'Mission',
  'Dataset facts',
  'Hardware',
  'Plan & decisions',
  'User directives',
  'Current status',
] as const;

export type MemorySection = (typeof SECTIONS)[number];

const TEMPLATE = SECTIONS.map((s) => `## ${s}\n\n_(empty)_\n`).join('\n');

const MAX_BYTES = 16 * 1024;

export function memoryFile(dataDir: string, sessionId: string): string {
  return path.join(dataDir, 'sessions', sessionId, 'FINETUNE.md');
}

export function ensureMemory(dataDir: string, sessionId: string): void {
  const file = memoryFile(dataDir, sessionId);
  if (!existsSync(file)) {
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, TEMPLATE, 'utf8');
  }
}

export function renderMemory(dataDir: string, sessionId: string): string {
  const file = memoryFile(dataDir, sessionId);
  if (!existsSync(file)) return '';
  try {
    return readFileSync(file, 'utf8').slice(0, MAX_BYTES);
  } catch {
    return '';
  }
}

/** Replace the body of one `## Section`; unknown section names are rejected. */
export function updateMemorySection(
  dataDir: string,
  sessionId: string,
  section: string,
  content: string,
): { ok: true } | { ok: false; error: string } {
  const match = SECTIONS.find((s) => s.toLowerCase() === section.trim().toLowerCase());
  if (!match) {
    return { ok: false, error: `unknown section "${section}" — valid: ${SECTIONS.join(', ')}` };
  }
  ensureMemory(dataDir, sessionId);
  const file = memoryFile(dataDir, sessionId);
  const current = readFileSync(file, 'utf8');
  const lines = current.split('\n');
  const start = lines.findIndex((l) => l.trim() === `## ${match}`);
  if (start === -1) {
    // Section header was lost somehow — append it fresh.
    const next = `${current.trimEnd()}\n\n## ${match}\n\n${content.trim()}\n`;
    if (next.length > MAX_BYTES) return { ok: false, error: 'memory file would exceed 16KiB cap' };
    writeFileSync(file, next, 'utf8');
    return { ok: true };
  }
  let end = lines.length;
  for (let i = start + 1; i < lines.length; i++) {
    if (lines[i].startsWith('## ')) {
      end = i;
      break;
    }
  }
  const next = [...lines.slice(0, start + 1), '', content.trim(), '', ...lines.slice(end)].join('\n');
  if (next.length > MAX_BYTES) return { ok: false, error: 'memory file would exceed 16KiB cap' };
  writeFileSync(file, next, 'utf8');
  return { ok: true };
}
