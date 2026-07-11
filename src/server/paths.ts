import path from 'node:path';

/**
 * Workspace containment guard. Every file/terminal tool resolves paths
 * through this — a path that escapes the session workspace is rejected,
 * whatever mix of ../, absolute segments or drive letters produced it.
 */
export function resolveInWorkspace(workspaceDir: string, relPath: string): string {
  const base = path.resolve(workspaceDir);
  const resolved = path.resolve(base, relPath ?? '.');
  const rel = path.relative(base, resolved);
  if (rel === '') return resolved;
  if (rel.startsWith('..') || path.isAbsolute(rel)) {
    throw new Error(`path escapes the session workspace: ${relPath}`);
  }
  return resolved;
}

/** Forward slashes for prompts/UI — consistent across platforms. */
export function toPosix(p: string): string {
  return p.replace(/\\/g, '/');
}
