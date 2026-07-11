import { existsSync } from 'node:fs';
import path from 'node:path';
import { runForeground } from './terminal';

/**
 * uv-managed per-session Python environments. Agents run everything through
 * `uv run` / `uv pip install` inside the workspace, so the venv is created
 * lazily the first time a uv command executes there.
 */
export async function ensureVenv(workspaceDir: string): Promise<{ created: boolean; note?: string }> {
  if (existsSync(path.join(workspaceDir, '.venv'))) return { created: false };
  const res = await runForeground({
    command: 'uv venv',
    cwd: workspaceDir,
    timeoutMs: 120_000,
  });
  if (res.exitCode !== 0) {
    return {
      created: false,
      note: `uv venv failed (exit ${res.exitCode}): ${res.output.slice(-500)}`,
    };
  }
  return { created: true };
}
