import { execFile } from 'node:child_process';
import os from 'node:os';
import { promisify } from 'node:util';

const execFileP = promisify(execFile);

export interface UvInfo {
  found: boolean;
  version?: string;
}

export interface GpuInfo {
  present: boolean;
  name?: string;
  vramTotalMb?: number;
  vramFreeMb?: number;
  source?: 'nvidia-smi' | 'cim';
}

export interface SysInfo {
  uv: UvInfo;
  gpu: GpuInfo;
  cpu: { cores: number; model: string };
  ramTotalGb: number;
  platform: NodeJS.Platform;
}

// Probes shell out to external binaries; cache briefly so ConfigBanner
// polling doesn't spawn processes on every request.
const CACHE_MS = 60_000;
let cache: { at: number; value: SysInfo } | null = null;

export async function probeUv(): Promise<UvInfo> {
  try {
    const { stdout } = await execFileP('uv', ['--version'], { timeout: 5_000 });
    return { found: true, version: stdout.trim() };
  } catch {
    return { found: false };
  }
}

export async function probeGpu(): Promise<GpuInfo> {
  // Primary: nvidia-smi (present on any machine with NVIDIA drivers).
  try {
    const { stdout } = await execFileP(
      'nvidia-smi',
      ['--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'],
      { timeout: 8_000 },
    );
    const first = stdout.trim().split('\n')[0];
    if (first) {
      const [name, total, free] = first.split(',').map((s) => s.trim());
      return {
        present: true,
        name,
        vramTotalMb: Number.parseInt(total, 10) || undefined,
        vramFreeMb: Number.parseInt(free, 10) || undefined,
        source: 'nvidia-smi',
      };
    }
  } catch {
    // fall through
  }
  // Fallback (Windows): CIM gives us the adapter name at least.
  if (process.platform === 'win32') {
    try {
      const { stdout } = await execFileP(
        'powershell.exe',
        ['-NoProfile', '-NonInteractive', '-Command', '(Get-CimInstance Win32_VideoController | Select-Object -First 1).Name'],
        { timeout: 10_000 },
      );
      const name = stdout.trim();
      if (name) return { present: false, name, source: 'cim' };
    } catch {
      // fall through
    }
  }
  return { present: false };
}

export async function getSysInfo(force = false): Promise<SysInfo> {
  if (!force && cache && Date.now() - cache.at < CACHE_MS) return cache.value;
  const [uv, gpu] = await Promise.all([probeUv(), probeGpu()]);
  const cpus = os.cpus();
  const value: SysInfo = {
    uv,
    gpu,
    cpu: { cores: cpus.length, model: cpus[0]?.model ?? 'unknown' },
    ramTotalGb: Math.round((os.totalmem() / 1024 ** 3) * 10) / 10,
    platform: process.platform,
  };
  cache = { at: Date.now(), value };
  return value;
}
