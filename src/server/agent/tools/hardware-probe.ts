import { z } from 'zod';
import { registerTool } from './index';
import { getSysInfo } from '../../sysinfo';

registerTool<Record<string, never>>({
  name: 'hardware_probe',
  description: 'Detect GPU (name/VRAM via nvidia-smi), CPU, RAM, platform, and uv availability. Ground truth for training-feasibility decisions — never assume hardware.',
  inputSchema: z.object({}),
  parallelSafe: true,
  async execute(_input, _ctx) {
    const sys = await getSysInfo(true);
    const gpu = sys.gpu.present
      ? `GPU: ${sys.gpu.name} — ${sys.gpu.vramTotalMb ?? '?'}MB VRAM total, ${sys.gpu.vramFreeMb ?? '?'}MB free (${sys.gpu.source})`
      : `GPU: none detected${sys.gpu.name ? ` (display adapter: ${sys.gpu.name})` : ''} — CPU-only machine`;
    return {
      text: [
        gpu,
        `CPU: ${sys.cpu.model} (${sys.cpu.cores} threads)`,
        `RAM: ${sys.ramTotalGb} GB`,
        `Platform: ${sys.platform}`,
        `uv: ${sys.uv.found ? sys.uv.version : 'NOT INSTALLED — generated Python cannot run until the user installs it (astral.sh/uv)'}`,
      ].join('\n'),
    };
  },
});
