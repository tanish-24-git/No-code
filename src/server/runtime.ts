import { loadConfig, type AppConfig } from './config';

/**
 * The long-lived harness runtime. Route handlers grab it via getRuntime();
 * it must survive dev-mode HMR recompiles, hence the globalThis stash
 * (plain module scope is wiped on every recompile).
 */
export class SessionManager {
  constructor(readonly config: AppConfig) {}

  /** Called once at boot from instrumentation.ts. */
  async resumeAll(): Promise<void> {
    // M1+: reload session checkpoints; M5+: re-attach training watchers,
    // synthesize wake notifications for runs that ended while we were down.
    console.log('[finetune-studio] runtime ready', {
      dataDir: this.config.dataDir,
      workspacesDir: this.config.workspacesDir,
      approvalMode: this.config.approvalMode,
    });
  }
}

const g = globalThis as unknown as { __ftRuntime?: SessionManager };

export function getRuntime(): SessionManager {
  return (g.__ftRuntime ??= new SessionManager(loadConfig()));
}
