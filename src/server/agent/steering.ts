/**
 * Mid-run user messages. The loop drains this at the top of every iteration
 * (interstitial injection between tool calls), so the user can steer a
 * running agent without waiting for the whole turn chain to finish.
 */
export class SteeringQueue {
  private queue: string[] = [];

  push(text: string): void {
    this.queue.push(text);
  }

  drain(): string[] {
    const out = this.queue;
    this.queue = [];
    return out;
  }

  get size(): number {
    return this.queue.length;
  }
}
