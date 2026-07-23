export class FrameQueue<T> {
  private items: T[] = [];
  private isPaused = false;

  enqueue(frame: T): void {
    this.items.push(frame);
  }

  next(): T | null {
    if (this.isPaused) return null;
    return this.items.shift() ?? null;
  }

  step(): T | null {
    return this.items.shift() ?? null;
  }

  latest(): T | null {
    const frame = this.items.at(-1) ?? null;
    this.items = [];
    return frame;
  }

  pause(): void {
    this.isPaused = true;
  }

  resume(): void {
    this.isPaused = false;
  }

  clear(): void {
    this.items = [];
  }

  get paused(): boolean {
    return this.isPaused;
  }

  get size(): number {
    return this.items.length;
  }
}
