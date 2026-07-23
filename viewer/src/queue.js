export class FrameQueue {
  #items = [];
  #paused = false;

  enqueue(frame) {
    this.#items.push(frame);
  }

  next() {
    if (this.#paused) return null;
    return this.#items.shift() ?? null;
  }

  step() {
    return this.#items.shift() ?? null;
  }

  latest() {
    const frame = this.#items.at(-1) ?? null;
    this.#items = [];
    return frame;
  }

  pause() {
    this.#paused = true;
  }

  resume() {
    this.#paused = false;
  }

  clear() {
    this.#items = [];
  }

  get paused() {
    return this.#paused;
  }

  get size() {
    return this.#items.length;
  }
}
