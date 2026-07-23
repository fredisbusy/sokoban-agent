import assert from "node:assert/strict";
import test from "node:test";

import { FrameQueue } from "../lib/frame-queue.ts";

test("queue pauses display without dropping streamed frames", () => {
  const queue = new FrameQueue();
  queue.enqueue({ id: 1 });
  queue.pause();
  assert.equal(queue.next(), null);
  assert.equal(queue.size, 1);
  assert.deepEqual(queue.step(), { id: 1 });
});

test("latest catches up and clears older frames", () => {
  const queue = new FrameQueue();
  queue.enqueue({ id: 1 });
  queue.enqueue({ id: 2 });
  assert.deepEqual(queue.latest(), { id: 2 });
  assert.equal(queue.size, 0);
});
