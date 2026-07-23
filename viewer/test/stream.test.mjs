import assert from "node:assert/strict";
import test from "node:test";

import {
  parseSseChunk,
  runIdFromEvent,
  streamRun,
  validateRunContext,
} from "../src/stream.js";

const RUN_CONTEXT = {
  prompt_name: "sokoban-strategy",
  prompt_commit: "abc123",
  model_name: "qwen3.6:27b-mlx",
  rationale_mode: "on",
  grounding_mode: "local-search",
};

test("SSE parser preserves incomplete chunks", () => {
  const first = parseSseChunk("event: updates\nid: 1\ndata: {\"execute\":");
  assert.equal(first.events.length, 0);
  const second = parseSseChunk(`${first.remainder}"ok"}\n\n`);
  assert.deepEqual(second.events, [{
    event: "updates",
    id: "1",
    data: { execute: "ok" },
  }]);
});

test("SSE parser supports multiline data and final frame", () => {
  const parsed = parseSseChunk(
    "event: updates\ndata: {\"node\":\ndata: {\"status\":\"ok\"}}\n\n",
    true,
  );
  assert.deepEqual(parsed.events[0].data, { node: { status: "ok" } });
});

test("run id is read from Agent Server metadata", () => {
  assert.equal(runIdFromEvent({ data: { run_id: "run-1" } }), "run-1");
  assert.equal(runIdFromEvent({ data: { run: { run_id: "run-2" } } }), "run-2");
});

test("run context rejects mutable prompt selectors", () => {
  assert.throws(
    () => validateRunContext({ ...RUN_CONTEXT, prompt_commit: "latest" }),
    /고정 commit/,
  );
  assert.throws(
    () => validateRunContext({ ...RUN_CONTEXT, model_name: "" }),
    /model_name/,
  );
});

test("stream run sends immutable prompt and model context to LangGraph", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url: String(url), options });
    return new Response("", { status: 200 });
  };

  try {
    await streamRun({
      apiUrl: "http://agent",
      threadId: "thread-1",
      assistantId: "sokoban_agent",
      input: { level_id: "tiny-walk", seed: 0, max_steps: 15 },
      context: RUN_CONTEXT,
      onEvent: () => {},
    });
    const body = JSON.parse(calls[0].options.body);
    assert.deepEqual(body.context, RUN_CONTEXT);
    assert.equal(body.assistant_id, "sokoban_agent");
    assert.equal(body.stream_mode, "updates");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("graph error events are not mistaken for resumable disconnects", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url: String(url), options });
    return new Response(
      "event: metadata\ndata: {\"run_id\":\"run-1\"}\n\n"
      + "event: error\ndata: {\"error\":\"PromptConfigurationError\"}\n\n",
      { status: 200, headers: { "Content-Type": "text/event-stream" } },
    );
  };

  try {
    await assert.rejects(
      streamRun({
        apiUrl: "http://agent",
        threadId: "thread-1",
        assistantId: "agent",
        input: {},
        context: RUN_CONTEXT,
        onEvent: () => {},
      }),
      /PromptConfigurationError/,
    );
    assert.equal(calls.length, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("dropped resumable stream rejoins from its last event", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  let pulls = 0;
  const broken = new ReadableStream({
    pull(controller) {
      pulls += 1;
      if (pulls === 1) {
        controller.enqueue(new TextEncoder().encode(
          "event: metadata\nid: 1\ndata: {\"run_id\":\"run-1\"}\n\n"
          + "event: updates\nid: 2\ndata: {\"observe\":{\"status\":\"ok\"}}\n\n",
        ));
      } else {
        controller.error(new Error("network dropped"));
      }
    },
  });
  const resumed = new Response(
    "event: updates\nid: 3\ndata: {\"execute\":{\"status\":\"success\"}}\n\n",
    { status: 200, headers: { "Content-Type": "text/event-stream" } },
  );
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url: String(url), options });
    return calls.length === 1
      ? new Response(broken, { status: 200 })
      : resumed;
  };

  try {
    const events = [];
    await streamRun({
      apiUrl: "http://agent",
      threadId: "thread-1",
      assistantId: "agent",
      input: {},
      context: RUN_CONTEXT,
      onEvent: (event) => events.push(event),
    });
    assert.deepEqual(events.map((event) => event.id), ["1", "2", "3"]);
    assert.match(calls[1].url, /threads\/thread-1\/runs\/run-1\/stream/);
    assert.equal(calls[1].options.headers["Last-Event-ID"], "2");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
