import assert from "node:assert/strict";
import test from "node:test";

import type { Client } from "@langchain/langgraph-sdk";

import {
  createThread,
  GraphRunError,
  streamRun,
  validateRunContext,
} from "../lib/stream.ts";
import type { RunContext, SseEvent } from "../lib/types.ts";

const RUN_CONTEXT: RunContext = {
  prompt_name: "sokoban-strategy",
  prompt_commit: "abc123",
  model_name: "qwen3.6:27b-mlx",
  rationale_mode: "on",
  grounding_mode: "local-search",
};

test("run context accepts automatic latest selector and rejects unresolved values", () => {
  assert.doesNotThrow(
    () => validateRunContext({ ...RUN_CONTEXT, prompt_commit: "latest" }),
  );
  assert.throws(
    () => validateRunContext({ ...RUN_CONTEXT, prompt_commit: "unresolved" }),
    /해석할 수 없습니다/,
  );
  assert.throws(
    () => validateRunContext({ ...RUN_CONTEXT, model_name: "" }),
    /model_name/,
  );
});

test("thread creation uses the official LangGraph client", async () => {
  const signal = new AbortController().signal;
  const calls: unknown[] = [];
  const client = {
    threads: {
      async create(options: unknown) {
        calls.push(options);
        return { thread_id: "thread-1" };
      },
    },
  } as unknown as Client;

  const threadId = await createThread("http://agent", signal, client);

  assert.equal(threadId, "thread-1");
  assert.deepEqual(calls, [{ signal }]);
});

test("stream run delegates resumable updates to the LangGraph SDK", async () => {
  const captured: Record<string, unknown> = {};
  const events: SseEvent[] = [];
  const client = {
    runs: {
      stream(
        threadId: string,
        assistantId: string,
        options: Record<string, unknown>,
      ) {
        Object.assign(captured, { threadId, assistantId, options });
        return chunks([
          { event: "metadata", id: "1", data: { run_id: "run-1" } },
          { event: "updates", id: "2", data: { observe: { status: "ok" } } },
        ]);
      },
    },
  } as unknown as Client;

  await streamRun({
    apiUrl: "http://agent",
    threadId: "thread-1",
    assistantId: "sokoban_agent",
    input: {
      level_id: "boxoban-medium-56",
      level_rows: ["#####", "#@$.#", "#####"],
      seed: 0,
      max_steps: 120,
    },
    context: RUN_CONTEXT,
    onEvent: (event) => events.push(event),
    client,
  });

  assert.equal(captured.threadId, "thread-1");
  assert.equal(captured.assistantId, "sokoban_agent");
  assert.deepEqual(captured.options, {
    input: {
      level_id: "boxoban-medium-56",
      level_rows: ["#####", "#@$.#", "#####"],
      seed: 0,
      max_steps: 120,
    },
    context: RUN_CONTEXT,
    streamMode: "updates",
    streamSubgraphs: true,
    streamResumable: true,
    onDisconnect: "continue",
    signal: undefined,
  });
  assert.deepEqual(events.map((event) => event.id), ["1", "2"]);
});

test("graph error events retain their server payload", async () => {
  const client = {
    runs: {
      stream() {
        return chunks([
          {
            event: "error",
            data: {
              error: "PromptConfigurationError",
              message: "prompt lookup failed",
            },
          },
        ]);
      },
    },
  } as unknown as Client;

  await assert.rejects(
    streamRun({
      apiUrl: "http://agent",
      threadId: "thread-1",
      assistantId: "agent",
      input: {},
      context: RUN_CONTEXT,
      onEvent: () => {},
      client,
    }),
    (error: unknown) => (
      error instanceof GraphRunError
      && error.message.includes("PromptConfigurationError")
      && error.message.includes("prompt lookup failed")
    ),
  );
});

async function* chunks(
  events: Array<{ event: string; id?: string; data: unknown }>,
) {
  for (const event of events) yield event;
}
