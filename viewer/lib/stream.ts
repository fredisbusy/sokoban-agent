import { Client } from "@langchain/langgraph-sdk";

import type { RunContext, SseEvent } from "./types.ts";

interface StreamRunOptions {
  apiUrl: string;
  threadId: string;
  assistantId: string;
  input: Record<string, unknown>;
  context: RunContext;
  signal?: AbortSignal;
  onEvent: (event: SseEvent) => void;
  client?: Client;
}

export async function createThread(
  apiUrl: string,
  signal?: AbortSignal,
  client = new Client({ apiUrl }),
): Promise<string> {
  const thread = await client.threads.create({ signal });
  if (!thread.thread_id) {
    throw new Error("Agent Server가 thread_id를 반환하지 않았습니다");
  }
  return thread.thread_id;
}

export async function streamRun({
  apiUrl,
  threadId,
  assistantId,
  input,
  context,
  signal,
  onEvent,
  client = new Client({ apiUrl }),
}: StreamRunOptions): Promise<void> {
  validateRunContext(context);
  const stream = client.runs.stream(threadId, assistantId, {
    input,
    context,
    streamMode: "updates",
    streamSubgraphs: true,
    streamResumable: true,
    onDisconnect: "continue",
    signal,
  });
  for await (const chunk of stream) {
    const event: SseEvent = {
      event: chunk.event,
      id: chunk.id ?? null,
      data: chunk.data,
    };
    if (event.event === "error") throw new GraphRunError(event.data);
    onEvent(event);
  }
}

export class GraphRunError extends Error {
  readonly payload: unknown;

  constructor(payload: unknown) {
    const record = isRecord(payload) ? payload : {};
    const message = typeof payload === "string" ? payload : record.message;
    super(
      [record.error, message]
        .filter((item): item is string => typeof item === "string" && item.length > 0)
        .join(": ") || "LangGraph run이 실패했습니다",
    );
    this.name = "GraphRunError";
    this.payload = payload;
  }
}

export function validateRunContext(context: RunContext): void {
  for (const key of ["prompt_name", "prompt_commit", "model_name"] as const) {
    if (context[key].trim() === "") throw new Error(`${key} 값을 입력하세요`);
  }
  if (context.prompt_commit.trim().toLowerCase() === "unresolved") {
    throw new Error("prompt_commit을 해석할 수 없습니다");
  }
  if (!["on", "off"].includes(context.rationale_mode)) {
    throw new Error("rationale_mode는 on 또는 off여야 합니다");
  }
  if (!["direct", "local-search"].includes(context.grounding_mode)) {
    throw new Error("grounding_mode는 direct 또는 local-search여야 합니다");
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
