import type { RunContext, SseEvent } from "./types.ts";

interface StreamRunOptions {
  apiUrl: string;
  threadId: string;
  assistantId: string;
  input: Record<string, unknown>;
  context: RunContext;
  signal?: AbortSignal;
  onEvent: (event: SseEvent) => void;
}

interface StreamProgress {
  lastEventId: string | null;
  runId: string | null;
}

export async function createThread(apiUrl: string, signal?: AbortSignal): Promise<string> {
  const response = await fetch(`${trimSlash(apiUrl)}/threads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
    signal,
  });
  if (!response.ok) throw new Error(await responseMessage(response, "thread 생성 실패"));
  const payload = await response.json() as { thread_id?: string };
  if (!payload.thread_id) throw new Error("Agent Server가 thread_id를 반환하지 않았습니다");
  return payload.thread_id;
}

export async function streamRun({
  apiUrl,
  threadId,
  assistantId,
  input,
  context,
  signal,
  onEvent,
}: StreamRunOptions): Promise<void> {
  validateRunContext(context);
  const response = await fetch(
    `${trimSlash(apiUrl)}/threads/${encodeURIComponent(threadId)}/runs/stream`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        assistant_id: assistantId,
        input,
        context,
        stream_mode: "updates",
        stream_subgraphs: true,
        stream_resumable: true,
        on_disconnect: "continue",
      }),
      signal,
    },
  );
  if (!response.ok) throw new Error(await responseMessage(response, "stream 시작 실패"));
  const progress: StreamProgress = { lastEventId: null, runId: null };
  try {
    await readEventStream(response, onEvent, progress);
  } catch (error) {
    if (!(error instanceof StreamDisconnectError) || signal?.aborted || !progress.runId) {
      throw error;
    }
    await resumeRun({ apiUrl, threadId, signal, onEvent, progress });
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

async function readEventStream(
  response: Response,
  onEvent: (event: SseEvent) => void,
  progress: StreamProgress,
): Promise<void> {
  if (!response.body) throw new Error("Agent Server stream 본문이 없습니다");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  while (true) {
    let chunk: ReadableStreamReadResult<Uint8Array>;
    try {
      chunk = await reader.read();
    } catch (error) {
      throw new StreamDisconnectError(error);
    }
    const { value, done } = chunk;
    pending += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const parsed = parseSseChunk(pending, done);
    pending = parsed.remainder;
    for (const event of parsed.events) {
      if (event.id) progress.lastEventId = event.id;
      progress.runId ??= runIdFromEvent(event);
      if (event.event === "error") throw new GraphRunError(event.data);
      onEvent(event);
    }
    if (done) break;
  }
}

class StreamDisconnectError extends Error {
  constructor(cause: unknown) {
    super(cause instanceof Error ? cause.message : "Agent Server stream 연결이 끊겼습니다");
    this.name = "StreamDisconnectError";
    this.cause = cause;
  }
}

async function resumeRun({
  apiUrl,
  threadId,
  signal,
  onEvent,
  progress,
}: Omit<StreamRunOptions, "assistantId" | "input" | "context"> & {
  progress: StreamProgress;
}): Promise<void> {
  const response = await fetch(
    `${trimSlash(apiUrl)}/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(progress.runId ?? "")}/stream?stream_mode=updates`,
    {
      headers: {
        Accept: "text/event-stream",
        ...(progress.lastEventId ? { "Last-Event-ID": progress.lastEventId } : {}),
      },
      signal,
    },
  );
  if (!response.ok) throw new Error(await responseMessage(response, "stream 복구 실패"));
  await readEventStream(response, onEvent, progress);
}

export function parseSseChunk(
  text: string,
  flush = false,
): { events: SseEvent[]; remainder: string } {
  const normalized = text.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");
  const remainder = flush ? "" : parts.pop() ?? "";
  const events = parts.map(parseFrame).filter((event): event is SseEvent => event !== null);
  return { events, remainder };
}

export function runIdFromEvent(event: Pick<SseEvent, "data">): string | null {
  if (!isRecord(event.data)) return null;
  if (typeof event.data.run_id === "string") return event.data.run_id;
  return isRecord(event.data.run) && typeof event.data.run.run_id === "string"
    ? event.data.run.run_id
    : null;
}

function parseFrame(frame: string): SseEvent | null {
  let event = "message";
  let id: string | null = null;
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("id:")) id = line.slice(3).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (data.length === 0) return null;
  const raw = data.join("\n");
  let parsed: unknown = raw;
  try {
    parsed = JSON.parse(raw);
  } catch {
    // Keep non-JSON protocol messages readable.
  }
  return { event, id, data: parsed };
}

async function responseMessage(response: Response, fallback: string): Promise<string> {
  const detail = await response.text();
  return `${fallback} (${response.status})${detail ? `: ${detail.slice(0, 240)}` : ""}`;
}

function trimSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
