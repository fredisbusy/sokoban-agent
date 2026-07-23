export async function createThread(apiUrl, signal) {
  const response = await fetch(`${trimSlash(apiUrl)}/threads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
    signal,
  });
  if (!response.ok) throw new Error(await responseMessage(response, "thread 생성 실패"));
  const payload = await response.json();
  if (!payload.thread_id) throw new Error("Agent Server가 thread_id를 반환하지 않았습니다");
  return payload.thread_id;
}

export async function streamRun({
  apiUrl,
  threadId,
  assistantId,
  input,
  signal,
  onEvent,
}) {
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
        stream_mode: "updates",
        stream_subgraphs: true,
        stream_resumable: true,
        on_disconnect: "continue",
      }),
      signal,
    },
  );
  if (!response.ok) throw new Error(await responseMessage(response, "stream 시작 실패"));
  const progress = { lastEventId: null, runId: null };
  try {
    await readEventStream(response, signal, onEvent, progress);
  } catch (error) {
    if (signal?.aborted || !progress.runId) throw error;
    await resumeRun({ apiUrl, threadId, signal, onEvent, progress });
  }
}

async function readEventStream(response, signal, onEvent, progress) {
  if (!response.body) throw new Error("Agent Server stream 본문이 없습니다");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  while (true) {
    const { value, done } = await reader.read();
    pending += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const parsed = parseSseChunk(pending, done);
    pending = parsed.remainder;
    for (const event of parsed.events) {
      if (event.id) progress.lastEventId = event.id;
      progress.runId ??= runIdFromEvent(event);
      onEvent(event);
    }
    if (done) break;
  }
}

async function resumeRun({ apiUrl, threadId, signal, onEvent, progress }) {
  const response = await fetch(
    `${trimSlash(apiUrl)}/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(progress.runId)}/stream?stream_mode=updates`,
    {
      headers: {
        Accept: "text/event-stream",
        ...(progress.lastEventId ? { "Last-Event-ID": progress.lastEventId } : {}),
      },
      signal,
    },
  );
  if (!response.ok) throw new Error(await responseMessage(response, "stream 복구 실패"));
  await readEventStream(response, signal, onEvent, progress);
}

export function parseSseChunk(text, flush = false) {
  const normalized = text.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");
  const remainder = flush ? "" : parts.pop() ?? "";
  const frames = flush ? parts : parts;
  const events = frames.map(parseFrame).filter(Boolean);
  return { events, remainder };
}

export function runIdFromEvent(event) {
  if (!event || typeof event.data !== "object" || event.data === null) return null;
  return event.data.run_id ?? event.data.run?.run_id ?? null;
}

function parseFrame(frame) {
  let event = "message";
  let id = null;
  const data = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("id:")) id = line.slice(3).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (data.length === 0) return null;
  const raw = data.join("\n");
  let parsed = raw;
  try {
    parsed = JSON.parse(raw);
  } catch {
    // Preserve non-JSON protocol messages for error reporting.
  }
  return { event, id, data: parsed };
}

async function responseMessage(response, fallback) {
  const detail = await response.text();
  return `${fallback} (${response.status})${detail ? `: ${detail.slice(0, 240)}` : ""}`;
}

function trimSlash(value) {
  return String(value).replace(/\/+$/, "");
}
