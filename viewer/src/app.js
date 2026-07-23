import { parseBoard } from "./board.js";
import { applyUpdate, normalizeEvent } from "./events.js";
import { FrameQueue } from "./queue.js";
import {
  createThread,
  GraphRunError,
  streamRun,
  validateRunContext,
} from "./stream.js";

const queue = new FrameQueue();
const elements = Object.fromEntries(
  [...document.querySelectorAll("[id]")].map((element) => [element.id, element]),
);
let graphState = {};
let timer = null;
let abortController = null;
let streamComplete = false;

elements["run-form"].addEventListener("submit", startRun);
elements["pause-button"].addEventListener("click", togglePause);
elements["step-button"].addEventListener("click", () => renderFrame(queue.step()));
elements["latest-button"].addEventListener("click", () => renderFrame(queue.latest()));
elements.speed.addEventListener("change", schedulePlayback);

async function startRun(event) {
  event.preventDefault();
  const input = {
    level_id: elements["level-id"].value,
    seed: nullableInteger(elements.seed.value),
    max_steps: Number(elements["max-steps"].value),
  };
  const context = {
    prompt_name: elements["prompt-name"].value.trim(),
    prompt_commit: elements["prompt-commit"].value.trim(),
    model_name: elements["model-name"].value.trim(),
    rationale_mode: elements["rationale-mode"].value,
    grounding_mode: elements["grounding-mode"].value,
  };
  try {
    validateRunContext(context);
  } catch (error) {
    setConnection("error", "설정 오류");
    setPhase("error");
    setError(error.message);
    return;
  }

  abortController?.abort();
  abortController = new AbortController();
  queue.clear();
  queue.resume();
  graphState = {};
  streamComplete = false;
  setError(null);
  setConnection("connecting", "Agent Server 연결 중");
  setPhase("planning");
  elements["run-button"].disabled = true;
  elements["level-label"].textContent = elements["level-id"].value;
  elements["pause-button"].textContent = "일시정지";

  try {
    const threadId = await createThread(elements["api-url"].value, abortController.signal);
    setConnection("live", "실시간 연결");
    await streamRun({
      apiUrl: elements["api-url"].value,
      threadId,
      assistantId: elements["assistant-id"].value,
      input,
      context,
      signal: abortController.signal,
      onEvent: (raw) => receiveEvent(raw, threadId),
    });
    streamComplete = true;
    schedulePlayback();
    setConnection("", queue.size > 0 ? "수신 완료 · 표시 중" : "실행 완료");
  } catch (error) {
    if (error.name !== "AbortError") {
      setPhase("error");
      if (error instanceof GraphRunError) {
        clearTimeout(timer);
        renderFrame(queue.latest());
        queue.pause();
        streamComplete = true;
        setConnection("error", "실행 오류");
        setPhase("error");
        setError(error.message);
      } else {
        setConnection("error", "연결 오류");
        setError(`${error.message} — 먼저 'make studio'가 실행 중인지 확인하세요.`);
      }
    }
  } finally {
    elements["run-button"].disabled = false;
  }
}

function receiveEvent(raw, threadId) {
  if (!["updates", "messages/partial", "message"].includes(raw.event)) return;
  const payload = Array.isArray(raw.data) && raw.data.length === 2 ? raw.data[1] : raw.data;
  if (!payload || typeof payload !== "object") return;
  const { node, state } = applyUpdate(graphState, payload);
  graphState = state;
  const frame = normalizeEvent({ id: raw.id, threadId, node, state });
  if (frame.board) queue.enqueue(frame);
  updateQueueCount();
  schedulePlayback();
}

function schedulePlayback() {
  clearTimeout(timer);
  if (queue.paused || queue.size === 0) {
    if (streamComplete && queue.size === 0) setConnection("", "실행 완료");
    return;
  }
  const delay = Number(elements.speed.value);
  timer = setTimeout(() => {
    renderFrame(queue.next());
    schedulePlayback();
  }, delay);
}

function togglePause() {
  if (queue.paused) {
    queue.resume();
    elements["pause-button"].textContent = "일시정지";
    schedulePlayback();
  } else {
    queue.pause();
    clearTimeout(timer);
    elements["pause-button"].textContent = "계속";
  }
}

function renderFrame(frame) {
  if (!frame) return;
  renderBoard(frame.board);
  setPhase(frame.phase);
  setText("node-value", frame.node);
  setText("status-value", statusLabel(frame));
  setText("action-value", frame.action);
  setText("step-value", `${frame.step ?? 0} / ${frame.maxSteps ?? "—"}`);
  setText("push-value", frame.pushCount);
  setText("event-value", shorten(frame.eventId));
  setText("hypothesis-value", frame.strategy.hypothesis);
  setText("assignment-value", frame.strategy.assignment);
  setText("subgoal-value", frame.strategy.subgoal);
  setText("protected-value", frame.strategy.protectedCells.join(", "));
  setText("risk-value", frame.strategy.risk);
  setText("expected-value", frame.effect.expected);
  setText("observed-value", frame.effect.observed);
  setText("revision-value", frame.revision);
  setText("llm-calls-value", frame.metrics.llmCalls);
  const tokenTotal = sumNullable(frame.metrics.promptTokens, frame.metrics.outputTokens);
  setText("tokens-value", tokenTotal);
  setText("search-value", frame.metrics.expandedStates ?? frame.metrics.localSearchCalls);
  updateQueueCount();
}

function renderBoard(boardText) {
  if (!boardText) return;
  const rows = parseBoard(boardText);
  const board = elements.board;
  const dimensionsChanged = board.dataset.rows !== String(rows.length)
    || board.dataset.columns !== String(rows[0].length);
  board.style.display = "grid";
  board.style.setProperty("--rows", rows.length);
  board.style.setProperty("--columns", rows[0].length);
  board.dataset.rows = rows.length;
  board.dataset.columns = rows[0].length;
  elements["empty-board"].hidden = true;

  board.querySelectorAll(".tile").forEach((tile) => tile.remove());
  for (const tile of rows.flat()) {
    const cell = document.createElement("span");
    cell.className = `tile${tile.wall ? " wall" : ""}${tile.target ? " target" : ""}`;
    cell.setAttribute("aria-hidden", "true");
    board.insertBefore(cell, board.querySelector(".piece"));
  }
  syncPlayer(board, rows.flat().find((tile) => tile.player), dimensionsChanged);
  syncBoxes(board, rows.flat().filter((tile) => tile.box), dimensionsChanged);
  board.setAttribute("aria-label", `Sokoban 보드, ${rows.length}행 ${rows[0].length}열`);
}

function syncPlayer(board, tile, reset) {
  let piece = board.querySelector(".piece.player");
  if (!tile) {
    piece?.remove();
    return;
  }
  if (!piece || reset) {
    piece?.remove();
    piece = makePiece("player");
    board.append(piece);
  }
  positionPiece(piece, tile);
}

function syncBoxes(board, tiles, reset) {
  const previous = reset ? [] : [...board.querySelectorAll(".piece.box")];
  if (reset) board.querySelectorAll(".piece.box").forEach((piece) => piece.remove());
  const available = [...previous];
  for (const tile of tiles) {
    let piece = closestPiece(available, tile);
    if (!piece) {
      piece = makePiece("box");
      board.append(piece);
    } else {
      available.splice(available.indexOf(piece), 1);
    }
    piece.classList.toggle("complete", tile.target);
    positionPiece(piece, tile);
  }
  available.forEach((piece) => piece.remove());
}

function closestPiece(pieces, tile) {
  return pieces.reduce((closest, piece) => {
    const distance = Math.abs(Number(piece.dataset.row) - tile.row)
      + Math.abs(Number(piece.dataset.column) - tile.column);
    return !closest || distance < closest.distance ? { piece, distance } : closest;
  }, null)?.piece ?? null;
}

function makePiece(type) {
  const piece = document.createElement("span");
  piece.className = `piece ${type}`;
  piece.setAttribute("aria-label", type === "box" ? "상자" : "플레이어");
  return piece;
}

function positionPiece(piece, tile) {
  piece.dataset.row = tile.row;
  piece.dataset.column = tile.column;
  piece.style.setProperty("--row", tile.row);
  piece.style.setProperty("--column", tile.column);
}

function setConnection(className, text) {
  elements["connection-dot"].className = `status-dot ${className}`.trim();
  elements["connection-text"].textContent = text;
}

function setPhase(phase) {
  const labels = {
    planning: "계획 중", validating: "검증 중", executing: "실행 중",
    reflecting: "성찰 중", success: "해결", deadlock: "데드락",
    truncated: "행동 제한", error: "오류", disconnected: "대기",
  };
  elements["phase-chip"].dataset.phase = phase;
  elements["phase-chip"].textContent = labels[phase] ?? phase;
}

function setText(id, value) {
  elements[id].textContent = value === null || value === undefined || value === "" ? "—" : String(value);
}

function setError(message) {
  elements["error-message"].hidden = !message;
  elements["error-message"].textContent = message ?? "";
}

function statusLabel(frame) {
  if (frame.success) return "성공";
  if (frame.deadlock) return "데드락";
  if (frame.truncated) return "행동 제한";
  return frame.status;
}

function updateQueueCount() {
  elements["queue-count"].textContent = `대기 ${queue.size}`;
}

function nullableInteger(value) {
  return value === "" ? null : Number(value);
}

function shorten(value) {
  const text = String(value ?? "—");
  return text.length > 18 ? `${text.slice(0, 8)}…${text.slice(-6)}` : text;
}

function sumNullable(left, right) {
  if (left === null && right === null) return null;
  return (left ?? 0) + (right ?? 0);
}
