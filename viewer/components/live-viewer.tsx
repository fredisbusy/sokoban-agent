"use client";

import {
  type FormEvent,
  useEffect,
  useRef,
  useState,
} from "react";

import { applyUpdate, normalizeEvent } from "../lib/events";
import { FrameQueue } from "../lib/frame-queue";
import {
  createThread,
  GraphRunError,
  streamRun,
  validateRunContext,
} from "../lib/stream";
import type { LevelOption } from "../lib/levels";
import type {
  GraphState,
  Phase,
  RunContext,
  SseEvent,
  ViewEvent,
} from "../lib/types";
import { Board } from "./board";
import { Inspector } from "./inspector";
import { RunForm } from "./run-form";

const AUTO_PROMPT_SELECTOR = "latest";

type ConnectionKind = "" | "connecting" | "live" | "error";

interface LiveViewerProps {
  levels: LevelOption[];
}

export function LiveViewer({ levels }: LiveViewerProps) {
  const defaultLevel = levels.find((level) => level.id === "tiny-walk")
    ?? levels[0];
  const queueRef = useRef(new FrameQueue<ViewEvent>());
  const graphStateRef = useRef<GraphState>({});
  const abortRef = useRef<AbortController | null>(null);
  const [frame, setFrame] = useState<ViewEvent | null>(null);
  const [queueSize, setQueueSize] = useState(0);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(350);
  const [streamComplete, setStreamComplete] = useState(false);
  const [running, setRunning] = useState(false);
  const [levelLabel, setLevelLabel] = useState(defaultLevel.displayName);
  const [phase, setPhase] = useState<Phase>("disconnected");
  const [connection, setConnection] = useState<{
    kind: ConnectionKind;
    text: string;
  }>({ kind: "", text: "연결 전" });
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    if (paused || queueSize === 0) {
      if (streamComplete && queueSize === 0) {
        setConnection({ kind: "", text: "실행 완료" });
      }
      return;
    }
    const timer = window.setTimeout(() => {
      const next = queueRef.current.next();
      if (next) {
        setFrame(next);
        setPhase(next.phase);
      }
      setQueueSize(queueRef.current.size);
    }, speed);
    return () => window.clearTimeout(timer);
  }, [paused, queueSize, speed, streamComplete]);

  async function startRun(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const context = runContext(form);
    try {
      validateRunContext(context);
    } catch (error) {
      setConnection({ kind: "error", text: "설정 오류" });
      setPhase("error");
      setErrorMessage(errorText(error));
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    queueRef.current.clear();
    graphStateRef.current = {};
    setQueueSize(0);
    setPaused(false);
    setStreamComplete(false);
    setFrame(null);
    setErrorMessage(null);
    setConnection({ kind: "connecting", text: "Agent Server 연결 중" });
    setPhase("planning");
    setRunning(true);
    const levelId = textField(form, "level_id");
    const selectedLevel = levels.find((level) => level.id === levelId);
    if (!selectedLevel) {
      setRunning(false);
      setPhase("error");
      setConnection({ kind: "error", text: "맵 설정 오류" });
      setErrorMessage(`알 수 없는 맵입니다: ${levelId}`);
      return;
    }
    setLevelLabel(selectedLevel.displayName);

    try {
      const apiUrl = textField(form, "api_url");
      const threadId = await createThread(apiUrl, controller.signal);
      setConnection({ kind: "live", text: "실시간 연결" });
      await streamRun({
        apiUrl,
        threadId,
        assistantId: textField(form, "assistant_id"),
        input: {
          level_id: levelId,
          level_rows: selectedLevel.rows,
          seed: nullableInteger(form.get("seed")),
          max_steps: Number(form.get("max_steps")),
        },
        context,
        signal: controller.signal,
        onEvent: (raw) => receiveEvent(raw, threadId),
      });
      setStreamComplete(true);
      setConnection({
        kind: "",
        text: queueRef.current.size > 0 ? "수신 완료 · 표시 중" : "실행 완료",
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setPhase("error");
      if (error instanceof GraphRunError) {
        const latest = queueRef.current.latest();
        if (latest) setFrame(latest);
        setQueueSize(0);
        setPaused(true);
        setStreamComplete(true);
        setConnection({ kind: "error", text: "실행 오류" });
        setErrorMessage(error.message);
      } else {
        setConnection({ kind: "error", text: "연결 오류" });
        setErrorMessage(
          `${errorText(error)} — 먼저 'make studio'가 실행 중인지 확인하세요.`,
        );
      }
    } finally {
      setRunning(false);
    }
  }

  function receiveEvent(raw: SseEvent, threadId: string) {
    if (!["updates", "messages/partial", "message"].includes(raw.event)) return;
    const payload = Array.isArray(raw.data) && raw.data.length === 2
      ? raw.data[1]
      : raw.data;
    if (!payload || typeof payload !== "object") return;
    const update = applyUpdate(graphStateRef.current, payload);
    graphStateRef.current = update.state;
    const nextFrame = normalizeEvent({
      id: raw.id,
      threadId,
      node: update.node,
      state: update.state,
    });
    if (nextFrame.board) {
      queueRef.current.enqueue(nextFrame);
      setQueueSize(queueRef.current.size);
    }
  }

  function stepOnce() {
    const next = queueRef.current.step();
    if (next) {
      setFrame(next);
      setPhase(next.phase);
    }
    setQueueSize(queueRef.current.size);
  }

  function catchUp() {
    const latest = queueRef.current.latest();
    if (latest) {
      setFrame(latest);
      setPhase(latest.phase);
    }
    setQueueSize(0);
  }

  return (
    <main className="shell">
      <header className="masthead">
        <div>
          <p className="eyebrow">LANGGRAPH LIVE OBSERVATION</p>
          <h1>Sokoban Agent</h1>
        </div>
        <div className="connection">
          <span className={`status-dot ${connection.kind}`.trim()} aria-hidden="true" />
          <span>{connection.text}</span>
        </div>
      </header>

      <RunForm
        levels={levels}
        running={running}
        onLevelChange={(level) => setLevelLabel(level.displayName)}
        onSubmit={startRun}
      />

      <section className="workspace">
        <article className="stage panel">
          <div className="stage-heading">
            <div>
              <p className="section-label">LIVE BOARD</p>
              <h2>{levelLabel}</h2>
            </div>
            <span className="phase-chip" data-phase={phase}>{phaseLabel(phase)}</span>
          </div>
          <div className="board-wrap"><Board board={frame?.board ?? null} /></div>
          <div className="playback">
            <button type="button" onClick={() => setPaused((value) => !value)}>
              {paused ? "계속" : "일시정지"}
            </button>
            <button type="button" onClick={stepOnce}>한 단계</button>
            <button type="button" onClick={catchUp}>최신으로</button>
            <label>속도
              <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
                <option value={700}>0.5×</option>
                <option value={350}>1×</option>
                <option value={175}>2×</option>
                <option value={0}>즉시</option>
              </select>
            </label>
            <span className="queue-count">대기 {queueSize}</span>
          </div>
        </article>
        <Inspector frame={frame} />
      </section>
      {errorMessage && <p className="error-message" role="alert">{errorMessage}</p>}
    </main>
  );
}

function runContext(form: FormData): RunContext {
  return {
    prompt_name: textField(form, "prompt_name"),
    prompt_commit: AUTO_PROMPT_SELECTOR,
    model_name: textField(form, "model_name"),
    rationale_mode: textField(form, "rationale_mode") as RunContext["rationale_mode"],
    grounding_mode: textField(form, "grounding_mode") as RunContext["grounding_mode"],
  };
}

function textField(form: FormData, name: string): string {
  return String(form.get(name) ?? "").trim();
}

function nullableInteger(value: FormDataEntryValue | null): number | null {
  return value === null || value === "" ? null : Number(value);
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function phaseLabel(phase: Phase): string {
  const labels: Record<Phase, string> = {
    planning: "계획 중",
    validating: "검증 중",
    executing: "실행 중",
    reflecting: "성찰 중",
    success: "해결",
    deadlock: "데드락",
    truncated: "행동 제한",
    error: "오류",
    disconnected: "대기",
  };
  return labels[phase];
}
