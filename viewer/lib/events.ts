import { boardFromObservation } from "./board.ts";
import type { GraphState, Phase, ViewEvent } from "./types.ts";

const PHASE_BY_NODE: Array<[RegExp, Phase]> = [
  [/reflect/i, "reflecting"],
  [/execute|ground/i, "executing"],
  [/validate|guard|repetition/i, "validating"],
  [/plan|strategy|prompt|analy|observe/i, "planning"],
];

export function applyGraphUpdate(
  previousState: GraphState,
  payload: unknown,
): { node: string; state: GraphState } {
  if (!isRecord(payload)) return { node: "unknown", state: previousState };
  const entries = Object.entries(payload);
  if (entries.length === 0) return { node: "unknown", state: previousState };
  const [node, update] = entries[0];
  return {
    node,
    state: { ...previousState, ...(isRecord(update) ? update : {}) },
  };
}

interface NormalizeEventInput {
  id: string | null;
  threadId: string;
  node: string;
  state: GraphState;
}

export function decodeAgenticEventV1({
  id,
  threadId,
  node,
  state,
}: NormalizeEventInput): ViewEvent {
  const meta = objectOrEmpty(state.meta);
  const planning = objectOrEmpty(state.planning);
  const info = objectOrEmpty(state.info);
  const executionState = objectOrEmpty(state.execution);
  const execution = objectOrEmpty(executionState.result);
  const reflection = objectOrEmpty(executionState.reflection);
  const hypothesis = objectOrEmpty(planning.strategy_hypothesis);
  const subgoal = objectOrEmpty(hypothesis.subgoal);
  const metrics = objectOrEmpty(state.metrics);
  const episodeMetrics = objectOrEmpty(metrics.episode);
  const llmMetrics = objectOrEmpty(metrics.llm);
  const searchMetrics = objectOrEmpty(metrics.local_search);
  const algorithmMetrics = objectOrEmpty(metrics.algorithm);
  const actions = arrayOrEmpty(execution.actions_executed);
  const history = arrayOrEmpty(state.action_history);
  const status = String(state.status ?? "running");
  const board = typeof state.board === "string"
    ? state.board
    : boardFromObservation(state.observation);

  return {
    eventId: id ?? `${node}-${Date.now()}`,
    threadId,
    node,
    phase: terminalPhase(state, info) ?? phaseFor(node, status),
    step: integer(info.steps, integer(state.action_count, history.length)),
    maxSteps: integer(meta.max_steps, null),
    board,
    action: stringOrNull(actions.at(-1) ?? history.at(-1)),
    actionCount: integer(state.action_count, history.length) ?? history.length,
    pushCount: integer(
      state.push_count,
      integer(episodeMetrics.push_count, integer(execution.push_count, null)),
    ),
    status,
    success: Boolean(state.success ?? info.success),
    deadlock: Boolean(state.deadlock ?? info.deadlock),
    truncated: Boolean(state.truncated ?? execution.truncated),
    strategy: {
      hypothesis: displayValue(
        hypothesis.summary
          ?? hypothesis.hypothesis,
      ),
      assignment: displayAssignments(
        hypothesis.assignments ?? hypothesis.assignment ?? hypothesis.box_goal_assignment,
      ),
      subgoal: displayValue(subgoal.summary ?? subgoal.description ?? subgoal),
      protectedCells: arrayOrEmpty(hypothesis.protected_constraints).map(displayValue),
      risk: displayValue(hypothesis.risk),
    },
    effect: {
      expected: displayValue(hypothesis.expected_effect),
      observed: displayValue(reflection),
    },
    revision: displayValue(arrayOrEmpty(state.plan_revisions).at(-1)),
    metrics: {
      llmCalls: integer(
        llmMetrics.calls,
        null,
      ),
      promptTokens: integer(
        llmMetrics.prompt_tokens,
        null,
      ),
      outputTokens: integer(
        llmMetrics.output_tokens,
        null,
      ),
      localSearchCalls: integer(
        searchMetrics.calls,
        null,
      ),
      expandedStates: integer(
        searchMetrics.expanded_states,
        integer(algorithmMetrics.expanded_states, null),
      ),
    },
  };
}

export function decodeBaselineEventV1({
  id,
  threadId,
  node,
  state,
}: NormalizeEventInput): ViewEvent {
  const proposal = objectOrEmpty(state.proposal);
  const metrics = objectOrEmpty(state.metrics);
  const episode = objectOrEmpty(metrics.episode);
  const llm = objectOrEmpty(metrics.llm);
  const algorithm = objectOrEmpty(metrics.algorithm);
  const history = arrayOrEmpty(state.action_history);
  return {
    eventId: id ?? `${node}-${Date.now()}`,
    threadId,
    node,
    phase: terminalPhase(state, {}) ?? phaseFor(node, String(state.status ?? "running")),
    step: integer(state.action_count, history.length),
    maxSteps: integer(state.max_steps, null),
    board: typeof state.board === "string"
      ? state.board
      : boardFromObservation(state.observation),
    action: stringOrNull(history.at(-1)),
    actionCount: integer(state.action_count, history.length) ?? history.length,
    pushCount: integer(episode.push_count, null),
    status: String(state.status ?? "running"),
    success: state.success === true,
    deadlock: state.deadlock === true,
    truncated: state.truncated === true,
    strategy: {
      hypothesis: displayValue(proposal.goal),
      assignment: null,
      subgoal: null,
      protectedCells: [],
      risk: displayValue(proposal.risk),
    },
    effect: { expected: null, observed: null },
    revision: null,
    metrics: {
      llmCalls: integer(llm.calls, null),
      promptTokens: integer(llm.prompt_tokens, null),
      outputTokens: integer(llm.output_tokens, null),
      localSearchCalls: null,
      expandedStates: integer(algorithm.expanded_states, null),
    },
  };
}

function terminalPhase(state: GraphState, info: GraphState): Phase | null {
  if (state.success === true || info.success === true || state.status === "success") return "success";
  if (state.deadlock === true || info.deadlock === true || state.status === "deadlock") return "deadlock";
  if (state.truncated === true || state.status === "step_limit") return "truncated";
  return null;
}

function phaseFor(node: string, status: string): Phase {
  if (/error|failed/i.test(status)) return "error";
  return PHASE_BY_NODE.find(([pattern]) => pattern.test(node))?.[1] ?? "planning";
}

function displayValue(value: unknown): string | null {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (typeof value === "boolean") return value ? "예" : "아니오";
  if (Array.isArray(value)) return value.map(displayValue).filter(Boolean).join(", ");
  if (isRecord(value)) {
    return Object.entries(value)
      .map(([key, item]) => `${key}: ${displayValue(item) ?? "—"}`)
      .join(" · ");
  }
  return String(value);
}

function displayAssignments(value: unknown): string | null {
  if (!Array.isArray(value)) return displayValue(value);
  return value.map((assignment) => {
    if (!isRecord(assignment)) return displayValue(assignment);
    const boxId = displayValue(assignment.box_id);
    const targetId = displayValue(assignment.target_id);
    const reason = displayValue(assignment.reason);
    const pair = boxId && targetId ? `${boxId} → ${targetId}` : null;
    return [pair, reason].filter(Boolean).join(" · ");
  }).filter(Boolean).join(", ");
}

function objectOrEmpty(value: unknown): GraphState {
  return isRecord(value) ? value : {};
}

function arrayOrEmpty(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function integer(value: unknown, fallback: number | null): number | null {
  return Number.isInteger(value) ? value as number : fallback;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function isRecord(value: unknown): value is GraphState {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
