import { boardFromObservation } from "./board.js";

const PHASE_BY_NODE = [
  [/reflect/i, "reflecting"],
  [/execute|ground/i, "executing"],
  [/validate|guard|repetition/i, "validating"],
  [/plan|strategy|prompt|analy|observe/i, "planning"],
];

export function applyUpdate(previousState, payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return { node: "unknown", state: previousState };
  }
  const entries = Object.entries(payload);
  if (entries.length === 0) return { node: "unknown", state: previousState };
  const [node, update] = entries[0];
  const safeUpdate = update && typeof update === "object" ? update : {};
  return { node, state: { ...previousState, ...safeUpdate } };
}

export function normalizeEvent({ id, threadId, node, state }) {
  const info = objectOrEmpty(state.info);
  const execution = objectOrEmpty(state.execution_result);
  const reflection = objectOrEmpty(state.reflection_result);
  const hypothesis = objectOrEmpty(state.strategy_hypothesis);
  const subgoal = objectOrEmpty(state.active_subgoal);
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
    maxSteps: integer(state.max_steps, null),
    board,
    action: actions.at(-1) ?? history.at(-1) ?? null,
    actionCount: integer(state.action_count, history.length),
    pushCount: integer(execution.push_count, integer(info.push_count, null)),
    status,
    success: Boolean(state.success ?? info.success),
    deadlock: Boolean(state.deadlock ?? info.deadlock),
    truncated: Boolean(state.truncated ?? execution.truncated),
    strategy: {
      hypothesis: displayValue(hypothesis.summary ?? hypothesis.hypothesis ?? state.planner_goal),
      assignment: displayValue(hypothesis.assignment ?? hypothesis.box_goal_assignment),
      subgoal: displayValue(subgoal.summary ?? subgoal.description ?? subgoal),
      protectedCells: arrayOrEmpty(state.protected_constraints).map(displayValue),
      risk: displayValue(hypothesis.risk ?? state.risk ?? state.guard_summary),
    },
    effect: {
      expected: displayValue(state.expected_effect),
      observed: displayValue(reflection),
    },
    revision: displayValue(arrayOrEmpty(state.plan_revisions).at(-1)),
    metrics: {
      llmCalls: integer(state.llm_calls, integer(state.planning_attempts, null)),
      promptTokens: integer(state.llm_prompt_tokens, null),
      outputTokens: integer(state.llm_output_tokens, null),
      localSearchCalls: integer(state.local_search_calls, null),
      expandedStates: integer(state.local_expanded_states, null),
    },
  };
}

function terminalPhase(state, info) {
  if (state.success === true || info.success === true || state.status === "success") return "success";
  if (state.deadlock === true || info.deadlock === true || state.status === "deadlock") return "deadlock";
  if (state.truncated === true || state.status === "step_limit") return "truncated";
  return null;
}

function phaseFor(node, status) {
  if (/error|failed/i.test(status)) return "error";
  return PHASE_BY_NODE.find(([pattern]) => pattern.test(node))?.[1] ?? "planning";
}

function displayValue(value) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (typeof value === "boolean") return value ? "예" : "아니오";
  if (Array.isArray(value)) return value.map(displayValue).filter(Boolean).join(", ");
  if (typeof value === "object") {
    return Object.entries(value)
      .map(([key, item]) => `${key}: ${displayValue(item) ?? "—"}`)
      .join(" · ");
  }
  return String(value);
}

function objectOrEmpty(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function arrayOrEmpty(value) {
  return Array.isArray(value) ? value : [];
}

function integer(value, fallback) {
  return Number.isInteger(value) ? value : fallback;
}
