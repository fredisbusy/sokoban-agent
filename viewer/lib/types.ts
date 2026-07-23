export type GraphState = Record<string, unknown>;

export type Phase =
  | "planning"
  | "validating"
  | "executing"
  | "reflecting"
  | "success"
  | "deadlock"
  | "truncated"
  | "error"
  | "disconnected";

export interface BoardTile {
  symbol: string;
  row: number;
  column: number;
  wall: boolean;
  target: boolean;
  box: boolean;
  player: boolean;
}

export interface ViewEvent {
  eventId: string;
  threadId: string;
  node: string;
  phase: Phase;
  step: number | null;
  maxSteps: number | null;
  board: string | null;
  action: string | null;
  actionCount: number;
  pushCount: number | null;
  status: string;
  success: boolean;
  deadlock: boolean;
  truncated: boolean;
  strategy: {
    hypothesis: string | null;
    assignment: string | null;
    subgoal: string | null;
    protectedCells: Array<string | null>;
    risk: string | null;
  };
  effect: {
    expected: string | null;
    observed: string | null;
  };
  revision: string | null;
  metrics: {
    llmCalls: number | null;
    promptTokens: number | null;
    outputTokens: number | null;
    localSearchCalls: number | null;
    expandedStates: number | null;
  };
}

export interface RunContext {
  prompt_name: string;
  prompt_commit: string;
  model_name: string;
  rationale_mode: "on" | "off";
  grounding_mode: "direct" | "local-search";
}

export interface SseEvent {
  event: string;
  id: string | null;
  data: unknown;
}
