"""LangGraph state contracts for structured Sokoban problem solving."""

from __future__ import annotations

from operator import add
from typing import Annotated, Final, Literal, NotRequired, TypedDict, cast

from pydantic import Field

from sokoban_agent.env.catalog import DEFAULT_LEVEL_CATALOG
from sokoban_agent.graph.agentic.metrics import AgenticMetrics

CURRENT_STATE_SCHEMA_VERSION: Final[Literal[2]] = 2
GRAPH_REVISION: Final = "agentic-v2"
DEFAULT_LEVEL_ID: Final = "tiny-push"
DEFAULT_SEED: Final = 0
DEFAULT_MAX_STEPS: Final = 15
LEVEL_ID_OPTIONS: Final = tuple(
    record.level_id for record in DEFAULT_LEVEL_CATALOG.records
)

AgenticStatus = Literal[
    "initialized",
    "analyzed",
    "prompt_resolved",
    "failure_memory_recalled",
    "strategy_input_composed",
    "strategy_memory_hit",
    "strategy_memory_miss",
    "strategy_schema_error",
    "strategy_proposed",
    "strategy_semantic_error",
    "strategy_ready",
    "cycle_detected",
    "repetition_checked",
    "grounding_memory_hit",
    "grounding_memory_miss",
    "subgoal_grounding_failed",
    "subgoal_grounded",
    "executed_until_push",
    "subgoal_completed",
    "reflection_failed",
    "observed",
    "success",
    "deadlock",
    "step_limit",
]


class AgenticInput(TypedDict):
    """JSON-safe input accepted by the structured agent graph."""

    level_id: NotRequired[
        Annotated[
            str,
            Field(
                default=DEFAULT_LEVEL_ID,
                description="Catalog level to run.",
                json_schema_extra={"enum": list(LEVEL_ID_OPTIONS)},
            ),
        ]
    ]
    seed: NotRequired[
        Annotated[
            int | None,
            Field(
                default=DEFAULT_SEED,
                description="Random seed used for reproducible runs.",
            ),
        ]
    ]
    max_steps: NotRequired[
        Annotated[
            int,
            Field(
                default=DEFAULT_MAX_STEPS,
                description="Maximum environment actions before termination.",
                gt=0,
            ),
        ]
    ]
    level_rows: NotRequired[list[str]]
    level_sha256: NotRequired[str]


class PromptReference(TypedDict):
    """Resolved prompt identity recorded for reproducibility."""

    name: str
    commit: str


class DecisionEvent(TypedDict):
    """One observable decision-stage event."""

    step: int
    stage: str
    summary: str
    action: NotRequired[str]
    pushed: NotRequired[bool]


class PlanRevisionPayload(TypedDict):
    """One observable change to an active strategy."""

    step: int
    disposition: str
    changed_fields: list[str]
    evidence: str


class StrategySchemaIssue(TypedDict):
    """One safe, model-actionable structured-output validation issue."""

    path: str
    code: str
    message: str


class AgenticInfoState(TypedDict, total=False):
    level_id: str
    steps: int
    success: bool
    deadlock: bool
    pushed: bool
    player_moved: bool
    invalid_move: bool
    boxes_on_targets: int


class ExecutionResultState(TypedDict):
    before_boxes: list[list[int]]
    after_boxes: list[list[int]]
    actions_requested: list[str]
    actions_executed: list[str]
    push_count: int
    invalid_move: bool
    truncated: bool


class ReflectionResultState(TypedDict, total=False):
    matched: bool
    summary: str
    observed_effect: dict[str, object]


class GroundingFailureState(TypedDict):
    kind: str
    message: str


class AgenticMetaState(TypedDict):
    state_schema_version: Literal[2]
    graph_revision: str
    level_id: str
    level_sha256: str
    seed: int | None
    max_steps: int
    level_rows: list[str]
    prompt: PromptReference
    prompt_resolved: bool
    model_name: str
    rationale_mode: Literal["on", "off"]
    grounding_mode: Literal["direct", "local-search"]
    memory_mode: Literal["off", "episode", "shared"]
    memory_namespace: str


class AgenticPlanningState(TypedDict):
    board_analysis: dict[str, object] | None
    strategy_hypothesis: dict[str, object] | None
    strategy_input: dict[str, object]
    strategy_attempts: int
    strategy_error: str | None
    strategy_schema_issues: list[StrategySchemaIssue]
    latest_strategy_feedback: list[str]
    strategy_violations: list[dict[str, object]]
    grounded_plan: dict[str, object] | None
    grounding_failure: GroundingFailureState | None
    grounding_cache_key: str | None
    completed_subgoals: list[dict[str, object]]


class AgenticMemoryState(TypedDict):
    attempt_keys: list[str]
    rejected_pushes: dict[str, list[str]]


class AgenticExecutionState(TypedDict):
    result: ExecutionResultState | None
    reflection: ReflectionResultState | None


class AgenticState(TypedDict):
    """Initialized checkpointable state owned by the structured StateGraph.

    Graph input is intentionally partial through ``AgenticInput``.  Once the
    initialize node has run, every field below is present unless it is marked
    ``NotRequired`` explicitly.
    """

    meta: AgenticMetaState
    planning: AgenticPlanningState
    memory: AgenticMemoryState
    execution: AgenticExecutionState
    observation: list[list[int]]
    info: AgenticInfoState
    status: AgenticStatus
    action_history: list[str]
    cycle_detected: bool
    metrics: AgenticMetrics
    push_count: int
    plan_revisions: Annotated[list[PlanRevisionPayload], add]
    feedback: Annotated[list[str], add]
    decision_events: Annotated[list[DecisionEvent], add]


class AgenticRuntimeContext(TypedDict, total=False):
    """JSON-safe assistant context supplied by LangGraph Agent Server."""

    prompt_name: str
    prompt_commit: str
    model_name: str
    rationale_mode: Literal["on", "off"]
    grounding_mode: Literal["direct", "local-search"]
    memory_mode: Literal["off", "episode", "shared"]
    memory_namespace: str


def active_subgoal(state: AgenticState) -> dict[str, object] | None:
    hypothesis = state["planning"]["strategy_hypothesis"]
    value = hypothesis.get("subgoal") if hypothesis is not None else None
    return cast(dict[str, object] | None, value)


def protected_constraints(
    state: AgenticState,
) -> list[dict[str, object]]:
    hypothesis = state["planning"]["strategy_hypothesis"]
    value = (
        hypothesis.get("protected_constraints")
        if hypothesis is not None
        else None
    )
    return cast(list[dict[str, object]], value or [])


def expected_effect(state: AgenticState) -> dict[str, object] | None:
    hypothesis = state["planning"]["strategy_hypothesis"]
    value = (
        hypothesis.get("expected_effect") if hypothesis is not None else None
    )
    return cast(dict[str, object] | None, value)


def grounded_actions(state: AgenticState) -> list[str]:
    plan = state["planning"]["grounded_plan"]
    if plan is None:
        return []
    player_actions = cast(list[str], plan.get("player_actions", []))
    push_action = plan.get("push_action")
    return [
        *player_actions,
        *([push_action] if isinstance(push_action, str) else []),
    ]
