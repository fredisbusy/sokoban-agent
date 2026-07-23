"""LangGraph state contracts for structured Sokoban problem solving."""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, NotRequired, TypedDict


class AgenticInput(TypedDict):
    """JSON-safe input accepted by the structured agent graph."""

    level_id: NotRequired[str]
    seed: NotRequired[int | None]
    max_steps: NotRequired[int]
    level_rows: NotRequired[list[str]]


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


class AgenticState(AgenticInput, total=False):
    """Checkpointable state owned by the structured agent StateGraph."""

    observation: list[list[int]]
    info: dict[str, object]
    prompt: PromptReference
    prompt_resolved: bool
    model_name: str
    rationale_mode: Literal["on", "off"]
    grounding_mode: Literal["direct", "local-search"]
    status: str
    board_analysis: dict[str, object] | None
    strategy_hypothesis: dict[str, object] | None
    strategy_input: dict[str, object]
    strategy_attempts: int
    strategy_error: str | None
    strategy_violations: list[dict[str, object]]
    active_subgoal: dict[str, object] | None
    grounded_plan: dict[str, object] | None
    grounded_actions: list[str]
    grounding_failure: dict[str, object] | None
    action_history: list[str]
    execution_result: dict[str, object] | None
    reflection_result: dict[str, object] | None
    completed_subgoals: list[dict[str, object]]
    attempt_keys: list[str]
    cycle_detected: bool
    strategy_proposals: int
    strategy_schema_rejections: int
    strategy_semantic_rejections: int
    llm_calls: int
    llm_elapsed_seconds: float
    llm_prompt_tokens: int
    llm_output_tokens: int
    local_search_calls: int
    local_expanded_states: int
    local_search_elapsed_seconds: float
    rule_checks: int
    reachability_calls: int
    subgoal_grounding_attempts: int
    subgoal_grounding_failures: int
    push_count: int
    effect_matches: int
    effect_mismatches: int
    protected_constraints: list[dict[str, object]]
    expected_effect: dict[str, object] | None
    failure_conditions: list[dict[str, object]]
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
