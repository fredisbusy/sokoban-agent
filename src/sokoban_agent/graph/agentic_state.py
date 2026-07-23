"""LangGraph state contracts for structured Sokoban problem solving."""

from __future__ import annotations

from operator import add
from typing import Annotated, NotRequired, TypedDict


class AgenticInput(TypedDict):
    """JSON-safe input accepted by the structured agent graph."""

    level_id: NotRequired[str]
    seed: NotRequired[int | None]
    max_steps: NotRequired[int]


class PromptReference(TypedDict):
    """Resolved prompt identity recorded for reproducibility."""

    name: str
    commit: str


class DecisionEvent(TypedDict):
    """One observable decision-stage event."""

    step: int
    stage: str
    summary: str


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
    model_name: str
    status: str
    board_analysis: dict[str, object] | None
    strategy_hypothesis: dict[str, object] | None
    active_subgoal: dict[str, object] | None
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
