"""JSON-safe state and helpers for the LangGraph Studio graph."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

import numpy as np

from sokoban_agent.env import Action

Route = Literal["llm_plan", "astar_guard", "validate", "execute", "end"]


class StudioInput(TypedDict):
    """Small input form shown by LangGraph Studio."""

    level_id: NotRequired[str]
    seed: NotRequired[int]
    max_steps: NotRequired[int]


class DecisionEvent(TypedDict):
    """One human-readable event shown in the Studio state."""

    step: int
    stage: str
    summary: str
    details: dict[str, object]


class StudioState(TypedDict):
    """Initialized JSON-serializable state inspected at every Studio node."""

    level_id: str
    seed: int
    max_steps: int
    observation: list[list[int]]
    board: str
    plan: list[str]
    proposed_plan: list[str]
    action_history: list[str]
    feedback: list[str]
    planning_attempts: int
    action_count: int
    success: bool
    deadlock: bool
    truncated: bool
    status: str
    planner_goal: str
    decision_summary: str
    risk: str
    guard_summary: str
    validation_summary: str
    execution_summary: str
    failure_reason: str | None
    decision_log: list[DecisionEvent]
    llm_calls: int
    llm_elapsed_seconds: float
    llm_prompt_tokens: int
    llm_output_tokens: int
    algorithm_calls: int
    algorithm_fallbacks: int
    algorithm_expanded_states: int
    algorithm_elapsed_seconds: float


def decision_event(
    step: int,
    stage: str,
    summary: str,
    details: dict[str, object],
) -> DecisionEvent:
    """Create one JSON-safe human-readable decision event."""

    return {
        "step": step,
        "stage": stage,
        "summary": summary,
        "details": details,
    }


def observation_from_state(state: StudioState) -> np.ndarray:
    """Restore the NumPy board used by the environment rules."""

    return np.asarray(state["observation"], dtype=np.uint8)


def guard_update(
    state: StudioState,
    actions: list[Action],
    summary: str,
    *,
    fallback: bool,
    expanded_states: int = 0,
    elapsed_seconds: float = 0.0,
) -> dict[str, object]:
    """Build the common successful A* guard state update."""

    names = [action.name for action in actions]
    return {
        "plan": names,
        "guard_summary": summary,
        "status": "A* 검사 완료",
        "decision_log": [
            *state["decision_log"],
            decision_event(
                state["action_count"],
                "A* 검사",
                summary,
                {"grounded_actions": names, "fallback": fallback},
            ),
        ],
        "algorithm_calls": state["algorithm_calls"] + int(expanded_states > 0),
        "algorithm_fallbacks": state["algorithm_fallbacks"] + int(fallback),
        "algorithm_expanded_states": (
            state["algorithm_expanded_states"] + expanded_states
        ),
        "algorithm_elapsed_seconds": (
            state["algorithm_elapsed_seconds"] + elapsed_seconds
        ),
    }
