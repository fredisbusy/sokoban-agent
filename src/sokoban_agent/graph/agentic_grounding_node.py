"""LangGraph node for bounded single-push subgoal grounding."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np

from sokoban_agent.graph.agentic_state import AgenticState
from sokoban_agent.planning.base import Observation
from sokoban_agent.planning.local_execution import (
    SubgoalGroundingError,
    ground_push_subgoal,
)
from sokoban_agent.planning.strategy import (
    BoardAnalysis,
    ProtectedConstraint,
    PushSubgoal,
)

GroundingRoute = Literal[
    "compose_strategy_input",
    "execute_until_push",
    "__end__",
]


def ground_agentic_subgoal(state: AgenticState) -> dict[str, object]:
    """Ground one approved push without searching for puzzle completion."""

    observation = cast(
        Observation,
        np.asarray(state["observation"], dtype=np.uint8),
    )
    analysis = BoardAnalysis.model_validate(state["board_analysis"])
    subgoal = PushSubgoal.model_validate(state["active_subgoal"])
    constraints = tuple(
        ProtectedConstraint.model_validate(payload)
        for payload in state["protected_constraints"]
    )
    try:
        plan = ground_push_subgoal(
            observation,
            analysis,
            subgoal,
            constraints,
        )
    except SubgoalGroundingError as error:
        failure = {"kind": error.kind, "message": str(error)}
        feedback = f"{error.kind}: {error}"
        return {
            "grounded_plan": None,
            "grounded_actions": [],
            "grounding_failure": failure,
            "status": "subgoal_grounding_failed",
            "feedback": [feedback],
            "decision_events": [
                {
                    "step": _step(state),
                    "stage": "ground_subgoal",
                    "summary": feedback,
                }
            ],
        }

    actions = [*plan.player_actions, plan.push_action]
    return {
        "grounded_plan": plan.model_dump(mode="json"),
        "grounded_actions": actions,
        "grounding_failure": None,
        "status": "subgoal_grounded",
        "decision_events": [
            {
                "step": _step(state),
                "stage": "ground_subgoal",
                "summary": (
                    f"플레이어 이동 {len(plan.player_actions)}회와 "
                    "push 1회를 접지했습니다"
                ),
            }
        ],
    }


def route_after_grounding(state: AgenticState) -> GroundingRoute:
    """Send grounding failures back to bounded strategy correction."""

    if state["grounding_failure"] is None:
        return "execute_until_push"
    if state["strategy_attempts"] < 3:
        return "compose_strategy_input"
    return "__end__"


def _step(state: AgenticState) -> int:
    steps = state["info"].get("steps")
    if not isinstance(steps, int):
        raise TypeError("graph info steps must be an integer")
    return steps
