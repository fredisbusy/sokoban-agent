"""LangGraph node for bounded single-push subgoal grounding."""

from __future__ import annotations

from time import perf_counter
from typing import Literal, cast

import numpy as np

from sokoban_agent.graph.agentic.metrics import update_agentic_metrics
from sokoban_agent.graph.agentic.state import (
    AgenticState,
    active_subgoal,
    protected_constraints,
)
from sokoban_agent.planning.agentic.grounding import (
    SubgoalGroundingError,
    ground_push_subgoal,
    ground_push_subgoal_direct,
)
from sokoban_agent.planning.agentic.models import (
    BoardAnalysis,
    ProtectedConstraint,
    PushSubgoal,
)
from sokoban_agent.planning.contracts import Observation

GroundingRoute = Literal[
    "remember_failure",
    "execute_until_push",
    "__end__",
]


def ground_agentic_subgoal(state: AgenticState) -> dict[str, object]:
    """Ground one approved push without searching for puzzle completion."""

    observation = cast(
        Observation,
        np.asarray(state["observation"], dtype=np.uint8),
    )
    planning = state["planning"]
    analysis = BoardAnalysis.model_validate(planning["board_analysis"])
    subgoal = PushSubgoal.model_validate(active_subgoal(state))
    constraints = tuple(
        ProtectedConstraint.model_validate(payload)
        for payload in protected_constraints(state)
    )
    local_search = state["meta"]["grounding_mode"] == "local-search"
    metrics = state["metrics"]
    grounder = ground_push_subgoal if local_search else ground_push_subgoal_direct
    started_at = perf_counter()
    try:
        plan = grounder(
            observation,
            analysis,
            subgoal,
            constraints,
        )
    except SubgoalGroundingError as error:
        elapsed = perf_counter() - started_at
        failure = {"kind": error.kind, "message": str(error)}
        feedback = f"{error.kind}: {error}"
        return {
            "planning": {
                **planning,
                "grounded_plan": None,
                "grounding_failure": failure,
                "latest_strategy_feedback": [feedback],
            },
            "metrics": update_agentic_metrics(
                metrics,
                strategy={
                    "subgoal_attempts": (
                        metrics["strategy"]["subgoal_attempts"] + 1
                    ),
                    "subgoal_grounding_failures": (
                        metrics["strategy"]["subgoal_grounding_failures"] + 1
                    ),
                },
                rules={
                    "checks": metrics["rules"]["checks"] + 1,
                    "reachability_calls": (
                        metrics["rules"]["reachability_calls"]
                        + int(local_search)
                    ),
                },
                local_search={
                    "calls": (
                        metrics["local_search"]["calls"] + int(local_search)
                    ),
                    "elapsed_seconds": (
                        metrics["local_search"]["elapsed_seconds"]
                        + (elapsed if local_search else 0.0)
                    ),
                },
            ),
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

    elapsed = perf_counter() - started_at
    return {
        "planning": {
            **planning,
            "grounded_plan": plan.model_dump(mode="json"),
            "grounding_failure": None,
        },
        "metrics": update_agentic_metrics(
            metrics,
            strategy={
                "subgoal_attempts": (
                    metrics["strategy"]["subgoal_attempts"] + 1
                )
            },
            rules={
                "checks": metrics["rules"]["checks"] + 1,
                "reachability_calls": (
                    metrics["rules"]["reachability_calls"]
                    + int(local_search)
                ),
            },
            local_search={
                "calls": (
                    metrics["local_search"]["calls"] + int(local_search)
                ),
                "expanded_states": (
                    metrics["local_search"]["expanded_states"]
                    + (
                        plan.expanded_player_states
                        if local_search
                        else 0
                    )
                ),
                "elapsed_seconds": (
                    metrics["local_search"]["elapsed_seconds"]
                    + (elapsed if local_search else 0.0)
                ),
            },
        ),
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

    if state["planning"]["grounding_failure"] is None:
        return "execute_until_push"
    return "remember_failure"


def _step(state: AgenticState) -> int:
    steps = state["info"].get("steps")
    if not isinstance(steps, int):
        raise TypeError("graph info steps must be an integer")
    return steps
