"""Deterministic keys and small helpers for agentic graph memory."""

from __future__ import annotations

import hashlib
import json
from typing import cast

import numpy as np

from sokoban_agent.env.rules import decode_observation
from sokoban_agent.graph.agentic_state import AgenticState
from sokoban_agent.planning.base import Observation
from sokoban_agent.planning.strategy import (
    BoardAnalysis,
    ProtectedConstraint,
    StrategyHypothesis,
)
from sokoban_agent.planning.strategy_decision import compact_board_analysis


def board_memory_key(state: AgenticState) -> str:
    """Key the exact board abstraction shown to the strategy model."""

    analysis = BoardAnalysis.model_validate(state["board_analysis"])
    return digest(compact_board_analysis(analysis))


def strategy_memory_key(state: AgenticState) -> str:
    """Key an exact rendered strategy decision context."""

    return digest(
        {
            "prompt": state["prompt"],
            "model": state["model_name"],
            "rationale": state["rationale_mode"],
            "grounding": state["grounding_mode"],
            "input": state["strategy_input"],
        }
    )


def grounding_memory_key(state: AgenticState) -> str:
    """Key an exact dynamic state and approved single-push subgoal."""

    return digest(
        {
            "observation": state["observation"],
            "subgoal": state["active_subgoal"],
            "constraints": state["protected_constraints"],
            "mode": state["grounding_mode"],
        }
    )


def topology_memory_key(observation: Observation) -> str:
    """Key immutable board geometry independently of boxes and player."""

    level, _ = decode_observation(observation)
    return digest(
        {
            "shape": list(level.shape),
            "walls": [list(value) for value in sorted(level.walls)],
            "targets": [list(value) for value in sorted(level.targets)],
        }
    )


def current_push_id(state: AgenticState) -> str:
    """Return the compact push identity of the active hypothesis."""

    hypothesis = StrategyHypothesis.model_validate(state["strategy_hypothesis"])
    return f"{hypothesis.subgoal.box_id}:{hypothesis.subgoal.direction}"


def copy_rejections(state: AgenticState) -> dict[str, list[str]]:
    """Copy checkpointed rejection memory before a node update."""

    return {
        key: list(values)
        for key, values in state["rejected_pushes"].items()
    }


def constraints(state: AgenticState) -> tuple[ProtectedConstraint, ...]:
    """Parse active protected constraints for deterministic revalidation."""

    return tuple(
        ProtectedConstraint.model_validate(value)
        for value in state["protected_constraints"]
    )


def observation(state: AgenticState) -> Observation:
    """Restore the NumPy observation used by domain modules."""

    return cast(
        Observation,
        np.asarray(state["observation"], dtype=np.uint8),
    )


def shared_memory(state: AgenticState) -> bool:
    """Return whether cross-thread LangGraph Store memory is enabled."""

    return state["memory_mode"] == "shared"


def memory_namespace(state: AgenticState, kind: str) -> tuple[str, ...]:
    """Build a bounded application namespace for one memory kind."""

    return ("sokoban-agent", state["memory_namespace"], kind)


def digest(value: object) -> str:
    """Hash a JSON-safe value with canonical ordering."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def memory_event(
    state: AgenticState,
    stage: str,
    summary: str,
) -> dict[str, object]:
    """Create one observable memory decision event."""

    steps = state["info"].get("steps")
    if not isinstance(steps, int):
        raise TypeError("graph info steps must be an integer")
    return {"step": steps, "stage": stage, "summary": summary}
