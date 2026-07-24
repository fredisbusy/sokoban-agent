"""Deterministic keys and small helpers for agentic graph memory."""

from __future__ import annotations

import hashlib
import json
from typing import cast

import numpy as np

from sokoban_agent.env.rules import decode_observation
from sokoban_agent.graph.agentic.state import (
    AgenticState,
    active_subgoal,
    protected_constraints,
)
from sokoban_agent.planning.agentic.decision import compact_board_analysis
from sokoban_agent.planning.agentic.models import (
    BoardAnalysis,
    ProtectedConstraint,
    StrategyHypothesis,
)
from sokoban_agent.planning.contracts import Observation


def board_memory_key(state: AgenticState) -> str:
    """Key the exact board abstraction shown to the strategy model."""

    analysis = BoardAnalysis.model_validate(
        state["planning"]["board_analysis"]
    )
    return digest(compact_board_analysis(analysis))


def strategy_memory_key(state: AgenticState) -> str:
    """Key an exact rendered strategy decision context."""

    return digest(
        {
            "prompt": state["meta"]["prompt"],
            "model": state["meta"]["model_name"],
            "rationale": state["meta"]["rationale_mode"],
            "grounding": state["meta"]["grounding_mode"],
            "input": state["planning"]["strategy_input"],
        }
    )


def grounding_memory_key(state: AgenticState) -> str:
    """Key an exact dynamic state and approved single-push subgoal."""

    return digest(
        {
            "observation": state["observation"],
            "subgoal": active_subgoal(state),
            "constraints": protected_constraints(state),
            "mode": state["meta"]["grounding_mode"],
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

    hypothesis = StrategyHypothesis.model_validate(
        state["planning"]["strategy_hypothesis"]
    )
    return f"{hypothesis.subgoal.box_id}:{hypothesis.subgoal.direction}"


def copy_rejections(state: AgenticState) -> dict[str, list[str]]:
    """Copy checkpointed rejection memory before a node update."""

    return {
        key: list(values)
        for key, values in state["memory"]["rejected_pushes"].items()
    }


def constraints(state: AgenticState) -> tuple[ProtectedConstraint, ...]:
    """Parse active protected constraints for deterministic revalidation."""

    return tuple(
        ProtectedConstraint.model_validate(value)
        for value in protected_constraints(state)
    )


def observation(state: AgenticState) -> Observation:
    """Restore the NumPy observation used by domain modules."""

    return cast(
        Observation,
        np.asarray(state["observation"], dtype=np.uint8),
    )


def shared_memory(state: AgenticState) -> bool:
    """Return whether cross-thread LangGraph Store memory is enabled."""

    return state["meta"]["memory_mode"] == "shared"


def memory_namespace(state: AgenticState, kind: str) -> tuple[str, ...]:
    """Build a bounded application namespace for one memory kind."""

    return ("sokoban-agent", state["meta"]["memory_namespace"], kind)


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
