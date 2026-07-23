"""LangGraph-first structured Sokoban agent entrypoint."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from sokoban_agent.env import SokobanEnv
from sokoban_agent.graph.agentic_state import (
    AgenticInput,
    AgenticRuntimeContext,
    AgenticState,
)
from sokoban_agent.planning.base import Observation
from sokoban_agent.planning.board_analysis import analyze_board
from sokoban_agent.planning.strategy import BoardAnalysis


def initialize_agentic_state(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
) -> dict[str, object]:
    """Reset the requested level into JSON-safe graph state."""

    context = runtime.context or {}
    level_id = state.get("level_id", "tiny-push")
    seed = state.get("seed", 0)
    max_steps = state.get("max_steps", 15)
    env = SokobanEnv(max_steps=max_steps)
    try:
        observation, raw_info = env.reset(
            seed=seed,
            options={"level_id": level_id},
        )
    finally:
        env.close()
    resolved_level_id = str(raw_info["level_id"])
    return {
        "level_id": resolved_level_id,
        "seed": seed,
        "max_steps": max_steps,
        "observation": observation.tolist(),
        "info": dict(raw_info),
        "prompt": {
            "name": context.get("prompt_name", "sokoban-strategy"),
            "commit": context.get("prompt_commit", "unresolved"),
        },
        "model_name": context.get("model_name", "unconfigured"),
        "status": "initialized",
        "board_analysis": None,
        "strategy_hypothesis": None,
        "active_subgoal": None,
        "protected_constraints": [],
        "expected_effect": None,
        "failure_conditions": [],
        "plan_revisions": [],
        "feedback": [],
        "decision_events": [
            {
                "step": 0,
                "stage": "initialize",
                "summary": f"{resolved_level_id} 레벨을 초기화했습니다",
            }
        ],
    }


def analyze_agentic_board(state: AgenticState) -> dict[str, object]:
    """Expose deterministic board facts as a checkpointed graph update."""

    previous_payload = state.get("board_analysis")
    previous = (
        BoardAnalysis.model_validate(previous_payload)
        if previous_payload is not None
        else None
    )
    observation = cast(
        Observation,
        np.asarray(state["observation"], dtype=np.uint8),
    )
    analysis = analyze_board(observation, previous=previous)
    steps = state["info"].get("steps")
    if not isinstance(steps, int):
        raise TypeError("graph info steps must be an integer")
    return {
        "board_analysis": analysis.model_dump(mode="json"),
        "status": "analyzed",
        "decision_events": [
            {
                "step": steps,
                "stage": "analyze",
                "summary": (
                    f"상자 {len(analysis.boxes)}개와 가능한 push "
                    f"{len(analysis.push_options)}개를 분석했습니다"
                ),
            }
        ],
    }


def build_agentic_graph(
    *,
    checkpointer: InMemorySaver | None = None,
) -> Any:
    """Compile the structured agent graph using LangGraph primitives."""

    builder = StateGraph(
        AgenticState,
        context_schema=AgenticRuntimeContext,
        input_schema=AgenticInput,
    )
    builder.add_node("initialize", initialize_agentic_state)
    builder.add_node("analyze", analyze_agentic_board)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "analyze")
    builder.add_edge("analyze", END)
    return builder.compile(checkpointer=checkpointer)


graph = build_agentic_graph()
