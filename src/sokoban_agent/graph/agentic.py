"""LangGraph-first structured Sokoban agent entrypoint."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from sokoban_agent.graph.agentic_state import (
    AgenticInput,
    AgenticRuntimeContext,
    AgenticState,
)


def initialize_agentic_state(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
) -> dict[str, object]:
    """Reset the configured environment into JSON-safe graph state."""

    env = runtime.context.env
    level_id = state.get("level_id", "tiny-push")
    seed = state.get("seed", 0)
    max_steps = state.get("max_steps", env.max_steps)
    if max_steps != env.max_steps:
        raise ValueError("input max_steps must match the runtime environment")
    observation, raw_info = env.reset(
        seed=seed,
        options={"level_id": level_id},
    )
    resolved_level_id = str(raw_info["level_id"])
    return {
        "level_id": resolved_level_id,
        "seed": seed,
        "max_steps": max_steps,
        "observation": observation.tolist(),
        "info": dict(raw_info),
        "prompt": {
            "name": runtime.context.prompt_name,
            "commit": runtime.context.prompt_commit,
        },
        "model_name": runtime.context.model_name,
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
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", END)
    return builder.compile(checkpointer=checkpointer)
