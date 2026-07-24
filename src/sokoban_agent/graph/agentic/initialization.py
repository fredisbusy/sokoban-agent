"""Agentic graph state initialization and first routing decision."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from langgraph.runtime import Runtime

from sokoban_agent.env import (
    DEFAULT_LEVEL_CATALOG,
    FixedLevelProvider,
    SokobanEnv,
    level_rows_sha256,
    parse_level,
)
from sokoban_agent.graph.agentic.metrics import initial_agentic_metrics
from sokoban_agent.graph.agentic.state import (
    CURRENT_STATE_SCHEMA_VERSION,
    DEFAULT_LEVEL_ID,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
    GRAPH_REVISION,
    AgenticInfoState,
    AgenticInput,
    AgenticRuntimeContext,
    AgenticState,
    AgenticStatus,
)


def initialize_agentic_state(
    state: AgenticInput,
    runtime: Runtime[AgenticRuntimeContext],
    *,
    resolve_model_name: Callable[[str | None], str],
) -> AgenticState:
    """Reset the requested level into JSON-safe graph state."""

    context = runtime.context or {}
    level_id = state.get("level_id", DEFAULT_LEVEL_ID)
    seed = state.get("seed", DEFAULT_SEED)
    max_steps = state.get("max_steps", DEFAULT_MAX_STEPS)
    supplied_rows = state.get("level_rows")
    if supplied_rows is None:
        catalog_record = DEFAULT_LEVEL_CATALOG.get(level_id)
        level_rows = list(catalog_record.rows)
        level_sha256 = catalog_record.sha256
    else:
        level_rows = supplied_rows
        level_sha256 = level_rows_sha256(level_rows)
    expected_sha256 = state.get("level_sha256")
    if expected_sha256 is not None and expected_sha256 != level_sha256:
        raise ValueError(
            f"{level_id} checksum mismatch: "
            f"expected {expected_sha256}, got {level_sha256}"
        )
    level_provider = FixedLevelProvider([parse_level(level_id, level_rows)])
    env = SokobanEnv(max_steps=max_steps, level_provider=level_provider)
    try:
        observation, raw_info = env.reset(
            seed=seed,
            options={"level_id": level_id},
        )
    finally:
        env.close()
    resolved_level_id = str(raw_info["level_id"])
    model_name = resolve_model_name(context.get("model_name"))
    status: AgenticStatus = (
        "success"
        if raw_info.get("success") is True
        else "deadlock"
        if raw_info.get("deadlock") is True
        else "initialized"
    )
    return {
        "meta": {
            "state_schema_version": CURRENT_STATE_SCHEMA_VERSION,
            "graph_revision": GRAPH_REVISION,
            "level_id": resolved_level_id,
            "level_sha256": level_sha256,
            "level_rows": level_rows,
            "seed": seed,
            "max_steps": max_steps,
            "prompt": {
                "name": context.get("prompt_name", "sokoban-strategy"),
                "commit": context.get("prompt_commit", "latest"),
            },
            "prompt_resolved": False,
            "model_name": model_name,
            "rationale_mode": context.get("rationale_mode", "on"),
            "grounding_mode": context.get(
                "grounding_mode", "local-search"
            ),
            "memory_mode": context.get("memory_mode", "episode"),
            "memory_namespace": context.get(
                "memory_namespace", "default"
            ),
        },
        "planning": {
            "board_analysis": None,
            "strategy_hypothesis": None,
            "strategy_input": {},
            "strategy_attempts": 0,
            "strategy_error": None,
            "strategy_schema_issues": [],
            "latest_strategy_feedback": [],
            "strategy_violations": [],
            "grounded_plan": None,
            "grounding_failure": None,
            "grounding_cache_key": None,
            "completed_subgoals": [],
        },
        "memory": {"attempt_keys": [], "rejected_pushes": {}},
        "execution": {"result": None, "reflection": None},
        "observation": observation.tolist(),
        "info": cast(AgenticInfoState, raw_info),
        "status": status,
        "action_history": [],
        "cycle_detected": False,
        "metrics": initial_agentic_metrics(),
        "push_count": 0,
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


def route_after_initialize(
    state: AgenticState,
) -> Literal["analyze", "__end__"]:
    """Skip planning when reset already produced a terminal board."""

    return (
        "__end__"
        if state["status"] in {"success", "deadlock", "step_limit"}
        else "analyze"
    )


def bind_initialize_node(
    resolve_model_name: Callable[[str | None], str],
) -> Callable[
    [AgenticInput, Runtime[AgenticRuntimeContext]],
    AgenticState,
]:
    """Bind model resolution without hiding the LangGraph node signature."""

    def initialize(
        state: AgenticInput,
        runtime: Runtime[AgenticRuntimeContext],
    ) -> AgenticState:
        return initialize_agentic_state(
            state,
            runtime,
            resolve_model_name=resolve_model_name,
        )

    return initialize
