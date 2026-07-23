"""LangGraph-first structured Sokoban agent entrypoint."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from sokoban_agent.env import (
    FixedLevelProvider,
    SokobanEnv,
    parse_level,
)
from sokoban_agent.graph.agentic_execution_nodes import (
    detect_agentic_repetition,
    execute_agentic_until_push,
    observe_agentic_state,
    reflect_agentic_execution,
    route_after_action,
    route_after_reflection,
    route_after_repetition,
)
from sokoban_agent.graph.agentic_grounding_node import (
    ground_agentic_subgoal,
    route_after_grounding,
)
from sokoban_agent.graph.agentic_state import (
    AgenticInput,
    AgenticRuntimeContext,
    AgenticState,
)
from sokoban_agent.graph.agentic_strategy_nodes import (
    StrategyNodes,
    route_after_strategy_proposal,
    route_after_strategy_verification,
)
from sokoban_agent.planning.base import Observation
from sokoban_agent.planning.board_analysis import analyze_board
from sokoban_agent.planning.strategy import BoardAnalysis
from sokoban_agent.planning.strategy_runtime import (
    PromptSource,
    StrategyGenerator,
    TransientAgenticError,
)


def initialize_agentic_state(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
) -> dict[str, object]:
    """Reset the requested level into JSON-safe graph state."""

    context = runtime.context or {}
    level_id = state.get("level_id", "tiny-push")
    seed = state.get("seed", 0)
    max_steps = state.get("max_steps", 15)
    level_rows = state.get("level_rows")
    level_provider = (
        FixedLevelProvider([parse_level(level_id, level_rows)])
        if level_rows is not None
        else None
    )
    env = (
        SokobanEnv(max_steps=max_steps, level_provider=level_provider)
        if level_provider is not None
        else SokobanEnv(max_steps=max_steps)
    )
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
        "prompt_resolved": False,
        "model_name": context.get("model_name", "unconfigured"),
        "rationale_mode": context.get("rationale_mode", "on"),
        "grounding_mode": context.get("grounding_mode", "local-search"),
        "status": "initialized",
        "board_analysis": None,
        "strategy_hypothesis": None,
        "strategy_input": {},
        "strategy_attempts": 0,
        "strategy_error": None,
        "strategy_violations": [],
        "active_subgoal": None,
        "grounded_plan": None,
        "grounded_actions": [],
        "grounding_failure": None,
        "action_history": [],
        "execution_result": None,
        "reflection_result": None,
        "completed_subgoals": [],
        "attempt_keys": [],
        "cycle_detected": False,
        "strategy_proposals": 0,
        "strategy_schema_rejections": 0,
        "strategy_semantic_rejections": 0,
        "llm_calls": 0,
        "llm_elapsed_seconds": 0.0,
        "llm_prompt_tokens": 0,
        "llm_output_tokens": 0,
        "local_search_calls": 0,
        "local_expanded_states": 0,
        "local_search_elapsed_seconds": 0.0,
        "rule_checks": 0,
        "reachability_calls": 0,
        "subgoal_grounding_attempts": 0,
        "subgoal_grounding_failures": 0,
        "push_count": 0,
        "effect_matches": 0,
        "effect_mismatches": 0,
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
        "rule_checks": state["rule_checks"] + 1,
        "reachability_calls": state["reachability_calls"] + 1,
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


def route_after_analysis(
    state: AgenticState,
) -> Literal["resolve_prompt", "compose_strategy_input"]:
    """Resolve the managed prompt once, then reuse its pinned commit."""

    return (
        "compose_strategy_input"
        if state["prompt_resolved"]
        else "resolve_prompt"
    )


def build_agentic_graph(
    *,
    checkpointer: InMemorySaver | None = None,
    prompt_source: PromptSource | None = None,
    strategy_generator: StrategyGenerator | None = None,
) -> Any:
    """Compile the structured agent graph using LangGraph primitives."""

    strategy_nodes = StrategyNodes(prompt_source, strategy_generator)
    retry_policy = RetryPolicy(
        initial_interval=0.01,
        backoff_factor=2.0,
        max_interval=0.1,
        max_attempts=3,
        jitter=False,
        retry_on=TransientAgenticError,
    )
    builder = StateGraph(
        AgenticState,
        context_schema=AgenticRuntimeContext,
        input_schema=AgenticInput,
    )
    builder.add_node("initialize", initialize_agentic_state)
    builder.add_node("analyze", analyze_agentic_board)
    builder.add_node(
        "resolve_prompt",
        strategy_nodes.resolve_prompt,
        retry_policy=retry_policy,
    )
    builder.add_node(
        "compose_strategy_input",
        strategy_nodes.compose_strategy_input,
    )
    builder.add_node(
        "propose_strategy",
        strategy_nodes.propose_strategy,
        retry_policy=retry_policy,
    )
    builder.add_node("verify_strategy", strategy_nodes.verify_strategy)
    builder.add_node("detect_repetition", detect_agentic_repetition)
    builder.add_node("ground_subgoal", ground_agentic_subgoal)
    builder.add_node("execute_until_push", execute_agentic_until_push)
    builder.add_node("reflect", reflect_agentic_execution)
    builder.add_node("observe", observe_agentic_state)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "analyze")
    builder.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "resolve_prompt": "resolve_prompt",
            "compose_strategy_input": "compose_strategy_input",
        },
    )
    builder.add_edge("resolve_prompt", "compose_strategy_input")
    builder.add_edge("compose_strategy_input", "propose_strategy")
    builder.add_conditional_edges(
        "propose_strategy",
        route_after_strategy_proposal,
        {
            "compose_strategy_input": "compose_strategy_input",
            "verify_strategy": "verify_strategy",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "verify_strategy",
        route_after_strategy_verification,
        {
            "compose_strategy_input": "compose_strategy_input",
            "detect_repetition": "detect_repetition",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "detect_repetition",
        route_after_repetition,
        {
            "ground_subgoal": "ground_subgoal",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "ground_subgoal",
        route_after_grounding,
        {
            "compose_strategy_input": "compose_strategy_input",
            "execute_until_push": "execute_until_push",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "execute_until_push",
        route_after_action,
        {
            "execute_until_push": "execute_until_push",
            "reflect": "reflect",
        },
    )
    builder.add_conditional_edges(
        "reflect",
        route_after_reflection,
        {"observe": "observe", "__end__": END},
    )
    builder.add_edge("observe", "analyze")
    return builder.compile(checkpointer=checkpointer)


graph = build_agentic_graph()
