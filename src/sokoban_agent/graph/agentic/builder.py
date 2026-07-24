"""LangGraph-first structured Sokoban agent entrypoint."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy

from sokoban_agent.graph.agentic.initialization import (
    bind_initialize_node,
    route_after_initialize,
)
from sokoban_agent.graph.agentic.memory.nodes import (
    recall_failed_decisions,
    recall_grounding,
    recall_strategy,
    remember_failure,
    remember_outcome,
    route_after_failure_memory,
    route_after_outcome_memory,
)
from sokoban_agent.graph.agentic.memory.static import get_static_board_facts
from sokoban_agent.graph.agentic.metrics import (
    update_agentic_metrics,
)
from sokoban_agent.graph.agentic.nodes.execution import (
    detect_agentic_repetition,
    execute_agentic_until_push,
    observe_agentic_state,
    reflect_agentic_execution,
    route_after_action,
    route_after_repetition,
)
from sokoban_agent.graph.agentic.nodes.grounding import (
    ground_agentic_subgoal,
    route_after_grounding,
)
from sokoban_agent.graph.agentic.nodes.strategy import (
    StrategyNodes,
    route_after_strategy_proposal,
    route_after_strategy_verification,
)
from sokoban_agent.graph.agentic.state import (
    AgenticInput,
    AgenticRuntimeContext,
    AgenticState,
)
from sokoban_agent.planning.agentic.analysis import analyze_board
from sokoban_agent.planning.agentic.models import BoardAnalysis
from sokoban_agent.planning.agentic.runtime import (
    PromptSource,
    StrategyGenerator,
    TransientAgenticError,
)
from sokoban_agent.planning.contracts import Observation


def analyze_agentic_board(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
) -> dict[str, object]:
    """Expose deterministic board facts as a checkpointed graph update."""

    planning = state["planning"]
    previous_payload = planning["board_analysis"]
    previous = (
        BoardAnalysis.model_validate(previous_payload)
        if previous_payload is not None
        else None
    )
    observation = cast(
        Observation,
        np.asarray(state["observation"], dtype=np.uint8),
    )
    static_facts, requests, hits, writes = get_static_board_facts(
        state,
        runtime,
        observation,
    )
    analysis = analyze_board(
        observation,
        previous=previous,
        static_facts=static_facts,
    )
    steps = state["info"].get("steps")
    if not isinstance(steps, int):
        raise TypeError("graph info steps must be an integer")
    metrics = state["metrics"]
    return {
        "planning": {
            **planning,
            "board_analysis": analysis.model_dump(mode="json"),
        },
        "metrics": update_agentic_metrics(
            metrics,
            rules={
                "checks": metrics["rules"]["checks"] + 1,
                "reachability_calls": (
                    metrics["rules"]["reachability_calls"] + 1
                ),
            },
            memory={
                "requests": metrics["memory"]["requests"] + requests,
                "hits": metrics["memory"]["hits"] + hits,
                "writes": metrics["memory"]["writes"] + writes,
                "analysis_cache_hits": (
                    metrics["memory"]["analysis_cache_hits"] + hits
                ),
            },
        ),
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
        if state["meta"]["prompt_resolved"]
        else "resolve_prompt"
    )


def build_agentic_graph(
    *,
    prompt_source: PromptSource,
    strategy_generator: StrategyGenerator,
    checkpointer: InMemorySaver | None = None,
    store: BaseStore | None = None,
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
    builder.add_node(
        "initialize",
        cast(Any, bind_initialize_node(strategy_nodes.resolve_model_name)),
    )
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
    builder.add_node("recall_failures", recall_failed_decisions)
    builder.add_node("recall_strategy", recall_strategy)
    builder.add_node(
        "propose_strategy",
        strategy_nodes.propose_strategy,
        retry_policy=retry_policy,
    )
    builder.add_node("verify_strategy", strategy_nodes.verify_strategy)
    builder.add_node("remember_failure", remember_failure)
    builder.add_node("detect_repetition", detect_agentic_repetition)
    builder.add_node("recall_grounding", recall_grounding)
    builder.add_node("ground_subgoal", ground_agentic_subgoal)
    builder.add_node("execute_until_push", execute_agentic_until_push)
    builder.add_node("reflect", reflect_agentic_execution)
    builder.add_node("remember_outcome", remember_outcome)
    builder.add_node("observe", observe_agentic_state)
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        route_after_initialize,
        {"analyze": "analyze", "__end__": END},
    )
    builder.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "resolve_prompt": "resolve_prompt",
            "compose_strategy_input": "recall_failures",
        },
    )
    builder.add_edge("resolve_prompt", "recall_failures")
    builder.add_edge("recall_failures", "compose_strategy_input")
    builder.add_edge("compose_strategy_input", "recall_strategy")
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
            "remember_failure": "remember_failure",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "remember_failure",
        route_after_failure_memory,
        {
            "compose_strategy_input": "compose_strategy_input",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "detect_repetition",
        route_after_repetition,
        {
            "ground_subgoal": "recall_grounding",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "ground_subgoal",
        route_after_grounding,
        {
            "remember_failure": "remember_failure",
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
    builder.add_edge("reflect", "remember_outcome")
    builder.add_conditional_edges(
        "remember_outcome",
        route_after_outcome_memory,
        {"observe": "observe", "__end__": END},
    )
    builder.add_edge("observe", "analyze")
    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
