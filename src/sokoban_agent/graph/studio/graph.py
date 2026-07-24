"""Compiled LangGraph entrypoint loaded by LangGraph Studio."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from sokoban_agent.graph.studio.nodes import (
    astar_guard,
    execute_action,
    initialize,
    llm_plan,
    route_after_execution,
    route_after_guard,
    route_after_llm,
    route_after_validation,
    validate_plan,
)
from sokoban_agent.graph.studio.state import StudioInput, StudioState


def build_studio_graph() -> Any:
    """Compile the graph loaded by the local LangGraph Agent Server."""

    builder = StateGraph(StudioState, input_schema=StudioInput)
    builder.add_node("initialize", initialize)
    builder.add_node("llm_plan", llm_plan)
    builder.add_node("astar_guard", astar_guard)
    builder.add_node("validate_plan", validate_plan)
    builder.add_node("execute_action", execute_action)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "llm_plan")
    builder.add_conditional_edges(
        "llm_plan",
        route_after_llm,
        {"llm_plan": "llm_plan", "astar_guard": "astar_guard", "end": END},
    )
    builder.add_conditional_edges(
        "astar_guard",
        route_after_guard,
        {"validate": "validate_plan", "end": END},
    )
    builder.add_conditional_edges(
        "validate_plan",
        route_after_validation,
        {"llm_plan": "llm_plan", "execute": "execute_action", "end": END},
    )
    builder.add_conditional_edges(
        "execute_action",
        route_after_execution,
        {"llm_plan": "llm_plan", "execute": "execute_action", "end": END},
    )
    return builder.compile()


graph = build_studio_graph()
