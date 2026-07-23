"""Compiled LangGraph entrypoint loaded by LangGraph Studio."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from sokoban_agent.graph.studio_nodes import (
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
from sokoban_agent.graph.studio_state import StudioInput, StudioState


def build_studio_graph() -> Any:
    """Compile the graph loaded by the local LangGraph Agent Server."""

    builder = StateGraph(StudioState, input_schema=StudioInput)
    builder.add_node("초기화", initialize)
    builder.add_node("LLM_계획", llm_plan)
    builder.add_node("AStar_검사", astar_guard)
    builder.add_node("계획_검증", validate_plan)
    builder.add_node("행동_실행", execute_action)
    builder.add_edge(START, "초기화")
    builder.add_edge("초기화", "LLM_계획")
    builder.add_conditional_edges(
        "LLM_계획",
        route_after_llm,
        {"llm_plan": "LLM_계획", "astar_guard": "AStar_검사", "end": END},
    )
    builder.add_conditional_edges(
        "AStar_검사",
        route_after_guard,
        {"validate": "계획_검증", "end": END},
    )
    builder.add_conditional_edges(
        "계획_검증",
        route_after_validation,
        {"llm_plan": "LLM_계획", "execute": "행동_실행", "end": END},
    )
    builder.add_conditional_edges(
        "행동_실행",
        route_after_execution,
        {"llm_plan": "LLM_계획", "execute": "행동_실행", "end": END},
    )
    return builder.compile()


graph = build_studio_graph()
