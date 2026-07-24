"""LangGraph Store adapter for reusable static board analysis."""

from __future__ import annotations

from langgraph.runtime import Runtime

from sokoban_agent.graph.agentic_memory_keys import (
    memory_namespace,
    shared_memory,
    topology_memory_key,
)
from sokoban_agent.graph.agentic_state import (
    AgenticRuntimeContext,
    AgenticState,
)
from sokoban_agent.planning.base import Observation
from sokoban_agent.planning.board_analysis import (
    StaticBoardFacts,
    analyze_static_board,
    dump_static_board_facts,
    load_static_board_facts,
)


def get_static_board_facts(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
    observation: Observation,
) -> tuple[StaticBoardFacts, int, int, int]:
    """Recall topology facts and return request, hit, and write increments."""

    if not shared_memory(state) or runtime.store is None:
        return analyze_static_board(observation), 0, 0, 0
    key = topology_memory_key(observation)
    item = runtime.store.get(memory_namespace(state, "board"), key)
    if item is not None:
        try:
            return load_static_board_facts(item.value), 1, 1, 0
        except ValueError:
            pass
    facts = analyze_static_board(observation)
    runtime.store.put(
        memory_namespace(state, "board"),
        key,
        dump_static_board_facts(facts),
        index=False,
    )
    return facts, 1, 0, 1
