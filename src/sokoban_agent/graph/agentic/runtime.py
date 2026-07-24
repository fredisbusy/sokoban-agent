"""Thin local invocation boundary for the shared compiled agentic graph."""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from sokoban_agent.graph.agentic.composition import (
    AgenticDependencies,
    build_production_agentic_graph,
    production_dependencies,
)
from sokoban_agent.graph.agentic.state import (
    CURRENT_STATE_SCHEMA_VERSION,
    AgenticInput,
    AgenticRuntimeContext,
    AgenticState,
)
from sokoban_agent.planning.agentic.runtime import (
    PromptSource,
    StrategyGenerator,
)


class CheckpointSchemaMismatch(RuntimeError):
    """Raised when a thread contains an incompatible agentic checkpoint."""


class AgenticGraphRunner:
    """Invoke the same StateGraph definition used by Agent Server and Studio."""

    def __init__(
        self,
        *,
        prompt_source: PromptSource | None = None,
        strategy_generator: StrategyGenerator | None = None,
        checkpointer: InMemorySaver | None = None,
        store: BaseStore | None = None,
    ) -> None:
        if (prompt_source is None) != (strategy_generator is None):
            raise ValueError(
                "prompt_source and strategy_generator must be provided together"
            )
        dependencies = (
            production_dependencies()
            if prompt_source is None or strategy_generator is None
            else AgenticDependencies(prompt_source, strategy_generator)
        )
        self.checkpointer = checkpointer or InMemorySaver()
        self.store = store or InMemoryStore()
        self.graph: Any = build_production_agentic_graph(
            checkpointer=self.checkpointer,
            store=self.store,
            dependencies=dependencies,
        )

    def run(
        self,
        graph_input: AgenticInput,
        *,
        context: AgenticRuntimeContext | None = None,
        thread_id: str | None = None,
    ) -> AgenticState:
        """Run one checkpointed episode without implementing a policy loop."""

        max_steps = graph_input.get("max_steps", 15)
        run_id = thread_id or f"agentic:{uuid4().hex}"
        config = {
            "configurable": {"thread_id": run_id},
            "recursion_limit": max_steps * 12 + 50,
        }
        checkpoint = self.graph.get_state(config)
        if checkpoint.values:
            meta = checkpoint.values.get("meta")
            version = (
                meta.get("state_schema_version")
                if isinstance(meta, dict)
                else None
            )
            if version != CURRENT_STATE_SCHEMA_VERSION:
                raise CheckpointSchemaMismatch(
                    "agentic checkpoint schema mismatch: "
                    f"expected {CURRENT_STATE_SCHEMA_VERSION}, got {version!r}; "
                    "start a new thread"
                )
        result = self.graph.invoke(
            graph_input,
            config,
            context=context,
        )
        return cast(AgenticState, result)
