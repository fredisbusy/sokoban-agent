"""Production composition root for the structured Agent Server graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore

from sokoban_agent.graph.agentic.builder import build_agentic_graph
from sokoban_agent.planning.agentic.runtime import (
    LangSmithPromptSource,
    LiteLLMStrategyGenerator,
    PromptSource,
    StrategyGenerator,
)
from sokoban_agent.planning.llm.client import LiteLLMClient, OllamaSettings


@dataclass(frozen=True, slots=True)
class AgenticDependencies:
    """Providers required by the structured graph definition."""

    prompt_source: PromptSource
    strategy_generator: StrategyGenerator


def production_dependencies(
    settings: OllamaSettings | None = None,
) -> AgenticDependencies:
    """Create one provider set whose lifetime matches one compiled graph."""

    return AgenticDependencies(
        prompt_source=LangSmithPromptSource(),
        strategy_generator=LiteLLMStrategyGenerator(
            LiteLLMClient(settings) if settings is not None else None
        ),
    )


def build_production_agentic_graph(
    *,
    checkpointer: InMemorySaver | None = None,
    store: BaseStore | None = None,
    dependencies: AgenticDependencies | None = None,
) -> Any:
    """Compile the production graph from an explicit provider set."""

    providers = dependencies or production_dependencies()
    return build_agentic_graph(
        prompt_source=providers.prompt_source,
        strategy_generator=providers.strategy_generator,
        checkpointer=checkpointer,
        store=store,
    )


graph = build_production_agentic_graph()
