import json
from collections.abc import Mapping
from typing import cast

from langgraph.store.memory import InMemoryStore

from sokoban_agent.graph import build_agentic_graph
from sokoban_agent.graph.agentic.state import AgenticState
from sokoban_agent.planning.agentic.runtime import (
    PromptReferenceValue,
    RenderedStrategyPrompt,
)
from sokoban_agent.planning.llm.client import CompletionMetrics, TextCompletion


class MemoryPromptSource:
    def __init__(self) -> None:
        self.rendered_variables: list[Mapping[str, object]] = []

    def resolve(self, name: str, selector: str) -> PromptReferenceValue:
        return PromptReferenceValue(name, selector)

    def render(
        self,
        reference: PromptReferenceValue,
        variables: Mapping[str, object],
    ) -> RenderedStrategyPrompt:
        self.rendered_variables.append(variables)
        return RenderedStrategyPrompt("system", "user")


class MemoryStrategyGenerator:
    def __init__(self, *push_ids: str) -> None:
        self.push_ids = list(push_ids)
        self.calls = 0

    def generate(
        self,
        prompt: RenderedStrategyPrompt,
        *,
        seed: int | None,
        response_schema: Mapping[str, object],
    ) -> TextCompletion:
        push_id = self.push_ids[self.calls]
        self.calls += 1
        protected_cells = (
            [{"row": 2, "col": 3}] if push_id == "B1:RIGHT" else []
        )
        return TextCompletion(
            json.dumps(
                {
                    "summary": f"{push_id}를 선택한다",
                    "push_id": push_id,
                    "target_id": "T1",
                    "protected_cells": protected_cells,
                    "risk": "예상 위치가 다르면 수정한다",
                },
                ensure_ascii=False,
            ),
            CompletionMetrics(prompt_tokens=20, output_tokens=10),
        )


def _context(
    *,
    grounding_mode: str = "local-search",
    memory_mode: str = "shared",
) -> dict[str, str]:
    return {
        "prompt_name": "sokoban-strategy",
        "prompt_commit": "fixture-commit",
        "model_name": "fixture-model",
        "grounding_mode": grounding_mode,
        "memory_mode": memory_mode,
        "memory_namespace": "memory-tests",
    }


def test_shared_store_skips_model_and_search_on_exact_second_run() -> None:
    prompt_source = MemoryPromptSource()
    generator = MemoryStrategyGenerator("B1:UP")
    graph = build_agentic_graph(
        store=InMemoryStore(),
        prompt_source=prompt_source,
        strategy_generator=generator,
    )
    graph_input = {"level_id": "tiny-push", "seed": 7, "max_steps": 15}

    first = cast(
        AgenticState,
        graph.invoke(
            graph_input,
            {"configurable": {"thread_id": "memory-first"}},
            context=_context(),
        ),
    )
    second = cast(
        AgenticState,
        graph.invoke(
            graph_input,
            {"configurable": {"thread_id": "memory-second"}},
            context=_context(),
        ),
    )

    assert first["status"] == second["status"] == "success"
    assert first["metrics"]["llm"]["calls"] == 1
    assert first["metrics"]["local_search"]["calls"] == 1
    assert second["metrics"]["llm"]["calls"] == 0
    assert second["metrics"]["local_search"]["calls"] == 0
    assert second["metrics"]["memory"]["llm_calls_saved"] == 1
    assert second["metrics"]["memory"]["analysis_cache_hits"] == 1
    assert second["metrics"]["memory"]["strategy_cache_hits"] == 1
    assert second["metrics"]["memory"]["grounding_cache_hits"] == 1
    assert generator.calls == 1
    assert any(
        event["stage"] == "recall_strategy"
        and "재사용" in event["summary"]
        for event in second["decision_events"]
    )


def test_episode_failure_memory_filters_rejected_push_before_retry() -> None:
    prompt_source = MemoryPromptSource()
    generator = MemoryStrategyGenerator("B1:RIGHT", "B1:UP", "B1:RIGHT")
    graph = build_agentic_graph(
        prompt_source=prompt_source,
        strategy_generator=generator,
    )

    result = cast(
        AgenticState,
        graph.invoke(
            {"level_id": "tiny-walk", "seed": 7, "max_steps": 15},
            context=_context(
                grounding_mode="local-search",
                memory_mode="episode",
            ),
        ),
    )

    assert result["status"] == "success"
    assert generator.calls == 3
    assert result["metrics"]["strategy"]["semantic_rejections"] == 1
    assert result["metrics"]["memory"]["rejected_pushes_filtered"] == 1
    retry_context = json.loads(
        str(prompt_source.rendered_variables[1]["board_analysis_json"])
    )
    retry_pushes = {
        option["push_id"] for option in retry_context["safe_push_options"]
    }
    assert "B1:RIGHT" not in retry_pushes
    assert "B1:UP" in retry_pushes
    assert any(
        event["stage"] == "remember_failure"
        for event in result["decision_events"]
    )
