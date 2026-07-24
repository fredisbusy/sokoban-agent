"""Nested JSON-safe metric channels for the structured agent graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, cast


class StrategyMetricsState(TypedDict):
    proposals: int
    schema_rejections: int
    semantic_rejections: int
    subgoal_attempts: int
    subgoal_grounding_failures: int
    effect_matches: int
    effect_mismatches: int


class LLMMetricsState(TypedDict):
    calls: int
    elapsed_seconds: float
    prompt_tokens: int
    output_tokens: int


class MemoryMetricsState(TypedDict):
    requests: int
    hits: int
    writes: int
    strategy_cache_hits: int
    grounding_cache_hits: int
    analysis_cache_hits: int
    llm_calls_saved: int
    rejected_pushes_filtered: int


class SearchMetricsState(TypedDict):
    calls: int
    expanded_states: int
    elapsed_seconds: float


class RuleMetricsState(TypedDict):
    checks: int
    reachability_calls: int


class AgenticMetrics(TypedDict):
    strategy: StrategyMetricsState
    llm: LLMMetricsState
    memory: MemoryMetricsState
    local_search: SearchMetricsState
    rules: RuleMetricsState


def initial_agentic_metrics() -> AgenticMetrics:
    """Return the complete required metric state for graph initialization."""

    return {
        "strategy": {
            "proposals": 0,
            "schema_rejections": 0,
            "semantic_rejections": 0,
            "subgoal_attempts": 0,
            "subgoal_grounding_failures": 0,
            "effect_matches": 0,
            "effect_mismatches": 0,
        },
        "llm": {
            "calls": 0,
            "elapsed_seconds": 0.0,
            "prompt_tokens": 0,
            "output_tokens": 0,
        },
        "memory": {
            "requests": 0,
            "hits": 0,
            "writes": 0,
            "strategy_cache_hits": 0,
            "grounding_cache_hits": 0,
            "analysis_cache_hits": 0,
            "llm_calls_saved": 0,
            "rejected_pushes_filtered": 0,
        },
        "local_search": {
            "calls": 0,
            "expanded_states": 0,
            "elapsed_seconds": 0.0,
        },
        "rules": {"checks": 0, "reachability_calls": 0},
    }


def update_agentic_metrics(
    metrics: AgenticMetrics,
    *,
    strategy: Mapping[str, int] | None = None,
    llm: Mapping[str, int | float] | None = None,
    memory: Mapping[str, int] | None = None,
    local_search: Mapping[str, int | float] | None = None,
    rules: Mapping[str, int] | None = None,
) -> AgenticMetrics:
    """Replace selected absolute metric values while retaining all keys."""

    return {
        "strategy": cast(
            StrategyMetricsState, {**metrics["strategy"], **(strategy or {})}
        ),
        "llm": cast(LLMMetricsState, {**metrics["llm"], **(llm or {})}),
        "memory": cast(
            MemoryMetricsState, {**metrics["memory"], **(memory or {})}
        ),
        "local_search": cast(
            SearchMetricsState,
            {**metrics["local_search"], **(local_search or {})},
        ),
        "rules": cast(
            RuleMetricsState, {**metrics["rules"], **(rules or {})}
        ),
    }
