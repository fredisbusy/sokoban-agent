"""Checkpoint-safe state contracts for the baseline Sokoban graph."""

from __future__ import annotations

from typing_extensions import TypedDict


class ProposalState(TypedDict):
    """Latest planner proposal and its validation provenance."""

    proposed_actions: list[str]
    goal: str | None
    risk: str | None
    guard_summary: str | None
    used_llm_actions: bool


class EpisodeMetricsState(TypedDict):
    """Environment and research measurements accumulated per episode."""

    validation_rejections: int
    total_reward: float
    push_count: int
    revisited_states: int
    repeated_plans: int


class PlanningMetricsState(TypedDict):
    """Planner-node measurements."""

    calls: int
    retries: int
    errors: int
    elapsed_seconds: float


class AlgorithmMetricsState(TypedDict):
    """Deterministic search measurements."""

    calls: int
    requests: int
    cache_hits: int
    failures: int
    fallbacks: int
    expanded_states: int
    elapsed_seconds: float


class GuardMetricsState(TypedDict):
    """Search-guard disposition and contribution measurements."""

    accepted: int
    suffix_added: int
    replaced: int
    failed: int
    proposed_actions: int
    legal_prefix_actions: int
    adopted_actions: int
    suffix_expanded_states: int
    reference_calls: int
    reference_action_count: int
    reference_expanded_states: int
    reference_elapsed_seconds: float
    expansions_saved: int


class LLMMetricsState(TypedDict):
    """Language-model measurements and validation attribution."""

    calls: int
    retries: int
    client_errors: int
    format_errors: int
    invalid_actions: int
    elapsed_seconds: float
    load_seconds: float
    prompt_eval_seconds: float
    eval_seconds: float
    prompt_tokens: int
    output_tokens: int


class BaselineMetrics(TypedDict):
    """Measurements grouped by the component that owns them."""

    episode: EpisodeMetricsState
    planning: PlanningMetricsState
    algorithm: AlgorithmMetricsState
    guard: GuardMetricsState
    llm: LLMMetricsState


class SokobanGraphState(TypedDict):
    """JSON-safe state required to checkpoint and resume one episode."""

    observation: list[list[int]]
    info: dict[str, object]
    seed: int | None
    level_id: str
    plan: list[str]
    proposal: ProposalState | None
    action_history: list[str]
    visited_state_keys: list[str]
    seen_plan_keys: list[str]
    feedback: list[str]
    planning_attempts: int
    truncated: bool
    failure_reason: str | None
    metrics: BaselineMetrics
