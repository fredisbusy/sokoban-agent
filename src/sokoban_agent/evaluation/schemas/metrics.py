"""Composable immutable metric groups for evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Calls, tokens, and wall time attributed to a language model."""

    calls: int = 0
    elapsed_seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class MemoryUsage:
    """Long-term memory requests and avoided model work."""

    requests: int = 0
    hits: int = 0
    writes: int = 0
    strategy_cache_hits: int = 0
    grounding_cache_hits: int = 0
    analysis_cache_hits: int = 0
    llm_calls_saved: int = 0
    rejected_pushes_filtered: int = 0


@dataclass(frozen=True, slots=True)
class SearchUsage:
    """Calls, expansions, and time for one search boundary."""

    calls: int = 0
    expanded_states: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class RuleUsage:
    """Deterministic validation and reachability work."""

    checks: int = 0
    reachability_calls: int = 0


@dataclass(frozen=True, slots=True)
class StrategyUsage:
    """Structured strategy, subgoal, effect, and revision measurements."""

    proposals: int = 0
    schema_rejections: int = 0
    semantic_rejections: int = 0
    protected_constraint_violations: int = 0
    effect_matches: int = 0
    effect_mismatches: int = 0
    subgoal_attempts: int = 0
    subgoal_grounding_failures: int = 0
    plan_revisions: int = 0
    assignment_revisions: int = 0
    hypothesis_revisions: int = 0
    subgoal_revisions: int = 0

    @property
    def subgoal_successes(self) -> int:
        return self.effect_matches

    @property
    def subgoal_failures(self) -> int:
        return self.subgoal_grounding_failures + self.effect_mismatches


@dataclass(frozen=True, slots=True)
class PromptIdentity:
    """Resolved prompt and model provenance for one episode."""

    name: str
    commit: str
    model_name: str
