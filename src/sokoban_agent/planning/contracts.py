"""Planner contracts consumed by the LangGraph runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from sokoban_agent.env import Action

Observation = NDArray[np.uint8]
PlanningErrorKind = Literal["client", "format", "search", "empty"]
GuardDisposition = Literal[
    "accepted",
    "suffix_added",
    "replaced",
    "failed",
]


@dataclass(frozen=True, slots=True)
class PlanningContext:
    """Immutable state exposed to one planning node invocation."""

    observation: Observation
    info: Mapping[str, object]
    action_history: tuple[Action, ...]
    feedback: tuple[str, ...]
    seed: int | None


@dataclass(frozen=True, slots=True)
class PlanningNarrative:
    """Human-readable planner diagnostics kept separate from execution data."""

    goal: str | None = None
    decision_summary: str | None = None
    risk: str | None = None
    guard_summary: str | None = None


@dataclass(frozen=True, slots=True)
class PlanningFailure:
    """A planner failure routed back through the graph."""

    message: str
    kind: PlanningErrorKind


@dataclass(frozen=True, slots=True)
class LLMPlanningMetrics:
    """Model-call work performed for one planning proposal."""

    calls: int = 0
    client_errors: int = 0
    format_errors: int = 0
    elapsed_seconds: float = 0.0
    load_seconds: float = 0.0
    prompt_eval_seconds: float = 0.0
    eval_seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class AlgorithmPlanningMetrics:
    """Deterministic search work performed for one planning proposal."""

    calls: int = 0
    requests: int = 0
    cache_hits: int = 0
    failures: int = 0
    fallbacks: int = 0
    expanded_states: int = 0
    elapsed_seconds: float = 0.0

    def plus(
        self,
        *,
        calls: int = 0,
        requests: int = 0,
        cache_hits: int = 0,
        failures: int = 0,
        fallbacks: int = 0,
        expanded_states: int = 0,
        elapsed_seconds: float = 0.0,
    ) -> AlgorithmPlanningMetrics:
        """Return a copy with incremental search work added."""

        return AlgorithmPlanningMetrics(
            calls=self.calls + calls,
            requests=self.requests + requests,
            cache_hits=self.cache_hits + cache_hits,
            failures=self.failures + failures,
            fallbacks=self.fallbacks + fallbacks,
            expanded_states=self.expanded_states + expanded_states,
            elapsed_seconds=self.elapsed_seconds + elapsed_seconds,
        )


@dataclass(frozen=True, slots=True)
class GuardPlanningMetrics:
    """Search-guard disposition and contribution measurements."""

    disposition: GuardDisposition | None = None
    proposed_actions: int = 0
    legal_prefix_actions: int = 0
    adopted_actions: int = 0
    suffix_expanded_states: int = 0
    reference_calls: int = 0
    reference_action_count: int = 0
    reference_expanded_states: int = 0
    reference_elapsed_seconds: float = 0.0
    expansions_saved: int = 0


@dataclass(frozen=True, slots=True)
class PlanningOutcome:
    """A proposal composed from execution data, diagnostics, and metrics."""

    actions: tuple[Action, ...] = ()
    proposed_actions: tuple[Action, ...] = ()
    narrative: PlanningNarrative = PlanningNarrative()
    failure: PlanningFailure | None = None
    llm: LLMPlanningMetrics = LLMPlanningMetrics()
    algorithm: AlgorithmPlanningMetrics = AlgorithmPlanningMetrics()
    guard: GuardPlanningMetrics = GuardPlanningMetrics()
    elapsed_seconds: float = 0.0

    @property
    def error(self) -> str | None:
        return self.failure.message if self.failure else None

    @property
    def error_kind(self) -> PlanningErrorKind | None:
        return self.failure.kind if self.failure else None

    @property
    def goal(self) -> str | None:
        return self.narrative.goal

    @property
    def decision_summary(self) -> str | None:
        return self.narrative.decision_summary

    @property
    def risk(self) -> str | None:
        return self.narrative.risk

    @property
    def guard_summary(self) -> str | None:
        return self.narrative.guard_summary

    @property
    def llm_calls(self) -> int:
        return self.llm.calls

    @property
    def llm_client_errors(self) -> int:
        return self.llm.client_errors

    @property
    def llm_format_errors(self) -> int:
        return self.llm.format_errors

    @property
    def llm_elapsed_seconds(self) -> float:
        return self.llm.elapsed_seconds

    @property
    def llm_load_seconds(self) -> float:
        return self.llm.load_seconds

    @property
    def llm_prompt_eval_seconds(self) -> float:
        return self.llm.prompt_eval_seconds

    @property
    def llm_eval_seconds(self) -> float:
        return self.llm.eval_seconds

    @property
    def llm_prompt_tokens(self) -> int:
        return self.llm.prompt_tokens

    @property
    def llm_output_tokens(self) -> int:
        return self.llm.output_tokens

    @property
    def algorithm_calls(self) -> int:
        return self.algorithm.calls

    @property
    def algorithm_requests(self) -> int:
        return self.algorithm.requests

    @property
    def algorithm_cache_hits(self) -> int:
        return self.algorithm.cache_hits

    @property
    def algorithm_failures(self) -> int:
        return self.algorithm.failures

    @property
    def algorithm_fallbacks(self) -> int:
        return self.algorithm.fallbacks

    @property
    def algorithm_expanded_states(self) -> int:
        return self.algorithm.expanded_states

    @property
    def algorithm_elapsed_seconds(self) -> float:
        return self.algorithm.elapsed_seconds

    @property
    def guard_disposition(self) -> GuardDisposition | None:
        return self.guard.disposition

    @property
    def guard_proposed_actions(self) -> int:
        return self.guard.proposed_actions

    @property
    def guard_legal_prefix_actions(self) -> int:
        return self.guard.legal_prefix_actions

    @property
    def guard_adopted_actions(self) -> int:
        return self.guard.adopted_actions

    @property
    def guard_suffix_expanded_states(self) -> int:
        return self.guard.suffix_expanded_states

    @property
    def guard_reference_calls(self) -> int:
        return self.guard.reference_calls

    @property
    def guard_reference_action_count(self) -> int:
        return self.guard.reference_action_count

    @property
    def guard_reference_expanded_states(self) -> int:
        return self.guard.reference_expanded_states

    @property
    def guard_reference_elapsed_seconds(self) -> float:
        return self.guard.reference_elapsed_seconds

    @property
    def guard_expansions_saved(self) -> int:
        return self.guard.expansions_saved


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A deterministic search plan and its measured work."""

    actions: tuple[Action, ...]
    expanded_states: int
    elapsed_seconds: float


class Planner(Protocol):
    """A replaceable planning node used by the Sokoban graph."""

    @property
    def name(self) -> str:
        """Return a stable experiment label."""
        ...

    def reset(self, *, seed: int | None = None) -> None:
        """Reset planner-local state for a new graph run."""
        ...

    def plan(self, context: PlanningContext) -> PlanningOutcome:
        """Return one or more primitive actions without mutating the game."""
        ...


class NoSolutionError(RuntimeError):
    """Raised when complete search proves that no solution exists."""


class SearchLimitError(RuntimeError):
    """Raised when search reaches its configured expansion limit."""
