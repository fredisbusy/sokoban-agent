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
class PlanningOutcome:
    """A proposed action sequence plus generic planning diagnostics."""

    actions: tuple[Action, ...] = ()
    proposed_actions: tuple[Action, ...] = ()
    goal: str | None = None
    decision_summary: str | None = None
    risk: str | None = None
    guard_summary: str | None = None
    error: str | None = None
    error_kind: PlanningErrorKind | None = None
    llm_calls: int = 0
    llm_client_errors: int = 0
    llm_format_errors: int = 0
    llm_elapsed_seconds: float = 0.0
    llm_load_seconds: float = 0.0
    llm_prompt_eval_seconds: float = 0.0
    llm_eval_seconds: float = 0.0
    llm_prompt_tokens: int = 0
    llm_output_tokens: int = 0
    algorithm_calls: int = 0
    algorithm_requests: int = 0
    algorithm_cache_hits: int = 0
    algorithm_failures: int = 0
    algorithm_fallbacks: int = 0
    algorithm_expanded_states: int = 0
    algorithm_elapsed_seconds: float = 0.0
    guard_disposition: GuardDisposition | None = None
    guard_proposed_actions: int = 0
    guard_legal_prefix_actions: int = 0
    guard_adopted_actions: int = 0
    guard_suffix_expanded_states: int = 0
    guard_reference_calls: int = 0
    guard_reference_action_count: int = 0
    guard_reference_expanded_states: int = 0
    guard_reference_elapsed_seconds: float = 0.0
    guard_expansions_saved: int = 0
    elapsed_seconds: float = 0.0


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
