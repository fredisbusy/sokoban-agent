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
    algorithm_fallbacks: int = 0
    algorithm_expanded_states: int = 0
    algorithm_elapsed_seconds: float = 0.0
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
