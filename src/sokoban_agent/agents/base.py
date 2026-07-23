"""Common contracts and expected stopping conditions for agents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from sokoban_agent.env import Action

Observation = NDArray[np.uint8]
AgentInfo = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class AgentDiagnostics:
    """Optional per-episode measurements reported by instrumented agents."""

    llm_calls: int = 0
    llm_retries: int = 0
    llm_client_errors: int = 0
    llm_format_errors: int = 0
    llm_invalid_actions: int = 0
    llm_elapsed_seconds: float = 0.0


@runtime_checkable
class ReportsAgentDiagnostics(Protocol):
    """Optional protocol used by the runner to collect diagnostics."""

    @property
    def diagnostics(self) -> AgentDiagnostics:
        """Return measurements for the current episode."""
        ...


class AgentStopped(RuntimeError):
    """An expected condition where an agent cannot return another action."""


class NoSolutionError(AgentStopped):
    """Raised when a complete search proves that no solution exists."""


class SearchLimitError(AgentStopped):
    """Raised when a search reaches its configured state expansion limit."""


class Agent(Protocol):
    """Minimal policy contract shared by baselines and future LLM agents."""

    @property
    def name(self) -> str:
        """Return a stable name used in experiment results."""
        ...

    def reset(
        self,
        observation: Observation,
        info: AgentInfo,
        *,
        seed: int | None = None,
    ) -> None:
        """Start one episode from its initial observation."""
        ...

    def act(
        self,
        observation: Observation,
        info: AgentInfo,
    ) -> Action:
        """Return the next primitive action."""
        ...
