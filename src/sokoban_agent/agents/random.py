"""Uniform random baseline."""

from __future__ import annotations

import numpy as np

from sokoban_agent.agents.base import AgentInfo, Observation
from sokoban_agent.env import Action


class RandomAgent:
    """Select each primitive action with equal probability."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    @property
    def name(self) -> str:
        """Return the experiment name."""

        return "random"

    def reset(
        self,
        observation: Observation,
        info: AgentInfo,
        *,
        seed: int | None = None,
    ) -> None:
        """Reset the independent random generator for one episode."""

        del observation, info
        self._rng = np.random.default_rng(seed)

    def act(
        self,
        observation: Observation,
        info: AgentInfo,
    ) -> Action:
        """Sample one valid action without mutating the observation."""

        del observation, info
        return Action(int(self._rng.integers(len(Action))))
