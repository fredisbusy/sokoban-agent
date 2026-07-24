"""Uniform-random planning node."""

from __future__ import annotations

import numpy as np

from sokoban_agent.env import Action
from sokoban_agent.planning.contracts import PlanningContext, PlanningOutcome


class RandomPlanner:
    """Propose one primitive action uniformly at random."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    @property
    def name(self) -> str:
        """Return the experiment name."""

        return "graph:random"

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the independent random generator."""

        self._rng = np.random.default_rng(seed)

    def plan(self, context: PlanningContext) -> PlanningOutcome:
        """Propose one action; the graph validates it before execution."""

        del context
        action = Action(int(self._rng.integers(len(Action))))
        return PlanningOutcome(actions=(action,))
