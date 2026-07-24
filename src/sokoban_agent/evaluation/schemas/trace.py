"""Exact episode trajectory contracts."""

from __future__ import annotations

from dataclasses import dataclass

from sokoban_agent.env import Action
from sokoban_agent.evaluation.schemas.episode import EpisodeResult
from sokoban_agent.planning.base import Observation


@dataclass(frozen=True, slots=True)
class EpisodeFrame:
    """One observed board and the action that led to it."""

    index: int
    observation: Observation
    action: Action | None
    invalid_move: bool
    pushed: bool
    success: bool
    deadlock: bool


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    """Episode result paired with its exact observed state sequence."""

    result: EpisodeResult
    frames: tuple[EpisodeFrame, ...]
