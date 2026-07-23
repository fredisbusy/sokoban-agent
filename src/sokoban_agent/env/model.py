"""Shared value objects for Sokoban environments and agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from sokoban_agent.env.levels import Position


class Action(IntEnum):
    """Four primitive actions shared by every agent."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Tile(IntEnum):
    """Integer encoding returned in board observations."""

    FLOOR = 0
    WALL = 1
    TARGET = 2
    BOX = 3
    PLAYER = 4
    BOX_ON_TARGET = 5
    PLAYER_ON_TARGET = 6


@dataclass(frozen=True, slots=True)
class SokobanState:
    """Immutable dynamic state used by the environment and search agents."""

    player: Position
    boxes: frozenset[Position]


@dataclass(frozen=True, slots=True)
class MoveResult:
    """Result of applying one action to an immutable state."""

    state: SokobanState
    invalid_move: bool
    pushed: bool
    box_left_target: bool
    box_entered_target: bool
