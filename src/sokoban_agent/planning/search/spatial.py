"""Shared deterministic spatial analysis for Sokoban planning."""

from __future__ import annotations

from collections import deque

from sokoban_agent.env import Action, SokobanState
from sokoban_agent.env.levels import Position, SokobanLevel

DIRECTIONS: tuple[tuple[Action, Position], ...] = (
    (Action.UP, (-1, 0)),
    (Action.RIGHT, (0, 1)),
    (Action.DOWN, (1, 0)),
    (Action.LEFT, (0, -1)),
)


def reachable_paths(
    level: SokobanLevel,
    state: SokobanState,
) -> tuple[set[Position], dict[Position, tuple[Action, ...]]]:
    """Return every player-reachable cell and one shortest path to it."""

    reachable = {state.player}
    paths: dict[Position, tuple[Action, ...]] = {state.player: ()}
    frontier = deque([state.player])
    while frontier:
        position = frontier.popleft()
        for action, delta in DIRECTIONS:
            destination = add_position(position, delta)
            if (
                destination in reachable
                or destination in level.walls
                or destination in state.boxes
                or not inside_level(level, destination)
            ):
                continue
            reachable.add(destination)
            paths[destination] = (*paths[position], action)
            frontier.append(destination)
    return reachable, paths


def target_pull_distances(
    level: SokobanLevel,
) -> dict[Position, dict[Position, int]]:
    """Calculate wall-aware reverse-pull distance from every target."""

    result: dict[Position, dict[Position, int]] = {}
    for target in level.targets:
        distances = {target: 0}
        frontier = deque([target])
        while frontier:
            box = frontier.popleft()
            for _, delta in DIRECTIONS:
                predecessor = subtract_position(box, delta)
                support = subtract_position(predecessor, delta)
                if (
                    predecessor in distances
                    or predecessor in level.walls
                    or support in level.walls
                    or not inside_level(level, predecessor)
                    or not inside_level(level, support)
                ):
                    continue
                distances[predecessor] = distances[box] + 1
                frontier.append(predecessor)
        result[target] = distances
    return result


def static_dead_squares(
    level: SokobanLevel,
    pull_distances: dict[Position, dict[Position, int]],
) -> frozenset[Position]:
    """Return non-target cells from which no target can be reverse-pulled."""

    target_reachable = {
        position
        for distances in pull_distances.values()
        for position in distances
    }
    return frozenset(
        (row, column)
        for row in range(level.height)
        for column in range(level.width)
        if (row, column) not in level.walls
        and (row, column) not in level.targets
        and (row, column) not in target_reachable
    )


def add_position(position: Position, delta: Position) -> Position:
    """Add a direction delta to a board position."""

    return position[0] + delta[0], position[1] + delta[1]


def subtract_position(position: Position, delta: Position) -> Position:
    """Subtract a direction delta from a board position."""

    return position[0] - delta[0], position[1] - delta[1]


def inside_level(level: SokobanLevel, position: Position) -> bool:
    """Return whether a position lies inside the rectangular board."""

    return (
        0 <= position[0] < level.height
        and 0 <= position[1] < level.width
    )
