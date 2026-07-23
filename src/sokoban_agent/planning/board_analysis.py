"""Convert one observation into stable, model-facing Sokoban facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sokoban_agent.env.levels import Position
from sokoban_agent.env.rules import decode_observation
from sokoban_agent.planning.base import Observation
from sokoban_agent.planning.spatial import (
    DIRECTIONS,
    add_position,
    inside_level,
    reachable_paths,
    static_dead_squares,
    subtract_position,
    target_pull_distances,
)
from sokoban_agent.planning.strategy import (
    BoardAnalysis,
    BoxFact,
    Cell,
    Direction,
    PullDistance,
    PushOption,
    TargetFact,
)


@dataclass(frozen=True, slots=True)
class StaticBoardFacts:
    """Topology-only spatial facts reusable across dynamic observations."""

    dead_squares: frozenset[Position]
    pull_distances: dict[Position, dict[Position, int]]


def analyze_static_board(observation: Observation) -> StaticBoardFacts:
    """Compute facts that depend only on walls, targets, and board shape."""

    level, _ = decode_observation(observation)
    pull_distances = target_pull_distances(level)
    return StaticBoardFacts(
        dead_squares=static_dead_squares(level, pull_distances),
        pull_distances=pull_distances,
    )


def analyze_board(
    observation: Observation,
    *,
    previous: BoardAnalysis | None = None,
    static_facts: StaticBoardFacts | None = None,
) -> BoardAnalysis:
    """Return deterministic spatial facts with stable logical identities."""

    level, state = decode_observation(observation)
    boxes = _box_facts(state.boxes, previous)
    targets = _target_facts(level.targets, previous)
    box_ids = {box.position: box.box_id for box in boxes}
    reachable, _ = reachable_paths(level, state)
    static = static_facts or analyze_static_board(observation)
    pull_distances = static.pull_distances
    dead_squares = static.dead_squares

    push_options: list[PushOption] = []
    for position in sorted(state.boxes):
        for action, delta in DIRECTIONS:
            support = subtract_position(position, delta)
            destination = add_position(position, delta)
            if (
                support not in reachable
                or not inside_level(level, destination)
                or destination in level.walls
                or destination in state.boxes
            ):
                continue
            push_options.append(
                PushOption(
                    box_id=box_ids[_cell(position)],
                    direction=cast(Direction, action.name),
                    support=_cell(support),
                    destination=_cell(destination),
                    creates_static_deadlock=(
                        destination in dead_squares
                        and destination not in level.targets
                    ),
                )
            )

    distances = [
        PullDistance(
            box_id=box.box_id,
            target_id=target.target_id,
            distance=pull_distances[_position(target.position)].get(
                _position(box.position)
            ),
        )
        for box in boxes
        for target in targets
    ]
    return BoardAnalysis(
        boxes=boxes,
        targets=targets,
        dead_squares=tuple(_cell(position) for position in sorted(dead_squares)),
        reachable_cells=tuple(_cell(position) for position in sorted(reachable)),
        push_options=tuple(push_options),
        reverse_pull_distances=tuple(distances),
    )


def dump_static_board_facts(
    facts: StaticBoardFacts,
) -> dict[str, object]:
    """Serialize topology facts for a LangGraph Store value."""

    return {
        "dead_squares": [list(position) for position in sorted(facts.dead_squares)],
        "pull_distances": [
            {
                "target": list(target),
                "distances": [
                    [position[0], position[1], distance]
                    for position, distance in sorted(distances.items())
                ],
            }
            for target, distances in sorted(facts.pull_distances.items())
        ],
    }


def load_static_board_facts(
    payload: dict[str, object],
) -> StaticBoardFacts:
    """Validate and deserialize topology facts from a LangGraph Store."""

    raw_dead_squares = payload.get("dead_squares")
    raw_pull_distances = payload.get("pull_distances")
    if not isinstance(raw_dead_squares, list) or not isinstance(
        raw_pull_distances, list
    ):
        raise ValueError("stored static board facts have an invalid shape")
    dead_squares = frozenset(_stored_position(item) for item in raw_dead_squares)
    pull_distances: dict[Position, dict[Position, int]] = {}
    for item in raw_pull_distances:
        if not isinstance(item, dict):
            raise ValueError("stored pull distance entry must be an object")
        target = _stored_position(item.get("target"))
        raw_distances = item.get("distances")
        if not isinstance(raw_distances, list):
            raise ValueError("stored pull distances must be a list")
        distances: dict[Position, int] = {}
        for raw_distance in raw_distances:
            if (
                not isinstance(raw_distance, list)
                or len(raw_distance) != 3
                or not all(isinstance(value, int) for value in raw_distance)
            ):
                raise ValueError("stored pull distance must be [row, col, distance]")
            distances[(raw_distance[0], raw_distance[1])] = raw_distance[2]
        pull_distances[target] = distances
    return StaticBoardFacts(
        dead_squares=dead_squares,
        pull_distances=pull_distances,
    )


def _box_facts(
    positions: frozenset[Position],
    previous: BoardAnalysis | None,
) -> tuple[BoxFact, ...]:
    if previous is None:
        return tuple(
            BoxFact(box_id=f"B{index}", position=_cell(position))
            for index, position in enumerate(sorted(positions), start=1)
        )
    previous_by_position = {
        _position(box.position): box.box_id for box in previous.boxes
    }
    stationary = positions & previous_by_position.keys()
    removed = set(previous_by_position) - positions
    added = set(positions) - previous_by_position.keys()
    if len(removed) != len(added) or len(removed) > 1:
        raise ValueError("box identity requires at most one push per observation")
    ids_by_position = {
        position: previous_by_position[position] for position in stationary
    }
    if removed:
        ids_by_position[next(iter(added))] = previous_by_position[
            next(iter(removed))
        ]
    return tuple(
        BoxFact(box_id=ids_by_position[position], position=_cell(position))
        for position in sorted(positions)
    )


def _target_facts(
    positions: frozenset[Position],
    previous: BoardAnalysis | None,
) -> tuple[TargetFact, ...]:
    if previous is None:
        return tuple(
            TargetFact(target_id=f"T{index}", position=_cell(position))
            for index, position in enumerate(sorted(positions), start=1)
        )
    previous_by_position = {
        _position(target.position): target.target_id
        for target in previous.targets
    }
    if set(previous_by_position) != set(positions):
        raise ValueError("targets changed between observations")
    return tuple(
        TargetFact(
            target_id=previous_by_position[position],
            position=_cell(position),
        )
        for position in sorted(positions)
    )


def _cell(position: Position) -> Cell:
    return Cell(row=position[0], col=position[1])


def _position(cell: Cell) -> Position:
    return cell.row, cell.col


def _stored_position(value: object) -> Position:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(part, int) for part in value)
    ):
        raise ValueError("stored position must be [row, col]")
    return value[0], value[1]
