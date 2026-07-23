"""Convert one observation into stable, model-facing Sokoban facts."""

from __future__ import annotations

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


def analyze_board(
    observation: Observation,
    *,
    previous: BoardAnalysis | None = None,
) -> BoardAnalysis:
    """Return deterministic spatial facts with stable logical identities."""

    level, state = decode_observation(observation)
    boxes = _box_facts(state.boxes, previous)
    targets = _target_facts(level.targets, previous)
    box_ids = {box.position: box.box_id for box in boxes}
    reachable, _ = reachable_paths(level, state)
    pull_distances = target_pull_distances(level)
    dead_squares = static_dead_squares(level, pull_distances)

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
