"""Push-based A* search for Sokoban."""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import count
from time import perf_counter

from sokoban_agent.env import Action, SokobanState
from sokoban_agent.env.levels import Position, SokobanLevel
from sokoban_agent.env.rules import decode_observation, is_success
from sokoban_agent.planning.base import (
    NoSolutionError,
    Observation,
    PlanningContext,
    PlanningOutcome,
    SearchLimitError,
    SearchResult,
)

_DIRECTIONS: tuple[tuple[Action, Position], ...] = (
    (Action.UP, (-1, 0)),
    (Action.RIGHT, (0, 1)),
    (Action.DOWN, (1, 0)),
    (Action.LEFT, (0, -1)),
)
_StateKey = tuple[frozenset[Position], Position]
_Parent = tuple[_StateKey, tuple[Action, ...]] | None
_INF = 10**9


def solve_astar(
    observation: Observation,
    *,
    max_expanded_states: int = 100_000,
) -> tuple[Action, ...]:
    """Return a primitive-action plan found by push-based A*."""

    return solve_astar_result(
        observation,
        max_expanded_states=max_expanded_states,
    ).actions


def solve_astar_result(
    observation: Observation,
    *,
    max_expanded_states: int = 100_000,
) -> SearchResult:
    """Search push transitions using a box-to-target matching heuristic."""

    started_at = perf_counter()
    if max_expanded_states <= 0:
        raise ValueError("max_expanded_states must be positive")
    level, initial = decode_observation(observation)
    if is_success(level, initial):
        return SearchResult((), 0, perf_counter() - started_at)

    pull_distances = _target_pull_distances(level)
    if _matching_heuristic(initial.boxes, level.targets, pull_distances) >= _INF:
        raise NoSolutionError("a box cannot reach any target")

    initial_reachable, _ = _reachable_paths(level, initial)
    initial_key = (initial.boxes, min(initial_reachable))
    states: dict[_StateKey, SokobanState] = {initial_key: initial}
    parents: dict[_StateKey, _Parent] = {initial_key: None}
    best_cost: dict[_StateKey, int] = {initial_key: 0}
    order = count()
    frontier: list[tuple[int, int, int, _StateKey]] = []
    initial_h = _matching_heuristic(
        initial.boxes, level.targets, pull_distances
    )
    heappush(frontier, (initial_h, 0, next(order), initial_key))
    expanded_states = 0

    while frontier:
        _, cost, _, key = heappop(frontier)
        if cost != best_cost.get(key):
            continue
        if expanded_states >= max_expanded_states:
            raise SearchLimitError(
                f"A* reached the {max_expanded_states} state expansion limit"
            )
        state = states[key]
        expanded_states += 1
        reachable, paths = _reachable_paths(level, state)

        for box in sorted(state.boxes):
            for action, delta in _DIRECTIONS:
                destination = _add(box, delta)
                support = _subtract(box, delta)
                if support not in reachable:
                    continue
                if destination in level.walls or destination in state.boxes:
                    continue
                boxes = frozenset((*state.boxes - {box}, destination))
                if _has_deadlock(level, boxes, pull_distances):
                    continue
                segment = (*paths[support], action)
                next_state = SokobanState(player=box, boxes=boxes)
                if is_success(level, next_state):
                    parents_key = _key_for(level, next_state)
                    parents[parents_key] = (key, segment)
                    return SearchResult(
                        _restore_plan(parents, parents_key),
                        expanded_states,
                        perf_counter() - started_at,
                    )

                next_reachable, _ = _reachable_paths(level, next_state)
                next_key = (boxes, min(next_reachable))
                next_cost = cost + len(segment)
                if next_cost >= best_cost.get(next_key, _INF):
                    continue
                heuristic = _matching_heuristic(
                    boxes, level.targets, pull_distances
                )
                if heuristic >= _INF:
                    continue
                states[next_key] = next_state
                parents[next_key] = (key, segment)
                best_cost[next_key] = next_cost
                heappush(
                    frontier,
                    (
                        next_cost + heuristic,
                        next_cost,
                        next(order),
                        next_key,
                    ),
                )

    raise NoSolutionError("A* exhausted every reachable push state")


class AStarPlanner:
    """Produce a complete grounded plan using push-based A*."""

    def __init__(self, *, max_expanded_states: int = 100_000) -> None:
        if max_expanded_states <= 0:
            raise ValueError("max_expanded_states must be positive")
        self.max_expanded_states = max_expanded_states

    @property
    def name(self) -> str:
        return "graph:astar"

    def reset(self, *, seed: int | None = None) -> None:
        del seed

    def plan(self, context: PlanningContext) -> PlanningOutcome:
        started_at = perf_counter()
        try:
            result = solve_astar_result(
                context.observation,
                max_expanded_states=self.max_expanded_states,
            )
        except (NoSolutionError, SearchLimitError) as error:
            elapsed = perf_counter() - started_at
            return PlanningOutcome(
                error=str(error),
                error_kind="search",
                algorithm_calls=1,
                algorithm_requests=1,
                algorithm_failures=1,
                algorithm_elapsed_seconds=elapsed,
                elapsed_seconds=elapsed,
            )
        return PlanningOutcome(
            actions=result.actions,
            algorithm_calls=1,
            algorithm_requests=1,
            algorithm_expanded_states=result.expanded_states,
            algorithm_elapsed_seconds=result.elapsed_seconds,
            elapsed_seconds=perf_counter() - started_at,
        )


def _reachable_paths(
    level: SokobanLevel,
    state: SokobanState,
) -> tuple[set[Position], dict[Position, tuple[Action, ...]]]:
    reachable = {state.player}
    paths: dict[Position, tuple[Action, ...]] = {state.player: ()}
    frontier = deque([state.player])
    while frontier:
        position = frontier.popleft()
        for action, delta in _DIRECTIONS:
            destination = _add(position, delta)
            if (
                destination in reachable
                or destination in level.walls
                or destination in state.boxes
            ):
                continue
            reachable.add(destination)
            paths[destination] = (*paths[position], action)
            frontier.append(destination)
    return reachable, paths


def _target_pull_distances(
    level: SokobanLevel,
) -> dict[Position, dict[Position, int]]:
    result: dict[Position, dict[Position, int]] = {}
    for target in level.targets:
        distances = {target: 0}
        frontier = deque([target])
        while frontier:
            box = frontier.popleft()
            for _, delta in _DIRECTIONS:
                predecessor = _subtract(box, delta)
                support = _subtract(predecessor, delta)
                if (
                    predecessor in distances
                    or predecessor in level.walls
                    or support in level.walls
                    or not _inside(level, predecessor)
                    or not _inside(level, support)
                ):
                    continue
                distances[predecessor] = distances[box] + 1
                frontier.append(predecessor)
        result[target] = distances
    return result


def _matching_heuristic(
    boxes: frozenset[Position],
    targets: frozenset[Position],
    pull_distances: dict[Position, dict[Position, int]],
) -> int:
    ordered_boxes = sorted(boxes)
    ordered_targets = sorted(targets)
    costs = [
        [pull_distances[target].get(box, _INF) for target in ordered_targets]
        for box in ordered_boxes
    ]
    best: dict[int, int] = {0: 0}
    for row in costs:
        next_best: dict[int, int] = {}
        for mask, current in best.items():
            for index, distance in enumerate(row):
                bit = 1 << index
                if mask & bit or distance >= _INF:
                    continue
                candidate = current + distance
                next_mask = mask | bit
                next_best[next_mask] = min(
                    candidate, next_best.get(next_mask, _INF)
                )
        best = next_best
    return min(best.values(), default=_INF)


def _has_deadlock(
    level: SokobanLevel,
    boxes: frozenset[Position],
    pull_distances: dict[Position, dict[Position, int]],
) -> bool:
    reachable_box_cells = {
        cell for distances in pull_distances.values() for cell in distances
    }
    if any(
        box not in level.targets and box not in reachable_box_cells
        for box in boxes
    ):
        return True
    for row in range(level.height - 1):
        for column in range(level.width - 1):
            square = {
                (row, column),
                (row + 1, column),
                (row, column + 1),
                (row + 1, column + 1),
            }
            if square <= level.walls | boxes and (square & boxes) - level.targets:
                return True
    return False


def _key_for(level: SokobanLevel, state: SokobanState) -> _StateKey:
    reachable, _ = _reachable_paths(level, state)
    return state.boxes, min(reachable)


def _restore_plan(
    parents: dict[_StateKey, _Parent],
    goal: _StateKey,
) -> tuple[Action, ...]:
    segments: list[tuple[Action, ...]] = []
    key = goal
    while parents[key] is not None:
        parent = parents[key]
        if parent is None:
            break
        key, segment = parent
        segments.append(segment)
    return tuple(action for segment in reversed(segments) for action in segment)


def _add(position: Position, delta: Position) -> Position:
    return position[0] + delta[0], position[1] + delta[1]


def _subtract(position: Position, delta: Position) -> Position:
    return position[0] - delta[0], position[1] - delta[1]


def _inside(level: SokobanLevel, position: Position) -> bool:
    return (
        0 <= position[0] < level.height
        and 0 <= position[1] < level.width
    )
