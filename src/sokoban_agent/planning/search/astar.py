"""Push-based A* search for Sokoban."""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import count
from time import perf_counter

from sokoban_agent.env import Action, SokobanState
from sokoban_agent.env.levels import Position, SokobanLevel
from sokoban_agent.env.rules import decode_observation, is_success
from sokoban_agent.planning.contracts import (
    AlgorithmPlanningMetrics,
    NoSolutionError,
    Observation,
    PlanningContext,
    PlanningFailure,
    PlanningOutcome,
    SearchLimitError,
    SearchResult,
)
from sokoban_agent.planning.search.spatial import (
    DIRECTIONS,
    add_position,
    reachable_paths,
    subtract_position,
    target_pull_distances,
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

    pull_distances = target_pull_distances(level)
    if _matching_heuristic(initial.boxes, level.targets, pull_distances) >= _INF:
        raise NoSolutionError("a box cannot reach any target")

    initial_reachable, _ = reachable_paths(level, initial)
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
        reachable, paths = reachable_paths(level, state)

        for box in sorted(state.boxes):
            for action, delta in DIRECTIONS:
                destination = add_position(box, delta)
                support = subtract_position(box, delta)
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

                next_reachable, _ = reachable_paths(level, next_state)
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
                failure=PlanningFailure(str(error), "search"),
                algorithm=AlgorithmPlanningMetrics(
                    calls=1,
                    requests=1,
                    failures=1,
                    elapsed_seconds=elapsed,
                ),
                elapsed_seconds=elapsed,
            )
        return PlanningOutcome(
            actions=result.actions,
            algorithm=AlgorithmPlanningMetrics(
                calls=1,
                requests=1,
                expanded_states=result.expanded_states,
                elapsed_seconds=result.elapsed_seconds,
            ),
            elapsed_seconds=perf_counter() - started_at,
        )


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
    reachable, _ = reachable_paths(level, state)
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
