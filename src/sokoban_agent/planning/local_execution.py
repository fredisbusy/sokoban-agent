"""Bounded grounding for one validated Sokoban push subgoal."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from sokoban_agent.env import Action, SokobanState
from sokoban_agent.env.levels import SokobanLevel
from sokoban_agent.env.rules import apply_action, decode_observation
from sokoban_agent.planning.base import Observation
from sokoban_agent.planning.spatial import reachable_paths
from sokoban_agent.planning.strategy import (
    BoardAnalysis,
    Cell,
    Direction,
    FailureKind,
    GroundedPushPlan,
    ProtectedConstraint,
    PushOption,
    PushSubgoal,
)


class SubgoalGroundingError(ValueError):
    """Structured rejection raised before any graph environment transition."""

    def __init__(self, kind: FailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def ground_push_subgoal(
    observation: Observation,
    analysis: BoardAnalysis,
    subgoal: PushSubgoal,
    protected_constraints: Sequence[ProtectedConstraint],
) -> GroundedPushPlan:
    """Ground player movement only as far as one explicitly selected push."""

    level, state, option = _validate_subgoal(
        observation,
        analysis,
        subgoal,
        protected_constraints,
    )
    reachable, paths = reachable_paths(level, state)
    support = _position(option.support)
    try:
        player_path = paths[support]
    except KeyError as error:
        raise SubgoalGroundingError(
            "support_unreachable",
            "플레이어가 push 지지 칸에 도달할 수 없습니다",
        ) from error

    current = state
    for action in player_path:
        move = apply_action(level, current, action)
        if move.invalid_move or move.pushed:
            raise SubgoalGroundingError(
                "unexpected_state",
                "지지 칸 경로가 현재 관찰과 일치하지 않습니다",
            )
        current = move.state

    _validate_push(level, current, subgoal)
    return GroundedPushPlan(
        box_id=subgoal.box_id,
        support=option.support,
        player_actions=tuple(
            cast(Direction, action.name) for action in player_path
        ),
        push_action=subgoal.direction,
        expanded_player_states=len(reachable),
    )


def ground_push_subgoal_direct(
    observation: Observation,
    analysis: BoardAnalysis,
    subgoal: PushSubgoal,
    protected_constraints: Sequence[ProtectedConstraint],
) -> GroundedPushPlan:
    """Ground only a push whose support already contains the player."""

    level, state, option = _validate_subgoal(
        observation,
        analysis,
        subgoal,
        protected_constraints,
    )
    if state.player != _position(option.support):
        raise SubgoalGroundingError(
            "support_unreachable",
            "플레이어가 현재 push 지지 칸에 있지 않습니다",
        )
    _validate_push(level, state, subgoal)
    return GroundedPushPlan(
        box_id=subgoal.box_id,
        support=option.support,
        player_actions=(),
        push_action=subgoal.direction,
        expanded_player_states=1,
    )


def _validate_subgoal(
    observation: Observation,
    analysis: BoardAnalysis,
    subgoal: PushSubgoal,
    protected_constraints: Sequence[ProtectedConstraint],
) -> tuple[SokobanLevel, SokobanState, PushOption]:
    level, state = decode_observation(observation)
    box = next(
        (fact for fact in analysis.boxes if fact.box_id == subgoal.box_id),
        None,
    )
    if box is None or _position(box.position) not in state.boxes:
        raise SubgoalGroundingError(
            "unexpected_state",
            "하위 목표의 상자가 현재 관찰과 일치하지 않습니다",
        )

    option = next(
        (
            candidate
            for candidate in analysis.push_options
            if candidate.box_id == subgoal.box_id
            and candidate.direction == subgoal.direction
            and candidate.destination == subgoal.destination
        ),
        None,
    )
    if option is None:
        destination = _position(subgoal.destination)
        kind: FailureKind = (
            "destination_blocked"
            if destination in level.walls or destination in state.boxes
            else "support_unreachable"
        )
        raise SubgoalGroundingError(
            kind,
            "현재 관찰에서 선택한 push의 사전 조건을 만족할 수 없습니다",
        )
    if option.creates_static_deadlock:
        raise SubgoalGroundingError(
            "static_deadlock",
            "선택한 push가 정적 dead square에 상자를 둡니다",
        )

    protected_cells = {
        _position(cell)
        for constraint in protected_constraints
        for cell in constraint.cells
    }
    if _position(option.destination) in protected_cells:
        raise SubgoalGroundingError(
            "protected_constraint_violated",
            "선택한 push가 보호 칸을 점유합니다",
        )
    return level, state, option


def _validate_push(
    level: SokobanLevel,
    state: SokobanState,
    subgoal: PushSubgoal,
) -> None:
    push_action = Action[subgoal.direction]
    pushed = apply_action(level, state, push_action)
    if (
        pushed.invalid_move
        or not pushed.pushed
        or _position(subgoal.destination) not in pushed.state.boxes
    ):
        raise SubgoalGroundingError(
            "unexpected_state",
            "마지막 행동이 의도한 상자를 push하지 못합니다",
        )


def _position(cell: Cell) -> tuple[int, int]:
    return cell.row, cell.col
