"""Compact model boundary for one strategy decision."""

from __future__ import annotations

from typing import cast

from pydantic import Field

from sokoban_agent.planning.strategy import (
    BoardAnalysis,
    BoxTargetAssignment,
    Cell,
    Direction,
    ExpectedEffect,
    FailureCondition,
    ProtectedConstraint,
    PushSubgoal,
    StrategyHypothesis,
    StrategyModel,
)


class StrategyDecision(StrategyModel):
    """Small model-authored patch expanded into a verified strategy artifact."""

    summary: str = Field(min_length=1)
    push_id: str = Field(pattern=r"^B[1-9][0-9]*:(UP|RIGHT|DOWN|LEFT)$")
    target_id: str = Field(pattern=r"^T[1-9][0-9]*$")
    protected_cells: tuple[Cell, ...]
    risk: str = Field(min_length=1)


def compact_board_analysis(analysis: BoardAnalysis) -> dict[str, object]:
    """Project deterministic facts down to the current safe-push decision."""

    distances: dict[str, dict[str, int | None]] = {
        box.box_id: {} for box in analysis.boxes
    }
    for item in analysis.reverse_pull_distances:
        distances[item.box_id][item.target_id] = item.distance
    return {
        "boxes": {
            box.box_id: [box.position.row, box.position.col]
            for box in analysis.boxes
        },
        "targets": {
            target.target_id: [target.position.row, target.position.col]
            for target in analysis.targets
        },
        "safe_push_options": [
            {
                "push_id": f"{option.box_id}:{option.direction}",
                "support": option.support.model_dump(mode="json"),
                "destination": option.destination.model_dump(mode="json"),
            }
            for option in analysis.push_options
            if not option.creates_static_deadlock
        ],
        "reverse_pull_distances": distances,
    }


def without_immediate_reverse(
    context: dict[str, object],
    previous_push_id: str | None,
) -> dict[str, object]:
    """Hide the one candidate that deterministically undoes the last push."""

    if previous_push_id is None:
        return context
    box_id, direction = previous_push_id.split(":")
    opposite = {
        "UP": "DOWN",
        "RIGHT": "LEFT",
        "DOWN": "UP",
        "LEFT": "RIGHT",
    }[direction]
    forbidden = f"{box_id}:{opposite}"
    options = context.get("safe_push_options")
    if not isinstance(options, list):
        raise TypeError("compact context safe_push_options must be a list")
    return {
        **context,
        "safe_push_options": [
            option
            for option in options
            if isinstance(option, dict) and option.get("push_id") != forbidden
        ],
    }


def materialize_strategy(
    analysis: BoardAnalysis,
    decision: StrategyDecision,
) -> StrategyHypothesis:
    """Fill deterministic fields without asking the model to repeat them."""

    box_id, direction_value = decision.push_id.split(":")
    direction = cast(Direction, direction_value)
    positions = {box.box_id: box.position for box in analysis.boxes}
    origin = positions.get(box_id, Cell(row=1, col=1))
    row_delta, col_delta = {
        "UP": (-1, 0),
        "RIGHT": (0, 1),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
    }[direction]
    destination = Cell(
        row=origin.row + row_delta,
        col=origin.col + col_delta,
    )
    summary = decision.summary[:240]
    protected = (
        (
            ProtectedConstraint(
                kind="keep_clear",
                cells=decision.protected_cells,
                reason=summary[:160],
            ),
        )
        if decision.protected_cells
        else ()
    )
    return StrategyHypothesis(
        summary=summary,
        assignments=(
            BoxTargetAssignment(
                box_id=box_id,
                target_id=decision.target_id,
                reason=summary[:160],
            ),
        ),
        protected_constraints=protected,
        subgoal=PushSubgoal(
            kind="push",
            box_id=box_id,
            target_id=decision.target_id,
            direction=direction,
            destination=destination,
        ),
        expected_effect=ExpectedEffect(
            box_id=box_id,
            from_position=origin,
            to_position=destination,
        ),
        failure_conditions=(
            FailureCondition(
                kind="unexpected_state",
                description=decision.risk[:200],
            ),
        ),
    )
