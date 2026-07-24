"""Execution, reflection, and repetition nodes for the agentic graph."""

from __future__ import annotations

import json
from typing import Literal, cast

import numpy as np

from sokoban_agent.env import Action
from sokoban_agent.env.rules import (
    apply_action,
    decode_observation,
    has_static_corner_deadlock,
    is_success,
    observation_for,
)
from sokoban_agent.graph.agentic_metrics import update_agentic_metrics
from sokoban_agent.graph.agentic_state import AgenticState
from sokoban_agent.planning.base import Observation
from sokoban_agent.planning.strategy import BoardAnalysis, ExpectedEffect
from sokoban_agent.planning.strategy_decision import compact_board_analysis

ExecutionRoute = Literal["observe", "__end__"]
ActionRoute = Literal["execute_until_push", "reflect"]
RepetitionRoute = Literal["ground_subgoal", "__end__"]


def detect_agentic_repetition(state: AgenticState) -> dict[str, object]:
    """Detect repetition at the same abstraction shown to the planner."""

    analysis = BoardAnalysis.model_validate(state["board_analysis"])
    key = json.dumps(
        {
            "board": compact_board_analysis(analysis),
            "subgoal": state["active_subgoal"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if key in state["attempt_keys"]:
        feedback = "동일한 보드와 하위 목표가 반복되었습니다"
        return {
            "cycle_detected": True,
            "status": "cycle_detected",
            "attempt_keys": state["attempt_keys"],
            "feedback": [f"cycle_detected: {feedback}"],
            "decision_events": [
                {
                    "step": _step(state),
                    "stage": "detect_repetition",
                    "summary": feedback,
                }
            ],
        }
    return {
        "cycle_detected": False,
        "status": "repetition_checked",
        "attempt_keys": [*state["attempt_keys"], key],
        "decision_events": [
            {
                "step": _step(state),
                "stage": "detect_repetition",
                "summary": "새 보드와 하위 목표 조합을 승인했습니다",
            }
        ],
    }


def route_after_repetition(state: AgenticState) -> RepetitionRoute:
    """Execute only a decision that has not already been attempted."""

    return "__end__" if state["cycle_detected"] else "ground_subgoal"


def execute_agentic_until_push(state: AgenticState) -> dict[str, object]:
    """Apply one grounded action so every primitive move is stream-visible."""

    observation = cast(
        Observation,
        np.asarray(state["observation"], dtype=np.uint8),
    )
    level, board = decode_observation(observation)
    steps = _step(state)
    requested = list(state["grounded_actions"])
    prior = state["execution_result"]
    if prior is None:
        before_boxes = _boxes(board.boxes)
        executed: list[str] = []
        push_count = 0
    else:
        before_boxes = cast(
            list[list[int]],
            prior["before_boxes"],
        )
        executed = list(
            cast(list[str], prior["actions_executed"])
        )
        push_count = cast(int, prior["push_count"])

    invalid_move = False
    action_name: str | None = None
    pushed = False
    if steps < state["max_steps"] and len(executed) < len(requested):
        action_name = requested[len(executed)]
        move = apply_action(level, board, Action[action_name])
        board = move.state
        steps += 1
        executed.append(action_name)
        invalid_move = move.invalid_move
        if move.pushed:
            push_count += 1
            pushed = True

    success = is_success(level, board)
    deadlock = not success and has_static_corner_deadlock(level, board)
    truncated = steps >= state["max_steps"] and not success and not deadlock
    next_observation = observation_for(level, board)
    info = {
        **state["info"],
        "steps": steps,
        "invalid_move": invalid_move,
        "pushed": pushed,
        "boxes_on_targets": len(board.boxes & level.targets),
        "success": success,
        "deadlock": deadlock,
    }
    result = {
        "before_boxes": before_boxes,
        "after_boxes": _boxes(board.boxes),
        "actions_requested": requested,
        "actions_executed": executed,
        "push_count": push_count,
        "invalid_move": invalid_move,
        "truncated": truncated,
    }
    return {
        "observation": next_observation.tolist(),
        "info": info,
        "action_history": [
            *state["action_history"],
            *([action_name] if action_name is not None else []),
        ],
        "push_count": state["push_count"] + int(pushed),
        "execution_result": result,
        "status": "executed_until_push",
        "decision_events": [
            {
                "step": steps,
                "stage": "execute_until_push",
                "summary": (
                    f"{action_name} 행동을 실행했습니다"
                    if action_name is not None
                    else "행동 한도 때문에 실행하지 못했습니다"
                ),
                **(
                    {"action": action_name, "pushed": pushed}
                    if action_name is not None
                    else {}
                ),
            }
        ],
    }


def route_after_action(state: AgenticState) -> ActionRoute:
    """Repeat the action node until its first push or a terminal condition."""

    execution = cast(dict[str, object], state["execution_result"])
    executed = cast(list[str], execution["actions_executed"])
    requested = cast(list[str], execution["actions_requested"])
    if (
        cast(int, execution["push_count"]) == 1
        or cast(bool, execution["invalid_move"])
        or cast(bool, execution["truncated"])
        or state["info"].get("success") is True
        or state["info"].get("deadlock") is True
        or len(executed) >= len(requested)
    ):
        return "reflect"
    return "execute_until_push"


def reflect_agentic_execution(state: AgenticState) -> dict[str, object]:
    """Compare the expected box effect with the bounded execution result."""

    effect = ExpectedEffect.model_validate(state["expected_effect"])
    execution = cast(dict[str, object], state["execution_result"])
    before = {
        tuple(item)
        for item in cast(list[list[int]], execution["before_boxes"])
    }
    after = {
        tuple(item)
        for item in cast(list[list[int]], execution["after_boxes"])
    }
    expected_from = (effect.from_position.row, effect.from_position.col)
    expected_to = (effect.to_position.row, effect.to_position.col)
    matched = (
        cast(int, execution["push_count"]) == 1
        and expected_from in before
        and expected_from not in after
        and expected_to in after
    )
    reflection = {
        "matched": matched,
        "box_id": effect.box_id,
        "expected_from": list(expected_from),
        "expected_to": list(expected_to),
    }
    success = state["info"].get("success") is True
    deadlock = state["info"].get("deadlock") is True
    truncated = cast(bool, execution["truncated"])

    if matched:
        status = "success" if success else "subgoal_completed"
        completed = [*state["completed_subgoals"], state["active_subgoal"]]
        metrics = state["metrics"]
        return {
            "reflection_result": reflection,
            "completed_subgoals": completed,
            "strategy_attempts": 0,
            "metrics": update_agentic_metrics(
                metrics,
                strategy={
                    "effect_matches": (
                        metrics["strategy"]["effect_matches"] + 1
                    )
                },
            ),
            "status": status,
            "decision_events": [
                {
                    "step": _step(state),
                    "stage": "reflect",
                    "summary": (
                        "퍼즐을 해결했습니다"
                        if success
                        else "예상한 push 효과를 확인했습니다"
                    ),
                }
            ],
        }

    if truncated:
        status = "step_limit"
    elif deadlock:
        status = "deadlock"
    else:
        status = "reflection_failed"
    evidence = (
        f"{effect.box_id}이 예상 위치 {expected_to}로 이동하지 않았습니다"
    )
    revision = {
        "step": _step(state),
        "disposition": "modify",
        "changed_fields": ["subgoal", "expected_effect"],
        "evidence": evidence,
    }
    metrics = state["metrics"]
    return {
        "reflection_result": reflection,
        "plan_revisions": [revision],
        "metrics": update_agentic_metrics(
            metrics,
            strategy={
                "effect_mismatches": (
                    metrics["strategy"]["effect_mismatches"] + 1
                )
            },
        ),
        "feedback": [f"unexpected_state: {evidence}"],
        "latest_strategy_feedback": [f"unexpected_state: {evidence}"],
        "status": status,
        "decision_events": [
            {
                "step": _step(state),
                "stage": "reflect",
                "summary": evidence,
            }
        ],
    }


def route_after_reflection(state: AgenticState) -> ExecutionRoute:
    """Continue observing only after a non-terminal reflection."""

    if state["status"] in {"success", "deadlock", "step_limit"}:
        return "__end__"
    return "observe"


def observe_agentic_state(state: AgenticState) -> dict[str, object]:
    """Checkpoint the post-execution observation before fresh analysis."""

    return {
        "status": "observed",
        "grounded_plan": None,
        "grounded_actions": [],
        "grounding_failure": None,
        "execution_result": None,
        "decision_events": [
            {
                "step": _step(state),
                "stage": "observe",
                "summary": "실행 후 보드를 다시 관찰했습니다",
            }
        ],
    }


def _step(state: AgenticState) -> int:
    steps = state["info"].get("steps")
    if not isinstance(steps, int):
        raise TypeError("graph info steps must be an integer")
    return steps


def _boxes(boxes: frozenset[tuple[int, int]]) -> list[list[int]]:
    return [[row, col] for row, col in sorted(boxes)]
