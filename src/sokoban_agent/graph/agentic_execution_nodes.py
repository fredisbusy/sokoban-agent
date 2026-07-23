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
from sokoban_agent.graph.agentic_state import AgenticState
from sokoban_agent.planning.base import Observation
from sokoban_agent.planning.strategy import ExpectedEffect

ExecutionRoute = Literal["observe", "__end__"]
RepetitionRoute = Literal["ground_subgoal", "__end__"]


def detect_agentic_repetition(state: AgenticState) -> dict[str, object]:
    """Terminate a repeated board-and-subgoal decision before execution."""

    key = json.dumps(
        {
            "observation": state["observation"],
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
    """Apply grounded actions and stop at the first push or step limit."""

    observation = cast(
        Observation,
        np.asarray(state["observation"], dtype=np.uint8),
    )
    level, board = decode_observation(observation)
    before_boxes = _boxes(board.boxes)
    steps = _step(state)
    requested = list(state["grounded_actions"])
    executed: list[str] = []
    invalid_move = False
    push_count = 0

    for action_name in requested:
        if steps >= state["max_steps"]:
            break
        move = apply_action(level, board, Action[action_name])
        board = move.state
        steps += 1
        executed.append(action_name)
        invalid_move = move.invalid_move
        if move.pushed:
            push_count += 1
            break
        if move.invalid_move:
            break

    success = is_success(level, board)
    deadlock = not success and has_static_corner_deadlock(level, board)
    truncated = steps >= state["max_steps"] and not success and not deadlock
    next_observation = observation_for(level, board)
    info = {
        **state["info"],
        "steps": steps,
        "invalid_move": invalid_move,
        "pushed": push_count == 1,
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
        "action_history": [*state["action_history"], *executed],
        "execution_result": result,
        "status": "executed_until_push",
        "decision_events": [
            {
                "step": steps,
                "stage": "execute_until_push",
                "summary": (
                    f"행동 {len(executed)}개를 실행하고 "
                    f"push {push_count}회에서 멈췄습니다"
                ),
            }
        ],
    }


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
        return {
            "reflection_result": reflection,
            "completed_subgoals": completed,
            "strategy_attempts": 0,
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
    return {
        "reflection_result": reflection,
        "plan_revisions": [revision],
        "feedback": [f"unexpected_state: {evidence}"],
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
