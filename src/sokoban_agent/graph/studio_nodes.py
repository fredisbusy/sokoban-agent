"""Nodes and routing for the LangGraph Studio Sokoban graph."""

from __future__ import annotations

from sokoban_agent.env import Action, SokobanEnv
from sokoban_agent.env.rules import (
    apply_action,
    decode_observation,
    has_static_corner_deadlock,
    is_success,
    observation_for,
)
from sokoban_agent.graph.studio_state import (
    Route,
    StudioInput,
    StudioState,
    decision_event,
    guard_update,
    observation_from_state,
)
from sokoban_agent.planning import (
    LLMPlanner,
    NoSolutionError,
    PlanningContext,
    SearchLimitError,
    solve_astar_result,
)
from sokoban_agent.planning.llm import LiteLLMClient, OllamaSettings
from sokoban_agent.planning.llm_planner import serialize_board


def initialize(state: StudioInput) -> StudioState:
    """Create a JSON-safe starting board from the requested level."""

    level_id = state.get("level_id", "tiny-push")
    seed = state.get("seed", 0)
    max_steps = state.get("max_steps", 15)
    env = SokobanEnv(max_steps=max_steps)
    try:
        observation, info = env.reset(
            seed=seed,
            options={"level_id": level_id},
        )
    finally:
        env.close()
    board = serialize_board(observation)
    return {
        "level_id": str(info["level_id"]),
        "seed": seed,
        "max_steps": max_steps,
        "observation": observation.tolist(),
        "board": board,
        "plan": [],
        "proposed_plan": [],
        "action_history": [],
        "feedback": [],
        "planning_attempts": 0,
        "action_count": 0,
        "success": False,
        "deadlock": False,
        "truncated": False,
        "status": "계획 준비",
        "planner_goal": "",
        "decision_summary": "",
        "risk": "",
        "guard_summary": "",
        "validation_summary": "",
        "execution_summary": "",
        "failure_reason": None,
        "decision_log": [
            decision_event(
                0,
                "초기화",
                f"{info['level_id']} 레벨을 시작했습니다",
                {"board": board, "seed": seed, "max_steps": max_steps},
            )
        ],
        "llm_calls": 0,
        "llm_elapsed_seconds": 0.0,
        "llm_prompt_tokens": 0,
        "llm_output_tokens": 0,
        "algorithm_calls": 0,
        "algorithm_fallbacks": 0,
        "algorithm_expanded_states": 0,
        "algorithm_elapsed_seconds": 0.0,
    }


def llm_plan(state: StudioState) -> StudioState:
    """Ask the configured model for actions and a Korean decision summary."""

    settings = OllamaSettings.from_env()
    planner = LLMPlanner(LiteLLMClient(settings), model_name=settings.model)
    planner.reset(seed=state["seed"])
    outcome = planner.plan(
        PlanningContext(
            observation=observation_from_state(state),
            info={"level_id": state["level_id"]},
            action_history=tuple(
                Action[name] for name in state["action_history"]
            ),
            feedback=tuple(state["feedback"]),
            seed=state["seed"],
        )
    )
    attempts = state["planning_attempts"] + 1
    proposed = [action.name for action in outcome.actions]
    error = outcome.error
    exhausted = error is not None and attempts >= 3
    summary = outcome.decision_summary or error or "판단 요약이 없습니다"
    details: dict[str, object] = {
        "goal": outcome.goal or "",
        "risk": outcome.risk or "",
        "proposed_actions": proposed,
        "error": error,
    }
    return {
        "plan": proposed,
        "proposed_plan": proposed,
        "planner_goal": outcome.goal or "",
        "decision_summary": summary,
        "risk": outcome.risk or "",
        "planning_attempts": attempts,
        "failure_reason": error if exhausted else None,
        "feedback": [
            *state["feedback"],
            *([error] if error is not None else []),
        ],
        "status": "LLM 계획 완료" if proposed else "LLM 계획 실패",
        "decision_log": [
            *state["decision_log"],
            decision_event(
                state["action_count"],
                "LLM 계획",
                summary,
                details,
            ),
        ],
        "llm_calls": state["llm_calls"] + outcome.llm_calls,
        "llm_elapsed_seconds": (
            state["llm_elapsed_seconds"] + outcome.llm_elapsed_seconds
        ),
        "llm_prompt_tokens": (
            state["llm_prompt_tokens"] + outcome.llm_prompt_tokens
        ),
        "llm_output_tokens": (
            state["llm_output_tokens"] + outcome.llm_output_tokens
        ),
    }


def astar_guard(state: StudioState) -> StudioState:
    """Check the model plan and append or substitute an A* solution."""

    observation = observation_from_state(state)
    level, board = decode_observation(observation)
    accepted: list[Action] = []
    reason: str | None = None
    solved = False
    for index, name in enumerate(state["plan"]):
        action = Action[name]
        move = apply_action(level, board, action)
        if move.invalid_move:
            reason = f"{index + 1}번째 행동 {name}이 막혀 있습니다"
            break
        if has_static_corner_deadlock(level, move.state):
            reason = f"{index + 1}번째 행동 {name}이 데드락을 만듭니다"
            break
        board = move.state
        accepted.append(action)
        if is_success(level, board):
            solved = True
            break

    if solved:
        return guard_update(
            state,
            accepted,
            "LLM 계획만으로 퍼즐을 해결할 수 있습니다",
            fallback=False,
        )

    if reason is None:
        try:
            suffix = solve_astar_result(observation_for(level, board))
        except (NoSolutionError, SearchLimitError):
            reason = "LLM 계획 이후 상태에서 A*가 해답을 찾지 못했습니다"
        else:
            return guard_update(
                state,
                [*accepted, *suffix.actions],
                f"LLM 계획이 안전하여 A*가 후속 행동 "
                f"{len(suffix.actions)}개를 보강했습니다",
                fallback=False,
                expanded_states=suffix.expanded_states,
                elapsed_seconds=suffix.elapsed_seconds,
            )

    try:
        replacement = solve_astar_result(observation)
    except (NoSolutionError, SearchLimitError) as error:
        summary = f"{reason}. A* 대체 계획도 실패했습니다: {error}"
        return {
            "plan": [],
            "guard_summary": summary,
            "failure_reason": summary,
            "status": "A* 검사 실패",
            "decision_log": [
                *state["decision_log"],
                decision_event(
                    state["action_count"],
                    "A* 검사",
                    summary,
                    {},
                ),
            ],
            "algorithm_calls": state["algorithm_calls"] + 1,
            "algorithm_fallbacks": state["algorithm_fallbacks"] + 1,
        }
    return guard_update(
        state,
        list(replacement.actions),
        f"{reason}. A*가 안전한 전체 계획 "
        f"{len(replacement.actions)}개 행동으로 대체했습니다",
        fallback=True,
        expanded_states=replacement.expanded_states,
        elapsed_seconds=replacement.elapsed_seconds,
    )


def validate_plan(state: StudioState) -> StudioState:
    """Validate the grounded plan before any environment transition."""

    level, board = decode_observation(observation_from_state(state))
    validated: list[Action] = []
    message = "전체 계획이 유효합니다"
    for index, name in enumerate(state["plan"]):
        action = Action[name]
        move = apply_action(level, board, action)
        if move.invalid_move:
            message = f"{index + 1}번째 행동 {name}이 막혀 있습니다"
            validated = []
            break
        if has_static_corner_deadlock(level, move.state):
            message = f"{index + 1}번째 행동 {name}이 데드락을 만듭니다"
            validated = []
            break
        board = move.state
        validated.append(action)
        if is_success(level, board):
            message = "계획이 유효하며 실행 중 퍼즐을 해결합니다"
            break
    plan = [action.name for action in validated]
    return {
        "plan": plan,
        "validation_summary": message,
        "status": "계획 검증 완료" if plan else "계획 검증 실패",
        "feedback": state["feedback"] if plan else [*state["feedback"], message],
        "failure_reason": None,
        "decision_log": [
            *state["decision_log"],
            decision_event(
                state["action_count"],
                "계획 검증",
                message,
                {"validated_actions": plan},
            ),
        ],
    }


def execute_action(state: StudioState) -> StudioState:
    """Execute one validated action and expose the resulting board."""

    action = Action[state["plan"][0]]
    level, board = decode_observation(observation_from_state(state))
    move = apply_action(level, board, action)
    next_observation = observation_for(level, move.state)
    action_count = state["action_count"] + 1
    success = is_success(level, move.state)
    deadlock = has_static_corner_deadlock(level, move.state)
    truncated = action_count >= state["max_steps"] and not success
    remaining = state["plan"][1:]
    if success:
        status = "성공"
    elif deadlock:
        status = "데드락"
    elif truncated:
        status = "행동 제한 도달"
    else:
        status = "행동 실행"
    summary = (
        f"{action.name} 행동을 실행했습니다. "
        f"남은 계획은 {len(remaining)}개입니다"
    )
    return {
        "observation": next_observation.tolist(),
        "board": serialize_board(next_observation),
        "plan": remaining,
        "action_history": [*state["action_history"], action.name],
        "action_count": action_count,
        "planning_attempts": 0,
        "success": success,
        "deadlock": deadlock,
        "truncated": truncated,
        "status": status,
        "execution_summary": summary,
        "decision_log": [
            *state["decision_log"],
            decision_event(
                action_count,
                "행동 실행",
                summary,
                {"action": action.name, "board": serialize_board(next_observation)},
            ),
        ],
    }


def route_after_llm(state: StudioState) -> Route:
    if state.get("plan"):
        return "astar_guard"
    if state.get("failure_reason") is not None:
        return "end"
    return "llm_plan"


def route_after_guard(state: StudioState) -> Route:
    return "validate" if state.get("plan") else "end"


def route_after_validation(state: StudioState) -> Route:
    if state.get("plan"):
        return "execute"
    if state.get("planning_attempts", 0) >= 3:
        return "end"
    return "llm_plan"


def route_after_execution(state: StudioState) -> Route:
    if state.get("success") or state.get("deadlock") or state.get("truncated"):
        return "end"
    return "execute" if state.get("plan") else "llm_plan"
