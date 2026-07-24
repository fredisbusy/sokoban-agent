"""LangGraph-first Sokoban baseline with checkpoint-owned execution state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from sokoban_agent.env import Action, SokobanEnv
from sokoban_agent.env.model import MoveResult
from sokoban_agent.env.rules import (
    apply_action,
    decode_observation,
    has_static_corner_deadlock,
    is_success,
    observation_for,
)
from sokoban_agent.graph.baseline.metrics import (
    execution_research_update,
    initial_baseline_metrics,
    observation_key,
    planning_research_update,
    validation_research_update,
)
from sokoban_agent.graph.baseline.state import SokobanGraphState
from sokoban_agent.planning import Observation, Planner, PlanningContext

StepObserver = Callable[
    [Observation, Action | None, Mapping[str, object]],
    None,
]
Route = Literal["plan", "validate", "execute", "end"]


class SokobanGraph:
    """Own the plan, validation, execution, and recovery state machine."""

    def __init__(
        self,
        env: SokobanEnv,
        planner: Planner,
        *,
        max_planning_attempts: int = 3,
        checkpointer: InMemorySaver | None = None,
    ) -> None:
        if max_planning_attempts <= 0:
            raise ValueError("max_planning_attempts must be positive")
        self.env = env
        self.planner = planner
        self.max_planning_attempts = max_planning_attempts
        self.checkpointer = checkpointer or InMemorySaver()
        self._observer: StepObserver | None = None
        self._graph: Any = self._build_graph()

    @property
    def name(self) -> str:
        """Return the planner-qualified graph name."""

        return self.planner.name

    def run(
        self,
        *,
        seed: int | None = None,
        level_id: str | None = None,
        thread_id: str | None = None,
        step_observer: StepObserver | None = None,
    ) -> SokobanGraphState:
        """Reset the environment and execute one checkpointed graph run."""

        options = {"level_id": level_id} if level_id is not None else None
        raw_observation, raw_info = self.env.reset(seed=seed, options=options)
        observation = cast(Observation, raw_observation)
        info = dict(raw_info)
        self.planner.reset(seed=seed)
        self._observer = step_observer
        initial = self._initial_state(observation, info, seed)
        if step_observer is not None:
            step_observer(observation.copy(), None, info)

        run_id = thread_id or (
            f"baseline-v2:{self.name}:{initial['level_id']}:{seed}:{uuid4().hex}"
        )
        config = {
            "configurable": {"thread_id": run_id},
            "recursion_limit": self.env.max_steps * 4
            + self.max_planning_attempts * 3
            + 10,
        }
        try:
            result = self._graph.invoke(initial, config)
        finally:
            self._observer = None
        return cast(SokobanGraphState, result)

    def _build_graph(self) -> Any:
        builder = StateGraph(SokobanGraphState)
        builder.add_node("plan", self._plan)
        builder.add_node("validate", self._validate)
        builder.add_node("execute", self._execute)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"plan": "plan", "validate": "validate", "end": END},
        )
        builder.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {"plan": "plan", "execute": "execute", "end": END},
        )
        builder.add_conditional_edges(
            "execute",
            self._route_after_execute,
            {"plan": "plan", "validate": "validate", "end": END},
        )
        return builder.compile(checkpointer=self.checkpointer)

    def _plan(self, state: SokobanGraphState) -> dict[str, object]:
        observation = _observation(state)
        context = PlanningContext(
            observation=observation,
            info=state["info"],
            action_history=tuple(
                Action[action_name] for action_name in state["action_history"]
            ),
            feedback=tuple(state["feedback"]),
            seed=state["seed"],
        )
        started_at = perf_counter()
        outcome = self.planner.plan(context)
        elapsed = max(outcome.elapsed_seconds, perf_counter() - started_at)
        attempts = state["planning_attempts"] + 1
        error = outcome.error
        if not outcome.actions and error is None:
            error = "플래너가 행동 계획을 반환하지 않았습니다"
        exhausted = error is not None and attempts >= self.max_planning_attempts
        retry = error is not None and not exhausted
        feedback = state["feedback"]
        if error is not None:
            feedback = [*feedback, error]
        proposed = outcome.proposed_actions or outcome.actions
        used_llm_actions = outcome.llm_calls > 0 and (
            outcome.guard_disposition is None
            or outcome.guard_disposition == "accepted"
        )
        return {
            "plan": [action.name for action in outcome.actions],
            "proposal": {
                "proposed_actions": [action.name for action in proposed],
                "goal": outcome.goal,
                "risk": outcome.risk,
                "guard_summary": outcome.guard_summary,
                "used_llm_actions": used_llm_actions,
            },
            "feedback": feedback,
            "planning_attempts": attempts,
            "failure_reason": error if exhausted else None,
            **planning_research_update(
                state,
                observation,
                outcome,
                elapsed_seconds=elapsed,
                error=error is not None,
                retry=retry,
            ),
        }

    def _validate(self, state: SokobanGraphState) -> dict[str, object]:
        level, board = decode_observation(_observation(state))
        validated: list[str] = []
        message: str | None = None
        for index, action_name in enumerate(state["plan"]):
            action = Action[action_name]
            move = apply_action(level, board, action)
            if move.invalid_move:
                message = (
                    f"계획의 {index + 1}번째 행동 "
                    f"{action.name}이 현재 보드에서 막혀 있습니다"
                )
                break
            board = move.state
            validated.append(action_name)
            if has_static_corner_deadlock(level, board):
                message = (
                    f"계획의 {index + 1}번째 행동 "
                    f"{action.name}이 데드락을 만듭니다"
                )
                break
            if is_success(level, board):
                return {"plan": validated}

        if message is None:
            return {}
        exhausted = state["planning_attempts"] >= self.max_planning_attempts
        retry = not exhausted
        return {
            "plan": [],
            "feedback": [*state["feedback"], message],
            "failure_reason": message if exhausted else None,
            "metrics": validation_research_update(state, retry=retry),
        }

    def _execute(self, state: SokobanGraphState) -> dict[str, object]:
        observation = _observation(state)
        level, board = decode_observation(observation)
        action = Action[state["plan"][0]]
        move = apply_action(level, board, action)
        if move.invalid_move:
            raise RuntimeError("validated plan became invalid before execution")

        success = is_success(level, move.state)
        deadlock = not success and has_static_corner_deadlock(level, move.state)
        steps = _step_count(state) + 1
        truncated = steps >= self.env.max_steps and not success and not deadlock
        reward = _reward(self.env, move, success=success, deadlock=deadlock)
        next_observation = observation_for(level, move.state)
        info = {
            **state["info"],
            "steps": steps,
            "invalid_move": False,
            "pushed": move.pushed,
            "boxes_on_targets": len(move.state.boxes & level.targets),
            "success": success,
            "deadlock": deadlock,
        }
        if self._observer is not None:
            self._observer(next_observation.copy(), action, info)
        return {
            "observation": next_observation.tolist(),
            "info": info,
            "plan": state["plan"][1:],
            "action_history": [*state["action_history"], action.name],
            "feedback": [],
            "planning_attempts": 0,
            **execution_research_update(
                state,
                next_observation,
                pushed=move.pushed,
                reward=reward,
            ),
            "truncated": truncated,
            "failure_reason": None,
        }

    def _route_after_plan(self, state: SokobanGraphState) -> Route:
        if state["plan"]:
            return "validate"
        if state["failure_reason"] is not None:
            return "end"
        return "plan"

    def _route_after_validate(self, state: SokobanGraphState) -> Route:
        if state["plan"]:
            return "execute"
        if state["failure_reason"] is not None:
            return "end"
        return "plan"

    @staticmethod
    def _route_after_execute(state: SokobanGraphState) -> Route:
        if (
            bool(state["info"]["success"])
            or bool(state["info"]["deadlock"])
            or state["truncated"]
        ):
            return "end"
        if state["plan"]:
            return "validate"
        return "plan"

    @staticmethod
    def _initial_state(
        observation: Observation,
        info: dict[str, object],
        seed: int | None,
    ) -> SokobanGraphState:
        return {
            "observation": observation.tolist(),
            "info": info,
            "seed": seed,
            "level_id": str(info["level_id"]),
            "plan": [],
            "proposal": None,
            "action_history": [],
            "visited_state_keys": [observation_key(observation)],
            "seen_plan_keys": [],
            "feedback": [],
            "planning_attempts": 0,
            "truncated": False,
            "failure_reason": None,
            "metrics": initial_baseline_metrics(),
        }


def _observation(state: SokobanGraphState) -> Observation:
    return cast(Observation, np.asarray(state["observation"], dtype=np.uint8))


def _step_count(state: SokobanGraphState) -> int:
    steps = state["info"].get("steps")
    if not isinstance(steps, int):
        raise TypeError("graph info steps must be an integer")
    return steps


def _reward(
    env: SokobanEnv,
    move: MoveResult,
    *,
    success: bool,
    deadlock: bool,
) -> float:
    reward = env.reward_config.step
    if move.pushed and move.box_left_target:
        reward += env.reward_config.box_off_target
    if move.pushed and move.box_entered_target:
        reward += env.reward_config.box_on_target
    if success:
        reward += env.reward_config.completion
    elif deadlock:
        reward += env.reward_config.deadlock
    return reward
