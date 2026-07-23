"""LangGraph-first Sokoban episode runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, Literal, cast
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from sokoban_agent.env import Action, SokobanEnv
from sokoban_agent.env.rules import (
    apply_action,
    decode_observation,
    has_static_corner_deadlock,
    is_success,
)
from sokoban_agent.graph.state import SokobanGraphState
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
            f"{self.name}:{initial['level_id']}:{seed}:{uuid4().hex}"
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
        context = PlanningContext(
            observation=state["observation"],
            info=state["info"],
            action_history=state["action_history"],
            feedback=state["feedback"],
            seed=state["seed"],
        )
        started_at = perf_counter()
        outcome = self.planner.plan(context)
        elapsed = max(outcome.elapsed_seconds, perf_counter() - started_at)
        attempts = state["planning_attempts"] + 1
        error = outcome.error
        if not outcome.actions and error is None:
            error = "planner returned no actions"
        exhausted = error is not None and attempts >= self.max_planning_attempts
        feedback = state["feedback"]
        if error is not None:
            feedback = (*feedback, error)
        return {
            "plan": outcome.actions,
            "feedback": feedback,
            "planning_attempts": attempts,
            "failure_reason": error if exhausted else None,
            "planning_calls": state["planning_calls"] + 1,
            "planning_retries": state["planning_retries"]
            + int(error is not None and not exhausted),
            "planning_errors": state["planning_errors"] + int(error is not None),
            "planning_elapsed_seconds": (
                state["planning_elapsed_seconds"] + elapsed
            ),
            "algorithm_calls": (
                state["algorithm_calls"] + outcome.algorithm_calls
            ),
            "algorithm_fallbacks": (
                state["algorithm_fallbacks"] + outcome.algorithm_fallbacks
            ),
            "algorithm_expanded_states": (
                state["algorithm_expanded_states"]
                + outcome.algorithm_expanded_states
            ),
            "algorithm_elapsed_seconds": (
                state["algorithm_elapsed_seconds"]
                + outcome.algorithm_elapsed_seconds
            ),
            "llm_calls": state["llm_calls"] + outcome.llm_calls,
            "llm_client_errors": (
                state["llm_client_errors"] + outcome.llm_client_errors
            ),
            "llm_format_errors": (
                state["llm_format_errors"] + outcome.llm_format_errors
            ),
            "llm_elapsed_seconds": (
                state["llm_elapsed_seconds"]
                + outcome.llm_elapsed_seconds
            ),
            "llm_load_seconds": (
                state["llm_load_seconds"] + outcome.llm_load_seconds
            ),
            "llm_prompt_eval_seconds": (
                state["llm_prompt_eval_seconds"]
                + outcome.llm_prompt_eval_seconds
            ),
            "llm_eval_seconds": (
                state["llm_eval_seconds"] + outcome.llm_eval_seconds
            ),
            "llm_prompt_tokens": (
                state["llm_prompt_tokens"] + outcome.llm_prompt_tokens
            ),
            "llm_output_tokens": (
                state["llm_output_tokens"] + outcome.llm_output_tokens
            ),
            "last_proposal_used_llm": outcome.llm_calls > 0,
        }

    def _validate(self, state: SokobanGraphState) -> dict[str, object]:
        level, board = decode_observation(state["observation"])
        validated: list[Action] = []
        message: str | None = None
        for index, action in enumerate(state["plan"]):
            move = apply_action(level, board, action)
            if move.invalid_move:
                message = (
                    f"plan action {index + 1} ({action.name}) is blocked"
                )
                break
            board = move.state
            validated.append(action)
            if has_static_corner_deadlock(level, board):
                message = (
                    f"plan action {index + 1} ({action.name}) causes deadlock"
                )
                break
            if is_success(level, board):
                return {"plan": tuple(validated)}

        if message is None:
            return {}
        exhausted = (
            state["planning_attempts"] >= self.max_planning_attempts
        )
        return {
            "plan": (),
            "feedback": (*state["feedback"], message),
            "invalid_moves": state["invalid_moves"] + 1,
            "planning_retries": state["planning_retries"] + int(not exhausted),
            "failure_reason": message if exhausted else None,
            "llm_invalid_actions": state["llm_invalid_actions"]
            + int(state["last_proposal_used_llm"]),
        }

    def _execute(self, state: SokobanGraphState) -> dict[str, object]:
        action = state["plan"][0]
        observation, reward, _, truncated, raw_info = self.env.step(action)
        info = dict(raw_info)
        if self._observer is not None:
            self._observer(observation.copy(), action, info)
        return {
            "observation": observation,
            "info": info,
            "plan": state["plan"][1:],
            "action_history": (*state["action_history"], action),
            "feedback": (),
            "planning_attempts": 0,
            "action_count": state["action_count"] + 1,
            "total_reward": state["total_reward"] + reward,
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
            "observation": observation,
            "info": info,
            "seed": seed,
            "level_id": str(info["level_id"]),
            "plan": (),
            "action_history": (),
            "feedback": (),
            "planning_attempts": 0,
            "action_count": 0,
            "invalid_moves": 0,
            "total_reward": 0.0,
            "truncated": False,
            "failure_reason": None,
            "planning_calls": 0,
            "planning_retries": 0,
            "planning_errors": 0,
            "planning_elapsed_seconds": 0.0,
            "algorithm_calls": 0,
            "algorithm_fallbacks": 0,
            "algorithm_expanded_states": 0,
            "algorithm_elapsed_seconds": 0.0,
            "llm_calls": 0,
            "llm_client_errors": 0,
            "llm_format_errors": 0,
            "llm_invalid_actions": 0,
            "llm_elapsed_seconds": 0.0,
            "llm_load_seconds": 0.0,
            "llm_prompt_eval_seconds": 0.0,
            "llm_eval_seconds": 0.0,
            "llm_prompt_tokens": 0,
            "llm_output_tokens": 0,
            "last_proposal_used_llm": False,
        }
