"""LangGraph Store-backed memory nodes for structured Sokoban decisions."""

from __future__ import annotations

from typing import Literal, cast

from langgraph.runtime import Runtime

from sokoban_agent.graph.agentic_memory_keys import (
    board_memory_key,
    constraints,
    copy_rejections,
    current_push_id,
    grounding_memory_key,
    memory_event,
    memory_namespace,
    observation,
    shared_memory,
    strategy_memory_key,
    topology_memory_key,
)
from sokoban_agent.graph.agentic_state import (
    AgenticRuntimeContext,
    AgenticState,
)
from sokoban_agent.planning.base import Observation
from sokoban_agent.planning.board_analysis import (
    StaticBoardFacts,
    analyze_static_board,
    dump_static_board_facts,
    load_static_board_facts,
)
from sokoban_agent.planning.local_execution import (
    SubgoalGroundingError,
    validate_grounded_push_plan,
)
from sokoban_agent.planning.strategy import (
    BoardAnalysis,
    GroundedPushPlan,
    PushSubgoal,
    StrategyHypothesis,
)

StrategyRecallRoute = Literal["propose_strategy", "verify_strategy"]
GroundingRecallRoute = Literal["ground_subgoal", "execute_until_push"]
FailureMemoryRoute = Literal["compose_strategy_input", "__end__"]
OutcomeMemoryRoute = Literal["observe", "__end__"]

_REJECTING_VIOLATIONS = {
    "unknown_box",
    "unavailable_push",
    "protected_cell_conflict",
    "static_deadlock_push",
    "immediate_reverse",
}


def get_static_board_facts(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
    observation: Observation,
) -> tuple[StaticBoardFacts, int, int, int]:
    """Recall topology facts and return request, hit, and write increments."""

    if not shared_memory(state) or runtime.store is None:
        return analyze_static_board(observation), 0, 0, 0
    key = topology_memory_key(observation)
    item = runtime.store.get(memory_namespace(state, "board"), key)
    if item is not None:
        try:
            return load_static_board_facts(item.value), 1, 1, 0
        except ValueError:
            pass
    facts = analyze_static_board(observation)
    runtime.store.put(
        memory_namespace(state, "board"),
        key,
        dump_static_board_facts(facts),
        index=False,
    )
    return facts, 1, 0, 1


def recall_failed_decisions(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
) -> dict[str, object]:
    """Merge cross-thread rejected pushes into checkpointed run memory."""

    rejected = copy_rejections(state)
    requests = hits = 0
    key = board_memory_key(state)
    if shared_memory(state) and runtime.store is not None:
        requests = 1
        item = runtime.store.get(memory_namespace(state, "failures"), key)
        if item is not None:
            stored = item.value.get("push_ids")
            if isinstance(stored, list) and all(
                isinstance(value, str) for value in stored
            ):
                rejected[key] = sorted(
                    {*rejected.get(key, []), *stored}
                )
                hits = 1
    count = len(rejected.get(key, []))
    return {
        "rejected_pushes": rejected,
        "memory_requests": state["memory_requests"] + requests,
        "memory_hits": state["memory_hits"] + hits,
        "status": "failure_memory_recalled",
        "decision_events": [
            memory_event(
                state,
                "recall_failures",
                f"현재 보드의 제외 push {count}개를 불러왔습니다",
            )
        ],
    }


def filter_rejected_pushes(
    state: AgenticState,
    model_context: dict[str, object],
) -> tuple[dict[str, object], int]:
    """Remove decisions already rejected at the planner-visible abstraction."""

    if state["memory_mode"] == "off":
        return model_context, 0
    rejected = set(state["rejected_pushes"].get(board_memory_key(state), []))
    options = model_context.get("safe_push_options")
    if not rejected or not isinstance(options, list):
        return model_context, 0
    filtered = [
        option
        for option in options
        if not (
            isinstance(option, dict)
            and isinstance(option.get("push_id"), str)
            and option["push_id"] in rejected
        )
    ]
    return {**model_context, "safe_push_options": filtered}, len(options) - len(
        filtered
    )


def recall_strategy(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
) -> dict[str, object]:
    """Recall an exact validated decision before making another model call."""

    requests = hits = 0
    hypothesis: dict[str, object] | None = None
    if shared_memory(state) and runtime.store is not None:
        requests = 1
        item = runtime.store.get(
            memory_namespace(state, "strategies"),
            strategy_memory_key(state),
        )
        if item is not None:
            try:
                hypothesis = StrategyHypothesis.model_validate(
                    item.value.get("hypothesis")
                ).model_dump(mode="json")
                hits = 1
            except (TypeError, ValueError):
                hypothesis = None
    return {
        "strategy_hypothesis": hypothesis,
        "strategy_memory_hit": hits == 1,
        "memory_requests": state["memory_requests"] + requests,
        "memory_hits": state["memory_hits"] + hits,
        "strategy_cache_hits": state["strategy_cache_hits"] + hits,
        "llm_calls_saved": state["llm_calls_saved"] + hits,
        "status": "strategy_memory_hit" if hits else "strategy_memory_miss",
        "decision_events": [
            memory_event(
                state,
                "recall_strategy",
                "검증된 전략을 재사용합니다"
                if hits
                else "재사용할 검증 전략이 없습니다",
            )
        ],
    }


def route_after_strategy_recall(state: AgenticState) -> StrategyRecallRoute:
    return "verify_strategy" if state["strategy_memory_hit"] else "propose_strategy"


def recall_grounding(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
) -> dict[str, object]:
    """Recall and linearly revalidate an exact single-push grounding plan."""

    requests = hits = 0
    plan: GroundedPushPlan | None = None
    cache_key = grounding_memory_key(state)
    if shared_memory(state) and runtime.store is not None:
        requests = 1
        item = runtime.store.get(
            memory_namespace(state, "grounding"),
            cache_key,
        )
        if item is not None:
            try:
                candidate = GroundedPushPlan.model_validate(
                    item.value.get("plan")
                )
                validate_grounded_push_plan(
                    observation(state),
                    BoardAnalysis.model_validate(state["board_analysis"]),
                    PushSubgoal.model_validate(state["active_subgoal"]),
                    constraints(state),
                    candidate,
                )
                plan = candidate
                hits = 1
            except (SubgoalGroundingError, TypeError, ValueError):
                plan = None
    actions = (
        [*plan.player_actions, plan.push_action] if plan is not None else []
    )
    return {
        "grounded_plan": (
            plan.model_dump(mode="json") if plan is not None else None
        ),
        "grounded_actions": actions,
        "grounding_failure": None,
        "grounding_memory_hit": hits == 1,
        "grounding_cache_key": cache_key,
        "memory_requests": state["memory_requests"] + requests,
        "memory_hits": state["memory_hits"] + hits,
        "grounding_cache_hits": state["grounding_cache_hits"] + hits,
        "subgoal_grounding_attempts": (
            state["subgoal_grounding_attempts"] + hits
        ),
        "rule_checks": state["rule_checks"] + hits,
        "status": "grounding_memory_hit" if hits else "grounding_memory_miss",
        "decision_events": [
            memory_event(
                state,
                "recall_grounding",
                "검증된 단일 push 경로를 재사용합니다"
                if hits
                else "재사용할 접지 경로가 없습니다",
            )
        ],
    }


def route_after_grounding_recall(state: AgenticState) -> GroundingRecallRoute:
    return "execute_until_push" if state["grounding_memory_hit"] else "ground_subgoal"


def remember_failure(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
) -> dict[str, object]:
    """Checkpoint and optionally persist a rejected push decision."""

    rejected = copy_rejections(state)
    push_id = _rejected_push_id(state)
    writes = 0
    if state["memory_mode"] != "off" and push_id is not None:
        key = board_memory_key(state)
        rejected[key] = sorted({*rejected.get(key, []), push_id})
        if shared_memory(state) and runtime.store is not None:
            runtime.store.put(
                memory_namespace(state, "failures"),
                key,
                {"push_ids": rejected[key]},
                index=False,
            )
            writes = 1
    return {
        "rejected_pushes": rejected,
        "memory_writes": state["memory_writes"] + writes,
        "status": state["status"],
        "decision_events": [
            memory_event(
                state,
                "remember_failure",
                f"{push_id} 결정을 제외 메모리에 기록했습니다"
                if push_id is not None
                else "수정 피드백만 유지하고 push는 제외하지 않았습니다",
            )
        ],
    }


def route_after_failure_memory(state: AgenticState) -> FailureMemoryRoute:
    return (
        "compose_strategy_input"
        if state["strategy_attempts"] < 2
        else "__end__"
    )


def remember_outcome(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
) -> dict[str, object]:
    """Persist only strategies and paths proven by the environment."""

    reflection = cast(dict[str, object], state["reflection_result"])
    writes = 0
    rejected = copy_rejections(state)
    matched = reflection.get("matched") is True
    if matched and shared_memory(state) and runtime.store is not None:
        runtime.store.put(
            memory_namespace(state, "strategies"),
            strategy_memory_key(state),
            {"hypothesis": state["strategy_hypothesis"]},
            index=False,
        )
        runtime.store.put(
            memory_namespace(state, "grounding"),
            cast(str, state["grounding_cache_key"]),
            {"plan": state["grounded_plan"]},
            index=False,
        )
        writes = 2
    elif not matched and state["memory_mode"] != "off":
        push_id = current_push_id(state)
        key = board_memory_key(state)
        rejected[key] = sorted({*rejected.get(key, []), push_id})
        if shared_memory(state) and runtime.store is not None:
            runtime.store.put(
                memory_namespace(state, "failures"),
                key,
                {"push_ids": rejected[key]},
                index=False,
            )
            writes = 1
    return {
        "rejected_pushes": rejected,
        "memory_writes": state["memory_writes"] + writes,
        "status": state["status"],
        "decision_events": [
            memory_event(
                state,
                "remember_outcome",
                "성공한 전략과 접지 경로를 저장했습니다"
                if matched and writes
                else (
                    "실패한 push를 제외 메모리에 기록했습니다"
                    if not matched
                    else "성공 결과를 에피소드 메모리에 유지했습니다"
                ),
            )
        ],
    }


def route_after_outcome_memory(state: AgenticState) -> OutcomeMemoryRoute:
    if state["status"] in {"success", "deadlock", "step_limit"}:
        return "__end__"
    return "observe"


def _rejected_push_id(state: AgenticState) -> str | None:
    if state["status"] == "subgoal_grounding_failed":
        return current_push_id(state)
    codes = {
        item.get("code")
        for item in state["strategy_violations"]
        if isinstance(item, dict)
    }
    return (
        current_push_id(state)
        if codes & _REJECTING_VIOLATIONS
        else None
    )
