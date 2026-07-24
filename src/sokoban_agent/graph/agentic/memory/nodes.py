"""LangGraph Store-backed memory nodes for structured Sokoban decisions."""

from __future__ import annotations

from typing import Literal, cast

from langgraph.runtime import Runtime
from langgraph.types import Command

from sokoban_agent.graph.agentic.memory.keys import (
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
)
from sokoban_agent.graph.agentic.metrics import update_agentic_metrics
from sokoban_agent.graph.agentic.state import (
    AgenticRuntimeContext,
    AgenticState,
    active_subgoal,
)
from sokoban_agent.planning.agentic.grounding import (
    SubgoalGroundingError,
    validate_grounded_push_plan,
)
from sokoban_agent.planning.agentic.models import (
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
    metrics = state["metrics"]
    return {
        "memory": {**state["memory"], "rejected_pushes": rejected},
        "metrics": update_agentic_metrics(
            metrics,
            memory={
                "requests": metrics["memory"]["requests"] + requests,
                "hits": metrics["memory"]["hits"] + hits,
            },
        ),
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

    if state["meta"]["memory_mode"] == "off":
        return model_context, 0
    rejected = set(
        state["memory"]["rejected_pushes"].get(
            board_memory_key(state), []
        )
    )
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
) -> Command[StrategyRecallRoute]:
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
    metrics = state["metrics"]
    return Command(
        update={
        "planning": {
            **state["planning"],
            "strategy_hypothesis": hypothesis,
        },
        "metrics": update_agentic_metrics(
            metrics,
            memory={
                "requests": metrics["memory"]["requests"] + requests,
                "hits": metrics["memory"]["hits"] + hits,
                "strategy_cache_hits": (
                    metrics["memory"]["strategy_cache_hits"] + hits
                ),
                "llm_calls_saved": (
                    metrics["memory"]["llm_calls_saved"] + hits
                ),
            },
        ),
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
        },
        goto="verify_strategy" if hits else "propose_strategy",
    )


def recall_grounding(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
) -> Command[GroundingRecallRoute]:
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
                    BoardAnalysis.model_validate(
                        state["planning"]["board_analysis"]
                    ),
                    PushSubgoal.model_validate(active_subgoal(state)),
                    constraints(state),
                    candidate,
                )
                plan = candidate
                hits = 1
            except (SubgoalGroundingError, TypeError, ValueError):
                plan = None
    metrics = state["metrics"]
    return Command(
        update={
        "planning": {
            **state["planning"],
            "grounded_plan": (
                plan.model_dump(mode="json")
                if plan is not None
                else None
            ),
            "grounding_failure": None,
            "grounding_cache_key": cache_key,
        },
        "metrics": update_agentic_metrics(
            metrics,
            memory={
                "requests": metrics["memory"]["requests"] + requests,
                "hits": metrics["memory"]["hits"] + hits,
                "grounding_cache_hits": (
                    metrics["memory"]["grounding_cache_hits"] + hits
                ),
            },
            strategy={
                "subgoal_attempts": (
                    metrics["strategy"]["subgoal_attempts"] + hits
                )
            },
            rules={"checks": metrics["rules"]["checks"] + hits},
        ),
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
        },
        goto="execute_until_push" if hits else "ground_subgoal",
    )


def remember_failure(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
) -> dict[str, object]:
    """Checkpoint and optionally persist a rejected push decision."""

    rejected = copy_rejections(state)
    push_id = _rejected_push_id(state)
    writes = 0
    if state["meta"]["memory_mode"] != "off" and push_id is not None:
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
    metrics = state["metrics"]
    return {
        "memory": {**state["memory"], "rejected_pushes": rejected},
        "metrics": update_agentic_metrics(
            metrics,
            memory={"writes": metrics["memory"]["writes"] + writes},
        ),
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
        if state["planning"]["strategy_attempts"] < 2
        else "__end__"
    )


def remember_outcome(
    state: AgenticState,
    runtime: Runtime[AgenticRuntimeContext],
) -> dict[str, object]:
    """Persist only strategies and paths proven by the environment."""

    reflection = cast(
        dict[str, object], state["execution"]["reflection"]
    )
    writes = 0
    rejected = copy_rejections(state)
    matched = reflection.get("matched") is True
    if matched and shared_memory(state) and runtime.store is not None:
        runtime.store.put(
            memory_namespace(state, "strategies"),
            strategy_memory_key(state),
            {"hypothesis": state["planning"]["strategy_hypothesis"]},
            index=False,
        )
        runtime.store.put(
            memory_namespace(state, "grounding"),
            cast(str, state["planning"]["grounding_cache_key"]),
            {"plan": state["planning"]["grounded_plan"]},
            index=False,
        )
        writes = 2
    elif not matched and state["meta"]["memory_mode"] != "off":
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
    metrics = state["metrics"]
    return {
        "memory": {**state["memory"], "rejected_pushes": rejected},
        "metrics": update_agentic_metrics(
            metrics,
            memory={"writes": metrics["memory"]["writes"] + writes},
        ),
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
        for item in state["planning"]["strategy_violations"]
        if isinstance(item, dict)
    }
    return (
        current_push_id(state)
        if codes & _REJECTING_VIOLATIONS
        else None
    )
