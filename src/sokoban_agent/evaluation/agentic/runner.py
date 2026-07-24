"""Evaluation through the same compiled graph used by LangGraph Studio."""

from __future__ import annotations

from time import perf_counter

from sokoban_agent.evaluation.schemas.episode import (
    AgenticEpisodeResult,
    EpisodeIdentity,
    EpisodeOutcome,
)
from sokoban_agent.evaluation.schemas.metrics import (
    LLMUsage,
    MemoryUsage,
    PromptIdentity,
    RuleUsage,
    SearchUsage,
    StrategyUsage,
)
from sokoban_agent.graph.agentic.runtime import AgenticGraphRunner
from sokoban_agent.graph.agentic.state import (
    AgenticInput,
    AgenticRuntimeContext,
    AgenticState,
)
from sokoban_agent.planning.agentic.runtime import (
    PromptSource,
    StrategyGenerator,
)


def run_agentic_episode(
    graph_input: AgenticInput,
    *,
    context: AgenticRuntimeContext,
    prompt_source: PromptSource | None = None,
    strategy_generator: StrategyGenerator | None = None,
    thread_id: str | None = None,
) -> AgenticEpisodeResult:
    """Run and measure one structured policy without a second action loop."""

    runner = AgenticGraphRunner(
        prompt_source=prompt_source,
        strategy_generator=strategy_generator,
    )
    started_at = perf_counter()
    state = runner.run(
        graph_input,
        context=context,
        thread_id=thread_id,
    )
    elapsed = perf_counter() - started_at
    info = state["info"]
    meta = state["meta"]
    prompt = meta["prompt"]
    rationale_mode = context.get("rationale_mode", "on")
    grounding_mode = context.get("grounding_mode", "local-search")
    metrics = state["metrics"]
    memory_metrics = metrics["memory"]
    local_metrics = metrics["local_search"]
    rule_metrics = metrics["rules"]
    llm_metrics = metrics["llm"]
    return AgenticEpisodeResult(
        identity=EpisodeIdentity(
            policy_name=(
                "structured-no-rationale"
                if rationale_mode == "off"
                else (
                    "structured-llm"
                    if grounding_mode == "direct"
                    else "structured-local-search"
                )
            ),
            level_id=meta["level_id"],
            seed=meta["seed"],
        ),
        outcome=EpisodeOutcome(
            status=state["status"],
            success=info.get("success") is True,
            deadlock=info.get("deadlock") is True,
            truncated=state["status"] == "step_limit",
            cycle_detected=state["cycle_detected"],
            action_sequence=tuple(state["action_history"]),
            push_count=state["push_count"],
        ),
        strategy=_strategy_usage(state),
        llm=LLMUsage(
            calls=llm_metrics["calls"],
            elapsed_seconds=llm_metrics["elapsed_seconds"],
            prompt_tokens=llm_metrics["prompt_tokens"],
            output_tokens=llm_metrics["output_tokens"],
        ),
        memory=MemoryUsage(**memory_metrics),
        local_search=SearchUsage(**local_metrics),
        rules=RuleUsage(
            checks=rule_metrics["checks"],
            reachability_calls=rule_metrics["reachability_calls"],
        ),
        prompt=PromptIdentity(
            name=prompt["name"],
            commit=prompt["commit"],
            model_name=meta["model_name"],
        ),
        elapsed_seconds=elapsed,
    )


def _strategy_usage(state: AgenticState) -> StrategyUsage:
    metrics = state["metrics"]["strategy"]
    revisions = state["plan_revisions"]
    return StrategyUsage(
        proposals=metrics["proposals"],
        schema_rejections=metrics["schema_rejections"],
        semantic_rejections=metrics["semantic_rejections"],
        protected_constraint_violations=sum(
            item.startswith("protected_constraint")
            for item in state["feedback"]
        ),
        effect_matches=metrics["effect_matches"],
        effect_mismatches=metrics["effect_mismatches"],
        subgoal_attempts=metrics["subgoal_attempts"],
        subgoal_grounding_failures=metrics["subgoal_grounding_failures"],
        plan_revisions=len(revisions),
        assignment_revisions=sum(
            "assignments" in revision["changed_fields"]
            for revision in revisions
        ),
        hypothesis_revisions=sum(
            "hypothesis" in revision["changed_fields"]
            for revision in revisions
        ),
        subgoal_revisions=sum(
            "subgoal" in revision["changed_fields"] for revision in revisions
        ),
    )
