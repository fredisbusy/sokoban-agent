"""Evaluation through the same compiled graph used by LangGraph Studio."""

from __future__ import annotations

from time import perf_counter

from sokoban_agent.evaluation.schemas.episode import AgenticEpisodeResult
from sokoban_agent.graph.agentic_runtime import AgenticGraphRunner
from sokoban_agent.graph.agentic_state import (
    AgenticInput,
    AgenticRuntimeContext,
)
from sokoban_agent.planning.strategy_runtime import (
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
    prompt = state["prompt"]
    feedback = state["feedback"]
    rationale_mode = context.get("rationale_mode", "on")
    grounding_mode = context.get("grounding_mode", "local-search")
    revisions = state["plan_revisions"]
    return AgenticEpisodeResult(
        policy_name=(
            "structured-no-rationale"
            if rationale_mode == "off"
            else (
                "structured-llm"
                if grounding_mode == "direct"
                else "structured-local-search"
            )
        ),
        level_id=state["level_id"],
        seed=state["seed"],
        status=state["status"],
        success=info.get("success") is True,
        deadlock=info.get("deadlock") is True,
        truncated=state["status"] == "step_limit",
        cycle_detected=state["cycle_detected"],
        action_count=len(state["action_history"]),
        action_sequence=tuple(state["action_history"]),
        push_count=state["push_count"],
        strategy_proposals=state["strategy_proposals"],
        strategy_schema_rejections=state["strategy_schema_rejections"],
        strategy_semantic_rejections=state["strategy_semantic_rejections"],
        plan_revision_count=len(state["plan_revisions"]),
        protected_constraint_violations=sum(
            item.startswith("protected_constraint") for item in feedback
        ),
        effect_matches=state["effect_matches"],
        effect_mismatches=state["effect_mismatches"],
        llm_calls=state["llm_calls"],
        llm_elapsed_seconds=state["llm_elapsed_seconds"],
        llm_prompt_tokens=state["llm_prompt_tokens"],
        llm_output_tokens=state["llm_output_tokens"],
        memory_requests=state["memory_requests"],
        memory_hits=state["memory_hits"],
        memory_writes=state["memory_writes"],
        strategy_cache_hits=state["strategy_cache_hits"],
        grounding_cache_hits=state["grounding_cache_hits"],
        analysis_cache_hits=state["analysis_cache_hits"],
        llm_calls_saved=state["llm_calls_saved"],
        rejected_pushes_filtered=state["rejected_pushes_filtered"],
        local_search_calls=state["local_search_calls"],
        local_expanded_states=state["local_expanded_states"],
        local_search_elapsed_seconds=state[
            "local_search_elapsed_seconds"
        ],
        rule_checks=state["rule_checks"],
        reachability_calls=state["reachability_calls"],
        subgoal_attempts=state["subgoal_grounding_attempts"],
        subgoal_successes=state["effect_matches"],
        subgoal_failures=(
            state["subgoal_grounding_failures"]
            + state["effect_mismatches"]
        ),
        assignment_revision_count=sum(
            "assignments" in revision["changed_fields"]
            for revision in revisions
        ),
        hypothesis_revision_count=len(revisions),
        actions_derived_from_subgoal=len(state["action_history"]),
        algorithm_calls=0,
        prompt_name=prompt["name"],
        prompt_commit=prompt["commit"],
        model_name=state["model_name"],
        elapsed_seconds=elapsed,
    )
