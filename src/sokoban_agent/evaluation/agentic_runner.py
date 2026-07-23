"""Evaluation through the same compiled graph used by LangGraph Studio."""

from __future__ import annotations

from time import perf_counter

from sokoban_agent.evaluation.agentic_results import AgenticEpisodeResult
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
    return AgenticEpisodeResult(
        policy_name=(
            "structured-no-rationale"
            if rationale_mode == "off"
            else "structured-local-search"
        ),
        level_id=state["level_id"],
        seed=state["seed"],
        status=state["status"],
        success=info.get("success") is True,
        deadlock=info.get("deadlock") is True,
        truncated=state["status"] == "step_limit",
        cycle_detected=state["cycle_detected"],
        action_count=len(state["action_history"]),
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
        local_search_calls=state["local_search_calls"],
        local_expanded_states=state["local_expanded_states"],
        algorithm_calls=0,
        prompt_name=prompt["name"],
        prompt_commit=prompt["commit"],
        model_name=state["model_name"],
        elapsed_seconds=elapsed,
    )
