"""Versioned flat artifact adapters for composed baseline results."""

from __future__ import annotations

from typing import Any

from sokoban_agent.evaluation.schemas.baseline import (
    AlgorithmUsage,
    BaselineEpisodeOutcome,
    BaselineLLMUsage,
    EpisodeResult,
    EpisodeTiming,
    GuardUsage,
    PlannerEpisodeIdentity,
    PlanningUsage,
)
from sokoban_agent.evaluation.schemas.baseline_summary import PlannerSummary
from sokoban_agent.evaluation.schemas.reference import ReferenceResult

EPISODE_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1


class BaselineEpisodeRowV1:
    """Read and write the stable 57-column pilot episode row."""

    @staticmethod
    def to_dict(result: EpisodeResult) -> dict[str, object]:
        identity = result.identity
        outcome = result.outcome
        planning = result.planning
        algorithm = result.algorithm
        llm = result.llm
        guard = result.guard
        reference = result.reference
        return {
            "planner_name": identity.planner_name,
            "level_id": identity.level_id,
            "seed": identity.seed,
            "success": outcome.success,
            "deadlock": outcome.deadlock,
            "truncated": outcome.truncated,
            "action_count": outcome.action_count,
            "invalid_moves": outcome.invalid_moves,
            "total_reward": outcome.total_reward,
            "elapsed_seconds": result.timing.elapsed_seconds,
            "failure_reason": outcome.failure_reason,
            "planning_calls": planning.calls,
            "planning_retries": planning.retries,
            "planning_errors": planning.errors,
            "planning_elapsed_seconds": planning.elapsed_seconds,
            "algorithm_calls": algorithm.calls,
            "algorithm_requests": algorithm.requests,
            "algorithm_cache_hits": algorithm.cache_hits,
            "algorithm_failures": algorithm.failures,
            "algorithm_fallbacks": algorithm.fallbacks,
            "algorithm_expanded_states": algorithm.expanded_states,
            "algorithm_elapsed_seconds": algorithm.elapsed_seconds,
            "llm_calls": llm.calls,
            "llm_retries": llm.retries,
            "llm_client_errors": llm.client_errors,
            "llm_format_errors": llm.format_errors,
            "llm_invalid_actions": llm.invalid_actions,
            "llm_elapsed_seconds": llm.elapsed_seconds,
            "llm_load_seconds": llm.load_seconds,
            "llm_prompt_eval_seconds": llm.prompt_eval_seconds,
            "llm_eval_seconds": llm.eval_seconds,
            "llm_prompt_tokens": llm.prompt_tokens,
            "llm_output_tokens": llm.output_tokens,
            "push_count": outcome.push_count,
            "revisited_states": outcome.revisited_states,
            "repeated_plans": outcome.repeated_plans,
            "guard_accepted": guard.accepted,
            "guard_suffix_added": guard.suffix_added,
            "guard_replaced": guard.replaced,
            "guard_failed": guard.failed,
            "guard_proposed_actions": guard.proposed_actions,
            "guard_legal_prefix_actions": guard.legal_prefix_actions,
            "guard_adopted_actions": guard.adopted_actions,
            "guard_suffix_expanded_states": guard.suffix_expanded_states,
            "guard_reference_calls": guard.reference_calls,
            "guard_reference_action_count": guard.reference_action_count,
            "guard_reference_expanded_states": guard.reference_expanded_states,
            "guard_reference_elapsed_seconds": guard.reference_elapsed_seconds,
            "guard_expansions_saved": guard.expansions_saved,
            "reference_solved": reference.solved,
            "reference_action_count": reference.action_count,
            "reference_push_count": reference.push_count,
            "reference_expanded_states": reference.expanded_states,
            "reference_elapsed_seconds": reference.elapsed_seconds,
            "action_overhead_vs_reference": (
                result.action_overhead_vs_reference
            ),
            "push_overhead_vs_reference": result.push_overhead_vs_reference,
            "policy_elapsed_seconds": result.policy_elapsed_seconds,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> EpisodeResult:
        required = (
            "planner_name",
            "level_id",
            "seed",
            "success",
            "deadlock",
            "truncated",
            "action_count",
            "invalid_moves",
            "total_reward",
            "elapsed_seconds",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"episode row is missing required keys: {missing}")
        unknown = sorted(set(payload) - set(BaselineEpisodeRowV1.columns()))
        if unknown:
            raise ValueError(f"episode row has unknown keys: {unknown}")
        seed = payload["seed"]
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool)
        ):
            raise ValueError("seed must be an integer or null")
        failure_reason = payload.get("failure_reason")
        if failure_reason is not None and not isinstance(failure_reason, str):
            raise ValueError("failure_reason must be a string or null")
        result = EpisodeResult(
            identity=PlannerEpisodeIdentity(
                planner_name=str(payload["planner_name"]),
                level_id=str(payload["level_id"]),
                seed=seed,
            ),
            outcome=BaselineEpisodeOutcome(
                success=_bool(payload["success"], "success"),
                deadlock=_bool(payload["deadlock"], "deadlock"),
                truncated=_bool(payload["truncated"], "truncated"),
                action_count=int(payload["action_count"]),
                invalid_moves=int(payload["invalid_moves"]),
                total_reward=float(payload["total_reward"]),
                failure_reason=failure_reason,
                push_count=int(payload.get("push_count", 0)),
                revisited_states=int(payload.get("revisited_states", 0)),
                repeated_plans=int(payload.get("repeated_plans", 0)),
            ),
            planning=PlanningUsage(
                calls=int(payload.get("planning_calls", 0)),
                retries=int(payload.get("planning_retries", 0)),
                errors=int(payload.get("planning_errors", 0)),
                elapsed_seconds=float(
                    payload.get("planning_elapsed_seconds", 0.0)
                ),
            ),
            algorithm=AlgorithmUsage(
                calls=int(payload.get("algorithm_calls", 0)),
                requests=int(payload.get("algorithm_requests", 0)),
                cache_hits=int(payload.get("algorithm_cache_hits", 0)),
                failures=int(payload.get("algorithm_failures", 0)),
                fallbacks=int(payload.get("algorithm_fallbacks", 0)),
                expanded_states=int(
                    payload.get("algorithm_expanded_states", 0)
                ),
                elapsed_seconds=float(
                    payload.get("algorithm_elapsed_seconds", 0.0)
                ),
            ),
            llm=_llm_usage(payload),
            guard=_guard_usage(payload),
            reference=ReferenceResult(
                solved=_bool(
                    payload.get("reference_solved", False),
                    "reference_solved",
                ),
                action_count=_optional_int(
                    payload.get("reference_action_count"),
                    "reference_action_count",
                ),
                push_count=_optional_int(
                    payload.get("reference_push_count"),
                    "reference_push_count",
                ),
                expanded_states=int(
                    payload.get("reference_expanded_states", 0)
                ),
                elapsed_seconds=float(
                    payload.get("reference_elapsed_seconds", 0.0)
                ),
            ),
            timing=EpisodeTiming(float(payload["elapsed_seconds"])),
        )
        _validate_derived(payload, result)
        return result

    @staticmethod
    def columns() -> tuple[str, ...]:
        """Return the exact ordered v1 artifact columns."""

        return tuple(BaselineEpisodeRowV1.to_dict(_EMPTY_RESULT))


def planner_summary_to_flat_dict(
    summary: PlannerSummary,
) -> dict[str, object]:
    outcome = summary.outcome
    actions = summary.actions
    timing = summary.timing
    planning = summary.planning
    algorithm = summary.algorithm
    llm = summary.llm
    guard = summary.guard
    reference = summary.reference
    return {
        "planner_name": summary.planner_name,
        "episode_count": outcome.episode_count,
        "success_count": outcome.success_count,
        "success_rate": outcome.success_rate,
        "deadlock_count": outcome.deadlock_count,
        "deadlock_rate": outcome.deadlock_rate,
        "truncated_count": outcome.truncated_count,
        "mean_actions": actions.mean_actions,
        "mean_actions_on_success": actions.mean_actions_on_success,
        "mean_invalid_moves": actions.mean_invalid_moves,
        "mean_elapsed_seconds": timing.mean_elapsed_seconds,
        "p50_elapsed_seconds": timing.p50_elapsed_seconds,
        "p95_elapsed_seconds": timing.p95_elapsed_seconds,
        "total_planning_calls": planning.total_calls,
        "total_planning_retries": planning.total_retries,
        "total_planning_errors": planning.total_errors,
        "mean_planning_elapsed_seconds": planning.mean_elapsed_seconds,
        "total_algorithm_calls": algorithm.total_calls,
        "total_algorithm_requests": algorithm.total_requests,
        "total_algorithm_cache_hits": algorithm.total_cache_hits,
        "total_algorithm_failures": algorithm.total_failures,
        "total_algorithm_fallbacks": algorithm.total_fallbacks,
        "total_algorithm_expanded_states": algorithm.total_expanded_states,
        "mean_algorithm_elapsed_seconds": algorithm.mean_elapsed_seconds,
        "total_llm_calls": llm.total_calls,
        "total_llm_retries": llm.total_retries,
        "total_llm_client_errors": llm.total_client_errors,
        "total_llm_format_errors": llm.total_format_errors,
        "total_llm_invalid_actions": llm.total_invalid_actions,
        "mean_llm_elapsed_seconds": llm.mean_elapsed_seconds,
        "p50_llm_elapsed_seconds": llm.p50_elapsed_seconds,
        "p95_llm_elapsed_seconds": llm.p95_elapsed_seconds,
        "total_llm_prompt_tokens": llm.total_prompt_tokens,
        "total_llm_output_tokens": llm.total_output_tokens,
        "llm_output_tokens_per_second": llm.output_tokens_per_second,
        "mean_pushes_on_success": actions.mean_pushes_on_success,
        "total_revisited_states": actions.total_revisited_states,
        "total_repeated_plans": actions.total_repeated_plans,
        "total_guard_accepted": guard.total_accepted,
        "total_guard_suffix_added": guard.total_suffix_added,
        "total_guard_replaced": guard.total_replaced,
        "total_guard_failed": guard.total_failed,
        "total_guard_proposed_actions": guard.total_proposed_actions,
        "total_guard_legal_prefix_actions": guard.total_legal_prefix_actions,
        "total_guard_adopted_actions": guard.total_adopted_actions,
        "guard_adoption_rate": guard.adoption_rate,
        "total_guard_suffix_expanded_states": (
            guard.total_suffix_expanded_states
        ),
        "total_guard_reference_calls": guard.total_reference_calls,
        "total_guard_reference_expanded_states": (
            guard.total_reference_expanded_states
        ),
        "total_guard_expansions_saved": guard.total_expansions_saved,
        "reference_solved_count": reference.solved_count,
        "mean_action_overhead_vs_reference": (
            reference.mean_action_overhead
        ),
        "mean_push_overhead_vs_reference": reference.mean_push_overhead,
        "mean_policy_elapsed_seconds": timing.mean_policy_elapsed_seconds,
    }


def _llm_usage(payload: dict[str, Any]) -> BaselineLLMUsage:
    return BaselineLLMUsage(
        calls=int(payload.get("llm_calls", 0)),
        retries=int(payload.get("llm_retries", 0)),
        client_errors=int(payload.get("llm_client_errors", 0)),
        format_errors=int(payload.get("llm_format_errors", 0)),
        invalid_actions=int(payload.get("llm_invalid_actions", 0)),
        elapsed_seconds=float(payload.get("llm_elapsed_seconds", 0.0)),
        load_seconds=float(payload.get("llm_load_seconds", 0.0)),
        prompt_eval_seconds=float(
            payload.get("llm_prompt_eval_seconds", 0.0)
        ),
        eval_seconds=float(payload.get("llm_eval_seconds", 0.0)),
        prompt_tokens=int(payload.get("llm_prompt_tokens", 0)),
        output_tokens=int(payload.get("llm_output_tokens", 0)),
    )


def _guard_usage(payload: dict[str, Any]) -> GuardUsage:
    return GuardUsage(
        accepted=int(payload.get("guard_accepted", 0)),
        suffix_added=int(payload.get("guard_suffix_added", 0)),
        replaced=int(payload.get("guard_replaced", 0)),
        failed=int(payload.get("guard_failed", 0)),
        proposed_actions=int(payload.get("guard_proposed_actions", 0)),
        legal_prefix_actions=int(
            payload.get("guard_legal_prefix_actions", 0)
        ),
        adopted_actions=int(payload.get("guard_adopted_actions", 0)),
        suffix_expanded_states=int(
            payload.get("guard_suffix_expanded_states", 0)
        ),
        reference_calls=int(payload.get("guard_reference_calls", 0)),
        reference_action_count=int(
            payload.get("guard_reference_action_count", 0)
        ),
        reference_expanded_states=int(
            payload.get("guard_reference_expanded_states", 0)
        ),
        reference_elapsed_seconds=float(
            payload.get("guard_reference_elapsed_seconds", 0.0)
        ),
        expansions_saved=int(payload.get("guard_expansions_saved", 0)),
    )


def _bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer or null")
    return value


def _validate_derived(
    payload: dict[str, Any],
    result: EpisodeResult,
) -> None:
    expected = {
        "action_overhead_vs_reference": (
            result.action_overhead_vs_reference
        ),
        "push_overhead_vs_reference": result.push_overhead_vs_reference,
        "policy_elapsed_seconds": result.policy_elapsed_seconds,
    }
    for field, value in expected.items():
        if field in payload and payload[field] != value:
            raise ValueError(
                f"{field} does not match values used to derive it"
            )


_EMPTY_RESULT = EpisodeResult(
    identity=PlannerEpisodeIdentity("", "", None),
    outcome=BaselineEpisodeOutcome(False, False, False, 0, 0, 0.0),
)
