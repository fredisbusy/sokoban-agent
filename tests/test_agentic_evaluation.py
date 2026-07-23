import json
from collections.abc import Mapping

from sokoban_agent.evaluation import run_agentic_episode
from sokoban_agent.planning.llm import CompletionMetrics, TextCompletion
from sokoban_agent.planning.strategy_runtime import (
    PromptReferenceValue,
    RenderedStrategyPrompt,
)


class FixturePromptSource:
    def resolve(self, name: str, selector: str) -> PromptReferenceValue:
        return PromptReferenceValue(name, selector)

    def render(
        self,
        reference: PromptReferenceValue,
        variables: Mapping[str, object],
    ) -> RenderedStrategyPrompt:
        del reference, variables
        return RenderedStrategyPrompt("system", "user")


class DirectPushGenerator:
    def __init__(self, *, column: int = 2) -> None:
        self.column = column

    def generate(
        self,
        prompt: RenderedStrategyPrompt,
        *,
        seed: int | None,
        response_schema: Mapping[str, object],
    ) -> TextCompletion:
        del prompt, seed, response_schema
        payload = {
            "summary": "B1을 T1으로 올린다",
            "assignments": [
                {
                    "box_id": "B1",
                    "target_id": "T1",
                    "reason": "목표가 바로 위에 있다",
                }
            ],
            "protected_constraints": [],
            "subgoal": {
                "kind": "push",
                "box_id": "B1",
                "target_id": "T1",
                "direction": "UP",
                "destination": {"row": 1, "col": self.column},
            },
            "expected_effect": {
                "box_id": "B1",
                "from_position": {"row": 2, "col": self.column},
                "to_position": {"row": 1, "col": self.column},
            },
            "failure_conditions": [
                {
                    "kind": "unexpected_state",
                    "description": "상자가 목표로 이동하지 않는다",
                }
            ],
        }
        return TextCompletion(json.dumps(payload), CompletionMetrics())


def test_agentic_evaluation_reads_metrics_from_shared_graph_state() -> None:
    result = run_agentic_episode(
        {
            "level_id": "held-out-one-push",
            "level_rows": [
                "#######",
                "#  .  #",
                "#  $  #",
                "#  @  #",
                "#######",
            ],
            "seed": 7,
            "max_steps": 15,
        },
        context={
            "prompt_name": "sokoban-strategy",
            "prompt_commit": "fixture-commit",
            "model_name": "fixture-model",
            "rationale_mode": "on",
            "grounding_mode": "local-search",
        },
        prompt_source=FixturePromptSource(),
        strategy_generator=DirectPushGenerator(column=3),
        thread_id="agentic-evaluation-contract",
    )

    assert result.success
    assert result.level_id == "held-out-one-push"
    assert result.action_count == 1
    assert result.push_count == 1
    assert result.strategy_proposals == 1
    assert result.llm_calls == 1
    assert result.local_search_calls == 1
    assert result.local_expanded_states > 0
    assert result.local_search_elapsed_seconds >= 0
    assert result.rule_checks >= 3
    assert result.reachability_calls >= 2
    assert result.subgoal_attempts == 1
    assert result.subgoal_successes == 1
    assert result.subgoal_failures == 0
    assert result.actions_derived_from_subgoal == 1
    assert result.effect_matches == 1
    assert result.algorithm_calls == 0
    assert result.prompt_commit == "fixture-commit"


def test_structured_llm_ablation_uses_direct_push_without_local_search() -> None:
    result = run_agentic_episode(
        {"level_id": "tiny-push", "seed": 7, "max_steps": 15},
        context={
            "prompt_name": "sokoban-strategy",
            "prompt_commit": "fixture-commit",
            "model_name": "fixture-model",
            "rationale_mode": "on",
            "grounding_mode": "direct",
        },
        prompt_source=FixturePromptSource(),
        strategy_generator=DirectPushGenerator(),
        thread_id="agentic-direct-grounding",
    )

    assert result.policy_name == "structured-llm"
    assert result.success
    assert result.local_search_calls == 0
    assert result.local_expanded_states == 0
