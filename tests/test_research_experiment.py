import json
from collections.abc import Mapping

from sokoban_agent.evaluation.research.cohort import (
    AgenticCohortManifest,
    AgenticLevelCase,
)
from sokoban_agent.evaluation.research.experiment import (
    ResearchRunConfig,
    run_research_experiment,
)
from sokoban_agent.planning import AStarPlanner
from sokoban_agent.planning.agentic.runtime import (
    PromptReferenceValue,
    RenderedStrategyPrompt,
)
from sokoban_agent.planning.llm.client import CompletionMetrics, TextCompletion


class FixturePromptSource:
    def resolve(self, name: str, selector: str) -> PromptReferenceValue:
        return PromptReferenceValue(name, selector)

    def render(
        self,
        reference: PromptReferenceValue,
        variables: Mapping[str, object],
    ) -> RenderedStrategyPrompt:
        return RenderedStrategyPrompt("system", json.dumps(variables))


class DirectPushGenerator:
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
                "destination": {"row": 1, "col": 3},
            },
            "expected_effect": {
                "box_id": "B1",
                "from_position": {"row": 2, "col": 3},
                "to_position": {"row": 1, "col": 3},
            },
            "failure_conditions": [
                {
                    "kind": "unexpected_state",
                    "description": "상자가 목표로 이동하지 않는다",
                }
            ],
        }
        return TextCompletion(
            json.dumps(payload),
            CompletionMetrics(prompt_tokens=10, output_tokens=20),
        )


def _manifest() -> AgenticCohortManifest:
    case = AgenticLevelCase(
        level_id="eval-fixture-direct",
        rows=(
            "#######",
            "#  .  #",
            "#  $  #",
            "#  @  #",
            "#######",
        ),
        height=5,
        width=7,
        box_count=1,
        layout_family="open-room",
        corridor_structure="none",
        trap_types=(),
        oracle_expectation="solved",
        difficulty="fixture",
    )
    return AgenticCohortManifest(
        schema_version=1,
        version="fixture-v1",
        split="test",
        description="fixture",
        development_level_ids=("tiny-push",),
        cohort_sha256="fixture-sha",
        levels=(case,),
    )


def test_research_experiment_runs_six_policies_on_identical_cases() -> None:
    experiment = run_research_experiment(
        _manifest(),
        ResearchRunConfig(
            prompt_name="sokoban-strategy",
            prompt_commit="fixture-commit",
            graph_id="sokoban_agent",
            graph_revision="fixture-graph",
            model_name="fixture-model",
            seeds=(7,),
            max_steps=15,
            oracle_max_expanded_states=10_000,
        ),
        primitive_planner=AStarPlanner(),
        full_guard_planner=AStarPlanner(),
        prompt_source=FixturePromptSource(),
        strategy_generator=DirectPushGenerator(),
    )

    assert {record.policy_name for record in experiment.records} == {
        "primitive-llm",
        "structured-llm",
        "structured-local-search",
        "structured-no-rationale",
        "current-full-guard",
        "astar-oracle",
    }
    assert {
        (record.level_id, record.seed) for record in experiment.records
    } == {("eval-fixture-direct", 7)}
    by_policy = {
        record.policy_name: record for record in experiment.records
    }
    direct_search = by_policy["structured-llm"].local_search
    local_search = by_policy["structured-local-search"].local_search
    oracle_search = by_policy["astar-oracle"].algorithm
    assert direct_search is not None and direct_search.calls == 0
    assert local_search is not None and local_search.calls == 1
    assert oracle_search is not None and oracle_search.calls == 1
    assert by_policy["astar-oracle"].outcome.success
    assert {record.level.difficulty for record in experiment.records} == {
        "fixture"
    }
    assert experiment.run_manifest["prompt_commit"] == "fixture-commit"
    assert experiment.run_manifest["cohort_sha256"] == "fixture-sha"
    assert len(experiment.summaries) == 6
    assert experiment.rationale_intervention.compared_cases == 1
    assert experiment.rationale_intervention.action_sequence_changes == 0
    payload = experiment.to_json_dict()
    assert payload["record_schema_version"] == 2
    records = payload["records"]
    assert isinstance(records, list)
    structured = next(
        record
        for record in records
        if record["policy_name"] == "structured-local-search"
    )
    assert structured["cycle_detected"] is False
    assert structured["prompt_commit"] == "fixture-commit"
    assert "strategy_schema_rejections" in structured
    assert "strategy_cache_hits" in structured
    assert structured["assignment_revision_count"] == 0
    assert "actions_derived_from_subgoal" not in structured
    oracle = next(
        record
        for record in records
        if record["policy_name"] == "astar-oracle"
    )
    assert oracle["strategy_proposals"] is None
    assert oracle["memory_requests"] is None
    assert oracle["algorithm_calls"] == 1
    assert not any(isinstance(value, dict) for value in structured.values())
    json.dumps(payload)


def test_research_run_requires_an_immutable_prompt_commit() -> None:
    try:
        ResearchRunConfig(
            prompt_name="sokoban-strategy",
            prompt_commit="unresolved",
            graph_id="sokoban_agent",
            graph_revision="fixture-graph",
            model_name="fixture-model",
            seeds=(0,),
        )
    except ValueError as error:
        assert "prompt commit" in str(error)
    else:
        raise AssertionError("unresolved prompt commit was accepted")
