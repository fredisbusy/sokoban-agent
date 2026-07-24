from dataclasses import fields

from sokoban_agent import evaluation
from sokoban_agent.evaluation import agentic_results, research_results, results
from sokoban_agent.evaluation.schemas.episode import (
    AgenticEpisodeResult,
    EpisodeResult,
    PlannerSummary,
)
from sokoban_agent.evaluation.schemas.research import (
    ResearchEpisodeRecord,
    ResearchExperiment,
)
from sokoban_agent.graph.agentic_state import AgenticState
from sokoban_agent.planning import PlanningOutcome


def test_evaluation_public_api_reexports_schema_contracts() -> None:
    assert evaluation.AgenticEpisodeResult is AgenticEpisodeResult
    assert evaluation.EpisodeResult is EpisodeResult
    assert evaluation.PlannerSummary is PlannerSummary
    assert evaluation.ResearchEpisodeRecord is ResearchEpisodeRecord
    assert evaluation.ResearchExperiment is ResearchExperiment


def test_legacy_result_modules_reexport_schema_contracts() -> None:
    assert agentic_results.AgenticEpisodeResult is AgenticEpisodeResult
    assert results.EpisodeResult is EpisodeResult
    assert results.PlannerSummary is PlannerSummary
    assert research_results.ResearchEpisodeRecord is ResearchEpisodeRecord
    assert research_results.ResearchExperiment is ResearchExperiment


def test_internal_result_contracts_are_composed_not_flat_metric_bags() -> None:
    assert {field.name for field in fields(PlanningOutcome)} == {
        "actions",
        "proposed_actions",
        "narrative",
        "failure",
        "llm",
        "algorithm",
        "guard",
        "elapsed_seconds",
    }
    assert {field.name for field in fields(AgenticEpisodeResult)} == {
        "identity",
        "outcome",
        "strategy",
        "llm",
        "memory",
        "local_search",
        "rules",
        "prompt",
        "elapsed_seconds",
    }
    assert "metrics" in AgenticState.__required_keys__
    assert "llm_calls" not in AgenticState.__required_keys__
    assert "rule_checks" not in AgenticState.__required_keys__
