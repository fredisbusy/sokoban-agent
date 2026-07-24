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
