from typing import Any

import pytest

from sokoban_agent.env import Action
from sokoban_agent.graph.studio import graph
from sokoban_agent.planning import LLMPlanner, PlanningOutcome
from sokoban_agent.planning.llm import OllamaSettings


def test_studio_graph_exposes_each_decision_stage() -> None:
    mermaid = graph.get_graph().draw_mermaid()

    assert "initialize" in mermaid
    assert "llm_plan" in mermaid
    assert "astar_guard" in mermaid
    assert "validate_plan" in mermaid
    assert "execute_action" in mermaid


def test_studio_graph_records_korean_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = OllamaSettings(
        api_base="http://ollama.test",
        model="test-model",
    )

    def fake_from_env(cls: type[OllamaSettings]) -> OllamaSettings:
        del cls
        return settings

    def fake_plan(self: Any, context: Any) -> PlanningOutcome:
        del self, context
        return PlanningOutcome(
            actions=(Action.UP,),
            proposed_actions=(Action.UP,),
            goal="상자를 위쪽 목표로 이동",
            decision_summary="상자 아래에서 위로 밀면 바로 해결됩니다",
            risk="벽 방향으로 밀지 않습니다",
            llm_calls=1,
        )

    monkeypatch.setattr(
        OllamaSettings,
        "from_env",
        classmethod(fake_from_env),
    )
    monkeypatch.setattr(LLMPlanner, "plan", fake_plan)

    result = graph.invoke(
        {"level_id": "tiny-push", "seed": 0, "max_steps": 5},
        {"recursion_limit": 30},
    )

    assert result["success"]
    assert result["status"] == "성공"
    assert result["action_history"] == ["UP"]
    assert result["planner_goal"] == "상자를 위쪽 목표로 이동"
    assert result["decision_summary"].endswith("해결됩니다")
    stages = [event["stage"] for event in result["decision_log"]]
    assert stages == ["초기화", "LLM 계획", "A* 검사", "계획 검증", "행동 실행"]
