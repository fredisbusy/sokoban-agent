from collections import deque
from collections.abc import Mapping

import pytest

from sokoban_agent.env import Action, SokobanEnv
from sokoban_agent.evaluation import run_episode
from sokoban_agent.planning import (
    LLMPlanner,
    Observation,
    PlanningContext,
    parse_action_response,
    serialize_board,
)


class SequenceClient:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = deque(responses)
        self.requests: list[dict[str, object]] = []

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        seed: int | None = None,
        response_format: Mapping[str, object] | None = None,
    ) -> str:
        self.requests.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "seed": seed,
                "response_format": response_format,
            }
        )
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


def _initial_state() -> tuple[Observation, dict[str, object]]:
    env = SokobanEnv()
    observation, info = env.reset(options={"level_id": "tiny-push"})
    env.close()
    return observation, info


def test_board_is_serialized_with_standard_sokoban_symbols() -> None:
    observation, _ = _initial_state()

    assert serialize_board(observation) == (
        "#####\n"
        "# . #\n"
        "# $ #\n"
        "# @ #\n"
        "#####"
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"action":"UP"}', Action.UP),
        ('{"action":"RIGHT"}', Action.RIGHT),
    ],
)
def test_structured_action_response_is_parsed(
    response: str,
    expected: Action,
) -> None:
    assert parse_action_response(response) is expected


@pytest.mark.parametrize(
    "response",
    ["", "UP", '{"action":"JUMP"}', '{"action":"UP","reason":"x"}'],
)
def test_invalid_action_response_is_rejected(response: str) -> None:
    with pytest.raises(ValueError):
        parse_action_response(response)


def test_llm_planner_reports_format_error_without_retrying_itself() -> None:
    observation, info = _initial_state()
    client = SequenceClient([""])
    planner = LLMPlanner(client, model_name="test-model")
    planner.reset(seed=17)

    outcome = planner.plan(PlanningContext(observation, info, (), (), 17))

    assert outcome.actions == ()
    assert outcome.error_kind == "format"
    assert outcome.llm_calls == 1
    assert outcome.llm_format_errors == 1
    assert client.requests[0]["seed"] == 17


def test_graph_retries_format_and_blocked_llm_proposals() -> None:
    client = SequenceClient(["", '{"action":"DOWN"}', '{"action":"UP"}'])
    planner = LLMPlanner(client, model_name="test-model")

    result = run_episode(
        SokobanEnv(),
        planner,
        seed=17,
        level_id="tiny-push",
        max_planning_attempts=3,
    )

    assert result.success
    assert result.llm_calls == 3
    assert result.planning_retries == 2
    assert result.llm_format_errors == 1
    assert result.llm_invalid_actions == 1
    assert "Rejected proposals:" in str(client.requests[2]["prompt"])


def test_graph_stops_after_client_errors_exhaust_attempts() -> None:
    planner = LLMPlanner(
        SequenceClient([RuntimeError("offline"), RuntimeError("offline")]),
        model_name="test-model",
    )

    result = run_episode(
        SokobanEnv(),
        planner,
        level_id="tiny-push",
        max_planning_attempts=2,
    )

    assert not result.success
    assert result.failure_reason == "request failed: RuntimeError"
    assert result.llm_calls == 2
    assert result.llm_client_errors == 2
