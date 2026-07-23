from collections import deque
from collections.abc import Mapping

import pytest

from sokoban_agent.agents import (
    Agent,
    AgentStopped,
    LLMAgent,
    Observation,
    parse_action_response,
    serialize_board,
)
from sokoban_agent.env import Action, SokobanEnv
from sokoban_agent.evaluation import run_episode


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


def test_llm_agent_retries_format_and_blocked_action() -> None:
    observation, info = _initial_state()
    client = SequenceClient(["", '{"action":"DOWN"}', '{"action":"UP"}'])
    agent = LLMAgent(
        client,
        model_name="test-model",
        max_attempts=3,
    )
    accepted_agent: Agent = agent
    accepted_agent.reset(observation, info, seed=17)

    action = accepted_agent.act(observation, info)

    assert action is Action.UP
    assert agent.diagnostics.llm_calls == 3
    assert agent.diagnostics.llm_retries == 2
    assert agent.diagnostics.llm_format_errors == 1
    assert agent.diagnostics.llm_invalid_actions == 1
    assert client.requests[0]["seed"] == 17
    assert client.requests[0]["response_format"] is not None
    assert "Rejected attempts:" in str(client.requests[2]["prompt"])


def test_llm_agent_stops_after_client_errors_exhaust_attempts() -> None:
    observation, info = _initial_state()
    agent = LLMAgent(
        SequenceClient([RuntimeError("offline"), RuntimeError("offline")]),
        model_name="test-model",
        max_attempts=2,
    )
    agent.reset(observation, info)

    with pytest.raises(AgentStopped, match="after 2 attempts"):
        agent.act(observation, info)

    assert agent.diagnostics.llm_calls == 2
    assert agent.diagnostics.llm_retries == 1
    assert agent.diagnostics.llm_client_errors == 2


def test_episode_records_llm_diagnostics() -> None:
    env = SokobanEnv()
    agent = LLMAgent(
        SequenceClient(['{"action":"UP"}']),
        model_name="test-model",
    )

    result = run_episode(env, agent, seed=3, level_id="tiny-push")

    assert result.success
    assert result.llm_calls == 1
    assert result.llm_retries == 0
    assert result.llm_elapsed_seconds >= 0
