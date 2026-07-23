from types import SimpleNamespace
from typing import Any

import pytest

from sokoban_agent.agents import llm
from sokoban_agent.agents.llm import OllamaClient, OllamaSettings


def test_settings_are_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_API_BASE", "http://ollama.local:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "30")

    settings = OllamaSettings.from_env(env_file=None)

    assert settings.api_base == "http://ollama.local:11434"
    assert settings.litellm_model == "ollama/qwen3:8b"
    assert settings.timeout_seconds == 30
    assert settings.temperature == 0


def test_client_passes_ollama_configuration_to_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        message = SimpleNamespace(content="UP")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(llm, "completion", fake_completion)
    settings = OllamaSettings(
        api_base="http://localhost:11434",
        model="llama3.2",
        timeout_seconds=45,
    )

    result = OllamaClient(settings).complete(
        "다음 행동은?",
        system_prompt="한 단어로 답해.",
    )

    assert result == "UP"
    assert captured["model"] == "ollama/llama3.2"
    assert captured["api_base"] == "http://localhost:11434"
    assert captured["timeout"] == 45
    assert captured["temperature"] == 0
    assert captured["messages"] == [
        {"role": "system", "content": "한 단어로 답해."},
        {"role": "user", "content": "다음 행동은?"},
    ]


def test_client_passes_seed_and_structured_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        message = SimpleNamespace(content='{"action":"UP"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(llm, "completion", fake_completion)
    client = OllamaClient(
        OllamaSettings(
            api_base="http://localhost:11434",
            temperature=0.25,
        )
    )
    response_format = {
        "type": "json_schema",
        "json_schema": {"schema": {"type": "object"}},
    }

    client.complete("다음 행동은?", seed=7, response_format=response_format)

    assert captured["seed"] == 7
    assert captured["temperature"] == 0.25
    assert captured["response_format"] == response_format


def test_api_base_requires_http_scheme() -> None:
    with pytest.raises(ValueError, match="must start with"):
        OllamaSettings(api_base="localhost:11434")
