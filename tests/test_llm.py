from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sokoban_agent.planning.llm import LiteLLMClient, OllamaSettings


def test_settings_are_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_API_BASE", "http://ollama.local:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.6:27b-mlx")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "2048")
    monkeypatch.setenv("OLLAMA_MAX_OUTPUT_TOKENS", "64")
    monkeypatch.setenv("OLLAMA_STRATEGY_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "1h")
    monkeypatch.setenv("OLLAMA_THINK", "true")

    settings = OllamaSettings.from_env(env_file=None)

    assert settings.api_base == "http://ollama.local:11434"
    assert settings.model == "qwen3.6:27b-mlx"
    assert settings.timeout_seconds == 30
    assert settings.temperature == 0
    assert settings.num_ctx == 2048
    assert settings.max_output_tokens == 64
    assert settings.strategy_max_output_tokens == 1024
    assert settings.keep_alive == "1h"
    assert settings.think


def test_structured_output_default_allows_complete_strategy_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_API_BASE", "http://ollama.local:11434")
    monkeypatch.delenv("OLLAMA_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv(
        "OLLAMA_STRATEGY_MAX_OUTPUT_TOKENS",
        raising=False,
    )

    settings = OllamaSettings.from_env(env_file=None)

    assert settings.max_output_tokens == 512
    assert settings.strategy_max_output_tokens == 2048


def test_client_uses_langchain_chat_model_with_structured_output() -> None:
    captured: dict[str, Any] = {}

    class ChatModelFake:
        def invoke(
            self,
            input: object,
            **kwargs: object,
        ) -> AIMessage:
            captured.update(messages=input, kwargs=kwargs)
            return AIMessage(
                content='{"actions":["UP"]}',
                usage_metadata={
                    "input_tokens": 42,
                    "output_tokens": 8,
                    "total_tokens": 50,
                },
            )

    settings = OllamaSettings(
        api_base="http://localhost:11434",
        model="qwen3.6:27b-mlx",
        timeout_seconds=45,
        num_ctx=2048,
        max_output_tokens=64,
    )
    schema = {"type": "object"}

    result = LiteLLMClient(settings, model=ChatModelFake()).complete(
        "다음 계획은?",
        system_prompt="JSON으로 답해.",
        seed=7,
        response_schema=schema,
        max_output_tokens=1024,
    )

    assert result.content == '{"actions":["UP"]}'
    assert result.metrics.prompt_tokens == 42
    assert result.metrics.output_tokens == 8
    assert captured["messages"] == [
        SystemMessage(content="JSON으로 답해."),
        HumanMessage(content="다음 계획은?"),
    ]
    assert captured["kwargs"] == {
        "response_format": schema,
        "seed": 7,
        "max_tokens": 1024,
    }


def test_api_base_requires_http_scheme() -> None:
    with pytest.raises(ValueError, match="must start with"):
        OllamaSettings(api_base="localhost:11434")


def test_boolean_environment_rejects_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_API_BASE", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_THINK", "sometimes")

    with pytest.raises(ValueError, match="boolean"):
        OllamaSettings.from_env(env_file=None)
