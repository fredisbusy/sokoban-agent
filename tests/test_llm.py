from typing import Any

import pytest

from sokoban_agent.planning import llm
from sokoban_agent.planning.llm import OllamaClient, OllamaSettings


def test_settings_are_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_API_BASE", "http://ollama.local:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.6:27b-mlx")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "2048")
    monkeypatch.setenv("OLLAMA_MAX_OUTPUT_TOKENS", "64")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "1h")
    monkeypatch.setenv("OLLAMA_THINK", "true")

    settings = OllamaSettings.from_env(env_file=None)

    assert settings.api_base == "http://ollama.local:11434"
    assert settings.model == "qwen3.6:27b-mlx"
    assert settings.timeout_seconds == 30
    assert settings.temperature == 0
    assert settings.num_ctx == 2048
    assert settings.max_output_tokens == 64
    assert settings.keep_alive == "1h"
    assert settings.think


def test_structured_output_default_allows_complete_strategy_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_API_BASE", "http://ollama.local:11434")
    monkeypatch.delenv("OLLAMA_MAX_OUTPUT_TOKENS", raising=False)

    settings = OllamaSettings.from_env(env_file=None)

    assert settings.max_output_tokens == 512


def test_client_passes_native_ollama_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post_json(
        url: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        captured.update(url=url, payload=payload, timeout=timeout)
        return {
            "message": {"content": '{"actions":["UP"]}'},
            "total_duration": 2_000_000_000,
            "load_duration": 100_000_000,
            "prompt_eval_count": 42,
            "prompt_eval_duration": 500_000_000,
            "eval_count": 8,
            "eval_duration": 800_000_000,
        }

    monkeypatch.setattr(llm, "_post_json", fake_post_json)
    settings = OllamaSettings(
        api_base="http://localhost:11434",
        model="qwen3.6:27b-mlx",
        timeout_seconds=45,
        num_ctx=2048,
        max_output_tokens=64,
    )
    schema = {"type": "object"}

    result = OllamaClient(settings).complete(
        "다음 계획은?",
        system_prompt="JSON으로 답해.",
        seed=7,
        response_schema=schema,
    )

    assert result.content == '{"actions":["UP"]}'
    assert result.metrics.total_seconds == 2
    assert result.metrics.prompt_tokens == 42
    assert result.metrics.output_tokens == 8
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["timeout"] == 45
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen3.6:27b-mlx"
    assert payload["think"] is False
    assert payload["keep_alive"] == "30m"
    assert payload["format"] == schema
    assert payload["messages"] == [
        {"role": "system", "content": "JSON으로 답해."},
        {"role": "user", "content": "다음 계획은?"},
    ]
    assert payload["options"] == {
        "temperature": 0,
        "num_ctx": 2048,
        "num_predict": 64,
        "seed": 7,
    }


def test_https_client_uses_verified_certifi_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    trusted_context = object()

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self) -> bytes:
            return b'{"message":{"content":"ok"}}'

    def fake_context(*, cafile: str) -> object:
        captured["cafile"] = cafile
        return trusted_context

    def fake_urlopen(
        request: object,
        *,
        timeout: float,
        context: object,
    ) -> FakeResponse:
        captured.update(request=request, timeout=timeout, context=context)
        return FakeResponse()

    monkeypatch.setattr(
        "sokoban_agent.planning.llm.certifi.where",
        lambda: "/trusted/cacert.pem",
    )
    monkeypatch.setattr(
        "sokoban_agent.planning.llm.ssl.create_default_context",
        fake_context,
    )
    monkeypatch.setattr(llm, "urlopen", fake_urlopen)

    response = llm._post_json(
        "https://ollama.example/api/chat",
        {"model": "fixture"},
        timeout=12,
    )

    assert response == {"message": {"content": "ok"}}
    assert captured["cafile"] == "/trusted/cacert.pem"
    assert captured["context"] is trusted_context
    assert captured["timeout"] == 12


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
