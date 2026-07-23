"""Native Ollama client with generation telemetry."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

_NANOSECONDS_PER_SECOND = 1_000_000_000


class OllamaSettings(BaseModel):
    """Ollama connection and bounded generation settings."""

    model_config = ConfigDict(frozen=True)

    api_base: str
    model: str = "qwen3.6:27b-mlx"
    timeout_seconds: float = Field(default=300.0, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    num_ctx: int = Field(default=4096, gt=0)
    max_output_tokens: int = Field(default=128, gt=0, le=512)
    keep_alive: str = "30m"
    think: bool = False

    @field_validator("api_base")
    @classmethod
    def validate_api_base(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            msg = "OLLAMA_API_BASE must start with http:// or https://"
            raise ValueError(msg)
        return normalized

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> OllamaSettings:
        """Load settings from a dotenv file and the process environment."""

        if env_file is not None:
            dotenv_path = str(env_file)
            if dotenv_path == ".env":
                dotenv_path = find_dotenv(filename=".env", usecwd=True)
            load_dotenv(dotenv_path=dotenv_path, override=False)
        api_base = os.getenv("OLLAMA_API_BASE")
        if not api_base:
            raise ValueError("OLLAMA_API_BASE is required; set it in .env")
        return cls(
            api_base=api_base,
            model=os.getenv("OLLAMA_MODEL", "qwen3.6:27b-mlx"),
            timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
            max_output_tokens=int(
                os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "128")
            ),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
            think=_env_bool("OLLAMA_THINK", default=False),
        )


@dataclass(frozen=True, slots=True)
class CompletionMetrics:
    """Timing and token counters returned by Ollama."""

    total_seconds: float = 0.0
    load_seconds: float = 0.0
    prompt_eval_seconds: float = 0.0
    eval_seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class TextCompletion:
    """Structured text plus server-side generation telemetry."""

    content: str
    metrics: CompletionMetrics


class OllamaClient:
    """Small native client for deterministic, observable Ollama calls."""

    def __init__(self, settings: OllamaSettings) -> None:
        self.settings = settings

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> OllamaClient:
        """Create a client from environment variables."""

        return cls(OllamaSettings.from_env(env_file))

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        seed: int | None = None,
        response_schema: Mapping[str, object] | None = None,
    ) -> TextCompletion:
        """Generate one non-streaming response and retain Ollama metrics."""

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        options: dict[str, int | float] = {
            "temperature": self.settings.temperature,
            "num_ctx": self.settings.num_ctx,
            "num_predict": self.settings.max_output_tokens,
        }
        if seed is not None:
            options["seed"] = seed

        payload: dict[str, object] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "think": self.settings.think,
            "keep_alive": self.settings.keep_alive,
            "options": options,
        }
        if response_schema is not None:
            payload["format"] = dict(response_schema)

        raw = _post_json(
            f"{self.settings.api_base}/api/chat",
            payload,
            timeout=self.settings.timeout_seconds,
        )
        message = raw.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("Ollama returned no message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty response")
        return TextCompletion(
            content=content,
            metrics=CompletionMetrics(
                total_seconds=_seconds(raw.get("total_duration")),
                load_seconds=_seconds(raw.get("load_duration")),
                prompt_eval_seconds=_seconds(
                    raw.get("prompt_eval_duration")
                ),
                eval_seconds=_seconds(raw.get("eval_duration")),
                prompt_tokens=_integer(raw.get("prompt_eval_count")),
                output_tokens=_integer(raw.get("eval_count")),
            ),
        )


def _post_json(
    url: str,
    payload: Mapping[str, object],
    *,
    timeout: float,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read())
    if not isinstance(decoded, dict):
        raise RuntimeError("Ollama returned a non-object response")
    return decoded


def _seconds(value: object) -> float:
    return _integer(value) / _NANOSECONDS_PER_SECOND


def _integer(value: object) -> int:
    return value if isinstance(value, int) else 0


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")
