"""LiteLLM-backed LangChain chat model configuration and telemetry."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel, ConfigDict, Field, field_validator


class OllamaSettings(BaseModel):
    """Ollama connection and bounded generation settings."""

    model_config = ConfigDict(frozen=True)

    api_base: str
    model: str = "qwen3.6:27b-mlx"
    timeout_seconds: float = Field(default=300.0, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    num_ctx: int = Field(default=4096, gt=0)
    max_output_tokens: int = Field(default=512, gt=0, le=512)
    strategy_max_output_tokens: int = Field(default=2048, gt=0, le=4096)
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
                os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "512")
            ),
            strategy_max_output_tokens=int(
                os.getenv("OLLAMA_STRATEGY_MAX_OUTPUT_TOKENS", "2048")
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


class ChatModel(Protocol):
    """Minimal LangChain chat-model boundary used by tests and adapters."""

    def invoke(
        self,
        input: object,
        **kwargs: object,
    ) -> AIMessage:
        """Invoke a LangChain-compatible chat model."""
        ...


class LiteLLMClient:
    """Use the official LiteLLM LangChain integration for model calls."""

    def __init__(
        self,
        settings: OllamaSettings,
        *,
        model: ChatModel | None = None,
    ) -> None:
        self.settings = settings
        self.model = model or cast(ChatModel, _chat_model(settings))

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = ".env",
    ) -> LiteLLMClient:
        """Create a client from environment variables."""

        return cls(OllamaSettings.from_env(env_file))

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        seed: int | None = None,
        response_schema: Mapping[str, object] | None = None,
        max_output_tokens: int | None = None,
    ) -> TextCompletion:
        """Generate through LangChain and normalize standard usage metadata."""

        output_tokens = (
            self.settings.max_output_tokens
            if max_output_tokens is None
            else max_output_tokens
        )
        if output_tokens < 1 or output_tokens > self.settings.num_ctx:
            raise ValueError(
                "max_output_tokens must be between 1 and num_ctx"
            )
        messages: list[SystemMessage | HumanMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        invocation: dict[str, object] = {"max_tokens": output_tokens}
        if seed is not None:
            invocation["seed"] = seed
        if response_schema is not None:
            invocation["response_format"] = dict(response_schema)
        message = self.model.invoke(messages, **invocation)
        content = message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LiteLLM returned an empty text response")
        usage = message.usage_metadata
        return TextCompletion(
            content=content,
            metrics=CompletionMetrics(
                prompt_tokens=(
                    _integer(usage.get("input_tokens")) if usage else 0
                ),
                output_tokens=(
                    _integer(usage.get("output_tokens")) if usage else 0
                ),
            ),
        )


def _chat_model(settings: OllamaSettings) -> ChatLiteLLM:
    model = (
        settings.model
        if "/" in settings.model
        else f"ollama/{settings.model}"
    )
    return ChatLiteLLM(
        model=model,
        api_base=settings.api_base,
        request_timeout=settings.timeout_seconds,
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
        num_ctx=settings.num_ctx,
        model_kwargs={
            "keep_alive": settings.keep_alive,
            "think": settings.think,
        },
    )


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
