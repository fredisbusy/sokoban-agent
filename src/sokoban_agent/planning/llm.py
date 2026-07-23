"""LiteLLM client configuration for Ollama."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from litellm import completion
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator


class OllamaSettings(BaseModel):
    """Ollama connection settings loaded from environment variables."""

    model_config = ConfigDict(frozen=True)

    api_base: str
    model: str = "llama3.2"
    timeout_seconds: float = Field(default=120.0, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @field_validator("api_base")
    @classmethod
    def validate_api_base(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            msg = "OLLAMA_API_BASE must start with http:// or https://"
            raise ValueError(msg)
        return normalized

    @property
    def litellm_model(self) -> str:
        """Return the provider-prefixed model name expected by LiteLLM."""

        if self.model.startswith("ollama/"):
            return self.model
        return f"ollama/{self.model}"

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> OllamaSettings:
        """Load settings from a dotenv file and the process environment."""

        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)
        api_base = os.getenv("OLLAMA_API_BASE")
        if not api_base:
            raise ValueError("OLLAMA_API_BASE is required; set it in .env")
        return cls(
            api_base=api_base,
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
        )


class OllamaClient:
    """Provider-neutral text client backed by LiteLLM and Ollama."""

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
        response_format: Mapping[str, object] | None = None,
    ) -> str:
        """Generate one text response from Ollama."""

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.settings.litellm_model,
            "messages": messages,
            "api_base": self.settings.api_base,
            "timeout": self.settings.timeout_seconds,
            "temperature": self.settings.temperature,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if response_format is not None:
            kwargs["response_format"] = dict(response_format)

        response = cast(ModelResponse, completion(**kwargs))
        if not response.choices:
            raise RuntimeError("Ollama returned no choices")
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Ollama returned an empty response")
        return content
