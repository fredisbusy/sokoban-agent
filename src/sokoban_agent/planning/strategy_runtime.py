"""External prompt and model adapters for structured strategy generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langsmith import Client
from langsmith.utils import (
    LangSmithAPIError,
    LangSmithAuthError,
    LangSmithConnectionError,
    LangSmithNotFoundError,
    LangSmithRateLimitError,
    LangSmithRequestTimeout,
    LangSmithUserError,
)

from sokoban_agent.planning.llm import LiteLLMClient, TextCompletion


class TransientAgenticError(RuntimeError):
    """A retryable prompt or model transport failure."""


class PromptConfigurationError(RuntimeError):
    """A non-retryable missing or invalid prompt configuration."""


@dataclass(frozen=True, slots=True)
class PromptReferenceValue:
    """A LangSmith prompt pinned to an immutable commit."""

    name: str
    commit: str


@dataclass(frozen=True, slots=True)
class RenderedStrategyPrompt:
    """Provider-neutral messages rendered from a managed prompt."""

    system_prompt: str
    user_prompt: str


class PromptSource(Protocol):
    """Prompt Management boundary used by observable graph nodes."""

    def resolve(self, name: str, selector: str) -> PromptReferenceValue:
        """Resolve a prompt selector to an immutable commit."""
        ...

    def render(
        self,
        reference: PromptReferenceValue,
        variables: Mapping[str, object],
    ) -> RenderedStrategyPrompt:
        """Render a pinned prompt without persisting its body in graph state."""
        ...


class StrategyGenerator(Protocol):
    """Structured model boundary used by the strategy proposal node."""

    def generate(
        self,
        prompt: RenderedStrategyPrompt,
        *,
        seed: int | None,
        response_schema: Mapping[str, object],
    ) -> TextCompletion:
        """Return one schema-constrained strategy response."""
        ...


class LangSmithPromptSource:
    """Resolve and render versioned prompts through the LangSmith SDK."""

    def __init__(self, client: Client | None = None) -> None:
        self._client = client or Client()

    def resolve(self, name: str, selector: str) -> PromptReferenceValue:
        """Pull a selector and retain the server-resolved commit hash."""

        canonical_name = self._canonical_name(name)
        prompt = self._pull(
            f"{canonical_name}:{selector}",
            skip_cache=True,
        )
        metadata = prompt.metadata or {}
        commit = metadata.get("lc_hub_commit_hash")
        if not isinstance(commit, str) or not commit:
            raise PromptConfigurationError(
                "LangSmith prompt did not expose a resolved commit hash"
            )
        return PromptReferenceValue(name=canonical_name, commit=commit)

    def render(
        self,
        reference: PromptReferenceValue,
        variables: Mapping[str, object],
    ) -> RenderedStrategyPrompt:
        """Render one pinned prompt into Ollama-compatible message text."""

        prompt = self._pull(f"{reference.name}:{reference.commit}")
        value = prompt.invoke(dict(variables))
        system_messages: list[str] = []
        user_messages: list[str] = []
        for message in value.to_messages():
            if not isinstance(message.content, str):
                raise PromptConfigurationError(
                    "strategy prompt messages must contain text"
                )
            if message.type == "system":
                system_messages.append(message.content)
            else:
                user_messages.append(message.content)
        if not system_messages or not user_messages:
            raise PromptConfigurationError(
                "strategy prompt requires system and user messages"
            )
        return RenderedStrategyPrompt(
            system_prompt="\n\n".join(system_messages),
            user_prompt="\n\n".join(user_messages),
        )

    def _pull(
        self,
        identifier: str,
        *,
        skip_cache: bool = False,
    ) -> BasePromptTemplate[PromptValue]:
        try:
            prompt = self._client.pull_prompt(
                identifier,
                skip_cache=skip_cache,
            )
        except (
            LangSmithAPIError,
            LangSmithConnectionError,
            LangSmithRateLimitError,
            LangSmithRequestTimeout,
        ) as error:
            raise TransientAgenticError(
                "LangSmith prompt request failed temporarily"
            ) from error
        except (
            LangSmithAuthError,
            LangSmithNotFoundError,
            LangSmithUserError,
        ) as error:
            raise PromptConfigurationError(
                f"LangSmith prompt configuration failed: {type(error).__name__}"
            ) from error
        return cast(BasePromptTemplate[PromptValue], prompt)

    def _canonical_name(self, name: str) -> str:
        if "/" in name:
            return name
        try:
            prompt = self._client.get_prompt(name)
        except (
            LangSmithAPIError,
            LangSmithConnectionError,
            LangSmithRateLimitError,
            LangSmithRequestTimeout,
        ) as error:
            raise TransientAgenticError(
                "LangSmith prompt metadata request failed temporarily"
            ) from error
        except (LangSmithAuthError, LangSmithUserError) as error:
            raise PromptConfigurationError(
                f"LangSmith prompt configuration failed: {type(error).__name__}"
            ) from error
        full_name = prompt.full_name if prompt is not None else None
        if not isinstance(full_name, str) or not full_name:
            raise PromptConfigurationError(
                f"LangSmith prompt {name!r} does not exist"
            )
        return full_name


class LiteLLMStrategyGenerator:
    """Generate a structured strategy through LiteLLM and LangChain."""

    def __init__(self, client: LiteLLMClient | None = None) -> None:
        self._client = client

    def generate(
        self,
        prompt: RenderedStrategyPrompt,
        *,
        seed: int | None,
        response_schema: Mapping[str, object],
    ) -> TextCompletion:
        """Call Ollama while translating transport failures for retry policy."""

        client = self._client or LiteLLMClient.from_env()
        try:
            return client.complete(
                prompt.user_prompt,
                system_prompt=prompt.system_prompt,
                seed=seed,
                response_schema=response_schema,
                max_output_tokens=client.settings.strategy_max_output_tokens,
            )
        except (OSError, RuntimeError) as error:
            raise TransientAgenticError(
                "Ollama strategy request failed temporarily"
            ) from error
