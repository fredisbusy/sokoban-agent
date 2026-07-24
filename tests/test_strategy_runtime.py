from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts.structured import StructuredPrompt
from langsmith import Client

import sokoban_agent.planning.agentic.runtime as runtime_module
from sokoban_agent.planning.agentic.runtime import (
    LangSmithPromptSource,
    LiteLLMStrategyGenerator,
    PromptConfigurationError,
    PromptReferenceValue,
    RenderedStrategyPrompt,
)
from sokoban_agent.planning.llm.client import (
    CompletionMetrics,
    LiteLLMClient,
    OllamaSettings,
    TextCompletion,
)


class PromptClientFake:
    def __init__(self, prompt: StructuredPrompt) -> None:
        self.prompt = prompt
        self.identifiers: list[tuple[str, bool]] = []
        self.prompt_lookups: list[str] = []

    def get_prompt(self, identifier: str) -> Any:
        self.prompt_lookups.append(identifier)
        return SimpleNamespace(full_name="workspace/sokoban-strategy")

    def pull_prompt(
        self,
        identifier: str,
        *,
        skip_cache: bool = False,
    ) -> Any:
        self.identifiers.append((identifier, skip_cache))
        return self.prompt


def test_langsmith_prompt_source_pins_commit_and_renders_messages() -> None:
    prompt = cast(
        StructuredPrompt,
        StructuredPrompt.from_messages_and_schema(
            [
                ("system", "system {level_id}"),
                ("human", "analysis {board_analysis_json}"),
            ],
            schema={"type": "object"},
        ),
    )
    prompt.metadata = {"lc_hub_commit_hash": "abc123def456"}
    client = PromptClientFake(prompt)
    source = LangSmithPromptSource(cast(Client, client))

    reference = source.resolve("sokoban-strategy", "production")
    rendered = source.render(
        reference,
        {
            "level_id": "tiny-push",
            "board_analysis_json": "{}",
        },
    )

    assert reference == PromptReferenceValue(
        name="workspace/sokoban-strategy",
        commit="abc123def456",
    )
    assert rendered.system_prompt == "system tiny-push"
    assert rendered.user_prompt == "analysis {}"
    assert client.identifiers == [
        ("workspace/sokoban-strategy:production", True),
        ("workspace/sokoban-strategy:abc123def456", False),
    ]
    assert client.prompt_lookups == ["sokoban-strategy"]


def test_strategy_generator_uses_the_larger_strategy_token_budget(
) -> None:
    captured: dict[str, Any] = {}

    class ChatModelFake:
        def invoke(self, input: object, **kwargs: object) -> AIMessage:
            captured.update(messages=input, kwargs=kwargs)
            return AIMessage(content="{}")

    settings = OllamaSettings(
        api_base="http://localhost:11434",
        num_ctx=4096,
        max_output_tokens=64,
        strategy_max_output_tokens=1536,
    )
    generator = LiteLLMStrategyGenerator(
        LiteLLMClient(settings, model=ChatModelFake())
    )

    generator.generate(
        RenderedStrategyPrompt(
            system_prompt="JSON으로 답해.",
            user_prompt="전략을 세워.",
        ),
        model_name=settings.model,
        seed=0,
        response_schema={"type": "object"},
    )

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["max_tokens"] == 1536


def test_strategy_generator_applies_requested_model_to_actual_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_settings = OllamaSettings(
        api_base="http://localhost:11434",
        model="env-model",
    )
    created_models: list[str] = []

    class ClientFake:
        def __init__(self, settings: OllamaSettings) -> None:
            self.settings = settings
            created_models.append(settings.model)

        def complete(self, *args: object, **kwargs: object) -> TextCompletion:
            return TextCompletion("{}", CompletionMetrics())

    monkeypatch.setattr(
        OllamaSettings,
        "from_env",
        classmethod(lambda cls: base_settings),
    )
    monkeypatch.setattr(runtime_module, "LiteLLMClient", ClientFake)
    generator = LiteLLMStrategyGenerator()

    model_name = generator.resolve_model_name("requested-model")
    generator.generate(
        RenderedStrategyPrompt("system", "user"),
        model_name=model_name,
        seed=0,
        response_schema={"type": "object"},
    )
    generator.generate(
        RenderedStrategyPrompt("system", "user"),
        model_name=model_name,
        seed=1,
        response_schema={"type": "object"},
    )

    assert model_name == "requested-model"
    assert created_models == ["requested-model"]


def test_strategy_generator_rejects_injected_client_model_mismatch() -> None:
    settings = OllamaSettings(
        api_base="http://localhost:11434",
        model="bound-model",
    )
    generator = LiteLLMStrategyGenerator(
        LiteLLMClient(settings, model=cast(Any, object()))
    )

    with pytest.raises(PromptConfigurationError, match="model mismatch"):
        generator.resolve_model_name("different-model")
