from types import SimpleNamespace
from typing import Any, cast

from langchain_core.messages import AIMessage
from langchain_core.prompts.structured import StructuredPrompt
from langsmith import Client

from sokoban_agent.planning.llm import LiteLLMClient, OllamaSettings
from sokoban_agent.planning.strategy_runtime import (
    LangSmithPromptSource,
    LiteLLMStrategyGenerator,
    PromptReferenceValue,
    RenderedStrategyPrompt,
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
        seed=0,
        response_schema={"type": "object"},
    )

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["max_tokens"] == 1536
