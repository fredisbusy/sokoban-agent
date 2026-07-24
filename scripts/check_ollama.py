"""Send one lightweight request to the configured Ollama server."""

from sokoban_agent.planning.llm.client import LiteLLMClient


def main() -> None:
    client = LiteLLMClient.from_env()
    response = client.complete("연결 확인이라고 짧게 답해줘.")
    print(response.content)
    print(
        {
            "total_seconds": response.metrics.total_seconds,
            "load_seconds": response.metrics.load_seconds,
            "prompt_tokens": response.metrics.prompt_tokens,
            "output_tokens": response.metrics.output_tokens,
        }
    )


if __name__ == "__main__":
    main()
