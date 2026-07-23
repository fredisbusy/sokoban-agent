"""Send one lightweight request to the configured Ollama server."""

from sokoban_agent.agents.llm import OllamaClient


def main() -> None:
    client = OllamaClient.from_env()
    response = client.complete("연결 확인이라고 짧게 답해줘.")
    print(response)


if __name__ == "__main__":
    main()

