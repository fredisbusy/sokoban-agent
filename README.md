# Sokoban Agent

Sokoban을 플레이하는 AI 에이전트를 처음부터 구현하며 연구하기 위한 Python
프로젝트 뼈대입니다.

## 시작하기

```bash
uv sync
cp .env.example .env
uv run jupyter lab
```

재사용 가능한 환경·에이전트 로직은 `src/`에, 탐색과 시각화는
`notebooks/`에 둡니다. 반복 가능한 실험 설정은 `configs/`, 실행 진입점은
`scripts/`, 검증 코드는 `tests/`에 둡니다.

현재는 라이브러리와 폴더 구조만 준비되어 있으며 Sokoban 환경이나 에이전트
구현은 포함하지 않습니다.

## Ollama 연결

LiteLLM을 통해 로컬 또는 원격 Ollama 서버를 호출합니다. 접속 주소와 모델은
코드가 아닌 `.env`에서 설정합니다.

```dotenv
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT_SECONDS=120
```

로컬 Ollama를 사용한다면 모델을 준비한 뒤 연결을 확인할 수 있습니다.

```bash
ollama pull llama3.2
uv run python scripts/check_ollama.py
```

Python 코드에서는 다음처럼 사용합니다.

```python
from sokoban_agent.agents.llm import OllamaClient

client = OllamaClient.from_env()
answer = client.complete("이 보드에서 다음 행동을 한 단어로 답해줘.")
```

## 선택 가능한 기존 환경

- `gym-sokoban`: 빠른 참고용이지만 오래된 Gym API 기반이라 직접 의존하지 않습니다.
- MiniHack Boxoban: 대규모 벤치마크에 적합하지만 첫 구현에는 무겁습니다.
- Griddly Sokoban: 규칙 실험에는 유연하지만 별도 프레임워크 학습이 필요합니다.
