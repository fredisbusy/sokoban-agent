# 의존성과 기술 선택

기본 설치는 Sokoban 환경과 기준선 실행에 필요한 패키지만 포함한다.
기능별 라이브러리는 배포 기능이면 optional extra, 로컬 연구 도구이면
dependency group으로 분리한다.

## 설치 경계

| 범위 | 구성 | 설치 |
| --- | --- | --- |
| core | Gymnasium, NumPy, LangGraph | `uv sync --no-dev` |
| LLM | LiteLLM, Pydantic, python-dotenv | `uv sync --no-dev --extra llm` |
| vision | OpenCV, Pillow | `uv sync --no-dev --extra vision` |
| dev | pytest, Ruff, mypy와 LLM 테스트 의존성 | `uv sync` |
| notebook | Jupyter, nbclient, nbformat, pandas, Matplotlib | `uv sync --group notebook` |

`dev`에는 optional LLM 기능까지 항상 테스트하기 위해 LLM 패키지를
의도적으로 다시 선언한다. 배포 패키지의 기본 의존성에는 포함되지 않는다.

## 기준선 탐색

BFS는 외부 그래프 패키지 대신 표준 라이브러리 `collections.deque`를
사용한다. Sokoban 상태 그래프는 실행 중 이웃을 생성하므로, NetworkX나
rustworkx를 사용하려면 같은 상태를 별도 그래프로 먼저 물질화해야 한다.
이동·밀기·데드락은 프로젝트의 순수 규칙 함수가 담당하고 BFS는 탐색 순서와
경로 복원만 담당한다.

## LangGraph

LangGraph는 선택 기능이 아니라 실행 코어다. `StateGraph`의 조건부 edge가
계획, 검증, 실행과 복구를 연결하고 `InMemorySaver`가 에피소드 상태를
`thread_id`별로 체크포인트한다. Planner 구현은 LangChain에 의존하지 않으며
알고리즘과 LiteLLM 클라이언트를 같은 그래프 경계에 연결한다.

## 분석

에피소드 결과와 집계는 core에서 immutable dataclass와 표준 라이브러리로
계산한다. pandas와 Matplotlib은 표와 차트가 필요한 노트북에서만 사용한다.

## LLM

LiteLLM은 Ollama 호출과 JSON Schema 응답 형식을 같은 인터페이스로
제공한다. Pydantic은 모델 응답을 `UP`, `RIGHT`, `DOWN`, `LEFT` 중 하나로
검증한다. 실험에서는 `.env`의 모델·temperature와 episode seed를 호출에
전달하고, Python 규칙 함수가 막힌 행동을 환경 실행 전에 거절한다.

## 향후 memory

현재 LangGraph 체크포인트는 한 프로세스 안의 실행 상태이며 의미 기반 장기
기억이 아니다. 실패 계획과 데드락을 세션 간 검색할 때는 먼저 JSONL 또는
SQLite 기준선을 만들고, 필요하면 외부 저장소의 검색 품질과 비용을 비교한다.
