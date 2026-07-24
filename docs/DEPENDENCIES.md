# 의존성과 기술 선택

기본 설치는 환경, LangGraph 실행 코어, 구조화 전략 모델과 Ollama Adapter에
필요한 Python 패키지를 포함한다. 현재 공개 모듈과 `sokoban-agentic`
진입점이 LLM 계약을 직접 사용하므로 LLM 패키지를 optional extra로
분리하지 않는다. 로컬 검증·노트북·Studio 도구만 dependency group으로
분리한다.

## 설치 경계

| 범위 | 구성 | 설치 |
| --- | --- | --- |
| runtime | Gymnasium, NumPy, LangGraph, LangChain, LangSmith, ChatLiteLLM, Pydantic, python-dotenv, certifi | `make sync` |
| dev | pytest, Ruff, mypy | `make sync` |
| notebook | IPython, JupyterLab, nbconvert, nbformat, pandas, Matplotlib | `make sync` |
| studio | LangGraph CLI, 로컬 in-memory Agent Server | `make sync` |
| viewer | Next.js, React, LangGraph SDK, TypeScript | `make viewer-sync` |

`make sync`는 `uv sync --all-groups`로 Python runtime과 모든 연구 도구를
설치한다. Viewer는 별도 Node.js 애플리케이션이므로 최초 한 번
`make viewer-sync`로 lockfile 기준 설치한다.

현재 보드 입력은 Gymnasium의 구조화된 `uint8` 배열이고, 터미널은 ANSI,
웹은 CSS Grid로 렌더링한다. 화면 이미지를 읽는 perception은 아직 구현하지
않았으므로 OpenCV와 Pillow를 직접 의존성으로 두지 않는다. 향후 perception을
구현할 때는 입력 계약과 테스트가 생긴 뒤 필요한 이미지 라이브러리를
선정한다.

## 기준선 탐색

BFS는 외부 그래프 패키지 대신 표준 라이브러리 `collections.deque`를
사용한다. Sokoban 상태 그래프는 실행 중 이웃을 생성하므로, NetworkX나
rustworkx를 사용하려면 같은 상태를 별도 그래프로 먼저 물질화해야 한다.
이동·밀기·데드락은 프로젝트의 순수 규칙 함수가 담당하고 BFS는 탐색 순서와
경로 복원만 담당한다.

## LangGraph

LangGraph는 선택 기능이 아니라 실행 코어다. `StateGraph`의 조건부 edge가
계획, 검증, 실행과 복구를 연결하고 `InMemorySaver`가 에피소드 상태를
`thread_id`별로 체크포인트한다. 알고리즘 Planner와 LLM Planner는 같은 graph
경계를 사용하되, 모델 호출은 `langchain-litellm`의 `ChatLiteLLM`을 사용한다.
provider HTTP client, retry loop, callback·trace 전달 계층을 자체 구현하지 않는다.
LangGraph의 node retry·checkpoint와 LangChain Runnable·structured output·
LangSmith tracing을 우선 사용한다.

### Studio

`studio` 그룹은 로컬 시각화와 디버깅에만 필요하다. `langgraph dev`가
`langgraph.json`이 가리키는 구조화 `StateGraph`를 로컬 Agent Server로
띄운다. `graph/agentic/builder.py`는 topology를 정의하고,
`graph/agentic/composition.py`는 production provider를 조립한다.
Agent Server·Studio는 composition root의 전역 graph를, CLI·평가는 같은
factory로 local compile한 graph를 사용한다. 원격 LangSmith 추적은 실행
환경의 데이터 정책에 따라 명시적으로 켜거나 끈다. 활성화할 때는 LangGraph
node와 `ChatLiteLLM` 호출을 하나의 trace로 관리하고 민감한 입력에는
LangSmith masking을 적용한다.

### prompt 관리

구조화된 전략 prompt의 실행 순서는 LangGraph node와 edge로 관리하고,
본문·commit·환경 tag는 LangSmith Prompt Management를 사용한다. 이는
LangSmith tracing 활성화와 별개다. LangSmith SDK는 runtime 의존성이며,
prompt 이름과 고정 commit은 runtime context로 주입한다.

자체 prompt registry, cache, version database와 승격 도구는 만들지 않는다.
연구 실행은 mutable tag 대신 commit을 고정하고, offline 단위 테스트에는
같은 graph node를 통과하는 고정 fixture를 사용한다.

## 분석

에피소드 결과와 집계는 core에서 immutable dataclass와 표준 라이브러리로
계산한다. pandas와 Matplotlib은 표와 차트가 필요한 노트북에서만 사용한다.

## LLM

`ChatLiteLLM`을 통해 Ollama를 호출하고 LangChain structured output과
Pydantic으로 1~8개의 `UP`, `RIGHT`, `DOWN`, `LEFT` 행동열을 검증한다.
실험에서는 `.env`의 모델, context, 출력 제한, thinking, temperature와
episode seed를 호출에 전달하고 Python 규칙 함수가 막힌 이동과 중간
데드락을 실행 전에 거절한다. 표준 usage metadata는 공통 결과에 기록한다.
provider 고유 응답 필드가 꼭 필요한 연구 지표라면 먼저 LiteLLM의
`response_metadata` 확장으로 수집하고, 이를 이유로 별도 HTTP client를
만들지 않는다.

## memory

에피소드 내부 실행 상태는 checkpointer가, thread 간 공유 가능한 분석·전략·
접지 결과는 LangGraph Store가 소유한다. `memory_mode=off|episode|shared`로
실험의 정보 누설 범위를 통제하며, 별도 저장소 라이브러리나 자체 memory
framework는 사용하지 않는다.
