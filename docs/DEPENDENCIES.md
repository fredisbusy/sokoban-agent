# 의존성과 기술 선택

기본 설치는 Sokoban 환경과 기준선 실행에 필요한 패키지만 포함한다.
기능별 라이브러리는 배포 기능이면 optional extra, 로컬 연구 도구이면
dependency group으로 분리한다.

## 설치 경계

| 범위 | 구성 | 설치 |
| --- | --- | --- |
| core | Gymnasium, NumPy, LangGraph | `make sync` |
| LLM | ChatLiteLLM, Pydantic, python-dotenv | `make sync` |
| vision | OpenCV, Pillow | `make sync` |
| dev | pytest, Ruff, mypy와 LLM 테스트 의존성 | `make sync` |
| notebook | Jupyter, nbclient, nbformat, pandas, Matplotlib | `make sync` |
| studio | LangGraph CLI, 로컬 in-memory Agent Server | `make sync` |

`dev`에는 optional LLM 기능까지 항상 테스트하기 위해 LLM 패키지를
의도적으로 다시 선언한다. 배포 패키지의 기본 의존성에는 포함되지 않는다.
로컬 개발에서는 `make sync`가 모든 extra와 dependency group을 함께 설치한다.

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
현재 `langgraph.json`의 baseline Studio 그래프를 읽어 로컬 Agent Server를
띄운다. 구조화된 에이전트로 전환할 때는 Studio 전용 workflow를 유지하지
않고 CLI·평가와 같은 compiled `StateGraph`를 가리키게 한다. 원격 LangSmith
추적은 실행 환경의 데이터 정책에 따라 명시적으로 켜거나 끈다. 활성화할
때는 LangGraph node와 `ChatLiteLLM` 호출을 하나의 trace로 관리하고 민감한
입력에는 LangSmith masking을 적용한다.

### prompt 관리

구조화된 전략 prompt의 실행 순서는 LangGraph node와 edge로 관리하고,
본문·commit·환경 tag는 LangSmith Prompt Management를 사용한다. 이는
LangSmith tracing 활성화와 별개다. 실제 구현 단계에서 LangSmith SDK는 LLM
optional extra에 추가하고 prompt 이름과 고정 commit을 runtime context로
주입한다.

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

## 향후 memory

현재 LangGraph 체크포인트는 한 프로세스 안의 실행 상태이며 의미 기반 장기
기억이 아니다. 실패 계획과 데드락을 세션 간 검색할 때는 LangGraph Store를
우선 사용하고, 저장 Adapter를 교체해야 할 실제 요구가 생길 때만 별도
저장소 seam을 추가한다.
