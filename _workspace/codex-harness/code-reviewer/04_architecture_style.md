# 아키텍처·유지보수성 검토

## P1. viewer와 graph state 계약이 이미 갈라졌다

- 근거:
  - agentic graph는 `state.metrics.llm`과 `state.metrics.local_search`에 기록한다.
  - `viewer/lib/events.ts:79-85`는 이전 top-level scalar만 읽는다.
  - `GraphState`는 `Record<string, unknown>`이라 TypeScript가 drift를 잡지 못한다.
- 영향:
  - 실제 structured run의 LLM/search 비용이 null 또는 잘못된 값으로 표시된다.
- 방향:
  - agentic update DTO와 runtime decoder를 만들고 중첩 metrics를 명시적으로
    매핑한다.
  - baseline 호환은 별도 adapter로 분리하고 실제 agentic state fixture를
    contract test로 공유한다.

## P1. viewer 검증이 root 품질 게이트에서 빠져 있다

- 근거:
  - `Makefile check`는 Python pytest/Ruff/mypy만 실행한다.
  - `viewer/package.json:10`의 `node --test test/*.test.ts`는 현재 Node 22.16.0에서
    `ERR_UNKNOWN_FILE_EXTENSION`으로 모든 테스트가 시작 전에 실패한다.
- 영향:
  - viewer contract drift가 완료 검증을 통과해도 발견되지 않는다.
- 방향:
  - 지원 Node 버전을 고정하고 `tsx --test` 또는 지원되는 type stripping
    runner를 사용한다.
  - viewer test/typecheck/build를 root `make check` 또는 CI에 포함한다.

## P2. 사용되지 않는 별도 Studio 정책 그래프가 남아 있다

- 근거:
  - `langgraph.json`은 agentic graph만 가리킨다.
  - `graph/studio/{graph,nodes,state}.py`는 별도 LLM/A* validate/execute workflow를
    구현하고 `tests/test_studio.py`가 이를 계속 보증한다.
- 영향:
  - 실제 runtime과 다른 정책이 “Studio graph” 이름으로 남아 잘못 사용될 수
    있고, 규칙·계측·오류 처리 변경이 이중으로 필요하다.
- 방향:
  - legacy Studio graph와 테스트를 제거한다. baseline 관찰 fixture가 필요하면
    `graph/baseline/legacy_studio.py`처럼 의도를 드러내고 주 graph와 혼동하지
    않게 한다.

## P2. core/LLM optional dependency 경계가 import graph와 맞지 않는다

- 근거:
  - `pyproject.toml`은 `langchain-litellm`을 `llm` extra에만 둔다.
  - public `sokoban_agent.graph` import는 builder를 거쳐
    `planning.agentic.runtime -> planning.llm.client -> langchain_litellm`을
    즉시 import한다.
- 영향:
  - 기본 설치에서 LangGraph baseline/public graph API import가 실패할 수 있다.
- 방향:
  - 실제 제품 경계에 맞게 LLM을 필수 의존성으로 올리거나, optional adapter를
    별도 module과 lazy composition으로 분리한다.

## P3. state의 핵심 transition contract가 string/dict 중심이다

- 근거:
  - `AgenticState.status`는 단순 `str`이고 execution/reflection payload도
    `dict[str, object]`다.
  - 여러 route와 viewer가 문자열 literal과 cast로 의미를 다시 해석한다.
- 방향:
  - terminal/non-terminal status의 `Literal` union과 JSON-safe TypedDict를
    정의하고, graph-to-viewer/evaluation adapter에서 exhaustive mapping을
    강제한다.
