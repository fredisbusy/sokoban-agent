# 성능 검토

## P1. 외부 adapter가 노드 호출마다 재생성된다

- 근거:
  - `StrategyNodes._prompt_source()`와 `_strategy_generator()`는 주입값이 없으면
    매 호출마다 새 adapter를 반환한다 (`strategy.py:320-324`).
  - generator는 다시 `LiteLLMClient.from_env()`를 호출한다
    (`planning/agentic/runtime.py:201-208`).
- 영향:
  - LangSmith client, 설정 로딩, ChatLiteLLM model 구성이 전략 호출마다
    반복되고 adapter cache/connection 수명을 활용하지 못한다.
- 방향:
  - graph build 시점에 adapter를 한 번 만들고 node bundle에 필수 의존성으로
    주입한다. 테스트 fixture도 같은 composition root를 사용한다.

## P1. evaluation이 매 episode graph와 store를 다시 만든다

- 근거:
  - `evaluation/agentic/runner.py:42-45`가 매 호출마다 runner를 만든다.
  - `graph/agentic/runtime.py:35-42`가 runner마다 checkpointer, store, compiled
    graph를 만든다.
- 영향:
  - evaluation API에서 `memory_mode=shared`여도 episode 사이 memory가 사라진다.
  - 15레벨 × seed × structured policy마다 graph compile 비용이 반복된다.
- 방향:
  - experiment scope의 `AgenticEvaluator` 또는 주입 가능한 runner를 두고,
    policy 격리는 namespace/thread로 관리한다.

## P2. A* reachability hot path가 필요 이상으로 경로를 물질화한다

- 근거:
  - `planning/search/spatial.py:18-41`은 모든 reachable cell의 전체 tuple path를
    복사한다.
  - `planning/search/astar.py:89, 113-114, 239-241`은 state key 계산에서 경로를
    버리면서도 이 함수를 호출한다.
- 영향:
  - bounded 200k-state oracle/guard에서 불필요한 path 복사가 탐색 상태 수만큼
    반복된다.
- 방향:
  - `reachable_region()`과 `path_to()`를 분리하거나 predecessor tree를
    반환해 실제 successor path에서만 복원한다.

## P3. Boxoban provider 초기화가 전체 source byte 수에 비례한다

- 근거:
  - `env/levels.py:218-240`은 모든 txt를 읽어 header를 세고 첫 파일을 다시
    파싱한다.
- 방향:
  - 첫 parse를 재사용하고, header streaming scan이나 고정 index manifest로
    파일 count와 parsing을 분리한다.
