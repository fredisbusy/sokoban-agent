# 정확성·회귀 검토

## P1. 기록된 모델과 실제 호출 모델이 다를 수 있다

- 근거:
  - `builder.py:103-112`는 context의 `model_name`을 state에 기록한다.
  - `strategy.py:320-324`는 기본 generator를 호출 시점마다 새로 만든다.
  - `planning/agentic/runtime.py:186-208`은 실제 client를 환경 변수에서 만든다.
- 영향:
  - CLI의 `--model`과 viewer의 model 입력은 provenance만 바꾸고 실제 호출
    모델은 바꾸지 않을 수 있다.
  - 연구 artifact와 shared-memory key가 실제 모델과 다른 이름을 기록할 수 있다.
- 방향:
  - graph composition root에서 `StrategyGenerator`와 immutable
    `EffectiveModelConfig`를 함께 만들고 state의 model identity는 generator에서
    파생한다.
  - context에는 허용된 model profile ID만 받고 서버가 실제 adapter를 선택한다.

## P1. 시작부터 terminal인 보드가 planner로 진입한다

- 근거:
  - `builder.py:89-113`은 `env.reset()`의 terminal info를 저장한다.
  - `builder.py:216-225, 278-288`은 success/deadlock 여부와 무관하게
    `analyze -> resolve_prompt/compose_strategy_input`으로 보낸다.
- 영향:
  - 이미 해결된 보드나 초기 deadlock 보드가 불필요한 모델 호출을 하고,
    schema/semantic/cycle 상태로 끝날 수 있다.
- 방향:
  - initialize 직후 canonical terminal router를 두고 `success`, `deadlock`,
    invalid input을 END로 보낸다.
  - solved-at-reset, deadlocked-at-reset graph/evaluation 회귀 테스트를 추가한다.

## P2. 같은 thread에 새 episode를 넣으면 reducer 이력이 섞인다

- 근거:
  - `state.py:103-105`의 revision/feedback/event는 `operator.add` reducer다.
  - `builder.py:141-149`의 빈 리스트는 기존 reducer 값을 지우지 않는다.
  - `tests/test_agentic_graph.py:197-239`는 같은 thread의 두 번째 invoke에서
    첫 episode event가 남는 동작을 고정한다.
- 영향:
  - action/metrics는 초기화되는데 event/revision/feedback은 누적되는 혼합 state가
    만들어질 수 있다.
- 방향:
  - thread ID와 episode ID를 같은 개념으로 강제하거나, multi-run thread가
    필요하면 episode 경계를 state와 reducer에 명시한다.

## P2. recursion limit이 graph 위상과 분리된 magic number다

- 근거:
  - `graph/agentic/runtime.py:53-58`의 limit은 `max_steps * 12 + 50`이다.
  - 실제 성공 push cycle은 memory/strategy/grounding/action/reflection 노드로
    약 13개 이상의 graph step을 사용한다.
- 영향:
  - 긴 push-only episode는 환경 `max_steps` 전에 `GraphRecursionError`로 끝날
    가능성이 있다.
- 방향:
  - phase별 최대 전이 수와 retry budget에서 계산하거나 별도
    `max_graph_steps` 정책을 명시하고 긴 synthetic fixture로 검증한다.

## P2. oracle 격리 테스트가 호출이 아니라 노드 이름만 검사한다

- `tests/test_agentic_graph.py:242-251`은 노드 이름에 oracle 관련 문자열이
  없는지만 확인한다.
- 전역 oracle adapter와 guard seam을 monkeypatch해 구조화 graph 전체 실행에서
  실제 호출이 0회인지 검증해야 한다.
