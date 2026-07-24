# 단계별 마이그레이션 계획

## P0. 외부 계약 고정

변경 전에 다음 golden/contract test를 추가한다.

- pilot episode JSONL writer의 canonical v1 key
- optional 계측 키가 빠진 기존 additive v1 row의 기본값 load-save round trip
- baseline summary exact column set
- research record/summary exact column set과 `null`/`0` 구분
- agentic strict msgpack checkpoint round trip
- old state schema checkpoint resume의 명시적 거절 fixture
- 실제 agentic update fixture에서 `ViewEventV1` projection
- CLI JSON exact key

완료 조건: 현재 artifact와 view event의 호환 기준이 자동 테스트로 고정된다.

## P1. adapter 선행 도입

내부 shape를 바꾸기 전에 다음 adapter를 추가한다.

- `BaselineEpisodeRowV1`
- `PlannerSummaryRowV1`
- `ResearchPolicySummaryRowV1`
- `AgenticCliResultV1`
- viewer의 agentic/baseline decoder

모든 `asdict(result)`, `asdict(summary)`,
`EpisodeResult(**payload)` 직접 호출을 adapter로 교체한다. pilot artifact에는
episode/summary schema version을 명시한다.

완료 조건: 내부 dataclass shape를 바꾸지 않은 상태에서 기존 JSON key와
DataFrame 컬럼이 동일하고, old JSONL resume가 통과한다.

## P2. legacy Studio 제거

- `src/sokoban_agent/graph/studio/` 제거
- `tests/test_studio.py` 제거
- `README.md`, `docs/PROJECT.md`, `docs/ARCHITECTURE.md`,
  `docs/DEPENDENCIES.md`의 별도 Studio graph 설명을 실제 agentic graph
  기준으로 갱신
- viewer legacy Studio fallback 제거
- 실제 `SokobanGraphState` fixture로 baseline decoder 지원을 계속 검증

완료 조건: `langgraph.json`의 유일 graph와 문서·테스트·viewer가 같은
정책을 가리킨다.

## P3. AgenticState payload 타입 강화

top-level shape를 유지한 채 먼저 다음 타입을 도입한다.

- `AgenticStatus` Literal
- `AgenticInfoState`
- `ExecutionResultState`
- `ReflectionResultState`
- strategy/grounding payload TypedDict
- node별 update TypedDict

완료 조건: 주요 node의 `dict[str, object]`와 terminal string cast가 사라지고
checkpoint JSON/msgpack round trip이 유지된다.

## P4. 평가 결과 객체 합성

작은 breaking commit으로 순차 진행한다.

1. `EpisodeResult`를 `BaselineEpisodeResult` 합성 구조로 전환
2. runner가 graph nested metrics에서 책임 객체를 바로 조립
3. `PlannerSummary`를 책임별 aggregate로 전환
4. `ResearchPolicySummary`를 책임별 aggregate로 전환
5. notebook generator는 flat adapter만 사용하도록 변경

이 단계에서 내부 계산 저장을 제거한다.

- episode overhead와 policy elapsed time
- summary success/deadlock/guard adoption rate
- research success/subgoal/effect rate와 total LLM tokens

v1 adapter는 기존 키를 property 계산으로 계속 제공한다. `total_reward`와
`guard_reference_action_count`를 v2에서 제외할지는 P0 usage snapshot을
확인한 뒤 별도 breaking decision으로 확정한다.

완료 조건: 내부 대형 constructor가 책임별 조립으로 바뀌고 v1 artifact
round trip과 notebook column contract가 유지된다.

## P5. AgenticState 책임별 중첩

- `state_schema_version`과 graph revision 증가
- case/runtime/analysis/strategy/grounding/memory snapshot 도입
- `failure_conditions` 최상위 복제 제거
- hypothesis를 canonical source로 삼아 `active_subgoal`,
  `protected_constraints`, `expected_effect` 최상위 복제 제거
- `grounded_actions`를 grounded plan에서 파생
- recall node가 update와 routing을 함께 수행하도록 `Command`를 사용해
  `strategy_memory_hit`, `grounding_memory_hit` 제거
- terminal status를 canonical source로 정리
- reducer history는 최상위 delta update 유지
- state version mismatch를 initialize/resume 경계에서 명시적으로 실패시키거나
  versioned graph/assistant identity로 구 thread 접근을 차단
- 새 run/thread만 허용하고 구 checkpoint migration은 제공하지 않음

완료 조건: graph node routing, memory recall, one-push execution, reflection,
streaming final state가 기존과 동일하며 state 중복 source가 없다. 같은
thread의 구 schema checkpoint resume는 조용히 진행되지 않고 명시적으로
거절된다. `Command` hit/miss 양쪽의 update stream도 golden test와 일치한다.

## P6. 정리

- 임시 compatibility code와 obsolete alias 제거
- architecture/live viewer 문서 갱신
- notebook generator로 notebook 재생성
- schema deprecation 정책 기록

## 우선순위

1. Studio 제거와 viewer decoder — 활성/비활성 graph 혼동 제거
2. Agentic typed payload — 주 runtime의 string/dict drift 차단
3. 평가 flat adapter와 nested result — 큰 constructor와 artifact 결합 해소
4. AgenticState 중첩 — checkpoint breaking change이므로 앞 단계 뒤 수행
5. Research summary 중첩 — 현재 위험이 가장 낮아 마지막에 수행 가능
