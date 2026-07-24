# 최종 교차 검토

## 반영한 필수 수정

- `guard_reference_action_count`는 생산 경로가 있는 raw diagnostic임을
  명시하고, 단순 미사용 삭제가 아니라 P0 의미 검토 대상으로 낮췄다.
- viewer의 baseline decoder를 선택 사항이 아닌 기존 회귀 계약으로 확정했다.
- artifact writer의 canonical key와 additive v1 reader의 누락 optional key
  허용을 분리했다.
- old checkpoint를 실제로 차단할 state version guard 또는 versioned
  graph/assistant identity를 완료 조건에 추가했다.
- 실행 순서를 adapter 선행, Studio 제거, state 타입 강화, 평가 합성,
  checkpoint-breaking state 중첩 순으로 통일했다.

## 타당성 확인

- `StudioState` 전체 제거 방향은 현재 `langgraph.json` 진입점과 import graph에
  부합한다.
- `AgenticState.failure_conditions` 최상위 복제 제거 후에도 canonical
  hypothesis에 실패 조건이 남는다.
- memory recall의 hit flag를 `Command(update=..., goto=...)`로 대체하는 것은
  update와 routing을 함께 수행하는 경우에만 Command를 쓴다는 프로젝트 원칙과
  부합한다.
- `policy_elapsed_seconds`는 adapter property로 파생 가능하며 기존 `max(0)`
  의미를 golden test로 보존해야 한다.

## 구현 후 확인

- `StudioState`와 전용 graph/test를 제거하고 활성 agentic graph 하나로
  문서·viewer 계약을 맞췄다.
- `AgenticState`를 `meta`, `planning`, `memory`, `execution` 스냅샷과 hot
  channel로 분리했다.
- 하위 목표·보호 제약·예상 효과·실패 조건은 strategy hypothesis만 canonical
  source로 사용하며, grounded action은 grounded plan에서 파생한다.
- memory recall은 hit flag 없이 `Command(update=..., goto=...)`를 사용한다.
- baseline과 research 결과는 책임별 합성 객체가 raw count를 저장하고,
  flat adapter가 기존 외부 컬럼과 파생 rate를 제공한다.

`total_reward`와 `guard_reference_action_count`는 기존 v1 artifact의 생산
diagnostic이므로 이번 breaking 내부 리팩터링에서도 외부 계약에 유지했다.

## 2026-07-24 P7 교차 검토

- graph topology와 production provider 조립이 분리되었고 `langgraph.json`은
  composition root를 가리킨다.
- CLI의 `--model`, state의 `meta.model_name`, 실제 LiteLLM client model이
  하나의 effective model 계약을 사용한다.
- 초기 성공·데드락은 analyze 이전에 종료하며 모델 호출이 발생하지 않는다.
- 연구 실행은 episode마다 graph를 compile하지 않고 runner 하나를 재사용한다.
- architecture graph inventory와 README 예제가 현재 nested metric 계약을
  반영한다.

잔여 후속은 manifest download path containment·size limit과 agentic runtime의
port/adapter 물리적 파일 분리다. 둘은 이번 graph 발견성·수명 변경과 별도
위험 축이므로 다음 단계로 분리한다.
