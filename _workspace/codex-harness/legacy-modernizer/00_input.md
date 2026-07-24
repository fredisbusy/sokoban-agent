# 리팩터링 계획 입력

## 요청

필드가 15개를 초과하는 다음 클래스에서 불필요한 필드와 관심사 분리가 필요한
필드를 선정하고 구현 전 단계의 리팩터링 계획을 수립한다.

- `EpisodeResult` — 57개
- `PlannerSummary` — 54개
- `ResearchPolicySummary` — 20개
- `AgenticState` — 45개
- `StudioState` — 31개

## 범위와 제약

- 이번 작업은 분석과 계획 수립만 수행한다.
- LangGraph checkpoint, 평가 결과 직렬화, notebook, CLI, viewer 소비 계약을
  근거로 생성·소비 경로를 확인한다.
- 필드 수를 맞추기 위한 기계적 분할 대신 변경 이유가 다른 책임을 분리한다.
- 제거 후보를 실제 미사용, 중복 저장, 파생 가능, artifact-only로 구분한다.
- 마이그레이션 계획에 회귀 테스트와 단계별 완료 조건을 포함한다.

## 2026-07-24 후속 구현 입력

현재 코드와 기존 P0–P6 진행 상태를 다시 확인하고, NestJS에 익숙한 개발자가
어떤 graph에 어떤 module이 사용되는지 찾을 수 있도록 구조 발견성과 실행
수명을 개선한다.

- 실제 호출 모델과 기록된 model provenance를 일치시킨다.
- graph topology와 production provider 조립을 분리한다.
- Agent Server·Studio·CLI·평가의 graph 진입점을 문서화한다.
- terminal 초기 상태와 experiment 단위 provider 수명을 테스트한다.
