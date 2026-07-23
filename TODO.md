# TODO

위에서 아래 순서로 진행한다. 항목은 완료 조건을 검증한 뒤에만 체크한다.

## 완료된 기반

- [x] 타일, 4방향 행동, Boxoban 호환 레벨 형식을 정의한다.
- [x] Gymnasium `reset`, `step`, `render`를 구현한다.
- [x] 이동, 상자 밀기, 승리, 무효 이동, 기본 데드락을 판정한다.
- [x] 고정 레벨과 지연 로딩 Boxoban 레벨 공급자를 구현한다.
- [x] `reset/step/win/deadlock`과 Gymnasium 호환성 테스트를 작성한다.
- [x] 내장 레벨을 조작하는 터미널 플레이 진입점을 만든다.
- [x] LiteLLM 기반 Ollama 텍스트 클라이언트와 연결 확인 스크립트를 만든다.

고정 1-box 퍼즐은 `level_id`와 고정 행동으로 재현할 수 있고, seed로 레벨
선택을 재현할 수 있다.

## 완료된 기준선과 측정

- [x] Boxoban 다중 파일 인덱스, 캐시 전환과 크기 오류 테스트를 보강한다.
- [x] core, LLM, notebook, vision 의존성의 설치 경계를 분리한다.
- [x] 공통 Agent 인터페이스와 Random Agent를 만든다.
- [x] 비교용 BFS 솔버를 만든다.
- [x] 에피소드 실행기에서 성공률, 행동 수, 무효 이동, 데드락, 시간을 기록한다.
- [x] 환경과 기준선 결과를 보여주는 첫 노트북을 만든다.

내장 레벨 2개와 seed 10개를 비교하는 재현 가능한 코드가
`notebooks/baseline_comparison.ipynb`에 있다. 저장 출력은 유지하지 않는다.

## 완료된 LLM 기준선

- [x] 보드를 직렬화하고 행동 출력을 구조화한다.
- [x] 빈 응답, 출력 형식 오류, 무효 행동에 재시도·복구 규칙을 둔다.
- [x] 모델, seed, 행동 제한을 설정으로 고정하는 실행기를 만든다.
- [x] LLM과 기준선의 성공률·효율·응답 시간을 비교한다.
- [x] 비교 에피소드의 실제 이동 경로를 노트북에서 재생한다.

완료 조건: 소형 고정/미지 레벨에서 반복 가능한 비교 결과를 남긴다.

LangGraph 전환 전 `Agent.act()` 실행 결과와 저장 출력은 폐기했다. 현재
비교 코드는 `notebooks/langgraph_planner_comparison.ipynb`에서 LLM 단독과
LLM+BFS Search Guard를 함께 실행한다.

## 완료된 LangGraph baseline

- [x] 에피소드 루프를 `plan → validate → execute` StateGraph로 이전한다.
- [x] Random, BFS, LLM을 공통 Planner 노드로 이전한다.
- [x] 계획 오류와 막힌 행동을 조건부 edge에서 재시도한다.
- [x] 에피소드별 `thread_id` 체크포인트와 일반 계획 지표를 기록한다.
- [x] 평가와 trajectory가 반드시 같은 그래프 실행기를 사용하게 한다.
- [x] LLM 등 주 Planner의 제안을 BFS로 검사·대체하는 하이브리드를 추가한다.
- [x] push 기반 A* 휴리스틱 Planner를 추가해 BFS guard와 비교한다.
- [x] LLM이 짧은 다중 행동 계획을 제안하고 전체 계획을 검증한다.
- [x] Search Guard가 BFS/A* 후속 경로와 상태별 캐시를 재사용한다.
- [x] Ollama와 알고리즘 세부 성능 지표를 분리해 기록한다.
- [x] LangGraph 전환 후 기준선과 LLM 비교 노트북을 다시 실행한다.
- [x] LangGraph Studio에서 계획·A* 검사·검증·실행 노드를 관찰한다.
- [x] LLM 프롬프트와 목표·판단·위험·검사 요약을 한국어로 기록한다.
- [x] LLM 기여도를 분리하는 5개 정책 ablation을 구현한다.
- [x] prefix 채택률, 상태 재방문, 반복 계획과 bounded A* reference 대비
  행동·push·탐색량을 계측한다.
- [x] 커밋·체크섬·레벨 ID를 고정한 Boxoban 30/50 레벨 파일럿 러너를 만든다.

완료 조건: 하이브리드 Planner가 LLM 단독과 같은 레벨에서 실행되고 알고리즘
검사·폴백 횟수, 재계획과 성공률을 비교할 수 있다.

## 지금 — LangGraph 기반 구조화 문제 해결 에이전트

현재 원시 행동 LLM과 Search Guard는 비교 baseline으로 보존한다. 새 주
정책에서는 전역 A*가 정답 suffix를 제공하거나 전체 계획을 대체하지 않는다.
그래프 구조·상태 전이·재시도·체크포인트는 자체 framework가 아니라
LangGraph 기능을 우선 사용한다. 세부 설계와 평가 원칙은
`docs/AGENTIC_PLANNING.md`를 따른다.

### 1. LangGraph 연구 계약

- [x] 보드 분석, 전략 가설, 상자-목표 배정, 보호 제약, 하위 목표, 예상 효과,
  실패 조건과 계획 수정을 표현하는 state schema와 JSON fixture를 정의한다.
- [x] 누적 decision·revision event에 LangGraph state reducer를 적용한다.
- [x] prompt·모델 설정은 Agent Server가 직렬화할 수 있는 LangGraph runtime
  context로 주입하고, 환경 관찰은 checkpoint 가능한 state로 유지한다.
- [x] 전역 bounded A*를 평가 oracle seam으로 분리하고 주 `StateGraph`가
  oracle 결과를 참조하지 않는 테스트를 작성한다.
- [x] 기존 원시 행동 LLM, full-guard와 A* 정책의 결과·지표 형식을 baseline
  계약으로 고정한다.

완료 조건: 모델 호출 없이 state update와 reducer를 재현하고, 주 compiled
graph에서 전역 oracle 호출이 0회임을 테스트로 증명한다.

### 2. 보드 분석

- [x] 관찰에서 상자·목표의 안정적인 논리 ID와 플레이어 도달 영역을 계산한다.
- [x] 가능한 push 방향, 정적 dead square와 목표별 reverse-pull 사실을
  하나의 보드 분석 Module에서 반환한다.
- [x] 회전, 좁은 통로, 복수 상자 fixture로 분석의 결정성과 규칙 정합성을
  검증한다.
- [x] `analyze` node update를 LangGraph Studio checkpoint에서 확인한다.

완료 조건: 전략 node가 공간 알고리즘의 구현 세부를 알지 않고
`BoardAnalysis` state만으로 동일한 사실을 소비한다.

### 3. prompt 수명 주기와 전략 Planner

- [ ] LangSmith Prompt Management에 구조화 전략 prompt를 만들고 연구
  실행에서 prompt commit을 고정한다.
- [x] `resolve_prompt → compose_strategy_input → propose_strategy`를
  관찰 가능한 LangGraph node로 구성한다.
- [x] prompt 이름·resolved commit·모델 설정을 결과에 기록한다.
- [x] LLM이 원시 방향 행동 대신 구조화된 전략 가설과 하위 목표 하나를
  반환하도록 schema를 추가한다.
- [x] transient prompt·모델 오류는 LangGraph retry policy로 처리하고,
  schema·의미 오류는 state와 conditional edge로 수정 경로에 보낸다.
- [x] 네트워크 없이 그래프를 시험할 고정 prompt와 모델 fixture를 만든다.
- [x] 존재하지 않는 상자·목표·칸, 모순된 배정과 보호 제약을 구조화된
  feedback으로 거절한다.
- [x] 자연어 근거를 제거해도 실행 필드만 유지되는 ablation을 만든다.

완료 조건: prompt가 Planner 내부 문자열에 숨지 않고 Studio trace에서
node·commit·입력 변수를 확인할 수 있다. 같은 commit·입력·seed 실행을
재현하고 형식·의미 오류를 환경 전이 전에 검출한다.

### 4. 하위 목표 검증과 국소 실행

- [x] 플레이어 위치와 단일 push 하위 목표의 사전 조건을 검증한다.
- [x] 검증된 하위 목표만 제한된 원시 행동으로 접지하는 국소 실행 Module을
  구현한다.
- [x] 보호 제약 위반, 접근 불가와 명백한 데드락을 실행 전에 거절한다.
- [x] 국소 도구가 전체 퍼즐 성공 상태를 목표로 탐색하지 못하게 계약
  테스트를 작성한다.

완료 조건: 세부 이동이 필요한 단일 push를 수행하되, 불가능하거나 장기
전략과 모순되는 하위 목표를 전역 A* 대체 없이 전략 node에 돌려보낸다.

### 5. 단일 StateGraph의 관찰·성찰·수정 루프

- [x] 구조화 정책의 CLI·평가와 Studio node·routing을 하나의 `StateGraph`로
  통합하고 `langgraph.json`이 이 graph를 직접 가리키게 한다. 기존 원시
  행동 정책은 비교 baseline으로만 보존한다.
- [x] 고정 전이는 normal edge, 상태 분기는 conditional edge, update와
  routing을 함께 해야 하는 node만 `Command`를 사용한다.
- [x] 실행은 최대 한 번의 push 후 반드시 `observe`로 돌아가게 한다.
- [x] 실행 전후 보드를 비교해 예상 효과의 충족 여부를 구조화한다.
- [x] 성공한 하위 목표를 완료하고 다음 하위 목표를 선택한다.
- [x] 반증된 가설과 수정된 필드를 `PlanRevision`으로 체크포인트한다.
- [x] 같은 상태·가설·하위 목표의 반복을 탐지하고 유한하게 종료한다.
- [x] 별도 checkpoint 저장 계층 없이 local은 `InMemorySaver`, Agent Server는
  제공 persistence를 사용한다.

완료 조건: 의도적인 실패 fixture에서 관련 하위 목표나 가설만 수정하고 같은
실패의 무한 반복을 막는다. CLI·평가·Studio가 동일한 compiled graph와
checkpoint history를 사용한다.

### 5.1 LangGraph 실시간 관찰 화면

세부 화면, stream 계약과 완료 기준은 `docs/LIVE_VIEWER.md`를 따른다.

- [x] Agent Server streaming run에서 node state update를 브라우저로 받는다.
- [x] 실제 `board` state를 CSS Grid의 벽·목표·상자·플레이어로 렌더링한다.
- [x] 행동 event마다 `@`, `$`, `*` 위치를 실시간으로 갱신한다.
- [x] 오른쪽 패널에 node, 행동, 전략 가설, 하위 목표, 보호 제약, 예상 효과와
  실제 결과를 같은 event ID 기준으로 표시한다.
- [x] graph node를 지연하지 않는 브라우저 표시 queue와 재생 속도 controls를
  구현한다.
- [x] 연결 복구, 성공·데드락·제한·오류와 reduced-motion을 검증한다.

완료 조건: `tiny-walk` run이 끝나기 전에 첫 행동이 웹에 나타나고, 표시된
최종 board가 graph 최종 state와 일치한다. 시각화 지연은 정책 시간에
포함하지 않는다.

### 6. 일반화와 인과적 기여 평가

- [ ] 배치, 보드 크기, 상자 수, 통로 구조와 함정 유형별 held-out manifest를
  만든다.
- [ ] 원시 행동 LLM, 구조화된 LLM, 구조화+국소 탐색, 설명 제거,
  current-full-guard와 A* oracle을 같은 사례로 비교한다.
- [ ] 하위 목표 성공률, 배정·가설 수정, 보호 제약 위반, 예상 효과 일치와
  설명-행동 인과 지표를 결과에 추가한다.
- [ ] 규칙·도달성·국소 탐색 호출과 LLM 토큰·시간 비용을 분리한다.
- [ ] prompt commit, graph 설정, 모델과 seed를 결과 manifest에 고정한다.
- [ ] 재현 가능한 비교 노트북과 trajectory를 만든다.

완료 조건: test 코호트가 prompt와 개발 fixture에서 격리되고, 모든 정책의
성공·비용·일반화·수정 능력을 같은 표에서 비교한다.

## 보류

- [ ] 실패 가설·전략을 검색하는 `recall` node와 LangGraph Store 기반 장기
  기억을 추가한다.
- [ ] 렌더링 화면을 읽는 perception과 인식 불확실성을 추가한다.
- [ ] 더 강한 동적 데드락 분석과 학습된 전략·가치 모델을 실험한다.
- [ ] 구조화된 에이전트 결과를 바탕으로 절차 생성 레벨과 자동 curriculum으로
  확장한다.
