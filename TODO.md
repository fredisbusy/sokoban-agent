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

## 지금 — LangGraph 중심 실행기

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

## 보류

- [ ] 렌더링 화면을 읽는 perception을 추가한다.
- [ ] `recall` 노드와 영속 저장소로 실패 계획·데드락 memory를 추가한다.
- [ ] Boxoban 파일럿 결과를 바탕으로 분할별 벤치마크 또는 절차 생성 레벨로
  확장한다.
