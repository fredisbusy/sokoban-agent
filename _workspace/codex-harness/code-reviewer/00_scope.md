# 코드 구조 리뷰 범위

## 요청

현재 코드 구조에서 리팩토링이 필요한 요소를 점검한다. 구현 변경은 하지 않는다.

## 기준

- `docs/PROJECT.md`의 LangGraph 중심 구조와 연구 재현성
- `TODO.md`의 현재 구조화 에이전트 완료 조건
- 정확성, 보안, 성능, 아키텍처, 유지보수성, 테스트 경계
- `src/`, `tests/`, `scripts/`의 직접 작성 Python 파일 400줄 제한

## 조사 범위

- `src/sokoban_agent/{env,planning,graph,evaluation}`
- `viewer/`
- `scripts/prepare_boxoban_research.py`
- `pyproject.toml`, `langgraph.json`, 관련 테스트와 아키텍처 문서

## 검증

- `UV_CACHE_DIR=/tmp/sokoban-uv-cache uv run pytest`: 134 passed
- `UV_CACHE_DIR=/tmp/sokoban-uv-cache uv run ruff check .`: passed
- `UV_CACHE_DIR=/tmp/sokoban-uv-cache uv run mypy`: passed
- `viewer/npm test`: 실패. Node 22.16.0이 `.ts` 테스트를 직접 로드하지 못함
- Viewer typecheck/build: `node_modules`가 없어 실행하지 못함

Python 파일은 모두 400줄 이하이며, 정확히 400줄인 파일은 2개다.

## 2026-07-24 추가 검토 범위

- LangGraph 공식 application structure와 graph API 기준 비교
- 현재 graph family와 entrypoint별 provider wiring 추적
- NestJS 개발자 관점의 module/provider 발견성 평가
- 제품 코드는 변경하지 않고 graph 관련 표적 테스트만 재검증
