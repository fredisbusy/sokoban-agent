# 의존성과 기술 선택

기본 설치는 Sokoban 환경과 기준선 실행에 필요한 패키지만 포함한다.
기능별 라이브러리는 배포 기능이면 optional extra, 로컬 연구 도구이면
dependency group으로 분리한다.

## 설치 경계

| 범위 | 구성 | 설치 |
| --- | --- | --- |
| core | Gymnasium, NumPy | `uv sync --no-dev` |
| LLM | LiteLLM, Pydantic, python-dotenv | `uv sync --no-dev --extra llm` |
| vision | OpenCV, Pillow | `uv sync --no-dev --extra vision` |
| dev | pytest, Ruff, mypy와 LLM 테스트 의존성 | `uv sync` |
| notebook | Jupyter, nbclient, nbformat, pandas, Matplotlib | `uv sync --group notebook` |

`dev`에는 optional LLM 기능까지 항상 테스트하기 위해 LLM 패키지를
의도적으로 다시 선언한다. 배포 패키지의 기본 의존성에는 포함되지 않는다.

## 기준선 탐색

BFS는 외부 그래프 패키지 대신 표준 라이브러리 `collections.deque`를
사용한다. Sokoban 상태 그래프는 실행 중 이웃을 생성하므로, NetworkX나
rustworkx를 사용하려면 같은 상태를 별도 그래프로 먼저 물질화해야 한다.
이동·밀기·데드락은 프로젝트의 순수 규칙 함수가 담당하고 BFS는 탐색 순서와
경로 복원만 담당한다.

## 분석

에피소드 결과와 집계는 core에서 immutable dataclass와 표준 라이브러리로
계산한다. pandas와 Matplotlib은 표와 차트가 필요한 노트북에서만 사용한다.

## 향후 memory

현재 BFS의 방문 집합은 한 에피소드 안의 탐색 상태이며 장기 기억이 아니다.
LLM 에이전트가 안정된 뒤 실패 계획과 데드락을 세션 간 검색할 요구가 생기면
먼저 JSONL 또는 SQLite 기준선을 만들고 mem0의 검색 품질, 재현성, 외부
모델·저장소 비용을 비교한다. 검토 전에는 core 의존성에 추가하지 않는다.
