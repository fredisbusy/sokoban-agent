LEVEL ?=

.PHONY: help sync play ollama-check lab studio viewer-sync viewer \
	boxoban-research-data \
	test lint typecheck check baseline-notebook experiment-notebook \
	agentic-notebook levels levels-check

help:
	@echo "설치"
	@echo "  make sync              개발·연구 의존성 설치"
	@echo ""
	@echo "실행"
	@echo "  make play              터미널 게임 실행"
	@echo "  make play LEVEL=이름   지정한 레벨 실행"
	@echo "  make ollama-check      Ollama 연결 확인"
	@echo "  make lab               JupyterLab 실행"
	@echo "  make studio            LangGraph Studio 실행"
	@echo "  make viewer-sync       관찰 화면 의존성 설치"
	@echo "  make viewer            실시간 Sokoban 관찰 화면 실행"
	@echo "  make boxoban-research-data 난이도별 연구 맵 다운로드·검증"
	@echo "  make levels            Boxoban+사용자 맵 catalog 생성"
	@echo ""
	@echo "검증과 실험"
	@echo "  make check             테스트, 린트, 타입 검사"
	@echo "  make test              테스트"
	@echo "  make lint              Ruff 린트"
	@echo "  make typecheck         mypy 타입 검사"
	@echo "  make levels-check      생성된 맵 catalog 최신 상태 확인"
	@echo "  make baseline-notebook 기준선 노트북 생성 및 실행"
	@echo "  make experiment-notebook 주 실험 노트북 생성 및 실행"
	@echo "  make agentic-notebook  구조화 일반화 실험 노트북 생성"

sync:
	uv sync --all-groups

play:
	uv run sokoban-play $(if $(LEVEL),--level $(LEVEL))

ollama-check:
	uv run python scripts/check_ollama.py

lab:
	uv run --group notebook jupyter lab

studio:
	uv run --group studio langgraph dev --host localhost

viewer-sync:
	cd viewer && npm ci

viewer:
	cd viewer && npm run dev

boxoban-research-data:
	uv run python scripts/prepare_boxoban_research.py --download

levels:
	uv run python scripts/build_level_catalog.py

levels-check:
	uv run python scripts/build_level_catalog.py --check

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run --group notebook mypy

check: levels-check test lint typecheck

baseline-notebook:
	uv run --group notebook python scripts/build_baseline_notebook.py
	uv run --group notebook python -m jupyter nbconvert \
		--execute --to notebook --inplace notebooks/baseline_comparison.ipynb

experiment-notebook:
	uv run --group notebook python scripts/build_langgraph_comparison_notebook.py
	uv run --group notebook python -m jupyter nbconvert \
		--execute --to notebook --inplace \
		notebooks/langgraph_planner_comparison.ipynb

agentic-notebook:
	uv run --group notebook python scripts/build_agentic_research_notebook.py
