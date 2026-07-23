LEVEL ?=

.PHONY: help sync sync-core sync-llm sync-vision sync-notebook sync-studio \
	play ollama-check lab studio test lint typecheck check \
	baseline-notebook experiment-notebook

help:
	@echo "설치"
	@echo "  make sync              개발 환경 설치"
	@echo "  make sync-core         core만 설치"
	@echo "  make sync-llm          core + LLM 설치"
	@echo "  make sync-vision       core + vision 설치"
	@echo "  make sync-notebook     notebook 도구 설치"
	@echo "  make sync-studio       LangGraph Studio 설치"
	@echo ""
	@echo "실행"
	@echo "  make play              터미널 게임 실행"
	@echo "  make play LEVEL=이름   지정한 레벨 실행"
	@echo "  make ollama-check      Ollama 연결 확인"
	@echo "  make lab               JupyterLab 실행"
	@echo "  make studio            LangGraph Studio 실행"
	@echo ""
	@echo "검증과 실험"
	@echo "  make check             테스트, 린트, 타입 검사"
	@echo "  make test              테스트"
	@echo "  make lint              Ruff 린트"
	@echo "  make typecheck         mypy 타입 검사"
	@echo "  make baseline-notebook 기준선 노트북 생성 및 실행"
	@echo "  make experiment-notebook 주 실험 노트북 생성 및 실행"

sync:
	uv sync

sync-core:
	uv sync --no-dev

sync-llm:
	uv sync --no-dev --extra llm

sync-vision:
	uv sync --no-dev --extra vision

sync-notebook:
	uv sync --group notebook

sync-studio:
	uv sync --group studio

play:
	uv run sokoban-play $(if $(LEVEL),--level $(LEVEL))

ollama-check:
	uv run python scripts/check_ollama.py

lab:
	uv run --group notebook jupyter lab

studio: sync-studio
	uv run --group studio langgraph dev

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run --group notebook mypy

check: test lint typecheck

baseline-notebook:
	uv run --group notebook python scripts/build_baseline_notebook.py
	uv run --group notebook python -m jupyter nbconvert \
		--execute --to notebook --inplace notebooks/baseline_comparison.ipynb

experiment-notebook:
	uv run --group notebook python scripts/build_langgraph_comparison_notebook.py
	uv run --group notebook python -m jupyter nbconvert \
		--execute --to notebook --inplace \
		notebooks/langgraph_planner_comparison.ipynb
