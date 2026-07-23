.PHONY: sync lab studio test lint typecheck

sync:
	uv sync

lab:
	uv run --group notebook jupyter lab

studio:
	uv sync --group studio
	uv run --group studio langgraph dev

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy
