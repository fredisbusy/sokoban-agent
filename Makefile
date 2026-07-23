.PHONY: sync lab test lint typecheck

sync:
	uv sync

lab:
	uv run jupyter lab

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy

