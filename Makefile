.PHONY: install
install:
	uv sync

.PHONY: update-deps
update-deps:
	uv lock --upgrade
	uv sync

.PHONY: build
build:
	uv build

.PHONY: pytest
pytest:
	uv run pytest tests/

.PHONY: pyright
pyright:
	uv run pyright .
