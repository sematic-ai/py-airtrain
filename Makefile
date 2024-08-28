SHELL := /bin/bash
PROJECT_NAME := airtrain
PY_VERSION := "3.11"

.PHONY: wheel
wheel:
	uvx pip wheel -w dist .

.PHONY: test-release
test-release: wheel
	uvx twine check dist/*airtrain*.whl
	uvx twine upload --repository testpypi dist/*airtrain*.whl

.PHONY: release
release: wheel
	uvx twine check dist/*airtrain*.whl
	uvx twine upload dist/*airtrain*.whl

.PHONY: py-prep
py-prep:
	uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh
	rm -rf ".venv" || echo "No virtualenv yet"
	uv venv --python $(PY_VERSION)
	uv tool install --force ruff==0.6.1
	uv add --editable .

.PHONY: sync
sync:
	uv sync

.PHONY: fix
fix:
	uvx ruff format
	uvx ruff check --fix --show-fixes src/airtrain

.PHONY: lint
lint:
	uvx ruff format --check src/airtrain
	uvx ruff check --fix src/airtrain
	uv run mypy --explicit-package-bases src/airtrain

.PHONY: test
test:
	uv run pytest ./
