.PHONY: install install-dev lint fmt typecheck unit synthetic db-up db-down all clean

PY ?= .venv/bin

install:
	$(PY)/pip install -e .

install-dev:
	$(PY)/pip install -e ".[dev]"

lint:
	$(PY)/ruff check src tests
	$(PY)/ruff format --check src tests

fmt:
	$(PY)/ruff check --fix src tests
	$(PY)/ruff format src tests

typecheck:
	$(PY)/mypy src

unit:
	$(PY)/pytest tests/unit -m "not integration and not pymc"

# Synthetic distributor bundle (demo profile) — the input to every local flow.
synthetic:
	$(PY)/mro generate-synthetic --profile demo --out data/synthetic

# Local PostgreSQL (pgvector image) for the RLS tenant-isolation suite.
db-up:
	docker compose up -d db

db-down:
	docker compose down -v

all: lint typecheck unit

clean:
	rm -rf data output .pytest_cache .ruff_cache .mypy_cache .coverage
