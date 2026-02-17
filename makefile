SHELL := /bin/sh

# Require uv
UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
  $(error uv is required but not found. Install it from https://github.com/astral-sh/uv)
endif

PYTHON_VERSION ?= 3.12
PY := $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: help install test lint format type-check fix clean build

help:
	@echo "Make targets:"
	@echo "  install         Install project and dev dependencies."
	@echo "  test            Run tests."
	@echo "  lint            Run ruff lint."
	@echo "  format          Run ruff format."
	@echo "  type-check      Run mypy check."
	@echo "  fix             Auto-fix linting and formatting."
	@echo "  clean           Remove caches and build artifacts."
	@echo "  build           Build distribution packages."

install:
	@if [ ! -d .venv ]; then $(UV) venv --python $(PYTHON_VERSION); fi
	@$(UV) sync --extra dev

test:
	@$(UV) run pytest tests/ -v --cov=laakhay/quantlab --cov-report=term-missing

lint:
	@$(UV) run ruff check .

format:
	@$(UV) run ruff format .

type-check:
	@$(UV) run mypy laakhay/quantlab

fix:
	@$(UV) run ruff check --fix .
	@$(UV) run ruff format .

clean:
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	@find . -name '*.pyc' -delete
	@rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build *.egg-info

build: clean
	@$(UV) build