SHELL := /bin/sh

# Require uv
UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
  $(error uv is required but not found. Install it from https://github.com/astral-sh/uv)
endif

PYTHON_VERSION ?= 3.12
PY := $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: help install test lint lint-fix format format-check type-check check fix clean build

help:
	@echo "Make targets:"
	@echo "  install         Install project and dev dependencies."
	@echo "  test            Run tests."
	@echo "  lint            Run ruff lint check."
	@echo "  lint-fix        Run ruff lint and auto-fix issues."
	@echo "  format          Run ruff format."
	@echo "  format-check    Check if code is formatted correctly."
	@echo "  type-check      Run mypy check."
	@echo "  check           Run all checks (lint + format check)."
	@echo "  fix             Auto-fix all fixable issues (lint + format)."
	@echo "  clean           Remove caches and build artifacts."
	@echo "  build           Build distribution packages."

install:
	@if [ ! -d .venv ]; then $(UV) venv --python $(PYTHON_VERSION); fi
	@$(UV) sync --extra dev

test:
	@$(UV) run pytest tests/ -v --cov=laakhay/quantlab --cov-report=term-missing

lint:
	@$(UV) run ruff check .

lint-fix:
	@$(UV) run ruff check --fix .

format:
	@$(UV) run ruff format .

format-check:
	@$(UV) run ruff format --check .

type-check:
	@$(UV) run mypy laakhay/quantlab

check: lint format-check

fix: lint-fix format

clean:
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	@find . -name '*.pyc' -delete
	@rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build *.egg-info

build: clean
	@$(UV) build