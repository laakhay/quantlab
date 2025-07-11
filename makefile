.PHONY: help venv venv-clean install install-dev test lint format type-check clean all

help:
	@echo "Available commands:"
	@echo "  make venv         Create virtual environment"
	@echo "  make venv-clean   Remove virtual environment"
	@echo "  make install      Install runtime dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linting"
	@echo "  make format       Format code"
	@echo "  make type-check   Run type checking"
	@echo "  make clean        Clean up cache files"
	@echo "  make all          Run all checks (format, lint, type-check, test)"

venv:
	python3 -m venv .venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

venv-clean:
	rm -rf .venv

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=laakhay --cov-report=term-missing

lint:
	ruff check laakhay/ tests/

format:
	black laakhay/ tests/
	isort laakhay/ tests/

type-check:
	mypy laakhay/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage .pytest_cache .mypy_cache .ruff_cache

all: format lint type-check test