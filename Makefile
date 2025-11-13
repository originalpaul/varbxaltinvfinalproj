.PHONY: help install setup test lint format clean data all

help:
	@echo "VARBX Due Diligence - Makefile Commands"
	@echo ""
	@echo "  make install    - Install project dependencies"
	@echo "  make setup      - Set up pre-commit hooks"
	@echo "  make test       - Run tests with coverage"
	@echo "  make lint       - Run linting (ruff and mypy)"
	@echo "  make format     - Format code (black and isort)"
	@echo "  make clean      - Remove generated files"
	@echo "  make data       - Download benchmark data via yfinance"
	@echo "  make all        - Run full pipeline (format, lint, test)"

install:
	pip install -r requirements.txt
	pip install -e .

setup:
	pre-commit install

test:
	pytest

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	isort src tests

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
	rm -rf htmlcov .coverage dist build

data:
	python -c "from src.data.loaders import download_benchmarks; download_benchmarks()"

all: format lint test

