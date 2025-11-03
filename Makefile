.PHONY: help install install-dev test test-unit test-integration lint format clean run train data

# Default target
help:
	@echo "AI Soccer Predictions - Available commands:"
	@echo ""
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run all tests"
	@echo "  test-unit    - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean build artifacts"
	@echo "  run          - Run the API server"
	@echo "  train        - Train models"
	@echo "  data         - Run data pipeline"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up    - Start services with Docker Compose"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

# Testing
test: install-dev
	pytest tests/ -v --cov=src

test-unit: install-dev
	pytest tests/unit/ -v --cov=src

test-integration: install-dev
	pytest tests/integration/ -v

# Code Quality
lint: install-dev
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format: install-dev
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ipynb_checkpoints/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Development
run: install
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod: install
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

train: install
	python scripts/train_models.py --all

data: install
	python scripts/data_pipeline.py --collect --process

# Docker
docker-build:
	docker build -t soccer-predictions-api:latest -f docker/Dockerfile.api .
	docker build -t soccer-predictions-training:latest -f docker/Dockerfile.training .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

# Model Management
models-list:
	python scripts/model_registry.py --list

models-evaluate:
	python scripts/evaluate_models.py --all

models-deploy:
	python scripts/deploy_models.py --model ensemble --version latest

# Data Management
data-collect:
	python scripts/data_pipeline.py --collect --sources all

data-process:
	python scripts/data_pipeline.py --process --clean --augment

data-validate:
	python scripts/data_pipeline.py --validate

# Monitoring
monitor-models:
	python scripts/monitor_performance.py --check-all

monitor-system:
	python scripts/monitor_system.py --all

# Documentation
docs-serve: install-dev
	mkdocs serve

docs-build: install-dev
	mkdocs build

# Database
db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

db-reset:
	alembic downgrade base && alembic upgrade head

# Utility
notebook:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

shell:
	ipython

profile:
	python -m cProfile -o profile.stats scripts/train_models.py --model transformer
	snakeviz profile.stats
