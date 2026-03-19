.PHONY: install install-dev lint format test test-gpu clean build publish docker

install:
	pip install -e .

install-dev:
	pip install -e ".[all]"
	pre-commit install

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:
	pytest tests/ --ignore=tests/test_training/test_gpu.py -v --cov=llm_forge --cov-report=term-missing

test-gpu:
	pytest tests/ -v -m gpu

clean:
	rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build: clean
	python -m build

publish: build
	twine upload dist/*

docker:
	docker build -t llm-forge:latest -f docker/Dockerfile .

docker-gpu:
	docker build -t llm-forge:gpu -f docker/Dockerfile.gpu .
