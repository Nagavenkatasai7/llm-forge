# Contributing to llm-forge

Thank you for your interest in contributing to llm-forge!

## Development Setup

```bash
git clone https://github.com/Nagavenkatasai7/llm-forge.git
cd llm-forge
pip install -e ".[dev]"
pre-commit install
```

## Code Standards

- **Formatting**: We use `ruff` for linting and formatting
- **Type hints**: All public functions must have type annotations
- **Testing**: New features must include tests
- **Docstrings**: Google-style docstrings on public functions

## Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `make test`
5. Run linting: `make lint`
6. Commit with a descriptive message
7. Push and open a Pull Request

## Running Tests

```bash
# CPU tests (no GPU required)
make test

# GPU tests (requires NVIDIA GPU)
make test-gpu

# Linting
make lint
```

## Areas for Contribution

- New data format loaders
- Additional cleaning filters
- Benchmark integrations
- Documentation improvements
- Bug fixes

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests.
Include output from `llm-forge info` in bug reports.
