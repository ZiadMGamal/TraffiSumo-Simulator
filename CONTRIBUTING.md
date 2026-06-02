# Contributing

Thank you for contributing to the Cooperative Smart Traffic MARL project.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment and install dependencies:

```bash
pip install -r requirements-dev.txt
cd frontend && npm install
```

3. Copy `.env.example` to `.env` and configure `SUMO_HOME` if available

## Running Without SUMO

Set `ENV_MODE=mock` or omit `SUMO_HOME` to use the built-in mock traffic environment for API and dashboard development.

## Code Standards

- Python 3.10+ with type hints where practical
- Follow existing module layout and naming conventions
- No inline comments unless explaining non-obvious logic
- Run `pytest tests/ -v` before submitting pull requests
- Run `ruff check` on changed Python files

## Pull Request Process

1. Create a feature branch from `main`
2. Add or update tests for new behavior
3. Update README if user-facing behavior changes
4. Ensure CI passes (GitHub Actions)
5. Submit PR with clear description and test plan

## Reporting Issues

Include OS, Python version, SUMO version, error logs, and steps to reproduce.
