# Contributing

Use Python 3.11 or 3.12 and `uv` for all environments and commands.

```bash
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --frozen --extra train --group dev
PYTHONPATH="$PWD/src" UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync ruff check .
PYTHONPATH="$PWD/src" UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync ruff format --check .
PYTHONPATH="$PWD/src" UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync mypy
PYTHONPATH="$PWD/src" UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync pytest
UV_PROJECT_ENVIRONMENT=.venv-train uv build
UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync python scripts/audit_release.py .
```

There is no hosted CI workflow; run this checklist from a clean checkout before
publishing. Changes to a paper configuration must include a regression test.
Do not commit model weights, dataset text, rankings, per-query outputs,
credentials, machine-specific paths, or scheduler configuration.
