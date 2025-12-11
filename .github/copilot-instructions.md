# GitHub Copilot Instructions for datasets-server-py

## Project Overview

**datasets-server-py** is a Python client library for the Hugging Face Datasets Viewer API. It provides both synchronous and asynchronous interfaces.

**Key Features:**
- Full API coverage for all 10 datasets-server endpoints
- Both sync and async clients
- Pydantic models for type-safe responses
- Token authentication integration with huggingface_hub

## Tech Stack

- **Language**: Python 3.8+
- **Package Manager**: UV
- **HTTP Client**: httpx (planned migration from requests)
- **Models**: Pydantic v2
- **Testing**: pytest, pytest-asyncio
- **Linting**: Ruff
- **Type Checking**: mypy

## Issue Tracking with bd

**CRITICAL**: This project uses **bd** for ALL task tracking. Do NOT create markdown TODO lists.

### Essential Commands

```bash
# Find work
bd ready --json                    # Unblocked issues
bd stale --days 30 --json          # Forgotten issues

# Create and manage
bd create "Title" -t bug|feature|task -p 0-4 --json
bd create "Subtask" --parent <epic-id> --json  # Hierarchical subtask
bd update <id> --status in_progress --json
bd close <id> --reason "Done" --json

# Search
bd list --status open --priority 1 --json
bd show <id> --json
```

### Workflow

1. **Check ready work**: `bd ready --json`
2. **Claim task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** `bd create "Found bug" -p 1 --deps discovered-from:<parent-id> --json`
5. **Complete**: `bd close <id> --reason "Done" --json`

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

## Project Structure

```
datasets-server-py/
├── src/datasets_server/
│   ├── __init__.py          # Public API exports
│   ├── _base.py             # Shared base functionality
│   ├── client.py            # Synchronous client
│   ├── async_client.py      # Asynchronous client
│   ├── models.py            # Pydantic response models
│   └── exceptions.py        # Custom exceptions
├── tests/
│   ├── test_client.py       # Sync client tests
│   ├── test_async_client.py # Async client tests
│   └── test_integration.py  # Real API tests
└── examples/
```

## Development Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,examples]"

# Quality checks
uv run ruff format src/ tests/
uv run ruff check src/ tests/
uv run mypy src/

# Testing
uv run pytest                              # Unit tests only
INTEGRATION_TESTS=1 uv run pytest          # All tests
uv run pytest --cov=datasets_server        # With coverage
```

## Coding Guidelines

### Testing
- Run `uv run pytest` before committing
- Use mocked responses for unit tests
- Integration tests require `INTEGRATION_TESTS=1`

### Code Style
- Run `uv run ruff check` and `uv run ruff format` before committing
- Follow existing patterns in the codebase
- Use type hints throughout
- Google-style docstrings

## Important Rules

- Use bd for ALL task tracking
- Always use `--json` flag for programmatic use
- Do NOT create markdown TODO lists
- Run tests before committing
- Follow huggingface_hub patterns for eventual integration

---

**For detailed workflows, see [AGENTS.md](../AGENTS.md)**
