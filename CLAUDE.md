**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads)
for issue tracking. Use `bd` commands instead of markdown TODOs.
See AGENTS.md for workflow details.

# Datasets Server API Python Client - Implementation Plan

## Executive Summary

This document outlines the implementation plan for `datasets-server-py`, a standalone Python client library for the Hugging Face Datasets Viewer API. The package will be designed for easy future integration into `huggingface_hub` while maintaining simplicity and usefulness as an independent tool. Development will use UV for modern Python packaging and dependency management. **The client will support both synchronous and asynchronous operations from the start.**

## Project Overview

### Package Name

`datasets-server-py` (or `datasets-viewer-py`)

### Goals

1. Create a simple, functional client for the Datasets Viewer API
2. **Provide both sync and async interfaces**
3. Validate user interest and gather feedback
4. Design for easy future integration into huggingface_hub
5. Minimize complexity while maximizing utility

### Non-Goals (for initial version)

1. Full huggingface_hub integration
2. Complex caching mechanisms
3. CLI interface

## Project Structure

```
datasets-server-py/
├── pyproject.toml              # UV/modern Python packaging
├── README.md                   # Documentation and examples
├── LICENSE                     # Apache 2.0
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml             # Basic CI/CD
├── src/
│   └── datasets_server/
│       ├── __init__.py        # Public API exports
│       ├── client.py          # Synchronous client
│       ├── async_client.py    # Asynchronous client
│       ├── _base.py           # Shared base functionality
│       ├── models.py          # Response models
│       ├── exceptions.py      # Custom exceptions
│       └── __version__.py     # Version info
├── tests/
│   ├── __init__.py
│   ├── test_client.py
│   ├── test_async_client.py
│   ├── test_models.py
│   ├── test_integration.py    # Integration tests
│   └── README.md              # Test documentation
├── examples/
│   ├── basic_usage.py
│   ├── async_usage.py
│   ├── search_and_filter.py
│   ├── sample_rows_demo.py    # Random sampling examples
│   └── explore_dataset.ipynb
├── CLAUDE.md                  # Development notes
└── HELPER_FUNCTIONS.md        # Helper function planning doc
```

## API Endpoints Reference

Base URL: `https://datasets-server.huggingface.co`

### 1. Dataset Validation

- **Endpoint**: `/is-valid`
- **Method**: GET
- **Parameters**: `dataset` (required)
- **Response**: `{viewer: bool, preview: bool, search: bool, filter: bool, statistics: bool}`

### 2. List Splits

- **Endpoint**: `/splits`
- **Method**: GET
- **Parameters**: `dataset` (required)
- **Response**: `{splits: [...], pending: [...], failed: [...]}`

### 3. Dataset Information

- **Endpoint**: `/info`
- **Method**: GET
- **Parameters**: `dataset` (required), `config` (optional)
- **Response**: Dataset metadata including features, description, citation, etc.

### 4. Dataset Size

- **Endpoint**: `/size`
- **Method**: GET
- **Parameters**: `dataset` (required)
- **Response**: Size information including rows and bytes

### 5. Parquet Files

- **Endpoint**: `/parquet`
- **Method**: GET
- **Parameters**: `dataset` (required)
- **Response**: List of Parquet file URLs with metadata

### 6. Preview Rows

- **Endpoint**: `/first-rows`
- **Method**: GET
- **Parameters**: `dataset` (required), `config` (optional), `split` (optional)
- **Response**: First 100 rows with features schema (max 1MB)

### 7. Get Rows

- **Endpoint**: `/rows`
- **Method**: GET
- **Parameters**: `dataset`, `config`, `split`, `offset`, `length` (max 100)
- **Response**: Paginated dataset rows

### 8. Search Dataset

- **Endpoint**: `/search`
- **Method**: GET
- **Parameters**: `dataset`, `config`, `split`, `query`, `offset`, `length`
- **Response**: Rows matching search query

### 9. Filter Dataset

- **Endpoint**: `/filter`
- **Method**: GET
- **Parameters**: `dataset`, `config`, `split`, `where`, `orderby`, `offset`, `length`
- **Response**: Filtered rows based on SQL-like conditions

### 10. Dataset Statistics

- **Endpoint**: `/statistics`
- **Method**: GET
- **Parameters**: `dataset`, `config`, `split`
- **Response**: Statistical analysis of dataset columns

## Implementation Notes

### Authentication

- Uses HuggingFace tokens for private/gated datasets
- Token retrieved from huggingface_hub if available
- Optional token parameter for explicit authentication

### Error Handling

- Custom exceptions for different error scenarios
- Clear error messages for debugging
- Proper handling of 404s, timeouts, and API errors

### Response Models

- Pydantic models for type safety and validation
- All responses properly typed
- Partial results handled gracefully

### Async Support

- Full async client with same API as sync client
- Proper session management
- Support for concurrent requests
- AsyncIterator for row iteration

### Development Practices

- UV for package management
- Ruff for linting and formatting
- pytest for testing (with pytest-asyncio)
- Type hints throughout
- Comprehensive docstrings

## Testing Strategy

1. **Unit tests** with mocked responses (50 tests)
2. **Async tests** with aioresponses
3. **Integration tests** with real datasets (21 tests)
   - Controlled by `INTEGRATION_TESTS=1` environment variable
   - Skip by default to avoid API calls during normal development
   - Test against verified public datasets
4. **Test coverage**: Currently at 87%

## Future Integration Path

This standalone package is designed to be easily integrated into huggingface_hub:

1. Move to `huggingface_hub.datasets_viewer` module
2. Reuse huggingface_hub utilities (headers, session, etc.)
3. Inherit from huggingface_hub exception classes
4. Follow huggingface_hub documentation patterns

## Success Metrics

1. All 10 API endpoints accessible ✅
2. Both sync and async clients working ✅
3. Clean, intuitive API ✅
4. Good performance for large datasets ✅
5. Easy installation and setup ✅
6. Positive user feedback ⏳

## Implementation Status

### Completed Features

- Full implementation of all 10 API endpoints
- Both synchronous and asynchronous clients
- Type-safe Pydantic models for all responses
- Comprehensive error handling with custom exceptions
- Token authentication integration with huggingface_hub
- Context manager support for proper resource cleanup
- Iterator support for paginated data access
- 87% test coverage with comprehensive test suite

### Recent Improvements

- Fixed DatasetSize model to match actual API response structure
- Resolved environment-dependent test failures
- Added proper exception chaining (fixed ruff B904 violations)
- Refactored magic numbers to named constants (MAX_ROWS_PER_REQUEST)
- Enhanced test coverage from 76% to 87%
- Fixed critical bug in sample_rows where API returns splits as dict, not list
- Added comprehensive integration test suite with real API calls
- Implemented pytest markers for test categorization (unit/integration)
- Created GitHub Actions CI/CD workflow

### Code Quality

- All ruff linting checks passing
- Type hints throughout the codebase
- Comprehensive docstrings for all public APIs
- Clean separation of concerns with base class architecture
- Follows modern Python packaging standards with UV

## Helper Functions (Planned)

The following helper functions have been identified for future implementation:

### Priority 1 - Core Helpers

- ✅ `sample_rows(dataset, config, split, n_samples, seed)` - Get random sample of rows (COMPLETED)
- `get_first_valid_split(dataset)` - Find the first valid split for a dataset
- `get_default_split(dataset)` - Get the default/recommended split

### Priority 2 - Data Analysis

- `find_text_columns(dataset, config, split)` - Identify text columns
- `find_numeric_columns(dataset, config, split)` - Identify numeric columns
- `get_feature_types(dataset, config, split)` - Get dict of column names to types

### Priority 3 - Dataset Health

- `get_dataset_summary(dataset)` - Get comprehensive dataset overview
- `check_dataset_health(dataset)` - Check if all expected features work

## Running Tests

### Unit Tests (default)

```bash
uv run pytest
```

### Integration Tests

```bash
# For bash/zsh
INTEGRATION_TESTS=1 uv run pytest

# For fish shell
env INTEGRATION_TESTS=1 uv run pytest
```

### Full Test Suite with Coverage

```bash
# For fish shell
env INTEGRATION_TESTS=1 uv run pytest --cov=datasets_server --cov-report=html
```

## CI/CD

- **Unit tests**: Run on every push/PR
- **Integration tests**: Run on schedule (daily), manual trigger, or PRs with 'integration' label
- **Python versions**: Tests run on Python 3.8-3.12
- **Code coverage**: Automated reporting to Codecov
