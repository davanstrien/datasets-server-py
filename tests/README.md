# Tests

This directory contains both unit tests and integration tests for the Datasets Server Python client.

## Running Tests

### Unit Tests Only (Default)
```bash
# Run all unit tests
uv run pytest

# Run with coverage
uv run pytest --cov=datasets_server

# Run specific test file
uv run pytest tests/test_client.py

# Run specific test
uv run pytest tests/test_client.py::TestDatasetsServerClient::test_sample_rows_basic
```

### Integration Tests
Integration tests make real API calls to the Hugging Face Datasets Server. They are skipped by default.

```bash
# Run all integration tests
INTEGRATION_TESTS=1 uv run pytest -m integration

# Run integration tests with verbose output
INTEGRATION_TESTS=1 uv run pytest -m integration -xvs

# Run specific integration test
INTEGRATION_TESTS=1 uv run pytest tests/test_integration.py::TestIntegrationSync::test_is_valid_real_dataset -m integration
```

### All Tests (Unit + Integration)
```bash
# Run all tests including integration
INTEGRATION_TESTS=1 uv run pytest -m ""
```

## Test Structure

- `test_client.py` - Unit tests for synchronous client
- `test_async_client.py` - Unit tests for asynchronous client  
- `test_models.py` - Unit tests for Pydantic models
- `test_integration.py` - Integration tests that call real API

## Test Datasets

Integration tests use these verified public datasets:
- `SetFit/ag_news` - News classification dataset
- `stanfordnlp/imdb` - Movie reviews dataset
- `davanstrien/haiku_dpo` - Small dataset for quick tests
- `librarian-bots/dataset_cards_with_metadata` - Dataset with rich metadata

## CI/CD

- Unit tests run on every commit
- Integration tests run:
  - Daily at 2 AM UTC
  - On PRs with the `integration` label
  - Manually via workflow dispatch