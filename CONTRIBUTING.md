# Contributing to Datasets Server Python Client

Thank you for your interest in contributing to the Datasets Server Python Client! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct (see CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Issues

- Check if the issue has already been reported
- Use the issue templates when available
- Provide a clear description of the problem
- Include steps to reproduce the issue
- Mention your Python version and operating system

### Suggesting Features

- Open an issue to discuss the feature before implementing
- Explain the use case and why it would be beneficial
- Consider how it fits with the project's goals

### Submitting Changes

1. **Fork the Repository**
   ```bash
   git clone https://github.com/davanstrien/datasets-server-py
   cd datasets-server-py
   ```

2. **Set Up Development Environment**
   ```bash
   # Install UV if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

5. **Run Quality Checks**
   ```bash
   # Format code
   uv run ruff format src/ tests/
   
   # Lint code
   uv run ruff check src/ tests/
   
   # Type checking
   uv run mypy src/
   
   # Run tests
   uv run pytest
   
   # Run tests with coverage
   uv run pytest --cov=datasets_server
   ```

6. **Commit Your Changes**
   - Use clear, descriptive commit messages
   - Follow conventional commit format if possible (e.g., `feat:`, `fix:`, `docs:`)
   
   ```bash
   git add .
   git commit -m "feat: add support for new endpoint"
   ```

7. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a pull request on GitHub.

## Development Guidelines

### Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Maximum line length is 119 characters
- Use type hints for all function signatures
- Write docstrings for all public functions and classes

### Testing

- Write tests for all new functionality
- Maintain or improve the current test coverage (87%+)
- Use pytest for all tests
- Mock external API calls in unit tests
- Integration tests should be marked with `@pytest.mark.integration`

### Documentation

- Update README.md if adding new features
- Add docstrings to all public APIs
- Include examples for complex functionality
- Update CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/)

### Type Safety

- All code must pass mypy type checking
- Use Pydantic models for data validation
- Avoid using `Any` type where possible

## Project Structure

```
src/datasets_server/
├── __init__.py          # Public API exports
├── client.py            # Synchronous client
├── async_client.py      # Asynchronous client
├── _base.py            # Shared base functionality
├── models.py           # Pydantic models
├── exceptions.py       # Custom exceptions
└── __version__.py      # Version information
```

## Running Integration Tests

Integration tests make real API calls and are disabled by default:

```bash
# Run integration tests
INTEGRATION_TESTS=1 uv run pytest -m integration

# Run all tests including integration
INTEGRATION_TESTS=1 uv run pytest
```

## Questions?

Feel free to open an issue for any questions about contributing. We're here to help!