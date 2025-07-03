# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `max_requests` parameter to `sample_rows` method for API-efficient sampling
  - Allows limiting the number of API calls when sampling from large datasets
  - Trades true randomness for API efficiency by sampling from dataset segments
  - Useful for scenarios where approximate randomness is acceptable

## [0.1.0] - 2025-07-02

### Added
- Initial release of datasets-server-py
- Full implementation of all 10 Datasets Viewer API endpoints
- Synchronous client with requests library
- Asynchronous client with aiohttp
- Type-safe Pydantic models for all API responses
- Custom exception classes for better error handling
- Context manager support for proper resource cleanup
- Iterator support for paginated data access
- Comprehensive test suite with 90% coverage
- Integration test suite with 21 tests calling real API
- GitHub Actions CI/CD workflow for automated testing
- Examples demonstrating basic usage, async operations, and advanced features
- Full documentation with installation and usage instructions

### Fixed
- DatasetSize model validation error - updated to match actual API response structure
- Environment-dependent test failures by properly mocking token retrieval
- Import errors in async test suite
- Exception chaining to follow Python best practices (ruff B904)
- Critical bug in sample_rows where API returns splits as dict, not list
- Updated all test mocks to match actual API response structure

### Changed
- Refactored magic number (100) to MAX_ROWS_PER_REQUEST constant
- Improved error messages with clearer formatting

### Development
- UV for modern Python packaging
- Ruff for linting and formatting
- mypy for type checking
- pytest with asyncio support
- GitHub Actions ready configuration

[Unreleased]: https://github.com/davanstrien/datasets-server-py/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/davanstrien/datasets-server-py/releases/tag/v0.1.0