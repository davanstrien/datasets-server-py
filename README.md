# Datasets Server Python Client

A Python client library for the Hugging Face Datasets Viewer API with both synchronous and asynchronous support.

## Features

- 🔄 **Sync/Async Support**: Choose between synchronous and asynchronous clients based on your needs
- 🔍 **Full API Coverage**: Access all Datasets Viewer API endpoints with a Pythonic interface
- 🎯 **Type Safety**: Pydantic models for all API responses ensure type safety and validation
- 🚀 **High Performance**: Async support enables efficient concurrent operations
- 🔐 **Authentication**: Seamless integration with Hugging Face authentication tokens
- 📊 **Rich Data Access**: Preview datasets, search content, filter rows, and analyze statistics without downloading

## Installation

Install directly from GitHub:

```bash
# Using pip
pip install git+https://github.com/davanstrien/datasets-server-py.git

# Using UV (recommended)
uv pip install git+https://github.com/davanstrien/datasets-server-py.git
```

For development:

```bash
git clone https://github.com/davanstrien/datasets-server-py
cd datasets-server-py
uv pip install -e ".[dev]"
```

## Quick Start

### Synchronous Usage

```python
from datasets_server import DatasetsServerClient

# Initialize client (uses HF token from environment if available)
client = DatasetsServerClient()

# Check dataset validity
validity = client.is_valid("stanfordnlp/imdb")
if validity.preview:
    # Preview first rows
    rows = client.preview("stanfordnlp/imdb")
    print(f"Dataset has {len(rows.rows)} preview rows")
    
# Search within a dataset
if validity.search:
    results = client.search(
        dataset="stanfordnlp/imdb",
        query="amazing movie",
        config="plain_text",
        split="train",
        length=5
    )
    print(f"Found {results.num_rows_total} matches")
```

### Asynchronous Usage

```python
import asyncio
from datasets_server import AsyncDatasetsServerClient

async def explore_datasets():
    async with AsyncDatasetsServerClient() as client:
        # Check multiple datasets concurrently
        datasets = ["SetFit/ag_news", "stanfordnlp/imdb", "davanstrien/haiku_dpo"]
        tasks = [client.is_valid(ds) for ds in datasets]
        validities = await asyncio.gather(*tasks)
        
        for dataset, validity in zip(datasets, validities):
            print(f"{dataset}: preview={validity.preview}, search={validity.search}")

asyncio.run(explore_datasets())
```

## API Reference

### Client Initialization

Both clients accept the same parameters:

```python
client = DatasetsServerClient(
    token="your-hf-token",  # Optional: defaults to cached token
    endpoint="https://custom-endpoint",  # Optional: custom API endpoint
    timeout=30.0  # Optional: request timeout in seconds
)
```

### Available Methods

All methods are available in both sync and async versions:

#### Dataset Validation
- `is_valid(dataset)` - Check if a dataset is valid and which features are available

#### Dataset Information
- `list_splits(dataset)` - List all configurations and splits
- `get_info(dataset, config=None)` - Get detailed dataset information
- `get_size(dataset)` - Get dataset size information
- `list_parquet_files(dataset)` - List available Parquet files

#### Data Access
- `preview(dataset, config=None, split=None)` - Preview first 100 rows
- `get_rows(dataset, config, split, offset=0, length=100)` - Get rows with pagination
- `iter_rows(dataset, config, split, batch_size=100)` - Iterate through all rows
- `sample_rows(dataset, config, split, n_samples, seed=None, max_requests=None)` - Get random sample of rows

#### Search and Filter
- `search(dataset, query, config, split, offset=0, length=100)` - Search text in dataset
- `filter(dataset, where, config, split, orderby=None, offset=0, length=100)` - Filter with SQL-like conditions

#### Statistics
- `get_statistics(dataset, config, split)` - Get statistical information about dataset columns

## Examples

### Explore a Dataset

```python
from datasets_server import DatasetsServerClient

client = DatasetsServerClient()

# Get basic information
info = client.get_info("SetFit/ag_news")
print(f"Description: {info.dataset_info.get('description', 'N/A')}")

# List available splits
splits = client.list_splits("SetFit/ag_news")
for split in splits:
    print(f"Config: {split.config}, Split: {split.split}")

# Get dataset statistics
stats = client.get_statistics("SetFit/ag_news", config="default", split="train")
print(f"Number of examples: {stats.num_examples:,}")
```

### Filter Dataset Rows

```python
# Filter for positive reviews (label = 1)
filtered = client.filter(
    dataset="stanfordnlp/imdb",
    config="plain_text",
    split="train",
    where='"label" = 1',
    length=10
)

for row in filtered.rows:
    print(f"Label: {row['row']['label']}, Text preview: {row['row']['text'][:100]}...")
```

### Sample Random Rows

```python
# Get a random sample of rows
sample = client.sample_rows(
    dataset="stanfordnlp/imdb",
    config="plain_text",
    split="train",
    n_samples=10,
    seed=42  # For reproducibility
)

for row in sample.rows:
    text_preview = row["row"]["text"][:100] + "..."
    label = "positive" if row["row"]["label"] == 1 else "negative"
    print(f"{label}: {text_preview}")

# API-efficient sampling with max_requests
# Limits API calls for large datasets
efficient_sample = client.sample_rows(
    dataset="stanfordnlp/imdb",
    config="plain_text",
    split="train",
    n_samples=50,
    seed=42,
    max_requests=5  # Use at most 5 API calls
)
# Note: max_requests trades randomness for API efficiency
```

### Concurrent Operations with Async

```python
import asyncio
from datasets_server import AsyncDatasetsServerClient

async def analyze_datasets(dataset_list):
    async with AsyncDatasetsServerClient() as client:
        # Get info for all datasets concurrently
        tasks = [client.get_info(ds) for ds in dataset_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for dataset, result in zip(dataset_list, results):
            if isinstance(result, Exception):
                print(f"{dataset}: Error - {result}")
            else:
                print(f"{dataset}: {result.dataset_info.get('num_rows', 'Unknown')} rows")

asyncio.run(analyze_datasets(["SetFit/ag_news", "stanfordnlp/imdb", "librarian-bots/dataset_cards_with_metadata"]))
```

## Error Handling

The client includes custom exceptions for better error handling:

```python
from datasets_server import (
    DatasetsServerClient,
    DatasetNotFoundError,
    DatasetServerError
)

client = DatasetsServerClient()

try:
    validity = client.is_valid("non-existent-dataset")
except DatasetNotFoundError as e:
    print(f"Dataset not found: {e}")
except DatasetServerError as e:
    print(f"API error: {e}")
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/davanstrien/datasets-server-py
cd datasets-server-py

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,examples]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=datasets_server

# Run only async tests
pytest tests/test_async_client.py
```

### Code Quality

```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Built to work seamlessly with the [Hugging Face Hub](https://huggingface.co/)
- Inspired by the design patterns of [huggingface_hub](https://github.com/huggingface/huggingface_hub)

## Links

- [Hugging Face Datasets Viewer Documentation](https://huggingface.co/docs/dataset-viewer)
- [Issue Tracker](https://github.com/davanstrien/datasets-server-py/issues)