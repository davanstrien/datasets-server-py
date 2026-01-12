# Copyright 2024 Daniel van Strien
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLI for exploring HuggingFace datasets via the Datasets Viewer API.

This CLI is designed for dataset exploration, not bulk data access.
For bulk operations, use the `parquet` command to get URLs and process with DuckDB/pandas.
"""

import json
import sys
from typing import Annotated, Optional

import typer

from datasets_server import __version__
from datasets_server.cli._cli_utils import (
    ConfigOpt,
    DatasetArg,
    FormatOpt,
    LimitOpt,
    OffsetOpt,
    OutputFormat,
    SplitOpt,
    TokenOpt,
    typer_factory,
)
from datasets_server.client import DatasetsServerClient
from datasets_server.exceptions import DatasetNotFoundError, DatasetServerError

app = typer_factory(
    help="Explore HuggingFace datasets via the Datasets Viewer API.",
)


def _get_client(token: Optional[str] = None) -> DatasetsServerClient:
    """Create a client instance."""
    return DatasetsServerClient(token=token)


def _output_json(data: object) -> None:
    """Output data as JSON."""
    print(json.dumps(data, indent=2, default=str))


def _output_table(headers: list[str], rows: list[list[object]]) -> None:
    """Output data as a simple table."""
    if not rows:
        print("No data")
        return

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def _handle_error(e: Exception) -> None:
    """Handle errors and exit with appropriate code."""
    if isinstance(e, DatasetNotFoundError):
        error_data = {"error": "DatasetNotFoundError", "message": str(e)}
    elif isinstance(e, DatasetServerError):
        error_data = {"error": type(e).__name__, "message": str(e)}
    else:
        error_data = {"error": "Error", "message": str(e)}

    print(json.dumps(error_data), file=sys.stderr)
    raise typer.Exit(code=1)


@app.command("is-valid")
def is_valid(
    dataset: DatasetArg,
    format: FormatOpt = OutputFormat.json,
    token: TokenOpt = None,
) -> None:
    """Check if a dataset is valid and which features are available."""
    try:
        client = _get_client(token)
        validity = client.is_valid(dataset)

        if format == OutputFormat.json:
            _output_json(validity.model_dump())
        else:
            headers = ["Feature", "Available"]
            rows = [
                ["viewer", validity.viewer],
                ["preview", validity.preview],
                ["search", validity.search],
                ["filter", validity.filter],
                ["statistics", validity.statistics],
            ]
            _output_table(headers, rows)
    except Exception as e:
        _handle_error(e)


@app.command("info")
def info(
    dataset: DatasetArg,
    config: ConfigOpt = None,
    format: FormatOpt = OutputFormat.json,
    token: TokenOpt = None,
) -> None:
    """Get dataset information including features, description, and citation."""
    try:
        client = _get_client(token)
        dataset_info = client.get_info(dataset, config=config)

        if format == OutputFormat.json:
            _output_json(dataset_info.model_dump())
        else:
            # For table format, show a summary
            info_dict = dataset_info.model_dump()
            print(f"Dataset: {dataset}")
            if config:
                print(f"Config: {config}")
            print(f"Partial: {dataset_info.partial}")
            if dataset_info.pending:
                print(f"Pending: {len(dataset_info.pending)} configs")
            if dataset_info.failed:
                print(f"Failed: {len(dataset_info.failed)} configs")
            print("\nDataset Info:")
            print(json.dumps(info_dict.get("dataset_info", {}), indent=2, default=str))
    except Exception as e:
        _handle_error(e)


@app.command("splits")
def splits(
    dataset: DatasetArg,
    format: FormatOpt = OutputFormat.json,
    token: TokenOpt = None,
) -> None:
    """List all available splits for a dataset."""
    try:
        client = _get_client(token)
        split_list = client.list_splits(dataset)

        if format == OutputFormat.json:
            _output_json([s.model_dump() for s in split_list])
        else:
            headers = ["Config", "Split"]
            rows = [[s.config, s.split] for s in split_list]
            _output_table(headers, rows)
    except Exception as e:
        _handle_error(e)


@app.command("size")
def size(
    dataset: DatasetArg,
    format: FormatOpt = OutputFormat.json,
    token: TokenOpt = None,
) -> None:
    """Get dataset size information."""
    try:
        client = _get_client(token)
        size_info = client.get_size(dataset)

        if format == OutputFormat.json:
            _output_json(size_info.model_dump())
        else:
            print(f"Dataset: {dataset}")
            print(f"Partial: {size_info.partial}")
            if size_info.size:
                print("\nSize Information:")
                print(json.dumps(size_info.size, indent=2, default=str))
    except Exception as e:
        _handle_error(e)


@app.command("parquet")
def parquet(
    dataset: DatasetArg,
    format: FormatOpt = OutputFormat.json,
    token: TokenOpt = None,
) -> None:
    """List parquet file URLs for a dataset.

    Use these URLs with DuckDB or pandas for bulk data processing.
    """
    try:
        client = _get_client(token)
        parquet_files = client.list_parquet_files(dataset)

        if format == OutputFormat.json:
            _output_json([f.model_dump() for f in parquet_files])
        else:
            headers = ["Config", "Split", "Filename", "Size"]
            rows = [[f.config, f.split, f.filename, f.size] for f in parquet_files]
            _output_table(headers, rows)
    except Exception as e:
        _handle_error(e)


@app.command("preview")
def preview(
    dataset: DatasetArg,
    config: ConfigOpt = None,
    split: SplitOpt = None,
    format: FormatOpt = OutputFormat.json,
    token: TokenOpt = None,
) -> None:
    """Get the first rows of a dataset (up to 100 rows).

    This is useful for quickly inspecting dataset structure and content.
    """
    try:
        client = _get_client(token)
        rows_data = client.preview(dataset, config=config, split=split)

        if format == OutputFormat.json:
            _output_json(rows_data.model_dump())
        else:
            print(f"Dataset: {dataset}")
            print(f"Total rows: {rows_data.num_rows_total}")
            print(f"Rows returned: {len(rows_data.rows)}")
            print(f"Partial: {rows_data.partial}")
            print("\nFeatures:")
            for feature in rows_data.features:
                print(f"  - {feature.get('name', 'unknown')}: {feature.get('type', 'unknown')}")
            print("\nFirst rows:")
            print(json.dumps(rows_data.rows[:5], indent=2, default=str))
            if len(rows_data.rows) > 5:
                print(f"  ... and {len(rows_data.rows) - 5} more rows")
    except Exception as e:
        _handle_error(e)


@app.command("sample")
def sample(
    dataset: DatasetArg,
    n: Annotated[
        int,
        typer.Option("--n", "-n", help="Number of rows to sample (max 100)."),
    ] = 10,
    config: ConfigOpt = None,
    split: SplitOpt = None,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="Random seed for reproducible sampling."),
    ] = None,
    format: FormatOpt = OutputFormat.json,
    token: TokenOpt = None,
) -> None:
    """Get a random sample of rows from a dataset.

    Maximum 100 rows per request to avoid API overload.
    """
    # Enforce max limit
    n = min(n, 100)

    try:
        client = _get_client(token)
        rows_data = client.sample_rows(dataset, config=config, split=split, n_samples=n, seed=seed)

        if format == OutputFormat.json:
            _output_json(rows_data.model_dump())
        else:
            print(f"Dataset: {dataset}")
            print(f"Sampled rows: {len(rows_data.rows)}")
            if seed is not None:
                print(f"Seed: {seed}")
            print("\nSampled data:")
            print(json.dumps(rows_data.rows, indent=2, default=str))
    except Exception as e:
        _handle_error(e)


@app.command("search")
def search(
    dataset: DatasetArg,
    query: Annotated[
        str,
        typer.Argument(help="Search query string."),
    ],
    config: ConfigOpt = None,
    split: SplitOpt = None,
    limit: LimitOpt = 10,
    offset: OffsetOpt = 0,
    format: FormatOpt = OutputFormat.json,
    token: TokenOpt = None,
) -> None:
    """Search for rows matching a query.

    This is for finding specific examples, not bulk extraction.
    """
    # Enforce max limit
    limit = min(limit, 100)

    try:
        client = _get_client(token)
        results = client.search(dataset, query=query, config=config, split=split, offset=offset, length=limit)

        if format == OutputFormat.json:
            _output_json(results.model_dump())
        else:
            print(f"Dataset: {dataset}")
            print(f"Query: {query}")
            print(f"Results: {len(results.rows)} (total: {results.num_rows_total})")
            print("\nMatching rows:")
            print(json.dumps(results.rows, indent=2, default=str))
    except Exception as e:
        _handle_error(e)


@app.command("stats")
def stats(
    dataset: DatasetArg,
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="The dataset configuration name (required)."),
    ],
    split: Annotated[
        str,
        typer.Option("--split", "-s", help="The dataset split (required)."),
    ],
    format: FormatOpt = OutputFormat.json,
    token: TokenOpt = None,
) -> None:
    """Get statistical information about dataset columns."""
    try:
        client = _get_client(token)
        statistics = client.get_statistics(dataset, config=config, split=split)

        if format == OutputFormat.json:
            _output_json(statistics.model_dump())
        else:
            print(f"Dataset: {dataset}")
            print(f"Config: {config}")
            print(f"Split: {split}")
            print(f"Number of examples: {statistics.num_examples}")
            print(f"Partial: {statistics.partial}")
            print("\nStatistics:")
            print(json.dumps(statistics.statistics, indent=2, default=str))
    except Exception as e:
        _handle_error(e)


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-v", help="Show version and exit."),
    ] = None,
) -> None:
    """Explore HuggingFace datasets via the Datasets Viewer API."""
    if version:
        print(f"datasets-server-py {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
