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
"""Tests for the CLI module."""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from datasets_server import __version__
from datasets_server.cli.main import app
from datasets_server.exceptions import DatasetNotFoundError
from datasets_server.models import (
    DatasetInfo,
    DatasetRows,
    DatasetSize,
    DatasetSplit,
    DatasetStatistics,
    DatasetValidity,
    ParquetFile,
)

runner = CliRunner()


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_help(self) -> None:
        """Test that --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Explore HuggingFace datasets" in result.stdout

    def test_version(self) -> None:
        """Test that --version works."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_no_args_shows_help(self) -> None:
        """Test that running without args shows help."""
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Commands:" in result.stdout


class TestIsValidCommand:
    """Test the is-valid command."""

    @patch("datasets_server.cli.main._get_client")
    def test_is_valid_json(self, mock_get_client: MagicMock) -> None:
        """Test is-valid command with JSON output."""
        mock_client = MagicMock()
        mock_client.is_valid.return_value = DatasetValidity(
            viewer=True, preview=True, search=True, filter=True, statistics=True
        )
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["is-valid", "test/dataset"])
        assert result.exit_code == 0

        output = json.loads(result.stdout)
        assert output["viewer"] is True
        assert output["preview"] is True
        assert output["search"] is True
        assert output["filter"] is True
        assert output["statistics"] is True

    @patch("datasets_server.cli.main._get_client")
    def test_is_valid_table(self, mock_get_client: MagicMock) -> None:
        """Test is-valid command with table output."""
        mock_client = MagicMock()
        mock_client.is_valid.return_value = DatasetValidity(
            viewer=True, preview=False, search=True, filter=False, statistics=True
        )
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["is-valid", "test/dataset", "--format", "table"])
        assert result.exit_code == 0
        assert "Feature" in result.stdout
        assert "Available" in result.stdout

    @patch("datasets_server.cli.main._get_client")
    def test_is_valid_not_found(self, mock_get_client: MagicMock) -> None:
        """Test is-valid command with dataset not found."""
        mock_client = MagicMock()
        mock_client.is_valid.side_effect = DatasetNotFoundError("Dataset not found")
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["is-valid", "nonexistent/dataset"])
        assert result.exit_code == 1


class TestSplitsCommand:
    """Test the splits command."""

    @patch("datasets_server.cli.main._get_client")
    def test_splits_json(self, mock_get_client: MagicMock) -> None:
        """Test splits command with JSON output."""
        mock_client = MagicMock()
        mock_client.list_splits.return_value = [
            DatasetSplit(dataset="test/dataset", config="default", split="train"),
            DatasetSplit(dataset="test/dataset", config="default", split="test"),
        ]
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["splits", "test/dataset"])
        assert result.exit_code == 0

        output = json.loads(result.stdout)
        assert len(output) == 2
        assert output[0]["split"] == "train"
        assert output[1]["split"] == "test"

    @patch("datasets_server.cli.main._get_client")
    def test_splits_table(self, mock_get_client: MagicMock) -> None:
        """Test splits command with table output."""
        mock_client = MagicMock()
        mock_client.list_splits.return_value = [
            DatasetSplit(dataset="test/dataset", config="default", split="train"),
        ]
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["splits", "test/dataset", "--format", "table"])
        assert result.exit_code == 0
        assert "Config" in result.stdout
        assert "Split" in result.stdout


class TestInfoCommand:
    """Test the info command."""

    @patch("datasets_server.cli.main._get_client")
    def test_info_json(self, mock_get_client: MagicMock) -> None:
        """Test info command with JSON output."""
        mock_client = MagicMock()
        mock_client.get_info.return_value = DatasetInfo(
            dataset_info={"default": {"description": "Test dataset"}},
            pending=[],
            failed=[],
            partial=False,
        )
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["info", "test/dataset"])
        assert result.exit_code == 0

        output = json.loads(result.stdout)
        assert "dataset_info" in output
        assert output["partial"] is False


class TestSizeCommand:
    """Test the size command."""

    @patch("datasets_server.cli.main._get_client")
    def test_size_json(self, mock_get_client: MagicMock) -> None:
        """Test size command with JSON output."""
        mock_client = MagicMock()
        mock_client.get_size.return_value = DatasetSize(
            size={"dataset": {"num_bytes_original_files": 1000, "num_bytes_parquet_files": 800, "num_rows": 100}},
            pending=[],
            failed=[],
            partial=False,
        )
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["size", "test/dataset"])
        assert result.exit_code == 0

        output = json.loads(result.stdout)
        assert "size" in output


class TestParquetCommand:
    """Test the parquet command."""

    @patch("datasets_server.cli.main._get_client")
    def test_parquet_json(self, mock_get_client: MagicMock) -> None:
        """Test parquet command with JSON output."""
        mock_client = MagicMock()
        mock_client.list_parquet_files.return_value = [
            ParquetFile(
                dataset="test/dataset",
                config="default",
                split="train",
                url="https://example.com/file.parquet",
                filename="file.parquet",
                size=1000,
            ),
        ]
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["parquet", "test/dataset"])
        assert result.exit_code == 0

        output = json.loads(result.stdout)
        assert len(output) == 1
        assert output[0]["filename"] == "file.parquet"


class TestPreviewCommand:
    """Test the preview command."""

    @patch("datasets_server.cli.main._get_client")
    def test_preview_json(self, mock_get_client: MagicMock) -> None:
        """Test preview command with JSON output."""
        mock_client = MagicMock()
        mock_client.preview.return_value = DatasetRows(
            features=[{"name": "text", "type": {"dtype": "string", "_type": "Value"}}],
            rows=[{"row_idx": 0, "row": {"text": "Hello"}, "truncated_cells": []}],
            num_rows_total=100,
            num_rows_per_page=100,
            partial=False,
        )
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["preview", "test/dataset"])
        assert result.exit_code == 0

        output = json.loads(result.stdout)
        assert "features" in output
        assert "rows" in output
        assert len(output["rows"]) == 1


class TestSampleCommand:
    """Test the sample command."""

    @patch("datasets_server.cli.main._get_client")
    def test_sample_json(self, mock_get_client: MagicMock) -> None:
        """Test sample command with JSON output."""
        mock_client = MagicMock()
        mock_client.sample_rows.return_value = DatasetRows(
            features=[{"name": "text", "type": {"dtype": "string", "_type": "Value"}}],
            rows=[{"row_idx": 42, "row": {"text": "Sampled"}, "truncated_cells": []}],
            num_rows_total=100,
            num_rows_per_page=1,
            partial=False,
        )
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["sample", "test/dataset", "-n", "5", "--seed", "42"])
        assert result.exit_code == 0

        output = json.loads(result.stdout)
        assert "rows" in output

    @patch("datasets_server.cli.main._get_client")
    def test_sample_enforces_max_limit(self, mock_get_client: MagicMock) -> None:
        """Test that sample enforces max 100 rows."""
        mock_client = MagicMock()
        mock_client.sample_rows.return_value = DatasetRows(
            features=[],
            rows=[],
            num_rows_total=1000,
            num_rows_per_page=100,
            partial=False,
        )
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["sample", "test/dataset", "-n", "500"])
        assert result.exit_code == 0

        # Verify that n_samples was capped to 100
        mock_client.sample_rows.assert_called_once()
        call_args = mock_client.sample_rows.call_args
        assert call_args.kwargs.get("n_samples") == 100


class TestSearchCommand:
    """Test the search command."""

    @patch("datasets_server.cli.main._get_client")
    def test_search_json(self, mock_get_client: MagicMock) -> None:
        """Test search command with JSON output."""
        mock_client = MagicMock()
        mock_client.search.return_value = DatasetRows(
            features=[{"name": "text", "type": {"dtype": "string", "_type": "Value"}}],
            rows=[{"row_idx": 10, "row": {"text": "Found it"}, "truncated_cells": []}],
            num_rows_total=1,
            num_rows_per_page=10,
            partial=False,
        )
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["search", "test/dataset", "query text"])
        assert result.exit_code == 0

        output = json.loads(result.stdout)
        assert "rows" in output


class TestStatsCommand:
    """Test the stats command."""

    @patch("datasets_server.cli.main._get_client")
    def test_stats_json(self, mock_get_client: MagicMock) -> None:
        """Test stats command with JSON output."""
        mock_client = MagicMock()
        mock_client.get_statistics.return_value = DatasetStatistics(
            num_examples=1000,
            statistics=[{"column_name": "text", "column_type": "string", "column_statistics": {}}],
            partial=False,
        )
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["stats", "test/dataset", "--config", "default", "--split", "train"])
        assert result.exit_code == 0

        output = json.loads(result.stdout)
        assert output["num_examples"] == 1000


class TestTokenOption:
    """Test that token option is passed correctly."""

    @patch("datasets_server.cli.main._get_client")
    def test_token_passed_to_client(self, mock_get_client: MagicMock) -> None:
        """Test that token is passed to the client."""
        mock_client = MagicMock()
        mock_client.is_valid.return_value = DatasetValidity(
            viewer=True, preview=True, search=True, filter=True, statistics=True
        )
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["is-valid", "test/dataset", "--token", "hf_test_token"])
        assert result.exit_code == 0

        mock_get_client.assert_called_once_with("hf_test_token")


class TestErrorHandling:
    """Test error handling."""

    @patch("datasets_server.cli.main._get_client")
    def test_error_exits_with_code_1(self, mock_get_client: MagicMock) -> None:
        """Test that errors cause exit code 1."""
        mock_client = MagicMock()
        mock_client.is_valid.side_effect = DatasetNotFoundError("Dataset not found: test")
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["is-valid", "test/dataset"])
        assert result.exit_code == 1


# Integration-style tests (marked for optional running)
@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests that hit the real API."""

    def test_is_valid_real_dataset(self) -> None:
        """Test is-valid against a real dataset."""
        result = runner.invoke(app, ["is-valid", "rajpurkar/squad_v2"])
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["viewer"] is True

    def test_splits_real_dataset(self) -> None:
        """Test splits against a real dataset."""
        result = runner.invoke(app, ["splits", "rajpurkar/squad_v2"])
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert len(output) > 0
