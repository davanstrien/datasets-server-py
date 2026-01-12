"""Tests for synchronous client."""

from unittest.mock import patch

import httpx
import pytest
import respx

from datasets_server import (
    DatasetNotFoundError,
    DatasetServerError,
    DatasetServerHTTPError,
    DatasetsServerClient,
)
from datasets_server.models import DatasetValidity


class TestDatasetsServerClient:
    """Test synchronous client."""

    def test_client_initialization(self):
        """Test client initialization with different parameters."""
        # Default initialization
        client = DatasetsServerClient()
        assert client.endpoint == "https://datasets-server.huggingface.co"
        assert client.timeout == 30.0
        assert client._token is None

        # Custom initialization
        client = DatasetsServerClient(
            token="test-token",
            endpoint="https://custom.endpoint",
            timeout=60.0,
        )
        assert client.endpoint == "https://custom.endpoint"
        assert client.timeout == 60.0
        assert client._token == "test-token"

    @patch("datasets_server._base.get_token", return_value=None)
    def test_headers_without_token(self, mock_get_token):
        """Test headers when no token is provided."""
        client = DatasetsServerClient()
        headers = client.headers
        assert "user-agent" in headers
        assert "authorization" not in headers

    def test_headers_with_token(self):
        """Test headers when token is provided."""
        client = DatasetsServerClient(token="test-token")
        headers = client.headers
        assert headers["authorization"] == "Bearer test-token"

    @respx.mock
    def test_is_valid_success(self):
        """Test successful is_valid call."""
        respx.get("https://datasets-server.huggingface.co/is-valid").mock(
            return_value=httpx.Response(
                200,
                json={
                    "viewer": True,
                    "preview": True,
                    "search": False,
                    "filter": False,
                    "statistics": True,
                },
            )
        )

        client = DatasetsServerClient()
        validity = client.is_valid("test-dataset")

        assert isinstance(validity, DatasetValidity)
        assert validity.viewer is True
        assert validity.preview is True
        assert validity.search is False

    @respx.mock
    def test_dataset_not_found(self):
        """Test handling of 404 errors."""
        respx.get("https://datasets-server.huggingface.co/is-valid").mock(
            return_value=httpx.Response(404, json={"error": "Not found"})
        )

        client = DatasetsServerClient()
        with pytest.raises(DatasetNotFoundError) as exc_info:
            client.is_valid("non-existent-dataset")

        assert "Dataset not found: non-existent-dataset" in str(exc_info.value)

    @respx.mock
    def test_api_error(self):
        """Test handling of other HTTP errors raises DatasetServerHTTPError with context."""
        # Use 400 Bad Request (not in RETRY_STATUS_CODES) to avoid retry delays
        respx.get("https://datasets-server.huggingface.co/is-valid").mock(
            return_value=httpx.Response(
                400,
                json={"error": "Bad Request"},
                headers={"x-request-id": "test-request-123"},
            )
        )

        client = DatasetsServerClient()
        with pytest.raises(DatasetServerHTTPError) as exc_info:
            client.is_valid("test-dataset")

        # Verify response context is captured
        error = exc_info.value
        assert error.status_code == 400
        assert error.request_id == "test-request-123"
        assert error.server_message == "Bad Request"
        assert error.response is not None
        assert "API error:" in str(error)

    @respx.mock
    def test_generic_exception(self):
        """Test handling of generic exceptions during requests."""
        respx.get("https://datasets-server.huggingface.co/is-valid").mock(side_effect=Exception("Network error"))

        client = DatasetsServerClient()
        with pytest.raises(DatasetServerError) as exc_info:
            client.is_valid("test-dataset")

        assert "Request failed: Network error" in str(exc_info.value)

    @respx.mock
    def test_list_splits(self):
        """Test list_splits method."""
        respx.get("https://datasets-server.huggingface.co/splits").mock(
            return_value=httpx.Response(
                200,
                json={
                    "splits": [
                        {"dataset": "squad", "config": "plain_text", "split": "train"},
                        {"dataset": "squad", "config": "plain_text", "split": "validation"},
                    ],
                    "pending": [],
                    "failed": [],
                },
            )
        )

        client = DatasetsServerClient()
        splits = client.list_splits("squad")

        assert len(splits) == 2
        assert splits[0].dataset == "squad"
        assert splits[0].config == "plain_text"
        assert splits[0].split == "train"
        assert splits[1].split == "validation"

    def test_get_rows_with_invalid_length(self):
        """Test get_rows with length > 100."""
        client = DatasetsServerClient()

        with pytest.raises(ValueError) as exc_info:
            client.get_rows("dataset", "config", "split", length=150)

        assert "Length cannot exceed 100" in str(exc_info.value)

    def test_context_manager(self):
        """Test client as context manager creates dedicated session."""
        with DatasetsServerClient() as client:
            assert client._session is not None
            assert client._owns_session is True
        # Session should be closed and ownership released after exiting context
        assert client._session is None
        assert client._owns_session is False

    def test_global_session_used_by_default(self):
        """Test that global session is used when not using context manager."""
        client = DatasetsServerClient()
        # _session should be None (uses global session via property)
        assert client._session is None
        # But session property returns the global session
        session = client.session
        assert session is not None

    @patch("datasets_server._base.get_token")
    def test_token_exception_handling(self, mock_get_token):
        """Test handling of exception when getting token from huggingface_hub."""
        mock_get_token.side_effect = Exception("Token error")

        client = DatasetsServerClient()
        assert client.token is None

    @respx.mock
    def test_iter_rows(self):
        """Test row iteration."""
        route = respx.get("https://datasets-server.huggingface.co/rows")

        # Queue multiple responses for pagination
        route.side_effect = [
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(100)],
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(100, 150)],
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [],
                },
            ),
        ]

        client = DatasetsServerClient()
        rows = list(client.iter_rows("dataset", "config", "split"))

        assert len(rows) == 150
        assert rows[0]["text"] == "Row 0"
        assert rows[149]["text"] == "Row 149"

    @respx.mock
    def test_get_info_with_config(self):
        """Test get_info method with optional config parameter."""
        respx.get("https://datasets-server.huggingface.co/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "dataset_info": {"description": "Test dataset"},
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            )
        )

        client = DatasetsServerClient()
        info = client.get_info("dataset", config="my_config")

        assert info.dataset_info["description"] == "Test dataset"

    @respx.mock
    def test_filter_with_where_clause(self):
        """Test filter method with WHERE clause."""
        respx.get("https://datasets-server.huggingface.co/filter").mock(
            return_value=httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}, {"name": "score", "type": "int32"}],
                    "rows": [{"row": {"text": "Test", "score": 5}}],
                    "num_rows_total": 1,
                },
            )
        )

        client = DatasetsServerClient()
        results = client.filter(
            dataset="dataset",
            config="config",
            split="split",
            where="score > 3",
            orderby="score DESC",
            offset=10,
            length=50,
        )

        assert len(results.rows) == 1
        assert results.rows[0]["row"]["score"] == 5

    @respx.mock
    def test_iter_rows_empty_dataset(self):
        """Test iter_rows with empty dataset."""
        respx.get("https://datasets-server.huggingface.co/rows").mock(
            return_value=httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [],
                },
            )
        )

        client = DatasetsServerClient()
        rows = list(client.iter_rows("empty-dataset", "config", "split"))

        assert len(rows) == 0

    @respx.mock
    def test_sample_rows_basic(self):
        """Test basic sample_rows functionality."""
        # Mock info endpoint
        respx.get("https://datasets-server.huggingface.co/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "dataset_info": {
                        "splits": {
                            "train": {"name": "train", "num_examples": 1000},
                            "test": {"name": "test", "num_examples": 100},
                        }
                    },
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            )
        )

        # Mock rows endpoint with multiple responses
        # With seed=42 and 1000 rows, the indices are [25, 114, 281, 654, 759]
        rows_route = respx.get("https://datasets-server.huggingface.co/rows")
        rows_route.side_effect = [
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(25, 115)],
                    "num_rows_total": 1000,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 281"}}],
                    "num_rows_total": 1000,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 654"}}],
                    "num_rows_total": 1000,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 759"}}],
                    "num_rows_total": 1000,
                },
            ),
        ]

        client = DatasetsServerClient()
        result = client.sample_rows("dataset", "config", "train", n_samples=5, seed=42)

        assert len(result.rows) == 5
        assert result.num_rows_total == 1000
        assert result.features == [{"name": "text", "type": "string"}]
        expected_texts = ["Row 25", "Row 114", "Row 281", "Row 654", "Row 759"]
        actual_texts = [row["row"]["text"] for row in result.rows]
        assert actual_texts == expected_texts

    @respx.mock
    def test_sample_rows_with_seed(self):
        """Test that sampling with seed is deterministic."""
        # Mock info endpoint (called twice)
        info_route = respx.get("https://datasets-server.huggingface.co/info")
        info_route.side_effect = [
            httpx.Response(
                200,
                json={
                    "dataset_info": {"splits": {"train": {"name": "train", "num_examples": 100}}},
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            ),
            httpx.Response(
                200,
                json={
                    "dataset_info": {"splits": {"train": {"name": "train", "num_examples": 100}}},
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            ),
        ]

        # Mock rows endpoints
        rows_route = respx.get("https://datasets-server.huggingface.co/rows")
        rows_route.side_effect = [
            httpx.Response(
                200,
                json={
                    "features": [{"name": "id", "type": "int32"}],
                    "rows": [{"row": {"id": i}} for i in range(8, 100)],
                    "num_rows_total": 100,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "id", "type": "int32"}],
                    "rows": [{"row": {"id": i}} for i in range(8, 100)],
                    "num_rows_total": 100,
                },
            ),
        ]

        client = DatasetsServerClient()

        # First sampling with seed
        result1 = client.sample_rows("dataset", "config", "train", n_samples=10, seed=123)
        rows1 = [row["row"]["id"] for row in result1.rows]

        # Second sampling with same seed should give same results
        result2 = client.sample_rows("dataset", "config", "train", n_samples=10, seed=123)
        rows2 = [row["row"]["id"] for row in result2.rows]

        assert rows1 == rows2

    def test_sample_rows_zero_samples(self):
        """Test sample_rows with n_samples=0."""
        client = DatasetsServerClient()
        result = client.sample_rows("dataset", "config", "train", n_samples=0)

        assert len(result.rows) == 0
        assert result.num_rows_total == 0
        assert result.features == []

    def test_sample_rows_negative_samples(self):
        """Test sample_rows with negative n_samples."""
        client = DatasetsServerClient()

        with pytest.raises(ValueError) as exc_info:
            client.sample_rows("dataset", "config", "train", n_samples=-5)

        assert "n_samples must be non-negative" in str(exc_info.value)

    @respx.mock
    def test_sample_rows_exceeds_dataset_size(self):
        """Test sample_rows when requesting more samples than available."""
        respx.get("https://datasets-server.huggingface.co/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "dataset_info": {"splits": {"train": {"name": "train", "num_examples": 50}}},
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            )
        )

        client = DatasetsServerClient()

        with pytest.raises(ValueError) as exc_info:
            client.sample_rows("dataset", "config", "train", n_samples=100)

        assert "Requested 100 samples but dataset only has 50 rows" in str(exc_info.value)

    @respx.mock
    def test_sample_rows_fallback_to_get_rows(self):
        """Test sample_rows when info doesn't contain split information."""
        # Mock info endpoint without splits
        respx.get("https://datasets-server.huggingface.co/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "dataset_info": {"description": "Test dataset"},
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            )
        )

        # Mock rows endpoint with multiple responses
        rows_route = respx.get("https://datasets-server.huggingface.co/rows")
        rows_route.side_effect = [
            # First call to get total rows
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 0"}}],
                    "num_rows_total": 1000,
                },
            ),
            # Sampling calls
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(25, 115)],
                    "num_rows_total": 1000,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 281"}}],
                    "num_rows_total": 1000,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 654"}}],
                    "num_rows_total": 1000,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 759"}}],
                    "num_rows_total": 1000,
                },
            ),
        ]

        client = DatasetsServerClient()
        result = client.sample_rows("dataset", "config", "train", n_samples=5, seed=42)

        assert len(result.rows) == 5
        assert result.num_rows_total == 1000

    @respx.mock
    def test_sample_rows_with_max_requests(self):
        """Test sample_rows with max_requests parameter."""
        respx.get("https://datasets-server.huggingface.co/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "dataset_info": {"splits": {"train": {"name": "train", "num_examples": 10000}}},
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            )
        )

        # Mock rows endpoint
        rows_route = respx.get("https://datasets-server.huggingface.co/rows")
        rows_route.side_effect = [
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(100)],
                    "num_rows_total": 10000,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(5000, 5100)],
                    "num_rows_total": 10000,
                },
            ),
        ]

        client = DatasetsServerClient()
        result = client.sample_rows("dataset", "config", "train", n_samples=10, seed=42, max_requests=2)

        assert len(result.rows) == 10
        assert result.num_rows_total == 10000

    @respx.mock
    def test_sample_rows_max_requests_one(self):
        """Test sample_rows with max_requests=1 (single API call)."""
        respx.get("https://datasets-server.huggingface.co/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "dataset_info": {"splits": {"train": {"name": "train", "num_examples": 50000}}},
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            )
        )

        respx.get("https://datasets-server.huggingface.co/rows").mock(
            return_value=httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}, {"name": "label", "type": "int"}],
                    "rows": [{"row": {"text": f"Row {i}", "label": i % 2}} for i in range(15000, 15100)],
                    "num_rows_total": 50000,
                },
            )
        )

        client = DatasetsServerClient()
        result = client.sample_rows("dataset", "config", "train", n_samples=5, seed=123, max_requests=1)

        assert len(result.rows) == 5
        assert result.num_rows_total == 50000
        # All samples should come from the same batch
        row_nums = [int(row["row"]["text"].split()[1]) for row in result.rows]
        assert all(15000 <= num < 15100 for num in row_nums)

    @respx.mock
    def test_sample_rows_max_requests_exceeds_samples(self):
        """Test sample_rows when max_requests > n_samples."""
        respx.get("https://datasets-server.huggingface.co/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "dataset_info": {"splits": {"train": {"name": "train", "num_examples": 1000}}},
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            )
        )

        # Mock rows endpoint for multiple segments
        rows_route = respx.get("https://datasets-server.huggingface.co/rows")
        rows_route.side_effect = [
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(100)],
                    "num_rows_total": 1000,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(200, 300)],
                    "num_rows_total": 1000,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(400, 500)],
                    "num_rows_total": 1000,
                },
            ),
        ]

        client = DatasetsServerClient()
        result = client.sample_rows("dataset", "config", "train", n_samples=3, seed=42, max_requests=5)

        assert len(result.rows) == 3
        assert result.num_rows_total == 1000
