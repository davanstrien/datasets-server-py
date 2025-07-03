"""Tests for synchronous client."""

from unittest.mock import Mock, patch

import pytest
import requests

from datasets_server import DatasetNotFoundError, DatasetServerError, DatasetsServerClient
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
        assert "User-Agent" in headers
        assert "Authorization" not in headers

    def test_headers_with_token(self):
        """Test headers when token is provided."""
        client = DatasetsServerClient(token="test-token")
        headers = client.headers
        assert headers["Authorization"] == "Bearer test-token"

    @patch("datasets_server.client.requests.Session")
    def test_is_valid_success(self, mock_session_class):
        """Test successful is_valid call."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "viewer": True,
            "preview": True,
            "search": False,
            "filter": False,
            "statistics": True,
        }

        # Mock session
        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Test
        client = DatasetsServerClient()
        validity = client.is_valid("test-dataset")

        # Assertions
        assert isinstance(validity, DatasetValidity)
        assert validity.viewer is True
        assert validity.preview is True
        assert validity.search is False
        mock_session.request.assert_called_once_with(
            method="GET",
            url="https://datasets-server.huggingface.co/is-valid",
            params={"dataset": "test-dataset"},
            headers=client.headers,
            timeout=30.0,
        )

    @patch("datasets_server.client.requests.Session")
    def test_dataset_not_found(self, mock_session_class):
        """Test handling of 404 errors."""
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)

        # Mock session
        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Test
        client = DatasetsServerClient()
        with pytest.raises(DatasetNotFoundError) as exc_info:
            client.is_valid("non-existent-dataset")

        assert "Dataset not found: non-existent-dataset" in str(exc_info.value)

    @patch("datasets_server.client.requests.Session")
    def test_api_error(self, mock_session_class):
        """Test handling of other HTTP errors."""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response, request=Mock()
        )

        # Mock session
        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Test
        client = DatasetsServerClient()
        with pytest.raises(DatasetServerError) as exc_info:
            client.is_valid("test-dataset")

        assert "API error:" in str(exc_info.value)

    @patch("datasets_server.client.requests.Session")
    def test_generic_exception(self, mock_session_class):
        """Test handling of generic exceptions during requests."""
        # Mock session to raise a generic exception
        mock_session = Mock()
        mock_session.request.side_effect = Exception("Network error")
        mock_session_class.return_value = mock_session

        client = DatasetsServerClient()
        with pytest.raises(DatasetServerError) as exc_info:
            client.is_valid("test-dataset")

        assert "Request failed: Network error" in str(exc_info.value)

    @patch("datasets_server.client.requests.Session")
    def test_list_splits(self, mock_session_class):
        """Test list_splits method."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "splits": [
                {"dataset": "squad", "config": "plain_text", "split": "train"},
                {"dataset": "squad", "config": "plain_text", "split": "validation"},
            ],
            "pending": [],
            "failed": [],
        }

        # Mock session
        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Test
        client = DatasetsServerClient()
        splits = client.list_splits("squad")

        # Assertions
        assert len(splits) == 2
        assert splits[0].dataset == "squad"
        assert splits[0].config == "plain_text"
        assert splits[0].split == "train"
        assert splits[1].split == "validation"

    @patch("datasets_server.client.requests.Session")
    def test_get_rows_with_invalid_length(self, mock_session_class):
        """Test get_rows with length > 100."""
        client = DatasetsServerClient()

        with pytest.raises(ValueError) as exc_info:
            client.get_rows("dataset", "config", "split", length=150)

        assert "Length cannot exceed 100" in str(exc_info.value)  # The MAX_ROWS_PER_REQUEST constant is 100

    def test_context_manager(self):
        """Test client as context manager."""
        with DatasetsServerClient() as client:
            assert client._session is not None
        # Session should be closed after exiting context

    @patch("datasets_server._base.get_token")
    def test_token_exception_handling(self, mock_get_token):
        """Test handling of exception when getting token from huggingface_hub."""
        # Simulate get_token raising an exception
        mock_get_token.side_effect = Exception("Token error")

        client = DatasetsServerClient()
        # Should not raise, but return None
        assert client.token is None

    @patch("datasets_server.client.requests.Session")
    def test_iter_rows(self, mock_session_class):
        """Test row iteration."""
        # Mock responses for pagination
        responses = [
            {
                "features": [{"name": "text", "type": "string"}],
                "rows": [{"row": {"text": f"Row {i}"}} for i in range(100)],
            },
            {
                "features": [{"name": "text", "type": "string"}],
                "rows": [{"row": {"text": f"Row {i}"}} for i in range(100, 150)],
            },
            {
                "features": [{"name": "text", "type": "string"}],
                "rows": [],  # Empty response to stop iteration
            },
        ]

        # Mock session with multiple responses
        mock_session = Mock()
        mock_responses = []
        for resp_data in responses:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = resp_data
            mock_responses.append(mock_resp)

        mock_session.request.side_effect = mock_responses
        mock_session_class.return_value = mock_session

        # Test iteration
        client = DatasetsServerClient()
        rows = list(client.iter_rows("dataset", "config", "split"))

        assert len(rows) == 150
        assert rows[0]["text"] == "Row 0"
        assert rows[149]["text"] == "Row 149"

    @patch("datasets_server.client.requests.Session")
    def test_get_info_with_config(self, mock_session_class):
        """Test get_info method with optional config parameter."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "dataset_info": {"description": "Test dataset"},
            "pending": [],
            "failed": [],
            "partial": False,
        }

        # Mock session
        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Test with config
        client = DatasetsServerClient()
        info = client.get_info("dataset", config="my_config")

        # Verify the request was made with config parameter
        mock_session.request.assert_called_once_with(
            method="GET",
            url="https://datasets-server.huggingface.co/info",
            params={"dataset": "dataset", "config": "my_config"},
            headers=client.headers,
            timeout=30.0,
        )
        assert info.dataset_info["description"] == "Test dataset"

    @patch("datasets_server.client.requests.Session")
    def test_filter_with_where_clause(self, mock_session_class):
        """Test filter method with WHERE clause."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "features": [{"name": "text", "type": "string"}, {"name": "score", "type": "int32"}],
            "rows": [{"row": {"text": "Test", "score": 5}}],
            "num_rows_total": 1,
        }

        # Mock session
        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Test filter with WHERE clause
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

        # Verify the request parameters
        mock_session.request.assert_called_once_with(
            method="GET",
            url="https://datasets-server.huggingface.co/filter",
            params={
                "dataset": "dataset",
                "config": "config",
                "split": "split",
                "where": "score > 3",
                "orderby": "score DESC",
                "offset": 10,
                "length": 50,
            },
            headers=client.headers,
            timeout=30.0,
        )
        assert len(results.rows) == 1
        assert results.rows[0]["row"]["score"] == 5

    @patch("datasets_server.client.requests.Session")
    def test_iter_rows_empty_dataset(self, mock_session_class):
        """Test iter_rows with empty dataset."""
        # Mock empty response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "features": [{"name": "text", "type": "string"}],
            "rows": [],  # Empty rows
        }

        # Mock session
        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Test iteration on empty dataset
        client = DatasetsServerClient()
        rows = list(client.iter_rows("empty-dataset", "config", "split"))

        assert len(rows) == 0

    @patch("datasets_server.client.requests.Session")
    def test_sample_rows_basic(self, mock_session_class):
        """Test basic sample_rows functionality."""
        # Mock info response
        mock_info_response = Mock()
        mock_info_response.status_code = 200
        mock_info_response.json.return_value = {
            "dataset_info": {
                "splits": {
                    "train": {"name": "train", "num_examples": 1000},
                    "test": {"name": "test", "num_examples": 100}
                }
            },
            "pending": [],
            "failed": [],
            "partial": False,
        }

        # Create mock rows response factory
        def create_rows_response(offset, length):
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "features": [{"name": "text", "type": "string"}],
                "rows": [{"row": {"text": f"Row {i}"}} for i in range(offset, min(offset + length, 1000))],
                "num_rows_total": 1000,
            }
            return response

        # Mock session
        mock_session = Mock()
        # With seed=42 and 1000 rows, the indices are [25, 114, 281, 654, 759]
        # The algorithm batches indices that are within MAX_ROWS_PER_REQUEST of each other
        mock_session.request.side_effect = [
            mock_info_response,
            create_rows_response(25, 90),   # Rows 25-114 (includes both 25 and 114)
            create_rows_response(281, 1),   # Row 281
            create_rows_response(654, 1),   # Row 654
            create_rows_response(759, 1),   # Row 759
        ]
        mock_session_class.return_value = mock_session

        # Test sampling
        client = DatasetsServerClient()
        result = client.sample_rows("dataset", "config", "train", n_samples=5, seed=42)

        assert len(result.rows) == 5
        assert result.num_rows_total == 1000
        assert result.features == [{"name": "text", "type": "string"}]
        # Verify we got the expected rows
        expected_texts = ["Row 25", "Row 114", "Row 281", "Row 654", "Row 759"]
        actual_texts = [row["row"]["text"] for row in result.rows]
        assert actual_texts == expected_texts

    @patch("datasets_server.client.requests.Session")
    def test_sample_rows_with_seed(self, mock_session_class):
        """Test that sampling with seed is deterministic."""
        # Mock info response
        mock_info_response = Mock()
        mock_info_response.status_code = 200
        mock_info_response.json.return_value = {
            "dataset_info": {
                "splits": {
                    "train": {"name": "train", "num_examples": 100}
                }
            },
            "pending": [],
            "failed": [],
            "partial": False,
        }

        # Mock rows responses
        def create_rows_response(offset, length):
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "features": [{"name": "id", "type": "int32"}],
                "rows": [{"row": {"id": i}} for i in range(offset, min(offset + length, 100))],
                "num_rows_total": 100,
            }
            return response

        # Mock session
        mock_session = Mock()
        # We'll need multiple calls for different tests
        mock_session.request.side_effect = [
            mock_info_response,
            create_rows_response(8, 100),  # First sampling
            mock_info_response,
            create_rows_response(8, 100),  # Second sampling with same seed
        ]
        mock_session_class.return_value = mock_session

        client = DatasetsServerClient()

        # First sampling with seed
        result1 = client.sample_rows("dataset", "config", "train", n_samples=10, seed=123)
        rows1 = [row["row"]["id"] for row in result1.rows]

        # Second sampling with same seed should give same results
        result2 = client.sample_rows("dataset", "config", "train", n_samples=10, seed=123)
        rows2 = [row["row"]["id"] for row in result2.rows]

        assert rows1 == rows2

    @patch("datasets_server.client.requests.Session")
    def test_sample_rows_zero_samples(self, mock_session_class):
        """Test sample_rows with n_samples=0."""
        client = DatasetsServerClient()
        result = client.sample_rows("dataset", "config", "train", n_samples=0)

        assert len(result.rows) == 0
        assert result.num_rows_total == 0
        assert result.features == []

    @patch("datasets_server.client.requests.Session")
    def test_sample_rows_negative_samples(self, mock_session_class):
        """Test sample_rows with negative n_samples."""
        client = DatasetsServerClient()

        with pytest.raises(ValueError) as exc_info:
            client.sample_rows("dataset", "config", "train", n_samples=-5)

        assert "n_samples must be non-negative" in str(exc_info.value)

    @patch("datasets_server.client.requests.Session")
    def test_sample_rows_exceeds_dataset_size(self, mock_session_class):
        """Test sample_rows when requesting more samples than available."""
        # Mock info response with small dataset
        mock_info_response = Mock()
        mock_info_response.status_code = 200
        mock_info_response.json.return_value = {
            "dataset_info": {
                "splits": {
                    "train": {"name": "train", "num_examples": 50}
                }
            },
            "pending": [],
            "failed": [],
            "partial": False,
        }

        # Mock session
        mock_session = Mock()
        mock_session.request.return_value = mock_info_response
        mock_session_class.return_value = mock_session

        client = DatasetsServerClient()

        with pytest.raises(ValueError) as exc_info:
            client.sample_rows("dataset", "config", "train", n_samples=100)

        assert "Requested 100 samples but dataset only has 50 rows" in str(exc_info.value)

    @patch("datasets_server.client.requests.Session")
    def test_sample_rows_fallback_to_get_rows(self, mock_session_class):
        """Test sample_rows when info doesn't contain split information."""
        # Mock info response without splits
        mock_info_response = Mock()
        mock_info_response.status_code = 200
        mock_info_response.json.return_value = {
            "dataset_info": {"description": "Test dataset"},
            "pending": [],
            "failed": [],
            "partial": False,
        }

        # Mock first get_rows call to get total rows
        mock_first_batch = Mock()
        mock_first_batch.status_code = 200
        mock_first_batch.json.return_value = {
            "features": [{"name": "text", "type": "string"}],
            "rows": [{"row": {"text": "Row 0"}}],
            "num_rows_total": 1000,
        }

        # Create mock rows response factory
        def create_rows_response(offset, length):
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "features": [{"name": "text", "type": "string"}],
                "rows": [{"row": {"text": f"Row {i}"}} for i in range(offset, min(offset + length, 1000))],
                "num_rows_total": 1000,
            }
            return response

        # Mock session
        mock_session = Mock()
        # With seed=42 and 1000 rows, the indices are [25, 114, 281, 654, 759]
        mock_session.request.side_effect = [
            mock_info_response,
            mock_first_batch,  # First call to get total rows
            create_rows_response(25, 90),   # Rows 25-114
            create_rows_response(281, 1),   # Row 281
            create_rows_response(654, 1),   # Row 654
            create_rows_response(759, 1),   # Row 759
        ]
        mock_session_class.return_value = mock_session

        client = DatasetsServerClient()
        result = client.sample_rows("dataset", "config", "train", n_samples=5, seed=42)

        assert len(result.rows) == 5
        assert result.num_rows_total == 1000


    @patch("datasets_server.client.requests.Session")
    def test_sample_rows_with_max_requests(self, mock_session_class):
        """Test sample_rows with max_requests parameter."""
        # Mock info response
        mock_info_response = Mock()
        mock_info_response.status_code = 200
        mock_info_response.json.return_value = {
            "dataset_info": {
                "splits": {
                    "train": {"name": "train", "num_examples": 10000}
                }
            },
            "pending": [],
            "failed": [],
            "partial": False,
        }

        # Mock rows responses for limited requests
        def create_rows_response(offset, length):
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "features": [{"name": "text", "type": "string"}],
                "rows": [{"row": {"text": f"Row {i}"}} for i in range(offset, min(offset + length, 10000))],
                "num_rows_total": 10000,
            }
            return response

        # Mock session
        mock_session = Mock()
        # With max_requests=2, we expect 2 API calls for get_rows
        mock_session.request.side_effect = [
            mock_info_response,
            create_rows_response(0, 100),    # First segment
            create_rows_response(5000, 100), # Second segment
        ]
        mock_session_class.return_value = mock_session

        client = DatasetsServerClient()
        result = client.sample_rows("dataset", "config", "train", n_samples=10, seed=42, max_requests=2)

        # Should have made exactly 3 API calls (1 info + 2 get_rows)
        assert mock_session.request.call_count == 3
        assert len(result.rows) == 10
        assert result.num_rows_total == 10000

    @patch("datasets_server.client.requests.Session")
    def test_sample_rows_max_requests_one(self, mock_session_class):
        """Test sample_rows with max_requests=1 (single API call)."""
        # Mock info response
        mock_info_response = Mock()
        mock_info_response.status_code = 200
        mock_info_response.json.return_value = {
            "dataset_info": {
                "splits": {
                    "train": {"name": "train", "num_examples": 50000}
                }
            },
            "pending": [],
            "failed": [],
            "partial": False,
        }

        # Mock single rows response
        def create_rows_response(offset, length):
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "features": [{"name": "text", "type": "string"}, {"name": "label", "type": "int"}],
                "rows": [
                    {"row": {"text": f"Row {i}", "label": i % 2}} 
                    for i in range(offset, min(offset + length, 50000))
                ],
                "num_rows_total": 50000,
            }
            return response

        # Mock session
        mock_session = Mock()
        mock_session.request.side_effect = [
            mock_info_response,
            create_rows_response(15000, 100),  # Single request somewhere in the middle
        ]
        mock_session_class.return_value = mock_session

        client = DatasetsServerClient()
        result = client.sample_rows("dataset", "config", "train", n_samples=5, seed=123, max_requests=1)

        # Should have made exactly 2 API calls (1 info + 1 get_rows)
        assert mock_session.request.call_count == 2
        assert len(result.rows) == 5
        assert result.num_rows_total == 50000
        # All samples should come from the same batch
        row_nums = [int(row["row"]["text"].split()[1]) for row in result.rows]
        assert all(15000 <= num < 15100 for num in row_nums)

    @patch("datasets_server.client.requests.Session")
    def test_sample_rows_max_requests_exceeds_samples(self, mock_session_class):
        """Test sample_rows when max_requests > n_samples."""
        # Mock info response
        mock_info_response = Mock()
        mock_info_response.status_code = 200
        mock_info_response.json.return_value = {
            "dataset_info": {
                "splits": {
                    "train": {"name": "train", "num_examples": 1000}
                }
            },
            "pending": [],
            "failed": [],
            "partial": False,
        }

        # Mock rows responses
        def create_rows_response(offset, length):
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "features": [{"name": "text", "type": "string"}],
                "rows": [{"row": {"text": f"Row {i}"}} for i in range(offset, min(offset + length, 1000))],
                "num_rows_total": 1000,
            }
            return response

        # Mock session
        mock_session = Mock()
        # With max_requests=5 but only n_samples=3, we should still get good distribution
        mock_session.request.side_effect = [
            mock_info_response,
            create_rows_response(0, 100),    # Segment 1
            create_rows_response(200, 100),  # Segment 2
            create_rows_response(400, 100),  # Segment 3
            # Segments 4 and 5 won't be called since we only need 3 samples
        ]
        mock_session_class.return_value = mock_session

        client = DatasetsServerClient()
        result = client.sample_rows("dataset", "config", "train", n_samples=3, seed=42, max_requests=5)

        # Should have made 4 API calls (1 info + 3 get_rows for first 3 segments)
        assert mock_session.request.call_count == 4
        assert len(result.rows) == 3
        assert result.num_rows_total == 1000
