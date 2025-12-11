"""Tests for asynchronous client."""

import httpx
import pytest
import respx

from datasets_server import AsyncDatasetsServerClient, DatasetNotFoundError, DatasetServerError
from datasets_server.models import DatasetValidity


class TestAsyncDatasetsServerClient:
    """Test asynchronous client."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test async client initialization."""
        client = AsyncDatasetsServerClient()
        assert client.endpoint == "https://datasets-server.huggingface.co"
        assert client.timeout == 30.0
        assert client._session is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with AsyncDatasetsServerClient() as client:
            assert client._session is not None
        # Session should be closed after exiting

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_valid_success(self):
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

        async with AsyncDatasetsServerClient() as client:
            validity = await client.is_valid("test-dataset")

        assert isinstance(validity, DatasetValidity)
        assert validity.viewer is True
        assert validity.preview is True
        assert validity.search is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_dataset_not_found(self):
        """Test handling of 404 errors."""
        respx.get("https://datasets-server.huggingface.co/is-valid").mock(
            return_value=httpx.Response(404, json={"error": "Not found"})
        )

        async with AsyncDatasetsServerClient() as client:
            with pytest.raises(DatasetNotFoundError) as exc_info:
                await client.is_valid("non-existent")

        assert "Dataset not found: non-existent" in str(exc_info.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_list_splits(self):
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

        async with AsyncDatasetsServerClient() as client:
            splits = await client.list_splits("squad")

        assert len(splits) == 2
        assert splits[0].split == "train"
        assert splits[1].split == "validation"

    @pytest.mark.asyncio
    @respx.mock
    async def test_concurrent_requests(self):
        """Test making concurrent requests."""
        import asyncio

        # Mock multiple endpoints
        datasets = ["dataset1", "dataset2", "dataset3"]
        for ds in datasets:
            respx.get("https://datasets-server.huggingface.co/is-valid", params={"dataset": ds}).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "viewer": True,
                        "preview": True,
                        "search": True,
                        "filter": True,
                        "statistics": True,
                    },
                )
            )

        async with AsyncDatasetsServerClient() as client:
            tasks = [client.is_valid(ds) for ds in datasets]
            results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(r.viewer for r in results)

    @pytest.mark.asyncio
    async def test_get_rows_with_invalid_length(self):
        """Test get_rows with length > 100."""
        async with AsyncDatasetsServerClient() as client:
            with pytest.raises(ValueError) as exc_info:
                await client.get_rows("dataset", "config", "split", length=150)

            assert "Length cannot exceed 100" in str(exc_info.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_iter_rows(self):
        """Test async row iteration."""
        route = respx.get("https://datasets-server.huggingface.co/rows")
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
        ]

        async with AsyncDatasetsServerClient() as client:
            rows = []
            async for row in client.iter_rows("test", "config", "split"):
                rows.append(row)

        assert len(rows) == 150
        assert rows[0]["text"] == "Row 0"
        assert rows[149]["text"] == "Row 149"

    @pytest.mark.asyncio
    @respx.mock
    async def test_search(self):
        """Test search functionality."""
        respx.get("https://datasets-server.huggingface.co/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Test result"}}],
                    "num_rows_total": 42,
                },
            )
        )

        async with AsyncDatasetsServerClient() as client:
            results = await client.search(
                dataset="squad",
                query="test",
                config="plain_text",
                split="train",
                length=10,
            )

        assert len(results.rows) == 1
        assert results.num_rows_total == 42

    @pytest.mark.asyncio
    async def test_close_session(self):
        """Test closing session manually when using context manager."""
        # When using context manager, a dedicated session is created
        async with AsyncDatasetsServerClient() as client:
            assert client._session is not None
            assert client._owns_session is True

        # Session should be closed and ownership released after context exit
        assert client._session is None
        assert client._owns_session is False

    @pytest.mark.asyncio
    async def test_global_session_used_by_default(self):
        """Test that global session is used when not using context manager."""
        client = AsyncDatasetsServerClient()
        # _session should be None (uses global session via property)
        assert client._session is None
        # But session property returns the global session
        session = client.session
        assert session is not None

    @pytest.mark.asyncio
    @respx.mock
    async def test_generic_exception(self):
        """Test handling of generic exceptions during requests."""
        respx.get("https://datasets-server.huggingface.co/is-valid").mock(
            side_effect=Exception("Network error")
        )

        async with AsyncDatasetsServerClient() as client:
            with pytest.raises(DatasetServerError) as exc_info:
                await client.is_valid("error-dataset")

        assert "Request failed: Network error" in str(exc_info.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_info_with_config(self):
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

        async with AsyncDatasetsServerClient() as client:
            info = await client.get_info("dataset", config="my_config")

        assert info.dataset_info["description"] == "Test dataset"

    @pytest.mark.asyncio
    @respx.mock
    async def test_filter_with_where_clause(self):
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

        async with AsyncDatasetsServerClient() as client:
            results = await client.filter(
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

    @pytest.mark.asyncio
    @respx.mock
    async def test_sample_rows_basic(self):
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

        async with AsyncDatasetsServerClient() as client:
            result = await client.sample_rows("dataset", "config", "train", n_samples=5, seed=42)

        assert len(result.rows) == 5
        assert result.num_rows_total == 1000
        assert result.features == [{"name": "text", "type": "string"}]

    @pytest.mark.asyncio
    @respx.mock
    async def test_sample_rows_with_seed(self):
        """Test that sampling with seed is deterministic."""
        # Mock info endpoint (called twice)
        info_route = respx.get("https://datasets-server.huggingface.co/info")
        info_route.side_effect = [
            httpx.Response(
                200,
                json={
                    "dataset_info": {
                        "splits": {
                            "train": {"name": "train", "num_examples": 100}
                        }
                    },
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            ),
            httpx.Response(
                200,
                json={
                    "dataset_info": {
                        "splits": {
                            "train": {"name": "train", "num_examples": 100}
                        }
                    },
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
                    "rows": [{"row": {"id": i}} for i in range(4, 99)],
                    "num_rows_total": 100,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "id", "type": "int32"}],
                    "rows": [{"row": {"id": i}} for i in range(4, 99)],
                    "num_rows_total": 100,
                },
            ),
        ]

        async with AsyncDatasetsServerClient() as client:
            # First sampling with seed
            result1 = await client.sample_rows("dataset", "config", "train", n_samples=10, seed=123)
            rows1 = [row["row"]["id"] for row in result1.rows]

            # Second sampling with same seed should give same results
            result2 = await client.sample_rows("dataset", "config", "train", n_samples=10, seed=123)
            rows2 = [row["row"]["id"] for row in result2.rows]

        assert rows1 == rows2

    @pytest.mark.asyncio
    async def test_sample_rows_zero_samples(self):
        """Test sample_rows with n_samples=0."""
        async with AsyncDatasetsServerClient() as client:
            result = await client.sample_rows("dataset", "config", "train", n_samples=0)

            assert len(result.rows) == 0
            assert result.num_rows_total == 0
            assert result.features == []

    @pytest.mark.asyncio
    async def test_sample_rows_negative_samples(self):
        """Test sample_rows with negative n_samples."""
        async with AsyncDatasetsServerClient() as client:
            with pytest.raises(ValueError) as exc_info:
                await client.sample_rows("dataset", "config", "train", n_samples=-5)

            assert "n_samples must be non-negative" in str(exc_info.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_sample_rows_exceeds_dataset_size(self):
        """Test sample_rows when requesting more samples than available."""
        respx.get("https://datasets-server.huggingface.co/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "dataset_info": {
                        "splits": {
                            "train": {"name": "train", "num_examples": 50}
                        }
                    },
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            )
        )

        async with AsyncDatasetsServerClient() as client:
            with pytest.raises(ValueError) as exc_info:
                await client.sample_rows("dataset", "config", "train", n_samples=100)

            assert "Requested 100 samples but dataset only has 50 rows" in str(exc_info.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_sample_rows_fallback_to_get_rows(self):
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

        # Mock rows endpoint
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

        async with AsyncDatasetsServerClient() as client:
            result = await client.sample_rows("dataset", "config", "train", n_samples=5, seed=42)

        assert len(result.rows) == 5
        assert result.num_rows_total == 1000

    @pytest.mark.asyncio
    @respx.mock
    async def test_sample_rows_with_max_requests(self):
        """Test sample_rows with max_requests parameter."""
        respx.get("https://datasets-server.huggingface.co/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "dataset_info": {
                        "splits": {
                            "train": {"name": "train", "num_examples": 10000}
                        }
                    },
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
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(100)],
                    "num_rows_total": 10000,
                },
            ),
        ]

        async with AsyncDatasetsServerClient() as client:
            result = await client.sample_rows("dataset", "config", "train", n_samples=10, seed=42, max_requests=2)

        assert len(result.rows) == 10
        assert result.num_rows_total == 10000

    @pytest.mark.asyncio
    @respx.mock
    async def test_sample_rows_max_requests_one(self):
        """Test sample_rows with max_requests=1 (single API call)."""
        respx.get("https://datasets-server.huggingface.co/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "dataset_info": {
                        "splits": {
                            "train": {"name": "train", "num_examples": 50000}
                        }
                    },
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
                    "rows": [
                        {"row": {"text": f"Row {i}", "label": i % 2}}
                        for i in range(15000, 15100)
                    ],
                    "num_rows_total": 50000,
                },
            )
        )

        async with AsyncDatasetsServerClient() as client:
            result = await client.sample_rows("dataset", "config", "train", n_samples=5, seed=123, max_requests=1)

        assert len(result.rows) == 5
        assert result.num_rows_total == 50000

    @pytest.mark.asyncio
    @respx.mock
    async def test_sample_rows_max_requests_concurrent(self):
        """Test that max_requests properly limits concurrent API calls."""
        respx.get("https://datasets-server.huggingface.co/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "dataset_info": {
                        "splits": {
                            "train": {"name": "train", "num_examples": 100000}
                        }
                    },
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
                    "features": [{"name": "text", "type": "string"}, {"name": "id", "type": "int"}],
                    "rows": [
                        {"row": {"text": f"Row {i}", "id": i}}
                        for i in range(100)
                    ],
                    "num_rows_total": 100000,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}, {"name": "id", "type": "int"}],
                    "rows": [
                        {"row": {"text": f"Row {i}", "id": i}}
                        for i in range(100)
                    ],
                    "num_rows_total": 100000,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}, {"name": "id", "type": "int"}],
                    "rows": [
                        {"row": {"text": f"Row {i}", "id": i}}
                        for i in range(100)
                    ],
                    "num_rows_total": 100000,
                },
            ),
            httpx.Response(
                200,
                json={
                    "features": [{"name": "text", "type": "string"}, {"name": "id", "type": "int"}],
                    "rows": [
                        {"row": {"text": f"Row {i}", "id": i}}
                        for i in range(100)
                    ],
                    "num_rows_total": 100000,
                },
            ),
        ]

        async with AsyncDatasetsServerClient() as client:
            result = await client.sample_rows("dataset", "config", "train", n_samples=20, seed=42, max_requests=4)

        assert len(result.rows) == 20
        assert result.num_rows_total == 100000
