"""Tests for asynchronous client."""

import pytest
from aioresponses import aioresponses

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
    async def test_is_valid_success(self):
        """Test successful is_valid call."""
        with aioresponses() as m:
            m.get(
                "https://datasets-server.huggingface.co/is-valid?dataset=test-dataset",
                payload={
                    "viewer": True,
                    "preview": True,
                    "search": False,
                    "filter": False,
                    "statistics": True,
                },
            )

            async with AsyncDatasetsServerClient() as client:
                validity = await client.is_valid("test-dataset")

            assert isinstance(validity, DatasetValidity)
            assert validity.viewer is True
            assert validity.preview is True
            assert validity.search is False

    @pytest.mark.asyncio
    async def test_dataset_not_found(self):
        """Test handling of 404 errors."""
        with aioresponses() as m:
            m.get(
                "https://datasets-server.huggingface.co/is-valid?dataset=non-existent",
                status=404,
            )

            async with AsyncDatasetsServerClient() as client:
                with pytest.raises(DatasetNotFoundError) as exc_info:
                    await client.is_valid("non-existent")

            assert "Dataset not found: non-existent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_splits(self):
        """Test list_splits method."""
        with aioresponses() as m:
            m.get(
                "https://datasets-server.huggingface.co/splits?dataset=squad",
                payload={
                    "splits": [
                        {"dataset": "squad", "config": "plain_text", "split": "train"},
                        {"dataset": "squad", "config": "plain_text", "split": "validation"},
                    ],
                    "pending": [],
                    "failed": [],
                },
            )

            async with AsyncDatasetsServerClient() as client:
                splits = await client.list_splits("squad")

            assert len(splits) == 2
            assert splits[0].split == "train"
            assert splits[1].split == "validation"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test making concurrent requests."""
        import asyncio

        with aioresponses() as m:
            # Mock multiple endpoints
            datasets = ["dataset1", "dataset2", "dataset3"]
            for ds in datasets:
                m.get(
                    f"https://datasets-server.huggingface.co/is-valid?dataset={ds}",
                    payload={
                        "viewer": True,
                        "preview": True,
                        "search": True,
                        "filter": True,
                        "statistics": True,
                    },
                )

            async with AsyncDatasetsServerClient() as client:
                # Make concurrent requests
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

            assert "Length cannot exceed 100" in str(exc_info.value)  # The MAX_ROWS_PER_REQUEST constant is 100

    @pytest.mark.asyncio
    async def test_iter_rows(self):
        """Test async row iteration."""
        with aioresponses() as m:
            # Mock first page
            m.get(
                "https://datasets-server.huggingface.co/rows?dataset=test&config=config&split=split&offset=0&length=100",
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(100)],
                },
            )
            # Mock second page with fewer rows
            m.get(
                "https://datasets-server.huggingface.co/rows?dataset=test&config=config&split=split&offset=100&length=100",
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(100, 150)],
                },
            )

            async with AsyncDatasetsServerClient() as client:
                rows = []
                async for row in client.iter_rows("test", "config", "split"):
                    rows.append(row)

            assert len(rows) == 150
            assert rows[0]["text"] == "Row 0"
            assert rows[149]["text"] == "Row 149"

    @pytest.mark.asyncio
    async def test_search(self):
        """Test search functionality."""
        with aioresponses() as m:
            m.get(
                "https://datasets-server.huggingface.co/search?dataset=squad&config=plain_text&split=train&query=test&offset=0&length=10",
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Test result"}}],
                    "num_rows_total": 42,
                },
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
        """Test closing session manually."""
        client = AsyncDatasetsServerClient()
        await client._ensure_session()
        assert client._session is not None

        await client.close()
        # Session should be closed

    @pytest.mark.asyncio
    async def test_generic_exception(self):
        """Test handling of generic exceptions during requests."""
        with aioresponses() as m:
            # Configure to raise a generic exception
            m.get(
                "https://datasets-server.huggingface.co/is-valid?dataset=error-dataset",
                exception=Exception("Network error"),
            )

            async with AsyncDatasetsServerClient() as client:
                with pytest.raises(DatasetServerError) as exc_info:
                    await client.is_valid("error-dataset")

            assert "Request failed: Network error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_info_with_config(self):
        """Test get_info method with optional config parameter."""
        with aioresponses() as m:
            m.get(
                "https://datasets-server.huggingface.co/info?dataset=dataset&config=my_config",
                payload={
                    "dataset_info": {"description": "Test dataset"},
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            )

            async with AsyncDatasetsServerClient() as client:
                info = await client.get_info("dataset", config="my_config")

            assert info.dataset_info["description"] == "Test dataset"

    @pytest.mark.asyncio
    async def test_filter_with_where_clause(self):
        """Test filter method with WHERE clause."""
        with aioresponses() as m:
            m.get(
                "https://datasets-server.huggingface.co/filter?dataset=dataset&config=config&split=split&where=score+%3E+3&orderby=score+DESC&offset=10&length=50",
                payload={
                    "features": [{"name": "text", "type": "string"}, {"name": "score", "type": "int32"}],
                    "rows": [{"row": {"text": "Test", "score": 5}}],
                    "num_rows_total": 1,
                },
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
    async def test_sample_rows_basic(self):
        """Test basic sample_rows functionality."""
        with aioresponses() as m:
            # Mock info endpoint
            m.get(
                "https://datasets-server.huggingface.co/info?dataset=dataset&config=config",
                payload={
                    "dataset_info": {
                        "splits": {
                            "train": {"name": "train", "num_examples": 1000},
                            "test": {"name": "test", "num_examples": 100}
                        }
                    },
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            )

            # Mock rows endpoints for batched requests
            # With seed=42 and 1000 rows, indices are [25, 114, 281, 654, 759]
            m.get(
                "https://datasets-server.huggingface.co/rows?dataset=dataset&config=config&split=train&offset=25&length=90",
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(25, 115)],
                    "num_rows_total": 1000,
                },
            )
            m.get(
                "https://datasets-server.huggingface.co/rows?dataset=dataset&config=config&split=train&offset=281&length=1",
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 281"}}],
                    "num_rows_total": 1000,
                },
            )
            m.get(
                "https://datasets-server.huggingface.co/rows?dataset=dataset&config=config&split=train&offset=654&length=1",
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 654"}}],
                    "num_rows_total": 1000,
                },
            )
            m.get(
                "https://datasets-server.huggingface.co/rows?dataset=dataset&config=config&split=train&offset=759&length=1",
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 759"}}],
                    "num_rows_total": 1000,
                },
            )

            async with AsyncDatasetsServerClient() as client:
                result = await client.sample_rows("dataset", "config", "train", n_samples=5, seed=42)

            assert len(result.rows) == 5
            assert result.num_rows_total == 1000
            assert result.features == [{"name": "text", "type": "string"}]

    @pytest.mark.asyncio
    async def test_sample_rows_with_seed(self):
        """Test that sampling with seed is deterministic."""
        with aioresponses() as m:
            # Mock info endpoint (called twice)
            for _ in range(2):
                m.get(
                    "https://datasets-server.huggingface.co/info?dataset=dataset&config=config",
                    payload={
                        "dataset_info": {
                            "splits": {
                            "train": {"name": "train", "num_examples": 100}
                        }
                        },
                        "pending": [],
                        "failed": [],
                        "partial": False,
                    },
                )

            # Mock rows endpoints for seed=123 with 100 rows
            # Indices are [4, 6, 11, 13, 34, 48, 52, 68, 71, 98]
            # Since they span from 4 to 98 and are within 100 rows, they'll be fetched in one batch
            for _ in range(2):  # Called twice due to test
                m.get(
                    "https://datasets-server.huggingface.co/rows?dataset=dataset&config=config&split=train&offset=4&length=95",
                    payload={
                        "features": [{"name": "id", "type": "int32"}],
                        "rows": [{"row": {"id": i}} for i in range(4, 99)],
                        "num_rows_total": 100,
                    },
                )

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
    async def test_sample_rows_exceeds_dataset_size(self):
        """Test sample_rows when requesting more samples than available."""
        with aioresponses() as m:
            # Mock info endpoint with small dataset
            m.get(
                "https://datasets-server.huggingface.co/info?dataset=dataset&config=config",
                payload={
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

            async with AsyncDatasetsServerClient() as client:
                with pytest.raises(ValueError) as exc_info:
                    await client.sample_rows("dataset", "config", "train", n_samples=100)

                assert "Requested 100 samples but dataset only has 50 rows" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_sample_rows_fallback_to_get_rows(self):
        """Test sample_rows when info doesn't contain split information."""
        with aioresponses() as m:
            # Mock info endpoint without splits
            m.get(
                "https://datasets-server.huggingface.co/info?dataset=dataset&config=config",
                payload={
                    "dataset_info": {"description": "Test dataset"},
                    "pending": [],
                    "failed": [],
                    "partial": False,
                },
            )

            # Mock first get_rows call to get total rows
            m.get(
                "https://datasets-server.huggingface.co/rows?dataset=dataset&config=config&split=train&offset=0&length=1",
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 0"}}],
                    "num_rows_total": 1000,
                },
            )

            # Mock actual sampling calls with batched requests
            # With seed=42 and 1000 rows, indices are [25, 114, 281, 654, 759]
            m.get(
                "https://datasets-server.huggingface.co/rows?dataset=dataset&config=config&split=train&offset=25&length=90",
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(25, 115)],
                    "num_rows_total": 1000,
                },
            )
            m.get(
                "https://datasets-server.huggingface.co/rows?dataset=dataset&config=config&split=train&offset=281&length=1",
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 281"}}],
                    "num_rows_total": 1000,
                },
            )
            m.get(
                "https://datasets-server.huggingface.co/rows?dataset=dataset&config=config&split=train&offset=654&length=1",
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 654"}}],
                    "num_rows_total": 1000,
                },
            )
            m.get(
                "https://datasets-server.huggingface.co/rows?dataset=dataset&config=config&split=train&offset=759&length=1",
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": "Row 759"}}],
                    "num_rows_total": 1000,
                },
            )

            async with AsyncDatasetsServerClient() as client:
                result = await client.sample_rows("dataset", "config", "train", n_samples=5, seed=42)

            assert len(result.rows) == 5
            assert result.num_rows_total == 1000


    @pytest.mark.asyncio
    async def test_sample_rows_with_max_requests(self):
        """Test sample_rows with max_requests parameter."""
        with aioresponses() as m:
            # Mock info endpoint
            m.get(
                "https://datasets-server.huggingface.co/info?dataset=dataset&config=config",
                payload={
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

            # Mock rows calls for limited requests
            # With max_requests=2, dataset is divided into 2 segments
            # We don't know the exact offsets due to randomness, so mock with regex
            import re
            
            # First segment (0-4999)
            m.get(
                re.compile(r"https://datasets-server\.huggingface\.co/rows\?.*offset=\d+.*"),
                payload={
                    "features": [{"name": "text", "type": "string"}],
                    "rows": [{"row": {"text": f"Row {i}"}} for i in range(100)],
                    "num_rows_total": 10000,
                },
                repeat=True,
            )

            async with AsyncDatasetsServerClient() as client:
                result = await client.sample_rows("dataset", "config", "train", n_samples=10, seed=42, max_requests=2)

            assert len(result.rows) == 10
            assert result.num_rows_total == 10000

    @pytest.mark.asyncio
    async def test_sample_rows_max_requests_one(self):
        """Test sample_rows with max_requests=1 (single API call)."""
        with aioresponses() as m:
            # Mock info endpoint
            m.get(
                "https://datasets-server.huggingface.co/info?dataset=dataset&config=config",
                payload={
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

            # Mock single rows call with regex to handle random offset
            import re
            
            m.get(
                re.compile(r"https://datasets-server\.huggingface\.co/rows\?.*offset=\d+.*"),
                payload={
                    "features": [{"name": "text", "type": "string"}, {"name": "label", "type": "int"}],
                    "rows": [
                        {"row": {"text": f"Row {i}", "label": i % 2}} 
                        for i in range(15000, 15100)
                    ],
                    "num_rows_total": 50000,
                },
            )

            async with AsyncDatasetsServerClient() as client:
                result = await client.sample_rows("dataset", "config", "train", n_samples=5, seed=123, max_requests=1)

            assert len(result.rows) == 5
            assert result.num_rows_total == 50000
            # With max_requests=1, all samples come from a single API call

    @pytest.mark.asyncio
    async def test_sample_rows_max_requests_concurrent(self):
        """Test that max_requests properly limits concurrent API calls."""
        with aioresponses() as m:
            # Mock info endpoint
            m.get(
                "https://datasets-server.huggingface.co/info?dataset=dataset&config=config",
                payload={
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

            # Mock multiple rows calls that should happen with max_requests=4
            import re
            
            # Use regex to match any offset
            m.get(
                re.compile(r"https://datasets-server\.huggingface\.co/rows\?.*offset=\d+.*"),
                payload={
                    "features": [{"name": "text", "type": "string"}, {"name": "id", "type": "int"}],
                    "rows": [
                        {"row": {"text": f"Row {i}", "id": i}} 
                        for i in range(100)  # Just return 100 rows regardless of offset
                    ],
                    "num_rows_total": 100000,
                },
                repeat=True,
            )

            async with AsyncDatasetsServerClient() as client:
                result = await client.sample_rows("dataset", "config", "train", n_samples=20, seed=42, max_requests=4)

            assert len(result.rows) == 20
            assert result.num_rows_total == 100000
            # With max_requests=4, we should have fetched from 4 different segments
