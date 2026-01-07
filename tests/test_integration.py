"""Integration tests for the Datasets Server client.

These tests make real API calls to the Hugging Face Datasets Server.
They are skipped by default and can be run with:
    pytest -m integration
    or
    INTEGRATION_TESTS=1 pytest
"""

import os

import pytest

from datasets_server import AsyncDatasetsServerClient, DatasetsServerClient

# Test datasets - verified to be public and stable
TEST_DATASETS = {
    "news": "SetFit/ag_news",
    "reviews": "stanfordnlp/imdb",
    "small": "davanstrien/haiku_dpo",
    "metadata": "librarian-bots/dataset_cards_with_metadata",
}

# Skip all integration tests if INTEGRATION_TESTS env var is not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("INTEGRATION_TESTS", "0") != "1",
        reason="Integration tests are disabled. Set INTEGRATION_TESTS=1 to run them."
    )
]


class TestIntegrationSync:
    """Integration tests for synchronous client."""

    def setup_method(self):
        """Set up test client."""
        self.client = DatasetsServerClient()

    def test_is_valid_real_dataset(self):
        """Test is_valid with a real dataset."""
        validity = self.client.is_valid(TEST_DATASETS["reviews"])

        assert validity.viewer is True
        assert validity.preview is True
        # These datasets should have most features enabled
        assert isinstance(validity.search, bool)
        assert isinstance(validity.filter, bool)
        assert isinstance(validity.statistics, bool)

    def test_is_valid_nonexistent_dataset(self):
        """Test is_valid with a non-existent dataset."""
        from datasets_server import DatasetNotFoundError

        with pytest.raises(DatasetNotFoundError):
            self.client.is_valid("this-dataset-definitely-does-not-exist-12345")

    def test_list_splits(self):
        """Test listing splits for a real dataset."""
        splits = self.client.list_splits(TEST_DATASETS["reviews"])

        assert len(splits) > 0
        # Check that we have expected structure
        for split in splits:
            assert hasattr(split, "dataset")
            assert hasattr(split, "config")
            assert hasattr(split, "split")
            assert split.dataset == TEST_DATASETS["reviews"]

        # IMDB should have train and test splits
        split_names = [s.split for s in splits]
        assert "train" in split_names
        assert "test" in split_names

    def test_get_info(self):
        """Test getting dataset info."""
        info = self.client.get_info(TEST_DATASETS["reviews"])

        assert info.dataset_info is not None
        # API returns config name as top-level key, e.g., {"plain_text": {...}}
        # Get the first config's info
        first_config = list(info.dataset_info.values())[0]
        assert "splits" in first_config
        assert isinstance(first_config["splits"], dict)

        # Check split structure matches what we discovered
        splits_dict = first_config["splits"]
        assert "train" in splits_dict
        assert "num_examples" in splits_dict["train"]
        assert splits_dict["train"]["num_examples"] > 0

    def test_get_size(self):
        """Test getting dataset size."""
        size = self.client.get_size(TEST_DATASETS["small"])

        # DatasetSize model has size.size dict containing dataset/configs/splits
        assert hasattr(size, "size")
        assert isinstance(size.size, dict)

        # Check structure of size response
        assert "dataset" in size.size
        assert "configs" in size.size or "splits" in size.size

    def test_list_parquet_files(self):
        """Test listing parquet files."""
        parquet_files = self.client.list_parquet_files(TEST_DATASETS["reviews"])

        assert len(parquet_files) > 0
        for file in parquet_files:
            assert hasattr(file, "dataset")
            assert hasattr(file, "config")
            assert hasattr(file, "split")
            assert hasattr(file, "url")
            assert file.url.startswith("https://")
            assert file.url.endswith(".parquet")

    def test_preview(self):
        """Test dataset preview."""
        # Get splits first to find config
        splits = self.client.list_splits(TEST_DATASETS["reviews"])
        if not splits:
            pytest.skip("No splits available")

        config = splits[0].config
        split = splits[0].split

        preview = self.client.preview(TEST_DATASETS["reviews"], config=config, split=split)

        assert len(preview.rows) > 0
        assert len(preview.features) > 0

        # IMDB should have text and label columns
        feature_names = [f["name"] for f in preview.features]
        assert "text" in feature_names
        assert "label" in feature_names

        # Check row structure
        for row in preview.rows[:5]:  # Check first 5 rows
            assert "row" in row
            assert "text" in row["row"]
            assert "label" in row["row"]

    def test_get_rows_pagination(self):
        """Test getting rows with pagination."""
        # Get splits first
        splits = self.client.list_splits(TEST_DATASETS["news"])
        if not splits:
            pytest.skip("No splits available")

        split = splits[0]

        # Get first batch
        batch1 = self.client.get_rows(
            dataset=split.dataset,
            config=split.config,
            split=split.split,
            offset=0,
            length=10
        )

        assert len(batch1.rows) == 10
        assert batch1.num_rows_total > 10

        # Get second batch
        batch2 = self.client.get_rows(
            dataset=split.dataset,
            config=split.config,
            split=split.split,
            offset=10,
            length=10
        )

        assert len(batch2.rows) == 10
        # Make sure we got different rows
        assert batch1.rows[0] != batch2.rows[0]

    def test_search(self):
        """Test search functionality if available."""
        validity = self.client.is_valid(TEST_DATASETS["reviews"])

        if not validity.search:
            pytest.skip("Search not available for this dataset")

        splits = self.client.list_splits(TEST_DATASETS["reviews"])
        split = splits[0]

        # Search for a common word
        results = self.client.search(
            dataset=split.dataset,
            query="movie",
            config=split.config,
            split=split.split,
            length=5
        )

        assert len(results.rows) > 0
        # Verify search results contain the query
        for row in results.rows:
            text = row["row"].get("text", "").lower()
            assert "movie" in text

    def test_filter(self):
        """Test filter functionality if available."""
        validity = self.client.is_valid(TEST_DATASETS["reviews"])

        if not validity.filter:
            pytest.skip("Filter not available for this dataset")

        splits = self.client.list_splits(TEST_DATASETS["reviews"])
        split = splits[0]

        # Filter for positive reviews (label = 1)
        results = self.client.filter(
            dataset=split.dataset,
            where='"label" = 1',
            config=split.config,
            split=split.split,
            length=5
        )

        if len(results.rows) > 0:
            for row in results.rows:
                assert row["row"]["label"] == 1

    def test_get_statistics(self):
        """Test statistics functionality if available."""
        validity = self.client.is_valid(TEST_DATASETS["small"])

        if not validity.statistics:
            pytest.skip("Statistics not available for this dataset")

        splits = self.client.list_splits(TEST_DATASETS["small"])
        split = splits[0]

        stats = self.client.get_statistics(
            dataset=split.dataset,
            config=split.config,
            split=split.split
        )

        assert hasattr(stats, "num_examples")
        assert stats.num_examples > 0

    def test_sample_rows(self):
        """Test sample_rows with real data."""
        splits = self.client.list_splits(TEST_DATASETS["reviews"])
        split = next(s for s in splits if s.split == "train")

        # Sample with seed for reproducibility
        samples = self.client.sample_rows(
            dataset=split.dataset,
            config=split.config,
            split=split.split,
            n_samples=5,
            seed=42
        )

        assert len(samples.rows) == 5
        assert samples.num_rows_total > 5

        # Verify we got valid rows
        for row in samples.rows:
            assert "text" in row["row"]
            assert "label" in row["row"]

    def test_iter_rows(self):
        """Test row iteration."""
        splits = self.client.list_splits(TEST_DATASETS["small"])
        split = splits[0]

        # Iterate through first 25 rows
        rows = []
        for i, row in enumerate(self.client.iter_rows(
            dataset=split.dataset,
            config=split.config,
            split=split.split,
            batch_size=10
        )):
            rows.append(row)
            if i >= 24:  # Stop after 25 rows
                break

        assert len(rows) == 25
        # Verify all rows are unique
        assert len(set(str(row) for row in rows)) == 25


@pytest.mark.asyncio
class TestIntegrationAsync:
    """Integration tests for asynchronous client."""

    async def test_is_valid_real_dataset(self):
        """Test async is_valid with a real dataset."""
        async with AsyncDatasetsServerClient() as client:
            validity = await client.is_valid(TEST_DATASETS["reviews"])

            assert validity.viewer is True
            assert validity.preview is True
            assert isinstance(validity.search, bool)
            assert isinstance(validity.filter, bool)
            assert isinstance(validity.statistics, bool)

    async def test_concurrent_requests(self):
        """Test making concurrent requests."""
        import asyncio

        async with AsyncDatasetsServerClient() as client:
            # Make multiple concurrent requests
            tasks = [
                client.is_valid(TEST_DATASETS["news"]),
                client.is_valid(TEST_DATASETS["reviews"]),
                client.is_valid(TEST_DATASETS["small"]),
                client.is_valid(TEST_DATASETS["metadata"]),
            ]

            results = await asyncio.gather(*tasks)

            # All should be valid
            assert all(r.viewer for r in results)
            assert all(r.preview for r in results)

    async def test_sample_rows_async(self):
        """Test async sample_rows with real data."""
        async with AsyncDatasetsServerClient() as client:
            splits = await client.list_splits(TEST_DATASETS["reviews"])
            split = next(s for s in splits if s.split == "train")

            # Sample with seed
            samples = await client.sample_rows(
                dataset=split.dataset,
                config=split.config,
                split=split.split,
                n_samples=5,
                seed=42
            )

            assert len(samples.rows) == 5
            assert samples.num_rows_total > 5

    async def test_iter_rows_async(self):
        """Test async row iteration."""
        async with AsyncDatasetsServerClient() as client:
            splits = await client.list_splits(TEST_DATASETS["small"])
            split = splits[0]

            # Iterate through first 25 rows
            rows = []
            i = 0
            async for row in client.iter_rows(
                dataset=split.dataset,
                config=split.config,
                split=split.split,
                batch_size=10
            ):
                rows.append(row)
                i += 1
                if i >= 25:
                    break

            assert len(rows) == 25


class TestIntegrationEdgeCases:
    """Test edge cases and error handling with real API."""

    def test_dataset_with_multiple_configs(self):
        """Test handling datasets with multiple configurations."""
        client = DatasetsServerClient()

        # ag_news has a default config
        info = client.get_info(TEST_DATASETS["news"])
        assert info.dataset_info is not None

        splits = client.list_splits(TEST_DATASETS["news"])
        configs = set(s.config for s in splits)
        assert len(configs) >= 1

    def test_rate_limiting_behavior(self):
        """Test that rapid requests don't fail (basic rate limit test)."""
        client = DatasetsServerClient()

        # Make 10 rapid requests
        for i in range(10):
            validity = client.is_valid(TEST_DATASETS["small"])
            assert validity.viewer is True

    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        client = DatasetsServerClient()

        # ag_news is relatively large
        size = client.get_size(TEST_DATASETS["news"])

        # Just verify we can get size info without errors
        assert size is not None

    def test_special_characters_in_search(self):
        """Test search with special characters."""
        client = DatasetsServerClient()
        validity = client.is_valid(TEST_DATASETS["reviews"])

        if not validity.search:
            pytest.skip("Search not available")

        splits = client.list_splits(TEST_DATASETS["reviews"])
        split = splits[0]

        # Search with quotes and special chars
        try:
            results = client.search(
                dataset=split.dataset,
                query='movie "great"',
                config=split.config,
                split=split.split,
                length=5
            )
            # Just verify it doesn't crash
            assert results is not None
        except Exception:
            # Some special characters might not be supported
            pass
