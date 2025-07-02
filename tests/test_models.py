"""Tests for response models."""

from datasets_server.models import (
    DatasetInfo,
    DatasetRows,
    DatasetSize,
    DatasetSplit,
    DatasetStatistics,
    DatasetValidity,
    ParquetFile,
)


class TestDatasetValidity:
    """Test DatasetValidity model."""

    def test_create_validity(self):
        """Test creating a DatasetValidity instance."""
        validity = DatasetValidity(
            viewer=True,
            preview=True,
            search=False,
            filter=False,
            statistics=True,
        )
        assert validity.viewer is True
        assert validity.preview is True
        assert validity.search is False
        assert validity.filter is False
        assert validity.statistics is True

    def test_validity_from_dict(self):
        """Test creating DatasetValidity from dictionary."""
        data = {
            "viewer": True,
            "preview": False,
            "search": True,
            "filter": True,
            "statistics": False,
        }
        validity = DatasetValidity(**data)
        assert validity.viewer is True
        assert validity.preview is False
        assert validity.search is True


class TestDatasetSplit:
    """Test DatasetSplit model."""

    def test_create_split(self):
        """Test creating a DatasetSplit instance."""
        split = DatasetSplit(
            dataset="squad",
            config="plain_text",
            split="train",
        )
        assert split.dataset == "squad"
        assert split.config == "plain_text"
        assert split.split == "train"


class TestParquetFile:
    """Test ParquetFile model."""

    def test_create_parquet_file(self):
        """Test creating a ParquetFile instance."""
        pf = ParquetFile(
            dataset="squad",
            config="plain_text",
            split="train",
            url="https://example.com/file.parquet",
            filename="file.parquet",
            size=1024,
        )
        assert pf.dataset == "squad"
        assert pf.size == 1024
        assert pf.url == "https://example.com/file.parquet"


class TestDatasetRows:
    """Test DatasetRows model."""

    def test_create_rows(self):
        """Test creating a DatasetRows instance."""
        rows = DatasetRows(
            features=[{"name": "text", "type": "string"}],
            rows=[{"row": {"text": "Hello"}}],
            num_rows_total=100,
            num_rows_per_page=10,
            partial=False,
        )
        assert len(rows.features) == 1
        assert len(rows.rows) == 1
        assert rows.num_rows_total == 100
        assert rows.partial is False

    def test_rows_with_defaults(self):
        """Test creating DatasetRows with default values."""
        rows = DatasetRows(
            features=[],
            rows=[],
        )
        assert rows.num_rows_total is None
        assert rows.num_rows_per_page is None
        assert rows.partial is False


class TestDatasetInfo:
    """Test DatasetInfo model."""

    def test_create_info(self):
        """Test creating a DatasetInfo instance."""
        info = DatasetInfo(
            dataset_info={"description": "Test dataset"},
            pending=["task1"],
            failed=["task2"],
            partial=True,
        )
        assert info.dataset_info["description"] == "Test dataset"
        assert len(info.pending) == 1
        assert len(info.failed) == 1
        assert info.partial is True

    def test_info_with_defaults(self):
        """Test creating DatasetInfo with default values."""
        info = DatasetInfo(dataset_info={})
        assert info.pending == []
        assert info.failed == []
        assert info.partial is False


class TestDatasetStatistics:
    """Test DatasetStatistics model."""

    def test_create_statistics(self):
        """Test creating a DatasetStatistics instance."""
        stats = DatasetStatistics(
            num_examples=1000,
            statistics=[{"column_name": "text", "column_type": "string"}],
            partial=False,
        )
        assert stats.num_examples == 1000
        assert len(stats.statistics) == 1
        assert stats.partial is False


class TestDatasetSizeAPIResponse:
    """Test DatasetSize model with actual API response structure."""

    def test_dataset_size_with_api_response(self):
        """Test DatasetSize with actual API response format."""
        # This is the actual structure returned by the API
        api_response = {
            "size": {
                "dataset": {
                    "dataset": "squad",
                    "num_bytes_original_files": 50000000,
                    "num_bytes_parquet_files": 40000000,
                    "num_rows": 87599,
                },
                "configs": [
                    {
                        "dataset": "squad",
                        "config": "plain_text",
                        "num_bytes_original_files": 50000000,
                        "num_bytes_parquet_files": 40000000,
                        "num_rows": 87599,
                    }
                ],
                "splits": [{"dataset": "squad", "config": "plain_text", "split": "train", "num_rows": 87599}],
            },
            "pending": [],
            "failed": [],
            "partial": False,
        }

        # This should not raise a validation error
        size = DatasetSize(**api_response)
        assert size.size["dataset"]["num_rows"] == 87599
        assert size.partial is False
        assert len(size.pending) == 0
