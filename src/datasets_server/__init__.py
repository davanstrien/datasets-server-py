"""Python client for Hugging Face Datasets Viewer API with sync/async support."""

from .__version__ import __version__
from .async_client import AsyncDatasetsServerClient
from .client import DatasetsServerClient
from .exceptions import (
    DatasetNotFoundError,
    DatasetNotValidError,
    DatasetServerError,
    DatasetServerHTTPError,
    DatasetServerTimeoutError,
)
from .models import (
    DatasetInfo,
    DatasetRows,
    DatasetSize,
    DatasetSplit,
    DatasetStatistics,
    DatasetValidity,
    ParquetFile,
)

__all__ = [
    # Clients
    "DatasetsServerClient",
    "AsyncDatasetsServerClient",
    # Models
    "DatasetValidity",
    "DatasetSplit",
    "DatasetInfo",
    "DatasetRows",
    "ParquetFile",
    "DatasetSize",
    "DatasetStatistics",
    # Exceptions
    "DatasetServerError",
    "DatasetServerHTTPError",
    "DatasetNotFoundError",
    "DatasetNotValidError",
    "DatasetServerTimeoutError",
    # Version
    "__version__",
]
