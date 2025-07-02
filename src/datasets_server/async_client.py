"""Asynchronous client for the Datasets Server API."""

from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

from ._base import BaseClient
from .exceptions import DatasetNotFoundError, DatasetServerError
from .models import (
    DatasetInfo,
    DatasetRows,
    DatasetSize,
    DatasetSplit,
    DatasetStatistics,
    DatasetValidity,
    ParquetFile,
)


class AsyncDatasetsServerClient(BaseClient):
    """Asynchronous client for the Hugging Face Datasets Viewer API.

    Examples:
        >>> import asyncio
        >>> from datasets_server import AsyncDatasetsServerClient
        >>>
        >>> async def main():
        ...     async with AsyncDatasetsServerClient() as client:
        ...         validity = await client.is_valid("squad")
        ...         if validity.preview:
        ...             rows = await client.preview("squad")
        >>>
        >>> asyncio.run(main())
    """

    MAX_ROWS_PER_REQUEST = 100  # Maximum rows allowed per API request

    def __init__(
        self,
        token: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize the asynchronous client.

        Args:
            token: Optional HuggingFace API token. If not provided, will attempt to use cached token.
            endpoint: Optional API endpoint URL. Defaults to official endpoint.
            timeout: Request timeout in seconds. Defaults to 30.0.
        """
        super().__init__(token, endpoint, timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    async def _ensure_session(self):
        """Ensure session exists."""
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

    async def _request(self, method: str, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            params: Optional query parameters

        Returns:
            Parsed JSON response

        Raises:
            DatasetNotFoundError: If dataset is not found (404)
            DatasetServerError: For other API errors
        """
        await self._ensure_session()
        url = f"{self.endpoint}{path}"

        try:
            async with self._session.request(method=method, url=url, params=params, headers=self.headers) as response:
                if response.status == 404:
                    raise DatasetNotFoundError(f"Dataset not found: {params.get('dataset', 'unknown')}")
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            raise DatasetServerError(f"API error: {e}") from e
        except DatasetNotFoundError:
            raise
        except Exception as e:
            raise DatasetServerError(f"Request failed: {e}") from e

    # Async API methods
    async def is_valid(self, dataset: str) -> DatasetValidity:
        """Check if a dataset is valid and which features are available.

        Args:
            dataset: The dataset repository ID (e.g., "squad" or "user/dataset").

        Returns:
            DatasetValidity object with boolean flags for each feature.

        Raises:
            DatasetNotFoundError: If the dataset doesn't exist.
            DatasetServerError: If the API returns an error.
        """
        data = await self._request("GET", "/is-valid", {"dataset": dataset})
        return DatasetValidity(**data)

    async def list_splits(self, dataset: str) -> List[DatasetSplit]:
        """List all configurations and splits for a dataset.

        Args:
            dataset: The dataset repository ID.

        Returns:
            List of DatasetSplit objects.
        """
        data = await self._request("GET", "/splits", {"dataset": dataset})
        return [DatasetSplit(**split) for split in data.get("splits", [])]

    async def get_info(self, dataset: str, config: Optional[str] = None) -> DatasetInfo:
        """Get detailed information about a dataset.

        Args:
            dataset: The dataset repository ID.
            config: Optional dataset configuration name.

        Returns:
            DatasetInfo object with metadata.
        """
        params = {"dataset": dataset}
        if config:
            params["config"] = config
        data = await self._request("GET", "/info", params)
        return DatasetInfo(**data)

    async def get_size(self, dataset: str) -> DatasetSize:
        """Get size information for a dataset.

        Args:
            dataset: The dataset repository ID.

        Returns:
            DatasetSize object with size details.
        """
        data = await self._request("GET", "/size", {"dataset": dataset})
        return DatasetSize(**data)

    async def list_parquet_files(self, dataset: str) -> List[ParquetFile]:
        """List Parquet files for a dataset.

        Args:
            dataset: The dataset repository ID.

        Returns:
            List of ParquetFile objects.
        """
        data = await self._request("GET", "/parquet", {"dataset": dataset})
        return [ParquetFile(**f) for f in data.get("parquet_files", [])]

    async def preview(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
    ) -> DatasetRows:
        """Preview the first 100 rows of a dataset.

        Args:
            dataset: The dataset repository ID.
            config: Optional dataset configuration name.
            split: Optional dataset split name.

        Returns:
            DatasetRows object with features and row data.
        """
        params = {"dataset": dataset}
        if config:
            params["config"] = config
        if split:
            params["split"] = split
        data = await self._request("GET", "/first-rows", params)
        return DatasetRows(**data)

    async def get_rows(
        self,
        dataset: str,
        config: str,
        split: str,
        offset: int = 0,
        length: int = 100,
    ) -> DatasetRows:
        """Get rows from a dataset with pagination.

        Args:
            dataset: The dataset repository ID.
            config: The dataset configuration name.
            split: The dataset split name.
            offset: Starting row index.
            length: Number of rows to retrieve (max MAX_ROWS_PER_REQUEST).

        Returns:
            DatasetRows object with features and row data.

        Raises:
            ValueError: If length exceeds MAX_ROWS_PER_REQUEST.
        """
        if length > self.MAX_ROWS_PER_REQUEST:
            raise ValueError(f"Length cannot exceed {self.MAX_ROWS_PER_REQUEST} rows per request")

        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
        data = await self._request("GET", "/rows", params)
        return DatasetRows(**data)

    async def iter_rows(
        self,
        dataset: str,
        config: str,
        split: str,
        batch_size: int = 100,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Iterate through all rows in a dataset.

        Args:
            dataset: The dataset repository ID.
            config: The dataset configuration name.
            split: The dataset split name.
            batch_size: Number of rows to fetch per request (max 100).

        Yields:
            Individual row dictionaries.
        """
        offset = 0
        while True:
            batch = await self.get_rows(dataset, config, split, offset, batch_size)
            if not batch.rows:
                break

            for row in batch.rows:
                yield row["row"]

            if len(batch.rows) < batch_size:
                break

            offset += batch_size

    async def search(
        self,
        dataset: str,
        query: str,
        config: str,
        split: str,
        offset: int = 0,
        length: int = 100,
    ) -> DatasetRows:
        """Search for text in a dataset.

        Args:
            dataset: The dataset repository ID.
            query: Text to search for.
            config: The dataset configuration name.
            split: The dataset split name.
            offset: Starting row index.
            length: Number of rows to retrieve (max MAX_ROWS_PER_REQUEST).

        Returns:
            DatasetRows object with matching rows.
        """
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "query": query,
            "offset": offset,
            "length": length,
        }
        data = await self._request("GET", "/search", params)
        return DatasetRows(**data)

    async def filter(
        self,
        dataset: str,
        where: str,
        config: str,
        split: str,
        orderby: Optional[str] = None,
        offset: int = 0,
        length: int = 100,
    ) -> DatasetRows:
        """Filter rows in a dataset using SQL-like conditions.

        Args:
            dataset: The dataset repository ID.
            where: SQL-like WHERE clause (e.g., '"column" > 5').
            config: The dataset configuration name.
            split: The dataset split name.
            orderby: Optional column to order results by.
            offset: Starting row index.
            length: Number of rows to retrieve (max MAX_ROWS_PER_REQUEST).

        Returns:
            DatasetRows object with filtered rows.
        """
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "where": where,
            "offset": offset,
            "length": length,
        }
        if orderby:
            params["orderby"] = orderby

        data = await self._request("GET", "/filter", params)
        return DatasetRows(**data)

    async def get_statistics(
        self,
        dataset: str,
        config: str,
        split: str,
    ) -> DatasetStatistics:
        """Get statistical information about a dataset.

        Args:
            dataset: The dataset repository ID.
            config: The dataset configuration name.
            split: The dataset split name.

        Returns:
            DatasetStatistics object with column statistics.
        """
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
        }
        data = await self._request("GET", "/statistics", params)
        return DatasetStatistics(**data)

    async def sample_rows(
        self,
        dataset: str,
        config: str,
        split: str,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> DatasetRows:
        """Get a random sample of rows from a dataset.

        Args:
            dataset: The dataset repository ID.
            config: The dataset configuration name.
            split: The dataset split name.
            n_samples: Number of rows to sample.
            seed: Optional random seed for reproducibility.

        Returns:
            DatasetRows object with sampled rows.

        Raises:
            ValueError: If n_samples is negative or exceeds available rows.
            DatasetServerError: If unable to fetch dataset information.
        """
        import random

        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")

        if n_samples == 0:
            # Return empty DatasetRows
            return DatasetRows(features=[], rows=[], num_rows_total=0)

        # Get dataset info to determine total number of rows
        info = await self.get_info(dataset, config)

        # Find the split info to get the number of rows
        total_rows = None
        if "dataset_info" in info.model_dump() and "splits" in info.dataset_info:
            splits_data = info.dataset_info["splits"]
            if isinstance(splits_data, dict) and split in splits_data:
                total_rows = splits_data[split]["num_examples"]

        if total_rows is None:
            # Try alternative approach: get first batch to determine structure
            first_batch = await self.get_rows(dataset, config, split, offset=0, length=1)
            if first_batch.num_rows_total is not None:
                total_rows = first_batch.num_rows_total
            else:
                # If we still don't know total rows, we'll have to estimate
                raise DatasetServerError(
                    f"Unable to determine total number of rows for dataset {dataset}/{config}/{split}"
                )

        if n_samples > total_rows:
            raise ValueError(
                f"Requested {n_samples} samples but dataset only has {total_rows} rows"
            )

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        # Generate unique random indices
        indices = sorted(random.sample(range(total_rows), n_samples))

        # Group indices into batches based on MAX_ROWS_PER_REQUEST
        sampled_rows = []
        features = None

        i = 0
        while i < len(indices):
            # Find consecutive indices that can be fetched in one request
            batch_start = indices[i]
            batch_indices = [indices[i]]

            j = i + 1
            while j < len(indices) and indices[j] - batch_start < self.MAX_ROWS_PER_REQUEST:
                batch_indices.append(indices[j])
                j += 1

            # Fetch this batch
            offset = batch_start
            # Calculate length to cover all indices in this batch
            if len(batch_indices) == 1:
                length = 1
            else:
                length = min(self.MAX_ROWS_PER_REQUEST, batch_indices[-1] - batch_start + 1)

            batch = await self.get_rows(dataset, config, split, offset=offset, length=length)

            if features is None:
                features = batch.features

            # Extract only the rows at our sampled indices
            for idx in batch_indices:
                row_idx = idx - batch_start
                if row_idx < len(batch.rows):
                    sampled_rows.append(batch.rows[row_idx])

            i = j

        return DatasetRows(
            features=features or [],
            rows=sampled_rows,
            num_rows_total=total_rows,
            partial=False,
        )

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
