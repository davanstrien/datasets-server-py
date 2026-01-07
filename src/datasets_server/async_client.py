"""Asynchronous client for the Datasets Server API."""

from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from ._base import BaseClient
from ._http import async_http_backoff, get_async_session
from .constants import DEFAULT_REQUEST_TIMEOUT, MAX_ROWS_PER_REQUEST, RETRY_STATUS_CODES
from .exceptions import DatasetNotFoundError, DatasetServerError, DatasetServerHTTPError
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

    By default, uses a global shared async HTTP session for efficiency. When used
    as an async context manager, creates a dedicated session that is closed on exit.

    Examples:
        >>> import asyncio
        >>> from datasets_server import AsyncDatasetsServerClient
        >>>
        >>> async def main():
        ...     # Using global session (recommended for most cases)
        ...     client = AsyncDatasetsServerClient()
        ...     validity = await client.is_valid("squad")
        ...
        ...     # Or with dedicated session
        ...     async with AsyncDatasetsServerClient() as client:
        ...         rows = await client.preview("squad")
        >>>
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        token: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        """Initialize the asynchronous client.

        Args:
            token: Optional HuggingFace API token. If not provided, will attempt to use cached token.
            endpoint: Optional API endpoint URL. Defaults to HF_DATASETS_SERVER_ENDPOINT.
            timeout: Request timeout in seconds. Defaults to DEFAULT_REQUEST_TIMEOUT.
        """
        super().__init__(token, endpoint, timeout)
        self._session: Optional[httpx.AsyncClient] = None
        self._owns_session = False

    @property
    def session(self) -> httpx.AsyncClient:
        """Get the async HTTP session, using global shared session by default."""
        if self._session is not None:
            return self._session
        return get_async_session(self.timeout)

    async def __aenter__(self) -> "AsyncDatasetsServerClient":
        """Async context manager entry - creates a dedicated session."""
        self._session = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        self._owns_session = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - closes dedicated session."""
        if self._owns_session and self._session is not None:
            await self._session.aclose()
            self._session = None
            self._owns_session = False

    async def _ensure_session(self) -> None:
        """Ensure session exists (deprecated, kept for compatibility)."""
        # This method is now a no-op since we use the session property
        pass

    async def _request(self, method: str, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async HTTP request to the API with automatic retry for transient errors.

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
        url = f"{self.endpoint}{path}"

        async def _do_request() -> httpx.Response:
            response = await self.session.request(
                method=method, url=url, params=params, headers=self.headers
            )
            response.raise_for_status()
            return response

        try:
            # Use async_http_backoff for automatic retry on transient errors
            response = await async_http_backoff(_do_request, retry_on=RETRY_STATUS_CODES)
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                dataset_name = params.get("dataset", "unknown") if params else "unknown"
                raise DatasetNotFoundError(
                    f"Dataset not found: {dataset_name}",
                    response=e.response,
                ) from e
            raise DatasetServerHTTPError(f"API error: {e}", response=e.response) from e
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
        if length > MAX_ROWS_PER_REQUEST:
            raise ValueError(f"Length cannot exceed {MAX_ROWS_PER_REQUEST} rows per request")

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
        max_requests: Optional[int] = None,
    ) -> DatasetRows:
        """Get a random sample of rows from a dataset.

        Args:
            dataset: The dataset repository ID.
            config: The dataset configuration name.
            split: The dataset split name.
            n_samples: Number of rows to sample.
            seed: Optional random seed for reproducibility.
            max_requests: Optional maximum number of API requests. If specified,
                sampling will be less random but more API-efficient. Default None
                uses true random sampling.

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

        # If max_requests is specified, use request-limited sampling
        if max_requests is not None and max_requests > 0:
            return await self._sample_rows_limited(
                dataset, config, split, n_samples, total_rows, max_requests
            )

        # Otherwise, use true random sampling
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
            while j < len(indices) and indices[j] - batch_start < MAX_ROWS_PER_REQUEST:
                batch_indices.append(indices[j])
                j += 1

            # Fetch this batch
            offset = batch_start
            # Calculate length to cover all indices in this batch
            if len(batch_indices) == 1:
                length = 1
            else:
                length = min(MAX_ROWS_PER_REQUEST, batch_indices[-1] - batch_start + 1)

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

    async def _sample_rows_limited(
        self,
        dataset: str,
        config: str,
        split: str,
        n_samples: int,
        total_rows: int,
        max_requests: int,
    ) -> DatasetRows:
        """Sample rows with limited API requests.

        This method divides the dataset into segments and fetches rows from
        random offsets within each segment, then samples from the collected rows.
        """
        import random

        # Calculate segment size and samples per segment
        segment_size = total_rows // max_requests
        base_samples_per_segment = n_samples // max_requests
        extra_samples = n_samples % max_requests

        all_rows = []
        features = None

        for i in range(max_requests):
            # Calculate segment boundaries
            segment_start = i * segment_size
            if i == max_requests - 1:
                # Last segment includes any remaining rows
                segment_end = total_rows
            else:
                segment_end = (i + 1) * segment_size

            # Calculate samples for this segment
            samples_from_segment = base_samples_per_segment
            if i < extra_samples:
                samples_from_segment += 1

            if samples_from_segment == 0:
                continue

            # Pick random offset within segment
            max_offset = segment_end - min(MAX_ROWS_PER_REQUEST, segment_end - segment_start)
            offset = random.randint(segment_start, max(segment_start, max_offset))

            # Fetch rows from this offset
            length = min(MAX_ROWS_PER_REQUEST, segment_end - offset)
            batch = await self.get_rows(dataset, config, split, offset=offset, length=length)

            if features is None:
                features = batch.features

            # Collect all rows from this batch
            all_rows.extend(batch.rows)

        # Sample from collected rows
        if len(all_rows) < n_samples:
            # If we couldn't fetch enough rows, return what we have
            sampled_rows = all_rows
        else:
            # Sample n_samples from all collected rows
            sampled_indices = random.sample(range(len(all_rows)), n_samples)
            sampled_rows = [all_rows[i] for i in sampled_indices]

        return DatasetRows(
            features=features or [],
            rows=sampled_rows,
            num_rows_total=total_rows,
            partial=False,
        )

    async def close(self):
        """Close the dedicated session if one exists.

        Note: This only closes sessions created via context manager or explicit
        creation. The global shared session is managed automatically.
        """
        if self._owns_session and self._session is not None:
            await self._session.aclose()
            self._session = None
            self._owns_session = False
