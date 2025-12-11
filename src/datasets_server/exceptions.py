"""Custom exceptions for the Datasets Server API client.

This module follows the huggingface_hub exception patterns for consistency.
"""

from typing import Optional

import httpx


class DatasetServerError(Exception):
    """Base exception for all dataset server errors."""

    pass


class DatasetServerHTTPError(DatasetServerError, OSError):
    """HTTP error with response context for debugging.

    This exception captures detailed information about HTTP errors, following
    the huggingface_hub HfHubHTTPError pattern.

    Attributes:
        request_id: The x-request-id header from the response, if available.
        status_code: The HTTP status code from the response.
        response: The full httpx Response object for detailed inspection.
        server_message: Extracted error message from the API response.

    Example:
        >>> try:
        ...     client.is_valid("nonexistent-dataset")
        ... except DatasetServerHTTPError as e:
        ...     print(f"Status: {e.status_code}")
        ...     print(f"Request ID: {e.request_id}")
    """

    def __init__(
        self,
        message: str,
        *,
        response: Optional[httpx.Response] = None,
    ):
        """Initialize the HTTP error with response context.

        Args:
            message: Human-readable error message.
            response: Optional httpx Response object with error details.
        """
        self.response = response
        self.request_id: Optional[str] = None
        self.status_code: Optional[int] = None
        self.server_message: Optional[str] = None

        if response is not None:
            self.request_id = response.headers.get("x-request-id")
            self.status_code = response.status_code

            # Try to extract error message from JSON response
            # Intentionally broad except: JSON parsing failures should never
            # prevent error handling from completing (matches huggingface_hub pattern)
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    self.server_message = error_data.get("error") or error_data.get("message")
            except Exception:
                pass

        # Build enhanced message with context
        full_message = message
        if self.request_id:
            full_message = f"{full_message} (Request ID: {self.request_id})"

        super().__init__(full_message)

    def __str__(self) -> str:
        """Return string representation with context."""
        parts = [super().__str__()]

        if self.server_message:
            parts.append(f"Server message: {self.server_message}")

        return " - ".join(parts)


class DatasetNotFoundError(DatasetServerHTTPError):
    """Raised when a dataset is not found (HTTP 404).

    This exception is raised when attempting to access a dataset that
    doesn't exist or that the user doesn't have permission to access.
    """

    def __init__(
        self,
        message: str,
        *,
        response: Optional[httpx.Response] = None,
    ):
        """Initialize the not found error.

        Args:
            message: Human-readable error message.
            response: Optional httpx Response object with error details.
        """
        super().__init__(message, response=response)


class DatasetNotValidError(DatasetServerError):
    """Raised when attempting operations on invalid datasets."""

    pass


class DatasetServerTimeoutError(DatasetServerError):
    """Raised when API requests timeout."""

    pass
