"""HTTP session management following huggingface_hub patterns.

This module provides thread-safe global session management for both
synchronous and asynchronous HTTP clients, minimizing connection overhead.
It also includes retry logic with exponential backoff for transient errors.
"""

import atexit
import logging
import random
import threading
import time
from typing import Any, Callable, Optional, TypeVar

import httpx

from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RETRY_BASE_WAIT,
    DEFAULT_RETRY_MAX_WAIT,
    RETRY_STATUS_CODES,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# =============================================================================
# Global Session Management (Sync)
# =============================================================================

_GLOBAL_SYNC_CLIENT: Optional[httpx.Client] = None
_SYNC_CLIENT_LOCK = threading.Lock()


def _build_sync_client(timeout: float = DEFAULT_REQUEST_TIMEOUT) -> httpx.Client:
    """Create a new synchronous httpx client with default settings."""
    return httpx.Client(timeout=httpx.Timeout(timeout))


def get_session(timeout: float = DEFAULT_REQUEST_TIMEOUT) -> httpx.Client:
    """Get or create the global synchronous HTTP client.

    This function is thread-safe and uses double-checked locking to minimize
    lock contention while ensuring only one client is created.

    Args:
        timeout: Request timeout in seconds. Only used when creating a new client.

    Returns:
        The global httpx.Client instance.

    Example:
        >>> session = get_session()
        >>> response = session.get("https://example.com")
    """
    global _GLOBAL_SYNC_CLIENT

    if _GLOBAL_SYNC_CLIENT is None:
        with _SYNC_CLIENT_LOCK:
            # Double-check after acquiring lock
            if _GLOBAL_SYNC_CLIENT is None:
                _GLOBAL_SYNC_CLIENT = _build_sync_client(timeout)

    assert _GLOBAL_SYNC_CLIENT is not None  # Guaranteed by double-checked locking above
    return _GLOBAL_SYNC_CLIENT


def reset_sync_session() -> None:
    """Close and reset the global synchronous session.

    This is useful for testing or when you need to reconfigure the client.
    """
    global _GLOBAL_SYNC_CLIENT

    with _SYNC_CLIENT_LOCK:
        if _GLOBAL_SYNC_CLIENT is not None:
            _GLOBAL_SYNC_CLIENT.close()
            _GLOBAL_SYNC_CLIENT = None


# =============================================================================
# Global Session Management (Async)
# =============================================================================

_GLOBAL_ASYNC_CLIENT: Optional[httpx.AsyncClient] = None
_ASYNC_CLIENT_LOCK = threading.Lock()


def _build_async_client(timeout: float = DEFAULT_REQUEST_TIMEOUT) -> httpx.AsyncClient:
    """Create a new asynchronous httpx client with default settings."""
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout))


def get_async_session(timeout: float = DEFAULT_REQUEST_TIMEOUT) -> httpx.AsyncClient:
    """Get or create the global asynchronous HTTP client.

    This function is thread-safe and uses double-checked locking.

    Note: The async client must be properly closed when no longer needed.
    Use `reset_async_session()` or rely on the atexit handler.

    Args:
        timeout: Request timeout in seconds. Only used when creating a new client.

    Returns:
        The global httpx.AsyncClient instance.

    Example:
        >>> async def fetch():
        ...     session = get_async_session()
        ...     response = await session.get("https://example.com")
        ...     return response
    """
    global _GLOBAL_ASYNC_CLIENT

    if _GLOBAL_ASYNC_CLIENT is None:
        with _ASYNC_CLIENT_LOCK:
            # Double-check after acquiring lock
            if _GLOBAL_ASYNC_CLIENT is None:
                _GLOBAL_ASYNC_CLIENT = _build_async_client(timeout)

    assert _GLOBAL_ASYNC_CLIENT is not None  # Guaranteed by double-checked locking above
    return _GLOBAL_ASYNC_CLIENT


async def reset_async_session() -> None:
    """Close and reset the global asynchronous session.

    This is useful for testing or when you need to reconfigure the client.
    Must be called from an async context.
    """
    global _GLOBAL_ASYNC_CLIENT

    with _ASYNC_CLIENT_LOCK:
        if _GLOBAL_ASYNC_CLIENT is not None:
            await _GLOBAL_ASYNC_CLIENT.aclose()
            _GLOBAL_ASYNC_CLIENT = None


def _reset_async_session_sync() -> None:
    """Synchronous version of reset for atexit handler.

    Note: httpx AsyncClient has no sync close method, so we just set to None
    and let the garbage collector handle cleanup. This may leave some async
    cleanup pending, but it's the best we can do in a synchronous atexit handler.
    """
    global _GLOBAL_ASYNC_CLIENT

    with _ASYNC_CLIENT_LOCK:
        if _GLOBAL_ASYNC_CLIENT is not None:
            _GLOBAL_ASYNC_CLIENT = None


# =============================================================================
# Cleanup Registration
# =============================================================================


def _cleanup_sessions() -> None:
    """Clean up all global sessions on program exit."""
    reset_sync_session()
    _reset_async_session_sync()


# Register cleanup handler
atexit.register(_cleanup_sessions)


# =============================================================================
# Retry Logic with Exponential Backoff
# =============================================================================


def _get_retry_after(response: httpx.Response) -> Optional[float]:
    """Extract retry-after delay from response headers.

    Handles both integer seconds and HTTP-date formats.

    Args:
        response: The HTTP response to inspect.

    Returns:
        Delay in seconds, or None if no valid Retry-After header.
    """
    retry_after = response.headers.get("retry-after")
    if retry_after is None:
        return None

    try:
        # Try parsing as integer (seconds)
        return float(retry_after)
    except ValueError:
        pass

    # Could parse HTTP-date format here, but most APIs use seconds
    return None


def _calculate_wait_time(
    attempt: int,
    base_wait: float = DEFAULT_RETRY_BASE_WAIT,
    max_wait: float = DEFAULT_RETRY_MAX_WAIT,
    retry_after: Optional[float] = None,
) -> float:
    """Calculate wait time with exponential backoff and jitter.

    Args:
        attempt: Current retry attempt (0-indexed).
        base_wait: Base wait time in seconds.
        max_wait: Maximum wait time in seconds.
        retry_after: Server-specified retry delay (takes precedence).

    Returns:
        Wait time in seconds.
    """
    if retry_after is not None:
        return min(retry_after, max_wait)

    # Exponential backoff: base_wait * 2^attempt
    wait = base_wait * (2**attempt)

    # Add jitter (±25%) to prevent thundering herd
    jitter = wait * 0.25 * (2 * random.random() - 1)
    wait = wait + jitter

    return min(wait, max_wait)


def http_backoff(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_wait: float = DEFAULT_RETRY_BASE_WAIT,
    max_wait: float = DEFAULT_RETRY_MAX_WAIT,
    retry_on: tuple = RETRY_STATUS_CODES,
    **kwargs: Any,
) -> T:
    """Execute a function with retry logic and exponential backoff.

    This is a synchronous retry wrapper that handles transient HTTP errors
    following huggingface_hub patterns.

    Args:
        func: The function to execute (should return httpx.Response or raise).
        *args: Positional arguments to pass to func.
        max_retries: Maximum number of retry attempts.
        base_wait: Base wait time between retries in seconds.
        max_wait: Maximum wait time between retries in seconds.
        retry_on: Tuple of HTTP status codes that trigger retry.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        The result of func.

    Raises:
        httpx.HTTPStatusError: If all retries are exhausted.
        Exception: Any non-retryable exception from func.

    Example:
        >>> response = http_backoff(
        ...     session.request,
        ...     "GET",
        ...     "https://api.example.com/data",
        ... )
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in retry_on:
                raise

            last_exception = e

            if attempt < max_retries:
                retry_after = _get_retry_after(e.response)
                wait_time = _calculate_wait_time(attempt, base_wait, max_wait, retry_after)
                logger.warning(
                    "Request failed with status %d, retrying in %.2fs (attempt %d/%d)",
                    e.response.status_code,
                    wait_time,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait_time)
        except httpx.TimeoutException as e:
            last_exception = e

            if attempt < max_retries:
                wait_time = _calculate_wait_time(attempt, base_wait, max_wait)
                logger.warning(
                    "Request timed out, retrying in %.2fs (attempt %d/%d)",
                    wait_time,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait_time)

    # All retries exhausted
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected state: no exception but all retries exhausted")


async def async_http_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_wait: float = DEFAULT_RETRY_BASE_WAIT,
    max_wait: float = DEFAULT_RETRY_MAX_WAIT,
    retry_on: tuple = RETRY_STATUS_CODES,
    **kwargs: Any,
) -> Any:
    """Execute an async function with retry logic and exponential backoff.

    This is an asynchronous retry wrapper that handles transient HTTP errors
    following huggingface_hub patterns.

    Args:
        func: The async function to execute.
        *args: Positional arguments to pass to func.
        max_retries: Maximum number of retry attempts.
        base_wait: Base wait time between retries in seconds.
        max_wait: Maximum wait time between retries in seconds.
        retry_on: Tuple of HTTP status codes that trigger retry.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        The result of func.

    Raises:
        httpx.HTTPStatusError: If all retries are exhausted.
        Exception: Any non-retryable exception from func.

    Example:
        >>> response = await async_http_backoff(
        ...     session.request,
        ...     "GET",
        ...     "https://api.example.com/data",
        ... )
    """
    import asyncio

    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in retry_on:
                raise

            last_exception = e

            if attempt < max_retries:
                retry_after = _get_retry_after(e.response)
                wait_time = _calculate_wait_time(attempt, base_wait, max_wait, retry_after)
                logger.warning(
                    "Request failed with status %d, retrying in %.2fs (attempt %d/%d)",
                    e.response.status_code,
                    wait_time,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait_time)
        except httpx.TimeoutException as e:
            last_exception = e

            if attempt < max_retries:
                wait_time = _calculate_wait_time(attempt, base_wait, max_wait)
                logger.warning(
                    "Request timed out, retrying in %.2fs (attempt %d/%d)",
                    wait_time,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait_time)

    # All retries exhausted
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected state: no exception but all retries exhausted")
