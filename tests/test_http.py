"""Tests for HTTP utilities including retry logic."""

import pytest
import httpx

from datasets_server._http import (
    _calculate_wait_time,
    _get_retry_after,
    http_backoff,
    async_http_backoff,
)


class TestGetRetryAfter:
    """Tests for _get_retry_after function."""

    def test_no_header(self):
        """Test when no Retry-After header is present."""
        response = httpx.Response(429)
        assert _get_retry_after(response) is None

    def test_integer_seconds(self):
        """Test parsing integer seconds format."""
        response = httpx.Response(429, headers={"retry-after": "5"})
        assert _get_retry_after(response) == 5.0

    def test_float_seconds(self):
        """Test parsing float seconds format."""
        response = httpx.Response(429, headers={"retry-after": "2.5"})
        assert _get_retry_after(response) == 2.5

    def test_invalid_format(self):
        """Test handling of invalid format."""
        response = httpx.Response(429, headers={"retry-after": "invalid"})
        assert _get_retry_after(response) is None


class TestCalculateWaitTime:
    """Tests for _calculate_wait_time function."""

    def test_first_attempt(self):
        """Test wait time for first attempt (attempt 0)."""
        wait = _calculate_wait_time(0, base_wait=1.0, max_wait=8.0)
        # Should be around 1.0 ± 25% jitter
        assert 0.75 <= wait <= 1.25

    def test_exponential_growth(self):
        """Test that wait time grows exponentially."""
        wait0 = _calculate_wait_time(0, base_wait=1.0, max_wait=100.0)
        wait1 = _calculate_wait_time(1, base_wait=1.0, max_wait=100.0)
        wait2 = _calculate_wait_time(2, base_wait=1.0, max_wait=100.0)

        # Each should roughly double (accounting for jitter)
        assert 0.5 <= wait0 <= 1.5
        assert 1.5 <= wait1 <= 2.5
        assert 3.0 <= wait2 <= 5.0

    def test_max_wait_cap(self):
        """Test that wait time is capped at max_wait."""
        wait = _calculate_wait_time(10, base_wait=1.0, max_wait=8.0)
        assert wait <= 8.0

    def test_retry_after_takes_precedence(self):
        """Test that server-specified retry_after takes precedence."""
        wait = _calculate_wait_time(0, base_wait=1.0, max_wait=8.0, retry_after=5.0)
        assert wait == 5.0

    def test_retry_after_capped_by_max(self):
        """Test that retry_after is still capped by max_wait."""
        wait = _calculate_wait_time(0, base_wait=1.0, max_wait=3.0, retry_after=5.0)
        assert wait == 3.0


class TestHttpBackoff:
    """Tests for synchronous http_backoff function."""

    def test_success_on_first_try(self):
        """Test successful call without retries."""
        call_count = 0

        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = http_backoff(success_func, max_retries=3)
        assert result == "success"
        assert call_count == 1

    def test_retry_on_500(self):
        """Test retry on 500 status code."""
        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                response = httpx.Response(500)
                raise httpx.HTTPStatusError(
                    "Server error", request=httpx.Request("GET", "http://test"), response=response
                )
            return "success"

        result = http_backoff(flaky_func, max_retries=3, base_wait=0.01, max_wait=0.1)
        assert result == "success"
        assert call_count == 3

    def test_retry_on_429(self):
        """Test retry on rate limit (429) with Retry-After header."""
        call_count = 0

        def rate_limited_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                response = httpx.Response(429, headers={"retry-after": "0.01"})
                raise httpx.HTTPStatusError(
                    "Rate limited", request=httpx.Request("GET", "http://test"), response=response
                )
            return "success"

        result = http_backoff(rate_limited_func, max_retries=3, base_wait=0.01, max_wait=0.1)
        assert result == "success"
        assert call_count == 2

    def test_no_retry_on_404(self):
        """Test that 404 errors are not retried."""
        call_count = 0

        def not_found_func():
            nonlocal call_count
            call_count += 1
            response = httpx.Response(404)
            raise httpx.HTTPStatusError(
                "Not found", request=httpx.Request("GET", "http://test"), response=response
            )

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            http_backoff(not_found_func, max_retries=3)

        assert exc_info.value.response.status_code == 404
        assert call_count == 1  # No retries

    def test_exhausted_retries(self):
        """Test that exception is raised after all retries exhausted."""
        call_count = 0

        def always_fails():
            nonlocal call_count
            call_count += 1
            response = httpx.Response(503)
            raise httpx.HTTPStatusError(
                "Service unavailable", request=httpx.Request("GET", "http://test"), response=response
            )

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            http_backoff(always_fails, max_retries=2, base_wait=0.01, max_wait=0.1)

        assert exc_info.value.response.status_code == 503
        assert call_count == 3  # Initial + 2 retries

    def test_retry_on_timeout(self):
        """Test retry on timeout exceptions."""
        call_count = 0

        def timeout_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.TimeoutException("Timeout")
            return "success"

        result = http_backoff(timeout_func, max_retries=3, base_wait=0.01, max_wait=0.1)
        assert result == "success"
        assert call_count == 2


class TestAsyncHttpBackoff:
    """Tests for asynchronous async_http_backoff function."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """Test successful call without retries."""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await async_http_backoff(success_func, max_retries=3)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_500(self):
        """Test retry on 500 status code."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                response = httpx.Response(500)
                raise httpx.HTTPStatusError(
                    "Server error", request=httpx.Request("GET", "http://test"), response=response
                )
            return "success"

        result = await async_http_backoff(flaky_func, max_retries=3, base_wait=0.01, max_wait=0.1)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_404(self):
        """Test that 404 errors are not retried."""
        call_count = 0

        async def not_found_func():
            nonlocal call_count
            call_count += 1
            response = httpx.Response(404)
            raise httpx.HTTPStatusError(
                "Not found", request=httpx.Request("GET", "http://test"), response=response
            )

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await async_http_backoff(not_found_func, max_retries=3)

        assert exc_info.value.response.status_code == 404
        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        """Test that exception is raised after all retries exhausted."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            response = httpx.Response(503)
            raise httpx.HTTPStatusError(
                "Service unavailable", request=httpx.Request("GET", "http://test"), response=response
            )

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await async_http_backoff(always_fails, max_retries=2, base_wait=0.01, max_wait=0.1)

        assert exc_info.value.response.status_code == 503
        assert call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry on timeout exceptions."""
        call_count = 0

        async def timeout_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.TimeoutException("Timeout")
            return "success"

        result = await async_http_backoff(timeout_func, max_retries=3, base_wait=0.01, max_wait=0.1)
        assert result == "success"
        assert call_count == 2
