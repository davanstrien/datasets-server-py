"""Tests for header building utilities."""

import platform
import sys
from unittest.mock import patch

from datasets_server.__version__ import __version__
from datasets_server._headers import LIBRARY_NAME, build_hf_headers, get_token_to_send


class TestBuildHfHeaders:
    """Tests for build_hf_headers function."""

    def test_basic_headers(self):
        """Test basic header generation without token."""
        headers = build_hf_headers()

        assert "user-agent" in headers
        assert "authorization" not in headers

        # Check user-agent format
        user_agent = headers["user-agent"]
        assert LIBRARY_NAME in user_agent
        assert __version__ in user_agent
        assert "python" in user_agent

    def test_headers_with_token(self):
        """Test header generation with token."""
        headers = build_hf_headers(token="hf_test_token")

        assert headers["authorization"] == "Bearer hf_test_token"
        assert "user-agent" in headers

    def test_custom_library_name(self):
        """Test custom library name in user-agent."""
        headers = build_hf_headers(library_name="custom-lib")

        assert "custom-lib" in headers["user-agent"]
        assert LIBRARY_NAME not in headers["user-agent"]

    def test_custom_library_version(self):
        """Test custom library version in user-agent."""
        headers = build_hf_headers(library_version="9.9.9")

        assert "9.9.9" in headers["user-agent"]

    def test_user_agent_contains_python_version(self):
        """Test that user-agent includes Python version."""
        headers = build_hf_headers()
        python_version = ".".join(map(str, sys.version_info[:3]))

        assert python_version in headers["user-agent"]

    def test_user_agent_contains_platform(self):
        """Test that user-agent includes platform info."""
        headers = build_hf_headers()
        os_name = platform.system().lower()

        assert os_name in headers["user-agent"]


class TestGetTokenToSend:
    """Tests for get_token_to_send function."""

    def test_explicit_token(self):
        """Test that explicit token is returned."""
        token = get_token_to_send(token="explicit_token")
        assert token == "explicit_token"

    def test_explicit_token_none(self):
        """Test fallback when token is None."""
        with patch("huggingface_hub.HfFolder") as mock_folder:
            mock_folder.get_token.return_value = "cached_token"
            token = get_token_to_send(token=None)
            assert token == "cached_token"

    def test_no_cached_token(self):
        """Test when no cached token is available."""
        with patch("huggingface_hub.HfFolder") as mock_folder:
            mock_folder.get_token.return_value = None
            token = get_token_to_send(token=None)
            assert token is None

    def test_exception_handled(self):
        """Test graceful handling of exceptions."""
        with patch("huggingface_hub.HfFolder") as mock_folder:
            mock_folder.get_token.side_effect = Exception("Test error")
            # Should not raise, just return None
            token = get_token_to_send(token=None)
            assert token is None
