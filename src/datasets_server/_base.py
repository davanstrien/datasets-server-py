"""Base client functionality shared between sync and async clients."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from huggingface_hub import get_token

from .__version__ import __version__


class BaseClient(ABC):
    """Base class for sync and async clients."""

    DEFAULT_ENDPOINT = "https://datasets-server.huggingface.co"

    def __init__(
        self,
        token: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize base client.

        Args:
            token: Optional HuggingFace API token. If not provided, will attempt to use cached token.
            endpoint: Optional API endpoint URL. Defaults to official endpoint.
            timeout: Request timeout in seconds. Defaults to 30.0.
        """
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self.timeout = timeout
        self._token = token

    @property
    def token(self) -> Optional[str]:
        """Get the authentication token."""
        if self._token is not None:
            return self._token
        # Try to get from huggingface_hub
        try:
            return get_token()
        except Exception:
            return None

    @property
    def headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {"User-Agent": f"datasets-server-py/{__version__}"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    @abstractmethod
    def _request(self, method: str, path: str, params: Optional[Dict] = None) -> Any:
        """Make HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            params: Optional query parameters

        Returns:
            Parsed JSON response
        """
        pass
