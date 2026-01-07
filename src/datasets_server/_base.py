"""Base client functionality shared between sync and async clients."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from huggingface_hub import get_token

from ._headers import build_hf_headers
from .constants import DEFAULT_REQUEST_TIMEOUT, HF_DATASETS_SERVER_ENDPOINT


class BaseClient(ABC):
    """Base class for sync and async clients."""

    def __init__(
        self,
        token: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        """Initialize base client.

        Args:
            token: Optional HuggingFace API token. If not provided, will attempt to use cached token.
            endpoint: Optional API endpoint URL. Defaults to HF_DATASETS_SERVER_ENDPOINT.
            timeout: Request timeout in seconds. Defaults to DEFAULT_REQUEST_TIMEOUT.
        """
        self.endpoint = endpoint or HF_DATASETS_SERVER_ENDPOINT
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
        """Build request headers with rich user-agent."""
        return build_hf_headers(token=self.token)

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
