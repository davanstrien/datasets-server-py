"""Header building utilities following huggingface_hub patterns.

This module provides functions for building HTTP headers with rich user-agent
information for API requests.
"""

import platform
import sys
from typing import Dict, Optional

from .__version__ import __version__

# Library identification
LIBRARY_NAME = "datasets-server-py"


def build_hf_headers(
    *,
    token: Optional[str] = None,
    library_name: str = LIBRARY_NAME,
    library_version: Optional[str] = None,
) -> Dict[str, str]:
    """Build HTTP headers following huggingface_hub conventions.

    Creates headers with a rich user-agent string that includes library name,
    version, Python version, and platform information. This helps with debugging
    and analytics on the server side.

    Args:
        token: Optional HuggingFace API token for authorization.
        library_name: Name of the library making the request.
        library_version: Version of the library. Defaults to package version.

    Returns:
        Dictionary of HTTP headers.

    Example:
        >>> headers = build_hf_headers(token="hf_xxx")
        >>> print(headers["user-agent"])
        datasets-server-py/0.1.0; python/3.10.12; platform/linux
    """
    if library_version is None:
        library_version = __version__

    # Build user-agent string with detailed environment info
    python_version = ".".join(map(str, sys.version_info[:3]))
    os_name = platform.system().lower()

    user_agent = f"{library_name}/{library_version}; python/{python_version}; {os_name}/{platform.release()}"

    headers = {"user-agent": user_agent}

    if token:
        headers["authorization"] = f"Bearer {token}"

    return headers


def get_token_to_send(token: Optional[str] = None) -> Optional[str]:
    """Get the token to send with requests.

    Follows huggingface_hub pattern of checking explicit token first,
    then falling back to cached token.

    Args:
        token: Explicitly provided token. If None, attempts to get cached token.

    Returns:
        Token string if available, None otherwise.
    """
    if token is not None:
        return token

    # Try to get cached token from huggingface_hub
    # Intentionally broad except: token retrieval failures (import errors, API changes)
    # should never prevent requests from being made (matches huggingface_hub pattern)
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        pass

    return None
