"""Constants for the datasets-server client.

This module centralizes configuration values following huggingface_hub patterns.
Environment variables can be used to override defaults.
"""

import os

# =============================================================================
# Endpoint Configuration
# =============================================================================

DEFAULT_DATASETS_SERVER_ENDPOINT = "https://datasets-server.huggingface.co"

# Allow endpoint override via environment variable
HF_DATASETS_SERVER_ENDPOINT = os.environ.get(
    "HF_DATASETS_SERVER_ENDPOINT",
    DEFAULT_DATASETS_SERVER_ENDPOINT,
)

# =============================================================================
# Request Configuration
# =============================================================================

# Default timeout for API requests in seconds
DEFAULT_REQUEST_TIMEOUT = 30.0

# Maximum number of rows that can be fetched per API request
MAX_ROWS_PER_REQUEST = 100

# =============================================================================
# Retry Configuration
# =============================================================================

# Default maximum number of retries for transient errors
DEFAULT_MAX_RETRIES = 3

# Base wait time between retries in seconds (will be multiplied exponentially)
DEFAULT_RETRY_BASE_WAIT = 1.0

# Maximum wait time between retries in seconds
DEFAULT_RETRY_MAX_WAIT = 8.0

# HTTP status codes that trigger automatic retry
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
