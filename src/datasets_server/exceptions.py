"""Custom exceptions for the Datasets Server API client."""


class DatasetServerError(Exception):
    """Base exception for all dataset server errors."""

    pass


class DatasetNotFoundError(DatasetServerError):
    """Raised when a dataset is not found."""

    pass


class DatasetNotValidError(DatasetServerError):
    """Raised when attempting operations on invalid datasets."""

    pass


class DatasetServerTimeoutError(DatasetServerError):
    """Raised when API requests timeout."""

    pass
