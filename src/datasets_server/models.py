"""Response models for the Datasets Server API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetValidity(BaseModel):
    """Response model for dataset validity check."""

    viewer: bool
    preview: bool
    search: bool
    filter: bool
    statistics: bool


class DatasetSplit(BaseModel):
    """Response model for dataset split information."""

    dataset: str
    config: str
    split: str


class ParquetFile(BaseModel):
    """Response model for Parquet file information."""

    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int


class DatasetRows(BaseModel):
    """Response model for dataset rows."""

    features: List[Dict[str, Any]]
    rows: List[Dict[str, Any]]
    num_rows_total: Optional[int] = None
    num_rows_per_page: Optional[int] = None
    partial: Optional[bool] = False


class DatasetSize(BaseModel):
    """Response model for dataset size information."""

    size: Dict[str, Any]
    pending: List[Any] = Field(default_factory=list)
    failed: List[Any] = Field(default_factory=list)
    partial: bool = False


class DatasetInfo(BaseModel):
    """Response model for dataset information."""

    dataset_info: Dict[str, Any]
    pending: List[Any] = Field(default_factory=list)
    failed: List[Any] = Field(default_factory=list)
    partial: bool = False


class DatasetStatistics(BaseModel):
    """Response model for dataset statistics."""

    num_examples: int
    statistics: List[Dict[str, Any]]
    partial: bool = False
