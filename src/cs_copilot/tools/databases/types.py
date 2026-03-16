#!/usr/bin/env python
# coding: utf-8
"""
Shared types and dataclasses for database toolkit.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class PaginationMode(Enum):
    """Pagination modes supported by different databases."""

    OFFSET_LIMIT = "offset_limit"
    CURSOR = "cursor"
    PAGE_NUMBER = "page_number"
    TOKEN = "token"


@dataclass
class DBConfig:
    """Database connection configuration."""

    uri: str
    api_key: Optional[str] = None
    timeout_s: float = 30.0
    retries: int = 3
    page_size: int = 100
    rate_limit: Optional[float] = None  # requests per second
    headers: Dict[str, str] = field(default_factory=dict)

    # Capability flags
    supports_sql: bool = False
    supports_http_api: bool = True
    pagination_mode: PaginationMode = PaginationMode.OFFSET_LIMIT


@dataclass
class QueryParams:
    """Parameters for database queries."""

    filters: Dict[str, Any] = field(default_factory=dict)
    fields: Optional[Sequence[str]] = None
    sort: Optional[Sequence[Tuple[str, str]]] = None  # (field, "asc"/"desc")
    limit: Optional[int] = None
    offset: int = 0

    # For cursor-based pagination
    cursor: Optional[str] = None

    # For page-based pagination
    page: Optional[int] = None

    # Additional query parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Record:
    """A single database record."""

    data: Dict[str, Any]
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultPage:
    """A page of query results."""

    records: List[Dict[str, Any]]
    total: Optional[int] = None
    next_offset: Optional[int] = None
    next_cursor: Optional[str] = None
    next_page: Optional[int] = None
    has_more: bool = False

    # Metadata about the query
    query_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryMetrics:
    """Metrics for query execution."""

    query_time_ms: float
    records_returned: int
    total_records: Optional[int] = None
    pages_fetched: int = 1
    retries: int = 0
    cache_hit: bool = False


# Type aliases for convenience
DatabaseRecord = Dict[str, Any]
QueryFilter = Dict[str, Union[str, int, float, bool, List[Any]]]
FieldMapping = Dict[str, str]  # source_field -> target_field
