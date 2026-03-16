#!/usr/bin/env python
# coding: utf-8
"""
Database toolkit package.

This package provides a unified interface for working with various databases
through the BaseDatabaseToolkit abstract class and specific implementations
like ChemblToolkit.
"""

import logging

logger = logging.getLogger(__name__)

from .base import (
    BaseDatabaseToolkit,
    ConnectionError,
    DatabaseError,
    NotFound,
    QueryTimeout,
    RateLimited,
    ValidationError,
)
from .types import DBConfig, PaginationMode, QueryMetrics, QueryParams, Record, ResultPage

# Import ChemblToolkit - now safe because it uses lazy initialization internally
from .chembl import ChemblToolkit

__all__ = [
    # Toolkit classes
    "BaseDatabaseToolkit",
    "ChemblToolkit",
    # Types and configurations
    "DBConfig",
    "QueryParams",
    "ResultPage",
    "Record",
    "QueryMetrics",
    "PaginationMode",
    # Exceptions
    "DatabaseError",
    "ConnectionError",
    "NotFound",
    "ValidationError",
    "RateLimited",
    "QueryTimeout",
]
