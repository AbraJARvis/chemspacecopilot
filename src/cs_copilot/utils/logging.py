#!/usr/bin/env python
# coding: utf-8
"""
Logging utilities for cs_copilot.

This module provides logging configuration and utilities to handle
common logging issues in Jupyter notebooks and multiprocessing environments.
"""

import logging
import warnings
from typing import Optional


def setup_logging(
    level: int = logging.INFO, suppress_warnings: bool = True, suppress_tqdm: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for cs_copilot.

    Args:
        level: Logging level (default: INFO)
        suppress_warnings: Whether to suppress common warnings
        suppress_tqdm: Whether to suppress tqdm-related warnings

    Returns:
        Configured logger instance
    """
    # Configure logging
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True
    )

    logger = logging.getLogger("cs_copilot")

    if suppress_warnings:
        # Suppress CUDA warnings
        warnings.filterwarnings("ignore", message="Can't initialize NVML")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

        # Suppress other common warnings
        warnings.filterwarnings("ignore", message=".*torch.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*numpy.*", category=DeprecationWarning)

    if suppress_tqdm:
        # Suppress tqdm warnings that occur in Jupyter notebooks
        warnings.filterwarnings("ignore", message=".*tqdm.*")

        # Try to fix tqdm issues by setting environment variable
        import os

        os.environ["TQDM_DISABLE"] = "1"

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name (default: 'cs_copilot')

    Returns:
        Logger instance
    """
    if name is None:
        name = "cs_copilot"
    return logging.getLogger(name)
