#!/usr/bin/env python
# coding: utf-8
"""
Base dimensionality reduction toolkit providing general dimensionality reduction techniques.
"""

import logging
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from agno.tools.toolkit import Toolkit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DRToolkitError(Exception):
    """Base exception for dimensionality reduction operations."""

    pass


class BaseDRToolkit(Toolkit):
    """
    Base toolkit for general dimensionality reduction operations.
    """

    def __init__(self, name: str = "base_dimensionality_reduction"):
        """Initialize the BaseDimensionalityReductionToolkit.

        Args:
            name: Name of the toolkit (used for identification)
        """
        super().__init__(name)
        # Register basic tools
        # self.register(self.pca_reduction)
        # self.register(self.standardize_data)

    def _validate_data(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate and convert input data to numpy array."""
        if data is None:
            raise DRToolkitError("Data cannot be None")

        if isinstance(data, pd.DataFrame):
            data = data.values
        elif not isinstance(data, np.ndarray):
            raise DRToolkitError("Data must be numpy array or pandas DataFrame")

        if data.size == 0:
            raise DRToolkitError("Data cannot be empty")

        if len(data.shape) != 2:
            raise DRToolkitError("Data must be 2-dimensional")

        return data

    def standardize_data(
        self,
        data: Union[np.ndarray, pd.DataFrame],  # TODO introduce other options
        standardize: bool = True,
    ) -> tuple:
        """Standardize data for dimensionality reduction."""
        data = self._validate_data(data)

        if not standardize:
            return data, None

        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data)
        return standardized_data, scaler

    def pca_reduction(
        self, data: Union[np.ndarray, pd.DataFrame], n_components: int = 2, standardize: bool = True
    ) -> Dict[str, Any]:
        """Perform Principal Component Analysis (PCA) dimensionality reduction."""
        data = self._validate_data(data)

        if standardize:
            data, scaler = self.standardize_data(data, standardize=True)
        else:
            scaler = None

        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)

        return {
            "reduced_data": reduced_data,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "model": pca,
            "scaler": scaler,
        }

    # TODO: Potentially include intrinsic dimension estimation
