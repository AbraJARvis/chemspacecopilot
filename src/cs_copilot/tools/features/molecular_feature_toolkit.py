#!/usr/bin/env python
# coding: utf-8
"""
Toolkit for explicit molecular feature generation from curated QSAR datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.tools.toolkit import Toolkit

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.base_chemistry import calc_morgan_fp
from cs_copilot.tools.chemistry.standardize import standardize_smiles_column


def _resolve_output_csv(output_csv: Optional[str], input_csv: str, suffix: str) -> str:
    if output_csv:
        return S3.path(output_csv)
    stem = Path(input_csv).stem or "molecular_features"
    return S3.path(f"features/{stem}{suffix}")


def _feature_column_names(prefix: str, n_bits: int) -> List[str]:
    return [f"{prefix}{index:04d}" for index in range(n_bits)]


class MolecularFeatureToolkit(Toolkit):
    """Explicit tools for transforming molecular datasets into tabular features."""

    def __init__(self):
        super().__init__("molecular_features")
        self.register(self.smiles_to_morgan_fingerprints)

    def smiles_to_morgan_fingerprints(
        self,
        input_csv: str,
        smiles_column: str = "smiles",
        output_csv: Optional[str] = None,
        radius: int = 2,
        n_bits: int = 2048,
        include_input_columns: bool = False,
        input_columns_to_keep: Optional[List[str]] = None,
        feature_prefix: str = "fp_",
    ) -> Dict[str, Any]:
        """
        Transform a SMILES column into a tabular CSV of Morgan fingerprints.

        This tool is intentionally explicit and testable on its own so future
        tabular backends (for example TabICL or tree models) can reuse the same
        featurization step without coupling it to training.
        """
        if radius != 2:
            raise ValueError(
                "The current Morgan helper supports radius=2 only in this V1 implementation."
            )
        if n_bits <= 0:
            raise ValueError("n_bits must be a positive integer.")
        if not feature_prefix:
            raise ValueError("feature_prefix cannot be empty.")

        with S3.open(input_csv, "r") as fh:
            df = pd.read_csv(fh)

        if smiles_column not in df.columns:
            raise ValueError(
                f"SMILES column '{smiles_column}' not found. Available columns: {list(df.columns)}"
            )

        working = standardize_smiles_column(df.copy(), smiles_column)
        if smiles_column != "smiles":
            working = working.rename(columns={smiles_column: "smiles"})

        invalid_mask = working["smiles"].isna()
        if invalid_mask.any():
            invalid_rows = int(invalid_mask.sum())
            raise ValueError(
                f"Cannot generate Morgan fingerprints: {invalid_rows} row(s) have invalid or missing standardized SMILES."
            )

        feature_columns = _feature_column_names(feature_prefix, n_bits)
        fingerprint_rows: List[Any] = []
        for smiles in working["smiles"].tolist():
            fingerprint = calc_morgan_fp(smiles, n_bits)
            if fingerprint is None:
                raise ValueError(
                    f"Could not compute Morgan fingerprint for standardized SMILES: {smiles}"
                )
            fingerprint_rows.append(fingerprint.tolist())

        feature_df = pd.DataFrame(fingerprint_rows, columns=feature_columns)

        if input_columns_to_keep is not None:
            missing_columns = [column for column in input_columns_to_keep if column not in working.columns]
            if missing_columns:
                raise ValueError(
                    f"Requested input columns to keep are missing: {missing_columns}"
                )
            base_df = working[input_columns_to_keep].copy()
        elif include_input_columns:
            base_df = working.copy()
        else:
            base_df = pd.DataFrame(index=working.index)

        output_df = pd.concat([base_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

        resolved_output_csv = _resolve_output_csv(output_csv, input_csv, "_morgan_fp.csv")
        with S3.open(resolved_output_csv, "w") as fh:
            output_df.to_csv(fh, index=False)

        kept_columns = list(base_df.columns)
        return {
            "output_csv": resolved_output_csv,
            "rows_in": int(len(df)),
            "rows_out": int(len(output_df)),
            "smiles_column": "smiles",
            "radius": radius,
            "n_bits": n_bits,
            "num_features": len(feature_columns),
            "feature_prefix": feature_prefix,
            "feature_columns_sample": feature_columns[:5],
            "input_columns_kept": kept_columns,
        }
