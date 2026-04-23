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
from rdkit import Chem
from rdkit.Chem import Descriptors

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.base_chemistry import calc_morgan_bit_fp
from cs_copilot.tools.chemistry.standardize import standardize_smiles_column


def _resolve_output_csv(output_csv: Optional[str], input_csv: str, suffix: str) -> str:
    if output_csv:
        return S3.path(output_csv)
    stem = Path(input_csv).stem or "molecular_features"
    return S3.path(f"features/{stem}{suffix}")


def _feature_column_names(prefix: str, n_bits: int) -> List[str]:
    return [f"{prefix}{index:04d}" for index in range(n_bits)]


_BASIC_RDKIT_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "FractionCSP3": Descriptors.FractionCSP3,
}


def _build_base_output_dataframe(
    working: pd.DataFrame,
    *,
    include_input_columns: bool,
    input_columns_to_keep: Optional[List[str]],
) -> pd.DataFrame:
    if input_columns_to_keep is not None:
        missing_columns = [column for column in input_columns_to_keep if column not in working.columns]
        if missing_columns:
            raise ValueError(f"Requested input columns to keep are missing: {missing_columns}")
        return working[input_columns_to_keep].copy()
    if include_input_columns:
        return working.copy()
    return pd.DataFrame(index=working.index)


class MolecularFeatureToolkit(Toolkit):
    """Explicit tools for transforming molecular datasets into tabular features."""

    def __init__(self):
        super().__init__("molecular_features")
        self.register(self.smiles_to_morgan_fingerprints)
        self.register(self.smiles_to_rdkit_descriptors)

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
            fingerprint = calc_morgan_bit_fp(smiles, n_bits)
            if fingerprint is None:
                raise ValueError(
                    f"Could not compute Morgan fingerprint for standardized SMILES: {smiles}"
                )
            fingerprint_rows.append(fingerprint.astype(int).tolist())

        feature_df = pd.DataFrame(fingerprint_rows, columns=feature_columns)

        base_df = _build_base_output_dataframe(
            working,
            include_input_columns=include_input_columns,
            input_columns_to_keep=input_columns_to_keep,
        )

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

    def smiles_to_rdkit_descriptors(
        self,
        input_csv: str,
        smiles_column: str = "smiles",
        output_csv: Optional[str] = None,
        descriptor_set: str = "basic",
        include_input_columns: bool = False,
        input_columns_to_keep: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Transform a SMILES column into a tabular CSV of basic RDKit descriptors.

        This V1 intentionally exposes a short, fixed descriptor set to keep the
        output stable, testable, and easy to reason about.
        """
        if descriptor_set != "basic":
            raise ValueError(
                "Only descriptor_set='basic' is supported in this V1 implementation."
            )

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
                f"Cannot generate RDKit descriptors: {invalid_rows} row(s) have invalid or missing standardized SMILES."
            )

        descriptor_rows: List[Dict[str, float]] = []
        descriptor_names = list(_BASIC_RDKIT_DESCRIPTOR_FUNCS.keys())
        descriptor_columns = [f"desc_{name}" for name in descriptor_names]

        for smiles in working["smiles"].tolist():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(
                    f"Could not compute RDKit descriptors for standardized SMILES: {smiles}"
                )
            descriptor_rows.append(
                {
                    f"desc_{name}": float(func(mol))
                    for name, func in _BASIC_RDKIT_DESCRIPTOR_FUNCS.items()
                }
            )

        descriptor_df = pd.DataFrame(descriptor_rows, columns=descriptor_columns)
        base_df = _build_base_output_dataframe(
            working,
            include_input_columns=include_input_columns,
            input_columns_to_keep=input_columns_to_keep,
        )
        output_df = pd.concat(
            [base_df.reset_index(drop=True), descriptor_df.reset_index(drop=True)], axis=1
        )

        resolved_output_csv = _resolve_output_csv(output_csv, input_csv, "_rdkit_desc.csv")
        with S3.open(resolved_output_csv, "w") as fh:
            output_df.to_csv(fh, index=False)

        kept_columns = list(base_df.columns)
        return {
            "output_csv": resolved_output_csv,
            "rows_in": int(len(df)),
            "rows_out": int(len(output_df)),
            "smiles_column": "smiles",
            "descriptor_set": descriptor_set,
            "num_descriptors": len(descriptor_columns),
            "descriptor_names": descriptor_names,
            "descriptor_columns_sample": descriptor_columns[:5],
            "input_columns_kept": kept_columns,
        }
