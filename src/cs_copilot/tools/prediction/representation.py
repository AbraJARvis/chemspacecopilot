#!/usr/bin/env python
# coding: utf-8
"""
Representation builders for pluggable predictive backends.

These builders sit between raw user datasets and prediction backends.
They are responsible for validating input columns, creating the backend-
specific representation, and persisting enough metadata to reproduce the same
transformation at inference time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from cs_copilot.tools.chemistry.standardize import standardize_smiles_column


@dataclass
class PreparedInput:
    """Materialized input ready to be consumed by a prediction backend."""

    prepared_csv: str
    representation_name: str
    source_input_csv: Optional[str] = None
    input_columns: List[str] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prepared_csv": self.prepared_csv,
            "representation_name": self.representation_name,
            "source_input_csv": self.source_input_csv,
            "input_columns": list(self.input_columns),
            "feature_columns": list(self.feature_columns),
            "target_columns": list(self.target_columns),
            "metadata": dict(self.metadata),
        }


@dataclass
class TrainingRecipe:
    """Declarative training recipe coupling representation + backend."""

    recipe_id: str
    backend_name: str
    representation_name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    default_train_args: Dict[str, Any] = field(default_factory=dict)
    selection_hints: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "backend_name": self.backend_name,
            "representation_name": self.representation_name,
            "display_name": self.display_name or self.recipe_id,
            "description": self.description or "",
            "default_train_args": dict(self.default_train_args),
            "selection_hints": dict(self.selection_hints),
        }


class RepresentationBuilder(ABC):
    """Contract for converting raw datasets into backend-ready representations."""

    representation_name = "base"

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """Return a lightweight description of the representation."""

    @abstractmethod
    def is_compatible(
        self,
        *,
        df: pd.DataFrame,
        task_type: str,
        target_columns: Optional[List[str]] = None,
    ) -> bool:
        """Return True when the raw dataset can be transformed by this builder."""

    @abstractmethod
    def prepare_training_input(
        self,
        *,
        df: pd.DataFrame,
        destination_csv: str,
        smiles_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
    ) -> PreparedInput:
        """Create a materialized training dataset for a backend."""

    @abstractmethod
    def prepare_inference_input(
        self,
        *,
        df: pd.DataFrame,
        destination_csv: str,
        smiles_column: str = "smiles",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PreparedInput:
        """Create a materialized inference dataset for a backend."""


def _resolve_smiles_column(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    candidates = [requested, "smiles", "SMILES", "canonical_smiles", "Smiles", "smi"]
    for candidate in candidates:
        if candidate and candidate in df.columns:
            return candidate
    raise ValueError(
        f"No SMILES column found. Tried {candidates}. Available columns: {list(df.columns)}"
    )


def _fingerprint_column_names(n_bits: int) -> List[str]:
    return [f"fp_{index:04d}" for index in range(n_bits)]


def _build_morgan_features(smiles_series: pd.Series, *, radius: int, n_bits: int) -> pd.DataFrame:
    if importlib.util.find_spec("rdkit") is None:
        raise RuntimeError("RDKit is required to build Morgan fingerprints.")

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    feature_columns = _fingerprint_column_names(n_bits)
    rows: List[List[int]] = []

    for smiles in smiles_series.tolist():
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("Cannot featurize empty SMILES values for Morgan fingerprints.")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Could not parse standardized SMILES for Morgan featurization: {smiles}")
        fingerprint = generator.GetFingerprint(molecule)
        rows.append([int(bit) for bit in fingerprint.ToBitString()])

    return pd.DataFrame(rows, columns=feature_columns)


class SmilesGraphBuilder(RepresentationBuilder):
    """Canonical SMILES builder for graph-native backends such as Chemprop."""

    representation_name = "smiles_graph"

    def describe(self) -> Dict[str, Any]:
        return {
            "representation_name": self.representation_name,
            "input_type": "smiles",
            "canonical_output_columns": ["smiles"],
            "notes": "Standardized canonical SMILES for graph-based molecular models.",
        }

    def is_compatible(
        self,
        *,
        df: pd.DataFrame,
        task_type: str,
        target_columns: Optional[List[str]] = None,
    ) -> bool:
        try:
            _resolve_smiles_column(df)
        except ValueError:
            return False
        if task_type and task_type not in {"regression", "classification"}:
            return False
        if target_columns:
            return all(column in df.columns for column in target_columns)
        return True

    def prepare_training_input(
        self,
        *,
        df: pd.DataFrame,
        destination_csv: str,
        smiles_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
    ) -> PreparedInput:
        resolved_smiles = _resolve_smiles_column(df, smiles_column)
        working = df.copy()
        working = standardize_smiles_column(working, resolved_smiles)
        if resolved_smiles != "smiles":
            working = working.rename(columns={resolved_smiles: "smiles"})
        if target_columns:
            missing_targets = [column for column in target_columns if column not in working.columns]
            if missing_targets:
                raise ValueError(f"Missing target columns for representation: {missing_targets}")

        destination = Path(destination_csv).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        working.to_csv(destination, index=False)
        return PreparedInput(
            prepared_csv=str(destination),
            representation_name=self.representation_name,
            input_columns=list(df.columns),
            target_columns=list(target_columns or []),
            metadata={
                "smiles_source_column": resolved_smiles,
                "canonical_smiles_column": "smiles",
            },
        )

    def prepare_inference_input(
        self,
        *,
        df: pd.DataFrame,
        destination_csv: str,
        smiles_column: str = "smiles",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PreparedInput:
        resolved_smiles = _resolve_smiles_column(
            df,
            (metadata or {}).get("smiles_source_column") or smiles_column,
        )
        working = df.copy()
        working = standardize_smiles_column(working, resolved_smiles)
        if resolved_smiles != "smiles":
            working = working.rename(columns={resolved_smiles: "smiles"})

        destination = Path(destination_csv).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        working.to_csv(destination, index=False)
        return PreparedInput(
            prepared_csv=str(destination),
            representation_name=self.representation_name,
            input_columns=list(df.columns),
            metadata={
                "smiles_source_column": resolved_smiles,
                "canonical_smiles_column": "smiles",
            },
        )


class MorganFingerprintBuilder(RepresentationBuilder):
    """Placeholder builder for fingerprint-based tabular models."""

    representation_name = "morgan_fp_2048"

    def describe(self) -> Dict[str, Any]:
        return {
            "representation_name": self.representation_name,
            "input_type": "smiles",
            "notes": "Morgan fingerprints suitable for tabular ML backends.",
            "feature_builder": "rdkit_morgan",
            "default_radius": 2,
            "default_n_bits": 2048,
            "status": "available",
        }

    def is_compatible(
        self,
        *,
        df: pd.DataFrame,
        task_type: str,
        target_columns: Optional[List[str]] = None,
    ) -> bool:
        try:
            _resolve_smiles_column(df)
        except ValueError:
            return False
        return task_type in {"regression", "classification"}

    def prepare_training_input(
        self,
        *,
        df: pd.DataFrame,
        destination_csv: str,
        smiles_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
    ) -> PreparedInput:
        resolved_smiles = _resolve_smiles_column(df, smiles_column)
        working = df.copy()
        working = standardize_smiles_column(working, resolved_smiles)
        if resolved_smiles != "smiles":
            working = working.rename(columns={resolved_smiles: "smiles"})

        targets = list(target_columns or [])
        missing_targets = [column for column in targets if column not in working.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns for representation: {missing_targets}")

        radius = 2
        n_bits = 2048
        features = _build_morgan_features(working["smiles"], radius=radius, n_bits=n_bits)
        prepared = pd.concat([features, working[targets].reset_index(drop=True)], axis=1)

        destination = Path(destination_csv).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(destination, index=False)
        feature_columns = list(features.columns)
        return PreparedInput(
            prepared_csv=str(destination),
            representation_name=self.representation_name,
            input_columns=list(df.columns),
            feature_columns=feature_columns,
            target_columns=targets,
            metadata={
                "smiles_source_column": resolved_smiles,
                "canonical_smiles_column": "smiles",
                "feature_builder": "rdkit_morgan",
                "radius": radius,
                "n_bits": n_bits,
                "feature_columns": feature_columns,
            },
        )

    def prepare_inference_input(
        self,
        *,
        df: pd.DataFrame,
        destination_csv: str,
        smiles_column: str = "smiles",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PreparedInput:
        resolved_smiles = _resolve_smiles_column(
            df,
            (metadata or {}).get("smiles_source_column") or smiles_column,
        )
        working = df.copy()
        working = standardize_smiles_column(working, resolved_smiles)
        if resolved_smiles != "smiles":
            working = working.rename(columns={resolved_smiles: "smiles"})

        radius = int((metadata or {}).get("radius", 2))
        n_bits = int((metadata or {}).get("n_bits", 2048))
        features = _build_morgan_features(working["smiles"], radius=radius, n_bits=n_bits)

        destination = Path(destination_csv).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(destination, index=False)
        feature_columns = list(features.columns)
        return PreparedInput(
            prepared_csv=str(destination),
            representation_name=self.representation_name,
            input_columns=list(df.columns),
            feature_columns=feature_columns,
            metadata={
                "smiles_source_column": resolved_smiles,
                "canonical_smiles_column": "smiles",
                "feature_builder": "rdkit_morgan",
                "radius": radius,
                "n_bits": n_bits,
                "feature_columns": feature_columns,
            },
        )
