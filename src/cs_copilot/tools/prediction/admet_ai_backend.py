#!/usr/bin/env python
# coding: utf-8
"""
ADMET-AI backend adapter.

This backend uses the official ADMET-AI Python package for inference rather than
forcing ADMET-AI assets through the Chemprop v2 CLI.  It is the correct path for
the ADMET-AI model family and avoids checkpoint-format mismatches.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .backend import (
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionExecutionError,
    PredictionModelRecord,
    PredictionTaskSpec,
)


class AdmetAIBackend(PredictionBackend):
    """Prediction backend built around the official ADMET-AI package."""

    backend_name = "admet_ai"

    def _package_version(self) -> Optional[str]:
        try:
            return importlib.metadata.version("admet-ai")
        except importlib.metadata.PackageNotFoundError:
            try:
                return importlib.metadata.version("admet_ai")
            except importlib.metadata.PackageNotFoundError:
                return None

    def is_available(self) -> bool:
        if self._package_version() is None:
            return False
        return importlib.util.find_spec("admet_ai") is not None

    def describe_environment(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "available": self.is_available(),
            "package_version": self._package_version(),
            "execution_mode": "python_api",
        }

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise InvalidPredictionInputError(f"Model path does not exist: {model_path}")
        return path

    def _load_model(self):
        if not self.is_available():
            raise PredictionExecutionError(
                "ADMET-AI backend is not available. Install the `admet-ai` package first."
            )

        try:
            from admet_ai import ADMETModel
        except Exception as exc:  # pragma: no cover - import guard
            raise PredictionExecutionError(
                "Failed to import ADMET-AI Python API."
            ) from exc

        try:
            return ADMETModel()
        except Exception as exc:
            raise PredictionExecutionError(
                "Failed to initialize the ADMET-AI model bundle."
            ) from exc

    def predict_from_csv(
        self,
        input_csv: str,
        model_record: PredictionModelRecord,
        preds_path: str,
        *,
        return_uncertainty: bool = False,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if return_uncertainty:
            raise InvalidPredictionInputError(
                "ADMET-AI backend does not currently expose uncertainty outputs in this integration."
            )

        input_path = Path(input_csv).expanduser()
        if not input_path.exists():
            raise InvalidPredictionInputError(f"Input CSV does not exist: {input_csv}")

        output_path = Path(preds_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_path)
        smiles_column = (model_record.task.smiles_columns or ["smiles"])[0]
        if smiles_column not in df.columns:
            raise InvalidPredictionInputError(
                f"SMILES column `{smiles_column}` not found in input CSV."
            )

        model = self._load_model()
        try:
            prediction_df = model.predict(smiles=df[smiles_column].astype(str).tolist())
        except Exception as exc:
            raise PredictionExecutionError(
                "ADMET-AI prediction failed through the Python API."
            ) from exc

        if not isinstance(prediction_df, pd.DataFrame):
            raise PredictionExecutionError(
                "ADMET-AI returned an unexpected prediction object."
            )

        prediction_df = prediction_df.reset_index()
        if "index" in prediction_df.columns and smiles_column not in prediction_df.columns:
            prediction_df = prediction_df.rename(columns={"index": smiles_column})

        target_columns = model_record.task.target_columns or []
        output_df = df.copy()
        added_columns = []

        if target_columns:
            missing_targets = [col for col in target_columns if col not in prediction_df.columns]
            if missing_targets:
                raise PredictionExecutionError(
                    "ADMET-AI predictions did not contain the requested target columns: "
                    f"{missing_targets}"
                )
            for column in target_columns:
                output_df[column] = prediction_df[column].values
                added_columns.append(column)
        else:
            merge_columns = [
                column for column in prediction_df.columns if column != smiles_column
            ]
            output_df = output_df.merge(prediction_df, on=smiles_column, how="left")
            added_columns.extend(merge_columns)

        output_df.to_csv(output_path, index=False)
        return {
            "backend": self.backend_name,
            "preds_path": str(output_path),
            "predicted_columns": added_columns,
            "num_rows": int(len(output_df)),
            "stdout": "",
            "stderr": "",
        }

    def train_model(
        self,
        train_csv: str,
        output_dir: str,
        task: PredictionTaskSpec,
        *,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise PredictionExecutionError(
            "Training through ADMET-AI is not supported in this integration. "
            "Use Chemprop training workflows for now."
        )
