#!/usr/bin/env python
# coding: utf-8
"""
Consensus ensemble backend for catalogued QSAR models.

The V1 ensemble artifact is intentionally simple: it references already
catalogued component models, executes each component backend, and aggregates
their predictions with a transparent consensus rule.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from cs_copilot.storage import S3

from .backend import (
    BackendNotAvailableError,
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionExecutionError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from .chemprop_backend import ChempropBackend
from .lightgbm_backend import LightGBMBackend
from .qsar_training_policy import safe_slug
from .tabicl_backend import TabICLBackend


ENSEMBLE_SCHEMA_VERSION = 1
DEFAULT_ENSEMBLE_KIND = "consensus_regression"
DEFAULT_AGGREGATION_STRATEGY = "median"
DEFAULT_UNCERTAINTY_STRATEGY = "component_disagreement_std"


def _strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()


def _component_slug(record: PredictionModelRecord, index: int) -> str:
    slug = safe_slug(record.model_id)
    return slug or f"component_{index + 1}"


class EnsembleBackend(PredictionBackend):
    """Prediction backend for transparent regression consensus ensembles."""

    backend_name = "ensemble"
    MODEL_EXTENSIONS = (".json",)

    def __init__(self, backends: Optional[Dict[str, PredictionBackend]] = None) -> None:
        self.backends = backends or {
            "chemprop": ChempropBackend(),
            "lightgbm": LightGBMBackend(),
            "tabicl": TabICLBackend(),
        }

    def is_available(self) -> bool:
        return True

    def describe_environment(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "available": True,
            "schema_version": ENSEMBLE_SCHEMA_VERSION,
            "capabilities": [
                "regression",
                "single_target",
                "catalogued_component_consensus",
                "component_disagreement_uncertainty",
            ],
            "supported_component_backends": sorted(self.backends),
        }

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise InvalidPredictionInputError(f"Ensemble artifact does not exist: {model_path}")
        if path.suffix not in self.MODEL_EXTENSIONS:
            raise InvalidPredictionInputError(
                f"Ensemble artifact must end with one of {self.MODEL_EXTENSIONS}: {model_path}"
            )
        payload = self._load_payload(path)
        self._validate_payload(payload, artifact_path=path)
        return path.resolve()

    def _load_payload(self, path: Path) -> Dict[str, Any]:
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            raise InvalidPredictionInputError(f"Could not read ensemble artifact {path}: {exc}") from exc

    def _validate_payload(self, payload: Dict[str, Any], *, artifact_path: Path) -> None:
        if int(payload.get("schema_version") or 0) != ENSEMBLE_SCHEMA_VERSION:
            raise InvalidPredictionInputError(
                f"Unsupported ensemble schema_version in {artifact_path}: {payload.get('schema_version')}"
            )
        if payload.get("ensemble_kind") != DEFAULT_ENSEMBLE_KIND:
            raise InvalidPredictionInputError(
                f"Unsupported ensemble_kind in {artifact_path}: {payload.get('ensemble_kind')}"
            )
        task = payload.get("task") or {}
        if task.get("task_type") != "regression":
            raise InvalidPredictionInputError("Ensemble V1 only supports regression tasks.")
        targets = list(task.get("target_columns") or [])
        if len(targets) != 1:
            raise InvalidPredictionInputError("Ensemble V1 requires exactly one target column.")
        components = payload.get("components") or []
        if len(components) < 2:
            raise InvalidPredictionInputError("An ensemble requires at least two component models.")
        for component in components:
            record_payload = component.get("record") or component
            record = PredictionModelRecord.from_dict(record_payload)
            if record.backend_name == self.backend_name:
                raise InvalidPredictionInputError("Nested ensemble components are not supported in V1.")
            if record.backend_name not in self.backends:
                raise InvalidPredictionInputError(
                    f"Unsupported ensemble component backend: {record.backend_name}"
                )
            if record.task.task_type != task.get("task_type"):
                raise InvalidPredictionInputError("All ensemble components must share the same task type.")
            if list(record.task.target_columns) != targets:
                raise InvalidPredictionInputError("All ensemble components must share the same target column.")
            if record.status == "deprecated":
                raise InvalidPredictionInputError(f"Deprecated component is not allowed: {record.model_id}")
            self.backends[record.backend_name].validate_model_path(record.model_path)

    def _component_records(self, payload: Dict[str, Any]) -> List[PredictionModelRecord]:
        records: List[PredictionModelRecord] = []
        for component in payload.get("components") or []:
            records.append(PredictionModelRecord.from_dict(component.get("record") or component))
        return records

    def _get_backend(self, backend_name: str) -> PredictionBackend:
        backend = self.backends.get(backend_name)
        if backend is None:
            raise InvalidPredictionInputError(f"Unsupported ensemble component backend: {backend_name}")
        if not backend.is_available():
            raise BackendNotAvailableError(f"Component backend is not available: {backend_name}")
        return backend

    def predict_from_csv(
        self,
        input_csv: str,
        model_record: PredictionModelRecord,
        preds_path: str,
        *,
        return_uncertainty: bool = False,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        artifact_path = self.validate_model_path(model_record.model_path)
        payload = self._load_payload(artifact_path)
        task = payload.get("task") or {}
        target_column = list(task.get("target_columns") or model_record.task.target_columns)[0]
        records = self._component_records(payload)
        output_path = Path(preds_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        component_predictions: List[pd.Series] = []
        component_columns: Dict[str, pd.Series] = {}
        component_paths: Dict[str, str] = {}
        temp_dir = output_path.parent / f"{output_path.stem}_components"
        temp_dir.mkdir(parents=True, exist_ok=True)

        expected_rows: Optional[int] = None
        for index, record in enumerate(records):
            backend = self._get_backend(record.backend_name)
            slug = _component_slug(record, index)
            component_pred_path = temp_dir / f"{slug}_predictions.csv"
            try:
                backend.predict_from_csv(
                    input_csv=input_csv,
                    model_record=record,
                    preds_path=str(component_pred_path),
                    return_uncertainty=False,
                )
            except Exception as exc:
                raise PredictionExecutionError(
                    f"Ensemble component prediction failed for {record.model_id}: {exc}"
                ) from exc
            with S3.open(str(component_pred_path), "r") as fh:
                component_df = _strip_unnamed_columns(pd.read_csv(fh))
            prediction_source = (
                "prediction"
                if "prediction" in component_df.columns
                else target_column
                if target_column in component_df.columns
                else None
            )
            if prediction_source is None:
                raise PredictionExecutionError(
                    f"Component {record.model_id} did not produce a prediction column."
                )
            series = pd.to_numeric(component_df[prediction_source], errors="coerce").astype(float)
            if series.isna().any():
                raise PredictionExecutionError(
                    f"Component {record.model_id} produced non-numeric predictions."
                )
            if expected_rows is None:
                expected_rows = int(len(series))
            elif int(len(series)) != expected_rows:
                raise PredictionExecutionError(
                    f"Component {record.model_id} produced {len(series)} rows; expected {expected_rows}."
                )
            component_predictions.append(series.reset_index(drop=True))
            column_name = f"prediction_{slug}"
            component_columns[column_name] = series.reset_index(drop=True)
            component_paths[column_name] = str(component_pred_path)

        matrix = pd.concat(component_predictions, axis=1)
        official = matrix.median(axis=1)
        mean = matrix.mean(axis=1)
        std = matrix.std(axis=1, ddof=0).fillna(0.0)
        minimum = matrix.min(axis=1)
        maximum = matrix.max(axis=1)

        output = pd.DataFrame(
            {
                "prediction": official.astype(float),
                "ensemble_prediction_median": official.astype(float),
                "ensemble_prediction_mean": mean.astype(float),
                "ensemble_prediction_std": std.astype(float),
                "ensemble_prediction_min": minimum.astype(float),
                "ensemble_prediction_max": maximum.astype(float),
                "ensemble_component_count": int(len(records)),
            }
        )
        output[target_column] = output["prediction"]
        for column_name, series in component_columns.items():
            output[column_name] = series.astype(float)

        with S3.open(str(output_path), "w") as fh:
            output.to_csv(fh, index=False)

        return {
            "preds_path": str(output_path),
            "rows": int(len(output)),
            "ensemble_kind": payload.get("ensemble_kind"),
            "aggregation_strategy": payload.get("aggregation_strategy", DEFAULT_AGGREGATION_STRATEGY),
            "uncertainty_strategy": payload.get(
                "uncertainty_strategy",
                DEFAULT_UNCERTAINTY_STRATEGY,
            ),
            "component_count": int(len(records)),
            "component_prediction_paths": component_paths,
            "applicability_domain": {},
            "applicability_domain_columns": [],
            "return_uncertainty": return_uncertainty,
            "mean_component_disagreement_std": (
                float(std.mean()) if len(std) else math.nan
            ),
        }

    def train_model(
        self,
        train_csv: str,
        output_dir: str,
        task: PredictionTaskSpec,
        *,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise InvalidPredictionInputError(
            "EnsembleBackend does not train component models directly. "
            "Create an ensemble from catalogued models or a robust benchmark campaign."
        )
