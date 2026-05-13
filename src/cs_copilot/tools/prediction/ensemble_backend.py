#!/usr/bin/env python
# coding: utf-8
"""Post-hoc consensus ensemble backend for catalogued QSAR models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

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


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser())


def _prediction_column(frame: pd.DataFrame, target_columns: list[str]) -> str:
    if "prediction" in frame.columns:
        return "prediction"
    for column in target_columns:
        if column in frame.columns:
            return column
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    if numeric:
        return numeric[-1]
    raise InvalidPredictionInputError("Component prediction output does not contain a numeric prediction column.")


class EnsembleBackend(PredictionBackend):
    """Aggregate predictions from already-persisted component models."""

    backend_name = "ensemble"

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
            "capabilities": ["regression", "post_hoc_consensus", "median_aggregation"],
            "component_backends": {
                name: backend.describe_environment() for name, backend in self.backends.items()
            },
        }

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise InvalidPredictionInputError(f"Ensemble model path does not exist: {model_path}")
        if path.name != "ensemble.json":
            raise InvalidPredictionInputError("Ensemble model artifact must be named ensemble.json.")
        payload = self._load_payload(path)
        if int(payload.get("schema_version", 0)) != 1:
            raise InvalidPredictionInputError("Unsupported ensemble schema_version.")
        if payload.get("ensemble_kind") != "catalog_consensus_regression":
            raise InvalidPredictionInputError("Unsupported ensemble_kind for this V1 backend.")
        if payload.get("aggregation_strategy") != "median":
            raise InvalidPredictionInputError("Only median aggregation is supported in ensemble V1.")
        components = payload.get("components")
        if not isinstance(components, list) or not components:
            raise InvalidPredictionInputError("Ensemble must contain at least one component.")
        return path.resolve()

    def _load_payload(self, path: Path) -> Dict[str, Any]:
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            raise InvalidPredictionInputError(f"Could not read ensemble.json: {exc}") from exc
        if not isinstance(payload, dict):
            raise InvalidPredictionInputError("ensemble.json must contain a JSON object.")
        return payload

    def _component_record(self, component: Dict[str, Any], ensemble_record: PredictionModelRecord) -> PredictionModelRecord:
        task_payload = component.get("task") or {}
        return PredictionModelRecord(
            model_id=str(component.get("model_id")),
            backend_name=str(component.get("backend_name")),
            model_path=str(component.get("model_path")),
            metadata_path=component.get("metadata_path"),
            display_name=component.get("display_name"),
            description=component.get("description"),
            version=component.get("version"),
            status=component.get("status", "workflow_demo"),
            known_metrics=dict(component.get("known_metrics") or {}),
            training_data_summary=dict(component.get("training_data_summary") or {}),
            inference_profile=dict(component.get("inference_profile") or {}),
            selection_hints=dict(component.get("selection_hints") or {}),
            applicability_domain=dict(component.get("applicability_domain") or {}),
            task=PredictionTaskSpec(
                task_type=task_payload.get("task_type") or ensemble_record.task.task_type,
                smiles_columns=task_payload.get("smiles_columns") or ensemble_record.task.smiles_columns,
                target_columns=task_payload.get("target_columns") or ensemble_record.task.target_columns,
            ),
        )

    def predict_from_csv(
        self,
        input_csv: str,
        model_record: PredictionModelRecord,
        preds_path: str,
        *,
        return_uncertainty: bool = False,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ensemble_path = self.validate_model_path(model_record.model_path)
        payload = self._load_payload(ensemble_path)
        components = payload.get("components") or []
        output_path = Path(preds_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir = output_path.parent / f"{output_path.stem}_components"
        work_dir.mkdir(parents=True, exist_ok=True)

        source_df = _read_csv(input_csv)
        component_columns: Dict[str, pd.Series] = {}
        component_paths: Dict[str, str] = {}
        failures: list[str] = []

        for component in components:
            backend_name = str(component.get("backend_name") or "")
            backend = self.backends.get(backend_name)
            if backend is None:
                failures.append(f"{component.get('model_id')}: unsupported backend `{backend_name}`")
                continue
            record = self._component_record(component, model_record)
            slug = safe_slug(str(component.get("component_slug") or component.get("model_id") or backend_name)) or backend_name
            component_path = work_dir / f"{slug}_predictions.csv"
            try:
                backend.predict_from_csv(
                    input_csv=input_csv,
                    model_record=record,
                    preds_path=str(component_path),
                    return_uncertainty=False,
                    extra_args=extra_args,
                )
                frame = _read_csv(component_path)
                column = _prediction_column(frame, record.task.target_columns)
                values = pd.to_numeric(frame[column], errors="coerce")
                if len(values) != len(source_df):
                    raise InvalidPredictionInputError(
                        f"Component `{record.model_id}` returned {len(values)} predictions for {len(source_df)} inputs."
                    )
                component_columns[f"prediction_{slug}"] = values.reset_index(drop=True)
                component_paths[slug] = str(component_path)
            except Exception as exc:
                failures.append(f"{component.get('model_id')}: {exc}")

        if failures:
            raise PredictionExecutionError("Ensemble component inference failed: " + " | ".join(failures))
        if not component_columns:
            raise PredictionExecutionError("No component predictions were produced.")

        component_df = pd.DataFrame(component_columns)
        aggregate = pd.DataFrame(
            {
                "prediction": component_df.median(axis=1),
                "ensemble_prediction_median": component_df.median(axis=1),
                "ensemble_prediction_mean": component_df.mean(axis=1),
                "ensemble_prediction_std": component_df.std(axis=1, ddof=0),
                "ensemble_prediction_min": component_df.min(axis=1),
                "ensemble_prediction_max": component_df.max(axis=1),
                "ensemble_component_count": len(component_columns),
            }
        )
        result_df = pd.concat([aggregate, component_df], axis=1)
        result_df.to_csv(output_path, index=False)
        return {
            "backend_name": self.backend_name,
            "predictions_path": str(output_path),
            "component_prediction_paths": component_paths,
            "component_count": len(component_columns),
            "aggregation_strategy": "median",
            "uncertainty_strategy": "component_disagreement_std",
            "return_uncertainty": return_uncertainty,
        }

    def train_model(
        self,
        train_csv: str,
        output_dir: str,
        task: PredictionTaskSpec,
        *,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise BackendNotAvailableError("EnsembleBackend is post-hoc only and does not train component models.")
