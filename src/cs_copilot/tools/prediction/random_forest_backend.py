#!/usr/bin/env python
# coding: utf-8
"""
Random forest backend adapter.

This backend targets tabular representations such as Morgan fingerprints and
keeps the same high-level contract as Chemprop so the agent can orchestrate
training and inference across heterogeneous model families.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .backend import (
    BackendNotAvailableError,
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionModelRecord,
    PredictionTaskSpec,
)


class RandomForestBackend(PredictionBackend):
    """Prediction backend powered by scikit-learn random forests."""

    backend_name = "random_forest"
    MODEL_EXTENSIONS = (".joblib", ".pkl")

    def _load_joblib(self):
        if importlib.util.find_spec("joblib") is None:
            raise BackendNotAvailableError("joblib is required for the random forest backend.")
        import joblib

        return joblib

    def _load_estimators(self):
        if importlib.util.find_spec("sklearn") is None:
            raise BackendNotAvailableError("scikit-learn is required for the random forest backend.")
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        return RandomForestClassifier, RandomForestRegressor

    def _package_version(self) -> Optional[str]:
        try:
            return importlib.metadata.version("scikit-learn")
        except importlib.metadata.PackageNotFoundError:
            return None

    def is_available(self) -> bool:
        return (
            importlib.util.find_spec("sklearn") is not None
            and importlib.util.find_spec("joblib") is not None
        )

    def describe_environment(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "available": self.is_available(),
            "package_version": self._package_version(),
            "model_extensions": list(self.MODEL_EXTENSIONS),
        }

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise InvalidPredictionInputError(f"Model path does not exist: {model_path}")
        if not path.is_file():
            raise InvalidPredictionInputError(f"Random forest model artifact must be a file: {model_path}")
        if path.suffix not in self.MODEL_EXTENSIONS:
            raise InvalidPredictionInputError(
                f"Random forest model artifact must end with one of {self.MODEL_EXTENSIONS}: {model_path}"
            )
        return path

    def _ensure_available(self) -> None:
        if not self.is_available():
            raise BackendNotAvailableError(
                "Random forest backend is not available. Install scikit-learn to enable it."
            )

    def _resolve_feature_columns(
        self,
        df: pd.DataFrame,
        *,
        model_record: Optional[PredictionModelRecord] = None,
        extra_args: Optional[Dict[str, Any]] = None,
        target_columns: Optional[List[str]] = None,
    ) -> List[str]:
        explicit = None
        if extra_args:
            explicit = extra_args.get("feature_columns")
        if explicit is None and model_record is not None:
            explicit = (
                model_record.feature_schema.get("feature_columns")
                or model_record.representation_metadata.get("feature_columns")
            )

        if explicit:
            missing = [column for column in explicit if column not in df.columns]
            if missing:
                raise InvalidPredictionInputError(
                    f"Missing required feature columns for random forest backend: {missing}"
                )
            return list(explicit)

        excluded = set(target_columns or [])
        feature_columns = [column for column in df.columns if column not in excluded]
        numeric_columns = [
            column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])
        ]
        if not numeric_columns:
            raise InvalidPredictionInputError(
                "Random forest backend requires numeric feature columns, but none were detected."
            )
        return numeric_columns

    def predict_from_csv(
        self,
        input_csv: str,
        model_record: PredictionModelRecord,
        preds_path: str,
        *,
        return_uncertainty: bool = False,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_available()
        model_path = self.validate_model_path(model_record.model_path)
        input_path = Path(input_csv).expanduser()
        output_path = Path(preds_path).expanduser()
        if not input_path.exists():
            raise InvalidPredictionInputError(f"Prediction input CSV does not exist: {input_csv}")

        df = pd.read_csv(input_path)
        joblib = self._load_joblib()
        feature_columns = self._resolve_feature_columns(
            df,
            model_record=model_record,
            extra_args=extra_args,
            target_columns=model_record.task.target_columns,
        )
        model = joblib.load(model_path)
        predictions = model.predict(df[feature_columns])
        target_column = (
            model_record.task.target_columns[0]
            if model_record.task.target_columns
            else extra_args.get("prediction_column", "prediction")
            if extra_args
            else "prediction"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({target_column: predictions}).to_csv(output_path, index=False)
        return {
            "backend_name": self.backend_name,
            "preds_path": str(output_path),
            "prediction_column": target_column,
            "feature_columns": feature_columns,
            "return_uncertainty": return_uncertainty,
            "applicability_domain": {},
            "applicability_domain_columns": [],
        }

    def train_model(
        self,
        train_csv: str,
        output_dir: str,
        task: PredictionTaskSpec,
        *,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_available()
        input_path = Path(train_csv).expanduser()
        if not input_path.exists():
            raise InvalidPredictionInputError(f"Training input CSV does not exist: {train_csv}")
        if not task.target_columns:
            raise InvalidPredictionInputError("Random forest training requires at least one target column.")

        df = pd.read_csv(input_path)
        joblib = self._load_joblib()
        RandomForestClassifier, RandomForestRegressor = self._load_estimators()
        target_column = task.target_columns[0]
        if target_column not in df.columns:
            raise InvalidPredictionInputError(
                f"Target column '{target_column}' is missing from training CSV."
            )

        feature_columns = self._resolve_feature_columns(
            df,
            extra_args=extra_args,
            target_columns=task.target_columns,
        )
        train_args = dict(extra_args or {})
        n_estimators = int(train_args.get("n_estimators", 300))
        random_state = int(train_args.get("random_state", 42))
        n_jobs = int(train_args.get("n_jobs", -1))

        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        X = df[feature_columns]
        y = df[target_column]

        if task.task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
            )

        model.fit(X, y)

        model_path = output_path / "model.joblib"
        metadata_path = output_path / "rf_training_summary.json"
        joblib.dump(model, model_path)

        metadata_payload = {
            "backend_name": self.backend_name,
            "task_type": task.task_type,
            "target_columns": list(task.target_columns),
            "feature_columns": feature_columns,
            "representation_name": train_args.get("representation_name", "morgan_fp_2048"),
            "training_params": {
                "n_estimators": n_estimators,
                "random_state": random_state,
                "n_jobs": n_jobs,
            },
            "model_path": str(model_path),
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2) + "\n")

        feature_importances = getattr(model, "feature_importances_", None)
        feature_schema = {
            "num_features": len(feature_columns),
            "feature_column_prefix": "fp_",
        }
        if feature_importances is not None:
            ranked = sorted(
                zip(feature_columns, feature_importances.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )[:25]
            feature_schema["top_feature_importances"] = [
                {"feature": column, "importance": float(value)}
                for column, value in ranked
            ]

        return {
            "backend_name": self.backend_name,
            "model_path": str(model_path),
            "best_model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "target_column": target_column,
            "feature_columns": feature_columns,
            "feature_schema": feature_schema,
            "training_params": metadata_payload["training_params"],
        }
