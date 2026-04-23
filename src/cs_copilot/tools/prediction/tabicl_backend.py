#!/usr/bin/env python
# coding: utf-8
"""
TabICLv2 backend adapter for tabular QSAR workflows.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import logging
import math
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from cs_copilot.storage import S3

from .backend import (
    BackendNotAvailableError,
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionExecutionError,
    PredictionModelRecord,
    PredictionTaskSpec,
)

logger = logging.getLogger(__name__)

DEFAULT_TABICL_CHECKPOINT_DIR = Path("data/model_assets/checkpoints/tabicl").resolve()
DEFAULT_TABICL_REGRESSOR_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"


def _strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()


def _coerce_split_sizes(split_sizes: Optional[List[float]]) -> List[float]:
    if not split_sizes:
        return [0.8, 0.1, 0.1]
    if len(split_sizes) != 3:
        raise InvalidPredictionInputError("split_sizes must contain exactly 3 values: train, val, test.")
    total = float(sum(split_sizes))
    if total <= 0:
        raise InvalidPredictionInputError("split_sizes must sum to a positive value.")
    normalized = [float(value) / total for value in split_sizes]
    if any(value <= 0 for value in normalized):
        raise InvalidPredictionInputError("split_sizes must all be positive.")
    return normalized


class TabICLBackend(PredictionBackend):
    """Prediction backend built around TabICLv2 regressors."""

    backend_name = "tabicl"
    MODEL_EXTENSIONS = (".pkl",)

    def _package_version(self) -> Optional[str]:
        try:
            return importlib.metadata.version("tabicl")
        except importlib.metadata.PackageNotFoundError:
            return None

    def is_available(self) -> bool:
        return importlib.util.find_spec("tabicl") is not None

    def describe_environment(self) -> Dict[str, Any]:
        checkpoint_path = DEFAULT_TABICL_CHECKPOINT_DIR / DEFAULT_TABICL_REGRESSOR_CHECKPOINT
        return {
            "backend_name": self.backend_name,
            "available": self.is_available(),
            "package_version": self._package_version(),
            "checkpoint_dir": str(DEFAULT_TABICL_CHECKPOINT_DIR),
            "default_checkpoint_version": DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
            "default_checkpoint_path": str(checkpoint_path),
            "default_checkpoint_present": checkpoint_path.exists(),
        }

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise InvalidPredictionInputError(f"Model path does not exist: {model_path}")
        if path.suffix not in self.MODEL_EXTENSIONS:
            raise InvalidPredictionInputError(
                f"TabICL model artifact must end with one of {self.MODEL_EXTENSIONS}: {model_path}"
            )
        return path.resolve()

    def _ensure_available(self) -> None:
        if not self.is_available():
            raise BackendNotAvailableError(
                "TabICL backend is not available. Install the optional `tabicl` dependency first. "
                f"Environment snapshot: {self.describe_environment()}"
            )

    def _resolve_checkpoint_config(self, extra_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        extra_args = dict(extra_args or {})
        checkpoint_version = str(
            extra_args.get("checkpoint_version") or DEFAULT_TABICL_REGRESSOR_CHECKPOINT
        )
        checkpoint_dir = Path(
            extra_args.get("checkpoint_dir") or DEFAULT_TABICL_CHECKPOINT_DIR
        ).expanduser()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / checkpoint_version
        allow_auto_download = bool(extra_args.get("allow_auto_download", True))
        return {
            "checkpoint_version": checkpoint_version,
            "checkpoint_dir": checkpoint_dir.resolve(),
            "checkpoint_path": checkpoint_path.resolve(),
            "allow_auto_download": allow_auto_download,
        }

    def _import_tabicl_regressor(self):
        self._ensure_available()
        from tabicl import TabICLRegressor

        return TabICLRegressor

    def _sanitize_train_extra_args(self, extra_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        raw = dict(extra_args or {})
        allowed = {
            "n_estimators",
            "norm_methods",
            "feat_shuffle_method",
            "outlier_threshold",
            "batch_size",
            "kv_cache",
            "model_path",
            "allow_auto_download",
            "checkpoint_version",
            "device",
            "use_amp",
            "use_fa3",
            "offload_mode",
            "disk_offload_dir",
            "random_state",
            "n_jobs",
            "verbose",
            "inference_config",
            "checkpoint_dir",
            "save_model_weights",
            "save_training_data",
            "save_kv_cache",
            "split_sizes",
            "split_type",
            "validation_protocol",
            "feature_columns",
        }
        dropped = sorted(key for key in raw if key not in allowed)
        sanitized = {key: value for key, value in raw.items() if key in allowed}
        if dropped:
            logger.warning(
                "Dropping unsupported TabICL train args for this V1 backend: %s",
                ", ".join(dropped),
            )
        return sanitized

    def _select_feature_columns(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
        extra_args: Dict[str, Any],
    ) -> List[str]:
        explicit = extra_args.get("feature_columns")
        if explicit:
            if isinstance(explicit, str):
                explicit = [explicit]
            missing = [column for column in explicit if column not in df.columns]
            if missing:
                raise InvalidPredictionInputError(f"Requested feature columns are missing: {missing}")
            return list(explicit)

        excluded = set(target_columns)
        numeric_columns = [
            column
            for column in df.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
        ]
        if not numeric_columns:
            raise InvalidPredictionInputError(
                "No numeric feature columns were found for TabICL. "
                "Provide a tabular dataset with numeric feature columns or pass feature_columns explicitly."
            )
        return numeric_columns

    def _compute_regression_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
        residuals = y_true - y_pred
        mse = float((residuals.pow(2)).mean())
        mae = float(residuals.abs().mean())
        rmse = float(math.sqrt(mse))
        centered = y_true - float(y_true.mean())
        ss_tot = float((centered.pow(2)).sum())
        ss_res = float((residuals.pow(2)).sum())
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else None
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "n": int(len(y_true)),
        }

    def _persist_checkpoint_if_possible(self, estimator: Any, destination: Path) -> Optional[str]:
        if destination.exists():
            return str(destination)

        candidates = [
            getattr(estimator, "model_path", None),
            getattr(estimator, "checkpoint_path", None),
            getattr(estimator, "checkpoint_file", None),
            getattr(estimator, "checkpoint", None),
            getattr(estimator, "_model_path", None),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            candidate_path = Path(str(candidate)).expanduser()
            if candidate_path.exists() and candidate_path.is_file():
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate_path, destination)
                    return str(destination)
                except Exception as exc:
                    logger.warning("Could not persist TabICL checkpoint to %s: %s", destination, exc)
                    return None
        return None

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
            raise InvalidPredictionInputError("TabICL V1 does not support predictive uncertainty export.")

        model_path = self.validate_model_path(model_record.model_path)
        try:
            with model_path.open("rb") as fh:
                estimator = pickle.load(fh)
        except Exception as exc:
            raise PredictionExecutionError(f"Could not load TabICL model artifact {model_path}: {exc}") from exc

        with S3.open(input_csv, "r") as fh:
            df = _strip_unnamed_columns(pd.read_csv(fh))

        target_columns = list(model_record.task.target_columns)
        feature_columns = list((model_record.inference_profile or {}).get("feature_columns", []))
        if not feature_columns:
            feature_columns = self._select_feature_columns(df, target_columns, {})

        missing_features = [column for column in feature_columns if column not in df.columns]
        if missing_features:
            raise InvalidPredictionInputError(f"Prediction input is missing feature columns: {missing_features}")

        X = df[feature_columns].copy()
        try:
            y_pred = estimator.predict(X)
        except Exception as exc:
            raise PredictionExecutionError(f"TabICL prediction failed: {exc}") from exc

        output = pd.DataFrame({"prediction": pd.Series(y_pred).astype(float)})
        with S3.open(preds_path, "w") as fh:
            output.to_csv(fh, index=False)

        return {
            "preds_path": preds_path,
            "rows": int(len(output)),
            "feature_columns": feature_columns,
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
        if task.task_type != "regression":
            raise InvalidPredictionInputError("TabICL V1 only supports regression tasks.")
        if len(task.target_columns) != 1:
            raise InvalidPredictionInputError("TabICL V1 requires exactly one target column.")

        sanitized_args = self._sanitize_train_extra_args(extra_args)
        checkpoint_cfg = self._resolve_checkpoint_config(sanitized_args)
        split_sizes = _coerce_split_sizes(sanitized_args.pop("split_sizes", None))
        random_state = int(sanitized_args.get("random_state", 42))
        split_type = str(sanitized_args.get("split_type", "random"))
        validation_protocol = str(sanitized_args.get("validation_protocol", "standard_qsar"))
        if split_type != "random":
            raise InvalidPredictionInputError("TabICL V1 currently supports only split_type='random'.")

        with S3.open(train_csv, "r") as fh:
            dataset = _strip_unnamed_columns(pd.read_csv(fh))

        target_column = task.target_columns[0]
        if target_column not in dataset.columns:
            raise InvalidPredictionInputError(f"Missing target column: {target_column}")

        feature_columns = self._select_feature_columns(dataset, [target_column], sanitized_args)
        working = dataset[feature_columns + [target_column] + [c for c in ("smiles", "Drug_ID") if c in dataset.columns]].copy()
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
        working = working.dropna(subset=[target_column]).reset_index(drop=True)
        if len(working) < 10:
            raise InvalidPredictionInputError("TabICL V1 requires at least 10 rows after target cleanup.")

        train_ratio, val_ratio, test_ratio = split_sizes
        train_val_df, test_df = train_test_split(
            working,
            test_size=test_ratio,
            random_state=random_state,
            shuffle=True,
        )
        relative_val_size = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=random_state,
            shuffle=True,
        )

        X_train = train_df[feature_columns].copy()
        y_train = train_df[target_column].astype(float).copy()
        X_test = test_df[feature_columns].copy()
        y_test = test_df[target_column].astype(float).copy()

        TabICLRegressor = self._import_tabicl_regressor()
        model_path_arg = str(checkpoint_cfg["checkpoint_path"]) if checkpoint_cfg["checkpoint_path"].exists() else None
        init_kwargs = {
            "model_path": model_path_arg,
            "allow_auto_download": checkpoint_cfg["allow_auto_download"],
            "checkpoint_version": checkpoint_cfg["checkpoint_version"],
            "random_state": random_state,
            "verbose": bool(sanitized_args.get("verbose", False)),
        }
        optional_keys = (
            "n_estimators",
            "norm_methods",
            "feat_shuffle_method",
            "outlier_threshold",
            "batch_size",
            "kv_cache",
            "device",
            "use_amp",
            "use_fa3",
            "offload_mode",
            "disk_offload_dir",
            "n_jobs",
            "inference_config",
        )
        for key in optional_keys:
            if key in sanitized_args:
                init_kwargs[key] = sanitized_args[key]

        estimator = TabICLRegressor(**init_kwargs)
        try:
            estimator.fit(X_train, y_train)
            y_pred = pd.Series(estimator.predict(X_test), index=X_test.index, dtype=float)
        except Exception as exc:
            raise PredictionExecutionError(f"TabICL training failed: {exc}") from exc

        checkpoint_path = checkpoint_cfg["checkpoint_path"]
        persisted_checkpoint = self._persist_checkpoint_if_possible(estimator, checkpoint_path)

        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        model_artifact_path = output_path / "tabicl_model.pkl"
        test_predictions_path = output_path / "test_predictions.csv"
        summary_path = output_path / "tabicl_training_summary.json"

        save_model_weights = bool(sanitized_args.get("save_model_weights", True))
        save_training_data = bool(sanitized_args.get("save_training_data", True))
        save_kv_cache = bool(sanitized_args.get("save_kv_cache", False))
        try:
            estimator.save(
                str(model_artifact_path),
                save_model_weights=save_model_weights,
                save_training_data=save_training_data,
                save_kv_cache=save_kv_cache,
            )
        except Exception:
            with model_artifact_path.open("wb") as fh:
                pickle.dump(estimator, fh)

        predictions_df = pd.DataFrame(
            {
                **(
                    {column: test_df[column].reset_index(drop=True) for column in ("Drug_ID", "smiles") if column in test_df.columns}
                ),
                "y_true": y_test.reset_index(drop=True),
                "y_pred": y_pred.reset_index(drop=True),
            }
        )
        with S3.open(str(test_predictions_path), "w") as fh:
            predictions_df.to_csv(fh, index=False)

        metrics = self._compute_regression_metrics(
            predictions_df["y_true"].astype(float),
            predictions_df["y_pred"].astype(float),
        )

        summary = {
            "backend_name": self.backend_name,
            "task_type": task.task_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "split_type": "random",
            "validation_protocol": validation_protocol,
            "split_sizes": split_sizes,
            "random_state": random_state,
            "checkpoint_version": checkpoint_cfg["checkpoint_version"],
            "checkpoint_path": persisted_checkpoint or str(checkpoint_path),
            "checkpoint_present_after_run": checkpoint_path.exists(),
            "metrics": {"test": metrics},
            "model_artifact_path": str(model_artifact_path),
            "test_predictions_path": str(test_predictions_path),
            "output_dir": str(output_path),
        }
        with S3.open(str(summary_path), "w") as fh:
            json.dump(summary, fh, indent=2)

        return {
            "backend_name": self.backend_name,
            "model_path": str(model_artifact_path),
            "output_dir": str(output_path),
            "checkpoint_version": checkpoint_cfg["checkpoint_version"],
            "checkpoint_path": persisted_checkpoint or str(checkpoint_path),
            "metrics": {"test": metrics},
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "target_column": target_column,
            "split_type": "random",
            "validation_protocol": validation_protocol,
            "split_sizes": split_sizes,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "test_predictions_path": str(test_predictions_path),
            "summary_path": str(summary_path),
            "save_model_weights": save_model_weights,
            "save_training_data": save_training_data,
            "save_kv_cache": save_kv_cache,
        }
