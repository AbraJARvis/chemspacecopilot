#!/usr/bin/env python
# coding: utf-8
"""
Toolkit exposing LightGBM-backed tabular QSAR workflows.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from .activity_cliffs import (
    MIN_TRAINING_ROWS,
    build_activity_cliff_comparison_metrics,
    build_random_oof_splits,
    choose_recommended_variant,
    compute_activity_cliff_annotation,
    merge_activity_cliff_args,
    parse_activity_cliff_config,
    strip_activity_cliff_args,
    write_activity_cliff_artifacts,
)
from .ad_builder import build_applicability_domain_from_training_data
from .backend import PredictionTaskSpec
from .chemprop_toolkit import _get_prediction_state, _write_active_training_marker
from .lightgbm_backend import LightGBMBackend, _strip_unnamed_columns as _backend_strip_unnamed_columns
from .qsar_plots import build_activity_cliff_feedback_plots, build_qsar_training_plots
from .qsar_training_policy import (
    assess_protocol_results,
    describe_compute_environment,
    project_now,
    resolve_training_profile,
    resolve_validation_protocol,
    safe_slug,
    summarize_training_durations,
)

logger = logging.getLogger(__name__)


def _strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()


class LightGBMToolkit(Toolkit):
    """Toolkit exposing LightGBM-backed QSAR orchestration for tabular datasets."""

    def __init__(self, backend: Optional[LightGBMBackend] = None):
        super().__init__("lightgbm_prediction")
        self.backend = backend or LightGBMBackend()
        self.register(self.describe_lightgbm_backend)
        self.register(self.describe_lightgbm_environment)
        self.register(self.is_lightgbm_available)
        self.register(self.validate_lightgbm_model_path)
        self.register(self.train_lightgbm_model)
        self.register(self.predict_with_lightgbm_from_csv)

    def is_lightgbm_available(self) -> bool:
        """Return whether the LightGBM backend is available in the current environment."""
        return self.backend.is_available()

    def describe_compute_environment(self) -> Dict[str, Any]:
        return describe_compute_environment()

    def _resolve_training_profile(self, compute_env: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_training_profile(compute_env)

    def _resolve_validation_protocol(
        self,
        *,
        requested_protocol: Optional[str],
        training_profile: str,
    ) -> Dict[str, Any]:
        return resolve_validation_protocol(
            requested_protocol=requested_protocol,
            training_profile=training_profile,
        )

    def _training_defaults_for_profile(self, profile: str) -> Dict[str, Any]:
        base = {
            "split_sizes": [0.8, 0.1, 0.1],
            "split_type": "random",
            "random_state": 42,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
        }
        if profile == "heavy_validation":
            return {
                **base,
                "n_estimators": 1000,
                "early_stopping_rounds": 50,
                "n_jobs": 8,
            }
        if profile == "local_standard":
            return {
                **base,
                "n_estimators": 500,
                "early_stopping_rounds": 50,
                "n_jobs": 4,
            }
        return {
            **base,
            "n_estimators": 300,
            "early_stopping_rounds": 30,
            "n_jobs": 1,
        }

    def _apply_training_profile(
        self,
        extra_args: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        requested = dict(extra_args or {})
        requested_profile = requested.pop("training_profile", None)
        requested_validation_protocol = requested.pop("validation_protocol", None)
        allow_heavy_compute = bool(requested.pop("allow_heavy_compute", False))

        compute_env = self.describe_compute_environment()
        resolved = self._resolve_training_profile(compute_env)
        profile = requested_profile or resolved["profile"]

        if not allow_heavy_compute and profile == "heavy_validation":
            profile = resolved["profile"]

        merged = {
            **self._training_defaults_for_profile(profile),
            **requested,
        }

        if not allow_heavy_compute:
            if profile == "local_light":
                merged["n_estimators"] = min(int(merged.get("n_estimators", 300)), 300)
                merged["early_stopping_rounds"] = min(
                    int(merged.get("early_stopping_rounds", 30)),
                    30,
                )
                merged["n_jobs"] = max(1, int(merged.get("n_jobs", 1)))
            elif profile == "local_standard":
                merged["n_estimators"] = min(int(merged.get("n_estimators", 500)), 500)
                merged["early_stopping_rounds"] = min(
                    int(merged.get("early_stopping_rounds", 50)),
                    50,
                )
                merged["n_jobs"] = max(1, int(merged.get("n_jobs", 4)))

        if profile == "heavy_validation":
            merged["n_estimators"] = max(int(merged.get("n_estimators", 1000)), 1000)
            merged["early_stopping_rounds"] = max(
                int(merged.get("early_stopping_rounds", 50)),
                50,
            )
            merged["n_jobs"] = max(int(merged.get("n_jobs", 8)), 8)

        return {
            "compute_environment": compute_env,
            "training_profile": profile,
            "profile_reason": resolved["reason"],
            "validation_protocol": requested_validation_protocol,
            "extra_args": merged,
        }

    def _resolve_lightgbm_run_artifacts(self, output_dir: Path) -> Dict[str, Path]:
        model_dir = output_dir / "model_0"
        return {
            "best_model_path": model_dir / "best.pkl",
            "test_predictions_path": model_dir / "test_predictions.csv",
            "config_path": output_dir / "config.toml",
            "splits_path": output_dir / "splits.json",
        }

    def _materialize_primary_protocol_artifacts(
        self,
        *,
        root_output_dir: Path,
        primary_run: Dict[str, Any],
    ) -> Dict[str, Optional[str]]:
        root_model_dir = root_output_dir / "model_0"
        root_model_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            primary_run.get("model_path"): root_model_dir / "best.pkl",
            primary_run.get("test_predictions_path"): root_model_dir / "test_predictions.csv",
            primary_run.get("config_path"): root_output_dir / "config.toml",
            primary_run.get("splits_path"): root_output_dir / "splits.json",
        }
        copied: Dict[str, Optional[str]] = {
            "best_model_path": None,
            "test_predictions_path": None,
            "config_path": None,
            "splits_path": None,
        }

        for source_raw, target_path in file_map.items():
            if not source_raw:
                continue
            source_path = Path(str(source_raw)).expanduser()
            if not source_path.exists():
                continue
            if source_path.resolve() != target_path.resolve():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
            if target_path.name == "best.pkl":
                copied["best_model_path"] = str(target_path)
            elif target_path.name == "test_predictions.csv":
                copied["test_predictions_path"] = str(target_path)
            elif target_path.name == "config.toml":
                copied["config_path"] = str(target_path)
            elif target_path.name == "splits.json":
                copied["splits_path"] = str(target_path)
        return copied

    def _build_applicability_domain(
        self,
        *,
        train_csv: str,
        primary_run: Dict[str, Any],
        primary_output_dir: Path,
        task: PredictionTaskSpec,
    ) -> Dict[str, Any]:
        splits_path = Path(primary_run.get("splits_path") or primary_output_dir / "splits.json")
        if not splits_path.exists():
            return {}
        split_payload = json.loads(splits_path.read_text())
        if not split_payload or "train" not in split_payload[0]:
            return {}

        dataset = _strip_unnamed_columns(pd.read_csv(Path(train_csv).expanduser()))
        train_indices = split_payload[0].get("train") or []
        smiles_column = task.smiles_columns[0] if task.smiles_columns else "smiles"
        ad_output_dir = primary_output_dir / "applicability_domain"
        return build_applicability_domain_from_training_data(
            dataset=dataset,
            train_indices=train_indices,
            smiles_column=smiles_column,
            output_dir=str(ad_output_dir),
            model_id=primary_output_dir.name,
        )

    def _summarize_training_resources(
        self,
        *,
        compute_env: Dict[str, Any],
        effective_train_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "execution_env": compute_env.get("execution_env"),
            "cpu_count": compute_env.get("cpu_count"),
            "gpu_available": compute_env.get("gpu_available"),
            "gpu_count": compute_env.get("gpu_count"),
            "gpu_name": compute_env.get("gpu_name"),
            "memory_gb_total": compute_env.get("memory_gb_total"),
            "n_estimators": effective_train_args.get("n_estimators"),
            "n_jobs": effective_train_args.get("n_jobs"),
            "num_leaves": effective_train_args.get("num_leaves"),
            "learning_rate": effective_train_args.get("learning_rate"),
            "device_type": effective_train_args.get("device_type"),
        }

    def describe_lightgbm_backend(self) -> Dict[str, Any]:
        """Describe the LightGBM backend defaults and current runtime support."""
        description = self.backend.describe_environment()
        description.update(
            {
                "default_task_type": "regression",
                "default_target_scope": "single_target",
                "default_representations": [
                    "morgan_only",
                    "rdkit_basic_only",
                    "morgan_rdkit_basic",
                    "morgan_rdkit_all",
                ],
            }
        )
        return description

    def describe_lightgbm_environment(self) -> Dict[str, Any]:
        """Alias for backend/environment inspection."""
        return self.describe_lightgbm_backend()

    def validate_lightgbm_model_path(self, model_path: str) -> Dict[str, Any]:
        """Validate a trained LightGBM model artifact path."""
        resolved = self.backend.validate_model_path(model_path)
        return {
            "model_path": str(resolved),
            "exists": resolved.exists(),
            "suffix": resolved.suffix,
        }

    def _normalize_json_list_argument(
        self,
        value: Optional[List[Any] | str],
        *,
        argument_name: str,
    ) -> Optional[List[Any]]:
        if value is None:
            return None
        if isinstance(value, str):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError(f"{argument_name} must be a list or a JSON-encoded list.")
            return parsed
        if not isinstance(value, list):
            raise ValueError(f"{argument_name} must be a list or a JSON-encoded list.")
        return value

    def _run_oof_predictions(
        self,
        *,
        train_csv: str,
        target_column: str,
        feature_columns: Optional[List[str]],
        categorical_feature_columns: Optional[List[str]],
        training_policy: Dict[str, Any],
        cliff_config: Any,
        random_state: int,
        output_dir: Path,
    ) -> pd.Series:
        dataset = _strip_unnamed_columns(pd.read_csv(Path(train_csv).expanduser()))
        folds = build_random_oof_splits(
            n_rows=len(dataset),
            n_folds=cliff_config.oof_folds,
            random_state=random_state,
        )
        oof_predictions = pd.Series(index=dataset.index, dtype=float)
        base_args = {
            **strip_activity_cliff_args(training_policy["extra_args"]),
            **self._training_defaults_for_profile("local_light"),
            "feature_columns": feature_columns,
            "categorical_feature_columns": categorical_feature_columns,
            "validation_protocol": "activity_cliff_oof",
        }
        for fold_index, split_payload in enumerate(folds, start=1):
            fold_dir = output_dir / "_activity_cliff_oof" / f"fold_{fold_index}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_result = self.backend.train_model(
                train_csv=train_csv,
                output_dir=str(fold_dir),
                task=PredictionTaskSpec(
                    task_type="regression",
                    smiles_columns=["smiles"],
                    target_columns=[target_column],
                ),
                extra_args={
                    **base_args,
                    "random_state": random_state + fold_index,
                    "split_payload": [split_payload],
                    "split_type": "random",
                },
            )
            preds_path = Path(fold_result["test_predictions_path"]).expanduser()
            predictions = _backend_strip_unnamed_columns(pd.read_csv(preds_path))
            pred_series = pd.to_numeric(predictions[target_column], errors="coerce")
            test_indices = split_payload["test"]
            if len(pred_series) != len(test_indices):
                raise ValueError("Activity-cliff OOF predictions length mismatch for LightGBM.")
            oof_predictions.iloc[test_indices] = pred_series.to_numpy(dtype=float)
        if oof_predictions.isna().any():
            raise ValueError("Activity-cliff OOF predictions were incomplete for LightGBM.")
        return oof_predictions.astype(float)

    def _train_lightgbm_single(
        self,
        *,
        train_csv: str,
        task_type: str,
        output_dir: str,
        target_columns: List[str],
        feature_columns: Optional[List[str]],
        categorical_feature_columns: Optional[List[str]],
        validation_protocol: Optional[str],
        split_type: str,
        split_sizes: Optional[List[float]],
        random_state: int,
        extra_args: Optional[Dict[str, Any]],
        agent: Optional[Agent],
    ) -> Dict[str, Any]:
        normalized_target_columns = list(target_columns)
        normalized_feature_columns = feature_columns
        normalized_categorical_feature_columns = categorical_feature_columns
        normalized_split_sizes = split_sizes

        resolved_output_dir = str(Path(output_dir).expanduser().resolve())
        root_output_path = Path(resolved_output_dir)
        root_output_path.mkdir(parents=True, exist_ok=True)

        requested_extra_args = dict(extra_args or {})
        requested_extra_args.setdefault("feature_columns", normalized_feature_columns)
        requested_extra_args.setdefault(
            "categorical_feature_columns",
            normalized_categorical_feature_columns,
        )
        requested_extra_args.setdefault("split_sizes", normalized_split_sizes)
        requested_extra_args.setdefault("random_state", random_state)
        requested_extra_args.setdefault("split_type", split_type)
        requested_extra_args.setdefault("validation_protocol", validation_protocol)

        training_policy = self._apply_training_profile(requested_extra_args)
        protocol_policy = self._resolve_validation_protocol(
            requested_protocol=training_policy.get("validation_protocol"),
            training_profile=training_policy["training_profile"],
        )
        trained_at = project_now()
        active_marker_path = root_output_path / ".training_in_progress"
        prediction_state = _get_prediction_state(agent) if agent is not None else None

        active_run_record = {
            "status": "running",
            "backend_name": "lightgbm",
            "train_csv": train_csv,
            "output_dir": resolved_output_dir,
            "validation_protocol": protocol_policy["protocol"],
            "training_profile": training_policy["training_profile"],
            "created_at": trained_at.isoformat(),
            "active_marker_path": str(active_marker_path),
            "current_split_label": None,
            "current_split_index": None,
            "total_splits": len(protocol_policy["split_runs"]),
            "progress_message": None,
        }
        if prediction_state is not None:
            prediction_state["active_training_run"] = dict(active_run_record)
        _write_active_training_marker(active_marker_path, active_run_record)

        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=["smiles"],
            target_columns=list(normalized_target_columns),
        )
        split_results: List[Dict[str, Any]] = []
        primary_run: Optional[Dict[str, Any]] = None
        total_started_at = project_now()
        multi_run_protocol = len(protocol_policy["split_runs"]) > 1

        try:
            for run_index, split_run in enumerate(protocol_policy["split_runs"], start=1):
                label = split_run["label"]
                run_output_dir = (
                    root_output_path / f"{safe_slug(label)}_split"
                    if multi_run_protocol
                    else root_output_path
                )
                run_output_dir.mkdir(parents=True, exist_ok=True)
                started_at = project_now()
                run_args = {
                    **training_policy["extra_args"],
                    "feature_columns": normalized_feature_columns,
                    "categorical_feature_columns": normalized_categorical_feature_columns,
                    "split_sizes": normalized_split_sizes,
                    "split_type": split_run["backend_split_type"],
                    "random_state": split_run["seed"],
                    "validation_protocol": protocol_policy["protocol"],
                }

                active_run_record["current_split_label"] = label
                active_run_record["current_split_index"] = run_index
                active_run_record["progress_message"] = (
                    "LightGBM training progress: "
                    f"run {run_index}/{len(protocol_policy['split_runs'])} - {label}"
                )
                if prediction_state is not None:
                    prediction_state["active_training_run"] = dict(active_run_record)
                _write_active_training_marker(active_marker_path, active_run_record)

                single_result = self.backend.train_model(
                    train_csv=train_csv,
                    output_dir=str(run_output_dir),
                    task=task,
                    extra_args=run_args,
                )

                if "scaffold" in label:
                    strategy = "scaffold"
                    strategy_family = "scaffold"
                elif "kmeans" in label:
                    strategy = "cluster_kmeans"
                    strategy_family = "cluster_kmeans"
                elif "random_seed_" in label:
                    strategy = label
                    strategy_family = "random"
                else:
                    strategy = "random"
                    strategy_family = "random"

                completed_at = project_now()
                single_result["strategy"] = strategy
                single_result["strategy_family"] = strategy_family
                single_result["strategy_label"] = label
                single_result["backend_split_type"] = split_run["backend_split_type"]
                single_result["seed"] = split_run["seed"]
                single_result["validation_protocol"] = protocol_policy["protocol"]
                single_result["output_dir"] = str(run_output_dir)
                single_result["started_at"] = single_result.get("started_at") or started_at.isoformat()
                single_result["completed_at"] = single_result.get("completed_at") or completed_at.isoformat()
                single_result["duration_seconds"] = single_result.get("duration_seconds") or round(
                    (completed_at - started_at).total_seconds(), 3
                )
                split_results.append(single_result)

                if split_run.get("primary") or primary_run is None:
                    primary_run = single_result
        finally:
            active_run_record["status"] = "completed" if primary_run is not None else "failed"
            active_run_record["completed_at"] = project_now().isoformat()
            if prediction_state is not None:
                prediction_state["active_training_run"] = None
            _write_active_training_marker(active_marker_path, active_run_record)

        if primary_run is None:
            raise ValueError("LightGBM validation protocol did not produce a primary run.")

        root_artifacts = self._materialize_primary_protocol_artifacts(
            root_output_dir=root_output_path,
            primary_run=primary_run,
        )
        ad_summary = self._build_applicability_domain(
            train_csv=train_csv,
            primary_run=primary_run,
            primary_output_dir=root_output_path,
            task=task,
        )
        plot_artifacts: Dict[str, str] = {}
        target_column = task.target_columns[0] if task.target_columns else None
        if target_column and root_artifacts.get("splits_path") and root_artifacts.get("test_predictions_path"):
            plots_output_dir = root_output_path / "artifacts" / "plots"
            try:
                plot_artifacts = build_qsar_training_plots(
                    train_csv=train_csv,
                    split_results=[
                        {
                            **item,
                            "splits_path": (
                                root_artifacts["splits_path"]
                                if item is primary_run
                                else item.get("splits_path")
                            ),
                            "test_predictions_path": (
                                root_artifacts["test_predictions_path"]
                                if item is primary_run
                                else item.get("test_predictions_path")
                            ),
                        }
                        for item in split_results
                    ],
                    primary_run={
                        **primary_run,
                        "splits_path": root_artifacts.get("splits_path") or primary_run.get("splits_path"),
                        "test_predictions_path": root_artifacts.get("test_predictions_path")
                        or primary_run.get("test_predictions_path"),
                    },
                    output_dir=str(plots_output_dir),
                    target_column=target_column,
                )
            except Exception:
                plot_artifacts = {}

        validation_assessment = assess_protocol_results(split_results)
        total_completed_at = project_now()
        result = dict(primary_run)
        result["output_dir"] = resolved_output_dir
        result["model_path"] = root_artifacts.get("best_model_path") or primary_run.get("model_path")
        result["summary_path"] = str(root_output_path / "cs_copilot_training_summary.json")
        result["config_path"] = root_artifacts.get("config_path") or primary_run.get("config_path")
        result["splits_path"] = root_artifacts.get("splits_path") or primary_run.get("splits_path")
        result["test_predictions_path"] = root_artifacts.get("test_predictions_path") or primary_run.get(
            "test_predictions_path"
        )
        result["validation_protocol"] = protocol_policy["protocol"]
        result["validation_protocol_reason"] = protocol_policy["reason"]
        result["split_results"] = split_results
        result["validation_assessment"] = validation_assessment
        result["compute_environment"] = training_policy["compute_environment"]
        result["training_profile"] = training_policy["training_profile"]
        result["profile_reason"] = training_policy["profile_reason"]
        result["effective_train_args"] = {
            **training_policy["extra_args"],
            "device_type": primary_run.get("effective_train_args", {}).get("device_type"),
        }
        result["training_resources"] = self._summarize_training_resources(
            compute_env=training_policy["compute_environment"],
            effective_train_args=result["effective_train_args"],
        )
        result["training_durations"] = summarize_training_durations(
            split_results=split_results,
            total_started_at=total_started_at,
            total_completed_at=total_completed_at,
        )
        result["applicability_domain"] = ad_summary
        result["plot_artifacts"] = plot_artifacts
        result["trained_at"] = trained_at.isoformat()
        result["trained_date"] = trained_at.strftime("%d/%m/%Y")
        result["trained_time"] = trained_at.strftime("%H:%M:%S")
        result["train_csv"] = train_csv
        result["target_columns"] = list(normalized_target_columns)
        result["feature_columns"] = list(
            normalized_feature_columns or (primary_run.get("feature_columns") or [])
        )
        result["categorical_feature_columns"] = list(
            normalized_categorical_feature_columns
            or (primary_run.get("categorical_feature_columns") or [])
        )
        result["canonical_summary_path"] = result["summary_path"]

        summary_path = Path(result["summary_path"])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2) + "\n")

        if prediction_state is not None:
            prediction_state["training_runs"].append(
                {
                    "train_csv": train_csv,
                    "output_dir": resolved_output_dir,
                    "task_type": task_type,
                    "smiles_columns": task.smiles_columns,
                    "target_columns": task.target_columns,
                    "validation_protocol": protocol_policy["protocol"],
                    "training_profile": training_policy["training_profile"],
                    "split_runs": [
                        {
                            "label": item["strategy_label"],
                            "strategy": item["strategy"],
                            "strategy_family": item.get("strategy_family"),
                            "output_dir": item["output_dir"],
                            "seed": item["seed"],
                        }
                        for item in split_results
                    ],
                }
            )
            prediction_state["active_training_run"] = None

        return result

    def train_lightgbm_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        target_columns: List[str] | str,
        feature_columns: Optional[List[str] | str] = None,
        categorical_feature_columns: Optional[List[str] | str] = None,
        validation_protocol: Optional[str] = None,
        split_type: str = "random",
        split_sizes: Optional[List[float] | str] = None,
        random_state: int = 42,
        activity_cliff_feedback: bool = False,
        activity_cliff_feedback_loops: int = 0,
        activity_cliff_step_percentile: float = 5.0,
        activity_cliff_similarity_threshold: float = 0.70,
        activity_cliff_k_neighbors: int = 10,
        activity_cliff_oof_folds: int = 5,
        activity_cliff_index: str = "sali",
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Train a LightGBM regressor with QSAR validation protocols."""
        normalized_target_columns = self._normalize_json_list_argument(
            target_columns,
            argument_name="target_columns",
        ) or []
        normalized_feature_columns = self._normalize_json_list_argument(
            feature_columns,
            argument_name="feature_columns",
        )
        normalized_categorical_feature_columns = self._normalize_json_list_argument(
            categorical_feature_columns,
            argument_name="categorical_feature_columns",
        )
        normalized_split_sizes = self._normalize_json_list_argument(
            split_sizes,
            argument_name="split_sizes",
        )
        merged_extra_args = merge_activity_cliff_args(
            extra_args=extra_args,
            activity_cliff_feedback=activity_cliff_feedback,
            activity_cliff_feedback_loops=activity_cliff_feedback_loops,
            activity_cliff_step_percentile=activity_cliff_step_percentile,
            activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
            activity_cliff_k_neighbors=activity_cliff_k_neighbors,
            activity_cliff_oof_folds=activity_cliff_oof_folds,
            activity_cliff_index=activity_cliff_index,
        )
        feedback_config = parse_activity_cliff_config(merged_extra_args)

        resolved_output_dir = str(Path(output_dir).expanduser().resolve())
        root_output_path = Path(resolved_output_dir)
        root_output_path.mkdir(parents=True, exist_ok=True)
        base_extra_args = dict(merged_extra_args)
        training_policy = self._apply_training_profile(base_extra_args)
        target_column = normalized_target_columns[0]
        base_dataset = _strip_unnamed_columns(pd.read_csv(Path(train_csv).expanduser()))

        ac_supported = (
            task_type == "regression"
            and len(normalized_target_columns) == 1
            and "smiles" in base_dataset.columns
        )
        if not ac_supported:
            return self._train_lightgbm_single(
                train_csv=train_csv,
                task_type=task_type,
                output_dir=output_dir,
                target_columns=normalized_target_columns,
                feature_columns=normalized_feature_columns,
                categorical_feature_columns=normalized_categorical_feature_columns,
                validation_protocol=validation_protocol,
                split_type=split_type,
                split_sizes=normalized_split_sizes,
                random_state=random_state,
                extra_args=strip_activity_cliff_args(merged_extra_args),
                agent=agent,
            )

        oof_predictions = self._run_oof_predictions(
            train_csv=train_csv,
            target_column=target_column,
            feature_columns=normalized_feature_columns,
            categorical_feature_columns=normalized_categorical_feature_columns,
            training_policy=training_policy,
            cliff_config=feedback_config,
            random_state=random_state,
            output_dir=root_output_path,
        )
        annotated_df = compute_activity_cliff_annotation(
            dataset=base_dataset,
            smiles_column="smiles",
            target_column=target_column,
            oof_predictions=oof_predictions,
            similarity_threshold=feedback_config.similarity_threshold,
            k_neighbors=feedback_config.k_neighbors,
            sali_flag_threshold=feedback_config.sali_flag_threshold,
            index_name=feedback_config.index_name,
        )
        cliff_summary = write_activity_cliff_artifacts(
            annotated_df=annotated_df,
            target_column=target_column,
            output_dir=str(root_output_path / "activity_cliff_feedback"),
            loops=feedback_config.loops,
            min_training_rows=MIN_TRAINING_ROWS,
        )

        variants: List[Dict[str, Any]] = []
        baseline_output_dir = (
            resolved_output_dir if not feedback_config.feedback_enabled else str(root_output_path / "baseline_loop_0")
        )
        baseline_result = self._train_lightgbm_single(
            train_csv=train_csv,
            task_type=task_type,
            output_dir=baseline_output_dir,
            target_columns=normalized_target_columns,
            feature_columns=normalized_feature_columns,
            categorical_feature_columns=normalized_categorical_feature_columns,
            validation_protocol=validation_protocol,
            split_type=split_type,
            split_sizes=normalized_split_sizes,
            random_state=random_state,
            extra_args=strip_activity_cliff_args(merged_extra_args),
            agent=agent,
        )
        baseline_variant = {
            "variant_id": "baseline_loop_0",
            "loop_index": 0,
            "removed_tiers": [],
            "removed_count": 0,
            "remaining_rows": int(len(base_dataset)),
            "filtered_training_csv": train_csv,
            "training_result": baseline_result,
            "persisted_model_id": str(
                baseline_result.get("model_id")
                or Path(baseline_result.get("model_path") or baseline_output_dir).stem
            ),
        }
        variants.append(baseline_variant)
        for item in cliff_summary["variants"]:
            training_result = self._train_lightgbm_single(
                train_csv=item["filtered_training_csv"],
                task_type=task_type,
                output_dir=str(root_output_path / item["variant_id"]),
                target_columns=normalized_target_columns,
                feature_columns=normalized_feature_columns,
                categorical_feature_columns=normalized_categorical_feature_columns,
                validation_protocol=validation_protocol,
                split_type=split_type,
                split_sizes=normalized_split_sizes,
                random_state=random_state,
                extra_args=strip_activity_cliff_args(merged_extra_args),
                agent=agent,
            )
            variants.append(
                {
                    **item,
                    "training_result": training_result,
                    "persisted_model_id": str(
                        training_result.get("model_id")
                        or Path(training_result.get("model_path") or item["variant_id"]).stem
                    ),
                }
            )

        recommended_variant = choose_recommended_variant(variants)
        feedback_plots = build_activity_cliff_feedback_plots(
            train_csv=train_csv,
            target_column=target_column,
            annotated_training_csv=cliff_summary["annotated_training_csv"],
            variants=variants,
            output_dir=str(root_output_path / "activity_cliff_feedback" / "plots"),
        )
        comparison_metrics = build_activity_cliff_comparison_metrics(variants)
        selected_result = dict(recommended_variant["training_result"])
        selected_result["activity_cliff_feedback"] = {
            "enabled": True,
            "mode": feedback_config.mode,
            "index_name": feedback_config.index_name,
            "loops_requested": feedback_config.loops,
            "annotated_training_csv": cliff_summary["annotated_training_csv"],
            "flagged_count": cliff_summary["flagged_count"],
            "priority_counts": cliff_summary["priority_counts"],
            "tiering_policy": cliff_summary["tiering_policy"],
            "variants": variants,
            "recommended_variant": recommended_variant["variant_id"],
            "comparison_metrics": comparison_metrics,
            "plot_artifacts": feedback_plots,
            "summary_path": cliff_summary["summary_path"],
            "warnings": cliff_summary.get("warnings") or [],
        }
        selected_result["plot_artifacts"] = {
            **(selected_result.get("plot_artifacts") or {}),
            **feedback_plots,
        }
        selected_result["summary_path"] = str(root_output_path / "cs_copilot_training_summary.json")
        selected_result["canonical_summary_path"] = selected_result["summary_path"]
        summary_path = Path(selected_result["summary_path"])
        summary_path.write_text(json.dumps(selected_result, indent=2) + "\n")
        return selected_result

    def predict_with_lightgbm_from_csv(
        self,
        input_csv: str,
        model_path: str,
        preds_path: str,
        target_columns: Optional[List[str] | str] = None,
        feature_columns: Optional[List[str] | str] = None,
        categorical_feature_columns: Optional[List[str] | str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run LightGBM batch prediction from a tabular CSV input file."""
        normalized_target_columns = self._normalize_json_list_argument(
            target_columns,
            argument_name="target_columns",
        ) or []
        normalized_feature_columns = self._normalize_json_list_argument(
            feature_columns,
            argument_name="feature_columns",
        ) or []
        normalized_categorical_feature_columns = self._normalize_json_list_argument(
            categorical_feature_columns,
            argument_name="categorical_feature_columns",
        ) or []

        from .backend import PredictionModelRecord

        model_record = PredictionModelRecord(
            model_id=Path(model_path).stem,
            backend_name=self.backend.backend_name,
            model_path=model_path,
            task=PredictionTaskSpec(
                task_type="regression",
                smiles_columns=["smiles"],
                target_columns=list(normalized_target_columns),
            ),
            inference_profile={
                "feature_columns": list(normalized_feature_columns),
                "categorical_feature_columns": list(normalized_categorical_feature_columns),
            },
        )
        return self.backend.predict_from_csv(
            input_csv=input_csv,
            model_record=model_record,
            preds_path=preds_path,
            extra_args=extra_args,
        )
