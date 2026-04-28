#!/usr/bin/env python
# coding: utf-8
"""
Toolkit exposing TabICLv2-backed tabular QSAR workflows.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from .ad_builder import build_applicability_domain_from_training_data
from .backend import PredictionExecutionError, PredictionTaskSpec
from .chemprop_toolkit import _get_prediction_state, _write_active_training_marker
from .qsar_plots import build_qsar_training_plots
from .qsar_training_policy import (
    assess_protocol_results,
    describe_compute_environment,
    project_now,
    resolve_training_profile,
    resolve_validation_protocol,
    safe_slug,
    summarize_training_durations,
)
from .tabicl_backend import (
    DEFAULT_TABICL_CHECKPOINT_DIR,
    DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
    TabICLBackend,
)

logger = logging.getLogger(__name__)


def _strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()


class TabICLToolkit(Toolkit):
    """Toolkit exposing Chemprop-style QSAR orchestration for TabICL."""

    def __init__(self, backend: Optional[TabICLBackend] = None):
        super().__init__("tabicl_prediction")
        self.backend = backend or TabICLBackend()
        self.register(self.describe_tabicl_backend)
        self.register(self.describe_tabicl_environment)
        self.register(self.is_tabicl_available)
        self.register(self.validate_tabicl_model_path)
        self.register(self.validate_tabicl_checkpoint_path)
        self.register(self.train_tabicl_model)
        self.register(self.predict_with_tabicl_from_csv)

    def is_tabicl_available(self) -> bool:
        """Return whether the TabICL backend is available in the current environment."""
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
            "n_jobs": 0,
            "verbose": False,
        }
        if profile == "heavy_validation":
            return {
                **base,
                "batch_size": 64,
                "n_estimators": 8,
                "kv_cache": False,
                "n_jobs": 8,
            }
        if profile == "local_standard":
            return {
                **base,
                "batch_size": 64,
                "n_estimators": 4,
                "kv_cache": False,
            }
        return {
            **base,
            "batch_size": 32,
            "n_estimators": 4,
            "kv_cache": False,
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
                merged["batch_size"] = min(int(merged.get("batch_size", 32)), 32)
                merged["n_estimators"] = min(int(merged.get("n_estimators", 4)), 4)
                merged["n_jobs"] = 0
                merged["kv_cache"] = False
            elif profile == "local_standard":
                merged["batch_size"] = min(int(merged.get("batch_size", 64)), 64)
                merged["n_estimators"] = min(int(merged.get("n_estimators", 4)), 4)
                merged["n_jobs"] = 0

        if profile == "heavy_validation":
            merged["batch_size"] = max(int(merged.get("batch_size", 64)), 64)
            merged["n_estimators"] = max(int(merged.get("n_estimators", 8)), 8)
            merged["n_jobs"] = max(int(merged.get("n_jobs", 8)), 8)
            merged["kv_cache"] = bool(merged.get("kv_cache", False))

        return {
            "compute_environment": compute_env,
            "training_profile": profile,
            "profile_reason": resolved["reason"],
            "validation_protocol": requested_validation_protocol,
            "extra_args": merged,
        }

    def _resolve_tabicl_run_artifacts(self, output_dir: Path) -> Dict[str, Path]:
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
            "batch_size": effective_train_args.get("batch_size"),
            "n_estimators": effective_train_args.get("n_estimators"),
            "n_jobs": effective_train_args.get("n_jobs"),
            "kv_cache": effective_train_args.get("kv_cache"),
        }

    def describe_tabicl_backend(self) -> Dict[str, Any]:
        """Describe the TabICL backend defaults and current runtime support."""
        description = self.backend.describe_environment()
        description.update(
            {
                "default_task_type": "regression",
                "default_checkpoint_dir": str(DEFAULT_TABICL_CHECKPOINT_DIR),
                "default_checkpoint_version": DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
                "supported_validation_protocols": [
                    "fast_local",
                    "standard_qsar",
                    "robust_qsar",
                    "challenging_qsar",
                ],
                "supported_split_families": ["random", "scaffold", "cluster_kmeans"],
                "default_tabular_feature_policy": {
                    "default_representation": "morgan_plus_rdkit",
                    "local_light": {"rdkit_descriptor_set": "basic"},
                    "local_standard": {"rdkit_descriptor_set": "basic"},
                    "heavy_validation": {"rdkit_descriptor_set": "all"},
                    "explicit_user_override": True,
                },
                "notes": [
                    "Supports Chemprop-style QSAR protocol names and split families.",
                    "Default TabICL tabular features combine Morgan fingerprints with RDKit descriptors.",
                    "Lighter profiles default to RDKit basic; heavy GPU-capable profiles default to RDKit all.",
                    "If the user explicitly requests Morgan only or RDKit only, that override should win.",
                    "The `.ckpt` checkpoint is a backend resource, not a trained model artifact.",
                    "`validate_tabicl_model_path` is intended for saved trained models such as `.pkl`.",
                ],
            }
        )
        return description

    def describe_tabicl_environment(self) -> Dict[str, Any]:
        """Return a lightweight runtime snapshot for TabICL availability."""
        snapshot = self.backend.describe_environment()
        snapshot["compute_environment"] = self.describe_compute_environment()
        return snapshot

    def validate_tabicl_model_path(self, model_path: str) -> Dict[str, Any]:
        """Validate a saved TabICL model artifact path."""
        resolved = self.backend.validate_model_path(model_path)
        return {
            "model_path": str(resolved),
            "exists": resolved.exists(),
            "suffix": resolved.suffix,
        }

    def validate_tabicl_checkpoint_path(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate the persisted TabICL base checkpoint path."""
        path = Path(checkpoint_path or (DEFAULT_TABICL_CHECKPOINT_DIR / DEFAULT_TABICL_REGRESSOR_CHECKPOINT))
        resolved = path.expanduser().resolve()
        if resolved.suffix != ".ckpt":
            raise ValueError(f"TabICL checkpoint must end with '.ckpt'. Received: {resolved}")
        return {
            "checkpoint_path": str(resolved),
            "exists": resolved.exists(),
            "suffix": resolved.suffix,
            "is_backend_resource": True,
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
        return list(value)

    def _build_active_run_record(
        self,
        *,
        train_csv: str,
        resolved_output_dir: str,
        protocol_policy: Dict[str, Any],
        training_policy: Dict[str, Any],
        trained_at,
        active_marker_path: Path,
        worker_pid: Optional[int] = None,
        worker_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "status": "running",
            "backend_name": "tabicl",
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
            "worker_pid": worker_pid,
            "worker_status": worker_status,
        }

    def _sync_training_run_state_from_result(
        self,
        *,
        prediction_state: Optional[Dict[str, Any]],
        train_csv: str,
        resolved_output_dir: str,
        task_type: str,
        task: PredictionTaskSpec,
        protocol_policy: Dict[str, Any],
        training_policy: Dict[str, Any],
        split_results: List[Dict[str, Any]],
    ) -> None:
        if prediction_state is None:
            return
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

    def _run_protocol_training(
        self,
        *,
        train_csv: str,
        task_type: str,
        resolved_output_dir: str,
        target_columns: List[str],
        feature_columns: Optional[List[str]],
        split_type: str,
        split_sizes: Optional[List[float]],
        random_state: int,
        extra_args: Optional[Dict[str, Any]],
        prediction_state: Optional[Dict[str, Any]] = None,
        active_marker_path: Optional[Path] = None,
        worker_pid: Optional[int] = None,
        worker_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        root_output_path = Path(resolved_output_dir)
        root_output_path.mkdir(parents=True, exist_ok=True)
        trained_at = project_now()

        requested_extra_args = dict(extra_args or {})
        requested_extra_args.setdefault("feature_columns", feature_columns)
        requested_extra_args.setdefault("split_sizes", split_sizes)
        requested_extra_args.setdefault("random_state", random_state)
        requested_extra_args.setdefault("split_type", split_type)
        requested_extra_args.setdefault("validation_protocol", requested_extra_args.get("validation_protocol"))

        training_policy = self._apply_training_profile(requested_extra_args)
        protocol_policy = self._resolve_validation_protocol(
            requested_protocol=training_policy.get("validation_protocol"),
            training_profile=training_policy["training_profile"],
        )
        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=["smiles"],
            target_columns=list(target_columns),
        )

        marker_path = active_marker_path or (root_output_path / ".training_in_progress")
        active_run_record = self._build_active_run_record(
            train_csv=train_csv,
            resolved_output_dir=resolved_output_dir,
            protocol_policy=protocol_policy,
            training_policy=training_policy,
            trained_at=trained_at,
            active_marker_path=marker_path,
            worker_pid=worker_pid,
            worker_status=worker_status,
        )
        if prediction_state is not None:
            prediction_state["active_training_run"] = dict(active_run_record)
        _write_active_training_marker(marker_path, active_run_record)

        split_results: List[Dict[str, Any]] = []
        primary_run: Optional[Dict[str, Any]] = None
        total_started_at = project_now()
        multi_run_protocol = len(protocol_policy["split_runs"]) > 1

        try:
            for run_index, split_run in enumerate(protocol_policy["split_runs"], start=1):
                label = split_run["label"]
                run_output_dir = (
                    root_output_path / f"{safe_slug(label)}_split" if multi_run_protocol else root_output_path
                )
                run_output_dir.mkdir(parents=True, exist_ok=True)
                started_at = project_now()
                run_args = {
                    **training_policy["extra_args"],
                    "feature_columns": feature_columns,
                    "split_sizes": split_sizes,
                    "split_type": split_run["backend_split_type"],
                    "random_state": split_run["seed"],
                    "validation_protocol": protocol_policy["protocol"],
                    "heartbeat_path": str(marker_path),
                    "heartbeat_label": label,
                    "heartbeat_run_index": run_index,
                    "heartbeat_total_runs": len(protocol_policy["split_runs"]),
                }
                run_args.setdefault("heartbeat_seconds", 120.0)
                run_args.setdefault("disk_offload_dir", str((run_output_dir / "disk_offload").resolve()))

                active_run_record["current_split_label"] = label
                active_run_record["current_split_index"] = run_index
                active_run_record["progress_message"] = (
                    f"TabICL training progress: run {run_index}/{len(protocol_policy['split_runs'])} - {label}"
                )
                active_run_record["worker_status"] = "running"
                if prediction_state is not None:
                    prediction_state["active_training_run"] = dict(active_run_record)
                _write_active_training_marker(marker_path, active_run_record)

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
            active_run_record["worker_status"] = "completed" if primary_run is not None else "failed"
            if prediction_state is not None:
                prediction_state["active_training_run"] = None
            _write_active_training_marker(marker_path, active_run_record)

        if primary_run is None:
            raise ValueError("TabICL validation protocol did not produce a primary run.")

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
        result["effective_train_args"] = training_policy["extra_args"]
        result["training_resources"] = self._summarize_training_resources(
            compute_env=training_policy["compute_environment"],
            effective_train_args=training_policy["extra_args"],
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
        result["target_columns"] = list(target_columns)
        result["feature_columns"] = list(feature_columns or (primary_run.get("feature_columns") or []))
        result["canonical_summary_path"] = result["summary_path"]

        summary_path = Path(result["summary_path"])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2) + "\n")

        self._sync_training_run_state_from_result(
            prediction_state=prediction_state,
            train_csv=train_csv,
            resolved_output_dir=resolved_output_dir,
            task_type=task_type,
            task=task,
            protocol_policy=protocol_policy,
            training_policy=training_policy,
            split_results=split_results,
        )
        return result

    def _write_worker_job(
        self,
        *,
        job_dir: Path,
        payload: Dict[str, Any],
    ) -> Path:
        job_dir.mkdir(parents=True, exist_ok=True)
        for stale_name in ("result.json", "error.json", "worker.log"):
            stale_path = job_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
        job_path = job_dir / "job.json"
        job_path.write_text(json.dumps(payload, indent=2) + "\n")
        return job_path

    def _run_training_worker(
        self,
        *,
        job_path: Path,
        worker_log_path: Path,
    ) -> Dict[str, Any]:
        result_path = job_path.parent / "result.json"
        error_path = job_path.parent / "error.json"
        command = [
            sys.executable,
            "-m",
            "cs_copilot.tools.prediction.tabicl_train_worker",
            str(job_path),
        ]
        worker_log_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.monotonic()
        with worker_log_path.open("w", encoding="utf-8") as log_fh:
            process = subprocess.Popen(
                command,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )

            result_seen_at: Optional[float] = None
            while True:
                return_code = process.poll()
                if result_path.exists() and result_seen_at is None:
                    result_seen_at = time.monotonic()
                if return_code is not None:
                    break
                if result_seen_at is not None and (time.monotonic() - result_seen_at) >= 2.0:
                    logger.warning(
                        "TabICL worker produced result.json but remained alive; terminating worker pid=%s.",
                        process.pid,
                    )
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        logger.warning("TabICL worker did not terminate cleanly; killing pid=%s.", process.pid)
                        process.kill()
                        process.wait(timeout=5.0)
                    break
                time.sleep(0.5)

        duration_seconds = round(time.monotonic() - start_time, 3)
        if result_path.exists():
            result = json.loads(result_path.read_text())
            result.setdefault("worker_duration_seconds", duration_seconds)
            return result

        if error_path.exists():
            payload = json.loads(error_path.read_text())
            message = payload.get("error_message") or "TabICL worker failed."
            traceback_text = payload.get("traceback")
            if traceback_text:
                raise RuntimeError(f"{message}\n{traceback_text}")
            raise RuntimeError(message)

        log_excerpt = worker_log_path.read_text(encoding="utf-8") if worker_log_path.exists() else ""
        raise RuntimeError(
            "TabICL worker exited without producing result.json or error.json. "
            f"worker_log={worker_log_path} details={log_excerpt[-4000:]}"
        )

    def train_tabicl_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        target_columns: List[str] | str,
        feature_columns: Optional[List[str] | str] = None,
        validation_protocol: Optional[str] = None,
        split_type: str = "random",
        split_sizes: Optional[List[float] | str] = None,
        random_state: int = 42,
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Train a TabICLv2 regressor with Chemprop-style QSAR validation protocols."""
        normalized_target_columns = self._normalize_json_list_argument(
            target_columns,
            argument_name="target_columns",
        ) or []
        normalized_feature_columns = self._normalize_json_list_argument(
            feature_columns,
            argument_name="feature_columns",
        )
        normalized_split_sizes = self._normalize_json_list_argument(
            split_sizes,
            argument_name="split_sizes",
        )

        resolved_output_dir = str(Path(output_dir).expanduser().resolve())
        root_output_path = Path(resolved_output_dir)
        root_output_path.mkdir(parents=True, exist_ok=True)

        requested_extra_args = dict(extra_args or {})
        requested_extra_args.setdefault("feature_columns", normalized_feature_columns)
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

        active_run_record = self._build_active_run_record(
            train_csv=train_csv,
            resolved_output_dir=resolved_output_dir,
            protocol_policy=protocol_policy,
            training_policy=training_policy,
            trained_at=trained_at,
            active_marker_path=active_marker_path,
            worker_status="starting",
        )
        if prediction_state is not None:
            prediction_state["active_training_run"] = dict(active_run_record)
        _write_active_training_marker(active_marker_path, active_run_record)

        job_dir = root_output_path / "_worker_job"
        worker_log_path = job_dir / "worker.log"
        job_payload = {
            "train_csv": train_csv,
            "task_type": task_type,
            "output_dir": resolved_output_dir,
            "target_columns": normalized_target_columns,
            "feature_columns": normalized_feature_columns,
            "split_type": split_type,
            "split_sizes": normalized_split_sizes,
            "random_state": random_state,
            "extra_args": requested_extra_args,
        }
        job_path = self._write_worker_job(job_dir=job_dir, payload=job_payload)

        try:
            result = self._run_training_worker(
                job_path=job_path,
                worker_log_path=worker_log_path,
            )
        except Exception as exc:
            active_run_record["status"] = "failed"
            active_run_record["worker_status"] = "failed"
            active_run_record["completed_at"] = project_now().isoformat()
            if prediction_state is not None:
                prediction_state["active_training_run"] = None
            _write_active_training_marker(active_marker_path, active_run_record)
            raise PredictionExecutionError(f"TabICL worker execution failed: {exc}") from exc

        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=["smiles"],
            target_columns=list(normalized_target_columns),
        )
        self._sync_training_run_state_from_result(
            prediction_state=prediction_state,
            train_csv=train_csv,
            resolved_output_dir=resolved_output_dir,
            task_type=task_type,
            task=task,
            protocol_policy=protocol_policy,
            training_policy=training_policy,
            split_results=list(result.get("split_results") or []),
        )
        if prediction_state is not None:
            prediction_state["active_training_run"] = None
        return result

    def predict_with_tabicl_from_csv(
        self,
        input_csv: str,
        model_path: str,
        preds_path: str,
        target_columns: Optional[List[str] | str] = None,
        feature_columns: Optional[List[str] | str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run TabICL batch prediction from a tabular CSV input file."""
        if isinstance(target_columns, str):
            parsed = json.loads(target_columns)
            if not isinstance(parsed, list):
                raise ValueError("target_columns must be a list or a JSON-encoded list.")
            target_columns = parsed
        if isinstance(feature_columns, str):
            parsed = json.loads(feature_columns)
            if not isinstance(parsed, list):
                raise ValueError("feature_columns must be a list or a JSON-encoded list.")
            feature_columns = parsed

        model_record_extra = dict(extra_args or {})
        if feature_columns:
            model_record_extra.setdefault("feature_columns", feature_columns)

        from .backend import PredictionModelRecord

        model_record = PredictionModelRecord(
            model_id=Path(model_path).stem,
            backend_name=self.backend.backend_name,
            model_path=model_path,
            task=PredictionTaskSpec(
                task_type="regression",
                smiles_columns=["smiles"],
                target_columns=list(target_columns or []),
            ),
            inference_profile={"feature_columns": list(feature_columns or [])},
        )
        return self.backend.predict_from_csv(
            input_csv=input_csv,
            model_record=model_record,
            preds_path=preds_path,
            extra_args=model_record_extra,
        )
