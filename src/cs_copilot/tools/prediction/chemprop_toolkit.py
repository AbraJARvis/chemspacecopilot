#!/usr/bin/env python
# coding: utf-8
"""
Toolkit for model registration, prediction, and future Chemprop training flows.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import torch
from agno.agent import Agent
from agno.tools.toolkit import Toolkit
from scipy.stats import kendalltau, spearmanr

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.standardize import standardize_smiles_column

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
from .backend import PredictionModelRecord, PredictionTaskSpec
from .catalog import DEFAULT_INTERNAL_MODEL_ROOT, PredictionModelCatalog
from .chemprop_backend import ChempropBackend
from .lightgbm_backend import LightGBMBackend
from .qsar_training_policy import (
    QSAR_HARDEST_SPLIT_R2_MIN,
    QSAR_RANDOM_STABILITY_R2_STD_MAX,
    assess_protocol_results,
    coerce_project_timezone,
    describe_compute_environment,
    project_now,
    resolve_training_profile,
    resolve_validation_protocol,
    safe_slug,
    summarize_training_durations,
)
from .qsar_plots import build_activity_cliff_feedback_plots, build_qsar_training_plots
from .tabicl_backend import TabICLBackend

def _get_prediction_state(agent: Agent) -> Dict[str, Any]:
    state = agent.session_state.setdefault("prediction_models", {})
    state.setdefault("registered", {})
    state.setdefault("last_prediction", {})
    state.setdefault("prediction_history", [])
    state.setdefault("catalog_recommendations", {})
    state.setdefault("training_runs", [])
    state.setdefault("active_training_run", None)
    return state


def _prediction_output_path(model_id: str, preds_path: Optional[str] = None) -> Path:
    if preds_path:
        return Path(preds_path).expanduser()
    return (Path(".files") / "prediction_outputs" / f"{model_id}_predictions.csv").resolve()


def _strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()


def _bundle_artifacts(bundle_path: Path, files: List[Path]) -> Path:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in files:
            if file_path.exists():
                zf.write(file_path, arcname=file_path.name)
    return bundle_path


def _relative_posix(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()


def _safe_display_token(value: str) -> str:
    token = value.replace("_", " ").strip()
    return " ".join(part.capitalize() for part in token.split())


def _extract_endpoint_and_dataset(train_csv: Optional[str], fallback_model_id: str) -> tuple[str, str]:
    source = Path(train_csv or fallback_model_id).stem.lower()
    for suffix in ("_curated", "_cleaned", "_dataset", "_training"):
        if source.endswith(suffix):
            source = source[: -len(suffix)]
            break
    parts = [part for part in source.split("_") if part]
    if len(parts) >= 2:
        endpoint = parts[0]
        dataset = "_".join(parts[1:])
    elif len(parts) == 1:
        endpoint = parts[0]
        dataset = "dataset"
    else:
        endpoint = safe_slug(fallback_model_id) or "endpoint"
        dataset = "dataset"
    return safe_slug(endpoint) or "endpoint", safe_slug(dataset) or "dataset"


def _canonical_model_id(
    *,
    endpoint: str,
    dataset: str,
    protocol: str,
    backend: str,
    representation: Optional[str],
    version: str,
    trained_at: datetime,
) -> str:
    version_token = version if str(version).startswith("v") else f"v{version}"
    date_token = trained_at.strftime("%d%m%Y")
    time_token = trained_at.strftime("%H%M%S")
    parts = [
        safe_slug(endpoint),
        safe_slug(dataset),
        safe_slug(protocol),
        safe_slug(backend),
    ]
    if representation:
        parts.append(safe_slug(representation))
    parts.extend([safe_slug(version_token), date_token, time_token])
    return "_".join(parts)


def _canonical_display_name(
    *,
    endpoint: str,
    dataset: str,
    protocol: str,
    backend: str,
    representation: Optional[str],
    version: str,
) -> str:
    version_token = version if str(version).startswith("v") else f"v{version}"
    parts = [
        _safe_display_token(endpoint),
        _safe_display_token(dataset),
        _safe_display_token(protocol),
        _safe_display_token(backend),
    ]
    if representation:
        parts.append(_safe_display_token(representation))
    parts.append(version_token)
    return " ".join(parts).strip()


def _write_active_training_marker(marker_path: Path, payload: Dict[str, Any]) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(payload, indent=2))


def _find_first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except Exception:
            continue
    return None


class ChempropToolkit(Toolkit):
    """Toolkit exposing Chemprop-backed property prediction workflows."""

    def __init__(self, backend: Optional[ChempropBackend] = None):
        super().__init__("chemprop_prediction")
        primary_backend = backend or ChempropBackend()
        tabicl_backend = TabICLBackend()
        lightgbm_backend = LightGBMBackend()
        self.backends = {
            primary_backend.backend_name: primary_backend,
            tabicl_backend.backend_name: tabicl_backend,
            lightgbm_backend.backend_name: lightgbm_backend,
        }
        self.backend = primary_backend
        self.catalog = PredictionModelCatalog.load()
        self.catalog.refresh_from_internal_store(persist=True)
        self.register(self.describe_backend)
        self.register(self.describe_backends)
        self.register(self.describe_catalog)
        self.register(self.list_catalog_models)
        self.register(self.summarize_catalog_model)
        self.register(self.recommend_catalog_model)
        self.register(self.register_catalog_model)
        self.register(self.persist_registered_model)
        self.register(self.register_model)
        self.register(self.list_registered_models)
        self.register(self.summarize_model)
        self.register(self.predict_from_csv)
        self.register(self.predict_from_smiles)
        self.register(self.export_prediction_summary)
        self.register(self.prepare_training_dataset)
        self.register(self.describe_compute_environment)
        self.register(self.train_model)

    def _detect_memory_limit_bytes(self) -> Optional[int]:
        candidates = [
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                raw = path.read_text().strip()
                if not raw or raw == "max":
                    continue
                value = int(raw)
                # Ignore absurdly large “no real limit” cgroup values.
                if value <= 0 or value > 1 << 60:
                    continue
                return value
            except Exception:
                continue
        return None

    def _resolve_chemprop_run_artifacts(self, output_dir: Path) -> Dict[str, Optional[Path]]:
        output_path = output_dir.expanduser().resolve()
        best_model_path = _find_first_existing_path(
            [
                output_path / "model_0" / "best.pt",
                output_path / "replicate_0" / "model_0" / "best.pt",
            ]
        )
        test_predictions_path = _find_first_existing_path(
            [
                output_path / "model_0" / "test_predictions.csv",
                output_path / "replicate_0" / "model_0" / "test_predictions.csv",
            ]
        )
        return {
            "best_model_path": best_model_path,
            "test_predictions_path": test_predictions_path,
            "config_path": output_path / "config.toml",
            "splits_path": output_path / "splits.json",
        }

    def _detect_physical_memory_bytes(self) -> Optional[int]:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
            if isinstance(page_size, int) and isinstance(page_count, int) and page_size > 0 and page_count > 0:
                return page_size * page_count
        except Exception:
            return None
        return None

    def _detect_disk_usage(self, base_path: Optional[Path] = None) -> Dict[str, Optional[float]]:
        target = (base_path or Path.cwd()).resolve()
        try:
            usage = shutil.disk_usage(target)
        except Exception:
            return {
                "disk_path": str(target),
                "disk_gb_total": None,
                "disk_gb_free": None,
                "disk_gb_used": None,
            }

        gib = 1024**3
        return {
            "disk_path": str(target),
            "disk_gb_total": round(usage.total / gib, 2),
            "disk_gb_free": round(usage.free / gib, 2),
            "disk_gb_used": round(usage.used / gib, 2),
        }

    def describe_compute_environment(self) -> Dict[str, Any]:
        """Describe the local compute budget used to choose safe training defaults."""
        return describe_compute_environment()

    def _resolve_training_profile(self, compute_env: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_training_profile(compute_env)

    def _training_defaults_for_profile(self, profile: str) -> Dict[str, Any]:
        if profile == "local_light":
            return {
                "epochs": 30,
                "batch_size": 32,
                "num_replicates": 1,
                "ensemble_size": 1,
                "num_workers": 0,
                "metric": "rmse",
                "split_type": "random",
                "split_sizes": [0.8, 0.1, 0.1],
            }
        if profile == "local_standard":
            return {
                "epochs": 50,
                "batch_size": 32,
                "num_replicates": 1,
                "ensemble_size": 1,
                "num_workers": 0,
                "metric": "rmse",
                "split_type": "random",
                "split_sizes": [0.8, 0.1, 0.1],
            }
        if profile == "heavy_validation":
            return {
                "epochs": 100,
                "batch_size": 64,
                "num_replicates": 3,
                "ensemble_size": 1,
                "num_workers": 16,
                "patience": 15,
                "metric": "rmse",
                "split_type": "random",
                "split_sizes": [0.8, 0.1, 0.1],
            }
        return {
            "epochs": 50,
            "batch_size": 32,
            "num_replicates": 1,
            "ensemble_size": 1,
            "num_workers": 0,
            "metric": "rmse",
            "split_type": "random",
            "split_sizes": [0.8, 0.1, 0.1],
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

        # Protect local machines unless heavy compute was explicitly authorized.
        if not allow_heavy_compute and profile in {"heavy_validation", "benchmark"}:
            profile = resolved["profile"]

        merged = {
            **self._training_defaults_for_profile(profile),
            **requested,
        }

        if not allow_heavy_compute:
            if profile == "local_light":
                merged["epochs"] = min(int(merged.get("epochs", 30)), 30)
                merged["batch_size"] = min(int(merged.get("batch_size", 32)), 32)
                merged["ensemble_size"] = 1
                merged["num_replicates"] = 1
                merged["num_workers"] = 0
            elif profile == "local_standard":
                merged["epochs"] = min(int(merged.get("epochs", 50)), 50)
                merged["batch_size"] = min(int(merged.get("batch_size", 32)), 32)
                merged["ensemble_size"] = min(int(merged.get("ensemble_size", 1)), 1)
                merged["num_replicates"] = min(int(merged.get("num_replicates", 1)), 1)
                merged["num_workers"] = 0

        if profile == "heavy_validation":
            # On high-compute GPU runs, treat the profile values as floor values:
            # the agent may request a more aggressive configuration, but not a slower one.
            merged["epochs"] = max(int(merged.get("epochs", 100)), 100)
            merged["batch_size"] = max(int(merged.get("batch_size", 64)), 64)
            merged["num_workers"] = max(int(merged.get("num_workers", 16)), 16)

        return {
            "compute_environment": compute_env,
            "training_profile": profile,
            "profile_reason": resolved["reason"],
            "validation_protocol": requested_validation_protocol,
            "extra_args": merged,
        }

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

    def _train_single_run(
        self,
        *,
        train_csv: str,
        task: PredictionTaskSpec,
        output_dir: str,
        train_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = self.backend.train_model(
            train_csv=train_csv,
            output_dir=output_dir,
            task=task,
            extra_args=train_args,
        )
        result.update(
            self._compute_training_metrics(
                train_csv=train_csv,
                output_dir=output_dir,
                task=task,
            )
        )
        return result

    def _summarize_training_resources(
        self,
        *,
        compute_env: Dict[str, Any],
        effective_train_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        cpu_count = int(compute_env.get("cpu_count") or 1)
        gpu_available = bool(compute_env.get("gpu_available"))
        gpu_count = int(compute_env.get("gpu_count") or 0)
        num_workers = int(effective_train_args.get("num_workers") or 0)
        batch_size = int(effective_train_args.get("batch_size") or 0)
        ensemble_size = int(effective_train_args.get("ensemble_size") or 1)
        num_replicates = int(effective_train_args.get("num_replicates") or 1)
        epochs = int(effective_train_args.get("epochs") or 0)

        gpu_devices_requested = 1 if gpu_available and gpu_count > 0 else 0
        cpu_processes_estimated = max(1, min(cpu_count, num_workers + 1))

        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "ensemble_size": ensemble_size,
            "num_replicates": num_replicates,
            "epochs": epochs,
            "cpu_cores_available": cpu_count,
            "cpu_processes_estimated": cpu_processes_estimated,
            "gpu_available": gpu_available,
            "gpu_count_available": gpu_count,
            "gpu_devices_requested": gpu_devices_requested,
            "gpu_name": compute_env.get("gpu_name"),
            "memory_gb_total": compute_env.get("memory_gb_total"),
            "disk_gb_free": compute_env.get("disk_gb_free"),
            "disk_gb_total": compute_env.get("disk_gb_total"),
            "execution_env": compute_env.get("execution_env"),
        }

    def _summarize_training_durations(
        self,
        *,
        split_results: List[Dict[str, Any]],
        total_started_at: datetime,
        total_completed_at: datetime,
    ) -> Dict[str, Any]:
        return summarize_training_durations(
            split_results=split_results,
            total_started_at=total_started_at,
            total_completed_at=total_completed_at,
        )

    def _aggregate_split_families(self, split_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        families: Dict[str, List[Dict[str, Any]]] = {}
        for item in split_results:
            family = item.get("strategy_family") or item.get("strategy")
            metrics = ((item.get("metrics") or {}).get("test") or {})
            if not family or not metrics:
                continue
            families.setdefault(family, []).append(item)

        aggregated: Dict[str, Any] = {}
        metric_names = ("mse", "mae", "rae", "rmse", "r2", "spearman", "kendall")
        for family, items in families.items():
            entry: Dict[str, Any] = {
                "family": family,
                "num_runs": len(items),
                "strategy_labels": [item.get("strategy_label") for item in items],
                "runs": [],
                "test_n_values": [],
            }
            for item in items:
                metrics = ((item.get("metrics") or {}).get("test") or {})
                entry["runs"].append(
                    {
                        "label": item.get("strategy_label"),
                        "seed": item.get("seed"),
                        "metrics": metrics,
                    }
                )
                if metrics.get("n") is not None:
                    entry["test_n_values"].append(metrics["n"])

            for metric_name in metric_names:
                values = [
                    float(((item.get("metrics") or {}).get("test") or {}).get(metric_name))
                    for item in items
                    if ((item.get("metrics") or {}).get("test") or {}).get(metric_name) is not None
                ]
                if not values:
                    continue
                mean_value = sum(values) / len(values)
                variance = (
                    sum((value - mean_value) ** 2 for value in values) / len(values)
                    if len(values) > 1
                    else 0.0
                )
                entry[f"{metric_name}_mean"] = mean_value
                entry[f"{metric_name}_std"] = math.sqrt(variance)
                if len(values) == 1:
                    entry[metric_name] = values[0]

            if entry["test_n_values"]:
                entry["test_n_mean"] = sum(entry["test_n_values"]) / len(entry["test_n_values"])
            aggregated[family] = entry

        return aggregated

    def _assess_protocol_results(self, split_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return assess_protocol_results(split_results)

    def _materialize_primary_protocol_artifacts(
        self,
        *,
        root_output_dir: Path,
        primary_output_dir: Path,
    ) -> Dict[str, Optional[str]]:
        root_output_dir.mkdir(parents=True, exist_ok=True)
        root_model_dir = root_output_dir / "model_0"
        root_model_dir.mkdir(parents=True, exist_ok=True)

        copied: Dict[str, Optional[str]] = {
            "best_model_path": None,
            "test_predictions_path": None,
            "config_path": None,
            "splits_path": None,
        }

        resolved_artifacts = self._resolve_chemprop_run_artifacts(primary_output_dir)
        file_map = {
            resolved_artifacts["best_model_path"]: root_model_dir / "best.pt",
            resolved_artifacts["test_predictions_path"]: root_model_dir / "test_predictions.csv",
            resolved_artifacts["config_path"]: root_output_dir / "config.toml",
            resolved_artifacts["splits_path"]: root_output_dir / "splits.json",
        }

        for source_path, target_path in file_map.items():
            if source_path and source_path.exists():
                if source_path.resolve() != target_path.resolve():
                    shutil.copy2(source_path, target_path)
                if target_path.name == "best.pt":
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
        model_id_hint: str,
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
            model_id=model_id_hint,
        )

    def describe_backend(
        self,
        backend_name: Optional[str] = None,
        __name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe Chemprop backend availability and version information.

        The agent sometimes passes lightweight selector kwargs such as
        `backend_name` or `__name`. We accept and ignore them here so a simple
        backend inspection never fails on argument noise.
        """
        return self.backend.describe_environment()

    def _infer_training_run_dir(self, record: PredictionModelRecord) -> Optional[Path]:
        model_path = Path(record.model_path).expanduser()
        if not model_path.exists():
            return None
        if model_path.parent.name == "model_0":
            return model_path.parent.parent
        return model_path.parent

    def _materialize_internal_model(
        self,
        *,
        record: PredictionModelRecord,
        train_csv: Optional[str] = None,
        model_id: Optional[str] = None,
        source_artifacts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run_dir = self._infer_training_run_dir(record)
        if run_dir is None or not run_dir.exists():
            return {"materialized": False, "reason": "training_run_not_found"}

        resolved_model_id = model_id or record.model_id
        model_root = DEFAULT_INTERNAL_MODEL_ROOT / resolved_model_id
        model_dir = model_root / "model"
        artifacts_dir = model_root / "artifacts"
        model_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        source_model_path = Path(record.model_path).expanduser()
        copied_files: Dict[str, str] = {}

        if source_model_path.exists():
            target_name = source_model_path.name
            if record.backend_name == "tabicl":
                target_name = "best.pkl"
            target_model_path = model_dir / target_name
            shutil.copy2(source_model_path, target_model_path)
            copied_files["model_path"] = _relative_posix(target_model_path, model_root)
        else:
            return {"materialized": False, "reason": "model_artifact_missing"}

        optional_artifacts = {
            "config_path": run_dir / "config.toml",
            "training_summary_path": (
                run_dir / "cs_copilot_training_summary.json"
                if (run_dir / "cs_copilot_training_summary.json").exists()
                else run_dir / "tabicl_training_summary.json"
                if record.backend_name == "tabicl"
                else run_dir / "cs_copilot_training_summary.json"
            ),
            "splits_path": run_dir / "splits.json",
            "test_predictions_path": (
                run_dir / "test_predictions.csv"
                if record.backend_name == "tabicl"
                else self._resolve_chemprop_run_artifacts(run_dir).get("test_predictions_path")
                or run_dir / "model_0" / "test_predictions.csv"
            ),
            "reference_store_path": run_dir / "applicability_domain" / "reference_fingerprints.npz",
            "reference_manifest_path": run_dir / "applicability_domain" / "reference_manifest.json",
            "applicability_domain_path": run_dir / "applicability_domain" / "applicability_domain.json",
        }
        plot_sources: Dict[str, Path] = {}

        if source_artifacts:
            for key in (
                "config_path",
                "training_summary_path",
                "splits_path",
                "test_predictions_path",
                "reference_store_path",
                "reference_manifest_path",
                "applicability_domain_path",
            ):
                raw_path = source_artifacts.get(key)
                if raw_path:
                    optional_artifacts[key] = Path(str(raw_path)).expanduser()
            for plot_name, raw_path in (source_artifacts.get("plot_artifacts") or {}).items():
                if raw_path:
                    plot_sources[plot_name] = Path(str(raw_path)).expanduser()

        for key, source_path in optional_artifacts.items():
            if source_path.exists():
                if key == "config_path":
                    target_path = model_dir / "config.toml"
                elif key == "training_summary_path":
                    target_path = artifacts_dir / "cs_copilot_training_summary.json"
                elif key == "splits_path":
                    target_path = artifacts_dir / "splits.json"
                elif key == "test_predictions_path":
                    target_path = artifacts_dir / "test_predictions.csv"
                else:
                    target_path = artifacts_dir / source_path.name
                shutil.copy2(source_path, target_path)
                copied_files[key] = _relative_posix(target_path, model_root)

        copied_plot_artifacts: Dict[str, str] = {}
        if plot_sources:
            plots_dir = artifacts_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            for plot_name, source_path in plot_sources.items():
                if not source_path.exists():
                    continue
                target_path = plots_dir / source_path.name
                shutil.copy2(source_path, target_path)
                copied_plot_artifacts[plot_name] = _relative_posix(target_path, model_root)
            if copied_plot_artifacts:
                copied_files["plot_artifacts"] = copied_plot_artifacts

        if train_csv:
            source_train_csv = Path(train_csv).expanduser()
            if source_train_csv.exists():
                target_train_csv = artifacts_dir / source_train_csv.name
                shutil.copy2(source_train_csv, target_train_csv)
                copied_files["training_dataset_path"] = _relative_posix(
                    target_train_csv, model_root
                )

        metadata = {
            "model_id": resolved_model_id,
            "display_name": record.display_name or resolved_model_id,
            "version": record.version or "1.0",
            "status": record.status,
            "owner": record.owner or "chemspacecopilot",
            "source": record.source or "internal_training",
            "backend_name": record.backend_name,
            "task": {
                "task_type": record.task.task_type,
                "smiles_columns": list(record.task.smiles_columns),
                "target_columns": list(record.task.target_columns),
                "reaction_columns": list(record.task.reaction_columns),
                "uncertainty_method": record.task.uncertainty_method,
                "calibration_method": record.task.calibration_method,
            },
            "description": record.description or "",
            "domain_summary": record.domain_summary or "",
            "strengths": list(record.strengths),
            "limitations": list(record.limitations),
            "recommended_for": list(record.recommended_for),
            "not_recommended_for": list(record.not_recommended_for),
            "known_metrics": dict(record.known_metrics),
            "training_data_summary": dict(record.training_data_summary),
            "inference_profile": dict(record.inference_profile),
            "selection_hints": dict(record.selection_hints),
            "tags": dict(record.tags),
            "artifacts": copied_files,
        }
        if copied_files.get("applicability_domain_path"):
            metadata["applicability_domain"] = {
                "available": True,
                "method": "hybrid_morgan_domain",
                "reference_store_path": copied_files.get("reference_store_path"),
                "reference_manifest_path": copied_files.get("reference_manifest_path"),
                "index_path": copied_files.get("applicability_domain_path"),
            }
        if copied_plot_artifacts:
            metadata["plot_artifacts"] = copied_plot_artifacts
        metadata_path = model_root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

        return {
            "materialized": True,
            "model_root": str(model_root),
            "model_path": str(model_root / copied_files["model_path"]),
            "metadata_path": str(metadata_path),
            "artifacts": copied_files,
        }

    def _sync_internal_metadata(
        self,
        *,
        metadata_path: Optional[str],
        record: PredictionModelRecord,
        governance_assessment: Optional[Dict[str, Any]] = None,
        status_reason: Optional[str] = None,
    ) -> None:
        if not metadata_path:
            return
        target = Path(metadata_path).expanduser()
        if not target.exists():
            return
        try:
            payload = json.loads(target.read_text())
        except Exception:
            payload = {}
        payload.update(
            {
                "model_id": record.model_id,
                "display_name": record.display_name or record.model_id,
                "version": record.version or "1.0",
                "status": record.status,
                "owner": record.owner or "chemspacecopilot",
                "source": record.source or "internal_training",
                "backend_name": record.backend_name,
                "description": record.description or "",
                "domain_summary": record.domain_summary or "",
                "known_metrics": dict(record.known_metrics),
                "training_data_summary": dict(record.training_data_summary),
                "trained_at": (record.training_data_summary or {}).get("trained_at"),
                "trained_date": (record.training_data_summary or {}).get("trained_date"),
                "trained_time": (record.training_data_summary or {}).get("trained_time"),
                "inference_profile": dict(record.inference_profile),
                "selection_hints": dict(record.selection_hints),
                "strengths": list(record.strengths),
                "limitations": list(record.limitations),
                "recommended_for": list(record.recommended_for),
                "not_recommended_for": list(record.not_recommended_for),
                "tags": dict(record.tags),
                "task": {
                    "task_type": record.task.task_type,
                    "smiles_columns": list(record.task.smiles_columns),
                    "target_columns": list(record.task.target_columns),
                    "reaction_columns": list(record.task.reaction_columns),
                    "uncertainty_method": record.task.uncertainty_method,
                    "calibration_method": record.task.calibration_method,
                },
            }
        )
        artifacts = payload.get("artifacts") or {}
        if artifacts.get("applicability_domain_path"):
            payload["applicability_domain"] = {
                "available": True,
                "method": "hybrid_morgan_domain",
                "reference_store_path": artifacts.get("reference_store_path"),
                "reference_manifest_path": artifacts.get("reference_manifest_path"),
                "index_path": artifacts.get("applicability_domain_path"),
            }
        if artifacts.get("plot_artifacts"):
            payload["plot_artifacts"] = artifacts.get("plot_artifacts")
        if governance_assessment:
            payload["governance_assessment"] = governance_assessment
        if status_reason:
            payload["status_reason"] = status_reason
        target.write_text(json.dumps(payload, indent=2) + "\n")

    def describe_backends(self) -> Dict[str, Any]:
        """Describe all configured prediction backends."""
        return {
            name: backend.describe_environment()
            for name, backend in self.backends.items()
        }

    def _get_backend(self, backend_name: str):
        backend = self.backends.get(backend_name)
        if backend is None:
            raise ValueError(f"Unsupported prediction backend: {backend_name}")
        return backend

    def describe_catalog(self) -> Dict[str, Any]:
        """Describe the persistent model catalog configured for prediction."""
        self.catalog.refresh_from_internal_store(persist=True)
        return {
            "catalog_path": str(self.catalog.source_path),
            "num_models": len(self.catalog.list_models()),
            "model_ids": [record.model_id for record in self.catalog.list_models()],
        }

    def _annotate_record(self, record: PredictionModelRecord) -> Dict[str, Any]:
        payload = record.as_dict()
        payload["backend_environment"] = self._get_backend(
            record.backend_name
        ).describe_environment()
        payload["model_path_exists"] = Path(record.model_path).expanduser().exists()
        return payload

    def list_catalog_models(
        self,
        allowed_statuses: Optional[List[str]] = None,
        include_unavailable_paths: bool = False,
    ) -> List[Dict[str, Any]]:
        """List models from the persistent catalog with runtime annotations."""
        self.catalog.refresh_from_internal_store(persist=True)
        available_backends = [
            name for name, backend in self.backends.items() if backend.is_available()
        ]
        candidates = self.catalog.search(
            allowed_statuses=allowed_statuses,
            backend_available=bool(available_backends),
            available_backend_names=available_backends,
            include_unavailable_paths=include_unavailable_paths,
        )
        return [candidate.as_dict() for candidate in candidates]

    def summarize_catalog_model(self, model_id: str) -> Dict[str, Any]:
        """Return the catalog metadata for one model, enriched with runtime checks."""
        return self._annotate_record(self.catalog.get_model(model_id))

    def recommend_catalog_model(
        self,
        task_type: str,
        target_hint: Optional[str] = None,
        domain_hint: Optional[str] = None,
        require_uncertainty: bool = False,
        allowed_statuses: Optional[List[str]] = None,
        preferred_backend: Optional[str] = None,
        include_unavailable_paths: bool = True,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Recommend the best catalog model for a requested task."""
        self.catalog.refresh_from_internal_store(persist=True)
        recommendation = self.catalog.recommend(
            task_type=task_type,
            target_hint=target_hint,
            domain_hint=domain_hint,
            require_uncertainty=require_uncertainty,
            allowed_statuses=allowed_statuses,
            preferred_backend=preferred_backend,
            backend_available=any(backend.is_available() for backend in self.backends.values()),
            available_backend_names=[
                name for name, backend in self.backends.items() if backend.is_available()
            ],
            include_unavailable_paths=include_unavailable_paths,
        )

        if agent is not None:
            prediction_state = _get_prediction_state(agent)
            prediction_state["catalog_recommendations"] = recommendation

        return recommendation

    def register_catalog_model(self, model_id: str, agent: Optional[Agent] = None) -> Dict[str, Any]:
        """Register a model from the persistent catalog into the current session."""
        if agent is None:
            raise ValueError("Agent is required to register a catalog model")

        self.catalog.refresh_from_internal_store(persist=True)
        record = self.catalog.get_model(model_id)
        return self.register_model(
            model_id=record.model_id,
            model_path=record.model_path,
            backend_name=record.backend_name,
            task_type=record.task.task_type,
            smiles_columns=record.task.smiles_columns,
            target_columns=record.task.target_columns,
            reaction_columns=record.task.reaction_columns,
            uncertainty_method=record.task.uncertainty_method,
            calibration_method=record.task.calibration_method,
            description=record.description,
            tags=record.tags,
            version=record.version,
            status=record.status,
            owner=record.owner,
            source=record.source,
            domain_summary=record.domain_summary,
            strengths=record.strengths,
            limitations=record.limitations,
            recommended_for=record.recommended_for,
            not_recommended_for=record.not_recommended_for,
            known_metrics=record.known_metrics,
            training_data_summary=record.training_data_summary,
            inference_profile=record.inference_profile,
            selection_hints=record.selection_hints,
            applicability_domain=record.applicability_domain,
            agent=agent,
        )

    def register_model(
        self,
        model_id: str,
        model_path: str,
        task_type: str,
        backend_name: Optional[str] = None,
        smiles_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        reaction_columns: Optional[List[str]] = None,
        uncertainty_method: Optional[str] = None,
        calibration_method: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        version: Optional[str] = None,
        status: str = "experimental",
        owner: Optional[str] = None,
        source: Optional[str] = None,
        domain_summary: Optional[str] = None,
        strengths: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        recommended_for: Optional[List[str]] = None,
        not_recommended_for: Optional[List[str]] = None,
        known_metrics: Optional[Dict[str, Any]] = None,
        training_data_summary: Optional[Dict[str, Any]] = None,
        inference_profile: Optional[Dict[str, Any]] = None,
        selection_hints: Optional[Dict[str, Any]] = None,
        applicability_domain: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Register a Chemprop model in session state for later use."""
        if agent is None:
            raise ValueError("Agent is required to register a model")

        backend_name = backend_name or self.backend.backend_name
        backend = self._get_backend(backend_name)
        validated_path = backend.validate_model_path(model_path)
        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=smiles_columns or ["smiles"],
            target_columns=target_columns or [],
            reaction_columns=reaction_columns or [],
            uncertainty_method=uncertainty_method,
            calibration_method=calibration_method,
        )
        record = PredictionModelRecord(
            model_id=model_id,
            backend_name=backend.backend_name,
            model_path=str(validated_path),
            metadata_path=None,
            task=task,
            description=description,
            tags=tags or {},
            version=version,
            status=status,
            owner=owner,
            source=source,
            domain_summary=domain_summary,
            strengths=strengths or [],
            limitations=limitations or [],
            recommended_for=recommended_for or [],
            not_recommended_for=not_recommended_for or [],
            known_metrics=known_metrics or {},
            training_data_summary=training_data_summary or {},
            inference_profile=inference_profile or {},
            selection_hints=selection_hints or {},
            applicability_domain=applicability_domain or {},
        )

        prediction_state = _get_prediction_state(agent)
        prediction_state["registered"][model_id] = record.as_dict()
        return record.as_dict()

    def persist_registered_model(
        self,
        model_id: str,
        status: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        source: Optional[str] = None,
        domain_summary: Optional[str] = None,
        owner: Optional[str] = None,
        version: Optional[str] = None,
        strengths: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        recommended_for: Optional[List[str]] = None,
        not_recommended_for: Optional[List[str]] = None,
        known_metrics: Optional[Dict[str, Any]] = None,
        training_data_summary: Optional[Dict[str, Any]] = None,
        inference_profile: Optional[Dict[str, Any]] = None,
        selection_hints: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Persist a session-registered model into the catalog JSON."""
        if agent is None:
            raise ValueError("Agent is required to persist a registered model")

        current = self._resolve_record(model_id, agent)
        prediction_state = _get_prediction_state(agent)
        training_runs = prediction_state.get("training_runs") or []
        train_csv = None
        matching_training_run = None
        inferred_run_dir = self._infer_training_run_dir(current)
        if inferred_run_dir is not None:
            inferred_output = str(inferred_run_dir.resolve())
            for run in reversed(training_runs):
                if str(Path(run.get("output_dir", "")).expanduser().resolve()) == inferred_output:
                    matching_training_run = run
                    train_csv = run.get("train_csv")
                    break

        governance_assessment = {}
        recommended_status = None
        applicability_domain = {}
        summary_payload: Dict[str, Any] = {}
        summary_path: Optional[Path] = None
        if matching_training_run:
            output_dir = Path(matching_training_run.get("output_dir", "")).expanduser()
            summary_candidates = [
                output_dir / "cs_copilot_training_summary.json",
                output_dir / "tabicl_training_summary.json",
            ]
            summary_path = next((path for path in summary_candidates if path.exists()), None)
            if summary_path is not None and summary_path.exists():
                try:
                    summary_payload = json.loads(summary_path.read_text())
                    applicability_domain = summary_payload.get("applicability_domain") or {}
                    governance_assessment = (
                        (summary_payload.get("validation_assessment") or {}).get("governance") or {}
                    )
                    recommended_status = governance_assessment.get("recommended_status")
                except Exception:
                    governance_assessment = {}
                    applicability_domain = {}
                    summary_payload = {}

        resolved_version = version or current.version or "1"
        trained_at_raw = summary_payload.get("trained_at")
        try:
            trained_at = coerce_project_timezone(trained_at_raw)
        except Exception:
            trained_at = project_now()
        train_csv_for_name = train_csv or summary_payload.get("train_csv") or current.source
        benchmark_dataset_name = (current.training_data_summary or {}).get("benchmark_dataset_name")
        benchmark_target_name = (current.training_data_summary or {}).get("benchmark_target_name")
        if benchmark_dataset_name and benchmark_target_name:
            endpoint_name = safe_slug(str(benchmark_dataset_name)) or "endpoint"
            dataset_name = safe_slug(str(benchmark_target_name)) or "dataset"
        else:
            endpoint_name, dataset_name = _extract_endpoint_and_dataset(train_csv_for_name, current.model_id)
        protocol_name = (
            summary_payload.get("validation_protocol")
            or matching_training_run.get("validation_protocol")
            if matching_training_run
            else "protocol"
        )
        representation_name = (
            (current.training_data_summary or {}).get("representation_name")
            or (current.inference_profile or {}).get("representation_name")
            or (current.selection_hints or {}).get("representation_name")
        )
        canonical_model_id = _canonical_model_id(
            endpoint=endpoint_name,
            dataset=dataset_name,
            protocol=str(protocol_name or "protocol"),
            backend=current.backend_name,
            representation=representation_name,
            version=str(resolved_version),
            trained_at=trained_at,
        )
        canonical_display_name = _canonical_display_name(
            endpoint=endpoint_name,
            dataset=dataset_name,
            protocol=str(protocol_name or "protocol"),
            backend=current.backend_name,
            representation=representation_name,
            version=str(resolved_version),
        )

        source_artifacts = {
            "training_summary_path": str(summary_path) if matching_training_run and summary_path.exists() else None,
            "config_path": summary_payload.get("config_path"),
            "splits_path": summary_payload.get("splits_path"),
            "test_predictions_path": summary_payload.get("test_predictions_path"),
            "reference_store_path": applicability_domain.get("reference_store_path"),
            "reference_manifest_path": applicability_domain.get("reference_manifest_path"),
            "applicability_domain_path": applicability_domain.get("applicability_domain_path"),
            "plot_artifacts": summary_payload.get("plot_artifacts") or {},
        }

        materialized = self._materialize_internal_model(
            record=current,
            train_csv=train_csv,
            model_id=canonical_model_id,
            source_artifacts=source_artifacts,
        )
        resolved_model_path = materialized.get("model_path", current.model_path)
        resolved_metadata_path = materialized.get("metadata_path", current.metadata_path)
        requested_status = status or current.status
        resolved_status = requested_status
        status_reason = None
        governed_statuses = {"validated", "robust_validated"}
        if requested_status in governed_statuses and recommended_status and recommended_status != requested_status:
            resolved_status = recommended_status
            status_reason = (
                f"Requested `{requested_status}` was adjusted by governance because the final "
                "validation gates did not support that status."
            )
        persisted_record = PredictionModelRecord(
            model_id=canonical_model_id,
            backend_name=current.backend_name,
            model_path=resolved_model_path,
            metadata_path=resolved_metadata_path,
            task=current.task,
            display_name=display_name or canonical_display_name,
            description=description or current.description,
            tags=tags or current.tags,
            version=resolved_version,
            status=resolved_status,
            owner=owner or current.owner,
            source=source or current.source,
            domain_summary=domain_summary or current.domain_summary,
            strengths=strengths or current.strengths,
            limitations=limitations or current.limitations,
            recommended_for=recommended_for or current.recommended_for,
            not_recommended_for=not_recommended_for or current.not_recommended_for,
            known_metrics=known_metrics or current.known_metrics,
            training_data_summary={
                **current.training_data_summary,
                **(training_data_summary or {}),
                "trained_at": trained_at.isoformat(),
                "trained_date": trained_at.strftime("%d/%m/%Y"),
                "trained_time": trained_at.strftime("%H:%M:%S"),
                "endpoint_name": endpoint_name,
                "dataset_name": dataset_name,
                "validation_protocol": str(protocol_name or "protocol"),
            },
            inference_profile={
                **current.inference_profile,
                **(inference_profile or {}),
            },
            selection_hints={
                **current.selection_hints,
                **(selection_hints or {}),
                "governance_recommended_status": recommended_status,
            },
            applicability_domain={
                **current.applicability_domain,
                **(applicability_domain or {}),
            },
        )

        self.catalog.upsert_model(persisted_record)
        self.catalog = PredictionModelCatalog.load(str(self.catalog.source_path))
        self._sync_internal_metadata(
            metadata_path=resolved_metadata_path,
            record=persisted_record,
            governance_assessment=governance_assessment,
            status_reason=status_reason,
        )

        prediction_state["registered"].pop(model_id, None)
        prediction_state["registered"][canonical_model_id] = persisted_record.as_dict()

        return {
            "catalog_path": str(self.catalog.source_path),
            "model_id": persisted_record.model_id,
            "status": persisted_record.status,
            "persisted": True,
            "materialized": bool(materialized.get("materialized")),
            "model_root": materialized.get("model_root"),
            "metadata_path": persisted_record.metadata_path,
            "governance_assessment": governance_assessment,
            "status_reason": status_reason,
            "record": persisted_record.as_dict(),
        }

    def list_registered_models(self, agent: Optional[Agent] = None) -> List[Dict[str, Any]]:
        """List models registered in the current session."""
        if agent is None:
            return []
        prediction_state = _get_prediction_state(agent)
        return list(prediction_state["registered"].values())

    def summarize_model(self, model_id: str, agent: Optional[Agent] = None) -> Dict[str, Any]:
        """Return the stored summary for a registered model."""
        if agent is None:
            raise ValueError("Agent is required to summarize a model")
        prediction_state = _get_prediction_state(agent)
        record = prediction_state["registered"].get(model_id)
        if not record:
            raise ValueError(f"Unknown model_id: {model_id}")
        return self._annotate_record(PredictionModelRecord.from_dict(record))

    def _resolve_record(self, model_id: str, agent: Agent) -> PredictionModelRecord:
        prediction_state = _get_prediction_state(agent)
        payload = prediction_state["registered"].get(model_id)
        if payload is None:
            raise ValueError(f"Unknown model_id: {model_id}")
        return PredictionModelRecord.from_dict(payload)

    def predict_from_csv(
        self,
        model_id: str,
        input_csv: str,
        smiles_column: str = "smiles",
        preds_path: Optional[str] = None,
        return_uncertainty: bool = False,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Run prediction from a CSV file and persist the result path in session state."""
        if agent is None:
            raise ValueError("Agent is required for prediction")

        record = self._resolve_record(model_id, agent)
        output_path = _prediction_output_path(model_id, preds_path)
        backend = self._get_backend(record.backend_name)
        local_input = Path(input_csv).expanduser()
        if local_input.exists():
            source_df = pd.read_csv(local_input)
        else:
            with S3.open(input_csv, "r") as fh:
                source_df = pd.read_csv(fh)

        df = source_df.copy()

        smiles_found = None
        smiles_candidates = [
            smiles_column,
            "smiles",
            "SMILES",
            "canonical_smiles",
            "Smiles",
            "smi",
        ]
        for candidate in smiles_candidates:
            if candidate and candidate in df.columns:
                smiles_found = candidate
                break

        if smiles_found is None:
            raise ValueError(
                f"No SMILES column found for prediction. Tried {smiles_candidates}. "
                f"Available columns: {list(df.columns)}"
            )

        df = standardize_smiles_column(df, smiles_found)
        if smiles_found != "smiles":
            df = df.rename(columns={smiles_found: "smiles"})
        local_input = Path(".files") / "prediction_inputs" / f"{model_id}_input.csv"
        local_input.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_input, index=False)

        result = backend.predict_from_csv(
            input_csv=str(local_input),
            model_record=record,
            preds_path=str(output_path),
            return_uncertainty=return_uncertainty,
        )

        predictions_only_df = pd.read_csv(output_path)
        prediction_columns = [
            column for column in predictions_only_df.columns if column not in source_df.columns
        ]
        if prediction_columns:
            enriched_output_df = pd.concat(
                [source_df.reset_index(drop=True), predictions_only_df[prediction_columns].reset_index(drop=True)],
                axis=1,
            )
            enriched_output_df.to_csv(output_path, index=False)

        preview_df = pd.read_csv(output_path)
        preview_columns = list(preview_df.columns)
        preview = preview_df.head(5).to_dict(orient="records")
        num_rows = int(len(preview_df))

        prediction_state = _get_prediction_state(agent)
        prediction_state["last_prediction"] = {
            "model_id": model_id,
            "input_csv": str(local_input),
            "preds_path": str(output_path),
            "return_uncertainty": return_uncertainty,
            "applicability_domain": result.get("applicability_domain") or {},
            "applicability_domain_columns": result.get("applicability_domain_columns") or [],
        }
        history_entry = {
            "model_id": model_id,
            "backend_name": record.backend_name,
            "task_type": record.task.task_type,
            "input_csv": str(local_input),
            "preds_path": str(output_path),
            "download_file_ref": str(output_path),
            "preview_columns": preview_columns,
            "preview": preview,
            "num_rows": num_rows,
            "applicability_domain": result.get("applicability_domain") or {},
            "applicability_domain_columns": result.get("applicability_domain_columns") or [],
        }
        prediction_state["prediction_history"].append(history_entry)
        result["preds_path"] = str(output_path)
        result["download_file_ref"] = str(output_path)
        result["preview_columns"] = preview_columns
        result["preview"] = preview
        result["num_rows"] = num_rows
        return result

    def predict_from_smiles(
        self,
        model_id: str,
        smiles: List[str],
        preds_path: Optional[str] = None,
        return_uncertainty: bool = False,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Run prediction from an in-memory list of SMILES by materializing a temporary CSV."""
        if agent is None:
            raise ValueError("Agent is required for prediction")

        if not smiles:
            raise ValueError("At least one SMILES string is required")

        input_path = Path(".files") / "prediction_inputs" / f"{model_id}_smiles_input.csv"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"smiles": smiles})
        df = standardize_smiles_column(df, "smiles")
        df.to_csv(input_path, index=False)

        result = self.predict_from_csv(
            model_id=model_id,
            input_csv=str(input_path),
            smiles_column="smiles",
            preds_path=preds_path,
            return_uncertainty=return_uncertainty,
            agent=agent,
        )
        result["num_smiles"] = len(smiles)
        return result

    def export_prediction_summary(
        self,
        summary_csv: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Export a consolidated CSV summary from prediction history."""
        if agent is None:
            raise ValueError("Agent is required to export a prediction summary")

        prediction_state = _get_prediction_state(agent)
        history = prediction_state.get("prediction_history") or []
        if not history:
            raise ValueError("No prediction history is available for summary export")

        summary_path = (
            Path(summary_csv).expanduser()
            if summary_csv
            else (Path(".files") / "prediction_outputs" / "prediction_summary.csv").resolve()
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        frames: List[pd.DataFrame] = []
        for item in history:
            preds_path = item.get("preds_path")
            model_id = item.get("model_id")
            task_type = item.get("task_type")
            if not preds_path or not Path(preds_path).exists():
                continue

            df = pd.read_csv(preds_path).copy()
            if df.empty:
                continue

            value_columns = [col for col in df.columns if col.lower() != "smiles"]
            if not value_columns:
                continue

            melted = df.melt(
                id_vars=["smiles"] if "smiles" in df.columns else None,
                value_vars=value_columns,
                var_name="prediction_column",
                value_name="predicted_value",
            )
            melted.insert(0, "model_id", model_id)
            melted.insert(1, "task_type", task_type)
            frames.append(melted)

        if not frames:
            raise ValueError("No readable prediction files were found for summary export")

        summary_df = pd.concat(frames, ignore_index=True)
        summary_df.to_csv(summary_path, index=False)

        prediction_outputs = agent.session_state.setdefault("prediction_outputs", {})
        prediction_outputs["latest_summary"] = str(summary_path)

        preview_columns = list(summary_df.columns)
        preview = summary_df.head(10).to_dict(orient="records")

        return {
            "summary_csv": str(summary_path),
            "download_file_ref": str(summary_path),
            "num_rows": int(len(summary_df)),
            "num_files": len(frames),
            "preview_columns": preview_columns,
            "preview": preview,
        }

    def _compute_training_metrics(
        self,
        *,
        train_csv: str,
        output_dir: str,
        task: PredictionTaskSpec,
    ) -> Dict[str, Any]:
        output_path = Path(output_dir).expanduser()
        resolved_artifacts = self._resolve_chemprop_run_artifacts(output_path)
        splits_path = resolved_artifacts["splits_path"]
        preds_path = resolved_artifacts["test_predictions_path"]

        if splits_path is None or preds_path is None or not splits_path.exists() or not preds_path.exists():
            return {}

        target_column = task.target_columns[0] if task.target_columns else None
        if not target_column:
            return {}

        dataset = _strip_unnamed_columns(pd.read_csv(Path(train_csv).expanduser()))
        predictions = _strip_unnamed_columns(pd.read_csv(preds_path))

        split_payload = json.loads(splits_path.read_text())
        if not split_payload or "test" not in split_payload[0]:
            return {}

        test_indices = split_payload[0]["test"]
        actual = dataset.iloc[test_indices].reset_index(drop=True)

        if target_column not in actual.columns or target_column not in predictions.columns:
            return {}

        if len(actual) != len(predictions):
            return {}

        actual_values = pd.to_numeric(actual[target_column], errors="coerce")
        predicted_values = pd.to_numeric(predictions[target_column], errors="coerce")
        valid_mask = actual_values.notna() & predicted_values.notna()
        if not valid_mask.any():
            return {}

        y_true = actual_values[valid_mask].astype(float)
        y_pred = predicted_values[valid_mask].astype(float)
        residuals = y_true - y_pred
        mse = float((residuals.pow(2)).mean())
        mae = float(residuals.abs().mean())
        rae_denom = float((y_true - float(y_true.mean())).abs().sum())
        rae_num = float(residuals.abs().sum())
        rae = float(rae_num / rae_denom) if rae_denom > 0 else None
        rmse = float(math.sqrt(mse))
        centered = y_true - float(y_true.mean())
        ss_tot = float((centered.pow(2)).sum())
        ss_res = float((residuals.pow(2)).sum())
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else None
        spearman = None
        kendall = None
        try:
            spearman_stat = spearmanr(y_true.to_numpy(), y_pred.to_numpy(), nan_policy="omit")
            spearman = float(spearman_stat.statistic) if spearman_stat.statistic is not None else None
        except Exception:
            spearman = None
        try:
            kendall_stat = kendalltau(y_true.to_numpy(), y_pred.to_numpy(), nan_policy="omit")
            kendall = float(kendall_stat.statistic) if kendall_stat.statistic is not None else None
        except Exception:
            kendall = None

        return {
            "best_model_path": str(resolved_artifacts["best_model_path"]) if resolved_artifacts.get("best_model_path") else None,
            "test_predictions_path": str(preds_path),
            "splits_path": str(splits_path),
            "metrics": {
                "test": {
                    "mse": mse,
                    "mae": mae,
                    "rae": rae,
                    "rmse": rmse,
                    "r2": r2,
                    "spearman": spearman,
                    "kendall": kendall,
                    "n": int(valid_mask.sum()),
                    "target_column": target_column,
                }
            },
        }

    def prepare_training_dataset(
        self,
        input_csv: str,
        smiles_column: str,
        target_columns: List[str] | str,
        output_csv: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalize a training CSV into the canonical format expected by later prediction tools."""
        with S3.open(input_csv, "r") as fh:
            df = pd.read_csv(fh)

        if isinstance(target_columns, str):
            try:
                parsed = json.loads(target_columns)
            except json.JSONDecodeError:
                target_columns = [target_columns]
            else:
                if isinstance(parsed, list):
                    target_columns = parsed
                else:
                    target_columns = [str(parsed)]

        df = standardize_smiles_column(df, smiles_column)
        missing_targets = [column for column in target_columns if column not in df.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")

        standardized = df[["smiles", *target_columns]].copy()
        destination = output_csv or "training/chemprop_training_dataset.csv"
        with S3.open(destination, "w") as fh:
            standardized.to_csv(fh, index=False)

        return {
            "output_csv": destination,
            "rows": int(len(standardized)),
            "columns": list(standardized.columns),
        }

    def _run_oof_predictions(
        self,
        *,
        train_csv: str,
        task: PredictionTaskSpec,
        target_column: str,
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
            "validation_protocol": "activity_cliff_oof",
            "num_replicates": 1,
            "ensemble_size": 1,
            "num_workers": 0,
        }
        for fold_index, split_payload in enumerate(folds, start=1):
            fold_dir = output_dir / "_activity_cliff_oof" / f"fold_{fold_index}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_frame = dataset.iloc[split_payload["train"]].reset_index(drop=True)
            val_frame = dataset.iloc[split_payload["val"]].reset_index(drop=True)
            test_frame = dataset.iloc[split_payload["test"]].reset_index(drop=True)
            train_path = fold_dir / "train.csv"
            val_path = fold_dir / "val.csv"
            test_path = fold_dir / "test.csv"
            train_frame.to_csv(train_path, index=False)
            val_frame.to_csv(val_path, index=False)
            test_frame.to_csv(test_path, index=False)
            self.backend.train_model(
                train_csv=str(train_path),
                output_dir=str(fold_dir),
                task=task,
                extra_args={
                    **base_args,
                    "data_seed": random_state + fold_index,
                    "split_type": "random",
                    "separate_val_path": str(val_path),
                    "separate_test_path": str(test_path),
                },
            )
            resolved_artifacts = self._resolve_chemprop_run_artifacts(fold_dir)
            preds_path = resolved_artifacts.get("test_predictions_path")
            if preds_path is None or not preds_path.exists():
                raise ValueError("Activity-cliff OOF predictions were not produced for Chemprop.")
            predictions = _strip_unnamed_columns(pd.read_csv(preds_path))
            pred_series = pd.to_numeric(predictions[target_column], errors="coerce")
            test_indices = split_payload["test"]
            if len(pred_series) != len(test_indices):
                raise ValueError("Activity-cliff OOF predictions length mismatch for Chemprop.")
            oof_predictions.iloc[test_indices] = pred_series.to_numpy(dtype=float)
        if oof_predictions.isna().any():
            raise ValueError("Activity-cliff OOF predictions were incomplete for Chemprop.")
        return oof_predictions.astype(float)

    def _train_model_single(
        self,
        *,
        train_csv: str,
        task_type: str,
        output_dir: str,
        smiles_columns: Optional[List[str]],
        target_columns: Optional[List[str]],
        reaction_columns: Optional[List[str]],
        extra_args: Optional[Dict[str, Any]],
        agent: Optional[Agent],
    ) -> Dict[str, Any]:
        resolved_output_dir = str(Path(output_dir).expanduser().resolve())
        root_output_path = Path(resolved_output_dir)
        active_marker_path = root_output_path / ".training_in_progress"
        trained_at = project_now()
        training_policy = self._apply_training_profile(extra_args)
        protocol_policy = self._resolve_validation_protocol(
            requested_protocol=training_policy.get("validation_protocol"),
            training_profile=training_policy["training_profile"],
        )
        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=smiles_columns or ["smiles"],
            target_columns=target_columns or [],
            reaction_columns=reaction_columns or [],
        )
        prediction_state = None
        qsar_training_state = None
        active_run_record = {
            "status": "running",
            "train_csv": train_csv,
            "output_dir": resolved_output_dir,
            "validation_protocol": protocol_policy["protocol"],
            "training_profile": training_policy["training_profile"],
            "created_at": trained_at.isoformat(),
            "active_marker_path": str(active_marker_path),
            "current_split_label": None,
        }

        if agent is not None:
            prediction_state = _get_prediction_state(agent)
            prediction_state["active_training_run"] = dict(active_run_record)
            qsar_training_state = agent.session_state.setdefault("qsar_training", {})
            qsar_training_state["active_run"] = dict(active_run_record)

        _write_active_training_marker(active_marker_path, active_run_record)

        split_results: List[Dict[str, Any]] = []
        primary_run: Optional[Dict[str, Any]] = None
        primary_output_dir: Optional[Path] = None
        total_started_at = project_now()
        multi_run_protocol = len(protocol_policy["split_runs"]) > 1

        try:
            for split_run in protocol_policy["split_runs"]:
                label = split_run["label"]
                run_output_dir = (
                    root_output_path / f"{safe_slug(label)}_split"
                    if multi_run_protocol
                    else root_output_path
                )
                run_args = {
                    **training_policy["extra_args"],
                    "split_type": split_run["backend_split_type"],
                    "data_seed": split_run["seed"],
                }

                active_run_record["current_split_label"] = label
                if prediction_state is not None:
                    prediction_state["active_training_run"] = dict(active_run_record)
                if qsar_training_state is not None:
                    qsar_training_state["active_run"] = dict(active_run_record)
                _write_active_training_marker(active_marker_path, active_run_record)

                single_result = self._train_single_run(
                    train_csv=train_csv,
                    task=task,
                    output_dir=str(run_output_dir),
                    train_args=run_args,
                )
                if "scaffold" in label:
                    strategy_name = "scaffold"
                    strategy_family = "scaffold"
                elif "kmeans" in label:
                    strategy_name = "cluster_kmeans"
                    strategy_family = "cluster_kmeans"
                elif "kennard" in label:
                    strategy_name = "distance_kennard_stone"
                    strategy_family = "distance_kennard_stone"
                elif "random_seed_" in label:
                    strategy_name = label
                    strategy_family = "random"
                else:
                    strategy_name = "random"
                    strategy_family = "random"
                single_result["strategy"] = strategy_name
                single_result["strategy_family"] = strategy_family
                single_result["strategy_label"] = label
                single_result["backend_split_type"] = split_run["backend_split_type"]
                single_result["seed"] = split_run["seed"]
                single_result["output_dir"] = str(run_output_dir)
                single_result["validation_protocol"] = protocol_policy["protocol"]
                split_results.append(single_result)

                if split_run.get("primary") or primary_run is None:
                    primary_run = single_result
                    primary_output_dir = run_output_dir

            if primary_run is None or primary_output_dir is None:
                raise ValueError("Training protocol did not produce a primary run.")

            if prediction_state is not None:
                prediction_state["training_runs"].append(
                    {
                        "train_csv": train_csv,
                        "output_dir": resolved_output_dir,
                        "task_type": task_type,
                        "smiles_columns": task.smiles_columns,
                        "target_columns": task.target_columns,
                        "validation_protocol": protocol_policy["protocol"],
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

            root_artifacts = self._materialize_primary_protocol_artifacts(
                root_output_dir=root_output_path,
                primary_output_dir=primary_output_dir,
            )
            validation_assessment = self._assess_protocol_results(split_results)
            ad_summary = self._build_applicability_domain(
                train_csv=train_csv,
                primary_run=primary_run,
                primary_output_dir=primary_output_dir,
                model_id_hint=Path(resolved_output_dir).name,
                task=task,
            )
            plot_artifacts: Dict[str, str] = {}
            target_column = task.target_columns[0] if task.target_columns else None
            if target_column:
                plots_output_dir = root_output_path / "artifacts" / "plots"
                try:
                    plot_artifacts = build_qsar_training_plots(
                        train_csv=train_csv,
                        split_results=split_results,
                        primary_run=primary_run,
                        output_dir=str(plots_output_dir),
                        target_column=target_column,
                    )
                except Exception:
                    plot_artifacts = {}

            result = dict(primary_run)
            result["output_dir"] = resolved_output_dir
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
            total_completed_at = project_now()
            result["training_durations"] = self._summarize_training_durations(
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

            training_summary_path = Path(resolved_output_dir) / "cs_copilot_training_summary.json"
            training_summary_path.parent.mkdir(parents=True, exist_ok=True)
            training_summary_path.write_text(json.dumps(result, indent=2))

            resolved_primary_artifacts = self._resolve_chemprop_run_artifacts(Path(resolved_output_dir))
            best_model_path = Path(
                root_artifacts.get("best_model_path")
                or resolved_primary_artifacts.get("best_model_path")
                or (Path(resolved_output_dir) / "model_0" / "best.pt")
            )
            config_path = Path(root_artifacts.get("config_path") or resolved_primary_artifacts["config_path"])
            splits_path = Path(root_artifacts.get("splits_path") or resolved_primary_artifacts["splits_path"])
            result["summary_path"] = str(training_summary_path)
            if best_model_path.exists():
                result["best_model_path"] = str(best_model_path)
                result["download_file_ref"] = str(best_model_path)
            result["summary_file_ref"] = str(training_summary_path)
            if root_artifacts.get("test_predictions_path"):
                result["test_predictions_file_ref"] = root_artifacts["test_predictions_path"]
            elif primary_run.get("test_predictions_path"):
                result["test_predictions_file_ref"] = primary_run["test_predictions_path"]
            if ad_summary.get("applicability_domain_path"):
                result["applicability_domain_file_ref"] = ad_summary["applicability_domain_path"]
            bundle_path = (
                Path(".files")
                / "prediction_outputs"
                / f"{Path(resolved_output_dir).name}_training_bundle.zip"
            ).resolve()
            bundle_files = [
                Path(train_csv).expanduser(),
                training_summary_path,
                best_model_path,
                config_path,
                splits_path,
            ]
            for item in split_results:
                if item.get("summary_path"):
                    bundle_files.append(Path(item["summary_path"]).expanduser())
                if item.get("test_predictions_path"):
                    bundle_files.append(Path(item["test_predictions_path"]).expanduser())
            for ad_key in (
                "reference_store_path",
                "reference_manifest_path",
                "applicability_domain_path",
            ):
                if ad_summary.get(ad_key):
                    bundle_files.append(Path(ad_summary[ad_key]).expanduser())
            for plot_path in plot_artifacts.values():
                bundle_files.append(Path(plot_path).expanduser())
            bundle = _bundle_artifacts(bundle_path, bundle_files)
            result["bundle_file_ref"] = str(bundle)
            return result
        except Exception as exc:
            active_run_record["status"] = "failed"
            active_run_record["error"] = str(exc)
            raise
        finally:
            active_run_record["completed_at"] = project_now().isoformat()
            if prediction_state is not None:
                prediction_state["active_training_run"] = None
            if qsar_training_state is not None:
                qsar_training_state["active_run"] = None
            _write_active_training_marker(active_marker_path, active_run_record)

    def train_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        smiles_columns: Optional[List[str] | str] = None,
        target_columns: Optional[List[str] | str] = None,
        reaction_columns: Optional[List[str] | str] = None,
        activity_cliff_feedback: bool = False,
        activity_cliff_feedback_loops: int = 1,
        activity_cliff_step_percentile: float = 5.0,
        activity_cliff_similarity_threshold: float = 0.70,
        activity_cliff_k_neighbors: int = 10,
        activity_cliff_oof_folds: int = 5,
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Launch Chemprop training and persist a lightweight training record."""
        if isinstance(smiles_columns, str):
            parsed = json.loads(smiles_columns)
            smiles_columns = parsed if isinstance(parsed, list) else [str(parsed)]
        if isinstance(target_columns, str):
            parsed = json.loads(target_columns)
            target_columns = parsed if isinstance(parsed, list) else [str(parsed)]
        if isinstance(reaction_columns, str):
            parsed = json.loads(reaction_columns)
            reaction_columns = parsed if isinstance(parsed, list) else [str(parsed)]
        merged_extra_args = merge_activity_cliff_args(
            extra_args=extra_args,
            activity_cliff_feedback=activity_cliff_feedback,
            activity_cliff_feedback_loops=activity_cliff_feedback_loops,
            activity_cliff_step_percentile=activity_cliff_step_percentile,
            activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
            activity_cliff_k_neighbors=activity_cliff_k_neighbors,
            activity_cliff_oof_folds=activity_cliff_oof_folds,
        )
        feedback_config = parse_activity_cliff_config(merged_extra_args)
        normalized_smiles_columns = smiles_columns or ["smiles"]
        normalized_target_columns = target_columns or []
        normalized_reaction_columns = reaction_columns or []

        if not feedback_config.enabled:
            return self._train_model_single(
                train_csv=train_csv,
                task_type=task_type,
                output_dir=output_dir,
                smiles_columns=normalized_smiles_columns,
                target_columns=normalized_target_columns,
                reaction_columns=normalized_reaction_columns,
                extra_args=merged_extra_args,
                agent=agent,
            )

        resolved_output_dir = str(Path(output_dir).expanduser().resolve())
        root_output_path = Path(resolved_output_dir)
        root_output_path.mkdir(parents=True, exist_ok=True)
        training_policy = self._apply_training_profile(merged_extra_args)
        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=normalized_smiles_columns,
            target_columns=normalized_target_columns,
            reaction_columns=normalized_reaction_columns,
        )
        target_column = normalized_target_columns[0]
        oof_predictions = self._run_oof_predictions(
            train_csv=train_csv,
            task=task,
            target_column=target_column,
            training_policy=training_policy,
            cliff_config=feedback_config,
            random_state=int((strip_activity_cliff_args(merged_extra_args)).get("data_seed", 42)),
            output_dir=root_output_path,
        )
        base_dataset = _strip_unnamed_columns(pd.read_csv(Path(train_csv).expanduser()))
        annotated_df = compute_activity_cliff_annotation(
            dataset=base_dataset,
            smiles_column=normalized_smiles_columns[0],
            target_column=target_column,
            oof_predictions=oof_predictions,
            similarity_threshold=feedback_config.similarity_threshold,
            k_neighbors=feedback_config.k_neighbors,
        )
        cliff_summary = write_activity_cliff_artifacts(
            annotated_df=annotated_df,
            target_column=target_column,
            output_dir=str(root_output_path / "activity_cliff_feedback"),
            loops=feedback_config.loops,
            step_percentile=feedback_config.step_percentile,
            min_training_rows=MIN_TRAINING_ROWS,
        )

        variants: List[Dict[str, Any]] = []
        baseline_result = self._train_model_single(
            train_csv=train_csv,
            task_type=task_type,
            output_dir=str(root_output_path / "baseline_top_0"),
            smiles_columns=normalized_smiles_columns,
            target_columns=normalized_target_columns,
            reaction_columns=normalized_reaction_columns,
            extra_args=strip_activity_cliff_args(merged_extra_args),
            agent=agent,
        )
        variants.append(
            {
                "variant_id": "baseline_top_0",
                "removed_percent": 0.0,
                "removed_count": 0,
                "filtered_training_csv": train_csv,
                "training_result": baseline_result,
            }
        )
        for item in cliff_summary["variants"]:
            training_result = self._train_model_single(
                train_csv=item["filtered_training_csv"],
                task_type=task_type,
                output_dir=str(root_output_path / item["variant_id"]),
                smiles_columns=normalized_smiles_columns,
                target_columns=normalized_target_columns,
                reaction_columns=normalized_reaction_columns,
                extra_args=strip_activity_cliff_args(merged_extra_args),
                agent=agent,
            )
            variants.append({**item, "training_result": training_result})

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
            "loops_requested": feedback_config.loops,
            "step_percentile": feedback_config.step_percentile,
            "annotated_training_csv": cliff_summary["annotated_training_csv"],
            "filtered_training_csvs": [item["filtered_training_csv"] for item in variants[1:]],
            "ranked_molecule_count": cliff_summary["ranked_molecule_count"],
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
        summary_path = Path(selected_result["summary_path"])
        summary_path.write_text(json.dumps(selected_result, indent=2) + "\n")
        return selected_result
