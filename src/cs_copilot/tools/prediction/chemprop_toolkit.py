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
from pathlib import Path
from typing import Any, Dict, List, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.standardize import standardize_smiles_column

from .admet_ai_backend import AdmetAIBackend
from .backend import PredictionModelRecord, PredictionTaskSpec
from .catalog import DEFAULT_INTERNAL_MODEL_ROOT, PredictionModelCatalog
from .chemprop_backend import ChempropBackend


def _get_prediction_state(agent: Agent) -> Dict[str, Any]:
    state = agent.session_state.setdefault("prediction_models", {})
    state.setdefault("registered", {})
    state.setdefault("last_prediction", {})
    state.setdefault("prediction_history", [])
    state.setdefault("catalog_recommendations", {})
    state.setdefault("training_runs", [])
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


class ChempropToolkit(Toolkit):
    """Toolkit exposing Chemprop-backed property prediction workflows."""

    def __init__(self, backend: Optional[ChempropBackend] = None):
        super().__init__("chemprop_prediction")
        primary_backend = backend or ChempropBackend()
        self.backends = {
            primary_backend.backend_name: primary_backend,
            "admet_ai": AdmetAIBackend(),
        }
        self.backend = primary_backend
        self.catalog = PredictionModelCatalog.load()
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

    def describe_compute_environment(self) -> Dict[str, Any]:
        """Describe the local compute budget used to choose safe training defaults."""
        cpu_count = os.cpu_count() or 1
        memory_limit_bytes = self._detect_memory_limit_bytes()
        memory_gb_total = (
            round(memory_limit_bytes / (1024**3), 2) if memory_limit_bytes else None
        )
        gpu_available = bool(
            os.getenv("CUDA_VISIBLE_DEVICES")
            and os.getenv("CUDA_VISIBLE_DEVICES", "").strip() not in {"", "-1"}
        )
        execution_env = "docker_local" if Path("/.dockerenv").exists() else "local"
        profile = self._resolve_training_profile(
            {
                "cpu_count": cpu_count,
                "memory_gb_total": memory_gb_total,
                "gpu_available": gpu_available,
                "execution_env": execution_env,
            }
        )
        return {
            "execution_env": execution_env,
            "cpu_count": cpu_count,
            "memory_gb_total": memory_gb_total,
            "gpu_available": gpu_available,
            "suggested_profile": profile["profile"],
            "profile_reason": profile["reason"],
        }

    def _resolve_training_profile(self, compute_env: Dict[str, Any]) -> Dict[str, Any]:
        cpu_count = int(compute_env.get("cpu_count") or 1)
        memory_gb_total = compute_env.get("memory_gb_total")
        gpu_available = bool(compute_env.get("gpu_available"))
        execution_env = compute_env.get("execution_env") or "local"

        if not gpu_available and execution_env == "docker_local":
            if memory_gb_total is None and cpu_count <= 8:
                return {
                    "profile": "local_light",
                    "reason": "CPU-only Docker environment on a modest local machine; defaulting to the safest single-run profile.",
                }
            if memory_gb_total is not None and memory_gb_total <= 8.5 and cpu_count <= 8:
                return {
                    "profile": "local_light",
                    "reason": "CPU-only Docker environment with limited RAM; using a conservative single-run configuration.",
                }
            if memory_gb_total is not None and memory_gb_total <= 16 and cpu_count <= 12:
                return {
                    "profile": "local_standard",
                    "reason": "CPU-only local environment; using a moderate single-run configuration.",
                }

        if gpu_available:
            return {
                "profile": "heavy_validation",
                "reason": "GPU detected; heavier validation settings are acceptable.",
            }

        return {
            "profile": "local_standard",
            "reason": "Defaulting to a moderate single-run local profile.",
        }

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
                "epochs": 75,
                "batch_size": 32,
                "num_replicates": 3,
                "ensemble_size": 1,
                "num_workers": 0,
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
                merged["ensemble_size"] = 1
                merged["num_replicates"] = 1
                merged["num_workers"] = 0
            elif profile == "local_standard":
                merged["epochs"] = min(int(merged.get("epochs", 50)), 50)
                merged["ensemble_size"] = min(int(merged.get("ensemble_size", 1)), 1)
                merged["num_replicates"] = min(int(merged.get("num_replicates", 1)), 1)
                merged["num_workers"] = 0

        return {
            "compute_environment": compute_env,
            "training_profile": profile,
            "profile_reason": resolved["reason"],
            "extra_args": merged,
        }

    def describe_backend(self) -> Dict[str, Any]:
        """Describe Chemprop backend availability and version information."""
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
            target_model_path = model_dir / source_model_path.name
            shutil.copy2(source_model_path, target_model_path)
            copied_files["model_path"] = _relative_posix(target_model_path, model_root)
        else:
            return {"materialized": False, "reason": "model_artifact_missing"}

        optional_artifacts = {
            "config_path": run_dir / "config.toml",
            "training_summary_path": run_dir / "cs_copilot_training_summary.json",
            "splits_path": run_dir / "splits.json",
            "test_predictions_path": run_dir / "model_0" / "test_predictions.csv",
        }

        for key, source_path in optional_artifacts.items():
            if source_path.exists():
                target_path = (
                    model_dir / source_path.name
                    if key == "config_path"
                    else artifacts_dir / source_path.name
                )
                shutil.copy2(source_path, target_path)
                copied_files[key] = _relative_posix(target_path, model_root)

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
        metadata_path = model_root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

        return {
            "materialized": True,
            "model_root": str(model_root),
            "model_path": str(model_root / copied_files["model_path"]),
            "metadata_path": str(metadata_path),
            "artifacts": copied_files,
        }

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
        inferred_run_dir = self._infer_training_run_dir(current)
        if inferred_run_dir is not None:
            inferred_output = str(inferred_run_dir.resolve())
            for run in reversed(training_runs):
                if str(Path(run.get("output_dir", "")).expanduser().resolve()) == inferred_output:
                    train_csv = run.get("train_csv")
                    break

        materialized = self._materialize_internal_model(
            record=current,
            train_csv=train_csv,
            model_id=model_id,
        )
        resolved_model_path = materialized.get("model_path", current.model_path)
        resolved_metadata_path = materialized.get("metadata_path", current.metadata_path)
        persisted_record = PredictionModelRecord(
            model_id=current.model_id,
            backend_name=current.backend_name,
            model_path=resolved_model_path,
            metadata_path=resolved_metadata_path,
            task=current.task,
            display_name=display_name or current.display_name,
            description=description or current.description,
            tags=tags or current.tags,
            version=version or current.version,
            status=status or current.status,
            owner=owner or current.owner,
            source=source or current.source,
            domain_summary=domain_summary or current.domain_summary,
            strengths=strengths or current.strengths,
            limitations=limitations or current.limitations,
            recommended_for=recommended_for or current.recommended_for,
            not_recommended_for=not_recommended_for or current.not_recommended_for,
            known_metrics=known_metrics or current.known_metrics,
            training_data_summary=training_data_summary or current.training_data_summary,
            inference_profile=inference_profile or current.inference_profile,
            selection_hints=selection_hints or current.selection_hints,
        )

        self.catalog.upsert_model(persisted_record)
        self.catalog = PredictionModelCatalog.load(str(self.catalog.source_path))

        prediction_state["registered"][model_id] = persisted_record.as_dict()

        return {
            "catalog_path": str(self.catalog.source_path),
            "model_id": persisted_record.model_id,
            "status": persisted_record.status,
            "persisted": True,
            "materialized": bool(materialized.get("materialized")),
            "model_root": materialized.get("model_root"),
            "metadata_path": persisted_record.metadata_path,
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

        if not Path(input_csv).is_absolute():
            local_input = Path(input_csv)
        else:
            local_input = Path(input_csv)

        if not local_input.exists():
            with S3.open(input_csv, "r") as fh:
                df = pd.read_csv(fh)
            df = standardize_smiles_column(df, smiles_column)
            local_input = Path(".files") / "prediction_inputs" / f"{model_id}_input.csv"
            local_input.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(local_input, index=False)

        result = backend.predict_from_csv(
            input_csv=str(local_input),
            model_record=record,
            preds_path=str(output_path),
            return_uncertainty=return_uncertainty,
        )

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
        splits_path = output_path / "splits.json"
        preds_path = output_path / "model_0" / "test_predictions.csv"

        if not splits_path.exists() or not preds_path.exists():
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
        rmse = float(math.sqrt(mse))
        centered = y_true - float(y_true.mean())
        ss_tot = float((centered.pow(2)).sum())
        ss_res = float((residuals.pow(2)).sum())
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else None

        return {
            "test_predictions_path": str(preds_path),
            "splits_path": str(splits_path),
            "metrics": {
                "test": {
                    "mse": mse,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "n": int(valid_mask.sum()),
                    "target_column": target_column,
                }
            },
        }

    def prepare_training_dataset(
        self,
        input_csv: str,
        smiles_column: str,
        target_columns: List[str],
        output_csv: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalize a training CSV into the canonical format expected by later prediction tools."""
        with S3.open(input_csv, "r") as fh:
            df = pd.read_csv(fh)

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

    def train_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        smiles_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        reaction_columns: Optional[List[str]] = None,
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Launch Chemprop training and persist a lightweight training record."""
        resolved_output_dir = str(Path(output_dir).expanduser().resolve())
        training_policy = self._apply_training_profile(extra_args)
        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=smiles_columns or ["smiles"],
            target_columns=target_columns or [],
            reaction_columns=reaction_columns or [],
        )
        result = self.backend.train_model(
            train_csv=train_csv,
            output_dir=resolved_output_dir,
            task=task,
            extra_args=training_policy["extra_args"],
        )

        if agent is not None:
            prediction_state = _get_prediction_state(agent)
            prediction_state["training_runs"].append(
                {
                    "train_csv": train_csv,
                    "output_dir": resolved_output_dir,
                    "task_type": task_type,
                    "smiles_columns": task.smiles_columns,
                    "target_columns": task.target_columns,
                }
            )

        metric_payload = self._compute_training_metrics(
            train_csv=train_csv,
            output_dir=resolved_output_dir,
            task=task,
        )
        result.update(metric_payload)
        result["compute_environment"] = training_policy["compute_environment"]
        result["training_profile"] = training_policy["training_profile"]
        result["profile_reason"] = training_policy["profile_reason"]
        result["effective_train_args"] = training_policy["extra_args"]

        training_summary_path = Path(resolved_output_dir) / "cs_copilot_training_summary.json"
        training_summary_path.parent.mkdir(parents=True, exist_ok=True)
        training_summary_path.write_text(json.dumps(result, indent=2))
        best_model_path = Path(resolved_output_dir) / "model_0" / "best.pt"
        config_path = Path(resolved_output_dir) / "config.toml"
        splits_path = Path(resolved_output_dir) / "splits.json"
        result["summary_path"] = str(training_summary_path)
        if best_model_path.exists():
            result["best_model_path"] = str(best_model_path)
            result["download_file_ref"] = str(best_model_path)
        result["summary_file_ref"] = str(training_summary_path)
        if metric_payload.get("test_predictions_path"):
            result["test_predictions_file_ref"] = metric_payload["test_predictions_path"]
        bundle_path = (
            Path(".files")
            / "prediction_outputs"
            / f"{Path(resolved_output_dir).name}_training_bundle.zip"
        ).resolve()
        bundle = _bundle_artifacts(
            bundle_path,
            [
                Path(train_csv).expanduser(),
                training_summary_path,
                best_model_path,
                config_path,
                splits_path,
                Path(metric_payload["test_predictions_path"]).expanduser()
                if metric_payload.get("test_predictions_path")
                else Path("/nonexistent"),
            ],
        )
        result["bundle_file_ref"] = str(bundle)
        return result
