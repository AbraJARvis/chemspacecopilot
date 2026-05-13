#!/usr/bin/env python
# coding: utf-8
"""
Toolkit for building catalogued QSAR consensus ensembles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from .backend import InvalidPredictionInputError, PredictionModelRecord, PredictionTaskSpec
from .catalog import DEFAULT_INTERNAL_MODEL_ROOT, PredictionModelCatalog
from .ensemble_backend import (
    DEFAULT_AGGREGATION_STRATEGY,
    DEFAULT_ENSEMBLE_KIND,
    DEFAULT_UNCERTAINTY_STRATEGY,
    ENSEMBLE_SCHEMA_VERSION,
    EnsembleBackend,
)
from .qsar_training_policy import project_now, safe_slug


def _get_prediction_state(agent: Agent) -> Dict[str, Any]:
    state = agent.session_state.setdefault("prediction_models", {})
    state.setdefault("registered", {})
    state.setdefault("last_prediction", {})
    state.setdefault("prediction_history", [])
    state.setdefault("catalog_recommendations", {})
    state.setdefault("training_runs", [])
    return state


def _coerce_list(value: Optional[List[str] | str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            if "," in stripped:
                return [item.strip() for item in stripped.split(",") if item.strip()]
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if isinstance(parsed, str):
            return [parsed]
        raise ValueError("Expected a list, scalar string, or JSON-encoded list.")
    return [str(item) for item in value]


def _coerce_status(status: Optional[str]) -> str:
    normalized = (status or "workflow_demo").strip().lower()
    if not normalized:
        return "workflow_demo"
    return normalized


PROMOTED_ENSEMBLE_STATUSES = {"validated", "robust_validated", "production"}
ABLATION_REPRESENTATIONS = {"morgan_only", "rdkit_basic_only", "rdkit_all_only"}


class EnsembleToolkit(Toolkit):
    """Create and register future-proof QSAR consensus ensemble artifacts."""

    def __init__(self, backend: Optional[EnsembleBackend] = None):
        super().__init__("ensemble_prediction")
        self.backend = backend or EnsembleBackend()
        self.catalog = PredictionModelCatalog.load()
        self.catalog.refresh_from_internal_store(persist=False)
        self.register(self.describe_ensemble_backend)
        self.register(self.create_ensemble_from_catalog_models)
        self.register(self.create_ensemble_from_catalog_search)

    def describe_ensemble_backend(self) -> Dict[str, Any]:
        """Describe the ensemble backend and V1 constraints."""
        return self.backend.describe_environment()

    def _resolve_record(self, model_id: str, agent: Optional[Agent] = None) -> PredictionModelRecord:
        if agent is not None:
            registered = _get_prediction_state(agent).get("registered") or {}
            if model_id in registered:
                return PredictionModelRecord.from_dict(registered[model_id])
        self.catalog.refresh_from_internal_store(persist=True)
        return self.catalog.get_model(model_id)

    def _validate_components(self, records: List[PredictionModelRecord]) -> PredictionTaskSpec:
        if len(records) < 2:
            raise InvalidPredictionInputError("An ensemble requires at least two component models.")
        first = records[0]
        if first.task.task_type != "regression":
            raise InvalidPredictionInputError("Ensemble V1 only supports regression tasks.")
        if len(first.task.target_columns) != 1:
            raise InvalidPredictionInputError("Ensemble V1 requires exactly one target column.")
        for record in records:
            if record.backend_name == self.backend.backend_name:
                raise InvalidPredictionInputError("Nested ensemble components are not supported in V1.")
            if record.status == "deprecated":
                raise InvalidPredictionInputError(f"Deprecated component is not allowed: {record.model_id}")
            if record.task.task_type != first.task.task_type:
                raise InvalidPredictionInputError("All ensemble components must use the same task type.")
            if list(record.task.target_columns) != list(first.task.target_columns):
                raise InvalidPredictionInputError("All ensemble components must share the same target column.")
            component_backend = self.backend.backends.get(record.backend_name)
            if component_backend is None:
                raise InvalidPredictionInputError(
                    f"Unsupported ensemble component backend: {record.backend_name}"
                )
            if not component_backend.is_available():
                raise InvalidPredictionInputError(
                    f"Component backend is not available: {record.backend_name}"
                )
            component_backend.validate_model_path(record.model_path)
        return PredictionTaskSpec(
            task_type=first.task.task_type,
            smiles_columns=list(first.task.smiles_columns or ["smiles"]),
            target_columns=list(first.task.target_columns),
            reaction_columns=[],
            uncertainty_method=DEFAULT_UNCERTAINTY_STRATEGY,
            calibration_method=None,
        )

    def _component_representation(self, record: PredictionModelRecord) -> str:
        return (
            str((record.training_data_summary or {}).get("representation_name") or "")
            or str((record.inference_profile or {}).get("representation_name") or "")
            or str((record.selection_hints or {}).get("representation_name") or "")
        )

    def _govern_ensemble_status(self, requested_status: str) -> tuple[str, Optional[str]]:
        normalized = _coerce_status(requested_status)
        if normalized in PROMOTED_ENSEMBLE_STATUSES:
            return (
                "workflow_demo",
                (
                    f"Requested status `{normalized}` was downgraded to `workflow_demo`: "
                    "catalogue ensembles are not promoted in V1 without a dedicated ensemble-level validation gate."
                ),
            )
        return normalized, None

    def _build_model_id(
        self,
        *,
        ensemble_name: Optional[str],
        target_columns: List[str],
        created_at: str,
    ) -> str:
        if ensemble_name:
            base = safe_slug(ensemble_name)
        else:
            target = safe_slug(target_columns[0]) if target_columns else "target"
            base = f"ensemble_{target}_consensus"
        suffix = safe_slug(created_at.replace(":", "").replace("+", "_"))
        return f"{base}_{suffix}"[:180].strip("_")

    def create_ensemble_from_catalog_models(
        self,
        model_ids: List[str] | str,
        ensemble_name: Optional[str] = None,
        aggregation_strategy: str = DEFAULT_AGGREGATION_STRATEGY,
        status: str = "workflow_demo",
        description: Optional[str] = None,
        source: str = "catalogue_explicit_request",
        known_metrics: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Create and persist a catalogued regression consensus ensemble."""
        ids = _coerce_list(model_ids)
        if len(ids) < 2:
            raise InvalidPredictionInputError("Provide at least two model_ids to create an ensemble.")
        if aggregation_strategy != DEFAULT_AGGREGATION_STRATEGY:
            raise InvalidPredictionInputError(
                "Ensemble V1 only supports aggregation_strategy='median'."
            )

        records = [self._resolve_record(model_id, agent=agent) for model_id in ids]
        task = self._validate_components(records)
        final_status, status_policy_note = self._govern_ensemble_status(status)
        component_warnings: List[str] = []
        ablation_components = [
            record.model_id
            for record in records
            if self._component_representation(record) in ABLATION_REPRESENTATIONS
        ]
        if ablation_components:
            component_warnings.append(
                "Ablation representation components are included because they were explicitly selected: "
                + ", ".join(ablation_components)
            )
        if status_policy_note:
            component_warnings.append(status_policy_note)
        created_at = project_now().isoformat()
        model_id = self._build_model_id(
            ensemble_name=ensemble_name,
            target_columns=task.target_columns,
            created_at=created_at,
        )
        model_root = DEFAULT_INTERNAL_MODEL_ROOT / model_id
        model_dir = model_root / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        ensemble_path = model_dir / "ensemble.json"

        component_payloads = []
        for index, record in enumerate(records):
            component_payloads.append(
                {
                    "component_index": index,
                    "model_id": record.model_id,
                    "backend_name": record.backend_name,
                    "status": record.status,
                    "display_name": record.display_name or record.model_id,
                    "known_metrics": dict(record.known_metrics or {}),
                    "training_data_summary": dict(record.training_data_summary or {}),
                    "selection_hints": dict(record.selection_hints or {}),
                    "record": record.as_dict(),
                }
            )

        artifact = {
            "schema_version": ENSEMBLE_SCHEMA_VERSION,
            "ensemble_kind": DEFAULT_ENSEMBLE_KIND,
            "aggregation_strategy": aggregation_strategy,
            "uncertainty_strategy": DEFAULT_UNCERTAINTY_STRATEGY,
            "created_at": created_at,
            "source": source,
            "provenance": {
                "created_by": "EnsembleToolkit.create_ensemble_from_catalog_models",
                "explicit_request_required": True,
                "requested_status": status,
                "final_status": final_status,
                "status_policy_note": status_policy_note,
            },
            "task": {
                "task_type": task.task_type,
                "smiles_columns": list(task.smiles_columns),
                "target_columns": list(task.target_columns),
                "reaction_columns": [],
            },
            "components": component_payloads,
            "future_extension_slots": {
                "weights": None,
                "calibration": None,
                "stacking": None,
                "split_families": None,
                "selection_protocol": None,
            },
            "warnings": component_warnings,
        }
        ensemble_path.write_text(json.dumps(artifact, indent=2) + "\n")

        inferred_protocol = None
        for protocol_name in ("robust_qsar", "standard_qsar", "challenging_qsar", "fast_local"):
            if protocol_name in source:
                inferred_protocol = protocol_name
                break
        training_summary = {
            "created_at": created_at,
            "component_count": len(records),
            "component_model_ids": [record.model_id for record in records],
            "aggregation_strategy": aggregation_strategy,
            "uncertainty_strategy": DEFAULT_UNCERTAINTY_STRATEGY,
        }
        if inferred_protocol:
            training_summary["benchmark_mode"] = inferred_protocol
            training_summary["validation_protocol"] = inferred_protocol
        if component_warnings:
            training_summary["component_warnings"] = component_warnings

        record = PredictionModelRecord(
            model_id=model_id,
            backend_name=self.backend.backend_name,
            model_path=str(ensemble_path.resolve()),
            metadata_path=str((model_root / "metadata.json").resolve()),
            display_name=ensemble_name or f"Consensus ensemble for {', '.join(task.target_columns)}",
            description=description
            or "Median consensus ensemble over compatible catalogued QSAR models.",
            tags={"model_family": "ensemble", "aggregation_strategy": aggregation_strategy},
            version="1.0",
            status=final_status,
            owner="chemspacecopilot",
            source=source,
            domain_summary="Consensus applicability is governed by component model compatibility and disagreement.",
            strengths=[
                "transparent component-level predictions",
                "robust median aggregation",
                "component disagreement reported as uncertainty proxy",
            ],
            limitations=[
                "not calibrated uncertainty",
                "not a stacked meta-learner",
                "requires all component inputs to be compatible at prediction time",
            ],
            recommended_for=["methodological consensus QSAR exploration"],
            not_recommended_for=["regulatory decisions without external validation"],
            known_metrics=dict(known_metrics or {}),
            training_data_summary=training_summary,
            inference_profile={
                "aggregation_strategy": aggregation_strategy,
                "uncertainty_strategy": DEFAULT_UNCERTAINTY_STRATEGY,
                "component_count": len(records),
            },
            selection_hints={
                "ensemble_kind": DEFAULT_ENSEMBLE_KIND,
                "component_model_ids": [record.model_id for record in records],
            },
            applicability_domain={},
            task=task,
        )
        metadata = {
            **record.as_dict(),
            "artifacts": {
                "model_path": "model/ensemble.json",
                "ensemble_path": "model/ensemble.json",
            },
            "ensemble": {
                "schema_version": ENSEMBLE_SCHEMA_VERSION,
                "ensemble_kind": DEFAULT_ENSEMBLE_KIND,
                "aggregation_strategy": aggregation_strategy,
                "uncertainty_strategy": DEFAULT_UNCERTAINTY_STRATEGY,
                "component_model_ids": [record.model_id for record in records],
                "warnings": component_warnings,
                "requested_status": status,
                "final_status": final_status,
            },
        }
        Path(record.metadata_path or model_root / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        self.catalog.upsert_model(record)
        self.catalog = PredictionModelCatalog.load(str(self.catalog.source_path))
        if agent is not None:
            state = _get_prediction_state(agent)
            state["registered"][model_id] = record.as_dict()

        return {
            "created": True,
            "model_id": model_id,
            "backend_name": self.backend.backend_name,
            "model_path": str(ensemble_path.resolve()),
            "metadata_path": record.metadata_path,
            "catalog_path": str(self.catalog.source_path),
            "status": record.status,
            "aggregation_strategy": aggregation_strategy,
            "uncertainty_strategy": DEFAULT_UNCERTAINTY_STRATEGY,
            "component_count": len(records),
            "component_model_ids": [record.model_id for record in records],
            "record": record.as_dict(),
        }

    def create_ensemble_from_catalog_search(
        self,
        target_hint: str,
        task_type: str = "regression",
        allowed_statuses: Optional[List[str] | str] = None,
        max_components: int = 5,
        ensemble_name: Optional[str] = None,
        aggregation_strategy: str = DEFAULT_AGGREGATION_STRATEGY,
        status: str = "workflow_demo",
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Create an ensemble from compatible persisted models matching a target hint."""
        statuses = _coerce_list(allowed_statuses) or [
            "production",
            "robust_validated",
            "validated",
            "experimental",
            "workflow_demo",
        ]
        self.catalog.refresh_from_internal_store(persist=False)
        recommendations = self.catalog.search(
            task_type=task_type,
            target_hint=target_hint,
            allowed_statuses=statuses,
            available_backend_names=[
                name for name, backend in self.backend.backends.items() if backend.is_available()
            ],
            include_unavailable_paths=False,
        )
        model_ids: List[str] = []
        seen_backend_representations: set[tuple[str, str]] = set()
        for item in recommendations:
            record = item.record
            if record.backend_name == self.backend.backend_name or record.status == "deprecated":
                continue
            representation = (
                str((record.training_data_summary or {}).get("representation_name") or "")
                or str((record.inference_profile or {}).get("representation_name") or "")
                or "default"
            )
            if representation in ABLATION_REPRESENTATIONS:
                continue
            family = (record.backend_name, representation)
            if family in seen_backend_representations:
                continue
            model_ids.append(record.model_id)
            seen_backend_representations.add(family)
            if len(model_ids) >= max(2, int(max_components)):
                break
        if len(model_ids) < 2:
            raise InvalidPredictionInputError(
                f"Could not find at least two compatible catalog models for target `{target_hint}`."
            )
        return self.create_ensemble_from_catalog_models(
            model_ids=model_ids,
            ensemble_name=ensemble_name or f"{target_hint}_catalog_consensus",
            aggregation_strategy=aggregation_strategy,
            status=status,
            description=(
                f"Median consensus ensemble built from catalog search for target `{target_hint}`."
            ),
            source="catalogue_explicit_search_request",
            agent=agent,
        )
