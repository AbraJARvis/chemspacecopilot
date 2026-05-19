#!/usr/bin/env python
# coding: utf-8
"""Backend-neutral prediction model registry and catalog facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from .backend import PredictionModelRecord, PredictionTaskSpec
from .catalog import PredictionModelCatalog
from .session_state import get_prediction_state


class PredictionRegistryToolkit(Toolkit):
    """Backend-neutral registry/catalog operations for prediction models."""

    def __init__(
        self,
        *,
        backends: Mapping[str, Any],
        catalog: Optional[PredictionModelCatalog] = None,
        default_backend_name: str = "chemprop",
        register_tools: bool = True,
    ):
        super().__init__("prediction_registry")
        self.backends = dict(backends)
        self.default_backend_name = default_backend_name
        self.catalog = catalog or PredictionModelCatalog.load()
        self.catalog.refresh_from_internal_store(persist=True)

        if register_tools:
            self.register(self.describe_backends)
            self.register(self.describe_catalog)
            self.register(self.list_catalog_models)
            self.register(self.summarize_catalog_model)
            self.register(self.recommend_catalog_model)
            self.register(self.register_catalog_model)
            self.register(self.register_model)
            self.register(self.list_registered_models)
            self.register(self.summarize_model)

    def describe_backends(self) -> Dict[str, Any]:
        """Describe all configured prediction backends."""
        return {
            name: backend.describe_environment()
            for name, backend in self.backends.items()
        }

    def get_backend(self, backend_name: str):
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

    def annotate_record(self, record: PredictionModelRecord) -> Dict[str, Any]:
        payload = record.as_dict()
        payload["backend_environment"] = self.get_backend(
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
        self.catalog.refresh_from_internal_store(persist=True)
        return self.annotate_record(self.catalog.get_model(model_id))

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
            prediction_state = get_prediction_state(agent)
            prediction_state["catalog_recommendations"] = recommendation

        return recommendation

    def register_catalog_model(self, model_id: str, agent: Optional[Agent] = None) -> Dict[str, Any]:
        """Register a model from the persistent catalog into the current session."""
        if agent is None:
            raise ValueError("Agent is required to register a catalog model")

        self.catalog.refresh_from_internal_store(persist=True)
        try:
            record = self.catalog.get_model(model_id)
        except ValueError as exc:
            available_ids = [record.model_id for record in self.catalog.list_models()]
            return {
                "registered": False,
                "error": str(exc),
                "model_id": model_id,
                "available_model_ids": available_ids,
                "usage_hint": (
                    "register_catalog_model only accepts an existing persistent catalog model_id. "
                    "For a newly trained session model, call persist_registered_model and use its returned "
                    "canonical model_id instead of inventing a display name."
                ),
            }
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
        """Register a prediction model in session state for later use."""
        if agent is None:
            raise ValueError("Agent is required to register a model")

        resolved_backend_name = backend_name or self.default_backend_name
        backend = self.get_backend(resolved_backend_name)
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

        prediction_state = get_prediction_state(agent)
        prediction_state["registered"][model_id] = record.as_dict()
        return record.as_dict()

    def list_registered_models(self, agent: Optional[Agent] = None) -> List[Dict[str, Any]]:
        """List models registered in the current session."""
        if agent is None:
            return []
        prediction_state = get_prediction_state(agent)
        return list(prediction_state["registered"].values())

    def summarize_model(self, model_id: str, agent: Optional[Agent] = None) -> Dict[str, Any]:
        """Return the stored summary for a registered model."""
        if agent is None:
            raise ValueError("Agent is required to summarize a model")
        return self.annotate_record(self.resolve_record(model_id, agent))

    def resolve_record(self, model_id: str, agent: Agent) -> PredictionModelRecord:
        prediction_state = get_prediction_state(agent)
        payload = prediction_state["registered"].get(model_id)
        if payload is None:
            raise ValueError(f"Unknown model_id: {model_id}")
        return PredictionModelRecord.from_dict(payload)
