#!/usr/bin/env python
# coding: utf-8
"""
Persistent catalog for predictive models.

The catalog is intentionally richer than session registration metadata so the
agent can select models using structured criteria instead of prompt-only
guesswork.  It is the first step toward a governed registry of validated
predictive assets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .backend import PredictionModelRecord


DEFAULT_MODEL_CATALOG_PATH = Path(__file__).with_name("model_catalog.json")
DEFAULT_INTERNAL_MODEL_ROOT = Path("data/model_assets/internal").resolve()
DEFAULT_ALLOWED_STATUSES = ("production", "validated", "experimental")
STATUS_WEIGHTS = {
    "production": 12,
    "validated": 8,
    "experimental": 4,
    "workflow_demo": -12,
    "deprecated": -20,
}


def _normalize_text(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _flatten_text(values: Iterable[Any]) -> str:
    parts: List[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            parts.append(_flatten_text(value.values()))
        elif isinstance(value, (list, tuple, set)):
            parts.append(_flatten_text(value))
        else:
            parts.append(str(value))
    return " ".join(parts).lower()


@dataclass
class CatalogRecommendation:
    """Structured recommendation result for a model search."""

    record: PredictionModelRecord
    score: int
    reasons: List[str]
    warnings: List[str]
    backend_available: bool
    model_path_exists: bool
    runtime_compatible: bool

    def as_dict(self) -> Dict[str, Any]:
        payload = self.record.as_dict()
        payload.update(
            {
                "score": self.score,
                "reasons": list(self.reasons),
                "warnings": list(self.warnings),
                "backend_available": self.backend_available,
                "model_path_exists": self.model_path_exists,
                "runtime_compatible": self.runtime_compatible,
            }
        )
        return payload


class PredictionModelCatalog:
    """Access layer for the prediction model catalog."""

    def __init__(
        self,
        records: List[PredictionModelRecord],
        source_path: Path,
        schema_version: int = 1,
    ):
        self.records = records
        self.source_path = source_path
        self.schema_version = schema_version

    @classmethod
    def load(cls, path: Optional[str] = None) -> "PredictionModelCatalog":
        source_path = Path(path).expanduser() if path else DEFAULT_MODEL_CATALOG_PATH
        payload = json.loads(source_path.read_text())
        records = [
            PredictionModelRecord.from_dict(record_payload)
            for record_payload in payload.get("models", [])
        ]
        return cls(
            records=records,
            source_path=source_path,
            schema_version=int(payload.get("schema_version", 1)),
        )

    def save(self) -> None:
        payload = {
            "schema_version": self.schema_version,
            "models": [record.as_dict() for record in self.records],
        }
        self.source_path.write_text(json.dumps(payload, indent=2) + "\n")

    def list_models(self) -> List[PredictionModelRecord]:
        return list(self.records)

    def get_model(self, model_id: str) -> PredictionModelRecord:
        for record in self.records:
            if record.model_id == model_id:
                return record
        raise ValueError(f"Unknown catalog model_id: {model_id}")

    def upsert_model(self, record: PredictionModelRecord) -> PredictionModelRecord:
        """Insert or replace a model record and persist the catalog."""
        for index, existing in enumerate(self.records):
            if existing.model_id == record.model_id:
                self.records[index] = record
                break
        else:
            self.records.append(record)

        self.records.sort(key=lambda item: item.model_id)
        self.save()
        return record

    def search(
        self,
        *,
        task_type: Optional[str] = None,
        target_hint: Optional[str] = None,
        domain_hint: Optional[str] = None,
        require_uncertainty: bool = False,
        allowed_statuses: Optional[List[str]] = None,
        preferred_backend: Optional[str] = None,
        backend_available: Optional[bool] = None,
        available_backend_names: Optional[List[str]] = None,
        include_unavailable_paths: bool = False,
    ) -> List[CatalogRecommendation]:
        allowed = {status.lower() for status in (allowed_statuses or DEFAULT_ALLOWED_STATUSES)}
        normalized_task = _normalize_text(task_type)
        normalized_target = _normalize_text(target_hint)
        normalized_domain = _normalize_text(domain_hint)
        normalized_backend = _normalize_text(preferred_backend)
        normalized_available_backends = {
            _normalize_text(name) for name in (available_backend_names or []) if name
        }

        recommendations: List[CatalogRecommendation] = []
        for record in self.records:
            status = _normalize_text(record.status)
            if allowed and status not in allowed:
                continue

            path_exists = Path(record.model_path).expanduser().exists()
            if not include_unavailable_paths and not path_exists:
                continue

            score = 0
            reasons: List[str] = []
            warnings: List[str] = []

            if normalized_task:
                if _normalize_text(record.task.task_type) != normalized_task:
                    continue
                score += 50
                reasons.append(f"Task type matches `{record.task.task_type}`.")

            if normalized_backend:
                if _normalize_text(record.backend_name) != normalized_backend:
                    continue
                score += 8
                reasons.append(f"Backend preference matches `{record.backend_name}`.")

            if record.task.uncertainty_method:
                score += 4
            elif require_uncertainty:
                continue

            if require_uncertainty:
                reasons.append(
                    f"Supports uncertainty via `{record.task.uncertainty_method}`."
                )

            status_weight = STATUS_WEIGHTS.get(status, 0)
            if status_weight:
                score += status_weight
            reasons.append(f"Catalog status is `{record.status}`.")

            target_text = _flatten_text(
                [
                    record.task.target_columns,
                    record.model_id,
                    record.display_name,
                    record.description,
                    record.recommended_for,
                    record.tags,
                ]
            )
            if normalized_target:
                if normalized_target in target_text:
                    score += 25
                    reasons.append(f"Target hint `{target_hint}` matches catalog metadata.")
                elif normalized_target in _flatten_text(record.not_recommended_for):
                    score -= 10
                    warnings.append(
                        f"Catalog flags this model as a weak fit for `{target_hint}`."
                    )

            domain_text = _flatten_text(
                [
                    record.domain_summary,
                    record.recommended_for,
                    record.training_data_summary,
                    record.selection_hints,
                ]
            )
            if normalized_domain:
                if normalized_domain in domain_text:
                    score += 18
                    reasons.append(
                        f"Domain hint `{domain_hint}` matches the model domain metadata."
                    )
                elif normalized_domain in _flatten_text(record.not_recommended_for):
                    score -= 8
                    warnings.append(
                        f"Catalog indicates limited suitability for `{domain_hint}`."
                    )

            if path_exists:
                score += 6
                reasons.append("Model artifact path is currently available.")
            else:
                score -= 18
                warnings.append("Model artifact path is not currently available.")

            if backend_available is True:
                score += 6
                reasons.append("Backend is available in the current environment.")
            elif backend_available is False:
                warnings.append("Backend is not available in the current environment.")

            runtime_validated = bool(
                (record.inference_profile or {}).get("runtime_validated", False)
            )
            runtime_compatible = True

            if normalized_available_backends:
                if _normalize_text(record.backend_name) in normalized_available_backends:
                    score += 10
                    reasons.append(
                        f"Backend `{record.backend_name}` is available in the active runtime."
                    )
                else:
                    runtime_compatible = False
                    score -= 24
                    warnings.append(
                        "Catalog metadata indicates this model uses a backend "
                        f"(`{record.backend_name}`) that is not currently available in the runtime."
                    )

            if runtime_validated:
                score += 8
                reasons.append("Runtime execution has been validated in this project.")
            else:
                runtime_compatible = False
                score -= 12
                warnings.append(
                    "Runtime execution has not yet been validated in this project."
                )

            recommendations.append(
                CatalogRecommendation(
                    record=record,
                    score=score,
                    reasons=reasons,
                    warnings=warnings,
                    backend_available=backend_available is not False,
                    model_path_exists=path_exists,
                    runtime_compatible=runtime_compatible and path_exists,
                )
            )

        recommendations.sort(key=lambda item: item.score, reverse=True)
        return recommendations

    def recommend(
        self,
        *,
        task_type: str,
        target_hint: Optional[str] = None,
        domain_hint: Optional[str] = None,
        require_uncertainty: bool = False,
        allowed_statuses: Optional[List[str]] = None,
        preferred_backend: Optional[str] = None,
        backend_available: Optional[bool] = None,
        available_backend_names: Optional[List[str]] = None,
        include_unavailable_paths: bool = False,
    ) -> Dict[str, Any]:
        candidates = self.search(
            task_type=task_type,
            target_hint=target_hint,
            domain_hint=domain_hint,
            require_uncertainty=require_uncertainty,
            allowed_statuses=allowed_statuses,
            preferred_backend=preferred_backend,
            backend_available=backend_available,
            available_backend_names=available_backend_names,
            include_unavailable_paths=include_unavailable_paths,
        )
        if not candidates:
            return {
                "catalog_path": str(self.source_path),
                "selected_model": None,
                "runnable_candidate": None,
                "alternatives": [],
                "selection_summary": (
                    "No compatible model was found in the catalog for the requested task."
                ),
            }

        selected = candidates[0]
        runnable_candidate = next(
            (candidate for candidate in candidates if candidate.runtime_compatible),
            None,
        )
        summary = " | ".join(selected.reasons[:3])
        if runnable_candidate is None:
            summary += " | No runtime-compatible candidate is currently validated."
        return {
            "catalog_path": str(self.source_path),
            "selected_model": selected.as_dict(),
            "runnable_candidate": runnable_candidate.as_dict() if runnable_candidate else None,
            "alternatives": [candidate.as_dict() for candidate in candidates[1:4]],
            "selection_summary": summary,
        }
