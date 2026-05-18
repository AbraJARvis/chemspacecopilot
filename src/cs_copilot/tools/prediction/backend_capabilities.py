#!/usr/bin/env python
# coding: utf-8
"""Machine-readable capability contracts for QSAR prediction backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class BackendCapabilities:
    """Static capabilities expected from a prediction backend."""

    backend_name: str
    can_train: bool
    can_predict: bool
    prediction_input_kinds: Tuple[str, ...]
    requires_feature_preparation: bool
    supported_task_types: Tuple[str, ...]
    supported_representations: Tuple[str, ...]
    supports_applicability_domain: bool
    supports_uncertainty: str
    supports_component_orchestration: bool = False

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


BACKEND_CAPABILITIES: Dict[str, BackendCapabilities] = {
    "chemprop": BackendCapabilities(
        backend_name="chemprop",
        can_train=True,
        can_predict=True,
        prediction_input_kinds=("smiles_csv",),
        requires_feature_preparation=False,
        supported_task_types=("regression",),
        supported_representations=("molecular_graph",),
        supports_applicability_domain=True,
        supports_uncertainty="none",
    ),
    "lightgbm": BackendCapabilities(
        backend_name="lightgbm",
        can_train=True,
        can_predict=True,
        prediction_input_kinds=("tabular_features_csv",),
        requires_feature_preparation=True,
        supported_task_types=("regression",),
        supported_representations=(
            "morgan_only",
            "rdkit_basic_only",
            "morgan_rdkit_basic",
            "morgan_rdkit_all",
        ),
        supports_applicability_domain=True,
        supports_uncertainty="none",
    ),
    "tabicl": BackendCapabilities(
        backend_name="tabicl",
        can_train=True,
        can_predict=True,
        prediction_input_kinds=("tabular_features_csv",),
        requires_feature_preparation=True,
        supported_task_types=("regression",),
        supported_representations=(
            "morgan_only",
            "rdkit_basic_only",
            "morgan_rdkit_basic",
            "morgan_rdkit_all",
        ),
        supports_applicability_domain=True,
        supports_uncertainty="none",
    ),
    "ensemble": BackendCapabilities(
        backend_name="ensemble",
        can_train=False,
        can_predict=True,
        prediction_input_kinds=("smiles_csv",),
        requires_feature_preparation=False,
        supported_task_types=("regression",),
        supported_representations=("catalog_consensus_regression",),
        supports_applicability_domain=False,
        supports_uncertainty="component_disagreement_std",
        supports_component_orchestration=True,
    ),
}


def get_backend_capabilities(
    backend_name: str,
    *,
    registry: Mapping[str, BackendCapabilities] | None = None,
) -> BackendCapabilities:
    """Return capabilities for a registered backend."""
    capabilities_registry = BACKEND_CAPABILITIES if registry is None else registry
    normalized = str(backend_name).strip().lower()
    try:
        return capabilities_registry[normalized]
    except KeyError as exc:
        raise KeyError(f"No backend capabilities registered for `{backend_name}`.") from exc


def describe_backend_capabilities() -> Dict[str, Dict[str, object]]:
    """Return all known backend capabilities as serializable dictionaries."""
    return {
        name: capabilities.as_dict()
        for name, capabilities in BACKEND_CAPABILITIES.items()
    }


def backend_requires_feature_preparation(
    backend_name: str,
    *,
    registry: Mapping[str, BackendCapabilities] | None = None,
) -> bool:
    """Return whether a backend needs tabular feature preparation for SMILES inputs."""
    return get_backend_capabilities(
        backend_name,
        registry=registry,
    ).requires_feature_preparation
