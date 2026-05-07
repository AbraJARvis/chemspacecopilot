"""Adapter for the official ChEMBL Structure Pipeline package."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from rdkit import Chem

from cs_copilot.tools.curation.backends.legacy_rdkit_v1 import (
    standardize_with_legacy_rdkit_v1,
)
from cs_copilot.tools.curation.identity import (
    has_explicit_stereochemistry,
    strip_stereochemistry_from_smiles,
)


def _format_checker_issues(issues: List[Tuple[Any, Any]]) -> Tuple[str, int]:
    if not issues:
        return "", 0
    normalized = []
    for first, second in issues:
        if isinstance(first, int):
            penalty, message = int(first), str(second)
        else:
            penalty, message = int(second), str(first)
        normalized.append((penalty, message))
    max_penalty = max(penalty for penalty, _message in normalized)
    text = "; ".join(f"{penalty}:{message}" for penalty, message in normalized)
    return text, max_penalty


def _mol_to_molblock_from_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToMolBlock(mol, kekulize=False)


def standardize_with_chembl_structure_v1(raw_smiles: pd.Series) -> Dict[str, Any]:
    """Standardize a SMILES series with ChEMBL, then apply QSAR identity policy."""

    try:
        from chembl_structure_pipeline import checker, standardizer
    except Exception as exc:
        legacy = standardize_with_legacy_rdkit_v1(raw_smiles)
        legacy.update(
            {
                "backend_name": "chembl_structure_v1",
                "used_backend_name": "legacy_rdkit_v1",
                "fallback_used": True,
                "fallback_reason": f"chembl_structure_pipeline unavailable: {exc}",
            }
        )
        return legacy

    rows = []
    for row_index, smiles in raw_smiles.items():
        raw = smiles if isinstance(smiles, str) else None
        molblock = _mol_to_molblock_from_smiles(raw) if raw else None
        if molblock is None:
            rows.append(
                {
                    "row_index": row_index,
                    "raw_smiles": raw,
                    "standardized_smiles": None,
                    "qsar_identity_smiles": None,
                    "curation_identity_key": None,
                    "curation_identity_key_type": "qsar_identity_smiles",
                    "curation_backend_status": "invalid_smiles",
                    "checker_issues": "",
                    "checker_max_penalty": 0,
                    "parent_structure_changed": False,
                    "stereochemistry_removed_for_identity": False,
                }
            )
            continue

        try:
            issues = checker.check_molblock(molblock)
            checker_issues, checker_max_penalty = _format_checker_issues(issues)
            standardized_molblock = standardizer.standardize_molblock(molblock)
            parent_molblock, _exclude = standardizer.get_parent_molblock(standardized_molblock)
            parent_mol = Chem.MolFromMolBlock(parent_molblock, sanitize=True, removeHs=True)
            standardized = (
                Chem.MolToSmiles(parent_mol, canonical=True) if parent_mol is not None else None
            )
            qsar_identity = (
                strip_stereochemistry_from_smiles(standardized) if standardized else None
            )
            parent_structure_changed = bool(standardized and raw and standardized != raw)
            stereo_removed = bool(
                standardized
                and qsar_identity
                and has_explicit_stereochemistry(standardized)
                and qsar_identity != standardized
            )
            status = "ok" if standardized and qsar_identity else "standardization_failed"
        except Exception as exc:
            standardized = None
            qsar_identity = None
            checker_issues = f"standardization_error:{exc}"
            checker_max_penalty = 9
            parent_structure_changed = False
            stereo_removed = False
            status = "standardization_failed"

        rows.append(
            {
                "row_index": row_index,
                "raw_smiles": raw,
                "standardized_smiles": standardized,
                "qsar_identity_smiles": qsar_identity,
                "curation_identity_key": qsar_identity,
                "curation_identity_key_type": "qsar_identity_smiles",
                "curation_backend_status": status,
                "checker_issues": checker_issues,
                "checker_max_penalty": checker_max_penalty,
                "parent_structure_changed": parent_structure_changed,
                "stereochemistry_removed_for_identity": stereo_removed,
            }
        )

    return {
        "backend_name": "chembl_structure_v1",
        "used_backend_name": "chembl_structure_v1",
        "fallback_used": False,
        "fallback_reason": None,
        "identity_column": "curation_identity_key",
        "standardization_map": pd.DataFrame(rows),
    }
