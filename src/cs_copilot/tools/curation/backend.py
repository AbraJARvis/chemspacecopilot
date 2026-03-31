#!/usr/bin/env python
# coding: utf-8
"""
Structured contracts for the QSAR dataset curation workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CurationRequest:
    """Declarative request for preparing a QSAR-ready dataset."""

    dataset_path: str
    task_type: str
    endpoint_name: Optional[str] = None
    dataset_id: Optional[str] = None
    preferred_smiles_column: Optional[str] = None
    preferred_target_columns: List[str] = field(default_factory=list)
    curation_policy: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "task_type": self.task_type,
            "endpoint_name": self.endpoint_name,
            "dataset_id": self.dataset_id,
            "preferred_smiles_column": self.preferred_smiles_column,
            "preferred_target_columns": list(self.preferred_target_columns),
            "curation_policy": self.curation_policy,
        }


@dataclass
class TargetSummary:
    """Compact numeric summary for target columns after curation."""

    column: str
    mean: Optional[float] = None
    std: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    median: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "median": self.median,
        }


@dataclass
class CurationResult:
    """Structured output from the curation workflow."""

    status: str
    ready_for_qsar: bool
    dataset_id: Optional[str]
    source_dataset_path: str
    curated_dataset_path: Optional[str]
    smiles_column_original: Optional[str]
    smiles_column_curated: Optional[str]
    target_columns_original: List[str]
    target_columns_curated: List[str]
    task_type: str
    rows_in: int
    rows_out: int
    invalid_smiles_removed: int = 0
    duplicate_rows_removed: int = 0
    missing_target_removed: int = 0
    target_summaries: List[TargetSummary] = field(default_factory=list)
    curation_actions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blocking_issues: List[str] = field(default_factory=list)
    report_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "ready_for_qsar": self.ready_for_qsar,
            "dataset_id": self.dataset_id,
            "source_dataset_path": self.source_dataset_path,
            "curated_dataset_path": self.curated_dataset_path,
            "smiles_column_original": self.smiles_column_original,
            "smiles_column_curated": self.smiles_column_curated,
            "target_columns_original": list(self.target_columns_original),
            "target_columns_curated": list(self.target_columns_curated),
            "task_type": self.task_type,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_removed": self.rows_in - self.rows_out,
            "invalid_smiles_removed": self.invalid_smiles_removed,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "missing_target_removed": self.missing_target_removed,
            "target_summaries": [summary.as_dict() for summary in self.target_summaries],
            "curation_actions": list(self.curation_actions),
            "warnings": list(self.warnings),
            "blocking_issues": list(self.blocking_issues),
            "report_path": self.report_path,
        }
