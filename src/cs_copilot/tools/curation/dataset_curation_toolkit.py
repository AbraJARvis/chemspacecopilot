#!/usr/bin/env python
# coding: utf-8
"""
Toolkit dedicated to QSAR dataset curation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.standardize import standardize_smiles_column

from .backend import CurationRequest, CurationResult, TargetSummary


def _get_curation_state(agent: Agent) -> Dict[str, Any]:
    state = agent.session_state.setdefault("qsar_curation", {})
    state.setdefault("last_request", {})
    state.setdefault("last_result", {})
    state.setdefault("history", [])
    return state


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path).expanduser()
    if path.exists():
        return pd.read_csv(path)
    with S3.open(dataset_path, "r") as fh:
        return pd.read_csv(fh)


class DatasetCurationToolkit(Toolkit):
    """Tools for preparing QSAR-ready datasets before training."""

    def __init__(self):
        super().__init__("qsar_dataset_curation")
        self.register(self.inspect_dataset_schema)
        self.register(self.identify_qsar_columns)
        self.register(self.curate_qsar_dataset)
        self.register(self.summarize_curated_dataset)
        self.register(self.write_curation_report)

    def inspect_dataset_schema(self, dataset_path: str) -> Dict[str, Any]:
        """Inspect columns, size, and a small preview for a dataset."""
        df = _load_dataset(dataset_path)
        return {
            "dataset_path": dataset_path,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
            "preview": df.head(5).to_dict(orient="records"),
        }

    def identify_qsar_columns(
        self,
        dataset_path: str,
        task_type: str,
        preferred_smiles_column: Optional[str] = None,
        preferred_target_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Identify the likely SMILES and target columns for QSAR work."""
        df = _load_dataset(dataset_path)
        columns = list(df.columns)

        smiles_column = None
        if preferred_smiles_column and preferred_smiles_column in columns:
            smiles_column = preferred_smiles_column
        else:
            for candidate in ("smiles", "SMILES", "Drug", "smi", "structure"):
                if candidate in columns:
                    smiles_column = candidate
                    break

        target_columns: List[str] = []
        preferred_target_columns = preferred_target_columns or []
        for column in preferred_target_columns:
            if column in columns:
                target_columns.append(column)

        if not target_columns:
            excluded = {smiles_column} if smiles_column else set()
            excluded.update({"Drug_ID", "compound_id", "id", "ID"})
            numeric_candidates = [
                column
                for column in columns
                if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
            ]
            if task_type == "regression":
                target_columns = numeric_candidates[:1]
            else:
                target_columns = numeric_candidates[:1]

        return {
            "dataset_path": dataset_path,
            "task_type": task_type,
            "smiles_column": smiles_column,
            "target_columns": target_columns,
            "all_columns": columns,
            "ready": bool(smiles_column and target_columns),
        }

    def curate_qsar_dataset(
        self,
        dataset_path: str,
        task_type: str,
        smiles_column: str,
        target_columns: List[str],
        output_csv: Optional[str] = None,
        dataset_id: Optional[str] = None,
        report_path: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Produce a QSAR-ready curated dataset and structured curation result."""
        request = CurationRequest(
            dataset_path=dataset_path,
            task_type=task_type,
            dataset_id=dataset_id,
            preferred_smiles_column=smiles_column,
            preferred_target_columns=target_columns,
        )
        df = _load_dataset(dataset_path)
        rows_in = int(len(df))
        working = df.copy()

        warnings: List[str] = []
        blocking_issues: List[str] = []
        actions: List[str] = []

        if smiles_column not in working.columns:
            blocking_issues.append(f"SMILES column not found: {smiles_column}")
        missing_targets = [column for column in target_columns if column not in working.columns]
        if missing_targets:
            blocking_issues.append(f"Target columns not found: {missing_targets}")

        if blocking_issues:
            result = CurationResult(
                status="blocked",
                ready_for_qsar=False,
                dataset_id=dataset_id,
                source_dataset_path=dataset_path,
                curated_dataset_path=None,
                smiles_column_original=smiles_column,
                smiles_column_curated=None,
                target_columns_original=list(target_columns),
                target_columns_curated=[],
                task_type=task_type,
                rows_in=rows_in,
                rows_out=0,
                blocking_issues=blocking_issues,
            )
            if agent is not None:
                state = _get_curation_state(agent)
                state["last_request"] = request.as_dict()
                state["last_result"] = result.as_dict()
                state["history"].append(result.as_dict())
            return result.as_dict()

        if smiles_column != "smiles":
            working = working.rename(columns={smiles_column: "smiles"})
            actions.append(f"rename {smiles_column} -> smiles")
        else:
            actions.append("use existing smiles column")

        working = standardize_smiles_column(working, "smiles")
        invalid_before_drop = int(working["smiles"].isna().sum())
        if invalid_before_drop:
            actions.append("standardize smiles")
            actions.append("remove invalid smiles")
        else:
            actions.append("standardize smiles")
        working = working.dropna(subset=["smiles"]).copy()

        missing_target_removed = 0
        curated_targets: List[str] = []
        for column in target_columns:
            numeric = pd.to_numeric(working[column], errors="coerce")
            missing_target_removed += int(numeric.isna().sum())
            working[column] = numeric
            curated_targets.append(column)

        if task_type == "regression":
            if not curated_targets:
                blocking_issues.append("No regression target column was retained after curation.")
            working = working.dropna(subset=curated_targets).copy()
            actions.append("coerce target columns to numeric")
            actions.append("remove missing target rows")

        duplicate_rows_removed = int(working.duplicated(subset=["smiles"]).sum())
        if duplicate_rows_removed:
            working = working.drop_duplicates(subset=["smiles"], keep="first").copy()
            actions.append("remove duplicate standardized smiles")
        else:
            actions.append("check duplicate standardized smiles")

        rows_out = int(len(working))
        if rows_out == 0:
            blocking_issues.append("Dataset is empty after curation.")

        target_summaries: List[TargetSummary] = []
        for column in curated_targets:
            series = pd.to_numeric(working[column], errors="coerce").dropna()
            if series.empty:
                warnings.append(f"Target column `{column}` is empty after curation.")
                continue
            target_summaries.append(
                TargetSummary(
                    column=column,
                    mean=float(series.mean()),
                    std=float(series.std()) if len(series) > 1 else 0.0,
                    minimum=float(series.min()),
                    maximum=float(series.max()),
                    median=float(series.median()),
                )
            )

        if invalid_before_drop == 0:
            warnings.append("No invalid SMILES were removed during curation.")
        if duplicate_rows_removed == 0:
            warnings.append("No duplicate standardized SMILES were removed.")

        if output_csv:
            curated_path = Path(output_csv).expanduser()
        else:
            stem = dataset_id or Path(dataset_path).stem or "qsar_dataset"
            curated_path = (Path(".files") / "qsar_curation" / f"{stem}_curated.csv").resolve()
        curated_path.parent.mkdir(parents=True, exist_ok=True)
        working.to_csv(curated_path, index=False)

        result = CurationResult(
            status="ready" if not blocking_issues else "blocked",
            ready_for_qsar=not blocking_issues,
            dataset_id=dataset_id or Path(dataset_path).stem,
            source_dataset_path=dataset_path,
            curated_dataset_path=str(curated_path),
            smiles_column_original=smiles_column,
            smiles_column_curated="smiles",
            target_columns_original=list(target_columns),
            target_columns_curated=curated_targets,
            task_type=task_type,
            rows_in=rows_in,
            rows_out=rows_out,
            invalid_smiles_removed=invalid_before_drop,
            duplicate_rows_removed=duplicate_rows_removed,
            missing_target_removed=missing_target_removed,
            target_summaries=target_summaries,
            curation_actions=actions,
            warnings=warnings,
            blocking_issues=blocking_issues,
            report_path=report_path,
        )

        if report_path:
            report_payload = result.as_dict()
            report_file = Path(report_path).expanduser()
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(json.dumps(report_payload, indent=2) + "\n")
            result.report_path = str(report_file)

        result_payload = result.as_dict()

        if agent is not None:
            state = _get_curation_state(agent)
            state["last_request"] = request.as_dict()
            state["last_result"] = result_payload
            state["history"].append(result_payload)

        return result_payload

    def summarize_curated_dataset(
        self,
        curated_dataset_path: str,
        smiles_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Summarize a curated dataset that is ready for QSAR training."""
        df = _load_dataset(curated_dataset_path)
        target_columns = target_columns or [
            column for column in df.columns if column != smiles_column
        ]

        target_summaries = []
        for column in target_columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if series.empty:
                continue
            target_summaries.append(
                TargetSummary(
                    column=column,
                    mean=float(series.mean()),
                    std=float(series.std()) if len(series) > 1 else 0.0,
                    minimum=float(series.min()),
                    maximum=float(series.max()),
                    median=float(series.median()),
                ).as_dict()
            )

        return {
            "curated_dataset_path": curated_dataset_path,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "smiles_column": smiles_column,
            "target_columns": target_columns,
            "target_summaries": target_summaries,
        }

    def write_curation_report(
        self,
        curation_result: Dict[str, Any],
        report_path: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Persist a curation result as a JSON report artifact."""
        dataset_id = curation_result.get("dataset_id") or "qsar_dataset"
        destination = (
            Path(report_path).expanduser()
            if report_path
            else (Path(".files") / "qsar_curation" / f"{dataset_id}_curation_report.json").resolve()
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(curation_result, indent=2) + "\n")

        payload = {
            "report_path": str(destination),
            "download_file_ref": str(destination),
            "dataset_id": dataset_id,
        }
        if agent is not None:
            state = _get_curation_state(agent)
            if state.get("last_result"):
                state["last_result"]["report_path"] = str(destination)
        return payload
