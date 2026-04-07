#!/usr/bin/env python
# coding: utf-8
"""
Toolkit dedicated to QSAR dataset curation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit
from rdkit import Chem

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


def _bundle_files(bundle_path: Path, files: List[Path]) -> Path:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in files:
            if file_path.exists():
                zf.write(file_path, arcname=file_path.name)
    return bundle_path


def _count_stereo_markers_removed(original: pd.Series, standardized: pd.Series) -> int:
    """Approximate how many rows lost explicit stereochemistry markers during standardization."""
    count = 0
    for raw_smiles, std_smiles in zip(original.fillna(""), standardized.fillna(""), strict=False):
        if not isinstance(raw_smiles, str) or not isinstance(std_smiles, str):
            continue
        if ("@" in raw_smiles or "/" in raw_smiles or "\\" in raw_smiles) and (
            "@" not in std_smiles and "/" not in std_smiles and "\\" not in std_smiles
        ):
            count += 1
    return count


_UNIT_COLUMN_PATTERN = re.compile(r"(?:^|_)(unit|units|uom|measurement_unit|conc_unit)(?:$|_)", re.I)


def _normalize_unit_value(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.lower().replace(" ", "")


def _infer_target_unit_from_context(
    *,
    dataset_path: str,
    dataset_id: Optional[str],
    target_columns: List[str],
) -> Optional[str]:
    context = " ".join(
        [
            Path(dataset_path).stem.lower(),
            (dataset_id or "").lower(),
            " ".join(column.lower() for column in target_columns),
        ]
    )
    if any(token in context for token in ("lipophilicity", "logp", "logd")):
        return "unitless_log_scale"
    return None


def _detect_target_unit_quality(
    df: pd.DataFrame,
    *,
    dataset_path: str,
    dataset_id: Optional[str],
    target_columns: List[str],
) -> Dict[str, Any]:
    unit_columns = [column for column in df.columns if _UNIT_COLUMN_PATTERN.search(column)]
    unit_column_value_map: Dict[str, List[str]] = {}
    distinct_units: List[str] = []

    for column in unit_columns:
        values = [
            normalized
            for normalized in (_normalize_unit_value(v) for v in df[column].tolist())
            if normalized is not None
        ]
        unique_values = sorted(set(values))
        unit_column_value_map[column] = unique_values
        distinct_units.extend(unique_values)

    distinct_units = sorted(set(distinct_units))
    inferred_unit = _infer_target_unit_from_context(
        dataset_path=dataset_path,
        dataset_id=dataset_id,
        target_columns=target_columns,
    )

    if distinct_units:
        return {
            "unit_columns_detected": unit_columns,
            "unit_values_detected": distinct_units,
            "target_unit_detected": distinct_units[0] if len(distinct_units) == 1 else None,
            "target_unit_source": "dataset_unit_column",
            "target_unit_homogeneous": len(distinct_units) <= 1,
            "unit_conflicts_detected": max(len(distinct_units) - 1, 0),
            "unit_column_value_map": unit_column_value_map,
        }

    return {
        "unit_columns_detected": [],
        "unit_values_detected": [inferred_unit] if inferred_unit else [],
        "target_unit_detected": inferred_unit,
        "target_unit_source": "context_inference" if inferred_unit else None,
        "target_unit_homogeneous": True if inferred_unit else None,
        "unit_conflicts_detected": 0,
        "unit_column_value_map": {},
    }


def _detect_target_outliers(
    df: pd.DataFrame,
    *,
    target_columns: List[str],
) -> Dict[str, Any]:
    outlier_counts: Dict[str, int] = {}
    outlier_bounds: Dict[str, Dict[str, float]] = {}
    total_flagged = 0

    for column in target_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(series) < 4:
            outlier_counts[column] = 0
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            outlier_counts[column] = 0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = int(((series < lower) | (series > upper)).sum())
        outlier_counts[column] = flagged
        total_flagged += flagged
        outlier_bounds[column] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
        }

    return {
        "outlier_method": "iqr_1.5",
        "outliers_flagged_total": total_flagged,
        "outliers_flagged_by_target": outlier_counts,
        "outlier_bounds_by_target": outlier_bounds,
        "outlier_policy": "flag_only",
    }


_METAL_ATOMIC_NUMBERS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
}


def _structure_flags(smiles: str) -> Dict[str, bool]:
    """Classify raw structures for early structural curation decisions."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "valid": False,
            "is_inorganic": False,
            "is_organometallic": False,
            "is_mixture": False,
            "has_salt_or_counterion_pattern": False,
        }

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    organic_fragment_count = 0
    has_metal = False
    has_carbon = False

    for frag in fragments:
        frag_has_carbon = any(atom.GetAtomicNum() == 6 for atom in frag.GetAtoms())
        if frag_has_carbon:
            organic_fragment_count += 1
        has_carbon = has_carbon or frag_has_carbon
        if any(atom.GetAtomicNum() in _METAL_ATOMIC_NUMBERS for atom in frag.GetAtoms()):
            has_metal = True

    is_organometallic = has_metal and has_carbon
    is_inorganic = not has_carbon
    is_mixture = organic_fragment_count > 1
    has_salt_or_counterion_pattern = len(fragments) > 1 and organic_fragment_count <= 1

    return {
        "valid": True,
        "is_inorganic": is_inorganic,
        "is_organometallic": is_organometallic,
        "is_mixture": is_mixture,
        "has_salt_or_counterion_pattern": has_salt_or_counterion_pattern,
    }


def _resolve_regression_duplicates(
    df: pd.DataFrame,
    target_columns: List[str],
    conflict_threshold: float,
) -> Dict[str, Any]:
    """Resolve duplicate standardized structures for regression datasets."""
    grouped_rows: List[Dict[str, Any]] = []
    duplicate_groups_detected = 0
    duplicate_groups_aggregated = 0
    duplicate_conflicting_groups = 0
    duplicate_conflicting_rows_removed = 0

    for _, group in df.groupby("smiles", dropna=False, sort=False):
        if len(group) == 1:
            grouped_rows.append(group.iloc[0].to_dict())
            continue

        duplicate_groups_detected += 1
        is_conflicting = False
        for target in target_columns:
            series = pd.to_numeric(group[target], errors="coerce").dropna()
            if len(series) <= 1:
                continue
            spread = float(series.max() - series.min())
            if spread > conflict_threshold:
                is_conflicting = True
                break

        if is_conflicting:
            duplicate_conflicting_groups += 1
            duplicate_conflicting_rows_removed += int(len(group))
            continue

        aggregated = group.iloc[0].to_dict()
        for target in target_columns:
            aggregated[target] = float(pd.to_numeric(group[target], errors="coerce").mean())
        grouped_rows.append(aggregated)
        duplicate_groups_aggregated += 1

    resolved_df = pd.DataFrame(grouped_rows, columns=df.columns)
    return {
        "dataframe": resolved_df,
        "duplicate_groups_detected": duplicate_groups_detected,
        "duplicate_groups_aggregated": duplicate_groups_aggregated,
        "duplicate_conflicting_groups": duplicate_conflicting_groups,
        "duplicate_conflicting_rows_removed": duplicate_conflicting_rows_removed,
    }


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
        duplicate_conflict_threshold: float = 1.0,
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
            duplicate_conflict_threshold=duplicate_conflict_threshold,
        )
        df = _load_dataset(dataset_path)
        rows_in = int(len(df))
        working = df.copy()

        warnings: List[str] = []
        blocking_issues: List[str] = []
        actions: List[str] = []
        curation_policy = {
            "structure_pipeline": [
                "remove inorganic structures",
                "remove organometallic structures",
                "remove mixtures with multiple organic fragments",
                "standardize remaining structures",
            ],
            "fragment_handling": "retain largest fragment parent for salt/counterion cases",
            "smiles_standardization": "cleanup -> fragment parent -> uncharge -> canonical tautomer",
            "tautomer_policy": "canonical_tautomer",
            "duplicate_policy": "aggregate_mean_if_spread_within_threshold_else_drop_conflicts",
            "duplicate_conflict_threshold": duplicate_conflict_threshold,
            "target_policy": "coerce_numeric -> remove_non_numeric -> remove_infinite -> remove_missing -> flag_constant_targets",
            "unit_policy": "detect explicit unit columns -> block on unresolved heterogeneous units -> infer unit context only when no explicit unit column exists",
            "outlier_policy": "detect_iqr_1.5 -> flag_only",
        }

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

        raw_smiles = working["smiles"].copy()
        flags = raw_smiles.apply(
            lambda s: _structure_flags(s) if isinstance(s, str) else _structure_flags("")
        )
        flags_df = pd.DataFrame(list(flags))

        inorganic_rows_removed = int(flags_df["is_inorganic"].sum())
        organometallic_rows_removed = int(flags_df["is_organometallic"].sum())
        mixture_rows_removed = int(flags_df["is_mixture"].sum())
        salt_or_counterion_rows_processed = int(flags_df["has_salt_or_counterion_pattern"].sum())

        structural_remove_mask = (
            flags_df["is_inorganic"] | flags_df["is_organometallic"] | flags_df["is_mixture"]
        )
        if structural_remove_mask.any():
            working = working.loc[~structural_remove_mask].copy()
            actions.append("remove inorganic structures")
            actions.append("remove organometallic structures")
            actions.append("remove mixtures with multiple organic fragments")
        else:
            actions.append("check inorganic / organometallic / mixture structures")

        original_smiles = working["smiles"].copy()
        working = standardize_smiles_column(working, "smiles")
        invalid_before_drop = int(working["smiles"].isna().sum())
        stereochemistry_markers_removed = _count_stereo_markers_removed(
            original_smiles, working["smiles"]
        )
        if invalid_before_drop:
            actions.append("standardize smiles")
            actions.append("remove invalid smiles")
        else:
            actions.append("standardize smiles")
        working = working.dropna(subset=["smiles"]).copy()

        missing_target_removed = 0
        non_numeric_target_removed = 0
        infinite_target_removed = 0
        curated_targets: List[str] = []
        for column in target_columns:
            numeric = pd.to_numeric(working[column], errors="coerce")
            raw_missing = int(working[column].isna().sum())
            coerced_missing = int(numeric.isna().sum())
            non_numeric_target_removed += max(coerced_missing - raw_missing, 0)
            infinite_mask = numeric.apply(lambda v: isinstance(v, (int, float)) and not math.isfinite(v))
            infinite_target_removed += int(infinite_mask.sum())
            numeric = numeric.mask(infinite_mask, other=pd.NA)
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
        duplicate_groups_detected = 0
        duplicate_groups_aggregated = 0
        duplicate_conflicting_groups = 0
        duplicate_conflicting_rows_removed = 0
        if duplicate_rows_removed:
            if task_type == "regression" and curated_targets:
                duplicate_resolution = _resolve_regression_duplicates(
                    working,
                    curated_targets,
                    conflict_threshold=duplicate_conflict_threshold,
                )
                working = duplicate_resolution["dataframe"]
                duplicate_groups_detected = duplicate_resolution["duplicate_groups_detected"]
                duplicate_groups_aggregated = duplicate_resolution["duplicate_groups_aggregated"]
                duplicate_conflicting_groups = duplicate_resolution["duplicate_conflicting_groups"]
                duplicate_conflicting_rows_removed = duplicate_resolution[
                    "duplicate_conflicting_rows_removed"
                ]
                actions.append(
                    "resolve duplicate standardized smiles using conflict-threshold aggregation"
                )
            else:
                working = working.drop_duplicates(subset=["smiles"], keep="first").copy()
                actions.append("remove duplicate standardized smiles")
        else:
            actions.append("check duplicate standardized smiles")

        rows_out = int(len(working))
        if rows_out == 0:
            blocking_issues.append("Dataset is empty after curation.")

        target_summaries: List[TargetSummary] = []
        constant_target_columns: List[str] = []
        for column in curated_targets:
            series = pd.to_numeric(working[column], errors="coerce").dropna()
            if series.empty:
                warnings.append(f"Target column `{column}` is empty after curation.")
                continue
            if series.nunique(dropna=True) <= 1:
                constant_target_columns.append(column)
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

        if task_type == "regression" and len(constant_target_columns) == len(curated_targets):
            blocking_issues.append("All retained target columns are constant after curation.")

        unit_quality = _detect_target_unit_quality(
            df=df,
            dataset_path=dataset_path,
            dataset_id=dataset_id or Path(dataset_path).stem,
            target_columns=curated_targets or target_columns,
        )
        outlier_quality = _detect_target_outliers(
            working,
            target_columns=curated_targets,
        )

        target_data_quality = {
            "numeric_targets_required": task_type == "regression",
            "non_numeric_target_removed": non_numeric_target_removed,
            "infinite_target_removed": infinite_target_removed,
            "missing_target_removed": missing_target_removed,
            "constant_target_columns": list(constant_target_columns),
            "target_ready_for_qsar": not bool(
                blocking_issues
                or (task_type == "regression" and len(constant_target_columns) == len(curated_targets))
            ),
        }
        target_data_quality.update(unit_quality)
        target_data_quality.update(outlier_quality)

        if unit_quality.get("unit_conflicts_detected", 0):
            blocking_issues.append(
                "Conflicting target units detected in the dataset; unit harmonization is required before QSAR training."
            )
        elif unit_quality.get("target_unit_detected"):
            actions.append(
                f"detect target unit context: {unit_quality['target_unit_detected']}"
            )

        if invalid_before_drop == 0:
            warnings.append("No invalid SMILES were removed during curation.")
        if inorganic_rows_removed:
            warnings.append(f"{inorganic_rows_removed} inorganic structures were removed.")
        if organometallic_rows_removed:
            warnings.append(f"{organometallic_rows_removed} organometallic structures were removed.")
        if mixture_rows_removed:
            warnings.append(f"{mixture_rows_removed} multi-component organic mixtures were removed.")
        if salt_or_counterion_rows_processed:
            warnings.append(
                f"{salt_or_counterion_rows_processed} rows were processed as salt/counterion cases via fragment-parent standardization."
            )
        if duplicate_rows_removed == 0:
            warnings.append("No duplicate standardized SMILES were removed.")
        elif duplicate_conflicting_groups:
            warnings.append(
                f"{duplicate_conflicting_groups} duplicate structure groups exceeded the conflict threshold and were removed."
            )
        if duplicate_groups_aggregated:
            warnings.append(
                f"{duplicate_groups_aggregated} duplicate structure groups were aggregated by mean target."
            )
        if stereochemistry_markers_removed:
            warnings.append(
                f"Standardization removed explicit stereochemistry markers for {stereochemistry_markers_removed} rows."
            )
        if non_numeric_target_removed:
            warnings.append(
                f"{non_numeric_target_removed} target values were non-numeric and removed during coercion."
            )
        if infinite_target_removed:
            warnings.append(
                f"{infinite_target_removed} target values were infinite and removed during curation."
            )
        if constant_target_columns:
            warnings.append(
                f"Constant target columns detected after curation: {', '.join(constant_target_columns)}."
            )
        if unit_quality.get("unit_conflicts_detected", 0):
            warnings.append(
                "Multiple target units were detected; the dataset was blocked pending unit harmonization."
            )
        elif unit_quality.get("target_unit_detected"):
            unit_source = unit_quality.get("target_unit_source") or "unknown_source"
            warnings.append(
                f"Target unit context detected as `{unit_quality['target_unit_detected']}` via {unit_source}."
            )
        else:
            warnings.append("No explicit target unit column was detected.")
        if outlier_quality.get("outliers_flagged_total", 0):
            warnings.append(
                f"{outlier_quality['outliers_flagged_total']} potential target outliers were flagged using the IQR rule."
            )

        target_data_quality["target_ready_for_qsar"] = not bool(
            blocking_issues
            or (task_type == "regression" and len(constant_target_columns) == len(curated_targets))
        )

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
            retained_columns=list(working.columns),
            invalid_smiles_removed=invalid_before_drop,
            inorganic_rows_removed=inorganic_rows_removed,
            organometallic_rows_removed=organometallic_rows_removed,
            mixture_rows_removed=mixture_rows_removed,
            salt_or_counterion_rows_processed=salt_or_counterion_rows_processed,
            duplicate_rows_removed=duplicate_rows_removed,
            duplicate_groups_detected=duplicate_groups_detected,
            duplicate_groups_aggregated=duplicate_groups_aggregated,
            duplicate_conflicting_groups=duplicate_conflicting_groups,
            duplicate_conflicting_rows_removed=duplicate_conflicting_rows_removed,
            missing_target_removed=missing_target_removed,
            non_numeric_target_removed=non_numeric_target_removed,
            infinite_target_removed=infinite_target_removed,
            constant_target_columns=constant_target_columns,
            stereochemistry_markers_removed=stereochemistry_markers_removed,
            target_summaries=target_summaries,
            curation_actions=actions,
            curation_policy=curation_policy,
            target_data_quality=target_data_quality,
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
        curated_dataset_path = curation_result.get("curated_dataset_path")
        if curated_dataset_path:
            curated_path = Path(curated_dataset_path).expanduser()
            bundle_path = (
                Path(".files")
                / "qsar_curation"
                / f"{dataset_id}_curation_bundle.zip"
            ).resolve()
            bundle = _bundle_files(bundle_path, [curated_path, destination])
            payload["bundle_file_ref"] = str(bundle)
        if agent is not None:
            state = _get_curation_state(agent)
            if state.get("last_result"):
                state["last_result"]["report_path"] = str(destination)
                if payload.get("bundle_file_ref"):
                    state["last_result"]["bundle_file_ref"] = payload["bundle_file_ref"]
        return payload
