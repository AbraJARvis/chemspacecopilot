#!/usr/bin/env python
# coding: utf-8
"""
Benchmark orchestration for multi-backend QSAR campaigns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from .chemprop_toolkit import ChempropToolkit
from .qsar_training_policy import describe_compute_environment, resolve_training_profile, safe_slug
from .tabicl_toolkit import TabICLToolkit
from ..features.molecular_feature_toolkit import MolecularFeatureToolkit

logger = logging.getLogger(__name__)


def _coerce_list(value: Optional[List[str] | str]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("Expected a list or JSON-encoded list.")
        return [str(item) for item in parsed]
    return [str(item) for item in value]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _benchmark_dataset_token(train_csv: str) -> str:
    stem = Path(train_csv).stem.lower()
    suffixes = (
        "_tabular_qsar",
        "_tabular",
        "_normalized",
        "_prepared",
        "_curated",
        "_cleaned",
        "_dataset",
        "_training",
    )
    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for trailing in ("_train", "_test"):
        if stem.endswith(trailing):
            stem = stem[: -len(trailing)]
            break
    return safe_slug(stem) or "dataset"


def _benchmark_target_token(target_columns: List[str]) -> str:
    if not target_columns:
        return "target"
    return safe_slug(str(target_columns[0])) or "target"


class BenchmarkToolkit(Toolkit):
    """Toolkit orchestrating multi-backend QSAR benchmark campaigns."""

    BENCHMARK_MODE_TO_PROTOCOL = {
        "benchmark_fast_local": "fast_local",
        "benchmark_standard_qsar": "standard_qsar",
        "benchmark_robust_qsar": "robust_qsar",
        "benchmark_challenging_qsar": "challenging_qsar",
    }

    TABICL_VARIANT_SPECS = {
        "tabicl_morgan_only": {
            "representation_name": "morgan_only",
            "use_morgan": True,
            "use_rdkit": False,
            "descriptor_set": None,
        },
        "tabicl_rdkit_basic_only": {
            "representation_name": "rdkit_basic_only",
            "use_morgan": False,
            "use_rdkit": True,
            "descriptor_set": "basic",
        },
        "tabicl_morgan_rdkit_basic": {
            "representation_name": "morgan_rdkit_basic",
            "use_morgan": True,
            "use_rdkit": True,
            "descriptor_set": "basic",
        },
        "tabicl_morgan_rdkit_all": {
            "representation_name": "morgan_rdkit_all",
            "use_morgan": True,
            "use_rdkit": True,
            "descriptor_set": "all",
        },
    }

    def __init__(
        self,
        *,
        chemprop_toolkit: Optional[ChempropToolkit] = None,
        tabicl_toolkit: Optional[TabICLToolkit] = None,
        molecular_feature_toolkit: Optional[MolecularFeatureToolkit] = None,
    ):
        super().__init__("benchmark_prediction")
        self.chemprop_toolkit = chemprop_toolkit or ChempropToolkit()
        self.tabicl_toolkit = tabicl_toolkit or TabICLToolkit()
        self.molecular_feature_toolkit = molecular_feature_toolkit or MolecularFeatureToolkit()
        self.register(self.benchmark_qsar_models)

    def _resolve_benchmark_protocol(self, benchmark_mode: str) -> str:
        normalized = benchmark_mode.strip().lower()
        protocol = self.BENCHMARK_MODE_TO_PROTOCOL.get(normalized)
        if protocol is None:
            raise ValueError(
                "Unsupported benchmark_mode. Expected one of "
                f"{sorted(self.BENCHMARK_MODE_TO_PROTOCOL)}."
            )
        return protocol

    def _resolve_compute_profile(self) -> Dict[str, Any]:
        compute_env = describe_compute_environment()
        resolved = resolve_training_profile(compute_env)
        return {
            "compute_environment": compute_env,
            "training_profile": resolved["profile"],
            "profile_reason": resolved["reason"],
        }

    def _resolve_backends(
        self,
        *,
        task_type: str,
        target_columns: List[str],
        requested_backends: Optional[List[str]],
    ) -> List[str]:
        available: List[str] = []
        if self.chemprop_toolkit.backend.is_available():
            available.append("chemprop")
        if (
            self.tabicl_toolkit.backend.is_available()
            and task_type == "regression"
            and len(target_columns) == 1
        ):
            available.append("tabicl")

        if requested_backends:
            filtered = [name for name in requested_backends if name in available]
            if not filtered:
                raise ValueError(
                    f"No requested backends are available or compatible: {requested_backends}"
                )
            return filtered
        return available

    def _expand_tabicl_candidates(
        self,
        *,
        include_candidate_variants: bool,
        requested_variants: Optional[List[str]],
        training_profile: str,
    ) -> List[Dict[str, Any]]:
        if requested_variants:
            unknown = [name for name in requested_variants if name not in self.TABICL_VARIANT_SPECS]
            if unknown:
                raise ValueError(f"Unknown TabICL benchmark variants: {unknown}")
            variant_ids = requested_variants
        elif include_candidate_variants:
            variant_ids = [
                "tabicl_morgan_only",
                "tabicl_rdkit_basic_only",
                "tabicl_morgan_rdkit_basic",
            ]
            if training_profile == "heavy_validation":
                variant_ids.append("tabicl_morgan_rdkit_all")
        else:
            variant_ids = [
                "tabicl_morgan_rdkit_all"
                if training_profile == "heavy_validation"
                else "tabicl_morgan_rdkit_basic"
            ]

        candidates: List[Dict[str, Any]] = []
        for candidate_id in variant_ids:
            spec = dict(self.TABICL_VARIANT_SPECS[candidate_id])
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "backend_name": "tabicl",
                    **spec,
                }
            )
        return candidates

    def _expand_candidates(
        self,
        *,
        task_type: str,
        target_columns: List[str],
        requested_backends: Optional[List[str]],
        include_candidate_variants: bool,
        requested_tabicl_variants: Optional[List[str]],
        training_profile: str,
    ) -> List[Dict[str, Any]]:
        backends = self._resolve_backends(
            task_type=task_type,
            target_columns=target_columns,
            requested_backends=requested_backends,
        )
        candidates: List[Dict[str, Any]] = []
        if "chemprop" in backends:
            candidates.append(
                {
                    "candidate_id": "chemprop_default",
                    "backend_name": "chemprop",
                    "representation_name": "molecular_graph",
                }
            )
        if "tabicl" in backends:
            candidates.extend(
                self._expand_tabicl_candidates(
                    include_candidate_variants=include_candidate_variants,
                    requested_variants=requested_tabicl_variants,
                    training_profile=training_profile,
                )
            )
        return candidates

    def _prepare_tabicl_candidate_dataset(
        self,
        *,
        candidate: Dict[str, Any],
        train_csv: str,
        smiles_column: str,
        target_columns: List[str],
        candidate_dir: Path,
    ) -> Dict[str, Any]:
        normalized_smiles_column = "smiles"
        base_columns = [normalized_smiles_column] + list(target_columns)
        feature_csvs: List[str] = []
        features_dir = candidate_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        base_csv_for_build = train_csv

        if smiles_column != normalized_smiles_column:
            normalized_base_csv = features_dir / "base_normalized.csv"
            source_df = pd.read_csv(Path(train_csv).expanduser())
            if smiles_column not in source_df.columns:
                raise ValueError(
                    f"SMILES column '{smiles_column}' not found in benchmark input columns: {list(source_df.columns)}"
                )
            normalized_df = source_df.copy()
            normalized_df = normalized_df.rename(columns={smiles_column: normalized_smiles_column})
            normalized_df.to_csv(normalized_base_csv, index=False)
            base_csv_for_build = str(normalized_base_csv)

        if candidate.get("use_morgan"):
            keep_columns = base_columns if not candidate.get("use_rdkit") else [smiles_column]
            morgan_result = self.molecular_feature_toolkit.smiles_to_morgan_fingerprints(
                input_csv=train_csv,
                smiles_column=smiles_column,
                output_csv=str(features_dir / "morgan_fp.csv"),
                input_columns_to_keep=keep_columns,
            )
            if not candidate.get("use_rdkit"):
                return {
                    "train_csv": morgan_result["output_csv"],
                    "representation_name": candidate["representation_name"],
                }
            feature_csvs.append(morgan_result["output_csv"])

        if candidate.get("use_rdkit"):
            keep_columns = base_columns if not candidate.get("use_morgan") else [smiles_column]
            rdkit_result = self.molecular_feature_toolkit.smiles_to_rdkit_descriptors(
                input_csv=train_csv,
                smiles_column=smiles_column,
                output_csv=str(features_dir / f"rdkit_{candidate['descriptor_set']}.csv"),
                descriptor_set=str(candidate["descriptor_set"]),
                input_columns_to_keep=keep_columns,
            )
            if not candidate.get("use_morgan"):
                return {
                    "train_csv": rdkit_result["output_csv"],
                    "representation_name": candidate["representation_name"],
                }
            feature_csvs.append(rdkit_result["output_csv"])

        tabular_output = candidate_dir / f"{Path(train_csv).stem}_{candidate['representation_name']}_tabular.csv"
        build_result = self.molecular_feature_toolkit.build_tabular_qsar_dataset(
            base_csv=base_csv_for_build,
            output_csv=str(tabular_output),
            feature_csvs=feature_csvs,
            join_on=[normalized_smiles_column],
            base_columns_to_keep=base_columns,
            drop_duplicate_feature_columns=True,
        )
        return {
            "train_csv": build_result["output_csv"],
            "representation_name": candidate["representation_name"],
        }

    def _train_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        train_csv: str,
        task_type: str,
        smiles_column: str,
        target_columns: List[str],
        benchmark_protocol: str,
        candidate_dir: Path,
        allow_heavy_compute: bool,
        training_profile: Optional[str],
        agent: Agent,
    ) -> Dict[str, Any]:
        requested_extra_args: Dict[str, Any] = {
            "validation_protocol": benchmark_protocol,
            "allow_heavy_compute": allow_heavy_compute,
        }
        if training_profile:
            requested_extra_args["training_profile"] = training_profile

        if candidate["backend_name"] == "chemprop":
            result = self.chemprop_toolkit.train_model(
                train_csv=train_csv,
                task_type=task_type,
                output_dir=str(candidate_dir),
                smiles_columns=[smiles_column],
                target_columns=list(target_columns),
                extra_args=requested_extra_args,
                agent=agent,
            )
            result["representation_name"] = candidate["representation_name"]
            result["candidate_train_csv"] = train_csv
            return result

        prepared = self._prepare_tabicl_candidate_dataset(
            candidate=candidate,
            train_csv=train_csv,
            smiles_column=smiles_column,
            target_columns=target_columns,
            candidate_dir=candidate_dir,
        )
        result = self.tabicl_toolkit.train_tabicl_model(
            train_csv=prepared["train_csv"],
            task_type=task_type,
            output_dir=str(candidate_dir),
            target_columns=list(target_columns),
            validation_protocol=benchmark_protocol,
            extra_args=requested_extra_args,
            agent=agent,
        )
        result["representation_name"] = prepared["representation_name"]
        result["candidate_train_csv"] = prepared["train_csv"]
        return result

    def _display_name_for_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        target_columns: List[str],
    ) -> str:
        target_label = ", ".join(target_columns)
        return f"{candidate['backend_name'].upper()} {candidate['representation_name']} benchmark model for {target_label}"

    def _register_and_persist_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        result: Dict[str, Any],
        benchmark_mode: str,
        benchmark_protocol: str,
        campaign_root: Path,
        task_type: str,
        smiles_column: str,
        target_columns: List[str],
        training_profile: str,
        agent: Agent,
    ) -> Dict[str, Any]:
        temporary_model_id = f"{candidate['candidate_id']}_session"
        model_path = result.get("best_model_path") or result.get("model_path")
        if not model_path:
            raise ValueError(f"Candidate {candidate['candidate_id']} did not produce a model artifact path.")

        metrics_payload = (
            ((result.get("validation_assessment") or {}).get("aggregated_split_metrics"))
            or result.get("metrics")
            or {}
        )
        self.chemprop_toolkit.register_model(
            model_id=temporary_model_id,
            model_path=str(model_path),
            backend_name=candidate["backend_name"],
            task_type=task_type,
            smiles_columns=[smiles_column],
            target_columns=list(target_columns),
            description=f"Benchmark candidate {candidate['candidate_id']} trained under {benchmark_mode}.",
            tags={
                "benchmark_mode": benchmark_mode,
                "candidate_id": candidate["candidate_id"],
                "representation_name": candidate["representation_name"],
            },
            version="1.0.0",
            status=((result.get("validation_assessment") or {}).get("governance") or {}).get(
                "recommended_status",
                "experimental",
            ),
            owner="qsar_training_agent",
            source=result.get("candidate_train_csv") or result.get("train_csv"),
            known_metrics=metrics_payload if isinstance(metrics_payload, dict) else {},
            training_data_summary={
                "benchmark_mode": benchmark_mode,
                "candidate_id": candidate["candidate_id"],
                "representation_name": candidate["representation_name"],
                "benchmark_dataset_name": _benchmark_dataset_token(train_csv),
                "benchmark_target_name": _benchmark_target_token(target_columns),
                "validation_protocol": benchmark_protocol,
                "training_profile": training_profile,
                "campaign_root": str(campaign_root),
                "trained_at": result.get("trained_at"),
            },
            inference_profile={
                "feature_columns": list(result.get("feature_columns") or []),
                "representation_name": candidate["representation_name"],
            },
            selection_hints={
                "benchmark_mode": benchmark_mode,
                "candidate_id": candidate["candidate_id"],
                "representation_name": candidate["representation_name"],
            },
            applicability_domain=result.get("applicability_domain") or {},
            agent=agent,
        )
        persisted = self.chemprop_toolkit.persist_registered_model(
            model_id=temporary_model_id,
            display_name=self._display_name_for_candidate(
                candidate=candidate,
                target_columns=target_columns,
            ),
            version="1.0.0",
            agent=agent,
        )
        return persisted

    def _candidate_summary_row(
        self,
        *,
        candidate: Dict[str, Any],
        result: Dict[str, Any],
        persisted: Dict[str, Any],
        benchmark_protocol: str,
    ) -> Dict[str, Any]:
        validation_assessment = result.get("validation_assessment") or {}
        aggregated = validation_assessment.get("aggregated_split_metrics") or {}
        governance = validation_assessment.get("governance") or {}
        hardest_split = validation_assessment.get("hardest_split")
        hardest_family = aggregated.get(hardest_split) if hardest_split else None
        random_family = aggregated.get("random") or {}
        scaffold_family = aggregated.get("scaffold") or {}
        kmeans_family = aggregated.get("cluster_kmeans") or {}
        hardest_r2 = _safe_float((hardest_family or {}).get("r2_mean") or (hardest_family or {}).get("r2"))
        delta_vs_random = validation_assessment.get("delta_vs_random") or {}
        hardest_delta_r2 = None
        if hardest_split:
            hardest_delta_r2 = _safe_float((delta_vs_random.get(hardest_split) or {}).get("r2"))

        row = {
            "candidate_id": candidate["candidate_id"],
            "model_id": persisted.get("model_id"),
            "backend": candidate["backend_name"],
            "representation": candidate["representation_name"],
            "protocol": benchmark_protocol,
            "random_r2": _safe_float(random_family.get("r2_mean") or random_family.get("r2")),
            "scaffold_r2": _safe_float(scaffold_family.get("r2_mean") or scaffold_family.get("r2")),
            "cluster_kmeans_r2": _safe_float(kmeans_family.get("r2_mean") or kmeans_family.get("r2")),
            "hardest_split": hardest_split,
            "hardest_split_r2": hardest_r2,
            "delta_r2": hardest_delta_r2,
            "random_family_r2_mean": _safe_float(random_family.get("r2_mean")),
            "random_family_r2_std": _safe_float(random_family.get("r2_std")),
            "train_time_s": _safe_float((result.get("training_durations") or {}).get("total_duration_seconds")),
            "status": persisted.get("status"),
            "internal_model_root": persisted.get("model_root"),
            "best_model_path": persisted.get("model_path"),
        }
        return row

    def _rank_summary_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def sort_key(row: Dict[str, Any]) -> tuple[float, float, float, float]:
            hardest_r2 = row.get("hardest_split_r2")
            delta_r2 = row.get("delta_r2")
            random_r2 = row.get("random_r2")
            train_time = row.get("train_time_s")
            return (
                -(hardest_r2 if hardest_r2 is not None else float("-inf")),
                -(delta_r2 if delta_r2 is not None else float("-inf")),
                -(random_r2 if random_r2 is not None else float("-inf")),
                train_time if train_time is not None else float("inf"),
            )

        return sorted(rows, key=sort_key)

    def _resolve_recommendations(
        self,
        rows: List[Dict[str, Any]],
        benchmark_protocol: str,
    ) -> Dict[str, Optional[str]]:
        ranked = self._rank_summary_rows(rows)
        best_overall = ranked[0]["candidate_id"] if ranked else None
        best_hardest = max(
            rows,
            key=lambda row: row.get("hardest_split_r2") if row.get("hardest_split_r2") is not None else float("-inf"),
            default=None,
        )
        best_fast = min(
            rows,
            key=lambda row: row.get("train_time_s") if row.get("train_time_s") is not None else float("inf"),
            default=None,
        )
        best_stability = None
        if benchmark_protocol == "robust_qsar":
            stability_rows = [row for row in rows if row.get("random_family_r2_std") is not None]
            if stability_rows:
                best_stability = min(
                    stability_rows,
                    key=lambda row: row.get("random_family_r2_std", float("inf")),
                )

        return {
            "best_overall_candidate": best_overall,
            "best_hardest_split_candidate": best_hardest["candidate_id"] if best_hardest else None,
            "best_stability_candidate": best_stability["candidate_id"] if best_stability else None,
            "best_fast_candidate": best_fast["candidate_id"] if best_fast else None,
            "recommended_candidate_for_followup": best_overall,
        }

    def _render_benchmark_report(
        self,
        *,
        benchmark_mode: str,
        benchmark_protocol: str,
        compute_environment: Dict[str, Any],
        candidate_rows: List[Dict[str, Any]],
        candidate_results: List[Dict[str, Any]],
        recommendations: Dict[str, Optional[str]],
    ) -> str:
        lines: List[str] = []
        lines.append(f"# Benchmark QSAR Report — {benchmark_mode}")
        lines.append("")
        lines.append("## Executive summary")
        lines.append("")
        lines.append(f"- Benchmark mode: `{benchmark_mode}`")
        lines.append(f"- Base protocol: `{benchmark_protocol}`")
        lines.append(f"- Candidates compared: `{len(candidate_rows)}`")
        for key, value in recommendations.items():
            if value:
                lines.append(f"- {key.replace('_', ' ')}: `{value}`")
        lines.append("")
        lines.append("## Benchmark context")
        lines.append("")
        lines.append(f"- Execution environment: `{compute_environment.get('execution_env')}`")
        lines.append(f"- CPU count: `{compute_environment.get('cpu_count')}`")
        lines.append(f"- GPU available: `{compute_environment.get('gpu_available')}`")
        lines.append(f"- Suggested profile: `{compute_environment.get('suggested_profile')}`")
        lines.append("")
        lines.append("## Candidate inventory")
        lines.append("")
        for row in candidate_rows:
            lines.append(
                f"- `{row['candidate_id']}` — backend `{row['backend']}`, representation `{row['representation']}`, model `{row['model_id']}`"
            )
        lines.append("")
        lines.append("## Leaderboard")
        lines.append("")
        lines.append("| candidate_id | backend | representation | hardest_split | hardest_split_r2 | random_r2 | scaffold_r2 | cluster_kmeans_r2 | delta_r2 | train_time_s | status |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        for row in self._rank_summary_rows(candidate_rows):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("candidate_id") or ""),
                        str(row.get("backend") or ""),
                        str(row.get("representation") or ""),
                        str(row.get("hardest_split") or ""),
                        f"{row['hardest_split_r2']:.3f}" if row.get("hardest_split_r2") is not None else "",
                        f"{row['random_r2']:.3f}" if row.get("random_r2") is not None else "",
                        f"{row['scaffold_r2']:.3f}" if row.get("scaffold_r2") is not None else "",
                        f"{row['cluster_kmeans_r2']:.3f}" if row.get("cluster_kmeans_r2") is not None else "",
                        f"{row['delta_r2']:.3f}" if row.get("delta_r2") is not None else "",
                        f"{row['train_time_s']:.3f}" if row.get("train_time_s") is not None else "",
                        str(row.get("status") or ""),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append("## Split-by-split comparative analysis")
        lines.append("")
        for item in candidate_results:
            lines.append(f"### {item['candidate_id']}")
            lines.append("")
            lines.append(f"- backend: `{item['backend_name']}`")
            lines.append(f"- representation: `{item['representation_name']}`")
            validation_assessment = item.get("validation_assessment") or {}
            lines.append(f"- hardest split: `{validation_assessment.get('hardest_split')}`")
            for split_result in item.get("split_results") or []:
                metrics = ((split_result.get("metrics") or {}).get("test") or {})
                lines.append(
                    f"- `{split_result.get('strategy_label')}`: "
                    f"R²={metrics.get('r2')}, RMSE={metrics.get('rmse')}, MAE={metrics.get('mae')}, MSE={metrics.get('mse')}"
                )
            lines.append("")
        lines.append("## Governance and recommendation")
        lines.append("")
        for key, value in recommendations.items():
            if value:
                lines.append(f"- {key.replace('_', ' ')}: `{value}`")
        lines.append("")
        lines.append("## Artifact and model-path appendix")
        lines.append("")
        for row in candidate_rows:
            lines.append(f"### {row['candidate_id']}")
            lines.append("")
            lines.append(f"- model_id: `{row['model_id']}`")
            lines.append(f"- internal_model_root: `{row['internal_model_root']}`")
            lines.append(f"- best_model_path: `{row['best_model_path']}`")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def benchmark_qsar_models(
        self,
        train_csv: str,
        task_type: str,
        target_columns: List[str] | str,
        smiles_column: str = "smiles",
        benchmark_mode: str = "benchmark_standard_qsar",
        backends: Optional[List[str] | str] = None,
        include_candidate_variants: bool = True,
        tabicl_candidate_variants: Optional[List[str] | str] = None,
        output_dir: str = ".files/benchmark_output",
        allow_heavy_compute: bool = False,
        training_profile: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Run a multi-backend benchmark campaign for a QSAR-ready dataset."""
        if agent is None:
            raise ValueError("Agent is required when running a benchmark campaign")

        target_columns = _coerce_list(target_columns) or []
        requested_backends = _coerce_list(backends)
        requested_tabicl_variants = _coerce_list(tabicl_candidate_variants)

        benchmark_protocol = self._resolve_benchmark_protocol(benchmark_mode)
        compute_payload = self._resolve_compute_profile()
        effective_training_profile = training_profile or compute_payload["training_profile"]

        candidates = self._expand_candidates(
            task_type=task_type,
            target_columns=target_columns,
            requested_backends=requested_backends,
            include_candidate_variants=include_candidate_variants,
            requested_tabicl_variants=requested_tabicl_variants,
            training_profile=effective_training_profile,
        )
        if not candidates:
            raise ValueError("No compatible benchmark candidates are available.")

        campaign_root = Path(output_dir).expanduser().resolve()
        campaign_root.mkdir(parents=True, exist_ok=True)

        candidate_rows: List[Dict[str, Any]] = []
        candidate_results: List[Dict[str, Any]] = []
        persisted_model_mapping: List[Dict[str, Any]] = []

        total_candidates = len(candidates)
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_dir = campaign_root / candidate["candidate_id"]
            candidate_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Benchmark progress: candidate %d/%d - %s (backend=%s, representation=%s) starting.",
                candidate_index,
                total_candidates,
                candidate["candidate_id"],
                candidate["backend_name"],
                candidate["representation_name"],
            )
            try:
                result = self._train_candidate(
                    candidate=candidate,
                    train_csv=train_csv,
                    task_type=task_type,
                    smiles_column=smiles_column,
                    target_columns=target_columns,
                    benchmark_protocol=benchmark_protocol,
                    candidate_dir=candidate_dir,
                    allow_heavy_compute=allow_heavy_compute,
                    training_profile=training_profile,
                    agent=agent,
                )
            except Exception:
                logger.exception(
                    "Benchmark progress: candidate %d/%d - %s failed.",
                    candidate_index,
                    total_candidates,
                    candidate["candidate_id"],
                )
                raise
            persisted = self._register_and_persist_candidate(
                candidate=candidate,
                result=result,
                benchmark_mode=benchmark_mode,
                benchmark_protocol=benchmark_protocol,
                campaign_root=campaign_root,
                task_type=task_type,
                smiles_column=smiles_column,
                target_columns=target_columns,
                training_profile=effective_training_profile,
                agent=agent,
            )
            result["candidate_id"] = candidate["candidate_id"]
            result["backend_name"] = candidate["backend_name"]
            result["representation_name"] = candidate["representation_name"]
            candidate_results.append(result)
            candidate_row = self._candidate_summary_row(
                candidate=candidate,
                result=result,
                persisted=persisted,
                benchmark_protocol=benchmark_protocol,
            )
            candidate_rows.append(candidate_row)
            logger.info(
                "Benchmark progress: candidate %d/%d - %s completed (status=%s, hardest_split_r2=%s).",
                candidate_index,
                total_candidates,
                candidate["candidate_id"],
                persisted.get("status"),
                candidate_row.get("hardest_split_r2"),
            )
            persisted_model_mapping.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "model_id": persisted.get("model_id"),
                    "backend_name": candidate["backend_name"],
                    "representation_name": candidate["representation_name"],
                    "internal_model_root": persisted.get("model_root"),
                    "catalog_status": persisted.get("status"),
                    "best_model_path": persisted.get("model_path"),
                }
            )

        ranked_rows = self._rank_summary_rows(candidate_rows)
        recommendations = self._resolve_recommendations(ranked_rows, benchmark_protocol)

        leaderboard_path = campaign_root / "leaderboard.csv"
        pd.DataFrame(ranked_rows).to_csv(leaderboard_path, index=False)

        report_path = campaign_root / "benchmark_report.md"
        report_path.write_text(
            self._render_benchmark_report(
                benchmark_mode=benchmark_mode,
                benchmark_protocol=benchmark_protocol,
                compute_environment=compute_payload["compute_environment"],
                candidate_rows=ranked_rows,
                candidate_results=candidate_results,
                recommendations=recommendations,
            )
        )

        benchmark_summary = {
            "benchmark_mode": benchmark_mode,
            "benchmark_protocol": benchmark_protocol,
            "train_csv": train_csv,
            "task_type": task_type,
            "target_columns": target_columns,
            "smiles_column": smiles_column,
            "compute_environment": compute_payload["compute_environment"],
            "training_profile": effective_training_profile,
            "candidate_inventory": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "backend_name": candidate["backend_name"],
                    "representation_name": candidate["representation_name"],
                    "validation_protocol": benchmark_protocol,
                    "training_profile": effective_training_profile,
                }
                for candidate in candidates
            ],
            "candidate_results": candidate_results,
            "persisted_model_mapping": persisted_model_mapping,
            "leaderboard_path": str(leaderboard_path),
            "report_path": str(report_path),
            **recommendations,
        }
        summary_path = campaign_root / "benchmark_summary.json"
        summary_path.write_text(json.dumps(benchmark_summary, indent=2) + "\n")

        return {
            "benchmark_mode": benchmark_mode,
            "benchmark_protocol": benchmark_protocol,
            "output_dir": str(campaign_root),
            "candidate_results": candidate_results,
            "persisted_model_mapping": persisted_model_mapping,
            "leaderboard_path": str(leaderboard_path),
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            **recommendations,
        }
