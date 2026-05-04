#!/usr/bin/env python
# coding: utf-8
"""
Shared helpers for SALI-based activity-cliff annotation and feedback loops.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import KFold, train_test_split

from cs_copilot.storage import S3


DEFAULT_ACTIVITY_CLIFF_INDEX = "sali"
DEFAULT_ACTIVITY_CLIFF_SIMILARITY_THRESHOLD = 0.70
DEFAULT_ACTIVITY_CLIFF_K_NEIGHBORS = 10
DEFAULT_ACTIVITY_CLIFF_OOF_FOLDS = 5
DEFAULT_SALI_FLAG_THRESHOLD = 0.35
MAX_ACTIVITY_CLIFF_LOOPS = 3
MIN_TRAINING_ROWS = 10
SALI_EPSILON = 1e-6


@dataclass
class ActivityCliffConfig:
    mode: str = "annotate_only"
    loops: int = 0
    index_name: str = DEFAULT_ACTIVITY_CLIFF_INDEX
    similarity_threshold: float = DEFAULT_ACTIVITY_CLIFF_SIMILARITY_THRESHOLD
    k_neighbors: int = DEFAULT_ACTIVITY_CLIFF_K_NEIGHBORS
    oof_folds: int = DEFAULT_ACTIVITY_CLIFF_OOF_FOLDS
    sali_flag_threshold: float = DEFAULT_SALI_FLAG_THRESHOLD

    @property
    def enabled(self) -> bool:
        return True

    @property
    def feedback_enabled(self) -> bool:
        return self.mode == "with_feedback_loops"


def _fingerprint_generator(radius: int = 2, nbits: int = 2048):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)


def _safe_slug(value: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "value"


def _coerce_loop_variant_id(loop_index: int, removed_tiers: List[str]) -> str:
    if loop_index <= 0:
        return "baseline_loop_0"
    joined = "_".join(removed_tiers) if removed_tiers else "none"
    return f"filtered_loop_{loop_index}_drop_{joined}"


def _count_priority_tiers(series: pd.Series) -> Dict[str, int]:
    counts = series.fillna("none").astype(str).value_counts().to_dict()
    return {
        "none": int(counts.get("none", 0)),
        "low": int(counts.get("low", 0)),
        "medium": int(counts.get("medium", 0)),
        "high": int(counts.get("high", 0)),
    }


def _reason_codes_for_row(row: pd.Series) -> str:
    codes: List[str] = []
    if float(row.get("activity_cliff_local_sali_norm", 0.0) or 0.0) >= 0.8:
        codes.append("HIGH_LOCAL_SALI")
    if int(row.get("activity_cliff_neighbor_count", 0) or 0) >= 3:
        codes.append("MULTI_NEIGHBOR_SUPPORT")
    if float(row.get("activity_cliff_max_activity_gap", 0.0) or 0.0) > 0:
        codes.append("EXTREME_LOCAL_GAP")
    if float(row.get("activity_cliff_residual_norm", 0.0) or 0.0) >= 0.8:
        codes.append("OOF_HARD_CASE")
    if not codes and bool(row.get("activity_cliff_flag", False)):
        codes.append("FLAGGED_BY_SALI")
    return "|".join(codes)


def _filter_justification_for_row(row: pd.Series) -> str:
    tier = str(row.get("activity_cliff_priority_tier") or "none")
    if tier == "none":
        return "Not flagged by SALI under the current dataset-aware policy."
    reasons = []
    if float(row.get("activity_cliff_local_sali_norm", 0.0) or 0.0) >= 0.8:
        reasons.append("strong local SALI signal")
    if int(row.get("activity_cliff_neighbor_count", 0) or 0) >= 3:
        reasons.append("supported by multiple close analogs")
    if float(row.get("activity_cliff_residual_norm", 0.0) or 0.0) >= 0.8:
        reasons.append("hard for the model under OOF")
    if not reasons:
        reasons.append("flagged by the dataset-aware SALI ranking")
    return f"{tier.title()} priority: " + ", ".join(reasons) + "."


def parse_activity_cliff_config(
    extra_args: Optional[Dict[str, Any]],
) -> ActivityCliffConfig:
    raw = dict(extra_args or {})
    feedback_requested = bool(raw.get("activity_cliff_feedback", False))
    loops = int(raw.get("activity_cliff_feedback_loops", 0) or 0)
    index_name = str(raw.get("activity_cliff_index", DEFAULT_ACTIVITY_CLIFF_INDEX) or DEFAULT_ACTIVITY_CLIFF_INDEX)
    similarity_threshold = float(
        raw.get(
            "activity_cliff_similarity_threshold",
            DEFAULT_ACTIVITY_CLIFF_SIMILARITY_THRESHOLD,
        )
    )
    k_neighbors = int(raw.get("activity_cliff_k_neighbors", DEFAULT_ACTIVITY_CLIFF_K_NEIGHBORS) or 10)
    oof_folds = int(raw.get("activity_cliff_oof_folds", DEFAULT_ACTIVITY_CLIFF_OOF_FOLDS) or 5)
    sali_flag_threshold = float(raw.get("activity_cliff_sali_flag_threshold", DEFAULT_SALI_FLAG_THRESHOLD))

    if feedback_requested and loops < 1:
        raise ValueError(
            "activity_cliff_feedback_loops must be in [1, 3] when activity_cliff_feedback is enabled."
        )
    if loops < 0:
        raise ValueError("activity_cliff_feedback_loops must be >= 0.")
    if loops > MAX_ACTIVITY_CLIFF_LOOPS:
        raise ValueError("activity_cliff_feedback_loops cannot exceed 3 in V2.")
    if index_name.lower() != DEFAULT_ACTIVITY_CLIFF_INDEX:
        raise ValueError("Only `activity_cliff_index=\"sali\"` is supported in V2.")
    if not (0.0 < similarity_threshold <= 1.0):
        raise ValueError("activity_cliff_similarity_threshold must be in (0, 1].")
    if k_neighbors < 1:
        raise ValueError("activity_cliff_k_neighbors must be >= 1.")
    if oof_folds < 2:
        raise ValueError("activity_cliff_oof_folds must be >= 2.")
    if not (0.0 < sali_flag_threshold <= 1.0):
        raise ValueError("activity_cliff_sali_flag_threshold must be in (0, 1].")

    mode = "with_feedback_loops" if feedback_requested or loops > 0 else "annotate_only"
    return ActivityCliffConfig(
        mode=mode,
        loops=loops,
        index_name=DEFAULT_ACTIVITY_CLIFF_INDEX,
        similarity_threshold=similarity_threshold,
        k_neighbors=k_neighbors,
        oof_folds=oof_folds,
        sali_flag_threshold=sali_flag_threshold,
    )


def merge_activity_cliff_args(
    *,
    extra_args: Optional[Dict[str, Any]],
    activity_cliff_feedback: bool = False,
    activity_cliff_feedback_loops: int = 0,
    activity_cliff_step_percentile: float = 5.0,
    activity_cliff_similarity_threshold: float = DEFAULT_ACTIVITY_CLIFF_SIMILARITY_THRESHOLD,
    activity_cliff_k_neighbors: int = DEFAULT_ACTIVITY_CLIFF_K_NEIGHBORS,
    activity_cliff_oof_folds: int = DEFAULT_ACTIVITY_CLIFF_OOF_FOLDS,
    activity_cliff_index: str = DEFAULT_ACTIVITY_CLIFF_INDEX,
) -> Dict[str, Any]:
    merged = dict(extra_args or {})
    merged["activity_cliff_feedback"] = bool(activity_cliff_feedback)
    merged["activity_cliff_feedback_loops"] = int(activity_cliff_feedback_loops)
    # Deprecated in V2, but preserved temporarily for compatibility with older callers.
    merged["activity_cliff_step_percentile"] = float(activity_cliff_step_percentile)
    merged["activity_cliff_similarity_threshold"] = float(activity_cliff_similarity_threshold)
    merged["activity_cliff_k_neighbors"] = int(activity_cliff_k_neighbors)
    merged["activity_cliff_oof_folds"] = int(activity_cliff_oof_folds)
    merged["activity_cliff_index"] = str(activity_cliff_index or DEFAULT_ACTIVITY_CLIFF_INDEX)
    return merged


def strip_activity_cliff_args(extra_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cleaned = dict(extra_args or {})
    for key in (
        "activity_cliff_feedback",
        "activity_cliff_feedback_loops",
        "activity_cliff_step_percentile",
        "activity_cliff_similarity_threshold",
        "activity_cliff_k_neighbors",
        "activity_cliff_oof_folds",
        "activity_cliff_index",
        "activity_cliff_sali_flag_threshold",
    ):
        cleaned.pop(key, None)
    return cleaned


def build_random_oof_splits(
    *,
    n_rows: int,
    n_folds: int,
    random_state: int = 42,
    val_fraction_within_train: float = 0.1,
) -> List[Dict[str, List[int]]]:
    if n_rows < max(MIN_TRAINING_ROWS, n_folds):
        raise ValueError("Dataset is too small to build activity-cliff OOF folds.")

    indices = np.arange(n_rows)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    payloads: List[Dict[str, List[int]]] = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(indices), start=1):
        if len(train_val_idx) < MIN_TRAINING_ROWS - 1:
            raise ValueError(
                f"OOF fold {fold_idx} leaves too few rows in the train/val portion."
            )
        val_size = max(1, int(round(len(train_val_idx) * val_fraction_within_train)))
        if val_size >= len(train_val_idx):
            val_size = max(1, len(train_val_idx) - 1)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=random_state + fold_idx,
            shuffle=True,
        )
        payloads.append(
            {
                "train": sorted(int(i) for i in train_idx),
                "val": sorted(int(i) for i in val_idx),
                "test": sorted(int(i) for i in test_idx),
            }
        )
    return payloads


def compute_oof_residual_norm(oof_df: pd.DataFrame, target_column: str) -> pd.Series:
    y_true = pd.to_numeric(oof_df[target_column], errors="coerce")
    y_pred = pd.to_numeric(oof_df["oof_prediction"], errors="coerce")
    residuals = (y_true - y_pred).abs()
    valid = residuals.dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(oof_df), dtype=float), index=oof_df.index)
    p95 = float(np.percentile(valid.to_numpy(dtype=float), 95))
    if p95 <= 0:
        return pd.Series(np.zeros(len(oof_df), dtype=float), index=oof_df.index)
    return residuals.fillna(0.0).astype(float).clip(lower=0.0).div(p95).clip(upper=1.0)


def _assign_priority_tiers(
    *,
    sali_norm: pd.Series,
    flagged_mask: pd.Series,
) -> pd.Series:
    tiers = pd.Series("none", index=sali_norm.index, dtype=object)
    flagged_indices = sali_norm[flagged_mask].sort_values(ascending=False, kind="mergesort").index.tolist()
    n_flagged = len(flagged_indices)
    if n_flagged == 0:
        return tiers

    high_end = int(math.ceil(n_flagged * 0.20))
    medium_end = int(math.ceil(n_flagged * 0.40))
    for rank, idx in enumerate(flagged_indices):
        if rank < high_end:
            tiers.at[idx] = "high"
        elif rank < medium_end:
            tiers.at[idx] = "medium"
        else:
            tiers.at[idx] = "low"
    return tiers


def compute_activity_cliff_annotation(
    *,
    dataset: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    oof_predictions: pd.Series,
    similarity_threshold: float,
    k_neighbors: int,
    sali_flag_threshold: float = DEFAULT_SALI_FLAG_THRESHOLD,
    index_name: str = DEFAULT_ACTIVITY_CLIFF_INDEX,
    radius: int = 2,
    nbits: int = 2048,
) -> pd.DataFrame:
    if smiles_column not in dataset.columns:
        raise ValueError(f"Missing smiles column: {smiles_column}")
    if target_column not in dataset.columns:
        raise ValueError(f"Missing target column: {target_column}")
    if len(oof_predictions) != len(dataset):
        raise ValueError("OOF predictions length must match dataset length.")

    working = dataset.copy().reset_index(drop=True)
    working["oof_prediction"] = pd.to_numeric(oof_predictions, errors="coerce")
    y_true = pd.to_numeric(working[target_column], errors="coerce")
    valid_activity = y_true.dropna()
    if valid_activity.empty:
        raise ValueError("Could not compute activity cliffs: target column is empty after numeric coercion.")

    generator = _fingerprint_generator(radius=radius, nbits=nbits)
    fps: List[Any] = []
    for smiles in working[smiles_column].tolist():
        try:
            mol = None
            if isinstance(smiles, str) and smiles.strip():
                from rdkit import Chem

                mol = Chem.MolFromSmiles(smiles)
            fp = generator.GetFingerprint(mol) if mol is not None else None
        except Exception:
            fp = None
        fps.append(fp)

    local_sali_raw = np.zeros(len(working), dtype=float)
    neighbor_counts = np.zeros(len(working), dtype=int)
    max_similarity = np.zeros(len(working), dtype=float)
    max_activity_gap = np.zeros(len(working), dtype=float)
    pair_sali_values: List[float] = []

    for idx, fp in enumerate(fps):
        if fp is None or pd.isna(y_true.iloc[idx]):
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        candidates: List[tuple[float, int, float, float]] = []
        for other_idx, sim in enumerate(sims):
            if other_idx == idx or fps[other_idx] is None or pd.isna(y_true.iloc[other_idx]):
                continue
            sim_value = float(sim)
            if sim_value < similarity_threshold:
                continue
            gap = abs(float(y_true.iloc[idx]) - float(y_true.iloc[other_idx]))
            if sim_value >= 1.0 and gap <= 0.0:
                continue
            denominator = max(1.0 - sim_value, SALI_EPSILON)
            sali_value = float(gap / denominator)
            pair_sali_values.append(sali_value)
            candidates.append((sim_value, other_idx, gap, sali_value))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        top_candidates = candidates[:k_neighbors]
        neighbor_counts[idx] = len(top_candidates)
        max_similarity[idx] = max(float(item[0]) for item in top_candidates)
        max_activity_gap[idx] = max(float(item[2]) for item in top_candidates)
        local_sali_raw[idx] = max(float(item[3]) for item in top_candidates)

    sali_p95 = float(np.percentile(pair_sali_values, 95)) if pair_sali_values else 0.0
    if sali_p95 <= 0:
        sali_p95 = 1.0

    residual_norm = compute_oof_residual_norm(working, target_column)
    residual_oof = (
        pd.to_numeric(working[target_column], errors="coerce")
        - pd.to_numeric(working["oof_prediction"], errors="coerce")
    ).abs()
    local_sali_norm = pd.Series(local_sali_raw, index=working.index).astype(float).div(sali_p95).clip(0.0, 1.0)
    flagged_mask = (local_sali_norm >= float(sali_flag_threshold)) & (pd.Series(neighbor_counts, index=working.index) >= 1)
    priority_tier = _assign_priority_tiers(sali_norm=local_sali_norm, flagged_mask=flagged_mask)

    working["activity_cliff_index_name"] = str(index_name)
    working["activity_cliff_local_sali_raw"] = pd.Series(local_sali_raw, index=working.index).astype(float)
    working["activity_cliff_local_sali_norm"] = local_sali_norm.astype(float)
    working["activity_cliff_neighbor_count"] = pd.Series(neighbor_counts, index=working.index).astype(int)
    working["activity_cliff_max_similarity"] = pd.Series(max_similarity, index=working.index).astype(float)
    working["activity_cliff_max_activity_gap"] = pd.Series(max_activity_gap, index=working.index).astype(float)
    working["activity_cliff_flag"] = flagged_mask.astype(bool)
    working["activity_cliff_priority_tier"] = priority_tier.astype(str)
    working["activity_cliff_residual_oof"] = residual_oof.fillna(0.0).astype(float)
    working["activity_cliff_residual_norm"] = residual_norm.astype(float).clip(0.0, 1.0)
    working["activity_cliff_reason_codes"] = working.apply(_reason_codes_for_row, axis=1)
    working["activity_cliff_filter_justification"] = working.apply(_filter_justification_for_row, axis=1)
    return working


def write_activity_cliff_artifacts(
    *,
    annotated_df: pd.DataFrame,
    target_column: str,
    output_dir: str,
    loops: int,
    min_training_rows: int = MIN_TRAINING_ROWS,
) -> Dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    annotated_path = output_path / "activity_cliff_annotated_training.csv"
    annotated_df.to_csv(annotated_path, index=False)

    tier_counts = _count_priority_tiers(annotated_df["activity_cliff_priority_tier"])
    flagged_count = int(bool_col.sum()) if (bool_col := annotated_df["activity_cliff_flag"].astype(bool)).size else 0

    variants: List[Dict[str, Any]] = []
    warnings: List[str] = []
    loop_tiers = {
        1: ["high"],
        2: ["high", "medium"],
        3: ["high", "medium", "low"],
    }
    for loop_index in range(1, loops + 1):
        removed_tiers = loop_tiers.get(loop_index, [])
        if not removed_tiers:
            continue
        kept_df = annotated_df.loc[
            ~annotated_df["activity_cliff_priority_tier"].isin(removed_tiers)
        ].copy()
        removed_count = int(len(annotated_df) - len(kept_df))
        if removed_count <= 0:
            warnings.append(
                f"Skipped activity-cliff loop {loop_index}; no compounds belonged to tiers {removed_tiers}."
            )
            continue
        if len(kept_df) < min_training_rows:
            warnings.append(
                f"Stopped activity-cliff loops before loop {loop_index}; only {len(kept_df)} rows would remain."
            )
            break
        variant_id = _coerce_loop_variant_id(loop_index, removed_tiers)
        filtered_path = output_path / f"activity_cliff_filtered_{variant_id}.csv"
        kept_df.reset_index(drop=True).to_csv(filtered_path, index=False)
        variants.append(
            {
                "variant_id": variant_id,
                "loop_index": int(loop_index),
                "removed_tiers": removed_tiers,
                "removed_count": removed_count,
                "remaining_rows": int(len(kept_df)),
                "filtered_training_csv": str(filtered_path),
            }
        )

    summary = {
        "index_name": str(annotated_df.get("activity_cliff_index_name", pd.Series([DEFAULT_ACTIVITY_CLIFF_INDEX])).iloc[0]),
        "target_column": target_column,
        "annotated_training_csv": str(annotated_path),
        "ranked_molecule_count": int(len(annotated_df)),
        "flagged_count": flagged_count,
        "loops_requested": int(loops),
        "priority_counts": tier_counts,
        "tiering_policy": {
            "strategy": "dataset_aware_quantiles_on_flagged_compounds",
            "high_fraction": 0.20,
            "medium_fraction": 0.20,
            "tier_source_column": "activity_cliff_local_sali_norm",
            "non_flagged_tier": "none",
        },
        "variants": variants,
        "warnings": warnings,
        "score_summary": {
            "activity_cliff_local_sali_raw_max": float(
                pd.to_numeric(annotated_df["activity_cliff_local_sali_raw"], errors="coerce").max() or 0.0
            ),
            "activity_cliff_local_sali_norm_max": float(
                pd.to_numeric(annotated_df["activity_cliff_local_sali_norm"], errors="coerce").max() or 0.0
            ),
        },
    }
    summary_path = output_path / "activity_cliff_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["summary_path"] = str(summary_path)
    return summary


def choose_recommended_variant(
    variants: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not variants:
        raise ValueError("At least one activity-cliff variant is required.")

    def sort_key(item: Dict[str, Any]) -> tuple[float, float]:
        metrics = ((item.get("training_result") or {}).get("metrics") or {}).get("test") or {}
        r2 = metrics.get("r2")
        rmse = metrics.get("rmse")
        safe_r2 = float(r2) if r2 is not None else float("-inf")
        safe_rmse = float(rmse) if rmse is not None else float("inf")
        return (safe_r2, -safe_rmse)

    return max(variants, key=sort_key)


def build_activity_cliff_comparison_metrics(
    variants: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in variants:
        training_result = item.get("training_result") or {}
        metrics = (training_result.get("metrics") or {}).get("test") or {}
        split_metrics: Dict[str, Dict[str, Any]] = {}
        for split_result in training_result.get("split_results") or []:
            label = str(split_result.get("strategy_label") or split_result.get("strategy") or "").strip().lower()
            split_metrics[label] = ((split_result.get("metrics") or {}).get("test") or {})
        rows.append(
            {
                "variant_id": item.get("variant_id"),
                "loop_index": item.get("loop_index", 0),
                "removed_tiers": item.get("removed_tiers") or [],
                "removed_count": item.get("removed_count"),
                "remaining_rows": item.get("remaining_rows"),
                "r2": metrics.get("r2"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "mse": metrics.get("mse"),
                "strategy_label": training_result.get("strategy_label"),
                "split_metrics": split_metrics,
            }
        )
    return rows


def load_csv(path: str) -> pd.DataFrame:
    with S3.open(path, "r") as fh:
        return pd.read_csv(fh)
