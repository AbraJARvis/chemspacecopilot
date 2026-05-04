#!/usr/bin/env python
# coding: utf-8
"""
Shared helpers for activity-cliff feedback loops in QSAR training.
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


DEFAULT_ACTIVITY_CLIFF_STEP_PERCENTILE = 5.0
DEFAULT_ACTIVITY_CLIFF_SIMILARITY_THRESHOLD = 0.70
DEFAULT_ACTIVITY_CLIFF_K_NEIGHBORS = 10
DEFAULT_ACTIVITY_CLIFF_OOF_FOLDS = 5
MIN_TRAINING_ROWS = 10


@dataclass
class ActivityCliffConfig:
    enabled: bool = False
    loops: int = 1
    step_percentile: float = DEFAULT_ACTIVITY_CLIFF_STEP_PERCENTILE
    similarity_threshold: float = DEFAULT_ACTIVITY_CLIFF_SIMILARITY_THRESHOLD
    k_neighbors: int = DEFAULT_ACTIVITY_CLIFF_K_NEIGHBORS
    oof_folds: int = DEFAULT_ACTIVITY_CLIFF_OOF_FOLDS


def _fingerprint_generator(radius: int = 2, nbits: int = 2048):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)


def _safe_slug(value: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "value"


def parse_activity_cliff_config(
    extra_args: Optional[Dict[str, Any]],
) -> ActivityCliffConfig:
    raw = dict(extra_args or {})
    enabled = bool(raw.get("activity_cliff_feedback", False))
    loops = int(raw.get("activity_cliff_feedback_loops", 1) or 1)
    step_percentile = float(
        raw.get("activity_cliff_step_percentile", DEFAULT_ACTIVITY_CLIFF_STEP_PERCENTILE)
    )
    similarity_threshold = float(
        raw.get(
            "activity_cliff_similarity_threshold",
            DEFAULT_ACTIVITY_CLIFF_SIMILARITY_THRESHOLD,
        )
    )
    k_neighbors = int(raw.get("activity_cliff_k_neighbors", DEFAULT_ACTIVITY_CLIFF_K_NEIGHBORS) or 10)
    oof_folds = int(raw.get("activity_cliff_oof_folds", DEFAULT_ACTIVITY_CLIFF_OOF_FOLDS) or 5)

    if enabled and loops < 1:
        raise ValueError("activity_cliff_feedback_loops must be >= 1 when activity_cliff_feedback is enabled.")
    if step_percentile <= 0:
        raise ValueError("activity_cliff_step_percentile must be > 0.")
    if step_percentile > 50:
        raise ValueError(
            "activity_cliff_step_percentile represents the percent of compounds removed at each loop "
            "(for example 5.0 for a 5% removal step). Values above 50 are rejected in V1 because they "
            "usually indicate an inverted retained-percentile input such as 95 instead of 5."
        )
    if not (0.0 < similarity_threshold <= 1.0):
        raise ValueError("activity_cliff_similarity_threshold must be in (0, 1].")
    if k_neighbors < 1:
        raise ValueError("activity_cliff_k_neighbors must be >= 1.")
    if oof_folds < 2:
        raise ValueError("activity_cliff_oof_folds must be >= 2.")

    return ActivityCliffConfig(
        enabled=enabled,
        loops=loops,
        step_percentile=step_percentile,
        similarity_threshold=similarity_threshold,
        k_neighbors=k_neighbors,
        oof_folds=oof_folds,
    )


def merge_activity_cliff_args(
    *,
    extra_args: Optional[Dict[str, Any]],
    activity_cliff_feedback: bool = False,
    activity_cliff_feedback_loops: int = 1,
    activity_cliff_step_percentile: float = DEFAULT_ACTIVITY_CLIFF_STEP_PERCENTILE,
    activity_cliff_similarity_threshold: float = DEFAULT_ACTIVITY_CLIFF_SIMILARITY_THRESHOLD,
    activity_cliff_k_neighbors: int = DEFAULT_ACTIVITY_CLIFF_K_NEIGHBORS,
    activity_cliff_oof_folds: int = DEFAULT_ACTIVITY_CLIFF_OOF_FOLDS,
) -> Dict[str, Any]:
    merged = dict(extra_args or {})
    merged["activity_cliff_feedback"] = bool(activity_cliff_feedback)
    merged["activity_cliff_feedback_loops"] = int(activity_cliff_feedback_loops)
    merged["activity_cliff_step_percentile"] = float(activity_cliff_step_percentile)
    merged["activity_cliff_similarity_threshold"] = float(activity_cliff_similarity_threshold)
    merged["activity_cliff_k_neighbors"] = int(activity_cliff_k_neighbors)
    merged["activity_cliff_oof_folds"] = int(activity_cliff_oof_folds)
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


def compute_activity_cliff_annotation(
    *,
    dataset: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    oof_predictions: pd.Series,
    similarity_threshold: float,
    k_neighbors: int,
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
    valid_mask = []
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
        valid_mask.append(fp is not None)

    non_nan_targets = y_true.dropna().to_numpy(dtype=float)
    pairwise_gap_p95 = float(np.percentile(np.abs(non_nan_targets[:, None] - non_nan_targets[None, :]).ravel(), 95))
    if pairwise_gap_p95 <= 0:
        pairwise_gap_p95 = 1.0

    residual_norm = compute_oof_residual_norm(working, target_column)
    cliff_scores = np.zeros(len(working), dtype=float)
    neighbor_counts = np.zeros(len(working), dtype=int)

    for idx, fp in enumerate(fps):
        if fp is None or pd.isna(y_true.iloc[idx]):
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        candidates: List[tuple[float, int]] = []
        for other_idx, sim in enumerate(sims):
            if other_idx == idx or fps[other_idx] is None or pd.isna(y_true.iloc[other_idx]):
                continue
            sim_value = float(sim)
            if sim_value >= similarity_threshold:
                candidates.append((sim_value, other_idx))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        top_candidates = candidates[:k_neighbors]
        neighbor_counts[idx] = len(top_candidates)
        best_score = 0.0
        for sim_value, other_idx in top_candidates:
            gap = abs(float(y_true.iloc[idx]) - float(y_true.iloc[other_idx]))
            gap_norm = min(gap / pairwise_gap_p95, 1.0)
            pair_score = sim_value * gap_norm
            if pair_score > best_score:
                best_score = pair_score
        cliff_scores[idx] = float(best_score)

    working["activity_cliff_score"] = pd.Series(cliff_scores, index=working.index).clip(0.0, 1.0)
    working["activity_cliff_residual_norm"] = residual_norm.astype(float).clip(0.0, 1.0)
    working["activity_cliff_suspicion_score"] = (
        working["activity_cliff_score"] * working["activity_cliff_residual_norm"]
    ).astype(float).clip(0.0, 1.0)
    working["activity_cliff_neighbor_count"] = neighbor_counts.astype(int)
    return working


def write_activity_cliff_artifacts(
    *,
    annotated_df: pd.DataFrame,
    target_column: str,
    output_dir: str,
    loops: int,
    step_percentile: float,
    min_training_rows: int = MIN_TRAINING_ROWS,
) -> Dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    annotated_path = output_path / "activity_cliff_annotated_training.csv"
    annotated_df.to_csv(annotated_path, index=False)

    n_rows = len(annotated_df)
    sorted_df = annotated_df.sort_values(
        by=[
            "activity_cliff_suspicion_score",
            "activity_cliff_residual_norm",
            "activity_cliff_score",
        ],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    variants: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for loop_index in range(1, loops + 1):
        removed_percent = min(loop_index * step_percentile, 99.0)
        remove_count = int(math.ceil((removed_percent / 100.0) * n_rows))
        if remove_count <= 0:
            continue
        remaining = n_rows - remove_count
        if remaining < min_training_rows:
            warnings.append(
                f"Stopped activity-cliff loops before top {removed_percent:.1f}% removal; only {remaining} rows would remain."
            )
            break

        kept_df = sorted_df.iloc[remove_count:].copy().reset_index(drop=True)
        variant_slug = _safe_slug(f"top_{int(round(removed_percent))}")
        filtered_path = output_path / f"activity_cliff_filtered_{variant_slug}.csv"
        kept_df.to_csv(filtered_path, index=False)
        variants.append(
            {
                "variant_id": f"filtered_top_{int(round(removed_percent))}",
                "removed_percent": float(removed_percent),
                "removed_count": int(remove_count),
                "filtered_training_csv": str(filtered_path),
                "remaining_rows": int(len(kept_df)),
            }
        )

    summary = {
        "target_column": target_column,
        "annotated_training_csv": str(annotated_path),
        "ranked_molecule_count": int(n_rows),
        "loops_requested": int(loops),
        "step_percentile": float(step_percentile),
        "variants": variants,
        "warnings": warnings,
        "score_summary": {
            "activity_cliff_score_max": float(annotated_df["activity_cliff_score"].max() or 0.0),
            "activity_cliff_suspicion_score_max": float(
                annotated_df["activity_cliff_suspicion_score"].max() or 0.0
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
        rows.append(
            {
                "variant_id": item.get("variant_id"),
                "removed_percent": item.get("removed_percent"),
                "removed_count": item.get("removed_count"),
                "r2": metrics.get("r2"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "mse": metrics.get("mse"),
                "strategy_label": training_result.get("strategy_label"),
            }
        )
    return rows


def load_csv(path: str) -> pd.DataFrame:
    with S3.open(path, "r") as fh:
        return pd.read_csv(fh)
