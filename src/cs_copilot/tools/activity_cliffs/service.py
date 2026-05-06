#!/usr/bin/env python
# coding: utf-8
"""Core Activity Cliff services.

This module intentionally treats SALI as the first registered index, not as the
framework itself. Training code should depend on the generic
``activity_cliff_*`` contract so later indexes can be added without changing
backend toolkits.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import matplotlib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from cs_copilot.storage import S3

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTIVITY_CLIFF_ANNOTATION_PREFIX = "activity_cliff_"
DEFAULT_ACTIVITY_CLIFF_INDEX = "sali"
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_TOP_K_NEIGHBORS = 10
DEFAULT_FLAG_THRESHOLD = 0.35
MAX_ACTIVITY_CLIFF_LOOPS = 3
MIN_VARIANT_TRAINING_ROWS = 10
SALI_EPSILON = 1e-6
PLOT_DPI = 300
ACTIVITY_CLIFF_ARG_KEYS = {
    "activity_cliff_index",
    "activity_cliff_feedback",
    "activity_cliff_feedback_loops",
    "activity_cliff_similarity_threshold",
    "activity_cliff_top_k_neighbors",
    "activity_cliff_k_neighbors",
    "activity_cliff_flag_threshold",
}


@dataclass(frozen=True)
class ActivityCliffConfig:
    index_name: str = DEFAULT_ACTIVITY_CLIFF_INDEX
    mode: str = "standard"
    feedback_loops: int = 0
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    top_k_neighbors: int = DEFAULT_TOP_K_NEIGHBORS
    flag_threshold: float = DEFAULT_FLAG_THRESHOLD

    @property
    def loops_enabled(self) -> bool:
        return self.mode == "with_feedback_loops" and self.feedback_loops > 0


class ActivityCliffIndex(Protocol):
    index_name: str

    def compute(
        self,
        *,
        dataset: pd.DataFrame,
        smiles_column: str,
        target_column: str,
        config: ActivityCliffConfig,
    ) -> pd.DataFrame:
        """Return index-specific and generic activity-cliff annotations."""


class ActivityCliffIndexRegistry:
    """Registry for pluggable activity-cliff indexes."""

    def __init__(self):
        self._indexes: Dict[str, ActivityCliffIndex] = {}

    def register(self, index: ActivityCliffIndex) -> None:
        self._indexes[index.index_name.lower()] = index

    def list_indexes(self) -> List[str]:
        return sorted(self._indexes)

    def get(self, index_name: str) -> ActivityCliffIndex:
        normalized = str(index_name or DEFAULT_ACTIVITY_CLIFF_INDEX).lower()
        if normalized not in self._indexes:
            raise ValueError(
                "Unsupported activity_cliff_index "
                f"`{index_name}`. Available indexes: {self.list_indexes()}."
            )
        return self._indexes[normalized]


class SALIIndex:
    """Structure-Activity Landscape Index implementation."""

    index_name = DEFAULT_ACTIVITY_CLIFF_INDEX

    def compute(
        self,
        *,
        dataset: pd.DataFrame,
        smiles_column: str,
        target_column: str,
        config: ActivityCliffConfig,
    ) -> pd.DataFrame:
        if smiles_column not in dataset.columns:
            raise ValueError(f"Missing smiles column for activity cliffs: {smiles_column}")
        if target_column not in dataset.columns:
            raise ValueError(f"Missing target column for activity cliffs: {target_column}")

        working = dataset.copy().reset_index(drop=True)
        y_true = pd.to_numeric(working[target_column], errors="coerce")
        if y_true.dropna().empty:
            raise ValueError("Cannot compute activity cliffs: target column has no numeric values.")

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps: List[Any] = []
        for smiles in working[smiles_column].tolist():
            try:
                mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
                fps.append(generator.GetFingerprint(mol) if mol is not None else None)
            except Exception:
                fps.append(None)

        raw_scores = np.zeros(len(working), dtype=float)
        neighbor_counts = np.zeros(len(working), dtype=int)
        max_similarity = np.zeros(len(working), dtype=float)
        max_activity_gap = np.zeros(len(working), dtype=float)
        pair_scores: List[float] = []

        for idx, fp in enumerate(fps):
            if fp is None or pd.isna(y_true.iloc[idx]):
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
            candidates: List[tuple[float, float, float]] = []
            for other_idx, sim in enumerate(sims):
                if other_idx == idx or fps[other_idx] is None or pd.isna(y_true.iloc[other_idx]):
                    continue
                sim_value = float(sim)
                if sim_value < config.similarity_threshold:
                    continue
                gap = abs(float(y_true.iloc[idx]) - float(y_true.iloc[other_idx]))
                if sim_value >= 1.0 and gap <= 0.0:
                    continue
                sali = gap / max(1.0 - sim_value, SALI_EPSILON)
                pair_scores.append(float(sali))
                candidates.append((sim_value, gap, float(sali)))
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0], reverse=True)
            top_candidates = candidates[: config.top_k_neighbors]
            neighbor_counts[idx] = len(top_candidates)
            max_similarity[idx] = max(item[0] for item in top_candidates)
            max_activity_gap[idx] = max(item[1] for item in top_candidates)
            raw_scores[idx] = max(item[2] for item in top_candidates)

        p95 = float(np.percentile(pair_scores, 95)) if pair_scores else 0.0
        if p95 <= 0:
            p95 = 1.0
        norm_scores = pd.Series(raw_scores, index=working.index).div(p95).clip(0.0, 1.0)

        working["activity_cliff_index_name"] = self.index_name
        working["activity_cliff_score_raw"] = pd.Series(raw_scores, index=working.index).astype(float)
        working["activity_cliff_score_norm"] = norm_scores.astype(float)
        working["activity_cliff_neighbor_count"] = pd.Series(
            neighbor_counts, index=working.index
        ).astype(int)
        working["activity_cliff_sali_raw"] = working["activity_cliff_score_raw"]
        working["activity_cliff_sali_norm"] = working["activity_cliff_score_norm"]
        working["activity_cliff_max_similarity"] = pd.Series(
            max_similarity, index=working.index
        ).astype(float)
        working["activity_cliff_max_activity_gap"] = pd.Series(
            max_activity_gap, index=working.index
        ).astype(float)
        return working


def default_registry() -> ActivityCliffIndexRegistry:
    registry = ActivityCliffIndexRegistry()
    registry.register(SALIIndex())
    return registry


def strip_activity_cliff_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith(ACTIVITY_CLIFF_ANNOTATION_PREFIX)].copy()


def split_activity_cliff_args(extra_args: Optional[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    raw = dict(extra_args or {})
    activity_args: Dict[str, Any] = {}
    cleaned: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in ACTIVITY_CLIFF_ARG_KEYS:
            normalized_key = (
                "activity_cliff_top_k_neighbors"
                if key == "activity_cliff_k_neighbors"
                else key
            )
            activity_args[normalized_key] = value
        else:
            cleaned[key] = value
    return cleaned, activity_args


def parse_activity_cliff_config(
    *,
    activity_cliff_index: str = DEFAULT_ACTIVITY_CLIFF_INDEX,
    activity_cliff_feedback: bool = False,
    activity_cliff_feedback_loops: int = 0,
    activity_cliff_similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    activity_cliff_top_k_neighbors: int = DEFAULT_TOP_K_NEIGHBORS,
    activity_cliff_flag_threshold: float = DEFAULT_FLAG_THRESHOLD,
    registry: Optional[ActivityCliffIndexRegistry] = None,
) -> ActivityCliffConfig:
    registry = registry or default_registry()
    registry.get(activity_cliff_index)
    loops = int(activity_cliff_feedback_loops or 0)
    if loops < 0:
        raise ValueError("activity_cliff_feedback_loops must be >= 0.")
    if loops > MAX_ACTIVITY_CLIFF_LOOPS:
        raise ValueError("activity_cliff_feedback_loops cannot exceed 3.")
    if bool(activity_cliff_feedback) and loops < 1:
        raise ValueError(
            "activity_cliff_feedback_loops must be between 1 and 3 when feedback loops are requested."
        )
    if not (0.0 < float(activity_cliff_similarity_threshold) <= 1.0):
        raise ValueError("activity_cliff_similarity_threshold must be in (0, 1].")
    if int(activity_cliff_top_k_neighbors) < 1:
        raise ValueError("activity_cliff_top_k_neighbors must be >= 1.")
    if not (0.0 < float(activity_cliff_flag_threshold) <= 1.0):
        raise ValueError("activity_cliff_flag_threshold must be in (0, 1].")
    return ActivityCliffConfig(
        index_name=str(activity_cliff_index or DEFAULT_ACTIVITY_CLIFF_INDEX).lower(),
        mode="with_feedback_loops" if loops > 0 or activity_cliff_feedback else "standard",
        feedback_loops=loops,
        similarity_threshold=float(activity_cliff_similarity_threshold),
        top_k_neighbors=int(activity_cliff_top_k_neighbors),
        flag_threshold=float(activity_cliff_flag_threshold),
    )


def _assign_tiers(score_norm: pd.Series, flagged: pd.Series) -> pd.Series:
    tiers = pd.Series("none", index=score_norm.index, dtype=object)
    ranked = score_norm[flagged].sort_values(ascending=False, kind="mergesort")
    n_flagged = len(ranked)
    if n_flagged == 0:
        return tiers
    high_end = max(1, int(math.ceil(n_flagged * 0.20)))
    medium_end = max(high_end, int(math.ceil(n_flagged * 0.40)))
    for rank, idx in enumerate(ranked.index.tolist()):
        if rank < high_end:
            tiers.at[idx] = "high"
        elif rank < medium_end:
            tiers.at[idx] = "medium"
        else:
            tiers.at[idx] = "low"
    return tiers


def _reason_codes(row: pd.Series) -> str:
    codes: List[str] = []
    if bool(row.get("activity_cliff_flag", False)):
        codes.append("HIGH_ACTIVITY_DISCONTINUITY")
    if float(row.get("activity_cliff_score_norm", 0.0) or 0.0) >= 0.8:
        codes.append("HIGH_NORMALIZED_INDEX_SCORE")
    if int(row.get("activity_cliff_neighbor_count", 0) or 0) >= 3:
        codes.append("MULTIPLE_SIMILAR_NEIGHBORS")
    return "|".join(codes) if codes else "NOT_FLAGGED"


def _justification(row: pd.Series) -> str:
    tier = str(row.get("activity_cliff_priority_tier") or "none")
    if tier == "none":
        return "Not flagged by the selected activity-cliff index under the current policy."
    return (
        f"{tier.title()} priority from the selected activity-cliff index: "
        f"normalized score={float(row.get('activity_cliff_score_norm', 0.0) or 0.0):.3f}, "
        f"qualified_neighbors={int(row.get('activity_cliff_neighbor_count', 0) or 0)}."
    )


def _priority_counts(series: pd.Series) -> Dict[str, int]:
    counts = series.fillna("none").astype(str).value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in ("none", "low", "medium", "high")}


def _plot_score_histogram(
    annotated: pd.DataFrame,
    output_path: Path,
    *,
    flag_threshold: float,
) -> Optional[str]:
    values = pd.to_numeric(annotated["activity_cliff_score_norm"], errors="coerce").dropna()
    if values.empty:
        return None
    flagged_values = values[values >= flag_threshold]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=30, color="#224f75", edgecolor="white", alpha=0.9)
    ax.axvline(
        flag_threshold,
        color="#b54a4a",
        linewidth=2,
        linestyle="--",
        label=f"Flag threshold {flag_threshold:g}",
    )
    if not flagged_values.empty:
        ax.hist(flagged_values, bins=12, color="#d98b36", edgecolor="white", alpha=0.95, label="Flagged")
    ax.set_title("Activity-cliff normalized score distribution")
    ax.set_xlabel("Normalized activity-cliff score")
    ax.set_ylabel("Compound count")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_tier_distribution(annotated: pd.DataFrame, output_path: Path) -> Optional[str]:
    counts = _priority_counts(annotated["activity_cliff_priority_tier"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = ["low", "medium", "high"]
    values = [counts[label] for label in labels]
    bars = ax.bar(labels, values, color=["#258d9a", "#d98b36", "#b54a4a"])
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values + [1]) * 0.03,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Flagged activity-cliff tier distribution")
    ax.set_xlabel("Tier")
    ax.set_ylabel("Flagged compound count")
    ax.set_ylim(0, max(values + [1]) * 1.25)
    ax.grid(alpha=0.2, linestyle="--", axis="y")
    ax.text(
        0.99,
        0.94,
        f"Not flagged: {counts['none']}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#5f6972",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_gap_vs_similarity(annotated: pd.DataFrame, output_path: Path) -> Optional[str]:
    if "activity_cliff_max_similarity" not in annotated.columns:
        return None
    x = pd.to_numeric(annotated["activity_cliff_max_similarity"], errors="coerce")
    y = pd.to_numeric(annotated["activity_cliff_max_activity_gap"], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tiers = annotated.loc[valid, "activity_cliff_priority_tier"].fillna("none").astype(str)
    none_mask = tiers == "none"
    if none_mask.any():
        ax.scatter(
            x[valid][none_mask],
            y[valid][none_mask],
            s=14,
            c="#a9b2ba",
            alpha=0.18,
            edgecolor="none",
            label="none",
        )
    for tier, color, size in (
        ("low", "#258d9a", 42),
        ("medium", "#d98b36", 52),
        ("high", "#b54a4a", 62),
    ):
        tier_mask = tiers == tier
        if tier_mask.any():
            ax.scatter(
                x[valid][tier_mask],
                y[valid][tier_mask],
                s=size,
                c=color,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
                label=tier,
            )
    ax.set_title("Activity gap versus structural similarity")
    ax.set_xlabel("Maximum qualified Tanimoto similarity")
    ax.set_ylabel("Maximum activity gap")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _write_plots(
    annotated: pd.DataFrame,
    output_dir: Path,
    *,
    flag_threshold: float,
) -> Dict[str, str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated = {
        "activity_cliff_score_histogram": _plot_score_histogram(
            annotated,
            plots_dir / "activity_cliff_sali_histogram.png",
            flag_threshold=flag_threshold,
        ),
        "activity_cliff_tier_distribution": _plot_tier_distribution(
            annotated, plots_dir / "activity_cliff_tier_distribution.png"
        ),
        "activity_gap_vs_similarity": _plot_gap_vs_similarity(
            annotated, plots_dir / "activity_gap_vs_similarity.png"
        ),
    }
    return {key: value for key, value in generated.items() if value}


def _short_variant_label(variant_id: str) -> str:
    if variant_id == "baseline_loop_0":
        return "loop 0"
    if "loop_1" in variant_id:
        return "loop 1"
    if "loop_2" in variant_id:
        return "loop 2"
    if "loop_3" in variant_id:
        return "loop 3"
    return variant_id


def _variant_order(variant_id: str) -> int:
    if variant_id == "baseline_loop_0":
        return 0
    for idx in (1, 2, 3):
        if f"loop_{idx}" in variant_id:
            return idx
    return 99


def _plot_loop_metric_comparison(
    table: pd.DataFrame,
    output_path: Path,
    *,
    metric: str,
    ylabel: str,
) -> Optional[str]:
    if table.empty or metric not in table.columns:
        return None
    rows = table.dropna(subset=[metric]).copy()
    if rows.empty:
        return None
    rows["variant_order"] = rows["variant_id"].astype(str).map(_variant_order)
    rows["variant_label"] = rows["variant_id"].astype(str).map(_short_variant_label)
    rows = rows.sort_values(["variant_order", "split"])
    pivot = rows.pivot_table(index="variant_label", columns="split", values=metric, aggfunc="first")
    if pivot.empty:
        return None

    ordered_labels = rows.drop_duplicates("variant_label").sort_values("variant_order")["variant_label"].tolist()
    pivot = pivot.reindex(ordered_labels)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"random": "#224f75", "scaffold": "#d98b36"}
    for split_name in [column for column in ("random", "scaffold") if column in pivot.columns]:
        ax.plot(
            pivot.index,
            pivot[split_name],
            marker="o",
            linewidth=2,
            color=colors.get(split_name, None),
            label=split_name,
        )
        for x_pos, value in enumerate(pivot[split_name].tolist()):
            if pd.notna(value):
                ax.text(x_pos, float(value), f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Activity-cliff loop {ylabel} comparison")
    ax.set_xlabel("Variant")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linestyle="--", axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_loop_delta_comparison(table: pd.DataFrame, output_path: Path) -> Optional[str]:
    if table.empty or "r2" not in table.columns or "rmse" not in table.columns:
        return None
    rows = table.copy()
    rows["variant_order"] = rows["variant_id"].astype(str).map(_variant_order)
    rows["variant_label"] = rows["variant_id"].astype(str).map(_short_variant_label)
    baseline = rows[rows["variant_id"] == "baseline_loop_0"].set_index("split")
    if baseline.empty:
        return None
    plot_rows: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        split = row.get("split")
        if split not in baseline.index:
            continue
        base = baseline.loc[split]
        plot_rows.append(
            {
                "variant_label": row["variant_label"],
                "variant_order": row["variant_order"],
                "split": split,
                "delta_r2": float(row["r2"]) - float(base["r2"]) if pd.notna(row.get("r2")) else np.nan,
                "delta_rmse": float(row["rmse"]) - float(base["rmse"]) if pd.notna(row.get("rmse")) else np.nan,
            }
        )
    delta = pd.DataFrame(plot_rows)
    if delta.empty:
        return None
    delta = delta[delta["variant_label"] != "loop 0"].sort_values(["variant_order", "split"])
    if delta.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    colors = {"random": "#224f75", "scaffold": "#d98b36"}
    for ax, metric, title, zero_label in (
        (axes[0], "delta_r2", "Delta R2 vs baseline", "baseline"),
        (axes[1], "delta_rmse", "Delta RMSE vs baseline", "baseline"),
    ):
        pivot = delta.pivot_table(index="variant_label", columns="split", values=metric, aggfunc="first")
        labels = delta.drop_duplicates("variant_label").sort_values("variant_order")["variant_label"].tolist()
        pivot = pivot.reindex(labels)
        for split_name in [column for column in ("random", "scaffold") if column in pivot.columns]:
            ax.plot(
                pivot.index,
                pivot[split_name],
                marker="o",
                linewidth=2,
                color=colors.get(split_name, None),
                label=split_name,
            )
        ax.axhline(0.0, color="#5f6972", linewidth=1, linestyle="--", label=zero_label)
        ax.set_title(title)
        ax.grid(alpha=0.2, linestyle="--", axis="y")
    axes[0].set_ylabel("Delta R2")
    axes[1].set_ylabel("Delta RMSE")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_variant_parity_from_predictions(
    predictions_path: Path,
    output_path: Path,
    *,
    label: str,
) -> Optional[str]:
    if not predictions_path.exists():
        return None
    predictions = pd.read_csv(predictions_path)
    if "y_true" not in predictions.columns or "y_pred" not in predictions.columns:
        return None
    frame = pd.DataFrame(
        {
            "y_true": pd.to_numeric(predictions["y_true"], errors="coerce"),
            "y_pred": pd.to_numeric(predictions["y_pred"], errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return None
    frame["residual"] = frame["y_true"] - frame["y_pred"]
    mae = float(frame["residual"].abs().mean())
    rmse = float((frame["residual"].pow(2).mean()) ** 0.5)
    centered = frame["y_true"] - float(frame["y_true"].mean())
    ss_tot = float((centered.pow(2)).sum())
    ss_res = float((frame["residual"].pow(2)).sum())
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else None

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(frame["y_true"], frame["y_pred"], s=18, alpha=0.55, color="#224f75")
    min_val = min(frame["y_true"].min(), frame["y_pred"].min())
    max_val = max(frame["y_true"].max(), frame["y_pred"].max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#b54a4a", linewidth=1.2)
    ax.set_title(f"Scaffold parity - {label}")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_aspect("equal", adjustable="box")
    metrics = [f"RMSE = {rmse:.3f}", f"MAE = {mae:.3f}"]
    if r2 is not None:
        metrics.insert(0, f"R2 = {r2:.3f}")
    ax.text(
        0.03,
        0.97,
        "\n".join(metrics),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    ax.grid(alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def build_activity_cliff_loop_comparison_plots(
    activity_cliffs: Dict[str, Any],
    *,
    output_dir: str,
) -> Dict[str, str]:
    table = pd.DataFrame(activity_cliffs.get("variant_comparison_table") or [])
    if table.empty:
        return {}
    comparison_dir = Path(output_dir).expanduser().resolve() / "loop_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Optional[str]] = {
        "loop_r2_comparison": _plot_loop_metric_comparison(
            table,
            comparison_dir / "activity_cliff_loop_r2_comparison.png",
            metric="r2",
            ylabel="R2",
        ),
        "loop_rmse_comparison": _plot_loop_metric_comparison(
            table,
            comparison_dir / "activity_cliff_loop_rmse_comparison.png",
            metric="rmse",
            ylabel="RMSE",
        ),
        "loop_delta_vs_baseline": _plot_loop_delta_comparison(
            table,
            comparison_dir / "activity_cliff_loop_delta_vs_baseline.png",
        ),
    }
    for variant in activity_cliffs.get("variant_training") or []:
        variant_id = str(variant.get("variant_id") or "")
        if not variant_id:
            continue
        scaffold_result = next(
            (
                result
                for result in (variant.get("split_results") or [])
                if result.get("strategy_label") == "scaffold"
            ),
            None,
        )
        if not scaffold_result or not scaffold_result.get("test_predictions_path"):
            continue
        key = f"parity_scaffold_{variant_id}"
        generated[key] = _plot_variant_parity_from_predictions(
            Path(str(scaffold_result["test_predictions_path"])).expanduser(),
            comparison_dir / f"parity_scaffold_{variant_id}.png",
            label=_short_variant_label(variant_id),
        )
    return {key: value for key, value in generated.items() if value}


def _variant_id(loop_index: int, removed_tiers: List[str]) -> str:
    return f"filtered_loop_{loop_index}_drop_{'_'.join(removed_tiers)}"


def _write_variants(
    *,
    annotated: pd.DataFrame,
    output_dir: Path,
    loops: int,
    min_training_rows: int,
) -> tuple[List[Dict[str, Any]], List[str]]:
    variants: List[Dict[str, Any]] = [
        {
            "variant_id": "baseline_loop_0",
            "loop_index": 0,
            "removed_tiers": [],
            "removed_count": 0,
            "removed_row_indices": [],
            "remaining_rows": int(len(annotated)),
            "filtered_training_csv": None,
        }
    ]
    warnings: List[str] = []
    tier_plan = {
        1: ["high"],
        2: ["high", "medium"],
        3: ["high", "medium", "low"],
    }
    clean_source = strip_activity_cliff_columns(annotated)
    for loop_index in range(1, loops + 1):
        removed_tiers = tier_plan[loop_index]
        mask = annotated["activity_cliff_priority_tier"].isin(removed_tiers)
        removed_count = int(mask.sum())
        if removed_count <= 0:
            warnings.append(
                f"Skipped {loop_index}; no compounds belonged to tiers {removed_tiers}."
            )
            continue
        kept = clean_source.loc[~mask].copy().reset_index(drop=True)
        if len(kept) < min_training_rows:
            warnings.append(
                f"Stopped before loop {loop_index}; only {len(kept)} rows would remain."
            )
            break
        variant = _variant_id(loop_index, removed_tiers)
        variant_path = output_dir / f"activity_cliff_{variant}.csv"
        kept.to_csv(variant_path, index=False)
        variants.append(
            {
                "variant_id": variant,
                "loop_index": loop_index,
                "removed_tiers": removed_tiers,
                "removed_count": removed_count,
                "removed_row_indices": [int(idx) for idx in annotated.index[mask].tolist()],
                "remaining_rows": int(len(kept)),
                "filtered_training_csv": str(variant_path),
            }
        )
    return variants, warnings


def prepare_activity_cliff_context(
    *,
    train_csv: str,
    output_dir: str,
    smiles_column: str = "smiles",
    target_column: str,
    activity_cliff_index: str = DEFAULT_ACTIVITY_CLIFF_INDEX,
    activity_cliff_feedback: bool = False,
    activity_cliff_feedback_loops: int = 0,
    activity_cliff_similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    activity_cliff_top_k_neighbors: int = DEFAULT_TOP_K_NEIGHBORS,
    activity_cliff_flag_threshold: float = DEFAULT_FLAG_THRESHOLD,
    min_training_rows: int = MIN_VARIANT_TRAINING_ROWS,
) -> Dict[str, Any]:
    registry = default_registry()
    config = parse_activity_cliff_config(
        activity_cliff_index=activity_cliff_index,
        activity_cliff_feedback=activity_cliff_feedback,
        activity_cliff_feedback_loops=activity_cliff_feedback_loops,
        activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
        activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
        activity_cliff_flag_threshold=activity_cliff_flag_threshold,
        registry=registry,
    )
    with S3.open(train_csv, "r") as fh:
        dataset = pd.read_csv(fh)
    index = registry.get(config.index_name)
    annotated = index.compute(
        dataset=dataset,
        smiles_column=smiles_column,
        target_column=target_column,
        config=config,
    )
    flagged = (
        pd.to_numeric(annotated["activity_cliff_score_norm"], errors="coerce") >= config.flag_threshold
    ) & (annotated["activity_cliff_neighbor_count"].astype(int) >= 1)
    annotated["activity_cliff_flag"] = flagged.astype(bool)
    annotated["activity_cliff_priority_tier"] = _assign_tiers(
        pd.to_numeric(annotated["activity_cliff_score_norm"], errors="coerce").fillna(0.0),
        annotated["activity_cliff_flag"],
    )
    annotated["activity_cliff_reason_codes"] = annotated.apply(_reason_codes, axis=1)
    annotated["activity_cliff_filter_justification"] = annotated.apply(_justification, axis=1)

    ac_dir = Path(output_dir).expanduser().resolve() / "activity_cliffs"
    ac_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = ac_dir / "activity_cliff_annotated_training.csv"
    annotated.to_csv(annotated_path, index=False)
    clean_training_path = ac_dir / "activity_cliff_training_clean.csv"
    strip_activity_cliff_columns(annotated).to_csv(clean_training_path, index=False)
    plot_artifacts = _write_plots(
        annotated,
        ac_dir,
        flag_threshold=config.flag_threshold,
    )
    variants, warnings = _write_variants(
        annotated=annotated,
        output_dir=ac_dir,
        loops=config.feedback_loops,
        min_training_rows=min_training_rows,
    )
    priority_counts = _priority_counts(annotated["activity_cliff_priority_tier"])
    summary = {
        "enabled": True,
        "mode": config.mode,
        "index_name": config.index_name,
        "available_indexes": registry.list_indexes(),
        "source_training_csv": train_csv,
        "clean_training_csv": str(clean_training_path),
        "annotated_training_csv": str(annotated_path),
        "target_column": target_column,
        "smiles_column": smiles_column,
        "ranked_molecule_count": int(len(annotated)),
        "flagged_count": int(annotated["activity_cliff_flag"].sum()),
        "priority_counts": priority_counts,
        "selection_policy": {
            "default_index": DEFAULT_ACTIVITY_CLIFF_INDEX,
            "explicit_index_required_for_non_default": True,
        },
        "index_parameters": {
            "similarity_metric": "tanimoto",
            "fingerprint": "morgan",
            "fingerprint_radius": 2,
            "fingerprint_bits": 2048,
            "similarity_threshold": config.similarity_threshold,
            "top_k_neighbors": config.top_k_neighbors,
            "flag_threshold": config.flag_threshold,
            "normalization": "pair_score_p95",
        },
        "tiering_policy": {
            "source_column": "activity_cliff_score_norm",
            "flag_rule": "score_norm >= flag_threshold and neighbor_count >= 1",
            "high": "top 20 percent of flagged compounds by normalized score",
            "medium": "next 20 percent of flagged compounds by normalized score",
            "low": "remaining flagged compounds",
            "none": "not flagged",
            "oof_residuals_used": False,
        },
        "annotation_columns": [
            column for column in annotated.columns if str(column).startswith("activity_cliff_")
        ],
        "feature_exclusion_prefixes": [ACTIVITY_CLIFF_ANNOTATION_PREFIX],
        "feedback_loops_requested": config.feedback_loops,
        "variants": variants,
        "recommended_variant": None,
        "evaluation_policy": "fixed_holdout_required_for_loop_comparison",
        "plot_artifacts": plot_artifacts,
        "warnings": warnings,
    }
    summary_path = ac_dir / "activity_cliff_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
