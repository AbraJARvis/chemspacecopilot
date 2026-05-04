#!/usr/bin/env python
# coding: utf-8
"""
Lightweight QSAR training plots persisted as model artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PLOT_DPI = 300


def _safe_metric_columns(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if target_column not in df.columns:
        return pd.DataFrame()
    predictions = pd.to_numeric(df[target_column], errors="coerce")
    cleaned = pd.DataFrame({"y_pred": predictions}).dropna()
    return cleaned


def _load_split_truth_and_predictions(
    *,
    dataset: pd.DataFrame,
    split_payload: List[Dict[str, Any]],
    predictions_path: Path,
    target_column: str,
) -> Optional[pd.DataFrame]:
    if not split_payload or "test" not in split_payload[0] or not predictions_path.exists():
        return None

    test_indices = split_payload[0].get("test") or []
    if not test_indices:
        return None

    actual = dataset.iloc[test_indices].reset_index(drop=True)
    predictions = pd.read_csv(predictions_path)
    if target_column not in actual.columns or target_column not in predictions.columns:
        return None
    if len(actual) != len(predictions):
        return None

    y_true = pd.to_numeric(actual[target_column], errors="coerce")
    y_pred = pd.to_numeric(predictions[target_column], errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    if not valid.any():
        return None

    frame = pd.DataFrame(
        {
            "y_true": y_true[valid].astype(float).reset_index(drop=True),
            "y_pred": y_pred[valid].astype(float).reset_index(drop=True),
        }
    )
    frame["residual"] = frame["y_true"] - frame["y_pred"]
    return frame


def _plot_target_distribution(
    values: pd.Series,
    output_path: Path,
    *,
    title: str = "Distribution de la cible apres curation",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=30, color="#224f75", edgecolor="white", alpha=0.9)
    mean_value = float(values.mean())
    median_value = float(values.median())
    ax.axvline(mean_value, color="#258d9a", linestyle="--", linewidth=1.5, label=f"Moyenne = {mean_value:.2f}")
    ax.axvline(median_value, color="#d98b36", linestyle="-.", linewidth=1.5, label=f"Mediane = {median_value:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Valeur cible Y (unitless_log_scale)")
    ax.set_ylabel("Nombre de composes")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.95,
        f"n = {len(values)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    ax.text(
        0.98,
        0.06,
        f"PNG exporte a {PLOT_DPI} dpi",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_split_distribution(
    dataset: pd.DataFrame,
    split_payload: List[Dict[str, Any]],
    target_column: str,
    output_path: Path,
) -> None:
    split_map = split_payload[0] if split_payload else {}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"train": "#224f75", "val": "#258d9a", "test": "#d98b36"}
    for split_name in ("train", "val", "test"):
        indices = split_map.get(split_name) or []
        if not indices:
            continue
        values = pd.to_numeric(dataset.iloc[indices][target_column], errors="coerce").dropna()
        if values.empty:
            continue
        ax.hist(
            values,
            bins=30,
            alpha=0.45,
            density=True,
            label=f"{split_name} (n={len(values)})",
            color=colors[split_name],
            edgecolor="white",
        )
    ax.set_title("Distribution normalisee de la cible par split")
    ax.set_xlabel("Valeur cible Y (unitless_log_scale)")
    ax.set_ylabel("Densite")
    ax.legend(title="Splits")
    ax.grid(alpha=0.2, linestyle="--")
    ax.text(
        0.98,
        0.06,
        f"Histogrammes normalises, PNG exporte a {PLOT_DPI} dpi",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_parity(
    frame: pd.DataFrame,
    label: str,
    output_path: Path,
    *,
    band_metric: str = "mae",
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    mae = float(frame["residual"].abs().mean())
    rmse = float((frame["residual"].pow(2).mean()) ** 0.5)
    centered = frame["y_true"] - float(frame["y_true"].mean())
    ss_tot = float((centered.pow(2)).sum())
    ss_res = float((frame["residual"].pow(2)).sum())
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else None
    point_alpha = 0.45 if len(frame) > 250 else 0.65
    ax.scatter(frame["y_true"], frame["y_pred"], s=18, alpha=point_alpha, color="#224f75")
    min_val = min(frame["y_true"].min(), frame["y_pred"].min())
    max_val = max(frame["y_true"].max(), frame["y_pred"].max())
    x_band = pd.Series([min_val, max_val], dtype=float)
    band_value = mae if band_metric == "mae" else rmse
    band_label = "MAE" if band_metric == "mae" else "RMSE"
    if band_value > 0:
        ax.fill_between(
            x_band,
            x_band - (2.0 * band_value),
            x_band + (2.0 * band_value),
            color="#258d9a",
            alpha=0.10,
            zorder=0,
            label=f"Bande ±2x {band_label}",
        )
        ax.fill_between(
            x_band,
            x_band - band_value,
            x_band + band_value,
            color="#258d9a",
            alpha=0.18,
            zorder=1,
            label=f"Bande ±1x {band_label}",
        )
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="#b54a4a",
        linewidth=1.2,
        label="Droite ideale y = x",
    )
    within_1x = float((frame["residual"].abs() <= band_value).mean() * 100.0) if band_value > 0 else 0.0
    within_2x = float((frame["residual"].abs() <= (2.0 * band_value)).mean() * 100.0) if band_value > 0 else 0.0
    ax.set_title(f"Observed vs Predicted ({label}, bandes {band_label})")
    ax.set_xlabel("Valeur observee Y")
    ax.set_ylabel("Valeur predite Y")
    ax.set_aspect("equal", adjustable="box")
    if band_value > 0:
        metrics_text = [f"MAE = {mae:.3f}", f"RMSE = {rmse:.3f}"]
        if r2 is not None:
            metrics_text.insert(0, f"R² = {r2:.3f}")
        ax.text(
            0.02,
            0.98,
            "\n".join(metrics_text + [f"{within_1x:.1f}% dans 1x {band_label}", f"{within_2x:.1f}% dans 2x {band_label}"]),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
        )
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.text(
        0.98,
        0.02,
        f"n = {len(frame)} | PNG exporte a {PLOT_DPI} dpi",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals(frame: pd.DataFrame, label: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mae = float(frame["residual"].abs().mean())
    point_alpha = 0.45 if len(frame) > 250 else 0.7
    ax.scatter(frame["y_pred"], frame["residual"], s=18, alpha=point_alpha, color="#258d9a")
    ax.axhline(0.0, linestyle="--", color="#b54a4a", linewidth=1.2)
    if mae > 0:
        ax.axhspan(-mae, mae, color="#258d9a", alpha=0.10, zorder=0)
    ax.set_title(f"Residuals vs Predicted ({label})")
    ax.set_xlabel("Valeur predite Y")
    ax.set_ylabel("Residuel (Y observe - Y predit)")
    ax.grid(alpha=0.2, linestyle="--")
    ax.text(
        0.98,
        0.95,
        f"MAE = {mae:.3f}\nn = {len(frame)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    ax.text(
        0.98,
        0.04,
        f"Bande coloree : ±1x MAE | PNG exporte a {PLOT_DPI} dpi",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _normalize_strategy_label(item: Dict[str, Any]) -> Optional[str]:
    strategy_family = str(item.get("strategy_family") or "").lower().strip()
    strategy = str(item.get("strategy") or "").lower().strip()
    strategy_label = str(item.get("strategy_label") or "").lower().strip()
    combined = " ".join([strategy_family, strategy, strategy_label])
    if "scaffold" in combined:
        return "scaffold"
    if "cluster" in combined or "kmeans" in combined:
        return "cluster_kmeans"
    if "random" in combined:
        return "random"
    return None


def _plot_seed_performance(split_results: List[Dict[str, Any]], output_path: Path) -> Optional[str]:
    rows: List[Dict[str, Any]] = []
    for item in split_results:
        strategy_label = str(item.get("strategy_label") or "")
        if "random_seed_" not in strategy_label:
            continue
        metrics = (item.get("metrics") or {}).get("test") or {}
        if not metrics:
            continue
        try:
            rows.append(
                {
                    "seed": strategy_label.replace("random_seed_", ""),
                    "r2": float(metrics["r2"]),
                    "rmse": float(metrics["rmse"]),
                    "mae": float(metrics["mae"]),
                }
            )
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    metric_specs = [
        ("r2", "R²", "#224f75"),
        ("rmse", "RMSE", "#258d9a"),
        ("mae", "MAE", "#d98b36"),
    ]
    for ax, (column, label, color) in zip(axes, metric_specs):
        ax.bar(df["seed"], df[column], color=color, alpha=0.85)
        ax.set_title(label)
        ax.set_xlabel("Seed")
        ax.grid(alpha=0.2, linestyle="--", axis="y")
    fig.suptitle("Comparaison des performances par seed")
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_activity_cliff_histogram(
    annotated_frame: pd.DataFrame,
    variants: List[Dict[str, Any]],
    output_path: Path,
) -> Optional[str]:
    if "activity_cliff_local_sali_norm" not in annotated_frame.columns:
        return None
    values = pd.to_numeric(
        annotated_frame["activity_cliff_local_sali_norm"], errors="coerce"
    ).dropna()
    if values.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(values, bins=30, color="#224f75", edgecolor="white", alpha=0.9)
    ordered = values.sort_values(ascending=False).reset_index(drop=True)
    colors = ["#b54a4a", "#d98b36", "#258d9a", "#6b8e23", "#6a5acd"]
    for idx, variant in enumerate(variants):
        removed_count = int(variant.get("removed_count") or 0)
        if removed_count <= 0 or removed_count > len(ordered):
            continue
        threshold = float(ordered.iloc[removed_count - 1])
        label = str(variant.get("variant_id") or f"variant_{idx}")
        ax.axvline(
            threshold,
            color=colors[idx % len(colors)],
            linestyle="--",
            linewidth=1.4,
            label=f"Seuil {label} = {threshold:.3f}",
        )
    ax.set_title("Distribution du score SALI local normalise")
    ax.set_xlabel("Activity cliff local SALI norm")
    ax.set_ylabel("Nombre de composes")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_error_coverage_curve(
    variant_frames: List[Dict[str, Any]],
    output_path: Path,
) -> Optional[str]:
    usable = [item for item in variant_frames if item.get("frame") is not None]
    if not usable:
        return None

    all_abs_residuals: List[float] = []
    for item in usable:
        frame = item["frame"]
        all_abs_residuals.extend(frame["residual"].abs().astype(float).tolist())
    if not all_abs_residuals:
        return None

    max_error = max(all_abs_residuals)
    if max_error <= 0:
        max_error = 1.0
    thresholds = np.linspace(0.0, max_error, 200)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.subplots_adjust(right=0.72)
    palette = ["#224f75", "#258d9a", "#d98b36", "#b54a4a", "#6a5acd", "#6b8e23"]
    coverage_summary_lines: List[str] = []
    for idx, item in enumerate(usable):
        frame = item["frame"]
        residuals = frame["residual"].abs().astype(float).to_numpy()
        coverages = [float(np.mean(residuals <= thr) * 100.0) for thr in thresholds]
        color = palette[idx % len(palette)]
        label = str(item.get("label") or f"variant_{idx}")
        ax.plot(thresholds, coverages, color=color, linewidth=1.8, label=label)
        rmse = float((frame["residual"].pow(2).mean()) ** 0.5)
        cov_1x = float(np.mean(residuals <= rmse) * 100.0)
        cov_2x = float(np.mean(residuals <= (2.0 * rmse)) * 100.0)
        ax.axvline(
            rmse,
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            label=f"RMSE {label}",
        )
        coverage_summary_lines.append(
            f"{label}: 1x RMSE={cov_1x:.1f}% | 2x RMSE={cov_2x:.1f}%"
        )
    ax.set_title("Couverture cumulative en fonction du seuil d'erreur")
    ax.set_xlabel("Seuil d'erreur absolue")
    ax.set_ylabel("Coverage cumulative (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    if coverage_summary_lines:
        fig.text(
            0.745,
            0.88,
            "\n".join(coverage_summary_lines),
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _variant_loop_label(variant_id: str) -> str:
    token = str(variant_id or "").strip()
    if token == "baseline_loop_0":
        return "loop 0"
    if token.startswith("filtered_loop_"):
        try:
            prefix = token.replace("filtered_loop_", "", 1)
            loop_number = int(prefix.split("_", 1)[0])
            return f"loop {loop_number}"
        except Exception:
            return token
    return token


def _build_variant_plot_key(base_key: str, variant_id: str) -> str:
    return f"{base_key}__{variant_id}"


def build_qsar_training_plots(
    *,
    train_csv: str,
    split_results: List[Dict[str, Any]],
    primary_run: Dict[str, Any],
    output_dir: str,
    target_column: str,
) -> Dict[str, str]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(Path(train_csv).expanduser())
    target_values = pd.to_numeric(dataset[target_column], errors="coerce").dropna()
    if target_values.empty:
        return {}

    generated: Dict[str, str] = {}

    target_plot = output_path / "target_distribution.png"
    _plot_target_distribution(target_values, target_plot)
    generated["target_distribution"] = str(target_plot)

    primary_splits_path = Path(primary_run.get("splits_path") or "")
    if primary_splits_path.exists():
        split_payload = json.loads(primary_splits_path.read_text())
        by_split_plot = output_path / "target_distribution_by_split.png"
        _plot_split_distribution(dataset, split_payload, target_column, by_split_plot)
        generated["target_distribution_by_split"] = str(by_split_plot)

    for item in split_results:
        strategy_label = _normalize_strategy_label(item)
        if strategy_label not in {"random", "scaffold", "cluster_kmeans"}:
            continue
        predictions_path = Path(item.get("test_predictions_path") or "")
        splits_path = Path(item.get("splits_path") or "")
        if not predictions_path.exists() or not splits_path.exists():
            continue
        split_payload = json.loads(splits_path.read_text())
        frame = _load_split_truth_and_predictions(
            dataset=dataset,
            split_payload=split_payload,
            predictions_path=predictions_path,
            target_column=target_column,
        )
        if frame is None or frame.empty:
            continue

        parity_path_mae = output_path / f"parity_plot_{strategy_label}.png"
        _plot_parity(frame, strategy_label, parity_path_mae, band_metric="mae")
        generated[f"parity_plot_{strategy_label}"] = str(parity_path_mae)

        parity_path_rmse = output_path / f"parity_plot_{strategy_label}_rmse.png"
        _plot_parity(frame, strategy_label, parity_path_rmse, band_metric="rmse")
        generated[f"parity_plot_{strategy_label}_rmse"] = str(parity_path_rmse)

        residuals_path = output_path / f"residuals_plot_{strategy_label}.png"
        _plot_residuals(frame, strategy_label, residuals_path)
        generated[f"residuals_plot_{strategy_label}"] = str(residuals_path)

    seed_plot_path = output_path / "seed_performance_comparison.png"
    rendered_seed_plot = _plot_seed_performance(split_results, seed_plot_path)
    if rendered_seed_plot:
        generated["seed_performance_comparison"] = rendered_seed_plot

    return generated


def build_activity_cliff_feedback_plots(
    *,
    train_csv: str,
    target_column: str,
    annotated_training_csv: str,
    variants: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, str]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    annotated_frame = pd.read_csv(Path(annotated_training_csv).expanduser())
    generated: Dict[str, str] = {}

    input_dataset = pd.read_csv(Path(train_csv).expanduser())
    input_target_values = pd.to_numeric(input_dataset[target_column], errors="coerce").dropna()
    if not input_target_values.empty:
        input_target_plot = output_path / "target_distribution_input_dataset.png"
        _plot_target_distribution(
            input_target_values,
            input_target_plot,
            title="Distribution de la cible — dataset d'entree",
        )
        generated["target_distribution__input_dataset"] = str(input_target_plot)

    histogram_path = output_path / "activity_cliff_histogram.png"
    rendered_hist = _plot_activity_cliff_histogram(annotated_frame, variants, histogram_path)
    if rendered_hist:
        generated["activity_cliff_histogram"] = rendered_hist

    if "activity_cliff_priority_tier" in annotated_frame.columns:
        tier_counts = (
            annotated_frame["activity_cliff_priority_tier"]
            .fillna("none")
            .astype(str)
            .value_counts()
            .reindex(["none", "low", "medium", "high"], fill_value=0)
        )
        if int(tier_counts.sum()) > 0:
            tier_path = output_path / "activity_cliff_tier_distribution.png"
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
            ax.bar(
                tier_counts.index.tolist(),
                tier_counts.astype(int).tolist(),
                color=["#cccccc", "#6b8e23", "#d98b36", "#b54a4a"],
                alpha=0.9,
            )
            ax.set_title("Distribution des tiers activity cliffs")
            ax.set_xlabel("Tier")
            ax.set_ylabel("Nombre de composes")
            ax.grid(alpha=0.2, linestyle="--", axis="y")
            fig.tight_layout()
            fig.savefig(tier_path, dpi=PLOT_DPI, bbox_inches="tight")
            plt.close(fig)
            generated["activity_cliff_tier_distribution"] = str(tier_path)

    if {
        "activity_cliff_max_similarity",
        "activity_cliff_max_activity_gap",
    }.issubset(annotated_frame.columns):
        scatter_df = annotated_frame[
            ["activity_cliff_max_similarity", "activity_cliff_max_activity_gap"]
        ].apply(pd.to_numeric, errors="coerce").dropna()
        if not scatter_df.empty:
            gap_path = output_path / "activity_gap_vs_similarity.png"
            fig, ax = plt.subplots(figsize=(6.8, 4.6))
            ax.scatter(
                scatter_df["activity_cliff_max_similarity"],
                scatter_df["activity_cliff_max_activity_gap"],
                color="#258d9a",
                alpha=0.45,
                s=28,
            )
            ax.set_title("Activity gap vs similarite locale")
            ax.set_xlabel("Similarite maximale")
            ax.set_ylabel("Ecart d'activite maximal")
            ax.grid(alpha=0.2, linestyle="--")
            fig.tight_layout()
            fig.savefig(gap_path, dpi=PLOT_DPI, bbox_inches="tight")
            plt.close(fig)
            generated["activity_gap_vs_similarity"] = str(gap_path)

    if {
        "activity_cliff_local_sali_norm",
        "activity_cliff_residual_norm",
    }.issubset(annotated_frame.columns):
        scatter_df = annotated_frame[
            ["activity_cliff_local_sali_norm", "activity_cliff_residual_norm"]
        ].apply(pd.to_numeric, errors="coerce").dropna()
        if not scatter_df.empty:
            sali_path = output_path / "residual_vs_sali.png"
            fig, ax = plt.subplots(figsize=(6.8, 4.6))
            ax.scatter(
                scatter_df["activity_cliff_local_sali_norm"],
                scatter_df["activity_cliff_residual_norm"],
                color="#224f75",
                alpha=0.45,
                s=28,
            )
            ax.set_title("Residuel OOF vs SALI local")
            ax.set_xlabel("Local SALI norm")
            ax.set_ylabel("Residual norm")
            ax.grid(alpha=0.2, linestyle="--")
            fig.tight_layout()
            fig.savefig(sali_path, dpi=PLOT_DPI, bbox_inches="tight")
            plt.close(fig)
            generated["residual_vs_sali"] = str(sali_path)

    variant_frames: List[Dict[str, Any]] = []
    for item in variants:
        training_result = item.get("training_result") or {}
        variant_train_csv = training_result.get("train_csv") or train_csv
        predictions_path = Path(training_result.get("test_predictions_path") or "")
        splits_path = Path(training_result.get("splits_path") or "")
        if not predictions_path.exists() or not splits_path.exists():
            continue
        split_payload = json.loads(splits_path.read_text())
        dataset = pd.read_csv(Path(variant_train_csv).expanduser())
        variant_target_values = pd.to_numeric(dataset[target_column], errors="coerce").dropna()
        variant_id = str(item.get("variant_id") or "variant")
        loop_label = _variant_loop_label(variant_id)
        if not variant_target_values.empty:
            target_plot = output_path / f"target_distribution_{variant_id}.png"
            _plot_target_distribution(
                variant_target_values,
                target_plot,
                title=f"Distribution de la cible — {variant_id} ({loop_label})",
            )
            generated[_build_variant_plot_key("target_distribution", variant_id)] = str(target_plot)

        split_results = training_result.get("split_results") or []
        for split_result in split_results:
            strategy_label = _normalize_strategy_label(split_result)
            if strategy_label not in {"random", "scaffold", "cluster_kmeans"}:
                continue
            split_predictions_path = Path(split_result.get("test_predictions_path") or "")
            split_splits_path = Path(split_result.get("splits_path") or "")
            if not split_predictions_path.exists() or not split_splits_path.exists():
                continue
            split_payload = json.loads(split_splits_path.read_text())
            split_frame = _load_split_truth_and_predictions(
                dataset=dataset,
                split_payload=split_payload,
                predictions_path=split_predictions_path,
                target_column=target_column,
            )
            if split_frame is None or split_frame.empty:
                continue

            title_label = f"{strategy_label}, {variant_id}, {loop_label}"
            parity_path_rmse = output_path / f"parity_plot_{strategy_label}_rmse_{variant_id}.png"
            _plot_parity(split_frame, title_label, parity_path_rmse, band_metric="rmse")
            generated[_build_variant_plot_key(f"parity_plot_{strategy_label}_rmse", variant_id)] = str(
                parity_path_rmse
            )

            residuals_path = output_path / f"residuals_plot_{strategy_label}_{variant_id}.png"
            _plot_residuals(split_frame, title_label, residuals_path)
            generated[_build_variant_plot_key(f"residuals_plot_{strategy_label}", variant_id)] = str(
                residuals_path
            )

        frame = _load_split_truth_and_predictions(
            dataset=dataset,
            split_payload=split_payload,
            predictions_path=predictions_path,
            target_column=target_column,
        )
        if frame is None or frame.empty:
            continue
        variant_frames.append({"label": variant_id, "frame": frame})

    curve_path = output_path / "error_coverage_curve.png"
    rendered_curve = _plot_error_coverage_curve(variant_frames, curve_path)
    if rendered_curve:
        generated["error_coverage_curve"] = rendered_curve

    return generated
