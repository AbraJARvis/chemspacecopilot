from __future__ import annotations

import json

import pandas as pd
import pytest

from cs_copilot.tools.prediction.chemprop_toolkit import _activity_cliff_variant_token
from cs_copilot.tools.prediction.activity_cliffs import (
    ActivityCliffConfig,
    build_activity_cliff_comparison_metrics,
    build_random_oof_splits,
    choose_recommended_variant,
    compute_activity_cliff_annotation,
    merge_activity_cliff_args,
    parse_activity_cliff_config,
    write_activity_cliff_artifacts,
)
from cs_copilot.tools.prediction.qsar_plots import build_activity_cliff_feedback_plots


def test_merge_and_parse_activity_cliff_args_round_trip():
    merged = merge_activity_cliff_args(
        extra_args={"epochs": 10},
        activity_cliff_feedback=True,
        activity_cliff_feedback_loops=2,
        activity_cliff_step_percentile=7.5,
        activity_cliff_similarity_threshold=0.8,
        activity_cliff_k_neighbors=6,
        activity_cliff_oof_folds=4,
    )

    config = parse_activity_cliff_config(merged)

    assert config == ActivityCliffConfig(
        enabled=True,
        loops=2,
        step_percentile=7.5,
        similarity_threshold=0.8,
        k_neighbors=6,
        oof_folds=4,
    )
    assert merged["epochs"] == 10


def test_build_random_oof_splits_covers_all_rows_once():
    folds = build_random_oof_splits(n_rows=20, n_folds=5, random_state=42)

    assert len(folds) == 5
    seen = []
    for fold in folds:
        assert set(fold) == {"train", "val", "test"}
        assert set(fold["train"]).isdisjoint(fold["val"])
        assert set(fold["train"]).isdisjoint(fold["test"])
        assert set(fold["val"]).isdisjoint(fold["test"])
        seen.extend(fold["test"])
    assert sorted(seen) == list(range(20))


def test_compute_activity_cliff_annotation_normalizes_scores():
    dataset = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "c1ccccc1", "CCCl"],
            "Y": [1.0, 1.2, 3.8, 2.0, 1.1],
        }
    )
    oof_predictions = pd.Series([1.1, 1.0, 2.0, 1.9, 1.0], dtype=float)

    annotated = compute_activity_cliff_annotation(
        dataset=dataset,
        smiles_column="smiles",
        target_column="Y",
        oof_predictions=oof_predictions,
        similarity_threshold=0.0,
        k_neighbors=3,
    )

    for column in (
        "activity_cliff_score",
        "activity_cliff_residual_norm",
        "activity_cliff_suspicion_score",
        "activity_cliff_neighbor_count",
    ):
        assert column in annotated.columns
    assert annotated["activity_cliff_score"].between(0.0, 1.0).all()
    assert annotated["activity_cliff_residual_norm"].between(0.0, 1.0).all()
    assert annotated["activity_cliff_suspicion_score"].between(0.0, 1.0).all()


def test_write_activity_cliff_artifacts_generates_cumulative_thresholds(tmp_path):
    annotated = pd.DataFrame(
        {
            "smiles": [f"CC{i}" for i in range(20)],
            "Y": [float(i) for i in range(20)],
            "activity_cliff_score": [0.9 - i * 0.01 for i in range(20)],
            "activity_cliff_residual_norm": [0.8 - i * 0.01 for i in range(20)],
            "activity_cliff_suspicion_score": [0.7 - i * 0.01 for i in range(20)],
            "activity_cliff_neighbor_count": [3] * 20,
        }
    )

    summary = write_activity_cliff_artifacts(
        annotated_df=annotated,
        target_column="Y",
        output_dir=str(tmp_path),
        loops=2,
        step_percentile=5.0,
        min_training_rows=10,
    )

    assert len(summary["variants"]) == 2
    assert summary["variants"][0]["variant_id"] == "filtered_top_5"
    assert summary["variants"][1]["variant_id"] == "filtered_top_10"
    assert summary["variants"][0]["removed_count"] == 1
    assert summary["variants"][1]["removed_count"] == 2
    assert (tmp_path / "activity_cliff_annotated_training.csv").exists()
    assert (tmp_path / "activity_cliff_summary.json").exists()


def test_choose_recommended_variant_prefers_higher_r2_then_lower_rmse():
    variants = [
        {
            "variant_id": "baseline_top_0",
            "training_result": {"metrics": {"test": {"r2": 0.7, "rmse": 0.5}}},
        },
        {
            "variant_id": "filtered_top_5",
            "training_result": {"metrics": {"test": {"r2": 0.7, "rmse": 0.4}}},
        },
        {
            "variant_id": "filtered_top_10",
            "training_result": {"metrics": {"test": {"r2": 0.8, "rmse": 0.6}}},
        },
    ]

    selected = choose_recommended_variant(variants)

    assert selected["variant_id"] == "filtered_top_10"
    comparison = build_activity_cliff_comparison_metrics(variants)
    assert comparison[0]["variant_id"] == "baseline_top_0"


def test_parse_activity_cliff_config_rejects_invalid_loops():
    with pytest.raises(ValueError):
        parse_activity_cliff_config(
            {
                "activity_cliff_feedback": True,
                "activity_cliff_feedback_loops": 0,
            }
        )


def test_activity_cliff_variant_token_from_summary_payload():
    assert _activity_cliff_variant_token({}) is None
    assert _activity_cliff_variant_token(
        {"activity_cliff_feedback": {"enabled": True, "recommended_variant": "baseline_top_0"}}
    ) == "ac_top_0"
    assert _activity_cliff_variant_token(
        {"activity_cliff_feedback": {"enabled": True, "recommended_variant": "filtered_top_10"}}
    ) == "ac_top_10"


def test_build_activity_cliff_feedback_plots_generates_variant_specific_artifacts(tmp_path):
    train_df = pd.DataFrame(
        {
            "smiles": [f"CC{i}" for i in range(20)],
            "Y": [float(i) / 10.0 + 4.0 for i in range(20)],
        }
    )
    train_csv = tmp_path / "train.csv"
    train_df.to_csv(train_csv, index=False)

    annotated_csv = tmp_path / "annotated.csv"
    annotated_df = train_df.copy()
    annotated_df["activity_cliff_score"] = [0.8] * len(annotated_df)
    annotated_df["activity_cliff_residual_norm"] = [0.4] * len(annotated_df)
    annotated_df["activity_cliff_suspicion_score"] = [0.3] * len(annotated_df)
    annotated_df["activity_cliff_neighbor_count"] = [2] * len(annotated_df)
    annotated_df.to_csv(annotated_csv, index=False)

    split_payload = [{"train": list(range(10)), "val": [10, 11], "test": [12, 13, 14, 15]}]
    variants = []
    for variant_id, removed_count in (("baseline_top_0", 0), ("filtered_top_5", 1)):
        variant_dir = tmp_path / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        splits_path = variant_dir / "splits.json"
        splits_path.write_text(json.dumps(split_payload))
        preds_path = variant_dir / "test_predictions.csv"
        pd.DataFrame({"Y": [4.8, 4.9, 5.0, 5.1]}).to_csv(preds_path, index=False)
        variant_train_csv = tmp_path / f"{variant_id}.csv"
        train_df.iloc[removed_count:].reset_index(drop=True).to_csv(variant_train_csv, index=False)
        variants.append(
            {
                "variant_id": variant_id,
                "removed_percent": float(removed_count * 5),
                "removed_count": removed_count,
                "filtered_training_csv": str(variant_train_csv),
                "training_result": {
                    "train_csv": str(variant_train_csv),
                    "test_predictions_path": str(preds_path),
                    "splits_path": str(splits_path),
                    "split_results": [
                        {
                            "strategy_label": "random",
                            "strategy": "random",
                            "strategy_family": "random",
                            "test_predictions_path": str(preds_path),
                            "splits_path": str(splits_path),
                        }
                    ],
                },
            }
        )

    generated = build_activity_cliff_feedback_plots(
        train_csv=str(train_csv),
        target_column="Y",
        annotated_training_csv=str(annotated_csv),
        variants=variants,
        output_dir=str(tmp_path / "plots"),
    )

    assert "target_distribution__input_dataset" in generated
    assert "target_distribution__baseline_top_0" in generated
    assert "target_distribution__filtered_top_5" in generated
    assert "parity_plot_random_rmse__baseline_top_0" in generated
    assert "parity_plot_random_rmse__filtered_top_5" in generated
    assert "residuals_plot_random__baseline_top_0" in generated
    assert "residuals_plot_random__filtered_top_5" in generated
    assert "error_coverage_curve" in generated
