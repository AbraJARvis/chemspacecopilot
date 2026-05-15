from __future__ import annotations

import json

import pandas as pd
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from cs_copilot.tools.activity_cliffs.service import (
    ActivityCliffIndexRegistry,
    SALIIndex,
    prepare_activity_cliff_context,
)
from cs_copilot.tools.activity_cliffs import attach_activity_cliff_variant_training
from cs_copilot.tools.prediction.backend import InvalidPredictionInputError, PredictionTaskSpec
from cs_copilot.tools.prediction.chemprop_toolkit import ChempropToolkit
from cs_copilot.tools.prediction.lightgbm_backend import LightGBMBackend
from cs_copilot.tools.prediction.tabicl_backend import TabICLBackend


def test_activity_cliff_framework_writes_generic_and_sali_columns(tmp_path):
    dataset = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CCBr"],
            "Y": [1.0, 1.2, 3.8, 1.1, 2.0, 1.0],
        }
    )
    train_csv = tmp_path / "train.csv"
    dataset.to_csv(train_csv, index=False)

    context = prepare_activity_cliff_context(
        train_csv=str(train_csv),
        output_dir=str(tmp_path / "run"),
        target_column="Y",
        activity_cliff_similarity_threshold=0.0,
        activity_cliff_top_k_neighbors=3,
    )
    annotated = pd.read_csv(context["annotated_training_csv"])

    assert context["enabled"] is True
    assert context["index_name"] == "sali"
    assert context["mode"] == "standard"
    assert context["summary_path"].endswith("activity_cliff_summary.json")
    summary = json.loads((tmp_path / "run" / "activity_cliffs" / "activity_cliff_summary.json").read_text())
    assert summary["index_parameters"]["fingerprint"] == "morgan_count"
    assert summary["index_parameters"]["fingerprint_dimensions"] == 2048
    assert summary["index_parameters"]["fingerprint_bits"] is None
    assert summary["index_parameters"]["similarity_metric"] == "count_tanimoto"
    assert summary["similarity_diagnostics"]["exact_similarity_policy"] == "reported_not_silently_corrected"
    assert "activity_cliff_score_raw" in annotated.columns
    assert "activity_cliff_score_norm" in annotated.columns
    assert "activity_cliff_sali_raw" in annotated.columns
    assert "activity_cliff_priority_tier" in annotated.columns
    assert set(context["priority_counts"]) == {"none", "low", "medium", "high"}


def test_morgan_count_similarity_reduces_known_binary_collisions():
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    pairs = [
        ("NC1CCCCC1", "NC1CCCCCC1"),
        ("C1CCCNCCC1", "C1CCCNCC1"),
    ]

    for left, right in pairs:
        left_mol = Chem.MolFromSmiles(left)
        right_mol = Chem.MolFromSmiles(right)
        bit_similarity = DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(left_mol),
            generator.GetFingerprint(right_mol),
        )
        count_similarity = DataStructs.TanimotoSimilarity(
            generator.GetCountFingerprint(left_mol),
            generator.GetCountFingerprint(right_mol),
        )

        assert bit_similarity == pytest.approx(1.0)
        assert count_similarity < bit_similarity
        assert count_similarity < 1.0


def test_activity_cliff_framework_rejects_unknown_index():
    registry = ActivityCliffIndexRegistry()
    registry.register(SALIIndex())

    with pytest.raises(ValueError, match="Available indexes"):
        registry.get("not_real")


def test_activity_cliff_framework_rejects_more_than_three_loops(tmp_path):
    dataset = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCI"],
            "Y": [1.0, 1.2, 3.8, 1.1, 1.0, 2.2],
        }
    )
    train_csv = tmp_path / "train.csv"
    dataset.to_csv(train_csv, index=False)

    with pytest.raises(ValueError, match="cannot exceed 3"):
        prepare_activity_cliff_context(
            train_csv=str(train_csv),
            output_dir=str(tmp_path / "run"),
            target_column="Y",
            activity_cliff_feedback=True,
            activity_cliff_feedback_loops=4,
        )


def test_lightgbm_auto_features_exclude_activity_cliff_columns():
    backend = LightGBMBackend()
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "CCC"],
            "Y": [1.0, 2.0],
            "fp_0001": [0.1, 0.2],
            "activity_cliff_score_norm": [0.9, 0.1],
        }
    )
    task = PredictionTaskSpec(task_type="regression", smiles_columns=["smiles"], target_columns=["Y"])

    features, _ = backend._select_feature_columns(df, task, {})

    assert features == ["fp_0001"]


def test_lightgbm_explicit_activity_cliff_feature_is_rejected():
    backend = LightGBMBackend()
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "CCC"],
            "Y": [1.0, 2.0],
            "activity_cliff_score_norm": [0.9, 0.1],
        }
    )
    task = PredictionTaskSpec(task_type="regression", smiles_columns=["smiles"], target_columns=["Y"])

    with pytest.raises(InvalidPredictionInputError):
        backend._select_feature_columns(df, task, {"feature_columns": ["activity_cliff_score_norm"]})


def test_tabicl_auto_features_exclude_activity_cliff_columns():
    backend = TabICLBackend()
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "CCC"],
            "Y": [1.0, 2.0],
            "fp_0001": [0.1, 0.2],
            "activity_cliff_score_norm": [0.9, 0.1],
        }
    )

    features = backend._select_feature_columns(df, ["Y"], {})

    assert features == ["fp_0001"]


def test_activity_cliff_feedback_training_selects_best_fixed_holdout_variant():
    activity_cliffs = {
        "enabled": True,
        "mode": "with_feedback_loops",
        "index_name": "sali",
        "feedback_loops_requested": 1,
        "index_parameters": {
            "fingerprint": "morgan_count",
            "fingerprint_radius": 2,
            "fingerprint_dimensions": 2048,
            "similarity_metric": "count_tanimoto",
            "similarity_threshold": 0.7,
            "top_k_neighbors": 10,
            "flag_threshold": 0.35,
            "normalization": "pair_score_p95",
        },
        "priority_counts": {"none": 8, "low": 1, "medium": 0, "high": 1},
        "flagged_count": 2,
        "ranked_molecule_count": 10,
        "variants": [
            {"variant_id": "baseline_loop_0", "loop_index": 0, "removed_tiers": [], "removed_count": 0},
            {
                "variant_id": "filtered_loop_1_drop_high",
                "loop_index": 1,
                "removed_tiers": ["high"],
                "removed_count": 1,
                "remaining_rows": 9,
            },
        ],
    }
    baseline = [
        {
            "strategy_label": "scaffold",
            "strategy_family": "scaffold",
            "metrics": {"test": {"r2": 0.5, "rmse": 0.7, "mae": 0.5, "mse": 0.49, "n": 2}},
            "source_train_count": 6,
            "effective_train_count": 6,
            "validation_count": 2,
            "test_count": 2,
        }
    ]
    variant_results = {
        "filtered_loop_1_drop_high": [
            {
                "strategy_label": "scaffold",
                "strategy_family": "scaffold",
                "metrics": {"test": {"r2": 0.6, "rmse": 0.6, "mae": 0.4, "mse": 0.36, "n": 2}},
                "source_train_count": 6,
                "effective_train_count": 5,
                "validation_count": 2,
                "test_count": 2,
                "removed_from_train_count": 1,
                "requested_exclusion_count": 1,
            }
        ]
    }

    updated = attach_activity_cliff_variant_training(
        activity_cliffs=activity_cliffs,
        baseline_split_results=baseline,
        variant_split_results=variant_results,
        backend_name="unit",
    )

    assert updated["feedback_evaluation_status"] == "evaluated"
    assert updated["recommended_variant"] == "filtered_loop_1_drop_high"
    assert updated["variant_comparison_table"][1]["removed_from_train_count"] == 1
    assert "boucles de retroaction ont ete evaluees" in updated["reporting_handoff"]["canonical_section_markdown"]


def test_chemprop_fixed_split_csv_removes_train_only_and_keeps_holdouts(tmp_path):
    dataset = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCI"],
            "Y": [1.0, 1.2, 1.4, 2.0, 2.2, 2.4],
        }
    )
    train_csv = tmp_path / "train.csv"
    dataset.to_csv(train_csv, index=False)
    toolkit = object.__new__(ChempropToolkit)

    fixed = toolkit._write_chemprop_fixed_split_csv(
        train_csv=str(train_csv),
        baseline_split_payload=[{"train": [0, 1, 2], "val": [3], "test": [4, 5]}],
        excluded_train_indices=[1, 4],
        output_dir=tmp_path / "variant",
    )
    fixed_df = pd.read_csv(fixed["train_csv"])

    assert fixed["excluded_from_train"] == [1]
    assert fixed["source_train_count"] == 3
    assert fixed["split_payload"][0]["train"] == [0, 1]
    assert fixed_df["cs_copilot_source_row_index"].tolist() == [0, 2, 3, 4, 5]
    assert fixed_df["cs_copilot_split"].tolist() == ["train", "train", "val", "test", "test"]
