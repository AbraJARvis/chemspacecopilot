from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cs_copilot.tools.prediction import catalog as catalog_module
from cs_copilot.tools.prediction.benchmark_toolkit import BenchmarkToolkit
from cs_copilot.tools.prediction.chemprop_toolkit import ChempropToolkit
from cs_copilot.tools.prediction.lightgbm_toolkit import LightGBMToolkit
from cs_copilot.tools.prediction.tabicl_toolkit import TabICLToolkit


def _fake_agent() -> SimpleNamespace:
    return SimpleNamespace(
        session_state={
            "prediction_models": {
                "registered": {},
                "last_prediction": {},
                "prediction_history": [],
                "catalog_recommendations": {},
                "training_runs": [],
                "active_training_run": None,
            }
        }
    )


def _write_fake_training_outputs(
    root: Path,
    *,
    model_name: str,
    model_suffix: str,
) -> tuple[Path, Path, Path, Path]:
    model_dir = root / "model_0"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name
    model_path.write_text("fake-model")

    test_predictions_path = model_dir / "test_predictions.csv"
    pd.DataFrame({"y_true": [1.0, 2.0], "y_pred": [1.1, 1.9]}).to_csv(test_predictions_path, index=False)

    config_path = root / "config.toml"
    config_path.write_text('backend_name = "fake"\n')

    splits_path = root / "splits.json"
    splits_path.write_text(json.dumps([{"train": [0], "val": [1], "test": [0, 1]}]))

    return model_path, test_predictions_path, config_path, splits_path


def _fake_validation_assessment(protocol: str, random_r2: float, hardest_name: str, hardest_r2: float) -> dict:
    aggregated = {
        "random": {"r2_mean": random_r2, "r2_std": 0.01 if protocol == "robust_qsar" else 0.0},
        hardest_name: {"r2_mean": hardest_r2},
    }
    if protocol == "robust_qsar":
        aggregated["random"]["num_runs"] = 3
    return {
        "hardest_split": hardest_name,
        "delta_vs_random": {hardest_name: {"r2": hardest_r2 - random_r2}},
        "aggregated_split_metrics": aggregated,
        "governance": {"recommended_status": "workflow_demo"},
    }


def test_resolve_benchmark_protocol_contract():
    toolkit = BenchmarkToolkit()
    assert toolkit._resolve_benchmark_protocol("benchmark_fast_local") == "fast_local"
    assert toolkit._resolve_benchmark_protocol("benchmark_standard_qsar") == "standard_qsar"
    assert toolkit._resolve_benchmark_protocol("benchmark_robust_qsar") == "robust_qsar"
    assert toolkit._resolve_benchmark_protocol("benchmark_challenging_qsar") == "challenging_qsar"
    assert toolkit._resolve_benchmark_protocol("fast_local") == "fast_local"
    assert toolkit._resolve_benchmark_protocol("standard_qsar") == "standard_qsar"
    assert toolkit._resolve_benchmark_protocol("robust_qsar") == "robust_qsar"
    assert toolkit._resolve_benchmark_protocol("challenging_qsar") == "challenging_qsar"


def test_benchmark_rejects_activity_cliff_feedback_v1():
    toolkit = BenchmarkToolkit()
    with pytest.raises(ValueError, match="Activity-cliff feedback loops are not supported"):
        toolkit.benchmark_qsar_models(
            train_csv="dataset.csv",
            task_type="regression",
            target_columns=["Y"],
            activity_cliff_feedback=True,
            agent=_fake_agent(),
        )


def test_expand_candidates_includes_heavy_tabicl_all():
    toolkit = BenchmarkToolkit()
    candidates = toolkit._expand_candidates(
        task_type="regression",
        target_columns=["Y"],
        requested_backends=["tabicl"],
        include_candidate_variants=True,
        requested_tabicl_variants=None,
        training_profile="heavy_validation",
    )
    candidate_ids = [item["candidate_id"] for item in candidates]
    assert "tabicl_morgan_only" in candidate_ids
    assert "tabicl_rdkit_basic_only" in candidate_ids
    assert "tabicl_morgan_rdkit_basic" in candidate_ids
    assert "tabicl_morgan_rdkit_all" in candidate_ids


def test_expand_candidates_includes_heavy_lightgbm_all():
    toolkit = BenchmarkToolkit()
    candidates = toolkit._expand_candidates(
        task_type="regression",
        target_columns=["Y"],
        requested_backends=["lightgbm"],
        include_candidate_variants=True,
        requested_tabicl_variants=None,
        training_profile="heavy_validation",
    )
    candidate_ids = [item["candidate_id"] for item in candidates]
    assert "lightgbm_morgan_only" in candidate_ids
    assert "lightgbm_rdkit_basic_only" in candidate_ids
    assert "lightgbm_morgan_rdkit_basic" in candidate_ids
    assert "lightgbm_morgan_rdkit_all" in candidate_ids


def test_rank_summary_rows_prefers_hardest_split_then_gap():
    toolkit = BenchmarkToolkit()
    ranked = toolkit._rank_summary_rows(
        [
            {
                "candidate_id": "a",
                "hardest_split_r2": 0.50,
                "delta_r2": -0.10,
                "random_r2": 0.70,
                "train_time_s": 100.0,
            },
            {
                "candidate_id": "b",
                "hardest_split_r2": 0.55,
                "delta_r2": -0.20,
                "random_r2": 0.90,
                "train_time_s": 50.0,
            },
        ]
    )
    assert ranked[0]["candidate_id"] == "b"


def test_benchmark_standard_qsar_persists_all_candidates(tmp_path, monkeypatch):
    catalog_path = tmp_path / "model_catalog.json"
    internal_root = tmp_path / "internal"
    train_csv = tmp_path / "dataset_curated.csv"
    pd.DataFrame({"smiles": ["CCO", "CCC"], "Y": [1.0, 2.0]}).to_csv(train_csv, index=False)

    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)

    import cs_copilot.tools.prediction.chemprop_toolkit as chemprop_module

    monkeypatch.setattr(chemprop_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    monkeypatch.setattr(
        chemprop_module.PredictionModelCatalog,
        "load",
        classmethod(lambda cls: cls(source_path=catalog_path)),
    )

    chemprop_toolkit = ChempropToolkit()
    lightgbm_toolkit = LightGBMToolkit()
    tabicl_toolkit = TabICLToolkit()
    monkeypatch.setattr(chemprop_toolkit.backend, "is_available", lambda: True)
    monkeypatch.setattr(lightgbm_toolkit.backend, "is_available", lambda: True)
    monkeypatch.setattr(tabicl_toolkit.backend, "is_available", lambda: True)

    toolkit = BenchmarkToolkit(
        chemprop_toolkit=chemprop_toolkit,
        lightgbm_toolkit=lightgbm_toolkit,
        tabicl_toolkit=tabicl_toolkit,
    )

    monkeypatch.setattr(
        toolkit,
        "_resolve_compute_profile",
        lambda: {
            "compute_environment": {
                "execution_env": "apptainer_local",
                "cpu_count": 48,
                "gpu_available": True,
                "suggested_profile": "heavy_validation",
            },
            "training_profile": "heavy_validation",
            "profile_reason": "test",
        },
    )

    def fake_prepare_tabular_candidate_dataset(*, candidate, train_csv, smiles_column, target_columns, candidate_dir):
        output = candidate_dir / f"{candidate['candidate_id']}_tabular.csv"
        pd.DataFrame(
            {
                "smiles": ["CCO", "CCC"],
                "Y": [1.0, 2.0],
                "feature_a": [0.1, 0.2],
                "feature_b": [0.3, 0.4],
            }
        ).to_csv(output, index=False)
        return {"train_csv": str(output), "representation_name": candidate["representation_name"]}

    monkeypatch.setattr(toolkit, "_prepare_tabular_candidate_dataset", fake_prepare_tabular_candidate_dataset)

    def fake_chemprop_train_model(self, train_csv, task_type, output_dir, smiles_columns=None, target_columns=None, extra_args=None, agent=None):
        root = Path(output_dir)
        model_path, _, _, _ = _write_fake_training_outputs(root, model_name="best.pt", model_suffix=".pt")
        result = {
            "best_model_path": str(model_path),
            "summary_path": str(root / "cs_copilot_training_summary.json"),
            "validation_assessment": _fake_validation_assessment("standard_qsar", 0.60, "scaffold", 0.52),
            "split_results": [
                {"strategy_label": "random", "metrics": {"test": {"r2": 0.60, "rmse": 0.70, "mae": 0.50, "mse": 0.49}}},
                {"strategy_label": "scaffold", "metrics": {"test": {"r2": 0.52, "rmse": 0.75, "mae": 0.55, "mse": 0.56}}},
            ],
            "training_durations": {"total_duration_seconds": 12.0},
            "metrics": {"test": {"r2": 0.60}},
            "applicability_domain": {},
            "train_csv": train_csv,
            "trained_at": "2026-04-27T12:00:00+02:00",
        }
        (root / "cs_copilot_training_summary.json").write_text(json.dumps(result))
        agent.session_state["prediction_models"]["training_runs"].append(
            {
                "train_csv": train_csv,
                "output_dir": str(root.resolve()),
                "task_type": task_type,
                "smiles_columns": smiles_columns or ["smiles"],
                "target_columns": target_columns or [],
                "validation_protocol": "standard_qsar",
            }
        )
        return result

    def fake_tabicl_train_model(self, train_csv, task_type, output_dir, target_columns, validation_protocol=None, extra_args=None, agent=None):
        root = Path(output_dir)
        model_path, test_predictions_path, config_path, splits_path = _write_fake_training_outputs(root, model_name="best.pkl", model_suffix=".pkl")
        (root / "test_predictions.csv").write_text(test_predictions_path.read_text())
        result = {
            "model_path": str(model_path),
            "summary_path": str(root / "cs_copilot_training_summary.json"),
            "config_path": str(config_path),
            "splits_path": str(splits_path),
            "test_predictions_path": str(root / "test_predictions.csv"),
            "validation_assessment": _fake_validation_assessment("standard_qsar", 0.58, "scaffold", 0.50),
            "split_results": [
                {"strategy_label": "random", "metrics": {"test": {"r2": 0.58, "rmse": 0.71, "mae": 0.51, "mse": 0.50}}},
                {"strategy_label": "scaffold", "metrics": {"test": {"r2": 0.50, "rmse": 0.77, "mae": 0.56, "mse": 0.59}}},
            ],
            "training_durations": {"total_duration_seconds": 10.0},
            "metrics": {"test": {"r2": 0.58}},
            "feature_columns": ["feature_a", "feature_b"],
            "applicability_domain": {},
            "train_csv": train_csv,
            "trained_at": "2026-04-27T12:00:00+02:00",
        }
        (root / "cs_copilot_training_summary.json").write_text(json.dumps(result))
        agent.session_state["prediction_models"]["training_runs"].append(
            {
                "train_csv": train_csv,
                "output_dir": str(root.resolve()),
                "task_type": task_type,
                "smiles_columns": ["smiles"],
                "target_columns": target_columns,
                "validation_protocol": validation_protocol or "standard_qsar",
                "training_profile": "heavy_validation",
            }
        )
        return result

    def fake_lightgbm_train_model(
        self,
        train_csv,
        task_type,
        output_dir,
        target_columns,
        feature_columns=None,
        categorical_feature_columns=None,
        validation_protocol=None,
        extra_args=None,
        agent=None,
    ):
        root = Path(output_dir)
        model_path, test_predictions_path, config_path, splits_path = _write_fake_training_outputs(
            root,
            model_name="best.pkl",
            model_suffix=".pkl",
        )
        (root / "test_predictions.csv").write_text(test_predictions_path.read_text())
        result = {
            "model_path": str(model_path),
            "summary_path": str(root / "cs_copilot_training_summary.json"),
            "config_path": str(config_path),
            "splits_path": str(splits_path),
            "test_predictions_path": str(root / "test_predictions.csv"),
            "validation_assessment": _fake_validation_assessment("standard_qsar", 0.57, "scaffold", 0.49),
            "split_results": [
                {"strategy_label": "random", "metrics": {"test": {"r2": 0.57, "rmse": 0.72, "mae": 0.52, "mse": 0.52}}},
                {"strategy_label": "scaffold", "metrics": {"test": {"r2": 0.49, "rmse": 0.78, "mae": 0.57, "mse": 0.61}}},
            ],
            "training_durations": {"total_duration_seconds": 9.0},
            "metrics": {"test": {"r2": 0.57}},
            "feature_columns": ["feature_a", "feature_b"],
            "categorical_feature_columns": [],
            "applicability_domain": {},
            "train_csv": train_csv,
            "trained_at": "2026-04-27T12:00:00+02:00",
        }
        (root / "cs_copilot_training_summary.json").write_text(json.dumps(result))
        agent.session_state["prediction_models"]["training_runs"].append(
            {
                "train_csv": train_csv,
                "output_dir": str(root.resolve()),
                "task_type": task_type,
                "smiles_columns": ["smiles"],
                "target_columns": target_columns,
                "validation_protocol": validation_protocol or "standard_qsar",
                "training_profile": "heavy_validation",
            }
        )
        return result

    monkeypatch.setattr(ChempropToolkit, "train_model", fake_chemprop_train_model)
    monkeypatch.setattr(LightGBMToolkit, "train_lightgbm_model", fake_lightgbm_train_model)
    monkeypatch.setattr(TabICLToolkit, "train_tabicl_model", fake_tabicl_train_model)

    agent = _fake_agent()
    output_dir = tmp_path / "benchmark_output"
    result = toolkit.benchmark_qsar_models(
        train_csv=str(train_csv),
        task_type="regression",
        target_columns=["Y"],
        smiles_column="smiles",
        benchmark_mode="benchmark_standard_qsar",
        output_dir=str(output_dir),
        agent=agent,
    )

    assert Path(result["summary_path"]).exists()
    assert Path(result["leaderboard_path"]).exists()
    assert Path(result["report_path"]).exists()

    candidate_ids = {item["candidate_id"] for item in result["persisted_model_mapping"]}
    assert "chemprop_default" in candidate_ids
    assert "lightgbm_morgan_only" in candidate_ids
    assert "lightgbm_rdkit_basic_only" in candidate_ids
    assert "lightgbm_morgan_rdkit_basic" in candidate_ids
    assert "lightgbm_morgan_rdkit_all" in candidate_ids
    assert "tabicl_morgan_only" in candidate_ids
    assert "tabicl_rdkit_basic_only" in candidate_ids
    assert "tabicl_morgan_rdkit_basic" in candidate_ids
    assert "tabicl_morgan_rdkit_all" in candidate_ids

    for item in result["persisted_model_mapping"]:
        model_root = Path(item["internal_model_root"])
        assert model_root.exists()
        assert (model_root / "metadata.json").exists()
        assert (model_root / "model").exists()
        assert (model_root / "artifacts").exists()

    leaderboard = pd.read_csv(result["leaderboard_path"])
    assert set(["candidate_id", "model_id", "backend", "representation", "hardest_split_r2"]).issubset(
        set(leaderboard.columns)
    )
    chemprop_row = leaderboard.loc[leaderboard["candidate_id"] == "chemprop_default"].iloc[0]
    assert chemprop_row["representation"] == "molecular_graph"
