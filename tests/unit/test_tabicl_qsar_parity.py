from pathlib import Path

import pandas as pd
import pytest

from cs_copilot.tools.io.pointer_pandas_tools import PointerPandasTools
from cs_copilot.tools.prediction.qsar_training_policy import (
    resolve_training_profile,
    resolve_validation_protocol,
)
from cs_copilot.tools.prediction.tabular_splitters import build_tabular_split_payload


def _sample_tabular_df() -> pd.DataFrame:
    smiles = [
        "CCO",
        "CCCO",
        "c1ccccc1",
        "c1ccncc1",
        "CC(=O)O",
        "CCN(CC)CC",
        "CCOC(=O)C",
        "CC(C)O",
        "CC(C)N",
        "CCCC",
        "CCS",
        "CCCl",
    ]
    return pd.DataFrame(
        {
            "smiles": smiles,
            "Y": [float(idx) / 10.0 for idx in range(len(smiles))],
            "desc_a": list(range(len(smiles))),
            "desc_b": [value * 2 for value in range(len(smiles))],
            "desc_c": [(-1) ** idx * (idx + 1) for idx in range(len(smiles))],
        }
    )


def test_tabicl_protocol_defaults_follow_training_profile():
    assert resolve_validation_protocol(
        requested_protocol=None,
        training_profile="local_light",
    )["protocol"] == "fast_local"
    assert resolve_validation_protocol(
        requested_protocol=None,
        training_profile="local_standard",
    )["protocol"] == "standard_qsar"
    assert resolve_validation_protocol(
        requested_protocol=None,
        training_profile="heavy_validation",
    )["protocol"] == "robust_qsar"


def test_robust_qsar_protocol_matches_chemprop_contract():
    payload = resolve_validation_protocol(
        requested_protocol="robust_qsar",
        training_profile="heavy_validation",
    )

    assert payload["protocol"] == "robust_qsar"
    assert [item["label"] for item in payload["split_runs"]] == [
        "random_seed_42",
        "random_seed_123",
        "random_seed_314",
        "scaffold",
    ]
    assert [item["backend_split_type"] for item in payload["split_runs"]] == [
        "random",
        "random",
        "random",
        "scaffold_balanced",
    ]


def test_training_profile_resolution_keeps_shared_names():
    local = resolve_training_profile(
        {
            "cpu_count": 8,
            "memory_gb_total": 8.0,
            "gpu_available": False,
            "execution_env": "docker_local",
        }
    )
    heavy = resolve_training_profile(
        {
            "cpu_count": 48,
            "memory_gb_total": 31.0,
            "gpu_available": True,
            "execution_env": "apptainer_local",
        }
    )

    assert local["profile"] == "local_light"
    assert heavy["profile"] == "heavy_validation"


def test_tabicl_light_profiles_keep_n_jobs_compatible_with_predict():
    from cs_copilot.tools.prediction.tabicl_toolkit import TabICLToolkit

    toolkit = TabICLToolkit()

    light = toolkit._apply_training_profile({"training_profile": "local_light"})
    standard = toolkit._apply_training_profile({"training_profile": "local_standard"})

    assert int(light["extra_args"]["n_jobs"]) >= 1
    assert int(standard["extra_args"]["n_jobs"]) >= 1


def test_random_split_payload_is_deterministic():
    df = _sample_tabular_df()

    first = build_tabular_split_payload(
        df=df,
        split_type="random",
        split_sizes=[0.8, 0.1, 0.1],
        random_state=42,
        smiles_column="smiles",
        feature_columns=["desc_a", "desc_b", "desc_c"],
    )
    second = build_tabular_split_payload(
        df=df,
        split_type="random",
        split_sizes=[0.8, 0.1, 0.1],
        random_state=42,
        smiles_column="smiles",
        feature_columns=["desc_a", "desc_b", "desc_c"],
    )

    assert first == second
    split = first[0]
    assert sorted(split["train"] + split["val"] + split["test"]) == list(range(len(df)))
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["val"]).isdisjoint(split["test"])


def test_scaffold_and_kmeans_splits_are_deterministic_and_complete():
    df = _sample_tabular_df()

    for split_type in ("scaffold_balanced", "kmeans"):
        payload = build_tabular_split_payload(
            df=df,
            split_type=split_type,
            split_sizes=[0.8, 0.1, 0.1],
            random_state=42,
            smiles_column="smiles",
            feature_columns=["desc_a", "desc_b", "desc_c"],
        )
        repeat = build_tabular_split_payload(
            df=df,
            split_type=split_type,
            split_sizes=[0.8, 0.1, 0.1],
            random_state=42,
            smiles_column="smiles",
            feature_columns=["desc_a", "desc_b", "desc_c"],
        )

        assert payload == repeat
        split = payload[0]
        assert sorted(split["train"] + split["val"] + split["test"]) == list(range(len(df)))
        assert split["train"]
        assert split["val"]
        assert split["test"]


def test_create_pandas_dataframe_rejects_json_for_read_csv(tmp_path):
    toolkit = PointerPandasTools()
    json_path = tmp_path / "training_summary.json"
    json_path.write_text('{"metrics": {"r2": 0.5}}')

    with pytest.raises(ValueError, match="JSON artifacts should not be loaded"):
        toolkit.create_pandas_dataframe(
            dataframe_name="summary_df",
            create_using_function="read_csv",
            function_parameters={"path_or_buf": str(json_path)},
        )


def test_pointer_pandas_to_csv_creates_parent_directories(tmp_path):
    toolkit = PointerPandasTools()
    toolkit.dataframes["tabular_ds"] = _sample_tabular_df()
    target = tmp_path / "nested" / "outputs" / "tabular_qsar_dataset.csv"

    toolkit.run_dataframe_operation(
        dataframe_name="tabular_ds",
        operation="to_csv",
        operation_parameters={"path_or_buf": str(target), "index": False},
    )

    assert target.exists()
