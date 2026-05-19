import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from cs_copilot.tools.prediction.backend import (
    InvalidPredictionInputError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from cs_copilot.tools.prediction.chemprop_backend import ChempropBackend
from cs_copilot.tools.prediction.chemprop_toolkit import ChempropToolkit
from cs_copilot.tools.prediction.prediction_inference_toolkit import PredictionInferenceToolkit
from cs_copilot.tools.prediction.prediction_registry_toolkit import PredictionRegistryToolkit
from cs_copilot.tools.prediction.backend_capabilities import (
    backend_requires_feature_preparation,
    backend_supports_component_orchestration,
    describe_backend_capabilities,
    get_backend_capabilities,
)
from cs_copilot.tools.prediction.session_state import (
    bundle_artifacts,
    discover_curation_artifacts_near_dataset,
    get_prediction_state,
    latest_curation_artifacts,
    write_active_training_marker,
)
from cs_copilot.tools.prediction.training_orchestration import (
    apply_training_profile,
    collect_training_bundle_files,
    materialize_primary_protocol_artifacts,
    normalize_json_list_argument,
    write_training_summary,
)


def test_backend_capabilities_registry_core_contracts():
    chemprop = get_backend_capabilities("chemprop")
    lightgbm = get_backend_capabilities("lightgbm")
    tabicl = get_backend_capabilities("tabicl")
    ensemble = get_backend_capabilities("ensemble")

    assert chemprop.requires_feature_preparation is False
    assert backend_requires_feature_preparation("lightgbm") is True
    assert tabicl.requires_feature_preparation is True
    assert ensemble.supports_component_orchestration is True
    assert backend_supports_component_orchestration("ensemble") is True
    assert ensemble.supports_uncertainty == "component_disagreement_std"
    assert lightgbm.supports_activity_cliff_feedback_loops is True
    assert chemprop.supports_activity_cliff_feedback_loops is False
    assert chemprop.gpu_support == "runtime_dependent"
    assert lightgbm.gpu_support == "supported_when_available"
    assert ensemble.gpu_support == "not_applicable"
    assert tabicl.catalog_model_filename == "best.pkl"
    assert "tabicl_training_summary.json" in tabicl.training_summary_filenames


def test_backend_capabilities_unknown_backend_is_clear():
    with pytest.raises(KeyError, match="No backend capabilities registered"):
        get_backend_capabilities("unknown_backend")


def test_describe_backend_capabilities_is_serializable():
    payload = describe_backend_capabilities()

    json.dumps(payload)
    assert payload["chemprop"]["backend_name"] == "chemprop"
    assert "morgan_rdkit_all" in payload["lightgbm"]["supported_representations"]
    assert payload["lightgbm"]["gpu_support"] == "supported_when_available"


def test_shared_prediction_state_helpers_initialize_backend_neutral_state(tmp_path):
    agent = SimpleNamespace(session_state={})

    state = get_prediction_state(agent)

    assert state["registered"] == {}
    assert state["prediction_history"] == []
    assert state["training_runs"] == []
    assert state["active_training_run"] is None

    marker = tmp_path / "run" / ".training_in_progress"
    write_active_training_marker(marker, {"status": "running", "backend_name": "fake"})
    assert json.loads(marker.read_text())["backend_name"] == "fake"


def test_shared_curation_artifact_helpers_are_backend_neutral(tmp_path):
    report = tmp_path / "dataset_curation_report.json"
    report.write_text("{}")
    artifacts_dir = tmp_path / "dataset_curation_artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "curation_manifest.json").write_text("{}")
    dataset = tmp_path / "dataset_curated.csv"
    dataset.write_text("smiles,pEC50\nCCO,5.0\n")

    discovered = discover_curation_artifacts_near_dataset(str(dataset))

    assert discovered["artifacts"]["curation_report_json"] == str(report)
    assert discovered["artifacts"]["manifest_json"] == str(artifacts_dir / "curation_manifest.json")

    agent = SimpleNamespace(
        session_state={
            "qsar_curation": {
                "last_result": {
                    "curation_backend_used": "chembl_structure_v1",
                    "curated_dataset_path": str(dataset),
                    "rows_in": 2,
                    "rows_out": 1,
                    "report_path": str(report),
                }
            }
        }
    )
    latest = latest_curation_artifacts(agent)

    assert latest["curation_backend"] == "chembl_structure_v1"
    assert latest["artifacts"]["curation_report_json"] == str(report)


def test_shared_bundle_artifacts_deduplicates_archive_names(tmp_path):
    first = tmp_path / "a" / "same.txt"
    second = tmp_path / "b" / "same.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first")
    second.write_text("second")

    bundle = bundle_artifacts(tmp_path / "bundle.zip", [first, second])

    assert bundle.exists()
    with ZipFile(bundle) as zf:
        names = zf.namelist()
    assert len(names) == 2
    assert len(set(names)) == 2


def test_training_orchestration_normalizes_agent_list_arguments():
    assert normalize_json_list_argument("pEC50", argument_name="target_columns") == ["pEC50"]
    assert normalize_json_list_argument('["smiles"]', argument_name="smiles_columns") == ["smiles"]
    assert normalize_json_list_argument("0.8,0.1,0.1", argument_name="split_sizes", coerce_numbers=True) == [
        0.8,
        0.1,
        0.1,
    ]


def test_training_orchestration_applies_profile_with_backend_specific_limits():
    compute_env = {
        "cpu_count": 48,
        "memory_gb_total": 31.0,
        "gpu_available": True,
        "execution_env": "apptainer_local",
    }

    def defaults(profile: str):
        return {"n_estimators": 1000 if profile == "heavy_validation" else 300}

    def limit(profile: str, args: dict, allow_heavy: bool):
        if profile == "local_light":
            args["n_estimators"] = min(int(args["n_estimators"]), 300)
        return args

    policy = apply_training_profile(
        {"training_profile": "heavy_validation", "n_estimators": 9999},
        defaults_for_profile=defaults,
        limit_profile_args=limit,
        compute_environment=compute_env,
        protected_profiles=("heavy_validation",),
    )

    assert policy["training_profile"] == "heavy_validation"
    assert policy["extra_args"]["n_estimators"] == 9999


def test_training_orchestration_materializes_summary_and_bundle_inputs(tmp_path):
    run_dir = tmp_path / "run"
    model = run_dir / "model_0" / "best.pkl"
    preds = run_dir / "model_0" / "test_predictions.csv"
    config = run_dir / "config.toml"
    splits = run_dir / "splits.json"
    model.parent.mkdir(parents=True)
    for path in (model, preds, config, splits):
        path.write_text(path.name)

    root = tmp_path / "root"
    artifacts = materialize_primary_protocol_artifacts(
        root_output_dir=root,
        primary_run={
            "model_path": str(model),
            "test_predictions_path": str(preds),
            "config_path": str(config),
            "splits_path": str(splits),
        },
        model_filename="best.pkl",
    )
    summary = write_training_summary(root / "cs_copilot_training_summary.json", {"ok": True})
    files = collect_training_bundle_files(
        train_csv=str(config),
        summary_path=summary,
        result={"model_path": artifacts["best_model_path"]},
        split_results=[],
        ad_summary={},
        plot_artifacts={},
    )

    assert Path(artifacts["best_model_path"]).exists()
    assert json.loads(summary.read_text())["ok"] is True
    assert Path(artifacts["best_model_path"]) in files


def test_describe_backends_includes_official_capabilities(monkeypatch):
    import cs_copilot.tools.prediction.chemprop_toolkit as toolkit_module

    catalog = SimpleNamespace(refresh_from_internal_store=lambda persist=True: None)
    monkeypatch.setattr(toolkit_module.PredictionModelCatalog, "load", lambda: catalog)

    toolkit = ChempropToolkit(include_prediction_summary_export=False)

    descriptions = toolkit.describe_backends()

    assert descriptions["chemprop"]["capabilities"]["backend_name"] == "chemprop"
    assert descriptions["lightgbm"]["capabilities"]["requires_feature_preparation"] is True
    assert descriptions["ensemble"]["capabilities"]["supports_component_orchestration"] is True


def test_chemprop_toolkit_uses_backend_neutral_registry_and_inference_facades(monkeypatch):
    import cs_copilot.tools.prediction.chemprop_toolkit as toolkit_module

    catalog = SimpleNamespace(refresh_from_internal_store=lambda persist=True: None)
    monkeypatch.setattr(toolkit_module.PredictionModelCatalog, "load", lambda: catalog)

    toolkit = ChempropToolkit(include_prediction_summary_export=False)

    assert isinstance(toolkit.registry_toolkit, PredictionRegistryToolkit)
    assert isinstance(toolkit.inference_toolkit, PredictionInferenceToolkit)
    assert toolkit.describe_backends() == toolkit.registry_toolkit.describe_backends()


def test_chemprop_backend_describe_environment_shape():
    backend = ChempropBackend()

    env = backend.describe_environment()

    assert env["backend_name"] == "chemprop"
    assert "available" in env
    assert "cli_path" in env
    assert "package_version" in env
    assert env["capabilities"]["backend_name"] == "chemprop"


def test_chemprop_backend_validate_model_path_rejects_missing(tmp_path):
    backend = ChempropBackend()

    missing_path = tmp_path / "missing.ckpt"

    with pytest.raises(InvalidPredictionInputError):
        backend.validate_model_path(str(missing_path))


def test_chemprop_backend_validate_model_path_accepts_ckpt(tmp_path):
    backend = ChempropBackend()
    model_path = tmp_path / "model.ckpt"
    model_path.write_text("placeholder")

    resolved = backend.validate_model_path(str(model_path))

    assert resolved == Path(model_path)


def test_prediction_model_record_as_dict():
    record = PredictionModelRecord(
        model_id="solubility_v1",
        backend_name="chemprop",
        model_path="/tmp/model.ckpt",
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["solubility"],
        ),
    )

    payload = record.as_dict()

    assert payload["model_id"] == "solubility_v1"
    assert payload["task"]["task_type"] == "regression"
    assert payload["task"]["target_columns"] == ["solubility"]
