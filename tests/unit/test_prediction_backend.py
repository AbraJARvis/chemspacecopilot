import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cs_copilot.tools.prediction.backend import (
    InvalidPredictionInputError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from cs_copilot.tools.prediction.chemprop_backend import ChempropBackend
from cs_copilot.tools.prediction.chemprop_toolkit import ChempropToolkit
from cs_copilot.tools.prediction.backend_capabilities import (
    backend_requires_feature_preparation,
    backend_supports_component_orchestration,
    describe_backend_capabilities,
    get_backend_capabilities,
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


def test_describe_backends_includes_official_capabilities(monkeypatch):
    import cs_copilot.tools.prediction.chemprop_toolkit as toolkit_module

    catalog = SimpleNamespace(refresh_from_internal_store=lambda persist=True: None)
    monkeypatch.setattr(toolkit_module.PredictionModelCatalog, "load", lambda: catalog)

    toolkit = ChempropToolkit(include_prediction_summary_export=False)

    descriptions = toolkit.describe_backends()

    assert descriptions["chemprop"]["capabilities"]["backend_name"] == "chemprop"
    assert descriptions["lightgbm"]["capabilities"]["requires_feature_preparation"] is True
    assert descriptions["ensemble"]["capabilities"]["supports_component_orchestration"] is True


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
