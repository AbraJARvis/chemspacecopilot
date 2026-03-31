from pathlib import Path

import pytest

from cs_copilot.tools.prediction.backend import (
    InvalidPredictionInputError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from cs_copilot.tools.prediction.chemprop_backend import ChempropBackend


def test_chemprop_backend_describe_environment_shape():
    backend = ChempropBackend()

    env = backend.describe_environment()

    assert env["backend_name"] == "chemprop"
    assert "available" in env
    assert "cli_path" in env
    assert "package_version" in env


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
