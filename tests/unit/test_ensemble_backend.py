from pathlib import Path

import pandas as pd
import pytest

from cs_copilot.tools.prediction.backend import (
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from cs_copilot.tools.prediction.ensemble_backend import EnsembleBackend
from cs_copilot.tools.prediction.ensemble_toolkit import EnsembleToolkit


class FakeRegressionBackend(PredictionBackend):
    backend_name = "fake"

    def is_available(self) -> bool:
        return True

    def describe_environment(self):
        return {"backend_name": self.backend_name, "available": True}

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path)
        if not path.exists():
            raise InvalidPredictionInputError(f"missing fake model: {model_path}")
        return path

    def predict_from_csv(self, input_csv, model_record, preds_path, *, return_uncertainty=False, extra_args=None):
        offset = float(model_record.tags.get("offset", 0.0))
        source = pd.read_csv(input_csv)
        pd.DataFrame({"prediction": source["feature"].astype(float) + offset}).to_csv(preds_path, index=False)
        return {"preds_path": preds_path, "rows": len(source)}

    def train_model(self, train_csv, output_dir, task, *, extra_args=None):
        raise NotImplementedError


def _record(tmp_path: Path, model_id: str, *, target: str = "Y", offset: float = 0.0, status: str = "validated"):
    model_path = tmp_path / f"{model_id}.fake"
    model_path.write_text("fake")
    return PredictionModelRecord(
        model_id=model_id,
        backend_name="fake",
        model_path=str(model_path),
        status=status,
        tags={"offset": str(offset)},
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=[target],
        ),
    )


def test_ensemble_backend_predicts_median_and_disagreement(tmp_path):
    backend = EnsembleBackend(backends={"fake": FakeRegressionBackend()})
    toolkit = EnsembleToolkit(backend=backend)
    records = [_record(tmp_path, "a", offset=0.0), _record(tmp_path, "b", offset=2.0)]

    task = toolkit._validate_components(records)
    ensemble_path = tmp_path / "ensemble.json"
    ensemble_path.write_text(
        toolkit_json := __import__("json").dumps(
            {
                "schema_version": 1,
                "ensemble_kind": "consensus_regression",
                "aggregation_strategy": "median",
                "uncertainty_strategy": "component_disagreement_std",
                "task": {
                    "task_type": task.task_type,
                    "smiles_columns": ["smiles"],
                    "target_columns": ["Y"],
                    "reaction_columns": [],
                },
                "components": [{"record": record.as_dict()} for record in records],
            }
        )
        + "\n"
    )
    assert toolkit_json

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"smiles": ["CCO", "CCC"], "feature": [1.0, 3.0]}).to_csv(input_csv, index=False)
    output_csv = tmp_path / "predictions.csv"
    ensemble_record = PredictionModelRecord(
        model_id="ensemble",
        backend_name="ensemble",
        model_path=str(ensemble_path),
        task=task,
    )

    result = backend.predict_from_csv(str(input_csv), ensemble_record, str(output_csv))
    output = pd.read_csv(output_csv)

    assert result["component_count"] == 2
    assert output["prediction"].tolist() == [2.0, 4.0]
    assert output["ensemble_prediction_mean"].tolist() == [2.0, 4.0]
    assert output["ensemble_prediction_std"].tolist() == [1.0, 1.0]
    assert {"prediction_a", "prediction_b"}.issubset(output.columns)


def test_ensemble_component_validation_rejects_mismatched_targets(tmp_path):
    toolkit = EnsembleToolkit(backend=EnsembleBackend(backends={"fake": FakeRegressionBackend()}))
    records = [_record(tmp_path, "a", target="Y"), _record(tmp_path, "b", target="Z")]

    with pytest.raises(InvalidPredictionInputError, match="target column"):
        toolkit._validate_components(records)


def test_ensemble_component_validation_rejects_deprecated_component(tmp_path):
    toolkit = EnsembleToolkit(backend=EnsembleBackend(backends={"fake": FakeRegressionBackend()}))
    records = [_record(tmp_path, "a"), _record(tmp_path, "b", status="deprecated")]

    with pytest.raises(InvalidPredictionInputError, match="Deprecated"):
        toolkit._validate_components(records)


def test_ensemble_governance_downgrades_promoted_status():
    toolkit = EnsembleToolkit(backend=EnsembleBackend(backends={"fake": FakeRegressionBackend()}))

    final_status, note = toolkit._govern_ensemble_status("validated")

    assert final_status == "workflow_demo"
    assert note is not None
    assert "dedicated ensemble-level validation gate" in note
