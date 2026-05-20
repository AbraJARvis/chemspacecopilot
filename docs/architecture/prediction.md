# Prediction Architecture

## Goal

QSAR prediction and training are backend-neutral at the agent boundary. Agents
talk to public toolkits that own one responsibility each; backend-specific
toolkits and backend adapters sit behind those public facades.

## Layering

```text
Agents
  |
  |-- DatasetCurationToolkit
  |-- QSARTrainingToolkit
  |-- ModelRegistryToolkit
  |-- PredictionInferenceToolkit
  |-- BenchmarkToolkit
  |-- EnsembleToolkit
  |-- QSARReportingToolkit
        |
        |-- training_orchestration.py
        |-- backend_factory.py
        |-- MolecularFeatureToolkit
        |-- ActivityCliffToolkit
        |
        |-- ChempropToolkit  -> ChempropBackend
        |-- LightGBMToolkit  -> LightGBMBackend
        |-- TabICLToolkit    -> TabICLBackend
        |-- EnsembleToolkit  -> EnsembleBackend
```

`ChempropToolkit`, `LightGBMToolkit`, and `TabICLToolkit` are backend-internal
training toolkits. They are not the registry, inference, or agent-facing
training facade.

## Public Toolkits

- `QSARTrainingToolkit`: single entry point for training workflows, including
  `prepare_training_dataset`, `train_qsar_model`, `train_chemprop_model`,
  `train_lightgbm_model`, and `train_tabicl_model`.
- `ModelRegistryToolkit`: catalog and session registry operations, including
  model registration, persistence, model summaries, catalog recommendations,
  and backend capability descriptions.
- `PredictionInferenceToolkit`: batch and direct inference for registered or
  catalog models.
- `BenchmarkToolkit`: explicit benchmark campaigns only. It compares candidates
  by delegating actual training to `QSARTrainingToolkit` and persistence to
  `ModelRegistryToolkit`.
- `EnsembleToolkit`: ensemble creation, summary, and evaluation workflows.

## Backend Construction

`backend_factory.py` centralizes default backend creation. Registry and
inference facades receive the same backend mapping so catalog records,
registered session models, and prediction calls agree on backend identity.

## Session State Contract

The dedicated prediction state lives under:

```python
session_state["prediction_models"] = {
    "catalog_recommendations": {
        "selected_model": {...},
        "alternatives": [...],
        "selection_summary": "...",
    },
    "registered": {
        "<model_id>": {
            "model_id": "...",
            "backend_name": "lightgbm",
            "model_path": "...",
            "status": "validated",
            "known_metrics": {...},
            "task": {
                "task_type": "regression",
                "smiles_columns": ["smiles"],
                "target_columns": ["pEC50"],
            },
            "tags": {...},
        }
    },
    "last_prediction": {...},
    "prediction_history": [...],
    "training_runs": [...],
}
```

## Persistent Model Catalog

The persistent catalog lives in:

```text
src/cs_copilot/tools/prediction/model_catalog.json
```

Each model entry can include:

- runtime identity: `model_id`, `display_name`, `backend_name`, `model_path`
- governance: `version`, `status`, `owner`, `source`
- scientific fit: `domain_summary`, `recommended_for`, `not_recommended_for`
- quality signals: `known_metrics`, `training_data_summary`
- operational hints: `inference_profile`, `selection_hints`
- user-facing caveats: `strengths`, `limitations`

## Canonical Data Contracts

### Input

- Preferred molecule column: `smiles`
- Optional identifier column: `compound_id`
- Optional split column for training flows: `split`
- Optional task targets: one or many endpoint columns

### Output

- Input columns preserved whenever possible
- Backend-specific prediction columns plus a canonical `prediction` when
  available
- Metadata tracked separately in session state:
  - `model_id`
  - `backend_name`
  - `task_type`
  - `preds_path`
  - `return_uncertainty`

## Non-Goals

- No registry or inference routing through `ChempropToolkit`.
- No implicit benchmark launch for ordinary `standard_qsar` or `robust_qsar`
  training requests.
- No hidden training workflow guessed from user text without explicit dataset,
  target, task, and protocol metadata.
