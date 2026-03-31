# Prediction Architecture

## Goal

Add predictive modeling to ChemSpace Copilot without coupling the agent layer to a
single ML library.

The initial backend target is **Chemprop**, but the architecture is designed so
that future backends can reuse the same agent and data contracts.

## Layering

```text
Agent (property_predictor)
        |
Toolkit (ChempropToolkit)
        |
Backend contract (PredictionBackend)
        |
Concrete backend (ChempropBackend)
```

## Why this shape

- The agent should reason about tasks and datasets, not Chemprop CLI flags.
- The toolkit should manage session state, file normalization, and output paths.
- The backend should encapsulate Chemprop-specific command construction.
- This keeps later support for other predictors realistic.

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
            "backend_name": "chemprop",
            "model_path": "...",
            "status": "validated",
            "strengths": [...],
            "limitations": [...],
            "known_metrics": {...},
            "task": {
                "task_type": "regression",
                "smiles_columns": ["smiles"],
                "target_columns": ["solubility"],
                "reaction_columns": [],
                "uncertainty_method": None,
                "calibration_method": None,
            },
            "tags": {...},
        }
    },
    "last_prediction": {...},
    "training_runs": [...],
}
```

## Persistent Model Catalog

The persistent catalog lives in:

```text
src/cs_copilot/tools/prediction/model_catalog.json
```

It acts as the source of truth for model selection metadata.  Each model entry
can include:

- runtime identity: `model_id`, `display_name`, `backend_name`, `model_path`
- governance: `version`, `status`, `owner`, `source`
- scientific fit: `domain_summary`, `recommended_for`, `not_recommended_for`
- quality signals: `known_metrics`, `training_data_summary`
- operational hints: `inference_profile`, `selection_hints`
- user-facing caveats: `strengths`, `limitations`

This allows the property predictor to justify why it chose a model, not just
which model it ran.

## Canonical Data Contracts

### Input

- Preferred molecule column: `smiles`
- Optional identifier column: `compound_id`
- Optional split column for training flows: `split`
- Optional task targets: one or many endpoint columns

### Output

- Input columns preserved whenever possible
- Prediction columns written by backend output
- Metadata tracked separately in session state:
  - `model_id`
  - `backend_name`
  - `task_type`
  - `preds_path`
  - `return_uncertainty`

## Near-Term Extension Points

- Add a `PredictionBackendRegistry` if multiple backends become active
- Add uncertainty-aware ranking and calibration workflows
- Add fingerprint extraction for GTM and chemoinformatics integration
- Add active learning loops on top of `training_runs`

## Deliberate Non-Goals of the First Iteration

- No hard dependency on Chemprop for the whole project
- No hidden training workflow guessed from user text without explicit metadata
