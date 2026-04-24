#!/usr/bin/env python
# coding: utf-8
"""
Toolkit exposing TabICLv2-backed tabular QSAR workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agno.tools.toolkit import Toolkit

from .backend import PredictionTaskSpec
from .tabicl_backend import (
    DEFAULT_TABICL_CHECKPOINT_DIR,
    DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
    TabICLBackend,
)


class TabICLToolkit(Toolkit):
    """Toolkit exposing a minimal TabICLv2 regression workflow."""

    def __init__(self, backend: Optional[TabICLBackend] = None):
        super().__init__("tabicl_prediction")
        self.backend = backend or TabICLBackend()
        self.register(self.describe_backend)
        self.register(self.describe_environment)
        self.register(self.is_available)
        self.register(self.validate_model_path)
        self.register(self.train_model)
        self.register(self.predict_from_csv)

    def is_available(self) -> bool:
        """Return whether the TabICL backend is available in the current environment."""
        return self.backend.is_available()

    def describe_backend(self) -> Dict[str, Any]:
        """Describe the TabICL backend defaults and current runtime support."""
        description = self.backend.describe_environment()
        description.update(
            {
                "default_task_type": "regression",
                "default_checkpoint_dir": str(DEFAULT_TABICL_CHECKPOINT_DIR),
                "default_checkpoint_version": DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
                "notes": [
                    "V1 supports regression only.",
                    "V1 expects a pre-built tabular dataset with numeric feature columns.",
                    "V1 expects the TabICL checkpoint to be provisioned in the persistent local checkpoint directory.",
                ],
            }
        )
        return description

    def describe_environment(self) -> Dict[str, Any]:
        """Return a lightweight runtime snapshot for TabICL availability."""
        return self.backend.describe_environment()

    def validate_model_path(self, model_path: str) -> Dict[str, Any]:
        """Validate a saved TabICL model artifact path."""
        resolved = self.backend.validate_model_path(model_path)
        return {
            "model_path": str(resolved),
            "exists": resolved.exists(),
            "suffix": resolved.suffix,
        }

    def train_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        target_columns: List[str] | str,
        feature_columns: Optional[List[str] | str] = None,
        validation_protocol: str = "standard_qsar",
        split_type: str = "random",
        split_sizes: Optional[List[float] | str] = None,
        random_state: int = 42,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train a TabICLv2 regressor on a prepared tabular QSAR dataset."""
        if isinstance(target_columns, str):
            parsed = json.loads(target_columns)
            if not isinstance(parsed, list):
                raise ValueError("target_columns must be a list or a JSON-encoded list.")
            target_columns = parsed
        if isinstance(feature_columns, str):
            parsed = json.loads(feature_columns)
            if not isinstance(parsed, list):
                raise ValueError("feature_columns must be a list or a JSON-encoded list.")
            feature_columns = parsed
        if isinstance(split_sizes, str):
            parsed = json.loads(split_sizes)
            if not isinstance(parsed, list):
                raise ValueError("split_sizes must be a list or a JSON-encoded list.")
            split_sizes = parsed

        merged_extra_args = dict(extra_args or {})
        merged_extra_args.setdefault("feature_columns", feature_columns)
        merged_extra_args.setdefault("split_sizes", split_sizes)
        merged_extra_args.setdefault("random_state", random_state)
        merged_extra_args.setdefault("validation_protocol", validation_protocol)
        merged_extra_args.setdefault("split_type", split_type)

        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=["smiles"],
            target_columns=list(target_columns),
        )
        return self.backend.train_model(
            train_csv=train_csv,
            output_dir=str(Path(output_dir).expanduser().resolve()),
            task=task,
            extra_args=merged_extra_args,
        )

    def predict_from_csv(
        self,
        input_csv: str,
        model_path: str,
        preds_path: str,
        target_columns: Optional[List[str] | str] = None,
        feature_columns: Optional[List[str] | str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run TabICL batch prediction from a tabular CSV input file."""
        if isinstance(target_columns, str):
            parsed = json.loads(target_columns)
            if not isinstance(parsed, list):
                raise ValueError("target_columns must be a list or a JSON-encoded list.")
            target_columns = parsed
        if isinstance(feature_columns, str):
            parsed = json.loads(feature_columns)
            if not isinstance(parsed, list):
                raise ValueError("feature_columns must be a list or a JSON-encoded list.")
            feature_columns = parsed

        model_record_extra = dict(extra_args or {})
        if feature_columns:
            model_record_extra.setdefault("feature_columns", feature_columns)

        from .backend import PredictionModelRecord

        model_record = PredictionModelRecord(
            model_id=Path(model_path).stem,
            backend_name=self.backend.backend_name,
            model_path=model_path,
            task=PredictionTaskSpec(
                task_type="regression",
                smiles_columns=["smiles"],
                target_columns=list(target_columns or []),
            ),
            inference_profile={"feature_columns": list(feature_columns or [])},
        )
        return self.backend.predict_from_csv(
            input_csv=input_csv,
            model_record=model_record,
            preds_path=preds_path,
            extra_args=model_record_extra,
        )
