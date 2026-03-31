#!/usr/bin/env python
# coding: utf-8
"""
Chemprop backend adapter.

This adapter intentionally keeps the rest of the codebase insulated from the
specific Chemprop CLI/API.  The initial implementation uses the Chemprop CLI
because it provides a stable operational path for training and prediction over
CSV files, which fits the project's S3/local file abstraction well.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .backend import (
    BackendNotAvailableError,
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionExecutionError,
    PredictionModelRecord,
    PredictionTaskSpec,
)

logger = logging.getLogger(__name__)


class ChempropBackend(PredictionBackend):
    """Prediction backend built around Chemprop v2 CLI commands."""

    backend_name = "chemprop"
    MODEL_EXTENSIONS = (".ckpt", ".pt")

    def _find_cli_path(self) -> Optional[str]:
        cli_path = shutil.which("chemprop")
        if cli_path:
            return cli_path

        candidate_roots = []

        venv_root = os.getenv("VIRTUAL_ENV")
        if venv_root:
            candidate_roots.append(Path(venv_root))

        candidate_roots.append(Path(sys.prefix))
        candidate_roots.append(Path("/app/.venv"))

        seen = set()
        for root in candidate_roots:
            if root in seen:
                continue
            seen.add(root)
            venv_cli = root / "bin" / "chemprop"
            if venv_cli.exists():
                return str(venv_cli)

        return None

    def _package_version(self) -> Optional[str]:
        try:
            return importlib.metadata.version("chemprop")
        except importlib.metadata.PackageNotFoundError:
            return None

    def is_available(self) -> bool:
        cli_path = self._find_cli_path()
        if cli_path:
            return True

        if self._package_version() is None:
            return False

        return importlib.util.find_spec("chemprop") is not None

    def describe_environment(self) -> Dict[str, Any]:
        version = self._package_version()

        return {
            "backend_name": self.backend_name,
            "available": self.is_available(),
            "cli_path": self._find_cli_path(),
            "package_version": version,
        }

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise InvalidPredictionInputError(f"Model path does not exist: {model_path}")

        if path.is_file() and path.suffix not in self.MODEL_EXTENSIONS:
            raise InvalidPredictionInputError(
                f"Chemprop model artifact must end with one of {self.MODEL_EXTENSIONS}: {model_path}"
            )

        return path

    def _ensure_available(self) -> None:
        if not self.is_available():
            env = self.describe_environment()
            raise BackendNotAvailableError(
                "Chemprop backend is not available. "
                "Install the optional dependency and ensure the `chemprop` CLI is on PATH. "
                f"Environment snapshot: {env}"
            )

    def _run_cli(self, args: list[str]) -> subprocess.CompletedProcess:
        self._ensure_available()
        cli_path = self._find_cli_path()
        if cli_path:
            args = [cli_path, *args[1:]]
        logger.info("Running Chemprop command: %s", " ".join(args))
        try:
            return subprocess.run(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            stdout = (exc.stdout or "").strip()
            stderr = (exc.stderr or "").strip()
            details = stderr or stdout or "Chemprop CLI exited with a non-zero status."
            raise PredictionExecutionError(
                "Chemprop execution failed. "
                f"Command: {' '.join(args)} | Details: {details}"
            ) from exc

    def predict_from_csv(
        self,
        input_csv: str,
        model_record: PredictionModelRecord,
        preds_path: str,
        *,
        return_uncertainty: bool = False,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        input_path = Path(input_csv).expanduser()
        if not input_path.exists():
            raise InvalidPredictionInputError(f"Input CSV does not exist: {input_csv}")

        model_path = self.validate_model_path(model_record.model_path)
        output_path = Path(preds_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        args = [
            "chemprop",
            "predict",
            "--test-path",
            str(input_path),
            "--model-paths",
            str(model_path),
            "--preds-path",
            str(output_path),
        ]

        if model_record.task.smiles_columns:
            args.extend(["--smiles-columns", *model_record.task.smiles_columns])

        if model_record.task.reaction_columns:
            args.extend(["--reaction-columns", *model_record.task.reaction_columns])

        if return_uncertainty and model_record.task.uncertainty_method:
            args.extend(["--uncertainty-method", model_record.task.uncertainty_method])
            if model_record.task.calibration_method:
                args.extend(["--calibration-method", model_record.task.calibration_method])

        for key, value in (extra_args or {}).items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            elif isinstance(value, (list, tuple)):
                args.extend([flag, *[str(item) for item in value]])
            elif value is not None:
                args.extend([flag, str(value)])

        completed = self._run_cli(args)
        return {
            "backend": self.backend_name,
            "command": args,
            "preds_path": str(output_path),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }

    def train_model(
        self,
        train_csv: str,
        output_dir: str,
        task: PredictionTaskSpec,
        *,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        input_path = Path(train_csv).expanduser()
        if not input_path.exists():
            raise InvalidPredictionInputError(f"Training CSV does not exist: {train_csv}")

        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        args = [
            "chemprop",
            "train",
            "--data-path",
            str(input_path),
            "--task-type",
            task.task_type,
            "--output-dir",
            str(output_path),
        ]

        if task.smiles_columns:
            args.extend(["--smiles-columns", *task.smiles_columns])

        if task.target_columns:
            args.extend(["--target-columns", *task.target_columns])

        if task.reaction_columns:
            args.extend(["--reaction-columns", *task.reaction_columns])

        for key, value in (extra_args or {}).items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            elif isinstance(value, (list, tuple)):
                args.extend([flag, *[str(item) for item in value]])
            elif value is not None:
                args.extend([flag, str(value)])

        completed = self._run_cli(args)
        return {
            "backend": self.backend_name,
            "command": args,
            "output_dir": str(output_path),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
