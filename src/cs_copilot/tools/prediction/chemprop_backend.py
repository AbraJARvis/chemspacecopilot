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

import json
import importlib.metadata
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .backend import (
    BackendNotAvailableError,
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionExecutionError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from cs_copilot.tools.chemistry.standardize import standardize_smiles_column

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

    def _sanitize_train_extra_args(
        self, extra_args: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Normalize or drop legacy Chemprop CLI arguments before training.

        Older prompts/runs may still emit v1/v2.0-era flags. Chemprop v2.1+
        rejects some of them hard (notably `num_folds`). We normalize the safe
        ones and silently drop those that only duplicate newer required args.
        """
        sanitized = dict(extra_args or {})

        if "num_folds" in sanitized and "num_replicates" not in sanitized:
            logger.warning(
                "Received deprecated Chemprop train arg `num_folds`; mapping it to `num_replicates`."
            )
            sanitized["num_replicates"] = sanitized.pop("num_folds")
        else:
            sanitized.pop("num_folds", None)

        if "save_dir" in sanitized:
            logger.warning(
                "Dropping deprecated/duplicate Chemprop train arg `save_dir`; `output_dir` is already set."
            )
            sanitized.pop("save_dir", None)

        return sanitized

    def _resolve_artifact_path(
        self,
        model_record: PredictionModelRecord,
        raw_path: Optional[str],
    ) -> Optional[Path]:
        if not raw_path:
            return None

        candidate = Path(raw_path).expanduser()
        candidates = [candidate]
        if not candidate.is_absolute():
            if model_record.metadata_path:
                candidates.append(Path(model_record.metadata_path).expanduser().parent / candidate)
            model_path = Path(model_record.model_path).expanduser()
            candidates.append(model_path.parent / candidate)
            candidates.append(model_path.parent.parent / candidate)

        for item in candidates:
            try:
                if item.exists():
                    return item.resolve()
            except Exception:
                continue
        return None

    def _bitrow_to_fingerprint(self, row: np.ndarray):
        bitstring = "".join("1" if int(value) else "0" for value in np.asarray(row).ravel())
        if hasattr(DataStructs, "CreateFromBitString"):
            return DataStructs.CreateFromBitString(bitstring)
        bitvect = DataStructs.ExplicitBitVect(len(bitstring))
        for idx, char in enumerate(bitstring):
            if char == "1":
                bitvect.SetBit(idx)
        return bitvect

    def _load_applicability_domain_assets(
        self,
        model_record: PredictionModelRecord,
    ) -> Optional[Dict[str, Any]]:
        applicability_domain = model_record.applicability_domain or {}
        index_path = self._resolve_artifact_path(
            model_record,
            applicability_domain.get("index_path") or applicability_domain.get("applicability_domain_path"),
        )
        store_path = self._resolve_artifact_path(
            model_record,
            applicability_domain.get("reference_store_path"),
        )

        if index_path is None or store_path is None:
            return None

        cache_key = (str(index_path), str(store_path))
        cache = getattr(self, "_ad_cache", {})
        if cache_key in cache:
            return cache[cache_key]

        try:
            index_payload = json.loads(index_path.read_text())
            store_payload = np.load(store_path, allow_pickle=True)
        except Exception as exc:
            logger.warning("Could not load applicability-domain artifacts for %s: %s", model_record.model_id, exc)
            return None

        fingerprint_matrix = store_payload.get("fingerprints")
        if fingerprint_matrix is None:
            logger.warning(
                "Applicability-domain store for %s is missing fingerprints.", model_record.model_id
            )
            return None

        reference_fps = [self._bitrow_to_fingerprint(row) for row in np.asarray(fingerprint_matrix)]
        if not reference_fps:
            return None

        manifest_path = self._resolve_artifact_path(
            model_record,
            applicability_domain.get("reference_manifest_path"),
        )
        manifest_payload: Dict[str, Any] = {}
        if manifest_path is not None and manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text())
            except Exception:
                manifest_payload = {}

        payload = {
            "index": index_payload,
            "manifest": manifest_payload,
            "reference_fps": reference_fps,
            "store_path": store_path,
            "index_path": index_path,
        }
        cache[cache_key] = payload
        setattr(self, "_ad_cache", cache)
        return payload

    def _compute_applicability_domain_scores(
        self,
        *,
        smiles_values: list[Any],
        model_record: PredictionModelRecord,
    ) -> Dict[str, list[Any]]:
        assets = self._load_applicability_domain_assets(model_record)
        if assets is None:
            return {}

        index_payload = assets["index"]
        reference_fps = assets["reference_fps"]
        if not reference_fps:
            return {}

        fingerprint_cfg = index_payload.get("fingerprint") or {}
        radius = int(fingerprint_cfg.get("radius", 2))
        nbits = int(fingerprint_cfg.get("nbits", 2048))
        thresholds = index_payload.get("thresholds") or {}
        in_domain_min = float(thresholds.get("in_domain_min", 0.5))
        edge_domain_min = float(thresholds.get("edge_domain_min", 0.35))

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

        ad_max_scores: list[Optional[float]] = []
        ad_support_scores: list[Optional[float]] = []
        ad_primary_status: list[Optional[str]] = []
        ad_support_status: list[Optional[str]] = []
        ad_status: list[Optional[str]] = []

        for smiles in smiles_values:
            if not isinstance(smiles, str) or not smiles.strip():
                ad_max_scores.append(None)
                ad_support_scores.append(None)
                ad_primary_status.append(None)
                ad_support_status.append(None)
                ad_status.append(None)
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                ad_max_scores.append(None)
                ad_support_scores.append(None)
                ad_primary_status.append(None)
                ad_support_status.append(None)
                ad_status.append(None)
                continue

            query_fp = generator.GetFingerprint(mol)
            sims = DataStructs.BulkTanimotoSimilarity(query_fp, reference_fps)
            if not sims:
                ad_max_scores.append(None)
                ad_support_scores.append(None)
                ad_primary_status.append(None)
                ad_support_status.append(None)
                ad_status.append(None)
                continue

            sims_arr = np.asarray(sims, dtype=float)
            max_sim = float(np.max(sims_arr))
            top_k = min(5, len(sims_arr))
            support_sim = float(np.mean(np.sort(sims_arr)[-top_k:])) if top_k > 0 else max_sim

            def _status(score: float) -> str:
                if score >= in_domain_min:
                    return "in_domain"
                if score >= edge_domain_min:
                    return "edge_of_domain"
                return "out_of_domain"

            primary_status = _status(max_sim)
            support_status = _status(support_sim)
            statuses = {primary_status, support_status}
            if "out_of_domain" in statuses:
                final_status = "out_of_domain"
            elif "edge_of_domain" in statuses:
                final_status = "edge_of_domain"
            else:
                final_status = "in_domain"

            ad_max_scores.append(max_sim)
            ad_support_scores.append(support_sim)
            ad_primary_status.append(primary_status)
            ad_support_status.append(support_status)
            ad_status.append(final_status)

        return {
            "ad_max_tanimoto": ad_max_scores,
            "ad_support_tanimoto": ad_support_scores,
            "ad_primary_status": ad_primary_status,
            "ad_support_status": ad_support_status,
            "ad_status": ad_status,
            "ad_threshold_in_domain": [in_domain_min] * len(ad_status),
            "ad_threshold_edge": [edge_domain_min] * len(ad_status),
            "ad_reference_size": [len(reference_fps)] * len(ad_status),
            "ad_method": [index_payload.get("method", "hybrid_morgan_domain")] * len(ad_status),
        }

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
        ad_columns: Dict[str, list[Any]] = {}
        ad_summary: Dict[str, Any] = {}
        try:
            input_df = pd.read_csv(input_path)
            smiles_column = model_record.task.smiles_columns[0] if model_record.task.smiles_columns else "smiles"
            if smiles_column in input_df.columns:
                standardized_df = standardize_smiles_column(input_df.copy(), smiles_column)
                smiles_values = standardized_df["smiles"].tolist()
                ad_columns = self._compute_applicability_domain_scores(
                    smiles_values=smiles_values,
                    model_record=model_record,
                )
                if ad_columns:
                    ad_frame = pd.DataFrame(ad_columns)
                    predictions_df = pd.read_csv(output_path)
                    if len(predictions_df) == len(ad_frame):
                        merged_df = pd.concat([predictions_df.reset_index(drop=True), ad_frame], axis=1)
                        merged_df.to_csv(output_path, index=False)
                        counts = merged_df["ad_status"].value_counts(dropna=False).to_dict()
                        ad_summary = {
                            "available": True,
                            "method": (ad_frame["ad_method"].dropna().iloc[0] if "ad_method" in ad_frame and not ad_frame["ad_method"].dropna().empty else None),
                            "reference_size": int(ad_frame["ad_reference_size"].dropna().iloc[0]) if "ad_reference_size" in ad_frame and not ad_frame["ad_reference_size"].dropna().empty else None,
                            "status_counts": {
                                str(key): int(value) for key, value in counts.items()
                            },
                        }
        except Exception as exc:
            logger.warning("Could not enrich prediction output with applicability domain: %s", exc)
            ad_columns = {}
            ad_summary = {}
        return {
            "backend": self.backend_name,
            "command": args,
            "preds_path": str(output_path),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "applicability_domain": ad_summary,
            "applicability_domain_columns": list(ad_columns.keys()),
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
        started_at = datetime.now().astimezone()

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

        sanitized_extra_args = self._sanitize_train_extra_args(extra_args)

        for key, value in sanitized_extra_args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            elif isinstance(value, (list, tuple)):
                args.extend([flag, *[str(item) for item in value]])
            elif value is not None:
                args.extend([flag, str(value)])

        completed = self._run_cli(args)
        completed_at = datetime.now().astimezone()
        return {
            "backend": self.backend_name,
            "command": args,
            "output_dir": str(output_path),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
        }
