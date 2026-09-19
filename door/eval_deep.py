"""Door GPU-track evaluator. Same protocol as the classical track (3-fold +
5-fold stratified CV + fixed train/test split, official metric), but each
fit/predict call runs on a NUS SoC cluster GPU node via SLURM instead of a
local Docker sandbox -- see cloud/cloud_harness.py for the anti-cheat
disclosure specific to that environment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "cloud"))

from contract import validate_public_contract, static_safety_scan  # noqa: E402
from metrics import door_score  # noqa: E402
from harness import stratified_kfold_indices  # noqa: E402 (pure sklearn split logic, reused as-is)
from cloud_harness import run_cv_cloud, run_candidate_cloud, code_length, CloudCandidateError  # noqa: E402

DATA_DIR = Path(os.environ.get("PS3_DATA_DIR", ROOT / "data" / "ps3_arrays"))
INITIAL_PROGRAM = EXPERIMENT_DIR / "baseline_deep.py"
CLOUD_TIMEOUT = int(os.environ.get("PS3_CLOUD_TIMEOUT", "600"))


def _load():
    Xtr = list(np.load(DATA_DIR / "door_train_x.npy", allow_pickle=True))
    ytr = np.asarray(np.load(DATA_DIR / "door_train_y.npy", allow_pickle=True))
    Xte = list(np.load(DATA_DIR / "door_test_x.npy", allow_pickle=True))
    yte = np.asarray(np.load(DATA_DIR / "door_test_y.npy", allow_pickle=True))
    return Xtr, ytr, Xte, yte


def evaluate(program_path: str) -> dict:
    try:
        enforce = os.environ.get("OPENEVO_GUARD_ENFORCE_SIGNATURES", "1") != "0"
        if enforce:
            violations = validate_public_contract(program_path, str(INITIAL_PROGRAM))
            if violations:
                return {
                    "combined_score": 0.0, "cv3_metric": 0.0, "traintest_metric": 0.0,
                    "cv5_metric": 0.0, "code_length": float(code_length(program_path)),
                    "contract_valid": 0.0, "error": "; ".join(violations),
                }
        safety_warnings = static_safety_scan(program_path)

        Xtr, ytr, Xte, yte = _load()

        cv3 = run_cv_cloud(Xtr, ytr, stratified_kfold_indices(ytr, 3), program_path, door_score, CLOUD_TIMEOUT)
        cv5 = run_cv_cloud(Xtr, ytr, stratified_kfold_indices(ytr, 5), program_path, door_score, CLOUD_TIMEOUT)
        test_preds = run_candidate_cloud(program_path, Xtr, ytr, Xte, timeout=CLOUD_TIMEOUT)
        traintest_metric = door_score(yte, test_preds)

        combined_score = 0.5 * cv3["mean"] + 0.5 * traintest_metric
        return {
            "combined_score": float(combined_score),
            "cv3_metric": float(cv3["mean"]),
            "cv5_metric": float(cv5["mean"]),
            "traintest_metric": float(traintest_metric),
            "code_length": float(code_length(program_path)),
            "contract_valid": 1.0,
            "safety_warnings": "; ".join(safety_warnings) if safety_warnings else "",
        }
    except CloudCandidateError as exc:
        return {
            "combined_score": 0.0, "cv3_metric": 0.0, "traintest_metric": 0.0,
            "cv5_metric": 0.0, "code_length": float(code_length(program_path)),
            "contract_valid": 1.0, "error": f"CloudCandidateError: {exc}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "combined_score": 0.0, "cv3_metric": 0.0, "traintest_metric": 0.0,
            "cv5_metric": 0.0, "code_length": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }
