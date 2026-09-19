from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(ROOT / "common"))

from contract import validate_public_contract, static_safety_scan  # noqa: E402
from harness import run_cv, stratified_kfold_indices, run_candidate_sandboxed, code_length, CandidateError  # noqa: E402
from metrics import rail_score  # noqa: E402

DATA_DIR = Path(os.environ.get("PS3_NX_DATA_DIR", ROOT / "data" / "nx_arrays"))
INITIAL_PROGRAM = EXPERIMENT_DIR / "baseline_classical_nx.py"
SANDBOX_TIMEOUT = int(os.environ.get("PS3_SANDBOX_TIMEOUT", "90"))


def _load():
    Xtr = list(np.load(DATA_DIR / "rail_train_x.npy", allow_pickle=True))
    ytr = np.asarray(np.load(DATA_DIR / "rail_train_y.npy", allow_pickle=True))
    Xte = list(np.load(DATA_DIR / "rail_test_x.npy", allow_pickle=True))
    yte = np.asarray(np.load(DATA_DIR / "rail_test_y.npy", allow_pickle=True))
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

        cv3 = run_cv(Xtr, ytr, stratified_kfold_indices(ytr, 3), program_path, rail_score, SANDBOX_TIMEOUT)
        cv5 = run_cv(Xtr, ytr, stratified_kfold_indices(ytr, 5), program_path, rail_score, SANDBOX_TIMEOUT)
        test_preds = run_candidate_sandboxed(program_path, Xtr, ytr, Xte, timeout=SANDBOX_TIMEOUT)
        traintest_metric = rail_score(yte, test_preds)

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
    except CandidateError as exc:
        return {
            "combined_score": 0.0, "cv3_metric": 0.0, "traintest_metric": 0.0,
            "cv5_metric": 0.0, "code_length": float(code_length(program_path)),
            "contract_valid": 1.0, "error": f"CandidateError: {exc}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "combined_score": 0.0, "cv3_metric": 0.0, "traintest_metric": 0.0,
            "cv5_metric": 0.0, "code_length": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }
