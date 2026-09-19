"""Shared, non-evolved evaluation harness for all 4 PS3 openevolve tracks.

This module is imported by each task's evaluator.py (never by candidate
programs, never placed inside an EVOLVE-BLOCK). It owns:
  - loading raw per-task arrays (the ONLY place data is loaded from disk)
  - building 3-fold / 5-fold stratified (or grouped) CV splits + the fixed
    train/test split
  - running each candidate's fit/predict inside the network-disabled Docker
    sandbox, so the evolved code never touches the network or the host
    filesystem outside its own tmp input/output files
  - computing each task's *official* competition metric from predictions vs.
    true labels -- the candidate model itself never sees a true label for
    anything it is asked to predict on.
"""
from __future__ import annotations

import pickle
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

SANDBOX_IMAGE = "ps3-evolve-sandbox:latest"
DEFAULT_TIMEOUT_S = 90


class CandidateError(Exception):
    pass


def run_candidate_sandboxed(
    program_path: str,
    X_train: Sequence[Any],
    y_train: np.ndarray,
    X_test: Sequence[Any],
    timeout: int = DEFAULT_TIMEOUT_S,
) -> np.ndarray:
    """Run fit(X_train,y_train) then predict(X_test) inside a network-disabled,
    resource-capped Docker container. Raises CandidateError on any failure
    (syntax error, exception, timeout, missing Model class, etc.)."""
    program_path = str(Path(program_path).resolve())
    with tempfile.TemporaryDirectory(prefix="ps3evolve-in-") as in_dir, \
         tempfile.TemporaryDirectory(prefix="ps3evolve-out-") as out_dir:
        in_dir_p, out_dir_p = Path(in_dir), Path(out_dir)
        candidate_copy = in_dir_p / "candidate.py"
        candidate_copy.write_text(Path(program_path).read_text(encoding="utf-8"), encoding="utf-8")
        sandbox_entry = Path(__file__).resolve().parent.parent / "sandbox" / "sandbox_entry.py"
        (in_dir_p / "sandbox_entry.py").write_bytes(sandbox_entry.read_bytes())
        payload = {
            "program_path": "/work/candidate.py",
            "X_train": list(X_train),
            "y_train": np.asarray(y_train),
            "X_test": list(X_test),
        }
        with open(in_dir_p / "input.pkl", "wb") as f:
            pickle.dump(payload, f)

        # Files are moved with `docker cp` instead of `-v` bind mounts: on
        # Docker Desktop/WSL2 every bind mount leaks a mount entry that is
        # never released, and after ~100k of them (a few hours of evolution)
        # the daemon fails every new container with "no space left on device".
        container_name = f"ps3evolve-{uuid.uuid4().hex[:12]}"

        def _docker(*args: str, t: int = 60) -> subprocess.CompletedProcess:
            return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=t)

        created = _docker(
            "create", "--name", container_name, "--network", "none",
            "--memory=3g", "--cpus=2", "--pids-limit=256", SANDBOX_IMAGE,
        )
        if created.returncode != 0:
            raise CandidateError(f"sandbox create failed: {created.stderr[-500:]}")
        result_path = out_dir_p / "result.pkl"
        try:
            copied = _docker("cp", f"{in_dir_p}/.", f"{container_name}:/work")
            if copied.returncode != 0:
                raise CandidateError(f"sandbox copy-in failed: {copied.stderr[-500:]}")
            try:
                proc = subprocess.run(
                    ["docker", "start", "-a", container_name],
                    capture_output=True, timeout=timeout, text=True,
                )
            except subprocess.TimeoutExpired:
                _docker("kill", container_name, t=30)
                raise CandidateError(f"timed out after {timeout}s")
            _docker("cp", f"{container_name}:/work/result.pkl", str(result_path))
        finally:
            _docker("rm", "-f", container_name, t=60)

        if not result_path.exists():
            raise CandidateError(
                f"sandbox produced no result (exit={proc.returncode}) stderr={proc.stderr[-800:]}"
            )
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        if result.get("error"):
            raise CandidateError(result["error"])
        return np.asarray(result["predictions"])


def stratified_kfold_indices(y: np.ndarray, n_splits: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def kfold_indices(n: int, n_splits: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.zeros(n)))


def leave_one_group_out_indices(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    return list(logo.split(np.zeros(len(groups)), groups=groups))


def run_cv(
    X: Sequence[Any],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    program_path: str,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    timeout: int = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    scores = []
    for tr_idx, te_idx in splits:
        X_tr = [X[i] for i in tr_idx]
        X_te = [X[i] for i in te_idx]
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        preds = run_candidate_sandboxed(program_path, X_tr, y_tr, X_te, timeout=timeout)
        scores.append(float(metric_fn(y_te, preds)))
    return {"fold_scores": scores, "mean": float(np.mean(scores)) if scores else 0.0}


def run_cv_grouped(
    X: Sequence[Any],
    y: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    program_path: str,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    timeout: int = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Like run_cv, but for metrics that need the group id of each held-out
    row (e.g. ACV's per-case rank-decay), not just (y_true, y_pred)."""
    scores = []
    for tr_idx, te_idx in splits:
        X_tr = [X[i] for i in tr_idx]
        X_te = [X[i] for i in te_idx]
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        g_te = groups[te_idx]
        preds = run_candidate_sandboxed(program_path, X_tr, y_tr, X_te, timeout=timeout)
        scores.append(float(metric_fn(y_te, preds, g_te)))
    return {"fold_scores": scores, "mean": float(np.mean(scores)) if scores else 0.0}


def group_kfold_indices(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(groups)), groups=groups))


def code_length(program_path: str) -> int:
    return len(Path(program_path).read_text(encoding="utf-8"))
