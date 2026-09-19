"""Exact official PS3 scoring formulas, quoted from each subsystem's Info Kit.

door_score: IoU-weighted F1 (Door_Subsystem_Info_Kit.md Sec 4). Because this
harness evaluates label prediction on the already-known true segment
boundaries (no boundary discovery), every predicted segment has IoU=1.0 with
its matching true segment and there are never extra/missing segments -- so
soft_recall == soft_precision == (correct / total), and the harmonic-mean
formula in the Info Kit collapses exactly to plain accuracy. This is not an
approximation; it's the same formula evaluated under a fixed-boundary setup.

rail_score: Macro F1 across Normal/Side I/Side II (Rail_Corrugation_Info_Kit
Sec 4) -- unweighted average of per-class F1.

shm_score: max(0, 1 - MAPE) between predicted and true cumulative damage
(SHM_Info_Kit Sec 5).

acv_score: mean per-case linear rank-decay score, score = (n-(r-1))/n where r
is the 1-indexed rank of the true faulty car by descending predicted fault
score within its case (ACV_Subsystem_Info_Kit Sec 4).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_percentage_error


def door_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def rail_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def shm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.clip(np.asarray(y_true, dtype=float), 1e-6, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, None)
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    return max(0.0, 1.0 - mape)


def acv_rank_decay(y_true: np.ndarray, fault_scores: np.ndarray, case_ids: np.ndarray) -> float:
    """y_true: 1 for the faulty car, 0 otherwise. fault_scores: higher = more
    likely faulty. case_ids: which case (group of ~8 cars) each row belongs to."""
    y_true = np.asarray(y_true)
    fault_scores = np.asarray(fault_scores, dtype=float)
    case_ids = np.asarray(case_ids)
    per_case_scores = []
    for case in np.unique(case_ids):
        mask = case_ids == case
        n = int(mask.sum())
        yt = y_true[mask]
        fs = fault_scores[mask]
        if yt.sum() == 0:
            continue
        faulty_idx = int(np.argmax(yt))
        # Ties are scored at their EXPECTED rank (random tie-break), never by array order: an all-tied case
        # (e.g. a model with no usable signal) must not get rank 1 just because the faulty car is listed first.
        greater = int((fs > fs[faulty_idx] + 1e-12).sum())
        ties = int((np.abs(fs - fs[faulty_idx]) <= 1e-12).sum()) - 1
        rank = 1 + greater + ties / 2.0
        per_case_scores.append((n - (rank - 1)) / n)
    return float(np.mean(per_case_scores)) if per_case_scores else 0.0
