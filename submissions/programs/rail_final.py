"""Rail classifier seeded with the promoted model from the independent `nebulax`
(ian-classical) work: source-aware three-view logistic ensemble with nested
inner-CV selection of the phase-branch settings.

Contract (frozen): `Model.__init__(self)`, `Model.fit(self, X, y)`,
`Model.predict(self, X)`.

X is a list of 1-D float vectors, one per recording, produced by label-free,
per-recording transforms of all 128 raw sensor channels + tachometer:
    X[i] = concat( v3[0:N_V3] | relative[N_V3:N_V3+N_REL] | phase[N_V3+N_REL:] )
  * v3       : per side/kind RMS, p99, crest, kurtosis, |skew|, spectral peak,
               band-energy and peak-frequency descriptors (+ side1-side2 diffs)
  * relative : v3 minus constant n_rows, plus scale-normalised side contrasts
               (rel_diff_, log_ratio_, side_sum_, side_max_)
  * phase    : tachometer phase/order-profile moments and harmonics
y holds raw string labels ("Normal" / "Side I" / "Side II") during fit;
predict() receives only X.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_V3, N_REL, N_PHASE = 96, 223, 96
CLASSES = np.array(["Normal", "Side I", "Side II"], dtype=object)

# EVOLVE-BLOCK-START
from sklearn.svm import SVC
_BIAS_GRID = [(bi, bii)
              for bi in (-0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.7)
              for bii in (-0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.7)]
_GRID = [(k, c, w, bi, bii)
         for k in (20, 40, 60, 80)
         for c in (0.003, 0.01, 0.03, 0.1, 0.3)
         for w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)
         for bi, bii in _BIAS_GRID]


def _views(A):
    return A[:, :N_V3], A[:, N_V3:N_V3 + N_REL], A[:, N_V3 + N_REL:]


def _proba(Xtr, ytr, Xte, k, c):
    m = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=min(k, Xtr.shape[1], len(Xtr) - 1)),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)
    out = np.zeros((len(Xte), len(CLASSES)))
    for j, cl in enumerate(m.classes_):
        out[:, list(CLASSES).index(cl)] = p[:, j]
    return out


def _retrieve(Xtr, ytr, Xte, k=5):
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd[sd < 1e-8] = 1.0
    tr = (np.nan_to_num(Xtr, nan=mu) - mu) / sd
    te = (np.nan_to_num(Xte, nan=mu) - mu) / sd
    d = ((te[:, None, :] - tr[None, :, :]) ** 2).mean(axis=2)
    kk = min(k, len(tr))
    nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    ww = 1.0 / (np.sqrt(np.take_along_axis(d, nn, axis=1)) + 1e-6)
    out = np.zeros((len(te), len(CLASSES)))
    for j, cl in enumerate(CLASSES):
        out[:, j] = (
            np.sum(ww * (ytr[nn] == cl), axis=1)
            / max(np.sum(ytr == cl), 1)
        )
    out += 1e-6
    return out / out.sum(axis=1, keepdims=True)


def _hierarchical(Xtr, ytr, Xte):
    """Fault gate followed by a Side-I/Side-II classifier."""
    fault = (np.asarray(ytr) != "Normal").astype(int)
    gate = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=min(80, Xtr.shape[1], len(Xtr) - 1)),
        LogisticRegression(C=0.03, class_weight="balanced", max_iter=5000),
    )
    mask = fault == 1
    side = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=min(80, Xtr.shape[1], max(2, mask.sum() - 1))),
        LogisticRegression(C=0.03, class_weight="balanced", max_iter=5000),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate.fit(Xtr, fault)
        side.fit(Xtr[mask], ytr[mask])
    pg = gate.predict_proba(Xte)[:, 1]
    ps = side.predict_proba(Xte)
    out = np.zeros((len(Xte), len(CLASSES)))
    out[:, 0] = 1.0 - pg
    for j, cl in enumerate(side.classes_):
        out[:, list(CLASSES).index(cl)] = pg * ps[:, j]
    return out


def _sparse_proba(Xtr, ytr, Xte):
    """Nonlinear regularized head for interactions among relative descriptors."""
    m = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=min(80, Xtr.shape[1], len(Xtr) - 1)),
        SVC(
            C=0.7, gamma="scale", probability=True,
            class_weight="balanced", random_state=173,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)
    out = np.zeros((len(Xte), len(CLASSES)))
    for j, cl in enumerate(m.classes_):
        out[:, list(CLASSES).index(cl)] = p[:, j]
    return out


def _base_and_phase(A_tr, y_tr, A_te, phase_kc):
    """Blend dense, sparse, hierarchical, and retrieval views."""
    v_tr, r_tr, p_tr = _views(A_tr)
    v_te, r_te, p_te = _views(A_te)
    joint_tr = np.hstack([v_tr, r_tr])
    joint_te = np.hstack([v_te, r_te])
    base = (
        0.12 * _proba(v_tr, y_tr, v_te, 50, 0.03)
        + 0.35 * _proba(r_tr, y_tr, r_te, 150, 0.01)
        + 0.15 * _proba(joint_tr, y_tr, joint_te, 160, 0.01)
        + 0.10 * _retrieve(r_tr, y_tr, r_te, 5)
        + 0.13 * _hierarchical(r_tr, y_tr, r_te)
        + 0.15 * _sparse_proba(r_tr, y_tr, r_te)
    )
    phase = {kc: _proba(p_tr, y_tr, p_te, kc[0], kc[1]) for kc in phase_kc}
    return base, phase


def _decide(base, phase, w, bi=0.0, bii=0.0):
    q = (1 - w) * base + w * phase
    bias = np.array([0.0, bi, bii])
    return CLASSES[np.argmax(np.log(np.maximum(q, 1e-9)) + bias, axis=1)]


class Model:
    def __init__(self) -> None:
        self._A = None
        self._y = None
        self._cfg = None

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        A = np.nan_to_num(np.vstack([np.asarray(x, dtype=float).ravel() for x in X]))
        y = np.asarray(y).astype(str)
        self._A, self._y = A, y
        counts = np.unique(y, return_counts=True)[1]
        n_splits = int(min(5, counts.min()))
        if n_splits < 2:
            self._cfg = (40, 0.03, 0.2, 0.0, 0.0)
            return
        phase_kc = sorted({(cfg[0], cfg[1]) for cfg in _GRID})
        preds = {cfg: [] for cfg in _GRID}
        for seed in (1701, 2718):
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
            for tr, te in cv.split(A, y):
                base, phase = _base_and_phase(A[tr], y[tr], A[te], phase_kc)
                for (k, c, w, bi, bii) in _GRID:
                    preds[(k, c, w, bi, bii)].append(
                        (te, _decide(base, phase[(k, c)], w, bi, bii)))
        best, best_key = None, None
        for cfg in _GRID:
            yy = np.concatenate([y[idx] for idx, _ in preds[cfg]])
            pp = np.concatenate([pred for _, pred in preds[cfg]]).astype(str)
            key = (f1_score(yy, pp, average="macro"),
                   balanced_accuracy_score(yy, pp))
            if best_key is None or key > best_key:
                best, best_key = cfg, key
        self._cfg = best

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        A_te = np.nan_to_num(np.vstack([np.asarray(x, dtype=float).ravel() for x in X]))
        k, c, w, bi, bii = self._cfg
        base, phase = _base_and_phase(self._A, self._y, A_te, [(k, c)])
        return _decide(base, phase[(k, c)], w, bi, bii).astype(object)
# EVOLVE-BLOCK-END
