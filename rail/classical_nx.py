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
_BIAS_VALUES = (0.0, 0.15, 0.3)
_GRID = [
    (k, c, w, bi, bii)
    for k in (20, 40, 60, 80)
    for c in (0.01, 0.03, 0.1)
    for w in (0.0, 0.1, 0.2, 0.3, 0.5)
    for bi in _BIAS_VALUES
    for bii in _BIAS_VALUES
]


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


def _base_and_phase(A_tr, y_tr, A_te, phase_kc):
    """Base (v3 + relative) probabilities and one phase probability per (k, C)."""
    v_tr, r_tr, p_tr = _views(A_tr)
    v_te, r_te, p_te = _views(A_te)
    base = 0.3 * _proba(v_tr, y_tr, v_te, 50, 0.03) + 0.7 * _proba(r_tr, y_tr, r_te, 150, 0.01)
    phase = {kc: _proba(p_tr, y_tr, p_te, kc[0], kc[1]) for kc in phase_kc}
    return base, phase


def _decide(base, phase, w, bi=0.2, bii=0.0):
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
        n_splits = int(min(3, counts.min()))
        if n_splits < 2:
            self._cfg = (40, 0.03, 0.2)
            return
        phase_kc = sorted({(k, c) for k, c, _, _, _ in _GRID})
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=1701)
        preds = {cfg: np.empty(len(y), dtype=object) for cfg in _GRID}
        for tr, te in cv.split(A, y):
            base, phase = _base_and_phase(A[tr], y[tr], A[te], phase_kc)
            for (k, c, w, bi, bii) in _GRID:
                preds[(k, c, w, bi, bii)][te] = _decide(
                    base, phase[(k, c)], w, bi, bii
                )
        best, best_key = None, None
        for cfg in _GRID:
            yp = preds[cfg].astype(str)
            key = (
                f1_score(y, yp, average="macro"),
                balanced_accuracy_score(y, yp),
            )
            if best_key is None or key > best_key:
                best, best_key = cfg, key
        self._cfg = best

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        A_te = np.nan_to_num(np.vstack([np.asarray(x, dtype=float).ravel() for x in X]))
        k, c, w, bi, bii = self._cfg
        base, phase = _base_and_phase(self._A, self._y, A_te, [(k, c)])
        return _decide(base, phase[(k, c)], w, bi, bii).astype(object)
# EVOLVE-BLOCK-END
