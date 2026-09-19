"""ACV fault-ranking model seeded with the promoted approach from the independent `nebulax` work:
a pairwise ranker over per-signal, case-median-centred car descriptors.

Contract (frozen): `Model.__init__(self)`, `Model.fit(self, X, y)`, `Model.predict(self, X)`.

X is a list of 1-D float vectors, one per car.  Each vector holds, for every signal the workbook can
expose (148 columns = union over all cases: `<signal>|mean`, `|std`, `|missing`, `|delta`), the value
centred on the car's own case median.  NaN means the car's case does not expose that signal -- cases expose
different subsets (five train cases and the deployment case share 4 signals; train case 4 exposes 32
unrelated ones).  y is 1 for the faulty car, 0 otherwise.  predict() returns one fault score per car
(higher = more likely faulty); cars are ranked within their case and scored by rank-decay.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

# EVOLVE-BLOCK-START
class Model:
    def __init__(self) -> None:
        self._F = None
        self._y = None

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        self._F = np.vstack([np.asarray(x, dtype=float) for x in X])
        self._y = np.asarray(y).astype(int)

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        Fp = np.vstack([np.asarray(x, dtype=float) for x in X])
        # only signals the scored case actually exposes, and that training also observed
        obs = np.flatnonzero((~np.isnan(Fp)).any(0) & (~np.isnan(self._F)).any(0))
        if len(obs) == 0 or (self._y == 1).sum() == 0:
            return np.zeros(len(Fp))
        Ftr = np.nan_to_num(self._F[:, obs])
        pos, neg = np.flatnonzero(self._y == 1), np.flatnonzero(self._y == 0)
        diffs, lab = [], []
        for p in pos:                      # directed faulty-vs-normal comparisons
            for n in neg:
                d = Ftr[p] - Ftr[n]
                diffs += [d, -d]
                lab += [1, 0]
        pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=min(5, len(obs), len(diffs) - 1)),
                             LogisticRegression(C=0.003, class_weight="balanced", max_iter=5000))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(np.asarray(diffs), np.asarray(lab))
            return np.asarray(pipe.decision_function(np.nan_to_num(Fp[:, obs])), dtype=float)
# EVOLVE-BLOCK-END
