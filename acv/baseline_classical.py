"""Baseline ACV fault-ranking model.

Contract (frozen): `Model.__init__(self)`, `Model.fit(self, X, y)`,
`Model.predict(self, X)`. X is a list of raw [8, T] float arrays, one per
car (4 raw per-timestep stats + 4 case-relative residual stats -- see
ps3_adapter.py's acv() for how these 8 channels are built from the original
per-car sensor columns). y is 1 for the faulty car, 0 otherwise, during fit.
predict() receives only X and must return one float "fault score" per row
(higher = more likely faulty) -- the harness ranks cars within each held-out
case by this score and scores by the official rank-decay formula. Cars are
NOT explicitly grouped by case in X/y; case membership is a harness-only
concept used for splitting/scoring, so the model must not assume ordering.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# EVOLVE-BLOCK-START
class Model:
    def __init__(self) -> None:
        pass

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        pass

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        return np.array([float(np.abs(np.asarray(x, dtype=float)).mean()) for x in X])
# EVOLVE-BLOCK-END
