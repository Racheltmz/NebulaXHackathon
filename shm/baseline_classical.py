"""Baseline SHM fatigue-damage regressor.

Contract (frozen): `Model.__init__(self)`, `Model.fit(self, X, y)`,
`Model.predict(self, X)`. X is a list of raw [1, 581119] float arrays
(a single long stress time series per example). y holds raw float
cumulative-damage targets during fit; predict() receives only X and must
return one float per row.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# EVOLVE-BLOCK-START
class Model:
    def __init__(self) -> None:
        self._mean_target = None

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        self._mean_target = float(np.mean(np.asarray(y, dtype=float)))

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        return np.full(len(X), self._mean_target if self._mean_target is not None else 0.0, dtype=float)
# EVOLVE-BLOCK-END
