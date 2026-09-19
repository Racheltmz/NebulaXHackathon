"""Baseline Door classifier.

Contract (frozen by the evolution harness -- do not rename or change these
signatures): `Model.__init__(self)`, `Model.fit(self, X, y)`,
`Model.predict(self, X)`.

X is a list of raw [channels, time] float arrays, one per door cycle
segment (variable length and channel count are NOT guaranteed equal across
tasks, but for Door every example has 16 channels). y is an array of raw
string labels ("Normal" / "Abnormal resistance") during fit; predict()
receives only X, never labels, and must return one label per row of X.
fit()/predict() must not read any file, open a socket, or import anything
beyond ordinary numeric/ML libraries -- they run in a network-disabled
sandbox that would reject such attempts anyway.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# EVOLVE-BLOCK-START
class Model:
    def __init__(self) -> None:
        self._majority_label = None

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        values, counts = np.unique(np.asarray(y, dtype=object), return_counts=True)
        self._majority_label = values[int(np.argmax(counts))]

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        return np.array([self._majority_label] * len(X), dtype=object)
# EVOLVE-BLOCK-END
