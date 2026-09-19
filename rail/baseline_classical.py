"""Baseline Rail corrugation classifier.

Contract (frozen): `Model.__init__(self)`, `Model.fit(self, X, y)`,
`Model.predict(self, X)`. X is a list of raw [5, 10000] float arrays
(speed + 4 aggregated vibration/shock channels). y holds raw string labels
("Normal" / "Side I" / "Side II") during fit; predict() receives only X.
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
