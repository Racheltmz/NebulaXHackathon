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
        self.model = None

    def _features(self, X: Sequence[np.ndarray]) -> np.ndarray:
        out = []
        for x in X:
            a = np.nan_to_num(
                np.asarray(x, dtype=float),
                nan=0.0, posinf=0.0, neginf=0.0
            )
            if a.ndim == 1:
                a = a[None, :]
            q = np.abs(a)
            d = np.abs(np.diff(a, axis=1))
            out.append(np.concatenate([
                a.mean(1),
                q.mean(1),
                q.std(1),
                np.percentile(q, 90, axis=1),
                np.percentile(q, 99, axis=1),
                q.max(1),
                np.std(a, axis=1),
                d.mean(1) if d.shape[1] else np.zeros(a.shape[0]),
                np.array([
                    q[4:].mean(),
                    np.percentile(q[4:], 95),
                    q[4:].max(),
                ]) if a.shape[0] > 4 else np.zeros(3),
            ]))
        return np.asarray(out)

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        f = self._features(X)
        self.mu = f.mean(0)
        self.scale = f.std(0) + 1e-6
        z = (f - self.mu) / self.scale
        y = np.asarray(y, dtype=int)
        p = z[y > 0]
        n = z[y == 0]
        raw_w = (p.mean(0) - n.mean(0)) / (
            p.std(0) + n.std(0) + 1e-3
        )
        self.w = np.clip(raw_w, -4.0, 4.0)
        self.pos = p
        self.pos_center = np.median(p, axis=0)
        self.neg_center = np.median(n, axis=0)

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        f = self._features(X)
        if not hasattr(self, "w"):
            return f.mean(1)
        z = (f - self.mu) / self.scale
        linear = z @ self.w + 0.08 * np.mean(z[:, 1:], axis=1)
        dist = np.mean(
            np.abs(z[:, None, :] - self.pos[None, :, :]),
            axis=2,
        )
        k = min(3, dist.shape[1])
        retrieval = -np.mean(np.partition(dist, k - 1, axis=1)[:, :k], axis=1)
        pos_dist = np.mean(np.abs(z - self.pos_center[None, :]), axis=1)
        neg_dist = np.mean(np.abs(z - self.neg_center[None, :]), axis=1)
        centroid = neg_dist - pos_dist
        linear = (linear - np.median(linear)) / (np.std(linear) + 1e-8)
        retrieval = (retrieval - np.median(retrieval)) / (
            np.std(retrieval) + 1e-8
        )
        centroid = (centroid - np.median(centroid)) / (
            np.std(centroid) + 1e-8
        )
        return np.asarray(
            0.70 * linear + 0.18 * retrieval + 0.12 * centroid,
            dtype=float,
        )
# EVOLVE-BLOCK-END
