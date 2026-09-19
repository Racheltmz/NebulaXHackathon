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
        self._model = None
        self._log_model = None
        self._rf_model = None
        self._gb_model = None
        self._mean_target = 0.0

    @staticmethod
    def _features(X: Sequence[np.ndarray]) -> np.ndarray:
        rows = []
        for z in X:
            a = np.nan_to_num(
                np.asarray(z, dtype=float).ravel(),
                nan=0.0, posinf=0.0, neginf=0.0
            )
            if not len(a):
                rows.append(np.zeros(288))
                continue
            d = np.diff(a)
            ad = np.abs(d)
            q = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
            aq = np.percentile(ad, [50, 75, 90, 95, 99])
            f = [
                np.mean(a), np.std(a), np.min(a), np.max(a),
                np.mean(np.abs(a)), np.sqrt(np.mean(a * a)),
                np.mean(np.abs(a) ** 3), np.mean(np.abs(a) ** 4),
                np.max(np.abs(a)), *q,
                np.mean(ad), np.std(d), np.max(ad), *aq,
                np.mean(d * d),
                np.mean(d[:-1] * d[1:]) if len(d) > 1 else 0.0,
            ]
            turn = np.flatnonzero(d[:-1] * d[1:] < 0) + 1 if len(d) > 1 else np.array([], dtype=int)
            if len(turn) > 1:
                r = np.abs(np.diff(a[turn]))
                f.extend([
                    len(turn) / len(a), np.mean(r), np.std(r),
                    *np.percentile(r, [75, 90, 95, 99]),
                    np.mean(r ** 2), np.mean(r ** 3),
                    np.mean(r ** 4), np.mean(r ** 5),
                    np.mean(r ** 6),
                    np.sum(r ** 3) / len(a),
                    np.sum(r ** 5) / len(a),
                    np.mean(np.sort(r)[-max(1, len(r) // 20):] ** 3) / len(a)
                ])
            else:
                f.extend([0.] * 15)
            edges = np.linspace(0, len(a), 32 + 1).astype(int)
            for i in range(32):
                s = a[edges[i]:edges[i + 1]]
                if len(s):
                    f.extend([
                        np.mean(s), np.std(s), np.max(np.abs(s)),
                        np.sqrt(np.mean(s * s)),
                        np.percentile(np.abs(s), 95),
                        np.mean(np.abs(s) ** 3),
                    ])
                else:
                    f.extend([0., 0., 0., 0., 0., 0., 0.])
            s = a[::max(1, len(a) // 4096)]
            p = np.abs(np.fft.rfft(s - np.mean(s))) ** 2
            cuts = np.linspace(0, len(p), 17).astype(int)
            f.extend(np.log1p([
                np.mean(p[cuts[i]:cuts[i + 1]]) for i in range(16)
            ]))
            rows.append(np.nan_to_num(f))
        return np.asarray(rows, dtype=float)

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        from sklearn.ensemble import ExtraTreesRegressor
        yy = np.asarray(y, dtype=float)
        self._mean_target = float(np.mean(yy)) if len(yy) else 0.0
        if len(yy) > 1:
            feats = self._features(X)
            self._model = ExtraTreesRegressor(
                n_estimators=240, min_samples_leaf=2, max_features=1.0,
                random_state=17, n_jobs=-1
            )
            self._model.fit(feats, yy)
            self._log_model = ExtraTreesRegressor(
                n_estimators=180, min_samples_leaf=2, max_features=0.8,
                random_state=31, n_jobs=-1
            )
            self._log_model.fit(
                feats, np.log1p(np.maximum(0.0, yy))
            )
            from sklearn.ensemble import RandomForestRegressor
            self._rf_model = RandomForestRegressor(
                n_estimators=180, min_samples_leaf=2, max_features=0.7,
                random_state=47, n_jobs=-1
            )
            self._rf_model.fit(feats, yy)
            from sklearn.ensemble import GradientBoostingRegressor
            self._gb_model = GradientBoostingRegressor(
                n_estimators=160, learning_rate=0.035, max_depth=2,
                min_samples_leaf=2, loss="huber", random_state=73
            )
            self._gb_model.fit(
                feats, np.log1p(np.maximum(0.0, yy))
            )

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        if self._model is None:
            return np.full(len(X), self._mean_target, dtype=float)
        feats = self._features(X)
        raw = self._model.predict(feats)
        if self._log_model is not None:
            logged = np.expm1(np.clip(
                self._log_model.predict(feats), -20.0, 30.0
            ))
            raw = 0.58 * raw + 0.27 * logged
        if self._rf_model is not None:
            raw = 0.82 * raw + 0.18 * self._rf_model.predict(feats)
        if self._gb_model is not None:
            boosted = np.expm1(np.clip(
                self._gb_model.predict(feats), -20.0, 30.0
            ))
            raw = 0.85 * raw + 0.15 * boosted
        return np.maximum(0.0, raw)
# EVOLVE-BLOCK-END
