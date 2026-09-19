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
        self._models = []
        self._fallback = 0.0

    @staticmethod
    def _features(a: np.ndarray) -> np.ndarray:
        z = np.asarray(a, dtype=float).reshape(-1)
        z = z[np.isfinite(z)]
        if z.size == 0:
            return np.zeros(231, dtype=float)

        q = np.percentile(z, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
        d = np.diff(z)
        ad = np.abs(d)
        centered = z - q[5]
        f = list(q)
        f += [
            np.mean(z), np.std(z), np.sqrt(np.mean(z * z)),
            np.mean(np.abs(centered)), np.mean(np.abs(z)),
            np.max(np.abs(z)), np.mean(ad), np.std(d),
            *np.percentile(ad, [25, 50, 75, 90, 99]).tolist(),
        ]
        # Damage is often driven by excursions around the local mean rather
        # than by signed stress. Add robust amplitude and high-order fatigue
        # descriptors while keeping them logarithmically bounded.
        ac = np.abs(centered)
        f.extend(np.percentile(ac, [50, 75, 90, 95, 99, 99.9]).tolist())
        for p in (1.5, 2, 3, 4):
            f.append(float(np.log1p(np.mean(ac ** p))))

        # Turning-point ranges provide compact fatigue-cycle proxies.
        if z.size > 2:
            s = np.sign(np.diff(z))
            s[s == 0] = 1
            ix = np.flatnonzero(s[1:] != s[:-1]) + 1
            tp = np.concatenate(([z[0]], z[ix], [z[-1]]))
            r = np.abs(np.diff(tp))
        else:
            r = np.abs(d)
        if r.size:
            rq = np.percentile(r, [10, 25, 50, 75, 90, 95, 99])
            f += list(rq)
            for p in (1, 2, 3, 4, 5, 6, 8):
                f.append(np.mean(r ** p))
        else:
            f += [0.0] * 14

        # Stable fatigue proxies emphasize cumulative large-cycle damage while
        # avoiding the extreme scale of untransformed high-order moments.
        if r.size:
            f.extend([
                float(np.log1p(np.mean(r ** 2))),
                float(np.log1p(np.mean(r ** 3))),
                float(np.log1p(np.mean(r ** 4))),
                float(np.log1p(np.mean(r ** 6))),
                float(np.log1p(np.sum(r > np.percentile(r, 90)))),
                float(np.mean(r > np.median(r))),
            ])
        else:
            f.extend([0.0] * 6)

        # Tail-weighted fatigue descriptors separate frequent moderate cycles
        # from a small number of severe cycles, which strongly affects damage.
        if r.size:
            rq = np.percentile(r, [50, 75, 90, 95, 99])
            for thr in rq[1:]:
                excess = np.maximum(r - thr, 0.0)
                f.extend([
                    float(np.log1p(np.mean(excess))),
                    float(np.log1p(np.mean(excess ** 2))),
                    float(np.log1p(np.mean(excess ** 3))),
                    float(np.mean(excess > 0.0)),
                ])
            top = np.sort(r)[-max(1, r.size // 100):]
            f.extend([
                float(np.log1p(np.mean(top))),
                float(np.log1p(np.mean(top ** 2))),
                float(np.log1p(np.mean(top ** 3))),
                float(np.log1p(np.sum(top ** 3))),
            ])
        else:
            f.extend([0.0] * 20)

        # Log-scaled multiscale damage and roughness descriptors.
        # These remain numerically stable while preserving high-order fatigue cues.
        for stride in (1, 4, 16, 64):
            u = z[::stride]
            if u.size > 1:
                du = np.abs(np.diff(u))
                for p in (1, 2, 3, 4):
                    f.append(float(np.log1p(np.mean(du ** p))))
                f.extend(np.log1p(np.percentile(du, [50, 75, 90, 95, 99])).tolist())
                acorr = np.corrcoef(u[:-1], u[1:])[0, 1] if u.size > 3 else 0.0
                f.append(float(np.nan_to_num(acorr)))
            else:
                f.extend([0.0] * 10)

        # Block peak-to-peak ranges capture slow, concentrated load excursions
        # that sample-to-sample turning points can miss.
        for nblock in (32, 128, 512, 2048):
            ranges = np.asarray([
                np.max(w) - np.min(w) for w in np.array_split(z, nblock)
                if w.size
            ], dtype=float)
            if ranges.size:
                f.extend(np.log1p(np.percentile(ranges, [25, 50, 75, 90, 95, 99])).tolist())
                f.extend([
                    float(np.log1p(np.mean(ranges ** 2))),
                    float(np.log1p(np.mean(ranges ** 4))),
                    float(np.log1p(np.sum(ranges > np.percentile(ranges, 90)))),
                ])
            else:
                f.extend([0.0] * 9)

        # Robust window statistics capture nonstationary loading.
        nwin = 16
        for w in np.array_split(z, nwin):
            if w.size:
                f.extend([np.mean(w), np.std(w), np.percentile(w, 90) -
                          np.percentile(w, 10), np.mean(np.abs(np.diff(w)))])
                # Window-local RMS and high-order amplitude capture
                # nonstationary loading and concentrated damage.
                f.extend([
                    np.sqrt(np.mean(w * w)),
                    np.percentile(np.abs(w), 95),
                    float(np.log1p(np.mean(np.abs(w) ** 3))),
                    float(np.log1p(np.mean(np.abs(w) ** 4))),
                ])
            else:
                f.extend([0.0] * 8)

        # Low-cost spectral energy descriptors.
        if z.size > 32:
            step = max(1, z.size // 4096)
            u = z[::step]
            u = u - np.mean(u)
            sp = np.abs(np.fft.rfft(u)) ** 2
            bands = np.array_split(sp[1:], 8)
            total = float(np.sum(sp[1:]) + 1e-12)
            f.extend([np.log1p(total)])
            f.extend([float(np.sum(b) / total) for b in bands])
        else:
            f.extend([0.0] * 9)
        return np.asarray(f, dtype=float)

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor

        F = np.vstack([self._features(a) for a in X])
        F = np.nan_to_num(F, nan=0.0, posinf=1e12, neginf=-1e12)
        target = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=1e12, neginf=0.0)
        self._fallback = float(np.mean(target)) if target.size else 0.0
        t = np.log1p(np.maximum(target, 0.0))
        self._models = [
            ExtraTreesRegressor(n_estimators=300, max_features=0.65,
                                min_samples_leaf=2, random_state=17,
                                n_jobs=-1),
            RandomForestRegressor(n_estimators=260, max_features=0.55,
                                  min_samples_leaf=3, random_state=31,
                                  n_jobs=-1),
            HistGradientBoostingRegressor(max_iter=260, learning_rate=0.035,
                                          max_leaf_nodes=15, min_samples_leaf=4,
                                          l2_regularization=0.35,
                                          random_state=53),
            HistGradientBoostingRegressor(max_iter=320, learning_rate=0.025,
                                          max_leaf_nodes=31, min_samples_leaf=5,
                                          l2_regularization=0.55,
                                          random_state=89),
            GradientBoostingRegressor(n_estimators=220, learning_rate=0.03,
                                      max_depth=2, min_samples_leaf=4,
                                      subsample=0.8, loss="huber",
                                      random_state=71),
        ]
        for model in self._models:
            model.fit(F, t)

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        if not self._models:
            return np.full(len(X), self._fallback, dtype=float)
        F = np.vstack([self._features(a) for a in X])
        F = np.nan_to_num(F, nan=0.0, posinf=1e12, neginf=-1e12)
        pred = np.mean([m.predict(F) for m in self._models], axis=0)
        return np.maximum(0.0, np.expm1(np.clip(pred, -20.0, 40.0)))
# EVOLVE-BLOCK-END
