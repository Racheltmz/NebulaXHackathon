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
        self._models = []
        self._label = None

    def _features(self, X: Sequence[np.ndarray]) -> np.ndarray:
        out = []
        for a in X:
            a = np.nan_to_num(np.asarray(a, dtype=float))
            rows = []
            for z in a:
                z = np.nan_to_num(z)
                scale = np.std(z) + 1e-8
                d = np.diff(z)
                q = np.percentile(z, [1, 5, 25, 50, 75, 95, 99])
                v = [np.mean(z), scale, np.sqrt(np.mean(z*z)),
                     np.mean(np.abs(z)), np.std(d), np.mean(d*d),
                     np.max(np.abs(z)), np.mean((z-z.mean())**3)/scale**3,
                     np.mean((z-z.mean())**4)/scale**4] + list(q)
                s = np.abs(np.fft.rfft(z-z.mean()))[1:]
                n = len(s)
                edges = np.linspace(0, n, 17, dtype=int)
                p = s*s
                total = p.sum() + 1e-8
                v += list(np.log1p([p[edges[i]:edges[i+1]].mean()
                                    for i in range(16)]))
                csum = np.cumsum(p) / total
                v += [np.argmax(s) / max(1, n), np.log1p(s.max()),
                      np.sum(p[:max(1, n//20)]) / total,
                      np.sum(np.arange(1, n + 1) * p) / (n * total),
                      np.searchsorted(csum, .50) / max(1, n),
                      np.searchsorted(csum, .90) / max(1, n),
                      -np.sum((p / total) * np.log(p / total + 1e-12))]
                rows.append(v)
            f = np.asarray(rows).ravel()
            if len(a) > 1:
                c = np.corrcoef(a)
                c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
                f = np.r_[f, c[np.triu_indices(len(a), 1)]]
            rms = np.sqrt(np.mean(a * a, axis=1))
            spread = np.std(a, axis=1)
            ref = rms[1] + 1e-12 if len(a) > 1 else 1.0
            f = np.r_[f, rms[1:] / ref,
                      spread[1:] / (spread[1] + 1e-12)]
            if len(a) >= 3:
                f = np.r_[f,
                          (rms[1] - rms[2]) / (rms[1] + rms[2] + 1e-12),
                          (spread[1] - spread[2]) /
                          (spread[1] + spread[2] + 1e-12)]
                cuts = np.linspace(0, a.shape[1], 9, dtype=int)
                g1 = np.sqrt(np.mean((a[1] - np.mean(a[1])) ** 2)) + 1e-12
                g2 = np.sqrt(np.mean((a[2] - np.mean(a[2])) ** 2)) + 1e-12
                for k in range(8):
                    u, v = a[1, cuts[k]:cuts[k + 1]], a[2, cuts[k]:cuts[k + 1]]
                    ru, rv = np.sqrt(np.mean(u * u)), np.sqrt(np.mean(v * v))
                    su, sv = np.std(u), np.std(v)
                    nu, nv = np.sqrt(np.mean((u - np.mean(u)) ** 2)) / g1, \
                             np.sqrt(np.mean((v - np.mean(v)) ** 2)) / g2
                    f = np.r_[f, (ru - rv) / (ru + rv + 1e-12),
                              (su - sv) / (su + sv + 1e-12),
                              nu - nv, nu, nv]
            out.append(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0))
        return np.asarray(out)

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        y = np.asarray(y, dtype=object)
        labels = np.unique(y)
        self._label = labels[0]
        self._models = []
        if len(labels) > 1:
            F = self._features(X)
            for c in (0.35, 0.6, 1.0):
                m = make_pipeline(
                    StandardScaler(),
                    SVC(C=c, gamma="scale", class_weight="balanced"))
                m.fit(F, y)
                self._models.append(m)

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        if not self._models:
            return np.full(len(X), self._label, dtype=object)
        F = self._features(X)
        pred = np.asarray([m.predict(F) for m in self._models], dtype=object)
        return np.asarray([
            max(np.unique(pred[:, i]), key=lambda z: np.sum(pred[:, i] == z))
            for i in range(len(X))
        ], dtype=object)
# EVOLVE-BLOCK-END
