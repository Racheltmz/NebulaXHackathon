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
        self._model = None
        self._label = None
        self._channels = 0

    def _features(self, X):
        out = []
        for a in X:
            a = np.nan_to_num(np.asarray(a, dtype=float))
            if a.ndim != 2:
                a = np.atleast_2d(a)
            z = np.zeros((self._channels, 35))
            for c in range(min(self._channels, a.shape[0])):
                v = a[c].ravel()
                if len(v) == 0:
                    continue
                d = np.diff(v)
                t = np.linspace(-1, 1, len(v))
                slope = np.polyfit(t, v, 1)[0] if len(v) > 1 else 0.
                q = np.percentile(v, [10, 25, 50, 75, 90])
                dq = np.percentile(d, [10, 25, 50, 75, 90]) if len(d) else np.zeros(5)
                seg = [x.mean() for x in np.array_split(v, 4)]
                u = np.linspace(0., 1., 8)
                shape = np.interp(u, np.linspace(0., 1., len(v)), v)
                if len(v) > 1 and np.std(v[:-1]) > 1e-12 and np.std(v[1:]) > 1e-12:
                    corr = np.corrcoef(v[:-1], v[1:])[0, 1]
                else:
                    corr = 0.
                z[c] = [v.mean(), v.std(), v.min(), v.max(), v.ptp(),
                        *q, np.mean(np.abs(d)) if len(d) else 0.,
                        np.mean(v * v), slope,
                        np.mean(np.abs(v - v.mean())),
                        v[0], v[-1], np.std(d) if len(d) else 0.,
                        *dq, corr, *seg, *shape]
            h = []
            for c in range(self._channels):
                v = a[c].ravel() if c < a.shape[0] else np.zeros(8)
                if len(v) == 0:
                    v = np.zeros(8)
                if len(v) < 2:
                    v = np.pad(v, (0, 2 - len(v)))
                h.append(np.interp(
                    np.linspace(0., 1., 16),
                    np.linspace(0., 1., len(v)),
                    (v - v.mean()) / (v.std() + 1e-8)))
            h = np.asarray(h)
            extra = []
            if a.shape[0]:
                cm = np.mean(a, axis=0)
                cs = np.std(a, axis=0)
                cr = np.ptp(a, axis=0)
                cmed = np.median(a, axis=0)
                cq25 = np.percentile(a, 25, axis=0)
                cq75 = np.percentile(a, 75, axis=0)
                cd = (np.mean(np.abs(np.diff(a, axis=0)), axis=0)
                      if a.shape[0] > 1 else np.zeros_like(cm))
                cv = (np.std(np.diff(a, axis=0), axis=0)
                      if a.shape[0] > 1 else np.zeros_like(cm))
                for v in (cm, cs, cr, cmed, cq25, cq75, cd, cv):
                    extra.extend(np.interp(
                        np.linspace(0., 1., 8),
                        np.linspace(0., 1., len(v)), v))
                if len(cm) > 1:
                    dc = np.diff(cm)
                    extra.extend([
                        dc.mean(), dc.std(), np.mean(np.abs(dc)),
                        np.max(np.abs(dc)),
                        np.percentile(np.abs(dc), 90)
                    ])
                    sp = np.abs(np.fft.rfft(cm - cm.mean()))
                    sp = sp[1:] if len(sp) > 1 else sp
                    extra.extend(np.interp(
                        np.linspace(0., 1., 8),
                        np.linspace(0., 1., max(1, len(sp))), sp))
                else:
                    extra.extend([0.] * 13)
                extra.extend([
                    cm.mean(), cm.std(), cs.mean(), cs.std(),
                    cr.mean(), cr.std(), cd.mean(), cd.std(),
                    cv.mean(), cv.std()
                ])
            z = np.concatenate([z.ravel(), h.mean(axis=0), h.std(axis=0),
                                np.asarray(extra)])
            out.append(z)
        return np.asarray(out)

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        y = np.asarray(y, dtype=object)
        vals, cnt = np.unique(y, return_counts=True)
        self._label = vals[int(np.argmax(cnt))]
        self._channels = max(np.asarray(a).shape[0] for a in X)
        f = self._features(X)
        if len(vals) > 1:
            self._model = make_pipeline(
                StandardScaler(),
                SVC(C=3.0, gamma="scale",
                    class_weight="balanced"))
            self._model.fit(f, y)

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        if self._model is None:
            return np.full(len(X), self._label, dtype=object)
        return np.asarray(self._model.predict(self._features(X)), dtype=object)
# EVOLVE-BLOCK-END
