"""SHM fatigue-damage regressor seeded with the promoted model from the
independent `nebulax` (ian-classical) work: five-view kernel/SVR blend fitted in
log-target space, followed by a clipped fold-local Ridge residual calibrator on
amplitude and short-window roughness covariates.

Contract (frozen): `Model.__init__(self)`, `Model.fit(self, X, y)`,
`Model.predict(self, X)`.

X is a list of 1-D float vectors, one per trace, produced by label-free,
per-trace transforms of the raw 581,120-sample stress series:
    X[i] = concat( multiscale[0:561] | generic[561:605] | rainflow[605:688] | temporal[688:838] )
  * multiscale: per window-count (8..128) quantiles/first/last/delta/slope/auc of
                mean, std, rms, ptp, |x| mean, max|x|, diff-std across windows
  * generic   : global moments, quantiles, turning-point ranges, spectral bands
  * rainflow  : four-point rainflow cycle-range quantiles and Miner damage sums
  * temporal  : 128-window temporal-evolution descriptors
y holds raw float cumulative-damage targets during fit; predict() receives only
X and must return one float per row (scored by max(0, 1 - MAPE)).
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

OFFSETS = [0, 561, 605, 688, 838]
# columns feeding the residual calibrator (generic ptp/std, multiscale amplitude
# quantiles, short-window difference-SD quantiles)
CAL_COLS = [568, 562, 54, 55, 56, 166, 167, 102, 103, 104]

# EVOLVE-BLOCK-START
def _tables(A):
    return [A[:, OFFSETS[i]:OFFSETS[i + 1]] for i in range(4)]


def _ker(Xtr, ytr, Xte, k, alpha, gamma, offset=1.0, power=0.0):
    sel = SelectKBest(lambda a, b: mutual_info_regression(a, b, random_state=0),
                      k=min(k, Xtr.shape[1], len(Xtr) - 1)).fit(Xtr, ytr)
    m = make_pipeline(StandardScaler(), KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma))
    w = np.maximum(ytr + 0.01, 1e-4) ** -power
    w = w / w.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(sel.transform(Xtr), np.log(ytr + offset), kernelridge__sample_weight=w)
    return np.maximum(np.exp(m.predict(sel.transform(Xte))) - offset, 1e-6)


def _svr(Xtr, ytr, Xte):
    sel = SelectKBest(lambda a, b: mutual_info_regression(a, b, random_state=0),
                      k=min(10, Xtr.shape[1], len(Xtr) - 1)).fit(Xtr, ytr)
    m = make_pipeline(StandardScaler(), SVR(C=10, gamma=0.01, epsilon=0.01))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(sel.transform(Xtr), np.log1p(ytr), svr__sample_weight=np.maximum(ytr, 1e-5) ** -0.5)
    return np.maximum(np.expm1(m.predict(sel.transform(Xte))), 1e-6)


def _backbone(A_tr, ytr, A_te):
    ms_tr, gen_tr, rain_tr, tmp_tr = _tables(A_tr)
    ms_te, gen_te, rain_te, tmp_te = _tables(A_te)
    ms = _ker(ms_tr, ytr, ms_te, 20, 0.01, 0.01, 0.01, 0.5)
    rel = _ker(gen_tr, ytr, gen_te, 5, 0.03, 0.03, 0.01, 0.5)
    gen = _ker(gen_tr, ytr, gen_te, 5, 0.1, 0.1, 1.0, 0.0)
    rain = _ker(rain_tr, ytr, rain_te, 3, 0.1, 0.1, 1.0, 0.0)
    tmp = _svr(tmp_tr, ytr, tmp_te)
    return 0.4 * ms + 0.6 * (0.4 * rel + 0.42 * gen + 0.108 * rain + 0.072 * tmp)


def _cal_features(A):
    return np.log(np.maximum(np.abs(A[:, CAL_COLS]), 1e-8))


class Model:
    def __init__(self) -> None:
        self._A = None
        self._y = None
        self._cal = None

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        A = np.nan_to_num(np.vstack([np.asarray(x, dtype=float).ravel() for x in X]))
        y = np.asarray(y, dtype=float)
        self._A, self._y = A, y
        oof = np.zeros(len(y))
        # Five folds give each fold-local backbone more training examples,
        # reducing calibration noise on this small dataset.
        for a, b in KFold(5, shuffle=True, random_state=1701).split(A):
            oof[b] = _backbone(A[a], y[a], A[b])
        residual = np.log(np.maximum(y, 1e-8)) - np.log(np.maximum(oof, 1e-8))
        z = np.c_[_cal_features(A), np.log(np.maximum(oof, 1e-8))]
        # Stronger regularization keeps the fold-local residual correction stable
        # with only 51 traces and many correlated amplitude descriptors.
        self._cal = make_pipeline(StandardScaler(), Ridge(alpha=0.03)).fit(z, residual)

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        A_te = np.nan_to_num(np.vstack([np.asarray(x, dtype=float).ravel() for x in X]))
        raw = _backbone(self._A, self._y, A_te)
        z = np.c_[_cal_features(A_te), np.log(np.maximum(raw, 1e-8))]
        corr = np.clip(self._cal.predict(z), -1.0, 1.0)
        return np.maximum(raw * np.exp(0.65 * corr), 1e-6).astype(float)
# EVOLVE-BLOCK-END
