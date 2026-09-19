#!/usr/bin/env python3
"""SHM: cumulative fatigue-damage regressor (data preprocessing + model + training, one file)

Turns each raw 581 120-sample stress trace into 838 label-free descriptors (multiscale window statistics,
rainflow cycle counting / Miner damage, temporal evolution) and regresses damage in log space with a kernel/SVR/ridge
blend followed by a clipped residual calibrator (scored by 1 - MAPE).

Usage
    python shm_model.py --data-root <path to the organisers' 02_Datasets folder> --out <output folder> [--jobs N]

Writes (organiser format, ready to zip):
    <out>/no_hard_label_retraining/shm_predictions.csv    model fitted on ALL labelled data, predicting the held-out set
    <out>/with_hard_label_retraining/shm_predictions.csv  the same model retrained on all labelled data + the confident held-out
                                            items (hard pseudo-labels, cutoff tuned on the labelled data)

Requirements (versions the submitted predictions were produced with):
    python 3.12, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.2, pandas 2.2.3, joblib
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence
from joblib import Parallel, delayed
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
import numpy as np
import pandas as pd


# =============================================================================
# 1. DATA PREPROCESSING (raw organiser files -> model inputs)
# =============================================================================
LABEL_ORDER = ["train17.csv", "train28.csv", "train55.csv", "train11.csv", "train36.csv", "train54.csv", "train50.csv", "train13.csv", "train01.csv", "train58.csv", "train07.csv", "train05.csv", "train46.csv", "train62.csv", "train33.csv", "train23.csv", "train20.csv", "train25.csv", "train15.csv", "train43.csv", "train40.csv", "train51.csv", "train27.csv", "train21.csv", "train29.csv", "train64.csv", "train52.csv", "train37.csv", "train57.csv", "train02.csv", "train38.csv", "train10.csv", "train04.csv", "train41.csv", "train45.csv", "train47.csv", "train14.csv", "train63.csv", "train48.csv", "train18.csv", "train19.csv", "train59.csv", "train60.csv", "train56.csv", "train09.csv", "train08.csv", "train34.csv", "train31.csv", "train16.csv", "train30.csv", "train39.csv", "train49.csv", "train24.csv", "train53.csv", "train26.csv", "train06.csv", "train44.csv", "train03.csv", "train32.csv", "train35.csv", "train61.csv", "train22.csv", "train42.csv", "train12.csv"]
# ^ order of the labelled traces (a fixed seed-7 shuffle); kept so the model's internal cross-validation folds, and
# therefore the submitted predictions, are reproduced exactly.


# ---- per-trace features from the raw 581 120-sample stress series (label-free, one trace at a time)
def shm_generic(x: np.ndarray) -> dict:
    x = x.astype("float32")
    a = np.abs(x)
    d = np.diff(x)
    q = np.quantile(x, [.01, .05, .1, .25, .5, .75, .9, .95, .99])
    aq = np.quantile(a, [.5, .9, .95, .99, .999])
    out = {"mean": x.mean(), "std": x.std(), "rms": np.sqrt(np.mean(x * x)), "abs_mean": a.mean(), "abs_std": a.std(),
           "min": x.min(), "max": x.max(), "ptp": np.ptp(x), "skew": skew(x), "kurtosis": kurtosis(x, fisher=False),
           "diff_std": d.std(), "diff_abs_mean": np.abs(d).mean(),
           "zero_cross": np.mean((x[:-1] - x.mean()) * (x[1:] - x.mean()) < 0), "n": len(x)}
    out.update({f"q{i}": v for i, v in enumerate(q)})
    out.update({f"absq{i}": v for i, v in enumerate(aq)})
    y = x[::16]
    peaks, _ = find_peaks(y, distance=2)
    troughs, _ = find_peaks(-y, distance=2)
    tp = np.sort(np.r_[peaks, troughs])
    ranges = np.abs(np.diff(y[tp])) if len(tp) > 1 else np.array([0.])
    out.update({"turning_points": len(tp), "range_mean": ranges.mean(), "range_std": ranges.std(),
                "range_p90": np.quantile(ranges, .9), "range_p99": np.quantile(ranges, .99), "range_max": ranges.max()})
    n = (len(x) // 64) * 64
    z = x[:n].reshape(-1, 64).mean(1)
    power = np.abs(np.fft.rfft(z - z.mean())) ** 2
    power[0] = 0
    freq = np.fft.rfftfreq(len(z), 64 / 10000)
    total = power.sum() + 1e-9
    out.update({"spec_peak_hz": freq[power.argmax()],
                "spec_entropy": float(-np.sum((power / total) * np.log(power / total + 1e-12)))})
    for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 500), (500, 5000)):
        out[f"band_{lo}_{hi}"] = power[(freq >= lo) & (freq < hi)].sum() / total
    return out


def shm_multiscale(x: np.ndarray) -> dict:
    x = x.astype("float32")
    out = {"n": float(len(x))}
    for nw in (8, 16, 32, 64, 128):
        z = x[:(len(x) // nw) * nw].reshape(nw, -1)
        t = np.linspace(0, 1, nw)
        vals = {"mean": z.mean(1), "std": z.std(1), "rms": np.sqrt(np.mean(z * z, 1)), "ptp": np.ptp(z, 1),
                "absmean": np.abs(z).mean(1), "maxabs": np.abs(z).max(1), "diffstd": np.diff(z, axis=1).std(1)}
        for name, v in vals.items():
            q = np.quantile(v, [.05, .1, .25, .5, .75, .9, .95, .99])
            for i, a in enumerate(q):
                out[f"w{nw}_{name}_q{i}"] = float(a)
            out[f"w{nw}_{name}_first"] = float(v[0])
            out[f"w{nw}_{name}_last"] = float(v[-1])
            out[f"w{nw}_{name}_delta"] = float(v[-1] - v[0])
            out[f"w{nw}_{name}_stdtime"] = float(v.std())
            out[f"w{nw}_{name}_max"] = float(v.max())
            out[f"w{nw}_{name}_min"] = float(v.min())
            out[f"w{nw}_{name}_slope"] = float(np.polyfit(t, v, 1)[0])
            out[f"w{nw}_{name}_auc"] = float(np.trapezoid(v, t) if hasattr(np, "trapezoid") else np.trapz(v, t))
    return out


def _reversals(x):
    peaks, _ = find_peaks(x)
    troughs, _ = find_peaks(-x)
    ix = np.sort(np.r_[0, peaks, troughs, len(x) - 1])
    y = x[ix]
    keep = np.r_[True, np.diff(y) != 0]
    return y[keep]


def _cycles(rev):
    stack = []
    out = []
    for v in rev:
        stack.append(float(v))
        while len(stack) >= 3:
            r1 = abs(stack[-2] - stack[-3])
            r2 = abs(stack[-1] - stack[-2])
            if r1 < r2:
                break
            if len(stack) == 3:
                out.append((r1, .5))
                del stack[0]
            else:
                out.append((r1, 1.0))
                del stack[-3:-1]
    out.extend((abs(stack[i + 1] - stack[i]), .5) for i in range(len(stack) - 1))
    if not out:
        return np.zeros(1), np.ones(1)
    r, c = np.asarray(out, dtype="float64").T
    return r, c


def shm_rainflow(x: np.ndarray) -> dict:
    x = x.astype("float32")
    out = {"n": float(len(x)), "std": float(x.std()), "ptp": float(np.ptp(x))}
    for ds in (4, 16, 64, 256):
        r, c = _cycles(_reversals(x[::ds]))
        out[f"rf{ds}_count"] = float(c.sum())
        out[f"rf{ds}_reversals"] = float(len(r))
        for q, v in zip((.5, .75, .9, .95, .99, .999, 1.0), np.quantile(r, [.5, .75, .9, .95, .99, .999, 1.0])):
            out[f"rf{ds}_q{q:g}"] = float(v)
        for m in (1, 2, 3, 4, 5, 6, 8, 10, 12):
            out[f"rf{ds}_damage{m}"] = float(np.sum(c * np.maximum(r, 1e-12) ** m))
        out[f"rf{ds}_range_mean"] = float(np.average(r, weights=c))
        out[f"rf{ds}_range_std"] = float(np.sqrt(np.average((r - np.average(r, weights=c)) ** 2, weights=c)))
    return out


def shm_temporal(x: np.ndarray, windows: int = 128) -> dict:
    x = x.astype("float32")
    n = (len(x) // windows) * windows
    z = x[:n].reshape(windows, -1)
    t = np.linspace(0, 1, windows)
    absz = np.abs(z)
    diff = np.diff(z, axis=1)
    series = {"mean": z.mean(1), "std": z.std(1), "rms": np.sqrt(np.mean(z * z, 1)), "ptp": np.ptp(z, 1),
              "abs_mean": absz.mean(1), "maxabs": absz.max(1), "diff_std": diff.std(1)}
    out = {"n": len(x)}
    w10 = max(1, windows // 10)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for name, v in series.items():
        q = np.quantile(v, [.05, .1, .25, .5, .75, .9, .95])
        out.update({f"{name}_q{i}": float(a) for i, a in enumerate(q)})
        out[f"{name}_first"] = float(v[0])
        out[f"{name}_last"] = float(v[-1])
        out[f"{name}_first10_mean"] = float(v[:w10].mean())
        out[f"{name}_last10_mean"] = float(v[-w10:].mean())
        out[f"{name}_delta"] = float(v[-1] - v[0])
        out[f"{name}_delta10"] = float(v[-w10:].mean() - v[:w10].mean())
        out[f"{name}_ratio10"] = float(v[-w10:].mean() / (abs(v[:w10].mean()) + 1e-8))
        out[f"{name}_std_time"] = float(v.std())
        out[f"{name}_max_time"] = float(v.max())
        out[f"{name}_min_time"] = float(v.min())
        out[f"{name}_slope"] = float(np.polyfit(t, v, 1)[0])
        out[f"{name}_slope_firsthalf"] = float(np.polyfit(t[:windows // 2], v[:windows // 2], 1)[0])
        out[f"{name}_slope_lasthalf"] = float(np.polyfit(t[windows // 2:], v[windows // 2:], 1)[0])
        out[f"{name}_auc"] = float(trap(v, t))
    out["ptp_over_std_mean"] = float(np.mean(series["ptp"] / (series["std"] + 1e-8)))
    out["ptp_slope_minus_std_slope"] = out["ptp_slope"] - out["std_slope"]
    return out


def shm_vector(x):
    """[ multiscale (561) | generic (44) | rainflow (83) | temporal (150) ]  (column order = dict insertion order)."""
    parts = [np.array(list(f(x).values()), dtype=np.float64)
             for f in (shm_multiscale, shm_generic, shm_rainflow, shm_temporal)]
    v = np.concatenate(parts)
    assert [len(p) for p in parts] == [561, 44, 83, 150]
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def _vector_from_file(path):
    return shm_vector(np.loadtxt(path, delimiter=","))


def load_data(data_root, jobs=4):
    base = Path(data_root) / "SHM"
    damage = pd.read_csv(base / "Train_Labels.csv").set_index("filename").damage
    X_lab = Parallel(n_jobs=jobs)(delayed(_vector_from_file)(base / "Train" / n) for n in LABEL_ORDER)
    y = np.array([float(damage[n]) for n in LABEL_ORDER])
    test_files = sorted((base / "Test").glob("*.csv"))
    X_test = Parallel(n_jobs=jobs)(delayed(_vector_from_file)(p) for p in test_files)
    return X_lab, y, X_test, [p.name for p in test_files]


def write_csv(folder, ids, pred):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file_id": ids, "prediction": np.maximum(pred, 1e-6)}).to_csv(folder / "shm_predictions.csv", index=False)


# =============================================================================
# 2. MODEL: the OpenEvolve program exactly as it was fitted for the submissions (unmodified)
# =============================================================================
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
    # Average two smooth multiscale kernels to reduce sensitivity to the
    # small-sample mutual-information feature selection and kernel bandwidth.
    ms_a = _ker(ms_tr, ytr, ms_te, 20, 0.01, 0.01, 0.01, 0.5)
    ms_b = _ker(ms_tr, ytr, ms_te, 10, 0.03, 0.03, 0.01, 0.5)
    ms = 0.70 * ms_a + 0.30 * ms_b
    rel = _ker(gen_tr, ytr, gen_te, 5, 0.03, 0.03, 0.01, 0.5)
    gen = _ker(gen_tr, ytr, gen_te, 5, 0.1, 0.1, 1.0, 0.0)
    rain = _ker(rain_tr, ytr, rain_te, 3, 0.1, 0.1, 1.0, 0.0)
    tmp = _svr(tmp_tr, ytr, tmp_te)
    # A heavily regularized linear log-target view reduces extrapolation error
    # when the nonlinear kernels become unstable on a small training fold.
    joined_tr = np.c_[gen_tr, rain_tr, tmp_tr]
    joined_te = np.c_[gen_te, rain_te, tmp_te]
    sel = SelectKBest(lambda a, b: mutual_info_regression(a, b, random_state=0),
                      k=min(12, joined_tr.shape[1], len(ytr) - 1)).fit(
                          joined_tr, ytr)
    lin = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    lin.fit(sel.transform(joined_tr), np.log1p(ytr))
    linear = np.maximum(np.expm1(lin.predict(sel.transform(joined_te))), 1e-6)
    nonlinear = 0.4 * ms + 0.6 * (0.4 * rel + 0.42 * gen + 0.108 * rain + 0.072 * tmp)
    # Increase the regularized linear contribution to reduce kernel extrapolation
    # variance on small cross-validation folds.
    return 0.88 * nonlinear + 0.12 * linear


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
        # Five folds provide more representative training-set sizes for the
        # residual calibrator while preserving strictly out-of-fold estimates.
        for a, b in KFold(5, shuffle=True, random_state=1701).split(A):
            oof[b] = _backbone(A[a], y[a], A[b])
        residual = np.log(np.maximum(y, 1e-8)) - np.log(np.maximum(oof, 1e-8))
        z = np.c_[_cal_features(A), np.log(np.maximum(oof, 1e-8))]
        # Stronger shrinkage improves stability of the fold-local residual model.
        self._cal = make_pipeline(StandardScaler(), Ridge(alpha=0.02)).fit(z, residual)

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        A_te = np.nan_to_num(np.vstack([np.asarray(x, dtype=float).ravel() for x in X]))
        raw = _backbone(self._A, self._y, A_te)
        z = np.c_[_cal_features(A_te), np.log(np.maximum(raw, 1e-8))]
        corr = np.clip(self._cal.predict(z), -1.25, 1.25)
        # Apply a conservative multiplicative correction for MAPE robustness.
        return np.maximum(raw * np.exp(0.60 * corr), 1e-6).astype(float)
# EVOLVE-BLOCK-END


# =============================================================================
# 3. HARD-LABEL RETRAINING (second output)
# =============================================================================
# Confidence = member agreement: N_COPIES copies of the model, each fitted on a stratified 80% subsample of the labelled
# traces, predict the held-out traces; a trace's confidence is the consistency (1 / spread of log-predictions) of the
# copies.  The RETRAIN_CUTOFF fraction of most-consistent traces is pseudo-labelled with the main model's predicted damage,
# the model is refitted on labelled + pseudo-labelled traces and predicts all held-out traces again.  The cutoff (keep the
# most consistent half) was tuned by 3-fold CV on the labelled data.
RETRAIN_CUTOFF = 0.5
N_COPIES = 10
COPY_SEED = 7
_FLAG_THR = None     # "flagged" = high-damage traces (top quartile of the labelled damage values), set in main()


def _is_flagged(y):
    return np.asarray(y, dtype=float) >= _FLAG_THR


def _subsample_indices(y, rng):
    flag = _is_flagged(y)
    pos, neg = np.flatnonzero(flag), np.flatnonzero(~flag)
    kp, kn = max(1, int(round(len(pos) * 0.8))), max(1, int(round(len(neg) * 0.8)))
    return [np.sort(np.concatenate([rng.choice(pos, kp, replace=False), rng.choice(neg, kn, replace=False)]))
            for _ in range(5)]


def _copies(X_lab, y, X_test):
    rng = np.random.default_rng(COPY_SEED)
    subsets = []
    while len(subsets) < N_COPIES:
        subsets += _subsample_indices(y, rng)
    outs = []
    for s in subsets[:N_COPIES]:
        m = Model()
        m.fit([X_lab[i] for i in s], y[s])
        outs.append(np.asarray(m.predict(X_test)))
    return outs


def _retrain_reg(X_lab, y, X_test, stage1, copies):
    conf = -np.log(np.maximum(np.asarray(copies, dtype=float), 1e-6)).std(axis=0)
    k = max(1, int(round(len(conf) * RETRAIN_CUTOFF)))
    chosen = np.argsort(-conf)[:k]
    m = Model()
    m.fit(list(X_lab) + [X_test[i] for i in chosen], np.concatenate([y, stage1[chosen]]))
    return np.exp(np.log(np.maximum(np.asarray(m.predict(X_test), dtype=float), 1e-6))), chosen


# =============================================================================
# 4. DRIVER: fit on ALL labelled data, predict the held-out set, write both variants
# =============================================================================

def main():
    global _FLAG_THR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True, help="the organisers' 02_Datasets folder")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--jobs", type=int, default=4, help="parallel workers for feature extraction")
    a = ap.parse_args()
    out = Path(a.out)
    X_lab, y, X_test, ids = load_data(a.data_root, a.jobs)
    _FLAG_THR = float(np.quantile(y, 0.75))
    print(f"{len(X_lab)} labelled traces (all used), {len(X_test)} held-out traces")
    model = Model()
    model.fit(X_lab, y)
    stage1 = np.exp(np.log(np.maximum(np.asarray(model.predict(X_test), dtype=float), 1e-6)))
    write_csv(out / "no_hard_label_retraining", ids, stage1)
    final, chosen = _retrain_reg(X_lab, y, X_test, stage1, _copies(X_lab, y, X_test))
    write_csv(out / "with_hard_label_retraining", ids, final)
    print(f"hard-label retraining: {len(chosen)}/{len(X_test)} traces pseudo-labelled (keep fraction {RETRAIN_CUTOFF}); "
          f"mean |relative change| {np.mean(np.abs(final - stage1) / stage1) * 100:.2f}%")


if __name__ == "__main__":
    main()
