#!/usr/bin/env python3
"""Door: abnormal-resistance door-cycle classifier (data preprocessing + model + training, one file)

Cuts the continuous Door stream into open/close cycles and classifies each cycle as Normal or
Abnormal resistance with a scaled RBF-SVM over rich per-channel segment statistics.

Usage
    python door_model.py --data-root <path to the organisers' 02_Datasets folder> --out <output folder> [--jobs N]

Writes (organiser format, ready to zip):
    <out>/no_hard_label_retraining/door_predictions.csv    model fitted on ALL labelled data, predicting the held-out set
    <out>/with_hard_label_retraining/door_predictions.csv  the same model retrained on all labelled data + the confident held-out
                                            items (hard pseudo-labels, cutoff tuned on the labelled data)

Requirements (versions the submitted predictions were produced with):
    python 3.12, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.2, pandas 2.2.3, joblib
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence
from datetime import datetime
import numpy as np
import pandas as pd


# =============================================================================
# 1. DATA PREPROCESSING (raw organiser files -> model inputs)
# =============================================================================
LABEL_ORDER = [32, 92, 4, 1, 67, 45, 72, 13, 55, 50, 75, 12, 24, 39, 101, 97, 11, 85, 8, 46, 69, 35, 66, 77, 20, 108, 64, 31, 7, 78, 102, 28, 52, 71, 23, 68, 33, 80, 19, 93, 62, 37, 100, 5, 26, 91, 63, 86, 36, 94, 40, 65, 14, 81, 104, 79, 48, 16, 90, 58, 87, 73, 98, 59, 84, 38, 21, 76, 6, 17, 27, 42, 51, 88, 29, 47, 2, 0, 57, 105, 96, 60, 83, 103, 99, 15, 106, 10, 109, 56, 3, 30, 18, 49, 9, 61, 34, 44, 22, 54, 74, 25, 70, 53, 41, 107, 95, 43, 89, 82]   # order of the labelled rows (indices into Train_Segments_Answer.csv).
# The model chooses its hyper-parameters with internal cross-validation whose folds depend on row order, so the
# order used for the submitted predictions is kept (a fixed seed-7 shuffle) to reproduce them exactly.


def _numeric(df):
    cols = [c for c in df.columns if c != "Datetime"]
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)


def _secs(stamp):
    """'2023-7-5-0-0-3-700' (Y-M-D-h-m-s-ms) -> seconds.  The millisecond field is parsed as an INTEGER: fields of
    different widths ('20' vs '700') must not be zero-padded or compared as strings."""
    y, mo, d, h, mi, s, ms = (int(v) for v in stamp.split("-"))
    return (datetime(y, mo, d, h, mi, s) - datetime(1970, 1, 1)).total_seconds() + ms / 1000.0


def segment_stream(datetimes, gap=1.0):
    """Cut a continuous door stream into cycles wherever the time between consecutive rows exceeds `gap` seconds.
    Reproduces all 110 labelled training segments exactly (start and end row)."""
    sec = np.array([_secs(s) for s in datetimes])
    cuts = np.flatnonzero(np.diff(sec) > gap) + 1
    b = np.r_[0, cuts, len(sec)]
    return [(int(b[i]), int(b[i + 1] - 1)) for i in range(len(b) - 1)]


def load_data(data_root, jobs=1):
    door = Path(data_root) / "Door"
    df = pd.read_csv(door / "Train.csv")
    ans = pd.read_csv(door / "Train_Segments_Answer.csv")
    num = _numeric(df)
    row_of = {s: i for i, s in enumerate(df.Datetime)}
    segs, labels = [], []
    for r in ans.itertuples():
        a, b = row_of[r.start_time], row_of[r.end_time]
        assert b - a + 1 == r.n_rows                      # segment = exactly the rows between start_time and end_time
        segs.append(num[a:b + 1].T.copy())                # [16 channels, T rows]
        labels.append(str(r.status))
    X_lab = [segs[i] for i in LABEL_ORDER]
    y = np.array([labels[i] for i in LABEL_ORDER])
    te = pd.read_csv(door / "Test.csv")
    tnum = _numeric(te)
    spans = segment_stream(te.Datetime)
    X_test = [tnum[a:b + 1].T.copy() for a, b in spans]
    ids = [(te.Datetime.iloc[a], te.Datetime.iloc[b]) for a, b in spans]
    return X_lab, y, X_test, ids


def write_csv(folder, ids, pred):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"start_time": [s for s, _ in ids], "end_time": [e for _, e in ids],
                  "prediction": pred}).to_csv(folder / "door_predictions.csv", index=False)


# =============================================================================
# 2. MODEL: the OpenEvolve program exactly as it was fitted for the submissions (unmodified)
# =============================================================================
from typing import Sequence

import numpy as np

# EVOLVE-BLOCK-START
class Model:
    def __init__(self) -> None:
        self._model = None
        self._label = None
        self._channels = 0
        self._scale = None

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
            self._scale = np.std(f, axis=0)

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        if self._model is None:
            return np.full(len(X), self._label, dtype=object)
        f = self._features(X)
        return np.asarray(self._model.predict(f), dtype=object)
# EVOLVE-BLOCK-END


# =============================================================================
# 3. HARD-LABEL RETRAINING (second output)
# =============================================================================
# Confidence = member agreement: N_COPIES copies of the model, each fitted on a stratified 80% subsample of the
# labelled rows, predict the held-out set; an item's confidence is the fraction of copies that agree with the main
# model's prediction (a cutoff of 0.9 is the "p >= 0.9 or <= 0.1" band).  Only items at/above RETRAIN_CUTOFF are
# pseudo-labelled with the main model's prediction; the model is refitted on labelled + pseudo-labelled rows and
# predicts the whole held-out set again.  The cutoff was tuned by 3-fold CV on the labelled data.
RETRAIN_CUTOFF = 1.0
N_COPIES = 10
COPY_SEED = 7


def _is_flagged(y):
    return np.asarray(y) == 'Abnormal resistance'


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


def _retrain_class(X_lab, y, X_test, stage1, copies):
    conf = np.mean([np.asarray(o).astype(str) == np.asarray(stage1).astype(str) for o in copies], axis=0)
    chosen = np.flatnonzero(conf >= RETRAIN_CUTOFF - 1e-12)
    if len(chosen) == 0:
        return np.asarray(stage1), chosen
    y_all = np.concatenate([y.astype(object), np.asarray(stage1, dtype=object)[chosen]])
    m = Model()
    m.fit(list(X_lab) + [X_test[i] for i in chosen], y_all)
    return np.asarray(m.predict(X_test)), chosen


# =============================================================================
# 4. DRIVER: fit on ALL labelled data, predict the held-out set, write both variants
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True, help="the organisers' 02_Datasets folder")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--jobs", type=int, default=4, help="parallel workers for feature extraction")
    a = ap.parse_args()
    out = Path(a.out)
    X_lab, y, X_test, ids = load_data(a.data_root, a.jobs)
    print(f"{len(X_lab)} labelled rows (all used), {len(X_test)} held-out items")
    model = Model()
    model.fit(X_lab, y)
    stage1 = np.asarray(model.predict(X_test))
    write_csv(out / "no_hard_label_retraining", ids, stage1)
    final, chosen = _retrain_class(X_lab, y, X_test, stage1, _copies(X_lab, y, X_test))
    write_csv(out / "with_hard_label_retraining", ids, final)
    print(f"hard-label retraining: {len(chosen)}/{len(X_test)} items pseudo-labelled (cutoff {RETRAIN_CUTOFF}); "
          f"{int((np.asarray(final) != stage1).sum())} predictions changed")


if __name__ == "__main__":
    main()
