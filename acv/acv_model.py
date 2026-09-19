#!/usr/bin/env python3
"""ACV: refrigerant-leak car ranking (data preprocessing + model + training, one file)

Ranks the 8 cars of a case from most to least likely faulty with a pairwise faulty-vs-normal logistic
ranker over per-signal descriptors that are centred on the median of the car's own case.

Usage
    python acv_model.py --data-root <path to the organisers' 02_Datasets folder> --out <output folder> [--jobs N]

Writes (organiser format, ready to zip):
    <out>/no_hard_label_retraining/acv_predictions.csv    model fitted on ALL labelled data, predicting the held-out set
    <out>/with_hard_label_retraining/acv_predictions.csv  the same model retrained on all labelled data + the confident held-out
                                            items (hard pseudo-labels, cutoff tuned on the labelled data)

Requirements (versions the submitted predictions were produced with):
    python 3.12, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.2, pandas 2.2.3, joblib
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence
import re as _re
import numpy as np
import pandas as pd


# =============================================================================
# 1. DATA PREPROCESSING (raw organiser files -> model inputs)
# =============================================================================
# Each ACV workbook is one CASE: 8 cars with exactly one refrigerant-leak fault.  Cases expose different signal
# subsets (five training cases and the held-out case share the same 8 signals; training case 4 exposes 63 unrelated
# ones), so a car is described by per-signal summary statistics centred on the median of its own case.
_ACV_PAT = _re.compile(r"Car (\d+) - (.*)")


def acv_case_features(path) -> pd.DataFrame:
    """Per-car, per-signal descriptors of one ACV case, centred on the case median
    (mirrors `case_rows` in nebulax src/ps3_acv_probe.py).  Columns: case, car, then
    `<signal>|mean|std|missing|delta` for every signal the case exposes."""
    from pathlib import Path
    path = Path(path)
    df = pd.read_excel(path)
    by = {}
    for col in df.columns:
        m = _ACV_PAT.fullmatch(str(col))
        if not m:
            continue
        car, s = m.groups()
        by.setdefault(car, {})[s] = pd.to_numeric(df[col], errors="coerce")
    rows = []
    for car, cols in sorted(by.items()):
        r = {"case": path.name, "car": car}
        for s, x in cols.items():
            a = x.to_numpy(float)
            valid = a[np.isfinite(a)]
            if not len(valid):
                continue
            r[f"{s}|mean"] = np.mean(valid)
            r[f"{s}|std"] = np.std(valid)
            r[f"{s}|missing"] = 1 - len(valid) / len(a)
            r[f"{s}|delta"] = valid[-1] - valid[0]
        rows.append(r)
    d = pd.DataFrame(rows)
    meta = d[["case", "car"]]
    x = d.drop(columns=["case", "car"]).apply(pd.to_numeric, errors="coerce")
    x = x.groupby(d.case).transform(lambda z: z - z.median())
    return pd.concat([meta, x], axis=1)


def load_data(data_root, jobs=1):
    d = Path(data_root) / "ACV"
    lab = pd.read_csv(d / "Train_Labels.csv")
    truth = dict(zip(lab.filename, lab.faulty_car.astype(int)))
    tr_files = sorted((d / "Train").glob("*.xlsx"))
    te_files = sorted((d / "Test").glob("*.xlsx"))
    tr = [acv_case_features(f) for f in tr_files]
    te = [acv_case_features(f) for f in te_files]
    cols = sorted({c for f in tr + te for c in f.columns if c not in ("case", "car")})   # union of all signals
    X_lab, y, X_test, ids = [], [], [], []
    for f, df in zip(tr_files, tr):
        for r in df.reindex(columns=["car"] + cols).itertuples(index=False):
            X_lab.append(np.asarray(r[1:], dtype=np.float64))          # NaN = signal not exposed by this case
            y.append(int(int(r[0]) == truth[f.name]))                  # 1 = the faulty car of its case
    for f, df in zip(te_files, te):
        for r in df.reindex(columns=["car"] + cols).itertuples(index=False):
            X_test.append(np.asarray(r[1:], dtype=np.float64))
            ids.append((f.name, str(r[0]).zfill(2)))
    return X_lab, np.asarray(y, dtype=int), X_test, ids


def write_csv(folder, ids, scores):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for fname in dict.fromkeys(f for f, _ in ids):
        idx = [i for i, (f, _) in enumerate(ids) if f == fname]
        order = sorted(idx, key=lambda i: -scores[i])                  # most -> least likely faulty
        rows.append({"file_id": fname, "ranked_cars": "|".join(ids[i][1] for i in order)})
    pd.DataFrame(rows).to_csv(folder / "acv_predictions.csv", index=False)


# =============================================================================
# 2. MODEL: the OpenEvolve program exactly as it was fitted for the submissions (unmodified)
# =============================================================================
import warnings
from typing import Sequence

import numpy as np

# EVOLVE-BLOCK-START
class Model:
    def __init__(self) -> None:
        self._F = None
        self._y = None

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        self._F = np.vstack([np.asarray(x, dtype=float) for x in X])
        self._y = np.asarray(y).astype(int)

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        Fp = np.vstack([np.asarray(x, dtype=float) for x in X])
        # only signals the scored case actually exposes, and that training also observed
        obs = np.flatnonzero((~np.isnan(Fp)).any(0) & (~np.isnan(self._F)).any(0))
        if len(obs) == 0 or (self._y == 1).sum() == 0:
            return np.zeros(len(Fp))
        Ftr = np.nan_to_num(self._F[:, obs])
        pos, neg = np.flatnonzero(self._y == 1), np.flatnonzero(self._y == 0)
        diffs, lab = [], []
        for p in pos:                      # directed faulty-vs-normal comparisons
            for n in neg:
                d = Ftr[p] - Ftr[n]
                diffs += [d, -d]
                lab += [1, 0]
        pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=min(5, len(obs), len(diffs) - 1)),
                             LogisticRegression(C=0.003, class_weight="balanced", max_iter=5000))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(np.asarray(diffs), np.asarray(lab))
            return np.asarray(pipe.decision_function(np.nan_to_num(Fp[:, obs])), dtype=float)
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
    return np.asarray(y) == 1


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


def _retrain_rank(X_lab, y, X_test, stage1, copies, groups_test):
    """Confidence of a case = fraction of copies whose top-ranked car equals the main model's top-ranked car;
    a confident case is pseudo-labelled as a whole (top car = faulty, the others = healthy)."""
    stage1 = np.asarray(stage1, dtype=float)
    conf = np.zeros(len(stage1))
    pseudo = np.zeros(len(stage1), dtype=int)
    for g in np.unique(groups_test):
        idx = np.flatnonzero(groups_test == g)
        top = idx[np.argmax(stage1[idx])]
        pseudo[top] = 1
        conf[idx] = np.mean([idx[np.argmax(np.asarray(o, dtype=float)[idx])] == top for o in copies])
    chosen = np.flatnonzero(conf >= RETRAIN_CUTOFF - 1e-12)
    if len(chosen) == 0:
        return stage1, chosen
    rows = np.flatnonzero(np.isin(groups_test, np.unique(groups_test[chosen])))
    m = Model()
    m.fit(list(X_lab) + [X_test[i] for i in rows], np.concatenate([y, pseudo[rows]]))
    return np.asarray(m.predict(X_test), dtype=float), chosen


# =============================================================================
# 4. DRIVER: fit on ALL labelled data, predict the held-out set, write both variants
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True, help="the organisers' 02_Datasets folder")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--jobs", type=int, default=1, help="unused (kept for a uniform command line)")
    a = ap.parse_args()
    out = Path(a.out)
    X_lab, y, X_test, ids = load_data(a.data_root)
    groups_test = np.array([sorted({f for f, _ in ids}).index(f) for f, _ in ids])
    print(f"{len(X_lab)} labelled cars (all used), {len(X_test)} held-out cars")
    model = Model()
    model.fit(X_lab, y)
    stage1 = np.asarray(model.predict(X_test), dtype=float)
    write_csv(out / "no_hard_label_retraining", ids, stage1)
    final, chosen = _retrain_rank(X_lab, y, X_test, stage1, _copies(X_lab, y, X_test), groups_test)
    write_csv(out / "with_hard_label_retraining", ids, final)
    print(f"hard-label retraining: {len(chosen)}/{len(X_test)} cars in confident cases (cutoff {RETRAIN_CUTOFF})")


if __name__ == "__main__":
    main()
