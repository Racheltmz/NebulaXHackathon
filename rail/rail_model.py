#!/usr/bin/env python3
"""Rail corrugation: Normal / Side I / Side II classifier (data preprocessing + model + training, one file)

Turns each 10 000-row, 128-channel recording into 415 label-free descriptors (spectral / amplitude / Side-I vs
Side-II contrasts / tachometer-phase) and classifies it with a blend of six regularised views plus a
cross-validated decision-bias search that maximises macro-F1.

Usage
    python rail_model.py --data-root <path to the organisers' 02_Datasets folder> --out <output folder> [--jobs N]

Writes (organiser format, ready to zip):
    <out>/no_hard_label_retraining/rail_predictions.csv    model fitted on ALL labelled data, predicting the held-out set
    <out>/with_hard_label_retraining/rail_predictions.csv  the same model retrained on all labelled data + the confident held-out
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
from scipy.signal import find_peaks  # noqa: F401
from scipy.stats import kurtosis, skew
import numpy as np
import pandas as pd


# =============================================================================
# 1. DATA PREPROCESSING (raw organiser files -> model inputs)
# =============================================================================
LABEL_ORDER = ["Train93.csv", "Train31.csv", "Train24.csv", "Train208.csv", "Train4.csv", "Train133.csv", "Train104.csv", "Train117.csv", "Train51.csv", "Train29.csv", "Train7.csv", "Train73.csv", "Train77.csv", "Train180.csv", "Train186.csv", "Train221.csv", "Train138.csv", "Train242.csv", "Train60.csv", "Train225.csv", "Train216.csv", "Train248.csv", "Train178.csv", "Train89.csv", "Train101.csv", "Train122.csv", "Train257.csv", "Train262.csv", "Train9.csv", "Train140.csv", "Train212.csv", "Train236.csv", "Train205.csv", "Train86.csv", "Train224.csv", "Train245.csv", "Train68.csv", "Train1.csv", "Train132.csv", "Train131.csv", "Train207.csv", "Train94.csv", "Train43.csv", "Train204.csv", "Train87.csv", "Train25.csv", "Train264.csv", "Train166.csv", "Train99.csv", "Train139.csv", "Train50.csv", "Train271.csv", "Train206.csv", "Train203.csv", "Train249.csv", "Train42.csv", "Train185.csv", "Train52.csv", "Train30.csv", "Train192.csv", "Train3.csv", "Train112.csv", "Train230.csv", "Train58.csv", "Train120.csv", "Train48.csv", "Train227.csv", "Train251.csv", "Train111.csv", "Train253.csv", "Train196.csv", "Train145.csv", "Train17.csv", "Train16.csv", "Train49.csv", "Train222.csv", "Train83.csv", "Train215.csv", "Train255.csv", "Train164.csv", "Train149.csv", "Train260.csv", "Train175.csv", "Train154.csv", "Train113.csv", "Train26.csv", "Train220.csv", "Train143.csv", "Train66.csv", "Train63.csv", "Train226.csv", "Train147.csv", "Train11.csv", "Train211.csv", "Train183.csv", "Train235.csv", "Train13.csv", "Train142.csv", "Train170.csv", "Train47.csv", "Train190.csv", "Train18.csv", "Train202.csv", "Train98.csv", "Train174.csv", "Train191.csv", "Train116.csv", "Train37.csv", "Train169.csv", "Train173.csv", "Train150.csv", "Train6.csv", "Train155.csv", "Train193.csv", "Train57.csv", "Train152.csv", "Train136.csv", "Train76.csv", "Train213.csv", "Train10.csv", "Train33.csv", "Train137.csv", "Train146.csv", "Train121.csv", "Train27.csv", "Train156.csv", "Train267.csv", "Train268.csv", "Train85.csv", "Train135.csv", "Train2.csv", "Train160.csv", "Train40.csv", "Train107.csv", "Train12.csv", "Train167.csv", "Train8.csv", "Train200.csv", "Train244.csv", "Train228.csv", "Train209.csv", "Train56.csv", "Train162.csv", "Train261.csv", "Train269.csv", "Train188.csv", "Train19.csv", "Train241.csv", "Train90.csv", "Train182.csv", "Train102.csv", "Train105.csv", "Train272.csv", "Train219.csv", "Train100.csv", "Train239.csv", "Train61.csv", "Train217.csv", "Train23.csv", "Train129.csv", "Train184.csv", "Train246.csv", "Train210.csv", "Train82.csv", "Train22.csv", "Train84.csv", "Train103.csv", "Train70.csv", "Train238.csv", "Train158.csv", "Train123.csv", "Train65.csv", "Train34.csv", "Train15.csv", "Train88.csv", "Train69.csv", "Train118.csv", "Train80.csv", "Train266.csv", "Train41.csv", "Train258.csv", "Train229.csv", "Train232.csv", "Train171.csv", "Train97.csv", "Train96.csv", "Train20.csv", "Train125.csv", "Train21.csv", "Train148.csv", "Train53.csv", "Train45.csv", "Train28.csv", "Train195.csv", "Train55.csv", "Train181.csv", "Train39.csv", "Train46.csv", "Train233.csv", "Train270.csv", "Train177.csv", "Train64.csv", "Train194.csv", "Train79.csv", "Train134.csv", "Train179.csv", "Train95.csv", "Train243.csv", "Train263.csv", "Train119.csv", "Train198.csv", "Train108.csv", "Train151.csv", "Train144.csv", "Train214.csv", "Train237.csv", "Train74.csv", "Train128.csv", "Train250.csv", "Train78.csv", "Train254.csv", "Train110.csv", "Train72.csv", "Train14.csv", "Train106.csv", "Train231.csv", "Train172.csv", "Train36.csv", "Train218.csv", "Train5.csv", "Train75.csv", "Train165.csv", "Train176.csv", "Train234.csv", "Train259.csv", "Train115.csv", "Train265.csv", "Train44.csv", "Train240.csv", "Train92.csv", "Train157.csv", "Train127.csv", "Train71.csv", "Train114.csv", "Train62.csv", "Train32.csv", "Train81.csv", "Train124.csv", "Train201.csv", "Train59.csv", "Train130.csv", "Train256.csv", "Train153.csv", "Train223.csv", "Train168.csv", "Train197.csv", "Train141.csv", "Train35.csv", "Train163.csv", "Train38.csv", "Train67.csv", "Train199.csv", "Train109.csv", "Train189.csv", "Train159.csv", "Train126.csv", "Train91.csv", "Train161.csv", "Train54.csv", "Train252.csv", "Train247.csv", "Train187.csv"]
# ^ order of the labelled recordings (a fixed seed-7 shuffle).  The model's internal cross-validation folds depend on
# row order, so the order used for the submitted predictions is kept to reproduce them exactly.


# ---- per-recording features: 129 columns = [tachometer, then 8 cars x 8 positions x (vibration, shock)] x 10 000 rows.
# Side I = positions 0,2,4,6 ; Side II = positions 1,3,5,7.  Every transform is label-free and per recording.
def rail_v1(df: pd.DataFrame) -> dict:
    x = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    vib = x[:, 0::2].reshape(len(x), 8, 8)
    shock = x[:, 1::2].reshape(len(x), 8, 8)
    out = {"n_rows": float(len(x)), "speed_mean": float(df.iloc[:, 0].mean())}
    for side, pos in (("side1", [0, 2, 4, 6]), ("side2", [1, 3, 5, 7])):
        for kind, a in (("vib", vib[:, :, pos]), ("shock", shock[:, :, pos])):
            z = a.reshape(len(x), -1)
            rms = np.sqrt(np.mean(z * z, axis=0))
            absz = np.abs(z)
            out[f"{side}_{kind}_rms_mean"] = float(rms.mean())
            out[f"{side}_{kind}_rms_max"] = float(rms.max())
            out[f"{side}_{kind}_p99_mean"] = float(np.quantile(absz, .99, axis=0).mean())
            out[f"{side}_{kind}_crest_mean"] = float((absz.max(0) / (rms + 1e-9)).mean())
            out[f"{side}_{kind}_kurt_mean"] = float(np.mean(kurtosis(z, axis=0, fisher=False, bias=False)))
            out[f"{side}_{kind}_skew_abs_mean"] = float(np.mean(np.abs(skew(z, axis=0, bias=False))))
            fft = np.abs(np.fft.rfft(z - z.mean(0), axis=0)) ** 2
            total = fft[1:].sum(0) + 1e-9
            out[f"{side}_{kind}_spec_peak_ratio"] = float(np.max(fft[1:], axis=0).mean() / total.mean())
            out[f"{side}_{kind}_high_ratio"] = float(fft[int(len(fft) * .25):].sum() / fft[1:].sum())
    for suffix in ("vib_rms_mean", "vib_rms_max", "vib_p99_mean", "vib_crest_mean", "vib_kurt_mean",
                   "vib_spec_peak_ratio", "vib_high_ratio", "shock_rms_mean", "shock_rms_max", "shock_p99_mean",
                   "shock_crest_mean", "shock_kurt_mean", "shock_spec_peak_ratio", "shock_high_ratio"):
        out[f"diff_{suffix}"] = out[f"side1_{suffix}"] - out[f"side2_{suffix}"]
    return out


def rail_extra(df: pd.DataFrame) -> dict:
    x = df.iloc[:, 1:].to_numpy(float)
    vib = x[:, 0::2].reshape(len(x), 8, 8)
    shock = x[:, 1::2].reshape(len(x), 8, 8)
    o = {}
    for side, ps in (("side1", [0, 2, 4, 6]), ("side2", [1, 3, 5, 7])):
        for kind, a in (("vib", vib), ("shock", shock)):
            z = a[:, :, ps].reshape(len(x), -1)
            z = z - z.mean(0)
            p = np.abs(np.fft.rfft(z, axis=0)) ** 2
            f = np.fft.rfftfreq(len(x), 1 / 10000)
            p[0] = 0
            total = p.sum(0) + 1e-9
            ix = p.argmax(0)
            o[f"{side}_{kind}_peak_hz_mean"] = float(np.average(f[ix], weights=np.maximum(p.max(0), 1e-9)))
            o[f"{side}_{kind}_peak_hz_median"] = float(np.median(f[ix]))
            o[f"{side}_{kind}_peak_ratio_median"] = float(np.median(p.max(0) / total))
            for lo, hi in ((0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 5000)):
                q = p[(f >= lo) & (f < hi)].sum() / total.sum()
                o[f"{side}_{kind}_band_{lo}_{hi}"] = float(q)
    for k in list(o):
        if k.startswith("side1_") and k.replace("side1_", "side2_") in o:
            o["diff_" + k[6:]] = o[k] - o[k.replace("side1_", "side2_")]
    return o


def rail_v3(df: pd.DataFrame) -> dict:
    r = rail_v1(df)
    r.update(rail_extra(df))
    return r


def rail_relative(v3: dict) -> dict:
    """Scale-normalised Side-I/Side-II contrasts (mirrors ps3_rail_relative_features.py)."""
    base = {k: v for k, v in v3.items() if k != "n_rows"}
    out = dict(base)
    for c in list(base):
        if c.startswith("side1_"):
            other = "side2_" + c[len("side1_"):]
            if other in base:
                a, b = float(base[c]), float(base[other])
                scale = abs(a) + abs(b) + 1e-8
                q = c[len("side1_"):]
                out[f"rel_diff_{q}"] = (a - b) / scale
                out[f"log_ratio_{q}"] = np.sign(a) * np.log1p(abs(a)) - np.sign(b) * np.log1p(abs(b))
                out[f"side_sum_{q}"] = a + b
                out[f"side_max_{q}"] = max(a, b)
    return out


NBINS = 64


def rail_phase(df: pd.DataFrame) -> dict:
    t = df.iloc[:, 0].to_numpy(float)
    x = df.iloc[:, 1:].to_numpy(float)
    n = len(x)
    edges = np.flatnonzero(np.diff(t) != 0) + 1
    if len(edges) >= 4:
        left, right = int(edges[0]), int(edges[-1])
        anchors = np.r_[left, edges[1:-1], right]
        phase = np.interp(np.arange(left, right), anchors, np.arange(len(anchors), dtype=float))
        phase = (phase - np.floor(phase))
        xx = x[left:right]
    else:
        phase = np.linspace(0, 1, n, endpoint=False)
        xx = x
    bi = np.minimum((phase * NBINS).astype(int), NBINS - 1)
    vib = xx[:, 0::2].reshape(len(xx), 8, 8)
    shock = xx[:, 1::2].reshape(len(xx), 8, 8)
    out = {'tacho_transitions': float(len(edges)),
           'tacho_speed_mps': float(max(len(edges) - 1, 0) / (2 * 90) * .85)}
    profiles = {}
    for side, pos in (('s1', [0, 2, 4, 6]), ('s2', [1, 3, 5, 7])):
        for kind, arr in (('v', vib), ('h', shock)):
            z = arr[:, :, pos].reshape(len(xx), -1)
            prof = np.zeros((NBINS, z.shape[1]))
            cnt = np.bincount(bi, minlength=NBINS).astype(float)
            for j in range(z.shape[1]):
                prof[:, j] = np.bincount(bi, weights=np.abs(z[:, j]), minlength=NBINS)
            prof /= np.maximum(cnt[:, None], 1)
            profiles[(side, kind)] = prof.mean(1)
            for stat, a in (('mean', prof.mean(1)), ('std', prof.std(1)),
                            ('peak', prof.max(1)), ('q90', np.quantile(prof, .9, axis=1))):
                out[f'{side}_{kind}_{stat}_mean'] = float(a.mean())
                out[f'{side}_{kind}_{stat}_std'] = float(a.std())
                out[f'{side}_{kind}_{stat}_max'] = float(a.max())
    for kind in ('v', 'h'):
        d = profiles[('s1', kind)] - profiles[('s2', kind)]
        s = profiles[('s1', kind)] + profiles[('s2', kind)] + 1e-9
        for h in (1, 2, 3, 4, 8):
            c = np.cos(2 * np.pi * h * np.arange(NBINS) / NBINS)
            q = np.sin(2 * np.pi * h * np.arange(NBINS) / NBINS)
            out[f'd_{kind}_cos{h}'] = float(np.dot(d, c) / NBINS)
            out[f'd_{kind}_sin{h}'] = float(np.dot(d, q) / NBINS)
            out[f'r_{kind}_cos{h}'] = float(np.dot(d / s, c) / NBINS)
            out[f'r_{kind}_sin{h}'] = float(np.dot(d / s, q) / NBINS)
        out[f'd_{kind}_abs_mean'] = float(np.mean(np.abs(d)))
        out[f'd_{kind}_abs_p90'] = float(np.quantile(np.abs(d), .9))
        out[f'd_{kind}_signed_std'] = float(np.std(d))
    return out


def rail_vector(df):
    """[ v3 (96) | relative (223) | phase (96) ] -- column order = insertion order of the three feature dicts."""
    v3 = rail_v3(df)
    rel = rail_relative(v3)
    ph = rail_phase(df)
    vec = np.array(list(v3.values()) + list(rel.values()) + list(ph.values()), dtype=np.float64)
    assert (len(v3), len(rel), len(ph)) == (96, 223, 96)
    return np.nan_to_num(vec)


def _vector_from_file(path):
    return rail_vector(pd.read_csv(path))


def load_data(data_root, jobs=4):
    base = Path(data_root) / "Rail_Corrugation"
    labels = pd.read_csv(base / "Train_Labels.csv").set_index("filename").label
    X_lab = Parallel(n_jobs=jobs)(delayed(_vector_from_file)(base / "Train" / n) for n in LABEL_ORDER)
    y = np.array([str(labels[n]) for n in LABEL_ORDER])
    test_files = sorted((base / "Test").glob("*.csv"))
    X_test = Parallel(n_jobs=jobs)(delayed(_vector_from_file)(p) for p in test_files)
    return X_lab, y, X_test, [p.name for p in test_files]


def write_csv(folder, ids, pred):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file_id": ids, "prediction": pred}).to_csv(folder / "rail_predictions.csv", index=False)


# =============================================================================
# 2. MODEL: the OpenEvolve program exactly as it was fitted for the submissions (unmodified)
# =============================================================================
import warnings
from typing import Sequence

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_V3, N_REL, N_PHASE = 96, 223, 96
CLASSES = np.array(["Normal", "Side I", "Side II"], dtype=object)

# EVOLVE-BLOCK-START
from sklearn.svm import SVC
_BIAS_GRID = [(bi, bii)
              for bi in (-0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.7)
              for bii in (-0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.7)]
_GRID = [(k, c, w, bi, bii)
         for k in (20, 40, 60, 80)
         for c in (0.003, 0.01, 0.03, 0.1, 0.3)
         for w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)
         for bi, bii in _BIAS_GRID]


def _views(A):
    return A[:, :N_V3], A[:, N_V3:N_V3 + N_REL], A[:, N_V3 + N_REL:]


def _proba(Xtr, ytr, Xte, k, c):
    m = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=min(k, Xtr.shape[1], len(Xtr) - 1)),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)
    out = np.zeros((len(Xte), len(CLASSES)))
    for j, cl in enumerate(m.classes_):
        out[:, list(CLASSES).index(cl)] = p[:, j]
    return out


def _retrieve(Xtr, ytr, Xte, k=5):
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd[sd < 1e-8] = 1.0
    tr = (np.nan_to_num(Xtr, nan=mu) - mu) / sd
    te = (np.nan_to_num(Xte, nan=mu) - mu) / sd
    d = ((te[:, None, :] - tr[None, :, :]) ** 2).mean(axis=2)
    kk = min(k, len(tr))
    nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    ww = 1.0 / (np.sqrt(np.take_along_axis(d, nn, axis=1)) + 1e-6)
    out = np.zeros((len(te), len(CLASSES)))
    for j, cl in enumerate(CLASSES):
        out[:, j] = (
            np.sum(ww * (ytr[nn] == cl), axis=1)
            / max(np.sum(ytr == cl), 1)
        )
    out += 1e-6
    return out / out.sum(axis=1, keepdims=True)


def _hierarchical(Xtr, ytr, Xte):
    """Fault gate followed by a Side-I/Side-II classifier."""
    fault = (np.asarray(ytr) != "Normal").astype(int)
    gate = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=min(80, Xtr.shape[1], len(Xtr) - 1)),
        LogisticRegression(C=0.03, class_weight="balanced", max_iter=5000),
    )
    mask = fault == 1
    side = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=min(80, Xtr.shape[1], max(2, mask.sum() - 1))),
        LogisticRegression(C=0.03, class_weight="balanced", max_iter=5000),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate.fit(Xtr, fault)
        side.fit(Xtr[mask], ytr[mask])
    pg = gate.predict_proba(Xte)[:, 1]
    ps = side.predict_proba(Xte)
    out = np.zeros((len(Xte), len(CLASSES)))
    out[:, 0] = 1.0 - pg
    for j, cl in enumerate(side.classes_):
        out[:, list(CLASSES).index(cl)] = pg * ps[:, j]
    return out


def _sparse_proba(Xtr, ytr, Xte):
    """Nonlinear regularized head for interactions among relative descriptors."""
    m = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=min(80, Xtr.shape[1], len(Xtr) - 1)),
        SVC(
            C=0.7, gamma="scale", probability=True,
            class_weight="balanced", random_state=173,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)
    out = np.zeros((len(Xte), len(CLASSES)))
    for j, cl in enumerate(m.classes_):
        out[:, list(CLASSES).index(cl)] = p[:, j]
    return out


def _base_and_phase(A_tr, y_tr, A_te, phase_kc):
    """Blend dense, sparse, hierarchical, and retrieval views."""
    v_tr, r_tr, p_tr = _views(A_tr)
    v_te, r_te, p_te = _views(A_te)
    joint_tr = np.hstack([v_tr, r_tr])
    joint_te = np.hstack([v_te, r_te])
    base = (
        0.12 * _proba(v_tr, y_tr, v_te, 50, 0.03)
        + 0.35 * _proba(r_tr, y_tr, r_te, 150, 0.01)
        + 0.15 * _proba(joint_tr, y_tr, joint_te, 160, 0.01)
        + 0.10 * _retrieve(r_tr, y_tr, r_te, 5)
        + 0.13 * _hierarchical(r_tr, y_tr, r_te)
        + 0.15 * _sparse_proba(r_tr, y_tr, r_te)
    )
    phase = {kc: _proba(p_tr, y_tr, p_te, kc[0], kc[1]) for kc in phase_kc}
    return base, phase


def _decide(base, phase, w, bi=0.0, bii=0.0):
    q = (1 - w) * base + w * phase
    bias = np.array([0.0, bi, bii])
    return CLASSES[np.argmax(np.log(np.maximum(q, 1e-9)) + bias, axis=1)]


class Model:
    def __init__(self) -> None:
        self._A = None
        self._y = None
        self._cfg = None

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        A = np.nan_to_num(np.vstack([np.asarray(x, dtype=float).ravel() for x in X]))
        y = np.asarray(y).astype(str)
        self._A, self._y = A, y
        counts = np.unique(y, return_counts=True)[1]
        n_splits = int(min(5, counts.min()))
        if n_splits < 2:
            self._cfg = (40, 0.03, 0.2, 0.0, 0.0)
            return
        phase_kc = sorted({(cfg[0], cfg[1]) for cfg in _GRID})
        preds = {cfg: [] for cfg in _GRID}
        for seed in (1701, 2718):
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
            for tr, te in cv.split(A, y):
                base, phase = _base_and_phase(A[tr], y[tr], A[te], phase_kc)
                for (k, c, w, bi, bii) in _GRID:
                    preds[(k, c, w, bi, bii)].append(
                        (te, _decide(base, phase[(k, c)], w, bi, bii)))
        best, best_key = None, None
        for cfg in _GRID:
            yy = np.concatenate([y[idx] for idx, _ in preds[cfg]])
            pp = np.concatenate([pred for _, pred in preds[cfg]]).astype(str)
            key = (f1_score(yy, pp, average="macro"),
                   balanced_accuracy_score(yy, pp))
            if best_key is None or key > best_key:
                best, best_key = cfg, key
        self._cfg = best

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        A_te = np.nan_to_num(np.vstack([np.asarray(x, dtype=float).ravel() for x in X]))
        k, c, w, bi, bii = self._cfg
        base, phase = _base_and_phase(self._A, self._y, A_te, [(k, c)])
        return _decide(base, phase[(k, c)], w, bi, bii).astype(object)
# EVOLVE-BLOCK-END


# =============================================================================
# 3. HARD-LABEL RETRAINING (second output)
# =============================================================================
# Confidence = member agreement: N_COPIES copies of the model, each fitted on a stratified 80% subsample of the
# labelled rows, predict the held-out set; an item's confidence is the fraction of copies that agree with the main
# model's prediction (a cutoff of 0.9 is the "p >= 0.9 or <= 0.1" band).  Only items at/above RETRAIN_CUTOFF are
# pseudo-labelled with the main model's prediction; the model is refitted on labelled + pseudo-labelled rows and
# predicts the whole held-out set again.  The cutoff was tuned by 3-fold CV on the labelled data.
RETRAIN_CUTOFF = 0.6
N_COPIES = 10
COPY_SEED = 7


def _is_flagged(y):
    return np.asarray(y) != 'Normal'


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
