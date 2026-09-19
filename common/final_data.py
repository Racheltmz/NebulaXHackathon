"""Labelled + held-out (test) data for the final ensembles.

Labelled data  = the organisers' Train set (all of it: our former train/test split is re-merged).
Held-out data  = the unlabelled organiser Test files (Door stream, ACV case, Rail 68 files, SHM 16).

Labelled arrays are read from the .npy files the evolution tracks were trained on
(data/ps3_arrays for raw inputs, data/nx_arrays for the Rail/SHM feature-vector inputs,
ps3_arrays door_* rebuilt by data/build_door_fixed.py); held-out inputs are produced from the raw
Test files with EXACTLY the same per-file transforms (validated in scripts/validate_transforms.py).

Env: PS3_RAW_DIR (02_Datasets), PS3_DATA_DIR (ps3_arrays), PS3_NX_DATA_DIR (nx_arrays),
     NEBULAX_DIR (checkout of the ian-classical branch; only its cached-feature CSV *headers* are read
     to fix the feature column order).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import nx_features as nxf

REPO = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("PS3_RAW_DIR", "/mnt/c/Users/ian/Desktop/nebulax/data/ps3_full/02_Datasets"))
ARR = Path(os.environ.get("PS3_DATA_DIR", REPO / "data" / "ps3_arrays"))
NXARR = Path(os.environ.get("PS3_NX_DATA_DIR", REPO / "data" / "nx_arrays"))
NEB = Path(os.environ.get("NEBULAX_DIR", "/mnt/c/Users/ian/Desktop/nebulax"))
ANALYSIS = NEB / "artifacts" / "analysis" / "ps3"


@dataclass
class TaskData:
    name: str
    y_lab: np.ndarray
    X_lab: dict = field(default_factory=dict)      # input_key -> list of arrays (aligned with y_lab)
    X_test: dict = field(default_factory=dict)     # input_key -> list of arrays
    test_ids: list = field(default_factory=list)   # file ids / (start,end) for output
    groups_lab: np.ndarray | None = None           # ACV case ids
    groups_test: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def _concat(split: str, prefix: str, base: Path):
    xs, ys = [], []
    for s in ("train", "test"):
        xs += list(np.load(base / f"{prefix}_{s}_x.npy", allow_pickle=True))
        ys.append(np.load(base / f"{prefix}_{s}_y.npy", allow_pickle=True))
    return xs, np.concatenate(ys)


# ------------------------------------------------------------------ Door ----


def _secs(stamp: str) -> float:
    y, mo, d, h, mi, s, ms = (int(v) for v in stamp.split("-"))
    return datetime(y, mo, d, h, mi, s).timestamp() + ms / 1000.0


def door_gap_segments(datetimes, gap: float = 1.0):
    """(start_row, end_row) inclusive segments of a continuous door stream, cut where the
    inter-row time gap exceeds `gap` seconds.  Integer-millisecond parsing; verified to
    reproduce all 110 training segments exactly (scripts/validate_transforms.py)."""
    sec = np.array([_secs(s) for s in datetimes])
    cuts = np.flatnonzero(np.diff(sec) > gap) + 1
    b = np.r_[0, cuts, len(sec)]
    return [(int(b[i]), int(b[i + 1] - 1)) for i in range(len(b) - 1)]


def _door_numeric(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c != "Datetime"]
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)


def door_data() -> TaskData:
    xs, ys = _concat("all", "door", ARR)
    te = pd.read_csv(RAW / "Door" / "Test.csv")
    num = _door_numeric(te)
    segs = door_gap_segments(te.Datetime)
    X = [num[a:b + 1].T.copy() for a, b in segs]
    ids = [(te.Datetime.iloc[a], te.Datetime.iloc[b]) for a, b in segs]
    return TaskData("door", ys.astype(str), {"raw": xs}, {"raw": X}, ids)


# ------------------------------------------------------------------- ACV ----


def acv_case_arrays(path: Path):
    """[8, T] per car (mean/std/q90/q10 of the car's signals + robust cross-car residuals),
    identical to ps3_prepare.acv().  Returns (list of arrays, list of zero-padded car ids)."""
    d = pd.read_excel(path)
    cars = sorted({c.split(" - ", 1)[0].split()[-1] for c in d.columns if c.startswith("Car ") and " - " in c})
    case_x = []
    for car in cars:
        cols = [c for c in d.columns if c.startswith(f"Car {car} - ")]
        a = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        case_x.append(np.stack([a.mean(1), a.std(1), np.quantile(a, .9, axis=1), np.quantile(a, .1, axis=1)], axis=0))
    stack = np.stack(case_x)
    med = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - med), axis=0) + 1e-5
    return [np.concatenate([a, (a - med) / mad], axis=0) for a in case_x], [str(c).zfill(2) for c in cars]


def acv_nx_matrix():
    """Per-car, per-signal feature vectors (nebulax `case_rows`): returns (X_lab, y_lab, X_test, test_ids, columns).
    Columns are the union over all cases; a signal a case does not expose is NaN (cases expose different
    subsets: five train cases and the test case share 4 signals, train case 4 exposes 32 unrelated ones)."""
    D = RAW / "ACV"
    lab = pd.read_csv(D / "Train_Labels.csv")
    truth = dict(zip(lab.filename, lab.faulty_car.astype(int)))
    tr_files = sorted((D / "Train").glob("*.xlsx"))
    te_files = sorted((D / "Test").glob("*.xlsx"))
    tr = [nxf.acv_case_features(f) for f in tr_files]
    te = [nxf.acv_case_features(f) for f in te_files]
    cols = sorted({c for f in tr + te for c in f.columns if c not in ("case", "car")})
    X_lab, y_lab, X_te, ids = [], [], [], []
    for f, d in zip(tr_files, tr):
        for r in d.reindex(columns=["car"] + cols).itertuples(index=False):
            X_lab.append(np.asarray(r[1:], dtype=np.float64))
            y_lab.append(int(int(r[0]) == truth[f.name]))
    for f, d in zip(te_files, te):
        for r in d.reindex(columns=["car"] + cols).itertuples(index=False):
            X_te.append(np.asarray(r[1:], dtype=np.float64))
            ids.append((f.name, str(r[0]).zfill(2)))
    return X_lab, np.asarray(y_lab, dtype=int), X_te, ids, cols


def acv_data() -> TaskData:
    xs, ys = _concat("all", "acv", ARR)
    groups = np.arange(len(xs)) // 8            # arrays were built case by case, 8 cars each
    assert len(xs) % 8 == 0 and len(xs) == 48
    nx_lab, nx_y, nx_te, nx_ids, _ = acv_nx_matrix()
    assert list(nx_y) == list(ys.astype(int)), "acv raw/nx label order mismatch"
    test_files = sorted((RAW / "ACV" / "Test").glob("*.xlsx"))
    X, ids, gt = [], [], []
    for gi, f in enumerate(test_files):
        arrs, cars = acv_case_arrays(f)
        X += arrs
        ids += [(f.name, c) for c in cars]
        gt += [gi] * len(arrs)
    assert [i[1] for i in ids] == [i[1] for i in nx_ids] and len(nx_te) == len(X), "acv raw/nx test order mismatch"
    return TaskData("acv", ys.astype(int), {"raw": xs, "nx": nx_lab}, {"raw": X, "nx": nx_te}, ids, groups, np.asarray(gt))


# ------------------------------------------------------------------ Rail ----


def rail_compact(a: np.ndarray) -> np.ndarray:
    """[5, T]: speed + mean vibration/shock of Side I and Side II (identical to ps3_prepare.rail)."""
    v = a[:, 1:].reshape(len(a), 8, 8, 2)
    return np.stack([a[:, 0], v[:, :, [0, 2, 4, 6], 0].mean((1, 2)), v[:, :, [0, 2, 4, 6], 1].mean((1, 2)),
                     v[:, :, [1, 3, 5, 7], 0].mean((1, 2)), v[:, :, [1, 3, 5, 7], 1].mean((1, 2))])


def _cols(csv: str, drop) -> list:
    return [c for c in pd.read_csv(ANALYSIS / csv, nrows=0).columns if c not in drop]


def rail_nx_vector(df: pd.DataFrame, layout=None) -> np.ndarray:
    layout = layout or {
        "v3": _cols("rail_file_features_v3.csv", {"file_id", "label"}),
        "rel": _cols("rail_relative_features.csv", {"file_id", "label"}),
        "phase": _cols("rail_phase_features.csv", {"file_id", "label"}),
    }
    v3 = nxf.rail_v3(df)
    rel = nxf.rail_relative(v3)
    ph = nxf.rail_phase(df)
    vec = [v3[c] for c in layout["v3"]] + [rel[c] for c in layout["rel"]] + [ph[c] for c in layout["phase"]]
    return np.nan_to_num(np.asarray(vec, dtype=np.float64))


def rail_data() -> TaskData:
    raw, y = _concat("all", "rail", ARR)
    nx, y2 = _concat("all", "rail", NXARR)
    assert list(y) == list(y2), "rail raw/nx label order mismatch"
    files = sorted((RAW / "Rail_Corrugation" / "Test").glob("*.csv"))
    layout = None
    Xr, Xn, ids = [], [], []
    for f in files:
        df = pd.read_csv(f)
        Xr.append(rail_compact(df.to_numpy(np.float32)))
        Xn.append(rail_nx_vector(df, layout))
        ids.append(f.name)
    return TaskData("rail", y.astype(str), {"raw": raw, "nx": nx}, {"raw": Xr, "nx": Xn}, ids)


# ------------------------------------------------------------------- SHM ----


def shm_nx_vector(x: np.ndarray, layout=None) -> np.ndarray:
    layout = layout or [
        ("shm_multiscale_features.csv", nxf.shm_multiscale), ("shm_features.csv", nxf.shm_generic),
        ("shm_rainflow_features.csv", nxf.shm_rainflow), ("shm_temporal_features.csv", nxf.shm_temporal)]
    parts = []
    for csv, fn in layout:
        cols = _cols(csv, {"file_id", "damage"})
        d = fn(x)
        parts.append(np.asarray([d[c] for c in cols], dtype=np.float64))
    v = np.concatenate(parts)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def shm_data() -> TaskData:
    raw, y = _concat("all", "shm", ARR)
    nx, y2 = _concat("all", "shm", NXARR)
    assert np.allclose(y.astype(float), y2.astype(float)), "shm raw/nx label order mismatch"
    files = sorted((RAW / "SHM" / "Test").glob("*.csv"))
    Xr, Xn, ids = [], [], []
    for f in files:
        Xr.append(pd.read_csv(f).select_dtypes(include="number").to_numpy(np.float32).T)
        Xn.append(shm_nx_vector(np.loadtxt(f, delimiter=",")))
        ids.append(f.name)
    return TaskData("shm", y.astype(float), {"raw": raw, "nx": nx}, {"raw": Xr, "nx": Xn}, ids)


LOADERS = {"door": door_data, "acv": acv_data, "rail": rail_data, "shm": shm_data}
