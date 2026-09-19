"""Fit the vendored `ian-model` models against the bundled PS3 training data.

That branch ships no trained weights — every model retrains inside `fit()` — and fitting at
request time is not an option (Rail alone is 272 files x 17 MB). So this script fits each model
once against the full training set and writes a joblib artifact that the app loads and only calls
`predict` on.

Run from the repo root after changing a model or the featurizer:

    python app/backend/scripts/fit_models.py            # all subsystems
    python app/backend/scripts/fit_models.py rail       # just one

Artifacts land in `app/backend/ml_artifacts/` as `<subsystem>_classical.joblib` — deliberately
distinct from any hand-trained artifact already there (e.g. SHM's `shm_model.joblib`, which comes
from the team's own notebook), so fitting can never clobber one. They unpickle by importing the
model class from `ml.models.*`, so those modules must stay importable at the same path.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
DATA = ROOT / "PS3" / "02_Datasets"
ARTIFACTS = BACKEND / "ml_artifacts"

sys.path.insert(0, str(BACKEND))

from ml import featurize  # noqa: E402
from ml.models import door_classical, rail_classical, shm_classical  # noqa: E402


def _log(msg):
    print(msg, flush=True)


def fit_rail():
    """272 recordings -> [5, T] each. Labels: Normal / Side I / Side II."""
    base = DATA / "Rail_Corrugation"
    labels = pd.read_csv(base / "Train_Labels.csv").set_index("filename")
    X, y = [], []
    for i, (fn, label) in enumerate(labels.label.items(), 1):
        X.append(featurize.rail_array((base / "Train" / fn).read_bytes()))
        y.append(str(label))
        if i % 50 == 0:
            _log(f"    {i}/{len(labels)} recordings")
    _log(f"    class counts: {pd.Series(y).value_counts().to_dict()}")
    model = rail_classical.Model()
    model.fit(X, np.asarray(y, dtype=object))
    return model, {"n_train": len(X), "classes": sorted(set(y))}


def fit_shm():
    """64 traces -> [C, T] each. Target: continuous cumulative damage."""
    base = DATA / "SHM"
    labels = pd.read_csv(base / "Train_Labels.csv").set_index("filename")
    X, y = [], []
    for i, (fn, damage) in enumerate(labels.damage.items(), 1):
        X.append(featurize.shm_array((base / "Train" / fn).read_bytes()))
        y.append(float(damage))
        if i % 20 == 0:
            _log(f"    {i}/{len(labels)} traces")
    model = shm_classical.Model()
    model.fit(X, np.asarray(y, dtype=float))
    return model, {"n_train": len(X), "target_range": [min(y), max(y)]}


def fit_door():
    """110 labelled segments cut out of the continuous training stream."""
    base = DATA / "Door"
    df = pd.read_csv(base / "Train.csv")
    ans = pd.read_csv(base / "Train_Segments_Answer.csv")
    pairs = list(zip(ans.start_time, ans.end_time))
    arrays = featurize.door_segment_arrays(df, pairs)
    X, y = [], []
    for arr, status in zip(arrays, ans.status):
        # The ian-model prep drops segments too short to carry a usable signal.
        if arr.shape[-1] > 4:
            X.append(arr)
            y.append(str(status))
    _log(f"    {len(X)} usable segments of {len(ans)}; {pd.Series(y).value_counts().to_dict()}")
    model = door_classical.Model()
    model.fit(X, np.asarray(y, dtype=object))
    return model, {"n_train": len(X), "classes": sorted(set(y))}


FITTERS = {"rail": fit_rail, "shm": fit_shm, "door": fit_door}
# ACV is not here on purpose: ml/acv.py fits nothing, so it has no artifact to build.


def main(argv):
    wanted = argv or list(FITTERS)
    unknown = [k for k in wanted if k not in FITTERS]
    if unknown:
        raise SystemExit(f"unknown subsystem(s) {unknown}; choose from {list(FITTERS)}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for key in wanted:
        _log(f"[{key}] fitting…")
        t0 = time.time()
        model, meta = FITTERS[key]()
        out = ARTIFACTS / f"{key}_classical.joblib"
        joblib.dump(model, out)
        _log(f"[{key}] {meta} -> {out.name} ({out.stat().st_size / 1024:.0f} KB, {time.time() - t0:.0f}s)\n")


if __name__ == "__main__":
    main(sys.argv[1:])
