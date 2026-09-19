#!/usr/bin/env python3
"""SHM: diverse-fold ensemble of the best classical programs (raw-trace programs and the nebulax-seeded
feature-vector programs; geometric mean in log space), OOF weight calibration, hard-label retraining on
the 16 held-out traces, and shm_predictions.csv (file_id, prediction = cumulative damage).
"Flagged" for this regression task = high-damage traces (top quartile of the labelled damage values): every
member sees all of them plus a different third of the low-damage traces.   usage: python shm/ensemble.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from common import ensemble_lib as E, final_data as fd  # noqa: E402
from common.metrics import shm_score  # noqa: E402


def build(data, top_p):
    thr = float(np.quantile(data.y_lab, 0.75))
    types = E.load_member_types(REPO / "shm", {"classical": ("raw", max(1, top_p - 1)), "classical_nx": ("nx", top_p + 1)}, top_p)
    return E.EnsembleTask(
        "shm", "reg", types, metric=lambda y, o, g: shm_score(np.asarray(y), np.asarray(o)),
        make_flag=lambda _y: (lambda v: np.asarray(v, dtype=float) >= thr))


def write(data, res, out):
    path = Path(out) / "shm_predictions.csv"
    pd.DataFrame({"file_id": data.test_ids, "prediction": np.maximum(res["final"], 1e-6)}).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    E.cli("shm", fd.shm_data, build, write)
