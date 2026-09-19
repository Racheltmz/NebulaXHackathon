#!/usr/bin/env python3
"""Door: diverse-fold ensemble of the best classical programs, OOF weight calibration, hard-label
retraining on the held-out stream, and door_predictions.csv (start_time, end_time, prediction).

Held-out segmentation: the continuous Test.csv stream is cut wherever the inter-row gap exceeds 1 s
(this rule reproduces all 110 training segments exactly).   usage: python door/ensemble.py --out submission
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from common import ensemble_lib as E, final_data as fd  # noqa: E402
from common.metrics import door_score  # noqa: E402


def build(data, top_p):
    return E.EnsembleTask(
        "door", "class", E.load_member_types(REPO / "door", {"classical": "raw"}, top_p),
        metric=lambda y, o, g: door_score(y, o),
        make_flag=lambda _y: (lambda v: np.asarray(v) == "Abnormal resistance"),
        classes=["Normal", "Abnormal resistance"], flagged_classes=["Abnormal resistance"])


def write(data, res, out):
    df = pd.DataFrame({"start_time": [s for s, _ in data.test_ids], "end_time": [e for _, e in data.test_ids],
                       "prediction": res["final"]})
    path = Path(out) / "door_predictions.csv"
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    E.cli("door", fd.door_data, build, write)
