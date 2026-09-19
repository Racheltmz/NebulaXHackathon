#!/usr/bin/env python3
"""Rail: diverse-fold ensemble of the best classical programs (raw compact-array programs and the
nebulax-seeded feature-vector programs), OOF weight calibration, hard-label retraining on the 68
held-out recordings, and rail_predictions.csv (file_id, prediction in Normal / Side I / Side II).
Flagged (faulty) = Side I and Side II: every member sees all of them, plus a different third of Normal.
usage: python rail/ensemble.py --out submission
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from common import ensemble_lib as E, final_data as fd  # noqa: E402
from common.metrics import rail_score  # noqa: E402


def build(data, top_p):
    types = E.load_member_types(REPO / "rail", {"classical": ("raw", max(1, top_p - 1)), "classical_nx": ("nx", top_p + 1)}, top_p)
    return E.EnsembleTask(
        "rail", "class", types, metric=lambda y, o, g: rail_score(y, o),
        make_flag=lambda _y: (lambda v: np.asarray(v) != "Normal"),
        classes=["Normal", "Side I", "Side II"], flagged_classes=["Side I", "Side II"])


def write(data, res, out):
    path = Path(out) / "rail_predictions.csv"
    pd.DataFrame({"file_id": data.test_ids, "prediction": res["final"]}).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    E.cli("rail", fd.rail_data, build, write)
