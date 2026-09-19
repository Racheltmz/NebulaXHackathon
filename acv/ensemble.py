#!/usr/bin/env python3
"""ACV: diverse-fold ensemble of the best classical programs (fault score per car, z-scored within the
case), leave-one-case-out weight calibration, hard-labelling of the held-out case's top-ranked car,
retraining, and acv_predictions.csv (file_id, ranked_cars = most -> least likely faulty, e.g. `03|05|...`).
usage: python acv/ensemble.py --out submission
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from common import ensemble_lib as E, final_data as fd  # noqa: E402
from common.metrics import acv_rank_decay  # noqa: E402


EXCLUDED_CASES = [3]   # train case 4 exposes 32 signals that no other case (nor the test case) has, so it cannot be
                       # validated from the rest; it stays in training but is not used to score or calibrate


def _metric(y, o, g):
    keep = ~np.isin(np.asarray(g), EXCLUDED_CASES)
    return acv_rank_decay(np.asarray(y)[keep], np.asarray(o)[keep], np.asarray(g)[keep])


def build(data, top_p):
    types = E.load_member_types(REPO / "acv", {"classical": ("raw", max(1, top_p - 1))}, top_p)
    seed = (REPO / "acv" / "baseline_classical_nx.py").read_text()
    types.append(E.MemberType("classical_nx:seed", seed, "nx"))
    return E.EnsembleTask("acv", "rank", types, metric=_metric, make_flag=lambda _y: (lambda v: np.asarray(v) == 1))


def write(data, res, out):
    rows = []
    for fname in dict.fromkeys(f for f, _ in data.test_ids):
        idx = [i for i, (f, _) in enumerate(data.test_ids) if f == fname]
        order = sorted(idx, key=lambda i: -res["final"][i])
        rows.append({"file_id": fname, "ranked_cars": "|".join(data.test_ids[i][1] for i in order)})
    path = Path(out) / "acv_predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    E.cli("acv", fd.acv_data, build, write)
