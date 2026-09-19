"""Build label-free per-recording feature-vector arrays for Rail and SHM from the
`nebulax` (ian-classical) feature tables, aligned to THIS project's existing
train/test split (ps3_prepare.split, seed 7).

Rail : concat[v3 | relative | phase]      (272 recordings, one 1-D vector each)
SHM  : concat[multiscale | generic | rainflow | temporal]   (64 traces)

Alignment is verified against the original label arrays -- any row-order
mismatch aborts.  Column-boundary / named-index constants are written to
nx_arrays/layout.json and baked into the seed programs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import os
NX = Path(os.environ.get("NEBULAX_DIR", "/mnt/c/Users/ian/Desktop/nebulax"))  # checkout of the ian-classical branch (has artifacts/analysis/ps3/*.csv)
ART = NX / "artifacts/analysis/ps3"
OLD = HERE / "ps3_arrays"
OUT = HERE / "nx_arrays"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from ps3_prepare import split  # noqa: E402  (the split used for the original arrays)


def clean(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def save(name, xs, ys, seed_split_ys):
    idx = list(range(len(xs)))
    tr_i, _, te_i, _ = split(idx, seed_split_ys)
    for tag, sel in (("train", tr_i), ("test", te_i)):
        arr = np.empty(len(sel), dtype=object)
        for k, i in enumerate(sel):
            arr[k] = np.asarray(xs[i], dtype=np.float64)
        np.save(OUT / f"{name}_{tag}_x.npy", arr, allow_pickle=True)
        np.save(OUT / f"{name}_{tag}_y.npy", np.asarray([ys[i] for i in sel]), allow_pickle=True)
    return tr_i, te_i


layout = {}

# ---------------- Rail ----------------
a = pd.read_csv(ART / "rail_file_features_v3.csv")
r = pd.read_csv(ART / "rail_relative_features.csv").set_index("file_id").loc[a.file_id]
p = pd.read_csv(ART / "rail_phase_features.csv").set_index("file_id").loc[a.file_id]
labels_rail = a.label.astype(str).tolist()
xv = clean(a.drop(columns=["file_id", "label"])).to_numpy(float)
xr = clean(r.drop(columns="label")).to_numpy(float)
xp = clean(p.drop(columns="label")).to_numpy(float)
xs = [np.concatenate([xv[i], xr[i], xp[i]]) for i in range(len(a))]
tr_i, te_i = save("rail", xs, labels_rail, labels_rail)
old_tr = np.load(OLD / "rail_train_y.npy", allow_pickle=True)
old_te = np.load(OLD / "rail_test_y.npy", allow_pickle=True)
assert list(old_tr) == [labels_rail[i] for i in tr_i], "RAIL train label order mismatch"
assert list(old_te) == [labels_rail[i] for i in te_i], "RAIL test label order mismatch"
layout["rail"] = {"n_v3": xv.shape[1], "n_rel": xr.shape[1], "n_phase": xp.shape[1],
                  "n_recordings": len(a), "classes": ["Normal", "Side I", "Side II"]}

# ---------------- SHM ----------------
names = ["shm_multiscale_features.csv", "shm_features.csv", "shm_rainflow_features.csv", "shm_temporal_features.csv"]
tabs = [pd.read_csv(ART / n) for n in names]
y_shm = tabs[0].damage.to_numpy(float)
for t in tabs[1:]:
    assert np.allclose(t.damage.to_numpy(float), y_shm), "SHM tables disagree on damage order"
    assert list(t.file_id) == list(tabs[0].file_id), "SHM tables disagree on file order"
mats = [clean(t.drop(columns=["file_id", "damage"])) for t in tabs]
xs = [np.concatenate([m.iloc[i].to_numpy(float) for m in mats]) for i in range(len(y_shm))]
tr_i, te_i = save("shm", xs, y_shm.tolist(), y_shm.tolist())
old_tr = np.load(OLD / "shm_train_y.npy", allow_pickle=True).astype(float)
old_te = np.load(OLD / "shm_test_y.npy", allow_pickle=True).astype(float)
assert np.allclose(old_tr, y_shm[tr_i]), "SHM train label order mismatch"
assert np.allclose(old_te, y_shm[te_i]), "SHM test label order mismatch"
sizes = [m.shape[1] for m in mats]
cols = [list(m.columns) for m in mats]
off = np.cumsum([0] + sizes)
def idx_of(table, name):
    return int(off[table] + cols[table].index(name))
need_multi = ["w8_ptp_q5", "w8_ptp_q6", "w8_ptp_q7", "w16_ptp_q5", "w16_ptp_q6",
              "w8_diffstd_q5", "w8_diffstd_q6", "w8_diffstd_q7"]
layout["shm"] = {"sizes": sizes, "offsets": off.tolist(), "n_traces": len(y_shm),
                 "calibrator_index": {"generic_ptp": idx_of(1, "ptp"), "generic_std": idx_of(1, "std"),
                                      **{n: idx_of(0, n) for n in need_multi}}}
(OUT / "layout.json").write_text(json.dumps(layout, indent=2))
print(json.dumps(layout, indent=2))
print("ALIGNMENT OK: train/test label orders match the original arrays exactly")
for f in sorted(OUT.glob("*.npy")):
    print(f.name, np.load(f, allow_pickle=True).shape)
