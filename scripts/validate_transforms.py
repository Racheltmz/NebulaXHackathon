#!/usr/bin/env python3
"""Prove the held-out (test-time) transforms reproduce the training arrays exactly.

Recomputes real *training* files with the same functions used on the Test files and
checks each result exists, element-for-element, in the stored labelled arrays.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import final_data as fd  # noqa: E402


def has_match(vec, pool, tol=1e-6):
    v = np.asarray(vec, dtype=np.float64)
    for a in pool:
        a = np.asarray(a, dtype=np.float64)
        if a.shape == v.shape and np.allclose(a, v, rtol=tol, atol=tol):
            return True
    return False


def has_match_rel(vec, pool, rel=1e-5):
    """ACV residual channels divide by a ~1e-5 MAD, so float32 rounding gives ~1e-6 *relative* noise."""
    v = np.asarray(vec, dtype=np.float64)
    for a in pool:
        a = np.asarray(a, dtype=np.float64)
        if a.shape == v.shape and np.max(np.abs(a - v)) <= rel * max(1.0, np.max(np.abs(a))):
            return True
    return False


ok = True
def report(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


# ---- Door: gap segmentation == answer segments; slices == stored arrays
df = pd.read_csv(fd.RAW / "Door" / "Train.csv"); ans = pd.read_csv(fd.RAW / "Door" / "Train_Segments_Answer.csv")
segs = fd.door_gap_segments(df.Datetime); pos = {s: i for i, s in enumerate(df.Datetime)}
truth = [(pos[s], pos[e]) for s, e in zip(ans.start_time, ans.end_time)]
report("door gap segmentation reproduces all answer segments", segs == truth, f"({len(segs)} segments)")
num = fd._door_numeric(df); xs, _ = fd._concat("all", "door", fd.ARR)
report("door segment slices match stored arrays", all(has_match(num[a:b + 1].T, xs) for a, b in segs[:25]), "(first 25)")

# ---- ACV: per-case arrays == stored arrays
xa, _ = fd._concat("all", "acv", fd.ARR)
arrs, cars = fd.acv_case_arrays(fd.RAW / "ACV" / "Train" / "acv_case_02.xlsx")
report("acv case arrays match stored arrays", all(has_match_rel(a, xa) for a in arrs), f"(cars {cars})")

# ---- Rail: compact + nx vectors for 3 training files
xr, _ = fd._concat("all", "rail", fd.ARR); xn, _ = fd._concat("all", "rail", fd.NXARR)
lab = pd.read_csv(fd.RAW / "Rail_Corrugation" / "Train_Labels.csv")
for name in [lab.filename.iloc[0], lab.filename.iloc[50], lab.filename.iloc[200]]:
    d = pd.read_csv(fd.RAW / "Rail_Corrugation" / "Train" / name)
    report(f"rail compact {name}", has_match(fd.rail_compact(d.to_numpy(np.float32)), xr))
    report(f"rail nx vector {name}", has_match(fd.rail_nx_vector(d), xn, tol=1e-5))

# ---- SHM: raw + nx vectors for 2 training files
xs_raw, _ = fd._concat("all", "shm", fd.ARR); xs_nx, _ = fd._concat("all", "shm", fd.NXARR)
lab = pd.read_csv(fd.RAW / "SHM" / "Train_Labels.csv")
for name in [lab.filename.iloc[3], lab.filename.iloc[40]]:
    p = fd.RAW / "SHM" / "Train" / name
    report(f"shm raw {name}", has_match(pd.read_csv(p).select_dtypes(include="number").to_numpy(np.float32).T, xs_raw))
    report(f"shm nx vector {name}", has_match(fd.shm_nx_vector(np.loadtxt(p, delimiter=",")), xs_nx, tol=1e-4))

print("ALL TRANSFORMS VALIDATED" if ok else "VALIDATION FAILED")
sys.exit(0 if ok else 1)
