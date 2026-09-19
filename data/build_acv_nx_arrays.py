"""ACV per-car feature-vector arrays (nebulax `case_rows` features) aligned to the existing ACV split
(train = first five cases, test = the sixth).  Labels are asserted equal to data/ps3_arrays.
NaN marks a signal a case does not expose (see common/final_data.acv_nx_matrix)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import final_data as fd  # noqa: E402

out = HERE / "nx_arrays"
out.mkdir(exist_ok=True)
X, y, _, _, cols = fd.acv_nx_matrix()
old_tr = np.load(HERE / "ps3_arrays" / "acv_train_y.npy", allow_pickle=True).astype(int)
old_te = np.load(HERE / "ps3_arrays" / "acv_test_y.npy", allow_pickle=True).astype(int)
assert list(y[:40]) == list(old_tr) and list(y[40:]) == list(old_te), "ACV label order mismatch"
for tag, sl in (("train", slice(0, 40)), ("test", slice(40, 48))):
    arr = np.empty(len(X[sl]), dtype=object)
    for i, v in enumerate(X[sl]):
        arr[i] = v
    np.save(out / f"acv_{tag}_x.npy", arr, allow_pickle=True)
    np.save(out / f"acv_{tag}_y.npy", y[sl], allow_pickle=True)
print("ACV nx arrays:", len(X), "cars x", len(cols), "features; alignment with ps3_arrays verified")
