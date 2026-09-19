"""Rebuild the Door arrays with the CORRECT segments.

The original ps3_prepare.door() selected each segment with a custom
timestamp->integer conversion that mis-orders millisecond fields of different
widths ("20" vs "700"), so 0/110 segments had the right length. Here every
segment is sliced by the exact row indices of its start_time/end_time strings
(verified: all 110 row counts equal the answer file's n_rows).

Same split (ps3_prepare.split, seed 7) and same array layout ([16, T] float32).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ps3_prepare import split  # noqa: E402


def door_segments(door_dir):
    door_dir = Path(door_dir)
    df = pd.read_csv(door_dir / "Train.csv")
    ans = pd.read_csv(door_dir / "Train_Segments_Answer.csv")
    pos = {s: i for i, s in enumerate(df.Datetime)}
    cols = [c for c in df.columns if c != "Datetime"]
    num = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    xs, ys = [], []
    for row in ans.itertuples():
        a, b = pos[row.start_time], pos[row.end_time]
        assert b - a + 1 == row.n_rows, (row.segment_id, b - a + 1, row.n_rows)
        xs.append(num[a:b + 1].T.copy())
        ys.append(str(row.status))
    return xs, ys


def main(root, out):
    xs, ys = door_segments(Path(root) / "Door")
    tx, ty, vx, vy = split(xs, ys)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, x, y in (("train", tx, ty), ("test", vx, vy)):
        arr = np.empty(len(x), dtype=object)
        for i, item in enumerate(x):
            arr[i] = item
        np.save(out / f"door_{name}_x.npy", arr, allow_pickle=True)
        np.save(out / f"door_{name}_y.npy", np.asarray(y), allow_pickle=True)
    print("segments:", len(xs), "train:", len(tx), "test:", len(vx),
          "| lengths min/median/max:", min(x.shape[1] for x in xs), int(np.median([x.shape[1] for x in xs])),
          max(x.shape[1] for x in xs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
