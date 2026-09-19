"""Raw PS3 file -> the `[C, T]` arrays the vendored models expect.

Ported from the `ian-model` branch's `data/ps3_prepare.py`. The evolved models in `models/` were
searched against arrays built exactly this way, so any change here silently invalidates them —
keep this file in step with that branch, not with what looks tidier.

One function per subsystem, each taking a file's bytes and returning what that subsystem's
`Model.predict` consumes.
"""

import io

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- Rail corrugation

# Columns 2-129 are 8 cars x 8 positions x (vibration, shock). Positions 1,3,5,7 (0-indexed
# 0,2,4,6) ride the Side I rail; 2,4,6,8 ride Side II. The evolved model sees speed plus the
# per-side mean vibration and shock — the side contrast is what separates Side I from Side II.
RAIL_SIDE_I = [0, 2, 4, 6]
RAIL_SIDE_II = [1, 3, 5, 7]


def rail_array(content: bytes) -> np.ndarray:
    """One recording -> [5, T]: speed, Side-I vib, Side-I shock, Side-II vib, Side-II shock."""
    a = pd.read_csv(io.BytesIO(content)).to_numpy(np.float32)
    v = a[:, 1:].reshape(len(a), 8, 8, 2)
    return np.stack([
        a[:, 0],
        v[:, :, RAIL_SIDE_I, 0].mean((1, 2)),
        v[:, :, RAIL_SIDE_I, 1].mean((1, 2)),
        v[:, :, RAIL_SIDE_II, 0].mean((1, 2)),
        v[:, :, RAIL_SIDE_II, 1].mean((1, 2)),
    ])


# ---------------------------------------------------------------- SHM


def shm_array(content: bytes) -> np.ndarray:
    """One stress trace -> [C, T], numeric columns transposed."""
    return pd.read_csv(io.BytesIO(content)).select_dtypes(include="number").to_numpy(np.float32).T


# ---------------------------------------------------------------- Door


def door_timestamp(s) -> int:
    """`Datetime` is dash-separated parts; zero-pad each so string order is time order."""
    return int("".join(f"{int(v):02d}" for v in str(s).split("-")))


def door_segment_arrays(df: pd.DataFrame, segments) -> list[np.ndarray]:
    """Rows of a continuous stream sliced into one [C, T] array per (start, end) segment."""
    t = df.Datetime.map(door_timestamp).to_numpy()
    cols = [c for c in df.columns if c != "Datetime"]
    out = []
    for start, end in segments:
        m = (t >= door_timestamp(start)) & (t <= door_timestamp(end))
        z = df.loc[m, cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32).T
        out.append(z)
    return out


# ---------------------------------------------------------------- ACV
#
# Kept for parity with the ian-model track; the shipped ACV predictor uses the peer-relative
# cooling residual in ml/acv.py instead, which scores higher on the labelled cases.


def acv_case_arrays(df: pd.DataFrame) -> tuple[list[str], list[np.ndarray]]:
    """One case file -> (car ids, one [8, T] array per car).

    Four per-timestep statistics across that car's own parameters, then the same four expressed
    relative to the rest of the train — faults are defined against the other cars in the same
    train, so the residual channels make that invariant explicit.
    """
    cars = sorted({c.split(" - ", 1)[0].split()[-1] for c in df.columns if c.startswith("Car ") and " - " in c})
    case_x = []
    for car in cars:
        cols = [c for c in df.columns if c.startswith(f"Car {car} - ")]
        a = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        case_x.append(np.stack([
            a.mean(1), a.std(1), np.quantile(a, 0.9, axis=1), np.quantile(a, 0.1, axis=1),
        ], axis=0))
    stack = np.stack(case_x)
    med = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - med), axis=0) + 1e-5
    return cars, [np.concatenate([a, (a - med) / mad], axis=0) for a in case_x]
