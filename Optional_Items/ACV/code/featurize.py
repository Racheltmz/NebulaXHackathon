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


def rail_array_from_frame(df: pd.DataFrame) -> np.ndarray:
    """One recording -> [5, T]: speed, Side-I vib, Side-I shock, Side-II vib, Side-II shock.

    Takes an already-parsed frame so a caller that also needs the raw rows (the dashboard
    telemetry does) parses the file once — these recordings are ~17 MB each.
    """
    a = df.to_numpy(np.float32)
    v = a[:, 1:].reshape(len(a), 8, 8, 2)
    return np.stack([
        a[:, 0],
        v[:, :, RAIL_SIDE_I, 0].mean((1, 2)),
        v[:, :, RAIL_SIDE_I, 1].mean((1, 2)),
        v[:, :, RAIL_SIDE_II, 0].mean((1, 2)),
        v[:, :, RAIL_SIDE_II, 1].mean((1, 2)),
    ])


def rail_array(content: bytes) -> np.ndarray:
    """As `rail_array_from_frame`, straight from file bytes."""
    return rail_array_from_frame(pd.read_csv(io.BytesIO(content)))


# ---------------------------------------------------------------- SHM


def shm_array(content: bytes) -> np.ndarray:
    """One stress trace -> [C, T], numeric columns transposed."""
    return pd.read_csv(io.BytesIO(content)).select_dtypes(include="number").to_numpy(np.float32).T


# ---------------------------------------------------------------- Door


def door_timestamp(s) -> int:
    """`Datetime` is dash-separated parts; zero-pad each so string order is time order."""
    return int("".join(f"{int(v):02d}" for v in str(s).split("-")))


def door_millis(s) -> int:
    """`Datetime` -> absolute milliseconds, for measuring real gaps between rows."""
    _, _, day, hour, minute, sec, milli = (int(v) for v in str(s).split("-"))
    return ((((day * 24 + hour) * 60 + minute) * 60) + sec) * 1000 + milli


# Rows inside one door cycle are sampled a uniform 20 ms apart; the stream is separate recordings
# laid end to end, and between them the clock jumps by tens of seconds. Anything above this
# threshold is therefore a recording boundary, not a slow sample. On the labelled training stream
# this recovers all 110 segments with exact boundaries (mean IoU 1.000), and the result is
# identical anywhere from 60 ms to 1000 ms — it is not a tuned number.
DOOR_GAP_MS = 500


def door_segment_bounds(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Split a continuous stream into (start_row, end_row) per door cycle."""
    t = df.Datetime.map(door_millis).to_numpy()
    starts = np.concatenate([[0], np.where(np.diff(t) > DOOR_GAP_MS)[0] + 1])
    ends = np.concatenate([starts[1:] - 1, [len(df) - 1]])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def door_arrays_from_bounds(df: pd.DataFrame, bounds) -> list[np.ndarray]:
    """One [C, T] array per (start_row, end_row) segment."""
    cols = [c for c in df.columns if c != "Datetime"]
    block = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    return [block[a : b + 1].T for a, b in bounds]


def door_segment_arrays(df: pd.DataFrame, segments) -> list[np.ndarray]:
    """One [C, T] array per (start_time, end_time) pair — used when boundaries are already
    known, as they are for the labelled training segments.

    Boundaries are located by exact match on the `Datetime` string rather than by comparing
    `door_timestamp` values. That encoding zero-pads each field to a *minimum* of two digits, so
    a millisecond field of 700 occupies three digits where 20 occupies two, and the concatenated
    integers stop being ordered by time. Slicing on them returned segments of 0 to 12,923 rows
    where the answer file says 137 to 190 — and the model would then be fitted on windows quite
    unlike the ones `door_segment_bounds` hands it at prediction time.
    """
    row_of = {str(dt): i for i, dt in enumerate(df.Datetime)}
    cols = [c for c in df.columns if c != "Datetime"]
    block = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    out = []
    for start, end in segments:
        a, b = row_of[str(start)], row_of[str(end)]
        out.append(block[a : b + 1].T)
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
