"""Door signals for the run dashboard: a compact, downsampled copy of the stream's 16 readings, so
the dashboard can plot them without re-reading the uploaded CSV. Display-only, like
ml/acv_telemetry.py — nothing here feeds the segmentation or `door_predictions.csv`.

The provided streams are cycles stitched together: the timestamps jump forward between one
open/close cycle and the next (Door info kit, Section 2.1). Those jumps are reported as `breaks`
so the dashboard can mark each cycle instead of drawing the skipped time as an empty stretch.
"""

import io
import re

import numpy as np
import pandas as pd

BINS = 1800  # points per series; a 6k-row stream keeps ~4 rows per point, an 18k-row one ~10
GAP_SECONDS = 1.0  # a jump in time bigger than this between two readings starts a new cycle

ANALOG = {
    "current": "Motor current(mA)",
    "voltage": "Motor Voltage(10mV)",
    "back_emf": "Motor electrodynamic force",
    "position": "Door leaf position",
    "opening_time": "Door opening time(.1s)",
    "closing_time": "Door closing time(.1s)",
}
# Recorded as 0 until a cycle has been timed, and a door can't open in 0 s — so 0 means no reading.
ZERO_MEANS_MISSING = {"opening_time", "closing_time"}

FLAGS = {
    "close_command": "Close command",
    "open_command": "Open command",
    "opening": "Door is opening",
    "closing": "Door is closing",
    "dcsr": "DCSR",
    "dcsl": "DCSL",
    "dlsr": "DLSR",
    "dlsl": "DLSL",
    "door_opened": "Door Opened",
    "door_locked": "Door Locked",
}


def _normalise(name) -> str:
    return re.sub(r"\s+", "", str(name)).lower()


def _epoch_seconds(series: pd.Series) -> np.ndarray:
    """The dataset's own `Year-Month-Date-Hour-Minute-Second-Millisecond` (hyphen-separated, not
    zero-padded) or anything pandas can parse, as seconds since the epoch. Naive timestamps are
    read as UTC, matching how the dashboard shows them."""
    parts = series.astype(str).str.split("-", expand=True)
    if parts.shape[1] == 7:
        p = parts.apply(pd.to_numeric, errors="coerce")
        stamps = pd.to_datetime(
            {"year": p[0], "month": p[1], "day": p[2], "hour": p[3], "minute": p[4], "second": p[5], "ms": p[6]},
            errors="coerce",
        )
    else:
        stamps = pd.to_datetime(series, errors="coerce")
    return (stamps - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype=float)


def build_telemetry(df: pd.DataFrame) -> dict | None:
    """Average each analog reading, and take the max (so a brief pulse survives) of each flag,
    over consecutive groups of rows. Returns None if the first column isn't a usable timestamp."""
    if df.shape[1] == 0:
        return None
    seconds = _epoch_seconds(df.iloc[:, 0])
    keep = ~np.isnan(seconds)
    if not keep.any():
        return None
    df, seconds = df.loc[keep].reset_index(drop=True), seconds[keep]

    rows = len(df)
    step = int(np.ceil(rows / min(rows, BINS)))
    starts = np.arange(0, rows, step)
    bins = len(starts)
    bin_of_row = np.arange(rows) // step

    by_name = {_normalise(c): c for c in df.columns[1:]}

    series, sources = {}, {}
    for key, name in ANALOG.items():
        column = by_name.get(_normalise(name))
        if column is None:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if key in ZERO_MEANS_MISSING:
            values = np.where(values == 0, np.nan, values)
        valid = ~np.isnan(values)
        if not valid.any():
            continue
        totals = np.bincount(bin_of_row[valid], weights=values[valid], minlength=bins)
        counts = np.bincount(bin_of_row[valid], minlength=bins)
        digits = 1 if key in ZERO_MEANS_MISSING else 0
        series[key] = [round(float(totals[i] / counts[i]), digits) if counts[i] else None for i in range(bins)]
        if digits == 0:
            series[key] = [None if v is None else int(v) for v in series[key]]
        sources[key] = column

    flags = {}
    for key, name in FLAGS.items():
        column = by_name.get(_normalise(name))
        if column is None:
            continue
        values = pd.to_numeric(df[column], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
        flags[key] = [int(v) for v in np.maximum.reduceat(values.astype(np.int8), starts)]
        sources[key] = column

    gaps = np.flatnonzero(np.diff(seconds) > GAP_SECONDS) + 1  # the row that starts each new cycle
    breaks = sorted({int(b) for b in bin_of_row[gaps] if b > 0})

    spacing = np.diff(seconds)
    spacing = spacing[(spacing > 0) & (spacing <= GAP_SECONDS)]
    return {
        "rows": rows,
        "rows_per_point": step,
        "sample_ms": round(float(np.median(spacing)) * 1000) if len(spacing) else None,
        "times": [round(float(seconds[s]), 3) for s in starts],
        "breaks": breaks,
        "series": series,
        "flags": flags,
        "sources": sources,
    }


def build_from_csv(content: bytes) -> dict | None:
    """The same, from an uploaded stream's raw bytes — for runs saved before signals were kept."""
    return build_telemetry(pd.read_csv(io.BytesIO(content)))
