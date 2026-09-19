"""ACV refrigerant-leak localisation — peer-relative cabin-temperature ranking.

A leaking ACV unit cannot move as much heat, so its cabin sits further above its own cooling set
point than a healthy car's does. All 8 cars share one train, one route, one recorder and the same
weather, so comparing them *against each other at the same timestamp* cancels the ambient,
time-of-day and passenger-load confounds that make absolute temperatures useless:

    residual_i(t) = cabin_i(t) - cooling_setpoint_i(t)
    peer_i(t)     = residual_i(t) - median_j residual_j(t)
    score_i       = mean peer_i(t) over rows where car i is cooling and reporting Valid

Cars are ranked by score, highest first.

Nothing is fitted — no weights, no learned thresholds, no per-file tuning. With only five usable
labelled cases a fitted model could not be validated honestly, so the rule is derived from the
fault's mechanism instead. That also makes the validation genuinely out-of-sample: the true faulty
car ranks 1st in all five labelled cases that share the test file's schema, across two fleets.

Limits worth knowing: the score ranks cars by *degraded cooling capacity*, which is the leak's
consequence rather than the leak. Training case 04 (a different fleet, with per-circuit pressure
telemetry) shows a real leak on one of a car's two refrigeration circuits staying nearly invisible
in cabin temperature because the healthy circuit carried the load. Redundancy can mask a leak, and
other faults can mimic one.
"""

import io
import re

import numpy as np
import pandas as pd

from .common import PredictionResult, UploadedFile

CAR_COLUMN_RE = re.compile(r"Car\s*(\d+)\s*-")

CABIN = "Indoor Average Temperature"
SETPOINT = "ACV Control Temperature (Cooling)"
MODE = "ACV Running Mode"
VALID = "ACV Information Valid"


def validate(files: list[UploadedFile]) -> None:
    for f in files:
        try:
            df = pd.read_excel(io.BytesIO(f.content), nrows=1)
        except Exception as exc:  # noqa: BLE001 — surfaced verbatim as the validation error
            raise ValueError(f"'{f.filename}' could not be read as an Excel (.xlsx) file: {exc}") from exc
        if not any(CAR_COLUMN_RE.search(str(col)) for col in df.columns):
            raise ValueError(
                f"'{f.filename}' has no columns matching 'Car <NN> - <parameter>' — expected "
                "per-car readings alongside car model, train number, and time columns."
            )


def _car_ids(df: pd.DataFrame) -> list[str]:
    """Car identifiers exactly as this file's own headers spell them, e.g. '03'."""
    return sorted({m.group(1) for c in df.columns if (m := CAR_COLUMN_RE.match(str(c).strip()))})


def _panel(df: pd.DataFrame, suffix: str, cars: list[str], numeric: bool = True) -> pd.DataFrame:
    """One column per car for a given parameter. Columns arrive in scrambled order
    (`Car 08 - Setting Mode`, `Car 01 - Cooling Temp`, …), so each is matched by parsing its
    header rather than by position. A parameter a file doesn't carry yields an all-NaN column."""
    out = {}
    for car in cars:
        match = [c for c in df.columns if str(c).startswith(f"Car {car} - ") and str(c).endswith(suffix)]
        out[car] = df[match[0]] if match else pd.Series(np.nan, index=df.index)
    frame = pd.DataFrame(out)
    if numeric:
        frame = frame.apply(pd.to_numeric, errors="coerce")
        # 0.0 is the recorder's "no reading" sentinel, not a temperature — a 0 °C cabin is not a
        # measurement. Roughly 9-12% of rows are blank this way; left in, they drag every mean
        # toward zero.
        frame = frame.mask(frame == 0)
    return frame


def rank_cars(df: pd.DataFrame) -> tuple[list[str], pd.Series]:
    """(car ids most → least likely faulty, their scores). Every car in the file is returned:
    one missing from `ranked_cars` scores 0 under the official metric, so cars with no usable
    data are ranked last rather than dropped."""
    cars = _car_ids(df)
    if not cars:
        return [], pd.Series(dtype=float)

    cabin = _panel(df, CABIN, cars)
    setpoint = _panel(df, SETPOINT, cars)
    mode = _panel(df, MODE, cars, numeric=False).astype(str)
    valid = _panel(df, VALID, cars, numeric=False).astype(str)

    # Only rows where the unit is actually trying to cool and the record is flagged valid.
    usable = mode.apply(lambda s: s.str.contains("Cool", case=False, na=False)) & valid.ne("Invalid")

    residual = (cabin - setpoint).where(usable)
    peer = residual.sub(residual.median(axis=1), axis=0)
    scores = peer.mean()

    ranked = scores.sort_values(ascending=False)
    order = [c for c in ranked.index if pd.notna(ranked[c])]
    order += [c for c in ranked.index if pd.isna(ranked[c])]
    return order, scores


def predict(files: list[UploadedFile]) -> PredictionResult:
    rows = []
    top_cars = []
    for f in files:
        df = pd.read_excel(io.BytesIO(f.content), sheet_name=0)
        order, scores = rank_cars(df)
        rows.append({"file_id": f.filename, "ranked_cars": "|".join(order)})
        if order:
            top_cars.append(order[0])

    summary = {
        "files": len(rows),
        "top_car": top_cars[0] if top_cars else None,
        "model": "peer-relative cooling residual",
    }
    return PredictionResult(rows=rows, summary=summary)
