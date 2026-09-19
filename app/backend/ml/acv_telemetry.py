"""Per-car telemetry for the ACV run dashboard: a compact, downsampled copy of each car's
temperatures and modes, so the dashboard can plot them without re-reading the uploaded workbook
(a full read takes seconds, and ~40s for the 483-column model B file).

Display-only, like ml/severity.py — nothing here feeds the ranking or the `acv_predictions.csv`.

Files record a different parameter set per car model (see the info kit, Section 2.1), so each
signal below lists the parameter names that can carry it and the first one a file has is used.
The name chosen is reported back (`sources`) so the dashboard can say where a series came from.
Model B's names are matched by value range and vocabulary, not documented — treat them as
best-effort.
"""

import re

import numpy as np
import pandas as pd

BUCKETS = 300  # time bins per series: ~15 min for a 3-4 day, 30 s-sampled file

CAR_COLUMN_RE = re.compile(r"Car\s*(\d+)\s*-\s*(.+)")

NUMERIC_SIGNALS = {
    "indoor": ["Indoor Average Temperature", "Passenger Cabin Temperature Detected Value"],
    "outdoor": [
        "Outdoor Average Temperature",
        "Outside Temperature Sensor Reading",  # model C; 'Invalid' on most cars
        "Fresh Air Temperature Detected Value",
    ],
    "control_cooling": ["ACV Control Temperature (Cooling)", "Target Temperature Value"],
    "control_heating": ["ACV Control Temperature (Heating)"],
}
CATEGORICAL_SIGNALS = {
    "setting_mode": ["ACV Setting Mode", "ACV Control Mode"],
    "running_mode": ["ACV Running Mode"],
    "load_halved": ["Load Halved"],
    "info_valid": ["ACV Information Valid"],
}

_ALL_PARAMETERS = {p for names in (*NUMERIC_SIGNALS.values(), *CATEGORICAL_SIGNALS.values()) for p in names}


def wanted_column(column) -> bool:
    """`usecols` filter: the identifier columns plus the per-car parameters plotted here."""
    name = str(column)
    if name in ("Time", "Car model", "Train number"):
        return True
    match = CAR_COLUMN_RE.match(name)
    return match is not None and match.group(2).strip() in _ALL_PARAMETERS


def _columns_by_car(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    by_car: dict[str, dict[str, str]] = {}
    for column in df.columns:
        match = CAR_COLUMN_RE.match(str(column))
        if match:
            by_car.setdefault(match.group(1).zfill(2), {})[match.group(2).strip()] = column
    return by_car


def _source(by_car: dict[str, dict[str, str]], candidates: list[str]) -> str | None:
    return next((p for p in candidates if any(p in cols for cols in by_car.values())), None)


def build_telemetry(df: pd.DataFrame) -> dict | None:
    """Downsample a case file to BUCKETS equal time bins: numeric signals are the bin's mean,
    categorical signals its most frequent value. A bin with no valid reading is null, so gaps in
    the recording stay gaps. A 0 on a temperature is dropped: it is the placeholder the files use
    for "no reading" (and for a heating setpoint that isn't set), never a real cabin temperature.
    Returns None if the file has no usable Time column.
    """
    if "Time" not in df.columns:
        return None
    secs = (pd.to_datetime(df["Time"], errors="coerce") - pd.Timestamp("1970-01-01")).dt.total_seconds()
    secs = secs.to_numpy(dtype=float)
    in_range = ~np.isnan(secs)
    if not in_range.any():
        return None

    start, end = secs[in_range].min(), secs[in_range].max()
    span = max(end - start, 1.0)
    with np.errstate(invalid="ignore"):
        bins = np.clip(np.floor((secs - start) / span * BUCKETS), 0, BUCKETS - 1)
    bins = np.where(in_range, bins, 0).astype(int)

    by_car = _columns_by_car(df)
    sources = {}
    for signal, candidates in {**NUMERIC_SIGNALS, **CATEGORICAL_SIGNALS}.items():
        source = _source(by_car, candidates)
        if source:
            sources[signal] = source

    # One shared category list per signal (first seen first), so a value has the same index on
    # every car and the dashboard can colour by name.
    categories: dict[str, list[str]] = {}
    for signal in CATEGORICAL_SIGNALS:
        if signal not in sources:
            continue
        seen: list[str] = []
        for cols in by_car.values():
            if sources[signal] in cols:
                for value in df[cols[sources[signal]]].dropna().astype(str).str.strip().unique():
                    if value and value not in seen:
                        seen.append(value)
        categories[signal] = seen

    cars = {}
    for car_id in sorted(by_car):
        series = {}
        for signal in NUMERIC_SIGNALS:
            column = by_car[car_id].get(sources.get(signal))
            if column is None:
                continue
            values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            values = np.where(values == 0, np.nan, values)
            valid = in_range & ~np.isnan(values)
            if not valid.any():
                continue  # e.g. model C's outdoor sensor, which reads "Invalid" on most cars
            totals = np.bincount(bins[valid], weights=values[valid], minlength=BUCKETS)
            counts = np.bincount(bins[valid], minlength=BUCKETS)
            series[signal] = [
                round(float(totals[i] / counts[i]), 1) if counts[i] else None for i in range(BUCKETS)
            ]

        for signal in CATEGORICAL_SIGNALS:
            column = by_car[car_id].get(sources.get(signal))
            if column is None:
                continue
            labels = df[column].astype("string").str.strip()
            codes = pd.Categorical(labels, categories=categories[signal]).codes
            valid = in_range & (codes >= 0)
            if not valid.any():
                continue
            size = len(categories[signal])
            counts = np.bincount(bins[valid] * size + codes[valid], minlength=BUCKETS * size)
            counts = counts.reshape(BUCKETS, size)
            series[signal] = [int(counts[i].argmax()) if counts[i].any() else None for i in range(BUCKETS)]

        cars[car_id] = series

    return {
        "t": [int(round(start + (i + 0.5) * span / BUCKETS)) for i in range(BUCKETS)],
        "cars": cars,
        "categories": categories,
        "sources": sources,
    }
