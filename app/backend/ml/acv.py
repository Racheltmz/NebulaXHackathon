"""ACV refrigerant-leak localisation — STUB.

No trained model exists yet for this subsystem (see docs/DESIGN.md Section 8). Placeholder
behaviour: lists every car found in the file's own column headers in the order they first
appear (i.e. unranked), so the output shape is already submission-ready — only the ordering
logic needs to change once a real model lands.
"""

import io
import logging
import re

import pandas as pd

from .acv_telemetry import build_telemetry, wanted_column
from .common import PredictionResult, UploadedFile

logger = logging.getLogger(__name__)

CAR_COLUMN_RE = re.compile(r"Car\s*(\d+)\s*-")


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


def _first_value(df: pd.DataFrame, column: str) -> str | None:
    """A file-level identifier column ('Car model', 'Train number') — constant within a file, so the
    first non-empty value is the file's. None if the column is absent or empty."""
    if column not in df.columns:
        return None
    values = df[column].dropna().astype(str).str.strip()
    return values.iloc[0] if len(values) and values.iloc[0] else None


def predict(files: list[UploadedFile]) -> PredictionResult:
    rows = []
    top_cars = []
    car_models = {}
    train_numbers = {}
    telemetry = {}
    for f in files:
        # Train number is text in the files ("0620"); pandas would otherwise turn it into 620.
        df = pd.read_excel(io.BytesIO(f.content), nrows=1, dtype={"Train number": str})
        car_models[f.filename] = _first_value(df, "Car model")
        train_numbers[f.filename] = _first_value(df, "Train number")
        car_ids = []
        for col in df.columns:
            match = CAR_COLUMN_RE.search(str(col))
            if match:
                car_id = match.group(1).zfill(2)
                if car_id not in car_ids:
                    car_ids.append(car_id)

        rows.append({"file_id": f.filename, "ranked_cars": "|".join(car_ids)})
        if car_ids:
            top_cars.append(car_ids[0])

        # Display-only, so it must never fail the run — the ranking above is what's being asked for.
        try:
            full = pd.read_excel(io.BytesIO(f.content), dtype={"Train number": str}, usecols=wanted_column)
            telemetry[f.filename] = build_telemetry(full)
        except Exception:  # noqa: BLE001
            logger.exception("Could not build ACV telemetry for %s", f.filename)

    # Kept in the summary, not the rows: rows map onto the exported acv_predictions.csv, which is
    # just file_id + ranked_cars, and onto prediction_rows columns that have no car_model.
    summary = {
        "files": len(rows),
        "top_car": top_cars[0] if top_cars else None,
        "car_models": car_models,
        "train_numbers": train_numbers,
        "telemetry": telemetry,
        "model": "stub",
    }
    return PredictionResult(rows=rows, summary=summary)
