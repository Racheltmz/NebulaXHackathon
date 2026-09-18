"""ACV refrigerant-leak localisation — STUB.

No trained model exists yet for this subsystem (see docs/DESIGN.md Section 8). Placeholder
behaviour: lists every car found in the file's own column headers in the order they first
appear (i.e. unranked), so the output shape is already submission-ready — only the ordering
logic needs to change once a real model lands.
"""

import io
import re

import pandas as pd

from .common import PredictionResult, UploadedFile

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


def predict(files: list[UploadedFile]) -> PredictionResult:
    rows = []
    top_cars = []
    for f in files:
        df = pd.read_excel(io.BytesIO(f.content), nrows=1)
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

    summary = {"files": len(rows), "top_car": top_cars[0] if top_cars else None, "model": "stub"}
    return PredictionResult(rows=rows, summary=summary)
