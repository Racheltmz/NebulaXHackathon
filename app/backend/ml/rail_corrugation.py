"""Rail corrugation 3-class classification — STUB.

No trained model exists yet for this subsystem (see docs/DESIGN.md Section 8). Placeholder
behaviour: classifies every file as "Normal", so the output shape is already submission-ready —
only the classification logic needs to change once a real model lands.
"""

import io

import pandas as pd

from .common import PredictionResult, UploadedFile

EXPECTED_COLUMNS = 129


def validate(files: list[UploadedFile]) -> None:
    for f in files:
        try:
            header = pd.read_csv(io.BytesIO(f.content), nrows=0)
        except Exception as exc:  # noqa: BLE001 — surfaced verbatim as the validation error
            raise ValueError(f"'{f.filename}' could not be read as CSV: {exc}") from exc
        if len(header.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"'{f.filename}' has {len(header.columns)} columns, expected {EXPECTED_COLUMNS} "
                "(rotating speed, then vibration + shock readings for each of 64 axle-box "
                "positions across 8 cars)."
            )


def predict(files: list[UploadedFile]) -> PredictionResult:
    rows = [{"file_id": f.filename, "label": "Normal"} for f in files]
    summary = {"Normal": len(rows), "Side I": 0, "Side II": 0, "model": "stub"}
    return PredictionResult(rows=rows, summary=summary)
