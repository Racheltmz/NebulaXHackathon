"""Door segment detection — STUB.

No trained model exists yet for this subsystem (see docs/DESIGN.md Section 8). This placeholder
lets the rest of the app (upload, storage, history, dashboard) be built and tested end-to-end
now; swap the body of `predict()` for the real segmentation + classification pipeline later
without touching any other layer.

Placeholder behaviour: treats each uploaded stream as a single unsegmented span from its first
to its last timestamp, labelled "Normal" (the schema only allows "Normal" or "Abnormal
resistance" — this is not a real detection).
"""

import io

import pandas as pd

from .common import PredictionResult, UploadedFile

EXPECTED_COLUMNS = 17


def validate(files: list[UploadedFile]) -> None:
    for f in files:
        try:
            header = pd.read_csv(io.BytesIO(f.content), nrows=0)
        except Exception as exc:  # noqa: BLE001 — surfaced verbatim as the validation error
            raise ValueError(f"'{f.filename}' could not be read as CSV: {exc}") from exc
        if len(header.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"'{f.filename}' has {len(header.columns)} columns, expected {EXPECTED_COLUMNS} "
                "(Datetime, motor current/voltage/back-EMF, door opening/closing time, "
                "close/open command, DCSR, DCSL, DLSR, DLSL, door opened/locked, door is "
                "opening/closing, door leaf position)."
            )


def predict(files: list[UploadedFile]) -> PredictionResult:
    rows = []
    for f in files:
        df = pd.read_csv(io.BytesIO(f.content))
        time_col = df.columns[0]
        rows.append(
            {
                "file_id": None,
                "start_time": str(df[time_col].iloc[0]),
                "end_time": str(df[time_col].iloc[-1]),
                "label": "Normal",
            }
        )

    summary = {"segments": len(rows), "abnormal": 0, "model": "stub"}
    return PredictionResult(rows=rows, summary=summary)
