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
