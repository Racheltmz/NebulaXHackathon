import io
import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class UploadedFile:
    filename: str
    content: bytes


PREVIEW_ROW_LIMIT = 25


def preview_table(filename: str, content: bytes, subsystem_key: str, limit: int = PREVIEW_ROW_LIMIT) -> dict:
    """Parses the first `limit` rows of an uploaded input file for display in the UI — not used
    by any prediction path, so it stays lenient (best-effort column names) rather than enforcing
    the strict per-subsystem contracts that `validate()` checks.
    """
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".xlsx":
            df = pd.read_excel(io.BytesIO(content), nrows=limit)
        elif subsystem_key == "shm":
            df = pd.read_csv(io.BytesIO(content), header=None, nrows=limit)
            df.columns = ["value"] if len(df.columns) == 1 else [f"value_{i + 1}" for i in range(len(df.columns))]
        else:
            df = pd.read_csv(io.BytesIO(content), nrows=limit)
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim as the preview error
        raise ValueError(f"Could not preview '{filename}': {exc}") from exc

    df = df.where(pd.notnull(df), None)
    return {
        "columns": [str(c) for c in df.columns],
        "rows": df.values.tolist(),
        "truncated": len(df) == limit,
    }


@dataclass
class PredictionResult:
    """rows: dicts matching the `prediction_rows` table shape (Section 4 of docs/DESIGN.md) —
    only the columns relevant to the subsystem need to be present, the rest are left null.
    summary: small rollup persisted onto `prediction_jobs.summary` for the history row / dashboard header.
    """

    rows: list[dict]
    summary: dict
