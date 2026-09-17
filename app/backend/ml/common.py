from dataclasses import dataclass


@dataclass
class UploadedFile:
    filename: str
    content: bytes


@dataclass
class PredictionResult:
    """rows: dicts matching the `prediction_rows` table shape (Section 4 of docs/DESIGN.md) —
    only the columns relevant to the subsystem need to be present, the rest are left null.
    summary: small rollup persisted onto `prediction_jobs.summary` for the history row / dashboard header.
    """

    rows: list[dict]
    summary: dict
