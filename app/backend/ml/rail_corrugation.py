"""Rail corrugation 3-class classification — STUB.

No trained model exists yet for this subsystem (see docs/DESIGN.md Section 8). Placeholder
behaviour: classifies every file as "Normal", so the output shape is already submission-ready —
only the classification logic needs to change once a real model lands.
"""

from .common import PredictionResult, UploadedFile


def predict(files: list[UploadedFile]) -> PredictionResult:
    rows = [{"file_id": f.filename, "label": "Normal"} for f in files]
    summary = {"Normal": len(rows), "Side I": 0, "Side II": 0, "model": "stub"}
    return PredictionResult(rows=rows, summary=summary)
