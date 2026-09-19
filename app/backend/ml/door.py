"""Door fault detection — find each open/close cycle in a continuous stream, then classify it.

Two stages, because the task is segmentation *and* classification:

1. **Segmentation.** The uploaded stream is separate door-cycle recordings laid end to end. Rows
   inside a cycle are a uniform 20 ms apart; between cycles the clock jumps by tens of seconds.
   Splitting on that gap recovers all 110 segments of the labelled training stream with exact
   boundaries (mean IoU 1.000). See `featurize.door_segment_bounds`.

2. **Classification.** Each segment goes to the evolved classical model vendored from the
   `ian-model` branch (`models/door_classical.py`) — an RBF SVC over per-channel statistics and
   derivative summaries — which labels it Normal or Abnormal resistance.

The split matters for scoring: the official metric is IoU-weighted F1 over predicted segments, so
a wrong boundary costs score even when the label is right. The evolved model was only ever scored
on *given* boundaries, where that term collapses to plain accuracy — the segmenter above is what
supplies the boundaries it never had to find.

The model carries no weights of its own (it retrains in `fit()`), so it is fitted once against the
110 labelled training segments by `scripts/fit_models.py` and loaded here.
"""

import io
import logging
from pathlib import Path

import joblib
import pandas as pd

from . import featurize
from .common import PredictionResult, UploadedFile
from .door_telemetry import build_telemetry

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = 17
ABNORMAL = "Abnormal resistance"

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml_artifacts" / "door_classical.joblib"

_model = None


def _load_model():
    """Loaded on first prediction rather than at import, so a missing artifact surfaces as a
    clear request-time error instead of taking the whole app down at startup."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Door model artifact is missing ({MODEL_PATH.name}). Build it with "
                "`python app/backend/scripts/fit_models.py door`."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


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
    model = _load_model()
    rows = []
    telemetry = {}
    for f in files:
        df = pd.read_csv(io.BytesIO(f.content))

        # Display-only, so it must never fail the run — the segments below are what's being asked for.
        try:
            telemetry[f.filename] = build_telemetry(df)
        except Exception:  # noqa: BLE001
            logger.exception("Could not build door telemetry for %s", f.filename)

        bounds = featurize.door_segment_bounds(df)
        arrays = featurize.door_arrays_from_bounds(df, bounds)

        # A segment of a handful of rows carries no usable cycle; the training prep dropped these
        # too, so the model has never seen one. Report it as Normal rather than guessing.
        scorable = [i for i, a in enumerate(arrays) if a.shape[-1] > 4]
        labels = ["Normal"] * len(arrays)
        if scorable:
            predicted = model.predict([arrays[i] for i in scorable])
            for i, label in zip(scorable, predicted):
                labels[i] = str(label)

        times = df.Datetime.astype(str).to_numpy()
        for (start, end), label in zip(bounds, labels):
            rows.append(
                {
                    "file_id": None,
                    "start_time": times[start],
                    "end_time": times[end],
                    "label": label,
                }
            )

    abnormal = sum(1 for r in rows if r["label"] == ABNORMAL)
    summary = {
        "segments": len(rows),
        "abnormal": abnormal,
        "telemetry": telemetry,
        "model": "gap segmentation + evolved classical SVC",
    }
    return PredictionResult(rows=rows, summary=summary)
