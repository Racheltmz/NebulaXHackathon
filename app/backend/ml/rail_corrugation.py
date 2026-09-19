"""Rail corrugation 3-class classification — Normal / Side I / Side II.

Runs the evolved classical model vendored from the `ian-model` branch (`models/rail_classical.py`):
three class-balanced SVCs at C = 0.35 / 0.6 / 1.0 over descriptors of a compact 5-channel view of
the recording, combined by majority vote.

Positions 1, 3, 5, 7 of each axle box ride the Side I rail and 2, 4, 6, 8 ride Side II, so the
128 sensor channels collapse to per-side mean vibration and shock alongside the tachometer — the
side contrast is what separates Side I from Side II, and a corrugated rail excites its own side.

The model carries no weights of its own (it retrains in `fit()`), so it is fitted once against all
272 labelled training recordings by `scripts/fit_models.py` and loaded here.
"""

import io
from pathlib import Path

import joblib
import pandas as pd

from . import featurize
from .common import PredictionResult, UploadedFile

EXPECTED_COLUMNS = 129
CLASSES = ("Normal", "Side I", "Side II")

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml_artifacts" / "rail_classical.joblib"

_model = None


def _load_model():
    """Loaded once on first prediction, not at import, so a missing artifact surfaces as a clear
    request-time error instead of taking the whole app down at startup."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Rail model artifact is missing ({MODEL_PATH.name}). Build it with "
                "`python app/backend/scripts/fit_models.py rail`."
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
                "(rotating speed, then vibration + shock readings for each of 64 axle-box "
                "positions across 8 cars)."
            )


def predict(files: list[UploadedFile]) -> PredictionResult:
    model = _load_model()
    arrays = [featurize.rail_array(f.content) for f in files]
    labels = [str(label) for label in model.predict(arrays)]

    rows = [{"file_id": f.filename, "label": label} for f, label in zip(files, labels)]
    summary = {cls: labels.count(cls) for cls in CLASSES}
    summary["model"] = "evolved classical (3x SVC vote)"
    return PredictionResult(rows=rows, summary=summary)
