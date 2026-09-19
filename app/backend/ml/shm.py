"""SHM cumulative fatigue damage regression.

Runs the evolved classical model vendored from the `ian-model` branch
(`models/shm_classical.py`): an ExtraTrees regressor on the raw target blended with a
log-target ExtraTrees, a random forest and a Huber-loss gradient booster, over multiscale
statistical descriptors of the stress trace.

The trace is passed through as [C, T] — numeric columns transposed, exactly as that branch's
`data/ps3_prepare.py` builds it — because the model's own feature extraction expects the raw
series, not precomputed damage features.

The model carries no weights of its own (it retrains in `fit()`), so it is fitted once against
the 64 labelled training traces by `scripts/fit_models.py` and loaded here.

The team's earlier hand-trained model is still in `ml_artifacts/shm_model.joblib` and its
rainflow feature code is preserved in `Optional_Items/SHM/code/shm.ipynb`, should a comparison
be wanted.
"""

import io
from pathlib import Path

import joblib
import numpy as np

from . import featurize
from .common import PredictionResult, UploadedFile

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml_artifacts" / "shm_classical.joblib"

_model = None


def _load_model():
    """Loaded on first prediction rather than at import, so a missing artifact surfaces as a
    clear request-time error instead of taking the whole app down at startup."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"SHM model artifact is missing ({MODEL_PATH.name}). Build it with "
                "`python app/backend/scripts/fit_models.py shm`."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def validate(files: list[UploadedFile]) -> None:
    for f in files:
        try:
            signal = np.loadtxt(io.BytesIO(f.content), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 — surfaced verbatim as the validation error
            raise ValueError(
                f"'{f.filename}' could not be parsed as a single column of raw numeric "
                f"readings with no header row: {exc}"
            ) from exc
        if signal.ndim != 1:
            raise ValueError(
                f"'{f.filename}' must contain exactly one column of readings, got shape "
                f"{signal.shape}."
            )
        if signal.size == 0:
            raise ValueError(f"'{f.filename}' is empty.")


def predict(files: list[UploadedFile]) -> PredictionResult:
    model = _load_model()
    arrays = [featurize.shm_array(f.content) for f in files]
    values = [float(v) for v in model.predict(arrays)]

    rows = [{"file_id": f.filename, "value": value} for f, value in zip(files, values)]
    summary = {}
    if values:
        summary = {
            "mean": round(float(np.mean(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
            "files": len(values),
            "model": "evolved classical (ExtraTrees/RF/GB blend)",
        }
    return PredictionResult(rows=rows, summary=summary)
