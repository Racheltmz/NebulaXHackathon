"""ACV refrigerant-leak localisation — rank the 8 cars, most to least likely faulty.

Runs the evolved classical model vendored from the `ian-model` branch
(`models/acv_classical.py`), which scores each car and blends three views of it: a
difference-of-means linear weighting, a k-nearest-neighbour retrieval term against the faulty
cars seen in training, and a distance-to-centroid term (0.70 / 0.18 / 0.12).

Each car becomes an [8, T] array: four per-timestep statistics across that car's own parameters,
then the same four expressed relative to the rest of the train. The residual channels matter
because a leak is only visible *relative to the other cars* — all 8 share one train, one route
and the same weather, so a car that cannot cool stands out against its peers rather than against
any absolute threshold.

The model carries no weights of its own (it retrains in `fit()`), so it is fitted once against
all six labelled training cases by `scripts/fit_models.py` and loaded here. `predict` returns a
continuous fault score per car; ranking them descending gives `ranked_cars`.
"""

import io
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import featurize
from .common import PredictionResult, UploadedFile

CAR_COLUMN_RE = re.compile(r"Car\s*(\d+)\s*-")

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml_artifacts" / "acv_classical.joblib"

_model = None


def _load_model():
    """Loaded on first prediction rather than at import, so a missing artifact surfaces as a
    clear request-time error instead of taking the whole app down at startup."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"ACV model artifact is missing ({MODEL_PATH.name}). Build it with "
                "`python app/backend/scripts/fit_models.py acv`."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


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
    model = _load_model()
    rows = []
    top_cars = []
    for f in files:
        # sheet_name=0 because the held-out test file's sheet is named in Chinese.
        df = pd.read_excel(io.BytesIO(f.content), sheet_name=0)
        cars, arrays = featurize.acv_case_arrays(df)
        scores = np.asarray(model.predict(arrays), dtype=float)
        # Every car in the file is listed: one missing from `ranked_cars` scores 0 under the
        # official rank-decay metric, so there is never a reason to drop one.
        order = [cars[i] for i in np.argsort(-scores)]
        rows.append({"file_id": f.filename, "ranked_cars": "|".join(order)})
        if order:
            top_cars.append(order[0])

    summary = {
        "files": len(rows),
        "top_car": top_cars[0] if top_cars else None,
        "model": "evolved classical (linear + retrieval + centroid)",
    }
    return PredictionResult(rows=rows, summary=summary)
