"""SHM cumulative fatigue damage regression.

Ported feature-extraction logic from Optional_Items/SHM/code/shm.ipynb — must stay in sync
with however shm_model.joblib was trained, since the model expects this exact feature set.
The joblib bundle stores its own `feature_cols`, so this module only needs to compute every
feature it might ask for; which subset actually feeds the model is decided by the artifact,
not by this file.
"""

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rainflow
from scipy import stats as sstats

from .common import PredictionResult, UploadedFile

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml_artifacts" / "shm_model.joblib"

PSEUDO_DAMAGE_EXPONENTS = (3, 5, 8, 10)

_bundle = None


def _load_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"SHM model artifact not found at {MODEL_PATH} — copy "
                "Optional_Items/SHM/model/shm_model.joblib there."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _load_signal(content: bytes) -> np.ndarray:
    return np.loadtxt(io.BytesIO(content), dtype=np.float64)


def validate(files: list[UploadedFile]) -> None:
    for f in files:
        try:
            signal = _load_signal(f.content)
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


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    avg = np.average(values, weights=weights)
    var = np.average((values - avg) ** 2, weights=weights)
    return float(np.sqrt(var))


def _raw_signal_features(signal: np.ndarray) -> dict:
    return {
        "raw_mean": float(signal.mean()),
        "raw_std": float(signal.std()),
        "raw_rms": float(np.sqrt(np.mean(signal**2))),
        "raw_min": float(signal.min()),
        "raw_max": float(signal.max()),
        "raw_peak_to_peak": float(signal.max() - signal.min()),
        "raw_skew": float(sstats.skew(signal)),
        "raw_kurtosis": float(sstats.kurtosis(signal)),
    }


def _rainflow_features(signal: np.ndarray) -> dict:
    cycles = list(rainflow.extract_cycles(signal))
    ranges = np.array([c[0] for c in cycles])
    means = np.array([c[1] for c in cycles])
    counts = np.array([c[2] for c in cycles])
    amplitudes = ranges / 2.0

    feats = {
        "rf_n_cycles": float(len(cycles)),
        "rf_total_cycle_count": float(counts.sum()),
        "rf_amp_mean": float(np.average(amplitudes, weights=counts)),
        "rf_amp_std": _weighted_std(amplitudes, counts),
        "rf_amp_max": float(amplitudes.max()),
        "rf_amp_p90": float(np.percentile(amplitudes, 90)),
        "rf_amp_p99": float(np.percentile(amplitudes, 99)),
        "rf_mean_stress_mean": float(np.average(means, weights=counts)),
        "rf_mean_stress_std": _weighted_std(means, counts),
    }
    for b in PSEUDO_DAMAGE_EXPONENTS:
        feats[f"rf_pseudo_damage_b{b}"] = float(np.sum(counts * amplitudes**b))
    return feats


def _extract_features(content: bytes) -> dict:
    signal = _load_signal(content)
    feats = {}
    feats.update(_raw_signal_features(signal))
    feats.update(_rainflow_features(signal))
    return feats


def predict(files: list[UploadedFile]) -> PredictionResult:
    bundle = _load_bundle()
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    rows = []
    values = []
    for f in files:
        feats = _extract_features(f.content)
        x = pd.DataFrame([feats])[feature_cols]
        value = float(model.predict(x)[0])
        rows.append({"file_id": f.filename, "value": value})
        values.append(value)

    summary = {}
    if values:
        summary = {
            "mean": round(float(np.mean(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
            "files": len(values),
        }
    return PredictionResult(rows=rows, summary=summary)
