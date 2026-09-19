"""Axle-box signals for the Rail Corrugation run dashboard: a compact, downsampled copy of the
1-second recording, so the dashboard can plot all 64 axle boxes without re-reading the CSV.

Display-only, like ml/acv_telemetry.py and ml/door_telemetry.py — nothing here feeds the
classification or `rail_predictions.csv`.

The raw file is 10,000 samples per channel across 128 channels, far too much to store or draw, and
a decimated waveform of an oscillating signal is meaningless anyway. So each channel is reduced to
the two things that actually carry corrugation:

- an **amplitude envelope** (RMS per time bin) — how hard that box is shaking through the second;
- a **spectrum** (up to 600 Hz) — corrugation excites a characteristic frequency band rather than
  raising the overall level, so this is where a corrugated rail separates from a healthy one
  (info kit, Section 1.1).

Positions 1, 3, 5, 7 ride the Side I rail and 2, 4, 6, 8 the Side II rail (info kit, Section 2.1),
so the per-side spectra below are the comparison the Side I / Side II call rests on.
"""

import io
import re

import numpy as np
import pandas as pd

TIME_BINS = 120  # amplitude-envelope points per channel, over the 1 s recording
# Corrugation excites a band well under 200 Hz, and everything above is flat noise that would
# squash the part worth reading — so the stored spectrum stops just past it.
SPECTRUM_HZ = 600.0
SPECTRUM_BINS = 128
SAMPLE_RATE = 10_000.0  # Hz (info kit, Section 2.1)
CORRUGATION_BAND = (40.0, 150.0)  # where the side-to-side comparison is taken

# Speed sensor: a 90-tooth wheel toggling 0/1, on a 0.85 m wheel (info kit, Section 2.1).
TEETH_PER_REV = 90.0
WHEEL_DIAMETER_M = 0.85

CHANNEL_RE = re.compile(r"(Vibration|Shock)\s+of\s+bearing\s+in\s+position\s+(\d+)\s+of\s+car\s+(\d+)", re.I)

SIDE_OF_POSITION = {p: ("I" if p % 2 == 1 else "II") for p in range(1, 9)}


def _round(values, digits=4):
    return [round(float(v), digits) for v in values]


def _bin_rms(values: np.ndarray, bins: int) -> list:
    """RMS of each equal slice — the envelope of an oscillating signal, not a decimation of it."""
    usable = (len(values) // bins) * bins
    if usable == 0:
        return _round(np.sqrt(values**2))
    block = values[:usable].reshape(bins, -1)
    return _round(np.sqrt((block**2).mean(axis=1)))


def _spectrum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum, Hann-windowed, up to SPECTRUM_HZ."""
    centred = values - values.mean()
    magnitude = np.abs(np.fft.rfft(centred * np.hanning(len(centred)))) / len(centred)
    freqs = np.fft.rfftfreq(len(centred), 1.0 / SAMPLE_RATE)
    keep = freqs <= SPECTRUM_HZ
    return freqs[keep], magnitude[keep]


def _reduce_spectrum(magnitude: np.ndarray, bins: int) -> np.ndarray:
    """Peak per bin — a mean would wash out the narrow peak that corrugation shows up as."""
    usable = (len(magnitude) // bins) * bins
    return magnitude[:usable].reshape(bins, -1).max(axis=1)


def _speed_kmh(toggles: np.ndarray) -> float | None:
    """Train speed from the 0/1 tooth signal: count transitions over the recording."""
    valid = toggles[~np.isnan(toggles)]
    if len(valid) < 2:
        return None
    transitions = int(np.count_nonzero(np.diff(valid) != 0))
    revolutions = (transitions / 2.0) / TEETH_PER_REV
    seconds = len(valid) / SAMPLE_RATE
    if seconds <= 0:
        return None
    return round(revolutions * np.pi * WHEEL_DIAMETER_M / seconds * 3.6, 1)


def build_telemetry(df: pd.DataFrame) -> dict | None:
    """Reduce one recording to per-axle-box envelopes, spectra and summary stats. Returns None if
    the file carries no recognisable `<Vibration|Shock> of bearing in position N of car M` columns.
    """
    channels: dict[tuple[int, int], dict[str, str]] = {}
    for column in df.columns:
        match = CHANNEL_RE.match(str(column).strip())
        if match:
            kind, position, car = match.group(1).lower(), int(match.group(2)), int(match.group(3))
            channels.setdefault((car, position), {})[kind] = column
    if not channels:
        return None

    freqs, _ = _spectrum(np.zeros(len(df)) if len(df) else np.zeros(2))
    spectrum_freqs = _round(_reduce_spectrum(freqs, SPECTRUM_BINS), 1)
    band = (np.array(spectrum_freqs) >= CORRUGATION_BAND[0]) & (np.array(spectrum_freqs) <= CORRUGATION_BAND[1])

    boxes = {}
    side_spectra: dict[str, list] = {"I": [], "II": []}
    for (car, position), columns in sorted(channels.items()):
        entry = {"car": car, "position": position, "side": SIDE_OF_POSITION.get(position)}
        for kind in ("vibration", "shock"):
            column = columns.get(kind)
            if column is None:
                continue
            values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0)
            if not len(values):
                continue
            entry[kind] = _bin_rms(values, TIME_BINS)
            entry[f"rms_{kind}"] = round(float(np.sqrt((values**2).mean())), 4)
            if kind == "vibration":
                _, magnitude = _spectrum(values)
                reduced = _reduce_spectrum(magnitude, SPECTRUM_BINS)
                entry["spectrum"] = _round(reduced)
                in_band = reduced[band] if band.any() else reduced
                entry["peak_amplitude"] = round(float(in_band.max()), 4)
                entry["peak_frequency"] = spectrum_freqs[int(np.argmax(reduced * band))] if band.any() else None
                if entry["side"] in side_spectra:
                    side_spectra[entry["side"]].append(reduced)
        boxes[f"{car}-{position}"] = entry

    sides = {}
    for side, spectra in side_spectra.items():
        if not spectra:
            continue
        mean_spectrum = np.mean(spectra, axis=0)
        in_band = mean_spectrum[band] if band.any() else mean_spectrum
        sides[side] = {
            "spectrum": _round(mean_spectrum),
            "peak_amplitude": round(float(in_band.max()), 4),
            "peak_frequency": spectrum_freqs[int(np.argmax(mean_spectrum * band))] if band.any() else None,
        }

    speed_column = next((c for c in df.columns if "speed" in str(c).lower()), None)
    speed = _speed_kmh(pd.to_numeric(df[speed_column], errors="coerce").to_numpy(dtype=float)) if speed_column else None

    return {
        "samples": int(len(df)),
        "duration_s": round(len(df) / SAMPLE_RATE, 3),
        "speed_kmh": speed,
        "time_bins": TIME_BINS,
        "spectrum_freqs": spectrum_freqs,
        "corrugation_band": list(CORRUGATION_BAND),
        "boxes": boxes,
        "sides": sides,
    }


def build_from_csv(content: bytes) -> dict | None:
    return build_telemetry(pd.read_csv(io.BytesIO(content)))
