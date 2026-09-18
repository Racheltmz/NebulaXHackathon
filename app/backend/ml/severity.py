"""Per-run severity shown in the History table — Normal / Elevated / Critical.

PS3 doesn't define a severity scale, so these rules are ours: each subsystem's own prediction
mapped onto one shared three-level scale, kept in this one file so the thresholds are easy to
change. Severity is display-only — it never goes into the `*_predictions.csv` downloads, whose
schema is fixed by the spec.
"""

# SHM damage is the fraction of fatigue life used (Miner's rule: 1.0 = failure, and none of the
# labelled data comes close — max 0.93), so the bands are fixed cut-offs on that 0–1 scale rather
# than anything relative to other records.
SHM_ELEVATED_THRESHOLD = 0.5  # half the fatigue life used
SHM_CRITICAL_THRESHOLD = 0.8  # within 20% of failure
DOOR_CRITICAL_ABNORMAL_SHARE = 0.5  # at least this share of a stream's segments abnormal

NORMAL, ELEVATED, CRITICAL = "normal", "elevated", "critical"


def compute_severity(subsystem: str, rows: list) -> dict | None:
    """rows: the run's PredictionRow objects. Returns {"level", "detail"}, or None when the
    subsystem's output can't be graded (ACV only ranks cars — it gives no fault magnitude)."""
    if not rows:
        return None

    if subsystem == "shm":
        values = [r.value for r in rows if r.value is not None]
        if not values:
            return None
        value = sum(values) / len(values)
        if value >= SHM_CRITICAL_THRESHOLD:
            level = CRITICAL
        elif value >= SHM_ELEVATED_THRESHOLD:
            level = ELEVATED
        else:
            level = NORMAL
        return {"level": level, "detail": f"Predicted cumulative damage {value:.4f}"}

    if subsystem == "rail_corrugation":
        faults = sorted({r.label for r in rows if r.label in ("Side I", "Side II")})
        if faults:
            return {"level": ELEVATED, "detail": f"Corrugation detected: {', '.join(faults)}"}
        return {"level": NORMAL, "detail": "No corrugation detected"}

    if subsystem == "door":
        abnormal = sum(1 for r in rows if r.label == "Abnormal resistance")
        detail = f"{abnormal} of {len(rows)} segments abnormal"
        if abnormal == 0:
            return {"level": NORMAL, "detail": detail}
        level = CRITICAL if abnormal / len(rows) >= DOOR_CRITICAL_ABNORMAL_SHARE else ELEVATED
        return {"level": level, "detail": detail}

    return None
