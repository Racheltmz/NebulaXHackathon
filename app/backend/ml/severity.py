"""Per-run severity shown in the History table — Normal / Elevated / Critical.

Only SHM is graded: its prediction is a damage magnitude, so it maps naturally onto a scale. The
other subsystems output a status or a ranking (Door: Normal / Abnormal resistance, Rail
Corrugation: Normal / Side I / Side II, ACV: a car ranking) that is shown as it is, with no
severity on top. PS3 doesn't define a severity scale, so these rules are ours, kept in this one
file so the thresholds are easy to change. Severity is display-only — it never goes into the
`*_predictions.csv` downloads, whose schema is fixed by the spec.
"""

# SHM damage is the fraction of fatigue life used (Miner's rule: 1.0 = failure, and none of the
# labelled data comes close — max 0.93), so the bands are fixed cut-offs on that 0–1 scale rather
# than anything relative to other records.
SHM_ELEVATED_THRESHOLD = 0.5  # half the fatigue life used
SHM_CRITICAL_THRESHOLD = 0.8  # within 20% of failure

NORMAL, ELEVATED, CRITICAL = "normal", "elevated", "critical"


def compute_severity(subsystem: str, rows: list) -> dict | None:
    """rows: the run's PredictionRow objects. Returns {"level", "detail"}, or None for every
    subsystem except SHM."""
    if subsystem != "shm" or not rows:
        return None

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
