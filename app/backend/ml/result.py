"""The one-glance result pill per run on the History page's All tab.

Each subsystem reports its own kind of output (SHM a damage grade, Door and Rail a status, ACV a
car ranking). This maps every one onto the same three tones so an engineer can scan the column by
colour: normal (green), warning (amber) and fault (red). Display-only — it never goes into the
`*_predictions.csv` downloads, whose schema is fixed by the spec.
"""

from .severity import compute_severity

NORMAL, WARNING, FAULT = "normal", "warning", "fault"

_SEVERITY_TONE = {"normal": NORMAL, "elevated": WARNING, "critical": FAULT}


def compute_result(subsystem: str, rows: list) -> dict | None:
    """rows: the run's PredictionRow objects. Returns {"label", "tone", "detail"}, or None when
    there is nothing to show."""
    if not rows:
        return None

    if subsystem == "shm":
        severity = compute_severity(subsystem, rows)
        if severity is None:
            return None
        level = severity["level"]
        return {"label": level.capitalize(), "tone": _SEVERITY_TONE[level], "detail": severity["detail"]}

    if subsystem == "door":
        abnormal = sum(1 for r in rows if r.label == "Abnormal resistance")
        return {
            "label": "Abnormal resistance" if abnormal else "Normal",
            "tone": FAULT if abnormal else NORMAL,
            "detail": f"{abnormal} of {len(rows)} segments abnormal",
        }

    if subsystem == "rail_corrugation":
        faults = sorted({r.label for r in rows if r.label in ("Side I", "Side II")})
        if faults:
            return {"label": ", ".join(faults), "tone": FAULT, "detail": f"Corrugation detected: {', '.join(faults)}"}
        return {"label": "Normal", "tone": NORMAL, "detail": "No corrugation detected"}

    if subsystem == "acv":
        top_car = next((r.ranked_cars.split("|")[0] for r in rows if r.ranked_cars), None)
        if not top_car:
            return None
        return {
            "label": f"Car {top_car}",
            "tone": FAULT,
            "detail": "Ranked first: the car most likely to have the refrigerant leak",
        }

    return None
