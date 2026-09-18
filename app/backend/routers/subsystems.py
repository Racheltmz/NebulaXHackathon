from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db import PredictionJob, PredictionRow, get_db
from ml.subsystems import SUBSYSTEMS
from schemas import SubsystemInfo

router = APIRouter(prefix="/api/subsystems", tags=["subsystems"])


@router.get("", response_model=list[SubsystemInfo])
def list_subsystems():
    return [
        SubsystemInfo(
            key=s.key,
            label=s.label,
            accepted_extensions=s.accepted_extensions,
            schema_hint=s.schema_hint,
            output_columns=s.output_columns,
        )
        for s in SUBSYSTEMS.values()
    ]


@router.get("/shm/damage-distribution")
def shm_damage_distribution(db: Session = Depends(get_db)):
    """Every predicted cumulative-damage value ever recorded for SHM, across all runs — lets the
    dashboard show where one run's value sits relative to everything else the app has predicted.
    """
    rows = (
        db.query(PredictionRow.value)
        .join(PredictionJob, PredictionJob.id == PredictionRow.job_id)
        .filter(PredictionJob.subsystem == "shm", PredictionRow.value.isnot(None))
        .all()
    )
    return {"values": [r[0] for r in rows]}
