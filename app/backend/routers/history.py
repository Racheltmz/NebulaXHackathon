from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from db import PredictionJob, get_db
from schemas import InputFileInfo, JobSummaryOut

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=list[JobSummaryOut])
def list_history(
    subsystem: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Every run, shared across anyone using the app (docs/DESIGN.md Section 7.3) — there's no
    per-user scoping since there's no login yet."""
    query = db.query(PredictionJob)
    if subsystem:
        query = query.filter(PredictionJob.subsystem == subsystem)
    query = query.order_by(PredictionJob.created_at.desc())

    return [
        JobSummaryOut(
            id=str(job.id),
            subsystem=job.subsystem,
            status=job.status,
            input_files=[InputFileInfo(**f) for f in job.input_files],
            summary=job.summary,
            created_at=job.created_at,
        )
        for job in query.all()
    ]
