from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session

import config
import storage
from db import PredictionJob, PredictionRow, get_db
from ml.export import rows_to_csv_bytes
from schemas import InputFileInfo, JobDetailOut, PredictionRowOut

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _get_job_or_404(db: Session, job_id: str) -> PredictionJob:
    job = db.get(PredictionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/{job_id}", response_model=JobDetailOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = _get_job_or_404(db, job_id)
    rows = db.query(PredictionRow).filter(PredictionRow.job_id == job.id).all()

    return JobDetailOut(
        id=str(job.id),
        subsystem=job.subsystem,
        status=job.status,
        input_files=[InputFileInfo(**f) for f in job.input_files],
        summary=job.summary,
        created_at=job.created_at,
        rows=[
            PredictionRowOut(
                file_id=r.file_id,
                start_time=r.start_time,
                end_time=r.end_time,
                label=r.label,
                ranked_cars=r.ranked_cars,
                value=r.value,
            )
            for r in rows
        ],
    )


@router.get("/{job_id}/download")
def download_job(job_id: str, db: Session = Depends(get_db)):
    job = _get_job_or_404(db, job_id)

    if job.output_storage_path:
        try:
            url = storage.create_signed_url(config.PREDICTIONS_BUCKET, job.output_storage_path)
            return RedirectResponse(url)
        except Exception:  # noqa: BLE001 — missing creds, missing bucket, etc. — fall back to regenerating below
            pass

    rows = db.query(PredictionRow).filter(PredictionRow.job_id == job.id).all()
    row_dicts = [
        {
            "file_id": r.file_id,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "label": r.label,
            "ranked_cars": r.ranked_cars,
            "value": r.value,
        }
        for r in rows
    ]
    csv_bytes = rows_to_csv_bytes(job.subsystem, row_dicts)
    filename = f"{job.subsystem}_predictions.csv"
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
