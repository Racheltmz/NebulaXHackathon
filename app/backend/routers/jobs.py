import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session

import config
import storage
from db import PredictionJob, PredictionRow, get_db
from ml.common import preview_table
from ml.door_telemetry import build_from_csv as build_door_telemetry
from ml.rail_telemetry import build_from_csv as build_rail_telemetry
from ml.export import rows_to_csv_bytes
from schemas import InputFileInfo, JobDetailOut, PredictionRowOut

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
logger = logging.getLogger(__name__)


def _get_job_or_404(db: Session, job_id: str) -> PredictionJob:
    job = db.get(PredictionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _get_input_file_or_404(job: PredictionJob, index: int) -> dict:
    if index < 0 or index >= len(job.input_files):
        raise HTTPException(status_code=404, detail="Input file not found")
    meta = job.input_files[index]
    if not meta.get("storage_path"):
        raise HTTPException(
            status_code=404,
            detail=f"'{meta['filename']}' was not saved to storage for this run.",
        )
    return meta


# Subsystems whose run dashboard plots the uploaded file's own readings, and the reducer for each.
# Both read a CSV of a few MB in well under a second, so a run saved before the readings were kept
# can have them rebuilt on the first page view rather than needing the file uploaded again. ACV is
# absent on purpose: its workbooks take seconds to tens of seconds to read, which is too slow here.
TELEMETRY_BUILDERS = {"door": build_door_telemetry, "rail_corrugation": build_rail_telemetry}


def _backfill_telemetry(db: Session, job: PredictionJob) -> None:
    build = TELEMETRY_BUILDERS.get(job.subsystem)
    if build is None:
        return
    summary = dict(job.summary or {})
    telemetry = dict(summary.get("telemetry") or {})
    changed = False
    for meta in job.input_files:
        name = meta["filename"]
        if name in telemetry or not meta.get("storage_path"):
            continue
        try:
            content = storage.download_file(config.UPLOADS_BUCKET, meta["storage_path"])
            telemetry[name] = build(content)
        except Exception:  # noqa: BLE001 — missing creds, missing file, unreadable CSV: just no chart
            logger.exception("Could not rebuild %s telemetry for %s", job.subsystem, name)
            continue
        changed = True
    if changed:
        job.summary = {**summary, "telemetry": telemetry}
        db.commit()


@router.get("/{job_id}", response_model=JobDetailOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = _get_job_or_404(db, job_id)
    _backfill_telemetry(db, job)
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


@router.get("/{job_id}/input-files/{index}/preview")
def preview_input_file(job_id: str, index: int, db: Session = Depends(get_db)):
    job = _get_job_or_404(db, job_id)
    meta = _get_input_file_or_404(job, index)

    try:
        content = storage.download_file(config.UPLOADS_BUCKET, meta["storage_path"])
    except Exception as exc:  # noqa: BLE001 — missing creds, missing bucket, etc.
        raise HTTPException(status_code=502, detail=f"Could not fetch file for preview: {exc}") from exc

    try:
        preview = preview_table(meta["filename"], content, job.subsystem)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"filename": meta["filename"], **preview}


@router.get("/{job_id}/input-files/{index}/download")
def download_input_file(job_id: str, index: int, db: Session = Depends(get_db)):
    job = _get_job_or_404(db, job_id)
    meta = _get_input_file_or_404(job, index)

    try:
        url = storage.create_signed_url(config.UPLOADS_BUCKET, meta["storage_path"])
    except Exception as exc:  # noqa: BLE001 — missing creds, missing bucket, etc.
        raise HTTPException(status_code=502, detail=f"Could not generate download link: {exc}") from exc
    return RedirectResponse(url)


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
