import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

import config
import storage
from db import PredictionJob, PredictionRow, get_db
from ml.common import UploadedFile
from ml.export import rows_to_csv_bytes
from ml.subsystems import get_subsystem

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("/{subsystem_key}")
async def run_prediction(
    subsystem_key: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """One upload = one run: every file becomes its own `prediction_jobs` row so it shows up as
    a separate entry in History. The frontend sends a multi-file selection as one request per file.
    """
    try:
        subsystem = get_subsystem(subsystem_key)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in subsystem.accepted_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{file.filename}' has an unsupported extension for {subsystem.label} — "
                f"expected one of {subsystem.accepted_extensions}."
            ),
        )
    uploaded = [UploadedFile(filename=file.filename, content=await file.read())]

    try:
        subsystem.validate_fn(uploaded)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = subsystem.predict_fn(uploaded)
    except Exception as exc:  # noqa: BLE001 — surfaced to the user as the job's error_message
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    job_id = uuid.uuid4()

    input_files_meta = []
    for f in uploaded:
        storage_path = f"{job_id}/{f.filename}"
        try:
            storage.upload_file(config.UPLOADS_BUCKET, storage_path, f.content)
        except Exception:  # noqa: BLE001 — missing creds, missing bucket, etc. — job still recorded either way
            storage_path = None
        input_files_meta.append(
            {"filename": f.filename, "storage_path": storage_path, "size_bytes": len(f.content)}
        )

    output_csv = rows_to_csv_bytes(subsystem_key, result.rows)
    output_storage_path = f"{job_id}/{subsystem_key}_predictions.csv"
    try:
        storage.upload_file(config.PREDICTIONS_BUCKET, output_storage_path, output_csv, "text/csv")
    except Exception:  # noqa: BLE001 — missing creds, missing bucket, etc.
        output_storage_path = None

    job = PredictionJob(
        id=job_id,
        subsystem=subsystem_key,
        status="done",
        input_files=input_files_meta,
        output_storage_path=output_storage_path,
        summary=result.summary,
    )
    db.add(job)
    db.flush()

    for row in result.rows:
        db.add(
            PredictionRow(
                job_id=job_id,
                file_id=row.get("file_id"),
                start_time=row.get("start_time"),
                end_time=row.get("end_time"),
                label=row.get("label"),
                ranked_cars=row.get("ranked_cars"),
                value=row.get("value"),
            )
        )
    db.commit()

    return {"job_id": str(job_id), "summary": result.summary, "rows": result.rows}
