from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db import PredictionJob, PredictionRow, get_db
from ml.export import rows_to_csv_bytes
from ml.severity import compute_severity
from ml.subsystems import SUBSYSTEMS
from schemas import HistoryRowOut, InputFileInfo, JobSummaryOut

router = APIRouter(prefix="/api/history", tags=["history"])


def _match_input_file(input_files: list[dict], file_id: str | None) -> tuple[int | None, str | None, bool]:
    """Which uploaded file a prediction row came from: by filename when the row carries a file_id
    (ACV/Rail/SHM), or the run's only file otherwise (Door rows are segments of one stream)."""
    index = None
    if file_id is not None:
        index = next((i for i, f in enumerate(input_files) if f["filename"] == file_id), None)
    elif len(input_files) == 1:
        index = 0
    if index is None:
        return None, None, False
    meta = input_files[index]
    return index, meta["filename"], bool(meta.get("storage_path"))


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
    jobs = query.all()

    rows_by_job = defaultdict(list)
    if jobs:
        for r in db.query(PredictionRow).filter(PredictionRow.job_id.in_([j.id for j in jobs])):
            rows_by_job[r.job_id].append(r)

    return [
        JobSummaryOut(
            id=str(job.id),
            subsystem=job.subsystem,
            status=job.status,
            input_files=[InputFileInfo(**f) for f in job.input_files],
            # ACV's per-car telemetry is large and only the run dashboard plots it.
            summary={k: v for k, v in (job.summary or {}).items() if k != "telemetry"},
            created_at=job.created_at,
            severity=compute_severity(job.subsystem, rows_by_job[job.id]),
        )
        for job in jobs
    ]


def _latest_rows(db: Session, subsystem: str) -> list[PredictionRow]:
    """Every finished run's rows for one subsystem, collapsed to "one row per file" — if the same
    file was run more than once the latest run wins. Door has no file_id (one row per detected
    segment), so its segments dedupe on their timestamps instead.
    """
    if subsystem not in SUBSYSTEMS:
        raise HTTPException(status_code=404, detail=f"Unknown subsystem '{subsystem}'")

    rows = (
        db.query(PredictionRow)
        .join(PredictionJob, PredictionJob.id == PredictionRow.job_id)
        .filter(PredictionJob.subsystem == subsystem, PredictionJob.status == "done")
        .order_by(PredictionJob.created_at.asc())
        .all()
    )
    latest: dict = {}
    for r in rows:
        latest[(r.start_time, r.end_time) if subsystem == "door" else r.file_id] = r
    return list(latest.values())


def _row_dict(r: PredictionRow) -> dict:
    return {
        "file_id": r.file_id,
        "start_time": r.start_time,
        "end_time": r.end_time,
        "label": r.label,
        "ranked_cars": r.ranked_cars,
        "value": r.value,
    }


@router.get("/dashboard")
def subsystem_dashboard(subsystem: str = Query(...), db: Session = Depends(get_db)):
    """Everything the per-subsystem dashboard on the History page charts: the latest prediction
    for every file that subsystem has been run on, plus how many runs that spans."""
    rows = _latest_rows(db, subsystem)
    dashboard = {
        "subsystem": subsystem,
        "runs": len({r.job_id for r in rows}),
        "rows": [_row_dict(r) for r in rows],
    }
    if subsystem == "acv":
        # Car model isn't a prediction_rows column — it lives on each run's summary. Runs from
        # before it was recorded have none, so they map to null.
        jobs = {
            job.id: job
            for job in db.query(PredictionJob).filter(PredictionJob.id.in_(list({r.job_id for r in rows})))
        }
        dashboard["car_models"] = {
            r.file_id: (jobs[r.job_id].summary or {}).get("car_models", {}).get(r.file_id) for r in rows
        }
    return dashboard


@router.get("/download")
def download_subsystem_predictions(subsystem: str = Query(...), db: Session = Depends(get_db)):
    """One `<subsystem>_predictions.csv` covering every finished run of that subsystem, in the
    schema PS3 Section 4.1 asks for — this is the file that goes into predictions.zip. Runs are
    one-file-each, so this stitches them back together.
    """
    rows = _latest_rows(db, subsystem)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No completed {SUBSYSTEMS[subsystem].label} runs yet — run a prediction first.",
        )

    return StreamingResponse(
        iter([rows_to_csv_bytes(subsystem, [_row_dict(r) for r in rows])]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{subsystem}_predictions.csv"'},
    )


@router.get("/{subsystem_key}/rows", response_model=list[HistoryRowOut])
def list_history_rows(subsystem_key: str, db: Session = Depends(get_db)):
    """Every prediction row for one subsystem, newest run first — one query so the History
    tab doesn't need a request per run."""
    results = (
        db.query(PredictionRow, PredictionJob)
        .join(PredictionJob, PredictionJob.id == PredictionRow.job_id)
        .filter(PredictionJob.subsystem == subsystem_key)
        .order_by(PredictionJob.created_at.desc(), PredictionRow.file_id, PredictionRow.start_time)
        .all()
    )

    rows_by_job = defaultdict(list)
    for row, job in results:
        rows_by_job[job.id].append(row)
    severity_by_job = {
        job_id: compute_severity(subsystem_key, job_rows) for job_id, job_rows in rows_by_job.items()
    }

    out = []
    for row, job in results:
        index, name, available = _match_input_file(job.input_files, row.file_id)
        summary = job.summary or {}
        out.append(
            HistoryRowOut(
                job_id=str(job.id),
                created_at=job.created_at,
                severity=severity_by_job[job.id],
                file_id=row.file_id,
                start_time=row.start_time,
                end_time=row.end_time,
                label=row.label,
                ranked_cars=row.ranked_cars,
                value=row.value,
                input_file_index=index,
                input_file_name=name,
                input_file_available=available,
                car_model=(summary.get("car_models") or {}).get(row.file_id),
                train_number=(summary.get("train_numbers") or {}).get(row.file_id),
            )
        )
    return out
