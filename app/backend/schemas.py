from datetime import datetime

from pydantic import BaseModel


class SubsystemInfo(BaseModel):
    key: str
    label: str
    accepted_extensions: list[str]
    schema_hint: str
    output_columns: list[str]


class InputFileInfo(BaseModel):
    filename: str
    storage_path: str | None = None
    size_bytes: int


class PredictionRowOut(BaseModel):
    file_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    label: str | None = None
    ranked_cars: str | None = None
    value: float | None = None


class SeverityOut(BaseModel):
    level: str  # "normal" | "elevated" | "critical"
    detail: str


class JobSummaryOut(BaseModel):
    id: str
    subsystem: str
    status: str
    input_files: list[InputFileInfo]
    summary: dict
    created_at: datetime
    severity: SeverityOut | None = None


class JobDetailOut(JobSummaryOut):
    rows: list[PredictionRowOut]


class HistoryRowOut(PredictionRowOut):
    """One prediction row plus enough of its parent run to render a per-prediction history table."""

    job_id: str
    created_at: datetime
    severity: SeverityOut | None = None
    input_file_index: int | None = None
    input_file_name: str | None = None
    input_file_available: bool = False
    car_model: str | None = None  # ACV only, from the run's summary
    train_number: str | None = None  # ACV only, from the run's summary
