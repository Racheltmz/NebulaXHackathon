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


class JobSummaryOut(BaseModel):
    id: str
    subsystem: str
    status: str
    input_files: list[InputFileInfo]
    summary: dict
    created_at: datetime


class JobDetailOut(JobSummaryOut):
    rows: list[PredictionRowOut]
