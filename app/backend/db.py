"""SQLAlchemy models + session handling for the Supabase Postgres database.

Table DDL lives in schema.sql (run once in the Supabase SQL editor) — these ORM classes map onto
tables created there, they do not create the schema themselves.

Auth/user tracking is deferred for now (see docs/DESIGN.md) — there is no `profiles` table and
`prediction_jobs` has no `user_id`. Re-add both together when login comes back.

The engine is created lazily: importing this module without DATABASE_URL set (e.g. before
Supabase credentials are configured) does not fail — only actually querying does.
"""

import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import Column, DateTime, Float, ForeignKey, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

import config


class Base(DeclarativeBase):
    pass


class PredictionJob(Base):
    __tablename__ = "prediction_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subsystem = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default="processing")
    input_files = Column(JSONB, nullable=False, default=list)
    output_storage_path = Column(Text, nullable=True)
    summary = Column(JSONB, nullable=False, default=dict)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class PredictionRow(Base):
    __tablename__ = "prediction_rows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("prediction_jobs.id"), nullable=False)
    file_id = Column(Text, nullable=True)
    start_time = Column(Text, nullable=True)
    end_time = Column(Text, nullable=True)
    label = Column(Text, nullable=True)
    ranked_cars = Column(Text, nullable=True)
    value = Column(Float, nullable=True)


_engine = None
_SessionLocal = None


def _get_session_factory():
    global _engine, _SessionLocal
    if _SessionLocal is None:
        if not config.DATABASE_URL:
            raise HTTPException(
                status_code=503,
                detail="DATABASE_URL is not configured yet — add your Supabase Postgres "
                "connection string to app/backend/.env before using any endpoint that touches "
                "the database.",
            )
        _engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _SessionLocal


def get_db():
    session: Session = _get_session_factory()()
    try:
        yield session
    finally:
        session.close()
