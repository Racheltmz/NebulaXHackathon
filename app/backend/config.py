"""Environment configuration. Every Supabase-related value is optional at import time —
routes that need it raise a clear 503 at request time instead of crashing app startup,
so the rest of the app (subsystem info, predict-without-persistence, etc.) stays usable
before credentials are supplied.
"""

import os

from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

DATABASE_URL = os.environ.get("DATABASE_URL")

UPLOADS_BUCKET = os.environ.get("SUPABASE_UPLOADS_BUCKET", "uploads")
PREDICTIONS_BUCKET = os.environ.get("SUPABASE_PREDICTIONS_BUCKET", "predictions")

CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
