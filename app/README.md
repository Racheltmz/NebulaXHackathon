# App

Full design rationale lives in [`../docs/DESIGN.md`](../docs/DESIGN.md) — this file is just
setup/run instructions.

## Status

No login — deferred by design (see docs/DESIGN.md Section 7.0). The home page goes straight to
Predict. Subsystem info, file upload, and prediction all work today without any credentials (SHM
uses its real trained model; Door/ACV/Rail Corrugation use documented placeholder logic — see
`backend/ml/`); persistence (history, downloads-after-the-fact, dashboards you revisit later)
needs Supabase configured per Section 2 below.

## 1. Backend (FastAPI)

```bash
cd app/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in once you have a Supabase project (Section 2)
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/api/health` — it reports which of database/storage are configured.

## 2. Supabase setup (do this once you're ready to enable persistence)

1. Create a free project at [supabase.com](https://supabase.com).
2. In the SQL Editor, run `app/backend/schema.sql` — creates `prediction_jobs` and
   `prediction_rows`.
3. In Storage, create two **private** buckets: `uploads` and `predictions` (or set
   `SUPABASE_UPLOADS_BUCKET` / `SUPABASE_PREDICTIONS_BUCKET` to whatever names you used).
4. Copy these into `app/backend/.env`:
   - `SUPABASE_URL` — Project Settings → API (the bare project URL, no `/rest/v1/` suffix).
   - `SUPABASE_SERVICE_ROLE_KEY` — Project Settings → API Keys → the **secret** key
     (`sb_secret_...`), not the publishable one.
   - `DATABASE_URL` — Project Settings → Database → Connection string (URI) — replace
     `[YOUR-PASSWORD]` and `[YOUR-PROJECT-REF]` with your real values.
5. Restart the backend.

## 3. Frontend (React + Vite)

```bash
cd app/frontend
npm install
cp .env.example .env   # VITE_API_BASE_URL defaults to http://localhost:8000
npm run dev
```

Visit `http://localhost:5173`.

## Folder map

```
backend/
  main.py            FastAPI app + CORS
  config.py          env vars (all optional at import time)
  db.py              SQLAlchemy models (prediction_jobs, prediction_rows)
  storage.py         Supabase Storage upload/signed-url wrapper
  schema.sql          run once in the Supabase SQL editor
  ml/                 one predict() per subsystem (shm.py is real, the other 3 are stubs)
  ml_artifacts/       shm_model.joblib
  routers/            subsystems, predict, jobs, history
frontend/
  src/pages/          Predict (home page), Dashboard, History
  src/components/     SubsystemSelector, FormatPanel, FileDropzone, ResultsTable, charts/
  src/lib/            apiClient (axios), downloadJob
  src/styles/         tokens.css (the three docs/DESIGN_*.md style refs) + global.css
```
