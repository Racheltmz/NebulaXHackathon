# App Design — PS3 Train Condition Monitoring App

> Status: **in progress.** `app/` is being built against this doc; it's updated as decisions
> change rather than being a frozen pre-build spec.
>
> **Auth is currently deferred** (see Section 7.0) — there is no Login/Register/Profile page and
> no per-user identity anywhere in the app right now. The home page (`/`) goes straight to the
> Predict page, and Predict/History/Dashboard are open to anyone who can reach the app. Supabase
> is still used for its Postgres database and Storage buckets (history + file persistence); only
> its Auth product is unused for now. Re-introducing login is a scoped addition later, not a
> rewrite — see Section 7.0 for what comes back and where.
>
> Visual style references (already chosen), three files each covering a different slice of the
> UI: [`DESIGN_FORM_MAIN.md`](DESIGN_FORM_MAIN.md) (Typeform) for page headers, the "how to use
> this app" explanation, and the upload/download controls; [`DESIGN_FORM_ELEMENTS.md`](DESIGN_FORM_ELEMENTS.md)
> (Buddy) for the subsystem-selection control, whose component vocabulary can also be reused on
> the dashboard page where it fits better than the alternative; and
> [`DESIGN_DASHBOARD.md`](DESIGN_DASHBOARD.md) (Dub) for the history/table page. This document
> covers the functional design: pages, data model, storage, and API — not visual styling, which
> the three style docs already own.

## 1. Purpose & Scope

A single web app, submitted once at the team-root level (per spec Section 4.1 item 3 /
"Submission folder structure"), that lets a **non-technical user**:

1. Land on the Predict page immediately (it's the home page — no login screen in front of it).
2. Pick one of the 4 subsystems (Door, ACV, Rail Corrugation, SHM).
3. Upload (or drag-and-drop) the held-out test file(s) for that subsystem.
4. Get predictions back on screen, download them as CSV, and see a dashboard visualising the
   result.
5. Look back at past runs in a history table.

This app is also literally how the team produces `predictions.zip` (item 2) — running the
organiser-distributed held-out test files through it — so its output CSV schema must exactly
match Section 4.1's table, per subsystem.

### Non-goals for v1
- No admin/moderation tooling.
- No retraining or model-management UI — models are fixed artifacts the backend loads.
- No real-time streaming ingestion — uploads are finite files, processed once per submission.
- No login/accounts for now — deferred (Section 7.0); everyone using the app shares one history.

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | React (Vite) + Axios | Requested. Vite for fast dev server / simple build. |
| Backend | FastAPI (Python) | Requested. Also the natural home for the model inference code (same language as the notebooks in `Optional_Items/*/code`). |
| Database + File storage | **Supabase** (free tier) | One free service covers both needs below (Section 3) instead of stitching together separate providers — least moving parts for a hackathon judge to reproduce/run. Supabase Auth is also part of this project but is unused while auth is deferred (Section 7.0). |
| ML inference | plain Python functions per subsystem, loaded once at backend startup | See Section 8. |

## 3. Storage & Infrastructure Decision: Supabase

The app needs two things that persist beyond a single request: **structured history/prediction
data** (for the history table and dashboards) and **uploaded/output files** (the raw input file +
the generated `*_predictions.csv`, downloadable later). Supabase's free tier provides both under
one project:

- **Postgres database** — relational tables for `prediction_jobs` and `prediction_rows`
  (Section 4).
- **Storage (object buckets)** — holds the original uploaded file(s) and the generated
  `*_predictions.csv`, so "download predictions" in the history page is a signed-URL fetch, not
  re-generating the file.

Free-tier ballpark (verify current limits at project creation, these change over time): ~500MB
database, ~1GB file storage, project auto-pauses after a week of inactivity (fine for a hackathon
demo; a request wakes it back up with a short delay). Dataset sizes here are small (largest raw
file is `Door/Train.csv` at ~1.1MB; individual test files are similarly small), so storage
headroom is not a concern.

Supabase also offers Auth, which is why it was picked over a plain Postgres+storage host even
though Auth is unused for now — adding login later (Section 7.0) is turning a feature back on in
the same project, not introducing a new provider.

**Alternative considered:** Firebase (Auth + Firestore + Storage) is an equally valid free
one-stop option. Supabase is preferred here because its relational Postgres model maps directly
onto the history table (one row per job, filterable/sortable by subsystem) without denormalizing
into a NoSQL shape.

## 4. Data Model (Postgres)

No `profiles` table and no per-user column anywhere — see Section 7.0 for what's deferred and
what comes back together when login returns.

### `prediction_jobs`
One row per "run the app once for a subsystem" — i.e. one row per history-table entry.

| Column | Type | Notes |
|---|---|---|
| `id` | uuid, PK | job id |
| `subsystem` | text | `door` \| `acv` \| `rail_corrugation` \| `shm` |
| `status` | text | `processing` \| `done` \| `failed` |
| `input_files` | jsonb | list of `{filename, storage_path, size_bytes}` for each uploaded file |
| `output_storage_path` | text | path to the generated `*_predictions.csv` in the `predictions` bucket |
| `summary` | jsonb | small precomputed rollup for the history-row / dashboard header (e.g. door: `{segments: 42, abnormal: 5}`; rail: `{Normal: 60, "Side I": 3, "Side II": 5}`; shm: `{mean: 0.41, max: 0.91}`; acv: `{top_car: "03", car_models: {"acv_case_01.xlsx": "A"}, train_numbers: {...}, telemetry: {"acv_case_01.xlsx": {...}}}` — `telemetry` is a ~90 KB downsampled per-car copy of the file's temperatures and modes for the run dashboard (ml/acv_telemetry.py); the history list endpoint strips it) |
| `error_message` | text, nullable | populated if `status = failed` |
| `created_at` | timestamptz | |

### `prediction_rows`
The parsed, queryable form of the output CSV — one row per **prediction record** (not per raw
CSV line), so the dashboard can query/aggregate without re-parsing a CSV file out of storage on
every view. This is the "store predictions efficiently to load" piece.

| Column | Type | Notes |
|---|---|---|
| `id` | bigserial, PK | |
| `job_id` | uuid, FK → `prediction_jobs.id` | |
| `file_id` | text, nullable | source filename (null for Door — see Section 6) |
| `start_time` / `end_time` | text, nullable | Door only |
| `label` | text, nullable | predicted class (Door/Rail) |
| `ranked_cars` | text, nullable | ACV only, `\|`-joined |
| `value` | double precision, nullable | SHM's numeric prediction |

Rationale for one wide-ish table over 4 subsystem-specific tables: the history/dashboard code
only ever needs "give me the rows for job X," and one table keeps that query and the ORM model
uncomplicated; the schema stays close to the union of what Section 4.1's four output formats
actually contain. Unused columns per subsystem are simply null.

The raw uploaded file(s) and the final `*_predictions.csv` are **not** duplicated into Postgres —
they live in Storage (Section 3) and `prediction_jobs` just points to them, keeping the database
small and downloads a direct file fetch.

## 5. Storage Bucket Layout

Two buckets:
- `uploads/{job_id}/<original_filename>` — the file(s) as submitted.
- `predictions/{job_id}/<subsystem>_predictions.csv` — the exact-schema output file, same one
  the "download" button serves and the one that goes into `predictions.zip` at submission time.

## 6. Subsystem Upload Contract

This drives the format hint shown on the Predict page once a subsystem is selected, upload
validation, and how "one upload" maps to "one or more predicted rows."

| Subsystem | Accepted file(s) | Cardinality | Required columns / shape (validated on upload) | Output rows produced |
|---|---|---|---|---|
| Door | `.csv` | 1+ files, each an independent continuous stream | 17 columns per **Door_Subsystem_Info_Kit.md** §2.2 (time + motor current/voltage/back-EMF + position/status flags), header row required | Many rows per file — one per detected open/close segment (`start_time`, `end_time`, `prediction`) |
| ACV | `.xlsx` | 1+ files, each one case | timestamp + car-model/train-number columns, plus per-car `Car <NN> - <parameter>` columns (exact parameter set may vary by file — validated by pattern, not fixed column list) | 1 row per file (`file_id`, `ranked_cars`) |
| Rail Corrugation | `.csv` | 1+ files, each one 1-second recording | 129 columns (speed + 64 axle-box vibration/shock channel pairs), header row present | 1 row per file (`file_id`, `prediction`) |
| SHM | `.csv` | 1+ files, each one stress time segment | single column of raw stress readings, no header | 1 row per file (`file_id`, `prediction`) |

**Assumption (flagging per spec Section 3.2's "state your assumption" guidance):** the official
held-out test set for Door and ACV is a single file each (`Test.csv`, `acv_test_case.xlsx`), but
the app accepts multiple files per subsystem uniformly, treating every uploaded file
independently. This keeps the upload UX identical across all 4 subsystems and lets a user
re-check several candidate files in one session; it doesn't change what a single official-test
run produces.

The final downloadable `*_predictions.csv` for a job concatenates all rows produced by all files
in that job, in the exact column schema from Section 4.1's table — so running the actual
organiser-distributed test file(s) through the app and downloading the result is submission-ready
as-is.

## 7. Pages

### 7.0 Auth pages (Login / Register / Profile) — deferred

**Removed for now, not just unstyled.** There's no Login/Register/Profile page, no
`AuthContext`/`ProtectedRoute`, and no Supabase JS client in the frontend; the backend has no
auth dependency at all — every request runs unauthenticated. The home page (`/`) is the Predict
page directly. This was a deliberate scope cut to focus on Predict/Dashboard/History first,
rather than a "not implemented yet" gap in those pages.

When login comes back, it's additive rather than a rewrite:
- Frontend: re-add `LoginPage`/`RegisterPage`/`ProfilePage`, an `AuthContext` wrapping the
  Supabase JS client, and a `ProtectedRoute` guard — the previous implementation (email/password
  via Supabase Auth, JWT held client-side) is the reference to rebuild from.
  Minimal, per the user's "just for user functionality" scope — login/register are thin wrappers
  around the Supabase Auth JS client (email + password); profile shows/edits a display name and
  logs out. Not styled in depth — reuse the same base component library as the rest of the app,
  no dedicated design pass needed.
- Backend: re-add a `profiles` table (`id` = `auth.users.id`, `display_name`, `created_at`), a
  `user_id` FK back on `prediction_jobs`, an auth dependency that verifies the Supabase JWT
  (`Authorization: Bearer <token>`) and resolves it to a user, and apply that dependency to
  `predict`/`jobs`/`history`.
- History (Section 7.3) gets its "Run by" column back once `user_id` exists to join against.

### 7.1 Main Prediction Page (`/`, the home page)
Styled per a split of the two form-oriented references: [`DESIGN_FORM_MAIN.md`](DESIGN_FORM_MAIN.md)
(Typeform) for the page header, the usage blurb, and the upload/download controls;
[`DESIGN_FORM_ELEMENTS.md`](DESIGN_FORM_ELEMENTS.md) (Buddy) specifically for the subsystem
selector control.

1. Short usage blurb at the top ("select a subsystem → upload your file(s) in the shown format →
   get predictions") — `DESIGN_FORM_MAIN.md` styling.
2. Subsystem selector (4 options: Door / ACV / Rail Corrugation / SHM) — `DESIGN_FORM_ELEMENTS.md`
   styling.
3. Once a subsystem is picked, a format panel appears showing that subsystem's required format
   (from the table in Section 6 — file type, one-line schema description, and a link/expand to
   see the full column list), so the user knows what to upload before they try.
4. Drag-and-drop + click-to-browse upload zone, accepting the file type(s) from Section 6,
   multiple files allowed.
5. Submit → calls the backend (Section 9), shows a loading state, then on success:
   - a "Download predictions.csv" button,
   - a "View dashboard" button/link to that job's dashboard (Section 7.2),
   - a compact inline preview of the result (first N rows or the `summary` rollup).
6. Client-side validation before submit: right extension(s) for the selected subsystem, and a
   friendly error if the backend rejects a file's shape (e.g. wrong column count) rather than a
   raw stack trace.

### 7.2 Dashboard / Visualisation Page (`/jobs/:jobId`)
Styled per [`DESIGN_FORM_ELEMENTS.md`](DESIGN_FORM_ELEMENTS.md) (Buddy) — its component
vocabulary (cards, tags/pills, accent-colour usage) is reused here rather than the Dub reference,
which is now reserved for the History page (Section 7.3). The page header (subsystem, run date,
uploader, download button) reuses `DESIGN_FORM_MAIN.md`'s header/button treatment for consistency
with the Predict page. One dashboard per prediction job, reached either straight after a
prediction run or via the History page. Layout differs by subsystem (kept intentionally simple
for v1, per the user's own note — richer visuals are a later iteration):

| Subsystem | v1 dashboard content |
|---|---|
| Door | Horizontal timeline of detected segments coloured by label (Normal vs Abnormal resistance) across the stream; a normal-vs-abnormal count tile; a table of segments. |
| ACV | Bar chart of cars ranked by fault likelihood (most-likely first), top car called out; one bar chart per uploaded case file. |
| Rail Corrugation | Class-distribution bar chart (Normal / Side I / Side II counts across uploaded files); a table of per-file predictions. |
| SHM | Bar chart of predicted cumulative damage per file, sorted descending, with a reference line at damage = 1.0 (Miner's-rule failure threshold, see Info Kit §1.3.1); a table of per-file values. |

Every dashboard shares a header (subsystem, run date, uploader, download button) and a table of
the raw prediction rows underneath the chart — the chart is the "at a glance" layer, the table is
the "check the actual numbers" layer.

### 7.3 History Page (`/history`)
Styled per [`DESIGN_DASHBOARD.md`](DESIGN_DASHBOARD.md) (Dub, table-heavy, matches that
reference's "Dashboard Table Row" component). One row per `prediction_jobs` entry, columns:

`Subsystem | Date | Status | Download | Dashboard →`

Every run is visible to everyone using the app — there's no per-user scoping since there's no
login (Section 7.0). A "Run by" column returns once `user_id` exists to populate it. Sortable/
filterable by subsystem at minimum for v1.

## 8. Model Integration

Backend exposes one Python function per subsystem behind a common interface, e.g.
`predict(files: list[UploadedFile]) -> PredictionResult` (a small dataclass: list of row dicts
matching Section 4's `prediction_rows` shape + the `summary` rollup for `prediction_jobs`). This
lets the API/DB/frontend layers be built and tested against all 4 subsystems today, independent
of how far along each subsystem's model is:

| Subsystem | Model status | Plan |
|---|---|---|
| SHM | **Done** — `Optional_Items/SHM/model/shm_model.joblib` (gradient-boosted regressor + rainflow-count features + fixed `feature_cols` list, per `Optional_Items/SHM/code/shm.ipynb`) | Load the joblib at backend startup; wrap its existing feature-extraction function to run on uploaded files. |
| Door | Not yet built | Stub returns a placeholder single "unclassified" segment per file until the real segmentation+classification model lands, so the app pipeline is testable end-to-end now. |
| ACV | Not yet built | Stub returns cars in file-column order (unranked) as a placeholder. |
| Rail Corrugation | Not yet built | Stub returns `"Normal"` for every file as a placeholder. |

This keeps "build the app" and "finish the other 3 models" as independent workstreams — swapping
a stub for a real model later is a one-file change in `backend/models/<subsystem>.py` with no
API/DB/frontend impact.

## 9. Backend API

No auth on any route right now (Section 7.0) — all of these are open.

| Method & path | Purpose |
|---|---|
| `GET /api/subsystems` | Returns the Section 6 contract (accepted extension(s), schema summary) for all 4 subsystems, so the frontend's format panel has a single source of truth instead of hardcoding it twice. |
| `POST /api/predict/{subsystem}` | Multipart upload of 1+ files. Runs that subsystem's `predict()` (Section 8) synchronously (files are small enough that this doesn't need a background job queue for v1), persists the job + rows + files, returns the created `prediction_jobs` row plus its rows/summary. |
| `GET /api/jobs/{job_id}` | Full job detail (summary + all prediction rows) — powers the dashboard page. |
| `GET /api/jobs/{job_id}/download` | Streams/redirects to the stored `*_predictions.csv` for that job. |
| `GET /api/history` | List of all `prediction_jobs`, newest first — powers the History page. |

## 10. `app/` Folder Structure

```
app/
├── frontend/                      # React + Vite
│   └── src/
│       ├── pages/                 # Predict (home page), Dashboard, History
│       ├── components/            # SubsystemSelector, FileDropzone, FormatPanel, ResultsTable, charts/
│       ├── lib/                   # apiClient (axios), downloadJob
│       └── ...
└── backend/                       # FastAPI
    ├── main.py
    ├── routers/                   # subsystems.py, predict.py, jobs.py, history.py
    ├── ml/                        # door.py, acv.py, rail_corrugation.py, shm.py — predict() per Section 8
    ├── ml_artifacts/              # shm_model.joblib (copied from Optional_Items/SHM/model)
    ├── storage.py                 # Supabase Storage client wrapper
    ├── db.py                      # Supabase Postgres connection / ORM models (Section 4)
    └── requirements.txt
```

No `auth.py`, no `profile.py` router, no Login/Register/Profile pages, no `AuthContext`/
`ProtectedRoute`/`supabaseClient` — see Section 7.0 for what comes back and where.

## 11. Open Points

1. **Supabase project**: needs a Supabase account/project created (free tier) with its URL +
   service-role key + database connection string supplied to the backend via `.env`
   (`app/backend/schema.sql` creates the tables; the Storage tab needs `uploads` and
   `predictions` buckets created manually).
2. **Auth timing** (Section 7.0): deferred by explicit request to focus on Predict/Dashboard/
   History first — revisit once those are solid.
3. **Multi-file upload per subsystem** (Section 6's assumption) — confirm this is desired UX
   versus restricting to exactly the official cardinality per subsystem.
4. **Synchronous inference** (Section 9) — fine while files stay small; would need a job
   queue/polling if that changes.
5. Door/ACV/Rail Corrugation model stubs (Section 8) — confirm it's fine for the app to ship
   against placeholder logic for those three until their models are ready, versus waiting.
