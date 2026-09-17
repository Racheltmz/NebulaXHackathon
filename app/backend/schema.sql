-- Run once in the Supabase project's SQL editor (Database > SQL Editor) after the project is
-- created. Matches app/backend/db.py's ORM models and docs/DESIGN.md Section 4.
--
-- Auth/user tracking is deferred for now (see docs/DESIGN.md) — there is no `profiles` table and
-- no `user_id` on prediction_jobs. Everyone using the app shares one history. Re-add a `profiles`
-- table + `user_id` FK + RLS together when login comes back.
--
-- Also create two private Storage buckets named `uploads` and `predictions` in the Storage tab
-- (or set SUPABASE_UPLOADS_BUCKET / SUPABASE_PREDICTIONS_BUCKET to whatever names you used).

create table if not exists public.prediction_jobs (
  id uuid primary key default gen_random_uuid(),
  subsystem text not null check (subsystem in ('door', 'acv', 'rail_corrugation', 'shm')),
  status text not null default 'processing' check (status in ('processing', 'done', 'failed')),
  input_files jsonb not null default '[]'::jsonb,
  output_storage_path text,
  summary jsonb not null default '{}'::jsonb,
  error_message text,
  created_at timestamptz not null default now()
);

create index if not exists prediction_jobs_created_at_idx on public.prediction_jobs (created_at desc);
create index if not exists prediction_jobs_subsystem_idx on public.prediction_jobs (subsystem);

create table if not exists public.prediction_rows (
  id uuid primary key default gen_random_uuid(),
  job_id uuid not null references public.prediction_jobs (id) on delete cascade,
  file_id text,
  start_time text,
  end_time text,
  label text,
  ranked_cars text,
  value double precision
);

create index if not exists prediction_rows_job_id_idx on public.prediction_rows (job_id);

-- The backend connects via DATABASE_URL as the `postgres` role, a superuser that always bypasses
-- RLS — so this has no effect on the app. It closes off the *other* way into these tables:
-- Supabase exposes every public-schema table over its REST API using the anon/publishable key,
-- which is discoverable by anyone even though this app doesn't use it. RLS with no policies
-- denies that path entirely by default, without needing any auth to define policies against.
alter table public.prediction_jobs enable row level security;
alter table public.prediction_rows enable row level security;

-- Migrating an existing install that already ran the old (auth-enabled) version of this file:
--   drop table if exists public.prediction_rows cascade;
--   drop table if exists public.prediction_jobs cascade;
--   drop table if exists public.profiles cascade;
--   drop trigger if exists on_auth_user_created on auth.users;
--   drop function if exists public.handle_new_user;
-- then re-run everything above.
