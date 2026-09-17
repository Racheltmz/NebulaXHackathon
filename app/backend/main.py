from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
from routers import history, jobs, predict, subsystems

app = FastAPI(title="PS3 Train Condition Monitoring API")


@app.middleware("http")
async def catch_unhandled_exceptions(request: Request, call_next):
    """Registered before CORSMiddleware below, which puts it *inside* CORSMiddleware in the
    actual middleware stack (Starlette wraps middleware in reverse registration order). That
    matters because `@app.exception_handler(Exception)` doesn't work for this — Starlette routes
    that specific handler through ServerErrorMiddleware, which sits outside CORSMiddleware no
    matter what, so its responses never carry CORS headers. Catching the exception here instead
    means the response flows back out through CORSMiddleware normally, so a backend crash (e.g. a
    bad DATABASE_URL) shows up as an actual error message on the frontend instead of a misleading
    "CORS policy" error in the browser console.
    """
    try:
        return await call_next(request)
    except Exception as exc:  # noqa: BLE001 — deliberate last-resort catch-all
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(subsystems.router)
app.include_router(predict.router)
app.include_router(jobs.router)
app.include_router(history.router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "database_configured": bool(config.DATABASE_URL),
        "storage_configured": bool(config.SUPABASE_URL and config.SUPABASE_SERVICE_ROLE_KEY),
    }
