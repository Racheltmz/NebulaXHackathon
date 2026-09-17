"""Thin wrapper around the Supabase Storage client.

The client is created lazily on first use so importing/starting the backend before Supabase
credentials are set doesn't fail — only the actual upload/download/sign-url calls do, with a
clear error.
"""

from supabase import Client, create_client

import config

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not (config.SUPABASE_URL and config.SUPABASE_SERVICE_ROLE_KEY):
            raise RuntimeError(
                "Storage is not configured yet — add SUPABASE_URL and "
                "SUPABASE_SERVICE_ROLE_KEY to app/backend/.env."
            )
        _client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
    return _client


def upload_file(bucket: str, path: str, content: bytes, content_type: str = "application/octet-stream") -> None:
    get_client().storage.from_(bucket).upload(
        path, content, {"content-type": content_type, "upsert": "true"}
    )


def create_signed_url(bucket: str, path: str, expires_in_seconds: int = 3600) -> str:
    result = get_client().storage.from_(bucket).create_signed_url(path, expires_in_seconds)
    return result["signedURL"] if "signedURL" in result else result["signedUrl"]
