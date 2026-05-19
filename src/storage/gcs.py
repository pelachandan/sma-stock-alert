"""
Google Cloud Storage client wrapper.
=====================================
Provides read/write helpers for the stock-alert-prod bucket.

Auth:
  - GitHub Actions: google-github-actions/auth@v2 sets up Application Default
    Credentials automatically (Workload Identity Federation — no JSON key needed).
  - Local dev: run `gcloud auth application-default login` once.
  - If no credentials are available, all operations silently no-op so local
    development without GCP access still works.

Bucket layout:
  historical-data/   AAPL.csv, MSFT.csv …
  weekly/            AAPL_weekly.csv …
  monthly/           AAPL_monthly.csv …
  config/            open_positions.json, rs_ranker_bought.json …
"""

import os
from pathlib import Path

GCS_BUCKET = os.getenv("GCS_BUCKET", "stock-alert-prod")

_client = None   # lazy singleton
_client_init_attempted = False
_client_init_error = None


def get_client():
    """Return a GCS client, or None if credentials are not available."""
    global _client, _client_init_attempted, _client_init_error
    if _client is not None:
        return _client
    if _client_init_attempted:
        return None
    _client_init_attempted = True
    try:
        from google.cloud import storage  # noqa: PLC0415
        _client = storage.Client()
        return _client
    except Exception as exc:
        _client_init_error = exc
        print(f"⚠️  [gcs] GCS unavailable — running in local-only mode ({exc})")
        return None


def _bucket():
    client = get_client()
    if client is None:
        return None
    return client.bucket(GCS_BUCKET)


# ──────────────────────────────────────────────
# Single-file operations
# ──────────────────────────────────────────────

def download_file(gcs_path: str, local_path: str | Path) -> bool:
    """
    Download a single file from GCS to local_path.
    Returns True on success, False if not found or GCS unavailable.
    """
    bucket = _bucket()
    if bucket is None:
        return False
    try:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            return False
        blob.download_to_filename(str(local_path))
        return True
    except Exception as exc:
        print(f"⚠️  [gcs] download_file({gcs_path}) failed: {exc}")
        return False


def upload_file(local_path: str | Path, gcs_path: str) -> bool:
    """
    Upload a local file to GCS.
    Returns True on success, False if file missing or GCS unavailable.
    """
    bucket = _bucket()
    if bucket is None:
        return False
    local_path = Path(local_path)
    if not local_path.exists():
        return False
    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        return True
    except Exception as exc:
        print(f"⚠️  [gcs] upload_file({gcs_path}) failed: {exc}")
        return False


def file_exists(gcs_path: str) -> bool:
    """Return True if the given GCS object exists."""
    bucket = _bucket()
    if bucket is None:
        return False
    try:
        return bucket.blob(gcs_path).exists()
    except Exception:
        return False


# ──────────────────────────────────────────────
# Bulk folder operations
# ──────────────────────────────────────────────

def sync_from_gcs(gcs_prefix: str, local_dir: str | Path) -> int:
    """
    Download all blobs under gcs_prefix into local_dir.
    Skips files that already exist locally (incremental).
    Returns number of files downloaded.
    """
    client = get_client()
    if client is None:
        return 0

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    try:
        blobs = list(client.list_blobs(GCS_BUCKET, prefix=gcs_prefix))
        for blob in blobs:
            # Derive local filename from the blob name
            relative = blob.name[len(gcs_prefix):].lstrip("/")
            if not relative:
                continue
            local_file = local_dir / relative
            if local_file.exists():
                continue  # already cached locally
            local_file.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file))
            downloaded += 1
    except Exception as exc:
        print(f"⚠️  [gcs] sync_from_gcs({gcs_prefix}) failed: {exc}")

    return downloaded


def sync_to_gcs(local_dir: str | Path, gcs_prefix: str, pattern: str = "*.csv") -> int:
    """
    Upload all files matching pattern in local_dir to GCS under gcs_prefix.
    Returns number of files uploaded.
    """
    bucket = _bucket()
    if bucket is None:
        return 0

    local_dir = Path(local_dir)
    if not local_dir.exists():
        return 0

    uploaded = 0
    try:
        for local_file in local_dir.glob(pattern):
            gcs_path = f"{gcs_prefix}/{local_file.name}"
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_file))
            uploaded += 1
    except Exception as exc:
        print(f"⚠️  [gcs] sync_to_gcs({gcs_prefix}) failed: {exc}")

    return uploaded
