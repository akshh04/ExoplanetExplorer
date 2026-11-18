"""Atomically store the last validated NASA response for offline recovery."""

import gzip
import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile


def path() -> Path:
    configured_path = os.environ.get("EXOPLANET_ARCHIVE_CACHE")
    return Path(configured_path) if configured_path else Path(__file__).parent / ".cache" / "archive.json.gz"


def read() -> tuple[str, datetime]:
    with gzip.open(path(), "rt", encoding="utf-8") as snapshot:
        payload = json.load(snapshot)
    fetched = datetime.fromisoformat(payload["fetched_at"])
    if fetched.tzinfo is None or not isinstance(payload["csv"], str):
        raise ValueError("Invalid archive snapshot")
    return payload["csv"], fetched


def write(csv: str, fetched_at: datetime) -> None:
    destination = path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as snapshot:
            json.dump({"fetched_at": fetched_at.isoformat(), "csv": csv}, snapshot)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
