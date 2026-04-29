import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

from .settings import SEARCH_RESULT_CACHE_DIR, SEARCH_RESULT_CACHE_TTL_HOURS


SEARCH_ID_PATTERN = re.compile(r"^search_\d{8}_\d{6}_[0-9a-f]{8}$")


class SearchResultCacheError(ValueError):
    pass


def _validate_search_id(search_id: str) -> str:
    search_id = str(search_id or "").strip()
    if not SEARCH_ID_PATTERN.match(search_id):
        raise SearchResultCacheError("Invalid search_id.")
    return search_id


def _cache_path(search_id: str) -> Path:
    return Path(SEARCH_RESULT_CACHE_DIR) / f"{_validate_search_id(search_id)}.json"


def save_search_result(result: Dict[str, Any]) -> Path:
    search_id = _validate_search_id(result.get("search_id", ""))
    cache_dir = Path(SEARCH_RESULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    final_path = _cache_path(search_id)
    tmp_path = final_path.with_suffix(".tmp")

    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
    tmp_path.replace(final_path)
    return final_path


def load_search_result(search_id: str) -> Dict[str, Any]:
    path = _cache_path(search_id)
    if not path.exists():
        raise SearchResultCacheError("Search result has expired or does not exist.")

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    result = payload.get("result")
    if not isinstance(result, dict):
        raise SearchResultCacheError("Cached search result is invalid.")
    return result


def clear_search_results(older_than_hours: int | None = None, clear_all: bool = False) -> int:
    cache_dir = Path(SEARCH_RESULT_CACHE_DIR)
    if not cache_dir.exists():
        return 0

    cutoff = None
    if not clear_all:
        hours = older_than_hours if older_than_hours is not None else SEARCH_RESULT_CACHE_TTL_HOURS
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    deleted = 0
    for path in cache_dir.glob("search_*.json"):
        should_delete = clear_all
        if not should_delete and cutoff is not None:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            should_delete = modified_at < cutoff

        if should_delete:
            path.unlink(missing_ok=True)
            deleted += 1

    return deleted
