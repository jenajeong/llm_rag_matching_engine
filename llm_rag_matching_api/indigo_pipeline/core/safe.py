import math
import re
from pathlib import Path
from typing import Any, Mapping


NULL_STRINGS = {"", "none", "null", "nan", "nat", "undefined"}


def is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in NULL_STRINGS:
        return True
    return False


def as_text(value: Any, default: str = "") -> str:
    if is_nullish(value):
        return default
    try:
        return str(value).strip()
    except Exception:
        return default


def clean_ws(value: Any, default: str = "") -> str:
    return re.sub(r"\s+", " ", as_text(value, default)).strip()


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def get_nested(mapping: Mapping[str, Any] | None, key: str, default: Any = "") -> Any:
    if not isinstance(mapping, Mapping):
        return default
    value = mapping.get(key, default)
    return default if is_nullish(value) else value


def split_csv(value: Any) -> set[str]:
    return {part.strip() for part in as_text(value).split(",") if part.strip()}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
