import json
from pathlib import Path
from typing import Dict, Iterable, List

from .source_keys import source_key_for_doc_type


def load_json_list(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def save_json_list(path: Path, data: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def existing_source_keys(path: Path, doc_type: str) -> set[str]:
    return {
        source_key_for_doc_type(doc_type, record)
        for record in load_json_list(path)
    }


def append_new_records(path: Path, doc_type: str, records: Iterable[Dict]) -> tuple[int, int]:
    existing = load_json_list(path)
    seen_keys = {
        source_key_for_doc_type(doc_type, record)
        for record in existing
    }

    appended = 0
    skipped = 0
    for record in records:
        source_key = source_key_for_doc_type(doc_type, record)
        if source_key in seen_keys:
            skipped += 1
            continue
        seen_keys.add(source_key)
        existing.append(record)
        appended += 1

    if appended:
        save_json_list(path, existing)

    return appended, skipped
