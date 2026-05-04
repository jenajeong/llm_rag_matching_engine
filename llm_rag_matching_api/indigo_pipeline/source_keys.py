import hashlib
import re
from typing import Any


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip().lower()


def stable_source_key(doc_type: str, *values: Any) -> str:
    raw = "|".join([_norm(doc_type), *(_norm(value) for value in values)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def article_source_key(record: dict) -> str:
    return stable_source_key("article", record.get("no"), record.get("title"), record.get("text"))


def patent_source_key(record: dict) -> str:
    return stable_source_key("patent", record.get("no"), record.get("title"), record.get("text"))


def project_source_key(record: dict) -> str:
    return stable_source_key("project", record.get("no"), record.get("title"), record.get("text"))


def source_key_for_doc_type(doc_type: str, record: dict) -> str:
    if doc_type == "article":
        return article_source_key(record)
    if doc_type == "patent":
        return patent_source_key(record)
    if doc_type == "project":
        return project_source_key(record)
    return stable_source_key(doc_type, record.get("no"), record.get("title"), record.get("text"))
