from typing import Any

from ..core.safe import as_text, split_csv


def merge_duplicate_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for entity in entities:
        name = as_text(entity.get("name")).upper()
        if not name:
            continue
        entity = dict(entity)
        entity["name"] = name
        entity["entity_type"] = as_text(entity.get("entity_type"), "UNKNOWN").upper()
        entity["description"] = as_text(entity.get("description"))
        entity["source_doc_id"] = as_text(entity.get("source_doc_id"))
        if name not in merged:
            merged[name] = entity
            continue
        existing = merged[name]
        _append_unique_text(existing, "description", entity.get("description"))
        _merge_doc_ids(existing, entity)
    return list(merged.values())


def merge_duplicate_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for relation in relations:
        source = as_text(relation.get("source_entity")).upper()
        target = as_text(relation.get("target_entity")).upper()
        if not source or not target:
            continue
        relation = dict(relation)
        relation["source_entity"] = source
        relation["target_entity"] = target
        relation["keywords"] = as_text(relation.get("keywords"))
        relation["description"] = as_text(relation.get("description"))
        relation["source_doc_id"] = as_text(relation.get("source_doc_id"))
        key = "|".join(sorted([source, target]))
        if key not in merged:
            merged[key] = relation
            continue
        existing = merged[key]
        old_keywords = split_csv(existing.get("keywords"))
        new_keywords = split_csv(relation.get("keywords"))
        existing["keywords"] = ",".join(sorted(old_keywords | new_keywords))
        _append_unique_text(existing, "description", relation.get("description"))
        _merge_doc_ids(existing, relation)
        existing["weight"] = max(int(existing.get("weight", 1) or 1), int(relation.get("weight", 1) or 1))
    return list(merged.values())


def _append_unique_text(target: dict[str, Any], key: str, value: Any) -> None:
    text = as_text(value)
    if not text:
        return
    current = as_text(target.get(key))
    if text not in current:
        target[key] = f"{current}\n{text}".strip() if current else text


def _merge_doc_ids(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    doc_ids = split_csv(target.get("source_doc_id")) | split_csv(incoming.get("source_doc_id"))
    target["source_doc_id"] = ",".join(sorted(doc_ids))
