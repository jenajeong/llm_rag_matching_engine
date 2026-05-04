from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcessedDocument:
    doc_id: str
    doc_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    name: str
    entity_type: str
    description: str
    source_doc_id: str


@dataclass
class Relation:
    source_entity: str
    target_entity: str
    keywords: str
    description: str
    source_doc_id: str
