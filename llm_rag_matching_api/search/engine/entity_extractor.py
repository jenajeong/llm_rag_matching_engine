import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from openai import OpenAI

from .cost_tracker import log_chat_usage
from .prompts import (
    COMPLETION_DELIMITER,
    RECORD_DELIMITER,
    TUPLE_DELIMITER,
    format_entity_extraction_prompt,
)
from .settings import LLM_MODEL, OPENAI_API_KEY


logger = logging.getLogger(__name__)


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


class EntityRelationExtractor:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL

    def extract_batch(
        self,
        documents: List[Dict],
        doc_type: str = "patent",
        batch_size: int = 10,
    ) -> Tuple[List[Entity], List[Relation], List[str]]:
        all_entities = []
        all_relations = []
        failed_doc_ids = []

        for i in range(0, len(documents), batch_size):
            for document in documents[i:i + batch_size]:
                doc_id = str(document.get("doc_id", "")).strip()
                text = document.get("text", "")
                if not doc_id or not text:
                    failed_doc_ids.append(doc_id)
                    continue

                try:
                    entities, relations = self.extract_from_document(doc_id, text, doc_type)
                except Exception as exc:
                    logger.exception("Entity extraction failed for %s: %s", doc_id, exc)
                    failed_doc_ids.append(doc_id)
                    continue

                if not entities:
                    failed_doc_ids.append(doc_id)
                    continue

                all_entities.extend(entities)
                all_relations.extend(relations)

        return all_entities, all_relations, failed_doc_ids

    def extract_from_document(
        self,
        doc_id: str,
        text: str,
        doc_type: str = "patent",
    ) -> Tuple[List[Entity], List[Relation]]:
        del doc_type
        prompt = format_entity_extraction_prompt(text[:8000])
        response = self._call_llm(prompt, doc_id=doc_id)
        if not response:
            return [], []
        return self._parse_response(response, doc_id)

    def _call_llm(self, prompt: str, doc_id: str = "", max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4096,
                )
                log_chat_usage(
                    component="entity_extraction",
                    model=self.model,
                    response=response,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                logger.warning(
                    "LLM extraction error for %s (%s/%s): %s",
                    doc_id,
                    attempt + 1,
                    max_retries,
                    exc,
                )
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 5)

        return ""

    def _parse_response(self, response: str, doc_id: str) -> Tuple[List[Entity], List[Relation]]:
        entities = []
        relations = []

        for record in response.split(RECORD_DELIMITER):
            record = record.strip()
            if not record or COMPLETION_DELIMITER in record:
                continue

            match = re.search(r'\("([^"]+)"' + re.escape(TUPLE_DELIMITER) + r"(.+)\)", record)
            if not match:
                continue

            record_type = match.group(1).lower()
            fields = [field.strip().strip('"') for field in match.group(2).split(TUPLE_DELIMITER)]

            if record_type == "entity" and len(fields) >= 3:
                entities.append(Entity(
                    name=fields[0].strip().upper(),
                    entity_type=fields[1].strip().upper(),
                    description=fields[2].strip(),
                    source_doc_id=doc_id,
                ))
            elif record_type == "relationship" and len(fields) >= 4:
                relations.append(Relation(
                    source_entity=fields[0].strip().upper(),
                    target_entity=fields[1].strip().upper(),
                    description=fields[2].strip(),
                    keywords=fields[3].strip(),
                    source_doc_id=doc_id,
                ))

        return entities, relations


def to_dicts(items: List[Entity | Relation]) -> List[Dict]:
    return [json.loads(json.dumps(item.__dict__, ensure_ascii=False)) for item in items]
