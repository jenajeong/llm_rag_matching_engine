import asyncio
import json
import logging
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI, OpenAI

from .. import config
from ..core.safe import as_text
from ..core.types import Entity, Relation
from ..cost_tracker import log_chat_usage
from .prompts import COMPLETION_DELIMITER, RECORD_DELIMITER, TUPLE_DELIMITER, format_entity_extraction_prompt

logger = logging.getLogger(__name__)


class EntityRelationExtractor:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or config.OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is required for entity extraction.")
        self.client = OpenAI(api_key=key, timeout=config.LLM_REQUEST_TIMEOUT)
        self.model = model or config.LLM_MODEL

    def extract_batch(self, documents: list[dict], doc_type: str = "patent") -> tuple[list[Entity], list[Relation]]:
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []
        for idx, doc in enumerate(documents):
            doc_id = as_text(doc.get("doc_id"), f"doc_{idx}")
            text = as_text(doc.get("text"))
            if not text:
                logger.warning("[%s] Empty text skipped", doc_id)
                continue
            try:
                entities, relations = self.extract_from_document(doc_id, text, doc_type)
                all_entities.extend(entities)
                all_relations.extend(relations)
                logger.info("[%s] Extracted %s entities, %s relations", doc_id, len(entities), len(relations))
            except Exception:
                logger.exception("[%s] Extraction failed", doc_id)
        return all_entities, all_relations

    def extract_from_document(self, doc_id: str, text: str, doc_type: str = "patent") -> tuple[list[Entity], list[Relation]]:
        text = as_text(text)[: config.ENTITY_EXTRACTION_MAX_CHARS]
        if not text:
            return [], []
        response = self._call_llm(format_entity_extraction_prompt(text), doc_id)
        return self._parse_response(response, doc_id)

    def _call_llm(self, prompt: str, doc_id: str, max_retries: int | None = None) -> str:
        import time

        retries = max_retries or config.LLM_MAX_RETRIES
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4096,
                )
                log_chat_usage("entity_extraction", self.model, response)
                return response.choices[0].message.content or ""
            except Exception as exc:
                logger.warning("[%s] LLM error attempt %s/%s: %s", doc_id, attempt + 1, retries, exc)
                if attempt < retries - 1:
                    time.sleep((attempt + 1) * 5)
        return ""

    def _parse_response(self, response: str, doc_id: str) -> tuple[list[Entity], list[Relation]]:
        entities: list[Entity] = []
        relations: list[Relation] = []
        if not response:
            return entities, relations
        for record in response.split(RECORD_DELIMITER):
            record = record.strip()
            if not record or COMPLETION_DELIMITER in record:
                continue
            match = re.search(r'\("([^"]+)"' + re.escape(TUPLE_DELIMITER) + r"(.+)\)", record, flags=re.DOTALL)
            if not match:
                continue
            record_type = match.group(1).strip().lower()
            fields = [field.strip().strip('"') for field in match.group(2).split(TUPLE_DELIMITER)]
            try:
                if record_type == "entity" and len(fields) >= 3:
                    name = as_text(fields[0]).upper()
                    if name:
                        entities.append(Entity(name, as_text(fields[1], "UNKNOWN").upper(), as_text(fields[2]), doc_id))
                elif record_type == "relationship" and len(fields) >= 4:
                    source = as_text(fields[0]).upper()
                    target = as_text(fields[1]).upper()
                    if source and target:
                        relations.append(Relation(source, target, as_text(fields[3]), as_text(fields[2]), doc_id))
            except Exception:
                logger.debug("[%s] Failed to parse record: %s", doc_id, record, exc_info=True)
        return entities, relations


class AsyncEntityRelationExtractor(EntityRelationExtractor):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        concurrency: int = 5,
        max_retries: int | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 20,
    ):
        key = api_key or config.OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is required for entity extraction.")
        self.client = AsyncOpenAI(api_key=key, timeout=config.LLM_REQUEST_TIMEOUT)
        self.model = model or config.LLM_MODEL
        self.concurrency = max(1, concurrency)
        self.max_retries = max_retries or config.LLM_MAX_RETRIES
        self.checkpoint_dir = Path(checkpoint_dir or config.CHECKPOINT_DIR)
        self.checkpoint_interval = checkpoint_interval
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.stats = {"success": 0, "failed": 0, "retries": 0}

    def _checkpoint_path(self, doc_type: str) -> Path:
        return self.checkpoint_dir / f"extraction_{doc_type}_checkpoint.json"

    def load_checkpoint(self, doc_type: str) -> Optional[dict]:
        path = self._checkpoint_path(doc_type)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_checkpoint(
        self,
        doc_type: str,
        processed_doc_ids: list[str],
        failed_doc_ids: list[str],
        entities: list[Entity],
        relations: list[Relation],
        total_docs: int,
    ) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_type": doc_type,
            "total_docs": total_docs,
            "processed_count": len(processed_doc_ids),
            "failed_count": len(failed_doc_ids),
            "processed_doc_ids": processed_doc_ids,
            "failed_doc_ids": failed_doc_ids,
            "entities": [asdict(e) for e in entities],
            "relations": [asdict(r) for r in relations],
            "last_saved_at": datetime.now().isoformat(),
            "stats": self.stats,
        }
        with self._checkpoint_path(doc_type).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    async def extract_batch_async(
        self,
        documents: list[dict],
        doc_type: str = "patent",
        progress_callback=None,
        resume: bool = True,
    ) -> tuple[list[Entity], list[Relation], list[str]]:
        self.semaphore = asyncio.Semaphore(self.concurrency)
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []
        processed_doc_ids: list[str] = []
        failed_doc_ids: list[str] = []
        total_docs = len(documents)

        if resume:
            checkpoint = self.load_checkpoint(doc_type)
            if checkpoint:
                processed_doc_ids = list(checkpoint.get("processed_doc_ids", []))
                failed_doc_ids = list(checkpoint.get("failed_doc_ids", []))
                all_entities = [Entity(**item) for item in checkpoint.get("entities", []) if isinstance(item, dict)]
                all_relations = [Relation(**item) for item in checkpoint.get("relations", []) if isinstance(item, dict)]
                self.stats = checkpoint.get("stats", self.stats)

        processed = set(processed_doc_ids)
        remaining = [doc for doc in documents if as_text(doc.get("doc_id")) not in processed]

        async def process_one(doc: dict):
            doc_id = as_text(doc.get("doc_id"), "unknown")
            text = as_text(doc.get("text"))
            if not text:
                return doc_id, [], [], False
            try:
                entities, relations = await self.extract_from_document_async(doc_id, text, doc_type)
                if progress_callback:
                    progress_callback(doc_id, len(entities), len(relations))
                return doc_id, entities, relations, bool(entities)
            except Exception:
                logger.exception("[%s] Async extraction failed", doc_id)
                return doc_id, [], [], False

        count = 0
        tasks = [asyncio.create_task(process_one(doc)) for doc in remaining]
        for completed in asyncio.as_completed(tasks):
            result = await completed
            doc_id, entities, relations, ok = result
            if ok:
                all_entities.extend(entities)
                all_relations.extend(relations)
                processed_doc_ids.append(doc_id)
                failed_doc_ids = [failed_id for failed_id in failed_doc_ids if failed_id != doc_id]
            else:
                failed_doc_ids.append(doc_id)
            count += 1
            if count % self.checkpoint_interval == 0:
                self._save_checkpoint(doc_type, processed_doc_ids, failed_doc_ids, all_entities, all_relations, total_docs)
        self._save_checkpoint(doc_type, processed_doc_ids, failed_doc_ids, all_entities, all_relations, total_docs)
        return all_entities, all_relations, failed_doc_ids

    async def extract_from_document_async(self, doc_id: str, text: str, doc_type: str = "patent") -> tuple[list[Entity], list[Relation]]:
        text = as_text(text)[: config.ENTITY_EXTRACTION_MAX_CHARS]
        if not text:
            return [], []
        response = await self._call_llm_async(format_entity_extraction_prompt(text), doc_id)
        return self._parse_response(response, doc_id)

    async def _call_llm_async(self, prompt: str, doc_id: str) -> str:
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=4096,
                    )
                log_chat_usage("entity_extraction_async", self.model, response)
                self.stats["success"] += 1
                return response.choices[0].message.content or ""
            except Exception as exc:
                self.stats["retries"] += 1
                logger.warning("[%s] Async LLM error attempt %s/%s: %s", doc_id, attempt + 1, self.max_retries, exc)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep((2 ** attempt) * 2)
        self.stats["failed"] += 1
        return ""
