import json
import logging
from pathlib import Path
from typing import Dict, List

from .embedder import Embedder
from .entity_extractor import EntityRelationExtractor, to_dicts
from .graph_store import GraphStore
from .settings import (
    DATA_TRAIN_ARTICLE_FILE,
    DATA_TRAIN_PATENT_FILE,
    DATA_TRAIN_PROJECT_FILE,
)
from .text_processor import TextProcessor
from .vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)

DATA_FILES = {
    "patent": DATA_TRAIN_PATENT_FILE,
    "article": DATA_TRAIN_ARTICLE_FILE,
    "project": DATA_TRAIN_PROJECT_FILE,
}


class IncrementalIndexBuilder:
    def __init__(self, doc_type: str, force_api: bool = False, store_dir: str | None = None):
        if doc_type not in DATA_FILES:
            raise ValueError(f"Unknown doc_type: {doc_type}")

        self.doc_type = doc_type
        self.force_api = force_api
        self.store_dir = store_dir
        self.vector_store = ChromaVectorStore(persist_dir=store_dir)
        self.graph_store = GraphStore(store_dir=store_dir, doc_type=doc_type)
        self.text_processor = TextProcessor()

    def run(
        self,
        data_file: str | None = None,
        limit: int | None = None,
        dry_run: bool = False,
        batch_size: int = 10,
    ) -> Dict:
        raw_data = self.load_data(data_file)
        unique_raw_data, duplicate_input_count = self._dedupe_raw_documents(raw_data)

        if limit is not None:
            unique_raw_data = unique_raw_data[:limit]

        new_raw_data, existing_count = self._filter_existing_documents(unique_raw_data)
        stats = {
            "doc_type": self.doc_type,
            "loaded": len(raw_data),
            "duplicate_input_skipped": duplicate_input_count,
            "already_indexed_skipped": existing_count,
            "new_candidates": len(new_raw_data),
            "dry_run": dry_run,
        }

        if dry_run or not new_raw_data:
            stats.update({
                "docs_processed": 0,
                "entities_stored": 0,
                "relations_stored": 0,
                "chunks_stored": 0,
            })
            return stats

        processed_docs = self._process_documents(new_raw_data)
        docs = [
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for doc in processed_docs
        ]

        extractor = EntityRelationExtractor()
        entities, relations, failed_doc_ids = extractor.extract_batch(
            documents=docs,
            doc_type=self.doc_type,
            batch_size=batch_size,
        )
        entity_dicts = self._merge_duplicate_entities(to_dicts(entities))
        relation_dicts = self._merge_duplicate_relations(to_dicts(relations))

        embedder = Embedder(force_api=self.force_api)
        entity_embeddings, relation_embeddings, chunk_embeddings = self._generate_embeddings(
            embedder,
            entity_dicts,
            relation_dicts,
            docs,
        )
        self._store_vector_data(
            entity_dicts,
            relation_dicts,
            docs,
            entity_embeddings,
            relation_embeddings,
            chunk_embeddings,
        )
        self.graph_store.add_entities_batch(entity_dicts)
        self.graph_store.add_relations_batch(relation_dicts)
        self.graph_store.save()

        stats.update({
            "docs_processed": len(docs),
            "failed_docs": len(failed_doc_ids),
            "entities_stored": len(entity_dicts),
            "relations_stored": len(relation_dicts),
            "chunks_stored": len(docs),
            "vector_store": self.vector_store.get_stats(),
            "graph_store": self.graph_store.get_stats(),
        })
        return stats

    def load_data(self, data_file: str | None = None) -> List[Dict]:
        path = Path(data_file) if data_file else Path(DATA_FILES[self.doc_type])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {path}")
        return data

    def _dedupe_raw_documents(self, raw_data: List[Dict]) -> tuple[List[Dict], int]:
        seen_doc_ids = set()
        unique = []
        duplicate_count = 0

        for item in raw_data:
            doc_id = str(item.get("no", "")).strip()
            if not doc_id:
                duplicate_count += 1
                continue
            if doc_id in seen_doc_ids:
                duplicate_count += 1
                continue
            seen_doc_ids.add(doc_id)
            unique.append(item)

        return unique, duplicate_count

    def _filter_existing_documents(self, raw_data: List[Dict]) -> tuple[List[Dict], int]:
        doc_ids = [str(item.get("no", "")).strip() for item in raw_data]
        new_doc_ids = set(self.vector_store.filter_new_doc_ids(self.doc_type, doc_ids))
        new_raw_data = [
            item
            for item in raw_data
            if str(item.get("no", "")).strip() in new_doc_ids
        ]
        return new_raw_data, len(raw_data) - len(new_raw_data)

    def _process_documents(self, raw_data: List[Dict]):
        docs = self.text_processor.process(self.doc_type, raw_data)
        logger.info("Text processing stats: %s", self.text_processor.get_stats())
        return docs

    def _generate_embeddings(self, embedder, entities: List[Dict], relations: List[Dict], docs: List[Dict]):
        entity_texts = [
            f"{entity.get('name', '')}\n{entity.get('description', '')}"
            for entity in entities
        ]
        relation_texts = [
            (
                f"{relation.get('keywords', '')}\t{relation['source_entity']}\n"
                f"{relation['target_entity']}\n{relation.get('description', '')}"
            )
            for relation in relations
        ]
        chunk_texts = [doc["text"] for doc in docs]

        entity_embeddings = embedder.encode(entity_texts) if entity_texts else None
        relation_embeddings = embedder.encode(relation_texts) if relation_texts else None
        chunk_embeddings = embedder.encode(chunk_texts) if chunk_texts else None
        return entity_embeddings, relation_embeddings, chunk_embeddings

    def _store_vector_data(
        self,
        entities: List[Dict],
        relations: List[Dict],
        docs: List[Dict],
        entity_embeddings,
        relation_embeddings,
        chunk_embeddings,
    ):
        if entities and entity_embeddings is not None:
            self.vector_store.add_entities(entities, entity_embeddings, doc_type=self.doc_type)
        if relations and relation_embeddings is not None:
            self.vector_store.add_relations(relations, relation_embeddings, doc_type=self.doc_type)
        if docs and chunk_embeddings is not None:
            chunks = [
                {
                    "doc_id": doc["doc_id"],
                    "text": doc["text"],
                    "title": doc.get("metadata", {}).get("title", ""),
                }
                for doc in docs
            ]
            self.vector_store.add_chunks(chunks, chunk_embeddings, doc_type=self.doc_type)

    def _merge_duplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        merged = {}
        for entity in entities:
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            if name not in merged:
                merged[name] = entity.copy()
                continue

            existing = merged[name]
            new_description = entity.get("description", "")
            if new_description and new_description not in existing.get("description", ""):
                existing["description"] = f"{existing.get('description', '')}\n{new_description}".strip()

            source_ids = set(str(existing.get("source_doc_id", "")).split(","))
            source_ids.add(str(entity.get("source_doc_id", "")))
            existing["source_doc_id"] = ",".join(sorted(source_ids - {""}))

        return list(merged.values())

    def _merge_duplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        merged = {}
        for relation in relations:
            source = relation.get("source_entity", "")
            target = relation.get("target_entity", "")
            if not source or not target:
                continue
            key = "|".join(sorted([source, target]))
            if key not in merged:
                merged[key] = relation.copy()
                continue

            existing = merged[key]
            old_keywords = {item.strip() for item in existing.get("keywords", "").split(",") if item.strip()}
            new_keywords = {item.strip() for item in relation.get("keywords", "").split(",") if item.strip()}
            existing["keywords"] = ",".join(sorted(old_keywords | new_keywords))

            new_description = relation.get("description", "")
            if new_description and new_description not in existing.get("description", ""):
                existing["description"] = f"{existing.get('description', '')}\n{new_description}".strip()

            source_ids = set(str(existing.get("source_doc_id", "")).split(","))
            source_ids.add(str(relation.get("source_doc_id", "")))
            existing["source_doc_id"] = ",".join(sorted(source_ids - {""}))

        return list(merged.values())
