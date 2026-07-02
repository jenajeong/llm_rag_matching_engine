import argparse
import asyncio
import hashlib
import json
import logging
import pickle
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import config
from ..core.safe import as_text
from ..cost_tracker import get_cost_tracker
from ..io import load_json_records
from ..preprocessing import TextProcessor
from .merge import merge_duplicate_entities, merge_duplicate_relations

if TYPE_CHECKING:
    from ..embedding import Embedder
    from ..llm import AsyncEntityRelationExtractor, EntityRelationExtractor
    from ..stores import ChromaVectorStore, GraphStore

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


logger = logging.getLogger(__name__)


def setup_logging(doc_type: str = "index") -> Path:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOG_DIR / f"build_index_{doc_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 기존 파일 핸들러만 제거 (콘솔 핸들러는 유지)
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.FileHandler)]

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 콘솔 핸들러가 없으면 추가
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_file


class IndexBuilder:
    def __init__(
        self,
        doc_type: str = "patent",
        force_api: bool = False,
        store_dir: str | Path | None = None,
        concurrency: int = 1,
        checkpoint_interval: int = 20,
        min_text_length: int | None = None,
    ):
        if doc_type not in config.TRAIN_FILES:
            raise ValueError(f"Unknown doc_type: {doc_type}")
        self.doc_type = doc_type
        self.store_dir = Path(store_dir) if store_dir else None
        self.concurrency = max(1, int(concurrency or 1))
        self.checkpoint_interval = checkpoint_interval
        self.text_processor = TextProcessor(min_text_length=min_text_length)
        self.use_async = self.concurrency > 1
        self._extractor = None
        self.force_api = force_api
        self._embedder = None
        self._vector_store = None
        self._graph_store = None
        self.stats: dict[str, Any] = {
            "docs_loaded": 0,
            "docs_processed": 0,
            "docs_already_indexed": 0,
            "docs_new": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "entities_after_merge": 0,
            "relations_after_merge": 0,
            "chunks_stored": 0,
            "failed_docs": 0,
            "errors": 0,
        }

    @property
    def embedder(self) -> "Embedder":
        if self._embedder is None:
            from ..embedding import Embedder

            self._embedder = Embedder(force_api=self.force_api)
        return self._embedder

    @property
    def extractor(self):
        if self._extractor is None:
            from ..llm import AsyncEntityRelationExtractor, EntityRelationExtractor

            self._extractor = (
                AsyncEntityRelationExtractor(concurrency=self.concurrency, checkpoint_interval=self.checkpoint_interval)
                if self.use_async
                else EntityRelationExtractor()
            )
        return self._extractor

    @property
    def vector_store(self) -> "ChromaVectorStore":
        if self._vector_store is None:
            from ..stores import ChromaVectorStore

            self._vector_store = ChromaVectorStore(persist_dir=self.store_dir)
        return self._vector_store

    @property
    def graph_store(self) -> "GraphStore":
        if self._graph_store is None:
            from ..stores import GraphStore

            self._graph_store = GraphStore(store_dir=self.store_dir, doc_type=self.doc_type)
        return self._graph_store

    def load_data(self, file_path: str | Path | None = None) -> list[dict[str, Any]]:
        path = Path(file_path) if file_path else config.TRAIN_FILES[self.doc_type]
        logger.info("Loading %s train data from %s", self.doc_type, path)
        records = load_json_records(path)
        self.stats["docs_loaded"] = len(records)
        return records

    def process_documents(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        processed = self.text_processor.process(self.doc_type, raw_data)
        docs = [{"doc_id": doc.doc_id, "doc_type": doc.doc_type, "text": doc.text, "metadata": doc.metadata} for doc in processed]
        self.stats["docs_processed"] = len(docs)
        logger.info("Processed %s/%s documents. Stats=%s", len(docs), len(raw_data), self.text_processor.get_stats())
        return docs

    def filter_new_documents(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        ChromaDB chunks 컬렉션에 이미 존재하는 문서를 제외하고
        새로 처리해야 할 문서만 반환한다.
        """
        doc_ids = [as_text(doc.get("doc_id")) for doc in docs]
        new_doc_ids = set(self.vector_store.filter_new_doc_ids(self.doc_type, doc_ids))

        new_docs = [doc for doc in docs if as_text(doc.get("doc_id")) in new_doc_ids]
        already_indexed = len(docs) - len(new_docs)

        self.stats["docs_already_indexed"] = already_indexed
        self.stats["docs_new"] = len(new_docs)

        logger.info(
            "Duplicate check — already indexed: %s, new: %s",
            already_indexed,
            len(new_docs),
        )
        return new_docs

    def _doc_candidate_ids(self, doc: dict[str, Any]) -> set[str]:
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        return {
            candidate_id
            for candidate_id in {
                as_text(doc.get("doc_id")),
                as_text(metadata.get("legacy_doc_id")),
                *self._doc_content_aliases(doc),
            }
            if candidate_id
        }

    def _doc_content_aliases(self, doc: dict[str, Any]) -> set[str]:
        text = as_text(doc.get("text")).strip()
        if not text:
            return set()
        normalized_text = " ".join(text.split())
        aliases = {f"text:{hashlib.sha1(normalized_text.encode('utf-8')).hexdigest()}"}

        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        title = as_text(metadata.get("title")).strip().lower()
        if title:
            aliases.add(f"title_text:{hashlib.sha1(f'{title}|{normalized_text}'.encode('utf-8')).hexdigest()}")
        return aliases

    def _existing_extraction_payload(self) -> tuple[set[str], dict[str, list[dict]], dict[str, list[dict]], dict[str, str]]:
        existing_doc_ids: set[str] = set()
        entities_by_doc: dict[str, list[dict]] = {}
        relations_by_doc: dict[str, list[dict]] = {}
        alias_to_doc_id: dict[str, str] = {}

        def ingest(payload: dict[str, Any]) -> None:
            existing_doc_ids.update(as_text(doc_id) for doc_id in payload.get("processed_doc_ids", []))
            for doc in payload.get("docs", []):
                if isinstance(doc, dict):
                    doc_id = as_text(doc.get("doc_id"))
                    if not doc_id:
                        continue
                    for candidate_id in self._doc_candidate_ids(doc):
                        existing_doc_ids.add(candidate_id)
                        alias_to_doc_id[candidate_id] = doc_id
            for entity in payload.get("entities", []):
                if not isinstance(entity, dict):
                    continue
                source_doc_id = as_text(entity.get("source_doc_id"))
                if not source_doc_id:
                    continue
                existing_doc_ids.add(source_doc_id)
                alias_to_doc_id.setdefault(source_doc_id, source_doc_id)
                entities_by_doc.setdefault(source_doc_id, []).append(entity)
            for relation in payload.get("relations", []):
                if not isinstance(relation, dict):
                    continue
                source_doc_id = as_text(relation.get("source_doc_id"))
                if not source_doc_id:
                    continue
                existing_doc_ids.add(source_doc_id)
                alias_to_doc_id.setdefault(source_doc_id, source_doc_id)
                relations_by_doc.setdefault(source_doc_id, []).append(relation)

        checkpoint_path = config.CHECKPOINT_DIR / f"extraction_{self.doc_type}_checkpoint.json"
        if checkpoint_path.exists():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                ingest(checkpoint)
            except Exception:
                logger.warning("Failed to read extraction checkpoint: %s", checkpoint_path, exc_info=True)

        artifact_path = self.get_extraction_file()
        if artifact_path.exists():
            try:
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                ingest(artifact)
            except Exception:
                logger.warning("Failed to read extraction artifact: %s", artifact_path, exc_info=True)

        split_artifact_root = config.CHECKPOINT_DIR / "split_index_runs"
        if split_artifact_root.exists():
            for split_artifact_path in split_artifact_root.glob(f"*/{self.doc_type}/artifacts/*.json"):
                try:
                    artifact = json.loads(split_artifact_path.read_text(encoding="utf-8"))
                    if artifact.get("doc_type") == self.doc_type:
                        ingest(artifact)
                except Exception:
                    logger.warning("Failed to read split extraction artifact: %s", split_artifact_path, exc_info=True)

        return {doc_id for doc_id in existing_doc_ids if doc_id}, entities_by_doc, relations_by_doc, alias_to_doc_id

    def _existing_extracted_doc_ids(self) -> set[str]:
        existing_doc_ids, _, _, _ = self._existing_extraction_payload()
        return existing_doc_ids

    def filter_unextracted_documents(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        existing_doc_ids = self._existing_extracted_doc_ids()
        if not existing_doc_ids:
            return docs

        def already_extracted(doc: dict[str, Any]) -> bool:
            return any(candidate_id in existing_doc_ids for candidate_id in self._doc_candidate_ids(doc))

        new_docs = [doc for doc in docs if not already_extracted(doc)]
        skipped = len(docs) - len(new_docs)
        if skipped:
            logger.info("Extraction duplicate check - already extracted: %s, remaining: %s", skipped, len(new_docs))
        return new_docs

    def split_existing_extractions(self, docs: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        existing_doc_ids, entities_by_doc, relations_by_doc, alias_to_doc_id = self._existing_extraction_payload()
        if not existing_doc_ids:
            return docs, [], [], []

        docs_to_extract: list[dict] = []
        reusable_docs: list[dict] = []
        reusable_entities: list[dict] = []
        reusable_relations: list[dict] = []

        for doc in docs:
            current_doc_id = as_text(doc.get("doc_id"))
            matched_alias = next((candidate for candidate in self._doc_candidate_ids(doc) if candidate in existing_doc_ids), "")
            if not matched_alias:
                docs_to_extract.append(doc)
                continue

            matched_doc_id = alias_to_doc_id.get(matched_alias, matched_alias)
            reusable_docs.append(doc)
            for entity in entities_by_doc.get(matched_doc_id, []):
                item = dict(entity)
                item["source_doc_id"] = current_doc_id
                reusable_entities.append(item)
            for relation in relations_by_doc.get(matched_doc_id, []):
                item = dict(relation)
                item["source_doc_id"] = current_doc_id
                reusable_relations.append(item)

        if reusable_docs:
            logger.info(
                "Reusing existing extraction payload: docs=%s, entities=%s, relations=%s, remaining=%s",
                len(reusable_docs),
                len(reusable_entities),
                len(reusable_relations),
                len(docs_to_extract),
            )
        return docs_to_extract, reusable_docs, reusable_entities, reusable_relations

    def extract_entities_relations(self, docs: list[dict[str, Any]], batch_size: int = 10) -> tuple[list[dict], list[dict], list[str]]:
        if not docs:
            return [], [], []
        if self.use_async:
            return self._extract_async(docs)
        return self._extract_sync(docs, batch_size=batch_size)

    def _extract_async(self, docs: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[str]]:
        started = time.time()
        total = len(docs)
        processed = 0

        def progress(doc_id: str, entity_count: int, relation_count: int) -> None:
            nonlocal processed
            processed += 1
            if processed % 100 == 0 or processed == total:
                elapsed = max(time.time() - started, 1)
                logger.info("Extraction progress: %s/%s (%.1f docs/min)", processed, total, processed / elapsed * 60)

        entities, relations, failed_doc_ids = asyncio.run(
            self.extractor.extract_batch_async(docs, doc_type=self.doc_type, progress_callback=progress)
        )
        return self._normalize_extraction_result(entities, relations, failed_doc_ids)

    def _extract_sync(self, docs: list[dict[str, Any]], batch_size: int = 10) -> tuple[list[dict], list[dict], list[str]]:
        all_entities = []
        all_relations = []
        extracted_doc_ids = set()
        for i in tqdm(range(0, len(docs), batch_size), desc=f"Extracting {self.doc_type}"):
            batch = docs[i: i + batch_size]
            try:
                entities, relations = self.extractor.extract_batch(batch, doc_type=self.doc_type)
                for entity in entities:
                    entity_dict = asdict(entity) if hasattr(entity, "__dataclass_fields__") else dict(entity)
                    all_entities.append(entity_dict)
                    extracted_doc_ids.add(as_text(entity_dict.get("source_doc_id")))
                for relation in relations:
                    all_relations.append(asdict(relation) if hasattr(relation, "__dataclass_fields__") else dict(relation))
            except Exception:
                logger.exception("Extraction batch failed at offset %s", i)
                self.stats["errors"] += 1
        all_doc_ids = {as_text(doc.get("doc_id")) for doc in docs}
        failed_doc_ids = sorted(all_doc_ids - extracted_doc_ids)
        return self._normalize_extraction_result(all_entities, all_relations, failed_doc_ids)

    def _normalize_extraction_result(self, entities, relations, failed_doc_ids):
        entity_dicts = [asdict(e) if hasattr(e, "__dataclass_fields__") else dict(e) for e in entities]
        relation_dicts = [asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r) for r in relations]
        self.stats["entities_extracted"] = len(entity_dicts)
        self.stats["relations_extracted"] = len(relation_dicts)
        self.stats["failed_docs"] = len(failed_doc_ids)
        logger.info("Extracted %s entities, %s relations; failed docs=%s", len(entity_dicts), len(relation_dicts), len(failed_doc_ids))
        return entity_dicts, relation_dicts, list(failed_doc_ids)

    def generate_embeddings(self, entities: list[dict], relations: list[dict], docs: list[dict]) -> tuple[Any, Any, Any]:
        entity_texts = [f"{as_text(e.get('name'))}\n{as_text(e.get('description'))}" for e in entities]
        relation_texts = [
            f"{as_text(r.get('keywords'))}\t{as_text(r.get('source_entity'))}\n{as_text(r.get('target_entity'))}\n{as_text(r.get('description'))}"
            for r in relations
        ]
        chunk_texts = [as_text(doc.get("text")) for doc in docs if as_text(doc.get("text"))]
        entity_embeddings = self.embedder.encode(entity_texts) if entity_texts else None
        relation_embeddings = self.embedder.encode(relation_texts) if relation_texts else None
        chunk_embeddings = self.embedder.encode(chunk_texts) if chunk_texts else None
        return entity_embeddings, relation_embeddings, chunk_embeddings

    def _iter_batches(self, items: list[Any], batch_size: int):
        batch_size = max(1, int(batch_size or 1))
        for index in range(0, len(items), batch_size):
            yield items[index:index + batch_size]

    def store_to_vector_db(self, entities, relations, docs, entity_embeddings, relation_embeddings, chunk_embeddings) -> None:
        if entities and entity_embeddings is not None:
            self.vector_store.add_entities(entities, entity_embeddings, self.doc_type)
        if relations and relation_embeddings is not None:
            self.vector_store.add_relations(relations, relation_embeddings, self.doc_type)
        if docs and chunk_embeddings is not None:
            chunks = [
                {"doc_id": doc["doc_id"], "text": doc["text"], "title": doc.get("metadata", {}).get("title", "")}
                for doc in docs
                if as_text(doc.get("text"))
            ]
            self.vector_store.add_chunks(chunks, chunk_embeddings, self.doc_type)
            self.stats["chunks_stored"] = len(chunks)
        logger.info("Chroma stats: %s", self.vector_store.get_stats())

    def store_to_graph_db(self, entities: list[dict], relations: list[dict]) -> None:
        self.graph_store.add_entities_batch(entities)
        self.graph_store.add_relations_batch(relations)
        self.graph_store.save()
        logger.info("Graph stats: %s", self.graph_store.get_stats())

    def store_payload_streaming(
        self,
        docs: list[dict],
        entities: list[dict],
        relations: list[dict],
        embedding_batch_size: int = 128,
        save_graph: bool = True,
    ) -> None:
        for entity_batch in self._iter_batches(entities, embedding_batch_size):
            embeddings = self.generate_embeddings(entity_batch, [], [])
            self.store_to_vector_db(entity_batch, [], [], *embeddings)
            self.graph_store.add_entities_batch(entity_batch)

        for relation_batch in self._iter_batches(relations, embedding_batch_size):
            embeddings = self.generate_embeddings([], relation_batch, [])
            self.store_to_vector_db([], relation_batch, [], *embeddings)
            self.graph_store.add_relations_batch(relation_batch)

        for doc_batch in self._iter_batches(docs, embedding_batch_size):
            embeddings = self.generate_embeddings([], [], doc_batch)
            self.store_to_vector_db([], [], doc_batch, *embeddings)

        if save_graph:
            self.graph_store.save()
            logger.info("Graph stats: %s", self.graph_store.get_stats())

    def get_extraction_file(self, path: str | Path | None = None) -> Path:
        if path:
            return Path(path)
        return config.CHECKPOINT_DIR / "extraction_artifacts" / f"{self.doc_type}_extraction.json"

    def prepare_documents(self, data_file: str | Path | None = None, clear: bool = False) -> list[dict[str, Any]]:
        raw_data = self.load_data(data_file)
        docs = self.process_documents(raw_data)
        if clear:
            self.stats["docs_new"] = len(docs)
            return docs
        return self.filter_new_documents(docs)

    def run_extract(
        self,
        data_file: str | Path | None = None,
        clear: bool = False,
        batch_size: int = 10,
        output_file: str | Path | None = None,
        prepared_docs_file: str | Path | None = None,
    ) -> dict[str, Any]:
        logger.info("Starting Indigo extraction phase for %s", self.doc_type)
        tracker = get_cost_tracker()
        tracker.start_task("index_extraction", description=f"{self.doc_type} GPT extraction")

        if prepared_docs_file:
            docs = json.loads(Path(prepared_docs_file).read_text(encoding="utf-8"))
            self.stats["docs_processed"] = len(docs)
            if not clear:
                docs = self.filter_new_documents(docs)
            else:
                self.stats["docs_new"] = len(docs)
        else:
            docs = self.prepare_documents(data_file=data_file, clear=clear)
        reusable_docs: list[dict] = []
        reusable_entities: list[dict] = []
        reusable_relations: list[dict] = []
        if not clear:
            docs, reusable_docs, reusable_entities, reusable_relations = self.split_existing_extractions(docs)
        if not docs:
            if reusable_docs or reusable_entities or reusable_relations:
                logger.info("All documents already extracted. Reusing existing extraction payload.")
                self.stats["entities_extracted"] = len(reusable_entities)
                self.stats["relations_extracted"] = len(reusable_relations)
                self.stats["entities_after_merge"] = len(reusable_entities)
                self.stats["relations_after_merge"] = len(reusable_relations)
                artifact_file = self._save_extraction_artifact(
                    output_file,
                    reusable_docs,
                    reusable_entities,
                    reusable_relations,
                    [],
                )
                cost_result = tracker.end_task(**self.stats)
                if cost_result:
                    logger.info("Estimated extraction API cost: $%.6f", cost_result.get("total_cost_usd", 0.0))
                return {"artifact_file": str(artifact_file), **self.stats}

            logger.info("All documents already indexed. Nothing to extract.")
            tracker.end_task(**self.stats)
            artifact_file = self._save_extraction_artifact(output_file, docs, [], [], [])
            return {"artifact_file": str(artifact_file), **self.stats}

        entities, relations, failed_doc_ids = self.extract_entities_relations(docs, batch_size=batch_size)
        self._save_failed_docs(failed_doc_ids, len(docs))

        if reusable_entities:
            entities = [*reusable_entities, *entities]
        if reusable_relations:
            relations = [*reusable_relations, *relations]
        artifact_docs = [*reusable_docs, *docs]

        if entities:
            entities = merge_duplicate_entities(entities)
            relations = merge_duplicate_relations(relations)
            self.stats["entities_after_merge"] = len(entities)
            self.stats["relations_after_merge"] = len(relations)
        else:
            logger.warning("No entities extracted. Chunks can still be stored in store phase.")

        self.stats["entities_extracted"] = len(entities)
        self.stats["relations_extracted"] = len(relations)
        artifact_file = self._save_extraction_artifact(output_file, artifact_docs, entities, relations, failed_doc_ids)
        cost_result = tracker.end_task(**self.stats)
        if cost_result:
            logger.info("Estimated extraction API cost: $%.6f", cost_result.get("total_cost_usd", 0.0))
        return {"artifact_file": str(artifact_file), **self.stats}

    def run_store(
        self,
        extraction_file: str | Path | None = None,
        clear: bool = False,
        embedding_batch_size: int | None = 128,
    ) -> dict[str, Any]:
        logger.info("Starting Indigo embedding/store phase for %s", self.doc_type)
        artifact = self._load_extraction_artifact(extraction_file)
        docs = artifact.get("docs", [])
        entities = artifact.get("entities", [])
        relations = artifact.get("relations", [])
        self.stats.update(artifact.get("stats", {}))

        if clear:
            self.vector_store.clear_all()
            self.graph_store.clear()

        batch_size = max(1, int(embedding_batch_size or 128))
        self.store_payload_streaming(docs, entities, relations, embedding_batch_size=batch_size)
        logger.info("Store phase complete. Stats=%s", self.stats)
        return self.stats

    def run_store_manifest(
        self,
        manifest_file: str | Path,
        clear: bool = False,
        embedding_batch_size: int = 128,
    ) -> dict[str, Any]:
        logger.info("Starting Indigo streaming store phase for %s from %s", self.doc_type, manifest_file)
        manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
        if manifest.get("doc_type") != self.doc_type:
            raise ValueError(f"Manifest doc_type mismatch: {manifest.get('doc_type')} != {self.doc_type}")

        if clear:
            self.vector_store.clear_all()
            self.graph_store.clear()

        artifact_files = [Path(path) for path in manifest.get("artifact_files", [])]
        totals = {
            "docs_processed": 0,
            "docs_new": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "entities_after_merge": 0,
            "relations_after_merge": 0,
            "failed_docs": 0,
        }
        for artifact_file in artifact_files:
            artifact = self._load_extraction_artifact(artifact_file)
            docs = artifact.get("docs", [])
            entities = artifact.get("entities", [])
            relations = artifact.get("relations", [])
            self.store_payload_streaming(
                docs,
                entities,
                relations,
                embedding_batch_size=embedding_batch_size,
                save_graph=False,
            )
            stats = artifact.get("stats", {})
            for key in totals:
                totals[key] += int(stats.get(key, 0) or 0)

        self.graph_store.save()
        self.stats.update(totals)
        self.stats["manifest_artifacts_stored"] = len(artifact_files)
        logger.info("Streaming store phase complete. Stats=%s", self.stats)
        return self.stats

    def run(
        self,
        data_file: str | Path | None = None,
        clear: bool = False,
        resume: bool = False,
        batch_size: int = 10,
        embedding_batch_size: int = 128,
    ):
        logger.info("Starting Indigo index build for %s", self.doc_type)
        tracker = get_cost_tracker()
        tracker.start_task("indexing", description=f"{self.doc_type} index build")
        checkpoint_file = config.CHECKPOINT_DIR / f"index_{self.doc_type}.pkl"

        if resume and checkpoint_file.exists():
            logger.info("Resuming from checkpoint: %s", checkpoint_file)
            with checkpoint_file.open("rb") as f:
                checkpoint = pickle.load(f)
            docs = checkpoint["docs"]
            entities = checkpoint["entities"]
            relations = checkpoint["relations"]
            self.stats.update(checkpoint.get("stats", {}))
        else:
            if clear:
                self.vector_store.clear_all()
                self.graph_store.clear()

            # 1. 데이터 로드
            raw_data = self.load_data(data_file)

            # 2. 텍스트 전처리 (text null이어도 title로 처리)
            docs = self.process_documents(raw_data)

            # 3. 이미 인덱싱된 문서 제외
            docs = self.filter_new_documents(docs)
            if not clear:
                docs = self.filter_unextracted_documents(docs)

            if not docs:
                logger.info("All documents already indexed. Nothing to do.")
                tracker.end_task(**self.stats)
                return self.stats

            # 4. GPT 엔티티/관계 추출
            entities, relations, failed_doc_ids = self.extract_entities_relations(docs, batch_size=batch_size)
            self._save_failed_docs(failed_doc_ids, len(docs))

            # 엔티티가 없어도 chunk/graph 저장은 계속 진행
            # (title만 있는 문서는 GPT가 엔티티를 못 뽑을 수 있음)
            if entities:
                entities = merge_duplicate_entities(entities)
                relations = merge_duplicate_relations(relations)
                self.stats["entities_after_merge"] = len(entities)
                self.stats["relations_after_merge"] = len(relations)
            else:
                logger.warning(
                    "No entities extracted — skipping entity/relation merge. "
                    "Chunks will still be stored."
                )

            # 5. 체크포인트 저장
            self._save_checkpoint(checkpoint_file, docs, entities, relations, failed_doc_ids)

        # 6. 벡터DB + 그래프DB 저장
        self.store_payload_streaming(docs, entities, relations, embedding_batch_size=embedding_batch_size)

        cost_result = tracker.end_task(**self.stats)
        if cost_result:
            logger.info("Estimated API cost: $%.6f", cost_result.get("total_cost_usd", 0.0))
        logger.info("Index build complete. Stats=%s", self.stats)
        return self.stats

    def retry_failed(self, max_docs: int | None = None):
        failed_file = config.LOG_DIR / f"failed_docs_{self.doc_type}.json"
        checkpoint_file = config.CHECKPOINT_DIR / f"index_{self.doc_type}.pkl"
        if not failed_file.exists() or not checkpoint_file.exists():
            logger.error("Missing failed-doc log or index checkpoint for %s", self.doc_type)
            return None
        failed = json.loads(failed_file.read_text(encoding="utf-8")).get("failed_doc_ids", [])
        with checkpoint_file.open("rb") as f:
            checkpoint = pickle.load(f)
        doc_map = {as_text(doc.get("doc_id")): doc for doc in checkpoint.get("docs", [])}
        docs = [doc_map[doc_id] for doc_id in failed if doc_id in doc_map]
        if max_docs:
            docs = docs[:max_docs]
        entities, relations, still_failed = self.extract_entities_relations(docs)
        if entities:
            entities = merge_duplicate_entities(entities)
            relations = merge_duplicate_relations(relations)
        embeddings = self.generate_embeddings(entities, relations, docs)
        self.store_to_vector_db(entities, relations, docs, *embeddings)
        self.store_to_graph_db(entities, relations)
        return {"retried": len(docs), "success": len(docs) - len(still_failed), "still_failed": len(still_failed)}

    def _save_extraction_artifact(self, path: str | Path | None, docs, entities, relations, failed_doc_ids) -> Path:
        artifact_file = self.get_extraction_file(path)
        artifact_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_type": self.doc_type,
            "docs": docs,
            "entities": entities,
            "relations": relations,
            "failed_doc_ids": failed_doc_ids,
            "stats": self.stats.copy(),
            "saved_at": datetime.now().isoformat(),
        }
        tmp_file = artifact_file.with_suffix(f"{artifact_file.suffix}.tmp")
        tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_file.replace(artifact_file)
        logger.info("Saved extraction artifact: %s", artifact_file)
        return artifact_file

    def _load_extraction_artifact(self, path: str | Path | None) -> dict[str, Any]:
        artifact_file = self.get_extraction_file(path)
        if not artifact_file.exists():
            raise FileNotFoundError(f"Extraction artifact not found: {artifact_file}")
        artifact = json.loads(artifact_file.read_text(encoding="utf-8"))
        if artifact.get("doc_type") != self.doc_type:
            raise ValueError(f"Artifact doc_type mismatch: {artifact.get('doc_type')} != {self.doc_type}")
        return artifact

    def _save_failed_docs(self, failed_doc_ids: list[str], total_docs: int) -> None:
        if not failed_doc_ids:
            return
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = config.LOG_DIR / f"failed_docs_{self.doc_type}.json"
        path.write_text(json.dumps({
            "doc_type": self.doc_type,
            "total_docs": total_docs,
            "failed_count": len(failed_doc_ids),
            "failed_doc_ids": failed_doc_ids,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_checkpoint(self, path: Path, docs, entities, relations, failed_doc_ids) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "docs": docs,
                "entities": entities,
                "relations": relations,
                "failed_doc_ids": failed_doc_ids,
                "stats": self.stats.copy(),
                "saved_at": datetime.now().isoformat(),
            }, f)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Indigo train-data graph/vector indexes.")
    parser.add_argument("--doc-type", choices=["patent", "article", "project"], default="patent")
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--force-api", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-retry", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--phase", choices=["full", "extract", "store"], default="full")
    parser.add_argument("--extraction-file", default=None)
    parser.add_argument("--manifest-file", default=None)
    parser.add_argument("--prepared-docs-file", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument(
        "--allow-extraction-file-store",
        action="store_true",
        help="Allow legacy store from one extraction JSON file. Prefer --manifest-file for large runs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log_file = setup_logging(args.doc_type)
    logger.info("Log file: %s", log_file)
    builder = IndexBuilder(
        doc_type=args.doc_type,
        force_api=args.force_api,
        store_dir=args.store_dir,
        concurrency=args.concurrency,
        checkpoint_interval=args.checkpoint_interval,
    )
    if args.retry_failed:
        result = builder.retry_failed(max_docs=args.max_retry)
    elif args.phase == "extract":
        result = builder.run_extract(
            data_file=args.data_file,
            clear=args.clear,
            batch_size=args.batch_size,
            output_file=args.extraction_file,
            prepared_docs_file=args.prepared_docs_file,
        )
    elif args.phase == "store":
        if args.manifest_file:
            result = builder.run_store_manifest(
                manifest_file=args.manifest_file,
                clear=args.clear,
                embedding_batch_size=args.embedding_batch_size,
            )
        elif args.allow_extraction_file_store:
            result = builder.run_store(
                extraction_file=args.extraction_file,
                clear=args.clear,
                embedding_batch_size=args.embedding_batch_size,
            )
        else:
            raise ValueError("--phase store requires --manifest-file unless --allow-extraction-file-store is set.")
    else:
        result = builder.run(
            args.data_file,
            clear=args.clear,
            resume=args.resume,
            batch_size=args.batch_size,
            embedding_batch_size=args.embedding_batch_size,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
