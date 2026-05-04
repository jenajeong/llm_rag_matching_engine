import argparse
import asyncio
import json
import logging
import pickle
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import config
from ..core.safe import as_text
from ..cost_tracker import get_cost_tracker
from ..embedding import Embedder
from ..io import load_json_records
from ..llm import AsyncEntityRelationExtractor, EntityRelationExtractor
from ..preprocessing import TextProcessor
from ..stores import ChromaVectorStore, GraphStore
from .merge import merge_duplicate_entities, merge_duplicate_relations

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
        self.text_processor = TextProcessor(min_text_length=min_text_length)
        self.use_async = self.concurrency > 1
        self.extractor = (
            AsyncEntityRelationExtractor(concurrency=self.concurrency, checkpoint_interval=checkpoint_interval)
            if self.use_async
            else EntityRelationExtractor()
        )
        self.embedder = Embedder(force_api=force_api)
        self.vector_store = ChromaVectorStore(persist_dir=self.store_dir)
        self.graph_store = GraphStore(store_dir=self.store_dir, doc_type=doc_type)
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

        Args:
            docs: 전처리된 문서 리스트

        Returns:
            아직 인덱싱되지 않은 문서 리스트
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

    def run(self, data_file: str | Path | None = None, clear: bool = False, resume: bool = False, batch_size: int = 10):
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

            # 2. 텍스트 전처리
            docs = self.process_documents(raw_data)

            # 3. 이미 인덱싱된 문서 제외 (신규 문서만 추출)
            docs = self.filter_new_documents(docs)

            if not docs:
                logger.info("All documents already indexed. Nothing to do.")
                tracker.end_task(**self.stats)
                return self.stats

            # 4. GPT 엔티티/관계 추출
            entities, relations, failed_doc_ids = self.extract_entities_relations(docs, batch_size=batch_size)
            self._save_failed_docs(failed_doc_ids, len(docs))

            if not entities:
                logger.warning("No entities extracted; stopping before embedding/storage.")
                tracker.end_task(**self.stats)
                return self.stats

            # 5. 중복 엔티티/관계 병합
            entities = merge_duplicate_entities(entities)
            relations = merge_duplicate_relations(relations)
            self.stats["entities_after_merge"] = len(entities)
            self.stats["relations_after_merge"] = len(relations)

            # 6. 체크포인트 저장
            self._save_checkpoint(checkpoint_file, docs, entities, relations, failed_doc_ids)

        # 7. 임베딩 생성
        embeddings = self.generate_embeddings(entities, relations, docs)

        # 8. 벡터DB + 그래프DB 저장
        self.store_to_vector_db(entities, relations, docs, *embeddings)
        self.store_to_graph_db(entities, relations)

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
        if not entities:
            return {"retried": len(docs), "success": 0, "still_failed": len(still_failed)}
        entities = merge_duplicate_entities(entities)
        relations = merge_duplicate_relations(relations)
        embeddings = self.generate_embeddings(entities, relations, [])
        self.store_to_vector_db(entities, relations, [], embeddings[0], embeddings[1], None)
        self.store_to_graph_db(entities, relations)
        return {"retried": len(docs), "success": len(docs) - len(still_failed), "still_failed": len(still_failed)}

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
    else:
        result = builder.run(args.data_file, clear=args.clear, resume=args.resume, batch_size=args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()