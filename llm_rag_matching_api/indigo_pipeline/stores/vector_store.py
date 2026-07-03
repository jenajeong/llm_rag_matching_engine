"""
ChromaDB 踰≫꽣 ??μ냼
6媛?而щ젆?섏쑝濡??뷀떚??愿怨꾨? 臾몄꽌 ??낅퀎濡????
"""

import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import asdict
import numpy as np

from ..config import RAG_STORE_DIR, TOP_K_RESULTS, SIMILARITY_THRESHOLD
from ..core.safe import as_text

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not installed. Run: pip install chromadb")
    raise


# ChromaDB 諛곗튂 ?ш린 ?쒗븳 (理쒕? 5461, ?덉쟾?섍쾶 5000 ?ъ슜)
CHROMADB_MAX_BATCH_SIZE = 5000

# 而щ젆???대쫫 ?곸닔
COLLECTIONS = {
    # ?뷀떚??愿怨?而щ젆??(LightRAG??
    "patent_entities": "patent_entities",
    "patent_relations": "patent_relations",
    "article_entities": "article_entities",
    "article_relations": "article_relations",
    "project_entities": "project_entities",
    "project_relations": "project_relations",
    # 泥?겕 而щ젆??(Naive RAG?? - 臾몄꽌 1媛?= 泥?겕 1媛?
    "patent_chunks": "patent_chunks",
    "article_chunks": "article_chunks",
    "project_chunks": "project_chunks",
}


class ChromaVectorStore:
    """ChromaDB 湲곕컲 踰≫꽣 ??μ냼 - 9媛?而щ젆??愿由?(?뷀떚??愿怨?泥?겕)

    ?깃????⑦꽩: ?щ윭 retriever媛 媛숈? ?몄뒪?댁뒪瑜?怨듭쑀?섏뿬
    ?숈떆 ?묎렐?쇰줈 ?명븳 HNSW ?몃뜳??異⑸룎 諛⑹?
    """

    _instances = {}  # persist_dir蹂꾨줈 ?몄뒪?댁뒪 愿由?

    def __new__(cls, persist_dir: str = None):
        # persist_dir蹂꾨줈 蹂꾨룄 ?몄뒪?댁뒪 ?앹꽦
        key = persist_dir or "default"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
            cls._instances[key]._initialized = False
        return cls._instances[key]

    def __init__(self, persist_dir: str = None):
        """
        ChromaDB 踰≫꽣 ??μ냼 珥덇린??

        Args:
            persist_dir: ?곴뎄 ????붾젆?좊━
        """
        # ?대? 珥덇린?붾릺?덉쑝硫??ㅽ궢
        if getattr(self, '_initialized', False):
            return

        self.persist_dir = Path(persist_dir or RAG_STORE_DIR) / "chromadb"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # ChromaDB ?대씪?댁뼵??珥덇린??(?곴뎄 ???
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # 9媛?而щ젆??珥덇린??
        self.collections = {}
        self._init_collections()

        print(f"ChromaDB initialized at: {self.persist_dir}")
        self._initialized = True

    def _init_collections(self):
        """9媛?而щ젆???앹꽦/濡쒕뱶"""
        import time

        for collection_name in COLLECTIONS.values():
            self.collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # 肄붿궗???좎궗???ъ슜
            )
            count = self.collections[collection_name].count()
            print(f"  - {collection_name}: {count} items")

        # HNSW ?몃뜳???덉젙?붾? ?꾪븳 ?湲?(ChromaDB 1.4.x ??대컢 ?댁뒋 ?닿껐)
        time.sleep(1)

    def _get_collection_name(self, doc_type: str, item_type: str) -> str:
        """
        臾몄꽌 ??낃낵 ?꾩씠????낆쑝濡?而щ젆???대쫫 寃곗젙

        Args:
            doc_type: patent / article / project
            item_type: entities / relations / chunks

        Returns:
            而щ젆???대쫫
        """
        return f"{doc_type}_{item_type}"

    def _make_chunk_id(self, doc_type: str, doc_id: str) -> str:
        """
        chunk ID ?앹꽦 (doc_type + doc_id 湲곕컲 ?댁떆)

        Args:
            doc_type: 臾몄꽌 ???
            doc_id: 臾몄꽌 ID

        Returns:
            chunk ID 臾몄옄??
        """
        raw_id = f"{doc_type}_c_{doc_id}"
        hash_suffix = hashlib.md5(raw_id.encode()).hexdigest()[:8]
        return f"{raw_id.replace(' ', '_')[:90]}_{hash_suffix}"

    def _add_new_only(
        self,
        collection,
        ids: List[str],
        embeddings,
        documents: List[str],
        metadatas: List[Dict],
    ) -> int:
        """
        Append-only insert. Existing ids are skipped so stored items are never
        overwritten by a later pipeline run.
        """
        if not ids:
            return 0

        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        inserted = 0
        for i in range(0, len(ids), CHROMADB_MAX_BATCH_SIZE):
            batch_end = min(i + CHROMADB_MAX_BATCH_SIZE, len(ids))
            batch_ids = ids[i:batch_end]
            unique_batch_ids = list(dict.fromkeys(batch_ids))
            existing = set(collection.get(ids=unique_batch_ids).get("ids", []) or [])

            seen_in_batch: set[str] = set()
            new_indexes = []
            for local_index, item_id in enumerate(batch_ids):
                if item_id in existing or item_id in seen_in_batch:
                    continue
                seen_in_batch.add(item_id)
                new_indexes.append(i + local_index)

            if not new_indexes:
                print(f"  Batch {i//CHROMADB_MAX_BATCH_SIZE + 1}: 0 new items")
                continue

            collection.add(
                ids=[ids[index] for index in new_indexes],
                embeddings=[embeddings_list[index] for index in new_indexes],
                documents=[documents[index] for index in new_indexes],
                metadatas=[metadatas[index] for index in new_indexes],
            )
            inserted += len(new_indexes)
            print(f"  Batch {i//CHROMADB_MAX_BATCH_SIZE + 1}: {len(new_indexes)} new items")

        return inserted

    # =========================================================
    # 以묐났 泥댄겕
    # =========================================================

    def filter_new_doc_ids(self, doc_type: str, doc_ids: List[str]) -> List[str]:
        """
        ChromaDB chunks 而щ젆?섏뿉 ?대? 議댁옱?섎뒗 臾몄꽌瑜??쒖쇅?섍퀬
        ?덈줈 泥섎━?댁빞 ??doc_id 紐⑸줉留?諛섑솚?쒕떎.

        Args:
            doc_type: 臾몄꽌 ???(patent / article / project)
            doc_ids: ?뺤씤??doc_id 紐⑸줉

        Returns:
            ?꾩쭅 ?몃뜳?깅릺吏 ?딆? doc_id 紐⑸줉
        """
        collection_name = self._get_collection_name(doc_type, "chunks")
        collection = self.collections.get(collection_name)

        # 而щ젆?섏씠 ?놁쑝硫??꾨? ?좉퇋
        if not collection:
            return [str(doc_id) for doc_id in doc_ids]

        normalized = [str(doc_id) for doc_id in doc_ids]

        # doc_id ??chunk_id 蹂??
        chunk_ids = [self._make_chunk_id(doc_type, doc_id) for doc_id in normalized]

        # ChromaDB??諛곗튂 議고쉶 (?대? ?덈뒗 chunk_id??metadata?먯꽌 doc_id 異붿텧)
        existing_doc_ids: set[str] = set()
        for i in range(0, len(chunk_ids), CHROMADB_MAX_BATCH_SIZE):
            batch = chunk_ids[i:i + CHROMADB_MAX_BATCH_SIZE]
            result = collection.get(ids=batch, include=["metadatas"])
            for meta in result.get("metadatas", []) or []:
                if meta:
                    existing_doc_ids.add(str(meta.get("doc_id", "")))

        # ?좉퇋 doc_id留?諛섑솚
        return [doc_id for doc_id in normalized if doc_id not in existing_doc_ids]

    # =========================================================
    # 異붽?
    # =========================================================

    def add_entities(
        self,
        entities: List[Dict],
        embeddings: np.ndarray,
        doc_type: str = "patent"
    ):
        """
        ?뷀떚?곕? 而щ젆?섏뿉 異붽?

        Args:
            entities: ?뷀떚???뺣낫 由ъ뒪??(name, entity_type, description, source_doc_id)
            embeddings: ?뷀떚???꾨쿋??踰≫꽣 (numpy array)
            doc_type: 臾몄꽌 ???(patent/article/project)
        """
        if len(entities) == 0:
            return

        collection_name = self._get_collection_name(doc_type, "entities")
        collection = self.collections[collection_name]

        ids = []
        documents = []
        metadatas = []

        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                continue
            entity["name"] = as_text(entity.get("name"))
            entity["source_doc_id"] = as_text(entity.get("source_doc_id"))
            entity["description"] = as_text(entity.get("description"))
            entity["entity_type"] = as_text(entity.get("entity_type"), "UNKNOWN")
            if not entity["name"]:
                continue
            # 怨좎쑀 ID ?앹꽦 (?댁떆 ?ъ슜?쇰줈 以묐났 諛⑹?)
            raw_id = f"{doc_type}_e_{entity['name']}_{entity['source_doc_id']}"
            hash_suffix = hashlib.md5(raw_id.encode()).hexdigest()[:8]
            entity_id = f"{raw_id.replace(' ', '_')[:90]}_{hash_suffix}"

            ids.append(entity_id)
            # LightRAG 諛⑹떇: name + description
            documents.append(f"{entity['name']}\n{entity.get('description', '')}")
            metadatas.append({
                "name": entity["name"],
                "entity_type": entity.get("entity_type", "UNKNOWN"),
                "source_doc_id": entity.get("source_doc_id", ""),
                "doc_type": doc_type
            })

        inserted = self._add_new_only(collection, ids, embeddings, documents, metadatas)
        print(f"Added {inserted}/{len(ids)} new entities to {collection_name}")

    def add_relations(
        self,
        relations: List[Dict],
        embeddings: np.ndarray,
        doc_type: str = "patent"
    ):
        """
        愿怨꾨? 而щ젆?섏뿉 異붽?

        Args:
            relations: 愿怨??뺣낫 由ъ뒪??(source_entity, target_entity, keywords, description, source_doc_id)
            embeddings: 愿怨??꾨쿋??踰≫꽣
            doc_type: 臾몄꽌 ???(patent/article/project)
        """
        if len(relations) == 0:
            return

        collection_name = self._get_collection_name(doc_type, "relations")
        collection = self.collections[collection_name]

        ids = []
        documents = []
        metadatas = []

        for i, relation in enumerate(relations):
            if not isinstance(relation, dict):
                continue
            source = as_text(relation.get("source_entity"))
            target = as_text(relation.get("target_entity"))
            if not source or not target:
                continue
            keywords = as_text(relation.get("keywords"))
            source_doc_id = as_text(relation.get("source_doc_id"))

            # Unique per relation and source document. Different professors/docs
            # can have the same semantic relation without overwriting each other.
            raw_id = f"{doc_type}_r_{source}_{target}_{source_doc_id}"
            hash_suffix = hashlib.md5(raw_id.encode()).hexdigest()[:8]
            rel_id = f"{raw_id.replace(' ', '_')[:90]}_{hash_suffix}"

            ids.append(rel_id)
            # LightRAG 諛⑹떇: keywords瑜?留??욎뿉 諛곗튂
            rel_text = f"{keywords}\t{source}\n{target}\n{as_text(relation.get('description'))}"
            documents.append(rel_text)
            metadatas.append({
                "source_entity": source,
                "target_entity": target,
                "keywords": keywords,
                "source_doc_id": source_doc_id,
                "doc_type": doc_type
            })

        inserted = self._add_new_only(collection, ids, embeddings, documents, metadatas)
        print(f"Added {inserted}/{len(ids)} new relations to {collection_name}")

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        doc_type: str = "patent"
    ):
        """
        臾몄꽌 泥?겕瑜?而щ젆?섏뿉 異붽? (Naive RAG??

        Args:
            chunks: 泥?겕 ?뺣낫 由ъ뒪??(doc_id, text, title ??
            embeddings: 泥?겕 ?꾨쿋??踰≫꽣
            doc_type: 臾몄꽌 ???(patent/article/project)
        """
        if len(chunks) == 0:
            return

        collection_name = self._get_collection_name(doc_type, "chunks")
        collection = self.collections[collection_name]

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            chunk["doc_id"] = as_text(chunk.get("doc_id"))
            chunk["text"] = as_text(chunk.get("text"))
            chunk["title"] = as_text(chunk.get("title"))
            if not chunk["doc_id"] or not chunk["text"]:
                continue
            chunk_id = self._make_chunk_id(doc_type, chunk["doc_id"])

            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "doc_id": chunk["doc_id"],
                "title": chunk.get("title", ""),
                "doc_type": doc_type
            })

        inserted = self._add_new_only(collection, ids, embeddings, documents, metadatas)
        print(f"Added {inserted}/{len(ids)} new chunks to {collection_name}")

    # =========================================================
    # 寃??
    # =========================================================

    def _append_query_results(self, all_results: List[Dict], results: Dict, doc_type: str, item_type: str) -> None:
        if not results or not results["ids"][0]:
            return
        for i, item_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0
            all_results.append({
                "id": item_id,
                "document": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "similarity": 1 - distance,
                "doc_type": doc_type,
                "item_type": item_type
            })

    def _deduplicate_entities_by_name(self, results: List[Dict]) -> List[Dict]:
        deduped: dict[str, Dict] = {}
        for item in sorted(results, key=lambda x: x["similarity"], reverse=True):
            metadata = item.get("metadata") or {}
            name = as_text(metadata.get("name")).upper() or as_text(item.get("id"))
            if name and name not in deduped:
                deduped[name] = item
        return list(deduped.values())

    def search_entities(
        self,
        query_embedding: Union[List[float], np.ndarray],
        doc_types: List[str] = None,
        top_k: int = None,
        deduplicate: bool = True,
        fetch_multiplier: int = 5,
        max_dedup_retries: int = 2
    ) -> List[Dict]:
        """
        ?뷀떚??寃??

        Args:
            query_embedding: 荑쇰━ ?꾨쿋??踰≫꽣
            doc_types: 寃?됲븷 臾몄꽌 ???由ъ뒪??(None?대㈃ ?꾩껜)
            top_k: 諛섑솚??寃곌낵 ??

        Returns:
            寃??寃곌낵 由ъ뒪??
        """
        if doc_types is None:
            doc_types = ["patent", "article", "project"]

        top_k = top_k or TOP_K_RESULTS

        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        attempts = max(1, max_dedup_retries + 1) if deduplicate else 1
        base_multiplier = max(1, int(fetch_multiplier or 1))
        best_results: List[Dict] = []

        for attempt in range(attempts):
            requested = top_k * (base_multiplier * (attempt + 1)) if deduplicate else top_k
            all_results = []
            total_available = 0

            for doc_type in doc_types:
                collection_name = self._get_collection_name(doc_type, "entities")
                if collection_name not in self.collections:
                    continue

                collection = self.collections[collection_name]
                count = collection.count()
                if count == 0:
                    continue
                total_available += count
                n_results = min(requested, count)

                try:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"]
                    )
                    self._append_query_results(all_results, results, doc_type, "entity")
                except Exception:
                    import time
                    time.sleep(0.5)
                    try:
                        results = collection.query(
                            query_embeddings=[query_embedding],
                            n_results=n_results,
                            include=["documents", "metadatas", "distances"]
                        )
                        self._append_query_results(all_results, results, doc_type, "entity")
                    except Exception as e2:
                        print(f"Search error in {collection_name}: {e2}")
                        continue

            all_results.sort(key=lambda x: x["similarity"], reverse=True)
            best_results = self._deduplicate_entities_by_name(all_results) if deduplicate else all_results
            if not deduplicate or len(best_results) >= top_k or len(all_results) >= total_available:
                break

        return best_results[:top_k]

    def search_relations(
        self,
        query_embedding: Union[List[float], np.ndarray],
        doc_types: List[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        愿怨?寃??

        Args:
            query_embedding: 荑쇰━ ?꾨쿋??踰≫꽣
            doc_types: 寃?됲븷 臾몄꽌 ???由ъ뒪??
            top_k: 諛섑솚??寃곌낵 ??

        Returns:
            寃??寃곌낵 由ъ뒪??
        """
        if doc_types is None:
            doc_types = ["patent", "article", "project"]

        top_k = top_k or TOP_K_RESULTS

        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        all_results = []

        for doc_type in doc_types:
            collection_name = self._get_collection_name(doc_type, "relations")
            if collection_name not in self.collections:
                continue

            collection = self.collections[collection_name]

            if collection.count() == 0:
                continue

            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )

                if results and results["ids"][0]:
                    for i, id in enumerate(results["ids"][0]):
                        distance = results["distances"][0][i] if results["distances"] else 0
                        similarity = 1 - distance

                        all_results.append({
                            "id": id,
                            "document": results["documents"][0][i] if results["documents"] else "",
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "similarity": similarity,
                            "doc_type": doc_type,
                            "item_type": "relation"
                        })

            except Exception as e:
                # HNSW ??대컢 ?댁뒋濡??명븳 ?ъ떆??
                import time
                time.sleep(0.5)
                try:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k,
                        include=["documents", "metadatas", "distances"]
                    )
                    if results and results["ids"][0]:
                        for i, id in enumerate(results["ids"][0]):
                            distance = results["distances"][0][i] if results["distances"] else 0
                            similarity = 1 - distance
                            all_results.append({
                                "id": id,
                                "document": results["documents"][0][i] if results["documents"] else "",
                                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                                "similarity": similarity,
                                "doc_type": doc_type,
                                "item_type": "relation"
                            })
                except Exception as e2:
                    print(f"Search error in {collection_name}: {e2}")
                    continue

        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]

    def search_chunks(
        self,
        query_embedding: Union[List[float], np.ndarray],
        doc_types: List[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        泥?겕 寃??(Naive RAG??

        Args:
            query_embedding: 荑쇰━ ?꾨쿋??踰≫꽣
            doc_types: 寃?됲븷 臾몄꽌 ???由ъ뒪??(None?대㈃ ?꾩껜)
            top_k: 諛섑솚??寃곌낵 ??

        Returns:
            寃??寃곌낵 由ъ뒪??
        """
        if doc_types is None:
            doc_types = ["patent", "article", "project"]

        top_k = top_k or TOP_K_RESULTS

        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        all_results = []

        for doc_type in doc_types:
            collection_name = self._get_collection_name(doc_type, "chunks")
            if collection_name not in self.collections:
                continue

            collection = self.collections[collection_name]

            if collection.count() == 0:
                continue

            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )

                if results and results["ids"][0]:
                    for i, id in enumerate(results["ids"][0]):
                        distance = results["distances"][0][i] if results["distances"] else 0
                        similarity = 1 - distance

                        all_results.append({
                            "id": id,
                            "document": results["documents"][0][i] if results["documents"] else "",
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "similarity": similarity,
                            "doc_type": doc_type,
                            "item_type": "chunk"
                        })

            except Exception as e:
                # HNSW ??대컢 ?댁뒋濡??명븳 ?ъ떆??
                import time
                time.sleep(0.5)
                try:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k,
                        include=["documents", "metadatas", "distances"]
                    )
                    if results and results["ids"][0]:
                        for i, id in enumerate(results["ids"][0]):
                            distance = results["distances"][0][i] if results["distances"] else 0
                            similarity = 1 - distance
                            all_results.append({
                                "id": id,
                                "document": results["documents"][0][i] if results["documents"] else "",
                                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                                "similarity": similarity,
                                "doc_type": doc_type,
                                "item_type": "chunk"
                            })
                except Exception as e2:
                    print(f"Search error in {collection_name}: {e2}")
                    continue

        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]

    def search_all(
        self,
        query_embedding: Union[List[float], np.ndarray],
        doc_types: List[str] = None,
        top_k: int = None
    ) -> Dict[str, List[Dict]]:
        """
        ?뷀떚?곗? 愿怨?紐⑤몢 寃??

        Args:
            query_embedding: 荑쇰━ ?꾨쿋??踰≫꽣
            doc_types: 寃?됲븷 臾몄꽌 ???由ъ뒪??
            top_k: 媛???낅퀎 諛섑솚??寃곌낵 ??

        Returns:
            {"entities": [...], "relations": [...]}
        """
        return {
            "entities": self.search_entities(query_embedding, doc_types, top_k),
            "relations": self.search_relations(query_embedding, doc_types, top_k)
        }

    # =========================================================
    # ?좏떥由ы떚
    # =========================================================

    def get_stats(self) -> Dict[str, int]:
        """而щ젆?섎퀎 ?듦퀎 諛섑솚"""
        stats = {}
        for name, collection in self.collections.items():
            stats[name] = collection.count()
        return stats

    def delete_by_doc_id(self, doc_id: str, doc_type: str = "patent"):
        """
        ?뱀젙 臾몄꽌???뷀떚??愿怨???젣

        Args:
            doc_id: ??젣??臾몄꽌 ID
            doc_type: 臾몄꽌 ???
        """
        for item_type in ["entities", "relations"]:
            collection_name = self._get_collection_name(doc_type, item_type)
            collection = self.collections.get(collection_name)
            if collection:
                collection.delete(where={"source_doc_id": doc_id})

        print(f"Deleted all items for doc_id: {doc_id}")

    def clear_all(self):
        """紐⑤뱺 而щ젆??珥덇린??(二쇱쓽!)"""
        for name in COLLECTIONS.values():
            self.client.delete_collection(name)

        self._init_collections()
        print("All collections cleared")
