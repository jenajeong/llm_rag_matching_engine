"""
ChromaDB 벡터 저장소
6개 컬렉션으로 엔티티/관계를 문서 타입별로 저장
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import asdict
import numpy as np

from .settings import RAG_STORE_DIR, TOP_K_RESULTS

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not installed. Run: pip install chromadb")
    raise


# ChromaDB 배치 크기 제한 (최대 5461, 안전하게 5000 사용)
CHROMADB_MAX_BATCH_SIZE = 5000

# 컬렉션 이름 상수
COLLECTIONS = {
    # 엔티티/관계 컬렉션 (LightRAG용)
    "patent_entities": "patent_entities",
    "patent_relations": "patent_relations",
    "article_entities": "article_entities",
    "article_relations": "article_relations",
    "project_entities": "project_entities",
    "project_relations": "project_relations",
    # 청크 컬렉션 (Naive RAG용) - 문서 1개 = 청크 1개
    "patent_chunks": "patent_chunks",
    "article_chunks": "article_chunks",
    "project_chunks": "project_chunks",
}


class ChromaVectorStore:
    """ChromaDB 기반 벡터 저장소 - 9개 컬렉션 관리 (엔티티/관계/청크)

    싱글톤 패턴: 여러 retriever가 같은 인스턴스를 공유하여
    동시 접근으로 인한 HNSW 인덱스 충돌 방지
    """

    _instances = {}  # persist_dir별로 인스턴스 관리

    def __new__(cls, persist_dir: str = None):
        # persist_dir별로 별도 인스턴스 생성
        key = persist_dir or "default"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
            cls._instances[key]._initialized = False
        return cls._instances[key]

    def __init__(self, persist_dir: str = None):
        """
        ChromaDB 벡터 저장소 초기화

        Args:
            persist_dir: 영구 저장 디렉토리
        """
        # 이미 초기화되었으면 스킵
        if getattr(self, '_initialized', False):
            return

        self.persist_dir = Path(persist_dir or RAG_STORE_DIR) / "chromadb"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # ChromaDB 클라이언트 초기화 (영구 저장)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # 6개 컬렉션 초기화
        self.collections = {}
        self._init_collections()

        print(f"ChromaDB initialized at: {self.persist_dir}")
        self._initialized = True

    def _init_collections(self):
        """9개 컬렉션 생성/로드"""
        import time

        for collection_name in COLLECTIONS.values():
            self.collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
            )
            count = self.collections[collection_name].count()
            print(f"  - {collection_name}: {count} items")

        # HNSW 인덱스 안정화를 위한 대기 (ChromaDB 1.4.x 타이밍 이슈 해결)
        time.sleep(1)

    def _get_collection_name(self, doc_type: str, item_type: str) -> str:
        """
        문서 타입과 아이템 타입으로 컬렉션 이름 결정

        Args:
            doc_type: patent / article / project
            item_type: entities / relations

        Returns:
            컬렉션 이름
        """
        return f"{doc_type}_{item_type}"

    def add_entities(
        self,
        entities: List[Dict],
        embeddings: np.ndarray,
        doc_type: str = "patent"
    ):
        """
        엔티티를 컬렉션에 추가

        Args:
            entities: 엔티티 정보 리스트 (name, entity_type, description, source_doc_id)
            embeddings: 엔티티 임베딩 벡터 (numpy array)
            doc_type: 문서 타입 (patent/article/project)
        """
        if len(entities) == 0:
            return

        collection_name = self._get_collection_name(doc_type, "entities")
        collection = self.collections[collection_name]

        # ChromaDB 형식으로 변환
        ids = []
        documents = []
        metadatas = []

        for i, entity in enumerate(entities):
            # 고유 ID 생성 (해시 사용으로 중복 방지)
            raw_id = f"{doc_type}_e_{entity['name']}_{entity['source_doc_id']}"
            hash_suffix = hashlib.md5(raw_id.encode()).hexdigest()[:8]
            entity_id = f"{raw_id.replace(' ', '_')[:90]}_{hash_suffix}"

            ids.append(entity_id)
            # LightRAG 방식: name + description
            documents.append(f"{entity['name']}\n{entity.get('description', '')}")
            metadatas.append({
                "name": entity["name"],
                "entity_type": entity.get("entity_type", "UNKNOWN"),
                "source_doc_id": entity.get("source_doc_id", ""),
                "doc_type": doc_type
            })

        # 임베딩을 리스트로 변환
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        # ChromaDB에 배치로 추가 (upsert로 중복 방지)
        total = len(ids)
        for i in range(0, total, CHROMADB_MAX_BATCH_SIZE):
            batch_end = min(i + CHROMADB_MAX_BATCH_SIZE, total)
            collection.upsert(
                ids=ids[i:batch_end],
                embeddings=embeddings_list[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            print(f"  Batch {i//CHROMADB_MAX_BATCH_SIZE + 1}: {batch_end - i} items")

        print(f"Added {len(entities)} entities to {collection_name}")

    def add_relations(
        self,
        relations: List[Dict],
        embeddings: np.ndarray,
        doc_type: str = "patent"
    ):
        """
        관계를 컬렉션에 추가

        Args:
            relations: 관계 정보 리스트 (source_entity, target_entity, keywords, description, source_doc_id)
            embeddings: 관계 임베딩 벡터
            doc_type: 문서 타입
        """
        if len(relations) == 0:
            return

        collection_name = self._get_collection_name(doc_type, "relations")
        collection = self.collections[collection_name]

        ids = []
        documents = []
        metadatas = []

        for relation in relations:
            # 고유 ID 생성 (해시 사용으로 중복 방지)
            raw_id = f"{doc_type}_r_{relation['source_entity']}_{relation['target_entity']}_{relation['source_doc_id']}"
            hash_suffix = hashlib.md5(raw_id.encode()).hexdigest()[:8]
            rel_id = f"{raw_id.replace(' ', '_')[:90]}_{hash_suffix}"

            ids.append(rel_id)

            # LightRAG 방식: keywords를 맨 앞에 배치
            keywords = relation.get("keywords", "")
            description = relation.get("description", "")
            rel_text = f"{keywords}\t{relation['source_entity']}\n{relation['target_entity']}\n{description}"
            documents.append(rel_text)

            metadatas.append({
                "source_entity": relation["source_entity"],
                "target_entity": relation["target_entity"],
                "keywords": keywords,
                "source_doc_id": relation.get("source_doc_id", ""),
                "doc_type": doc_type
            })

        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        # ChromaDB에 배치로 추가
        total = len(ids)
        for i in range(0, total, CHROMADB_MAX_BATCH_SIZE):
            batch_end = min(i + CHROMADB_MAX_BATCH_SIZE, total)
            collection.upsert(
                ids=ids[i:batch_end],
                embeddings=embeddings_list[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            print(f"  Batch {i//CHROMADB_MAX_BATCH_SIZE + 1}: {batch_end - i} items")

        print(f"Added {len(relations)} relations to {collection_name}")

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        doc_type: str = "patent"
    ):
        """
        문서 청크를 컬렉션에 추가 (Naive RAG용)

        Args:
            chunks: 청크 정보 리스트 (doc_id, text, title 등)
            embeddings: 청크 임베딩 벡터
            doc_type: 문서 타입 (patent/article/project)
        """
        if len(chunks) == 0:
            return

        collection_name = self._get_collection_name(doc_type, "chunks")
        collection = self.collections[collection_name]

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            # 고유 ID (해시 사용으로 중복 방지)
            raw_id = f"{doc_type}_c_{chunk['doc_id']}"
            hash_suffix = hashlib.md5(raw_id.encode()).hexdigest()[:8]
            chunk_id = f"{raw_id.replace(' ', '_')[:90]}_{hash_suffix}"

            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "doc_id": chunk["doc_id"],
                "title": chunk.get("title", ""),
                "doc_type": doc_type
            })

        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        # ChromaDB에 배치로 추가
        total = len(ids)
        for i in range(0, total, CHROMADB_MAX_BATCH_SIZE):
            batch_end = min(i + CHROMADB_MAX_BATCH_SIZE, total)
            collection.upsert(
                ids=ids[i:batch_end],
                embeddings=embeddings_list[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            print(f"  Batch {i//CHROMADB_MAX_BATCH_SIZE + 1}: {batch_end - i} items")

        print(f"Added {len(chunks)} chunks to {collection_name}")

    def search_chunks(
        self,
        query_embedding: Union[List[float], np.ndarray],
        doc_types: List[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        청크 검색 (Naive RAG용)

        Args:
            query_embedding: 쿼리 임베딩 벡터
            doc_types: 검색할 문서 타입 리스트 (None이면 전체)
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트
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
                # HNSW 타이밍 이슈로 인한 재시도
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

    def search_entities(
        self,
        query_embedding: Union[List[float], np.ndarray],
        doc_types: List[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        엔티티 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            doc_types: 검색할 문서 타입 리스트 (None이면 전체)
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        if doc_types is None:
            doc_types = ["patent", "article", "project"]

        top_k = top_k or TOP_K_RESULTS

        # numpy array를 리스트로 변환
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        all_results = []

        for doc_type in doc_types:
            collection_name = self._get_collection_name(doc_type, "entities")
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

                # 결과 정리
                if results and results["ids"][0]:
                    for i, id in enumerate(results["ids"][0]):
                        distance = results["distances"][0][i] if results["distances"] else 0
                        similarity = 1 - distance  # cosine distance를 similarity로 변환

                        all_results.append({
                            "id": id,
                            "document": results["documents"][0][i] if results["documents"] else "",
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "similarity": similarity,
                            "doc_type": doc_type,
                            "item_type": "entity"
                        })

            except Exception as e:
                # HNSW 타이밍 이슈로 인한 재시도
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
                                "item_type": "entity"
                            })
                except Exception as e2:
                    print(f"Search error in {collection_name}: {e2}")
                    continue

        # 유사도 기준 정렬 후 상위 k개 반환
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]

    def search_relations(
        self,
        query_embedding: Union[List[float], np.ndarray],
        doc_types: List[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        관계 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            doc_types: 검색할 문서 타입 리스트
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트
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
                # HNSW 타이밍 이슈로 인한 재시도
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

    def search_all(
        self,
        query_embedding: Union[List[float], np.ndarray],
        doc_types: List[str] = None,
        top_k: int = None
    ) -> Dict[str, List[Dict]]:
        """
        엔티티와 관계 모두 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            doc_types: 검색할 문서 타입 리스트
            top_k: 각 타입별 반환할 결과 수

        Returns:
            {"entities": [...], "relations": [...]}
        """
        return {
            "entities": self.search_entities(query_embedding, doc_types, top_k),
            "relations": self.search_relations(query_embedding, doc_types, top_k)
        }

    def get_stats(self) -> Dict[str, int]:
        """컬렉션별 통계 반환"""
        stats = {}
        for name, collection in self.collections.items():
            stats[name] = collection.count()
        return stats

    def delete_by_doc_id(self, doc_id: str, doc_type: str = "patent"):
        """
        특정 문서의 엔티티/관계 삭제

        Args:
            doc_id: 삭제할 문서 ID
            doc_type: 문서 타입
        """
        for item_type in ["entities", "relations"]:
            collection_name = self._get_collection_name(doc_type, item_type)
            collection = self.collections.get(collection_name)

            if collection:
                collection.delete(where={"source_doc_id": doc_id})

        print(f"Deleted all items for doc_id: {doc_id}")

    def clear_all(self):
        """모든 컬렉션 초기화 (주의!)"""
        for name in COLLECTIONS.values():
            self.client.delete_collection(name)

        self._init_collections()
        print("All collections cleared")


if __name__ == "__main__":
    import numpy as np

    # 테스트
    print("Testing ChromaVectorStore...")
    store = ChromaVectorStore()

    # 샘플 엔티티
    test_entities = [
        {
            "name": "딥러닝",
            "entity_type": "TECHNOLOGY",
            "description": "심층 신경망 기반 기계학습 기술",
            "source_doc_id": "patent_001"
        },
        {
            "name": "의료영상분석",
            "entity_type": "DOMAIN",
            "description": "CT, MRI 등 의료 영상 분석 분야",
            "source_doc_id": "patent_001"
        }
    ]

    # 샘플 임베딩 (실제로는 Embedder 사용)
    test_embeddings = np.random.rand(2, 1536)  # OpenAI 임베딩 차원

    # 엔티티 추가
    store.add_entities(test_entities, test_embeddings, doc_type="patent")

    # 통계 확인
    print("\nCollection stats:")
    for name, count in store.get_stats().items():
        print(f"  - {name}: {count}")

    # 검색 테스트
    query_embedding = np.random.rand(1536)
    results = store.search_entities(query_embedding, doc_types=["patent"], top_k=5)

    print(f"\nSearch results: {len(results)} items")
    for r in results:
        print(f"  - {r['metadata'].get('name')}: {r['similarity']:.4f}")
