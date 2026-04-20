"""
HybridRetriever
LightRAG 스타일 Hybrid 검색 (Local + Global)
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from openai import OpenAI

from .cost_tracker import log_chat_usage
from .embedder import Embedder
from .graph_store import GraphStore
from .prompts import format_keyword_extraction_prompt, RAG_RESPONSE_PROMPT
from .settings import OPENAI_API_KEY, LLM_MODEL, RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD
from .vector_store import ChromaVectorStore


@dataclass
class RetrievalResult:
    """검색 결과"""
    entity_name: str
    entity_type: str
    description: str
    source_doc_ids: List[str]
    similarity: float
    search_type: str  # "local" or "global"


class HybridRetriever:
    """LightRAG 스타일 Hybrid 검색기 (Local + Global)"""

    def __init__(
        self,
        doc_types: List[str] = None,
        force_api: bool = False,
        embedder: Embedder = None,
        vector_store: ChromaVectorStore = None
    ):
        """
        Hybrid 검색기 초기화

        Args:
            doc_types: 검색할 문서 타입 리스트 (기본: patent, article, project)
            force_api: OpenAI API 강제 사용 여부
            embedder: 외부에서 주입할 Embedder 인스턴스 (None이면 내부 생성)
            vector_store: 외부에서 주입할 ChromaVectorStore 인스턴스 (None이면 내부 생성)
        """
        self.doc_types = doc_types or ["patent", "article", "project"]

        # OpenAI 클라이언트 (키워드 추출용)
        self.llm_client = OpenAI(api_key=OPENAI_API_KEY)
        self.llm_model = LLM_MODEL

        # 임베딩 모델 (외부 주입 또는 내부 생성)
        self.embedder = embedder or Embedder(force_api=force_api)

        # 벡터/그래프 저장소 (외부 주입 또는 내부 생성)
        self.vector_store = vector_store or ChromaVectorStore()
        self.graph_stores = {
            doc_type: GraphStore(doc_type=doc_type)
            for doc_type in self.doc_types
        }

        print(f"HybridRetriever initialized for doc_types: {self.doc_types}")

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        LLM으로 쿼리에서 키워드 추출 (한글 + 영어)

        Args:
            query: 사용자 쿼리

        Returns:
            Dict with keys: high_level, high_level_en, low_level, low_level_en
        """
        prompt = format_keyword_extraction_prompt(query)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800
            )

            # 비용 추적
            log_chat_usage(
                component="keyword_extraction",
                model=self.llm_model,
                response=response
            )

            content = response.choices[0].message.content.strip()

            # GPT-4o-mini가 ```json ... ``` 으로 감싸는 경우 방어
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*\n?", "", content)
                content = re.sub(r"\n?```\s*$", "", content)
                content = content.strip()

            # JSON 파싱
            result = json.loads(content)
            keywords = {
                "high_level": result.get("high_level_keywords", []),
                "high_level_en": result.get("high_level_keywords_en", []),
                "low_level": result.get("low_level_keywords", []),
                "low_level_en": result.get("low_level_keywords_en", [])
            }

            print(f"Extracted keywords - High: {keywords['high_level']}, Low: {keywords['low_level']}")
            print(f"Extracted keywords (EN) - High: {keywords['high_level_en']}, Low: {keywords['low_level_en']}")
            return keywords

        except Exception as e:
            print(f"Keyword extraction error: {e}")
            # 실패 시 쿼리 자체를 키워드로 사용 (영어는 빈 리스트)
            return {
                "high_level": [query],
                "high_level_en": [],
                "low_level": [query],
                "low_level_en": []
            }

    def _search_entities_by_keywords(
        self,
        keywords: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        키워드로 엔티티 검색 (내부 헬퍼 함수)

        Args:
            keywords: 키워드 리스트
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        if not keywords:
            return []

        keyword_text = " ".join(keywords)
        query_embedding = self.embedder.encode(keyword_text)

        entity_results = self.vector_store.search_entities(
            query_embedding=query_embedding,
            doc_types=self.doc_types,
            top_k=top_k
        )

        return entity_results

    def _local_search(
        self,
        keywords_ko: List[str],
        keywords_en: List[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Local Search: 엔티티 검색 → 연결된 관계 확장
        Article의 경우 한글/영어 각각 검색 후 병합

        Args:
            keywords_ko: 한글 low_level 키워드 리스트
            keywords_en: 영어 low_level 키워드 리스트 (Article용)
            top_k: 각 언어별 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        if not keywords_ko:
            return []

        # 한글 검색 (모든 doc_type)
        ko_results = self._search_entities_by_keywords(keywords_ko, top_k)

        # 영어 검색 (article만, 영어 키워드가 있을 때만)
        en_results = []
        if keywords_en and "article" in self.doc_types:
            en_results = self._search_entities_by_keywords(keywords_en, top_k)

        # 결과 병합 (중복 제거)
        all_results = self._merge_search_results(ko_results, en_results)

        # 결과에 search_type 추가 및 그래프 확장
        results = []
        seen_entities = set()

        for entity in all_results:
            entity_name = entity["metadata"].get("name", "")
            if entity_name in seen_entities:
                continue
            seen_entities.add(entity_name)

            # 그래프에서 1-hop 이웃 조회
            doc_type = entity.get("doc_type", "patent")
            if doc_type in self.graph_stores:
                neighbors = self.graph_stores[doc_type].get_neighbors(
                    entity_name, direction="both", hop=1
                )
                entity["neighbors"] = neighbors

            entity["search_type"] = "local"
            results.append(entity)

        return results

    def _merge_search_results(
        self,
        results_a: List[Dict],
        results_b: List[Dict]
    ) -> List[Dict]:
        """
        두 검색 결과 병합 (doc_id 기준 중복 제거, 높은 similarity 유지)

        Args:
            results_a: 첫 번째 검색 결과
            results_b: 두 번째 검색 결과

        Returns:
            병합된 결과 리스트 (similarity 내림차순)
        """
        doc_map = {}

        for r in results_a + results_b:
            doc_id = r.get('metadata', {}).get('source_doc_id')
            if doc_id is None:
                continue

            if doc_id not in doc_map:
                doc_map[doc_id] = r
            elif r.get('similarity', 0) > doc_map[doc_id].get('similarity', 0):
                doc_map[doc_id] = r

        # similarity 내림차순 정렬
        merged = list(doc_map.values())
        merged.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        return merged

    def _search_relations_by_keywords(
        self,
        keywords: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        키워드로 관계 검색 (내부 헬퍼 함수)

        Args:
            keywords: 키워드 리스트
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        if not keywords:
            return []

        keyword_text = " ".join(keywords)
        query_embedding = self.embedder.encode(keyword_text)

        relation_results = self.vector_store.search_relations(
            query_embedding=query_embedding,
            doc_types=self.doc_types,
            top_k=top_k
        )

        return relation_results

    def _global_search(
        self,
        keywords_ko: List[str],
        keywords_en: List[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Global Search: 관계 검색 → 연결된 엔티티 확장
        Article의 경우 한글/영어 각각 검색 후 병합

        Args:
            keywords_ko: 한글 high_level 키워드 리스트
            keywords_en: 영어 high_level 키워드 리스트 (Article용)
            top_k: 각 언어별 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        if not keywords_ko:
            return []

        # 한글 검색 (모든 doc_type)
        ko_results = self._search_relations_by_keywords(keywords_ko, top_k)

        # 영어 검색 (article만, 영어 키워드가 있을 때만)
        en_results = []
        if keywords_en and "article" in self.doc_types:
            en_results = self._search_relations_by_keywords(keywords_en, top_k)

        # 결과 병합 (중복 제거)
        all_results = self._merge_search_results(ko_results, en_results)

        # 관계에서 연결된 엔티티 정보 추가
        results = []
        for relation in all_results:
            source_entity = relation["metadata"].get("source_entity", "")
            target_entity = relation["metadata"].get("target_entity", "")

            # 그래프에서 엔티티 정보 및 관계 정보 조회
            doc_type = relation.get("doc_type", "patent")
            if doc_type in self.graph_stores:
                graph_store = self.graph_stores[doc_type]
                source_info = graph_store.get_entity(source_entity)
                target_info = graph_store.get_entity(target_entity)
                relation["source_entity_info"] = source_info
                relation["target_entity_info"] = target_info

                # 그래프에서 관계의 전체 정보 조회 (description, keywords 포함)
                graph_relation = graph_store.get_relations_between(source_entity, target_entity)
                if graph_relation:
                    relation["relation_description"] = graph_relation.get("description", "")
                    relation["relation_keywords"] = graph_relation.get("keywords", [])

            relation["search_type"] = "global"
            results.append(relation)

        return results

    def _merge_results(
        self,
        local_results: List[Dict],
        global_results: List[Dict],
        similarity_threshold: float = None
    ) -> List[Dict]:
        """
        Local/Global 결과 병합 (dedup + similarity 임계값 필터링)

        Args:
            local_results: Local Search 결과
            global_results: Global Search 결과
            similarity_threshold: similarity 임계값 (이 값 이상만 반환)

        Returns:
            병합된 결과 리스트 (doc_id 기준 dedup, similarity 내림차순, 임계값 이상만)
        """
        if similarity_threshold is None:
            similarity_threshold = SIMILARITY_THRESHOLD

        # 1. 전체 결과 합치기
        all_results = local_results + global_results

        # 2. doc_id 기준 dedup (max similarity)
        doc_map = {}
        for r in all_results:
            doc_id = r.get('metadata', {}).get('source_doc_id')
            if not doc_id:
                continue

            doc_id = str(doc_id)
            similarity = r.get('similarity', 0)

            if doc_id not in doc_map:
                doc_map[doc_id] = r
            else:
                # max similarity로 갱신
                if similarity > doc_map[doc_id].get('similarity', 0):
                    doc_map[doc_id] = r

        # 3. similarity 내림차순 정렬
        deduped = list(doc_map.values())
        deduped.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        # 4. similarity 임계값 필터링 (top-k 대신)
        filtered = [r for r in deduped if r.get('similarity', 0) >= similarity_threshold]

        return filtered

    def _enrich_with_original_content(
        self,
        merged_results: List[Dict]
    ) -> List[Dict]:
        """
        merged_results에 원본 chunk content 추가

        GraphRAG 엔티티/관계의 짧은 description 대신
        원본 문서의 전체 content를 추가하여 평가 정확도 향상

        Args:
            merged_results: 병합된 검색 결과

        Returns:
            original_content가 추가된 결과 리스트
        """
        if not merged_results:
            return merged_results

        # 1. (doc_type, doc_id) 쌍 추출
        doc_keys = []
        for r in merged_results:
            doc_id = r.get('metadata', {}).get('source_doc_id')
            doc_type = r.get('doc_type')
            if doc_id and doc_type:
                doc_keys.append((doc_type, str(doc_id)))

        if not doc_keys:
            return merged_results

        # 2. 각 (doc_type, doc_id)에 대해 해당 컬렉션에서만 chunk 검색
        doc_contents = {}
        for doc_type, doc_id in set(doc_keys):
            collection_name = f"{doc_type}_chunks"
            if collection_name not in self.vector_store.collections:
                continue

            collection = self.vector_store.collections[collection_name]
            try:
                results = collection.get(
                    where={"doc_id": str(doc_id)},
                    include=["documents"]
                )
                if results and results["documents"]:
                    # (doc_type, doc_id) 조합을 키로 사용
                    doc_contents[(doc_type, doc_id)] = results["documents"][0]
            except:
                pass

        # 3. merged_results에 original_content 추가
        for r in merged_results:
            doc_id = str(r.get('metadata', {}).get('source_doc_id', ''))
            doc_type = r.get('doc_type', '')
            key = (doc_type, doc_id)
            if key in doc_contents:
                r['original_content'] = doc_contents[key]

        return merged_results

    def retrieve(
        self,
        query: str,
        retrieval_top_k: int = None,
        similarity_threshold: float = None,
        mode: str = "hybrid",
        keywords: Dict[str, List[str]] = None
    ) -> Dict:
        """
        쿼리에 대한 관련 문서/엔티티 검색

        Args:
            query: 사용자 쿼리
            retrieval_top_k: Local/Global 검색 시 각각 가져올 개수 (기본: RETRIEVAL_TOP_K)
            similarity_threshold: similarity 임계값 (기본: SIMILARITY_THRESHOLD)
            mode: 검색 모드 ("hybrid", "local", "global")
            keywords: 미리 추출된 키워드 딕셔너리. None이면 내부에서 추출
                     형식: {"high_level": [], "high_level_en": [], "low_level": [], "low_level_en": []}

        Returns:
            검색 결과 딕셔너리
        """
        retrieval_top_k = retrieval_top_k or RETRIEVAL_TOP_K
        similarity_threshold = similarity_threshold if similarity_threshold is not None else SIMILARITY_THRESHOLD

        # 1. 키워드 추출 (외부에서 전달되면 재사용)
        if keywords:
            kw = keywords
        else:
            kw = self._extract_keywords(query)

        # 2. 검색 수행 (retrieval_top_k 사용)
        # Article의 경우 한글+영어 각각 검색, 그 외는 한글만
        local_results = []
        global_results = []

        if mode in ("hybrid", "local"):
            local_results = self._local_search(
                keywords_ko=kw["low_level"],
                keywords_en=kw.get("low_level_en", []),
                top_k=retrieval_top_k
            )

        if mode in ("hybrid", "global"):
            global_results = self._global_search(
                keywords_ko=kw["high_level"],
                keywords_en=kw.get("high_level_en", []),
                top_k=retrieval_top_k
            )

        # 3. 결과 병합 (similarity_threshold 기반 필터링)
        if mode == "hybrid":
            merged_results = self._merge_results(local_results, global_results, similarity_threshold)
        elif mode == "local":
            merged_results = [r for r in local_results if r.get('similarity', 0) >= similarity_threshold]
        else:
            merged_results = [r for r in global_results if r.get('similarity', 0) >= similarity_threshold]

        # 4. 원본 chunk content 추가 (평가용)
        merged_results = self._enrich_with_original_content(merged_results)

        return {
            "query": query,
            "high_level_keywords": kw["high_level"],
            "high_level_keywords_en": kw.get("high_level_en", []),
            "low_level_keywords": kw["low_level"],
            "low_level_keywords_en": kw.get("low_level_en", []),
            "local_results": local_results,
            "global_results": global_results,
            "merged_results": merged_results,
            "mode": mode,
            "similarity_threshold": similarity_threshold
        }

    def _format_context(self, results: List[Dict]) -> str:
        """
        검색 결과를 LLM 컨텍스트 형식으로 변환

        Args:
            results: 검색 결과 리스트

        Returns:
            포맷팅된 컨텍스트 문자열
        """
        context_parts = []

        for i, r in enumerate(results):
            search_type = r.get("search_type", "unknown")
            metadata = r.get("metadata", {})

            if search_type == "local":
                # 엔티티 정보
                name = metadata.get("name", "N/A")
                entity_type = metadata.get("entity_type", "N/A")
                description = r.get("document", "")
                source = metadata.get("source_doc_id", "N/A")

                context_parts.append(
                    f"[엔티티 {i+1}] {name} ({entity_type})\n"
                    f"  설명: {description}\n"
                    f"  출처: {source}"
                )

                # 이웃 정보 추가
                neighbors = r.get("neighbors", [])
                if neighbors:
                    neighbor_names = [n["name"] for n in neighbors[:3]]
                    context_parts.append(f"  관련: {', '.join(neighbor_names)}")

            else:
                # 관계 정보
                source_entity = metadata.get("source_entity", "N/A")
                target_entity = metadata.get("target_entity", "N/A")
                keywords = metadata.get("keywords", "")
                description = r.get("document", "")

                context_parts.append(
                    f"[관계 {i+1}] {source_entity} → {target_entity}\n"
                    f"  키워드: {keywords}\n"
                    f"  설명: {description}"
                )

        return "\n\n".join(context_parts)

    def generate_response(
        self,
        query: str,
        contexts: List[Dict],
        response_type: str = "간결하게 2-3문장으로 답변"
    ) -> str:
        """
        검색 결과를 바탕으로 자연어 응답 생성

        Args:
            query: 사용자 쿼리
            contexts: 검색 결과 리스트
            response_type: 응답 형식 지정

        Returns:
            생성된 자연어 응답
        """
        if not contexts:
            return "검색 결과가 없습니다."

        # 컨텍스트 포맷팅
        context_data = self._format_context(contexts)

        # 프롬프트 생성
        prompt = RAG_RESPONSE_PROMPT.format(
            response_type=response_type,
            context_data=context_data
        )

        # 사용자 질문 추가
        full_prompt = f"{prompt}\n\n---Question---\n{query}"

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
                max_tokens=1000
            )

            # 비용 추적
            log_chat_usage(
                component="rag_response",
                model=self.llm_model,
                response=response
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Response generation error: {e}")
            return f"응답 생성 중 오류가 발생했습니다: {e}"

    def query(
        self,
        query: str,
        top_k: int = None,
        mode: str = "hybrid",
        generate: bool = True
    ) -> Dict:
        """
        검색 + 응답 생성 통합 메서드

        Args:
            query: 사용자 쿼리
            top_k: 검색 결과 수
            mode: 검색 모드
            generate: 응답 생성 여부

        Returns:
            검색 결과 + 생성된 응답
        """
        # 검색 수행
        results = self.retrieve(query, top_k=top_k, mode=mode)

        # 응답 생성
        if generate:
            response = self.generate_response(
                query=query,
                contexts=results["merged_results"]
            )
            results["response"] = response
        else:
            results["response"] = None

        return results


if __name__ == "__main__":
    # 테스트
    print("Testing HybridRetriever...")

    retriever = HybridRetriever(doc_types=["patent"])

    # 테스트 쿼리
    test_query = "딥러닝을 활용한 의료영상 진단 전문가를 찾아줘"
    print(f"\nQuery: {test_query}")

    # 검색 + 응답 생성
    results = retriever.query(test_query, top_k=5, generate=True)

    print(f"\nHigh-level keywords: {results['high_level_keywords']}")
    print(f"Low-level keywords: {results['low_level_keywords']}")
    print(f"\nLocal results: {len(results['local_results'])}")
    print(f"Global results: {len(results['global_results'])}")
    print(f"Merged results: {len(results['merged_results'])}")

    print("\n=== Generated Response ===")
    print(results['response'])
