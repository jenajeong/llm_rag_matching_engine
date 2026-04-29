import uuid
from datetime import datetime
from typing import Any, Dict, List

from .ahp_config import DEFAULT_TYPE_WEIGHTS
from .professor_aggregator import ProfessorAggregator
from .ranker import ProfessorRanker
from .retriever import HybridRetriever
from .result_cache import save_search_result
from .settings import OPENAI_API_KEY, RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD


VALID_DOC_TYPES = ("patent", "article", "project")


def normalize_doc_types(doc_types: List[str] | None) -> List[str]:
    if not doc_types:
        return list(VALID_DOC_TYPES)
    return [doc_type for doc_type in doc_types if doc_type in VALID_DOC_TYPES]


def normalize_keywords_if_duplicate_query(keywords: Dict[str, Any], query: str) -> Dict[str, List[str]]:
    high = list(keywords.get("high_level") or [])
    low = list(keywords.get("low_level") or [])
    if not query or (len(high) != 1 or len(low) != 1):
        return {"high_level": high, "low_level": low}
    if high[0] != query or low[0] != query:
        return {"high_level": high, "low_level": low}
    return {"high_level": [query], "low_level": [query]}


def convert_rag_results(raw_results: Dict[str, Any]) -> Dict[str, Any]:
    docs_dict: Dict[tuple[str, str], Dict[str, Any]] = {}

    for result in raw_results.get("merged_results", []):
        doc_no = str(result.get("metadata", {}).get("source_doc_id", ""))
        if not doc_no:
            continue

        doc_type = result.get("doc_type", "unknown")
        metadata = result.get("metadata", {})

        if metadata.get("name") is not None:
            match_info = {
                "search_type": "local",
                "similarity": result.get("similarity", 0),
                "matched_entity": {
                    "name": metadata.get("name", ""),
                    "entity_type": metadata.get("entity_type", ""),
                    "description": result.get("document", ""),
                },
                "neighbors_1hop": [
                    {
                        "name": neighbor.get("name", ""),
                        "entity_type": neighbor.get("entity_type", ""),
                        "relation_keywords": neighbor.get("relation_keywords", []),
                        "relation_description": neighbor.get("relation_description", ""),
                    }
                    for neighbor in result.get("neighbors", [])
                ],
            }
        else:
            match_info = {
                "search_type": "global",
                "similarity": result.get("similarity", 0),
                "matched_relation": {
                    "source_entity": metadata.get("source_entity", ""),
                    "target_entity": metadata.get("target_entity", ""),
                    "keywords": metadata.get("keywords", ""),
                    "description": result.get("document", ""),
                },
                "source_entity_info": result.get("source_entity_info"),
                "target_entity_info": result.get("target_entity_info"),
            }

        docs_dict[(doc_no, doc_type)] = {
            "no": doc_no,
            "data_type": doc_type,
            "matches": [match_info],
        }

    retrieved_docs = sorted(
        docs_dict.values(),
        key=lambda document: document["matches"][0].get("similarity", 0) if document.get("matches") else 0,
        reverse=True,
    )

    return {
        "query": raw_results.get("query", ""),
        "keywords": {
            "high_level": raw_results.get("high_level_keywords", []),
            "low_level": raw_results.get("low_level_keywords", []),
        },
        "retrieved_docs": retrieved_docs,
    }


def recommend_professors(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = str(payload.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Add it to the .env file.")

    doc_types = normalize_doc_types(payload.get("doc_types"))
    retrieval_top_k = payload.get("retrieval_top_k") or RETRIEVAL_TOP_K
    similarity_threshold = payload.get("similarity_threshold") or SIMILARITY_THRESHOLD

    retriever = HybridRetriever(
        doc_types=doc_types,
        force_api=bool(payload.get("force_api", False)),
    )
    raw_rag_results = retriever.retrieve(
        query=query,
        retrieval_top_k=retrieval_top_k,
        similarity_threshold=similarity_threshold,
        mode="hybrid",
    )

    rag_results = convert_rag_results(raw_rag_results)
    professor_data = ProfessorAggregator().aggregate_by_professor(rag_results=rag_results, doc_types=doc_types)
    ranked_professors = ProfessorRanker().rank_professors(professor_data, DEFAULT_TYPE_WEIGHTS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_keywords = {
        "high_level": raw_rag_results.get("high_level_keywords", []),
        "low_level": raw_rag_results.get("low_level_keywords", []),
    }
    ahp_results = {
        "query": query,
        "keywords": normalize_keywords_if_duplicate_query(raw_keywords, query),
        "timestamp": timestamp,
        "total_professors": len(ranked_professors),
        "type_weights": DEFAULT_TYPE_WEIGHTS,
        "ranked_professors": ranked_professors,
    }

    result = {
        "search_id": f"search_{timestamp}_{uuid.uuid4().hex[:8]}",
        "query": query,
        "doc_types": doc_types,
        "rag_results": rag_results,
        "ahp_results": ahp_results,
    }
    save_search_result(result)
    return result
