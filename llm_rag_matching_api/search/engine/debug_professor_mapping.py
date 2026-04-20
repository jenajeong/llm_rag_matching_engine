import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .professor_aggregator import ProfessorAggregator
from .ranker import ProfessorRanker


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def summarize_match(document: Dict[str, Any]) -> str:
    matches = document.get("matches") or []
    if not matches:
        return "no matches"

    match = matches[0]
    if match.get("search_type") == "local":
        entity = match.get("matched_entity") or {}
        return f"local:{entity.get('name', '')}"

    relation = match.get("matched_relation") or {}
    return f"global:{relation.get('source_entity', '')}->{relation.get('target_entity', '')}"


def summarize_documents(professor_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for professor_id, data in professor_data.items():
        professor_info = data.get("professor_info") or {}
        documents = data.get("documents") or {}
        summary.append(
            {
                "professor_id": professor_id,
                "name": professor_info.get("NM"),
                "emp_no": professor_info.get("EMP_NO"),
                "sq": professor_info.get("SQ"),
                "document_counts": {
                    doc_type: len(documents.get(doc_type, []))
                    for doc_type in ("article", "patent", "project")
                },
            }
        )

    summary.sort(
        key=lambda item: (
            item["document_counts"]["article"]
            + item["document_counts"]["patent"]
            + item["document_counts"]["project"]
        ),
        reverse=True,
    )
    return summary


def unresolved_documents(
    rag_results: Dict[str, Any],
    aggregator: ProfessorAggregator,
) -> List[Dict[str, Any]]:
    unresolved = []

    for document in rag_results.get("retrieved_docs", []):
        doc_type = document.get("data_type", "")
        raw_doc_id = str(document.get("no", ""))
        resolved_docs = aggregator._load_original_documents(doc_type, raw_doc_id)
        if resolved_docs:
            continue

        unresolved.append(
            {
                "doc_type": doc_type,
                "raw_doc_id": raw_doc_id,
                "match": summarize_match(document),
            }
        )

        if len(unresolved) >= 10:
            break

    return unresolved


def debug_mapping(response_path: Path) -> Dict[str, Any]:
    response = load_json(response_path)
    rag_results = response.get("rag_results", {})

    aggregator = ProfessorAggregator()
    professor_data = aggregator.aggregate_by_professor(rag_results)
    ranked_professors = ProfessorRanker(aggregator=aggregator).rank_professors(professor_data)

    return {
        "response_path": str(response_path),
        "retrieved_docs": len(rag_results.get("retrieved_docs", [])),
        "resolved_professors": len(professor_data),
        "ranked_professors": len(ranked_professors),
        "top_professors": summarize_documents(professor_data)[:10],
        "top_ranked": [
            {
                "rank": professor.get("rank"),
                "name": (professor.get("professor_info") or {}).get("NM"),
                "emp_no": (professor.get("professor_info") or {}).get("EMP_NO"),
                "total_score": professor.get("total_score"),
                "scores_by_type": professor.get("scores_by_type"),
            }
            for professor in ranked_professors[:10]
        ],
        "unresolved_examples": unresolved_documents(rag_results, aggregator),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug professor aggregation without calling the API")
    parser.add_argument("response_json", type=Path, help="Path to a saved API response JSON")
    args = parser.parse_args()

    summary = debug_mapping(args.response_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
