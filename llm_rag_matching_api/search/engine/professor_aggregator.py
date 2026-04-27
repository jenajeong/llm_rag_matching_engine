import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from .settings import (
    DATA_TRAIN_ARTICLE_FILE,
    DATA_TRAIN_PATENT_FILE,
    DATA_TRAIN_PROJECT_FILE,
)


class ProfessorAggregator:
    def __init__(self):
        self._data_cache = {
            "patent": None,
            "article": None,
            "project": None,
        }
        self._index_cache = {
            "patent": None,
            "article": None,
            "project": None,
        }

    def aggregate_by_professor(self, rag_results: Dict[str, Any], doc_types: list[str] | None = None) -> Dict[str, Dict[str, Any]]:
        if doc_types is None:
            doc_types = ["patent", "article", "project"]

        professor_data = defaultdict(
            lambda: {
                "professor_info": None,
                "documents": {"patent": [], "article": [], "project": []},
            }
        )

        for document in rag_results.get("retrieved_docs", []):
            doc_no = str(document.get("no", ""))
            doc_type = document.get("data_type", "")
            if not doc_no or doc_type not in doc_types:
                continue

            original_documents = self._load_original_documents(doc_type, doc_no)
            for original_doc in original_documents:
                professor_info = self._extract_professor_info(original_doc)
                if not professor_info:
                    continue

                raw_id = professor_info.get("SQ") or professor_info.get("EMP_NO")
                professor_id = self._normalize_professor_id(raw_id)
                if not professor_id:
                    continue

                if professor_data[professor_id]["professor_info"] is None:
                    professor_data[professor_id]["professor_info"] = professor_info

                professor_data[professor_id]["documents"][doc_type].append(original_doc)

        return self._merge_same_professor(dict(professor_data))

    def _normalize_professor_id(self, raw_id: Any) -> str:
        if raw_id is None:
            return ""

        value = str(raw_id).strip()
        if not value:
            return ""

        try:
            number = float(value)
        except (TypeError, ValueError):
            return value

        return str(int(number)) if number == int(number) else value

    def _merge_same_professor(self, professor_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not professor_data:
            return professor_data

        parent: Dict[str, str] = {}

        def find(item: str) -> str:
            if item not in parent:
                parent[item] = item
            if parent[item] != item:
                parent[item] = find(parent[item])
            return parent[item]

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for professor_id, data in professor_data.items():
            info = data.get("professor_info") or {}
            sq = self._normalize_professor_id(info.get("SQ"))
            emp_no = self._normalize_professor_id(info.get("EMP_NO"))

            find(professor_id)
            if sq:
                find(sq)
                union(professor_id, sq)
            if emp_no:
                find(emp_no)
                union(professor_id, emp_no)

        merged = defaultdict(
            lambda: {
                "professor_info": None,
                "documents": {"patent": [], "article": [], "project": []},
            }
        )

        for professor_id, data in professor_data.items():
            canonical = find(professor_id)
            if merged[canonical]["professor_info"] is None:
                merged[canonical]["professor_info"] = data["professor_info"]
            for doc_type in ("patent", "article", "project"):
                merged[canonical]["documents"][doc_type].extend(data["documents"].get(doc_type, []))

        return dict(merged)

    def _extract_professor_info(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        professor_info = document.get("professor_info")
        if not professor_info:
            return None
        if not (professor_info.get("SQ") or professor_info.get("EMP_NO")):
            return None
        return professor_info

    def _split_document_ids(self, raw_doc_id: str) -> list[str]:
        return [part.strip() for part in str(raw_doc_id).split(",") if part.strip()]

    def _load_original_documents(self, doc_type: str, raw_doc_id: str) -> list[Dict[str, Any]]:
        file_paths = {
            "patent": DATA_TRAIN_PATENT_FILE,
            "article": DATA_TRAIN_ARTICLE_FILE,
            "project": DATA_TRAIN_PROJECT_FILE,
        }

        file_path = Path(file_paths.get(doc_type, ""))
        if not file_path.exists():
            return []

        if self._data_cache[doc_type] is None:
            with open(file_path, "r", encoding="utf-8") as file:
                self._data_cache[doc_type] = json.load(file)
            self._index_cache[doc_type] = {
                str(item.get("no")): item
                for item in self._data_cache[doc_type]
                if item.get("no") is not None
            }

        index = self._index_cache[doc_type] or {}
        resolved_documents = []
        seen_doc_ids = set()

        for doc_id in self._split_document_ids(raw_doc_id):
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            document = index.get(doc_id)
            if document:
                resolved_documents.append(document)

        return resolved_documents
