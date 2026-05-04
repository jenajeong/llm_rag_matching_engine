import hashlib
import re
from collections import Counter
from typing import Any

from ..core.safe import as_dict, as_text, clean_ws, get_nested, is_nullish
from ..core.types import ProcessedDocument


class TextProcessor:
    MIN_TEXT_LENGTH = 10  # title만 있는 경우를 위해 기존 100 → 10으로 완화

    def __init__(self, min_text_length: int | None = None):
        self.min_text_length = min_text_length or self.MIN_TEXT_LENGTH
        self.stats: Counter[str] = Counter()

    def get_stats(self) -> dict[str, int]:
        return dict(self.stats)

    def process(self, doc_type: str, records: list[dict[str, Any]]) -> list[ProcessedDocument]:
        if doc_type == "patent":
            return self.process_patents(records)
        if doc_type == "article":
            return self.process_articles(records)
        if doc_type == "project":
            return self.process_projects(records)
        raise ValueError(f"Unknown doc_type: {doc_type}")

    def process_patents(self, records: list[dict[str, Any]]) -> list[ProcessedDocument]:
        self._reset()
        docs: list[ProcessedDocument] = []
        for record in records:
            self.stats["total"] += 1
            metadata = as_dict(record.get("metadata"))
            doc_id = self._doc_id(record, "patent")
            title = self._title(record)
            abstract = self._body(record)

            # title도 없고 text도 없으면 스킵
            if not title and not abstract:
                self.stats["skipped_empty_text"] += 1
                continue

            # text가 없으면 title만으로 구성
            if not abstract:
                text = self._join_labeled(("특허명", title))
                self.stats["title_only"] += 1
            elif self._starts_with_title(abstract, title):
                text = f"[요약] {abstract}"
            else:
                text = self._join_labeled(("특허명", title), ("요약", abstract))

            self._append_if_valid(docs, doc_id, "patent", text, {
                "title": title,
                "register_status": clean_ws(get_nested(metadata, "kipris_register_status")),
                "application_date": clean_ws(get_nested(metadata, "kipris_application_date")),
                "raw_metadata": metadata,
            })
        return docs

    def process_articles(self, records: list[dict[str, Any]]) -> list[ProcessedDocument]:
        self._reset()
        docs: list[ProcessedDocument] = []
        for record in records:
            self.stats["total"] += 1
            doc_id = self._doc_id(record, "article")
            title = self._title(record)
            abstract = self._body(record)

            # title도 없고 text도 없으면 스킵
            if not title and not abstract:
                self.stats["skipped_empty_text"] += 1
                continue

            # text가 없으면 title만으로 구성
            if not abstract:
                text = self._join_labeled(("논문명", title))
                self.stats["title_only"] += 1
            else:
                text = self._join_labeled(("논문명", title), ("초록", abstract))

            self._append_if_valid(docs, doc_id, "article", text, {
                "title": title,
                "raw_metadata": as_dict(record.get("metadata")),
            })
        return docs

    def process_projects(self, records: list[dict[str, Any]]) -> list[ProcessedDocument]:
        self._reset()
        docs: list[ProcessedDocument] = []
        for record in records:
            self.stats["total"] += 1
            doc_id = self._doc_id(record, "project")
            title = self._title(record)
            content = self._body(record)

            # title도 없고 text도 없으면 스킵
            if not title and not content:
                self.stats["skipped_empty_text"] += 1
                continue

            # text가 없으면 title만으로 구성
            if not content:
                text = self._join_labeled(("과제명", title))
                self.stats["title_only"] += 1
            else:
                text = self._join_labeled(("과제명", title), ("내용", content))

            self._append_if_valid(docs, doc_id, "project", text, {
                "title": title,
                "raw_metadata": as_dict(record.get("metadata")),
            })
        return docs

    def _reset(self) -> None:
        self.stats = Counter()

    def _append_if_valid(
        self,
        docs: list[ProcessedDocument],
        doc_id: str,
        doc_type: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        text = clean_ws(text)
        if len(text) <= self.min_text_length:
            self.stats["skipped_short"] += 1
            return
        docs.append(ProcessedDocument(doc_id=doc_id, doc_type=doc_type, text=text, metadata=metadata))
        self.stats["processed"] += 1

    def _doc_id(self, record: dict[str, Any], doc_type: str) -> str:
        for key in ("no", "id", "doc_id", "source_id", "SQ", "APPL_NO"):
            value = record.get(key)
            if not is_nullish(value):
                return as_text(value)
        seed = "|".join([doc_type, self._title(record), self._body(record)[:200]])
        return f"{doc_type}_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]}"

    def _title(self, record: dict[str, Any]) -> str:
        for key in ("title", "THSS_NM", "INVENTION_TITLE", "KOR_INVENTION_TITLE", "SBJT_NM", "name"):
            value = record.get(key)
            if not is_nullish(value):
                return clean_ws(value)
        return ""

    def _body(self, record: dict[str, Any]) -> str:
        for key in ("text", "abstract", "summary", "content", "description", "ABSTRACT", "RSCH_GOAL", "RSCH_CONT"):
            value = record.get(key)
            if not is_nullish(value):
                return clean_ws(value)
        pieces = []
        for value in record.values():
            if isinstance(value, str) and len(value.strip()) >= 20:
                pieces.append(value)
        return clean_ws(" ".join(pieces))

    def _join_labeled(self, *pairs: tuple[str, str]) -> str:
        return "\n".join(f"[{label}] {value}" for label, value in pairs if clean_ws(value))

    def _starts_with_title(self, body: str, title: str) -> bool:
        if not title:
            return False
        return self._norm_for_compare(body).startswith(self._norm_for_compare(title))

    def _norm_for_compare(self, value: str) -> str:
        value = re.sub(r"\s+", "", as_text(value))
        value = re.sub(r"[^\w가-힣]", "", value)
        return value.lower()