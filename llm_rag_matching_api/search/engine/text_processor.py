import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProcessedDocument:
    doc_id: str
    doc_type: str
    text: str
    metadata: Dict


class TextProcessor:
    MIN_TEXT_LENGTH = 100

    def __init__(self):
        self.stats = {
            "total": 0,
            "processed": 0,
            "skipped_empty": 0,
            "skipped_short": 0,
        }

    def process_patents(self, patent_data: List[Dict]) -> List[ProcessedDocument]:
        self._reset_stats()
        processed_docs = []

        for patent in patent_data:
            self.stats["total"] += 1
            doc_id = str(patent.get("no", "")).strip()
            title = str(patent.get("title", "")).strip()
            abstract = str(patent.get("text", "")).strip()
            metadata_source = patent.get("metadata", {}) or {}

            if not doc_id or not abstract:
                self.stats["skipped_empty"] += 1
                continue

            if self._starts_with_title(abstract, title):
                text = f"[Abstract] {abstract}"
                if len(abstract) <= self.MIN_TEXT_LENGTH:
                    self.stats["skipped_short"] += 1
                    continue
            else:
                text = f"[Patent Title] {title}\n[Abstract] {abstract}"

            text = self._clean_text(text)
            processed_docs.append(ProcessedDocument(
                doc_id=doc_id,
                doc_type="patent",
                text=text,
                metadata={
                    "register_status": metadata_source.get("kipris_register_status", ""),
                    "application_date": metadata_source.get("kipris_application_date", ""),
                    "title": title,
                },
            ))
            self.stats["processed"] += 1

        return processed_docs

    def process_articles(self, article_data: List[Dict]) -> List[ProcessedDocument]:
        self._reset_stats()
        processed_docs = []

        for article in article_data:
            self.stats["total"] += 1
            doc_id = str(article.get("no", "")).strip()
            title = str(article.get("title", "")).strip()
            abstract = str(article.get("text", "")).strip()

            if not doc_id or not abstract:
                self.stats["skipped_empty"] += 1
                continue
            if len(abstract) <= self.MIN_TEXT_LENGTH:
                self.stats["skipped_short"] += 1
                continue

            text = self._clean_text(f"[Article Title] {title}\n[Abstract] {abstract}")
            processed_docs.append(ProcessedDocument(
                doc_id=doc_id,
                doc_type="article",
                text=text,
                metadata={"title": title},
            ))
            self.stats["processed"] += 1

        return processed_docs

    def process_projects(self, project_data: List[Dict]) -> List[ProcessedDocument]:
        self._reset_stats()
        processed_docs = []

        for project in project_data:
            self.stats["total"] += 1
            doc_id = str(project.get("no", "")).strip()
            title = str(project.get("title", "")).strip()
            content = str(project.get("text", "")).strip()

            if not doc_id or not content:
                self.stats["skipped_empty"] += 1
                continue

            text = self._clean_text(f"[Project Title] {title}\n[Content] {content}")
            if len(text) <= self.MIN_TEXT_LENGTH:
                self.stats["skipped_short"] += 1
                continue

            processed_docs.append(ProcessedDocument(
                doc_id=doc_id,
                doc_type="project",
                text=text,
                metadata={"title": title},
            ))
            self.stats["processed"] += 1

        return processed_docs

    def process(self, doc_type: str, raw_data: List[Dict]) -> List[ProcessedDocument]:
        if doc_type == "patent":
            return self.process_patents(raw_data)
        if doc_type == "article":
            return self.process_articles(raw_data)
        if doc_type == "project":
            return self.process_projects(raw_data)
        raise ValueError(f"Unknown doc_type: {doc_type}")

    def get_stats(self) -> Dict:
        return self.stats

    def _reset_stats(self):
        self.stats = {
            "total": 0,
            "processed": 0,
            "skipped_empty": 0,
            "skipped_short": 0,
        }

    def _starts_with_title(self, abstract: str, title: str) -> bool:
        if not title:
            return False
        return self._normalize_for_comparison(abstract).startswith(
            self._normalize_for_comparison(title)
        )

    def _normalize_for_comparison(self, text: str) -> str:
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^\w가-힣]", "", text)
        return text.lower()

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
