import base64
import html
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from search.engine.cost_tracker import log_chat_usage
from search.engine.service import normalize_keywords_if_duplicate_query, recommend_professors
from search.engine.settings import (
    LLM_MODEL,
    OPENAI_API_KEY,
    REPORT_MAX_TOKENS,
    REPORT_RESULTS_DIR,
    REPORT_SUMMARY_MAX_CHARS,
)


DOC_TYPE_LABELS = {
    "article": "논문",
    "patent": "특허",
    "project": "연구 과제",
}


def _clean_inline(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _truncate(value: str, max_chars: int) -> str:
    value = _clean_inline(value)
    return value[:max_chars] + "..." if len(value) > max_chars else value


def _escape_html(value: str) -> str:
    return html.escape(value or "", quote=True)


class ReportGenerationError(ValueError):
    pass


class ReportGenerator:
    def __init__(self, output_dir: Optional[str | Path] = None, api_key: Optional[str] = None):
        self.output_dir = Path(output_dir or REPORT_RESULTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ReportGenerationError("OPENAI_API_KEY is not set. Add it to the .env file.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = LLM_MODEL

    def generate_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        recommendation = self._resolve_recommendation(payload)
        ahp_results = recommendation.get("ahp_results") or payload.get("ahp_results")
        rag_results = recommendation.get("rag_results") or payload.get("rag_results")

        if not isinstance(ahp_results, dict):
            raise ReportGenerationError("ahp_results is required. Pass the full recommendation response or ahp_results/rag_results.")

        timestamp = ahp_results.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
        input_data = self._prepare_input_json(ahp_results)
        report_text = self._generate_report_text(input_data)
        report_html = markdown_to_html(report_text)
        pdf_path = self.save_pdf(report_html, timestamp=timestamp)

        return {
            "report_id": f"report_{timestamp}_{uuid.uuid4().hex[:8]}",
            "search_id": recommendation.get("search_id") or payload.get("search_id"),
            "query": ahp_results.get("query", ""),
            "timestamp": timestamp,
            "report_text": report_text,
            "report_html": report_html,
            "pdf_path": str(pdf_path),
            "pdf_filename": pdf_path.name,
            "input_data": input_data,
            "rag_results": rag_results,
            "ahp_results": ahp_results,
            "model": self.model,
        }

    def _resolve_recommendation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(payload.get("recommendation"), dict):
            return payload["recommendation"]
        if isinstance(payload.get("search_result"), dict):
            return payload["search_result"]
        if isinstance(payload.get("ahp_results"), dict):
            return payload
        if payload.get("query"):
            return recommend_professors(payload)
        raise ReportGenerationError("Pass the first endpoint response, or provide query to run recommendation first.")

    def _prepare_input_json(self, ahp_results: Dict[str, Any]) -> Dict[str, Any]:
        query = ahp_results.get("query", "")
        keywords = normalize_keywords_if_duplicate_query(ahp_results.get("keywords", {}), query)
        professors = []

        for index, professor in enumerate(ahp_results.get("ranked_professors", [])[:3], 1):
            info = professor.get("professor_info") or {}
            documents = professor.get("documents") or {}
            document_scores = professor.get("document_scores") or {}
            professor_documents = []

            for doc_type in ("article", "patent", "project"):
                docs = documents.get(doc_type) or []
                scores = {
                    str(score.get("no", "")): score.get("score", 0.0)
                    for score in (document_scores.get(doc_type) or [])
                }
                docs_with_scores = [(doc, scores.get(str(doc.get("no", "")), 0.0)) for doc in docs]
                docs_with_scores.sort(key=lambda item: item[1], reverse=True)

                for doc, _score in docs_with_scores[:3]:
                    professor_documents.append(
                        {
                            "type": doc_type,
                            "type_ko": DOC_TYPE_LABELS.get(doc_type, doc_type),
                            "title": _clean_inline(doc.get("title", "")),
                            "summary": _truncate(doc.get("text", ""), REPORT_SUMMARY_MAX_CHARS),
                            "year": doc.get("year", ""),
                        }
                    )

            professors.append(
                {
                    "number": index,
                    "name": info.get("NM", ""),
                    "department": _clean_inline(f"{info.get('COLG_NM', '')} {info.get('HG_NM', '')}"),
                    "contact": info.get("EMAIL", "") or "-",
                    "documents": professor_documents,
                }
            )

        return {
            "query": query,
            "keywords": {
                "high_level": keywords.get("high_level", []),
                "low_level": keywords.get("low_level", []),
            },
            "professors": professors,
        }

    def _generate_report_text(self, input_data: Dict[str, Any]) -> str:
        prompt = build_report_prompt(input_data)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "너는 문서 근거 기반 교수 추천 보고서를 작성하는 한국어 보고서 전문가다.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=REPORT_MAX_TOKENS,
        )
        log_chat_usage(component="report_generation", model=self.model, response=response)
        return (response.choices[0].message.content or "").strip()

    def save_pdf(self, report_html: str, timestamp: Optional[str] = None) -> Path:
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = self.output_dir / f"report_{timestamp}.pdf"
        html_document = build_pdf_html(report_html)

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise ReportGenerationError("playwright is required for PDF generation. Install playwright and chromium.") from error

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 794, "height": 1123})
            page.set_content(html_document, wait_until="load")
            page.emulate_media(media="print")
            page.pdf(
                path=str(pdf_path),
                format="A4",
                margin={"top": "18mm", "right": "18mm", "bottom": "18mm", "left": "18mm"},
                print_background=True,
            )
            browser.close()

        return pdf_path


def build_report_prompt(input_data: Dict[str, Any]) -> str:
    return f"""
아래 JSON만 근거로 공식 교수 추천 보고서를 작성해줘.

작성 규칙:
- 한국어로 작성한다.
- AHP 점수, 총점, 내부 계산값은 표시하지 않는다.
- 교수는 input JSON의 professors 배열 순서대로 1, 2, 3번만 표시한다.
- 각 교수 블록에는 소속, 이메일, 사용자가 검색한 내용과 관련 있는 문서 근거를 포함한다.
- 문서 유형은 [논문], [특허], [연구 과제] 순서로 묶는다.
- 없는 문서 유형은 출력하지 않는다.
- 문서마다 제목, 연도, 2~3문장의 관련성 설명을 쓴다.
- 마크다운으로 출력한다.

출력 형식:
# AI 기반 검색 결과

**사용자 검색어:** ...

---

### 추천 교수 및 관련 정보

##### **1.** **교수명 교수**
**소속:** ...
**이메일:** ...

**사용자 검색어 관련 자료**
**[논문]**
  **문서 제목** (2024)
  - 관련성 설명

---

### 유의사항 및 문의 안내

- 추천 결과는 시스템에 등록된 문서와 검색 조건을 기반으로 제공됩니다.
- 실제 협업 가능 여부는 담당 부서 확인이 필요합니다.

입력 JSON:
{json.dumps(input_data, ensure_ascii=False, indent=2)}
""".strip()


def markdown_to_html(markdown_text: str) -> str:
    try:
        import markdown as markdown_lib

        return markdown_lib.markdown(markdown_text, extensions=["extra", "nl2br"])
    except ImportError:
        escaped = _escape_html(markdown_text)
        return "<pre>" + escaped + "</pre>"


def build_pdf_html(body_html: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
    color: #1e3a5f;
    font-size: 13px;
    line-height: 1.72;
    margin: 0;
    word-break: keep-all;
    overflow-wrap: break-word;
  }}
  h1 {{ font-size: 22px; margin: 0 0 14px; }}
  h3 {{ font-size: 17px; margin: 18px 0 10px; }}
  h5 {{ font-size: 15px; margin: 16px 0 6px; }}
  p {{ margin: 6px 0; }}
  ul {{ padding-left: 18px; }}
  li {{ margin: 4px 0; }}
  strong {{ font-weight: 700; }}
  hr {{ border: none; border-top: 1px solid #c8d3df; margin: 16px 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ border: 1px solid #ccd6e0; padding: 6px 8px; }}
  th {{ background: #e8eef4; }}
</style>
</head>
<body>
{body_html}
</body>
</html>"""


def encode_pdf_base64(pdf_path: str | Path) -> str:
    return base64.b64encode(Path(pdf_path).read_bytes()).decode("ascii")
