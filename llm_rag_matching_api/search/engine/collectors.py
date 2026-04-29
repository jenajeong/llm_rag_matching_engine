import json
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .indigo_database import (
    get_article_candidates,
    get_db_connection,
    get_original_patent_data,
    get_patent_register_ids,
    get_project_rows_with_professor,
)
from .json_store import append_new_records, existing_source_keys, load_json_list
from .settings import (
    ARTICLE_DATA_FILE,
    KIPRIS_API_KEY,
    PATENT_DATA_FILE,
    PROJECT_DATA_FILE,
)
from .source_keys import (
    article_source_key,
    patent_source_key,
    project_source_key,
)


class PatentCollector:
    output_file = PATENT_DATA_FILE

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or KIPRIS_API_KEY

    def collect(self, limit: int | None = None, dry_run: bool = True, sleep_seconds: float = 1.0) -> Dict:
        conn = get_db_connection()
        try:
            candidates = get_patent_register_ids(conn, limit=limit)
            existing_keys = existing_source_keys(self.output_file, "patent")
            new_candidates = [
                candidate
                for candidate in candidates
                if patent_source_key(candidate) not in existing_keys
            ]

            stats = {
                "doc_type": "patent",
                "output_file": str(self.output_file),
                "db_candidates": len(candidates),
                "already_collected_skipped": len(candidates) - len(new_candidates),
                "new_candidates": len(new_candidates),
                "dry_run": dry_run,
                "api_success": 0,
                "api_failed": 0,
                "appended": 0,
            }
            if dry_run or not new_candidates:
                return stats

            if not self.api_key:
                raise ValueError("KIPRIS_API_KEY is not set")

            collected = []
            for index, candidate in enumerate(new_candidates):
                register_id = candidate.get("ptnt_rgstr_id", "")
                clean_register_id = candidate.get("ptnt_rgstr_id_clean", "")
                original_data = get_original_patent_data(conn, register_id) or {}
                kipris_data = self.fetch_patent_data(
                    register_id=clean_register_id,
                    mbr_sn=candidate.get("mbr_sn", ""),
                    professor_info=candidate.get("professor_info") or {},
                )
                if kipris_data:
                    merged = {**original_data, **kipris_data}
                    if candidate.get("professor_info"):
                        merged["professor_info"] = candidate["professor_info"]
                    if (merged.get("professor_info") or {}).get("SQ"):
                        collected.append(merged)
                        stats["api_success"] += 1
                    else:
                        stats["api_failed"] += 1
                else:
                    stats["api_failed"] += 1

                if index < len(new_candidates) - 1 and sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            appended, skipped = append_new_records(self.output_file, "patent", collected)
            stats["appended"] = appended
            stats["append_skipped"] = skipped
            return stats
        finally:
            conn.close()

    def fetch_patent_data(self, register_id: str, mbr_sn: str = "", professor_info: Dict | None = None) -> Optional[Dict]:
        encoded_register_id = urllib.parse.quote(register_id, safe="")
        url = (
            "https://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getAdvancedSearch"
            f"?registerNumber={encoded_register_id}"
            f"&ServiceKey={self.api_key}"
            "&numOfRows=10"
            "&pageNo=1"
        )
        response = requests.get(url, timeout=30)
        if response.text.strip().startswith(("<!DOCTYPE", "<html")):
            return None

        root = ET.fromstring(response.content)
        success_yn = root.findtext(".//successYN", default="")
        result_msg = root.findtext(".//resultMsg", default="")
        result_code = root.findtext(".//resultCode", default="")
        if success_yn == "N" or (result_msg and "ERROR" in result_msg.upper()):
            if result_code in {"20", "21", "22"} or "LIMIT" in result_msg.upper() or "QUOTA" in result_msg.upper():
                raise RuntimeError(f"KIPRIS quota or rate limit reached: {result_msg}")
            return None

        item = next(iter(root.findall(".//item")), None)
        if item is None:
            return None

        data = {
            "ptnt_rgstr_id": register_id,
            "mbr_sn": mbr_sn,
            "kipris_index_no": item.findtext("indexNo", default=""),
            "kipris_register_status": item.findtext("registerStatus", default=""),
            "kipris_register_number": item.findtext("registerNumber", default=""),
            "kipris_application_date": item.findtext("applicationDate", default=""),
            "kipris_abstract": item.findtext("astrtCont", default="").strip(),
            "kipris_application_name": item.findtext("inventionTitle", default=""),
            "kipris_application_number": item.findtext("applicationNumber", default=""),
            "kipris_total_count": root.findtext(".//totalCount", default="0"),
        }
        if professor_info:
            data["professor_info"] = professor_info
        return data


class ArticleCollector:
    output_file = ARTICLE_DATA_FILE

    def collect_candidates(self, limit: int | None = None, dry_run: bool = True, min_year: int = 2015) -> Dict:
        conn = get_db_connection()
        try:
            candidates = get_article_candidates(conn, min_year=min_year, limit=limit)
            existing_keys = existing_source_keys(self.output_file, "article")
            new_candidates = [
                candidate
                for candidate in candidates
                if article_source_key(candidate) not in existing_keys
            ]
            return {
                "doc_type": "article",
                "output_file": str(self.output_file),
                "db_candidates": len(candidates),
                "already_collected_skipped": len(candidates) - len(new_candidates),
                "new_candidates": len(new_candidates),
                "dry_run": dry_run,
                "note": "Article DB candidate collection is safe. EBSCO crawling is intentionally not automatic.",
            }
        finally:
            conn.close()


class ProjectCollector:
    output_file = PROJECT_DATA_FILE

    def __init__(self, json_file1: str | None = None, json_file2: str | None = None):
        self.json_file1 = Path(json_file1) if json_file1 else PROJECT_DATA_FILE.parent / "project_source_1.json"
        self.json_file2 = Path(json_file2) if json_file2 else PROJECT_DATA_FILE.parent / "project_source_2.json"

    def collect(self, limit: int | None = None, dry_run: bool = True) -> Dict:
        conn = get_db_connection()
        try:
            db_rows = get_project_rows_with_professor(conn, limit=limit)
            source_rows = self._load_source_rows()
            merged = self._merge_rows(db_rows, source_rows)
            existing_keys = existing_source_keys(self.output_file, "project")
            new_records = [
                record
                for record in merged
                if project_source_key(record) not in existing_keys
            ]

            stats = {
                "doc_type": "project",
                "output_file": str(self.output_file),
                "db_candidates": len(db_rows),
                "source_rows": len(source_rows),
                "merged_records": len(merged),
                "already_collected_skipped": len(merged) - len(new_records),
                "new_candidates": len(new_records),
                "dry_run": dry_run,
                "appended": 0,
            }
            if dry_run or not new_records:
                return stats

            appended, skipped = append_new_records(self.output_file, "project", new_records)
            stats["appended"] = appended
            stats["append_skipped"] = skipped
            return stats
        finally:
            conn.close()

    def _load_source_rows(self) -> List[Dict]:
        rows = []
        for path in [self.json_file1, self.json_file2]:
            rows.extend(load_json_list(path))
        return rows

    def _merge_rows(self, db_rows: List[Dict], source_rows: List[Dict]) -> List[Dict]:
        source_index = {}
        for source in source_rows:
            project_name = self._first_value(source, ["과제명(국문)", "excel_과제명(국문)", "PRJ_NM", "project_name"])
            year = self._first_value(source, ["기준년도", "excel_기준년도", "year"])
            if project_name:
                source_index.setdefault(self._project_key(project_name, year), source)

        merged = []
        for db_row in db_rows:
            professor_info = db_row.get("professor_info") or {}
            if not professor_info.get("SQ"):
                continue

            project_name = db_row.get("PRJ_NM")
            year = self._year_from_db_row(db_row)
            source = source_index.get(self._project_key(project_name, year)) or source_index.get(self._project_key(project_name, None))
            if not source:
                continue

            merged_record = dict(db_row)
            for key, value in source.items():
                merged_record[f"excel_{key}"] = value
            merged_record["professor_info"] = professor_info
            merged.append(merged_record)

        return merged

    def _project_key(self, project_name, year) -> str:
        name = "" if project_name is None else str(project_name).strip().lower()
        year_text = "" if year is None else str(year).strip()
        return f"{name}|{year_text}"

    def _year_from_db_row(self, row: Dict):
        date_value = row.get("RCH_ST_DT")
        if not date_value:
            return None
        text = str(date_value)
        return text[:4] if len(text) >= 4 and text[:4].isdigit() else None

    def _first_value(self, row: Dict, keys: List[str]):
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return value
        return None
