import hashlib
import re
from typing import Any, Dict


def normalize_key_part(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_register_number(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[^0-9A-Za-z]", "", text).upper()


def stable_hash(*parts: Any) -> str:
    joined = "|".join(normalize_key_part(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def patent_source_key(record: Dict) -> str:
    register_number = (
        record.get("ptnt_rgstr_id_clean")
        or record.get("ptnt_rgstr_id")
        or record.get("kipris_register_number")
        or record.get("kipris_application_number")
    )
    normalized = normalize_register_number(register_number)
    if normalized:
        return f"patent:{normalized}"
    return f"patent:hash:{stable_hash(record.get('kipris_application_name'), record.get('mbr_sn'))}"


def article_source_key(record: Dict) -> str:
    emp_no = record.get("EMP_NO") or record.get("emp_no")
    title = record.get("THSS_NM") or record.get("title")
    return f"article:{stable_hash(emp_no, title)}"


def project_source_key(record: Dict) -> str:
    project_name = record.get("PRJ_NM") or record.get("title")
    start_date = record.get("RCH_ST_DT") or record.get("year")
    professor_id = (
        record.get("PRJ_RSPR_EMP_ID")
        or (record.get("professor_info") or {}).get("EMP_NO")
        or (record.get("professor_info") or {}).get("SQ")
    )
    return f"project:{stable_hash(project_name, start_date, professor_id)}"


def source_key_for_doc_type(doc_type: str, record: Dict) -> str:
    if doc_type == "patent":
        return patent_source_key(record)
    if doc_type == "article":
        return article_source_key(record)
    if doc_type == "project":
        return project_source_key(record)
    raise ValueError(f"Unknown doc_type: {doc_type}")
