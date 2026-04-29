import ast
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .json_store import load_json_list, save_json_list
from .settings import (
    ARTICLE_DATA_FILE,
    DATA_TRAIN_ARTICLE_FILE,
    DATA_TRAIN_PATENT_FILE,
    DATA_TRAIN_PROJECT_FILE,
    PATENT_DATA_FILE,
    PROJECT_DATA_FILE,
)


RAW_FILES = {
    "patent": PATENT_DATA_FILE,
    "article": ARTICLE_DATA_FILE,
    "project": PROJECT_DATA_FILE,
}

TRAIN_FILES = {
    "patent": DATA_TRAIN_PATENT_FILE,
    "article": DATA_TRAIN_ARTICLE_FILE,
    "project": DATA_TRAIN_PROJECT_FILE,
}


def filter_raw_data(doc_type: str, dry_run: bool = True) -> Dict:
    raw_data = load_json_list(RAW_FILES[doc_type])
    if doc_type == "patent":
        filtered, filter_stats = filter_patent_data(raw_data)
    elif doc_type == "article":
        filtered, filter_stats = filter_article_data(raw_data)
    elif doc_type == "project":
        filtered, filter_stats = filter_project_data(raw_data)
    else:
        raise ValueError(f"Unknown doc_type: {doc_type}")

    stats = {
        "doc_type": doc_type,
        "raw_file": str(RAW_FILES[doc_type]),
        "train_file": str(TRAIN_FILES[doc_type]),
        "raw_count": len(raw_data),
        "filtered_count": len(filtered),
        "dry_run": dry_run,
        "filter_stats": filter_stats,
    }

    if not dry_run:
        save_json_list(TRAIN_FILES[doc_type], filtered)
        stats["written"] = len(filtered)
    else:
        stats["written"] = 0

    return stats


def filter_patent_data(patents: List[Dict]) -> Tuple[List[Dict], Dict]:
    filtered = []
    stats = {
        "total": len(patents),
        "null_field_filtered": 0,
        "date_filtered": 0,
        "text_preprocessing_failed": 0,
        "text_preprocessing_passed": 0,
    }
    required_fields = {
        "kipris_application_name",
        "mbr_sn",
        "kipris_abstract",
        "tech_invnt_se",
        "kipris_register_status",
    }

    for patent in patents:
        if any(not has_value(patent.get(field)) for field in required_fields):
            stats["null_field_filtered"] += 1
            continue

        year, parsed = parse_year(patent.get("kipris_application_date"))
        if not parsed or year < 2015:
            stats["date_filtered"] += 1
            continue

        text, is_valid = preprocess_text(patent.get("kipris_abstract"), min_length=100, max_length=5000)
        if not is_valid:
            stats["text_preprocessing_failed"] += 1
            continue

        stats["text_preprocessing_passed"] += 1
        filtered.append({
            "data_type": "patent",
            "no": len(filtered) + 1,
            "text": text,
            "title": patent.get("kipris_application_name"),
            "year": year,
            "professor_info": patent.get("professor_info"),
            "metadata": {
                "kipris_register_status": patent.get("kipris_register_status"),
                "kipris_application_date": patent.get("kipris_application_date"),
                "kipris_register_number": patent.get("kipris_register_number"),
                "kipris_application_number": patent.get("kipris_application_number"),
                "ptnt_rgstr_id": patent.get("ptnt_rgstr_id"),
            },
        })

    return filtered, stats


def filter_article_data(articles: List[Dict]) -> Tuple[List[Dict], Dict]:
    filtered = []
    stats = {
        "total": len(articles),
        "year_filtered": 0,
        "metadata_filtered": 0,
        "abstract_null": 0,
        "text_preprocessing_failed": 0,
        "text_preprocessing_passed": 0,
    }

    for article in articles:
        year = parse_year_value(article.get("YY") or article.get("year") or article.get("PUBLSH_DT"))
        if year is None or year < 2015:
            stats["year_filtered"] += 1
            continue

        metadata = {
            "THSS_PATICP_GBN": article.get("THSS_PATICP_GBN"),
            "JRNL_GBN": article.get("JRNL_GBN"),
        }
        if has_invalid_metadata(metadata):
            stats["metadata_filtered"] += 1
            continue

        abstract = select_article_abstract(article)
        if not abstract:
            stats["abstract_null"] += 1
            continue

        text, is_valid = preprocess_text(abstract, min_length=100, max_length=5000)
        if not is_valid:
            stats["text_preprocessing_failed"] += 1
            continue

        stats["text_preprocessing_passed"] += 1
        filtered.append({
            "data_type": "article",
            "no": len(filtered) + 1,
            "text": text,
            "title": article.get("THSS_NM") or article.get("title"),
            "year": year,
            "professor_info": article.get("professor_info"),
            "metadata": metadata,
        })

    return filtered, stats


def filter_project_data(projects: List[Dict]) -> Tuple[List[Dict], Dict]:
    filtered = []
    stats = {
        "total": len(projects),
        "year_filtered": 0,
        "text_preprocessing_failed": 0,
        "text_preprocessing_passed": 0,
    }

    for project in projects:
        year = parse_project_year(project)
        if year is None or year < 2015:
            stats["year_filtered"] += 1
            continue

        objective = project.get("excel_연구목표요약") or project.get("excel_?곌뎄紐⑺몴?붿빟") or ""
        content = project.get("excel_연구내용요약") or project.get("excel_?곌뎄?댁슜?붿빟") or ""
        summary = " ".join(str(part).strip() for part in [objective, content] if str(part).strip())
        if not summary:
            stats["text_preprocessing_failed"] += 1
            continue

        text, is_valid = preprocess_text(summary, min_length=100, max_length=5000)
        if not is_valid:
            stats["text_preprocessing_failed"] += 1
            continue

        stats["text_preprocessing_passed"] += 1
        filtered.append({
            "data_type": "project",
            "no": len(filtered) + 1,
            "text": text,
            "title": project.get("PRJ_NM"),
            "year": year,
            "professor_info": project.get("professor_info"),
            "metadata": {
                "PRJ_RSPR_EMP_ID": project.get("PRJ_RSPR_EMP_ID"),
                "TOT_RND_AMT": project.get("TOT_RND_AMT"),
                "RCH_ST_DT": project.get("RCH_ST_DT"),
                "excel_base_year": project.get("excel_기준년도") or project.get("excel_湲곗??꾨룄"),
                "excel_project_name_kr": project.get("excel_과제명(국문)") or project.get("excel_怨쇱젣紐?援?Ц)"),
                "excel_expected_effect_summary": project.get("excel_기대효과요약") or project.get("excel_湲곕??④낵?붿빟"),
                "excel_연구목표요약": objective,
                "excel_연구내용요약": content,
            },
        })

    return filtered, stats


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, list) and not value:
        return False
    return True


def parse_year(date_value: Any) -> tuple[Optional[int], bool]:
    year = parse_year_value(date_value)
    return year, year is not None


def parse_year_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.year
    text = str(value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        year = int(text[:4])
        if 1900 <= year <= 2100:
            return year
    return None


def parse_project_year(project: Dict) -> Optional[int]:
    return (
        parse_year_value(project.get("excel_기준년도"))
        or parse_year_value(project.get("excel_湲곗??꾨룄"))
        or parse_year_value(project.get("RCH_ST_DT"))
    )


def has_invalid_metadata(metadata: Dict) -> bool:
    if not metadata:
        return True
    for value in metadata.values():
        if value is None or (isinstance(value, str) and not value.strip()):
            return True
        if isinstance(value, str) and value.strip() == "기타학술지(비정기개발학술지)":
            return True
    return False


def select_article_abstract(article: Dict) -> Optional[str]:
    for key in ["abstract", "abstract_description", "abstract_translated", "text"]:
        value = article.get(key)
        if not has_value(value):
            continue
        if isinstance(value, list):
            return select_text_from_list(value)
        if isinstance(value, str):
            parsed = parse_list_string(value)
            if parsed:
                return select_text_from_list(parsed)
            return value
        return str(value)
    return None


def parse_list_string(text: str) -> Optional[List]:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    try:
        parsed = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, list) else None


def select_text_from_list(values: List) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def remove_formulas_and_symbols(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\\\(.*?\\\)", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s.,!?;:()\[\]{}\"'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: Any, min_length: int = 100, max_length: int = 5000) -> Tuple[Optional[str], bool]:
    if text is None:
        return None, False
    cleaned = remove_formulas_and_symbols(str(text))
    if len(cleaned) < min_length or len(cleaned) > max_length:
        return None, False
    return cleaned, True
