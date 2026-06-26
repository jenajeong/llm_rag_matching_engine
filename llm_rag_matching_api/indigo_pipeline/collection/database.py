"""
Indigo collection data access.

The pipeline used to read directly from MariaDB. Indigo now exposes the same
source tables through a paginated POST API, so this module keeps the old helper
function names while fetching data from the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from indigo_pipeline.config import (
    INDIGO_API_KEY,
    INDIGO_API_TIMEOUT,
    INDIGO_API_URL,
)


TABLE_PATENT = "tb_inu_tech"
TABLE_EMPLOYEE = "v_emp1"
TABLE_ARTICLE = "v_emp1_3"
TABLE_PROJECT = "vw_inu_prj_info"
TABLE_INVENTOR = "tb_inu_tech_invntr_v2026_0115"

CAT_EMPLOYEE = "emp1"
CAT_ARTICLE = "emp1_3"
CAT_PATENT = "inu_tech"
CAT_INVENTOR = "inu_tech_invntr"
CAT_PROJECT = "inu_prj_info"

CAT_BY_TABLE = {
    TABLE_EMPLOYEE: CAT_EMPLOYEE,
    TABLE_ARTICLE: CAT_ARTICLE,
    TABLE_PATENT: CAT_PATENT,
    TABLE_INVENTOR: CAT_INVENTOR,
    TABLE_PROJECT: CAT_PROJECT,
}

VALID_CATS = {CAT_EMPLOYEE, CAT_ARTICLE, CAT_PATENT, CAT_INVENTOR, CAT_PROJECT}

# Patent columns
COL_PATENT_APP_ID = "tech_aplct_id"
COL_PATENT_MBR_SN = "mbr_sn"
COL_PATENT_PROJECT_NAME = "tech_nm"
COL_PATENT_REGISTER_ID = "ptnt_rgstr_id"

# Professor columns
COL_EMP_SQ = "SQ"
COL_EMP_NO = "EMP_NO"
COL_EMP_NM = "NM"
COL_EMP_GEN_GBN = "GEN_GBN"
COL_EMP_BIRTH_DT = "BIRTH_DT"
COL_EMP_NAT_GBN = "NAT_GBN"
COL_EMP_RECHER_REG_NO = "RECHER_REG_NO"
COL_EMP_WKGD_NM = "WKGD_NM"
COL_EMP_COLG_NM = "COLG_NM"
COL_EMP_HG_NM = "HG_NM"
COL_EMP_HOOF_GBN = "HOOF_GBN"
COL_EMP_HANDP_NO = "HANDP_NO"
COL_EMP_OFCE_TELNO = "OFCE_TELNO"
COL_EMP_EMAIL = "EMAIL"

EMPLOYEE_COLUMNS = [
    COL_EMP_SQ,
    COL_EMP_NO,
    COL_EMP_NM,
    COL_EMP_GEN_GBN,
    COL_EMP_BIRTH_DT,
    COL_EMP_NAT_GBN,
    COL_EMP_RECHER_REG_NO,
    COL_EMP_WKGD_NM,
    COL_EMP_COLG_NM,
    COL_EMP_HG_NM,
    COL_EMP_HOOF_GBN,
    COL_EMP_HANDP_NO,
    COL_EMP_OFCE_TELNO,
    COL_EMP_EMAIL,
]

# Article columns
COL_ARTICLE_EMP_NO = "EMP_NO"
COL_ARTICLE_THSS_NM = "THSS_NM"
COL_ARTICLE_PUBLSH_DT = "PUBLSH_DT"

# Project columns
COL_PROJECT_PRJ_NM = "PRJ_NM"
COL_PROJECT_RSPR_EMP_ID = "PRJ_RSPR_EMP_ID"

TARGET_TABLE = TABLE_PATENT
TARGET_ID_COLUMN = COL_PATENT_APP_ID


@dataclass
class IndigoApiClient:
    url: str = INDIGO_API_URL
    key: str = INDIGO_API_KEY
    timeout: int = INDIGO_API_TIMEOUT
    _cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def fetch_page(self, cat: str, page: int = 1) -> Dict[str, Any]:
        if cat not in VALID_CATS:
            raise ValueError(f"Invalid Indigo API cat: {cat}")

        response = requests.post(
            self.url,
            data={"key": self.key, "cat": cat, "page": page or 1},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        status = payload.get("status")
        if str(status).lower() not in {"1", "true", "success", "s", "y", "ok"}:
            raise RuntimeError(f"Indigo API failed for cat={cat}, page={page}: {payload}")

        return payload

    def fetch_all(self, cat: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if cat in self._cache:
            rows = self._cache[cat]
            return rows[:limit] if limit else rows

        all_rows: List[Dict[str, Any]] = []
        page = 1

        while True:
            payload = self.fetch_page(cat, page)
            data = payload.get("data") or {}
            datas = data.get("datas") or []
            if isinstance(datas, dict):
                datas = [datas]

            all_rows.extend([_with_column_aliases(row) for row in datas if isinstance(row, dict)])

            record = _to_int(data.get("record"))
            rows_per_page = _to_int(data.get("rows")) or len(datas) or 100

            if limit and len(all_rows) >= limit:
                break
            if not datas:
                break
            if record and len(all_rows) >= record:
                break
            if len(datas) < rows_per_page:
                break

            page += 1

        self._cache[cat] = all_rows
        return all_rows[:limit] if limit else all_rows

    def close(self) -> None:
        self._cache.clear()


def get_db_connection() -> IndigoApiClient:
    print("Indigo API client ready.")
    return IndigoApiClient()


def close_db_connection(conn: Optional[IndigoApiClient]):
    if conn:
        conn.close()
        print("Indigo API client closed.")


def test_connection():
    conn = None
    try:
        conn = get_db_connection()
        payload = conn.fetch_page(CAT_EMPLOYEE, page=1)
        count = len((payload.get("data") or {}).get("datas") or [])
        print(f"Indigo API connection test succeeded: {count} rows on page 1")
    except Exception as e:
        print(f"Indigo API connection test failed: {e}")
    finally:
        close_db_connection(conn)


def get_api_rows(cat_or_table: str, conn: Optional[IndigoApiClient] = None, limit: Optional[int] = None) -> List[Dict]:
    cat = CAT_BY_TABLE.get(cat_or_table, cat_or_table)
    client = conn or get_db_connection()
    try:
        return client.fetch_all(cat, limit=limit)
    finally:
        if conn is None:
            close_db_connection(client)


def get_api_dataframe(cat_or_table: str, conn: Optional[IndigoApiClient] = None, limit: Optional[int] = None) -> pd.DataFrame:
    return pd.DataFrame(get_api_rows(cat_or_table, conn=conn, limit=limit))


def get_article_data(conn: IndigoApiClient, min_year: int = 2015) -> pd.DataFrame:
    df = get_api_dataframe(CAT_ARTICLE, conn)
    _ensure_columns(df, [COL_ARTICLE_EMP_NO, COL_ARTICLE_THSS_NM, COL_ARTICLE_PUBLSH_DT])

    df = df[
        df[COL_ARTICLE_EMP_NO].notna()
        & df[COL_ARTICLE_THSS_NM].notna()
        & df[COL_ARTICLE_PUBLSH_DT].notna()
    ].copy()
    df[COL_ARTICLE_EMP_NO] = df[COL_ARTICLE_EMP_NO].astype(str).str.strip()
    df[COL_ARTICLE_THSS_NM] = df[COL_ARTICLE_THSS_NM].astype(str).str.strip()
    df[COL_ARTICLE_PUBLSH_DT] = pd.to_datetime(df[COL_ARTICLE_PUBLSH_DT], errors="coerce")

    if min_year:
        df = df[df[COL_ARTICLE_PUBLSH_DT].dt.year >= min_year]

    df = df.drop_duplicates(subset=[COL_ARTICLE_EMP_NO, COL_ARTICLE_THSS_NM]).reset_index(drop=True)
    df = df.sort_values(COL_ARTICLE_PUBLSH_DT, ascending=False).reset_index(drop=True)
    return df


def get_patent_statistics(conn: IndigoApiClient) -> Dict[str, int]:
    patent_df = get_api_dataframe(CAT_PATENT, conn)
    matched_df = _patent_professor_join(conn)

    total_records = len(patent_df)
    with_register = _non_empty_mask(patent_df, COL_PATENT_REGISTER_ID).sum()

    return {
        "total_records": int(total_records),
        "records_with_register_id": int(with_register),
        "records_matched_with_professor": int(matched_df[COL_PATENT_REGISTER_ID].dropna().nunique())
        if COL_PATENT_REGISTER_ID in matched_df
        else 0,
    }


def get_patent_application_ids(conn: IndigoApiClient, limit: Optional[int] = None) -> List[Dict]:
    patent_df = get_api_dataframe(CAT_PATENT, conn)
    emp_df = get_api_dataframe(CAT_EMPLOYEE, conn)
    _ensure_columns(patent_df, [COL_PATENT_APP_ID, COL_PATENT_MBR_SN])
    _ensure_columns(emp_df, EMPLOYEE_COLUMNS)

    df = patent_df[
        _non_empty_mask(patent_df, COL_PATENT_APP_ID) & _non_empty_mask(patent_df, COL_PATENT_MBR_SN)
    ].copy()
    df["_mbr_sn_key"] = df[COL_PATENT_MBR_SN].map(_key)
    emp_df = emp_df.copy()
    emp_df["_sq_key"] = emp_df[COL_EMP_SQ].map(_key)
    df = df.merge(emp_df[EMPLOYEE_COLUMNS + ["_sq_key"]], left_on="_mbr_sn_key", right_on="_sq_key", how="inner")
    if limit:
        df = df.head(limit)

    result = []
    for _, row in df.iterrows():
        result.append(
            {
                "tech_aplct_id": _str_value(row.get(COL_PATENT_APP_ID)),
                "mbr_sn": _str_value(row.get(COL_PATENT_MBR_SN)),
                "professor_info": _professor_info(row),
            }
        )
    return result


def get_patent_register_ids(conn: IndigoApiClient, limit: Optional[int] = None, verbose: bool = False) -> List[Dict]:
    df = _patent_professor_join(conn)
    if limit:
        df = df.head(limit)

    if verbose:
        print("[Indigo API]")
        print(f"  - cat={CAT_PATENT}, cat={CAT_INVENTOR}, cat={CAT_EMPLOYEE}")
        print("  - join: patent.mbr_sn = inventor.mbr_sn, inventor name/dept = professor name/dept")
        print("  - filter: ptnt_rgstr_id present, inventor.mbr_sn present/non-zero, invntr_se = A00")
        print(f"  - rows: {len(df):,}")

    result = []
    for _, row in df.iterrows():
        register_id = _str_value(row.get(COL_PATENT_REGISTER_ID)).strip()
        if not register_id:
            continue
        result.append(
            {
                "ptnt_rgstr_id": register_id,
                "ptnt_rgstr_id_clean": register_id.replace("-", ""),
                "tech_nm": _str_value(row.get(COL_PATENT_PROJECT_NAME)).strip(),
                "invntr_nm": _str_value(row.get("invntr_nm")).strip(),
                "invntr_co_nm": _str_value(row.get("invntr_co_nm")).strip(),
                "mbr_sn": _str_value(row.get(COL_PATENT_MBR_SN)),
                "professor_info": _professor_info(row),
            }
        )
    return result


def get_project_statistics(conn: IndigoApiClient) -> Dict[str, int]:
    project_df = get_api_dataframe(CAT_PROJECT, conn)
    project_prof = get_project_with_professor_info(conn)

    with_name = _non_empty_mask(project_df, COL_PROJECT_PRJ_NM).sum() if COL_PROJECT_PRJ_NM in project_df else 0
    matched = {
        item["project_data"].get(COL_PROJECT_PRJ_NM)
        for item in project_prof
        if item.get("professor_info") and item.get("project_data", {}).get(COL_PROJECT_PRJ_NM)
    }

    return {
        "total_records": int(len(project_df)),
        "records_with_project_name": int(with_name),
        "records_matched_with_professor": len(matched),
    }


def get_project_data(conn: IndigoApiClient, limit: Optional[int] = None) -> List[Dict]:
    return [_row_to_dict(row) for row in get_api_rows(CAT_PROJECT, conn=conn, limit=limit)]


def get_project_with_professor_info(conn: IndigoApiClient, limit: Optional[int] = None) -> List[Dict]:
    project_df = get_api_dataframe(CAT_PROJECT, conn)
    emp_df = get_api_dataframe(CAT_EMPLOYEE, conn)
    _ensure_columns(project_df, [COL_PROJECT_RSPR_EMP_ID])
    _ensure_columns(emp_df, EMPLOYEE_COLUMNS)

    project_df = project_df.copy()
    emp_df = emp_df.copy()
    project_df["_emp_key"] = project_df[COL_PROJECT_RSPR_EMP_ID].map(_key)
    emp_df["_emp_key"] = emp_df[COL_EMP_NO].map(_key)
    df = project_df.merge(emp_df[EMPLOYEE_COLUMNS + ["_emp_key"]], on="_emp_key", how="left", suffixes=("", "_emp"))
    if limit:
        df = df.head(limit)

    project_columns = [col for col in project_df.columns if col != "_emp_key"]
    result = []
    for _, row in df.iterrows():
        project_data = {col: _json_value(row.get(col)) for col in project_columns}
        has_professor = bool(_str_value(row.get(COL_EMP_SQ)))
        result.append(
            {
                "project_data": project_data,
                "professor_info": _professor_info(row) if has_professor else None,
            }
        )
    return result


def _patent_professor_join(conn: IndigoApiClient) -> pd.DataFrame:
    patent_df = get_api_dataframe(CAT_PATENT, conn)
    inventor_df = get_api_dataframe(CAT_INVENTOR, conn)
    emp_df = get_api_dataframe(CAT_EMPLOYEE, conn)

    _ensure_columns(patent_df, [COL_PATENT_REGISTER_ID, COL_PATENT_MBR_SN, COL_PATENT_PROJECT_NAME])
    _ensure_columns(inventor_df, ["mbr_sn", "invntr_nm", "invntr_co_nm", "invntr_se"])
    _ensure_columns(emp_df, EMPLOYEE_COLUMNS)

    patent_df = patent_df[
        _non_empty_mask(patent_df, COL_PATENT_REGISTER_ID) & _non_empty_mask(patent_df, COL_PATENT_MBR_SN)
    ].copy()
    inventor_df = inventor_df[
        _non_empty_mask(inventor_df, "mbr_sn")
        & (inventor_df["mbr_sn"].map(_key) != "0")
        & (inventor_df["invntr_se"].astype(str).str.strip() == "A00")
    ].copy()

    patent_df["_mbr_key"] = patent_df[COL_PATENT_MBR_SN].map(_key)
    inventor_df["_mbr_key"] = inventor_df["mbr_sn"].map(_key)
    merged = patent_df.merge(inventor_df, on="_mbr_key", how="inner", suffixes=("", "_inv"))

    merged["_name_key"] = merged["invntr_nm"].map(_norm)
    merged["_dept_key"] = merged["invntr_co_nm"].map(_norm)
    emp_df = emp_df.copy()
    emp_df["_name_key"] = emp_df[COL_EMP_NM].map(_norm)
    emp_df["_dept_key"] = emp_df[COL_EMP_HG_NM].map(_norm)

    return merged.merge(emp_df[EMPLOYEE_COLUMNS + ["_name_key", "_dept_key"]], on=["_name_key", "_dept_key"], how="inner")


def _ensure_columns(df: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA


def _with_column_aliases(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    for key, value in row.items():
        if isinstance(key, str):
            normalized.setdefault(key.upper(), value)
    return normalized


def _non_empty_mask(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].notna() & (df[column].astype(str).str.strip() != "")


def _professor_info(row: pd.Series) -> Dict[str, str]:
    return {col: _str_value(row.get(col)) for col in EMPLOYEE_COLUMNS}


def _row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _json_value(value) for key, value in row.items()}


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return value
    return str(value)


def _str_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _key(value: Any) -> str:
    return _str_value(value).strip()


def _norm(value: Any) -> str:
    return _str_value(value).strip().lower()


def _to_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    test_connection()
