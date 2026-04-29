from typing import Dict, List, Optional

from .settings import (
    INDIGO_DB_CONNECT_TIMEOUT,
    INDIGO_DB_HOST,
    INDIGO_DB_NAME,
    INDIGO_DB_PASSWORD,
    INDIGO_DB_PORT,
    INDIGO_DB_USER,
)


TABLE_PATENT = "tb_inu_tech"
TABLE_EMPLOYEE = "v_emp1"
TABLE_ARTICLE = "v_emp1_3"
TABLE_PROJECT = "vw_inu_prj_info"
TABLE_INVENTOR = "tb_inu_tech_invntr_v2026_0115"

EMPLOYEE_COLUMNS = [
    "SQ",
    "EMP_NO",
    "NM",
    "GEN_GBN",
    "BIRTH_DT",
    "NAT_GBN",
    "RECHER_REG_NO",
    "WKGD_NM",
    "COLG_NM",
    "HG_NM",
    "HOOF_GBN",
    "HANDP_NO",
    "OFCE_TELNO",
    "EMAIL",
]


def get_db_connection():
    try:
        import mariadb
    except ImportError as exc:
        raise ImportError("mariadb is required. Install project requirements first.") from exc

    missing = [
        name
        for name, value in {
            "INDIGO_DB_HOST": INDIGO_DB_HOST,
            "INDIGO_DB_USER": INDIGO_DB_USER,
            "INDIGO_DB_PASSWORD": INDIGO_DB_PASSWORD,
            "INDIGO_DB_NAME": INDIGO_DB_NAME,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing DB settings: {', '.join(missing)}")

    return mariadb.connect(
        user=INDIGO_DB_USER,
        password=INDIGO_DB_PASSWORD,
        host=INDIGO_DB_HOST,
        port=INDIGO_DB_PORT,
        database=INDIGO_DB_NAME,
        connect_timeout=INDIGO_DB_CONNECT_TIMEOUT,
        autocommit=True,
    )


def fetch_all_dicts(conn, query: str, params: Optional[tuple] = None) -> List[Dict]:
    cursor = conn.cursor()
    cursor.execute(query, params or ())
    columns = [column[0] for column in cursor.description]
    return [
        {
            column: _stringify_value(value)
            for column, value in zip(columns, row)
        }
        for row in cursor.fetchall()
    ]


def fetch_one_dict(conn, query: str, params: Optional[tuple] = None) -> Optional[Dict]:
    cursor = conn.cursor()
    cursor.execute(query, params or ())
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [column[0] for column in cursor.description]
    return {
        column: _stringify_value(value)
        for column, value in zip(columns, row)
    }


def get_patent_register_ids(conn, limit: int | None = None) -> List[Dict]:
    employee_select = ",\n            ".join(f"e.{column}" for column in EMPLOYEE_COLUMNS)
    query = f"""
        SELECT DISTINCT
            t.ptnt_rgstr_id,
            REPLACE(t.ptnt_rgstr_id, '-', '') AS ptnt_rgstr_id_clean,
            t.tech_nm,
            t.mbr_sn,
            inv.invntr_nm,
            inv.invntr_co_nm,
            {employee_select}
        FROM {TABLE_PATENT} t
        INNER JOIN {TABLE_INVENTOR} inv
            ON CAST(t.mbr_sn AS CHAR) = CAST(inv.mbr_sn AS CHAR)
        INNER JOIN {TABLE_EMPLOYEE} e
            ON CAST(inv.invntr_nm AS CHAR) COLLATE utf8mb4_unicode_ci = CAST(e.NM AS CHAR) COLLATE utf8mb4_unicode_ci
            AND CAST(inv.invntr_co_nm AS CHAR) COLLATE utf8mb4_unicode_ci = CAST(e.HG_NM AS CHAR) COLLATE utf8mb4_unicode_ci
        WHERE t.ptnt_rgstr_id IS NOT NULL
            AND t.ptnt_rgstr_id != ''
            AND inv.mbr_sn != 0
            AND inv.mbr_sn IS NOT NULL
            AND inv.invntr_se = 'A00'
    """
    if limit:
        query += f"\n        LIMIT {int(limit)}"

    rows = fetch_all_dicts(conn, query)
    for row in rows:
        row["professor_info"] = _professor_info_from_row(row)
    return rows


def get_original_patent_data(conn, register_id: str) -> Optional[Dict]:
    return fetch_one_dict(
        conn,
        f"SELECT * FROM {TABLE_PATENT} WHERE ptnt_rgstr_id = ? LIMIT 1",
        (register_id,),
    )


def get_article_candidates(conn, min_year: int = 2015, limit: int | None = None) -> List[Dict]:
    query = f"""
        SELECT EMP_NO, THSS_NM, PUBLSH_DT
        FROM {TABLE_ARTICLE}
        WHERE EMP_NO IS NOT NULL
          AND THSS_NM IS NOT NULL
          AND PUBLSH_DT IS NOT NULL
          AND YEAR(PUBLSH_DT) >= ?
        ORDER BY PUBLSH_DT DESC
    """
    if limit:
        query += f"\n        LIMIT {int(limit)}"
    rows = fetch_all_dicts(conn, query, (min_year,))

    seen = set()
    unique = []
    for row in rows:
        key = (str(row.get("EMP_NO", "")).strip(), str(row.get("THSS_NM", "")).strip())
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def get_project_rows_with_professor(conn, limit: int | None = None) -> List[Dict]:
    employee_select = ",\n            ".join(f"e.{column} AS emp_{column}" for column in EMPLOYEE_COLUMNS)
    query = f"""
        SELECT
            p.*,
            {employee_select}
        FROM {TABLE_PROJECT} p
        LEFT JOIN {TABLE_EMPLOYEE} e
            ON CAST(p.PRJ_RSPR_EMP_ID AS CHAR) = CAST(e.EMP_NO AS CHAR)
        WHERE p.PRJ_NM IS NOT NULL
          AND p.PRJ_NM != ''
    """
    if limit:
        query += f"\n        LIMIT {int(limit)}"

    rows = fetch_all_dicts(conn, query)
    for row in rows:
        row["professor_info"] = {
            column: row.get(f"emp_{column}", "")
            for column in EMPLOYEE_COLUMNS
        }
        for column in EMPLOYEE_COLUMNS:
            row.pop(f"emp_{column}", None)
    return rows


def _professor_info_from_row(row: Dict) -> Dict:
    return {
        column: row.get(column, "")
        for column in EMPLOYEE_COLUMNS
    }


def _stringify_value(value):
    if value is None:
        return None
    return str(value)
