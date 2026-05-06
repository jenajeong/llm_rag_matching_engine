"""
데이터베이스 연결 설정
MariaDB 접속 정보 및 연결 함수를 제공합니다.
"""

import mariadb
import pandas as pd
from typing import Optional, List, Dict

from indigo_pipeline.config import (
    INDIGO_DB_AUTOCOMMIT,
    INDIGO_DB_CONNECT_TIMEOUT,
    INDIGO_DB_HOST,
    INDIGO_DB_NAME,
    INDIGO_DB_PASSWORD,
    INDIGO_DB_PORT,
    INDIGO_DB_SSL,
    INDIGO_DB_USER,
)


# MariaDB 접속 정보
DB_CONFIG = {
    "HOST": INDIGO_DB_HOST,
    "PORT": INDIGO_DB_PORT,
    "USER": INDIGO_DB_USER,
    "PASSWORD": INDIGO_DB_PASSWORD,
    "DATABASE": INDIGO_DB_NAME,
    "CONNECT_TIMEOUT": INDIGO_DB_CONNECT_TIMEOUT,
    "AUTOCOMMIT": INDIGO_DB_AUTOCOMMIT,
    "SSL": INDIGO_DB_SSL,
}

# 테이블명 정의
TABLE_PATENT = "tb_inu_tech"
TABLE_EMPLOYEE = "v_emp1"
TABLE_ARTICLE = "v_emp1_3"
TABLE_PROJECT = "vw_inu_prj_info"
TABLE_INVENTOR = "tb_inu_tech_invntr_v2026_0115"

# 특허 테이블 컬럼명
COL_PATENT_APP_ID = "tech_aplct_id"
COL_PATENT_MBR_SN = "mbr_sn"
COL_PATENT_PROJECT_NAME = "tech_nm"
COL_PATENT_REGISTER_ID = "ptnt_rgstr_id"

# 교수 테이블 컬럼명
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

# 논문 테이블 컬럼명
COL_ARTICLE_EMP_NO = "EMP_NO"
COL_ARTICLE_THSS_NM = "THSS_NM"
COL_ARTICLE_PUBLSH_DT = "PUBLSH_DT"

# 연구과제 테이블 컬럼명
COL_PROJECT_PRJ_NM = "PRJ_NM"
COL_PROJECT_RSPR_EMP_ID = "PRJ_RSPR_EMP_ID"

# 작업 대상 테이블 (하위 호환성)
TARGET_TABLE = TABLE_PATENT
TARGET_ID_COLUMN = COL_PATENT_APP_ID


def get_db_connection() -> mariadb.Connection:
    """
    MariaDB 데이터베이스 연결을 생성하고 반환합니다.
    
    Returns:
        mariadb.Connection: 데이터베이스 연결 객체
        
    Raises:
        mariadb.Error: 데이터베이스 연결 실패 시
    """
    try:
        conn = mariadb.connect(
            user=DB_CONFIG["USER"],
            password=DB_CONFIG["PASSWORD"],
            host=DB_CONFIG["HOST"],
            port=DB_CONFIG["PORT"],
            database=DB_CONFIG["DATABASE"],
            connect_timeout=DB_CONFIG["CONNECT_TIMEOUT"],
            autocommit=DB_CONFIG["AUTOCOMMIT"],
            ssl=DB_CONFIG["SSL"]
        )
        print("MariaDB 연결 성공!")
        return conn
    except mariadb.Error as e:
        print("MariaDB 연결 실패!")
        print("오류 코드:", e.errno)
        print("오류 메시지:", e.msg)
        raise


def close_db_connection(conn: Optional[mariadb.Connection]):
    """
    데이터베이스 연결을 안전하게 종료합니다.
    
    Args:
        conn: 종료할 연결 객체
    """
    if conn:
        try:
            conn.close()
            print("연결 종료 완료.")
        except Exception as e:
            print(f"연결 종료 중 오류: {e}")


def test_connection():
    """데이터베이스 연결을 테스트합니다."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print(f"연결 테스트 성공: {result}")
    except Exception as e:
        print(f"연결 테스트 실패: {e}")
    finally:
        close_db_connection(conn)


def get_article_data(conn: mariadb.Connection, min_year: int = 2015) -> pd.DataFrame:
    """
    논문 데이터를 조회합니다.
    
    Args:
        conn: 데이터베이스 연결 객체
        min_year: 최소 게재 연도 (기본값: 2015)
        
    Returns:
        논문 데이터 DataFrame (EMP_NO, THSS_NM, PUBLSH_DT)
    """
    query = f"""
        SELECT
            {COL_ARTICLE_EMP_NO},
            {COL_ARTICLE_THSS_NM},
            {COL_ARTICLE_PUBLSH_DT}
        FROM {TABLE_ARTICLE}
        WHERE {COL_ARTICLE_EMP_NO} IS NOT NULL
          AND {COL_ARTICLE_THSS_NM} IS NOT NULL
          AND {COL_ARTICLE_PUBLSH_DT} IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    
    # 전처리
    df[COL_ARTICLE_EMP_NO] = df[COL_ARTICLE_EMP_NO].astype(str).str.strip()
    df[COL_ARTICLE_THSS_NM] = df[COL_ARTICLE_THSS_NM].astype(str).str.strip()
    df[COL_ARTICLE_PUBLSH_DT] = pd.to_datetime(df[COL_ARTICLE_PUBLSH_DT], errors="coerce")
    
    # 연도 필터링
    if min_year:
        df = df[df[COL_ARTICLE_PUBLSH_DT].dt.year >= min_year]
    
    # 중복 제거
    df = df.drop_duplicates(subset=[COL_ARTICLE_EMP_NO, COL_ARTICLE_THSS_NM]).reset_index(drop=True)
    
    # 게재일자 순 정렬 (최신순)
    df = df.sort_values(COL_ARTICLE_PUBLSH_DT, ascending=False).reset_index(drop=True)
    
    return df


def get_patent_statistics(conn: mariadb.Connection) -> Dict[str, int]:
    """
    특허 데이터 통계 정보를 조회합니다.
    
    Args:
        conn: 데이터베이스 연결 객체
        
    Returns:
        통계 정보 딕셔너리
    """
    stats = {}
    
    # 1. 특허 테이블 전체 데이터 개수
    query_total = f"SELECT COUNT(*) as total_count FROM {TABLE_PATENT}"
    df_total = pd.read_sql(query_total, conn)
    stats['total_records'] = int(df_total.iloc[0]['total_count'])
    
    # 2. 등록번호(ptnt_rgstr_id)가 있는 데이터 개수
    query_with_register_id = f"""
        SELECT COUNT(*) as count_with_register_id
        FROM {TABLE_PATENT}
        WHERE {COL_PATENT_REGISTER_ID} IS NOT NULL 
            AND {COL_PATENT_REGISTER_ID} != ''
    """
    df_with_register_id = pd.read_sql(query_with_register_id, conn)
    stats['records_with_register_id'] = int(df_with_register_id.iloc[0]['count_with_register_id'])
    
    # 3. 교수 사번 매칭된 데이터 개수 (발명자 테이블을 통한 매핑)
    query_matched = f"""
        SELECT COUNT(DISTINCT t.{COL_PATENT_REGISTER_ID}) as matched_count
        FROM {TABLE_PATENT} t
        INNER JOIN {TABLE_INVENTOR} inv 
            ON CAST(t.{COL_PATENT_MBR_SN} AS CHAR) = CAST(inv.mbr_sn AS CHAR)
        INNER JOIN {TABLE_EMPLOYEE} e 
            ON CAST(inv.invntr_nm AS CHAR) COLLATE utf8mb4_unicode_ci = CAST(e.{COL_EMP_NM} AS CHAR) COLLATE utf8mb4_unicode_ci
            AND CAST(inv.invntr_co_nm AS CHAR) COLLATE utf8mb4_unicode_ci = CAST(e.{COL_EMP_HG_NM} AS CHAR) COLLATE utf8mb4_unicode_ci
        WHERE t.{COL_PATENT_REGISTER_ID} IS NOT NULL 
            AND t.{COL_PATENT_REGISTER_ID} != ''
            AND inv.mbr_sn != 0
            AND inv.mbr_sn IS NOT NULL
            AND inv.invntr_se = 'A00'
    """
    df_matched = pd.read_sql(query_matched, conn)
    stats['records_matched_with_professor'] = int(df_matched.iloc[0]['matched_count'])
    
    return stats


def get_patent_application_ids(conn: mariadb.Connection, limit: Optional[int] = None) -> List[Dict]:
    """
    특허 출원번호와 교수 정보를 조회합니다.
    
    Args:
        conn: 데이터베이스 연결 객체
        limit: 가져올 최대 개수 (None이면 전체)
        
    Returns:
        [{"tech_aplct_id": "...", "mbr_sn": "...", "professor_info": {...}}, ...] 형태의 리스트
    """
    query = f"""
        SELECT DISTINCT 
            t.{COL_PATENT_APP_ID}, 
            t.{COL_PATENT_MBR_SN},
            e.{COL_EMP_SQ},
            e.{COL_EMP_NO},
            e.{COL_EMP_NM},
            e.{COL_EMP_GEN_GBN},
            e.{COL_EMP_BIRTH_DT},
            e.{COL_EMP_NAT_GBN},
            e.{COL_EMP_RECHER_REG_NO},
            e.{COL_EMP_WKGD_NM},
            e.{COL_EMP_COLG_NM},
            e.{COL_EMP_HG_NM},
            e.{COL_EMP_HOOF_GBN},
            e.{COL_EMP_HANDP_NO},
            e.{COL_EMP_OFCE_TELNO},
            e.{COL_EMP_EMAIL}
        FROM {TABLE_PATENT} t
        INNER JOIN {TABLE_EMPLOYEE} e ON CAST(t.{COL_PATENT_MBR_SN} AS CHAR) = CAST(e.{COL_EMP_SQ} AS CHAR)
        WHERE t.{COL_PATENT_APP_ID} IS NOT NULL 
            AND t.{COL_PATENT_APP_ID} != ''
            AND t.{COL_PATENT_MBR_SN} IS NOT NULL
            AND t.{COL_PATENT_MBR_SN} != ''
    """
    if limit:
        query += f" LIMIT {limit}"
    query += ";"
    
    df = pd.read_sql(query, conn)
    
    # 딕셔너리 리스트로 변환
    application_list = []
    for _, row in df.iterrows():
        if pd.notna(row[COL_PATENT_APP_ID]):
            # 교수 정보 추출
            professor_info = {
                "SQ": str(row[COL_EMP_SQ]) if pd.notna(row[COL_EMP_SQ]) else "",
                "EMP_NO": str(row[COL_EMP_NO]) if pd.notna(row[COL_EMP_NO]) else "",
                "NM": str(row[COL_EMP_NM]) if pd.notna(row[COL_EMP_NM]) else "",
                "GEN_GBN": str(row[COL_EMP_GEN_GBN]) if pd.notna(row[COL_EMP_GEN_GBN]) else "",
                "BIRTH_DT": str(row[COL_EMP_BIRTH_DT]) if pd.notna(row[COL_EMP_BIRTH_DT]) else "",
                "NAT_GBN": str(row[COL_EMP_NAT_GBN]) if pd.notna(row[COL_EMP_NAT_GBN]) else "",
                "RECHER_REG_NO": str(row[COL_EMP_RECHER_REG_NO]) if pd.notna(row[COL_EMP_RECHER_REG_NO]) else "",
                "WKGD_NM": str(row[COL_EMP_WKGD_NM]) if pd.notna(row[COL_EMP_WKGD_NM]) else "",
                "COLG_NM": str(row[COL_EMP_COLG_NM]) if pd.notna(row[COL_EMP_COLG_NM]) else "",
                "HG_NM": str(row[COL_EMP_HG_NM]) if pd.notna(row[COL_EMP_HG_NM]) else "",
                "HOOF_GBN": str(row[COL_EMP_HOOF_GBN]) if pd.notna(row[COL_EMP_HOOF_GBN]) else "",
                "HANDP_NO": str(row[COL_EMP_HANDP_NO]) if pd.notna(row[COL_EMP_HANDP_NO]) else "",
                "OFCE_TELNO": str(row[COL_EMP_OFCE_TELNO]) if pd.notna(row[COL_EMP_OFCE_TELNO]) else "",
                "EMAIL": str(row[COL_EMP_EMAIL]) if pd.notna(row[COL_EMP_EMAIL]) else "",
            }
            
            application_list.append({
                "tech_aplct_id": str(row[COL_PATENT_APP_ID]),
                "mbr_sn": str(row[COL_PATENT_MBR_SN]) if pd.notna(row[COL_PATENT_MBR_SN]) else "",
                "professor_info": professor_info
            })
    
    return application_list


def get_patent_register_ids(conn: mariadb.Connection, limit: Optional[int] = None, verbose: bool = False) -> List[Dict]:
    """
    특허 등록번호(ptnt_rgstr_id)와 교수 정보를 조회합니다.
    발명자 테이블을 통한 매핑 방식을 사용합니다.
    
    Args:
        conn: 데이터베이스 연결 객체
        limit: 가져올 최대 개수 (None이면 전체)
        verbose: 쿼리 정보를 출력할지 여부
        
    Returns:
        [{"ptnt_rgstr_id": "...", "ptnt_rgstr_id_clean": "...", "tech_nm": "...", "mbr_sn": "...", "professor_info": {...}}, ...] 형태의 리스트
    """
    query = f"""
        SELECT DISTINCT 
            t.{COL_PATENT_REGISTER_ID}, 
            t.{COL_PATENT_PROJECT_NAME},
            t.{COL_PATENT_MBR_SN},
            inv.invntr_nm,
            inv.invntr_co_nm,
            e.{COL_EMP_SQ},
            e.{COL_EMP_NO},
            e.{COL_EMP_NM},
            e.{COL_EMP_GEN_GBN},
            e.{COL_EMP_BIRTH_DT},
            e.{COL_EMP_NAT_GBN},
            e.{COL_EMP_RECHER_REG_NO},
            e.{COL_EMP_WKGD_NM},
            e.{COL_EMP_COLG_NM},
            e.{COL_EMP_HG_NM},
            e.{COL_EMP_HOOF_GBN},
            e.{COL_EMP_HANDP_NO},
            e.{COL_EMP_OFCE_TELNO},
            e.{COL_EMP_EMAIL}
        FROM {TABLE_PATENT} t
        INNER JOIN {TABLE_INVENTOR} inv 
            ON CAST(t.{COL_PATENT_MBR_SN} AS CHAR) = CAST(inv.mbr_sn AS CHAR)
        INNER JOIN {TABLE_EMPLOYEE} e 
            ON CAST(inv.invntr_nm AS CHAR) COLLATE utf8mb4_unicode_ci = CAST(e.{COL_EMP_NM} AS CHAR) COLLATE utf8mb4_unicode_ci
            AND CAST(inv.invntr_co_nm AS CHAR) COLLATE utf8mb4_unicode_ci = CAST(e.{COL_EMP_HG_NM} AS CHAR) COLLATE utf8mb4_unicode_ci
        WHERE t.{COL_PATENT_REGISTER_ID} IS NOT NULL 
            AND t.{COL_PATENT_REGISTER_ID} != ''
            AND inv.mbr_sn != 0
            AND inv.mbr_sn IS NOT NULL
            AND inv.invntr_se = 'A00'
    """
    if limit:
        query += f" LIMIT {limit}"
    query += ";"
    
    if verbose:
        print(f"[실행 쿼리]")
        print(query)
        print(f"[쿼리 설명]")
        print(f"  - 테이블: {TABLE_PATENT} (별칭: t) INNER JOIN {TABLE_INVENTOR} (별칭: inv) INNER JOIN {TABLE_EMPLOYEE} (별칭: e)")
        print(f"  - 조인 조건 1: t.{COL_PATENT_MBR_SN} = inv.mbr_sn")
        print(f"  - 조인 조건 2: inv.invntr_nm = e.{COL_EMP_NM} AND inv.invntr_co_nm = e.{COL_EMP_HG_NM}")
        print(f"  - 필터 조건:")
        print(f"    * t.{COL_PATENT_REGISTER_ID} IS NOT NULL AND != ''")
        print(f"    * inv.mbr_sn != 0 AND IS NOT NULL")
        print(f"    * inv.invntr_se = 'A00'")
        print(f"  - 조회 컬럼: 특허 등록번호, 특허명(tech_nm), 발명자 정보, 교수 정보 (SQ, EMP_NO, NM, 등)")
        if limit:
            print(f"  - 제한: LIMIT {limit}")
        else:
            print(f"  - 제한: 없음 (전체 조회)")
    
    df = pd.read_sql(query, conn)
    
    # 딕셔너리 리스트로 변환
    register_id_list = []
    for _, row in df.iterrows():
        if pd.notna(row[COL_PATENT_REGISTER_ID]):
            # 등록번호에서 - 제거
            register_id = str(row[COL_PATENT_REGISTER_ID]).strip()
            register_id_clean = register_id.replace("-", "")
            
            # 특허명 가져오기 (확인용)
            tech_nm = str(row[COL_PATENT_PROJECT_NAME]).strip() if pd.notna(row[COL_PATENT_PROJECT_NAME]) else ""
            
            # 발명자 정보 (확인용)
            invntr_nm = str(row['invntr_nm']).strip() if pd.notna(row['invntr_nm']) else ""
            invntr_co_nm = str(row['invntr_co_nm']).strip() if pd.notna(row['invntr_co_nm']) else ""
            
            # 교수 정보 추출
            professor_info = {
                "SQ": str(row[COL_EMP_SQ]) if pd.notna(row[COL_EMP_SQ]) else "",
                "EMP_NO": str(row[COL_EMP_NO]) if pd.notna(row[COL_EMP_NO]) else "",
                "NM": str(row[COL_EMP_NM]) if pd.notna(row[COL_EMP_NM]) else "",
                "GEN_GBN": str(row[COL_EMP_GEN_GBN]) if pd.notna(row[COL_EMP_GEN_GBN]) else "",
                "BIRTH_DT": str(row[COL_EMP_BIRTH_DT]) if pd.notna(row[COL_EMP_BIRTH_DT]) else "",
                "NAT_GBN": str(row[COL_EMP_NAT_GBN]) if pd.notna(row[COL_EMP_NAT_GBN]) else "",
                "RECHER_REG_NO": str(row[COL_EMP_RECHER_REG_NO]) if pd.notna(row[COL_EMP_RECHER_REG_NO]) else "",
                "WKGD_NM": str(row[COL_EMP_WKGD_NM]) if pd.notna(row[COL_EMP_WKGD_NM]) else "",
                "COLG_NM": str(row[COL_EMP_COLG_NM]) if pd.notna(row[COL_EMP_COLG_NM]) else "",
                "HG_NM": str(row[COL_EMP_HG_NM]) if pd.notna(row[COL_EMP_HG_NM]) else "",
                "HOOF_GBN": str(row[COL_EMP_HOOF_GBN]) if pd.notna(row[COL_EMP_HOOF_GBN]) else "",
                "HANDP_NO": str(row[COL_EMP_HANDP_NO]) if pd.notna(row[COL_EMP_HANDP_NO]) else "",
                "OFCE_TELNO": str(row[COL_EMP_OFCE_TELNO]) if pd.notna(row[COL_EMP_OFCE_TELNO]) else "",
                "EMAIL": str(row[COL_EMP_EMAIL]) if pd.notna(row[COL_EMP_EMAIL]) else "",
            }
            
            register_id_list.append({
                "ptnt_rgstr_id": register_id,  # 원본 등록번호
                "ptnt_rgstr_id_clean": register_id_clean,  # - 제거된 등록번호
                "tech_nm": tech_nm,  # 특허명 (확인용)
                "invntr_nm": invntr_nm,  # 발명자 이름 (확인용)
                "invntr_co_nm": invntr_co_nm,  # 발명자 소속 (확인용)
                "mbr_sn": str(row[COL_PATENT_MBR_SN]) if pd.notna(row[COL_PATENT_MBR_SN]) else "",
                "professor_info": professor_info
            })
    
    return register_id_list


def get_project_statistics(conn: mariadb.Connection) -> Dict[str, int]:
    """
    연구과제 데이터 통계 정보를 조회합니다.
    
    Args:
        conn: 데이터베이스 연결 객체
        
    Returns:
        통계 정보 딕셔너리
    """
    stats = {}
    
    # 1. 연구과제 테이블 전체 데이터 개수
    query_total = f"SELECT COUNT(*) as total_count FROM {TABLE_PROJECT}"
    df_total = pd.read_sql(query_total, conn)
    stats['total_records'] = int(df_total.iloc[0]['total_count'])
    
    # 2. PRJ_NM이 있는 데이터 개수
    query_with_name = f"""
        SELECT COUNT(*) as count_with_name
        FROM {TABLE_PROJECT}
        WHERE {COL_PROJECT_PRJ_NM} IS NOT NULL 
            AND {COL_PROJECT_PRJ_NM} != ''
    """
    df_with_name = pd.read_sql(query_with_name, conn)
    stats['records_with_project_name'] = int(df_with_name.iloc[0]['count_with_name'])
    
    # 3. 교수 정보 매칭된 데이터 개수
    query_matched = f"""
        SELECT COUNT(DISTINCT p.{COL_PROJECT_PRJ_NM}) as matched_count
        FROM {TABLE_PROJECT} p
        INNER JOIN {TABLE_EMPLOYEE} e ON CAST(p.{COL_PROJECT_RSPR_EMP_ID} AS CHAR) = CAST(e.{COL_EMP_NO} AS CHAR)
        WHERE p.{COL_PROJECT_PRJ_NM} IS NOT NULL 
            AND p.{COL_PROJECT_PRJ_NM} != ''
            AND p.{COL_PROJECT_RSPR_EMP_ID} IS NOT NULL
            AND p.{COL_PROJECT_RSPR_EMP_ID} != ''
    """
    df_matched = pd.read_sql(query_matched, conn)
    stats['records_matched_with_professor'] = int(df_matched.iloc[0]['matched_count'])
    
    return stats


def get_project_data(conn: mariadb.Connection, limit: Optional[int] = None) -> List[Dict]:
    """
    연구과제 데이터를 조회합니다.
    
    Args:
        conn: 데이터베이스 연결 객체
        limit: 가져올 최대 개수 (None이면 전체)
        
    Returns:
        연구과제 데이터 리스트 (모든 컬럼 포함)
    """
    query = f"SELECT * FROM {TABLE_PROJECT}"
    if limit:
        query += f" LIMIT {limit}"
    query += ";"
    
    df = pd.read_sql(query, conn)
    
    # 딕셔너리 리스트로 변환
    project_list = []
    for _, row in df.iterrows():
        project_data = {}
        for col in df.columns:
            value = row[col]
            if pd.notna(value):
                # 숫자 타입은 그대로, 문자열은 str로 변환
                if isinstance(value, (int, float)):
                    project_data[col] = value
                else:
                    project_data[col] = str(value)
            else:
                project_data[col] = None
        project_list.append(project_data)
    
    return project_list


def get_project_with_professor_info(conn: mariadb.Connection, limit: Optional[int] = None) -> List[Dict]:
    """
    연구과제 데이터와 교수 정보를 함께 조회합니다.
    PRJ_RSPR_EMP_ID와 EMP_NO를 매핑합니다.
    
    Args:
        conn: 데이터베이스 연결 객체
        limit: 가져올 최대 개수 (None이면 전체)
        
    Returns:
        [{"project_data": {...}, "professor_info": {...}}, ...] 형태의 리스트
    """
    query = f"""
        SELECT 
            p.*,
            e.{COL_EMP_SQ},
            e.{COL_EMP_NO},
            e.{COL_EMP_NM},
            e.{COL_EMP_GEN_GBN},
            e.{COL_EMP_BIRTH_DT},
            e.{COL_EMP_NAT_GBN},
            e.{COL_EMP_RECHER_REG_NO},
            e.{COL_EMP_WKGD_NM},
            e.{COL_EMP_COLG_NM},
            e.{COL_EMP_HG_NM},
            e.{COL_EMP_HOOF_GBN},
            e.{COL_EMP_HANDP_NO},
            e.{COL_EMP_OFCE_TELNO},
            e.{COL_EMP_EMAIL}
        FROM {TABLE_PROJECT} p
        LEFT JOIN {TABLE_EMPLOYEE} e ON CAST(p.{COL_PROJECT_RSPR_EMP_ID} AS CHAR) = CAST(e.{COL_EMP_NO} AS CHAR)
    """
    if limit:
        query += f" LIMIT {limit}"
    query += ";"
    
    df = pd.read_sql(query, conn)
    
    # 딕셔너리 리스트로 변환
    project_list = []
    for _, row in df.iterrows():
        # 프로젝트 데이터 추출 (교수 정보 컬럼 제외)
        project_data = {}
        professor_info = {}
        
        for col in df.columns:
            if col in [COL_EMP_SQ, COL_EMP_NO, COL_EMP_NM, COL_EMP_GEN_GBN, 
                      COL_EMP_BIRTH_DT, COL_EMP_NAT_GBN, COL_EMP_RECHER_REG_NO,
                      COL_EMP_WKGD_NM, COL_EMP_COLG_NM, COL_EMP_HG_NM, 
                      COL_EMP_HOOF_GBN, COL_EMP_HANDP_NO, COL_EMP_OFCE_TELNO, COL_EMP_EMAIL]:
                # 교수 정보 컬럼
                value = row[col]
                if pd.notna(value):
                    if isinstance(value, (int, float)):
                        professor_info[col] = value
                    else:
                        professor_info[col] = str(value)
                else:
                    professor_info[col] = None
            else:
                # 프로젝트 데이터 컬럼
                value = row[col]
                if pd.notna(value):
                    if isinstance(value, (int, float)):
                        project_data[col] = value
                    else:
                        project_data[col] = str(value)
                else:
                    project_data[col] = None
        
        # professor_info 딕셔너리 키 이름 변경 (COL_EMP_* -> 실제 키 이름)
        professor_info_formatted = {
            "SQ": professor_info.get(COL_EMP_SQ, ""),
            "EMP_NO": professor_info.get(COL_EMP_NO, ""),
            "NM": professor_info.get(COL_EMP_NM, ""),
            "GEN_GBN": professor_info.get(COL_EMP_GEN_GBN, ""),
            "BIRTH_DT": professor_info.get(COL_EMP_BIRTH_DT, ""),
            "NAT_GBN": professor_info.get(COL_EMP_NAT_GBN, ""),
            "RECHER_REG_NO": professor_info.get(COL_EMP_RECHER_REG_NO, ""),
            "WKGD_NM": professor_info.get(COL_EMP_WKGD_NM, ""),
            "COLG_NM": professor_info.get(COL_EMP_COLG_NM, ""),
            "HG_NM": professor_info.get(COL_EMP_HG_NM, ""),
            "HOOF_GBN": professor_info.get(COL_EMP_HOOF_GBN, ""),
            "HANDP_NO": professor_info.get(COL_EMP_HANDP_NO, ""),
            "OFCE_TELNO": professor_info.get(COL_EMP_OFCE_TELNO, ""),
            "EMAIL": professor_info.get(COL_EMP_EMAIL, ""),
        }
        
        project_list.append({
            "project_data": project_data,
            "professor_info": professor_info_formatted if professor_info.get(COL_EMP_SQ) else None
        })
    
    return project_list


if __name__ == "__main__":
    # 연결 테스트
    test_connection()

