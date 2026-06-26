import mariadb
import pandas as pd
import json
from typing import List, Dict, Optional
from pathlib import Path
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from indigo_pipeline.collection.database import (
    CAT_ARTICLE,
    CAT_EMPLOYEE,
    get_db_connection, 
    close_db_connection,
    get_api_dataframe,
    TABLE_EMPLOYEE,
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
    COL_EMP_EMAIL
)
from indigo_pipeline.config import ARTICLE_DATA_FILE, ARTICLE_PAPER_NO_PROFESSOR_FILE


def normalize(text):
    if not text:
        return ""
    return str(text).strip().lower()


def get_professor_info_by_emp_no(conn: mariadb.Connection, emp_no: str) -> Optional[Dict]:
    if not emp_no or not str(emp_no).strip():
        return None
    
    emp_no_clean = str(emp_no).strip()
    
    try:
        df = get_api_dataframe(CAT_EMPLOYEE, conn)
        df = df[df[COL_EMP_NO].astype(str).str.strip() == emp_no_clean].head(1)
        if df.empty:
            return None
        
        row = df.iloc[0]
        
        return {
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
    except Exception as e:
        print(f"교수 정보 조회 실패 (EMP_NO: {emp_no}): {e}")
        return None


def load_article_map(conn):
    df = get_api_dataframe(CAT_ARTICLE, conn)

    article_map = {}
    for _, row in df.iterrows():
        title = normalize(row["THSS_NM"])
        article_map[title] = row.to_dict()

    return article_map


def load_paper_json() -> List[Dict]:
    paper_file = ARTICLE_PAPER_NO_PROFESSOR_FILE
    paper_path = Path(paper_file)
    
    if not paper_path.exists():
        print(f"⚠️ 파일 없음: {paper_file}")
        return []
    
    with open(paper_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_professor_info_to_articles(articles: List[Dict], conn: mariadb.Connection, article_map) -> List[Dict]:
    professor_cache = {}
    results = []

    for article in articles:
        article_with_prof = article.copy()

        # 컬럼명 변환
        if "제목" in article_with_prof:
            article_with_prof["title_raw"] = article_with_prof.pop("제목")

        if "저자" in article_with_prof:
            article_with_prof["authors"] = article_with_prof.pop("저자")

        if "소속기관" in article_with_prof:
            article_with_prof["source"] = article_with_prof.pop("소속기관")

        if "저자 키워드" in article_with_prof:
            article_with_prof["keywords_merged"] = article_with_prof.pop("저자 키워드")
        
        if "Abstract" in article_with_prof:
            article_with_prof["abstract"] = article_with_prof.pop("Abstract")

        title = normalize(article_with_prof.get("THSS_NM"))
        meta = article_map.get(title)

        if meta:
            article_with_prof["THSS_PATICP_GBN"] = meta.get("THSS_PATICP_GBN")
            article_with_prof["JRNL_GBN"] = meta.get("JRNL_GBN")
            article_with_prof["YY"] = meta.get("YY")
        else:
            article_with_prof["THSS_PATICP_GBN"] = None
            article_with_prof["JRNL_GBN"] = None
            article_with_prof["YY"] = None

        # EMP_NO 처리
        emp_no_raw = article.get("EMP_NO")
        emp_no = None

        if isinstance(emp_no_raw, list):
            if len(emp_no_raw) > 0:
                emp_no = str(emp_no_raw[0]).strip()
        elif emp_no_raw:
            emp_no = str(emp_no_raw).strip()

        if emp_no:
            if emp_no in professor_cache:
                professor_info = professor_cache[emp_no]
            else:
                professor_info = get_professor_info_by_emp_no(conn, emp_no)
                professor_cache[emp_no] = professor_info

            article_with_prof["professor_info"] = professor_info
        else:
            article_with_prof["professor_info"] = None

        results.append(article_with_prof)

    return results


def save_article_json(articles: List[Dict]):
    output_file = ARTICLE_DATA_FILE
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def main():
    conn = None
    try:
        conn = get_db_connection()

        articles = load_paper_json()
        if not articles:
            print("데이터 없음")
            return

        article_map = load_article_map(conn)

        result = add_professor_info_to_articles(articles, conn, article_map)

        # =========================
        #  논문 extra 추가 (여기만 추가됨)
        # =========================
        print("논문 extra 추가 시작")

        existing_titles = set()
        for item in result:
            title = item.get("THSS_NM")
            if isinstance(title, str):
                existing_titles.add(normalize(title))

        df_all = get_api_dataframe(CAT_ARTICLE, conn)
        if "YY" in df_all.columns:
            years = pd.to_numeric(df_all["YY"], errors="coerce")
            df_all = df_all[years >= 2015]

        extra_items = []

        for _, row in df_all.iterrows():
            title = normalize(row["THSS_NM"])

            if not title or title in existing_titles:
                continue

            emp_no = str(row["EMP_NO"]) if pd.notna(row["EMP_NO"]) else None

            if emp_no:
                professor_info = get_professor_info_by_emp_no(conn, emp_no)
            else:
                professor_info = None

            extra_items.append({
                "EMP_NO": emp_no,
                "THSS_NM": row["THSS_NM"],
                "abstract": None,
                "abstract_missing": True,
                "THSS_PATICP_GBN": row.get("THSS_PATICP_GBN"),
                "JRNL_GBN": row.get("JRNL_GBN"),
                "YY": row.get("YY"),
                "professor_info": professor_info
            })

        print(f"extra 추가: {len(extra_items)}")

        result = result + extra_items

        # =========================

        save_article_json(result)

        print(f"완료: {len(result)}개 처리됨")

    except Exception as e:
        print(e)
    finally:
        close_db_connection(conn)


if __name__ == "__main__":
    main()
