import json
from pathlib import Path
import pandas as pd
import sys

# =========================
# 경로 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR.parent))

from indigo_pipeline.collection.database import (
    CAT_EMPLOYEE,
    CAT_PROJECT,
    get_api_dataframe,
    get_db_connection,
    close_db_connection,
)
from indigo_pipeline.config import PROJECT_DATA_FILE

PROJECT_PATH = Path(PROJECT_DATA_FILE)


# =========================
# JSON 로드
# =========================
def load_json(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 기존 PRJ_NO set
# =========================
def get_existing_ids(data):
    s = set()
    for item in data:
        key = item.get("PRJ_NO")
        if key:
            s.add(str(key))
    return s


# =========================
# 교수 map
# =========================
def get_professor_map(conn):
    df = get_api_dataframe(CAT_EMPLOYEE, conn)

    prof_map = {}

    for _, r in df.iterrows():
        emp_no = str(r["EMP_NO"]).strip()

        prof_map[emp_no] = {
            "SQ": str(r["SQ"]) if pd.notna(r["SQ"]) else None,
            "EMP_NO": r["EMP_NO"],
            "NM": r["NM"],
            "GEN_GBN": r["GEN_GBN"],
            "BIRTH_DT": str(r["BIRTH_DT"]) if pd.notna(r["BIRTH_DT"]) else None,
            "NAT_GBN": r["NAT_GBN"],
            "RECHER_REG_NO": r["RECHER_REG_NO"],
            "WKGD_NM": r["WKGD_NM"],
            "COLG_NM": r["COLG_NM"],
            "HG_NM": r["HG_NM"],
            "HOOF_GBN": r["HOOF_GBN"],
            "HANDP_NO": r["HANDP_NO"],
            "OFCE_TELNO": r["OFCE_TELNO"],
            "EMAIL": r["EMAIL"]
        }

    return prof_map


# =========================
# 변환 (DB → JSON)
# =========================
def convert(row, professor_info):
    def safe(v):
        return str(v) if pd.notna(v) else None

    data = {}

    for col in row.index:
        data[col] = safe(row[col])

    data["professor_info"] = professor_info

    return data


# =========================
# MAIN
# =========================
def main():
    conn = None

    try:
        print("📁 JSON 로드")
        json_data = load_json(PROJECT_PATH)
        print(f"기존 데이터: {len(json_data)}")

        existing_ids = get_existing_ids(json_data)
        print(f"기존 PRJ_NO: {len(existing_ids)}")

        conn = get_db_connection()

        print("📌 프로젝트 전체 조회")
        df = get_api_dataframe(CAT_PROJECT, conn)
        print(f"DB 개수: {len(df)}")

        print("📌 교수 매핑 로딩")
        prof_map = get_professor_map(conn)

        new_items = []
        skipped = 0
        matched = 0

        for _, row in df.iterrows():
            key = row["PRJ_NO"]

            if not key:
                continue

            key = str(key)

            if key in existing_ids:
                skipped += 1
                continue

            emp_id = str(row["PRJ_RSPR_EMP_ID"]) if pd.notna(row["PRJ_RSPR_EMP_ID"]) else None
            professor_info = prof_map.get(emp_id)

            if professor_info:
                matched += 1

            new_items.append(convert(row, professor_info))

        print(f"추가 데이터: {len(new_items)}")
        print(f"중복 스킵: {skipped}")
        print(f"교수 매칭: {matched}")

        final_data = json_data + new_items

        print(f"최종 개수 (중복 제거 없음): {len(final_data)}")

        with open(PROJECT_PATH, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        print(f"저장 완료: {len(final_data)}")

    except Exception as e:
        print("❌ 오류:", e)

    finally:
        close_db_connection(conn)


if __name__ == "__main__":
    main()
