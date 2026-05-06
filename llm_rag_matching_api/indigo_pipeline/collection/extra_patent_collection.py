import json
from pathlib import Path
import pandas as pd
import sys

# =========================
# 프로젝트 루트
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR.parent))

from indigo_pipeline.collection.database import get_db_connection, close_db_connection
from indigo_pipeline.config import PATENT_DATA_FILE

# =========================
# JSON 경로
# =========================
PATENT_PATH = Path(PATENT_DATA_FILE)


# =========================
# 문자열 정규화
# =========================
def normalize(text):
    if text is None:
        return None
    return str(text).replace(" ", "").lower()


# =========================
# JSON 로드
# =========================
def load_json(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 기존 tech_nm set
# =========================
def get_existing_names(data):
    s = set()
    for item in data:
        name = item.get("tech_nm")
        if isinstance(name, str):
            s.add(normalize(name))
    return s


# =========================
# 교수 map
# =========================
def get_professor_map(conn):
    df = pd.read_sql("SELECT * FROM v_emp1", conn)

    prof_map = {}

    for _, r in df.iterrows():
        name = normalize(r["NM"])
        dept = normalize(r["HG_NM"])

        if not name:
            continue

        if name not in prof_map:
            prof_map[name] = []

        prof_map[name].append({
            "dept": dept,
            "data": {
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
        })

    return prof_map


# =========================
# 발명자 map (inu_tech_sn 기준)
# =========================
def build_inventor_map(conn):
    df = pd.read_sql("SELECT * FROM tb_inu_tech_invntr_v2026_0115", conn)

    inv_map = {}

    for _, r in df.iterrows():
        key = str(r["inu_tech_sn"])

        if key not in inv_map:
            inv_map[key] = []

        name = normalize(r["invntr_nm"])
        dept = normalize(r["invntr_co_nm"])

        inv_map[key].append((name, dept))

    return inv_map


# =========================
# 학과/소속 매칭
# =========================
def dept_match(inv_dept, prof_dept):
    if not inv_dept or not prof_dept:
        return False

    return (
        inv_dept == prof_dept
        or inv_dept in prof_dept
        or prof_dept in inv_dept
    )


# =========================
# 교수 매칭
# - 동명이인은 DB에서 자동 판단
# =========================
def match_professor(row, inv_map, prof_map):
    key = str(row["inu_tech_sn"])
    inventors = inv_map.get(key)

    if not inventors:
        return None

    for name, dept in inventors:

        if name not in prof_map:
            continue

        candidates = prof_map[name]

        # 이름이 유일한 교수 → 이름만으로 매칭
        if len(candidates) == 1:
            return candidates[0]["data"]

        # 동명이인 교수 → 이름 + 학과/소속까지 비교
        for c in candidates:
            if dept_match(dept, c["dept"]):
                return c["data"]

        # 동명이인인데 학과/소속 매칭 실패 → 잘못 매칭하지 않고 다음 발명자 확인
        continue

    return None


# =========================
# 변환
# =========================
def convert(row, professor_info):
    def safe(v):
        return str(v) if pd.notna(v) else None

    return {
        "inu_tech_sn": safe(row["inu_tech_sn"]),
        "mbr_sn": safe(row["mbr_sn"]),
        "tech_nm": safe(row["tech_nm"]),
        "ptnt_rgstr_id": safe(row["ptnt_rgstr_id"]),
        "tech_invnt_se": safe(row["tech_invnt_se"]),
        "tech_invnt_right_se": safe(row["tech_invnt_right_se"]),
        "inu_tech_stage_se": safe(row["inu_tech_stage_se"]),

        "kipris_abstract": None,
        "kipris_application_name": None,
        "kipris_application_date": None,
        "kipris_register_status": None,
        "kipris_application_number": None,
        "kipris_total_count": None,

        "professor_info": professor_info,
        "kipris_missing": True
    }


# =========================
# MAIN
# =========================
def main():
    conn = None

    try:
        print("📁 JSON 로드")
        json_data = load_json(PATENT_PATH)
        print(f"기존 JSON 개수: {len(json_data)}")

        existing_names = get_existing_names(json_data)

        conn = get_db_connection()

        print("📌 특허 전체 조회")
        patent_df = pd.read_sql("SELECT * FROM tb_inu_tech", conn)
        print(f"DB 전체 개수: {len(patent_df)}")

        print("📌 발명자 매핑 생성")
        inv_map = build_inventor_map(conn)

        print("📌 교수 매핑 생성")
        prof_map = get_professor_map(conn)

        new_items = []
        skipped = 0
        matched = 0

        for _, row in patent_df.iterrows():
            name = row["tech_nm"]

            if not isinstance(name, str):
                continue

            if normalize(name) in existing_names:
                skipped += 1
                continue

            professor_info = match_professor(row, inv_map, prof_map)

            if professor_info:
                matched += 1

            new_items.append(convert(row, professor_info))

        print(f"추가 데이터: {len(new_items)}")
        print(f"중복 스킵: {skipped}")
        print(f"교수 매칭 성공: {matched}")

        final_data = json_data + new_items

        with open(PATENT_PATH, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        print(f"최종 개수: {len(final_data)}")

    except Exception as e:
        print("❌ 오류:", e)

    finally:
        close_db_connection(conn)


if __name__ == "__main__":
    main()
