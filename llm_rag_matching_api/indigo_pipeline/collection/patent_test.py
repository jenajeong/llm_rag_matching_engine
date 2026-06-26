"""Patent name mapping smoke test.

This script compares patent names from the Indigo API collection helpers with
an Excel source file. It intentionally uses the same patent-professor join path
as the production collector.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from indigo_pipeline.collection.database import (
    close_db_connection,
    get_db_connection,
    get_patent_register_ids,
)


def normalize(text: Any) -> str:
    if text is None or pd.isna(text):
        return ""
    return str(text).replace(" ", "").strip().lower()


def get_patent_names_with_professor(conn: Any) -> List[Dict]:
    patents = []
    for row in get_patent_register_ids(conn):
        tech_name = str(row.get("tech_nm", "")).strip()
        if not tech_name:
            continue
        patents.append(
            {
                "tech_nm": tech_name,
                "mbr_sn": str(row.get("mbr_sn", "")),
                "professor_info": row.get("professor_info") or {},
            }
        )
    return patents


def load_excel_data(excel_file: str) -> pd.DataFrame:
    excel_path = Path(excel_file)
    if not excel_path.exists():
        print(f"[ERROR] Excel file not found: {excel_file}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(excel_path, header=0, engine="openpyxl")
        print(f"[Excel] rows={len(df):,}, columns={len(df.columns)}")
        print(f"[Excel] sample columns={df.columns.tolist()[:10]}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read Excel: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()


def find_invention_title_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "발명 명칭",
        "발명의 명칭",
        "발명명칭",
        "inventionTitle",
        "Invention Title",
        "tech_nm",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    for col in df.columns:
        col_text = str(col).lower()
        if "발명" in col_text or "invention" in col_text:
            return col

    return None


def match_patent_data(db_patents: List[Dict], excel_df: pd.DataFrame, invention_title_col: str) -> int:
    db_by_name: Dict[str, List[Dict]] = {}
    for patent in db_patents:
        key = normalize(patent.get("tech_nm"))
        if key:
            db_by_name.setdefault(key, []).append(patent)

    matched_count = 0
    for _, row in excel_df.iterrows():
        title = row.get(invention_title_col)
        key = normalize(title)
        if not key:
            continue

        matched = db_by_name.get(key, [])
        if matched:
            matched_count += 1
            if matched_count <= 5:
                print(f"\n[MATCH #{matched_count}] {str(title)[:80]}")
                for idx, patent in enumerate(matched[:3], 1):
                    prof_name = patent.get("professor_info", {}).get("NM", "")
                    print(f"  [{idx}] professor={prof_name}, mbr_sn={patent.get('mbr_sn', '')}")

    return matched_count


def main() -> None:
    conn = None
    try:
        print("=" * 70)
        print("[1] Indigo API patent data")
        print("=" * 70)
        conn = get_db_connection()
        db_patents = get_patent_names_with_professor(conn)
        print(f"[Indigo] patent names with professor info: {len(db_patents):,}")

        print("\n" + "=" * 70)
        print("[2] Excel source")
        print("=" * 70)
        excel_df = load_excel_data("data/patent/kipris_inu_source_data.xlsx")
        if excel_df.empty:
            return

        invention_title_col = find_invention_title_column(excel_df)
        if not invention_title_col:
            print("[ERROR] Invention title column not found.")
            return

        print(f"[Excel] invention title column: {invention_title_col}")

        print("\n" + "=" * 70)
        print("[3] Mapping result")
        print("=" * 70)
        matched_count = match_patent_data(db_patents, excel_df, invention_title_col)
        print(f"[Result] matched rows: {matched_count:,}")
        print(f"[Result] Indigo patents: {len(db_patents):,}")
        print(f"[Result] Excel rows: {len(excel_df):,}")
        if len(excel_df):
            print(f"[Result] match ratio: {matched_count / len(excel_df) * 100:.1f}%")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()
    finally:
        close_db_connection(conn)


if __name__ == "__main__":
    main()
