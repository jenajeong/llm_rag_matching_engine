"""Print patent collection statistics from the Indigo API source."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from indigo_pipeline.collection.database import (
    CAT_INVENTOR,
    CAT_PATENT,
    COL_PATENT_REGISTER_ID,
    get_api_dataframe,
    get_db_connection,
    close_db_connection,
    get_patent_statistics,
)


def _non_empty_count(df, column):
    if column not in df.columns:
        return 0
    return int((df[column].notna() & (df[column].astype(str).str.strip() != "")).sum())


def check_patent_statistics():
    print("=" * 70)
    print("[Patent collection statistics - Indigo API]")
    print("=" * 70)

    conn = None
    try:
        conn = get_db_connection()
        patent_df = get_api_dataframe(CAT_PATENT, conn)
        inventor_df = get_api_dataframe(CAT_INVENTOR, conn)
        stats = get_patent_statistics(conn)

        total_records = stats["total_records"]
        records_with_register_id = stats["records_with_register_id"]
        records_matched_with_professor = stats["records_matched_with_professor"]

        inventor_mbr_count = _non_empty_count(inventor_df, "mbr_sn")
        inventor_a00_count = (
            int((inventor_df["invntr_se"].astype(str).str.strip() == "A00").sum())
            if "invntr_se" in inventor_df.columns
            else 0
        )

        print(f"Patent rows: {total_records:,}")
        print(f"Rows with {COL_PATENT_REGISTER_ID}: {records_with_register_id:,}")
        print(f"Inventor rows with mbr_sn: {inventor_mbr_count:,}")
        print(f"Inventor rows with invntr_se=A00: {inventor_a00_count:,}")
        print(f"Patent rows matched with professor info: {records_matched_with_professor:,}")

        if total_records:
            print(f"Final usable ratio / total: {records_matched_with_professor / total_records * 100:.1f}%")
        if records_with_register_id:
            print(
                "Final usable ratio / rows with register id: "
                f"{records_matched_with_professor / records_with_register_id * 100:.1f}%"
            )

        missing_register = total_records - records_with_register_id
        print(f"Rows without register id: {missing_register:,}")
        print("=" * 70)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()
    finally:
        close_db_connection(conn)


if __name__ == "__main__":
    check_patent_statistics()
