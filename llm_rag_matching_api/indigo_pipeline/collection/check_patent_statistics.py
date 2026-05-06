"""
특허 데이터 수집 과정의 단계별 통계 확인
각 단계에서 몇 개의 데이터가 있었고, 몇 개가 필터링되었는지 확인
"""

import mariadb
import pandas as pd
import sys
from pathlib import Path

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from indigo_pipeline.collection.database import (
    get_db_connection,
    close_db_connection,
    TABLE_PATENT,
    TABLE_INVENTOR,
    TABLE_EMPLOYEE,
    COL_PATENT_REGISTER_ID,
    COL_PATENT_MBR_SN,
    COL_PATENT_PROJECT_NAME,
    COL_EMP_NM,
    COL_EMP_HG_NM,
    COL_EMP_SQ
)


def check_patent_statistics():
    """특허 데이터 수집 과정의 단계별 통계를 확인합니다."""
    
    print("=" * 70)
    print("[특허 데이터 수집 과정 단계별 통계]")
    print("=" * 70)
    
    conn = None
    try:
        conn = get_db_connection()
        
        # 1단계: 특허 테이블 전체 데이터 개수
        print("\n[1단계] 특허 테이블 전체 데이터")
        query1 = f"SELECT COUNT(*) as total_count FROM {TABLE_PATENT}"
        df1 = pd.read_sql(query1, conn)
        total_records = int(df1.iloc[0]['total_count'])
        print(f"  - 전체 특허 데이터: {total_records:,}개")
        
        # 2단계: 등록번호(ptnt_rgstr_id)가 있는 데이터
        print("\n[2단계] 등록번호가 있는 데이터")
        query2 = f"""
            SELECT COUNT(*) as count_with_register_id
            FROM {TABLE_PATENT}
            WHERE {COL_PATENT_REGISTER_ID} IS NOT NULL 
                AND {COL_PATENT_REGISTER_ID} != ''
        """
        df2 = pd.read_sql(query2, conn)
        records_with_register_id = int(df2.iloc[0]['count_with_register_id'])
        print(f"  - 등록번호가 있는 데이터: {records_with_register_id:,}개")
        print(f"  - 필터링된 데이터: {total_records - records_with_register_id:,}개 (등록번호 없음)")
        print(f"  - 필터링률: {(total_records - records_with_register_id) / total_records * 100:.1f}%")
        
        # 3단계: 발명자 테이블과 조인 가능한 데이터 (mbr_sn 매칭)
        print("\n[3단계] 발명자 테이블과 조인 가능한 데이터 (mbr_sn 매칭)")
        query3 = f"""
            SELECT COUNT(DISTINCT t.{COL_PATENT_REGISTER_ID}) as matched_count
            FROM {TABLE_PATENT} t
            INNER JOIN {TABLE_INVENTOR} inv 
                ON CAST(t.{COL_PATENT_MBR_SN} AS CHAR) = CAST(inv.mbr_sn AS CHAR)
            WHERE t.{COL_PATENT_REGISTER_ID} IS NOT NULL 
                AND t.{COL_PATENT_REGISTER_ID} != ''
                AND inv.mbr_sn != 0
                AND inv.mbr_sn IS NOT NULL
        """
        df3 = pd.read_sql(query3, conn)
        records_with_inventor = int(df3.iloc[0]['matched_count'])
        print(f"  - 발명자 테이블과 조인 가능: {records_with_inventor:,}개")
        print(f"  - 필터링된 데이터: {records_with_register_id - records_with_inventor:,}개 (발명자 정보 없음 또는 mbr_sn=0)")
        print(f"  - 필터링률: {(records_with_register_id - records_with_inventor) / records_with_register_id * 100:.1f}%")
        
        # 4단계: invntr_se = 'A00' 필터링
        print("\n[4단계] 발명자 구분이 'A00'인 데이터")
        query4 = f"""
            SELECT COUNT(DISTINCT t.{COL_PATENT_REGISTER_ID}) as matched_count
            FROM {TABLE_PATENT} t
            INNER JOIN {TABLE_INVENTOR} inv 
                ON CAST(t.{COL_PATENT_MBR_SN} AS CHAR) = CAST(inv.mbr_sn AS CHAR)
            WHERE t.{COL_PATENT_REGISTER_ID} IS NOT NULL 
                AND t.{COL_PATENT_REGISTER_ID} != ''
                AND inv.mbr_sn != 0
                AND inv.mbr_sn IS NOT NULL
                AND inv.invntr_se = 'A00'
        """
        df4 = pd.read_sql(query4, conn)
        records_with_a00 = int(df4.iloc[0]['matched_count'])
        print(f"  - invntr_se = 'A00'인 데이터: {records_with_a00:,}개")
        print(f"  - 필터링된 데이터: {records_with_inventor - records_with_a00:,}개 (A00가 아닌 발명자)")
        if records_with_inventor > 0:
            print(f"  - 필터링률: {(records_with_inventor - records_with_a00) / records_with_inventor * 100:.1f}%")
        
        # 5단계: 교수 정보와 매칭 가능한 데이터 (이름 + 학과명 매칭)
        print("\n[5단계] 교수 정보와 매칭 가능한 데이터 (이름 + 학과명 매칭)")
        query5 = f"""
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
        df5 = pd.read_sql(query5, conn)
        records_matched_with_professor = int(df5.iloc[0]['matched_count'])
        print(f"  - 교수 정보 매칭 가능: {records_matched_with_professor:,}개")
        print(f"  - 필터링된 데이터: {records_with_a00 - records_matched_with_professor:,}개 (교수 정보 매칭 실패)")
        if records_with_a00 > 0:
            print(f"  - 필터링률: {(records_with_a00 - records_matched_with_professor) / records_with_a00 * 100:.1f}%")
        
        # 최종 요약
        print("\n" + "=" * 70)
        print("[최종 요약]")
        print("=" * 70)
        print(f"1. 특허 테이블 전체: {total_records:,}개")
        print(f"2. 등록번호가 있는 데이터: {records_with_register_id:,}개")
        print(f"   → 필터링: {total_records - records_with_register_id:,}개 ({(total_records - records_with_register_id) / total_records * 100:.1f}%)")
        print(f"3. 발명자 테이블 조인 가능: {records_with_inventor:,}개")
        print(f"   → 필터링: {records_with_register_id - records_with_inventor:,}개 ({(records_with_register_id - records_with_inventor) / records_with_register_id * 100:.1f}%)")
        print(f"4. invntr_se = 'A00': {records_with_a00:,}개")
        if records_with_inventor > 0:
            print(f"   → 필터링: {records_with_inventor - records_with_a00:,}개 ({(records_with_inventor - records_with_a00) / records_with_inventor * 100:.1f}%)")
        print(f"5. 교수 정보 매칭 가능: {records_matched_with_professor:,}개")
        if records_with_a00 > 0:
            print(f"   → 필터링: {records_with_a00 - records_matched_with_professor:,}개 ({(records_with_a00 - records_matched_with_professor) / records_with_a00 * 100:.1f}%)")
        print(f"\n[최종 수집 가능 데이터]")
        print(f"  - KIPRIS API로 수집 가능한 특허: {records_matched_with_professor:,}개")
        print(f"  - 전체 대비 비율: {records_matched_with_professor / total_records * 100:.1f}%")
        print(f"  - 등록번호 있는 데이터 대비 비율: {records_matched_with_professor / records_with_register_id * 100:.1f}%")
        print("=" * 70)
        
    except mariadb.Error as e:
        print(f"[오류] 데이터베이스 오류: {e}")
    except Exception as e:
        print(f"[오류] 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            close_db_connection(conn)


if __name__ == "__main__":
    check_patent_statistics()
