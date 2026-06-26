"""
특허 데이터 매핑 테스트
엑셀 파일의 '발명의 명칭'과 데이터베이스의 tech_nm을 매핑하여 개수를 확인합니다.
"""

import mariadb
import pandas as pd
from typing import List, Dict
from pathlib import Path
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from indigo_pipeline.collection.database import (
    CAT_EMPLOYEE,
    CAT_PATENT,
    EMPLOYEE_COLUMNS,
    get_db_connection,
    close_db_connection,
    get_api_dataframe,
    COL_PATENT_PROJECT_NAME,
    COL_PATENT_MBR_SN,
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


def get_patent_names_with_professor(conn: mariadb.Connection) -> List[Dict]:
    """
    데이터베이스에서 특허명(tech_nm)과 교수 정보를 가져옵니다.
    (tech_nm이 있고, v_emp1 테이블의 SQ와 매칭되는 것만)
    
    Args:
        conn: 데이터베이스 연결 객체
        
    Returns:
        [{"tech_nm": "...", "mbr_sn": "...", "professor_info": {...}}, ...] 형태의 리스트
    """
    patent_df = get_api_dataframe(CAT_PATENT, conn)
    emp_df = get_api_dataframe(CAT_EMPLOYEE, conn)

    df = patent_df[
        patent_df[COL_PATENT_PROJECT_NAME].notna()
        & (patent_df[COL_PATENT_PROJECT_NAME].astype(str).str.strip() != "")
        & patent_df[COL_PATENT_MBR_SN].notna()
        & (patent_df[COL_PATENT_MBR_SN].astype(str).str.strip() != "")
    ].copy()
    df["_mbr_key"] = df[COL_PATENT_MBR_SN].astype(str).str.strip()
    emp_df = emp_df.copy()
    emp_df["_sq_key"] = emp_df[COL_EMP_SQ].astype(str).str.strip()
    df = df.merge(emp_df[EMPLOYEE_COLUMNS + ["_sq_key"]], left_on="_mbr_key", right_on="_sq_key", how="inner")
    df = df.drop_duplicates(subset=[COL_PATENT_PROJECT_NAME, COL_PATENT_MBR_SN])
    
    # 딕셔너리 리스트로 변환
    patent_list = []
    for _, row in df.iterrows():
        if pd.notna(row[COL_PATENT_PROJECT_NAME]):
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
            
            patent_list.append({
                "tech_nm": str(row[COL_PATENT_PROJECT_NAME]).strip(),
                "mbr_sn": str(row[COL_PATENT_MBR_SN]) if pd.notna(row[COL_PATENT_MBR_SN]) else "",
                "professor_info": professor_info
            })
    
    return patent_list


def load_excel_data(excel_file: str) -> pd.DataFrame:
    """
    엑셀 파일을 읽어옵니다.
    
    Args:
        excel_file: 엑셀 파일 경로
        
    Returns:
        DataFrame
    """
    excel_path = Path(excel_file)
    if not excel_path.exists():
        print(f"[오류] 파일이 존재하지 않습니다: {excel_file}")
        return pd.DataFrame()
    
    print(f"[엑셀 파일 읽기] {excel_path.name}")
    try:
        # 첫 번째 행을 헤더로 사용
        df = pd.read_excel(excel_path, header=0, engine='openpyxl')
        print(f"  - 행 수: {len(df):,}개")
        print(f"  - 컬럼 수: {len(df.columns)}개")
        print(f"  - 컬럼 목록: {df.columns.tolist()[:10]}...")
        return df
    except Exception as e:
        print(f"  - 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def find_invention_title_column(df: pd.DataFrame) -> str:
    """
    '발명의 명칭' 컬럼을 찾습니다.
    
    Args:
        df: DataFrame
        
    Returns:
        컬럼명 또는 None
    """
    # 다양한 가능한 컬럼명 확인
    possible_names = ['발명의 명칭', '발명의명칭', 'inventionTitle', 'Invention Title']
    
    for col in df.columns:
        col_str = str(col).strip()
        # '발명'과 '명칭'이 모두 포함된 경우
        if '발명' in col_str and '명칭' in col_str:
            return col
        if col_str in possible_names:
            return col
    
    # 찾지 못한 경우 모든 컬럼 출력
    print(f"[경고] '발명의 명칭' 컬럼을 찾을 수 없습니다.")
    print(f"[사용 가능한 컬럼]")
    for i, col in enumerate(df.columns):
        print(f"  {i}: {col}")
    return None


def match_patent_data(db_patents: List[Dict], excel_df: pd.DataFrame, invention_title_col: str) -> int:
    """
    데이터베이스의 tech_nm과 엑셀의 '발명의 명칭'을 매핑합니다.
    
    Args:
        db_patents: 데이터베이스 특허 데이터 리스트
        excel_df: 엑셀 DataFrame
        invention_title_col: '발명의 명칭' 컬럼명
        
    Returns:
        매핑된 데이터 개수
    """
    matched_count = 0
    
    # 데이터베이스 특허명을 딕셔너리로 인덱싱 (중복 허용)
    db_patents_by_name = {}
    for patent in db_patents:
        tech_nm = patent.get("tech_nm", "").strip()
        if tech_nm:
            if tech_nm not in db_patents_by_name:
                db_patents_by_name[tech_nm] = []
            db_patents_by_name[tech_nm].append(patent)
    
    print(f"\n[매핑 시작]")
    print(f"  - 데이터베이스 특허명 개수: {len(db_patents_by_name):,}개")
    print(f"  - 엑셀 데이터 행 수: {len(excel_df):,}개")
    print(f"  - 매핑 키: DB.tech_nm <-> Excel.{invention_title_col}")
    
    # 엑셀 데이터와 매핑
    for idx, excel_row in excel_df.iterrows():
        excel_invention_title = excel_row.get(invention_title_col)
        
        # 발명의 명칭이 없는 엑셀 행은 건너뛰기
        if pd.isna(excel_invention_title) or not str(excel_invention_title).strip():
            continue
        
        excel_invention_title_clean = str(excel_invention_title).strip()
        
        # 매칭되는 데이터베이스 특허 찾기
        matched_db_patents = db_patents_by_name.get(excel_invention_title_clean, [])
        
        if matched_db_patents:
            matched_count += len(matched_db_patents)
            if matched_count <= 5:  # 처음 5개만 상세 출력
                print(f"\n[매핑 발견 #{matched_count}]")
                print(f"  엑셀 발명의 명칭: {excel_invention_title_clean[:50]}...")
                print(f"  매칭된 DB 특허 수: {len(matched_db_patents)}개")
                for i, db_patent in enumerate(matched_db_patents[:3], 1):  # 최대 3개만
                    prof_name = db_patent.get("professor_info", {}).get("NM", "알 수 없음")
                    print(f"    [{i}] 교수: {prof_name}, 사번: {db_patent.get('mbr_sn', '')}")
    
    return matched_count


def main():
    """메인 함수"""
    conn = None
    
    try:
        # 데이터베이스 연결
        print("=" * 70)
        print("[1단계: 데이터베이스 연결]")
        print("=" * 70)
        conn = get_db_connection()
        
        # 데이터베이스에서 특허명과 교수 정보 가져오기
        print("\n[2단계: 데이터베이스 데이터 조회]")
        print("=" * 70)
        print("[쿼리] tech_nm과 교수 정보 조회 (교수 사번 매칭된 데이터만)")
        db_patents = get_patent_names_with_professor(conn)
        print(f"[조회 결과] 총 {len(db_patents):,}개의 특허명 조회됨")
        
        # 엑셀 파일 읽기
        print("\n[3단계: 엑셀 파일 읽기]")
        print("=" * 70)
        excel_file = "data/patent/kipris_inu_source_data.xlsx"
        excel_df = load_excel_data(excel_file)
        
        if excel_df.empty:
            print("[오류] 엑셀 데이터를 읽을 수 없습니다.")
            return
        
        # '발명의 명칭' 컬럼 찾기
        print("\n[4단계: 컬럼 확인]")
        print("=" * 70)
        invention_title_col = find_invention_title_column(excel_df)
        
        if not invention_title_col:
            print("[오류] '발명의 명칭' 컬럼을 찾을 수 없습니다.")
            return
        
        print(f"[확인] '발명의 명칭' 컬럼: {invention_title_col}")
        
        # 매핑 수행
        print("\n[5단계: 데이터 매핑]")
        print("=" * 70)
        matched_count = match_patent_data(db_patents, excel_df, invention_title_col)
        
        # 결과 출력
        print("\n" + "=" * 70)
        print("[매핑 결과]")
        print("=" * 70)
        print(f"[매핑된 데이터 개수] {matched_count:,}개")
        print(f"[데이터베이스 특허명 개수] {len(db_patents):,}개")
        print(f"[엑셀 데이터 행 수] {len(excel_df):,}개")
        if len(excel_df) > 0:
            print(f"[매핑률] {matched_count / len(excel_df) * 100:.1f}% (엑셀 데이터 대비)")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[오류] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        close_db_connection(conn)


if __name__ == "__main__":
    main()
