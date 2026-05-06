"""
KIPRIS 특허 데이터 수집기
MariaDB의 tb_inu_tech 테이블에서 ptnt_rgstr_id(특허 등록번호)를 기반으로 
KIPRIS에서 특허 데이터를 수집하고 JSON 파일로 저장합니다.
등록번호에서 하이픈(-)을 제거한 후 검색합니다.
"""

import mariadb
import pandas as pd
import requests
import time
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from pathlib import Path
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from indigo_pipeline.collection.database import (
    get_db_connection, 
    close_db_connection,
    TARGET_TABLE, 
    COL_PATENT_REGISTER_ID
)
from indigo_pipeline.config import KIPRIS_API_KEY, PATENT_DATA_FILE

def normalize_register_id(value):
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    value = value.split(".")[0]
    value = value.replace("-", "").replace(" ", "")

    if len(value) > 9 and value.endswith("0000"):
        value = value[:-4]

    return value


class KIPRISCollector:
    """KIPRIS API를 사용하여 특허 데이터를 수집하는 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    # 전체 등록번호 가져오는 함수로 변경
    def get_patent_register_ids(self, conn, limit=None, verbose=False):

        query = f"""
        SELECT DISTINCT {COL_PATENT_REGISTER_ID}
        FROM {TARGET_TABLE}
        WHERE {COL_PATENT_REGISTER_ID} IS NOT NULL
        AND {COL_PATENT_REGISTER_ID} != ''
        """

        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql(query, conn)

        result = []
        for _, row in df.iterrows():
            raw_id = str(row[COL_PATENT_REGISTER_ID]).strip()
            clean_id = normalize_register_id(raw_id)

            if not clean_id:
                continue

            result.append({
                "ptnt_rgstr_id": raw_id,
                "ptnt_rgstr_id_clean": clean_id,
                "mbr_sn": "",
                "professor_info": {}
            })

        return result

    def fetch_patent_data(self, register_id: str, mbr_sn: str = "", professor_info: Dict = None) -> Optional[Dict]:

        if not self.api_key:
            print("API 키가 설정되지 않았습니다.")
            return None
        
        try:
            import urllib.parse
            encoded_register_id = urllib.parse.quote(register_id, safe='')

            url = (
                f"https://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getAdvancedSearch"
                f"?registerNumber={encoded_register_id}"
                f"&ServiceKey={self.api_key}"
                f"&numOfRows=10"
                f"&pageNo=1"
            )

            response = requests.get(url, timeout=30)

            if response.text.strip().startswith('<!DOCTYPE') or response.text.strip().startswith('<html'):
                return None

            try:
                root = ET.fromstring(response.content)
            except:
                root = ET.fromstring(response.text)

            success_yn = root.findtext(".//successYN", default="")
            result_msg = root.findtext(".//resultMsg", default="")

            if success_yn == "N" or (result_msg and "ERROR" in result_msg.upper()):
                return None

            items = root.findall(".//item")

            if items:
                item = items[0]

                result_data = {
                    "ptnt_rgstr_id": register_id,
                    "mbr_sn": mbr_sn,
                    "kipris_register_number": item.findtext("registerNumber", default=""),
                    "kipris_application_number": item.findtext("applicationNumber", default=""),
                    "kipris_application_date": item.findtext("applicationDate", default=""),
                    "kipris_register_status": item.findtext("registerStatus", default=""),
                    "kipris_abstract": item.findtext("astrtCont", default="").strip(),
                    "kipris_application_name": item.findtext("inventionTitle", default="")
                }

                if professor_info:
                    result_data["professor_info"] = professor_info

                return result_data
            else:
                return None

        except Exception:
            return None
    
    def collect_and_save(self, limit: Optional[int] = None):

        conn = None
        collected_data = []
        api_success_count = 0
        api_fail_count = 0
        
        try:
            conn = get_db_connection()
            
            print("\n" + "=" * 70)
            print("[1단계: 데이터베이스 쿼리]")
            print("=" * 70)
            
            register_id_list = self.get_patent_register_ids(conn, limit, verbose=True)
            
            if not register_id_list:
                print("[경고] 처리할 특허 등록번호가 없습니다.")
                return
            
            print(f"[쿼리 결과] 총 {len(register_id_list):,}개의 등록번호 조회됨")

            # 기존 JSON 데이터 로드 (이미 수집된 등록번호 확인용)
            patent_output_file = Path(PATENT_DATA_FILE)
            existing_ids = set()
            existing_data = []

            if patent_output_file.exists():
                try:
                    with open(patent_output_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)

                        for item in existing_data:
                            rid = item.get("ptnt_rgstr_id")

                            if not rid:
                                rid = item.get("kipris_register_number")

                            rid = normalize_register_id(rid)

                            if rid:
                                existing_ids.add(rid)

                    print(f"[기존 데이터] {len(existing_ids):,}개의 등록번호가 이미 존재합니다.")
                except:
                    print("[기존 데이터] JSON 로드 실패 → 무시하고 진행")

            total = len(register_id_list)

            print("\n" + "=" * 70)
            print("[2단계: KIPRIS API 데이터 수집]")
            print("=" * 70)

            for idx, register_info in enumerate(register_id_list, 1):

                original_register_id = register_info["ptnt_rgstr_id"]
                clean_register_id = register_info["ptnt_rgstr_id_clean"]

                # 기존에 수집된 등록번호는 skip
                if clean_register_id in existing_ids:
                    print(f"[SKIP] 이미 수집됨: {clean_register_id}")
                    continue

                print(f"[{idx}/{total}] 처리 중: {clean_register_id}")

                try:
                    kipris_data = self.fetch_patent_data(clean_register_id)

                    if kipris_data:
                        api_success_count += 1
                        collected_data.append(kipris_data)
                    else:
                        api_fail_count += 1

                except Exception:
                    api_fail_count += 1
                    continue

                if idx < total:
                    time.sleep(1)

            print("\n" + "=" * 70)
            print("[3단계: 저장]")
            print("=" * 70)

            final_data = existing_data + collected_data

            patent_output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(patent_output_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)

            print(f"[저장 완료] 기존: {len(existing_data)} / 신규: {len(collected_data)} / 총: {len(final_data)}")
            print(f"[API 결과] 성공: {api_success_count} / 실패: {api_fail_count}")

        finally:
            close_db_connection(conn)


if __name__ == "__main__":
    collector = KIPRISCollector(api_key=KIPRIS_API_KEY)
    collector.collect_and_save(limit=None)
