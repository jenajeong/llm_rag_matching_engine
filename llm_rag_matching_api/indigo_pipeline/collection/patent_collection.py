"""
KIPRIS ?뱁뿀 ?곗씠???섏쭛湲?
MariaDB??tb_inu_tech ?뚯씠釉붿뿉??ptnt_rgstr_id(?뱁뿀 ?깅줉踰덊샇)瑜?湲곕컲?쇰줈
KIPRIS?먯꽌 ?뱁뿀 ?곗씠?곕? ?섏쭛?섍퀬 JSON ?뚯씪濡???ν빀?덈떎.
?깅줉踰덊샇?먯꽌 ?섏씠??-)???쒓굅????寃?됲빀?덈떎.
"""

import requests
import time
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from pathlib import Path
import sys

# ?곸쐞 ?붾젆?좊━瑜?寃쎈줈??異붽?
sys.path.append(str(Path(__file__).parent.parent.parent))
from indigo_pipeline.collection.database import (
    get_db_connection,
    close_db_connection,
    get_patent_register_ids as load_patent_register_ids,
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
    """KIPRIS API collector."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    # ?꾩껜 ?깅줉踰덊샇 媛?몄삤???⑥닔濡?蹂寃?
    def get_patent_register_ids(self, conn, limit=None, verbose=False):
        result = load_patent_register_ids(conn, limit=limit, verbose=verbose)
        for item in result:
            item["ptnt_rgstr_id_clean"] = normalize_register_id(item.get("ptnt_rgstr_id"))
        return [item for item in result if item.get("ptnt_rgstr_id_clean")]

    def fetch_patent_data(self, register_id: str, mbr_sn: str = "", professor_info: Dict = None) -> Optional[Dict]:

        if not self.api_key:
            print("API ?ㅺ? ?ㅼ젙?섏? ?딆븯?듬땲??")
            return None

        try:
            url = "https://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getAdvancedSearch"

            params = {
                "registerNumber": register_id,
                "ServiceKey": self.api_key,
                "numOfRows": 10,
                "pageNo": 1
            }

            response = requests.get(url, params=params, timeout=30)

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
                    "kipris_application_name": item.findtext("inventionTitle", default=""),
                    "kipris_total_count": root.findtext(".//totalCount", default="1")
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
            print("[1?④퀎: ?곗씠?곕쿋?댁뒪 荑쇰━]")
            print("=" * 70)

            register_id_list = self.get_patent_register_ids(conn, limit, verbose=True)

            if not register_id_list:
                print("[寃쎄퀬] 泥섎━???뱁뿀 ?깅줉踰덊샇媛 ?놁뒿?덈떎.")
                return

            print(f"[QUERY] {len(register_id_list):,} patent register IDs found.")

            # 湲곗〈 JSON ?곗씠??濡쒕뱶 (?대? ?섏쭛???깅줉踰덊샇 ?뺤씤??
            patent_output_file = Path(PATENT_DATA_FILE)
            existing_complete_ids = set()
            existing_missing_map = {}
            existing_data = []

            if patent_output_file.exists():
                try:
                    with open(patent_output_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)

                        for idx, item in enumerate(existing_data):
                            rid = item.get("ptnt_rgstr_id")

                            if not rid:
                                rid = item.get("kipris_register_number")

                            rid = normalize_register_id(rid)

                            if not rid:
                                continue

                            has_kipris = (
                                item.get("kipris_register_number")
                                or item.get("kipris_application_number")
                                or item.get("kipris_application_name")
                                or item.get("kipris_abstract")
                            )

                            if has_kipris and not item.get("kipris_missing"):
                                existing_complete_ids.add(rid)
                            else:
                                if rid not in existing_missing_map:
                                    existing_missing_map[rid] = []
                                existing_missing_map[rid].append(idx)

                    print(f"[湲곗〈 ?곗씠?? {len(existing_complete_ids):,}媛쒖쓽 ?깅줉踰덊샇媛 ?대? 議댁옱?⑸땲??")
                    print(f"[湲곗〈 ?곗씠?? {len(existing_missing_map):,}媛쒖쓽 ?깅줉踰덊샇??KIPRIS ?뺣낫媛 鍮꾩뼱 ?덉뒿?덈떎.")
                except:
                    print("[湲곗〈 ?곗씠?? JSON 濡쒕뱶 ?ㅽ뙣 ??臾댁떆?섍퀬 吏꾪뻾")

            total = len(register_id_list)

            print("\n" + "=" * 70)
            print("[2?④퀎: KIPRIS API ?곗씠???섏쭛]")
            print("=" * 70)

            for idx, register_info in enumerate(register_id_list, 1):

                original_register_id = register_info["ptnt_rgstr_id"]
                clean_register_id = register_info["ptnt_rgstr_id_clean"]

                # 湲곗〈???섏쭛???깅줉踰덊샇??skip
                if clean_register_id in existing_complete_ids:
                    print(f"[SKIP] ?대? ?섏쭛?? {clean_register_id}")
                    continue

                print(f"[{idx}/{total}] 泥섎━ 以? {clean_register_id}")

                try:
                    kipris_data = self.fetch_patent_data(clean_register_id)

                    if kipris_data:
                        api_success_count += 1

                        if clean_register_id in existing_missing_map:
                            for data_idx in existing_missing_map[clean_register_id]:
                                existing_data[data_idx].update({
                                    "kipris_register_number": kipris_data.get("kipris_register_number"),
                                    "kipris_application_number": kipris_data.get("kipris_application_number"),
                                    "kipris_application_date": kipris_data.get("kipris_application_date"),
                                    "kipris_register_status": kipris_data.get("kipris_register_status"),
                                    "kipris_abstract": kipris_data.get("kipris_abstract"),
                                    "kipris_application_name": kipris_data.get("kipris_application_name"),
                                    "kipris_total_count": kipris_data.get("kipris_total_count"),
                                    "kipris_missing": False
                                })

                            print(f"[UPDATE] 湲곗〈 ?곗씠??KIPRIS 蹂닿컯 ?꾨즺: {clean_register_id}")
                        else:
                            collected_data.append(kipris_data)
                            print(f"[ADD] ?좉퇋 ?곗씠??異붽?: {clean_register_id}")

                    else:
                        api_fail_count += 1

                except Exception:
                    api_fail_count += 1
                    continue

                if idx < total:
                    time.sleep(1)

            print("\n" + "=" * 70)
            print("[STEP 3: SAVE]")
            print("=" * 70)

            final_data = existing_data + collected_data

            patent_output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(patent_output_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)

            print(f"[????꾨즺] 湲곗〈: {len(existing_data)} / ?좉퇋: {len(collected_data)} / 珥? {len(final_data)}")
            print(f"[API 寃곌낵] ?깃났: {api_success_count} / ?ㅽ뙣: {api_fail_count}")

        finally:
            close_db_connection(conn)


if __name__ == "__main__":
    collector = KIPRISCollector(api_key=KIPRIS_API_KEY)
    collector.collect_and_save(limit=None)
