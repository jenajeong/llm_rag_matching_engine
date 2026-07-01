"""
?뱦 EBSCO ?쇰Ц ?곗씠???섏쭛湲?

[?꾩껜 ?먮쫫]
1. MariaDB(v_emp1_3)?먯꽌 ?쇰Ц ?곗씠??議고쉶
2. ?쇰Ц ?쒕ぉ(THSS_NM) 湲곗? 以묐났 ?쒓굅
3. EBSCO 寃????議댁옱 ?щ? ?먮퀎
4. ?곸꽭 ?섏씠吏 吏꾩엯 ??硫뷀??곗씠??異붿텧
5. JSON ?꾩쟻 ???(以묎컙 ????ы븿)

[?듭떖 ?뱀쭠]
- ?쇰Ц ?⑥쐞 泥섎━ (THSS_NM 湲곗?)
- ?ъ떎??媛??援ъ“ (processed_titles)
- 以묎컙 ??μ쑝濡??곗씠???먯떎 諛⑹?
- ?ㅽ뙣/?먮윭??湲곕줉?섏뿬 ?ш???諛⑹?
"""

import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright
import time
import json
from pathlib import Path
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# ?곸쐞 ?붾젆?좊━瑜?寃쎈줈??異붽?
sys.path.append(str(Path(__file__).parent.parent.parent))
from indigo_pipeline.collection.database import get_db_connection, close_db_connection, get_article_data, COL_ARTICLE_EMP_NO, COL_ARTICLE_THSS_NM, COL_ARTICLE_PUBLSH_DT
from indigo_pipeline.config import (
    ARTICLE_DATA_FILE,
    ARTICLE_PAPER_NO_PROFESSOR_FILE,
    EBSCO_PASSWORD,
    EBSCO_USERNAME,
    INDIGO_BROWSER_HEADLESS,
)


BASE_DIR = Path(__file__).resolve().parent.parent

SEARCH_INPUT_SELECTOR = "input#search-input, input[data-auto='search-input'], input[type='search']"
INVALID_JOURNAL_LABELS = {
    "湲고??숈닠吏(鍮꾩젙湲곕컻?됲븰?좎?)",
    "기타학술지(비정기발간학술지)",
}


def has_collection_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    if pd.isna(value):
        return False
    return bool(str(value).strip())


def is_valid_article_metadata(row) -> bool:
    for column in ("THSS_PATICP_GBN", "JRNL_GBN"):
        if column not in row or not has_collection_value(row[column]):
            return False

    jrnl_gbn = str(row["JRNL_GBN"]).strip()
    return jrnl_gbn not in INVALID_JOURNAL_LABELS


def apply_article_prefilter(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    if "YY" in df.columns:
        years = pd.to_numeric(df["YY"], errors="coerce")
        df = df[years >= 2015].copy()
        print(f"[Article] filter-article 연도 조건 사전 적용: {before}건 -> {len(df)}건")
        before = len(df)

    missing_columns = [column for column in ("THSS_PATICP_GBN", "JRNL_GBN") if column not in df.columns]
    if missing_columns:
        print(f"[Article] 메타데이터 컬럼 없음: {missing_columns}. 메타데이터 사전 필터는 건너뜁니다.")
        return df

    df = df[df.apply(is_valid_article_metadata, axis=1)].copy()
    print(f"[Article] filter-article 메타데이터 조건 사전 적용: {before}건 -> {len(df)}건")
    return df.reset_index(drop=True)


def goto_search_page(page, search_url: str, timeout_ms: int = 60000) -> bool:
    try:
        page.goto(search_url, wait_until="domcontentloaded", timeout=timeout_ms)
        return True
    except PlaywrightTimeoutError as e:
        print(f"  [EBSCO] 검색 페이지 이동 시간 초과. 페이지 로딩을 중단하고 복구합니다: {e}")
        try:
            page.evaluate("window.stop()")
        except Exception:
            pass
        return False


def login_to_ebsco_if_needed(page) -> None:
    prompt_input = page.locator("input#prompt-input, input[data-auto='prompt-input']").first
    try:
        prompt_input.wait_for(state="visible", timeout=8000)
    except PlaywrightTimeoutError:
        return

    if not EBSCO_USERNAME or not EBSCO_PASSWORD:
        raise RuntimeError("EBSCO_USERNAME and EBSCO_PASSWORD are required for EBSCO login.")

    print("[EBSCO] 로그인이 필요합니다. 로그인합니다.")
    prompt_input.fill(EBSCO_USERNAME)
    prompt_input.press("Enter")

    password_input = page.locator("input[data-auto='password-input'], input[name='password-input']").first
    password_input.wait_for(state="visible", timeout=15000)
    password_input.fill(EBSCO_PASSWORD)
    password_input.press("Enter")

    try:
        page.wait_for_load_state("networkidle", timeout=30000)
    except PlaywrightTimeoutError:
        pass

    try:
        page.wait_for_selector(SEARCH_INPUT_SELECTOR, timeout=30000)
    except PlaywrightTimeoutError:
        print("[EBSCO] 로그인 요청 완료. 아직 검색창이 보이지 않습니다.")


def open_ebsco_context(browser, search_url: str):
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(15000)
    page.set_default_navigation_timeout(60000)
    goto_search_page(page, search_url)
    login_to_ebsco_if_needed(page)
    return context, page


def ensure_ebsco_search_ready(page, search_url: str, attempts: int = 2):
    for attempt in range(1, attempts + 1):
        login_to_ebsco_if_needed(page)
        try:
            search_input = page.locator(SEARCH_INPUT_SELECTOR).first
            search_input.wait_for(state="visible", timeout=8000)
            return search_input
        except PlaywrightTimeoutError:
            print(f"  [EBSCO] 검색창 준비 안 됨 ({attempt}/{attempts}). 검색 페이지를 다시 불러옵니다.")
            goto_search_page(page, search_url)
            login_to_ebsco_if_needed(page)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except PlaywrightTimeoutError:
                pass

    return None

# ==============================
# ?뵻 DB ?곌껐 諛??쇰Ц ?곗씠??議고쉶
# ==============================
# - 2015???댁긽 ?쇰Ц ?꾪꽣留?
# - 以묐났 ?쒓굅 + 理쒖떊???뺣젹
# - ?댄썑 寃??????곗씠?곕줈 ?ъ슜

def main() -> None:
    conn = None
    try:
        conn = get_db_connection()
    
        # 논문 데이터 조회 (2015년 이상)
        print("\n[Article] 논문 데이터 조회 중...")
        df_emp = get_article_data(conn, min_year=2015)
        df_emp = apply_article_prefilter(df_emp)
    
        print(f"[Article] 2015년 이후 필터링/중복 제거 완료: {len(df_emp)}건")
        print("[Article] 게재일자 기준 최신순 정렬 완료")
    
        # 확인
        print("\n[Article] (EMP_NO, THSS_NM) 쌍 개수:", len(df_emp))
        print("\n[Article] 미리보기 TOP 10")
        print(df_emp[[COL_ARTICLE_EMP_NO, COL_ARTICLE_THSS_NM, COL_ARTICLE_PUBLSH_DT]].head(10))

    except Exception as e:
        print("Indigo API 데이터 조회 실패!")
        print("오류:", e)
        sys.exit(1)

    EBSCO_URL = "https://research.ebsco.com/c/4zvbuh/search"

    # ==============================
    # 1. EMP_NO 留ㅽ븨 ?앹꽦 (以묐났 ?쒓굅 ?ы븿)
    # ==============================
    emp_map = {}
    for _, row in df_emp.iterrows():
        title = str(row[COL_ARTICLE_THSS_NM]).strip()
        emp_no = str(row[COL_ARTICLE_EMP_NO]).strip()

        if title:
            if title not in emp_map:
                emp_map[title] = set()  
            emp_map[title].add(emp_no)  


    # ==============================
    # 2. THSS_NM 湲곗? dedup
    # ==============================
    seen_titles = set()
    queries = []

    for title, emp_set in emp_map.items():
        if title not in seen_titles and title:
            seen_titles.add(title)
            queries.append({
                "THSS_NM": title,
                "EMP_NO": list(emp_set)  # ???ㅼ떆 list濡?蹂??
            })

    total_queries = len(queries)
    target_titles = {q["THSS_NM"] for q in queries}
    print(f"\n총 {total_queries}개의 논문 제목을 검색합니다.")

    # 湲곗〈 寃곌낵 ?뚯씪???덉쑝硫?濡쒕뱶 (以묎컙 ??μ슜)
    output_file = Path(ARTICLE_PAPER_NO_PROFESSOR_FILE)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing_results = []
    processed_titles = set()

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
            before_existing = len(existing_results)
            existing_results = [
                item for item in existing_results
                if str(item.get('THSS_NM', '')).strip() in target_titles
            ]
            removed_existing = before_existing - len(existing_results)
            if removed_existing:
                print(f"기존 결과 중 이번 수집 대상이 아닌 {removed_existing}건은 제외합니다.")
            for item in existing_results:
                if "EMP_NO" in item and not isinstance(item["EMP_NO"], list):
                    item["EMP_NO"] = [str(item["EMP_NO"])]
            processed_titles = set([str(item.get('THSS_NM', '')) for item in existing_results if item.get('THSS_NM')])
            print(f"기존 논문 결과 로드 완료: {len(existing_results)}건")
            print(f"이미 처리된 제목: {len(processed_titles)}건")
            remaining_titles = [q["THSS_NM"] for q in queries if q["THSS_NM"] not in processed_titles]
            print(f"이번 실행에서 새로 검색할 제목: {len(remaining_titles)}건")
            # 湲곗〈 ?뚯씪??EMP_NO媛 ?놁쑝硫?異붽? (?명솚?깆쓣 ?꾪빐)
            if existing_results and 'EMP_NO' not in existing_results[0].keys():
                print("  [경고] 기존 결과 파일에 EMP_NO 컬럼이 없습니다. 새 결과에는 EMP_NO가 포함됩니다.")
    except FileNotFoundError:
        print("기존 결과 파일이 없습니다. 새로 시작합니다.")
    except Exception as e:
        print(f"기존 결과 파일 로드 중 오류가 발생했습니다. 계속 진행합니다: {e}")

    results = existing_results.copy() if existing_results else []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=INDIGO_BROWSER_HEADLESS)
        context, page = open_ebsco_context(browser, EBSCO_URL)
        # input("寃??媛?ν븳 ?곹깭硫?Enter...")  # ?ㅻ뱶由ъ뒪 紐⑤뱶?먯꽌??遺덊븘??

        # 以묎컙 ???二쇨린 (5媛쒕쭏?????
        save_interval = 5
        start_idx = len(results)  # ?대? 泥섎━??媛쒖닔
    
        try:
            for i, q_dict in enumerate(queries, 1):
                q = q_dict["THSS_NM"]  # 寃?됱뼱??THSS_NM
                emp_no = q_dict["EMP_NO"]  # EMP_NO ??μ슜
            
                # ?대? 泥섎━???쇰Ц? 嫄대꼫?곌린
                if str(q) in processed_titles:
                    print(f"[{i}/{total_queries}] 이미 처리됨 - 건너뜀: {q[:50]}...")
                    continue
            
                print(f"[{i}/{total_queries}] Searching: {q}")

                try:
                    # ?꾩옱 URL ?뺤씤 - ?덊솕硫댁씠硫?寃???섏씠吏濡??대룞
                    current_url = page.url
                    if "search" not in current_url:
                        print("  [정보] 홈화면 감지, 검색 페이지로 이동 중...")
                        goto_search_page(page, EBSCO_URL)
                        login_to_ebsco_if_needed(page)
                        time.sleep(1.5)

                    search_input = ensure_ebsco_search_ready(page, EBSCO_URL)
                    if search_input is None:
                        print("  [EBSCO] 검색창 복구 실패. 새 브라우저 세션으로 한 번 더 시도합니다.")
                        try:
                            context.close()
                        except Exception:
                            pass
                        context, page = open_ebsco_context(browser, EBSCO_URL)
                        search_input = ensure_ebsco_search_ready(page, EBSCO_URL, attempts=1)
                        if search_input is None:
                            print("  [EBSCO] 검색창을 찾지 못했습니다. 실패로 기록하고 다음으로 진행합니다.")
                            results.append({
                                "EMP_NO": list(emp_no),
                                "THSS_NM": q,
                                "has_result": 0,
                                "ebsco_error": "search_input_unavailable"
                            })
                            processed_titles.add(str(q))
                            continue

                    # 寃???낅젰 ?꾨뱶 ?대━????寃??
                    try:
                        search_input.click()
                        time.sleep(0.5)
                        # ?꾩껜 ?좏깮 ????젣
                        page.keyboard.press("Control+A")
                        time.sleep(0.3)
                        page.keyboard.press("Backspace")
                        time.sleep(0.5)
                    except:
                        pass
                
                    # 寃?됱뼱 ?낅젰
                    search_input.fill(q, timeout=15000)
                    time.sleep(0.5)
                    search_input.press("Enter", timeout=10000)

                    # 寃곌낵 濡쒕뵫 ?湲?(??異⑸텇???湲??쒓컙)
                    try:
                        page.wait_for_load_state("networkidle", timeout=15000)
                    except PlaywrightTimeoutError:
                        print("  [정보] networkidle 대기 시간 초과. 현재 로드된 화면 기준으로 계속 진행합니다.")
                    time.sleep(3.0)  # SPA ?뚮뜑 ?덉젙??(?ㅻ뱶由ъ뒪 紐⑤뱶?먯꽌????湲??湲??꾩슂)
                
                    # 寃??寃곌낵 ?붿냼媛 ?섑????뚭퉴吏 紐낆떆?곸쑝濡??湲??쒕룄 (理쒕? 5珥?
                    try:
                        # 寃??寃곌낵 ?먮뒗 "寃곌낵 ?놁쓬" 硫붿떆吏媛 ?섑????뚭퉴吏 ?湲?
                        page.wait_for_selector('h3[data-auto="result-item-title"]', timeout=5000)
                    except:
                        # ?붿냼媛 ?섑??섏? ?딆븘??怨꾩냽 吏꾪뻾 (?대? networkidle濡?異⑸텇???湲고뻽?쇰?濡?
                        pass
                
                    time.sleep(1.0)  # 異붽? ?덉젙???湲?

                    # ?덊솕硫댁쑝濡?由щ떎?대젆?몃릺?덈뒗吏 ?ㅼ떆 ?뺤씤
                    current_url_after = page.url
                    if "search" not in current_url_after:
                        print("  [정보] 검색 후 홈화면으로 이동됨 - 검색 실패")
                        results.append({
                            "EMP_NO": list(emp_no),
                            "THSS_NM": q,
                            "has_result": 0
                        })
                        processed_titles.add(str(q))
                        continue

                    has_result = 0  # 湲곕낯媛? ?놁쓬

                    # ??寃??寃곌낵 ?덉쓬 ?뺤씤: ?щ윭 諛⑸쾿?쇰줈 ?쒕룄 (?ъ떆???ы븿)
                    # 諛⑸쾿 1: data-auto ?띿꽦 ?ъ슜 (媛???덉젙??
                    result_cnt_1 = page.locator('h3[data-auto="result-item-title"]').count()
                    result_cnt_link = page.locator('a[data-auto="result-item-title__link"]').count()
                
                    # 諛⑸쾿 2: mark ?쒓렇 ?뺤씤 (寃??寃곌낵媛 ?덉쓣 ???섏씠?쇱씠?몃릺??mark ?쒓렇)
                    # h3 ?대???mark ?쒓렇留??뺤씤?섏뿬 ???뺥솗?섍쾶 ?먮떒
                    result_cnt_mark_in_h3 = page.locator('h3[data-auto="result-item-title"] mark').count()
                    result_cnt_mark_all = page.locator('mark').count()
                
                    # 諛⑸쾿 3: ?대옒?ㅻ챸 ?⑦꽩 ?ъ슜 (遺遺??쇱튂 - 諛깆뾽??
                    result_cnt_2 = page.locator('div[class*="result-item-header__title"]').count()
                    result_cnt_3 = page.locator('div[class*="result-item-header"]').count()
                
                    # 諛⑸쾿 4: 寃??寃곌낵 媛쒖닔 ?띿뒪???뺤씤 ("寃??寃곌낵: X嫄?)
                    result_count_text = page.locator('text=/寃??寃곌낵.*嫄?').count()
                
                    # 泥?踰덉㎏ ?쒕룄?먯꽌 ?꾨Т寃껊룄 李얠? 紐삵븯硫?異붽? ?湲????ъ떆??
                    if (result_cnt_1 == 0 and result_cnt_link == 0 and result_cnt_mark_in_h3 == 0 and 
                        result_cnt_mark_all < 2 and result_count_text == 0):
                        time.sleep(2.0)  # 異붽? ?湲????ъ떆??
                        result_cnt_1 = page.locator('h3[data-auto="result-item-title"]').count()
                        result_cnt_link = page.locator('a[data-auto="result-item-title__link"]').count()
                        result_cnt_mark_in_h3 = page.locator('h3[data-auto="result-item-title"] mark').count()
                        result_cnt_mark_all = page.locator('mark').count()
                        result_cnt_2 = page.locator('div[class*="result-item-header__title"]').count()
                        result_cnt_3 = page.locator('div[class*="result-item-header"]').count()
                        result_count_text = page.locator('text=/寃??寃곌낵.*嫄?').count()
                
                    # 理쒖쥌 寃곌낵 媛쒖닔 ?먮떒
                    # h3 ?붿냼媛 ?덇굅?? 留곹겕媛 ?덉쑝硫?寃??寃곌낵媛 ?덈떎怨??먮떒
                    result_cnt = max(result_cnt_1, result_cnt_link, result_cnt_2, result_cnt_3)
                
                    # h3 ?대???mark ?쒓렇媛 1媛??댁긽 ?덉쑝硫??뺤떎??寃??寃곌낵媛 ?덈떎怨??먮떒
                    # (?쒓났??HTML 援ъ“: h3 ?덉뿉 ?щ윭 mark ?쒓렇媛 ?덉쓬)
                    if result_cnt_mark_in_h3 >= 1:
                        result_cnt = max(result_cnt, 1)
                    # ?꾩껜 mark ?쒓렇媛 2媛??댁긽 ?덉쑝硫?寃??寃곌낵媛 ?덉쓣 媛?μ꽦???믪쓬
                    elif result_cnt_mark_all >= 2:
                        result_cnt = max(result_cnt, 1)
                
                    # 寃??寃곌낵 媛쒖닔 ?띿뒪?멸? ?덉쑝硫?寃곌낵媛 ?덈떎怨??먮떒
                    if result_count_text > 0:
                        result_cnt = max(result_cnt, 1)
                
                    # 寃??寃곌낵 ?놁쓬 ?뺤씤: ?띿뒪??湲곕컲?쇰줈 ?뺤씤
                    no_result_cnt = page.locator('text=/No results|결과 없음|검색 결과 없음|철자를 확인/i').count()
                
                    # ?덊솕硫댁씤吏 ?뺤씤 (寃???낅젰 ?꾨뱶媛 ?녾굅??寃???섏씠吏媛 ?꾨땶 寃쎌슦)
                    is_home_page = "search" not in page.url or page.locator(SEARCH_INPUT_SELECTOR).count() == 0

                    # ?붾쾭源??뺣낫 異쒕젰
                    if result_cnt == 0 and no_result_cnt == 0:
                        print(f"  [디버그] h3={result_cnt_1}, link={result_cnt_link}, header_title={result_cnt_2}, header={result_cnt_3}, mark_in_h3={result_cnt_mark_in_h3}, mark_all={result_cnt_mark_all}, result_count_text={result_count_text}, no_result={no_result_cnt}, is_home={is_home_page}")

                    if is_home_page:
                        has_result = 0
                        print("  [정보] 홈화면으로 이동됨. 검색 실패로 처리합니다. (0)")
                        results.append({
                            "EMP_NO": list(emp_no),
                            "THSS_NM": q,
                            "has_result": 0
                        })
                        processed_titles.add(str(q))  # 泥섎━ ?꾨즺 ?쒖떆
                    elif no_result_cnt > 0:
                        has_result = 0
                        print("  [정보] 검색 결과 없음 (0)")
                        # 寃??寃곌낵 ?놁쓬??JSON?????(?ъ떎?????ㅼ떆 寃?됲븯吏 ?딅룄濡?
                        results.append({
                            "EMP_NO": list(emp_no),
                            "THSS_NM": q,
                            "has_result": 0
                        })
                        processed_titles.add(str(q))  # 泥섎━ ?꾨즺 ?쒖떆
                    elif result_cnt > 0 or result_cnt_mark_in_h3 >= 1 or result_cnt_mark_all >= 2 or result_count_text > 0:
                        has_result = 1
                        print(f"  [정보] 검색 결과 있음 (1) - h3={result_cnt_1}, link={result_cnt_link}, mark_in_h3={result_cnt_mark_in_h3}, mark_all={result_cnt_mark_all}, result_count_text={result_count_text}")
                    
                        # 寃??寃곌낵媛 ?덉쑝硫?泥?踰덉㎏ 寃곌낵???곸꽭 ?섏씠吏濡??대룞?섏뿬 硫뷀??곗씠??異붿텧
                        paper_metadata = {"EMP_NO": list(emp_no), "THSS_NM": q, "has_result": 1}
                    
                        try:
                            # 泥?踰덉㎏ 寃곌낵 留곹겕 李얘린
                            first_link = page.locator('a[data-auto="result-item-title__link"]').first
                            if first_link.count() > 0:
                                # ????뿉???닿린 (?먮뒗 ?꾩옱 ?섏씠吏?먯꽌 ?대룞)
                                href = first_link.get_attribute("href")
                                if href:
                                    # ?곷? 寃쎈줈瑜??덈? 寃쎈줈濡?蹂??
                                    if href.startswith("/"):
                                        detail_url = "https://research.ebsco.com" + href
                                    else:
                                        detail_url = href
                                
                                    # ?곸꽭 ?섏씠吏濡??대룞
                                    page.goto(detail_url, wait_until="domcontentloaded", timeout=30000)
                                    time.sleep(1.5)  # ?섏씠吏 濡쒕뵫 ?湲?
                                
                                    # 硫뷀??곗씠??異붿텧
                                    metadata_div = page.locator('div[data-auto="record-html-metadata"] article')

                                    # ?뵦 異붽?: abstract ?곹깭 珥덇린??
                                    abstract = None
                                    has_abstract = False

                                    if metadata_div.count() > 0:
                                        # JavaScript濡?硫뷀??곗씠??異붿텧 (???덉젙??
                                        metadata_dict = page.evaluate("""
                                            () => {
                                                const article = document.querySelector('div[data-auto="record-html-metadata"] article');
                                                if (!article) return {};
                                            
                                                const result = {};
                                                const h3Elements = article.querySelectorAll('h3');
                                            
                                                h3Elements.forEach(h3 => {
                                                    const key = h3.textContent.trim();
                                                    // h3 ?ㅼ쓬??泥?踰덉㎏ ul 李얘린
                                                    let nextSibling = h3.nextElementSibling;
                                                    while (nextSibling && nextSibling.tagName !== 'UL') {
                                                        nextSibling = nextSibling.nextElementSibling;
                                                    }
                                                
                                                    if (nextSibling && nextSibling.tagName === 'UL') {
                                                        const liElements = nextSibling.querySelectorAll('li');
                                                        const values = Array.from(liElements).map(li => li.textContent.trim()).filter(v => v);
                                                        if (values.length > 0) {
                                                            result[key] = values.length === 1 ? values[0] : values;
                                                        }
                                                    }
                                                });
                                            
                                                return result;
                                            }
                                        """)
                                    
                                        # 異붿텧??硫뷀??곗씠?곕? paper_metadata??異붽?
                                        paper_metadata.update(metadata_dict)
                                    
                                        # =========================
                                        # 珥덈줉 議댁옱 ?щ? ?먮떒 (DB 湲곕컲)
                                        # =========================
                                        abstract_candidates = ["珥덈줉", "Abstract", "Description", "?ㅻ챸", "?ㅻ챸(踰덉뿭??"]

                                        abstract_text = None

                                        for field in abstract_candidates:
                                            if field in paper_metadata and paper_metadata[field]:
                                                if isinstance(paper_metadata[field], list):
                                                    abstract_text = " ".join(paper_metadata[field]).strip()
                                                else:
                                                    abstract_text = str(paper_metadata[field]).strip()
                                                break

                                        has_abstract = abstract_text is not None

                                        if has_abstract:
                                            abstract = abstract_text  # ?뵦 異붽?
                                            paper_metadata["Abstract"] = abstract_text

                                    else:
                                        print("  [정보] EBSCO 메타데이터 없음. SCOPUS 확인으로 진행합니다.")

                                    # =========================
                                    # SCOPUS fallback
                                    # =========================
                                    if not has_abstract:
                                        print("  [정보] 초록 없음. SCOPUS 조회를 시도합니다.")

                                        scopus_page = None

                                        try:
                                            dropdown_btn = page.locator('button[data-auto="dropdown-button"]')

                                            if dropdown_btn.count() > 0:
                                                dropdown_btn.first.click()
                                                scopus_item = page.get_by_text("SCOPUS", exact=False).first
                                                try:
                                                    scopus_item.wait_for(state="visible", timeout=5000)
                                                except PlaywrightTimeoutError:
                                                    scopus_item = page.locator('[role="menuitem"]').filter(has_text="SCOPUS").first

                                                if scopus_item.count() > 0:
                                                    print("  [정보] SCOPUS 메뉴 발견")

                                                    with page.expect_popup(timeout=10000) as popup_info:
                                                        scopus_item.first.click()

                                                    scopus_page = popup_info.value
                                                    try:
                                                        scopus_page.wait_for_load_state("domcontentloaded", timeout=15000)
                                                    except PlaywrightTimeoutError:
                                                        print("  [정보] SCOPUS 페이지 로딩 대기 시간 초과. 현재 화면 기준으로 확인합니다.")

                                                    scopus_page.wait_for_selector('[data-testid="document-details-abstract"]', timeout=5000)

                                                    locator = scopus_page.locator('[data-testid="document-details-abstract"] span')

                                                    if locator.count() > 0:
                                                        abstract = " ".join(locator.all_inner_texts()).strip()
                                                        has_abstract = True  # ?뵦 異붽?
                                                        paper_metadata["Abstract_scopus"] = abstract
                                                        if not paper_metadata.get("Abstract"):
                                                            paper_metadata["Abstract"] = abstract
                                                        print("  [정보] SCOPUS 초록 추출 성공")

                                        except Exception as e:
                                            print(f"  [경고] SCOPUS 조회 실패: {e}")

                                        finally:
                                            try:
                                                for p in context.pages:
                                                    if p != page:
                                                        p.close()
                                            except:
                                                pass
                                
                                    # ==============================
                                    #  3李??щ·留?蹂닿컯 (EBSCO ?ъ떆??
                                    # ==============================
                                    # - SCOPUS源뚯? ?ㅽ뙣??寃쎌슦 異붽? ?쒕룄
                                    # - ?숈씪 ?섏씠吏?먯꽌 ?ы깘??(?뚮뜑留?吏?????
                                    if not has_abstract:  
                                        try:
                                            print("  [정보] 3차 재시도: EBSCO DOM 재확인")

                                            # 硫뷀??곗씠???곸뿭 ?ㅼ떆 ?먯깋
                                            metadata_retry = page.evaluate("""
                                                () => {
                                                    const article = document.querySelector('div[data-auto="record-html-metadata"] article');
                                                    if (!article) return {};

                                                    const result = {};
                                                    const h3Elements = article.querySelectorAll('h3');

                                                    h3Elements.forEach(h3 => {
                                                        const key = h3.textContent.trim();
                                                        let next = h3.nextElementSibling;

                                                        while (next && next.tagName !== 'UL') {
                                                            next = next.nextElementSibling;
                                                        }

                                                        if (next && next.tagName === 'UL') {
                                                            const values = Array.from(next.querySelectorAll('li'))
                                                                .map(li => li.textContent.trim())
                                                                .filter(v => v);

                                                            if (values.length > 0) {
                                                                result[key] = values.length === 1 ? values[0] : values;
                                                            }
                                                        }
                                                    });

                                                    return result;
                                                }
                                            """)

                                            # abstract ?ы솗??
                                            for field in ["珥덈줉", "Abstract", "Description"]:
                                                if field in metadata_retry and metadata_retry[field]:
                                                    if isinstance(metadata_retry[field], list):
                                                        abstract = " ".join(metadata_retry[field]).strip()
                                                    else:
                                                        abstract = str(metadata_retry[field]).strip()

                                                    has_abstract = True  # ?뵦 異붽?
                                                    paper_metadata["Abstract"] = abstract
                                                    print("  [정보] 3차 재시도에서 초록 발견")
                                                    break

                                        except Exception as e:
                                            print(f"  [경고] 3차 재시도 실패: {e}")

                                    extracted_fields = [k for k in paper_metadata.keys() if k not in ['THSS_NM', 'has_result', 'metadata_error', 'EMP_NO']]
                                    print(f"  [정보] 메타데이터 추출 완료: {len(extracted_fields)}개 필드 ({', '.join(extracted_fields[:3])}{'...' if len(extracted_fields) > 3 else ''})")
                                
                                else:
                                    print("  [경고] 결과 링크 href를 찾을 수 없음")
                                    paper_metadata["metadata_error"] = "결과 링크 href 없음"
                            else:
                                print("  [경고] 결과 링크를 찾을 수 없음")
                                paper_metadata["metadata_error"] = "결과 링크 요소 없음"
                            
                        except Exception as e:
                            print(f"  [경고] 메타데이터 추출 중 오류: {str(e)}")
                            paper_metadata["metadata_error"] = f"메타데이터 추출 오류: {str(e)}"
                            # ?ㅻ쪟媛 諛쒖깮?대룄 寃???섏씠吏濡??뚯븘媛湲??쒕룄
                            try:
                                if "search" not in page.url:
                                    goto_search_page(page, EBSCO_URL)
                                    login_to_ebsco_if_needed(page)
                                    time.sleep(0.5)
                            except:
                                pass
                    
                        results.append(paper_metadata)
                        processed_titles.add(str(q))  # 泥섎━ ?꾨즺 ?쒖떆
                    
                    else:
                        has_result = 0
                        print("  [경고] 결과 판별 실패(로딩/구조 문제) (0) - JSON에 저장하지 않음")
                        # 寃곌낵 ?먮퀎 ?ㅽ뙣??寃쎌슦 results??異붽??섏? ?딆쓬 (湲곕줉留??섍퀬 ??μ? ????
                        processed_titles.add(str(q))  # 泥섎━ ?꾨즺 ?쒖떆留?(以묐났 諛⑹?)

                    # 二쇨린?곸쑝濡?以묎컙 ???(?ㅻ쪟 諛쒖깮 ???곗씠???먯떎 諛⑹?)
                    # 寃곌낵 ?먮퀎 ?ㅽ뙣????ν븯吏 ?딆쑝誘濡?results?먮뒗 ??긽 ?좏슚???곗씠?곕쭔 ?덉쓬
                    current_processed = len(results) - start_idx
                    if current_processed > 0 and current_processed % save_interval == 0:
                        try:
                            # data ?대뜑?????
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(results, f, ensure_ascii=False, indent=2)
                            print(f"  [정보] 중간 저장 완료 ({len(results)}개 결과 저장됨)")
                        except Exception as e:
                            print(f"  [경고] 중간 저장 실패. 계속 진행합니다: {e}")

                    # ?ㅼ쓬 寃??以鍮?(?덊솕硫댁씠 ?꾨땺 ?뚮쭔 ?대━??
                    current_url_before_clear = page.url
                    if "search" in current_url_before_clear and page.locator(SEARCH_INPUT_SELECTOR).count() > 0:
                        try:
                            clear_btn = page.locator('button[aria-label="Clear"]')
                            if clear_btn.count() > 0:
                                clear_btn.click()
                                time.sleep(0.5)
                            else:
                                # Clear 踰꾪듉???놁쑝硫??섎룞?쇰줈 ?대━??
                                page.click(SEARCH_INPUT_SELECTOR)
                                time.sleep(0.3)
                                page.keyboard.press("Control+A")
                                time.sleep(0.2)
                                page.keyboard.press("Backspace")
                                time.sleep(0.3)
                        except Exception as e:
                            # ?대━???ㅽ뙣 ???섎룞?쇰줈 ?대━??
                            try:
                                page.click(SEARCH_INPUT_SELECTOR)
                                time.sleep(0.3)
                                page.keyboard.press("Control+A")
                                time.sleep(0.2)
                                page.keyboard.press("Backspace")
                                time.sleep(0.3)
                            except:
                                pass

                    time.sleep(2.0)  # 寃??媛꾧꺽 (?덈Т 鍮좊Ⅴ硫??쒕쾭媛 李⑤떒?????덉쓬, 2珥덈㈃ 異⑸텇)
                
                except Exception as e:
                    # 媛쒕퀎 ?쇰Ц 泥섎━ 以??ㅻ쪟 諛쒖깮 (?ㅼ쓬 ?쇰Ц?쇰줈 怨꾩냽 吏꾪뻾)
                    print(f"  [오류] 오류 발생. 다음 논문으로 진행합니다: {str(e)}")
                    error_result = {
                        "EMP_NO": list(emp_no),
                        "THSS_NM": q,
                        "has_result": 0,
                        "metadata_error": f"처리 중 오류: {str(e)}"
                    }
                    results.append(error_result)
                    processed_titles.add(str(q))  # 泥섎━ ?꾨즺 ?쒖떆
                
                    # 二쇨린?곸쑝濡????
                    current_processed = len(results) - start_idx
                    if current_processed > 0 and current_processed % save_interval == 0:
                        try:
                            # data ?대뜑?????
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(results, f, ensure_ascii=False, indent=2)
                            print(f"  [정보] 중간 저장 완료 ({len(results)}개 결과 저장됨)")
                        except Exception as save_error:
                            print(f"  [경고] 중간 저장 실패: {save_error}")
                
                    continue  # ?ㅼ쓬 ?쇰Ц?쇰줈 怨꾩냽

        except KeyboardInterrupt:
            print("\n\n사용자에 의해 중단되었습니다.")
            raise  # KeyboardInterrupt???ㅼ떆 諛쒖깮?쒖폒???뺤긽 醫낅즺 泥섎━
        
        except Exception as e:
            print(f"\n\n예상치 못한 오류 발생: {str(e)}")
            print("지금까지 수집한 데이터를 저장합니다...")
        
        finally:
            # 釉뚮씪?곗? 醫낅즺
            try:
                browser.close()
            except:
                pass
        
            # 理쒖쥌 寃곌낵 ???(?ㅻ쪟 諛쒖깮?대룄 ???
            # 寃곌낵 ?먮퀎 ?ㅽ뙣??results???ы븿?섏? ?딆쑝誘濡???ν븷 ?곗씠?곕쭔 ?덉쓬
            try:
                if results:
                    # data ?대뜑?????
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"\n최종 결과를 '{output_file}' 파일로 저장했습니다. (총 {len(results)}개)")
                    print("\n=== 검색 결과 및 메타데이터 예시 ===")
                    # JSON ?곗씠?곗쓽 泥섏쓬 10媛쒕쭔 異쒕젰
                    for i, item in enumerate(results[:10], 1):
                        print(f"{i}. {item.get('THSS_NM', 'N/A')[:50]}... (has_result: {item.get('has_result', 0)})")
                    if len(results) > 10:
                        print(f"... and {len(results) - 10} more")
                else:
                    print("\n저장할 결과가 없습니다.")
            except Exception as e:
                print(f"\n최종 저장 중 오류 발생: {e}")
                print("필요하면 results 변수의 데이터를 수동으로 확인하세요.")
            finally:
                # DB ?곌껐 醫낅즺
                close_db_connection(conn)


if __name__ == "__main__":
    main()

