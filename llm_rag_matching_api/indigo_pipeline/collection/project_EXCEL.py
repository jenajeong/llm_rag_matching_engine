from playwright.sync_api import sync_playwright, TimeoutError
from pathlib import Path
import pandas as pd
import json
import re
import sys
from indigo_pipeline.config import (
    INDIGO_BROWSER_HEADLESS,
    INDIGO_BROWSER_SLOW_MO,
    NTIS_ID,
    NTIS_PASSWORD,
    PROJECT_DATA_DIR,
)

print("현재 실행 Python:", sys.executable)

# =========================
# 로그인 정보
# =========================
ID = NTIS_ID
PW = NTIS_PASSWORD


# =========================
# 경로 설정
# =========================
PROJECT_DIR = Path(PROJECT_DATA_DIR)
PROJECT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 기관 목록
# =========================
ORG_LIST = [
    {
        "name": "인천대학교산학협력단",
        "url": "https://www.ntis.go.kr/orginfo/profile/profileMain.do?orgCd=0005161",
        "excel_path": PROJECT_DIR / "ntis_인천대학교산학협력단.xlsx",
        "json_path": PROJECT_DIR / "ntis_인천대학교산학협력단.json"
    },
    {
        "name": "인천대학교",
        "url": "https://www.ntis.go.kr/orginfo/profile/profileMain.do?orgCd=0009411",
        "excel_path": PROJECT_DIR / "ntis_인천대학교.xlsx",
        "json_path": PROJECT_DIR / "ntis_인천대학교.json"
    }
]


# =========================
# 컬럼명 정리
# =========================
def clean_column_name(col):
    col = str(col).strip()
    col = re.sub(r"\s+", " ", col)
    return col


# =========================
# Excel → JSON 변환
# =========================
def excel_to_json(excel_path, json_path):
    print(f"Excel 읽는 중: {excel_path}")

    # -----------------------------
    # 1. Excel 읽기
    # -----------------------------
    df = pd.read_excel(excel_path, engine="openpyxl")

    # -----------------------------
    # 2. 첫 번째 행을 컬럼명으로 사용
    # -----------------------------
    header = df.iloc[0].tolist()

    header = [
        clean_column_name(c) if pd.notna(c) else f"column_{i}"
        for i, c in enumerate(header)
    ]

    # -----------------------------
    # 3. 첫 번째 행 제거 후 컬럼명 재설정
    # -----------------------------
    df = df.iloc[1:].copy()
    df.columns = header

    # -----------------------------
    # 4. 완전히 빈 행 제거
    # -----------------------------
    df = df.dropna(how="all")

    # -----------------------------
    # 5. NO 컬럼 기준으로 실제 데이터만 남기기
    # -----------------------------
    if "NO" in df.columns:
        df = df[df["NO"].notna()]
        df = df[df["NO"].astype(str).str.strip() != ""]

    # -----------------------------
    # 6. NaN → None
    # -----------------------------
    df = df.where(pd.notna(df), None)

    # -----------------------------
    # 7. JSON 저장
    # -----------------------------
    data = df.to_dict(orient="records")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"JSON 저장 완료: {json_path}")
    print(f"   행 개수: {len(data)}")


# =========================
# NTIS 로그인
# =========================
def login_ntis(context):
    main_page = context.new_page()

    # -----------------------------
    # 1. NTIS 메인 접속
    # -----------------------------
    main_page.goto("https://www.ntis.go.kr/ThMain.do")
    main_page.wait_for_load_state("domcontentloaded")
    main_page.wait_for_timeout(3000)

    print("메인 접속:", main_page.url)

    # -----------------------------
    # 2. 시작하기 클릭
    # -----------------------------
    main_page.click(".btn-login-toggle")
    main_page.wait_for_timeout(1000)
    print("시작하기 클릭 완료")

    # -----------------------------
    # 3. 로그인 버튼 클릭
    # - 팝업으로 열리는 경우를 우선 처리
    # -----------------------------
    login_page = None
    is_popup_login = False

    try:
        with context.expect_page(timeout=8000) as popup_info:
            main_page.locator("a:has-text('로그인')").first.click()

        login_page = popup_info.value
        is_popup_login = True
        login_page.wait_for_load_state("domcontentloaded")
        print("로그인 팝업 열림:", login_page.url)

    except TimeoutError:
        print("팝업 감지 실패 → 같은 페이지 또는 이미 열린 로그인창 확인")

        try:
            main_page.locator("a:has-text('로그인')").first.click(timeout=5000)
        except Exception as e:
            print("로그인 버튼 클릭 재시도 실패:", e)

        main_page.wait_for_timeout(3000)
        login_page = main_page
        print("현재 페이지를 로그인 페이지로 사용:", login_page.url)

    # -----------------------------
    # 4. 로그인 입력창 대기
    # -----------------------------
    login_page.wait_for_selector("#userid", timeout=15000)
    login_page.wait_for_selector("#password", timeout=15000)
    login_page.wait_for_selector("#btnIdPasswordLogin", timeout=15000)

    print("로그인 입력창 확인")

    # -----------------------------
    # 5. ID / PW 입력
    # -----------------------------
    login_page.fill("#userid", ID)
    login_page.fill("#password", PW)

    login_page.wait_for_timeout(500)

    # -----------------------------
    # 6. 로그인 버튼 클릭
    # - 클릭 후 로그인창이 닫힐 수 있음
    # -----------------------------
    print("로그인 버튼 클릭 직전 URL:", login_page.url)

    try:
        login_page.locator("#btnIdPasswordLogin").click(force=True)
        print("로그인 버튼 클릭 완료")
    except Exception as e:
        print("로그인 버튼 클릭 실패:", e)

    # -----------------------------
    # 7. 로그인 처리 대기
    # -----------------------------
    main_page.wait_for_timeout(8000)

    # -----------------------------
    # 8. 로그인창 상태 확인
    # -----------------------------
    try:
        if login_page.is_closed():
            print("로그인창 닫힘")
        else:
            print("로그인창 아직 열려 있음:", login_page.url)
    except Exception as e:
        print("로그인창 상태 확인 실패:", e)

    # -----------------------------
    # 9. 메인 페이지 로그인 상태 확인
    # -----------------------------
    try:
        if is_popup_login:
            main_page.bring_to_front()
            main_page.reload()
            main_page.wait_for_timeout(5000)

            print("메인 페이지 URL:", main_page.url)
            print("메인 페이지 제목:", main_page.title())
        else:
            login_page.wait_for_timeout(5000)
            print("현재 페이지 URL:", login_page.url)
            print("현재 페이지 제목:", login_page.title())

    except Exception as e:
        print("메인 페이지 확인 실패:", e)

    return main_page


# =========================
# 기관 과제 Excel 다운로드
# =========================
def download_org_project_excel(page, org):
    print("\n==============================")
    print(f" 기관 처리 시작: {org['name']}")
    print("==============================")

    # -----------------------------
    # 1. 기관 프로필 페이지 접속
    # -----------------------------
    page.goto(org["url"])
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(5000)

    print("기관 페이지 접속:", page.url)

    # -----------------------------
    # 2. 전체(2010 ~ 2024) 클릭
    # -----------------------------
    page.wait_for_selector("button[name='searchYearAll']", timeout=15000)
    page.click("button[name='searchYearAll']")
    page.wait_for_timeout(3000)

    print("전체 기간 선택 완료")

    # -----------------------------
    # 3. 관련 과제 다운로드 클릭
    # -----------------------------
    page.wait_for_selector("#pjtExcelDown", timeout=15000)
    page.click("#pjtExcelDown")
    page.wait_for_timeout(2000)

    print("관련 과제 다운로드 버튼 클릭 완료")

    # -----------------------------
    # 4. 동의 체크
    # -----------------------------
    page.wait_for_selector("#layerAgreeAll", timeout=15000)

    agree = page.locator("#layerAgreeAll")
    if not agree.is_checked():
        agree.check()

    page.wait_for_timeout(1000)

    print("동의 체크 완료")

    # -----------------------------
    # 5. 최종 다운로드
    # -----------------------------
    page.wait_for_selector("#layerBtnDownload", timeout=15000)

    with page.expect_download(timeout=60000) as download_info:
        page.click("#layerBtnDownload")

    download = download_info.value
    download.save_as(str(org["excel_path"]))

    print(f" Excel 저장 완료: {org['excel_path']}")

    # -----------------------------
    # 6. Excel → JSON 변환
    # -----------------------------
    excel_to_json(org["excel_path"], org["json_path"])


# =========================
# MAIN
# =========================
def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=INDIGO_BROWSER_HEADLESS,
            slow_mo=INDIGO_BROWSER_SLOW_MO,
        )

        context = browser.new_context(
            accept_downloads=True
        )

        try:
            # -----------------------------
            # 1. 로그인
            # -----------------------------
            main_page = login_ntis(context)

            # -----------------------------
            # 2. 기관별 다운로드 + JSON 변환
            # -----------------------------
            for org in ORG_LIST:
                download_org_project_excel(main_page, org)

            print("\n 전체 완료")
            print(f"저장 폴더: {PROJECT_DIR}")

        except Exception as e:
            print("오류 발생:", e)

        finally:
            browser.close()
            print("브라우저 종료 완료")


if __name__ == "__main__":
    run()
