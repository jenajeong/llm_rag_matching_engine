from playwright.sync_api import sync_playwright
from datetime import datetime
import os
from indigo_pipeline.config import KIPRIS_PORTAL_ID, KIPRIS_PORTAL_PASSWORD

ID = KIPRIS_PORTAL_ID
PW = KIPRIS_PORTAL_PASSWORD

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # -----------------------------
        # 1. 메인 접속
        # -----------------------------
        page.goto("https://plus.kipris.or.kr/portal/main.do")

        # -----------------------------
        # 2. 로그인
        # -----------------------------
        page.click("text=로그인")
        page.fill("#mId", ID)
        page.fill("#mPass", PW)
        page.click(".submit-login")

        page.wait_for_timeout(3000)
        print("로그인 완료")

        # -----------------------------
        # 3. 메뉴 이동 (hover)
        # -----------------------------
        page.goto(
            "https://plus.kipris.or.kr/portal/data/request/apiFsmtmList.do?menuNo=290005",
            wait_until="domcontentloaded"
        )

        page.wait_for_timeout(5000)
        print("Open API 페이지 진입")

        # -----------------------------
        # 4. API 선택 (체크박스)
        # -----------------------------
        page.locator(".checkBox").nth(1).click()

        # -----------------------------
        # 5. 장바구니 (confirm 처리 포함)
        # -----------------------------
        page.on("dialog", lambda dialog: dialog.accept())
        page.click("text=장바구니")

        page.wait_for_timeout(2000)

        # -----------------------------
        # 6. 장바구니 페이지 이동
        # -----------------------------
        page.click("a[href*='cartList.do']")
        page.wait_for_timeout(5000)
        print("장바구니 페이지 진입")

        # -----------------------------
        # 7. 무료 선택
        # -----------------------------
        page.select_option("#motnTgtTpcd", value="KP242")
        page.wait_for_timeout(1000)

        # -----------------------------
        # 8. 이용기간 설정 (현재년도 12/31)
        # -----------------------------
        year = datetime.now().year
        end_date = f"{year}1231"

        page.evaluate(f"""
            document.querySelector('#end_date').value = '{end_date}';
            document.querySelector('#end_date').dispatchEvent(new Event('change'));
        """)

        page.wait_for_timeout(1000)

        # -----------------------------
        # 9. API 체크
        # -----------------------------
        checkbox = page.locator("input[name='checkBox_API']").first
        if not checkbox.is_checked():
            checkbox.check()

        # -----------------------------
        # 10. 서비스명 입력
        # -----------------------------
        page.fill("#utilSvcNm", "해당없음")

        # -----------------------------
        # 11. 약관 동의
        # -----------------------------
        agree = page.locator("#require")
        if not agree.is_checked():
            agree.check()

        # -----------------------------
        # 12. 신청하기 (dialog 처리 포함)
        # -----------------------------
        page.on("dialog", lambda dialog: dialog.accept())
        page.click(".shopRequest")

        page.wait_for_timeout(5000)

        print("API 신청 자동화 완료")

        input("확인 후 종료")
        browser.close()


if __name__ == "__main__":
    run()
