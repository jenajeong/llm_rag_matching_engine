from playwright.sync_api import sync_playwright
from datetime import datetime
import os
from indigo_pipeline.config import (
    INDIGO_BROWSER_HEADLESS,
    KIPRIS_PORTAL_ID,
    KIPRIS_PORTAL_PASSWORD,
)

ID = KIPRIS_PORTAL_ID
PW = KIPRIS_PORTAL_PASSWORD

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=INDIGO_BROWSER_HEADLESS)
        context = browser.new_context()
        page = context.new_page()

        # -----------------------------
        # 1. 硫붿씤 ?묒냽
        # -----------------------------
        page.goto("https://plus.kipris.or.kr/portal/main.do")

        # -----------------------------
        # 2. 濡쒓렇??
        # -----------------------------
        page.click("text=로그인")
        page.fill("#mId", ID)
        page.fill("#mPass", PW)
        page.click(".submit-login")

        page.wait_for_timeout(3000)
        print("濡쒓렇???꾨즺")

        # -----------------------------
        # 3. 硫붾돱 ?대룞 (hover)
        # -----------------------------
        page.goto(
            "https://plus.kipris.or.kr/portal/data/request/apiFsmtmList.do?menuNo=290005",
            wait_until="domcontentloaded"
        )

        page.wait_for_timeout(5000)
        print("Open API ?섏씠吏 吏꾩엯")

       # dialog ?먮룞 泥섎━ ?깅줉
        def handle_dialog(dialog):
            print("?뚮┝李??댁슜:", dialog.message)
            dialog.accept()

        page.on("dialog", handle_dialog)

        # -----------------------------
        # 4. API ?좏깮 泥댄겕諛뺤뒪 ?대┃
        # -----------------------------
        checkbox = page.locator("div.checkBox").nth(1)
        checkbox.wait_for(state="visible", timeout=5000)
        checkbox.click(force=True)

        page.wait_for_timeout(500)

        # -----------------------------
        # 5. ?λ컮援щ땲 ?대┃
        # -----------------------------
        page.locator('a[href="javascript:cart();"]').click(force=True)

        page.wait_for_timeout(2000)
        # -----------------------------
        # 6. ?λ컮援щ땲 ?섏씠吏 ?대룞
        # -----------------------------
        page.click("a[href*='cartList.do']")
        page.wait_for_timeout(5000)
        print("?λ컮援щ땲 ?섏씠吏 吏꾩엯")

        # -----------------------------
        # 7. 臾대즺 ?좏깮
        # -----------------------------
        page.select_option("#motnTgtTpcd", value="KP242")
        page.wait_for_timeout(1000)

        # -----------------------------
        # 8. ?댁슜湲곌컙 ?ㅼ젙 (?꾩옱?꾨룄 12/31)
        # -----------------------------
        year = datetime.now().year
        end_date = f"{year}1231"

        page.evaluate(f"""
            document.querySelector('#end_date').value = '{end_date}';
            document.querySelector('#end_date').dispatchEvent(new Event('change'));
        """)

        page.wait_for_timeout(1000)

        # -----------------------------
        # 9. API 泥댄겕
        # -----------------------------
        checkbox = page.locator("input[name='checkBox_API']").first
        if not checkbox.is_checked():
            checkbox.check()

        # -----------------------------
        # 10. ?쒕퉬?ㅻ챸 ?낅젰
        # -----------------------------
        page.fill("#utilSvcNm", "해당없음")

        # -----------------------------
        # 11. ?쎄? ?숈쓽
        # -----------------------------
        agree = page.locator("#require")
        if not agree.is_checked():
            agree.check()

        # -----------------------------
        # 12. ?좎껌?섍린 (dialog 泥섎━ ?ы븿)
        # -----------------------------
        page.on("dialog", lambda dialog: dialog.accept())
        page.click(".shopRequest")

        page.wait_for_timeout(5000)

        print("API ?좎껌 ?먮룞???꾨즺")

        input("확인 후 종료")
        browser.close()


if __name__ == "__main__":
    run()
