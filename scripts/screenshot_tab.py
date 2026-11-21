"""Capture a named Streamlit tab, e.g. python screenshot_tab.py 'Model performance' out.png."""
import sys

from playwright.sync_api import sync_playwright

tab = sys.argv[1]
out = sys.argv[2]

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path="/usr/bin/chromium",
        args=["--no-sandbox", "--disable-gpu"],
    )
    page = browser.new_page(viewport={"width": 1440, "height": 1200})
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(5000)
    page.get_by_role("tab", name=tab).click()
    page.wait_for_timeout(3000)
    page.screenshot(path=out, full_page=False)
    browser.close()
print(f"saved {out}")
