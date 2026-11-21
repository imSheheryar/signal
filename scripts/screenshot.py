"""Capture the running Streamlit dashboard to a PNG."""
import sys

from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
out = sys.argv[2] if len(sys.argv) > 2 else "artifacts/dashboard_home.png"

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path="/usr/bin/chromium",
        args=["--no-sandbox", "--disable-gpu"],
    )
    page = browser.new_page(viewport={"width": 1440, "height": 1200}, device_scale_factor=1)
    page.goto(url, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(8000)
    page.screenshot(path=out, full_page=False)
    browser.close()
print(f"saved {out}")
