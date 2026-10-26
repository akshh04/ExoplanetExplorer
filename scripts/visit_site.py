"""Open the hosted dashboard in a browser and verify that its data has loaded."""

import re
import time

from playwright.sync_api import sync_playwright


URL = "https://exoexplorer.streamlit.app/"


def visit(page, timeout=240):
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    deadline = time.monotonic() + timeout
    wake_button = page.get_by_role(
        "button", name=re.compile(r"yes, get this app back up|wake.*app", re.I)
    )
    while time.monotonic() < deadline:
        if page.get_by_text("Planets shown", exact=True).is_visible():
            print("Dashboard loaded: Planets shown metric is visible.")
            return
        if wake_button.count() and wake_button.first.is_visible():
            wake_button.first.click(timeout=10000)
        page.wait_for_timeout(1000)
    raise RuntimeError("Dashboard did not load its planet data within the timeout.")


if __name__ == "__main__":
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            visit(browser.new_page())
        finally:
            browser.close()
