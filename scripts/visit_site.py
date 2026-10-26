"""Open the hosted dashboard in a browser and verify that its data has loaded."""

import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from playwright.sync_api import Error, sync_playwright


URL = "https://exoexplorer.streamlit.app/"


def describe_page(page):
    """Expose useful cloud diagnostics without cookies or URL query strings."""
    details = []
    for frame in page.frames:
        try:
            details.append({
                "host": urlsplit(frame.url).hostname,
                "text": frame.locator("body").inner_text(timeout=3000)[:2500],
            })
        except Error:
            continue
    return json.dumps(details, ensure_ascii=False)


def visit(page, timeout=300):
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    deadline = time.monotonic() + timeout
    last_wake = last_retry = 0.0
    while time.monotonic() < deadline:
        # Community Cloud can wrap the app in an iframe. Locators on page alone
        # never reach its metrics or a wake-up control inside a child frame.
        for frame in page.frames:
            try:
                metric = frame.get_by_test_id("stMetric").filter(
                    has=frame.get_by_text("Planets shown", exact=True)
                )
                if metric.count() and metric.first.is_visible():
                    value = metric.first.get_by_test_id("stMetricValue").inner_text(timeout=3000)
                    if int(value.replace(",", "").strip()) > 0:
                        print(f"Dashboard ready: {value} planets loaded.", flush=True)
                        return value
                wake = frame.get_by_role(
                    "button", name=re.compile(r"yes, get this app back up|wake.*app", re.I)
                )
                if wake.count() and wake.first.is_visible() and time.monotonic() - last_wake > 30:
                    wake.first.click(timeout=10000)
                    last_wake = time.monotonic()
                    print("Clicked Streamlit's wake-up button.", flush=True)
                retry = frame.get_by_role("button", name="Try again", exact=True)
                if retry.count() and retry.first.is_visible() and time.monotonic() - last_retry > 60:
                    retry.first.click(timeout=10000)
                    last_retry = time.monotonic()
                    print("Retrying the app's data load.", flush=True)
            except (Error, ValueError):
                # Waking the app replaces frames and navigates the page.
                continue
        page.wait_for_timeout(1000)
    raise RuntimeError("Dashboard did not load its planet data. " + describe_page(page))


if __name__ == "__main__":
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        Path("visit-results").mkdir(exist_ok=True)
        try:
            for attempt in range(1, 4):
                try:
                    count = visit(page)
                    message = f"Cloud browser verified the live dashboard: {count} planets loaded."
                    print("::notice::" + message, flush=True)
                    break
                except (Error, RuntimeError) as error:
                    message = str(error)
                    print(f"Attempt {attempt} failed: {message}", flush=True)
                    page.screenshot(path=f"visit-results/attempt-{attempt}.png", full_page=True)
                    if attempt == 3:
                        # Check annotations remain readable through GitHub's API
                        # even when downloading the log archive is unavailable.
                        annotation = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
                        print("::error::" + annotation, flush=True)
                        raise
                    page.wait_for_timeout(15000)
        finally:
            browser.close()
