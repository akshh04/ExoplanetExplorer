"""Chromium smoke coverage against a local, synthetic NASA endpoint response."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8510
URL = f"http://127.0.0.1:{PORT}"


def check_dashboard(browser):
    page = browser.new_page(viewport={"width": 375, "height": 900})
    page.set_default_timeout(20000)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def click_navigation(label):
        radio = page.get_by_role("radio", name=label, exact=True)
        page.get_by_test_id("stRadioOption").filter(has=radio).click()

    page.goto(URL)
    expect(page.get_by_role("heading", name="Planet catalogue")).to_be_visible()
    page.wait_for_function("localStorage.getItem('exoplanet-explorer.preferences.v1') !== null")
    search_input = page.get_by_role("combobox", name="Find a planet")
    expect(search_input).to_be_visible()
    page.wait_for_function("document.querySelector('[data-testid=stApp]')?.dataset.testScriptState === 'notRunning'")
    # Streamlit may remount inputs as browser preferences hydrate on first load.
    for _ in range(3):
        page.keyboard.press("/")
        if search_input.evaluate("(input) => input === document.activeElement"):
            break
        page.wait_for_timeout(350)
    expect(search_input).to_be_focused()
    page.get_by_role("button", name="Filters", exact=True).click()
    drawer = page.get_by_role("dialog")
    expect(drawer).to_be_visible()
    expect(drawer.get_by_role("checkbox", name="Exclude planets with unknown distance")).not_to_be_checked()
    drawer.get_by_role("button", name="Apply filters").click()
    expect(drawer).to_have_count(0)

    for width in (320, 375, 768, 1440):
        page.set_viewport_size({"width": width, "height": 900})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), width
        assert min(page.locator(".st-key-explorer-navigation [data-testid=stRadioOption]")
                   .evaluate_all("(nodes) => nodes.map(node => node.getBoundingClientRect().height)")) >= 44

    click_navigation("Charts")
    expect(page).to_have_url(f"{URL}/?view=Trends")
    click_navigation("Orbits")
    expect(page).to_have_url(f"{URL}/?view=Orbits")
    expect(page.get_by_role("heading", name="Orbital period vs. planet size")).to_be_visible()
    page.go_back()
    expect(page.get_by_role("heading", name="The pace of discovery")).to_be_visible()
    page.go_forward()
    expect(page.get_by_role("heading", name="Orbital period vs. planet size")).to_be_visible()

    search = page.get_by_role("combobox", name="Find a planet")
    search.fill("Test World 8 b")
    page.get_by_role("option", name="Test World 8 b", exact=True).click()
    expect(page.get_by_role("heading", name="Individual planet profile")).to_be_visible()
    page.get_by_role("button", name="Save planet", exact=True).click()
    page.wait_for_function(
        "JSON.parse(localStorage.getItem('exoplanet-explorer.preferences.v1')).favorites.includes('Test World 8 b')",
    )
    click_navigation("Saved")
    expect(page.get_by_role("button", name="Preview Test World 8 b")).to_be_visible()
    page.reload()
    expect(page.get_by_role("button", name="Preview Test World 8 b")).to_be_visible()
    expect(page.get_by_test_id("stException")).to_have_count(0)
    assert not errors, errors
    page.close()


def main():
    with TemporaryDirectory() as directory:
        temporary = Path(directory)
        entrypoint = temporary / "fixture_dashboard.py"
        entrypoint.write_text(
            f"""import runpy
import sys
from pathlib import Path
from unittest.mock import patch
import requests

root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))
response = requests.Response()
response.status_code = 200
response.encoding = "utf-8"
response._content = (root / "tests/fixtures/archive.csv").read_bytes()
response._content_consumed = True
with patch("requests.get", return_value=response):
    runpy.run_path(str(root / "app.py"), run_name="__main__")
""",
            encoding="utf-8",
        )
        server = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(entrypoint),
             "--server.address=127.0.0.1", f"--server.port={PORT}",
             "--server.headless=true", "--browser.gatherUsageStats=false"],
            cwd=ROOT,
            env={**os.environ, "EXOPLANET_ARCHIVE_CACHE": str(temporary / "archive.json.gz")},
        )
        try:
            for _ in range(60):
                try:
                    with urlopen(f"{URL}/_stcore/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if server.poll() is not None:
                        raise RuntimeError("Streamlit exited before becoming ready")
                    time.sleep(0.25)
            else:
                raise RuntimeError("Streamlit health endpoint did not become ready")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path=os.environ.get("CHROMIUM_EXECUTABLE"))
                check_dashboard(browser)
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=10)
    print("Chromium dashboard checks passed.")


if __name__ == "__main__":
    main()
