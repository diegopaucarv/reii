"""Real-runtime validation of Tab E (run_every fragment + CRUD split) via Playwright.

Starts the real Streamlit server, opens the app in headless Chromium, clicks
Tab E, and verifies:
1. The controls view renders ("🚀 Workflows de análisis").
2. The CRUD editor renders ("⚙️ Configuración de agentes IA") — separate fragment.
3. No Streamlit exceptions in the app.
4. The fragment re-runs every 2s (run_every) — check via the "⏳ Ejecutando"
   indicator appearing when a run is started (best-effort; a real run takes
   minutes, so we only verify the controls + CRUD + no exceptions here).
"""

import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

PORT = 8601
URL = f"http://localhost:{PORT}"


def main():
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/reii/dashboard.py",
            "--server.headless",
            "true",
            "--server.port",
            str(PORT),
            "--browser.gatherUsageStats",
            "false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            errors = []
            page.on(
                "console",
                lambda msg: (
                    errors.append(f"console: {msg.text}")
                    if msg.type == "error"
                    else None
                ),
            )
            page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))

            # Wait for server to be up
            for _ in range(60):
                try:
                    page.goto(URL, wait_until="domcontentloaded", timeout=5000)
                    break
                except Exception:
                    time.sleep(1)
            else:
                print("FAIL: server never came up")
                sys.exit(1)

            # Wait for the app to render (snapshot load ~1-2s, then header)
            page.wait_for_selector("text=ALCESTE", timeout=120_000)
            print("OK: app rendered (header found)")

            # Click Tab E
            tab_e = page.get_by_text("E · Discurso por clase", exact=True)
            tab_e.click()
            time.sleep(3)

            # Verify controls view
            controls = page.get_by_text("🚀 Workflows de análisis", exact=True)
            if controls.count() == 0:
                print("FAIL: controls view not found in Tab E")
                body = page.locator("body").inner_text()
                sys.stdout.buffer.write(
                    ("  page text sample: " + body[:800] + "\n").encode(
                        "utf-8", errors="replace"
                    )
                )
                sys.exit(1)
            print("OK: controls view rendered ('🚀 Workflows de análisis')")

            # Verify CRUD editor (separate fragment)
            crud = page.get_by_text("⚙️ Configuración de agentes IA", exact=True)
            if crud.count() == 0:
                print("FAIL: CRUD editor not found in Tab E")
                sys.exit(1)
            print("OK: CRUD editor rendered ('⚙️ Configuración de agentes IA')")

            # Check for Streamlit exception elements
            exc = page.locator(".stException")
            if exc.count() > 0:
                print(f"FAIL: {exc.count()} Streamlit exception(s) on page")
                print(exc.first.inner_text()[:800])
                sys.exit(1)
            print("OK: no Streamlit exceptions")

            # Check for the "Ejecutar" button (workflow run)
            run_btn = page.get_by_role("button", name="▶ Ejecutar")
            if run_btn.count() == 0:
                print("WARN: no '▶ Ejecutar' button found (no workflows in ia/?)")
            else:
                print(f"OK: {run_btn.count()} '▶ Ejecutar' button(s) found")

            if errors:
                print("WARN: console/page errors observed:")
                for e in errors[:10]:
                    print("  ", e[:200])
            else:
                print("OK: no console/page errors")

            print("RUNTIME VALIDATION PASSED")
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()
