"""Runs scripts/test_fragment_tab.py under AppTest and asserts initial render.

NOTE: AppTest does NOT support fragment-scoped reruns (clicking a widget inside
a fragment triggers a full app rerun in AppTest). So we only validate:
1. The fragment renders inside the tab without errors.
2. The fragment's content appears in the active tab.
"""

from streamlit.testing.v1 import AppTest


def main():
    at = AppTest.from_file("scripts/test_fragment_tab.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Initial run raised: {at.exception}"
    assert at.session_state["app_runs"] == 1, at.session_state["app_runs"]
    assert at.session_state["frag_runs"] == 1, at.session_state["frag_runs"]
    texts = [t.value for t in at.markdown]
    assert any("frag_runs=1" in t for t in texts), texts
    print("PASS: initial run — fragment rendered inside tab A, no errors")

    # Click the button inside the fragment: AppTest does a FULL rerun (known
    # limitation), but the important thing is there's no exception and the
    # fragment re-renders correctly (no accumulation, no path-policy error).
    at.button[0].click()
    at.run()
    assert not at.exception, f"After click raised: {at.exception}"
    texts = [t.value for t in at.markdown]
    frag_lines = [t for t in texts if "frag_runs=" in t]
    assert len(frag_lines) == 1, f"Accumulation detected: {frag_lines}"
    print(f"PASS: click re-rendered fragment cleanly ({frag_lines[0]!r})")

    print("ALL PASS")


if __name__ == "__main__":
    main()
