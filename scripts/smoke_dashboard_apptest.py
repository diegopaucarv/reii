"""AppTest smoke: run the full dashboard and assert no exception.

Needs PYTHONPATH=src (src-layout, not installed). The dashboard loads the
workflow snapshot from data/ and renders all tabs; this validates that the
recent performance fixes (cached corpus search, cached charts, lemma network
split, stats cache, _e_prepare re-derivation, paragraph card cleanup) do not
raise at import/render time.
"""

from streamlit.testing.v1 import AppTest


def main():
    at = AppTest.from_file("src/reii/dashboard.py", default_timeout=300)
    at.run()
    if at.exception:
        print("FAIL: dashboard AppTest raised an exception")
        for exc in at.exception:
            print(exc)
        raise SystemExit(1)
    print("PASS: dashboard AppTest ran without exceptions")
    print(f"  markdown elements: {len(at.markdown)}")
    print(f"  dataframes:        {len(at.dataframe)}")
    print(f"  buttons:           {len(at.button)}")


if __name__ == "__main__":
    main()
