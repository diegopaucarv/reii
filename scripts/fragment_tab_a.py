"""One-off: wrap tab A's body in a @st.fragment called inside the tab.

Streamlit fragments can't render widgets into externally created containers
(tabs), so the fragment must be CALLED inside `with tab_a:` and its body must
use `st.*` directly (rendering into the fragment's own root container).

Transformation:
    BEFORE:
        if tab_a.open:
            with tab_a:
                <body at 8-space indent>

    AFTER:
        @st.fragment
        def _render_tab_a():
            <body at 4-space indent, no `with tab_a:`>

        if tab_a.open:
            with tab_a:
                _render_tab_a()

Idempotent: if `def _render_tab_a():` is already present, exits without
touching the file.

Usage:  python scripts/fragment_tab_a.py [path-to-dashboard.py]
"""

import re
import sys

DEFAULT = "src/reii/dashboard.py"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    if any(re.match(r"^def _render_tab_a\(\):\s*$", ln) for ln in lines):
        print("Already transformed; nothing to do.")
        return 0

    # Locate the tab A block: `if tab_a.open:` followed by `    with tab_a:`
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^if tab_a\.open:\s*$", line) and i + 1 < len(lines):
            if re.match(r"^    with tab_a:\s*$", lines[i + 1]):
                start = i
                break
    if start is None:
        print("ERROR: `if tab_a.open:` + `with tab_a:` block not found")
        return 1

    # Body starts after `with tab_a:` (line start+2), ends before `elif tab_b.open:`
    body_start = start + 2
    body_end = None
    for i in range(body_start, len(lines)):
        if re.match(r"^elif tab_b\.open:\s*$", lines[i]):
            body_end = i
            break
    if body_end is None:
        print("ERROR: `elif tab_b.open:` not found after tab A body")
        return 1

    # Verify every body line has >= 4 spaces of indentation (so de-indent is safe)
    for i in range(body_start, body_end):
        ln = lines[i]
        if ln.strip() and not ln.startswith("    "):
            print(f"ERROR: body line {i + 1} has < 4 spaces indent: {ln!r}")
            return 1

    out = lines[:start]
    out.append("@st.fragment\n")
    out.append("def _render_tab_a():\n")
    for i in range(body_start, body_end):
        out.append(lines[i][4:])  # de-indent by 4
    out.append("\n")
    out.append("if tab_a.open:\n")
    out.append("    with tab_a:\n")
    out.append("        _render_tab_a()\n")
    out.extend(lines[body_end:])

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)

    print(f"Transformed tab A block (lines {start + 1}-{body_end}):")
    print(f"  body: lines {body_start + 1}-{body_end} de-indented by 4")
    print("  call site added after body")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
