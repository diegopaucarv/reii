"""One-off: wrap the 5 top-level tab blocks in `if/elif tab_X.open:` guards.

Streamlit 1.45+ exposes `tab.open` (bool) on each tab container. Guarding each
`with tab_X:` block with `if/elif tab_X.open:` makes rendering lazy: only the
active tab's content executes on each rerun, instead of all five tabs.

Idempotent: if `if tab_a.open:` is already present, the script exits without
touching the file.

Usage:  python scripts/lazy_tabs.py [path-to-dashboard.py]
"""

import re
import sys

DEFAULT = "src/reii/dashboard.py"
TAB_ORDER = ["tab_a", "tab_b", "tab_c", "tab_d", "tab_e"]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    if any(re.match(r"^if tab_a\.open:\s*$", ln) for ln in lines):
        print("Already transformed; nothing to do.")
        return 0

    starts: dict = {}
    for i, line in enumerate(lines):
        m = re.match(r"^with (tab_[a-e]):\s*$", line)
        if m:
            starts[m.group(1)] = i

    missing = [t for t in TAB_ORDER if t not in starts]
    if missing:
        print(f"ERROR: missing tab blocks: {missing}")
        return 1

    # Footer: first column-0 'st.markdown(' after tab_e's start.
    footer_idx = None
    for i in range(starts["tab_e"] + 1, len(lines)):
        if lines[i].startswith("st.markdown("):
            footer_idx = i
            break
    if footer_idx is None:
        print("ERROR: footer not found after tab_e")
        return 1

    bounds = []
    for idx, t in enumerate(TAB_ORDER):
        end = starts[TAB_ORDER[idx + 1]] if idx < len(TAB_ORDER) - 1 else footer_idx
        bounds.append((t, starts[t], end))

    out: list = []
    cursor = 0
    for idx, (t, start, end) in enumerate(bounds):
        out.extend(lines[cursor:start])
        kw = "if" if idx == 0 else "elif"
        out.append(f"{kw} {t}.open:\n")
        out.append(f"    with {t}:\n")
        for line in lines[start + 1 : end]:
            out.append("    " + line)
        cursor = end
    out.extend(lines[cursor:])

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)

    print(f"Transformed {len(bounds)} tab blocks:")
    for t, s, e in bounds:
        print(f"  {t}: lines {s + 1}-{e} ({e - s} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
