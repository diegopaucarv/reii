"""Content-integrity check: _render_tab_e fragment vs backup's `with tab_e:` block.

The fragment was restructured:
- BEFORE view (controls) moved into `else:` branch with `return`.
- AFTER view (results) kept after the else.
- Manual polling block (time.sleep(2); st.rerun()) removed.
- CRUD editor moved to _render_tab_e_crud (separate fragment).

So we compare:
1. The RESULTS section of the fragment (after the `else: return`) against the
   backup's results section (after the old polling block).
2. The CONTROLS section against the backup's controls section.

Differences expected: line-wrapping from de-indentation, moved comment blocks,
removed polling block, removed CRUD section.
"""

import re
import sys
from difflib import SequenceMatcher

BACKUP = "scripts/dashboard.py.bak_lazy"
CURRENT = "src/reii/dashboard.py"


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def norm(text):
    """Strip whitespace-only differences (de-indentation, blank lines)."""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s:
            lines.append(s)
    return "\n".join(lines)


def extract_backup_tab_e(backup_text):
    lines = backup_text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == "with tab_e:":
            start = i
            break
    assert start is not None, "with tab_e: not found in backup"
    # The block runs to end of file (tab E is last)
    return "\n".join(lines[start + 1 :])


def extract_fragment(current_text, name):
    lines = current_text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith(f"def {name}("):
            start = i
            break
    assert start is not None, f"def {name} not found"
    # Find the end: next top-level def, decorator, or dispatch chain at column 0
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if (
            lines[i]
            and not lines[i][0].isspace()
            and lines[i].strip().startswith(("@", "def ", "if tab_a.open:"))
        ):
            end = i
            break
    return "\n".join(lines[start:end])


def ratio(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def main():
    backup = read(BACKUP)
    current = read(CURRENT)

    backup_tab_e = extract_backup_tab_e(backup)
    frag = extract_fragment(current, "_render_tab_e")
    crud = extract_fragment(current, "_render_tab_e_crud")

    # Split fragment into controls (before `return`) and results (after)
    frag_lines = frag.splitlines()
    ret_idx = None
    for i, ln in enumerate(frag_lines):
        if ln.strip() == "return":
            ret_idx = i
            break
    assert ret_idx is not None, "no bare `return` found in fragment"
    frag_controls = "\n".join(frag_lines[: ret_idx + 1])
    frag_results = "\n".join(frag_lines[ret_idx + 1 :])

    # Split backup into controls and results. The backup has the polling block
    # right after the "running" info section, then the results. Find the marker.
    b_lines = backup_tab_e.splitlines()
    # The results section starts after the polling block. Find "DESIGN TOKENS"
    # or the first results marker. In the backup, after the polling block there's
    # the results section. Let's find the "if ds_state["finished"]" equivalent.
    # Actually the backup structure was: controls, then polling, then results.
    # Find the polling block marker.
    poll_idx = None
    for i, ln in enumerate(b_lines):
        if "time.sleep(2)" in ln:
            poll_idx = i
            break
    assert poll_idx is not None, "polling block not found in backup"
    # Controls = everything before the polling block
    b_controls = "\n".join(b_lines[:poll_idx])
    # Results = everything after the polling block (skip the st.rerun() line)
    b_results = "\n".join(b_lines[poll_idx + 2 :])

    r_controls = ratio(frag_controls, b_controls)
    r_results = ratio(frag_results, b_results)

    print(f"controls similarity: {r_controls:.4f}")
    print(f"results similarity:  {r_results:.4f}")

    # CRUD section should be present in the backup's controls (it was at the end)
    crud_in_backup = "Configuración de agentes IA" in backup_tab_e
    print(f"CRUD section present in backup tab E: {crud_in_backup}")
    print(f"CRUD fragment has {len(crud.splitlines())} lines")

    ok = r_controls > 0.80 and r_results > 0.80
    print("CONTENT-INTEGRITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
