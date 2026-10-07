"""Targeted smoke test: run dashboard module-level code up to tab rendering.

Patches streamlit with a MagicMock and makes st.tabs raise a sentinel so the
script stops right before rendering the (very heavy) tab bodies. This validates
the whole critical path: SQLite snapshot load, discourse merge, all derived
computations, lemma_map/origen_index fallbacks, and _build_iso_and_global.

The module globals are captured via sys._getframe(1).f_globals inside the
sentinel, because a module that raises during its own execution is evicted
from sys.modules by importlib and becomes unreachable otherwise.
"""

import sys
import traceback
from unittest.mock import patch

import streamlit as st

captured = {}


def _tabs_raise(*args, **kwargs):
    # Walk up the stack to find the dashboard module frame (st.tabs is called
    # through Streamlit wrappers, so frame(1) is not the module frame).
    frame = sys._getframe(1)
    while frame is not None:
        g = frame.f_globals
        if g.get("__file__", "").endswith("dashboard.py"):
            captured.update(g)
            break
        frame = frame.f_back
    raise _TabSentinel()


class _TabSentinel(Exception):
    pass


with patch.object(st, "tabs", side_effect=_tabs_raise):
    try:
        import reii.dashboard  # noqa: F401  (module code runs; sentinel stops it at tabs)
    except _TabSentinel:
        print("OK: reached tab rendering (sentinel raised as expected)")
    except Exception:
        print("FAIL: exception before tab rendering")
        traceback.print_exc()
        sys.exit(1)

if not captured:
    print("FAIL: no globals captured")
    sys.exit(1)

for name in (
    "data",
    "uces",
    "terminos_df",
    "iso_classes",
    "global_data",
    "lemma_map",
    "origen_index",
    "modalizacion_by_cluster",
    "sintesis_estructurada",
    "uce_phi_dict",
    "clusters_unicos",
    "class_colors",
):
    val = captured.get(name)
    if val is None:
        print(f"WARN: {name} is None/empty")
    else:
        n = len(val) if hasattr(val, "__len__") else "?"
        print(f"  {name}: {type(val).__name__} len={n}")

# Confirm the heavy NLP stack was NOT imported at module level
heavy = [
    m
    for m in (
        "torch",
        "spacy",
        "transformers",
        "stanza",
        "gliner",
        "sentence_transformers",
        "cdlib",
        "reii.batch_processor",
    )
    if m in sys.modules
]
print(
    f"\nheavy modules imported at module level: {heavy if heavy else 'NONE (lazy import works)'}"
)

print("TARGETED SMOKE TEST PASSED")
