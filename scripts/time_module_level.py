"""Measure the module-level derived computations in dashboard.py.

Loads the snapshot (cached) and times the sections that run on EVERY full
rerun: discourse merge, lemma_map build, and the various normalizations.
"""

import sys
import time
from unittest.mock import patch

import streamlit as st

captured = {}


def _tabs_raise(*args, **kwargs):
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


t0 = time.perf_counter()
with patch.object(st, "tabs", side_effect=_tabs_raise):
    try:
        import reii.dashboard  # noqa: F401
    except _TabSentinel:
        pass
t_total = time.perf_counter() - t0
print(f"TOTAL module-level (incl. first-run data load): {t_total:.2f}s")

# Second import: cache hit for snapshot, but derived code still runs
t0 = time.perf_counter()
with patch.object(st, "tabs", side_effect=_tabs_raise):
    try:
        import reii.dashboard  # noqa: F401  (re-import; snapshot cache is warm)
    except _TabSentinel:
        pass
t_reload = time.perf_counter() - t0
print(f"TOTAL module-level (re-import, cache hit): {t_reload:.2f}s")
