"""Measure @st.cache_data snapshot cache-hit cost (the dominant module-level cost)."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st  # noqa: E402

from reii.backend.dashboard_adapter import DashboardAdapter  # noqa: E402

DB = "data/workflow_data.db"


@st.cache_data(show_spinner=False)
def _load_snapshot(_version: str):
    adapter = DashboardAdapter(DB)
    return adapter.snapshot()


_version = DashboardAdapter(DB).version_hash()

# Cold call
t0 = time.perf_counter()
snap = _load_snapshot(_version)
t_cold = time.perf_counter() - t0
print(f"cold: {t_cold:.2f}s")

# Warm call (cache hit)
t0 = time.perf_counter()
snap2 = _load_snapshot(_version)
t_warm = time.perf_counter() - t0
print(f"warm (cache hit): {t_warm:.2f}s")

# Warm call again
t0 = time.perf_counter()
snap3 = _load_snapshot(_version)
t_warm2 = time.perf_counter() - t0
print(f"warm2 (cache hit): {t_warm2:.2f}s")
