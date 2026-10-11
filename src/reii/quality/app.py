"""Página independiente:  streamlit run src/reii/quality/app.py [-- --dsn <DSN>]"""

import argparse
import sys

import streamlit as st

from reii.quality.dashboard_panel import render_quality_panel
from reii.quality.store import QualityStore

st.set_page_config(page_title="REII · Calidad y niveles", layout="wide")
_ap = argparse.ArgumentParser()
_ap.add_argument("--dsn", default=None)
_args, _ = _ap.parse_known_args(sys.argv[1:])
render_quality_panel(QualityStore.open(_args.dsn) if _args.dsn else None)
