"""Minimal validation: @st.fragment called inside a tab, body uses st.* directly.

Verifies:
1. The fragment renders inside the tab (content appears in tab A).
2. Clicking a widget inside the fragment triggers a FRAGMENT rerun (not full app).
3. No StreamlitFragmentWidgetsNotAllowedOutsideError.
4. No element accumulation across fragment reruns.
"""

import streamlit as st

if "app_runs" not in st.session_state:
    st.session_state.app_runs = 0
if "frag_runs" not in st.session_state:
    st.session_state.frag_runs = 0

st.session_state.app_runs += 1


@st.fragment
def _render_tab_a():
    st.session_state.frag_runs += 1
    st.button("Click me", key="btn_a")
    st.write(f"frag_runs={st.session_state.frag_runs}")


tab_a, tab_b = st.tabs(["A", "B"], on_change="rerun")

if tab_a.open:
    with tab_a:
        _render_tab_a()
elif tab_b.open:
    with tab_b:
        st.write("Tab B content")

st.write(f"app_runs={st.session_state.app_runs}")
