"""Panel Streamlit: selector de corridas y niveles, calibración configurable e indicadores.

Uso dentro de dashboard.py (pestaña nueva):  ``render_quality_panel()``
Uso independiente:                           ``streamlit run src/reii/quality/app.py``

Cambiar de corrida o de nivel es una consulta: nunca recalcula CDH, AFC ni φ.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reii.quality import calibration as cal
from reii.quality import consensus as cons
from reii.quality import metrics as M
from reii.quality import service
from reii.quality.store import QualityStore

FRAGILE_JACCARD = 0.60  # clases por debajo se marcan frágiles (criterio propuesto)


def _store(store: Optional[QualityStore]) -> QualityStore:
    if store is not None:
        return store
    if "_quality_store" not in st.session_state:
        st.session_state["_quality_store"] = QualityStore.open()
    return st.session_state["_quality_store"]


def _pct(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def _badge(text: str, color: str) -> None:
    st.markdown(
        f"<span style='background:{color};color:#fff;padding:2px 8px;border-radius:10px;"
        f"font-size:0.8em'>{text}</span>",
        unsafe_allow_html=True,
    )


def _select_run(store: QualityStore, k: str) -> Optional[str]:
    runs = store.list_runs()
    if runs.empty:
        st.info("Aún no hay corridas de calidad. Ejecuta `python -m reii.quality.cli migrate-legacy`.")
        return None
    sm = pd.DataFrame(list(runs["summary"]))
    runs = pd.concat([runs.drop(columns=["summary"]), sm], axis=1)
    runs["pareto"] = M.pareto_mask(
        runs, maximize=["retention_estricto", "jaccard_mean", "ari", "precision_estricto"], minimize=["aurc"]
    )
    show_all = st.toggle("Mostrar todas las corridas (no solo las no dominadas)", value=False, key=f"{k}_all")
    cand = runs if show_all else runs[runs["pareto"] | (runs["kind"] == "legacy")]
    labels = {
        r.run_id: f"{r.label} · {r.kind} · {str(r.created_at)[:16]} · ret {_pct(getattr(r, 'retention_estricto', None))}"
        + (" ★" if r.pareto else "")
        for r in cand.itertuples()
    }
    active = store.get_active()
    ids = list(labels)
    idx = ids.index(active["run_id"]) if active and active["run_id"] in ids else 0
    run_id = st.selectbox("Corrida", ids, index=idx, format_func=labels.get, key=f"{k}_run")
    if st.button("Fijar como corrida por defecto", key=f"{k}_setactive"):
        store.set_active("default", run_id, st.session_state.get(f"{k}_tier", "estricto"))
        store.log(run_id, "set_active")
        st.success("Corrida fijada.")
    return run_id


def _profile_controls(profile: Dict[str, Any], calibrated: bool, k: str) -> Dict[str, Any]:
    p = copy.deepcopy(profile)
    with st.expander("Calibración y umbrales de los niveles", expanded=False):
        st.caption(
            "Los umbrales definen qué UCEs entran en cada nivel. "
            + ("Hay calibración: los umbrales son probabilidades de acierto (p_cal)."
               if calibrated else "Sin calibración: los umbrales se aplican directamente sobre el puntaje de acuerdo.")
        )
        for t in p["tiers"]:
            if t["kind"] == "s_min":
                continue
            if calibrated:
                t["threshold"] = st.slider(f"{t['name']} · p_cal ≥", 0.5, 0.99, float(t["threshold"]), 0.01, key=f"{k}_thr_{t['tier_id']}")
            else:
                t["fallback_s"] = st.slider(f"{t['name']} · acuerdo ≥", 0.0, 1.0, float(t.get("fallback_s", 0.5)), 0.01, key=f"{k}_fb_{t['tier_id']}")
        if cal.is_exploratory(p):
            _badge("MODO EXPLORATORIO — umbrales distintos a los declarados", "#C0392B")
    return p


def _gold_ui(store: QualityStore, run_id: str, df: pd.DataFrame, k: str) -> None:
    st.subheader("Estándar humano")
    gold = store.load_gold(run_id)
    st.caption(f"{gold['uce_id'].nunique() if not gold.empty else 0} UCEs codificadas en esta corrida.")
    coder = st.text_input("Codificador/a", value="codificador1", key=f"{k}_coder")
    n = st.slider("Tamaño de la muestra", 20, 200, 60, 10, key=f"{k}_nsample")
    if st.button("Generar muestra estratificada", key=f"{k}_sample"):
        st.session_state[f"{k}_sample_df"] = service.stratified_sample(df, n)
    sample = st.session_state.get(f"{k}_sample_df")
    if sample is not None and len(sample):
        txt = df.get("texto")
        view = pd.DataFrame(
            {"uce_id": sample["uce_id"], "clase_asignada": sample["cluster_core"].astype("string"),
             "nivel": sample["tier_min"].astype("string"), "correcta": False}
        )
        edited = st.data_editor(view, hide_index=True, key=f"{k}_editor", disabled=["uce_id", "clase_asignada", "nivel"])
        if st.button("Guardar codificación", key=f"{k}_savegold"):
            rows = [{"uce_id": r.uce_id, "coder": coder, "is_correct": bool(r.correcta)} for r in edited.itertuples()]
            store.save_gold(run_id, rows)
            store.log(run_id, "save_gold", {"n": len(rows), "coder": coder})
            st.success(f"{len(rows)} juicios guardados.")
    min_n = st.number_input("Mínimo de UCEs para recalibrar", 30, 500, 60, key=f"{k}_minn")
    if st.button("Recalibrar con el estándar humano", key=f"{k}_recal"):
        fit = service.recalibrate(store, run_id, int(min_n))
        st.success(f"Calibración v{fit['version']} guardada (n={fit['n']}).") if fit else st.warning(
            "No hay suficientes juicios (o falta variación entre correctas e incorrectas)."
        )


def render_quality_panel(store: Optional[QualityStore] = None, key_prefix: str = "q") -> None:
    k = key_prefix
    store = _store(store)
    st.header("Calidad y niveles de resultado")
    run_id = _select_run(store, k)
    if run_id is None:
        return

    base = cal.default_profile()
    saved = store.load_calibration(run_id)
    profile = _profile_controls(base, saved is not None, k)
    res = service.evaluate_run(store, run_id, profile, persist=False, calibration=saved)
    df, tiers = res["df"], res["tiers"]
    names = {t["tier_id"]: t["name"] for t in profile["tiers"]}
    tier_id = st.radio("Nivel", list(names), format_func=names.get, horizontal=True, key=f"{k}_tier")
    m = tiers[tier_id]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retención", _pct(m["retention"]))
    c2.metric("Precisión esperada", _pct(m.get("precision")) if m.get("precision") is not None else "sin estándar")
    cov = (res["run"]["summary"].get("extras") or {}).get("coverage") or {}
    c3.metric("Cobertura vocab. (tokens)", _pct(cov.get("cov_tokens")), help="Ocurrencias de formas analizadas / ocurrencias totales")
    c4.metric("Cobertura vocab. (tipos)", _pct(cov.get("cov_types")), help="Definición anterior: formas analizadas / formas distintas")
    if m["n_projected"]:
        _badge(f"{m['n_projected']} UCEs proyectadas (no inductivas)", "#8E44AD")
    if not res["nested_ok"]:
        st.error("Los niveles no están anidados: revisa los umbrales.")
    if res["run"]["kind"] == "legacy":
        st.caption("Corrida migrada: solo se conserva la estabilidad final (is_stable); el puntaje de acuerdo es binario.")

    tab_ind, tab_cls, tab_uce, tab_gold, tab_cmp = st.tabs(
        ["Indicadores", "Clases", "Detalle por UCE", "Estándar humano", "Comparar / guardar"]
    )
    with tab_ind:
        fun = pd.DataFrame(res["funnel"])
        fig = go.Figure(go.Bar(x=fun["pct"], y=fun["step"], orientation="h", text=[f"{p:.0%}" for p in fun["pct"]]))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=24, b=0), title="Embudo de retención", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, key=f"{k}_funnel")
        if res["curve"] is not None:
            cv = res["curve"]
            f2 = go.Figure(go.Scatter(x=cv["coverage"], y=cv["precision"], mode="lines"))
            f2.update_layout(height=260, margin=dict(l=0, r=0, t=24, b=0), title=f"Curva riesgo-cobertura (AURC={res['summary']['aurc']:.3f})",
                             xaxis_title="cobertura", yaxis_title="precisión")
            st.plotly_chart(f2, use_container_width=True, key=f"{k}_rc")
        else:
            st.caption("La curva riesgo-cobertura aparece cuando hay estándar humano codificado.")
        bias = {kk: v for kk, v in m.items() if kk.startswith(("smd_", "doc_retention"))}
        if bias:
            st.markdown("**Sesgo de selección** (retenidas vs descartadas)")
            st.dataframe(pd.DataFrame([bias]).T.rename(columns={0: "valor"}), use_container_width=True)
            if abs(bias.get("smd_n_tokens", 0) or 0) >= 0.10:
                st.warning("Diferencia de longitud |SMD| ≥ 0.10 entre retenidas y descartadas.")
    with tab_cls:
        jac = (res["run"]["summary"].get("extras") or {}).get("jaccard_by_class") or {}
        rows = []
        for i, (c, n_) in enumerate(m["class_sizes"].items()):
            j = jac.get(str(i))
            rows.append({"clase": c, "UCEs": n_, "Jaccard (A1↔A2)": j,
                         "frágil": bool(j is not None and j < FRAGILE_JACCARD)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Jaccard aproximado: compara las dos clasificaciones de conteo de palabras (A1↔A2).")
    with tab_uce:
        uid = st.selectbox("UCE", df["uce_id"].tolist(), key=f"{k}_uce")
        row = df[df["uce_id"] == uid].iloc[0]
        st.json({
            "clase_inductiva": None if pd.isna(row["cluster_core"]) else int(row["cluster_core"]),
            "proyectada": None if pd.isna(row["projected_cluster_id"]) else int(row["projected_cluster_id"]),
            "s_acuerdo": row["s_acuerdo"], "n_comparaciones": None if pd.isna(row["n_comparisons"]) else int(row["n_comparisons"]),
            "p_cal": None if pd.isna(row["p_cal"]) else float(row["p_cal"]),
            "nivel_minimo": row["tier_min"], "origen_puntaje": row["s_source"],
            "comparaciones": {c: bool(row[c]) for c in cons.COMPARISONS if c in row and row[c]},
        })
    with tab_gold:
        _gold_ui(store, run_id, df, k)
    with tab_cmp:
        ph = res["profile_hash"]
        if st.button("Guardar este resultado (corrida + nivel + umbrales)", key=f"{k}_savetier"):
            service.evaluate_run(store, run_id, profile, persist=True, calibration=saved)
            store.log(run_id, "save_tier_result", {"profile_hash": ph, "exploratorio": cal.is_exploratory(profile)})
            st.success(f"Guardado ({ph}). Pueden coexistir varios resultados por umbral.")
        saved_res = store.list_tier_results(run_id=run_id)
        if not saved_res.empty:
            flat = pd.DataFrame(
                [{"nivel": r.tier_id, "perfil": r.profile_hash, "retención": r.metrics.get("retention"),
                  "precisión": r.metrics.get("precision"), "UCEs": r.metrics.get("n_assigned")} for r in saved_res.itertuples()]
            )
            st.dataframe(flat, use_container_width=True, hide_index=True)
        export = {"run_id": run_id, "tier": tier_id, "profile": profile, "profile_hash": ph,
                  "exploratorio": cal.is_exploratory(profile), "metricas": m}
        st.download_button("Exportar vista activa (JSON)", json.dumps(export, ensure_ascii=False, default=str, indent=2),
                           file_name=f"vista_{run_id}_{tier_id}.json", mime="application/json", key=f"{k}_export")
        st.download_button("Exportar asignaciones (CSV)",
                           df[["uce_id", "doc_id", f"shown_cluster_{tier_id}", "s_acuerdo", "p_cal", "tier_min"]].to_csv(index=False),
                           file_name=f"asignaciones_{run_id}_{tier_id}.csv", mime="text/csv", key=f"{k}_exportcsv")
