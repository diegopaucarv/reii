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


def _fmt(x: Any, pct: bool = False, nd: int = 3) -> str:
    """Formato de un valor que puede ser None/NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}"


def _corpus(store: QualityStore, run_id: str, k: str) -> Dict[str, Any]:
    """Texto/lingüística de las UCEs de la corrida (cache por sesión; verificado por huella)."""
    key = f"{k}_corpus_{run_id}"
    if key not in st.session_state:
        st.session_state[key] = service.load_corpus_for_run(store, run_id)
    return st.session_state[key]


def _gold_ui(store: QualityStore, run_id: str, df: pd.DataFrame, k: str) -> None:
    st.subheader("Estándar humano")
    status = st.empty()

    def _count_caption() -> None:
        g = store.load_gold(run_id)
        status.caption(f"{g['uce_id'].nunique() if not g.empty else 0} UCEs codificadas en esta corrida.")

    _count_caption()
    coder = st.text_input("Codificador/a", value="codificador1", key=f"{k}_coder")
    n = st.slider("Tamaño de la muestra", 20, 200, 60, 10, key=f"{k}_nsample")
    if st.button("Generar muestra estratificada", key=f"{k}_sample"):
        st.session_state[f"{k}_sample_df"] = service.stratified_sample(df, n)
    sample = st.session_state.get(f"{k}_sample_df")
    if sample is not None and len(sample):
        corpus = _corpus(store, run_id, k)
        if corpus["status"] == "ok":
            tx = dict(zip(corpus["df"]["uce_id"], corpus["df"]["texto"]))
        else:
            tx = {}
            st.warning(f"No se pudo cargar el texto de las UCEs ({corpus['reason']}): sin el texto no se puede codificar con criterio.")
        ids = sample["uce_id"].astype(str)
        view = pd.DataFrame(
            {
                "uce_id": ids,
                "texto": [(tx.get(u, "")[:240] + ("…" if len(tx.get(u, "")) > 240 else "")) for u in ids],
                "clase_pipeline": sample["cluster_core"].astype("string"),
                "nivel": sample["tier_min"].astype("string"),
                "correcta": False,
                "clase_humana": pd.array([pd.NA] * len(sample), dtype="Int64"),
            }
        )
        edited = st.data_editor(
            view,
            hide_index=True,
            key=f"{k}_editor",
            disabled=["uce_id", "texto", "clase_pipeline", "nivel"],
            column_config={
                "texto": st.column_config.TextColumn("Texto", width="large"),
                "correcta": st.column_config.CheckboxColumn("¿Correcta?"),
                "clase_humana": st.column_config.NumberColumn(
                    "Clase humana", min_value=0, max_value=99, step=1,
                    help="Opcional. Si la clase del pipeline es incorrecta, escribe la clase que "
                         "consideras correcta (mismos ids que «clase_pipeline»). Si la escribes, "
                         "«¿Correcta?» se deriva de ella. Necesaria para Kappa de clase y C_ext.",
                ),
            },
        )
        if st.button("Guardar codificación", key=f"{k}_savegold"):
            rows = service.gold_rows_from_editor(edited, coder)
            store.save_gold(run_id, rows)
            store.log(run_id, "save_gold", {"n": len(rows), "coder": coder})
            n_cls = sum(1 for r in rows if r["human_cluster"] is not None)
            st.success(f"{len(rows)} juicios guardados ({n_cls} con clase humana explícita).")
            _count_caption()
    gold = store.load_gold(run_id)  # recargar: lo guardado arriba aún no estaba en la lectura inicial

    coders = sorted(gold["coder"].unique().tolist()) if not gold.empty else []
    if len(coders) >= 2:
        st.subheader("Concordancia entre codificadores")
        c1, c2 = st.columns(2)
        ca = c1.selectbox("Codificador A", coders, key=f"{k}_ca")
        cb = c2.selectbox("Codificador B", [c for c in coders if c != ca], key=f"{k}_cb")
        if ca and cb and ca != cb:
            ag = service.coder_agreement(gold, df, ca, cb)
            if ag["binary"]:
                b = ag["binary"]
                st.metric("Kappa · correcta/incorrecta", _fmt(b["kappa"]),
                          help="Con muy pocas incorrectas el kappa se deprime aunque el % de acuerdo sea alto.")
                st.caption(f"n = {b['n']} UCEs · acuerdo simple {_fmt(b['pct_agreement'], pct=True)}")
            else:
                st.caption(f"Solo {ag['n_common']} UCEs codificadas por ambos; se necesitan ≥{ag['min_n']}.")
            if ag["class"]:
                c = ag["class"]
                st.metric("Kappa · clase", _fmt(c["kappa"]))
                st.caption(f"n = {c['n']} UCEs con clase (explícita o inferida de «correcta») en ambos · acuerdo simple {_fmt(c['pct_agreement'], pct=True)}")
            elif ag["binary"]:
                st.caption(f"Kappa de clase: solo {ag['n_class']} UCEs tienen clase en ambos codificadores; se necesitan ≥{ag['min_n']}.")
    elif not gold.empty:
        st.caption("El kappa entre codificadores aparece cuando hay ≥2 codificadores con nombres distintos.")

    min_n = st.number_input("Mínimo de UCEs para recalibrar", 30, 500, 60, key=f"{k}_minn")
    if st.button("Recalibrar con el estándar humano", key=f"{k}_recal"):
        fit = service.recalibrate(store, run_id, int(min_n))
        st.success(f"Calibración v{fit['version']} guardada (n={fit['n']}).") if fit else st.warning(
            "No hay suficientes juicios (o falta variación entre correctas e incorrectas)."
        )


def _c_ext_block(m: Dict[str, Any], tier_id: str) -> None:
    cx = m.get("c_ext")
    if cx:
        st.markdown("**Coherencia externa (vs. estándar humano)**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Exactitud (clase)", _fmt(cx.get("accuracy"), pct=True))
        c2.metric("Kappa (clase)", _fmt(cx.get("kappa")))
        c3.metric("Exactitud alineada", _fmt(cx.get("accuracy_aligned"), pct=True),
                  help="Tras emparejar clases pipeline↔humano con Hungarian: útil si el codificador usó su propia numeración.")
        c4.metric("NMI (clase)", _fmt(cx.get("nmi")))
        st.caption(
            f"n = {m['c_ext_n']} UCEs del nivel «{tier_id}» con clase humana "
            f"({m.get('c_ext_n_explicit', 0)} con clase explícita). Exactitud y kappa suponen que la clase humana usa los ids "
            "del pipeline; exactitud alineada y NMI no. La muestra es estratificada: no estima proporciones poblacionales."
        )
    elif "c_ext_n" in m:
        st.caption(f"C_ext no disponible: {m['c_ext_n']} UCEs del nivel con clase humana; se necesitan ≥{service.MIN_CODED} "
                   "(rellena «clase humana» o marca «correcta» en el editor).")


def _silhouette_block(store: QualityStore, run_id: str, k: str) -> None:
    st.markdown("**Validez interna (Silhouette de las clases, nivel UCE)**")
    run = store.get_run(run_id)
    sil = ((run or {}).get("summary") or {}).get("extras", {}).get("silhouette")
    if not (isinstance(sil, dict) and sil.get("status") == "ok"):
        if st.button("Calcular Silhouette", key=f"{k}_silbtn",
                     help="Se recalcula desde la tabla de UCEs y se guarda en la corrida."):
            with st.spinner("Calculando Silhouette…"):
                sil = service.ensure_silhouette(store, run_id)
        else:
            st.caption("Aún no calculado para esta corrida.")
            return
    if sil.get("status") != "ok":
        st.warning(f"Silhouette no disponible: {sil.get('reason', 'sin detalle')}.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("χ² (perfiles AFC)", _fmt(sil.get("chi2")), help="Geometría del AFC/ALCESTE sobre la matriz binaria UCE×término.")
    c2.metric("Coseno · binaria", _fmt(sil.get("binary_cosine")))
    c3.metric("Coseno · PPMI", _fmt(sil.get("ppmi_cosine")),
              help="Solo si el pipeline corrió con PPMI al guardarse la corrida (corridas recalculadas no lo tienen).")
    src = {"cdh": "vocabulario del CDH", "derived_tsj": "vocabulario derivado (tsj)"}.get(sil.get("vocab_source"), sil.get("vocab_source"))
    chk = {"hash": " · corpus verificado por huella", "ids_only": " · corpus verificado solo por ids"}.get(sil.get("corpus_check"), "")
    st.caption(f"n = {sil.get('n_labelled')} UCEs con clase ({sil.get('n_classes')} clases), {src}{chk}. "
               "Aproximación a nivel UCE: el CDH clasifica unidades de contexto y cada UCE hereda su etiqueta. Sin SBERT.")


def _compare_block(store: QualityStore, run_id: str, tier_id: str, profile: Dict[str, Any], k: str) -> None:
    st.subheader("Comparar con otra corrida")
    runs = store.list_runs()
    others = [r.run_id for r in runs.itertuples() if r.run_id != run_id]
    if not others:
        st.caption("Solo hay una corrida guardada. Ejecuta otra con distinto `classification_mode` para comparar.")
        return
    names = {r.run_id: f"{r.label} · {str(r.created_at)[:16]}" for r in runs.itertuples()}
    other = st.selectbox("Corrida de referencia", others, format_func=names.get, key=f"{k}_other_run")
    cmp = service.compare_runs(store, run_id, other, tier_id, profile=profile)
    if not cmp["same_corpus"]:
        st.warning("Las corridas no tienen la misma huella de corpus: la comparación solo tiene sentido si analizan los mismos textos.")
    if cmp["n_common"] < service.MIN_CODED:
        st.warning(f"Solo {cmp['n_common']} UCEs clasificadas en ambas corridas en el nivel «{tier_id}»: FMI/NMI/ARI no son fiables.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("FMI", _fmt(cmp["fmi"]))
    c2.metric("NMI", _fmt(cmp["nmi"]))
    c3.metric("ARI", _fmt(cmp["ari"]))
    st.caption(f"n = {cmp['n_common']} UCEs comunes (esta corrida: {cmp['n_a']}, referencia: {cmp['n_b']}) · nivel {tier_id}. "
               "Solo cuentan las UCEs que ambas corridas incluyen en el nivel, lo que tiende a favorecer el acuerdo.")
    jac = pd.DataFrame(
        [{"clase (esta corrida)": a, "clase emparejada (referencia)": cmp["class_match"].get(a), "Jaccard": j}
         for a, j in cmp["per_class_jaccard"].items()]
    )
    st.dataframe(jac, use_container_width=True, hide_index=True)
    st.dataframe(cmp["confusion"], use_container_width=True)
    if cmp["silhouette_a"] is not None or cmp["silhouette_b"] is not None:
        c4, c5 = st.columns(2)
        c4.metric(f"Silhouette ({cmp['silhouette_space']}) · esta corrida", _fmt(cmp["silhouette_a"]))
        c5.metric(f"Silhouette ({cmp['silhouette_space']}) · referencia", _fmt(cmp["silhouette_b"]))
    else:
        st.caption("Silhouette no calculado en estas corridas (pestaña «Indicadores» → Calcular Silhouette).")


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
        _c_ext_block(m, tier_id)
        _silhouette_block(store, run_id, k)
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
        _compare_block(store, run_id, tier_id, profile, k)
        st.divider()
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
