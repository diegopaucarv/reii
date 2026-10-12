"""Figuras del discurso: resultados del análisis IA, para el final de la pestaña C.

Muestra, para la clase seleccionada, cada figura (``trait``) con sus citas como Evidencia
(tarjetas con la cita resaltada) y KWIC (cita con contexto), igual que las metáforas de la
sección de isotopías. Los datos vienen de la tabla ``annotations`` (no del JSON de estado).
"""

from __future__ import annotations

import html
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

CONF_COLOR = {"alta": "#5DC88A", "media": "#E8A838", "baja": "#E86450"}
CONF_RANK = {"alta": 3, "media": 2, "baja": 1}


def _esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""))


def _alpha(color: str, a: float = 0.2) -> str:
    """#rgb / #rrggbb -> rgba(...); cualquier otro valor se devuelve igual."""
    c = (color or "").strip()
    if c.startswith("#") and len(c) in (4, 7):
        h = "".join(ch * 2 for ch in c[1:]) if len(c) == 4 else c[1:]
        try:
            r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{a})"
        except ValueError:
            pass
    return c


def resolve_span(text: str, span: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Posición (inicio, fin) de la cita en el texto: offsets si son válidos, si no búsqueda."""
    s, e, q = (
        span.get("start_char", -1),
        span.get("end_char", -1),
        (span.get("quote") or "").strip(),
    )
    if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text):
        if not q or text[s:e].strip().lower() == q.lower():
            return s, e
    if q:
        m = re.search(re.escape(q), text, flags=re.IGNORECASE)
        if m:
            return m.start(), m.end()
    return None


def highlight(text: str, spans: Iterable[Tuple[int, int]], color: str) -> str:
    merged: List[List[int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, pos = [], 0
    for s, e in merged:
        out.append(_esc(text[pos:s]))
        out.append(
            f'<mark style="background:{_alpha(color)};color:inherit;border-bottom:1px solid {color};padding:0 1px">{_esc(text[s:e])}</mark>'
        )
        pos = e
    out.append(_esc(text[pos:]))
    return "".join(out)


def build_frame(
    annotations_by_uce: Dict[str, List[Dict[str, Any]]], uces: Iterable[Dict[str, Any]]
) -> pd.DataFrame:
    """Una fila por cita (span). Une cada anotación con su UCE (texto, clase, documento)."""
    by_id = {str(u.get("uce_id") or u.get("id")): u for u in uces}
    rows = []
    for uid, anns in annotations_by_uce.items():
        u = by_id.get(str(uid))
        if u is None:
            continue
        text = u.get("texto") or ""
        for a in anns or []:
            spans = a.get("spans") or [
                {"quote": a.get("quote", ""), "start_char": -1, "end_char": -1}
            ]
            for sp in spans:
                pos = resolve_span(text, sp)
                rows.append(
                    {
                        "uce_id": str(uid),
                        "trait": a.get("trait") or "?",
                        "agent": a.get("agent") or "?",
                        "subtype": a.get("subtype") or "",
                        "confidence": a.get("confidence") or "baja",
                        "quote": sp.get("quote") or "",
                        "start": pos[0] if pos else -1,
                        "end": pos[1] if pos else -1,
                        "reasoning": (a.get("metadata") or {}).get("reasoning", ""),
                        "cluster_id": u.get("cluster_id"),
                        "doc_id": u.get("doc_id"),
                        "texto": text,
                    }
                )
    df = pd.DataFrame(
        rows,
        columns=[
            "uce_id",
            "trait",
            "agent",
            "subtype",
            "confidence",
            "quote",
            "start",
            "end",
            "reasoning",
            "cluster_id",
            "doc_id",
            "texto",
        ],
    )
    if not df.empty:
        df["conf_rank"] = df["confidence"].map(CONF_RANK).fillna(1).astype(int)
    return df


def evidence_card(group: pd.DataFrame, color: str) -> str:
    r = group.iloc[0]
    spans = [(int(s), int(e)) for s, e in zip(group["start"], group["end"]) if s >= 0]
    conf = r["confidence"]
    cc = CONF_COLOR.get(conf, "#888")
    reason = (
        f'<div style="font-size:11px;color:var(--text-low);font-style:italic;margin-top:6px">{_esc(r["reasoning"])}</div>'
        if r["reasoning"]
        else ""
    )
    sub = f" · {_esc(r['subtype'])}" if r["subtype"] else ""
    return (
        f'<div style="border-left:3px solid {color};padding:10px 12px;margin-bottom:8px;border-radius:0 6px 6px 0;'
        f'background:var(--bg-card,transparent)">'
        f'<div style="font-family:var(--font-mono,monospace);font-size:10px;color:var(--text-low);margin-bottom:6px">'
        f"{_esc(r['uce_id'])} · doc {_esc(r['doc_id'])} · {_esc(r['agent'])}{sub} "
        f'<span style="color:{cc};border:.5px solid {cc}66;border-radius:8px;padding:0 6px;margin-left:4px">{_esc(conf)}</span></div>'
        f'<div style="font-size:13px;line-height:1.6;color:var(--text-hi)">{highlight(r["texto"], spans, color)}</div>{reason}</div>'
    )


def kwic_rows(group_df: pd.DataFrame, width: int = 45) -> str:
    rows = []
    for r in group_df.itertuples():
        if r.start < 0:
            continue
        left = _esc(r.texto[max(0, r.start - width) : r.start]).rjust(0)
        mid = _esc(r.texto[r.start : r.end])
        right = _esc(r.texto[r.end : r.end + width])
        rows.append(
            f'<tr><td style="text-align:right;color:var(--text-low);padding:2px 6px;white-space:nowrap">…{left}</td>'
            f'<td style="font-weight:600;color:var(--text-hi);padding:2px 6px;white-space:nowrap">{mid}</td>'
            f'<td style="color:var(--text-low);padding:2px 6px;white-space:nowrap">{right}…</td></tr>'
        )
    if not rows:
        return '<div style="font-size:12px;color:var(--text-low)">Sin posiciones localizables para el KWIC.</div>'
    return (
        '<div style="overflow-x:auto"><table style="font-family:var(--font-mono,monospace);font-size:11px;'
        'border-collapse:collapse">' + "".join(rows) + "</table></div>"
    )


ALL_WF, NO_WF = "__all__", "__none__"


def workflow_options(
    df: pd.DataFrame, workflows: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Opciones del selector: clave -> {label, agents|None}. Cuenta las citas de cada pipeline."""
    opts: Dict[str, Dict[str, Any]] = {
        ALL_WF: {"label": f"Todos los pipelines · {len(df)} citas", "agents": None}
    }
    known: set = set()
    for w in workflows:
        ag = set(w.get("agents") or [])
        known |= ag
        n = int(df["agent"].isin(ag).sum()) if not df.empty else 0
        opts[w["file"]] = {
            "label": f"{w.get('name') or w['file']} · {n} citas",
            "agents": ag,
        }
    orphans = set(df["agent"].unique()) - known if not df.empty else set()
    if orphans:
        opts[NO_WF] = {
            "label": f"Sin pipeline (agentes ya no configurados) · {int(df['agent'].isin(orphans).sum())} citas",
            "agents": orphans,
        }
    return opts


def _pick_workflow(
    st, df: pd.DataFrame, workflows: List[Dict[str, Any]], key_prefix: str
) -> Tuple[pd.DataFrame, str]:
    opts = workflow_options(df, workflows)
    choice = st.selectbox(
        "Pipeline discursivo",
        list(opts),
        format_func=lambda k: opts[k]["label"],
        key=f"{key_prefix}_wf",
    )
    agents = opts[choice]["agents"]
    return (df if agents is None else df[df["agent"].isin(agents)]), choice


def summarize(df: pd.DataFrame, n_uces_scope: int) -> Dict[str, Any]:
    if df.empty:
        return {
            "n_spans": 0,
            "n_uces": 0,
            "coverage": 0.0,
            "n_agents": 0,
            "pct_alta": 0.0,
            "n_traits": 0,
        }
    return {
        "n_spans": int(len(df)),
        "n_uces": int(df["uce_id"].nunique()),
        "coverage": df["uce_id"].nunique() / n_uces_scope if n_uces_scope else 0.0,
        "n_agents": int(df["agent"].nunique()),
        "n_traits": int(df["trait"].nunique()),
        "pct_alta": float((df["confidence"] == "alta").mean()),
    }


def render_discourse_results(
    annotations_by_uce: Dict[str, List[Dict[str, Any]]],
    uces: List[Dict[str, Any]],
    class_id: Optional[int],
    class_color: str = "#7C6BF8",
    sec_rule: Optional[Callable[[str], None]] = None,
    key_prefix: str = "dr",
    per_trait_limit: int = 8,
    workflows: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """``workflows``: [{file, name, agents}] — un pipeline discursivo por JSON de ``ia/``."""
    import streamlit as st

    df_all = build_frame(annotations_by_uce, uces)
    if df_all.empty:
        # Estado vacío honesto: no se ha ejecutado el pipeline discursivo, así que
        # NO mostramos el encabezado de sección como si hubiera resultados.
        st.info(
            "Todavía no hay análisis del discurso en la base de datos. Ejecútalo en la última pestaña "
            "(paso 3 · Discurso) y se sincronizará automáticamente."
        )
        st.caption(
            "La sección «Figuras del discurso» aparecerá aquí con sus pestañas de Evidencia/KWIC "
            "una vez que el pipeline haya generado anotaciones."
        )
        return
    (sec_rule or st.subheader)("Figuras del discurso · análisis IA")
    all_classes = st.toggle(
        "Todas las clases", value=class_id is None, key=f"{key_prefix}_all"
    )
    scope_ids = {
        str(u.get("uce_id") or u.get("id"))
        for u in uces
        if all_classes or u.get("cluster_id") == class_id
    }
    df = df_all[df_all["uce_id"].isin(scope_ids)]
    wf_choice = ALL_WF
    if workflows:
        df, wf_choice = _pick_workflow(st, df, workflows, key_prefix)
    sm = summarize(df, len(scope_ids))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Figuras (citas)", sm["n_spans"])
    c2.metric("UCEs anotadas", sm["n_uces"], f"{sm['coverage']:.0%} de la selección")
    c3.metric("Tipos de figura", sm["n_traits"])
    c4.metric("Confianza alta", f"{sm['pct_alta']:.0%}")
    if df.empty:
        st.info("La clase seleccionada no tiene figuras anotadas.")
        return
    f1, f2, f3 = st.columns([2, 2, 1])
    agents = sorted(df["agent"].unique())
    # la clave incluye el pipeline: al cambiarlo, el filtro de agentes se reinicia
    pick_agents = f1.multiselect(
        "Agente", agents, default=agents, key=f"{key_prefix}_agents_{wf_choice}"
    )
    confs = [c for c in ("alta", "media", "baja") if c in set(df["confidence"])]
    pick_conf = f2.multiselect(
        "Confianza", confs, default=confs, key=f"{key_prefix}_conf"
    )
    show_all = f3.checkbox("Ver todas", value=False, key=f"{key_prefix}_showall")
    df = df[df["agent"].isin(pick_agents) & df["confidence"].isin(pick_conf)]
    order = (
        df.groupby("trait")
        .agg(n=("quote", "size"), best=("conf_rank", "max"))
        .sort_values(["n", "best"], ascending=False)
    )
    for ti, (trait, info) in enumerate(order.iterrows()):
        g = df[df["trait"] == trait].sort_values(
            ["conf_rank", "uce_id"], ascending=[False, True]
        )
        mix = Counter(g["confidence"])
        chips = " ".join(
            f'<span style="color:{CONF_COLOR[c]};font-size:10px">{mix[c]} {c}</span>'
            for c in ("alta", "media", "baja")
            if mix.get(c)
        )
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:10px 0 4px;border-bottom:.5px solid var(--border2,#8884)">'
            f'<span style="font-family:var(--font-mono,monospace);font-size:12px;font-weight:600;color:var(--text-hi)">{_esc(trait)}</span>'
            f'<span style="font-size:11px;color:var(--text-low)">{int(info["n"])} citas · {g["uce_id"].nunique()} UCEs</span>{chips}</div>',
            unsafe_allow_html=True,
        )
        shown = g if show_all else g.head(per_trait_limit)
        ev, kw = st.tabs([f"Evidencia {ti + 1}", f"KWIC {ti + 1}"])
        with ev:
            st.markdown(
                "".join(
                    evidence_card(grp, class_color)
                    for _, grp in shown.groupby(
                        ["uce_id", "agent", "subtype"], sort=False
                    )
                ),
                unsafe_allow_html=True,
            )
        with kw:
            st.markdown(kwic_rows(shown), unsafe_allow_html=True)
        if not show_all and len(g) > per_trait_limit:
            st.caption(
                f"Mostrando {per_trait_limit} de {len(g)} citas. Activa «Ver todas» para ver el resto."
            )


def render_discourse_results_from_db(
    class_id, class_colors, uces, sec_rule=None, conn_factory=None
) -> None:
    """Lee las anotaciones de la BD (con caché por firma de la tabla) y dibuja la sección."""
    import streamlit as st

    from reii.pipelines import discourse_db as ddb

    if conn_factory is None:
        from reii.backend.sql.db import get_conn as conn_factory  # type: ignore[assignment]

    @st.cache_data(show_spinner=False)
    def _load(_sig: Tuple[Any, Any]):
        with conn_factory() as conn:
            return ddb.load_annotations_by_uce(conn)

    with conn_factory() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n, MAX(annotation_id) AS m FROM annotations"
        ).fetchone()
    sig = (row["n"], row["m"]) if row else (0, None)
    ann = _load(sig) if sig[0] else {}
    color = (
        (class_colors or {}).get(class_id, "#7C6BF8")
        if class_id is not None
        else "#7C6BF8"
    )
    render_discourse_results(
        ann, uces, class_id, color, sec_rule, workflows=list_workflows()
    )


def list_workflows() -> List[Dict[str, Any]]:
    """Pipelines discursivos configurados (un JSON de ``ia/`` cada uno), con su nombre y agentes."""
    try:
        from reii.ia_config_editor import (
            get_workflow_agents,
            list_workflow_files,
            load_workflow_names,
        )

        names = load_workflow_names()
        return [
            {"file": f, "name": names.get(f, f), "agents": get_workflow_agents(f)}
            for f in list_workflow_files()
        ]
    except Exception:
        return []
