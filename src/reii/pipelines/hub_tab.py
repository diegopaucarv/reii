"""Última pestaña: configuración y ejecución de los tres pipelines.

Solo configura y ejecuta; los resultados se ven en las pestañas A–D y en la última sección
de la C. La pestaña se construye con un ``HubContext`` (inyección de dependencias) para que
el dashboard le pase sus ejecutores y para poder probarla sin base de datos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from reii.pipelines.registry import BY_ID, HEADLINE, PIPELINES

STATE_LABEL = {
    "sin_datos": ("Sin datos", "#7F8C8D"),
    "al_dia": ("Al día", "#2ECC71"),
    "desactualizado": ("Desactualizado", "#E8A838"),
    "ejecutando": ("Ejecutando…", "#3498DB"),
}
LEVEL_ICON = {"ok": "✅", "warn": "⚠️", "error": "❌", "info": "ℹ️"}
ESSENTIAL_KEYS = [
    "classification_mode",
    "use_bigrams",
    "use_trigrams",
    "stem_backend",
    "min_uce_words",
    "uce_target_size",
    "tsj",
    "use_rf_shap",
    "multivariate_metadata",
    "random_state",
]


@dataclass
class HubContext:
    get_status: Callable[[], Dict[str, Any]]
    session: Any
    get_effective_config: Callable[[], Dict[str, Any]]  # {clave: (valor, origen)}
    config_categories: Dict[str, List[str]]
    save_config: Callable[[Dict[str, Any]], Dict[str, Any]]
    reset_config: Optional[Callable[[], None]]
    start_workflow: Callable[[Dict[str, Any]], None]
    list_workflows: Callable[[], List[Dict[str, Any]]]  # [{file,name,agents,progress}]
    rename_workflow: Callable[[str, str], None]
    start_discourse: Callable[..., None]  # (file, store, label, force, source)
    run_fix: Callable[[str], str]
    render_agents_crud: Optional[Callable[[], None]] = None
    render_transcripciones: Optional[Callable[[], None]] = (
        None  # gestión de transcripciones (subida/cola/log/exportación)
    )
    render_entrevistas: Optional[Callable[[], None]] = (
        None  # editores de entrevistas (guía, prompts, localismos)
    )
    extra: Dict[str, Any] = field(default_factory=dict)


def _pill(text: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:#fff;padding:2px 9px;border-radius:10px;"
        f"font-size:11px'>{text}</span>"
    )


def _running(ctx: HubContext, pid: str) -> bool:
    return bool((ctx.session.get(BY_ID[pid].state_key) or {}).get("running"))


def _system_map(ctx: HubContext, status: Dict[str, Any]) -> None:
    cols = st.columns(len(PIPELINES))
    for col, p in zip(cols, PIPELINES):
        info = status.get(p.pid, {"state": "sin_datos", "metrics": []})
        state = "ejecutando" if _running(ctx, p.pid) else info["state"]
        label, color = STATE_LABEL[state]
        with col, st.container(border=True):
            st.markdown(f"**{p.icon} {p.number} · {p.title}**")
            st.markdown(_pill(label, color), unsafe_allow_html=True)
            st.caption(p.summary)
            for name, val in info.get("metrics", []):
                st.markdown(
                    f"<span style='font-size:12px'>{name}: <b>{val}</b></span>",
                    unsafe_allow_html=True,
                )
            st.caption(f"Se ve en: {p.shown_in}")
    st.caption(
        "Flujo: 1 · Transcripciones → 2 · Clasificación → 3 · Discurso (IA) → resultados en las pestañas A–D y C."
    )


def _checks(ctx: HubContext, status: Dict[str, Any]) -> None:
    checks = status.get("checks", [])
    bad = [c for c in checks if c["level"] in ("warn", "error")]
    title = (
        f"Coherencia entre lo que corre y lo que se muestra ({len(bad)} aviso{'s' if len(bad) != 1 else ''})"
        if bad
        else "Coherencia entre lo que corre y lo que se muestra (todo en orden)"
    )
    with st.expander(title, expanded=bool(bad)):
        for c in checks:
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                f"{LEVEL_ICON.get(c['level'], '•')} **{c['label']}** — {c['detail']}"
            )
            if c.get("fix"):
                if c2.button(
                    _FIX_LABEL.get(c["fix"], "Arreglar"),
                    key=f"hub_fix_{c['id']}_{c['fix']}",
                ):
                    st.success(ctx.run_fix(c["fix"]))
                    st.rerun()


_FIX_LABEL = {
    "sync_annotations": "Sincronizar",
    "export_input": "Regenerar",
    "unify_config": "Unificar",
}


def _how_it_works(pid: str) -> None:
    p = BY_ID[pid]
    with st.expander("¿Qué hace y qué toca este paso?", expanded=False):
        st.markdown(p.summary)
        st.markdown("**Lee:** " + " · ".join(p.reads))
        st.markdown("**Escribe:** " + " · ".join(p.writes))
        st.markdown("**Configuración:** " + p.config)
        st.markdown("**Dónde ver los resultados:** " + p.shown_in)


@st.fragment(run_every=2)
def _run_panel(ctx: HubContext, pid: str) -> None:
    p = BY_ID[pid]
    state = ctx.session.get(p.state_key) or {}
    log = ctx.session.get(p.state_key.replace("_state", "_log")) or []
    if state.get("running"):
        st.info(f"⏳ {p.title}: en ejecución…")
    elif state.get("finished"):
        if state.get("exit_code") == 0:
            st.success(
                f"✅ {p.title} terminó sin errores. Resultados en: {p.shown_in}."
            )
        else:
            st.error(
                f"❌ {p.title} terminó con error (código {state.get('exit_code')}). Revisa el registro."
            )
    if log:
        with st.expander(
            f"Registro ({len(log)} líneas)", expanded=bool(state.get("running"))
        ):
            st.code("\n".join(log[-300:]))


# ── Design tokens (paleta consistente con el resto del dashboard) ──────
_ACC = "#7c8cff"
_MINT = "#50fa7b"
_AMBER = "#ffb86c"
_CORAL = "#f97b6b"
_VIO = "#bd93f9"


def _section_header(icon: str, title: str, sub: str = "") -> None:
    """Cabecera de sección con acento de color, estilo 'app'."""
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin:6px 0 2px;'>"
        f"<span style='font-size:20px'>{icon}</span>"
        f"<div><div style='font-size:16px;font-weight:700;color:var(--text-hi);"
        f"letter-spacing:.01em'>{title}</div>"
        f"<div style='font-size:11px;color:var(--text-low);letter-spacing:.03em'>"
        f"{sub}</div></div></div>"
        f"<div style='height:2px;background:linear-gradient(90deg,{_ACC},transparent);"
        f"border-radius:2px;margin:2px 0 10px;'></div>",
        unsafe_allow_html=True,
    )


def _config_widget(key: str, value: Any, origin: str, prefix: str):
    label = f"{key} {'✎' if origin == 'override' else ''}".strip()
    wkey = f"{prefix}_{key}"
    if isinstance(value, bool):
        return st.radio(
            label, [True, False], index=0 if value else 1, horizontal=True, key=wkey
        )
    if isinstance(value, int):
        return st.number_input(label, value=value, step=1, key=wkey)
    if isinstance(value, float):
        return st.number_input(label, value=value, step=0.01, key=wkey)
    if isinstance(value, (list, tuple)):
        import json

        raw = st.text_area(
            label, value=json.dumps(list(value), ensure_ascii=False), key=wkey
        )
        try:
            return json.loads(raw)
        except ValueError:
            st.caption("JSON inválido: se conserva el valor anterior.")
            return list(value)
    return st.text_input(label, value="" if value is None else str(value), key=wkey)


def _tab_workflow(ctx: HubContext) -> None:
    _how_it_works("workflow")
    if ctx.render_transcripciones is not None:
        with st.expander(
            "📥 Transcripciones y exportación al workflow", expanded=False
        ):
            st.caption(
                "Sube transcripciones, revisa su estado, procesa la cola y exporta "
                "al workflow principal. Los parámetros del análisis se aplican al exportar."
            )
            try:
                ctx.render_transcripciones()
            except Exception as e:
                st.error(f"⚠️ No se pudo cargar el gestor de transcripciones: {e}")
    eff = ctx.get_effective_config()
    mode = st.radio(
        "Modo de procesamiento",
        ["classic", "transformer"],
        horizontal=True,
        key="hub_wf_mode",
        format_func=lambda v: {
            "classic": "📊 Classic (rápido, sin GPU)",
            "transformer": "🧠 Transformer (requiere GPU)",
        }[v],
    )
    updates: Dict[str, Any] = {}
    ess = [k for k in ESSENTIAL_KEYS if k in eff]
    with st.expander("Parámetros esenciales", expanded=True):
        for k in ess:
            v, origin = eff[k]
            new = _config_widget(k, v, origin, "hub_ess")
            if new != v:
                updates[k] = new
    with st.expander("Todos los parámetros por categoría", expanded=False):
        for cat, keys in ctx.config_categories.items():
            with st.expander(cat.replace("_", " ").capitalize(), expanded=False):
                for k in keys:
                    if k not in eff or k in ess:
                        continue
                    v, origin = eff[k]
                    new = _config_widget(k, v, origin, f"hub_{cat}")
                    if new != v:
                        updates[k] = new
    b1, b2 = st.columns([1, 1])
    if b1.button(
        f"💾 Guardar {len(updates)} cambio(s)" if updates else "💾 Guardar",
        key="hub_cfg_save",
        disabled=not updates,
    ):
        rep = ctx.save_config(updates)
        st.success(
            f"Guardado en ambos canales ({len(rep.get('pushed', []))} claves en la BD)."
        )
    if ctx.reset_config and b2.button("↩ Restaurar defaults", key="hub_cfg_reset"):
        ctx.reset_config()
        st.rerun()
    st.divider()
    if st.button(
        "▶ Ejecutar clasificación",
        key="hub_run_workflow",
        type="primary",
        disabled=_running(ctx, "workflow"),
    ):
        ctx.start_workflow({"workflow_mode": mode, **updates})
        st.rerun()
    _run_panel(ctx, "workflow")


def _tab_discourse(ctx: HubContext) -> None:
    _how_it_works("discourse")
    source = st.radio(
        "Fuente de las UCEs",
        ["db", "json"],
        horizontal=True,
        key="hub_dc_source",
        format_func=lambda v: {
            "db": "Base de datos (recomendado: lo que se anota es lo que se muestra)",
            "json": "workflow_data.json (comportamiento anterior)",
        }[v],
    )
    with st.expander("Opciones de ejecución", expanded=False):
        c1, c2, c3 = st.columns(3)
        store = c1.checkbox("Guardar estado", value=True, key="hub_dc_store")
        force = c2.checkbox(
            "Re-procesar todo",
            value=False,
            key="hub_dc_force",
            help="Ignora lo ya procesado y vuelve a analizar todas las UCEs.",
        )
        label = c3.text_input("Etiqueta (opcional)", key="hub_dc_label") or None
    st.caption(
        "Al terminar, las anotaciones se sincronizan con la base de datos y aparecen en la última sección de la pestaña C."
    )
    wfs = ctx.list_workflows()
    if not wfs:
        st.warning("No se encontraron workflows en `ia/` (archivos JSON de agentes).")
    running = _running(ctx, "discourse")
    for w in wfs:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 1])
            new_name = c1.text_input(
                "Nombre del workflow", value=w["name"], key=f"hub_wfname_{w['file']}"
            )
            c1.caption(f"{len(w['agents'])} agentes: {', '.join(w['agents'])}")
            c2.progress(min(1.0, max(0.0, float(w.get("progress", 0.0)))))
            c2.caption(
                f"{float(w.get('progress', 0.0)) * 100:.0f}% procesado"
                if w.get("progress")
                else "Sin datos todavía"
            )
            if c3.button(
                "▶ Ejecutar",
                key=f"hub_wfrun_{w['file']}",
                disabled=running,
                use_container_width=True,
            ):
                if new_name != w["name"]:
                    ctx.rename_workflow(w["file"], new_name)
                ctx.start_discourse(w["file"], store, label, force, source)
                st.rerun()
    _run_panel(ctx, "discourse")
    if ctx.render_agents_crud:
        with st.expander("Agentes y prompts (ia/*.json)", expanded=False):
            if not running:
                ctx.render_agents_crud()
            else:
                st.caption("Disponible cuando termine la ejecución.")


def render_pipeline_hub(ctx: HubContext) -> None:
    _section_header("⚙️", "Pipelines y configuración", HEADLINE)
    status = ctx.get_status()
    _system_map(ctx, status)
    _checks(ctx, status)
    t1, t2 = st.tabs(["2 · Clasificación", "3 · Análisis del discurso (IA)"])
    with t1:
        _tab_workflow(ctx)
    with t2:
        _tab_discourse(ctx)
    if ctx.render_entrevistas is not None:
        st.divider()
        _section_header(
            "📝", "Entrevistas y prompts", "Guía de preguntas, prompts y localismos"
        )
        with st.expander("Editores de entrevistas y prompts", expanded=False):
            try:
                ctx.render_entrevistas()
            except Exception as e:
                st.error(f"⚠️ No se pudieron cargar los editores de entrevistas: {e}")
