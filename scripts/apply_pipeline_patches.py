#!/usr/bin/env python3
"""Conecta reii.pipelines con dashboard.py y backend/migrate.py (idempotente, reversible).

  D1 workflow_config  : _workflow_runner copia los parámetros también a ConfigStore.
  D2 discourse_db     : _discourse_runner lee la entrada desde la BD y sincroniza al terminar.
  D3 tab_c_figures    : final de la pestaña C = figuras del discurso (desde la BD).
  D4 tab_e_hub        : pestaña E = Pipelines y configuración (solo configura y ejecuta).
  M1 migrate_ids      : migrate_annotations reconcilia ids en vez de descartar anotaciones.

Uso: python scripts/apply_pipeline_patches.py [--repo RUTA] [--dry-run] [--revert]
Copias de seguridad: <archivo>.pipelines.bak. Respeta CRLF/LF.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

MARK = "[pipeline-patch]"

GLUE = '''# ── %(m)s Pestaña E: Pipelines y configuración ─────────────────────────────
@st.cache_data(ttl=10, show_spinner=False)
def _pipeline_status_cached(_nonce: int = 0):
    from reii.backend.config_store import ConfigStore
    from reii.backend.sql.db import get_conn
    from reii.config import DISCOURSE_STATE_PATH, WORKFLOW_CONFIG_PATH, WORKFLOW_DB_PATH
    from reii.pipelines.status import collect_status

    try:
        overrides = ConfigStore().get_all()
    except Exception:
        overrides = {}
    with get_conn() as conn:
        return collect_status(
            conn,
            {
                "workflow_db": WORKFLOW_DB_PATH,
                "discourse_state": DISCOURSE_STATE_PATH,
                "workflow_config": WORKFLOW_CONFIG_PATH,
            },
            store_overrides=overrides,
        )


def _pipeline_run_fix(fix_id: str) -> str:
    from reii.backend.config_store import ConfigStore
    from reii.config import DISCOURSE_STATE_PATH, WORKFLOW_CONFIG_PATH, WORKFLOW_DB_PATH
    from reii.pipelines import config_sync, runtime

    _pipeline_status_cached.clear()
    if fix_id == "sync_annotations":
        rep = runtime.sync_after_run(DISCOURSE_STATE_PATH)
        _load_discourse.clear()
        return f"{rep['n_written']} de {rep['n_annotations']} anotaciones sincronizadas (sin resolver: {rep['n_unmatched']})."
    if fix_id == "export_input":
        runtime.prepare_discourse_input("db", WORKFLOW_DB_PATH)
        return "Entrada del análisis regenerada desde la base de datos."
    if fix_id == "unify_config":
        config_sync.save_unified(ConfigStore().get_all(), WORKFLOW_CONFIG_PATH)
        return "workflow_config.json igualado a la configuración de la base de datos."
    return "Acción desconocida."


def _pipeline_start_workflow(params):
    from reii.batch_processor import BatchProcessor

    ss = st.session_state
    ss.setdefault("workflow_state", {"running": False, "finished": False, "exit_code": None})
    ss.setdefault("workflow_log", [])
    try:
        exportado = BatchProcessor(db_path=BATCH_DB_PATH).exportar_para_workflow()
        note = f"📤 Se exportaron {len(exportado['json'])} entrevista(s) a `data/txt_outputs/tmp/`."
    except Exception as e:
        note = f"⚠️ No se pudo exportar ({type(e).__name__}: {e}); se usan los JSON ya existentes."
    ss.workflow_log = [note]
    ss.workflow_state.update(running=True, finished=False, exit_code=None)
    threading.Thread(
        target=_workflow_runner, args=(ss.workflow_log, ss.workflow_state, params), daemon=True
    ).start()


def _pipeline_start_discourse(wf_file, store, workflow, force, source):
    ss = st.session_state
    ss.setdefault("discourse_state", {"running": False, "finished": False, "exit_code": None, "refreshed": False})
    ss.discourse_log = []
    ss.discourse_state.update(running=True, finished=False, exit_code=None, refreshed=False)
    _pipeline_status_cached.clear()
    threading.Thread(
        target=_discourse_runner,
        args=(ss.discourse_log, ss.discourse_state, wf_file, store, workflow, force, source),
        daemon=True,
    ).start()


def _render_pipeline_hub():
    from reii.backend.config_store import CONFIG_CATEGORIES, ConfigStore
    from reii.config import WORKFLOW_CONFIG_PATH
    from reii.ia_config_editor import (
        get_workflow_agents,
        list_workflow_files,
        load_discourse_state_progress,
        load_workflow_names,
        save_workflow_names,
    )
    from reii.pipelines import config_sync
    from reii.pipelines.hub_tab import HubContext, render_pipeline_hub

    ss = st.session_state
    if ss.get("discourse_state", {}).get("finished") and not ss["discourse_state"].get("refreshed"):
        ss["discourse_state"]["refreshed"] = True
        _load_discourse.clear()
        _pipeline_status_cached.clear()

    def _effective():
        try:
            overrides = ConfigStore().get_all()
        except Exception:
            overrides = {}
        return config_sync.effective_config(data.get("config") or {}, overrides)

    def _save(updates):
        _pipeline_status_cached.clear()
        return config_sync.save_unified(updates, WORKFLOW_CONFIG_PATH)

    def _reset():
        store = ConfigStore()
        for k in list(store.get_all()):
            store.reset(k)
        _pipeline_status_cached.clear()

    def _workflows():
        total, by_agent = load_discourse_state_progress()
        names = load_workflow_names()
        out = []
        for f in list_workflow_files():
            agents = get_workflow_agents(f)
            done = sum(by_agent.get(a, 0) for a in agents)
            out.append({"file": f, "name": names.get(f, f), "agents": agents,
                        "progress": min(1.0, done / (len(agents) * total)) if total and agents else 0.0})
        return out

    def _rename(f, n):
        names = load_workflow_names()
        names[f] = n
        save_workflow_names(names)

    def _open_batch():
        ss.show_batch_view = True
        st.rerun()

    render_pipeline_hub(
        HubContext(
            get_status=_pipeline_status_cached,
            session=ss,
            batch_open=_open_batch,
            get_effective_config=_effective,
            config_categories=CONFIG_CATEGORIES,
            save_config=_save,
            reset_config=_reset,
            start_workflow=_pipeline_start_workflow,
            list_workflows=_workflows,
            rename_workflow=_rename,
            start_discourse=_pipeline_start_discourse,
            run_fix=_pipeline_run_fix,
            render_agents_crud=_render_tab_e_crud,
        )
    )


''' % {"m": MARK}


def _find(lines, needle, start=0):
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    return -1


def p_workflow_config(t):
    if MARK + " d1" in t:
        return t, "ya aplicado"
    lines = t.split("\n")
    i = _find(lines, "def _workflow_runner(")
    j = _find(lines, "json.dump(params, f, ensure_ascii=False, indent=2)", i)
    if i < 0 or j < 0:
        return t, "ANCLA NO ENCONTRADA"
    ind = lines[j][: len(lines[j]) - len(lines[j].lstrip())][:-4]
    block = [
        f"{ind}try:  # {MARK} d1",
        f"{ind}    from reii.pipelines.config_sync import push_params_to_store",
        "",
        f"{ind}    _pp = push_params_to_store(params)",
        f'{ind}    log.append(f"⚙️ Configuración enviada a la BD: {{len(_pp[\'pushed\'])}} claves.")',
        f"{ind}except Exception as _qe:",
        f'{ind}    log.append(f"⚠️ No se pudo enviar la configuración a la BD: {{_qe}}")',
    ]
    lines[j + 1 : j + 1] = block
    return "\n".join(lines), "aplicado"


def p_discourse_db(t):
    if MARK + " d2" in t:
        return t, "ya aplicado"
    s = t.index("def _discourse_runner(")
    e = t.index("\ndef ", s + 10) if "\ndef " in t[s + 10 :] else len(t)
    body = t[s:e]
    need = ["    force: bool = False,\n) -> None:", "            mon = DebugOrchestrator(", "                workflow_data_path=WORKFLOW_DB_PATH,",
            "            mon.run_workflow(workflow_file, store=store, workflow=workflow, force=force)\n"]
    if any(n not in body for n in need):
        return t, "ANCLA NO ENCONTRADA"
    body = body.replace("    force: bool = False,\n) -> None:", f"    force: bool = False,\n    source: str = \"db\",  # {MARK} d2\n) -> None:", 1)
    body = body.replace("            mon = DebugOrchestrator(",
        "            from reii.pipelines.runtime import prepare_discourse_input, sync_after_run\n\n"
        "            _q_input = prepare_discourse_input(source, WORKFLOW_DB_PATH)\n"
        "            mon = DebugOrchestrator(", 1)
    body = body.replace("                workflow_data_path=WORKFLOW_DB_PATH,", "                workflow_data_path=_q_input,", 1)
    body = body.replace("            mon.run_workflow(workflow_file, store=store, workflow=workflow, force=force)\n",
        "            mon.run_workflow(workflow_file, store=store, workflow=workflow, force=force)\n"
        "            try:\n                sync_after_run(DISCOURSE_STATE_PATH)\n"
        "            except Exception as _qe:\n                log.append(f\"⚠️ No se pudo sincronizar con la BD: {_qe}\")\n", 1)
    return t[:s] + body + t[e:], "aplicado"


def p_tab_c(t):
    if MARK + " d3" in t:
        return t, "ya aplicado"
    anchor = "@st.fragment\ndef _render_tab_d():"
    if anchor not in t:
        return t, "ANCLA NO ENCONTRADA"
    block = (
        f"    # ── Figuras del discurso (análisis IA, desde la BD) · {MARK} d3 ──\n"
        "    try:\n"
        "        from reii.pipelines.discourse_results import render_discourse_results_from_db\n\n"
        "        render_discourse_results_from_db(\n"
        '            locals().get("cur_c"), globals().get("class_colors"), uces, sec_rule=_sec_rule\n'
        "        )\n"
        "    except Exception as _qe:\n"
        '        st.error(f"Figuras del discurso no disponibles: {_qe}")\n\n\n'
    )
    return t.replace(anchor, block + anchor, 1), "aplicado"


def p_tab_e(t):
    if MARK + " d4" in t:
        return t, "ya aplicado"
    old_call = ("elif tab_e.open:\n    with tab_e:\n        _render_tab_e()\n"
                '        _ds = st.session_state.get("discourse_state", {})\n'
                '        if not _ds.get("finished") and not _ds.get("running"):\n'
                "            _render_tab_e_crud()\n")
    if old_call not in t or not re.search(r'"E · [^"\n]*",\n', t) or "\nif tab_a.open:" not in t:
        return t, "ANCLA NO ENCONTRADA"
    t = re.sub(r'"E · [^"\n]*",\n', '"E · Pipelines y configuración",\n', t, count=1)
    t = t.replace(old_call, f"elif tab_e.open:  # {MARK} d4\n    with tab_e:\n        _render_pipeline_hub()\n", 1)
    t = t.replace("\nif tab_a.open:", "\n" + GLUE + "if tab_a.open:", 1)
    return t, "aplicado"


def p_migrate(t):
    if MARK + " m1" in t:
        return t, "ya aplicado"
    anchor = '        for a in flat:\n            if a["uce_id"] not in valid_uce_ids:'
    if anchor not in t:
        return t, "ANCLA NO ENCONTRADA"
    new = (
        f"        from reii.pipelines.discourse_db import build_id_resolver  # {MARK} m1\n\n"
        "        _resolve = build_id_resolver(\n"
        "            {\n"
        '                "uce_id": str(u.get("id") or u.get("uce_id") or ""),\n'
        '                "doc_id": _doc_id_of(u),\n'
        '                "local_idx": u.get("local_idx"),\n'
        '                "section_id": (u.get("metadata") or {}).get("section_id", 0),\n'
        "            }\n"
        '            for u in (data.get("uces") or [])\n'
        '            if (u.get("id") or u.get("uce_id"))\n'
        "        )\n"
        "        for a in flat:\n"
        '            a["uce_id"] = _resolve(a["uce_id"]) or a["uce_id"]\n'
        '            if a["uce_id"] not in valid_uce_ids:'
    )
    return t.replace(anchor, new, 1), "aplicado"


PATCHES = [
    ("src/reii/dashboard.py", "workflow_config", p_workflow_config),
    ("src/reii/dashboard.py", "discourse_db", p_discourse_db),
    ("src/reii/dashboard.py", "tab_c_figures", p_tab_c),
    ("src/reii/dashboard.py", "tab_e_hub", p_tab_e),
    ("src/reii/backend/migrate.py", "migrate_ids", p_migrate),
]


def _read(path):
    raw = open(path, "rb").read().decode("utf-8")
    return raw.replace("\r\n", "\n"), "\r\n" in raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    files = sorted({p[0] for p in PATCHES})
    if a.revert:
        for rel in files:
            path = os.path.join(a.repo, rel)
            if os.path.exists(path + ".pipelines.bak"):
                shutil.copy2(path + ".pipelines.bak", path)
                print(f"restaurado {rel}")
        return 0
    cache, status = {}, 0
    for rel, name, fn in PATCHES:
        if rel not in cache:
            cache[rel] = _read(os.path.join(a.repo, rel))
        text, crlf = cache[rel]
        new, msg = fn(text)
        print(f"{name:16s} {rel}: {msg}")
        status |= msg.startswith("ANCLA")
        cache[rel] = (new, crlf)
    if not a.dry_run and not status:
        for rel, (text, crlf) in cache.items():
            path = os.path.join(a.repo, rel)
            if not os.path.exists(path + ".pipelines.bak"):
                shutil.copy2(path, path + ".pipelines.bak")
            open(path, "wb").write((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    return int(status)


if __name__ == "__main__":
    sys.exit(main())
