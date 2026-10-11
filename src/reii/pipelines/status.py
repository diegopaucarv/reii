"""Estado y chequeos de coherencia entre lo que corre y lo que se muestra.

Funciones puras sobre una conexión (psycopg o SQLite) y rutas de archivos. Cada chequeo
devuelve {id, label, level, detail, fix}; ``fix`` es el identificador de la acción que
la pestaña ofrece como botón (o None).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from reii.pipelines import config_sync
from reii.pipelines import discourse_db as ddb

TOLERANCE_S = 60.0


def file_info(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"exists": False, "size": 0, "mtime": None}
    st = os.stat(path)
    return {"exists": True, "size": st.st_size, "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)}


def parse_ts(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip().replace(" ", "T", 1)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if len(s) > 3 and s[-3] in "+-" and ":" not in s[-5:]:
        s += ":00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _scalar(conn, sql: str, params=()) -> Any:
    try:
        row = conn.execute(sql, params).fetchone()
        return None if row is None else next(iter(dict(row).values()))
    except Exception:
        return None


def read_state_summary(path: Optional[str]) -> Dict[str, Any]:
    """Resumen del JSON de estado del discurso (sin devolver el contenido)."""
    if not path or not os.path.exists(path):
        return {"exists": False, "state": {}, "n_uces": 0, "n_annotations": 0, "by_agent": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        return {"exists": True, "state": {}, "n_uces": 0, "n_annotations": 0, "by_agent": {}, "unreadable": True}
    by_agent: Dict[str, int] = {}
    n = 0
    for anns in (state.get("annotations_by_uce") or {}).values():
        for a in anns or []:
            if isinstance(a, dict):
                n += 1
                by_agent[a.get("agent") or "?"] = by_agent.get(a.get("agent") or "?", 0) + 1
    return {"exists": True, "state": state, "n_uces": len(state.get("annotations_by_uce") or {}),
            "n_annotations": n, "by_agent": by_agent, "last_workflow": state.get("last_workflow")}


def _chk(cid, label, level, detail, fix=None) -> Dict[str, Any]:
    return {"id": cid, "label": label, "level": level, "detail": detail, "fix": fix}


def collect_status(conn, paths: Dict[str, str], state_summary: Optional[Dict[str, Any]] = None,
                   store_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Estado de los tres pipelines + chequeos. ``paths``: workflow_db, discourse_state, workflow_config."""
    n_docs = _scalar(conn, "SELECT COUNT(*) FROM documents")
    n_uces = _scalar(conn, "SELECT COUNT(*) FROM uces") or 0
    n_stable = _scalar(conn, "SELECT COUNT(*) FROM uces WHERE is_stable = 1") or 0
    n_clusters = _scalar(conn, "SELECT COUNT(*) FROM clusters") or 0
    last_uces = parse_ts(_scalar(conn, "SELECT MAX(created_at) FROM uces"))
    n_ann = _scalar(conn, "SELECT COUNT(*) FROM annotations") or 0
    n_ann_uces = _scalar(conn, "SELECT COUNT(DISTINCT uce_id) FROM annotations") or 0
    shap_len = _scalar(conn, "SELECT LENGTH(value_json) FROM kv_store WHERE key = 'shap_analysis'")
    cfg_raw = _scalar(conn, "SELECT value_json FROM kv_store WHERE key = 'config'")
    try:
        cfg = json.loads(cfg_raw) if cfg_raw else {}
    except (TypeError, ValueError):
        cfg = {}
    use_rf = (store_overrides or {}).get("use_rf_shap", cfg.get("use_rf_shap"))

    wf_file = file_info(paths.get("workflow_db"))
    st_file = file_info(paths.get("discourse_state"))
    ss = state_summary if state_summary is not None else read_state_summary(paths.get("discourse_state"))

    checks: List[Dict[str, Any]] = []
    # 1 · clasificación existente
    if n_uces == 0:
        checks.append(_chk("classification", "Clasificación en la BD", "error",
                           "No hay UCEs en la base de datos. Ejecuta el paso 2 (Clasificación).", None))
    else:
        checks.append(_chk("classification", "Clasificación en la BD", "ok",
                           f"{n_uces} UCEs · {n_stable} estables · {n_clusters} clases."))
    # 2 · frescura de la entrada del discurso
    if wf_file["exists"] and last_uces is not None:
        gap = (last_uces - wf_file["mtime"]).total_seconds()
        if gap > TOLERANCE_S:
            checks.append(_chk("input_freshness", "Entrada del análisis de discurso", "warn",
                               "workflow_data.json es anterior a la última clasificación en la BD: el análisis "
                               "anotaría UCEs que ya no coinciden con lo que se muestra.", "export_input"))
        else:
            checks.append(_chk("input_freshness", "Entrada del análisis de discurso", "ok",
                               "workflow_data.json está al día respecto a la BD."))
    elif not wf_file["exists"]:
        checks.append(_chk("input_freshness", "Entrada del análisis de discurso", "info",
                           "No existe workflow_data.json; la entrada puede generarse desde la BD.", "export_input"))
    # 3 · sincronización de anotaciones estado → BD
    if ss["exists"] and ss["n_annotations"] and n_uces:
        resolve = ddb.build_id_resolver(ddb._fetch_uce_rows(conn))
        ids = list((ss["state"].get("annotations_by_uce") or {}))
        unmatched = [i for i in ids if resolve(i) is None]
        if unmatched:
            checks.append(_chk("annotations_ids", "Ids de las anotaciones", "warn",
                               f"{len(unmatched)} de {len(ids)} UCEs anotadas no existen en la BD actual "
                               f"(p. ej. {unmatched[:2]}). Se anotó una clasificación anterior.", None))
        if n_ann < ss["n_annotations"] - sum(len(ss["state"]["annotations_by_uce"][i] or []) for i in unmatched):
            checks.append(_chk("annotations_sync", "Anotaciones en la BD", "warn",
                               f"El estado tiene {ss['n_annotations']} anotaciones y la tabla annotations {n_ann}. "
                               "La pestaña C lee de la BD.", "sync_annotations"))
        else:
            checks.append(_chk("annotations_sync", "Anotaciones en la BD", "ok",
                               f"{n_ann} anotaciones en {n_ann_uces} UCEs."))
    elif n_ann:
        checks.append(_chk("annotations_sync", "Anotaciones en la BD", "ok", f"{n_ann} anotaciones en {n_ann_uces} UCEs."))
    else:
        checks.append(_chk("annotations_sync", "Anotaciones en la BD", "info",
                           "Todavía no hay anotaciones del discurso en la BD."))
    # 4 · canales de configuración
    json_params = config_sync.read_json_params(paths.get("workflow_config"))
    diffs = config_sync.diff_channels(store_overrides or {}, json_params)
    if diffs:
        shown = ", ".join(d["key"] for d in diffs[:4])
        checks.append(_chk("config_channels", "Configuración del workflow", "warn",
                           f"workflow_config.json y la BD difieren en: {shown}.", "unify_config"))
    else:
        checks.append(_chk("config_channels", "Configuración del workflow", "ok",
                           "Los dos canales de configuración coinciden."))
    # 5 · RF/SHAP
    if use_rf and not shap_len:
        checks.append(_chk("rf", "Random Forest / SHAP", "warn",
                           "use_rf_shap está activo pero no hay resultados guardados: el RF falló o se omitió en la "
                           "última corrida (revisa el log y multivariate_metadata)."))
    elif use_rf is False and not shap_len:
        checks.append(_chk("rf", "Random Forest / SHAP", "info", "use_rf_shap está desactivado: la pestaña D no tendrá métricas RF."))
    elif shap_len:
        checks.append(_chk("rf", "Random Forest / SHAP", "ok", "Hay resultados RF/SHAP guardados."))

    def _pill(has: bool, stale: bool) -> str:
        return "sin_datos" if not has else ("desactualizado" if stale else "al_dia")

    stale_disc = any(c["id"] in ("annotations_sync", "annotations_ids") and c["level"] == "warn" for c in checks)
    stale_wf = any(c["id"] in ("config_channels",) and c["level"] == "warn" for c in checks)
    return {
        "batch": {"state": _pill(bool(n_docs), False),
                  "metrics": [("Documentos", n_docs if n_docs is not None else "—")]},
        "workflow": {"state": _pill(n_uces > 0, stale_wf),
                     "metrics": [("UCEs", n_uces), ("Estables", n_stable), ("Clases", n_clusters)],
                     "last_run": last_uces},
        "discourse": {"state": _pill(bool(n_ann or ss["n_annotations"]), stale_disc),
                      "metrics": [("Anotaciones (BD)", n_ann), ("UCEs anotadas", n_ann_uces),
                                  ("Anotaciones (estado)", ss["n_annotations"])],
                      "last_run": st_file["mtime"], "by_agent": ss["by_agent"]},
        "checks": checks,
        "files": {"workflow_db": wf_file, "discourse_state": st_file},
    }
