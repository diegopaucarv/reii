#!/usr/bin/env python3
"""Diagnostica por qué la pestaña D no muestra métricas RF/SHAP.

Distingue entre: el RF no se ejecutó, se ejecutó y falló, hay datos viejos/incompletos,
o los datos existen y el problema sería de lectura.

Uso:
  python scripts/diagnose_rf.py                       # lee PostgreSQL (REII_DATABASE_URL)
  python scripts/diagnose_rf.py --dsn postgresql://...
  python scripts/diagnose_rf.py --json RUTA/workflow_data.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any, Dict, List, Optional

# Claves que consume dashboard.py (make_rf_triangulation, SHAP, confusión, MCA...)
REQUIRED = {
    "Triangulación de importancias": ["feature_importance", "permutation_importance", "shap_importance"],
    "Balanced accuracy (CV)": ["cv_balanced_accuracy"],
    "SHAP por clase": ["shap_per_class_mean_abs", "feature_names"],
    "SHAP beeswarm": ["shap_values", "raw_X", "feature_names"],
    "Matriz de confusión": ["confusion_matrix"],
    "Diagnóstico (PCA / tasa de error)": ["raw_X", "y_true", "oof_predictions"],
}


def check_environment() -> Dict[str, Any]:
    out = {}
    for mod in ("shap", "sklearn", "sklearn.ensemble", "sklearn.inspection"):
        try:
            importlib.import_module(mod)
            out[mod] = True
        except Exception as e:  # noqa: BLE001
            out[mod] = f"NO: {type(e).__name__}: {e}"
    return out


def analyze(
    shap: Any,
    config: Dict[str, Any],
    stable_cluster_ids: Optional[List[int]] = None,
    n_stable_uces: Optional[int] = None,
    doc_meta_columns: Optional[List[str]] = None,
    env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Función pura: devuelve {verdict, reasons, sections}."""
    reasons: List[str] = []
    sections: Dict[str, bool] = {}
    use_rf = config.get("use_rf_shap")
    meta_cols = config.get("multivariate_metadata") or []

    if env:
        bad = [k for k, v in env.items() if v is not True]
        if bad:
            reasons.append(f"Dependencias RF/SHAP no importables en este entorno: {', '.join(bad)}. "
                           "El pipeline desactiva el RF en silencio (_RF_SHAP_AVAILABLE=False).")

    present = isinstance(shap, dict) and bool(shap)
    if not present:
        verdict = "SIN_DATOS"
        if use_rf is False:
            reasons.append("use_rf_shap=False en la configuración de la última corrida: el RF no se ejecutó.")
        elif use_rf is None:
            reasons.append("No se pudo leer use_rf_shap de la configuración (el valor por defecto del código es False).")
        else:
            reasons.append("use_rf_shap=True pero no hay resultado guardado: el RF no corrió, devolvió "
                           "{'error': ...} (que el pipeline descarta sin guardarlo) o la corrida terminó antes.")
        if meta_cols and doc_meta_columns is not None:
            missing = [c for c in meta_cols if c not in doc_meta_columns]
            if len(missing) == len(meta_cols):
                reasons.append(f"Ninguna de las columnas de multivariate_metadata {meta_cols} aparece en los "
                               "metadatos: el RF responde 'No suitable metadata columns' y se omite.")
        reasons.append("El dashboard muestra el mismo aviso en todos estos casos: no distingue 'no se ejecutó' de 'falló'.")
        return {"verdict": verdict, "reasons": reasons, "sections": {k: False for k in REQUIRED}}

    if "error" in shap:
        reasons.append(f"El resultado guardado contiene error: {shap.get('error')!r}.")
        return {"verdict": "ERROR_GUARDADO", "reasons": reasons, "sections": {k: False for k in REQUIRED}}

    for name, keys in REQUIRED.items():
        sections[name] = all(bool(shap.get(k)) for k in keys)
    missing = [n for n, ok in sections.items() if not ok]
    if missing:
        reasons.append("Datos parciales; no se dibujan: " + "; ".join(missing))

    classes = sorted({int(c) for c in shap.get("classes", [])})
    if stable_cluster_ids is not None and classes:
        now = sorted({int(c) for c in stable_cluster_ids if c is not None and int(c) >= 0})
        if now != classes:
            reasons.append(f"Posibles datos viejos: el RF tiene clases {classes} y la clasificación actual {now}.")
    if n_stable_uces is not None and shap.get("y_true") and len(shap["y_true"]) != n_stable_uces:
        reasons.append(f"y_true tiene {len(shap['y_true'])} filas y hay {n_stable_uces} UCEs estables: el RF "
                       "usó otro conjunto (preprocesamiento con outliers o corrida anterior).")
    stale = any(r.startswith(("Posibles datos viejos", "y_true tiene")) for r in reasons)
    verdict = "DATOS_VIEJOS" if stale else ("PARCIAL" if missing else "OK_DATOS_PRESENTES")
    if verdict == "OK_DATOS_PRESENTES":
        reasons.append("Los datos están completos: si la pestaña D sigue vacía, el problema sería de lectura/caché del dashboard.")
    return {"verdict": verdict, "reasons": reasons, "sections": sections}


def load_postgres(dsn: Optional[str]) -> Dict[str, Any]:
    from reii.backend.sql.db import get_conn

    with get_conn(dsn) as conn:
        def kv(key):
            r = conn.execute("SELECT value_json, updated_at FROM kv_store WHERE key = %s", (key,)).fetchone()
            return (json.loads(r["value_json"]), r["updated_at"]) if r else (None, None)

        shap, shap_ts = kv("shap_analysis")
        cfg, _ = kv("config")
        cfg = dict(cfg or {})
        for r in conn.execute("SELECT key, value_json FROM workflow_config WHERE key IN ('use_rf_shap','multivariate_metadata')").fetchall():
            cfg[r["key"]] = json.loads(r["value_json"])  # sobrescrituras del dashboard
        rows = conn.execute("SELECT cluster_id FROM uces WHERE is_stable = 1").fetchall()
        meta = set()
        for r in conn.execute("SELECT metadata_json FROM documents").fetchall():
            meta |= set(json.loads(r["metadata_json"] or "{}"))
    return {"shap": shap, "shap_ts": shap_ts, "config": cfg,
            "stable_cluster_ids": [r["cluster_id"] for r in rows], "n_stable": len(rows),
            "doc_meta_columns": sorted(meta)}


def load_json(path: str) -> Dict[str, Any]:
    d = json.load(open(path, encoding="utf-8"))
    uces = d.get("uces", [])
    stable = [u for u in uces if u.get("is_stable")]
    meta = set()
    for v in (d.get("doc_metadata") or {}).values():
        meta |= set(v or {})
    return {"shap": d.get("shap_analysis"), "shap_ts": None, "config": d.get("config") or {},
            "stable_cluster_ids": [u.get("cluster_id") for u in stable], "n_stable": len(stable),
            "doc_meta_columns": sorted(meta)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    src = load_json(a.json) if a.json else load_postgres(a.dsn)
    env = check_environment()
    res = analyze(src["shap"], src["config"], src["stable_cluster_ids"], src["n_stable"], src["doc_meta_columns"], env)
    print(f"use_rf_shap (configuración efectiva): {src['config'].get('use_rf_shap')!r}")
    print(f"multivariate_metadata: {src['config'].get('multivariate_metadata')!r}")
    print(f"columnas de metadatos de documentos: {src['doc_meta_columns']}")
    print(f"shap_analysis: {'ausente' if src['shap'] is None else type(src['shap']).__name__} "
          f"(claves: {sorted(src['shap'])[:12] if isinstance(src['shap'], dict) else '-'})  guardado: {src['shap_ts']}")
    print(f"entorno: {env}")
    print(f"\nVEREDICTO: {res['verdict']}")
    for r in res["reasons"]:
        print(" -", r)
    print("\nSecciones de la pestaña D:")
    for k, ok in res["sections"].items():
        print(f"  [{'x' if ok else ' '}] {k}")
    return 0 if res["verdict"].startswith("OK") else 1


if __name__ == "__main__":
    sys.exit(main())
