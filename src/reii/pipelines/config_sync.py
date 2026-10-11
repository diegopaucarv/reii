"""Una sola configuración para los dos canales del workflow.

Hoy el modo *classic* lee ``workflow_config.json`` y el modo *transformer* lee la tabla
``workflow_config`` (ConfigStore). El dashboard escribe solo el JSON, así que opciones como
``use_rf_shap`` no llegan al modo transformer. ``push_params_to_store`` copia a ConfigStore
los parámetros que son campos válidos del workflow; ``save_unified`` escribe en ambos.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional


def valid_keys() -> List[str]:
    from reii.backend.config_store import ALL_CONFIG_KEYS  # importa psycopg: carga tardía

    return list(ALL_CONFIG_KEYS)


def split_params(params: Dict[str, Any], keys: Optional[Iterable[str]] = None):
    keys = set(keys if keys is not None else valid_keys())
    return ({k: v for k, v in params.items() if k in keys}, {k: v for k, v in params.items() if k not in keys})


def push_params_to_store(params: Dict[str, Any], store=None, keys: Optional[Iterable[str]] = None, by: str = "dashboard-run") -> Dict[str, Any]:
    """Copia a ConfigStore los parámetros válidos. Devuelve {'pushed': [...], 'ignored': [...]}."""
    valid, ignored = split_params(params, keys)
    if valid:
        if store is None:
            from reii.backend.config_store import ConfigStore

            store = ConfigStore()
        store.set_config_many(valid, by)
    return {"pushed": sorted(valid), "ignored": sorted(ignored)}


def read_json_params(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def diff_channels(store_values: Dict[str, Any], json_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Claves definidas en ambos canales con valores distintos."""
    out = []
    for k in sorted(set(store_values) & set(json_params)):
        if store_values[k] != json_params[k]:
            out.append({"key": k, "store": store_values[k], "json": json_params[k]})
    return out


def save_unified(updates: Dict[str, Any], json_path: str, store=None, by: str = "dashboard") -> Dict[str, Any]:
    """Guarda en ConfigStore (claves válidas) y mezcla en el JSON que lee el modo classic."""
    res = push_params_to_store(updates, store=store, by=by)
    current = read_json_params(json_path)
    current.update(updates)
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
    os.replace(tmp, json_path)
    return res


def effective_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Valores efectivos con su origen: {clave: (valor, 'override'|'ultima_corrida')}."""
    out = {k: (v, "ultima_corrida") for k, v in (base or {}).items()}
    for k, v in (overrides or {}).items():
        out[k] = (v, "override")
    return out
