# reii/backend/steps.py
"""
Framework de steps cacheados con dependencias explícitas.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from reii.backend.sql.db import get_conn, transaction


# ════════════════════════════════════════════════════════════════════════════
# Hashing
# ════════════════════════════════════════════════════════════════════════════
def _canonical_json(obj: Any) -> str:
    """JSON determinista: sorted keys, sin espacios, floats normalizados."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _hash_inputs(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


# ════════════════════════════════════════════════════════════════════════════
# Contexto compartido entre steps
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class WorkflowContext:
    """
    Estado vivo que fluye entre steps. Cada step lee lo que necesita y
    escribe su output acá bajo `ctx.outputs[nombre]`.

    No incluye nada que el step no necesite explícitamente — el contrato
    es que cada step declara sus inputs vía @cached_step(deps=[...]).
    """

    config: Any
    corpus_raw: Any
    outputs: Dict[str, Any] = field(default_factory=dict)
    steps_dir: Path = field(default_factory=lambda: Path("steps"))
    force: Set[str] = field(default_factory=set)  # steps a forzar re-corrida


# ════════════════════════════════════════════════════════════════════════════
# Decorador
# ════════════════════════════════════════════════════════════════════════════
def cached_step(
    name: str,
    version: int = 1,
    deps: Optional[List[str]] = None,
    inputs_fn: Optional[Callable[[WorkflowContext], Any]] = None,
):
    """
    Marca una función como step cacheable.

    Parámetros
    ----------
    name       : identificador único del step
    version    : bumpear cuando cambia la lógica del step
    deps       : lista de nombres de steps de los que depende
    inputs_fn  : función que devuelve el payload a hashear. Por defecto
                 hashea (config, outputs de deps).
    """
    deps = deps or []

    def decorator(fn):
        @wraps(fn)
        def wrapper(ctx: WorkflowContext, *args, **kwargs):
            # 1. Construir payload de hash
            if inputs_fn is not None:
                payload = inputs_fn(ctx)
            else:
                payload = {
                    "config_subset": _config_subset(ctx.config, name),
                    "deps": {d: _hash_output(ctx.outputs.get(d)) for d in deps},
                }
            input_hash = _hash_inputs(payload)

            # 2. ¿Cache válido?
            force = name in ctx.force
            cached = None if force else _load_cached(name, version, input_hash)

            if cached is not None:
                ctx.outputs[name] = cached
                print(f"   [step] {name}  (cached)")
                return cached

            # 3. Ejecutar
            print(f"   [step] {name}  (running)")
            t0 = time.time()
            result = fn(ctx, *args, **kwargs)
            duration_ms = int((time.time() - t0) * 1000)

            # 4. Persistir
            ctx.outputs[name] = result
            _save_cached(name, version, input_hash, result, duration_ms)
            print(f"   [step] {name}  ({duration_ms / 1000:.2f}s)")
            return result

        wrapper._step_name = name
        wrapper._step_version = version
        wrapper._step_deps = deps
        return wrapper

    return decorator


# ════════════════════════════════════════════════════════════════════════════
# Storage
# ════════════════════════════════════════════════════════════════════════════
def _load_cached(name: str, version: int, input_hash: str) -> Optional[Any]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT output_json FROM step_cache
            WHERE step_name = %s AND step_version = %s AND input_hash = %s
            """,
            (name, version, input_hash),
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["output_json"])
        except (json.JSONDecodeError, TypeError):
            return None


def _save_cached(
    name: str, version: int, input_hash: str, output: Any, duration_ms: int
) -> None:
    payload = _canonical_json(output)
    with get_conn() as conn:
        with transaction(conn):
            conn.execute(
                """
                INSERT INTO step_cache
                  (step_name, step_version, input_hash, output_json,
                   created_at, duration_ms, n_bytes)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s)
                ON CONFLICT (step_name, step_version, input_hash) DO UPDATE SET
                  output_json = EXCLUDED.output_json, created_at = CURRENT_TIMESTAMP,
                  duration_ms = EXCLUDED.duration_ms, n_bytes = EXCLUDED.n_bytes
                """,
                (name, version, input_hash, payload, duration_ms, len(payload)),
            )


def _hash_output(obj: Any) -> str:
    if obj is None:
        return "none"
    return _hash_inputs(obj)


# ════════════════════════════════════════════════════════════════════════════
# Config subset (qué parte del config afecta a cada step)
# ════════════════════════════════════════════════════════════════════════════
# Mapeo declarativo: qué claves del Config afectan a qué step.
# Si no se declara nada para un step, se hashea el Config entero.
_STEP_CONFIG_KEYS: Dict[str, List[str]] = {
    "segment": ["spacy_model", "min_uce_words", "uce_target_size"],
    "lemmatize": ["spacy_model", "stem_backend", "use_bigrams", "use_trigrams"],
    "build_ucs": ["min_forms_uc", "tsj", "min_term_abs_freq"],
    "classify": [
        "min_forms_uc",
        "tsj",
        "min_cluster_size_cdh",
        "swap_iterations",
        "min_r2_threshold",
        "n_perm_cdh",
        "classification_mode",
    ],
    "retrofit": ["embedding_model_name", "ppmi_k"],
    "afc": ["pseudocount"],
    "network": ["network_cooccurrence_threshold", "network_weight_method"],
    "rf_shap": ["rf_n_estimators", "rf_max_depth", "multivariate_metadata"],
}


def _config_subset(config: Any, step_name: str) -> Dict[str, Any]:
    keys = _STEP_CONFIG_KEYS.get(step_name)
    if keys is None:
        # Sin declaración → hash del config completo (seguro pero sobre-invalida)
        try:
            return {k: getattr(config, k) for k in vars(config)}
        except TypeError:
            return {"_repr": repr(config)}
    return {k: getattr(config, k, None) for k in keys}
