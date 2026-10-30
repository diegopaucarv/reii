"""Invariantes del allowlist de configuracion (dashboard <-> pipeline).

``ConfigStore.set_config_many`` filtra por ``ALL_CONFIG_KEYS``, mientras el
camino de aplicacion (``main_workflow``) filtra por ``__dataclass_fields__``.
Si una clave esta en el segundo y no en el primero, la fila de la BD se
aplicaria pero nadie podria escribirla: el knob queda invisible en el
dashboard aunque el optimizador lo mueva (#24-#27, #73).

Estos tests fijan las dos direcciones y las exclusiones deliberadas.
"""

from __future__ import annotations

import ast
from pathlib import Path

from reii.backend.config_store import ALL_CONFIG_KEYS, CONFIG_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "src" / "reii" / "main_workflow.py"

# Knobs que el optimizador mueve o que best_params.json persiste.
REQUIRED = {
    "hdbscan_uce_min_cluster_frac",
    "hdbscan_uce_min_cluster_frac_loose",
    "hdbscan_uce_min_cluster_floor",
    "hdbscan_uce_min_samples_ratio",
    "cdh_enforce_min_leaf",
    "method_c_backend",
    "method_c_hdbscan_selectable",
    "method_c_target_k",
    "dense_cdh_min_r2_1d",
    "dense_cdh_min_r2_dense",
    "dense_cdh_loose_factor",
    "retention_policy",
    "method_selection_max_class_share",
    "method_selection_null_mode",
    "optimize_coverage_gate_select",
}

# Excluidas a proposito; ver el comentario de ``lexico_y_gramatica``.
EXCLUDED = {"content_pos", "run_mode", "corpus_name"}


def _config_fields() -> set:
    """Campos de ``Config`` leidos con ``ast`` (importarlo cuesta ~60 s)."""
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            return {
                s.target.id
                for s in node.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            }
    raise AssertionError("no existe Config en main_workflow.py")


def test_required_knobs_are_writable() -> None:
    missing = sorted(REQUIRED - set(ALL_CONFIG_KEYS))
    assert not missing, "knobs aplicables pero no escribibles: %s" % missing


def test_excluded_knobs_stay_out() -> None:
    present = sorted(EXCLUDED & set(ALL_CONFIG_KEYS))
    assert not present, "claves excluidas por diseno: %s" % present


def test_allowlist_is_subset_of_config_without_duplicates() -> None:
    unknown = sorted(set(ALL_CONFIG_KEYS) - _config_fields())
    assert not unknown, "claves del allowlist sin campo en Config: %s" % unknown
    flat = [key for keys in CONFIG_CATEGORIES.values() for key in keys]
    dups = sorted({key for key in flat if flat.count(key) > 1})
    assert not dups, "claves en dos categorias: %s" % dups
