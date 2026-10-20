"""Regresión: defectos que inflan el coste del optimizador (sin imports pesados).

1. Un cache coref recargado de disco no acierta por tipo de clave (str vs int).
2. Optimizador._perturb_params devuelve copias idénticas (Random re-sembrado en
   cada llamada), así que el warm-start encola trials duplicados.
"""
import ast
import random
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reii.coref_cache import (
    atomic_write_json,
    build_cache_document,
    get_doc_chains,
    load_cache,
)

SRC = Path(__file__).resolve().parent.parent / "src" / "reii" / "main_workflow.py"


def test_naive_lookup_misses_after_disk_roundtrip(tmp_path):
    """Documenta el defecto: `doc_id in cache` falla tras recargar (UCE.doc_id es int)."""
    chains = {2: {0: [{"mentions": []}], 1: []}}
    p = str(tmp_path / "c.json")
    atomic_write_json(p, build_cache_document("k", "fp", chains))
    loaded = load_cache(p, "k")
    assert 0 in chains[2]
    assert 0 not in loaded[2]  # ← el re-extract por trial


def test_get_doc_chains_hits_after_roundtrip(tmp_path):
    chains = {2: {0: [{"mentions": [1]}], 1: []}}
    p = str(tmp_path / "c.json")
    atomic_write_json(p, build_cache_document("k", "fp", chains))
    loaded = load_cache(p, "k")
    assert get_doc_chains(loaded[2], 0) == [{"mentions": [1]}]
    assert get_doc_chains(loaded[2], 1) == []  # lista vacía ≠ ausente
    assert get_doc_chains(loaded[2], 99) is None
    assert get_doc_chains(chains[2], 0) == [{"mentions": [1]}]  # en memoria también


def _load_perturb():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_perturb_params":
            src = textwrap.dedent(ast.get_source_segment(SRC.read_text(encoding="utf-8"), node))
            ns = {"Dict": dict, "Any": object}
            exec(src, ns)
            return ns["_perturb_params"]
    raise AssertionError("_perturb_params no encontrado")


class _Cfg:
    random_state = 42


class _Opt:
    config_base = _Cfg()
    _search_space_cached = {
        "a": (2, 9), "b": [0.1, 0.2, 0.3, 0.4], "c": [1, 2, 3, 4, 5], "d": (0.0, 1.0),
    }


def test_perturb_copies_must_differ():
    perturb = _load_perturb()
    params = {"a": 5, "b": 0.2, "c": 3, "d": 0.5}
    copies = [perturb(_Opt(), params, salt=i) for i in range(2)]
    # Antes del fix: TypeError (sin `salt`) y, sin él, copias idénticas.
    assert copies[0] != copies[1], "las 2 copias perturbadas del warm-start son idénticas"
