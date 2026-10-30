"""Contrato del perfil ``--mode classic`` (ALCESTE Reinert puro).

No importa ``reii.main_workflow``: ese modulo abre Postgres, lee ``data/`` y
carga spaCy/Stanza al importarse (minutos). Aqui se parsea su fuente con
``ast`` y se ejecuta, aislada, la funcion que aplica el perfil.

Mientras el diff de ``main_workflow.py`` no este aplicado (ver
``docs/design_v5/CLASSIC_MODE.md``), los tests del contrato se saltan con un
motivo explicito; ``test_checker_script_passes`` corre siempre.
"""

import ast
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import types
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "src" / "reii" / "main_workflow.py"
CHECKER = ROOT / "scripts" / "playground" / "classic_profile_check.py"

SOURCE = MAIN.read_text(encoding="utf-8", errors="replace")
TREE = ast.parse(SOURCE)

_MISSING = object()


def _config_node() -> ast.ClassDef:
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            return node
    raise AssertionError("no encuentro class Config")


def config_fields() -> Dict[str, str]:
    """nombre -> anotacion de los campos de Config."""
    out = {}
    for stmt in _config_node().body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            out[stmt.target.id] = ast.unparse(stmt.annotation)
    return out


def config_defaults() -> Dict[str, Any]:
    """nombre -> default literal de Config (los no literales se omiten)."""
    out: Dict[str, Any] = {}
    for stmt in _config_node().body:
        if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
            continue
        try:
            out[stmt.target.id] = ast.literal_eval(stmt.value)
        except (TypeError, ValueError):
            continue
    return out


def module_value(name: str) -> Any:
    """Valor literal de una constante de modulo (p.ej. _CLASSIC_SPEC_FIELDS)."""
    for node in ast.walk(TREE):
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"no existe {name} en main_workflow.py")


def read_profile() -> Optional[Dict[str, Any]]:
    """CLASSIC_OVERRIDES del pipeline, o None si el diff no esta aplicado."""
    for node in ast.walk(TREE):
        ok = False
        if isinstance(node, ast.Assign):
            ok = any(
                isinstance(t, ast.Name) and t.id == "CLASSIC_OVERRIDES"
                for t in node.targets
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            ok = node.target.id == "CLASSIC_OVERRIDES"
        if not ok:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None
    return None


PROFILE = read_profile()

requiere_diff = pytest.mark.skipif(
    PROFILE is None,
    reason=(
        "diff pendiente: falta CLASSIC_OVERRIDES en src/reii/main_workflow.py "
        "(docs/design_v5/CLASSIC_MODE.md)"
    ),
)


def effective_value(key: str) -> Any:
    """Valor efectivo en classic: override si existe, si no default de Config."""
    if PROFILE is not None and key in PROFILE:
        return PROFILE[key]
    return config_defaults().get(key, _MISSING)


def function_source(name: str) -> str:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(SOURCE, node)
            assert segment
            return segment
    raise AssertionError(f"no existe {name} en main_workflow.py")


def enclosing_if_tests(node: ast.AST) -> List[str]:
    parents = {}
    for parent in ast.walk(TREE):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    tests = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.If):
            tests.append(ast.unparse(current.test))
        current = parents.get(current)
    return tests


def compatible(annotation: str, value: Any) -> bool:
    base = annotation.strip()
    if base.startswith("Optional[") and base.endswith("]"):
        base = base[len("Optional[") : -1]
    if base.startswith("List[") or base.startswith("list["):
        if not isinstance(value, list):
            return False
        inner = base[base.index("[") + 1 : -1]
        return all(compatible(inner, item) for item in value)
    if base in ("int", "float") and isinstance(value, bool):
        return False
    if base == "float":
        return isinstance(value, (int, float))
    if base == "int":
        return isinstance(value, int)
    if base == "str":
        return isinstance(value, str)
    if base == "bool":
        return isinstance(value, bool)
    return True


def profile_dataclass() -> type:
    """Dataclass minimo con lo que apply_classic_mode lee y escribe.

    Los 8 overrides + pseudocount (que el perfil NO toca) + los tres destinos
    de aislamiento que la funcion escribe sobre el Config.
    """
    defaults: Dict[str, Any] = dict(PROFILE or {})
    defaults.update(
        {
            "pseudocount": 0.0,
            "db_local_path": "",
            "optimize_storage": None,
            "optimize_study_name": "reii_search",
        }
    )
    fields = []
    for key, value in defaults.items():
        if isinstance(value, list):
            fields.append((key, List[int], dataclasses.field(default_factory=list)))
        else:
            fields.append((key, type(value), dataclasses.field(default=value)))
    return dataclasses.make_dataclass("Config", fields)


def classic_namespace(tmp_path: Path, *, optimize_only: bool) -> Dict[str, Any]:
    """Ejecuta run_spec_id y apply_classic_mode reales, aislados del pipeline."""

    def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    fake_os = types.SimpleNamespace(path=os.path, sep=os.sep, makedirs=_noop)
    namespace: Dict[str, Any] = {
        "sys": sys,
        "os": fake_os,
        "dataclasses": dataclasses,
        "json": json,
        "hashlib": hashlib,
        "Any": typing.Any,
        "Dict": typing.Dict,
        "List": typing.List,
        "Optional": typing.Optional,
        "Tuple": typing.Tuple,
        "Config": profile_dataclass(),
        "CLASSIC_DATA_DIR": str(tmp_path / "classic_data"),
        "CLASSIC_OUTPUT_DIR": str(tmp_path / "classic_output"),
        "CLASSIC_OVERRIDES": dict(PROFILE or {}),
        "_CLASSIC_SPEC_FIELDS": module_value("_CLASSIC_SPEC_FIELDS"),
        "_OPTIMIZE_ONLY": optimize_only,
    }
    exec(function_source("run_spec_id"), namespace)  # noqa: S102
    exec(function_source("apply_classic_mode"), namespace)  # noqa: S102
    return namespace


# ---------------------------------------------------------------------------
# 1. el perfil existe
# ---------------------------------------------------------------------------
@requiere_diff
def test_classic_profile_exists():
    assert isinstance(PROFILE, dict)
    assert len(PROFILE) >= 8, f"perfil sospechosamente corto: {sorted(PROFILE)}"


# ---------------------------------------------------------------------------
# 2. todas sus claves existen en Config, con tipo compatible
# ---------------------------------------------------------------------------
@requiere_diff
def test_all_profile_keys_exist_in_config():
    annotations = config_fields()
    missing = sorted(key for key in PROFILE if key not in annotations)
    assert not missing, f"claves sin campo en Config: {missing}"


@requiere_diff
def test_profile_values_have_compatible_types():
    fields = config_fields()
    for key, value in PROFILE.items():
        annotation = fields[key]
        assert compatible(annotation, value), (
            f"{key}={value!r} no encaja con {annotation!r}"
        )


@requiere_diff
def test_config_annotations_are_readable():
    """Sanity: el parser ve una cantidad razonable de campos."""
    assert len(config_fields()) > 100


# ---------------------------------------------------------------------------
# 3. cdh_enforce_min_leaf is False + innovacion apagada
# ---------------------------------------------------------------------------
@requiere_diff
def test_cdh_enforce_min_leaf_is_false():
    assert PROFILE.get("cdh_enforce_min_leaf") is False


@requiere_diff
def test_innovation_switches_are_off():
    assert PROFILE.get("classification_mode") == "wc_only"  # sin coref ni HDBSCAN
    assert PROFILE.get("optimize") is False
    assert PROFILE.get("use_liminal_projection") is False
    assert PROFILE.get("auto_calibrate_uce_size") is False
    assert PROFILE.get("run_mode") == "classic"


@requiere_diff
def test_valores_historicos_son_los_defaults_efectivos():
    """Los numeros del paper son el valor EFECTIVO del escenario classic.

    Se afirma el valor efectivo (default de Config o override del perfil), no
    la pertenencia a CLASSIC_OVERRIDES: si alguien cambia un default del
    pipeline, classic deja de ser fiel a 2012 y este test lo caza.
    """
    esperado = {
        "uce_target_size": 26,
        "tsj": 3,
        "min_cluster_size_cdh": 0.1,
        "swap_iterations": 2,
        "n_perm_cdh": 100,
        "pseudocount": 0.0,
        "min_forms_uc": [10, 14],
    }
    for key, expected in esperado.items():
        got = effective_value(key)
        assert got is not _MISSING, f"{key} no esta ni en el perfil ni en Config"
        assert got == expected, f"{key}: efectivo {got!r}, se esperaba {expected!r}"


# ---------------------------------------------------------------------------
# 4. incompatibilidad con --optimize-only + aplicacion correcta
# ---------------------------------------------------------------------------
@requiere_diff
def test_optimize_only_is_rejected_loudly(tmp_path, capsys):
    ns = classic_namespace(tmp_path, optimize_only=True)
    with pytest.raises(SystemExit) as excinfo:
        ns["apply_classic_mode"](ns["Config"](), object())
    assert excinfo.value.code == 2
    out = capsys.readouterr().out
    assert "[X]" in out
    assert "--optimize-only" in out


@requiere_diff
def test_profile_is_applied(tmp_path, capsys):
    ns = classic_namespace(tmp_path, optimize_only=False)
    uc_config = object()
    classic, uc = ns["apply_classic_mode"](ns["Config"](), uc_config)
    assert uc is uc_config
    for key, expected in PROFILE.items():
        assert getattr(classic, key) == expected, f"{key} no se aplico"
    assert classic.run_mode == "classic"
    # El paper pide 0.1 pero la CDH no consume epsilon: se declara el default.
    assert classic.pseudocount == 0.0
    assert classic.db_local_path.startswith(ns["CLASSIC_DATA_DIR"])
    out = capsys.readouterr().out
    assert "[CLI] --mode classic" in out
    assert ns["CLASSIC_DATA_DIR"] in out


# ---------------------------------------------------------------------------
# 5. select / wc_only no cambian
# ---------------------------------------------------------------------------
@requiere_diff
def test_profile_is_applied_only_under_classic_mode():
    calls = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_classic_mode"
    ]
    assert len(calls) == 1, f"esperaba 1 call site, hay {len(calls)}"
    tests = enclosing_if_tests(calls[0])
    assert any("classic" in test for test in tests), tests


@requiere_diff
def test_select_and_wc_only_keep_their_own_path():
    """El override directo de classification_mode sigue excluyendo classic."""
    replaces = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
        and any(kw.arg == "classification_mode" for kw in node.keywords)
    ]
    assert replaces, "desaparecio el override de --mode select|wc_only"
    tests = enclosing_if_tests(replaces[0])
    assert any("classic" in test for test in tests), tests


@requiere_diff
def test_mode_enum_keeps_select_and_wc_only():
    modes = set()
    for node in ast.walk(TREE):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_CLI_MODES_VALIDOS"
            for t in node.targets
        ):
            modes = set(ast.literal_eval(node.value))
    assert {"select", "wc_only", "classic"} <= modes, sorted(modes)


@requiere_diff
def test_default_run_mode_is_modern():
    for stmt in _config_node().body:
        if (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "run_mode"
        ):
            assert isinstance(stmt.value, ast.Constant)
            assert stmt.value.value == "modern"
            return
    pytest.fail("falta run_mode en Config")


# ---------------------------------------------------------------------------
# ancla siempre activa: el validador estatico
# ---------------------------------------------------------------------------
def test_checker_script_passes():
    proc = subprocess.run(
        [sys.executable, str(CHECKER)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "=> estado:" in proc.stdout
    # El checker ya valida el contrato real: nada de "perfil pendiente".
    assert "PENDIENTE" not in proc.stdout
    assert "CLASSIC_OVERRIDES" in proc.stdout


def _source_without_module_assignment(name: str) -> str:
    """``SOURCE`` sin la asignacion de modulo ``name`` (p.ej. CLASSIC_OVERRIDES)."""
    for node in TREE.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if isinstance(target, ast.Name) and target.id == name:
            lines = SOURCE.splitlines(keepends=True)
            return "".join(lines[: node.lineno - 1] + lines[node.end_lineno :])
    raise AssertionError(f"no existe {name} a nivel de modulo")


# Forma minima del pipeline DESPUES de borrar el bloque CLASSIC_OVERRIDES. Se
# usa un fichero corto a proposito: parsear las 11.5k lineas reales cuesta ~1 s
# en este entorno y el test debe quedar < 2 s.
_STRIPPED_PIPELINE = """class Config:
    run_mode: str = "modern"


def apply_classic_mode(config, uc_config):
    return config, uc_config, CLASSIC_OVERRIDES
"""


def test_checker_fails_without_classic_overrides(tmp_path):
    """Guard: sin CLASSIC_OVERRIDES el checker falla (no hay fallback silencioso).

    ``--pipeline`` apunta a la forma que deja el bloque borrado. El pipeline real
    si declara la asignacion y el helper la quita del texto; aqui solo se valida
    el guard, sin importar ``reii`` ni parsear el fichero completo. < 2 s.
    """
    stripped = _source_without_module_assignment("CLASSIC_OVERRIDES")
    assert "CLASSIC_OVERRIDES: Dict[str, Any] = {" not in stripped

    pipeline = tmp_path / "main_workflow.py"
    pipeline.write_text(_STRIPPED_PIPELINE, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(CHECKER), "--pipeline", str(pipeline)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        cwd=str(ROOT),
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "CLASSIC_OVERRIDES" in proc.stdout
    assert "=> estado:" in proc.stdout
