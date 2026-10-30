#!/usr/bin/env python3
"""Probe del modo clasico (ALCESTE 2012) -- sin docker, sin data/, <60 s.

Valida, SIN instanciar el workflow y SIN importar ``reii.main_workflow``
(importarlo arrastra Stanza: >4 min):

  1. La ``Config`` real del repo (extraida por AST) y la tabla de overrides
     clasicos: la diferencia es EXACTAMENTE la tabla, nada mas.
  2. La politica de parametros desconocidos con 4 fixtures en memoria.
  3. La clave de spec/hash: clasico != moderno (calculada en memoria).
  4. El guard de excepciones del optimizador, con una funcion que lanza.

Si el diff de ``docs/design_v5/CLASSIC_MODE.md`` todavia no se aplico a
``src/reii/main_workflow.py``, el probe lo detecta, avisa con ``[!]`` y usa la
implementacion de referencia de este mismo archivo (misma semantica, sin
verificar el archivo fuente). Con el diff aplicado, valida el fuente real.

Uso:
    python scripts/playground/classic_mode_probe.py
    python scripts/playground/classic_mode_probe.py --verbose

Exit != 0 si algun assert falla.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import json
import logging
import re
import sys
import traceback
import typing
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
MAIN_WORKFLOW = ROOT / "src" / "reii" / "main_workflow.py"
CONFIG_PY = ROOT / "src" / "reii" / "config.py"

# ─────────────────────────────────────────────────────────────────────────────
# Tabla canonica: es la de CLASSIC_MODE.md §2. Si el doc y esto divergen, el
# probe falla (el doc es la fuente, esto es su espejo ejecutable).
# ─────────────────────────────────────────────────────────────────────────────
EXPECTED_OVERRIDES: Dict[str, Any] = {
    "run_mode": "classic",
    "classification_mode": "wc_only",
    "min_forms_uc": [10, 14],
    "cdh_enforce_min_leaf": False,
    "optimize": False,
    "use_embeddings": False,
    "use_liminal_projection": False,
    "auto_calibrate_uce_size": False,
}

# Defaults modernos que la tabla asume. Si cambian, el doc esta desactualizado.
# OJO: classification_mode del DATACLASS es "wc_only"; el default OPERATIVO
# es "select", fijado explicitamente en el bloque __main__ (ver probe_config).
MODERN_BASELINE: Dict[str, Any] = {
    "classification_mode": "wc_only",
    "min_forms_uc": [13, 17],
    "cdh_enforce_min_leaf": True,
    "optimize": True,
    "use_embeddings": True,
    "use_liminal_projection": True,
    "auto_calibrate_uce_size": True,
    "pseudocount": 0.0,
    "tsj": 3,
    "uce_target_size": 26,
    "min_cluster_size_cdh": 0.10,
}

SPEC_FIELDS = (
    "classification_mode",
    "min_forms_uc",
    "tsj",
    "pseudocount",
    "cdh_enforce_min_leaf",
    "min_cluster_size_cdh",
    "min_r2_threshold",
    "uce_target_size",
    "min_uce_words",
    "stem_backend",
    "use_bigrams",
    "use_trigrams",
    "use_embeddings",
    "random_state",
)

NON_CONFIG_PARAM_KEYS = {
    "cdh_target_k",
    "min_forms_uc_1",
    "forms_gap",
    "similarity_threshold",
    "coref_weight",
    "uc_window_size",
}

SENTINEL = (-2.0, -1.0, 0.0)
MAX_TB_CHARS = 4096

_LOGGER = logging.getLogger("classic_mode_probe")
_FAILS: List[str] = []


def ok(msg: str) -> None:
    print(f"[ok] {msg}")


def warn(msg: str) -> None:
    print(f"[!]  {msg}")


def fail(msg: str, detail: str = "") -> None:
    _FAILS.append(msg)
    print(f"[X]  {msg}")
    if detail:
        for line in str(detail).rstrip().splitlines():
            print(f"     {line}")


def check(cond: bool, msg: str, detail: str = "") -> bool:
    if cond:
        ok(msg)
    else:
        fail(msg, detail)
    return bool(cond)


# ─────────────────────────────────────────────────────────────────────────────
# Carga de simbolos desde el fuente, sin importar el modulo
# ─────────────────────────────────────────────────────────────────────────────
def _load_config_module():
    """Carga ``src/reii/config.py`` como modulo suelto (sin el paquete reii)."""
    spec = importlib.util.spec_from_file_location("_probe_reii_config", CONFIG_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"no se pudo cargar {CONFIG_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract_defs(path: Path, names: set) -> Dict[str, ast.AST]:
    """Devuelve {nombre: nodo} para clases/funciones/assigns de nivel superior."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            found[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found[target.id] = node
    return found


def _exec_nodes(nodes: Dict[str, ast.AST], ns: Dict[str, Any], names: set) -> None:
    """Ejecuta en ``ns`` solo los nodos pedidos (preservando orden de fuente)."""
    # dataclasses resuelve tipos via sys.modules[cls.__module__]: el modulo
    # sintetico tiene que existir o @dataclass revienta con AttributeError.
    mod_name = str(ns.get("__name__") or "_probe_main_workflow")
    if mod_name not in sys.modules:
        import types as _types

        sys.modules[mod_name] = _types.ModuleType(mod_name)
    ordered = [nodes[n] for n in names if n in nodes]
    mod = ast.Module(body=ordered, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, "<classic_mode_probe>", "exec"), ns)  # noqa: S102


def load_config_class(cfgmod):
    """Config real del repo via AST (sin importar main_workflow)."""
    nodes = _extract_defs(MAIN_WORKFLOW, {"Config"})
    if "Config" not in nodes:
        raise RuntimeError(f"no se encontro 'class Config' en {MAIN_WORKFLOW}")
    ns: Dict[str, Any] = {
        "__name__": "_probe_main_workflow",
        "dataclass": dataclasses.dataclass,
        "field": dataclasses.field,
        "List": typing.List,
        "Tuple": typing.Tuple,
        "Dict": typing.Dict,
        "Optional": typing.Optional,
        "Any": typing.Any,
        "SPACY_MODEL": getattr(cfgmod, "SPACY_MODEL", ""),
        "TOGETHER_API_KEY": getattr(cfgmod, "TOGETHER_API_KEY", ""),
        "LLM_MODEL": getattr(cfgmod, "LLM_MODEL", ""),
        "CORPUS_NAME": getattr(cfgmod, "CORPUS_NAME", ""),
        "WORKFLOW_DB_PATH": getattr(cfgmod, "WORKFLOW_DB_PATH", ""),
        "DATABASE_URL": getattr(cfgmod, "DATABASE_URL", ""),
    }
    _exec_nodes(nodes, ns, {"Config"})
    return ns["Config"]


def load_source_helper(name: str, extra_ns: Dict[str, Any]):
    """Intenta extraer una funcion/constante del fuente. None si no existe."""
    nodes = _extract_defs(MAIN_WORKFLOW, {name})
    if name not in nodes:
        return None
    ns: Dict[str, Any] = {
        "__name__": "_probe_main_workflow",
        "logging": logging,
        "logger": _LOGGER,
        "traceback": traceback,
        "json": json,
        "hashlib": hashlib,
        "os": __import__("os"),
        "List": typing.List,
        "Dict": typing.Dict,
        "Any": typing.Any,
        "_SENTINEL_TRIAL": SENTINEL,
        "_MAX_TB_CHARS": MAX_TB_CHARS,
    }
    ns.update(extra_ns)
    try:
        _exec_nodes(nodes, ns, {name})
    except Exception as exc:  # noqa: BLE001
        warn(f"no se pudo ejecutar '{name}' del fuente: {exc}")
        return None
    return ns.get(name)


# ─────────────────────────────────────────────────────────────────────────────
# Implementaciones de referencia (misma semantica que el diff del doc)
# ─────────────────────────────────────────────────────────────────────────────
def ref_spec_id(config) -> str:
    if str(getattr(config, "run_mode", "modern")) != "classic":
        return "modern"
    payload = {k: getattr(config, k, None) for k in SPEC_FIELDS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "classic:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def make_resolver(config_cls):
    known = set(config_cls.__dataclass_fields__) | NON_CONFIG_PARAM_KEYS

    def resolve(
        params: Dict[str, Any], where: str = "best_params.json"
    ) -> Dict[str, Any]:
        if not params:
            return {}
        valid, ignored = {}, []
        for key, value in params.items():
            if key in known:
                valid[key] = value
            else:
                ignored.append(str(key))
        if ignored:
            print(
                f"   [!] {where}: {len(ignored)} clave(s) desconocida(s) "
                f"ignorada(s): {', '.join(sorted(ignored))}"
            )
            _LOGGER.warning("%s: claves ignoradas=%s", where, sorted(ignored))
            for key in sorted(ignored):
                print(f"       - {key} = {params[key]!r}")
        if not valid:
            raise ValueError(
                f"{where}: ninguna de las {len(params)} claves es aplicable "
                f"(ignoradas: {', '.join(sorted(ignored))})."
            )
        return valid

    return resolve


def ref_guard(evaluate, trial, failures: List[str]):
    try:
        return evaluate(trial)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        failures.append(f"#{trial.number} {type(exc).__name__}: {exc}")
        del failures[:-50]
        trial.set_user_attr("trial_failed", True)
        trial.set_user_attr("exception_type", type(exc).__name__)
        trial.set_user_attr("exception_msg", str(exc)[:500])
        trial.set_user_attr("exception_tb", tb[-MAX_TB_CHARS:])
        _LOGGER.warning("Trial %d exception: %s", trial.number, exc)
        return SENTINEL


# ─────────────────────────────────────────────────────────────────────────────
# Pruebas
# ─────────────────────────────────────────────────────────────────────────────
def build_classic(modern, config_cls):
    """Aplica la tabla sobre una instancia real; avisa de campos inexistentes."""
    fields = set(config_cls.__dataclass_fields__)
    applied, missing = {}, []
    for key, value in EXPECTED_OVERRIDES.items():
        applied[key] = value
        if key not in fields:
            missing.append(key)
    out = dataclasses.replace(
        modern, **{k: v for k, v in applied.items() if k in fields}
    )
    for key in missing:
        setattr(out, key, applied[key])  # atributo no-field (diff sin aplicar)
    return out, missing


def probe_config(config_cls, verbose: bool) -> Tuple[Any, Any]:
    print("\n== 1. Config clasica vs moderna ==")
    modern = config_cls()
    classic, missing = build_classic(modern, config_cls)

    if missing:
        warn(
            "campos de la tabla que aun NO existen en Config (diff sin aplicar): "
            + ", ".join(missing)
        )
    else:
        ok("todos los campos de la tabla existen en Config")

    bad = {
        k: (getattr(modern, k, "<falta>"), v)
        for k, v in MODERN_BASELINE.items()
        if getattr(modern, k, "<falta>") != v
    }
    check(
        not bad,
        "defaults modernos coinciden con la tabla del doc",
        f"desviaciones: {bad}",
    )

    changed = {
        f.name
        for f in dataclasses.fields(config_cls)
        if getattr(modern, f.name) != getattr(classic, f.name)
    }
    expected_changed = {
        k
        for k, v in EXPECTED_OVERRIDES.items()
        if k in config_cls.__dataclass_fields__ and getattr(modern, k) != v
    }
    noop = {
        k
        for k, v in EXPECTED_OVERRIDES.items()
        if k in config_cls.__dataclass_fields__ and getattr(modern, k) == v
    }
    check(
        changed == expected_changed,
        f"la diferencia es EXACTAMENTE la tabla ({len(expected_changed)} campos)",
        "campos que cambiaron de mas: "
        f"{sorted(changed - expected_changed)}; de menos: "
        f"{sorted(expected_changed - changed)}",
    )
    if noop:
        warn(
            "entradas de la tabla que ya valian eso en el dataclass (no-op): "
            + ", ".join(sorted(noop))
        )

    # El default del dataclass no es el default de la corrida: __main__ fija
    # classification_mode="select" a mano. Si eso cambia, la tabla del doc
    # ("moderno = select") deja de ser cierta.
    src = MAIN_WORKFLOW.read_text(encoding="utf-8")
    check(
        re.search(r'classification_mode\s*=\s*"select"', src) is not None,
        "el default OPERATIVO del pipeline es 'select' (fijado en __main__)",
        'no se encontro classification_mode="select" en main_workflow.py',
    )
    check(
        getattr(classic, "run_mode", None) == "classic",
        "run_mode='classic' marcado para trazabilidad",
    )
    check(
        classic.pseudocount == 0.0,
        "pseudocount sigue 0.0 (paper pide 0.1; epsilon no implementado)",
        f"observado: {classic.pseudocount!r}",
    )

    print("\n   campo                          moderno        clasico")
    for name in sorted(set(MODERN_BASELINE) | expected_changed):
        print(
            f"   {name:<30} {str(getattr(modern, name, '-')):<14} "
            f"{str(getattr(classic, name, '-'))}"
        )
    return modern, classic


def probe_params(config_cls, verbose: bool) -> None:
    print("\n== 2. politica de parametros desconocidos ==")
    from_source = load_source_helper("resolve_best_params", {"Config": config_cls})
    if callable(from_source):
        ok("usando resolve_best_params del FUENTE")
        resolve = from_source
    else:
        warn("resolve_best_params no esta en el fuente -> referencia del probe")
        resolve = make_resolver(config_cls)

    fixtures = [
        (
            "todo valido",
            {"min_forms_uc_1": 11, "forms_gap": 3, "tsj": 4},
            {"min_forms_uc_1": 11, "forms_gap": 3, "tsj": 4},
            None,
        ),
        (
            "una clave basura",
            {"tsj": 4, "quantum_flux": 1},
            {"tsj": 4},
            ["quantum_flux"],
        ),
        (
            "todas basura",
            {"foo": 1, "bar": 2},
            None,
            ["bar", "foo"],
        ),
        (
            "valida + basura",
            {"cdh_target_k": 8, "min_cluster_size_cdh": 0.0625, "bogus": 0},
            {"cdh_target_k": 8, "min_cluster_size_cdh": 0.0625},
            ["bogus"],
        ),
        (
            "params de UCBuilder (no son campos de Config)",
            {"similarity_threshold": 0.35, "coref_weight": 0.1, "uc_window_size": 3},
            {"similarity_threshold": 0.35, "coref_weight": 0.1, "uc_window_size": 3},
            None,
        ),
    ]

    for label, params, expected, ignored in fixtures:
        print(f"\n   fixture: {label}  ->  {params}")
        try:
            observed = resolve(params, where=f"fixture:{label}")
            raised = None
        except ValueError as exc:
            observed, raised = None, exc
        print(f"     esperado : {expected if expected is not None else 'ValueError'}")
        print(f"     observado: {observed if raised is None else 'ValueError'}")
        if raised is None:
            check(
                observed == expected,
                f"{label}: filtrado correcto",
                f"esperado {expected}, observado {observed}",
            )
        else:
            check(
                expected is None,
                f"{label}: no debia lanzar",
                f"ValueError inesperado: {raised}",
            )
    if verbose:
        print("   (los avisos [!] de arriba son la salida esperada del filtro)")


def probe_spec(config_cls, modern, classic, verbose: bool) -> None:
    print("\n== 3. clave de spec/hash ==")
    from_source = load_source_helper("run_spec_id", {"Config": config_cls})
    if callable(from_source):
        ok("usando run_spec_id del FUENTE")
        spec_id = from_source
    else:
        warn("run_spec_id no esta en el fuente -> referencia del probe")
        spec_id = ref_spec_id

    s_modern, s_classic = spec_id(modern), spec_id(classic)
    print(f"   moderno  -> {s_modern}")
    print(f"   clasico  -> {s_classic}")

    check(s_modern == "modern", "el moderno conserva la clave literal 'modern'")
    check(
        s_classic.startswith("classic:") and len(s_classic) == len("classic:") + 12,
        "el clasico usa 'classic:<12 hex>'",
        f"observado: {s_classic}",
    )
    check(s_classic != s_modern, "spec clasico != spec moderno (no comparten cache)")

    variant = dataclasses.replace(classic, min_forms_uc=[11, 15])
    check(
        spec_id(variant) != s_classic,
        "cambiar un parametro de fidelidad cambia el spec_id",
        f"{spec_id(variant)} vs {s_classic}",
    )


class _FakeTrial:
    def __init__(self, number: int = 81):
        self.number = number
        self.attrs: Dict[str, Any] = {}

    def set_user_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value


def probe_guard(config_cls, verbose: bool) -> None:
    print("\n== 4. guard de excepciones del optimizador ==")
    from_source = load_source_helper(
        "guarded_trial_eval",
        {"Config": config_cls, "SENTINEL": SENTINEL},
    )
    if callable(from_source):
        ok("usando guarded_trial_eval del FUENTE")
        guard = from_source
    else:
        warn("guarded_trial_eval no esta en el fuente -> referencia del probe")
        guard = ref_guard

    def boom(_trial):
        raise RuntimeError("svd no converge en el trial 81")

    trial, failures = _FakeTrial(), []
    out = guard(boom, trial=trial, failures=failures)

    print(f"   salida      -> {out}")
    print(f"   attrs       -> {sorted(trial.attrs)}")
    print(f"   failures    -> {failures}")

    check(out == SENTINEL, "una excepcion devuelve _SENTINEL_TRIAL", f"observado {out}")
    check(
        trial.attrs.get("trial_failed") is True,
        "user_attr trial_failed=True",
        f"attrs: {trial.attrs}",
    )
    check(
        trial.attrs.get("exception_type") == "RuntimeError",
        "user_attr exception_type",
        f"observado: {trial.attrs.get('exception_type')!r}",
    )
    tb = str(trial.attrs.get("exception_tb") or "")
    check(
        "RuntimeError" in tb and "boom" in tb,
        "user_attr exception_tb contiene el traceback",
        f"len={len(tb)}",
    )
    check(
        len(tb) <= MAX_TB_CHARS,
        f"traceback truncado a <= {MAX_TB_CHARS} B",
        f"len={len(tb)}",
    )
    check(len(failures) == 1, "contador de fallos incrementado", f"failures={failures}")

    # Y no debe romper el camino feliz.
    trial_ok = _FakeTrial(7)
    good = guard(lambda t: (0.5, 0.6, -7.0), trial=trial_ok, failures=[])
    check(good == (0.5, 0.6, -7.0), "el camino feliz pasa intacto", f"observado {good}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="   [log] %(levelname)s %(message)s"
    )

    print("PROBE MODO CLASICO (ALCESTE 2012) -- sin docker, sin data/, sin Stanza")
    print(f"   fuente -> {MAIN_WORKFLOW.relative_to(ROOT)}")

    if not MAIN_WORKFLOW.exists():
        fail(f"no existe {MAIN_WORKFLOW}")
        return 1

    try:
        cfgmod = _load_config_module()
        config_cls = load_config_class(cfgmod)
        ok(f"Config extraida por AST ({len(config_cls.__dataclass_fields__)} campos)")
    except Exception as exc:  # noqa: BLE001
        fail(f"no se pudo extraer Config: {exc}", traceback.format_exc())
        return 1

    modern, classic = probe_config(config_cls, args.verbose)
    probe_params(config_cls, args.verbose)
    probe_spec(config_cls, modern, classic, args.verbose)
    probe_guard(config_cls, args.verbose)

    print("\n== RESULTADO ==")
    if _FAILS:
        print(f"[X]  {len(_FAILS)} assert(s) fallaron:")
        for item in _FAILS:
            print(f"     - {item}")
        return 1
    print("[ok] todos los asserts pasaron")
    if not all(k in config_cls.__dataclass_fields__ for k in EXPECTED_OVERRIDES):
        print(
            "[!]  el diff de CLASSIC_MODE.md aun NO esta aplicado a "
            "src/reii/main_workflow.py: este run valido la tabla y la semantica "
            "de referencia, no el fuente."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
