#!/usr/bin/env python3
"""Valida el contrato del perfil ``classic`` (ALCESTE 2012) de forma estatica.

No importa ``reii.main_workflow`` (arrastra Stanza/spaCy/DB y tarda minutos).
Solo parsea su fuente con ``ast``.

Contrato que valida (docs/design_v5/CLASSIC_MODE.md)
---------------------------------------------------
1. ``CLASSIC_OVERRIDES`` existe y es un dict literal con EXACTAMENTE las 8
   claves del perfil (ni mas ni menos) y los valores del contrato.
2. Cada clave existe como campo de ``Config`` y su tipo encaja con la
   anotacion.
3. ``apply_classic_mode(config, uc_config)`` existe con 2 parametros y su
   cuerpo rechaza ``--optimize-only`` (``_OPTIMIZE_ONLY`` + ``sys.exit(2)``).
4. ``apply_classic_mode`` NO fija ``pseudocount``: el doc lo deja en 0.0 a
   proposito (seccion 3), asi que no puede ser un override.
5. Tabla de valores EFECTIVOS: clave | perfil classic | default del pipeline |
   origen. Los numeros historicos (tsj=3, uce_target_size=26, n_perm_cdh=100,
   min_cluster_size_cdh=0.10, swap_iterations=2, min_r2_threshold=0.05,
   pseudocount=0.0) son defaults del pipeline, no overrides del perfil.

Si ``CLASSIC_OVERRIDES`` no existe se sale con codigo 1: no hay fallback a
ninguna copia canonica. Ese fallback es lo que daba por bueno un perfil que el
codigo ya no implementa.

Uso:
    python scripts/playground/classic_profile_check.py                  # texto
    python scripts/playground/classic_profile_check.py --json           # JSON
    python scripts/playground/classic_profile_check.py --pipeline RUTA  # otro fichero

Sin red, sin docker, sin escrituras. < 10 s. Exit 0 solo si el contrato se
cumple.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE = ROOT / "src" / "reii" / "main_workflow.py"

# El contrato real: 8 overrides (docs/design_v5/CLASSIC_MODE.md seccion 2).
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

# Numeros del paper que NO son overrides: son defaults del pipeline (2.1). Se
# reportan en la tabla para que nadie los confunda con fidelidad del perfil.
HISTORICAL_DEFAULTS: Tuple[str, ...] = (
    "tsj",
    "uce_target_size",
    "n_perm_cdh",
    "min_cluster_size_cdh",
    "swap_iterations",
    "min_r2_threshold",
    "pseudocount",
)

_MISSING = object()


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------
def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def class_annotations(tree: ast.Module, class_name: str) -> Dict[str, str]:
    """nombre -> anotacion (sin evaluar) de los campos de ``class <name>``."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            out: Dict[str, str] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    out[stmt.target.id] = ast.unparse(stmt.annotation)
            return out
    return {}


def class_defaults(tree: ast.Module, class_name: str) -> Dict[str, Any]:
    """nombre -> default literal de ``class <name>`` (``<factory>`` si no)."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            out: Dict[str, Any] = {}
            for stmt in node.body:
                if not (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                ):
                    continue
                out[stmt.target.id] = _literal_or_factory(stmt.value)
            return out
    return {}


def config_annotations(tree: ast.Module) -> Dict[str, str]:
    """nombre -> anotacion de los campos de ``class Config``."""
    return class_annotations(tree, "Config")


def config_defaults(tree: ast.Module) -> Dict[str, Any]:
    """nombre -> default de ``class Config`` (``<factory>`` si no es literal)."""
    return class_defaults(tree, "Config")


def _literal_or_factory(node: Optional[ast.AST]) -> Any:
    if node is None:
        return "<sin default>"
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "field":
        for kw in node.keywords:
            if kw.arg == "default_factory":
                if isinstance(kw.value, ast.Name):  # default_factory=list
                    return []
                if isinstance(kw.value, ast.Lambda):
                    try:
                        return ast.literal_eval(kw.value.body)
                    except (ValueError, SyntaxError):
                        return "<factory>"
                return "<factory>"
        return "<factory>"
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return "<expr>"


def find_overrides(tree: ast.Module) -> Optional[Dict[str, Any]]:
    """``CLASSIC_OVERRIDES`` del pipeline, o None si no existe / no es literal."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "CLASSIC_OVERRIDES" for t in targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError, TypeError):
            return None
        return value if isinstance(value, dict) else None
    return None


def find_function(tree: ast.Module, name: str) -> Optional[ast.AST]:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    return None


def _mentions_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(sub, ast.Name) and sub.id == name for sub in ast.walk(node))


def _exits_with_two(node: ast.AST) -> bool:
    """True si el cuerpo llama a ``exit(2)`` / ``SystemExit(2)`` (o lo eleva)."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        fname = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if fname not in ("exit", "SystemExit") or not sub.args:
            continue
        arg = sub.args[0]
        if isinstance(arg, ast.Constant) and arg.value == 2:
            return True
    return False


def sets_pseudocount(node: ast.AST) -> bool:
    """True si el cuerpo escribe ``pseudocount`` (atributo, item, kw o setattr)."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and sub.attr == "pseudocount"
            and isinstance(sub.ctx, ast.Store)
        ):
            return True
        if isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store):
            sl = sub.slice
            if isinstance(sl, ast.Constant) and sl.value == "pseudocount":
                return True
        if isinstance(sub, ast.Call):
            if any(kw.arg == "pseudocount" for kw in sub.keywords):
                return True
            func = sub.func
            fname = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if fname == "setattr" and len(sub.args) >= 2:
                arg = sub.args[1]
                if isinstance(arg, ast.Constant) and arg.value == "pseudocount":
                    return True
    return False


# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------
_ALLOWED: Dict[str, Tuple[type, ...]] = {
    "bool": (bool,),
    "str": (str,),
    "int": (int,),
    "float": (int, float),
    "list": (list,),
    "dict": (dict,),
    "tuple": (tuple,),
}


def _base_type(annotation: str) -> str:
    a = annotation.strip()
    if a.startswith("Optional[") and a.endswith("]"):
        a = a[len("Optional[") : -1]
    if a.startswith("Union[") and a.endswith("]"):
        a = a.split(",")[0].strip()
    return a


def type_problem(annotation: str, value: Any) -> Optional[str]:
    """None si ``value`` encaja con la anotacion; si no, motivo."""
    base = _base_type(annotation)
    is_list = base.startswith("List[") or base.startswith("list[")
    is_tuple = base.startswith("Tuple[") or base.startswith("tuple[")
    if is_list:
        if not isinstance(value, list):
            return f"se esperaba list, llego {type(value).__name__}"
        inner = base[base.index("[") + 1 : -1]
        for item in value:
            if type_problem(inner, item) is not None:
                return f"elemento {item!r} incompatible con List[{inner}]"
        return None
    if is_tuple:
        return (
            None
            if isinstance(value, (tuple, list))
            else f"se esperaba tuple, llego {type(value).__name__}"
        )
    allowed = _ALLOWED.get(base)
    if allowed is None:
        return None  # anotacion no trivial: no opinamos
    if isinstance(value, bool) and base in ("int", "float"):
        return f"se esperaba {base}, llego bool"
    if base == "int" and isinstance(value, float) and not float(value).is_integer():
        return "se esperaba int, llego float no entero"
    if not isinstance(value, allowed):
        return f"se esperaba {base}, llego {type(value).__name__}"
    return None


# ---------------------------------------------------------------------------
# Presentacion
# ---------------------------------------------------------------------------
def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    return str(value)


def render_table(rows: List[Tuple[str, str, str, str]]) -> str:
    header = ("clave", "perfil classic", "default del pipeline", "origen")
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(header))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _fail(message: str) -> int:
    print(f"[X] {message}")
    print("=> estado: FALLO")
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="imprime el informe JSON")
    ap.add_argument("--quiet", action="store_true", help="solo errores y tabla")
    ap.add_argument(
        "--pipeline",
        type=Path,
        default=DEFAULT_PIPELINE,
        help="ruta de main_workflow.py (default: %(default)s)",
    )
    args = ap.parse_args(argv)

    pipeline: Path = args.pipeline
    problems: List[str] = []

    def say(msg: str) -> None:
        # Con --json no se mezcla prosa con el payload (la linea de estado si va).
        if not args.quiet and not args.json:
            print(msg)

    if not pipeline.exists():
        return _fail(f"no encuentro {pipeline}")

    try:
        tree = ast.parse(read(pipeline))
    except SyntaxError as exc:
        return _fail(f"no puedo parsear {pipeline}: {exc}")

    annotations = config_annotations(tree)
    defaults = config_defaults(tree)
    uc_defaults = class_defaults(tree, "UCBuilderConfig")

    # -- 1. CLASSIC_OVERRIDES: existe, claves exactas ----------------------
    overrides = find_overrides(tree)
    if overrides is None:
        return _fail(
            "no existe CLASSIC_OVERRIDES (o no es un dict literal) en "
            f"{_rel(pipeline)}; el contrato usa CLASSIC_OVERRIDES y no hay copia "
            "canonica de respaldo"
        )

    say(f"-> pipeline : {_rel(pipeline)} ({len(annotations)} campos en Config)")
    say(f"-> perfil   : CLASSIC_OVERRIDES ({len(overrides)} claves)")

    expected_keys = set(EXPECTED_OVERRIDES)
    got_keys = set(overrides)
    missing = sorted(expected_keys - got_keys)
    extra = sorted(got_keys - expected_keys)
    if missing:
        problems.append(f"CLASSIC_OVERRIDES: faltan claves {missing}")
    if extra:
        problems.append(f"CLASSIC_OVERRIDES: sobran claves {extra} (deriva)")
    if not missing and not extra:
        say(
            f"[ok] CLASSIC_OVERRIDES: {len(expected_keys)}/{len(expected_keys)} "
            "claves esperadas, sin claves de mas ni de menos"
        )

    value_drift = [
        f"{key}={fmt(overrides[key])} (esperado {fmt(EXPECTED_OVERRIDES[key])})"
        for key in sorted(expected_keys & got_keys)
        if overrides[key] != EXPECTED_OVERRIDES[key]
    ]
    if value_drift:
        problems.append(
            "CLASSIC_OVERRIDES: valores fuera del contrato -> " + "; ".join(value_drift)
        )

    # -- 2. cada clave es campo de Config con tipo compatible -------------
    type_ok = 0
    for key in sorted(expected_keys & got_keys):
        if key not in annotations:
            problems.append(f"{key}: no existe como campo de Config")
            continue
        why = type_problem(annotations[key], overrides[key])
        if why:
            problems.append(f"{key}: {why} (anotacion {annotations[key]})")
        else:
            type_ok += 1
    for key in extra:
        if key not in annotations:
            problems.append(f"{key}: clave extra que no existe como campo de Config")
    if type_ok == len(expected_keys):
        say(
            f"[ok] {type_ok}/{len(expected_keys)} claves existen en Config y el "
            "tipo encaja"
        )

    # -- 3. apply_classic_mode: firma + guarda --optimize-only ------------
    fn = find_function(tree, "apply_classic_mode")
    n_params = 0
    has_optimize_only = False
    exits_two = False
    touches_epsilon = False
    if fn is None:
        problems.append("no existe apply_classic_mode(config, uc_config)")
    else:
        a = fn.args  # type: ignore[attr-defined]
        n_params = len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs)
        has_optimize_only = _mentions_name(fn, "_OPTIMIZE_ONLY")
        exits_two = _exits_with_two(fn)
        touches_epsilon = sets_pseudocount(fn)

        if n_params == 2:
            say("[ok] apply_classic_mode con 2 parametros (config, uc_config)")
        else:
            problems.append(
                f"apply_classic_mode tiene {n_params} parametros, se esperaban 2"
            )

        if has_optimize_only and exits_two:
            say(
                "[ok] apply_classic_mode rechaza --optimize-only "
                "(_OPTIMIZE_ONLY + exit 2)"
            )
        else:
            falta = []
            if not has_optimize_only:
                falta.append("_OPTIMIZE_ONLY")
            if not exits_two:
                falta.append("SystemExit/exit 2")
            problems.append("apply_classic_mode no menciona " + " ni ".join(falta))

    # -- 3b. no fija pseudocount (epsilon queda en el default) ------------
    if fn is not None:
        if touches_epsilon:
            problems.append(
                "apply_classic_mode fija pseudocount; el perfil debe dejarlo en "
                "el default 0.0 (docs/design_v5/CLASSIC_MODE.md seccion 3)"
            )
        else:
            say(
                "[ok] apply_classic_mode no fija pseudocount (queda en el default "
                "del pipeline)"
            )

    # -- 4. valores historicos: defaults del pipeline, no del perfil ------
    leaked = [key for key in HISTORICAL_DEFAULTS if key in overrides]
    if leaked:
        problems.append(
            "valores historicos que NO deben ser overrides del perfil: "
            + ", ".join(leaked)
        )
    absent = [
        key
        for key in HISTORICAL_DEFAULTS
        if key not in defaults and key not in uc_defaults and key not in overrides
    ]
    if absent:
        problems.append(
            "valores historicos sin default en el pipeline: " + ", ".join(absent)
        )
    if not leaked and not absent:
        say(
            f"[ok] {len(HISTORICAL_DEFAULTS)} valores historicos son defaults del "
            "pipeline, no overrides del perfil"
        )

    # -- 5. tabla de valores EFECTIVOS ------------------------------------
    keys = sorted(overrides) + [k for k in HISTORICAL_DEFAULTS if k not in overrides]
    rows: List[Tuple[str, str, str, str]] = []
    for key in keys:
        is_override = key in overrides
        if is_override:
            effective = overrides[key]
        else:
            effective = defaults.get(key, uc_defaults.get(key, _MISSING))
        default_v = defaults.get(key, uc_defaults.get(key, _MISSING))
        rows.append(
            (
                key,
                fmt(effective) if effective is not _MISSING else "<ausente>",
                fmt(default_v) if default_v is not _MISSING else "<ausente>",
                "override" if is_override else "default",
            )
        )

    if not args.json and not args.quiet:
        print()
        print(render_table(rows))
        print()

    # -- 6. informe y estado ----------------------------------------------
    ok = not problems
    payload = {
        "ok": ok,
        "estado": "OK" if ok else "FALLO",
        "pipeline": _rel(pipeline),
        "expected_overrides": {
            k: EXPECTED_OVERRIDES[k] for k in sorted(EXPECTED_OVERRIDES)
        },
        "classic_overrides": {k: overrides[k] for k in sorted(overrides)},
        "missing_keys": missing,
        "extra_keys": extra,
        "value_drift": value_drift,
        "apply_classic_mode": {
            "found": fn is not None,
            "n_params": n_params,
            "mentions_optimize_only": has_optimize_only,
            "exits_2": exits_two,
            "sets_pseudocount": touches_epsilon,
        },
        "effective_values": [
            {
                "clave": key,
                "perfil_classic": classic_v,
                "default_pipeline": default_v,
                "origen": origin,
            }
            for key, classic_v, default_v, origin in rows
        ],
        "problems": problems,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))

    for problem in problems:
        print(f"[X] {problem}")
    print(f"=> estado: {'OK' if ok else 'FALLO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
