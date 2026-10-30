#!/usr/bin/env python3
"""Valida el perfil ``classic`` de ``--mode classic`` de forma estatica.

No importa ``reii.main_workflow`` (arrastra Stanza/spaCy/DB y tarda minutos) ni
``reii.main_workflow_clasico``. Solo parsea ambos ficheros con ``ast``.

Que comprueba
-------------
1. ``CLASSIC_PROFILE`` (si ya esta en ``main_workflow.py``) o, mientras el diff
   esta pendiente, la copia canonica de este script: cada clave existe como
   campo de ``Config`` y su tipo es compatible con la anotacion. Evita el fallo
   clasico de ``--mode classic`` con una clave inexistente.
2. Tabla ``clave | fork | pipeline | perfil classic | ¿igual al fork?`` usando
   los valores EFECTIVOS (default del dataclass + kwargs del ``Config(...)``
   dentro de ``if __name__ == "__main__"``), no solo el default.
3. ``cdh_enforce_min_leaf is False`` y las claves de innovacion desactivadas.
4. Imprime el perfil como JSON ordenado.

Uso:
    python scripts/playground/classic_profile_check.py            # texto + JSON
    python scripts/playground/classic_profile_check.py --json     # solo JSON

Sin red, sin docker, sin escrituras. < 10 s. Exit 0 si el contrato se cumple.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "src" / "reii" / "main_workflow.py"
FORK = ROOT / "src" / "reii" / "main_workflow_clasico.py"

# ---------------------------------------------------------------------------
# Copia canonica del perfil. Es la fuente de verdad MIENTRAS el diff no este
# aplicado (docs/design_v5/MODE_CLASSIC.md, seccion 5). Cuando el diff este
# aplicado, este script exige que CLASSIC_PROFILE == CANONICAL_CLASSIC_PROFILE
# (guarda anti-deriva), asi que no hay dos definiciones vivas.
# ---------------------------------------------------------------------------
CANONICAL_CLASSIC_PROFILE: Dict[str, Any] = {
    # -- trazabilidad y guardas -------------------------------------------
    "run_mode": "classic",  # campo nuevo (diff): "modern" | "classic"
    "classification_mode": "wc_only",  # solo Metodo A: sin coref (B), sin HDBSCAN (C)
    "optimize": False,  # ALCESTE 2012 fija los hiperparametros
    # -- nucleo CDH (paper 2, paso 2) ------------------------------------
    "cdh_enforce_min_leaf": False,  # sin piso de hoja: cortes 1-vs-(n-1) validos
    "use_ppmi": False,  # campo nuevo (diff): matriz logica binaria
    "min_cluster_size_cdh": 0.10,  # sigue como regla de PARADA de la recursion
    "swap_iterations": 2,
    "min_r2_threshold": 0.05,
    "n_perm_cdh": 100,
    "pseudocount": 0.0,  # epsilon: knob sin consumidor (ver doc, seccion 4)
    # -- segmentacion (paper 2, paso 1) ----------------------------------
    "uce_target_size": 18,  # valor fijo del fork/alceste_config
    "auto_calibrate_uce_size": False,  # sin calibracion bayesiana
    "min_forms_uc": [10, 14],  # paper: dos corridas de 10 vs 14 palabras
    "tsj": 3,
    # -- capas REII que no deben reportarse en A -------------------------
    "use_liminal_projection": False,  # proyeccion abductiva: no es ALCESTE
}

# Claves que el perfil usa pero que aun no existen en Config porque el diff de
# src/reii/main_workflow.py no se ha aplicado. Se declaran con su tipo esperado
# para que un typo SIGA fallando (no es una amnistia general).
PENDING_DIFF_KEYS: Dict[str, str] = {"run_mode": "str", "use_ppmi": "bool"}

# Semantica del fork para claves que el fork NO declara. True = el perfil
# reproduce lo que el fork HACE (aunque no lo nombre).
FORK_ABSENT_EQUIVALENT: Dict[str, Tuple[bool, str]] = {
    "run_mode": (False, "el fork no lo tiene (metadato de corrida)"),
    "use_ppmi": (True, "el fork nunca construye PPMI => equivale a False"),
    "cdh_enforce_min_leaf": (True, "el fork no tiene el fix => equivale a False"),
    "auto_calibrate_uce_size": (True, "el fork no calibra => equivale a False"),
    "use_liminal_projection": (True, "el fork no proyecta => equivale a False"),
    "classification_mode": (
        False,
        "el fork tiene un solo metodo; su __main__ pedia 'all'",
    ),
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------
def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def config_annotations(tree: ast.Module) -> Dict[str, str]:
    """nombre -> anotacion (sin evaluar) de los campos de ``class Config``."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            out: Dict[str, str] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    out[stmt.target.id] = ast.unparse(stmt.annotation)
            return out
    return {}


def config_defaults(tree: ast.Module) -> Dict[str, Any]:
    """nombre -> default literal de ``class Config`` (``<factory>`` si no es literal)."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Config":
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


def _literal_or_factory(node: Optional[ast.AST]) -> Any:
    if node is None:
        return "<sin default>"
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "field":
        for kw in node.keywords:
            if kw.arg == "default_factory":
                body = None
                if isinstance(kw.value, ast.Lambda):
                    body = kw.value.body
                elif isinstance(kw.value, ast.Name):  # default_factory=list
                    return []
                if body is not None:
                    try:
                        return ast.literal_eval(body)
                    except (ValueError, SyntaxError):
                        return "<factory>"
        return "<factory>"
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return "<expr>"


def _main_config_kwargs(tree: ast.Module) -> Dict[str, ast.AST]:
    """kwargs del/los ``Config(...)`` dentro de ``if __name__ == "__main__"``."""
    out: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and getattr(sub.func, "id", None) == "Config"
                ):
                    for kw in sub.keywords:
                        if kw.arg:
                            out[kw.arg] = kw.value
    return out


def _call_configs(tree: ast.Module) -> List[ast.Call]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Config"
    ]


def effective_config(path: Path) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    """(valores efectivos, anotaciones, kwargs desconocidos en __main__)."""
    tree = ast.parse(read(path))
    ann = config_annotations(tree)
    defaults = config_defaults(tree)
    main_kw = _main_config_kwargs(tree)

    effective = dict(defaults)
    unknown: List[str] = []
    for key, node in main_kw.items():
        if key not in ann:
            unknown.append(key)
            continue
        effective[key] = _literal_or_factory(node)
    return effective, ann, unknown


def find_profile(tree: ast.Module) -> Optional[Dict[str, Any]]:
    """``CLASSIC_PROFILE: Dict[str, Any] = {...}`` del pipeline, o None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CLASSIC_PROFILE" for t in node.targets
        ):
            src = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "CLASSIC_PROFILE"
        ):
            src = node.value
        else:
            continue
        try:
            value = ast.literal_eval(src)  # type: ignore[arg-type]
        except (ValueError, SyntaxError):
            return None
        return value if isinstance(value, dict) else None
    return None


# ---------------------------------------------------------------------------
# Comprobaciones
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


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    return str(value)


def render_table(rows: List[Tuple[str, str, str, str]]) -> str:
    header = ("clave", "fork", "pipeline", "perfil classic", "igual al fork")
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(header))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json", action="store_true", help="imprime solo el JSON del perfil"
    )
    ap.add_argument("--quiet", action="store_true", help="solo errores y avisos")
    args = ap.parse_args(argv)

    ok = True
    problems: List[str] = []

    def say(msg: str) -> None:
        if not args.quiet:
            print(msg)

    if not PIPELINE.exists():
        print(f"[X] no encuentro {PIPELINE}")
        return 1

    tree = ast.parse(read(PIPELINE))
    ann = config_annotations(tree)
    pipe_eff, _, _ = effective_config(PIPELINE)
    fork_eff, _, fork_unknown = (
        effective_config(FORK) if FORK.exists() else ({}, {}, [])
    )

    implemented = find_profile(tree)
    profile = implemented if implemented is not None else CANONICAL_CLASSIC_PROFILE
    source = (
        "CLASSIC_PROFILE (implementado)"
        if implemented is not None
        else "CANONICAL_CLASSIC_PROFILE (diff pendiente)"
    )

    say(f"-> pipeline : {PIPELINE.relative_to(ROOT)} ({len(ann)} campos en Config)")
    say(f"-> fork     : {FORK.relative_to(ROOT)} ({len(fork_eff)} campos efectivos)")
    say(f"-> perfil   : {source}")

    # -- 1. claves del perfil contra Config --------------------------------
    missing = [k for k in profile if k not in ann]
    pending = [k for k in missing if k in PENDING_DIFF_KEYS]
    fatal_missing = [k for k in missing if k not in PENDING_DIFF_KEYS]

    for key in profile:
        if key in ann:
            why = type_problem(ann[key], profile[key])
            if why:
                ok = False
                problems.append(f"{key}: {why} (anotacion {ann[key]})")
        elif key in PENDING_DIFF_KEYS:
            why = type_problem(PENDING_DIFF_KEYS[key], profile[key])
            if why:
                ok = False
                problems.append(f"{key} (pendiente): {why}")
        else:
            ok = False
            problems.append(f"{key}: no existe como campo de Config")

    if not fatal_missing:
        say(
            f"[ok] {len(profile) - len(pending)}/{len(profile)} claves existen en Config y el tipo encaja"
        )
    if pending:
        say(
            f"[!] {len(pending)} clave(s) pendientes del diff: {', '.join(sorted(pending))} "
            "(ver docs/design_v5/MODE_CLASSIC.md, seccion 5)"
        )

    # -- 2. invariantes del perfil ----------------------------------------
    if profile.get("cdh_enforce_min_leaf") is False:
        say("[ok] cdh_enforce_min_leaf=False (sin piso duro de hoja)")
    else:
        ok = False
        problems.append("cdh_enforce_min_leaf debe ser False en el perfil classic")

    innovation = {
        "classification_mode": "wc_only",
        "optimize": False,
        "use_liminal_projection": False,
        "auto_calibrate_uce_size": False,
        "use_ppmi": False,
    }
    for key, expected in innovation.items():
        got = profile.get(key, "<ausente>")
        if got == expected:
            say(f"[ok] {key}={fmt(expected)} (innovacion desactivada)")
        else:
            ok = False
            problems.append(
                f"{key} debe ser {fmt(expected)} en classic, es {fmt(got)!r}"
            )

    # -- 3. tabla comparativa ---------------------------------------------
    rows: List[Tuple[str, str, str, str]] = []
    same, differ, n_a = 0, 0, 0
    for key in sorted(profile):
        fork_v = fmt(fork_eff[key]) if key in fork_eff else "—"
        pipe_v = fmt(pipe_eff[key]) if key in pipe_eff else "—"
        if key in fork_eff:
            eq = fork_eff[key] == profile[key]
        elif key in FORK_ABSENT_EQUIVALENT:
            eq = FORK_ABSENT_EQUIVALENT[key][0]
        else:
            eq = None
        if eq is True:
            same += 1
        elif eq is False:
            differ += 1
        else:
            n_a += 1
        rows.append(
            (
                key,
                fork_v,
                pipe_v,
                fmt(profile[key]),
                "si" if eq is True else ("no" if eq is False else "n/a"),
            )
        )

    if not args.json:
        print()
        print(render_table(rows))
        print()

    # -- 4. estado del fork ------------------------------------------------
    if fork_unknown:
        say(
            f"[!] el fork NO es ejecutable: Config() rechaza {len(fork_unknown)} kwarg(s) "
            f"que su __main__ pasa (p.ej. {', '.join(sorted(fork_unknown)[:3])}...)"
        )
    if differ:
        say(
            f"[!] {differ} clave(s) del perfil NO coinciden con el fork: "
            "son desviaciones deliberadas (paper) o bugs del fork; ver doc."
        )
    say(f"[ok] {same} clave(s) coinciden con el fork, {n_a} son n/a")

    # -- 5. salida ---------------------------------------------------------
    payload = {
        "profile": {k: profile[k] for k in sorted(profile)},
        "n_overrides": len(profile),
        "source": "implemented" if implemented is not None else "canonical",
        "diff_pending": implemented is None,
        "pending_keys": sorted(pending),
        "fork_runnable": not fork_unknown,
        "fork_unknown_kwargs": sorted(fork_unknown),
        "matches_fork": same,
        "differs_from_fork": differ,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))

    if implemented is not None:
        if implemented != CANONICAL_CLASSIC_PROFILE:
            ok = False
            problems.append(
                "CLASSIC_PROFILE del pipeline != CANONICAL_CLASSIC_PROFILE (deriva)"
            )

    if problems:
        for p in problems:
            print(f"[X] {p}")
        print("=> estado: FALLO")
        return 1

    print(
        f"=> estado: {'APLICADO' if implemented is not None else 'PENDIENTE (perfil canonico valido)'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
