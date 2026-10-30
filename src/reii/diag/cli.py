"""CLI de diagnóstico de REII: ``python -m reii.diag <comando> [flags]``.

(La forma larga ``python -m reii.diag.cli`` es equivalente.)

  preflight        ¿los artefactos (best_params, estudio, selección) son del
                   corpus en disco? No muta nada; ``--exec`` lanza el plan.
  optuna-db        radiografía READ-ONLY de la BD de Optuna (centinelas, gates,
                   frente de Pareto, ρ parámetro↔objetivo, sign flip).
  uce-size         forma y coste de la segmentación según ``uce_target_size``
                   (lee ``data/diag/bench_uce*.log``).
  trials           estabilidad del optimizador entre semillas TPE.
  db-hygiene       inventario / backup / borrado / vacuum de la BD de Optuna.
  data-hygiene     inventario / rotacion de los temporales de data/ (no toca
                   los canonicos).
  multiverse       multiverso de calidad sobre PostgreSQL (una fila por
                   especificación × target).
  selection-audit  auditoría de la selección de método persistida.
  term-metrics     φ sin Yates, flag FDR y plano factorial F9 desde el
                   artefacto persistido (no relanza el pipeline).

Cada comando reenvía sus propios flags: ``... <comando> --help`` los lista.
En el host sin ``pip install -e .`` usar el shim de la raíz: ``python diag.py``.

Códigos de salida: el del comando (0 ok); 2 = comando desconocido; 130 =
Ctrl-C. Ningún comando de este CLI lanza el pipeline salvo que él mismo lo
declare (``preflight --exec``, ``trials``, ``uce-size --force-run``).
"""

from __future__ import annotations

import importlib
import sys

from reii.diag.common import utf8_stdout

# nombre del comando -> (módulo dentro de reii.diag, ayuda de una línea)
COMMANDS = {
    "preflight": (
        "preflight",
        "artefactos y estudio listos para correr (no muta nada)",
    ),
    "optuna-db": ("optuna_db", "diagnóstico read-only de la BD de Optuna"),
    "uce-size": ("uce_size", "forma y coste de la segmentación por uce_target_size"),
    "trials": ("trials", "estabilidad del optimizador entre semillas"),
    "db-hygiene": ("db_hygiene", "inventario/backup/borrado de la BD de Optuna"),
    "data-hygiene": (
        "data_hygiene",
        "inventario/rotacion de los temporales de data/ (no toca los canonicos)",
    ),
    "multiverse": ("multiverse", "multiverso de calidad (PostgreSQL)"),
    "selection-audit": ("selection_audit", "auditoría de la selección de método"),
    "term-metrics": (
        "term_metrics",
        "φ sin Yates, flag FDR y plano factorial F9 (no relanza nada)",
    ),
}

PROG = "reii.diag"


def _print_help() -> None:
    print(__doc__.splitlines()[0].strip())
    print()
    print("  uso: python -m reii.diag <comando> [flags]")
    print("       (host sin `pip install -e .`: python diag.py <comando> [flags])")
    print()
    width = max(len(name) for name in COMMANDS)
    for name, (_mod, help_) in COMMANDS.items():
        print(f"  {name:<{width}}  {help_}")
    print()
    print("  ... <comando> --help   lista los flags de ese comando")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    utf8_stdout()  # consola cp1252: los acentos de la ayuda no deben reventar
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return 0

    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"[X] comando desconocido: {name!r}")
        print()
        _print_help()
        return 2

    module_name = f"{__package__}.{COMMANDS[name][0]}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        print(f"[X] no se pudo importar {module_name}: {exc}")
        return 2

    try:
        code = module.run(rest)
    except KeyboardInterrupt:
        print("\n[!] interrumpido")
        return 130
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
