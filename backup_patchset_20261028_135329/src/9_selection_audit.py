#!/usr/bin/env python3
"""
Script 9 - Auditor de la seleccion de metodo (A / B / C) persistida.

Pregunta que responde: la eleccion automatica de metodo, SE MANTIENE? Si con el
mismo corpus el ganador cambia al mover uce_target_size, entonces la eleccion no
es identificable: es un artefacto de la segmentacion, no una propiedad de los
datos. Eso es exactamente el dato que hay que ver antes de escribir nada.

Fuente de datos (no recalcula nada):
  * data/method_selection_log.jsonl  -> una linea JSON por corrida, con
    context = {corpus_fingerprint, uce_target_size, mode, n_uces, seed,
               written_at} y selection = {...}.
    Lo escribe reii/method_selection.persist_selection desde
    DoubleClassifier._persist_method_selection (modo classification_mode=select).
  * data/method_selection.json       -> snapshot de la ultima corrida.

Agrupacion: por corpus_fingerprint (hash de los archivos de entrada). NO por
mode (todos son "select") ni por target (es la variable que se barre).

Uso (dentro del contenedor: necesita data/ montado, no la raiz):
  docker compose --profile batch run --rm workflow python -u src/9_selection_audit.py
  ... --history data/method_selection_log.jsonl --json data/diag/selection_audit.json
  ... --strict            # sale con 1 si el ganador cambia dentro de un corpus

Salida: tabla ASCII + veredicto. Sin emojis ni box-drawing (consola cp1252).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from reii.config import DATA_DIR
from reii.method_selection import audit_history, audit_text, read_history

DEFAULT_HISTORY = DATA_DIR / "method_selection_log.jsonl"
DEFAULT_SNAPSHOT = DATA_DIR / "method_selection.json"


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _load_snapshot(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main() -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--history", default=str(DEFAULT_HISTORY))
    ap.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    ap.add_argument("--json", default=None, help="ruta para volcar el auditor")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 si el ganador cambia dentro de un mismo corpus",
    )
    a = ap.parse_args()

    history = Path(a.history)
    records = read_history(str(history))
    if not records:
        print(f"[X] sin historial en {history}")
        print(
            "    se llena al correr con --mode select "
            "(una linea por corrida, append-only)"
        )
        return 2

    print(audit_text(records))

    snap = _load_snapshot(Path(a.snapshot))
    if snap:
        ctx = snap.get("context") or {}
        sel = snap.get("selection") or {}
        print("")
        print(f"ULTIMA CORRIDA ({ctx.get('written_at') or '?'})")
        print(
            f"  corpus={ctx.get('corpus_fingerprint')}  "
            f"target={ctx.get('uce_target_size')}  n_uces={ctx.get('n_uces')}"
        )
        print(
            f"  ganador={sel.get('winner')}  K={sel.get('winner_k')}  "
            f"ARI={sel.get('winner_ari')}  coverage={sel.get('winner_coverage')}"
        )

    report = audit_history(records)
    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nResultados: {out}")

    unstable = bool(report["instability"])
    print("")
    if unstable:
        print("VEREDICTO: la eleccion NO es estable -> no se persiste como decision.")
        return 1 if a.strict else 0
    print("VEREDICTO: eleccion estable en todos los targets medidos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
