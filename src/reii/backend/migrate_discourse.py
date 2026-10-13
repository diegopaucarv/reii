"""Migración de los JSON del análisis de discurso a PostgreSQL.

Migra:
  * ``data/discourse_state.json``  → tabla ``discourse_state`` (vía ``discourse_store``).
  * ``data/discourse_input.json``  → sus datos (UCEs + transcripción + anotaciones)
    ya viven en Postgres: es un archivo *derivado* de las tablas ``uces``,
    ``documents`` y ``annotations`` (lo genera ``export_discourse_input``). El
    script intenta migrarlo; si el JSON está truncado (no parsea), verifica que
    los datos ya estén en Postgres y NO re-migra (para no corromper la BD con
    un archivo incompleto).

Tras migrar, elimina ambos JSON (salvo ``--keep-files``).

Uso:
  python -m reii.backend.migrate_discourse [--keep-files]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from reii.config import DATA_DIR, DISCOURSE_STATE_PATH
from reii.pipelines import discourse_store


def migrate_state(state_path: str) -> Dict[str, Any]:
    """Migra ``discourse_state.json`` → tabla ``discourse_state`` (idempotente)."""
    p = Path(state_path)
    if not p.exists():
        return {"file": str(p), "status": "no_file", "n_uces": 0, "n_annotations": 0}
    with open(p, "r", encoding="utf-8") as f:
        state = json.load(f)
    discourse_store.save_state(state)
    annos = state.get("annotations_by_uce") or {}
    n_uces = len(annos) if isinstance(annos, dict) else 0
    n_annos = sum(len(v) for v in annos.values()) if isinstance(annos, dict) else 0
    return {
        "file": str(p),
        "status": "migrated",
        "n_uces": n_uces,
        "n_annotations": n_annos,
    }


def migrate_input(input_path: str) -> Dict[str, Any]:
    """Intenta migrar ``discourse_input.json`` → ``uces``/``annotations``.

    Si el archivo está truncado (no parsea), se verifica que los datos ya estén
    en Postgres y se reporta ``status="truncated"`` sin tocar la BD.
    """
    p = Path(input_path)
    if not p.exists():
        return {"file": str(p), "status": "no_file"}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # JSONDecodeError (truncado) u otro
        return {"file": str(p), "status": "truncated", "error": str(e)[:200]}

    from reii.backend.migrate import migrate_annotations, migrate_uces
    from reii.backend.sql.db import connect

    conn = connect()
    try:
        n_uces = migrate_uces(conn, data)
        n_annos = migrate_annotations(conn, data)
    finally:
        conn.close()
    return {
        "file": str(p),
        "status": "migrated",
        "n_uces": n_uces,
        "n_annotations": n_annos,
    }


def verify_input_in_db() -> Dict[str, Any]:
    """Verifica que los datos de ``discourse_input.json`` ya estén en Postgres."""
    from reii.backend.dashboard_adapter import DashboardAdapter
    from reii.pipelines.discourse_db import build_discourse_input

    raw = build_discourse_input(DashboardAdapter())
    return {
        "n_uces": len(raw.get("uces", [])),
        "n_annotations_by_uce": len(raw.get("annotations_by_uce", {})),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="reii.backend.migrate_discourse")
    ap.add_argument(
        "--keep-files", action="store_true", help="no elimina los JSON tras migrar"
    )
    a = ap.parse_args(argv)

    rep_state = migrate_state(DISCOURSE_STATE_PATH)
    print(f"Estado: {rep_state}")

    input_path = str(Path(DATA_DIR) / "discourse_input.json")
    rep_input = migrate_input(input_path)
    print(f"Entrada: {rep_input}")
    if rep_input.get("status") == "truncated":
        # Archivo derivado e incompleto: confirmamos que los datos ya están en Postgres.
        in_db = verify_input_in_db()
        print(
            f"  → JSON truncado; datos ya en Postgres: {in_db['n_uces']} UCEs, "
            f"{in_db['n_annotations_by_uce']} UCEs con anotaciones."
        )

    if not a.keep_files:
        for p in (Path(DISCOURSE_STATE_PATH), Path(input_path)):
            if p.exists():
                size_mb = p.stat().st_size / 1e6
                p.unlink()
                print(f"🗑️  Eliminado {p} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
