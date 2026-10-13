"""Pasos que rodean la ejecución del análisis de discurso: preparar la entrada y sincronizar."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from reii.pipelines import discourse_db as ddb

INPUT_NAME = "discourse_input.json"


def prepare_discourse_input(
    source: str, workflow_db_path: str, adapter=None, data_dir: Optional[str] = None
) -> str:
    """Devuelve la ruta del JSON de entrada del análisis.

    Genera ``discourse_input.json`` desde la BD (ids = uces.uce_id), así lo que se
    anota coincide con lo que muestra el dashboard.
    """
    if adapter is None:
        from reii.backend.dashboard_adapter import DashboardAdapter

        adapter = DashboardAdapter()
    folder = data_dir or os.path.dirname(workflow_db_path) or "."
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, INPUT_NAME)
    rep = ddb.export_discourse_input(adapter, path)
    print(
        f"📦 Entrada generada desde la BD: {rep['n_uces']} UCEs ({rep['n_uces_con_anotaciones']} con anotaciones previas)"
    )
    return path


def sync_after_run(
    state_path: str, dsn: Optional[str] = None, replace: str = "agents"
) -> Dict[str, Any]:
    """Escribe las anotaciones del estado en la tabla ``annotations``.

    El estado se lee de PostgreSQL (``discourse_store``); ``state_path`` se
    conserva por compatibilidad.
    """
    from reii.backend.sql.db import get_conn
    from reii.pipelines import discourse_store

    state = discourse_store.load_state()
    with get_conn(dsn) as conn:
        rep = ddb.sync_state_to_db(conn, state, replace=replace)
    print(
        f"🔗 Anotaciones sincronizadas con la BD: {rep['n_written']}/{rep['n_annotations']} "
        f"(sin resolver: {rep['n_unmatched']})"
    )
    return rep


def main(argv=None) -> int:
    """CLI:  python -m reii.pipelines.runtime sync|export-input [opciones]"""
    import argparse

    from reii.config import DISCOURSE_STATE_PATH, WORKFLOW_DB_PATH

    ap = argparse.ArgumentParser(prog="reii.pipelines.runtime")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser(
        "sync", help="sincroniza discourse_state.json con la tabla annotations"
    )
    s.add_argument("--state", default=DISCOURSE_STATE_PATH)
    s.add_argument("--dsn", default=None)
    s.add_argument("--replace", choices=["agents", "all"], default="agents")
    sub.add_parser("export-input", help="genera discourse_input.json desde la BD")
    a = ap.parse_args(argv)
    if a.cmd == "sync":
        sync_after_run(a.state, a.dsn, a.replace)
    else:
        prepare_discourse_input("db", WORKFLOW_DB_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
