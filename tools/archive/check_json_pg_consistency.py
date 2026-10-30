"""
Diagnóstico: consistencia JSON (data/workflow_data.json) vs PostgreSQL.

Compara los counts de cada tabla derivada del JSON con los counts reales
en PostgreSQL. El JSON es la fuente de verdad de Database.data; si difieren,
un _save() del pipeline reescribiría PG desde el JSON y "corregiría" (o
borraría) datos — por eso conviene detectar la divergencia antes.

Uso (desde el host, con reii-db levantado):

    python scripts/check_json_pg_consistency.py

El script intenta primero el DSN del contenedor (reii-db) y, si no resuelve,
prueba localhost:5433 (puerto publicado por docker-compose). También puedes
forzar el DSN con REII_DATABASE_URL.

Salida: tabla doc | json | pg | diff para cada colección.
Exit code: 0 si todo coincide, 1 si hay divergencias.
"""

from __future__ import annotations

import json
import os
import sys

import psycopg
from psycopg.rows import dict_row

JSON_PATH = os.environ.get("REII_WORKFLOW_DB_PATH", "data/workflow_data.json")
DSN = os.environ.get("REII_DATABASE_URL", "postgresql://reii:reii@reii-db:5432/reii")


def json_counts(data: dict) -> dict:
    net = data.get("network") or {}
    edges = net.get("edges", []) if isinstance(net, dict) else []
    return {
        "documents": len(data.get("doc_metadata", {}) or {}),
        "uces": len(data.get("uces", []) or []),
        "ucs": len(data.get("ucs", []) or []),
        "clusters": len(
            {
                u.get("cluster_id")
                for u in (data.get("uces", []) or [])
                if u.get("cluster_id") is not None
            }
        ),
        "terms": len(data.get("terminos", []) or []),
        "network_edges": len(edges),
    }


def pg_counts(conn: psycopg.Connection) -> dict:
    row = conn.execute(
        """
        SELECT
          (SELECT count(*) FROM documents) AS documents,
          (SELECT count(*) FROM uces) AS uces,
          (SELECT count(*) FROM ucs) AS ucs,
          (SELECT count(*) FROM clusters) AS clusters,
          (SELECT count(*) FROM terms) AS terms,
          (SELECT count(*) FROM network_edges) AS network_edges
        """
    ).fetchone()
    return dict(row)


def _connect() -> psycopg.Connection:
    """Conecta probando el DSN del contenedor y, si falla, localhost:5433."""
    candidates = [DSN]
    if "reii-db" in DSN:
        # El DSN del contenedor trae reii-db:5432; en el host el puerto
        # publicado es 5433. Reemplazamos host:puerto completos.
        candidates.append(DSN.replace("reii-db:5432", "127.0.0.1:5433"))
    last_err: Exception | None = None
    for dsn in candidates:
        try:
            return psycopg.connect(dsn, row_factory=dict_row)
        except psycopg.OperationalError as e:
            last_err = e
    raise last_err  # type: ignore[misc]


def main() -> int:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    jc = json_counts(data)

    conn = _connect()
    try:
        pc = pg_counts(conn)
    finally:
        conn.close()

    print(f"{'colección':<16} {'json':>8} {'pg':>8} {'diff':>6}")
    ok = True
    for key in ("documents", "uces", "ucs", "clusters", "terms", "network_edges"):
        diff = jc[key] - pc[key]
        status = "OK" if diff == 0 else "DIVERGE"
        if diff != 0:
            ok = False
        print(f"{key:<16} {jc[key]:>8} {pc[key]:>8} {diff:>6}  {status}")

    if not ok:
        print(
            "\n[FATAL] JSON y PostgreSQL divergen. No corras un _save() "
            "hasta resolverlo (o usa --force con conocimiento de causa)."
        )
        return 1
    print("\n[OK] JSON y PostgreSQL consistentes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
