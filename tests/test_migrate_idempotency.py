"""Test de idempotencia de la migración REII (rollback, no corrompe la BD).

Ejecuta la migración completa DOS veces dentro de transacciones que se
revientan (rollback) y verifica que los conteos se mantienen estables y
que no hay huérfanos. La BD real queda intacta.
"""

import json
import os
import sys

os.environ.setdefault("REII_DATABASE_URL", "postgresql://reii:reii@localhost:5433/reii")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from reii.backend import migrate as M  # noqa: E402
from reii.backend.sql.db import connect  # noqa: E402

TABLES = (
    "documents",
    "uces",
    "ucs",
    "clusters",
    "terms",
    "annotations",
    "network_edges",
    "kv_store",
)

EXPECTED = {
    "documents": 28,
    "uces": 5809,
    "ucs": 1757,
    "clusters": 8,
    "terms": 7968,
    "annotations": 0,
    "network_edges": 3179,
    "kv_store": 23,
}


def main() -> int:
    with open("data/workflow_data.json", encoding="utf-8") as f:
        data = json.load(f)

    conn = connect()
    try:

        def counts():
            return {
                t: conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"]
                for t in TABLES
            }

        before = counts()
        print("BEFORE (BD real):", before)

        results = []
        for i in (1, 2):
            try:
                with conn.transaction():
                    M.migrate_documents(conn, data)
                    M.migrate_uces(conn, data)
                    M.migrate_ucs(conn, data)
                    M.migrate_clusters(conn, data)
                    M.migrate_terms(conn, data)
                    M.migrate_annotations(conn, data)
                    M.migrate_network(conn, data)
                    M.migrate_kv(conn, data)
                    c = counts()
                    orphan_ucs = conn.execute(
                        "SELECT count(*) AS n FROM ucs u LEFT JOIN documents d"
                        " ON u.doc_id = d.doc_id WHERE d.doc_id IS NULL"
                    ).fetchone()["n"]
                    orphan_uces = conn.execute(
                        "SELECT count(*) AS n FROM uces u LEFT JOIN documents d"
                        " ON u.doc_id = d.doc_id WHERE d.doc_id IS NULL"
                    ).fetchone()["n"]
                    empty_ucs = conn.execute(
                        "SELECT count(*) AS n FROM ucs WHERE doc_id = ''"
                    ).fetchone()["n"]
                    empty_uces = conn.execute(
                        "SELECT count(*) AS n FROM uces WHERE doc_id = ''"
                    ).fetchone()["n"]
                    print(f"RUN{i}:", c)
                    print(
                        f"RUN{i} huérfanos ucs/uces: {orphan_ucs}/{orphan_uces}"
                        f"  doc_id vacío ucs/uces: {empty_ucs}/{empty_uces}"
                    )
                    results.append(c)
                    raise RuntimeError("ROLLBACK")
            except RuntimeError:
                pass

        after = counts()
        print("AFTER (BD real):", after)
        print("BD real intacta:", before == after)

        ok = True
        for i, c in enumerate(results, 1):
            for t, expected in EXPECTED.items():
                if c[t] != expected:
                    print(f"FALLO RUN{i} {t}: esperado {expected}, obtenido {c[t]}")
                    ok = False
        if len(results) == 2 and results[0] != results[1]:
            print("FALLO: RUN1 != RUN2 (no idempotente)")
            ok = False
        print("RESULTADO:", "OK" if ok else "FALLO")
        return 0 if ok else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
