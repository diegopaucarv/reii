"""
Verifica que la migración JSON → SQLite preservó los datos.
NO falla: reporta. Útil para correr tras cada migración.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from reii.backend.sql.db import get_conn


def verify(json_path: str, db_path: str) -> bool:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ok = True
    with get_conn(db_path) as conn:
        # ── counts ─────────────────────────────────────────────────────────
        checks = [
            (
                "uces",
                len(data.get("uces", []) or []),
                conn.execute("SELECT COUNT(*) FROM uces").fetchone()[0],
            ),
            (
                "ucs",
                len(data.get("ucs", []) or []),
                conn.execute("SELECT COUNT(*) FROM ucs").fetchone()[0],
            ),
            (
                "terms",
                len(data.get("terminos", []) or []),
                conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0],
            ),
        ]

        print("\n═══ Counts ═══")
        for name, expected, actual in checks:
            flag = "OK " if expected == actual else "!! "
            if expected != actual:
                ok = False
            print(f"  {flag}{name:<14} JSON={expected:>6}  SQL={actual:>6}")

        # ── UCEs estables por cluster ──────────────────────────────────────
        print("\n═══ UCEs por cluster ═══")
        rows = conn.execute("""
            SELECT cluster_id, COUNT(*) AS n,
                   SUM(is_stable) AS n_stable
            FROM uces WHERE cluster_id IS NOT NULL
            GROUP BY cluster_id ORDER BY cluster_id
        """).fetchall()
        for r in rows:
            print(
                f"  cluster {r['cluster_id']:>3}: {r['n']:>6} UCEs "
                f"({r['n_stable']} estables)"
            )

        # ── FKs huérfanas ──────────────────────────────────────────────────
        print("\n═══ Integridad referencial ═══")
        orphans_uces = conn.execute("""
            SELECT COUNT(*) FROM uces u
            LEFT JOIN documents d ON d.doc_id = u.doc_id
            WHERE d.doc_id IS NULL
        """).fetchone()[0]
        orphans_ucs = conn.execute("""
            SELECT COUNT(*) FROM ucs c
            LEFT JOIN documents d ON d.doc_id = c.doc_id
            WHERE d.doc_id IS NULL
        """).fetchone()[0]
        orphans_ann = conn.execute("""
            SELECT COUNT(*) FROM annotations a
            LEFT JOIN uces u ON u.uce_id = a.uce_id
            WHERE u.uce_id IS NULL
        """).fetchone()[0]

        for name, n in [
            ("uces→documents", orphans_uces),
            ("ucs→documents", orphans_ucs),
            ("annotations→uces", orphans_ann),
        ]:
            flag = "OK " if n == 0 else "!! "
            if n != 0:
                ok = False
            print(f"  {flag}{name:<24} huérfanas={n}")

        # ── Muestra aleatoria de UCEs ──────────────────────────────────────
        print("\n═══ Muestra de UCEs (SQL) ═══")
        sample = conn.execute("""
            SELECT uce_id, doc_id, local_idx, seccion,
                   substr(texto, 1, 60) AS snippet
            FROM uces ORDER BY RANDOM() LIMIT 3
        """).fetchall()
        for r in sample:
            print(
                f"  {r['uce_id']:<24} doc={r['doc_id']:<8} "
                f"sec={r['seccion'] or '?':<20} :: {r['snippet']}"
            )

    print("\n✅ Verificación OK" if ok else "\n❌ Verificación FALLÓ")
    return ok


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--db", required=True)
    args = ap.parse_args()
    sys.exit(0 if verify(args.json, args.db) else 1)
