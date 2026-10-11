"""CLI:  python -m reii.quality.cli <comando> [--dsn DSN]

  init                         crea las tablas quality_*
  migrate-legacy [--json P]    migra los resultados ya almacenados (Postgres o JSON)
  runs                         lista corridas y resumen
  evaluate RUN_ID              recalcula niveles/indicadores y los persiste
  set-active RUN_ID [--tier T] fija la corrida por defecto
  selftest                     prueba extremo a extremo con datos sintéticos (SQLite en memoria)
  serve                        lanza el panel Streamlit independiente
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from reii.quality import calibration as cal
from reii.quality import migrate_legacy as ml
from reii.quality import service
from reii.quality.store import QualityStore


def _selftest() -> int:
    rng = np.random.default_rng(0)
    n = 300
    uces = []
    for i in range(n):
        c = int(rng.integers(0, 4))
        stable = rng.random() < 0.55
        uces.append({"uce_id": f"u{i}", "doc_id": f"d{i % 10}", "cluster_id": c if stable else None,
                     "is_stable": stable, "n_tokens": int(rng.integers(8, 30)),
                     "lemmas": ["a", "b", f"t{c}"], "stems": ["a", "b", f"t{c}"]})
    s = QualityStore.open(":memory:")
    rep = ml.migrate(s, {"uces": uces, "vocabulario": ["a", "b", "t0", "t1"], "pairwise_stability": {}, "config": {}, "origin": "selftest"})
    r = service.evaluate_run(s, rep["run_id"], persist=False)
    ok = r["nested_ok"] and abs(r["tiers"]["estricto"]["retention"] - rep["retention"]) < 1e-9
    print(json.dumps({"migracion": rep, "niveles": {k: round(v["retention"], 3) for k, v in r["tiers"].items()}, "anidado": r["nested_ok"]}, ensure_ascii=False))
    print("SELFTEST", "OK" if ok else "FALLÓ")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="reii.quality.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=None, help="PostgreSQL o sqlite:///archivo.db (por defecto REII_QUALITY_DSN / REII_DATABASE_URL)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("migrate-legacy")
    p.add_argument("--json", default=None, help="archivo JSON del workflow (db_local_path); si se omite se lee PostgreSQL")
    p.add_argument("--source-dsn", default=None, help="DSN de PostgreSQL con los resultados previos (por defecto REII_DATABASE_URL)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--label", default=None)
    sub.add_parser("runs")
    p = sub.add_parser("evaluate"); p.add_argument("run_id")
    p = sub.add_parser("set-active"); p.add_argument("run_id"); p.add_argument("--tier", default="estricto")
    sub.add_parser("selftest")
    sub.add_parser("serve")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        return _selftest()
    if a.cmd == "serve":
        app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
        extra = ["--", "--dsn", a.dsn] if a.dsn else []
        return subprocess.call([sys.executable, "-m", "streamlit", "run", app, *extra])

    store = QualityStore.open(a.dsn)
    if a.cmd == "init":
        print("tablas quality_* listas")
    elif a.cmd == "migrate-legacy":
        legacy = ml.load_legacy_json(a.json) if a.json else ml.load_legacy_postgres(a.source_dsn)
        print(json.dumps(ml.migrate(store, legacy, dry_run=a.dry_run, label=a.label), ensure_ascii=False, indent=2))
    elif a.cmd == "runs":
        df = store.list_runs()
        if df.empty:
            print("sin corridas")
        else:
            for r in df.itertuples():
                s = r.summary
                print(f"{r.run_id}  {r.kind:8s} {r.label}  n={r.n_uces}  ret_estricto={s.get('retention_estricto')}  ret_amplio={s.get('retention_amplio')}")
    elif a.cmd == "evaluate":
        r = service.evaluate_run(store, a.run_id, cal.default_profile(), persist=True)
        print(json.dumps({k: {"retencion": round(v["retention"], 4), "UCEs": v["n_assigned"]} for k, v in r["tiers"].items()}, ensure_ascii=False))
    elif a.cmd == "set-active":
        store.set_active("default", a.run_id, a.tier)
        print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
