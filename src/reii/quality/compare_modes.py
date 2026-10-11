"""Tablas comparativas entre corridas (modos de clasificación) para el artículo.

    python -m reii.quality.compare_modes --runs run-a,run-b,run-c \\
        --tier estricto --out comparison_table.csv [--compute-silhouette]

Escribe dos CSV (esquemas distintos, por eso no se mezclan en uno):
  * ``comparison_table.csv``        una fila por corrida (retención, ARI, Jaccard,
                                    cobertura, Silhouette, C_ext, precisión/recall)
  * ``comparison_table_pairs.csv``  una fila por par de corridas (FMI, NMI, ARI,
                                    n común, Jaccard medio emparejado, Silhouette)

No tiene lógica de métricas propia: reutiliza ``service.evaluate_run``,
``service.compare_runs`` y ``service.ensure_silhouette``. Con ``--compute-silhouette``
recalcula (y guarda en la corrida) el Silhouette que falte; sin él solo lee lo ya guardado.
"""

from __future__ import annotations

import argparse
import itertools
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from reii.quality import calibration as cal
from reii.quality import service
from reii.quality.store import QualityStore


def _num(x: Any) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def build_tables(
    store: QualityStore,
    run_ids: List[str],
    tier_id: str = "estricto",
    profile: Optional[Dict[str, Any]] = None,
    space: str = "chi2",
    compute_silhouette: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(tabla por corrida, tabla por par de corridas)."""
    profile = profile or cal.default_profile()
    rows = []
    for rid in run_ids:
        if compute_silhouette:
            service.ensure_silhouette(store, rid)
        res = service.evaluate_run(store, rid, profile, persist=False)
        run, t, sm = res["run"], res["tiers"][tier_id], res["summary"]
        sil = (run["summary"].get("extras") or {}).get("silhouette") or {}
        if compute_silhouette and sil.get("status") != "ok":
            sil = {}
        cx = t.get("c_ext") or {}
        rows.append(
            {
                "run_id": rid,
                "label": run.get("label"),
                "mode": run.get("mode"),
                "kind": run.get("kind"),
                "tier": tier_id,
                "n_total": t["n_total"],
                "n_assigned": t["n_assigned"],
                "retention": t["retention"],
                "n_clusters": t["n_clusters"],
                "ari": _num(sm.get("ari")),
                "jaccard_mean": _num(sm.get("jaccard_mean")),
                "cov_types_active": _num(sm.get("cov_types_active")),
                "silhouette_chi2": sil.get("chi2"),
                "silhouette_binary_cosine": sil.get("binary_cosine"),
                "silhouette_ppmi_cosine": sil.get("ppmi_cosine"),
                "precision": _num(t.get("precision")),
                "recall": _num(t.get("recall")),
                "c_ext_n": t.get("c_ext_n"),
                "c_ext_accuracy": cx.get("accuracy"),
                "c_ext_kappa": cx.get("kappa"),
                "c_ext_accuracy_aligned": cx.get("accuracy_aligned"),
                "c_ext_nmi": cx.get("nmi"),
            }
        )
    per_run = pd.DataFrame(rows)
    labels = dict(zip(per_run["run_id"], per_run["label"]))
    pairs = []
    for a, b in itertools.combinations(run_ids, 2):
        c = service.compare_runs(store, a, b, tier_id, profile=profile, space=space)
        jac = list(c["per_class_jaccard"].values())
        pairs.append(
            {
                "run_a": a, "run_b": b, "label_a": labels[a], "label_b": labels[b],
                "tier": tier_id, "n_a": c["n_a"], "n_b": c["n_b"], "n_common": c["n_common"],
                "same_corpus": c["same_corpus"],
                "fmi": _num(c["fmi"]), "nmi": _num(c["nmi"]), "ari": _num(c["ari"]),
                "jaccard_matched_mean": float(np.mean(jac)) if jac else None,
                f"silhouette_{space}_a": c["silhouette_a"], f"silhouette_{space}_b": c["silhouette_b"],
            }
        )
    return per_run, pd.DataFrame(pairs)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="reii.quality.compare_modes", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="run_id separados por coma (mínimo 1)")
    ap.add_argument("--tier", default="estricto")
    ap.add_argument("--out", default="comparison_table.csv")
    ap.add_argument("--dsn", default=None, help="PostgreSQL o sqlite:///archivo.db")
    ap.add_argument("--space", default="chi2", choices=list(service.SIL_SPACES),
                    help="espacio del Silhouette en la tabla de pares")
    ap.add_argument("--compute-silhouette", action="store_true",
                    help="recalcula y guarda el Silhouette de las corridas que no lo tengan")
    a = ap.parse_args(argv)

    run_ids = [r.strip() for r in a.runs.split(",") if r.strip()]
    store = QualityStore.open(a.dsn)
    try:
        runs_df = store.list_runs()
        known = set(runs_df["run_id"]) if not runs_df.empty else set()
        missing = [r for r in run_ids if r not in known]
        if missing:
            raise SystemExit(f"corridas inexistentes: {', '.join(missing)}")
        per_run, pairs = build_tables(store, run_ids, a.tier, space=a.space,
                                      compute_silhouette=a.compute_silhouette)
    finally:
        store.close()
    per_run.to_csv(a.out, index=False)
    stem, ext = os.path.splitext(a.out)
    pairs_path = f"{stem}_pairs{ext or '.csv'}"
    pairs.to_csv(pairs_path, index=False)
    print(f"{a.out}: {len(per_run)} corridas · {pairs_path}: {len(pairs)} pares")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
