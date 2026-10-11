"""Migra los resultados ya almacenados (corrida actual del sistema completo) a una corrida 'legacy'.

Fuentes:
  * PostgreSQL: tablas ``uces`` / ``terms`` / ``kv_store`` (lo que lee el dashboard).
  * JSON: el archivo ``db_local_path`` del workflow (``data["uces"]``, ``vocabulario``...).

Limitación conocida: las banderas por comparación (stable_wc, stable_sim,
stable_emb, stable_cross_*) son atributos dinámicos que ``UCE.to_dict()`` y
``migrate_uces`` no persisten. Para una corrida previa solo se conserva
``is_stable`` (+ ``stability_method``), de modo que ``s_acuerdo`` vale 1.0 para las
estables y 0.0 para el resto y se marca ``s_source = legacy_is_stable``. Los puntajes
completos solo existen en corridas nuevas guardadas con el hook.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from reii.quality import calibration as cal
from reii.quality import metrics as M
from reii.quality import service
from reii.quality.store import QualityStore

PG_UCE_SQL = (
    "SELECT uce_id, doc_id, local_idx, texto, n_tokens, cluster_id, is_stable, stability_method, "
    "projected_cluster_id, projection_distance, projection_margin, projection_ratio, linguistic_json "
    "FROM uces ORDER BY doc_id, local_idx"
)
KV_KEYS = ("config", "pairwise_stability", "vocabulario")


def load_legacy_postgres(dsn: Optional[str] = None) -> Dict[str, Any]:
    from reii.backend.sql.db import get_conn  # psycopg

    with get_conn(dsn) as conn:
        rows = conn.execute(PG_UCE_SQL).fetchall()
        kv = {}
        for r in conn.execute(
            "SELECT key, value_json FROM kv_store WHERE key = ANY(%s)", (list(KV_KEYS),)
        ).fetchall():
            try:
                kv[r["key"]] = json.loads(r["value_json"])
            except (TypeError, json.JSONDecodeError):
                kv[r["key"]] = None
        terms = [
            r["term"]
            for r in conn.execute("SELECT DISTINCT term FROM terms").fetchall()
        ]
    uces = []
    for r in rows:
        d = dict(r)
        ling = json.loads(d.pop("linguistic_json") or "{}")
        d.update(
            {
                k: ling.get(k)
                for k in (
                    "lemmas",
                    "stems",
                    "bigram_stems",
                    "trigram_stems",
                    "bigrams",
                    "trigrams",
                )
            }
        )
        d["is_stable"] = bool(d.get("is_stable"))
        uces.append(d)
    vocab = kv.get("vocabulario") or sorted(terms)
    return {
        "uces": uces,
        "vocabulario": vocab,
        "pairwise_stability": kv.get("pairwise_stability") or {},
        "config": kv.get("config") or {},
        "origin": "postgres",
    }


def load_legacy_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    uces = []
    for u in data.get("uces", []) or []:
        d = dict(u)
        d["uce_id"] = str(d.get("uce_id") or d.get("id") or "")
        uces.append(d)
    return {
        "uces": uces,
        "vocabulario": data.get("vocabulario") or [],
        "pairwise_stability": data.get("pairwise_stability") or {},
        "config": data.get("config") or {},
        "origin": f"json:{path}",
    }


def build_legacy_run(
    legacy: Dict[str, Any], label: Optional[str] = None
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for u in legacy["uces"]:
        uid = str(u.get("uce_id") or u.get("id") or "")
        if not uid:
            continue
        stable = bool(u.get("is_stable"))
        cid = u.get("cluster_id")
        rows.append(
            {
                "uce_id": uid,
                "doc_id": None if u.get("doc_id") is None else str(u.get("doc_id")),
                "cluster_core": cid
                if (stable and cid is not None and cid >= 0)
                else None,
                "s_acuerdo": 1.0 if stable else 0.0,
                "s_margen": u.get("projection_margin"),
                "n_comparisons": 1,
                "s_source": "legacy_is_stable",
                "projected_cluster_id": u.get("projected_cluster_id"),
                "n_tokens": u.get("n_tokens") or len(u.get("lemmas") or []) or None,
                "comps": {"stability_method": u.get("stability_method")},
            }
        )
    df = pd.DataFrame(rows)
    sig = "\n".join(
        sorted(
            f"{r['uce_id']}:{r['cluster_core']}:{r['s_acuerdo']}:{r['projected_cluster_id']}"
            for r in rows
        )
    )
    run_id = "legacy-" + hashlib.sha1(sig.encode()).hexdigest()[:12]
    corpus_hash = hashlib.sha1(
        "\n".join(sorted(r["uce_id"] for r in rows)).encode()
    ).hexdigest()[:16]

    cfg = legacy.get("config") or {}
    extras: Dict[str, Any] = {
        "vocab_scope": "uces_estables",
        "origen": legacy.get("origin"),
    }
    use_stems = cfg.get("stem_backend", "snowball") != "none"
    use_bigrams = cfg.get("use_bigrams", True)
    use_trigrams = cfg.get("use_trigrams", True)
    tsj = int(cfg.get("tsj", 3))
    vocab = legacy.get("vocabulario") or []
    all_uces = legacy["uces"]
    if any(u.get("stems") or u.get("lemmas") for u in all_uces):

        def _terms(u: Dict[str, Any]) -> List[str]:
            return list(M.iter_terms(u, use_stems, use_bigrams, use_trigrams))

        corpus_term_lists = [_terms(u) for u in all_uces]
        stable_term_lists = [_terms(u) for u in all_uces if u.get("is_stable")]
        extras["coverage"] = M.corpus_coverage(
            stable_term_lists, corpus_term_lists, tsj
        )
        by_class: Dict[int, List[List[str]]] = {}
        for u in all_uces:
            if not u.get("is_stable"):
                continue
            cid = u.get("cluster_id")
            if cid is None or cid < 0:
                continue
            by_class.setdefault(int(cid), []).append(_terms(u))
        if by_class:
            extras["coverage_by_class"] = M.coverage_by_class(
                by_class, corpus_term_lists, tsj
            )
        extras["null_coverage"] = M.null_coverage(
            stable_term_lists, corpus_term_lists, tsj, n_perm=200
        )
    elif vocab and any(u.get("stems") or u.get("lemmas") for u in all_uces):
        tl = [
            list(M.iter_terms(u, use_stems, use_bigrams, use_trigrams))
            for u in all_uces
        ]
        extras["coverage"] = M.vocabulary_coverage(tl, vocab)
    pw = legacy.get("pairwise_stability") or {}
    if pw.get("overlap_aa"):
        extras["jaccard_by_class"] = {
            str(k): v
            for k, v in M.jaccard_from_overlap(np.array(pw["overlap_aa"])).items()
        }
    for k in ("ari_wc", "ari_coref", "ari_emb"):
        if k in pw:
            extras[k] = pw[k]
    extras["ari"] = pw.get("ari_wc") or pw.get("ari_coref") or pw.get("ari_emb")
    run = {
        "run_id": run_id,
        "label": label or "Corrida previa migrada (sistema completo)",
        "kind": "legacy",
        "mode": cfg.get("classification_mode"),
        "seed": cfg.get("random_state"),
        "corpus_hash": corpus_hash,
        "code_version": "legacy",
        "config": {
            k: (
                "***"
                if any(s in k.lower() for s in ("api_key", "secret", "password", "dsn"))
                else v
            )
            for k, v in cfg.items()
        },
        "summary": {"extras": extras},
        "n_uces": len(df),
        "n_uces_total": len(legacy["uces"]),
    }
    return run, df


def migrate(
    store: QualityStore,
    legacy: Dict[str, Any],
    dry_run: bool = False,
    label: Optional[str] = None,
    scope: str = "default",
) -> Dict[str, Any]:
    run, df = build_legacy_run(legacy, label)
    report = {
        "run_id": run["run_id"],
        "n_uces": len(df),
        "n_stable": int((df["cluster_core"].notna()).sum()),
        "n_projected": int(df["projected_cluster_id"].notna().sum()),
        "retention": float(df["cluster_core"].notna().mean()) if len(df) else 0.0,
        "dry_run": dry_run,
    }
    if dry_run or df.empty:
        return report
    existed = store.get_run(run["run_id"]) is not None
    store.save_run(run)
    store.save_assignments(run["run_id"], df)
    service.evaluate_run(store, run["run_id"], cal.default_profile(), persist=True)
    store.log(
        run["run_id"], "migrate_legacy", {k: report[k] for k in ("n_uces", "n_stable")}
    )
    if store.get_active(scope) is None:
        store.set_active(
            scope, run["run_id"], "estricto", cal.profile_hash(cal.default_profile())
        )
    report["already_present"] = existed
    return report
