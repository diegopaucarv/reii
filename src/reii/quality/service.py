"""Evaluación de una corrida: niveles, indicadores y persistencia de resultados.

Un mismo (run_id, tier_id) puede tener varios resultados si se evalúa con perfiles
de umbral distintos: la clave es (run_id, tier_id, profile_hash).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from reii.quality import calibration as cal
from reii.quality import consensus as cons
from reii.quality import metrics as M
from reii.quality.store import QualityStore


def expand_comps(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte el dict ``comps`` en columnas booleanas (wc, sim, emb, cross_*)."""
    out = df.copy()
    comps = (
        out["comps"] if "comps" in out else pd.Series([{}] * len(out), index=out.index)
    )
    for name in cons.COMPARISONS:
        out[name] = [bool((c or {}).get(name, False)) for c in comps]
    return out


def gold_consensus(gold: pd.DataFrame) -> pd.DataFrame:
    """Un juicio por UCE: mayoría entre codificadores (empate = incorrecto)."""
    if gold is None or gold.empty:
        return pd.DataFrame(columns=["uce_id", "is_correct"])
    g = gold.groupby("uce_id")["is_correct"].mean().reset_index()
    g["is_correct"] = (g["is_correct"] > 0.5).astype(int)
    return g


def tier_metrics(
    df: pd.DataFrame,
    tier_id: str,
    gold: Optional[pd.DataFrame] = None,
    n_total: Optional[int] = None,
) -> Dict[str, Any]:
    flag = df[f"in_{tier_id}"].astype(bool)
    shown = df[f"shown_cluster_{tier_id}"]
    n_total = int(n_total if n_total is not None else len(df))
    inductive = df["cluster_core"].notna() & (df["cluster_core"] >= 0)
    n_in = int(flag.sum())
    out: Dict[str, Any] = {
        "tier_id": tier_id,
        "n_total": n_total,
        "n_assigned": n_in,
        "retention": n_in / n_total if n_total else 0.0,
        "n_projected": int((flag & ~inductive).sum()),
        "n_clusters": int(shown.dropna().nunique()),
        "class_sizes": {
            str(int(k)): int(v)
            for k, v in shown.dropna().astype(int).value_counts().sort_index().items()
        },
        "p_cal_mean": float(df.loc[flag, "p_cal"].mean())
        if df["p_cal"].notna().any() and n_in
        else None,
    }
    out.update(M.selection_bias(df, flag))
    if gold is not None and not gold.empty:
        g = gold_consensus(gold).merge(
            df[["uce_id", f"in_{tier_id}", "s_acuerdo", "cluster_core"]], on="uce_id"
        )
        g = g[g["cluster_core"].notna() & (g["cluster_core"] >= 0)]
        if len(g):
            out.update(
                M.tier_validity(g[f"in_{tier_id}"].astype(bool), g["is_correct"])
            )
    return out


def evaluate_run(
    store: QualityStore,
    run_id: str,
    profile: Optional[Dict[str, Any]] = None,
    persist: bool = True,
    calibration: Optional[Dict[str, Any]] = None,
    min_share: Optional[float] = None,
) -> Dict[str, Any]:
    """Calcula indicadores por nivel para una corrida y (opcional) los persiste.

    Devuelve {"df": df_con_niveles, "tiers": {tier_id: metrics}, "run": {...},
              "funnel": [...], "profile": profile, "profile_hash": ..., "curve": DataFrame|None}
    """
    profile = profile or cal.default_profile()
    ph = cal.profile_hash(profile)
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(f"corrida inexistente: {run_id}")
    df = expand_comps(store.load_assignments(run_id))
    calibration = calibration or store.load_calibration(run_id)
    df = cal.assign_tiers(df, profile, calibration)
    gold = store.load_gold(run_id)

    n_total = run.get("n_uces_total") or run.get("n_uces") or len(df)
    tiers = {
        t["tier_id"]: tier_metrics(df, t["tier_id"], gold, n_total)
        for t in profile["tiers"]
    }
    tier_ids = [t["tier_id"] for t in profile["tiers"]]
    nested = cal.check_nesting(df, tier_ids)

    extras = run["summary"].get("extras", {})
    funnel = M.retention_funnel(df)
    curve = None
    aurc_v = None
    if not gold.empty:
        g = gold_consensus(gold).merge(
            df[["uce_id", "s_acuerdo", "s_margen", "cluster_core"]], on="uce_id"
        )
        g = g[g["cluster_core"].notna() & (g["cluster_core"] >= 0)]
        if len(g) >= 5:
            g["rank"] = cons.rank_score(g)
            curve = M.risk_coverage(g["rank"], g["is_correct"])
            aurc_v = M.aurc(curve)

    jac = extras.get("jaccard_by_class") or {}
    jac_mean = float(np.mean(list(jac.values()))) if jac else None
    first, last = tier_ids[0], tier_ids[-1]
    summary = {
        "extras": extras,
        "retention_estricto": tiers[first]["retention"],
        "retention_amplio": tiers[last]["retention"],
        "n_clusters": tiers[first]["n_clusters"],
        "jaccard_mean": jac_mean,
        "ari": extras.get("ari"),
        "precision_estricto": tiers[first].get("precision"),
        "aurc": aurc_v,
        "cov_tokens": (extras.get("coverage") or {}).get("cov_tokens"),
        "cov_types": (extras.get("coverage") or {}).get("cov_types"),
        "cov_types_total": (extras.get("coverage") or {}).get("cov_types_total"),
        "cov_types_active": (extras.get("coverage") or {}).get("cov_types_active"),
        "nested_ok": nested,
        "last_profile_hash": ph,
    }
    if persist:
        store.save_profile(
            ph, profile["name"], profile, prereg=not cal.is_exploratory(profile)
        )
        for tid, m in tiers.items():
            store.save_tier_result(run_id, tid, ph, m)
        store.update_summary(run_id, summary)
    return {
        "df": df,
        "tiers": tiers,
        "run": run,
        "funnel": funnel,
        "profile": profile,
        "profile_hash": ph,
        "curve": curve,
        "summary": summary,
        "nested_ok": nested,
        "calibration": calibration,
    }


def recalibrate(
    store: QualityStore, run_id: str, min_n: int = 60
) -> Optional[Dict[str, Any]]:
    """Ajusta la calibración isotónica con el estándar humano y la guarda (versionada)."""
    df = store.load_assignments(run_id)
    g = gold_consensus(store.load_gold(run_id))
    if df.empty or g.empty:
        return None
    m = g.merge(df[["uce_id", "s_acuerdo", "cluster_core"]], on="uce_id")
    m = m[m["cluster_core"].notna() & (m["cluster_core"] >= 0)]
    fit = cal.fit_isotonic(m["s_acuerdo"], m["is_correct"], min_n=min_n)
    if fit is None:
        return None
    version = store.save_calibration(run_id, fit)
    fit["version"] = version
    store.log(run_id, "recalibrate", {"n_gold": fit["n"], "version": version})
    p = cal.apply_calibration(fit, df["s_acuerdo"].fillna(0.0))
    store.update_p_cal(run_id, dict(zip(df["uce_id"], p)))
    return fit


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Muestra estratificada por (nivel mínimo, clase) para codificación humana."""
    d = df.copy()
    d["_stratum"] = (
        d["tier_min"].fillna("fuera").astype(str)
        + "|"
        + d["cluster_core"].astype("string").fillna("NA")
    )
    rng = np.random.default_rng(seed)
    per = max(1, n // max(1, d["_stratum"].nunique()))
    parts = []
    for _, g in d.groupby("_stratum"):
        idx = rng.permutation(len(g))[:per]
        parts.append(g.iloc[idx])
    s = pd.concat(parts)
    if len(s) < n:
        rest = d.drop(s.index)
        s = pd.concat([s, rest.sample(min(n - len(s), len(rest)), random_state=seed)])
    return s.head(n).drop(columns="_stratum")
