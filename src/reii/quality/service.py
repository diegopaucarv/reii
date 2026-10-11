"""Evaluación de una corrida: niveles, indicadores y persistencia de resultados.

Un mismo (run_id, tier_id) puede tener varios resultados si se evalúa con perfiles
de umbral distintos: la clave es (run_id, tier_id, profile_hash).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from reii.quality import calibration as cal
from reii.quality import consensus as cons
from reii.quality import mapping as mp
from reii.quality import metrics as M
from reii.quality.store import QualityStore

logger = logging.getLogger(__name__)

MIN_CODED = 10  # mínimo de UCEs codificadas para reportar C_ext / kappa
SIL_SAMPLE = 3000  # tope de filas para el Silhouette (O(n²))
SIL_SPACES = ("chi2", "binary_cosine", "ppmi_cosine")


def _clean(d: Dict[str, Any]) -> Dict[str, Any]:
    """NaN/inf -> None (JSON válido al persistir en metrics_json)."""
    return {
        k: (None if isinstance(v, float) and not np.isfinite(v) else v)
        for k, v in d.items()
    }


def _int_or_none(x: Any) -> Optional[int]:
    try:
        if x is None or pd.isna(x):
            return None
        return int(float(x))
    except (TypeError, ValueError):
        return None


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
        out.update(_c_ext(gold, df, tier_id))
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
        "silhouette_chi2": (extras.get("silhouette") or {}).get("chi2"),
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


# ═══════════════════════════════════════════════════════════════════════════
# Estándar humano: clase humana efectiva, consenso, C_ext y acuerdo entre codificadores
# ═══════════════════════════════════════════════════════════════════════════


def resolve_human_classes(gold: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Clase humana efectiva por (uce_id, codificador).

    * ``human_cluster`` explícita  -> esa clase                       (source="explicit")
    * sin clase y ``is_correct``   -> la clase inductiva del pipeline (source="inferred_correct")
    * sin clase e incorrecta       -> desconocida (NaN)               (source="none")

    Inferir la clase cuando el codificador marca "correcta" es imprescindible: si
    C_ext solo usara las filas con clase explícita, solo vería los desacuerdos
    (el codificador solo escribe clase cuando el pipeline se equivocó) y la
    exactitud saldría sesgada hacia 0. La inferencia solo aplica a UCEs con
    clase inductiva (``cluster_core``), que es la que el editor muestra.
    """
    cols = ["uce_id", "coder", "is_correct", "human_cluster", "cluster_core",
            "human_class", "class_source"]
    if gold is None or gold.empty:
        return pd.DataFrame(columns=cols)
    g = gold.merge(df[["uce_id", "cluster_core"]], on="uce_id", how="left")
    pipe = pd.to_numeric(g["cluster_core"], errors="coerce").astype(float)
    hc = pd.to_numeric(g["human_cluster"], errors="coerce").astype(float)
    ok = g["is_correct"].astype(bool).to_numpy() & pipe.notna().to_numpy() & (pipe >= 0).to_numpy()
    inferred = pd.Series(np.where(ok, pipe, np.nan), index=g.index)
    g["human_class"] = hc.where(hc.notna(), inferred)
    g["class_source"] = np.where(hc.notna(), "explicit", np.where(inferred.notna(), "inferred_correct", "none"))
    return g[cols]


def consensus_human_class(resolved: pd.DataFrame) -> pd.DataFrame:
    """Una clase humana por UCE: moda entre codificadores; el empate se descarta
    (a diferencia de ``gold_consensus``, aquí no hay un valor por defecto sensato)."""
    out_cols = ["uce_id", "human_class", "n_coders", "n_explicit"]
    r = resolved.dropna(subset=["human_class"]) if len(resolved) else resolved
    rows = []
    for uid, grp in r.groupby("uce_id"):
        vc = grp["human_class"].astype(int).value_counts()
        if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]:
            continue
        rows.append(
            {
                "uce_id": uid,
                "human_class": int(vc.index[0]),
                "n_coders": int(len(grp)),
                "n_explicit": int((grp["class_source"] == "explicit").sum()),
            }
        )
    return pd.DataFrame(rows, columns=out_cols)


def _c_ext(gold: pd.DataFrame, df: pd.DataFrame, tier_id: str) -> Dict[str, Any]:
    """C_ext del nivel: clases mostradas por el pipeline vs clase humana de consenso,
    sobre las UCEs que el nivel incluye. ``c_ext`` es None si hay < MIN_CODED."""
    shown = f"shown_cluster_{tier_id}"
    hc = consensus_human_class(resolve_human_classes(gold, df))
    m = hc.merge(df[["uce_id", shown]], on="uce_id").dropna(subset=[shown])
    out: Dict[str, Any] = {"c_ext": None, "c_ext_n": int(len(m))}
    if len(m) >= MIN_CODED:
        out["c_ext"] = _clean(
            M.external_coherence(m[shown].astype(int), m["human_class"].astype(int))
        )
        out["c_ext_n_explicit"] = int((m["n_explicit"] > 0).sum())
    return out


def gold_rows_from_editor(edited: pd.DataFrame, coder: str) -> List[Dict[str, Any]]:
    """Filas para ``store.save_gold`` a partir del editor (uce_id, clase_pipeline,
    correcta, clase_humana). Si hay clase humana, ``is_correct`` se DERIVA de ella
    (coincide con la clase del pipeline) para que ambas columnas no se contradigan."""
    rows = []
    for r in edited.to_dict(orient="records"):
        pipe = _int_or_none(r.get("clase_pipeline"))
        hc = _int_or_none(r.get("clase_humana"))
        correct = (pipe is not None and pipe == hc) if hc is not None else bool(r.get("correcta"))
        rows.append(
            {"uce_id": str(r["uce_id"]), "coder": coder, "is_correct": bool(correct), "human_cluster": hc}
        )
    return rows


def coder_agreement(
    gold: pd.DataFrame, df: pd.DataFrame, coder_a: str, coder_b: str, min_n: int = MIN_CODED
) -> Dict[str, Any]:
    """Kappa y % de acuerdo entre dos codificadores: correcta/incorrecta y clase."""
    res = resolve_human_classes(gold, df)
    a, b = res[res["coder"] == coder_a], res[res["coder"] == coder_b]
    m = a.merge(b, on="uce_id", suffixes=("_a", "_b"))
    out: Dict[str, Any] = {"n_common": int(len(m)), "n_class": 0, "binary": None, "class": None, "min_n": min_n}
    if len(m) >= min_n:
        out["binary"] = _clean(
            {
                "n": int(len(m)),
                "kappa": M.cohen_kappa(m["is_correct_a"], m["is_correct_b"]),
                "pct_agreement": M.percent_agreement(m["is_correct_a"], m["is_correct_b"]),
            }
        )
    mc = m.dropna(subset=["human_class_a", "human_class_b"])
    out["n_class"] = int(len(mc))
    if len(mc) >= min_n:
        out["class"] = _clean(
            {
                "n": int(len(mc)),
                "kappa": M.cohen_kappa(mc["human_class_a"], mc["human_class_b"]),
                "pct_agreement": M.percent_agreement(mc["human_class_a"], mc["human_class_b"]),
            }
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Corpus de la corrida (tabla ``uces``), Silhouette y comparación entre corridas
# ═══════════════════════════════════════════════════════════════════════════


def corpus_fingerprint(pairs: Iterable[Tuple[Any, Any]]) -> str:
    """Huella del corpus a partir de (uce_id, texto). La usa también ``hooks`` al
    guardar la corrida, para que ambas se calculen igual por construcción."""
    items = sorted(
        f"{uid}|{hashlib.sha1((texto or '').encode()).hexdigest()[:8]}"
        for uid, texto in pairs
    )
    return hashlib.sha1("\n".join(items).encode()).hexdigest()[:16]


def load_corpus_for_run(
    store: QualityStore, run_id: str, allow_unverified: bool = False
) -> Dict[str, Any]:
    """Lee ``uces`` (uce_id, texto, linguistic_json) y verifica que sea el corpus de ESTA corrida.

    ``Database._save()`` reconstruye ``uces`` en cada ejecución del pipeline, así que
    la tabla describe la ÚLTIMA corrida, no necesariamente la pedida. Se compara la
    huella del corpus guardada en la corrida; las corridas migradas (sin huella de
    textos) solo se verifican por ids. Devuelve {status, reason, check, df}.
    """
    run = store.get_run(run_id)
    if run is None:
        return {"status": "unavailable", "reason": f"corrida inexistente: {run_id}", "check": None, "df": None}
    try:
        rows = store.conn.execute("SELECT uce_id, texto, linguistic_json FROM uces").fetchall()
    except Exception as e:  # sin tabla uces / DSN distinto al del pipeline
        return {"status": "unavailable", "reason": f"tabla uces no disponible ({type(e).__name__})", "check": None, "df": None}
    if not rows:
        return {"status": "unavailable", "reason": "la tabla uces está vacía", "check": None, "df": None}
    ids = [str(r["uce_id"]) for r in rows]
    own = set(store.load_assignments(run_id)["uce_id"].astype(str))
    if not own or not own.issubset(set(ids)):
        return {"status": "unavailable", "reason": "la tabla uces no contiene las UCEs de esta corrida (¿se ejecutó otro corpus después?)", "check": None, "df": None}
    check = "ids_only"
    if run.get("kind") == "pipeline" and run.get("corpus_hash"):
        got = corpus_fingerprint((str(r["uce_id"]), r["texto"]) for r in rows)
        if got != run["corpus_hash"]:
            if not allow_unverified:
                return {"status": "unavailable", "reason": "la tabla uces corresponde a otra ejecución (huella del corpus distinta)", "check": None, "df": None}
        else:
            check = "hash"
    ling = []
    for r in rows:
        v = r["linguistic_json"]
        ling.append(v if isinstance(v, dict) else json.loads(v or "{}"))
    df = pd.DataFrame({"uce_id": ids, "texto": [r["texto"] or "" for r in rows], "ling": ling})
    return {"status": "ok", "reason": "", "check": check, "df": df}


def silhouette_report(
    term_lists: List[List[str]],
    labels: Iterable[int],
    vocab: Iterable[str],
    *,
    vocab_source: str,
    ppmi_builder: Any = None,
    sample_size: int = SIL_SAMPLE,
    seed: int = 42,
    origin: str = "pipeline",
) -> Dict[str, Any]:
    """Silhouette de las clases a nivel UCE en tres espacios sobre la matriz UCE x término:

    chi2          : perfiles χ² (geometría del AFC/ALCESTE) de la matriz binaria
    binary_cosine : coseno sobre la matriz binaria
    ppmi_cosine   : coseno sobre la matriz PPMI (solo si el pipeline corrió con PPMI)

    Es una aproximación: el CDH clasifica unidades de contexto (UC) y cada UCE hereda
    la etiqueta de su UC, de modo que aquí se mide en el espacio de UCEs.
    """
    y = np.asarray(list(labels), dtype=int)
    vocab = list(vocab)
    n_lab, k = int((y >= 0).sum()), len(np.unique(y[y >= 0]))
    base = {"status": "unavailable", "origin": origin, "vocab_source": vocab_source,
            "vocab_size": len(vocab), "n_labelled": n_lab, "n_classes": k, "sample_size": sample_size, "seed": seed}
    if n_lab < MIN_CODED or k < 2 or not vocab:
        return {**base, "reason": "se necesitan >=10 UCEs con clase, >=2 clases y vocabulario"}
    X = M.binary_term_matrix(term_lists, vocab)
    vals: Dict[str, Optional[float]] = {
        "chi2": M.silhouette_labels(X, y, "chi2", sample_size, seed),
        "binary_cosine": M.silhouette_labels(X, y, "cosine", sample_size, seed),
        "ppmi_cosine": None,
    }
    if ppmi_builder is not None:
        try:
            vals["ppmi_cosine"] = M.silhouette_labels(ppmi_builder.build(X, vocab), y, "cosine", sample_size, seed)
        except Exception as e:  # PPMI es opcional
            logger.debug("silhouette PPMI falló: %s", e)
    if all(v is None for v in vals.values()):
        return {**base, "reason": "no calculable"}
    return {**base, **vals, "status": "ok", "n_used": min(n_lab, sample_size)}


def ensure_silhouette(
    store: QualityStore,
    run_id: str,
    persist: bool = True,
    sample_size: int = SIL_SAMPLE,
    allow_unverified: bool = False,
) -> Dict[str, Any]:
    """Silhouette de la corrida: lo devuelve si ya está en ``summary.extras`` (corridas
    nuevas, calculado por el hook) y, si no, lo recalcula desde ``uces`` y lo persiste.

    El recálculo usa la configuración GUARDADA en la corrida (stems, n-gramas, tsj) y
    deriva el vocabulario con el mismo criterio del pipeline (frecuencia documental
    >= tsj y absoluta >= min_term_abs_freq); no hay PPMI porque necesita SUBTLEX.
    """
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(f"corrida inexistente: {run_id}")
    summary = dict(run.get("summary") or {})
    extras = dict(summary.get("extras") or {})
    cached = extras.get("silhouette")
    if isinstance(cached, dict) and cached.get("status") == "ok":
        return cached
    corpus = load_corpus_for_run(store, run_id, allow_unverified)
    if corpus["status"] != "ok":
        return {"status": "unavailable", "reason": corpus["reason"]}
    a = store.load_assignments(run_id)
    lab = {str(u): int(c) for u, c in zip(a["uce_id"], a["cluster_core"]) if pd.notna(c) and c >= 0}
    cdf = corpus["df"]
    cdf = cdf[cdf["uce_id"].isin(lab)].sort_values("uce_id")
    cfg = run.get("config") or {}
    use_stems = cfg.get("stem_backend", "snowball") != "none"
    bi, tri = bool(cfg.get("use_bigrams", True)), bool(cfg.get("use_trigrams", True))
    terms = [list(M.iter_terms(l, use_stems, bi, tri)) for l in cdf["ling"]]
    vocab = M.prune_vocabulary(terms, int(cfg.get("tsj", 3)), int(cfg.get("min_term_abs_freq", 2)))
    rep = silhouette_report(
        terms, [lab[u] for u in cdf["uce_id"]], vocab,
        vocab_source="derived_tsj", sample_size=sample_size, origin="lazy",
    )
    rep["corpus_check"] = corpus["check"]
    if persist and rep["status"] == "ok":
        extras["silhouette"] = rep
        summary["extras"] = extras
        store.update_summary(run_id, summary)  # update_summary reemplaza todo: se fusionó arriba
    return rep


def compare_runs(
    store: QualityStore,
    run_id_a: str,
    run_id_b: str,
    tier_id: str = "estricto",
    profile: Optional[Dict[str, Any]] = None,
    space: str = "chi2",
    compute_silhouette: bool = False,
) -> Dict[str, Any]:
    """FMI / NMI / ARI entre dos corridas en las UCEs que ambas incluyen en ``tier_id``.

    Los niveles se asignan con ``profile`` y la calibración propia de cada corrida.
    Devuelve n_common, fmi, nmi, ari, same_corpus, confusion (K_a x K_b),
    class_match / per_class_jaccard (por clase de A, emparejamiento Hungarian) y
    silhouette_a / silhouette_b en el espacio ``space`` (None si no está calculado;
    con ``compute_silhouette=True`` se recalcula lo que falte).
    """
    profile = profile or cal.default_profile()
    if tier_id not in {t["tier_id"] for t in profile["tiers"]}:
        raise ValueError(f"nivel inexistente en el perfil: {tier_id}")
    runs, labels = {}, {}
    for rid in (run_id_a, run_id_b):
        run = store.get_run(rid)
        if run is None:
            raise KeyError(f"corrida inexistente: {rid}")
        d = cal.assign_tiers(store.load_assignments(rid), profile, store.load_calibration(rid))
        d = d[d[f"in_{tier_id}"].astype(bool) & d[f"shown_cluster_{tier_id}"].notna()]
        runs[rid] = run
        labels[rid] = {str(u): int(c) for u, c in zip(d["uce_id"], d[f"shown_cluster_{tier_id}"])}
    la, lb = labels[run_id_a], labels[run_id_b]
    out: Dict[str, Any] = {
        "tier_id": tier_id,
        "n_a": len(la),
        "n_b": len(lb),
        **M.cross_partition_agreement(la, lb),
        "same_corpus": bool(runs[run_id_a].get("corpus_hash") and runs[run_id_a].get("corpus_hash") == runs[run_id_b].get("corpus_hash")),
    }
    common = sorted(set(la) & set(lb))
    out["confusion"] = (
        pd.crosstab(pd.Series([la[u] for u in common], name=f"A · {run_id_a[:12]}"),
                    pd.Series([lb[u] for u in common], name=f"B · {run_id_b[:12]}"))
        if common else pd.DataFrame()
    )
    ov, ra, rb = mp.overlap_matrix(la, lb)
    jac = M.jaccard_from_overlap(ov)
    hung = mp.hungarian_mapping(ov)
    out["per_class_jaccard"] = {int(ra[i]): j for i, j in jac.items()}
    out["class_match"] = {int(ra[i]): int(rb[next(iter(js))]) for i, js in hung.items()}
    for key, rid in (("silhouette_a", run_id_a), ("silhouette_b", run_id_b)):
        sil = (runs[rid].get("summary") or {}).get("extras", {}).get("silhouette")
        if compute_silhouette and not (isinstance(sil, dict) and sil.get("status") == "ok"):
            sil = ensure_silhouette(store, rid)
        out[key] = sil.get(space) if isinstance(sil, dict) and sil.get("status") == "ok" else None
    out["silhouette_space"] = space
    return out
