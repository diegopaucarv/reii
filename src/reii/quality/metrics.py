"""Indicadores de evaluación (funciones puras, sin acceso a base de datos)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau

# NumPy 2.0 renombró np.trapz a np.trapezoid; soportar ambos.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# ── Vocabulario ─────────────────────────────────────────────────────────────


def iter_terms(
    uce: Dict,
    use_stems: bool = True,
    use_bigrams: bool = True,
    use_trigrams: bool = True,
) -> Iterable[str]:
    """Réplica de ``MatrizBuilder._iter_terms`` para UCEs en forma de dict."""
    base = uce.get("stems") if use_stems else uce.get("lemmas")
    for s in base or []:
        yield s
    if use_bigrams:
        for b in (uce.get("bigram_stems") if use_stems else uce.get("bigrams")) or []:
            yield "_".join(b)
    if use_trigrams:
        for t in (uce.get("trigram_stems") if use_stems else uce.get("trigrams")) or []:
            yield "_".join(t)


def vocabulary_coverage(
    term_lists: Iterable[Sequence[str]], vocab: Iterable[str]
) -> Dict[str, float]:
    """Cobertura de vocabulario con dos definiciones, rotuladas:

    cov_types  : formas analizadas / formas distintas        (definición del dashboard)
    cov_tokens : ocurrencias de formas analizadas / ocurrencias totales
    Ambas sobre el MISMO universo de términos (el que se pasa en term_lists).

    NOTA: esta métrica mide qué fracción del vocabulario de ``term_lists`` está
    en ``vocab`` (el inventario analizado). Para la cobertura del vocabulario del
    CORPUS por las UCEs (la "riqueza de vocabulario" de ALCESTE), usar
    :func:`corpus_coverage`, que invierte la dirección y aplica el filtro TSJ.
    """
    vocab = set(vocab)
    freq: Dict[str, int] = {}
    for terms in term_lists:
        for t in terms:
            freq[t] = freq.get(t, 0) + 1
    n_types = len(freq)
    n_tokens = sum(freq.values())
    in_voc_types = sum(1 for t in freq if t in vocab)
    in_voc_tokens = sum(c for t, c in freq.items() if t in vocab)
    hapax = sum(1 for c in freq.values() if c == 1)
    return {
        "n_types": n_types,
        "n_tokens": n_tokens,
        "n_vocab": len(vocab),
        "n_types_in_vocab": in_voc_types,
        "cov_types": in_voc_types / n_types if n_types else 0.0,
        "cov_tokens": in_voc_tokens / n_tokens if n_tokens else 0.0,
        "hapax_pct": hapax / n_types if n_types else 0.0,
    }


def corpus_vocabulary_stats(
    corpus_term_lists: Iterable[Sequence[str]], tsj: int = 3
) -> Dict[str, object]:
    """Estadísticas del vocabulario del corpus (types como unidad).

    Replica la distinción de ALCESTE entre el inventario total (incluye hápax) y
    el inventario *activo* (types con frecuencia >= ``tsj``, que son los únicos
    que pueden establecer coocurrencias y entrar a la matriz de clasificación).

    Devuelve:
        freq            : {type: frecuencia en el corpus}
        n_types_total   : types distintos (incluye hápax)
        n_types_active  : types con frecuencia >= tsj
        n_hapax         : types con frecuencia == 1
        hapax_pct       : n_hapax / n_types_total
        ttr             : type-token ratio (n_types_total / n_tokens)
        active_types    : set de types activos
    """
    freq: Dict[str, int] = {}
    for terms in corpus_term_lists:
        for t in terms:
            freq[t] = freq.get(t, 0) + 1
    n_types_total = len(freq)
    n_tokens = sum(freq.values())
    n_hapax = sum(1 for c in freq.values() if c == 1)
    active_types = {t for t, c in freq.items() if c >= tsj}
    return {
        "freq": freq,
        "n_types_total": n_types_total,
        "n_types_active": len(active_types),
        "n_hapax": n_hapax,
        "hapax_pct": n_hapax / n_types_total if n_types_total else 0.0,
        "ttr": n_types_total / n_tokens if n_tokens else 0.0,
        "active_types": active_types,
    }


def corpus_coverage(
    uce_term_lists: Iterable[Sequence[str]],
    corpus_term_lists: Iterable[Sequence[str]],
    tsj: int = 3,
) -> Dict[str, float]:
    """Cobertura del vocabulario del CORPUS por las UCEs (types como unidad).

    Esta es la "riqueza de vocabulario" comparable a ALCESTE: qué fracción de los
    types del corpus aparece en las UCEs retenidas. Reporta dos definiciones:

    cov_types_total  : types en UCEs / types totales del corpus (incluye hápax)
    cov_types_active : types en UCEs / types activos del corpus (frecuencia >= tsj)

    La brecha entre ambas es la medida del sesgo de selección: si las UCEs
    retenidas concentran el vocabulario frecuente y descartan los hápax,
    ``cov_types_active`` será mucho mayor que ``cov_types_total``.
    """
    stats = corpus_vocabulary_stats(corpus_term_lists, tsj)
    freq = stats["freq"]
    active = stats["active_types"]
    uce_types = {t for terms in uce_term_lists for t in terms}
    n_total = int(stats["n_types_total"])
    n_active = int(stats["n_types_active"])
    in_total = sum(1 for t in uce_types if t in freq)
    in_active = sum(1 for t in uce_types if t in active)
    return {
        "n_types_total": n_total,
        "n_types_active": n_active,
        "n_hapax": int(stats["n_hapax"]),
        "hapax_pct": float(stats["hapax_pct"]),
        "ttr": float(stats["ttr"]),
        "n_uce_types": len(uce_types),
        "cov_types_total": in_total / n_total if n_total else 0.0,
        "cov_types_active": in_active / n_active if n_active else 0.0,
    }


def coverage_by_class(
    uce_term_lists_by_class: Dict[object, Iterable[Sequence[str]]],
    corpus_term_lists: Iterable[Sequence[str]],
    tsj: int = 3,
) -> Dict[object, Dict[str, float]]:
    """Cobertura de vocabulario por clase (types como unidad).

    ``uce_term_lists_by_class``: {clase: [lista de términos por UCE]}.
    Devuelve {clase: {n_types, cov_types_active}} para detectar clases
    léxicamente ricas vs. pobres.
    """
    stats = corpus_vocabulary_stats(corpus_term_lists, tsj)
    active = stats["active_types"]
    n_active = int(stats["n_types_active"])
    out: Dict[object, Dict[str, float]] = {}
    for cls, term_lists in uce_term_lists_by_class.items():
        uce_types = {t for terms in term_lists for t in terms}
        in_active = sum(1 for t in uce_types if t in active)
        out[cls] = {
            "n_types": len(uce_types),
            "cov_types_active": in_active / n_active if n_active else 0.0,
        }
    return out


def null_coverage(
    uce_term_lists: Iterable[Sequence[str]],
    corpus_term_lists: Iterable[Sequence[str]],
    tsj: int = 3,
    n_perm: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Cobertura esperada por azar (modelo nulo por permutación).

    Compara la cobertura observada de las UCEs contra la distribución de
    seleccionar el mismo número de types al azar del vocabulario activo del
    corpus. Un z-score alto indica que la cobertura observada supera lo que
    cabría esperar por azar; un p_value bajo refuerza esa conclusión.
    """
    stats = corpus_vocabulary_stats(corpus_term_lists, tsj)
    active = list(stats["active_types"])
    n_active = len(active)
    uce_types = {t for terms in uce_term_lists for t in terms}
    n_uce_types = len(uce_types)
    if n_active == 0:
        return {
            "observed": 0.0,
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "z_score": float("nan"),
            "p_value": float("nan"),
        }
    observed = sum(1 for t in uce_types if t in active) / n_active
    rng = np.random.default_rng(seed)
    k = min(n_uce_types, n_active)
    nulls = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        sample = rng.choice(active, size=k, replace=True)
        nulls[i] = len(set(sample)) / n_active
    std = float(nulls.std(ddof=1))
    z = (observed - float(nulls.mean())) / std if std > 0 else float("nan")
    return {
        "observed": observed,
        "null_mean": float(nulls.mean()),
        "null_std": std,
        "z_score": z,
        "p_value": float((nulls >= observed).mean()),
    }


# ── Retención y embudo ──────────────────────────────────────────────────────

FUNNEL_ORDER = [
    ("wc", "A · conteo de palabras"),
    ("sim", "B · coref/similitud"),
    ("emb", "C · embeddings"),
    ("cross_ab", "A↔B"),
    ("cross_bc", "B↔C"),
    ("cross_ac", "A↔C"),
]


def retention_funnel(df: pd.DataFrame, total: Optional[int] = None) -> List[Dict]:
    """Embudo acumulativo A -> A∩B -> ... usando las columnas booleanas de df.

    Solo incluye pasos cuyas columnas existen y tienen al menos un True.
    """
    total = int(total if total is not None else len(df))
    out = [{"step": "UCEs totales", "n": total, "pct": 1.0 if total else 0.0}]
    mask = pd.Series(True, index=df.index)
    for col, name in FUNNEL_ORDER:
        if col not in df.columns or not df[col].astype(bool).any():
            continue
        mask = mask & df[col].astype(bool)
        n = int(mask.sum())
        out.append({"step": f"∩ {name}", "n": n, "pct": n / total if total else 0.0})
    return out


# ── Estabilidad por clase ───────────────────────────────────────────────────


def jaccard_from_overlap(ov: np.ndarray) -> Dict[int, float]:
    """Jaccard de cada clase de A con su pareja Hungarian en B (índices de fila)."""
    ov = np.asarray(ov, float)
    if ov.size == 0:
        return {}
    r, c = linear_sum_assignment(-ov)
    rs, cs = ov.sum(axis=1), ov.sum(axis=0)
    out = {}
    for i, j in zip(r, c):
        union = rs[i] + cs[j] - ov[i, j]
        out[int(i)] = float(ov[i, j] / union) if union > 0 else 0.0
    return out


def jaccard_by_class(
    a: Dict[str, int], b: Dict[str, int], mapping: Dict[int, List[int]]
) -> Dict[int, float]:
    """Jaccard por clase de A frente a la unión de sus clases emparejadas en B."""
    common = set(a) & set(b)
    out = {}
    for k, targets in mapping.items():
        in_a = {u for u in common if a[u] == k}
        in_b = {u for u in common if b[u] in targets}
        union = len(in_a | in_b)
        out[int(k)] = len(in_a & in_b) / union if union else 0.0
    return out


# ── Sesgo de selección ──────────────────────────────────────────────────────


def smd(x: pd.Series, retained: pd.Series) -> float:
    """Diferencia estandarizada de medias (retenidas vs descartadas)."""
    x = pd.to_numeric(x, errors="coerce")
    r, d = x[retained], x[~retained]
    r, d = r.dropna(), d.dropna()
    if len(r) < 2 or len(d) < 2:
        return float("nan")
    pooled = np.sqrt((r.var(ddof=1) + d.var(ddof=1)) / 2)
    return float((r.mean() - d.mean()) / pooled) if pooled > 0 else 0.0


def selection_bias(df: pd.DataFrame, retained: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "n_tokens" in df:
        out["smd_n_tokens"] = smd(df["n_tokens"], retained)
    if "doc_id" in df and len(df):
        rate = retained.groupby(df["doc_id"]).mean()
        out["doc_retention_min"] = float(rate.min())
        out["doc_retention_max"] = float(rate.max())
        out["doc_retention_iqr"] = float(rate.quantile(0.75) - rate.quantile(0.25))
    return out


# ── Riesgo-cobertura y validez contra estándar humano ───────────────────────


def risk_coverage(scores: Sequence[float], correct: Sequence[int]) -> pd.DataFrame:
    """Curva riesgo-cobertura: ordena por puntaje descendente y acumula la precisión."""
    s = np.asarray(scores, float)
    y = np.asarray(correct, float)
    order = np.argsort(-s, kind="stable")
    y = y[order]
    n = len(y)
    if n == 0:
        return pd.DataFrame(columns=["coverage", "precision", "risk"])
    cum = np.cumsum(y)
    k = np.arange(1, n + 1)
    prec = cum / k
    return pd.DataFrame({"coverage": k / n, "precision": prec, "risk": 1 - prec})


def aurc(curve: pd.DataFrame) -> float:
    """Área bajo la curva riesgo-cobertura (menor es mejor)."""
    if curve.empty:
        return float("nan")
    return float(_trapz(curve["risk"].values, curve["coverage"].values))


def bootstrap_ci(
    values: Sequence[float],
    stat=np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    v = np.asarray(values, float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array(
        [stat(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)]
    )
    return (
        float(stat(v)),
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )


def tier_validity(
    included: Sequence[bool], correct: Sequence[int], seed: int = 42
) -> Dict[str, float]:
    """Precisión (entre incluidas) y recall (entre las correctas) con IC bootstrap.

    Requiere una muestra de estándar humano representativa de TODA la corrida
    (incluidas y no incluidas); si solo se codificaron incluidas, el recall no
    es interpretable.
    """
    inc = np.asarray(included, bool)
    y = np.asarray(correct, float)
    out = {"n_gold": int(len(y)), "n_gold_included": int(inc.sum())}
    if inc.sum() > 0:
        m, lo, hi = bootstrap_ci(y[inc], seed=seed)
        out.update(precision=m, precision_lo=lo, precision_hi=hi)
    else:
        out.update(
            precision=float("nan"), precision_lo=float("nan"), precision_hi=float("nan")
        )
    if y.sum() > 0:
        rec = (y * inc).sum() / y.sum()
        out["recall"] = float(rec)
    else:
        out["recall"] = float("nan")
    return out


# ── Comparaciones entre niveles / corridas ──────────────────────────────────


def kendall_top_terms(a: Dict[str, float], b: Dict[str, float], k: int = 50) -> float:
    """Tau de Kendall entre los rangos de los k términos principales de dos perfiles."""
    ta = [t for t, _ in sorted(a.items(), key=lambda x: -x[1])[:k]]
    tb = [t for t, _ in sorted(b.items(), key=lambda x: -x[1])[:k]]
    union = list(dict.fromkeys(ta + tb))
    ra = {t: i for i, t in enumerate(ta)}
    rb = {t: i for i, t in enumerate(tb)}
    xa = [ra.get(t, k) for t in union]
    xb = [rb.get(t, k) for t in union]
    if len(union) < 2:
        return float("nan")
    tau, _ = kendalltau(xa, xb)
    return float(tau)


def null_zscore(observed: float, null_values: Sequence[float]) -> float:
    v = np.asarray(null_values, float)
    if len(v) < 2 or v.std(ddof=1) == 0:
        return float("nan")
    return float((observed - v.mean()) / v.std(ddof=1))


def pareto_mask(
    df: pd.DataFrame, maximize: Sequence[str], minimize: Sequence[str] = ()
) -> pd.Series:
    """True para filas no dominadas. Ignora columnas ausentes o totalmente nulas."""
    cols_max = [c for c in maximize if c in df and df[c].notna().any()]
    cols_min = [c for c in minimize if c in df and df[c].notna().any()]
    if not cols_max and not cols_min:
        return pd.Series(True, index=df.index)
    M = (
        pd.concat(
            [df[c].astype(float) for c in cols_max]
            + [-df[c].astype(float) for c in cols_min],
            axis=1,
        )
        .fillna(-np.inf)
        .to_numpy()
    )
    n = len(M)
    keep = np.ones(n, bool)
    for i in range(n):
        dom = np.all(M >= M[i], axis=1) & np.any(M > M[i], axis=1)
        if dom.any():
            keep[i] = False
    return pd.Series(keep, index=df.index)
