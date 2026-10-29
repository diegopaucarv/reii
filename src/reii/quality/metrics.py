"""Indicadores de evaluación (funciones puras, sin acceso a base de datos)."""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
from sklearn.metrics import (
    adjusted_rand_score,
    cohen_kappa_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    silhouette_score,
)

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


# ── Matrices de términos y validez interna (Silhouette) ─────────────────────


def prune_vocabulary(
    term_lists: Iterable[Sequence[str]], tsj: int = 3, min_abs: int = 2
) -> List[str]:
    """Réplica de ``MatrizBuilder.construir_vocabulario`` para listas de términos.

    Conserva los términos con frecuencia documental >= ``tsj`` y frecuencia
    absoluta >= ``min_abs``. Se usa cuando el vocabulario real del CDH no se
    persistió (corridas ya guardadas).
    """
    doc_freq: Dict[str, int] = {}
    raw_freq: Dict[str, int] = {}
    for terms in term_lists:
        for t in terms:
            raw_freq[t] = raw_freq.get(t, 0) + 1
        for t in set(terms):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    min_abs = max(1, int(min_abs))
    return sorted(t for t, c in doc_freq.items() if c >= tsj and raw_freq[t] >= min_abs)


def binary_term_matrix(
    term_lists: Sequence[Sequence[str]], vocab: Sequence[str]
) -> sparse.csr_matrix:
    """Matriz binaria UCE x término (CSR, int32), misma construcción que el pipeline."""
    vocab = list(dict.fromkeys(vocab))
    idx = {t: i for i, t in enumerate(vocab)}
    rows: List[int] = []
    cols: List[int] = []
    for i, terms in enumerate(term_lists):
        for t in set(terms):
            j = idx.get(t)
            if j is not None:
                rows.append(i)
                cols.append(j)
    return sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(term_lists), len(vocab)),
        dtype=np.int32,
    ).tocsr()


def chi2_profiles(matrix) -> Tuple[Optional[sparse.csr_matrix], np.ndarray]:
    """Perfiles fila escalados de modo que la distancia euclídea sea la distancia χ².

    Z_ij = (x_ij / x_i.) / sqrt(x_.j / x_..): es la geometría del AFC/ALCESTE (la
    misma de ``consensus.chi2_margin``). Descarta filas sin términos y columnas
    sin masa. Devuelve ``(Z, keep_rows)``; ``Z`` es None si no hay masa.
    """
    X = (
        matrix.tocsr().astype(np.float64)
        if sparse.issparse(matrix)
        else sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))
    )
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    col_mass = np.asarray(X.sum(axis=0)).ravel()
    keep_rows = row_sums > 0
    keep_cols = col_mass > 0
    total = float(col_mass.sum())
    if total <= 0 or keep_rows.sum() == 0 or keep_cols.sum() == 0:
        return None, keep_rows
    X = X[keep_rows][:, keep_cols]
    P = sparse.diags(1.0 / row_sums[keep_rows]) @ X
    marg = col_mass[keep_cols] / total
    Z = P @ sparse.diags(1.0 / np.sqrt(marg))
    return Z.tocsr(), keep_rows


def silhouette_labels(
    matrix,
    labels: Sequence[int],
    metric: str = "cosine",
    sample_size: Optional[int] = 3000,
    seed: int = 42,
) -> Optional[float]:
    """Silhouette medio de ``labels`` en el espacio de ``matrix`` (la matriz la aporta el llamador).

    metric : "cosine" | "euclidean" | "chi2" (perfiles χ² + euclídea).
    Ignora etiquetas < 0 y filas vacías. Si hay más de ``sample_size`` filas se
    submuestrea con semilla fija (O(n²)). Devuelve None si no es calculable
    (< 3 filas, < 2 clases o una clase por fila).
    """
    y = np.asarray(labels).astype(int)
    if matrix.shape[0] != len(y):
        raise ValueError("matrix y labels deben tener el mismo número de filas")
    keep = y >= 0
    X = matrix[keep] if sparse.issparse(matrix) else np.asarray(matrix)[keep]
    y = y[keep]
    if metric == "chi2":
        X, ok = chi2_profiles(X)
        if X is None:
            return None
        y, metric = y[ok], "euclidean"
    elif metric == "cosine":
        norms = (
            np.asarray(abs(X).sum(axis=1)).ravel()
            if sparse.issparse(X)
            else np.abs(X).sum(axis=1)
        )
        ok = norms > 0
        X, y = X[ok], y[ok]
    if len(y) < 3 or len(np.unique(y)) < 2:
        return None
    if sample_size and len(y) > sample_size:
        idx = np.random.default_rng(seed).choice(len(y), size=sample_size, replace=False)
        X, y = X[idx], y[idx]
    k = len(np.unique(y))
    if k < 2 or k > len(y) - 1:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(silhouette_score(X, y, metric=metric))


# ── Acuerdo entre particiones, entre codificadores y contra el estándar humano ──


def cross_partition_agreement(
    labels_a: Mapping[str, int], labels_b: Mapping[str, int]
) -> Dict[str, float]:
    """FMI / NMI / ARI entre dos particiones, en las UCEs comunes con etiqueta >= 0.

    Invariante a la permutación de etiquetas. Pura y sin mutar sus entradas.
    """
    pairs = [
        (int(labels_a[u]), int(labels_b[u]))
        for u in sorted(set(labels_a) & set(labels_b))
        if labels_a[u] is not None
        and labels_b[u] is not None
        and labels_a[u] >= 0
        and labels_b[u] >= 0
    ]
    out = {
        "n_common": len(pairs),
        "fmi": float("nan"),
        "nmi": float("nan"),
        "ari": float("nan"),
    }
    if len(pairs) < 2:
        return out
    la, lb = (list(x) for x in zip(*pairs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["fmi"] = float(fowlkes_mallows_score(la, lb))
        out["nmi"] = float(normalized_mutual_info_score(la, lb))
        out["ari"] = float(adjusted_rand_score(la, lb))
    return out


def _paired(a: Sequence, b: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(pd.Series(list(a)), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(list(b)), errors="coerce").to_numpy(float)
    if len(x) != len(y):
        raise ValueError("a y b deben tener la misma longitud")
    ok = ~(np.isnan(x) | np.isnan(y))
    return x[ok].astype(int), y[ok].astype(int)


def percent_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    """Proporción de ítems con la misma etiqueta (NaN si no hay pares)."""
    x, y = _paired(a, b)
    return float((x == y).mean()) if len(x) else float("nan")


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Kappa de Cohen. NaN si hay < 2 pares comunes o el kappa es degenerado
    (una sola categoría observada en total: acuerdo perfecto sin variación)."""
    x, y = _paired(a, b)
    if len(x) < 2 or len(set(x) | set(y)) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = cohen_kappa_score(x, y)
    return float(k) if np.isfinite(k) else float("nan")


def external_coherence(
    assigned: Sequence[int], gold: Sequence[int]
) -> Dict[str, float]:
    """C_ext: coincidencia entre las clases del pipeline y las del estándar humano.

    accuracy         : proporción con la misma etiqueta (requiere el MISMO espacio de ids)
    kappa            : kappa de Cohen entre ambas etiquetas (mismo requisito)
    accuracy_aligned : exactitud tras emparejar clases con Hungarian (si el codificador
                       usó su propia numeración)
    nmi, ari         : invariantes a la numeración de las clases
    """
    x, y = _paired(assigned, gold)
    n = len(x)
    out = {
        "n": n,
        "accuracy": float("nan"),
        "kappa": float("nan"),
        "accuracy_aligned": float("nan"),
        "nmi": float("nan"),
        "ari": float("nan"),
    }
    if n == 0:
        return out
    out["accuracy"] = float((x == y).mean())
    out["kappa"] = cohen_kappa(x, y)
    ca, cb = np.unique(x), np.unique(y)
    ov = np.zeros((len(ca), len(cb)))
    ia = {int(v): i for i, v in enumerate(ca)}
    ib = {int(v): i for i, v in enumerate(cb)}
    for u, v in zip(x, y):
        ov[ia[int(u)], ib[int(v)]] += 1
    r, c = linear_sum_assignment(-ov)
    out["accuracy_aligned"] = float(ov[r, c].sum() / n)
    if n >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["nmi"] = float(normalized_mutual_info_score(x, y))
            out["ari"] = float(adjusted_rand_score(x, y))
    return out


# ── Coherencia interna, diversidad de particion y RMSD de proporciones ──────


def _row_normalize(X):
    """Normaliza filas a norma unitaria (densa o dispersa), respetando ceros."""
    if sparse.issparse(X):
        X = X.tocsr().astype(float)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        return sparse.diags(1.0 / np.where(norms > 0, norms, 1.0)) @ X
    X = np.asarray(X, float)
    norms = np.linalg.norm(X, axis=1)
    return X / np.where(norms > 0, norms, 1.0)[:, None]


def _cos_to_centroid(Xn, y, classes):
    """Coseno al centroide de cada clase: por clase y por UCE, y n negativos."""
    per_class: List[Tuple[int, float]] = []
    per_item: List[float] = []
    n_neg = 0
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        if len(idx) == 0:
            continue
        mu = np.asarray(Xn[idx].mean(axis=0)).ravel()
        norm = float(np.linalg.norm(mu))
        cos = (
            np.zeros(len(idx))
            if norm <= 0
            else np.asarray(Xn[idx] @ (mu / norm)).ravel()
        )
        n_neg += int((cos < 0.0).sum())
        cos = np.clip(cos, 0.0, 1.0)
        per_class.append((len(idx), float(cos.mean())))
        per_item.extend(cos.tolist())
    return per_class, np.asarray(per_item, float), n_neg


def internal_coherence(
    matrix,
    labels: Sequence[int],
    space_id: str,
    metric: str = "cosine",
    max_pairwise: int = 1500,
    n_boot: int = 500,
    n_perm: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """C_in: homogeneidad intra-clase medida en un espacio DECLARADO.

    ``space_id`` es obligatorio: un C_in no significa nada sin decir en que
    espacio se midio (embedding de 384d, perfiles chi2, tf-idf...). La funcion
    no adivina el espacio: lo recibe y lo devuelve en el resultado.

    Para cada clase ``C_k`` con ``n_k`` elementos y centroide normalizado
    ``mu_k``, el score de la UCE ``i`` es ``max(0, cos(x_i, mu_k))`` (truncado
    a [0,1]: un coseno negativo es anti-similitud y se reporta aparte en
    ``frac_negative``). Se devuelven tres versiones:

    ``cin``           media MACRO de las medias por clase (la del paper).
    ``cin_weighted``  ponderada por tamano de clase.
    ``cin_pairwise``  media de similitudes pairwise intra-clase (control del
                      sesgo del centroide hacia arriba para clases pequenas).

    En d alta el valor absoluto no es interpretable: se anade ``excess =
    cin - null_mean``, donde el nulo permuta las etiquetas (preserva tamanos
    de clase y espacio, rompe la asociacion UCE-clase). Reportar ``cin`` sin
    ``excess`` no distingue "clases coherentes" de "corpus homogeneo".

    ``ci_low``/``ci_high`` es el IC bootstrap de la media macro (sobre las K
    medias de clase); ``ci_low_weighted``/``ci_high_weighted`` lo es de la
    ponderada (sobre las n UCEs). Con K chico el IC macro es conservador por
    construccion: se remuestrean K valores.

    Ignora etiquetas < 0. Devuelve NaN (con ``n`` y ``k`` poblados) si hay
    menos de 3 UCEs o menos de 2 clases.
    """
    from scipy import sparse  # noqa: F401  (import local: ver _row_normalize)

    y = np.asarray(list(labels)).astype(int)
    X_full = matrix if hasattr(matrix, "shape") else np.asarray(matrix, float)
    if X_full.shape[0] != len(y):
        raise ValueError("matrix y labels deben tener el mismo numero de filas")
    keep = y >= 0
    X = X_full[keep] if sparse.issparse(X_full) else np.asarray(X_full, float)[keep]
    y = y[keep]
    n = int(len(y))
    classes = np.unique(y)
    k = int(len(classes))
    out: Dict[str, Any] = {
        "cin": float("nan"),
        "cin_weighted": float("nan"),
        "cin_pairwise": float("nan"),
        "space_id": space_id,
        "metric": metric,
        "n": n,
        "k": k,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "ci_low_weighted": float("nan"),
        "ci_high_weighted": float("nan"),
        "frac_negative": float("nan"),
        "null_mean": float("nan"),
        "null_std": float("nan"),
        "excess": float("nan"),
        "z_score": float("nan"),
        "p_value": float("nan"),
    }
    if n < 3 or k < 2:
        return out
    Xn = _row_normalize(X)
    per_class, per_item, n_neg = _cos_to_centroid(Xn, y, classes)
    if not per_class:
        return out
    coin = np.array([m for _, m in per_class])
    counts = np.array([c for c, _ in per_class], float)
    cin = float(coin.mean())
    out["cin"] = cin
    out["cin_weighted"] = float((counts * coin).sum() / counts.sum())
    out["frac_negative"] = n_neg / float(n)
    out["ci_low"], out["ci_high"] = bootstrap_ci(
        coin, np.mean, n_boot=n_boot, seed=seed
    )[1:]
    out["ci_low_weighted"], out["ci_high_weighted"] = bootstrap_ci(
        per_item, np.mean, n_boot=n_boot, seed=seed
    )[1:]
    # pairwise: O(n_c^2) por clase, con tope por clase
    pair = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        if len(idx) < 2:
            continue
        if len(idx) > max_pairwise:
            idx = np.random.default_rng(seed).choice(idx, max_pairwise, replace=False)
        gram = Xn[idx] @ Xn[idx].T
        sim = np.asarray(gram.todense()) if sparse.issparse(gram) else np.asarray(gram)
        iu = np.triu_indices(len(idx), 1)
        pair.append(float(np.clip(sim[iu], 0.0, 1.0).mean()))
    if pair:
        out["cin_pairwise"] = float(np.mean(pair))
    # nulo: permutacion de etiquetas (rompe la asociacion UCE-clase)
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        pc, _, _ = _cos_to_centroid(Xn, rng.permutation(y), classes)
        nulls[i] = float(np.mean([m for _, m in pc]))
    std = float(nulls.std(ddof=1))
    out["null_mean"] = float(nulls.mean())
    out["null_std"] = std
    out["excess"] = cin - out["null_mean"]
    out["z_score"] = (cin - out["null_mean"]) / std if std > 0 else float("nan")
    out["p_value"] = float((nulls >= cin).mean())
    return out


def _entropy_stats(y, k_ref: Optional[int] = None) -> Dict[str, float]:
    """Entropia y tamanos de una particion; ``k_ref`` fija K en el denominador."""
    counts = np.bincount(y)
    counts = counts[counts > 0]
    n = int(counts.sum())
    p = counts / n
    h = float(-(p * np.log(p)).sum())
    k = int(len(p))
    k_norm = int(k_ref) if k_ref else k
    return {
        "n": n,
        "k": k,
        "entropy": h,
        "entropy_norm": h / np.log(k_norm) if k_norm > 1 else float("nan"),
        "simpson": float(1.0 - (p * p).sum()),
        "effective_k": float(np.exp(h)),
        "max_share": float(p.max()),
    }


def partition_diversity(
    labels: Sequence[int], n_boot: int = 500, n_null: int = 500, seed: int = 42
) -> Dict[str, Any]:
    """d: diversidad de la particion (entropia normalizada, Simpson, Hill).

    Propiedad de la PARTICION, no del contenido: solo depende del vector de
    tamanos de clase. NO es ``diversidad_semantica`` (la metrica por UCE del
    pipeline, ~0.82): aquella mide dispersion lexica dentro de una UCE, esta
    mide equilibrio entre clases.

    ``entropy``       H en nats.
    ``entropy_norm``  H / ln K en [0,1] (la ``d`` que pide el paper).
    ``simpson``       1 - sum p_k^2 (Gini-Simpson, en [0,1)).
    ``effective_k``   exp(H): "numero efectivo de clases" (Hill).
    ``max_share``     max p_k (coincide con el gate de dominancia sobre lo
                      asignado cuando no hay ruido).

    Nulo: la permutacion de etiquetas es VACUA aqui (invariante: permutar no
    cambia ningun p_k), y se declara con ``label_permutation_invariant``. El
    nulo que se reporta es de composicion, Dirichlet(1) con el mismo K: mide
    si la particion esta mas (o menos) balanceada de lo que daria un reparto
    aleatorio uniforme. El p-valor es bilateral.

    ``ci_low``/``ci_high`` es el IC bootstrap de ``entropy_norm`` resampleando
    UCEs con K FIJO en el denominador (si K variara en cada remuestra, el IC
    estaria sesgado al alza por desaparicion de clases raras).

    ``entropy_norm`` no es comparable entre particiones con distinto K
    (normaliza por ln K); para comparar se usa ``effective_k``.
    """
    y = np.asarray(list(labels)).astype(int)
    y = y[y >= 0]
    base = _entropy_stats(y) if len(y) else None
    out: Dict[str, Any] = (
        dict(base)
        if base
        else {
            "n": 0,
            "k": 0,
            "entropy": float("nan"),
            "entropy_norm": float("nan"),
            "simpson": float("nan"),
            "effective_k": float("nan"),
            "max_share": float("nan"),
        }
    )
    out.update(
        {
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "null_mean": float("nan"),
            "null_q05": float("nan"),
            "null_q95": float("nan"),
            "p_value": float("nan"),
            "label_permutation_invariant": True,
            "space_id": "cluster_id",
        }
    )
    if base is None or base["k"] < 2:
        return out
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = _entropy_stats(
            rng.choice(y, size=len(y), replace=True), k_ref=base["k"]
        )["entropy_norm"]
    out["ci_low"] = float(np.quantile(boots, 0.025))
    out["ci_high"] = float(np.quantile(boots, 0.975))
    comp = rng.dirichlet(np.ones(base["k"]), size=n_null)
    h = -(comp * np.log(comp)).sum(axis=1) / np.log(base["k"])
    out["null_mean"] = float(h.mean())
    out["null_q05"] = float(np.quantile(h, 0.05))
    out["null_q95"] = float(np.quantile(h, 0.95))
    p_lo = float((h <= base["entropy_norm"]).mean())
    p_hi = float((h >= base["entropy_norm"]).mean())
    out["p_value"] = float(min(1.0, 2.0 * min(p_lo, p_hi)))
    return out


def _rmsd_core(ra: np.ndarray, ob: np.ndarray) -> Dict[str, float]:
    """RMSD de proporciones con emparejamiento Hungarian de clases."""
    ru, ri = np.unique(ra, return_inverse=True)
    ou, oi = np.unique(ob, return_inverse=True)
    n = len(ra)
    ov = np.zeros((len(ru), len(ou)))
    np.add.at(ov, (ri, oi), 1.0)
    row, col = ov.sum(axis=1), ov.sum(axis=0)
    r, c = linear_sum_assignment(-ov)
    pairs = [(row[i] / n, col[j] / n) for i, j in zip(r, c)]
    used_r, used_c = set(r.tolist()), set(c.tolist())
    pairs += [(row[i] / n, 0.0) for i in range(len(ru)) if i not in used_r]
    pairs += [(0.0, col[j] / n) for j in range(len(ou)) if j not in used_c]
    pr = np.array([p for p, _ in pairs])
    po = np.array([q for _, q in pairs])
    return {
        "rmsd": float(np.sqrt(np.mean((pr - po) ** 2))) if len(pr) else float("nan"),
        "k_ref": int(len(ru)),
        "k_obs": int(len(ou)),
        "k_union": int(len(pr)),
        "n_unmatched": int(len(pr) - len(r)),
    }


def rmsd_proportions(
    labels_ref: Mapping[str, int],
    labels_obs: Mapping[str, int],
    reference_id: str,
    n_boot: int = 300,
    n_null: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """RMSD entre los vectores de proporcion de clase de dos particiones.

    Se usan las UCEs comunes con etiqueta >= 0 en ambas. Las clases se
    emparejan con Hungarian maximizando el solape de la matriz de confusion, de
    modo que el estadistico es invariante a la numeracion de las clases y
    conservador: el emparejamiento se elige para que las proporciones se
    parezcan lo mas posible.

    ``reference_id`` es OBLIGATORIO y se devuelve en la salida. Sin el, el
    numero no es interpretable:

    * referencia externa (codificacion humana) -> parte de la validez externa;
    * otra salida del pipeline -> concordancia entre metodos, NO validez;
    * particion derivada del mismo dato -> ilustracion del calculo.

    ADVERTENCIA (medida, no teorica): esto mide concordancia de PERFIL de
    tamanos, NO de asignacion. El emparejamiento maximiza el solape, de modo
    que el estadistico queda dominado por los marginales: en el self-test,
    una particion ALEATORIA con el mismo vector de tamanos da RMSD menor que
    la particion real. El nulo por permutacion de etiquetas es valido (el
    emparejamiento depende de la asociacion) pero su dispersion es chica, y
    un p-valor alto significa "no hay evidencia de acuerdo", no "hay
    desacuerdo". Para acuerdo a nivel de UCE usar ``cross_partition_agreement``
    (FMI/NMI/ARI). Recomendacion: reportar el RMSD como DESCRIPTOR de
    perfiles y nunca como evidencia de validez.

    ``k_union`` son los pares usados y ``n_unmatched`` las clases sin pareja
    (penalizan que las dos particiones tengan distinto K). El IC bootstrap
    remuestrea UCEs y RECALCULA el emparejamiento en cada remuestra.
    """
    common = [
        u
        for u in sorted(set(labels_ref) & set(labels_obs))
        if labels_ref[u] is not None
        and labels_obs[u] is not None
        and int(labels_ref[u]) >= 0
        and int(labels_obs[u]) >= 0
    ]
    out: Dict[str, Any] = {
        "rmsd": float("nan"),
        "n": len(common),
        "k_ref": 0,
        "k_obs": 0,
        "k_union": 0,
        "n_unmatched": 0,
        "reference_id": reference_id,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "null_mean": float("nan"),
        "null_std": float("nan"),
        "null_q05": float("nan"),
        "null_q95": float("nan"),
        "p_value": float("nan"),
    }
    if len(common) < 2:
        return out
    ra = np.array([int(labels_ref[u]) for u in common])
    ob = np.array([int(labels_obs[u]) for u in common])
    out.update(_rmsd_core(ra, ob))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(common), len(common))
        boots[i] = _rmsd_core(ra[idx], ob[idx])["rmsd"]
    out["ci_low"] = float(np.quantile(boots, 0.025))
    out["ci_high"] = float(np.quantile(boots, 0.975))
    nulls = np.empty(n_null)
    for i in range(n_null):
        nulls[i] = _rmsd_core(ra, rng.permutation(ob))["rmsd"]
    out["null_mean"] = float(nulls.mean())
    out["null_std"] = float(nulls.std(ddof=1))
    out["null_q05"] = float(np.quantile(nulls, 0.05))
    out["null_q95"] = float(np.quantile(nulls, 0.95))
    out["p_value"] = float((nulls >= out["rmsd"]).mean())
    return out


# ── Estabilidad por remuestreo y nulo de etiquetas (STABILITY_NULL.md) ──────

#: `ci_low` minimo para declarar ESTABLE. Medido: ruido <= 0.15, estructura
#: >= 0.74 (STABILITY_NULL.md §2). No subir sin nueva evidencia.
STABILITY_CI_LOW_MIN = 0.50

#: `bootstrap_ari` medio por debajo del cual no hay senal recuperable.
#: Debe coincidir con `method_selection.stability_mean_min`.
STABILITY_MEAN_MIN = 0.30


def _r2_partition(X, labels) -> float:
    """R2 denso `1 - SSE/TSS` de `labels` sobre `X`. NaN si no es calculable.

    Es el estadistico por defecto del nulo porque es **el que el clusterer
    optimiza**: por eso el nulo de etiquetas no puede rechazar ruido
    (STABILITY_NULL.md §1.1). Se expone tambien como utilidad de auditoria.
    """
    try:
        A = np.asarray(X, dtype=float)
        y = np.asarray(labels).astype(int)
    except (TypeError, ValueError):
        return float("nan")
    if A.ndim != 2 or y.ndim != 1 or A.shape[0] != y.shape[0]:
        return float("nan")
    keep = y >= 0
    if int(keep.sum()) < 3 or np.unique(y[keep]).size < 2:
        return float("nan")
    A, y = A[keep], y[keep]
    mu = A.mean(axis=0, keepdims=True)
    tss = float(np.sum((A - mu) ** 2))
    if not np.isfinite(tss) or tss <= 0.0:
        return float("nan")
    sse = 0.0
    for c in np.unique(y):
        Ac = A[y == c]
        sse += float(np.sum((Ac - Ac.mean(axis=0, keepdims=True)) ** 2))
    value = 1.0 - sse / tss
    return float(value) if np.isfinite(value) else float("nan")


def _stability_verdict(mean: float, ci_low: float) -> str:
    """Veredicto de §3: ESTABLE | SIN_SENAL | FRAGIL | NO_DISPONIBLE."""
    if not math.isfinite(float(mean)):
        return "NO_DISPONIBLE"
    if float(mean) < STABILITY_MEAN_MIN:
        return "SIN_SENAL"
    if math.isfinite(float(ci_low)) and float(ci_low) >= STABILITY_CI_LOW_MIN:
        return "ESTABLE"
    return "FRAGIL"


def bootstrap_stability(
    X,
    labels,
    *,
    predict_fn: Optional[Callable] = None,
    n_boot: int = 10,
    frac: float = 0.8,
    replace: bool = False,
    seed: int = 42,
    min_n: int = 30,
    alpha: float = 0.05,
    stable_threshold: float = 0.75,
) -> Dict[str, Any]:
    """Estabilidad por submuestreo de UCEs + reajuste con el mismo K.

    Contrato de `predict_fn`: `predict_fn(X_sub) -> labels` de longitud
    `len(X_sub)`. El K, el piso y el `random_state` los cierra el llamador, de
    modo que la funcion no se acopla a ward, kmeans ni dense_cdh.

    En cada replica se submuuestrea `ceil(frac*n)` filas (sin reemplazo por
    defecto), se reajusta el clusterer y se mide `ARI(pred, labels)` sobre las
    filas comunes con etiqueta `>= 0`. `labels` es la particion de REFERENCIA
    (la del ajuste completo).

    Devuelve el bloque de STABILITY_NULL.md §4:
    `bootstrap_ari`, `bootstrap_ari_ci_low`, `bootstrap_ari_ci_high`,
    `bootstrap_n_boot`, `coverage_stable`, `stability_kind`,
    `stability_verdict` (+ `ok`, `reason`, `bootstrap_std`, `n`, `n_failed`).

    Precondiciones (todas devuelven `ok=False` + `nan`, NUNCA excepcion):
    `X` denso 2-D, `labels` 1-D de la misma longitud, `n >= min_n`,
    `predict_fn` callable, `n_boot >= 1`, `0 < frac <= 1`.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "reason": "",
        "bootstrap_ari": float("nan"),
        "bootstrap_ari_ci_low": float("nan"),
        "bootstrap_ari_ci_high": float("nan"),
        "bootstrap_n_boot": int(n_boot) if isinstance(n_boot, int) else 0,
        "bootstrap_std": float("nan"),
        "coverage_stable": float("nan"),
        "stability_kind": "",
        "stability_verdict": "NO_DISPONIBLE",
        "n": 0,
        "n_failed": 0,
    }
    try:
        A = np.asarray(X, dtype=float)
        y = np.asarray(labels).astype(int)
    except (TypeError, ValueError):
        out["reason"] = "X_o_labels_invalidos"
        return out
    if A.ndim != 2 or y.ndim != 1 or A.shape[0] != y.shape[0]:
        out["reason"] = "forma_invalida_o_longitudes_distintas"
        return out
    n = int(A.shape[0])
    out["n"] = n
    if n < int(min_n):
        out["reason"] = f"n={n}_menor_que_min_n={int(min_n)}"
        return out
    if not callable(predict_fn):
        out["reason"] = "predict_fn_no_es_callable"
        return out
    if int(n_boot) < 1:
        out["reason"] = "n_boot<1"
        return out
    frac = float(frac)
    if not (0.0 < frac <= 1.0):
        out["reason"] = "frac_fuera_de_(0,1]"
        return out
    if not np.isfinite(A).all():
        out["reason"] = "X_con_NaN_o_inf"
        return out

    n_sub = int(min(n, max(3, math.ceil(frac * n))))
    rng = np.random.default_rng(int(seed))
    ari_vals: List[float] = []
    hits = np.zeros(n, dtype=int)
    seen = np.zeros(n, dtype=int)
    fails = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(int(n_boot)):
            idx = rng.choice(n, size=n_sub, replace=bool(replace))
            if replace:
                idx = np.unique(idx)
            if idx.size < 3:
                fails += 1
                continue
            try:
                pred = np.asarray(predict_fn(A[idx])).ravel().astype(int)
            except Exception:
                fails += 1
                continue
            if pred.size != idx.size:
                fails += 1
                continue
            ys = y[idx]
            keep = ys >= 0
            if int(keep.sum()) < 2:
                fails += 1
                continue
            if np.unique(ys[keep]).size < 2 or np.unique(pred[keep]).size < 2:
                fails += 1
                continue
            value = float(adjusted_rand_score(ys[keep], pred[keep]))
            if not np.isfinite(value):
                fails += 1
                continue
            ari_vals.append(value)

            # Alineacion Hungarian para poder contar UCEs estables.
            ca, cb = np.unique(ys[keep]), np.unique(pred[keep])
            ia = {int(v): i for i, v in enumerate(ca)}
            ib = {int(v): j for j, v in enumerate(cb)}
            ov = np.zeros((ca.size, cb.size))
            for u, v in zip(ys[keep], pred[keep]):
                ov[ia[int(u)], ib[int(v)]] += 1
            rows, cols = linear_sum_assignment(-ov)
            mapped = {int(cb[j]): int(ca[i]) for i, j in zip(rows, cols)}
            for pos, uce in enumerate(idx):
                if ys[pos] < 0:
                    continue
                seen[uce] += 1
                if mapped.get(int(pred[pos])) == int(ys[pos]):
                    hits[uce] += 1

    out["n_failed"] = int(fails)
    if not ari_vals:
        out["reason"] = "todas_las_replicas_fallaron"
        return out
    arr = np.asarray(ari_vals, dtype=float)
    out["bootstrap_ari"] = float(arr.mean())
    out["bootstrap_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    out["bootstrap_ari_ci_low"] = float(np.quantile(arr, alpha / 2.0))
    out["bootstrap_ari_ci_high"] = float(np.quantile(arr, 1.0 - alpha / 2.0))
    assigned = y >= 0
    if assigned.any():
        need = np.ceil(float(stable_threshold) * np.maximum(seen, 1))
        ok_rows = (seen > 0) & (hits >= need) & assigned
        out["coverage_stable"] = float(int(ok_rows.sum()) / float(n))
    out["stability_kind"] = "bootstrap"
    out["stability_verdict"] = _stability_verdict(
        out["bootstrap_ari"], out["bootstrap_ari_ci_low"]
    )
    out["ok"] = True
    return out


def null_partition_test(
    X,
    labels,
    *,
    stat_fn: Optional[Callable] = None,
    n_perm: int = 50,
    seed: int = 42,
    min_n: int = 30,
) -> Dict[str, Any]:
    """Nulo por permutacion de ETIQUETAS. `stat_fn(X, labels) -> float`.

    Por defecto `_r2_partition`. AVISO (STABILITY_NULL.md §1.1): el estadistico
    observado procede de una particion AJUSTADA a X, asi que este nulo responde
    "mejor que etiquetas aleatorias", no "hay estructura". Sobre ruido puro
    devuelve `p = 1/(1+n_perm)` por sesgo de seleccion. **No es un gate.**

    Devuelve `null_p` (una cola superior, correccion +1), `null_stat`,
    `null_n_perm`, `null_z`, `null_mean`, `null_std`. Nunca lanza.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "reason": "",
        "null_p": float("nan"),
        "null_stat": float("nan"),
        "null_n_perm": int(n_perm) if isinstance(n_perm, int) else 0,
        "null_z": float("nan"),
        "null_mean": float("nan"),
        "null_std": float("nan"),
    }
    try:
        A = np.asarray(X, dtype=float)
        y = np.asarray(labels).astype(int)
    except (TypeError, ValueError):
        out["reason"] = "X_o_labels_invalidos"
        return out
    if A.ndim != 2 or y.ndim != 1 or A.shape[0] != y.shape[0]:
        out["reason"] = "forma_invalida_o_longitudes_distintas"
        return out
    if int(A.shape[0]) < int(min_n):
        out["reason"] = f"n={int(A.shape[0])}_menor_que_min_n={int(min_n)}"
        return out
    if int(n_perm) < 1:
        out["reason"] = "n_perm<1"
        return out
    stat = stat_fn if callable(stat_fn) else _r2_partition
    try:
        observed = float(stat(A, y))
    except Exception:
        observed = float("nan")
    if not np.isfinite(observed):
        out["reason"] = "estadistico_observado_NaN"
        return out
    out["null_stat"] = observed
    rng = np.random.default_rng(int(seed))
    nulls: List[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(int(n_perm)):
            try:
                value = float(stat(A, rng.permutation(y)))
            except Exception:
                continue
            if np.isfinite(value):
                nulls.append(value)
    if not nulls:
        out["reason"] = "todos_los_nulos_NaN"
        return out
    null = np.asarray(nulls, dtype=float)
    out["null_mean"] = float(null.mean())
    out["null_std"] = float(null.std(ddof=1)) if null.size > 1 else 0.0
    if out["null_std"] > 0.0:
        out["null_z"] = float((observed - out["null_mean"]) / out["null_std"])
    out["null_p"] = float((1.0 + float(np.sum(null >= observed))) / (null.size + 1.0))
    out["ok"] = True
    return out
