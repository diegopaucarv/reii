"""Puntaje de consenso por UCE.

s_acuerdo = fracción (ponderada) de comparaciones disponibles en que la UCE es
estable. Las comparaciones cuyo método no corrió se excluyen del denominador y se
cuenta cuántas entraron (n_comparisons).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

COMPARISONS = ("wc", "sim", "emb", "cross_ab", "cross_bc", "cross_ac")


def agreement_from_stable_sets(
    all_ids: Sequence[str],
    stable_sets: Dict[str, Iterable[str]],
    weights: Optional[Dict[str, float]] = None,
    available: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """stable_sets: nombre_comparación -> ids estables en esa comparación.

    ``available``: comparaciones que realmente corrieron (por defecto, las que
    tienen al menos una UCE estable). Devuelve un DataFrame indexado por uce_id con
    una columna booleana por comparación, ``n_comparisons`` y ``s_acuerdo``.
    """
    names = [n for n in stable_sets]
    sets = {n: set(stable_sets[n]) for n in names}
    avail = list(available) if available is not None else [n for n in names if sets[n]]
    df = pd.DataFrame(index=pd.Index(list(all_ids), name="uce_id"))
    for n in names:
        df[n] = [u in sets[n] for u in df.index]
    w = {n: float((weights or {}).get(n, 1.0)) for n in avail}
    tot = sum(w.values())
    if not avail or tot <= 0:
        df["n_comparisons"] = 0
        df["s_acuerdo"] = np.nan
        return df
    num = sum(df[n].astype(float) * w[n] for n in avail)
    df["n_comparisons"] = len(avail)
    df["s_acuerdo"] = num / tot
    return df


def chi2_margin(matrix, labels: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Margen χ² normalizado a [0, 1] de cada fila respecto a su clase.

    Perfiles fila (matriz binaria normalizada) frente a centroides de clase, con
    masas marginales globales, igual que ``_project_liminal_uces``. Devuelve
    (d2_segundo - d2_mejor) / d2_segundo; NaN para filas sin términos o con una
    sola clase disponible.
    """
    labels = np.asarray(labels)
    mat = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, float)
    mat = mat.astype(np.float64)
    n = mat.shape[0]
    out = np.full(n, np.nan)
    classes = np.unique(labels[labels >= 0])
    if len(classes) < 2:
        return out
    row_sums = mat.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    prof = mat / safe
    col_mass = mat.sum(axis=0)
    col_mass = np.where(col_mass > 0, col_mass, eps)
    marg = col_mass / col_mass.sum()
    cents = np.vstack([prof[labels == k].mean(axis=0) for k in classes])
    d2 = np.stack(
        [((prof - c) ** 2 / marg).sum(axis=1) for c in cents], axis=1
    )  # n x K
    d2s = np.sort(d2, axis=1)
    best, second = d2s[:, 0], d2s[:, 1]
    ok = (row_sums[:, 0] > 0) & (second > eps)
    out[ok] = (second[ok] - best[ok]) / second[ok]
    return out


def rank_score(df: pd.DataFrame, margin_weight: float = 0.01) -> pd.Series:
    """Valor para ordenar UCEs: acuerdo primero, margen χ² como desempate."""
    m = df["s_margen"].fillna(0.0) if "s_margen" in df else 0.0
    return df["s_acuerdo"].fillna(0.0) + margin_weight * m
