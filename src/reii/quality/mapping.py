"""Alineación entre dos particiones sobre las mismas UCEs.

``hungarian``   : asignación 1-1 (comportamiento actual de
                  ``DoubleClassifier._hungarian_stability_direct``).
``many_to_one`` : parte del Hungarian y, solo si K difiere, permite que una clase
                  absorba hasta ``max_merge`` clases de la otra partición cuando
                  el solapamiento relativo supera ``min_share``. Con K igual y sin
                  clases sueltas coincide con el Hungarian.

Una UCE es estable si su clase en A y su clase en B quedan emparejadas por el
mapeo. Las UCEs con etiqueta < 0 (ruido) nunca son estables.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

Labels = Dict[str, int]


def overlap_matrix(a: Labels, b: Labels) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matriz de solapamiento (clases reales de A x clases reales de B) en UCEs comunes."""
    common = set(a) & set(b)
    real_a = np.unique([v for v in a.values() if v >= 0])
    real_b = np.unique([v for v in b.values() if v >= 0])
    ov = np.zeros((len(real_a), len(real_b)), dtype=float)
    if len(real_a) == 0 or len(real_b) == 0:
        return ov, real_a, real_b
    ia = {int(v): i for i, v in enumerate(real_a)}
    ib = {int(v): i for i, v in enumerate(real_b)}
    for u in common:
        la, lb = int(a[u]), int(b[u])
        if la >= 0 and lb >= 0:
            ov[ia[la], ib[lb]] += 1
    return ov, real_a, real_b


def hungarian_mapping(ov: np.ndarray) -> Dict[int, Set[int]]:
    """Índices de fila -> {índice de columna} usando asignación óptima 1-1."""
    if ov.size == 0:
        return {}
    r, c = linear_sum_assignment(-ov)
    return {int(i): {int(j)} for i, j in zip(r, c)}


def many_to_one_mapping(
    ov: np.ndarray, min_share: float = 0.5, max_merge: int = 2
) -> Dict[int, Set[int]]:
    """Hungarian + absorción restringida de clases sueltas.

    - Fila a sin pareja (K_A > K_B): se une a la columna con mayor solapamiento si
      ese solapamiento cubre >= min_share de a y la columna aún admite fusiones.
    - Columna b sin pareja (K_B > K_A): se une a la fila cuyo solapamiento cubre
      >= min_share de b y que aún admite fusiones.
    ``max_merge`` limita cuántas clases del otro lado puede absorber una clase
    (incluida la pareja inicial), para evitar que todo colapse en una sola.
    """
    mapping = hungarian_mapping(ov)
    if ov.size == 0:
        return mapping
    n_a, n_b = ov.shape
    row_sum = ov.sum(axis=1)
    col_sum = ov.sum(axis=0)

    matched_rows = set(mapping)
    matched_cols = {j for s in mapping.values() for j in s}
    col_load: Dict[int, int] = {j: 1 for j in matched_cols}  # filas por columna

    # filas sueltas -> columna dominante
    for i in range(n_a):
        if i in matched_rows or row_sum[i] <= 0:
            continue
        for j in np.argsort(-ov[i]):
            j = int(j)
            if ov[i, j] <= 0:
                break
            if ov[i, j] / row_sum[i] < min_share:
                break
            if col_load.get(j, 0) < max_merge:
                mapping.setdefault(i, set()).add(j)
                col_load[j] = col_load.get(j, 0) + 1
                break

    # columnas sueltas -> fila dominante
    for j in range(n_b):
        if j in matched_cols or col_sum[j] <= 0:
            continue
        for i in np.argsort(-ov[:, j]):
            i = int(i)
            if ov[i, j] <= 0:
                break
            if ov[i, j] / col_sum[j] < min_share:
                break
            if len(mapping.get(i, set())) < max_merge:
                mapping.setdefault(i, set()).add(j)
                break
    return mapping


def stability(
    a: Labels,
    b: Labels,
    strategy: str = "hungarian",
    min_share: float = 0.5,
    max_merge: int = 2,
    n_total: Optional[int] = None,
) -> Dict[str, object]:
    """Estabilidad entre dos particiones. Devuelve un dict con:

    stable   : {uce_id: clase_en_A} para UCEs emparejadas
    mapping  : {clase_A: [clases_B]}
    overlap, rows, cols, ari, n_common, n_stable, k_a, k_b, strategy
    n_total  : total de UCEs del corpus (denominador de retención)
    retention: n_stable / n_total (si n_total se provee; si no, n_stable / n_common)
    """
    common = sorted(set(a) & set(b))
    ov, real_a, real_b = overlap_matrix(a, b)
    out: Dict[str, object] = {
        "stable": {},
        "mapping": {},
        "overlap": ov,
        "rows": real_a,
        "cols": real_b,
        "ari": 0.0,
        "n_common": len(common),
        "n_stable": 0,
        "k_a": int(len(real_a)),
        "k_b": int(len(real_b)),
        "strategy": strategy,
    }
    if not common or ov.size == 0:
        return out
    if strategy == "hungarian":
        idx_map = hungarian_mapping(ov)
    elif strategy == "many_to_one":
        idx_map = many_to_one_mapping(ov, min_share=min_share, max_merge=max_merge)
    else:
        raise ValueError(f"strategy desconocida: {strategy!r}")
    mapping = {
        int(real_a[i]): sorted(int(real_b[j]) for j in js) for i, js in idx_map.items()
    }
    stable = {
        u: int(a[u])
        for u in common
        if a[u] >= 0 and b[u] >= 0 and int(b[u]) in mapping.get(int(a[u]), [])
    }
    arr_a = np.array([a[u] for u in common])
    arr_b = np.array([b[u] for u in common])
    valid = (arr_a >= 0) & (arr_b >= 0)
    ari = (
        float(adjusted_rand_score(arr_a[valid], arr_b[valid]))
        if valid.sum() >= 2
        else 0.0
    )
    denom = int(n_total) if n_total is not None else len(common)
    out.update(
        stable=stable,
        mapping=mapping,
        ari=ari,
        n_stable=len(stable),
        n_total=denom,
        retention=len(stable) / denom if denom else 0.0,
    )
    return out


def hungarian_compatible(
    uce_to_ca: Labels,
    uce_to_cb: Labels,
    strategy: str = "many_to_one",
    label: str = "",
    min_share: float = 0.5,
    max_merge: int = 2,
):
    """Misma firma de retorno que ``_hungarian_stability_direct`` del workflow:

    (stable, ari, arr_a[valid], arr_b[valid], overlap, mapping_a_b)
    ``mapping_a_b`` conserva el formato Dict[int, int] (clase principal de B).
    """
    res = stability(uce_to_ca, uce_to_cb, strategy, min_share, max_merge)
    common = sorted(set(uce_to_ca) & set(uce_to_cb))
    arr_a = np.array([uce_to_ca[u] for u in common])
    arr_b = np.array([uce_to_cb[u] for u in common])
    valid = (arr_a >= 0) & (arr_b >= 0) if len(common) else np.array([], dtype=bool)
    mapping_primary = {k: int(v[0]) for k, v in res["mapping"].items() if v}  # type: ignore[union-attr]
    tag = f" [{label}]" if label else ""
    n_c = int(res["n_common"])  # type: ignore[arg-type]
    n_s = int(res["n_stable"])  # type: ignore[arg-type]
    print(
        f"   Stability/{strategy}{tag}: {n_s}/{n_c} stable "
        f"({100.0 * n_s / max(1, n_c):.1f}%)  ARI={res['ari']:.3f}  "
        f"K={res['k_a']}x{res['k_b']}"
    )
    return (
        res["stable"],
        res["ari"],
        arr_a[valid] if len(common) else np.array([]),
        arr_b[valid] if len(common) else np.array([]),
        res["overlap"],
        mapping_primary,
    )
