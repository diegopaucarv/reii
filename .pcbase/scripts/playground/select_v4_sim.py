#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precision de seleccion de metodo con verdad conocida: ``select-v3`` vs ``select-v4``.

QUE MIDE
--------
Genera corpus sinteticos con estructura PLANTADA de cuatro tipos y comprueba
que la regla de seleccion elige la familia de metodo que deberia:

    lexical  -> clases separadas por vocabulario; A/B deben ganar
    dense    -> blobs en el espacio vectorial; solo C debe ganar
    null     -> ni lexico ni geometria; nadie debe ganar (abstencion)
    mixed    -> lexico fuerte + geometria debil; A/B deben ganar

Para cada metodo se construyen las metricas COMO LAS CONSTRUIRIA el pipeline
(``M4``, espejo de ``MethodMetrics`` + los campos nuevos), y se aplican las
dos reglas:

    select-v3  -> ``reii.method_selection.select_method`` (codigo real, v3)
    select-v4  -> ``select_v4_reference`` (espejo del diff de
                  ``docs/design_v5/SELECT_V4.md``)

El motor de estabilidad (bootstrap) y de contraste contra el nulo
(permutacion de etiquetas) se implementa aqui como REFERENCIA; el documento
fija las mismas firmas para ``reii/quality/metrics.py``. Los ARI y las
silhouettes se calculan con ``reii.quality.metrics`` (modulo liviano: no
importa Stanza ni ``reii.main_workflow``).

POR QUE WARD PARA C
--------------------
``ward`` es el peor caso: no recibe el piso, su pase tight/loose es identico y
``ARI(C1,C2) == 1.0`` por construccion, incluso sobre ruido. Si la regla
sobrevive a ``ward``, sobrevive a ``dense_cdh`` (que ademas tiene el gate
interno ``r2_dense >= 0.08``).

QUE NO HACE
-----------
No usa red, Docker, la BD, ni escribe en ``data/``, ``output/`` ni ``models/``.
No importa ``reii.main_workflow``. Salida ASCII puro (consola cp1252).

Uso:
    python scripts/playground/select_v4_sim.py
    python scripts/playground/select_v4_sim.py --reps 6 --n 300
    python scripts/playground/select_v4_sim.py --skip-determinism
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ── Regla vigente (v3), del codigo real ────────────────────────────────────
from reii.method_selection import MethodMetrics, select_method  # noqa: E402

# ── Metricas livianas del proyecto (ARI y silhouette) ──────────────────────
from reii.quality.metrics import (  # noqa: E402
    binary_term_matrix,
    chi2_profiles,
    cross_partition_agreement,
    prune_vocabulary,
    silhouette_labels,
)

SEED = 42
NAN = float("nan")


# ══════════════════════════════════════════════════════════════════════════
# Metrica v4 (espejo del diff). Los campos nuevos llevan default.
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class M4:
    method: str
    k: int
    ari: float  # perturbacion VALIDA del backend (v4 la usa)
    coverage: float
    noise_frac: float = 0.0
    tiny_frac: float = 0.0
    perturbation: str = ""
    available: bool = True
    assigned_frac: Optional[float] = None
    min_class_size: int = 0
    median_class_size: float = 0.0
    max_class_share: float = 0.0
    # ── v4 ──────────────────────────────────────────────────────────────
    backend: str = ""
    diagnostic_only: bool = False
    ari_source: str = "two_pass"
    #: estabilidad del pase tight/loose, solo diagnostico cuando ari_source
    #: es bootstrap (en densos vale 1.0 por construccion).
    ari_two_pass: float = NAN
    evidence_p: float = NAN
    evidence_z: float = NAN
    evidence_stat: str = ""
    bootstrap_ari_mean: float = NAN
    bootstrap_ari_median: float = NAN
    bootstrap_ari_p05: float = NAN
    bootstrap_ari_p95: float = NAN
    coverage_bootstrap: float = NAN
    n_boot: float = 0.0
    n_null: float = 0.0

    @property
    def usable_coverage(self) -> float:
        return max(0.0, float(self.coverage) - float(self.tiny_frac))

    @property
    def assign_coverage(self) -> float:
        return max(0.0, 1.0 - float(self.noise_frac))

    @property
    def dominance_corpus(self) -> float:
        return float(self.max_class_share) * self.assign_coverage

    def to_v3(self) -> MethodMetrics:
        """Lo que ve ``select-v3``: para densos, el ARI tautologico 1.0."""
        ari = self.ari_two_pass if np.isfinite(self.ari_two_pass) else self.ari
        return MethodMetrics(
            method=self.method,
            k=self.k,
            ari=float(ari),
            coverage=float(self.coverage),
            noise_frac=float(self.noise_frac),
            tiny_frac=float(self.tiny_frac),
            perturbation=self.perturbation,
            available=bool(self.available),
            assigned_frac=self.assigned_frac,
            min_class_size=int(self.min_class_size),
            median_class_size=float(self.median_class_size),
            max_class_share=float(self.max_class_share),
        )


# ══════════════════════════════════════════════════════════════════════════
# Motor de referencia (las firmas que el diff de SELECT_V4.md fija)
# ══════════════════════════════════════════════════════════════════════════
def _ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """ARI entre dos arrays alineados, ignorando < 0 (via modulo del proyecto)."""
    da = {f"u{i}": int(v) for i, v in enumerate(labels_a) if int(v) >= 0}
    db = {f"u{i}": int(v) for i, v in enumerate(labels_b) if int(v) >= 0}
    return float(cross_partition_agreement(da, db)["ari"])


def _align(ref: np.ndarray, other: np.ndarray, k: int) -> np.ndarray:
    """Reetiqueta ``other`` para que sus ids coincidan con ``ref`` (Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    m = np.zeros((k, k), dtype=int)
    for a, b in zip(ref, other):
        if a >= 0 and b >= 0 and a < k and b < k:
            m[int(a), int(b)] += 1
    rows, cols = linear_sum_assignment(-m)
    mapping = {int(c): int(r) for r, c in zip(rows, cols)}
    return np.array(
        [mapping.get(int(v), -1) if v >= 0 else -1 for v in other], dtype=int
    )


def bootstrap_stability(
    matrix: np.ndarray,
    labels: np.ndarray,
    refit,
    k: int,
    *,
    n_boot: int = 100,
    subsample_frac: float = 0.80,
    seed: int = SEED,
) -> Dict[str, float]:
    """Estabilidad por submuestreo de UCEs (perturbacion real, no tautologica).

    Reajusta el backend sobre ``subsample_frac * n`` UCEs SIN reemplazo, B veces,
    y mide ARI entre la particion de referencia y la reajustada en las UCEs
    comunes. Agrega por media y mediana; el IC es percentil (p05, p95) sobre las
    B replicas, no un IC analitico.

    Devuelve ``ari_mean``, ``ari_median``, ``ari_p05``, ``ari_p95``,
    ``consistency`` (fraccion de UCEs cuya etiqueta coincide con la de
    referencia en >= 50 % de las replicas en que aparece) y ``n_boot``.
    """
    rng = np.random.default_rng(seed)
    n = int(len(labels))
    m = max(int(round(subsample_frac * n)), k + 1)
    n_ids = int(max(k, labels.max() + 1))
    vote = np.zeros((n, n_ids), dtype=float)
    present = np.zeros(n, dtype=float)
    aris: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.choice(n, size=m, replace=False)
        lab_b = np.asarray(refit(matrix[idx], k), dtype=int)
        if lab_b.shape[0] != m:
            continue
        aligned = _align(labels[idx], lab_b, n_ids)
        aris.append(_ari(labels[idx], lab_b))
        present[idx] += 1.0
        ok = aligned >= 0
        vote[idx[ok], aligned[ok]] += 1.0
    if not aris:
        return {
            "ari_mean": NAN,
            "ari_median": NAN,
            "ari_p05": NAN,
            "ari_p95": NAN,
            "consistency": NAN,
            "n_boot": 0.0,
        }
    arr = np.asarray(aris, float)
    seen = present > 0
    frac = vote[seen].max(axis=1) / np.maximum(present[seen], 1.0)
    return {
        "ari_mean": float(arr.mean()),
        "ari_median": float(np.median(arr)),
        "ari_p05": float(np.percentile(arr, 5)),
        "ari_p95": float(np.percentile(arr, 95)),
        "consistency": float((frac >= 0.5).mean()) if seen.any() else NAN,
        "n_boot": float(len(arr)),
    }


def permutation_null(
    matrix,
    labels: Sequence[int],
    *,
    metric: str = "euclidean",
    statistic: str = "silhouette",
    n_perm: int = 40,
    sample_size: Optional[int] = 200,
    seed: int = SEED,
) -> Dict[str, float]:
    """Contraste contra un nulo por permutacion de etiquetas.

    Permuta las etiquetas entre las UCEs ASIGNADAS conservando el tamano de cada
    clase (y la mascara de ruido). El estadistico es invariante a la
    reetiquetacion, asi que solo cambia la geometria relativa clase/punto.

    ``p_value`` = (1 + #{nulo >= obs}) / (1 + n_perm); monotono y nunca 0.

    ADVERTENCIA: ``r2_dense`` NO admite este nulo (no es funcion de las
    etiquetas: permutarlas lo deja invariante). ``dense_cdh`` usa el gate
    interno fijo de BACKEND_C.md y, ademas, esta silhouette.
    """
    from sklearn.metrics import silhouette_score

    y = np.asarray(labels, dtype=int)
    keep = y >= 0
    if keep.sum() < 3 or len(np.unique(y[keep])) < 2:
        return {
            "stat": NAN,
            "null_mean": NAN,
            "null_sd": NAN,
            "z": NAN,
            "p_value": NAN,
            "n_perm": 0.0,
        }
    X = np.asarray(matrix)[keep]
    y_keep = y[keep]
    rng = np.random.default_rng(seed)
    if sample_size and len(y_keep) > sample_size:
        sel = rng.choice(len(y_keep), size=int(sample_size), replace=False)
        Xs, ys = X[sel], y_keep[sel]
    else:
        Xs, ys = X, y_keep
    obs = silhouette_labels(Xs, ys, metric=metric, sample_size=None)
    if obs is None:
        return {
            "stat": NAN,
            "null_mean": NAN,
            "null_sd": NAN,
            "z": NAN,
            "p_value": NAN,
            "n_perm": 0.0,
        }
    null = np.empty(int(n_perm), dtype=float)
    for b in range(int(n_perm)):
        yp = rng.permutation(ys)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            null[b] = float(silhouette_score(Xs, yp, metric=metric))
    mu = float(null.mean())
    sd = float(null.std(ddof=1)) if len(null) > 1 else NAN
    z = float((obs - mu) / sd) if sd and sd > 0 else NAN
    p = float((1.0 + float((null >= obs).sum())) / (1.0 + len(null)))
    return {
        "stat": float(obs),
        "null_mean": mu,
        "null_sd": sd,
        "z": z,
        "p_value": p,
        "n_perm": float(len(null)),
    }


def data_null_evidence(
    matrix,
    labels: Sequence[int],
    refit,
    k: int,
    *,
    metric: str = "euclidean",
    n_null: int = 20,
    sample_size: Optional[int] = 200,
    seed: int = SEED,
    label_perm_n: int = 0,
) -> Dict[str, float]:
    """Contraste contra un nulo de DATOS (no de etiquetas).

    Por que no sirve permutar etiquetas: ``ward`` y ``kmeans`` MINIMIZAN la
    dispersion intra-clase, asi que su silhouette es positiva por construccion
    frente a etiquetas aleatorias. Medido: sobre ruido puro, la permutacion de
    etiquetas da ``z ~ 8`` y ``p = 1/(1+n_perm)`` (falso positivo). Es el mismo
    defecto que ``r2_1d`` en BACKEND_C.md:6: el estadistico se optimiza.

    El nulo correcto fija el ALGORITMO y rompe la ESTRUCTURA: se reajusta el
    mismo backend, con el mismo K, sobre datos sin estructura con la misma
    media y covarianza. Es la logica de *gap statistic* adaptada a silhouette.

    ``p_value`` = (1 + #{nulo >= obs}) / (1 + n_null). Con ``n_null = 20`` el
    minimo es 1/21 = 0.048, asi que ``alpha = 0.05`` exige que el observado
    supere a TODOS los nulos.
    """
    X = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=int)
    obs = silhouette_labels(X, y, metric=metric, sample_size=sample_size)
    out = {
        "stat": NAN,
        "null_mean": NAN,
        "null_sd": NAN,
        "z": NAN,
        "p_value": NAN,
        "n_null": 0.0,
        "lab_perm_p": NAN,
        "lab_perm_z": NAN,
    }
    if obs is None or X.shape[0] < 3:
        return out
    mu = X.mean(axis=0)
    d = int(X.shape[1])
    cov = np.atleast_2d(np.cov(X, rowvar=False))
    cov = cov + 1e-9 * np.eye(d)
    rng = np.random.default_rng(seed)
    null: List[float] = []
    for _ in range(int(n_null)):
        Xn = rng.multivariate_normal(mu, cov, size=X.shape[0])
        lab_n = np.asarray(refit(Xn, k), dtype=int)
        s = silhouette_labels(Xn, lab_n, metric=metric, sample_size=sample_size)
        if s is not None:
            null.append(float(s))
    if not null:
        return out
    arr = np.asarray(null, float)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else NAN
    out.update(
        {
            "stat": float(obs),
            "null_mean": float(arr.mean()),
            "null_sd": sd,
            "z": float((obs - arr.mean()) / sd) if sd and sd > 0 else NAN,
            "p_value": float((1.0 + float((arr >= obs).sum())) / (1.0 + len(arr))),
            "n_null": float(len(arr)),
        }
    )
    if label_perm_n > 0:
        lp = permutation_null(
            X, y, metric=metric, n_perm=label_perm_n, sample_size=sample_size, seed=seed
        )
        out["lab_perm_p"] = float(lp["p_value"])
        out["lab_perm_z"] = float(lp["z"])
    return out


# ══════════════════════════════════════════════════════════════════════════
# Corpus sinteticos con estructura plantada
# ══════════════════════════════════════════════════════════════════════════
def _bag(rng, words: Sequence[str], size: int) -> List[str]:
    return [
        str(w)
        for w in rng.choice(list(words), size=min(size, len(words)), replace=False)
    ]


def make_corpus(scenario: str, n: int, seed: int) -> Dict:
    """Devuelve ``terms`` (lexico), ``X`` (espacio vectorial) y ``k``."""
    rng = np.random.default_rng(seed)
    d = 20

    if scenario == "lexical":
        k = 5
        block = [[f"c{c}w{j}" for j in range(28)] for c in range(k)]
        shared = [f"sh{j}" for j in range(10)]
        allw = [w for b in block for w in b]
        lab = rng.integers(0, k, size=n)
        n_noise = int(round(0.08 * n))
        noisy = set(rng.choice(n, size=n_noise, replace=False).tolist())
        terms = []
        for i in range(n):
            src = allw if i in noisy else block[int(lab[i])]
            terms.append(_bag(rng, src, 10) + _bag(rng, shared, 3))
        X = rng.normal(0.0, 1.0, size=(n, d))
        return {"terms": terms, "X": X, "k": k, "name": scenario}

    if scenario == "dense":
        k = 4
        centers = rng.normal(0.0, 2.8, size=(k, d))
        w = np.linspace(1.0, 0.55, k)
        w = w / w.sum()
        n_cl = n - int(0.10 * n)
        sizes = np.maximum(15, np.round(w * n_cl).astype(int))
        sizes[-1] += n_cl - int(sizes.sum())
        lab = np.concatenate([np.full(int(s), i) for i, s in enumerate(sizes)])
        Xc = centers[lab] + rng.normal(0.0, 0.9, size=(len(lab), d))
        nn = n - len(lab)
        X = np.vstack([Xc, rng.uniform(-8.0, 8.0, size=(nn, d))])[rng.permutation(n)]
        vocab = [f"v{j}" for j in range(200)]
        terms = [_bag(rng, vocab, 10) for _ in range(n)]
        return {"terms": terms, "X": X, "k": k, "name": scenario}

    if scenario == "mixed":
        k = 4
        block = [[f"m{c}w{j}" for j in range(24)] for c in range(k)]
        shared = [f"mx{j}" for j in range(8)]
        lab = rng.integers(0, k, size=n)
        terms = [
            _bag(rng, block[int(lab[i])], 9) + _bag(rng, shared, 3) for i in range(n)
        ]
        centers = rng.normal(0.0, 1.3, size=(k, d))
        X = centers[lab] + rng.normal(0.0, 0.9, size=(n, d))
        return {"terms": terms, "X": X, "k": k, "name": scenario}

    # null
    k = 4
    vocab = [f"n{j}" for j in range(200)]
    terms = [_bag(rng, vocab, 10) for _ in range(n)]
    X = rng.normal(0.0, 1.0, size=(n, d))
    return {"terms": terms, "X": X, "k": k, "name": "null"}


# ══════════════════════════════════════════════════════════════════════════
# Particiones A/B (lexicas) y C (densa)
# ══════════════════════════════════════════════════════════════════════════
def _kmeans(X: np.ndarray, k: int, seed: int, n_init: int = 10) -> np.ndarray:
    """KMeans determinista. ``n_init=3`` solo para los reajustes del nulo
    (ahorran ~60 % del runtime y no cambian la conclusion: el nulo solo
    necesita *una* particion plausible de datos sin estructura)."""
    from sklearn.cluster import KMeans

    return (
        KMeans(n_clusters=int(k), n_init=int(n_init), random_state=seed)
        .fit_predict(X)
        .astype(int)
    )


def lexical_space(terms: List[List[str]], tight: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Perfiles chi2 (A) y vectores L2-normalizados (B) del lexico.

    ``tight`` filtra a los tipos con frecuencia documental >= cuantil 0.70 del
    vocabulario retenido. Es la perturbacion de A/B (ventana/umbral lexico):
    cambia la geometria, no el numero de UCEs.
    """
    vocab = prune_vocabulary(terms, tsj=2, min_abs=2)
    if not vocab:
        z = np.zeros((len(terms), 1))
        return z, z
    if tight:
        df = {w: 0 for w in vocab}
        for t in terms:
            for w in set(t):
                if w in df:
                    df[w] += 1
        thr = float(np.quantile(list(df.values()), 0.70))
        kept = [w for w in vocab if df[w] >= thr]
        vocab = kept or vocab
    M = binary_term_matrix(terms, vocab)
    Z, keep = chi2_profiles(M)
    raw = np.asarray(M.todense(), dtype=float)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    cos = raw / norms
    if Z is None or int(keep.sum()) != len(terms):
        # filas sin masa: se cae al espacio coseno para no romper el index 1:1
        return raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12), raw / norms
    return Z.toarray(), raw / norms


def ab_partitions(terms: List[List[str]], k: int, seed: int) -> Dict[str, np.ndarray]:
    """A1/A2 (chi2) y B1/B2 (coseno), tight vs loose por umbral de vocabulario."""
    Z_loose, C_loose = lexical_space(terms, tight=False)
    Z_tight, C_tight = lexical_space(terms, tight=True)
    return {
        "A1": _kmeans(np.asarray(Z_loose, float), k, seed + 1),
        "A2": _kmeans(np.asarray(Z_tight, float), k, seed + 2),
        "B1": _kmeans(np.asarray(C_loose, float), k, seed + 3),
        "B2": _kmeans(np.asarray(C_tight, float), k, seed + 4),
    }


def ward_fit(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    return (
        AgglomerativeClustering(n_clusters=int(k), linkage="ward")
        .fit_predict(X)
        .astype(int)
    )


def _class_stats(lab: np.ndarray) -> Tuple[float, float, int]:
    """(max_class_share, tiny_frac, K) de una particion cruda."""
    a = lab[lab >= 0]
    if a.size == 0:
        return 0.0, 0.0, 0
    sizes = np.bincount(a).astype(float)
    sizes = sizes[sizes > 0]
    n = len(lab)
    tiny = float(sum(s for s in sizes if s < max(2, int(0.01 * n))) / n)
    return float(sizes.max() / a.size), tiny, int(len(sizes))


def build_metrics(
    scenario: Dict, *, n_null: int, n_boot: int, seed: int
) -> Tuple[List[M4], Dict[str, Dict]]:
    """Metricas por metodo + diagnosticos crudos (para las tablas del informe)."""
    terms, X, k = scenario["terms"], scenario["X"], int(scenario["k"])
    labs = ab_partitions(terms, k, seed)

    out: List[M4] = []
    diag: Dict[str, Dict] = {}

    for m, l1, l2, space in (
        ("A", labs["A1"], labs["A2"], "chi2"),
        ("B", labs["B1"], labs["B2"], "cosine"),
    ):
        ari = _ari(l1, l2)
        # Las etiquetas de KMeans son ids ARBITRARIOS: comparar crudo da ~0.
        # El pipeline alinea por Hungarian (_hungarian_stability_direct) antes
        # de contar "misma etiqueta en tight y loose"; se replica aqui.
        kk_lab = int(max(l1.max(), l2.max()) + 1)
        aligned = _align(l1, l2, kk_lab)
        agree = (l1 == aligned) & (l1 >= 0)
        share, tiny, kk = _class_stats(l1)
        out.append(
            M4(
                method=m,
                k=kk,
                ari=float(ari),
                ari_two_pass=float(ari),
                coverage=float(agree.mean()),
                tiny_frac=tiny,
                perturbation=f"{m}1<->{m}2",
                max_class_share=share,
                assigned_frac=1.0,
                ari_source="two_pass",
            )
        )
        diag[m] = {"ari_two_pass": ari, "labels": l1, "space": space}

    # C: ward. C1 y C2 son la MISMA corrida (el backend no recibe el piso).
    c1 = ward_fit(X, k)
    share, tiny, kk = _class_stats(c1)
    boot = bootstrap_stability(
        X, c1, ward_fit, k, n_boot=n_boot, subsample_frac=0.80, seed=seed + 11
    )
    perm = data_null_evidence(
        X,
        c1,
        ward_fit,
        k,
        metric="euclidean",
        n_null=n_null,
        sample_size=200,
        seed=seed + 12,
        label_perm_n=40,
    )
    out.append(
        M4(
            method="C",
            k=kk,
            # v4: `ari` es la estabilidad bajo la perturbacion VALIDA (bootstrap).
            ari=float(boot["ari_mean"]),
            ari_two_pass=1.0,  # lo que ve v3: identidad por construccion
            coverage=1.0,  # tight/loose identico: 1.0 tautologico
            tiny_frac=tiny,
            perturbation="bootstrap(80%)",
            max_class_share=share,
            assigned_frac=1.0,
            backend="ward",
            ari_source="bootstrap",
            evidence_p=float(perm["p_value"]),
            evidence_z=float(perm["z"]),
            evidence_stat="silhouette",
            bootstrap_ari_mean=float(boot["ari_mean"]),
            bootstrap_ari_median=float(boot["ari_median"]),
            bootstrap_ari_p05=float(boot["ari_p05"]),
            bootstrap_ari_p95=float(boot["ari_p95"]),
            coverage_bootstrap=float(boot["consistency"]),
            n_boot=float(boot["n_boot"]),
            n_null=float(perm["n_null"]),
        )
    )
    diag["C"] = {
        "ari_two_pass": 1.0,
        "ari_bootstrap": boot["ari_mean"],
        "coverage_bootstrap": boot["consistency"],
        "evidence_p": perm["p_value"],
        "evidence_z": perm["z"],
        "silhouette": perm["stat"],
        "null_mean": perm["null_mean"],
        "labels": c1,
        "space": "euclidean",
        "lab_perm_p": perm["lab_perm_p"],
        "lab_perm_z": perm["lab_perm_z"],
        "null_sd": perm["null_sd"],
    }
    return out, diag


def add_evidence_ab(
    metrics: List[M4], diag: Dict[str, Dict], scenario: Dict, *, n_null: int, seed: int
) -> List[M4]:
    """Evidencia de A/B en SU espacio lexico, con el MISMO backend (kmeans)."""
    Zl, Cl = lexical_space(scenario["terms"], tight=False)
    spaces = {"A": np.asarray(Zl, float), "B": np.asarray(Cl, float)}
    out: List[M4] = []
    for m in metrics:
        if m.method in ("A", "B"):
            k = int(m.k)
            perm = data_null_evidence(
                spaces[m.method],
                diag[m.method]["labels"],
                lambda Xn, kk: _kmeans(Xn, kk, seed + 7, n_init=3),
                k,
                metric="euclidean",
                n_null=n_null,
                sample_size=200,
                seed=seed + 100 + (0 if m.method == "A" else 1),
            )
            m = replace(
                m,
                evidence_p=float(perm["p_value"]),
                evidence_z=float(perm["z"]),
                evidence_stat="silhouette",
                n_null=float(perm["n_null"]),
            )
            diag[m.method].update(
                {
                    "evidence_p": perm["p_value"],
                    "evidence_z": perm["z"],
                    "silhouette": perm["stat"],
                    "null_mean": perm["null_mean"],
                    "null_sd": perm["null_sd"],
                }
            )
        out.append(m)
    return out


# ══════════════════════════════════════════════════════════════════════════
# select-v4 (referencia) — espejo del diff de docs/design_v5/SELECT_V4.md
# ══════════════════════════════════════════════════════════════════════════
ORDER = {"A": 0, "B": 1, "C": 2}


def _v4_feasible(
    m: M4,
    *,
    c0: float,
    noise_max: float,
    coverage_floor: float,
    k_bounds: Tuple[int, int],
    z_min: float,
    stability_floor: Dict[str, float],
) -> Optional[str]:
    if not m.available:
        return "no produjo particion"
    if m.diagnostic_only:
        return f"backend de diagnostico {m.backend or '?'} (no seleccionable)"
    if not np.isfinite(float(m.ari)):
        return "estabilidad no finita"
    lo, hi = k_bounds
    if not (lo <= int(m.k) <= hi):
        return f"K={int(m.k)} fuera de [{lo}, {hi}]"
    if float(m.noise_frac) > noise_max:
        return f"ruido {float(m.noise_frac):.3f} > max {noise_max:.2f}"
    c = m.dominance_corpus
    if c > c0:
        return (
            f"clase dominante {float(m.max_class_share):.2f} = {c:.2f} del corpus"
            f" > {c0:.2f}"
        )
    if m.usable_coverage < coverage_floor:
        return f"coverage util {m.usable_coverage:.3f} < {coverage_floor:.2f}"
    if not np.isfinite(float(m.evidence_z)):
        return "sin contraste contra el nulo (evidencia no calculada)"
    if float(m.evidence_z) < z_min:
        return (
            f"sin evidencia contra el nulo: z={float(m.evidence_z):.2f} < "
            f"{z_min:.1f} (p={float(m.evidence_p):.3f})"
        )
    floor = float(stability_floor.get(str(m.ari_source), 0.0))
    if float(m.ari) < floor:
        return f"estabilidad {float(m.ari):.3f} < piso {floor:.2f} ({m.ari_source})"
    return None


def select_v4_reference(
    metrics: Sequence[M4],
    *,
    c0: float = 0.70,
    noise_max: float = 0.50,
    coverage_floor: float = 0.15,
    k_bounds: Tuple[int, int] = (3, 20),
    z_min: float = 3.0,
    margin_z: float = 1.0,
    stability_floor: Optional[Dict[str, float]] = None,
) -> Dict:
    """Regla select-v4: gates -> ranking (-z, orden) -> banderas.

    ``coverage`` NO participa del ranking: en los backends densos vale 1.0 por
    construccion (asignan el 100 % de las UCEs), asi que premiarla equivale a
    premiar a C siempre. Queda como **gate** (``coverage_floor``) y como
    **bandera** (``criteria_conflict``).

    ``ORDER`` (A<B<C) solo decide empates *exactos* de ``z``, que en la practica
    no ocurren (``z`` es continuo); es una preferencia pre-registrada por el
    metodo que replica ALCESTE, no una medida.
    """
    floors = dict(stability_floor or {"two_pass": 0.15, "bootstrap": 0.35})
    viable: List[M4] = []
    discarded: List[Dict] = []
    for m in metrics:
        why = _v4_feasible(
            m,
            c0=c0,
            noise_max=noise_max,
            coverage_floor=coverage_floor,
            k_bounds=k_bounds,
            z_min=z_min,
            stability_floor=floors,
        )
        if why is None:
            viable.append(m)
        else:
            discarded.append({"method": m.method, "why": why})
    if not viable:
        return {
            "winner": None,
            "winner_z": NAN,
            "runner_up": None,
            "runner_up_z": NAN,
            "gap_z": NAN,
            "decided_by": "abstencion",
            "discarded": discarded,
            "ambiguous": False,
            "criteria_conflict": False,
            "abstained": True,
            "reason": "ningun metodo cumple viabilidad",
        }
    ranked = sorted(
        viable,
        key=lambda m: (
            -float(m.evidence_z),
            ORDER.get(m.method, 99),
        ),
    )
    win = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    gap = abs(float(win.evidence_z) - float(runner.evidence_z)) if runner else NAN
    ambiguous = bool(runner is not None and gap <= margin_z)
    conflict = bool(runner is not None and runner.usable_coverage > win.usable_coverage)
    return {
        "winner": win.method,
        "winner_z": float(win.evidence_z),
        "runner_up": runner.method if runner else None,
        "runner_up_z": float(runner.evidence_z) if runner else NAN,
        "gap_z": float(gap) if runner else NAN,
        "decided_by": ("unico" if runner is None else ("z" if gap > 1e-9 else "orden")),
        "discarded": discarded,
        "ambiguous": ambiguous,
        "criteria_conflict": conflict,
        "abstained": False,
        "reason": "exceso estandarizado sobre el nulo de datos",
    }


# ══════════════════════════════════════════════════════════════════════════
# Ejecucion
# ══════════════════════════════════════════════════════════════════════════
EXPECTED = {
    "lexical": {"A", "B"},
    "dense": {"C"},
    "null": {None},
    "mixed": {"A", "B"},
}


def _v3(metrics: List[M4]) -> Dict:
    sel = select_method(
        [m.to_v3() for m in metrics],
        margin=0.02,
        coverage_floor=0.15,
        k_bounds=(3, 20),
        min_class_share=0.01,
        noise_max=0.50,
        max_class_share_max=0.90,
    )
    return {"winner": sel.winner, "abstained": sel.abstained}


def run(reps: int, n: int, n_null: int, n_boot: int, *, verbose: bool) -> Dict:
    rows: Dict[str, Dict] = {}
    calib: Dict[str, List[float]] = {
        "C_null_boot": [],
        "C_lex_boot": [],
        "C_dense_boot": [],
        "z_C_null": [],
        "p_C_null": [],
        "lp_p_C_null": [],
        "lp_z_C_null": [],
        "z_A_lex": [],
        "z_B_lex": [],
        "z_C_lex": [],
        "z_C_dense": [],
        "z_A_dense": [],
        "z_A_null": [],
        "z_B_null": [],
        "z_A_mixed": [],
        "z_C_mixed": [],
        "ari_A_null": [],
        "ari_B_null": [],
        "ari_A_lex": [],
        "ari_B_lex": [],
        "cons_null": [],
        "cons_dense": [],
        "gap_mixed": [],
        "gap_all": [],
    }
    flags: Dict[str, int] = {"amb4": 0, "conf4": 0, "tie4": 0, "n4": 0}
    decisions: Dict[str, List] = {}
    for scenario in ("lexical", "dense", "null", "mixed"):
        acc3 = acc4 = cw3 = cw4 = abst4 = 0
        n_runs = 0
        dec: List[Tuple] = []
        for r in range(reps):
            seed = SEED + 1000 * (1 + r) + 17 * len(scenario)
            corpus = make_corpus(scenario, n, seed)
            metrics, diag = build_metrics(
                corpus, n_null=n_null, n_boot=n_boot, seed=seed
            )
            metrics = add_evidence_ab(metrics, diag, corpus, n_null=n_null, seed=seed)
            r3 = _v3(metrics)
            r4 = select_v4_reference(metrics)
            exp = EXPECTED[scenario]
            acc3 += int(r3["winner"] in exp)
            acc4 += int(r4["winner"] in exp)
            if "C" not in exp:
                cw3 += int(r3["winner"] == "C")
                cw4 += int(r4["winner"] == "C")
            abst4 += int(r4["winner"] is None)
            dec.append(
                (r3["winner"], r4["winner"], r4["ambiguous"], r4["criteria_conflict"])
            )
            flags["n4"] += 1
            flags["amb4"] += int(r4["ambiguous"])
            flags["conf4"] += int(r4["criteria_conflict"])
            flags["tie4"] += int(r4["decided_by"] == "orden")
            if np.isfinite(r4["gap_z"]):
                calib["gap_all"].append(float(r4["gap_z"]))
                if scenario == "mixed":
                    calib["gap_mixed"].append(float(r4["gap_z"]))
            n_runs += 1
            if scenario == "null":
                calib["C_null_boot"].append(diag["C"]["ari_bootstrap"])
                calib["z_C_null"].append(diag["C"]["evidence_z"])
                calib["p_C_null"].append(diag["C"]["evidence_p"])
                calib["lp_p_C_null"].append(diag["C"]["lab_perm_p"])
                calib["lp_z_C_null"].append(diag["C"]["lab_perm_z"])
                calib["cons_null"].append(diag["C"]["coverage_bootstrap"])
                calib["z_A_null"].append(diag["A"]["evidence_z"])
                calib["z_B_null"].append(diag["B"]["evidence_z"])
                calib["ari_A_null"].append(diag["A"]["ari_two_pass"])
                calib["ari_B_null"].append(diag["B"]["ari_two_pass"])
            if scenario == "lexical":
                calib["C_lex_boot"].append(diag["C"]["ari_bootstrap"])
                calib["z_A_lex"].append(diag["A"]["evidence_z"])
                calib["z_B_lex"].append(diag["B"]["evidence_z"])
                calib["z_C_lex"].append(diag["C"]["evidence_z"])
                calib["ari_A_lex"].append(diag["A"]["ari_two_pass"])
                calib["ari_B_lex"].append(diag["B"]["ari_two_pass"])
            if scenario == "dense":
                calib["C_dense_boot"].append(diag["C"]["ari_bootstrap"])
                calib["z_C_dense"].append(diag["C"]["evidence_z"])
                calib["z_A_dense"].append(diag["A"]["evidence_z"])
                calib["cons_dense"].append(diag["C"]["coverage_bootstrap"])
            if scenario == "mixed":
                calib["z_A_mixed"].append(diag["A"]["evidence_z"])
                calib["z_C_mixed"].append(diag["C"]["evidence_z"])
            if verbose and r == 0:
                print(
                    f"    {scenario:8} rep0: v3={str(r3['winner']):5} "
                    f"v4={str(r4['winner']):5} | C boot={diag['C']['ari_bootstrap']:.3f} "
                    f"p={diag['C']['evidence_p']:.3f} z={diag['C']['evidence_z']:6.2f} "
                    f"| lab-perm p={diag['C']['lab_perm_p']:.3f} "
                    f"z={diag['C']['lab_perm_z']:5.2f} cons={diag['C']['coverage_bootstrap']:.2f}"
                )
        rows[scenario] = {
            "acc3": acc3 / n_runs,
            "acc4": acc4 / n_runs,
            "cwin3": (cw3 / n_runs) if "C" not in EXPECTED[scenario] else NAN,
            "cwin4": (cw4 / n_runs) if "C" not in EXPECTED[scenario] else NAN,
            "abst4": abst4 / n_runs,
            "n": n_runs,
        }
        decisions[scenario] = dec
    return {"rows": rows, "calib": calib, "flags": flags, "decisions": decisions}


def case_probe() -> None:
    """Caso concreto del encargo: C denso, nu=0, silhouette POR DEBAJO del nulo.

    Dos variantes, porque la respuesta depende de QUE gate lo detiene:
      (a) a=0.837 tal como se midio -> lo descarta el gate de dominancia
          (c = 0.837 > 0.70), antes incluso de mirar la evidencia;
      (b) a=0.60 (dominancia tolerable) -> lo descarta la EVIDENCIA
          (z = -0.28 < 3), que es la pieza nueva.
    """
    a = M4(
        method="A",
        k=4,
        ari=0.31,
        ari_two_pass=0.31,
        coverage=0.40,
        max_class_share=0.34,
        ari_source="two_pass",
        evidence_p=0.004,
        evidence_z=12.4,
        evidence_stat="silhouette",
    )
    for label, share in (("a=0.837 (medido)", 0.837), ("a=0.600", 0.600)):
        c = M4(
            method="C",
            k=4,
            ari=0.62,
            ari_two_pass=1.0,
            coverage=1.0,
            perturbation="bootstrap(80%)",
            backend="dense_cdh",
            ari_source="bootstrap",
            max_class_share=share,
            assigned_frac=1.0,
            evidence_p=0.61,
            evidence_z=-0.28,
            evidence_stat="silhouette",
            bootstrap_ari_mean=0.62,
            n_null=20.0,
        )
        r4 = select_v4_reference([a, c])
        why_c = [d["why"] for d in r4["discarded"] if d["method"] == "C"]
        r3 = _v3([a, c])
        print(f"  [caso concreto] C denso {label}, nu=0, z=-0.28, p=0.61")
        print(
            f"    select-v3 => winner={r3['winner']} "
            f"(ARI 1.0 por construccion, coverage 1.0, a={share:.3f} < 0.90)"
        )
        print(
            f"    select-v4 => winner={r4['winner']}"
            f"{'  [descartado C] ' + why_c[0] if why_c else ''}"
        )

    # (c) C denso SIN dominancia y CON evidencia: debe ganar.
    #     Se usan los valores MEDIDOS en el escenario denso: A cae por debajo
    #     del nulo (z=-5.6) y C lo supera holgadamente (z=26).
    a_dense = M4(
        method="A",
        k=4,
        ari=0.06,
        ari_two_pass=0.06,
        coverage=0.18,
        max_class_share=0.29,
        ari_source="two_pass",
        evidence_p=0.99,
        evidence_z=-5.58,
        evidence_stat="silhouette",
    )
    c_ok = M4(
        method="C",
        k=4,
        ari=0.96,
        ari_two_pass=1.0,
        coverage=1.0,
        perturbation="bootstrap(80%)",
        backend="ward",
        ari_source="bootstrap",
        max_class_share=0.41,
        assigned_frac=1.0,
        evidence_p=0.0001,
        evidence_z=26.0,
        evidence_stat="silhouette",
        bootstrap_ari_mean=0.96,
        n_null=20.0,
    )
    r4 = select_v4_reference([a_dense, c_ok])
    print(
        f"  [control] C denso a=0.41 z=26.0 vs A z=-5.6 => select-v4 "
        f"winner={r4['winner']} (C debe ganar: la regla no es anti-C)"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=16)
    ap.add_argument(
        "--skip-determinism",
        action="store_true",
        help="omite la doble corrida de verificacion (ahorra ~20 s)",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    print(
        f"SELECT-V4 SIM | n={args.n} reps={args.reps} "
        f"n_null={args.n_null} n_boot={args.n_boot} seed={SEED}"
    )
    print("  ARI/silhouette: reii.quality.metrics | C = ward (peor caso tautologico)")
    print("")
    out = run(args.reps, args.n, args.n_null, args.n_boot, verbose=True)
    dt = time.perf_counter() - t0

    print("")
    print("  precision de seleccion (ganador dentro del conjunto esperado)")
    print(
        f"  {'escenario':10}{'esperado':10}{'v3':>7}{'v4':>7}"
        f"{'C-gana v3':>11}{'C-gana v4':>11}{'abst v4':>9}"
    )
    print("  " + "-" * 66)
    for scen, r in out["rows"].items():
        exp = ",".join(sorted(str(x) for x in EXPECTED[scen]))
        c3 = "        n/a" if not np.isfinite(r["cwin3"]) else f"{r['cwin3']:>11.2f}"
        c4 = "        n/a" if not np.isfinite(r["cwin4"]) else f"{r['cwin4']:>11.2f}"
        print(
            f"  {scen:10}{exp:10}{r['acc3']:>7.2f}{r['acc4']:>7.2f}"
            f"{c3}{c4}{r['abst4']:>9.2f}"
        )
    print("")

    cal = out["calib"]
    fl = out["flags"]
    if cal["C_null_boot"]:
        v = np.asarray(cal["C_null_boot"], float)
        sd = v.std(ddof=1) if len(v) > 1 else NAN
        print("  calibracion de C (=ward) sobre ruido puro:")
        print(
            f"    bootstrap ARI  media={v.mean():.3f} sd={sd:.3f} "
            f"p95={np.percentile(v, 95):.3f} max={v.max():.3f}"
        )
        print(f"    consistencia bootstrap media={np.mean(cal['cons_null']):.3f}")
        print(
            f"    nulo de DATOS   z mediana={np.nanmedian(cal['z_C_null']):6.2f} "
            f"p mediana={np.nanmedian(cal['p_C_null']):.3f} "
            f"frac(z>=3)={np.mean(np.asarray(cal['z_C_null']) >= 3.0):.2f}"
        )
        print(
            f"    nulo de ETIQUETAS z mediana={np.nanmedian(cal['lp_z_C_null']):6.2f} "
            f"p mediana={np.nanmedian(cal['lp_p_C_null']):.3f} "
            f"frac(p<=0.05)={np.mean(np.asarray(cal['lp_p_C_null']) <= 0.05):.2f}"
            "  <== FALSO POSITIVO: no se usa para decidir"
        )
    print("  evidencia (z sobre el nulo de datos) por escenario:")
    for name, keys in (
        ("lexico ", ("z_A_lex", "z_B_lex", "z_C_lex")),
        ("denso  ", ("z_A_dense", "z_C_dense")),
        ("ruido  ", ("z_A_null", "z_B_null", "z_C_null")),
        ("mixto  ", ("z_A_mixed", "z_C_mixed")),
    ):
        parts = []
        for k in keys:
            if cal[k]:
                parts.append(f"{k[2]}={np.nanmedian(cal[k]):6.2f}")
        if parts:
            print(f"    {name} " + "  ".join(parts))
    if cal["C_dense_boot"] and cal["C_lex_boot"]:
        print("  bootstrap ARI de C (media por escenario):")
        print(
            f"    ruido={np.nanmean(cal['C_null_boot']):.3f}  "
            f"lexico={np.nanmean(cal['C_lex_boot']):.3f}  "
            f"denso={np.nanmean(cal['C_dense_boot']):.3f}"
        )
    if cal["ari_A_lex"] and cal["ari_A_null"]:
        print("  ARI two_pass de A/B (para calibrar el piso de A/B):")
        print(
            f"    A: lexico={np.nanmean(cal['ari_A_lex']):.3f} "
            f"ruido={np.nanmean(cal['ari_A_null']):.3f}  |  "
            f"B: lexico={np.nanmean(cal['ari_B_lex']):.3f} "
            f"ruido={np.nanmean(cal['ari_B_null']):.3f}"
        )
    if cal["gap_mixed"]:
        print(
            f"  margen |z_win - z_run| en mixto: mediana={np.nanmedian(cal['gap_mixed']):.2f} "
            f"min={np.nanmin(cal['gap_mixed']):.2f}"
        )
    if fl["n4"]:
        print(
            f"  banderas v4: ambiguous={fl['amb4']}/{fl['n4']} "
            f"criteria_conflict={fl['conf4']}/{fl['n4']} "
            f"desempate por orden={fl['tie4']}/{fl['n4']}"
        )
    print("")
    case_probe()
    print("")
    print(f"  runtime: {dt:.1f} s")

    if not args.skip_determinism:
        # Determinismo EXACTO: misma configuracion de 1 repeticion, dos veces.
        # Los seeds dependen de (r, escenario), no de reps, asi que la corrida
        # de 1 rep debe reproducir la decision de la rep 0 de la corrida grande.
        t1 = time.perf_counter()
        a = run(1, args.n, args.n_null, args.n_boot, verbose=False)
        b = run(1, args.n, args.n_null, args.n_boot, verbose=False)
        same_ab = a["decisions"] == b["decisions"]
        same_main = all(
            a["decisions"][scen][0] == out["decisions"][scen][0]
            for scen in out["decisions"]
        )
        ok = bool(same_ab and same_main)
        print(
            f"  determinismo (2 corridas identicas de 1 rep + rep0 de la grande): "
            f"{'[ok]' if ok else '[X]'} ({time.perf_counter() - t1:.1f} s)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
