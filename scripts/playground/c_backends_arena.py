#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Arena de backends para el paso "clasificacion densa" de Method C (REII).

Compara los cuatro backends candidatos sobre la MISMA matriz PCA(30):

    hdbscan          -> HDBSCAN plano (el Method C actual; solo diagnostico)
    dense_cdh        -> biseccion divisiva best-first por PC1 local + corte 1-D
    ward             -> AgglomerativeClustering(n_clusters=K, linkage="ward")
    kmeans_spherical -> L2-normalizar + KMeans(n_clusters=K, n_init=10, seed=42)

Por (backend, K) reporta silhouette, ARI entre K y K+2, CV de tamanos de clase,
max_class_share, n_noise, runtime y verificacion de determinismo (dos corridas).

POR QUE NO IMPORTA `reii.main_workflow`
---------------------------------------
Importar main_workflow arrastra Stanza y tarda >4 min (cuello de botella medido).
Este script replica la matematica del backend en funciones puras:
  * `corte_optimo_1d` es una COPIA LITERAL de `ClasificadorDescendente._corte_optimo`
    (mismo argsort, mismo SS_between, misma restriccion min_side).
  * el resto (PC1 local, refinamiento por centroide, PCA(30), KMeans, Ward) usa
    los mismos estimadores de sklearn y los mismos hiperparametros.
La equivalencia es de *especificacion*, no de importacion: los numeros de este
script son la medicion real sobre datos reales, pero el codigo de produccion es
el diff que acompana a docs/design_v5/BACKEND_C.md.

DATOS REALES (solo lectura)
---------------------------
`data/workflow_data.json` corresponde a un bench con uce_target=26 (600 UCEs).
Hechos verificados en ese artefacto:
  * solo 235 de 600 UCEs tienen `embedding` (son las `is_stable=True`); las otras
    365 tienen `embedding=None` y se excluyen.
  * `embedding` es 384-d (SentenceTransformer, con contexto). Method C en
    produccion usa fastText-retrofit 300-d (centroide ponderado por SUBTLEX-CD).
    => MISMO corpus, ESPACIO DISTINTO. Silhouettes/ARIs son indicativos y NO
    comparables 1:1 con la corrida de produccion.
  * `retrofitted_vectors_significant` esta vacio en este artefacto, asi que NO se
    puede reconstruir el espacio fastText sin cargar los modelos (prohibido aqui).
  * `cluster_id` (7 clases, todas las 235 asignadas) se usa solo como particion
    de referencia, nunca como entrada de ningun backend.

Uso:
    python scripts/playground/c_backends_arena.py
    python scripts/playground/c_backends_arena.py --ks 3,5,8 --datasets real_ctx
Salida ASCII puro (consola cp1252), sin red, sin docker, sin escrituras.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

# Un solo hilo ANTES de importar numpy: en n=600, d=30 el overhead de OpenMP/
# OpenBLAS domina (KMeans medido: 800 ms con N hilos vs 128 ms con 1). Los
# backends no se benefician de paralelismo a este tamano de matriz.
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
DATA_PATH = os.path.join(REPO_ROOT, "data", "workflow_data.json")
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

SEED = 42
PCA_DIMS = 30
# Gate r2_dense de produccion (calibrado en la arena: ruido 0.036-0.064 vs
# estructura real 0.10-0.24). En la arena se deja APAGADO por defecto para
# medir el control de K sin confundirlo con el gate; el barrido [gate] lo mide.
DENSE_MIN_R2_DENSE = 0.08
DENSE_MIN_R2_DENSE_ARENA = 0.0


# ══════════════════════════════════════════════════════════════════════════
# Metricas: se reutilizan las del proyecto (no se reimplementan)
# ══════════════════════════════════════════════════════════════════════════


def load_project_metrics():
    """Devuelve (silhouette_labels, cross_partition_agreement, reutilizado)."""
    try:
        from reii.quality.metrics import (  # import liviano (~15 s, sin Stanza)
            cross_partition_agreement,
            silhouette_labels,
        )

        return silhouette_labels, cross_partition_agreement, True
    except Exception as exc:  # pragma: no cover - solo si el entorno no tiene reii
        print(f"[!] no se pudo importar reii.quality.metrics: {exc}")
        print("[!] fallback a sklearn.metrics (mismas formulas, sin el wrapper)")

        from sklearn.metrics import adjusted_rand_score, silhouette_score

        def _sil(matrix, labels, metric="euclidean", sample_size=None, seed=42):
            y = np.asarray(labels).astype(int)
            keep = y >= 0
            X = np.asarray(matrix)[keep]
            y = y[keep]
            if len(y) < 3 or len(np.unique(y)) < 2:
                return None
            return float(silhouette_score(X, y, metric=metric))

        def _agree(labels_a, labels_b):
            common = [
                u
                for u in sorted(set(labels_a) & set(labels_b))
                if labels_a[u] >= 0 and labels_b[u] >= 0
            ]
            if len(common) < 2:
                return {"n_common": len(common), "ari": float("nan")}
            a = [int(labels_a[u]) for u in common]
            b = [int(labels_b[u]) for u in common]
            return {"n_common": len(common), "ari": float(adjusted_rand_score(a, b))}

        return _sil, _agree, False


# ══════════════════════════════════════════════════════════════════════════
# Datos
# ══════════════════════════════════════════════════════════════════════════


def pca30(X: np.ndarray) -> np.ndarray:
    """PCA(30) con random_state=42, igual que Method C.

    Si d <= 30 no se reduce (evita una rotacion gratuita en los datos
    sinteticos, que ya viven en 30-d).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] <= PCA_DIMS:
        return X
    from sklearn.decomposition import PCA

    return PCA(n_components=PCA_DIMS, random_state=SEED).fit_transform(X)


def load_real_dataset(path: str = DATA_PATH) -> Dict:
    """Carga workflow_data.json en SOLO LECTURA y arma la matriz real."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no existe {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    uces = data.get("uces") or []
    ids, rows, ref = [], [], []
    for uce in uces:
        emb = uce.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        ids.append(str(uce.get("id")))
        rows.append(emb)
        ref.append(uce.get("cluster_id"))
    if not rows:
        raise RuntimeError("workflow_data.json no tiene embeddings utilizables")
    X = np.asarray(rows, dtype=np.float64)
    ref_arr = np.asarray([-1 if c is None else int(c) for c in ref], dtype=int)
    return {
        "name": "real_ctx",
        "X": pca30(X),
        "ids": ids,
        "n_in": len(uces),
        "d_in": int(X.shape[1]),
        "ref_labels": ref_arr,
        "note": (
            f"{len(ids)}/{len(uces)} UCEs con embedding "
            f"({X.shape[1]}-d SentenceTransformer; Method C usa fastText 300-d)"
        ),
    }


def synth_blobs(
    n: int = 528,
    k: int = 8,
    d: int = 30,
    sep: float = 2.6,
    noise_frac: float = 0.12,
    seed: int = 7,
) -> Dict:
    """Blobs gaussianos con K_true=k y una fraccion de puntos uniformes."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, sep, size=(k, d))
    n_noise = int(round(n * noise_frac))
    n_clustered = n - n_noise
    # tamanos decrecientes pero todos >= 4 % de n (estructura real, no satelites)
    w = np.linspace(1.0, 0.45, k)
    w = w / w.sum()
    sizes = np.maximum(20, np.round(w * n_clustered).astype(int))
    sizes[-1] += n_clustered - int(sizes.sum())
    labels = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    X = centers[labels] + rng.normal(0.0, 0.9, size=(len(labels), d))
    span = float(np.abs(centers).max()) + 3.0
    Xn = rng.uniform(-span, span, size=(n_noise, d))
    X = np.vstack([X, Xn])
    lab = np.concatenate([labels, np.full(n_noise, -1)])
    perm = rng.permutation(n)
    return {
        "name": "synth_blobs",
        "X": X[perm],
        "ids": [f"syn{i:04d}" for i in range(n)],
        "n_in": n,
        "d_in": d,
        "ref_labels": lab[perm],
        "note": f"K_true={k}, {noise_frac:.0%} puntos uniformes, sep={sep}",
    }


def synth_null(n: int = 528, d: int = 30, seed: int = 11) -> Dict:
    """Gaussiana isotropica: no hay estructura. Control negativo."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d))
    return {
        "name": "synth_null",
        "X": X,
        "ids": [f"nul{i:04d}" for i in range(n)],
        "n_in": n,
        "d_in": d,
        "ref_labels": None,
        "note": "gaussiana isotropica 30-d (sin estructura)",
    }


# ══════════════════════════════════════════════════════════════════════════
# Backends
# ══════════════════════════════════════════════════════════════════════════


def corte_optimo_1d(coord: np.ndarray, min_side: int = 1) -> Tuple[float, np.ndarray]:
    """COPIA LITERAL de ClasificadorDescendente._corte_optimo (main_workflow).

    Corte que maximiza SS_between restringido a k en [min_side, n-min_side].
    En 1-D es el optimo exacto de 2-means con esa restriccion. Devuelve
    (R2 del corte, labels 0/1 ordenadas por `coord`).
    """
    n = len(coord)
    if n < 2:
        return 0.0, np.zeros(n, dtype=int)

    order = np.argsort(coord)
    sorted_c = coord[order]
    total = sorted_c.sum()
    mean = total / n

    cum = np.cumsum(sorted_c)
    left_sum = cum[:-1]
    right_sum = total - left_sum
    left_n = np.arange(1, n)
    right_n = n - left_n

    ss_between = left_sum**2 / left_n + right_sum**2 / right_n - total**2 / n

    lo = max(1, int(min_side))
    hi = n - lo
    if hi < lo:
        return 0.0, np.zeros(n, dtype=int)
    k_best = lo + int(np.argmax(ss_between[lo - 1 : hi]))

    ss_total = float(np.sum((sorted_c - mean) ** 2))
    var_inter = float(ss_between[k_best - 1]) / (ss_total + 1e-12)

    labels = np.zeros(n, dtype=int)
    labels[order[k_best:]] = 1
    return var_inter, labels


def _pc1_local(B: np.ndarray, nz: np.ndarray, fallback: np.ndarray):
    """PC1 local del bloque, calculada SOLO con filas no-cero.

    Devuelve (v, degenerada). Sin RNG: `eigh` de una matriz simetrica es
    determinista; el signo se canoniza exigiendo positivo el componente de
    mayor valor absoluto (la particion es invariante al signo).
    """
    if int(nz.sum()) < 3:
        return fallback, True
    Bn = B[nz]
    Bc = Bn - Bn.mean(axis=0, keepdims=True)
    cov = (Bc.T @ Bc) / max(1, Bc.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    l1 = float(vals[order[0]])
    l2 = float(vals[order[1]]) if len(order) > 1 else 0.0
    if not np.isfinite(l1) or l1 <= 1e-12 or (l1 - l2) <= 1e-9 * max(l1, 1e-12):
        return fallback, True  # subespacio degenerado: eje global canonico
    v = np.asarray(vecs[:, order[0]], dtype=np.float64)
    k = int(np.argmax(np.abs(v)))
    if v[k] < 0:
        v = -v
    return v, False


def dense_refine(B: np.ndarray, labels: np.ndarray, floor: int, max_iter: int = 5):
    """Refinamiento por centroide: Lloyd determinista con reversion al piso.

    Espejo de `_intercambio` + reversion de main_workflow: si una iteracion
    rompe el piso (o vacia una clase) se descarta y se conserva la particion
    anterior. Sin RNG, sin semillas.
    """
    cur = labels.copy()
    reverted = 0
    for _ in range(int(max_iter)):
        m0 = cur == 0
        m1 = cur == 1
        if not m0.any() or not m1.any():
            break
        c0 = B[m0].mean(axis=0)
        c1 = B[m1].mean(axis=0)
        d0 = np.einsum("ij,ij->i", B - c0, B - c0)
        d1 = np.einsum("ij,ij->i", B - c1, B - c1)
        new = (d1 < d0).astype(int)
        if np.array_equal(new, cur):
            break
        if int(np.bincount(new, minlength=2).min()) < int(floor):
            reverted += 1
            break
        cur = new
    return cur, reverted


def r2_dense(B: np.ndarray, labels: np.ndarray) -> float:
    """1 - SSE/TSS en las d dims de la matriz reducida (R2 del corte denso)."""
    mu = B.mean(axis=0, keepdims=True)
    tss = float(np.sum((B - mu) ** 2))
    if tss <= 0:
        return 0.0
    sse = 0.0
    for cls in np.unique(labels):
        Bc = B[labels == cls]
        sse += float(np.sum((Bc - Bc.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - sse / tss


def dense_cdh(
    X: np.ndarray,
    target_k: int,
    floor: int,
    min_r2_1d: float = 0.0,
    min_r2_dense: float = 0.0,
    refine_iter: int = 5,
    max_depth: int = 10,
) -> Tuple[np.ndarray, Dict]:
    """Biseccion divisiva best-first por PC1 local + corte 1-D (sin -1).

    A diferencia de la recursion pura de DENSE_CDH_DESIGN.md, este esquema
    elige en cada paso la hoja con mayor ganancia (r2_denso) entre TODAS las
    hojas: eso da control real de K (se para al llegar a target_k) y deja los
    splits mas debiles para el final. Cada hoja se evalua una sola vez, asi que
    el coste total es O(n*d^2*profundidad) y no O(n^2).

    `min_r2_1d` es el gate del eje (por defecto 0 = apagado): se midio que en
    ruido isotropico el corte 1-D optimo explica ~2/pi = 0.637, asi que un
    umbral tipo min_r2_threshold=0.05 NO discrimina. `min_r2_dense` es el gate
    calibrado sobre el R2 denso (0.08 en produccion; ver BACKEND_C.md).
    """
    n, d = X.shape
    nz = np.abs(X).sum(axis=1) > 1e-12
    fallback = np.zeros(d)
    fallback[0] = 1.0
    stats = {
        "r2_1d": [],
        "r2_dense": [],
        "depths": [],
        "degenerate_pc1": 0,
        "reverted_floor": 0,
        "rejected_min_r2": 0,
        "rejected_min_r2_dense": 0,
        "splits": 0,
    }

    def evaluate(leaf: Dict) -> Optional[Dict]:
        idx = leaf["idx"]
        if len(idx) < 2 * int(floor) or leaf["depth"] >= int(max_depth):
            return None
        B = X[idx]
        m = nz[idx]
        v, degen = _pc1_local(B, m, fallback)
        center = B[m].mean(axis=0) if int(m.sum()) > 0 else np.zeros(B.shape[1])
        coord = (B - center) @ v
        r2_1d, lab = corte_optimo_1d(coord, min_side=int(floor))
        if len(np.unique(lab)) < 2:
            return None
        if r2_1d < float(min_r2_1d):
            return {"rejected": True, "r2_1d": float(r2_1d)}
        lab_ref, reverted = dense_refine(B, lab, int(floor), refine_iter)
        if int(np.bincount(lab_ref, minlength=2).min()) < int(floor):
            lab_ref = lab  # reversion al piso (espejo de main_workflow:4144-4147)
            reverted += 1
        return {
            "rejected": False,
            "labels": lab_ref,
            "r2_1d": float(r2_1d),
            "r2_dense": float(r2_dense(B, lab_ref)),
            "degenerate_pc1": bool(degen),
            "reverted": int(reverted),
        }

    leaves: List[Dict] = [
        {"idx": np.arange(n), "depth": 0, "cand": None, "done": False}
    ]
    n_zero_rows = int((~nz).sum())

    if n >= 2 * int(floor):
        while len(leaves) < int(target_k):
            best_key, best_i, best_c = None, None, None
            for i, leaf in enumerate(leaves):
                if not leaf["done"]:
                    leaf["cand"] = evaluate(leaf)
                    leaf["done"] = True
                    c = leaf["cand"]
                    if c is not None and c.get("rejected"):
                        stats["rejected_min_r2"] += 1
                c = leaf["cand"]
                if c is None or c.get("rejected"):
                    continue
                key = (float(c["r2_dense"]), -int(leaf["idx"][0]))
                if best_key is None or key > best_key:
                    best_key, best_i, best_c = key, i, c
            if best_c is None:
                break
            # Gate calibrado sobre el R2 denso. Best-first ya eligio la hoja de
            # MAYOR ganancia: si esa no pasa el umbral, ninguna otra pasa.
            if float(best_c["r2_dense"]) < float(min_r2_dense):
                stats["rejected_min_r2_dense"] += 1
                break
            leaf = leaves[best_i]
            idx = leaf["idx"]
            lab = best_c["labels"]
            left, right = idx[lab == 0], idx[lab == 1]
            stats["r2_1d"].append(best_c["r2_1d"])
            stats["r2_dense"].append(best_c["r2_dense"])
            stats["depths"].append(int(leaf["depth"]))
            stats["degenerate_pc1"] += int(best_c["degenerate_pc1"])
            stats["reverted_floor"] += int(best_c["reverted"])
            stats["splits"] += 1
            leaves[best_i] = {
                "idx": left,
                "depth": leaf["depth"] + 1,
                "cand": None,
                "done": False,
            }
            leaves.insert(
                best_i + 1,
                {"idx": right, "depth": leaf["depth"] + 1, "cand": None, "done": False},
            )

    leaves.sort(key=lambda lf: int(lf["idx"][0]))
    labels = np.full(n, -1, dtype=int)
    for label, leaf in enumerate(leaves):
        labels[leaf["idx"]] = label
    diag = {
        "bisection_depths": [int(x) for x in stats["depths"]],
        "r2_1d": [round(float(x), 4) for x in stats["r2_1d"]],
        "r2_dense": [round(float(x), 4) for x in stats["r2_dense"]],
        "mean_r2_1d": float(np.mean(stats["r2_1d"])) if stats["r2_1d"] else 0.0,
        "mean_r2_dense": float(np.mean(stats["r2_dense"]))
        if stats["r2_dense"]
        else 0.0,
        "degenerate_pc1": int(stats["degenerate_pc1"]),
        "reverted_floor": int(stats["reverted_floor"]),
        "rejected_min_r2": int(stats["rejected_min_r2"]),
        "rejected_min_r2_dense": int(stats["rejected_min_r2_dense"]),
        "n_splits": int(stats["splits"]),
        "n_zero_rows": n_zero_rows,
        "floor": int(floor),
        "min_r2_1d": float(min_r2_1d),
        "min_r2_dense": float(min_r2_dense),
    }
    return labels, diag


def ward_backend(X: np.ndarray, k: int) -> Tuple[np.ndarray, Dict]:
    """Ward sobre PCA(30). Determinista; O(n^2) en memoria (full tree)."""
    from sklearn.cluster import AgglomerativeClustering

    ac = AgglomerativeClustering(
        n_clusters=int(k), linkage="ward", compute_distances=True
    )
    labels = ac.fit_predict(X).astype(int)
    heights = np.asarray(ac.distances_, dtype=float)
    n = X.shape[0]
    used = heights[: max(0, int(n) - int(k))]
    diag = {
        "merge_heights": [round(float(h), 4) for h in used.tolist()],
        "cut_height": float(used[-1]) if len(used) else 0.0,
        "mean_merge_height": float(used.mean()) if len(used) else 0.0,
        "n_merges": int(len(used)),
        "linkage": "ward",
        "metric": "euclidean",
    }
    return labels, diag


def kmeans_spherical(X: np.ndarray, k: int) -> Tuple[np.ndarray, Dict]:
    """L2-normalizar + KMeans(n_init=10, random_state=42). K exacto."""
    from sklearn.cluster import KMeans

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    Xn = X / norms
    km = KMeans(n_clusters=int(k), n_init=10, random_state=SEED)
    labels = km.fit_predict(Xn).astype(int)
    diag = {
        "inertia": round(float(km.inertia_), 4),
        "n_iter": int(km.n_iter_),
        "n_init": 10,
        "random_state": SEED,
        "normalized_input": True,
    }
    return labels, diag


def hdbscan_backend(X: np.ndarray, min_cluster_size: int, min_samples: int):
    """HDBSCAN plano: el Method C actual. No controla K; puede emitir -1.

    Se usa gen_min_span_tree=False y prediction_data=False solo por velocidad
    (no cambian las etiquetas); produccion los deja en True.
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        cluster_selection_epsilon=0.0,
        metric="euclidean",
        gen_min_span_tree=False,
        prediction_data=False,
    )
    labels = clusterer.fit_predict(X).astype(int)
    diag = {
        "cluster_persistence": [
            round(float(p), 4)
            for p in list(getattr(clusterer, "cluster_persistence_", []))
        ],
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "epsilon": 0.0,
    }
    return labels, diag


# ══════════════════════════════════════════════════════════════════════════
# Contrato de resultado (identico a _clasificar_umbral_embedding + extras)
# ══════════════════════════════════════════════════════════════════════════


def fit_backend(
    backend: str,
    X: np.ndarray,
    ids: Sequence[str],
    target_k: int,
    floor: int,
    min_r2_dense: float = 0.0,
) -> Dict:
    """Devuelve el dict con el contrato comun de Method C."""
    if backend == "dense_cdh":
        labels, diag = dense_cdh(X, target_k, floor, min_r2_dense=min_r2_dense)
    elif backend == "ward":
        labels, diag = ward_backend(X, target_k)
    elif backend == "kmeans_spherical":
        labels, diag = kmeans_spherical(X, target_k)
    elif backend == "hdbscan":
        mcs = int(floor)
        ms = max(2, int(round(0.25 * mcs)))
        labels, diag = hdbscan_backend(X, mcs, ms)
    else:
        raise ValueError(f"backend desconocido: {backend!r}")

    assigned = labels[labels >= 0]
    sizes = np.bincount(assigned).tolist() if assigned.size else []
    uce_to_label = {
        str(ids[i]): int(labels[i]) for i in range(len(ids)) if int(labels[i]) >= 0
    }
    diag["n_zero_rows"] = int((np.abs(X).sum(axis=1) < 1e-12).sum())
    return {
        "method": "embedding",
        "backend": backend,
        "target_k": int(target_k),
        "min_cluster_size": int(floor),
        "uce_to_label": uce_to_label,
        "labels_array": labels,
        "uce_ids": [str(i) for i in ids],
        "embedding_matrix": X,
        "cluster_persistence": list(diag.get("cluster_persistence", [])),
        "n_clusters": int(len(set(assigned.tolist()))),
        "n_noise": int(np.sum(labels < 0)),
        "partition_sizes": [int(s) for s in sizes],
        "n_assigned": int(assigned.size),
        "diagnostics": diag,
    }


# ══════════════════════════════════════════════════════════════════════════
# Arena
# ══════════════════════════════════════════════════════════════════════════

BACKENDS = ("hdbscan", "dense_cdh", "ward", "kmeans_spherical")


def floor_for(n: int, k: int) -> int:
    """Piso espejo de main_workflow: max(5, int(n * 1/(2K))) con frac=1/(2K)."""
    return max(5, int(n * (1.0 / (2.0 * max(1, int(k))))))


def cv_sizes(sizes: Sequence[int]) -> float:
    if not sizes:
        return float("nan")
    arr = np.asarray(sizes, dtype=float)
    return float(arr.std() / max(arr.mean(), 1e-12))


def fmt(value: Optional[float], width: int = 6, dec: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return " " * (width - 3) + "n/a"
    return f"{value:{width}.{dec}f}"


def run_arena(
    datasets: List[Dict], ks: List[int], min_r2_dense: float, verbose: bool
) -> int:
    print("=" * 78)
    print(" ARENA DE BACKENDS DENSO - Method C (REII)  [solo lectura, sin red]")
    print("=" * 78)

    for ds in datasets:
        X = ds["X"]
        n, d = X.shape
        print()
        print(
            f"[data] {ds['name']}: n={n} d_in={ds.get('d_in', d)} -> d={d} "
            f"(PCA<=30, seed=42)"
        )
        print(f"       {ds['note']}")
        ref = ds.get("ref_labels")
        if ref is not None:
            assigned = ref[ref >= 0]
            sizes = np.bincount(assigned).tolist() if assigned.size else []
            dom = max(sizes) / max(1, sum(sizes)) if sizes else float("nan")
            print(
                f"       particion de referencia: K={len(set(assigned.tolist()))} "
                f"max_class_share={dom:.3f} (no es entrada de ningun backend)"
            )

        t_ds = time.perf_counter()
        fits: Dict[Tuple[str, int], Dict] = {}

        # ── pasada 1: ajustar cada (backend, K) y verificar determinismo ──
        for backend in BACKENDS:
            for k in ks:
                floor = floor_for(n, k)
                t0 = time.perf_counter()
                try:
                    res1 = fit_backend(
                        backend, X, ds["ids"], k, floor, min_r2_dense=min_r2_dense
                    )
                except Exception as exc:
                    print(f"[!] {backend} K={k} fallo: {type(exc).__name__}: {exc}")
                    continue
                ms = (time.perf_counter() - t0) * 1000.0

                # segunda corrida: determinismo + runtime (min de las dos)
                t1 = time.perf_counter()
                try:
                    res2 = fit_backend(
                        backend, X, ds["ids"], k, floor, min_r2_dense=min_r2_dense
                    )
                    det_ok = bool(
                        np.array_equal(res1["labels_array"], res2["labels_array"])
                        and res1["uce_to_label"] == res2["uce_to_label"]
                    )
                except Exception:
                    det_ok = False
                # min de las dos corridas: la primera paga caches de CPU frias
                ms = min(ms, (time.perf_counter() - t1) * 1000.0)

                sizes = res1["partition_sizes"]
                dom = (
                    (max(sizes) / res1["n_assigned"])
                    if sizes and res1["n_assigned"]
                    else float("nan")
                )
                fits[(backend, int(k))] = {
                    "res": res1,
                    "ms": ms,
                    "det": det_ok,
                    "sil": silhouette_labels(
                        X, res1["labels_array"], metric="euclidean"
                    ),
                    "dom": dom,
                    "cv": cv_sizes(sizes),
                }

        # ── pasada 2: tabla con ARI(K, K+2) ya calculado ─────────────────
        header = (
            f"{'backend':17}{'K':>3}{'Kach':>5}{'silh':>7}{'ARI K/K+2':>11}"
            f"{'cv_sz':>7}{'dom':>7}{'noise':>7}{'ms':>8}  det"
        )
        print()
        print(header)
        print("-" * len(header))
        rows: List[Tuple] = []
        for backend in BACKENDS:
            for k in ks:
                item = fits.get((backend, int(k)))
                if item is None:
                    continue
                res = item["res"]
                nxt = fits.get((backend, int(k) + 2))
                ari = float("nan")
                if nxt is not None:
                    agree = cross_partition_agreement(
                        {u: int(v) for u, v in res["uce_to_label"].items()},
                        {u: int(v) for u, v in nxt["res"]["uce_to_label"].items()},
                    )
                    ari = float(agree["ari"])
                rows.append(
                    (
                        backend,
                        int(k),
                        res["n_clusters"],
                        item["sil"],
                        item["cv"],
                        item["dom"],
                        res["n_noise"],
                        item["ms"],
                        item["det"],
                        res["diagnostics"],
                        ari,
                    )
                )
                print(
                    f"{backend:17}{k:>3}{res['n_clusters']:>5}{fmt(item['sil'], 7)}"
                    f"{fmt(ari, 11)}{fmt(item['cv'], 7)}{fmt(item['dom'], 7)}"
                    f"{res['n_noise']:>7}{item['ms']:>8.1f}  "
                    f"{'[ok]' if item['det'] else '[!]'}"
                )

        elapsed = time.perf_counter() - t_ds
        print()
        print(
            f"[dataset] {ds['name']}: {len(fits)} celdas (backend x K), "
            f"{2 * len(fits)} corridas en {elapsed:.2f} s"
        )
        gate_sweep(X, ds["ids"], n)

        if verbose:
            print_diagnostics(rows)

    return 0


def gate_sweep(
    X: np.ndarray,
    ids: Sequence[str],
    n: int,
    k: int = 8,
    thresholds: Sequence[float] = (0.0, 0.05, 0.08, 0.12, 0.20),
) -> None:
    """Efecto del gate r2_dense sobre K alcanzado (una corrida por umbral).

    Es la medicion que fija el default de produccion: el umbral debe dejar
    pasar los cortes de estructura real y frenar los de ruido isotropico.
    """
    floor = floor_for(n, k)
    print()
    print(
        f"[gate r2_dense] dense_cdh, K objetivo={k}, floor={floor}, "
        "r2_1d sin gate (0.0)"
    )
    print(f"  {'umbral':>7}{'Kach':>6}{'silh':>8}{'splits':>8}{'r2_dense_med':>14}")
    for thr in thresholds:
        res = fit_backend("dense_cdh", X, ids, k, floor, min_r2_dense=float(thr))
        sil = silhouette_labels(X, res["labels_array"], metric="euclidean")
        diag = res["diagnostics"]
        print(
            f"  {thr:>7.2f}{res['n_clusters']:>6}{fmt(sil, 8)}"
            f"{diag['n_splits']:>8}{diag['mean_r2_dense']:>14.3f}"
        )


def print_diagnostics(rows: List[Tuple]) -> None:
    print()
    print("[diag] bloques propios de cada backend (fila = backend/K)")
    for backend, k, kach, sil, cvs, dom, noise, ms, det, diag, ari in rows:
        if backend == "dense_cdh":
            print(
                f"  dense_cdh K={k} -> Kach={kach} splits={diag['n_splits']} "
                f"mean_r2_1d={diag['mean_r2_1d']:.3f} "
                f"mean_r2_dense={diag['mean_r2_dense']:.3f} "
                f"rejected_r2_1d={diag['rejected_min_r2']} "
                f"rejected_r2_dense={diag['rejected_min_r2_dense']} "
                f"reverted={diag['reverted_floor']} degen_pc1={diag['degenerate_pc1']}"
            )
        elif backend == "ward":
            print(
                f"  ward K={k} -> Kach={kach} cut_height={diag['cut_height']:.3f} "
                f"n_merges={diag['n_merges']}"
            )
        elif backend == "kmeans_spherical":
            print(
                f"  kmeans K={k} -> Kach={kach} inertia={diag['inertia']:.1f} "
                f"n_iter={diag['n_iter']}"
            )
        else:
            pers = diag.get("cluster_persistence") or []
            pmax = max(pers) if pers else 0.0
            print(
                f"  hdbscan K={k} -> Kach={kach} noise={noise} "
                f"max_persistence={pmax:.3f} mcs={diag['min_cluster_size']} "
                f"ms={diag['min_samples']}"
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arena de backends densos de Method C")
    ap.add_argument(
        "--ks",
        default="3,5,8,11,14,18,20",
        help="lista de K objetivo separada por comas",
    )
    ap.add_argument(
        "--datasets",
        default="real_ctx,synth_blobs,synth_null",
        help="subconjunto de real_ctx,synth_blobs,synth_null",
    )
    ap.add_argument(
        "--min-r2-dense",
        type=float,
        default=DENSE_MIN_R2_DENSE_ARENA,
        help="gate r2_dense de dense_cdh (0.0 = target-K exacto; produccion 0.08)",
    )
    ap.add_argument(
        "--quiet-diag", action="store_true", help="omite bloques de diagnostico"
    )
    args = ap.parse_args(argv)

    ks = [int(x) for x in str(args.ks).split(",") if x.strip()]
    wanted = {x.strip() for x in str(args.datasets).split(",") if x.strip()}

    t_start = time.perf_counter()
    print("[init] importando metricas del proyecto...")
    t_imp = time.perf_counter()
    global silhouette_labels, cross_partition_agreement
    silhouette_labels, cross_partition_agreement, reused = load_project_metrics()
    print(
        f"[init] metricas listas en {time.perf_counter() - t_imp:.1f} s "
        f"({'reii.quality.metrics' if reused else 'fallback sklearn'})"
    )

    datasets: List[Dict] = []
    if "real_ctx" in wanted:
        try:
            datasets.append(load_real_dataset())
        except Exception as exc:
            print(f"[!] datos reales no disponibles: {type(exc).__name__}: {exc}")
    if "synth_blobs" in wanted:
        datasets.append(synth_blobs())
    if "synth_null" in wanted:
        datasets.append(synth_null())
    if not datasets:
        print("[!] sin datasets: nada que medir")
        return 0

    # warm-up: paga los imports perezosos (sklearn.cluster, hdbscan) y las
    # caches de CPU FUERA del cronometro de cada celda.
    t_w = time.perf_counter()
    Xw = np.random.default_rng(0).normal(size=(60, 8))
    wid = [f"w{i}" for i in range(Xw.shape[0])]
    for backend in BACKENDS:
        try:
            fit_backend(backend, Xw, wid, 3, 5, min_r2_dense=0.0)
        except Exception:
            pass
    print(f"[init] warm-up de backends en {time.perf_counter() - t_w:.1f} s")

    run_arena(datasets, ks, float(args.min_r2_dense), verbose=not args.quiet_diag)

    print()
    print("=" * 78)
    print("NOTAS")
    print("  * backends densos (dense_cdh, ward, kmeans_spherical) NUNCA emiten -1:")
    print("    n_noise=0 y 'dom' = clase mayor / UCEs asignadas (denominador sin")
    print("    dilucion). Con ruido de HDBSCAN el mismo 442 se divide por menos")
    print("    UCEs y el gate de dominancia (0.90) cambia de veredicto.")
    print("  * hdbscan NO controla K: 'K' es solo el piso min_cluster_size")
    print("    (n/(2K), espejo de cdh_target_k) y 'Kach' es lo que emerge.")
    print("  * dense_cdh: 'Kach' < 'K' significa que el gate r2_dense o el piso")
    print("    impidieron seguir partiendo (no fabrica splits).")
    print("  * r2_1d NO discrimina: en ruido isotropico el corte 1-D optimo explica")
    print("    ~2/pi = 0.637 de la varianza del eje (ver fila K=20 de synth_null).")
    print("    Un umbral tipo min_r2_threshold=0.05 nunca rechaza. El gate util es")
    print(f"    r2_dense (default de produccion: {DENSE_MIN_R2_DENSE:.2f}).")
    print("  * ARI(K, K+2) mide ANIDAMIENTO, no validez: los backends jerarquicos")
    print("    (dense_cdh/ward) dan ~0.9-1.0 incluso sobre ruido puro.")
    print("  * space mismatch en real_ctx: 384-d SentenceTransformer != fastText")
    print("    300-d de Method C; los valores son indicativos, no comparables 1:1.")
    print(f"[total] {time.perf_counter() - t_start:.2f} s")
    return 0


if __name__ == "__main__":
    silhouette_labels = None
    cross_partition_agreement = None
    sys.exit(main())
