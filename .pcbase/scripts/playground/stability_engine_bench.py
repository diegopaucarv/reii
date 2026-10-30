#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Banco de pruebas del MOTOR DE ESTABILIDAD (docs/design_v5/STABILITY_ENGINE.md).

Mide, sin red y sin pipeline:

  1. bootstrap_ari   : ARI de submuestreo + reajuste, contra la particion completa.
  2. permutation_null: nulo por permutacion de etiquetas (sirve para r2 y silhouette).
  3. r2_dense        : 1 - SSE/TSS en d dimensiones (funcion pura).
  4. stability_report: agregador serializable.

QUE SE IMPORTA Y QUE NO
-----------------------
Prohibido importar ``reii.main_workflow`` (>4 min por Stanza). Este script NO
lo hace. Las metricas del proyecto (``reii.quality.metrics``) se importan solo
con ``--with-reii`` (cuesta 12.6 s medidos de import en este entorno) para
verificar que ``silhouette_precomputed`` coincide con ``silhouette_labels``.

Las funciones del motor se incluyen aqui VERBATIM (seccion "COPIA CANONICA"):
son el mismo texto que el diff de ``src/reii/quality/stability.py``. El script
existe porque las reglas de trabajo prohiben escribir en ``src/**``; cuando se
aplique el diff, este bloque puede reemplazarse por
``from reii.quality.stability import ...`` sin cambiar ni un numero.

DATOS
-----
Sinteticos con verdad conocida: blobs gaussianos (sep 1.2 / 1.8 / 2.6, 12 %
uniforme, K_true=8) y gaussiana isotropica (control negativo). Se reutilizan
``synth_blobs``/``synth_null``/``load_real_dataset`` y los tres backends de
``scripts/playground/c_backends_arena.py`` (misma matematica ya medida).

``data/workflow_data.json`` se abre en SOLO LECTURA. Aviso de espacio: sus
embeddings son 384-d SentenceTransformer con contexto, mientras Method C usa
fastText-retrofit 300-d; y solo 235 de las 600 UCEs traen embedding. Los
numeros de ``real_ctx`` son indicativos, NO comparables 1:1 con produccion.

DETERMINISMO
------------
``OMP/OPENBLAS/MKL_NUM_THREADS=1`` se fijan ANTES de importar numpy. Ademas se
compara el mismo backend con 1 y 4 hilos via ``threadpoolctl.threadpool_limits``
(la unica palanca que funciona despues de importar numpy).

Uso:
    python scripts/playground/stability_engine_bench.py
    python scripts/playground/stability_engine_bench.py --with-reii
Salida ASCII puro (cp1252), exit 0.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import warnings
from typing import Callable, Dict, List, Optional

# Un hilo antes de numpy: reproducible y mas rapido a este tamano (ver BACKEND_C).
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
PLAY_DIR = os.path.join(REPO_ROOT, "scripts", "playground")
for _p in (SRC_DIR, PLAY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import c_backends_arena as arena  # noqa: E402

SEED = 42
N_BOOT = 20
N_BOOT_MATRIX = 5  # matriz: |mean(B=5)-mean(B=20)| medido 0.015 (TABLA 2)
N_PERM = 200
N_PERM_SIL = 100  # silhouette: 100 permutaciones bastan (p minimo 0.010)
MIN_N = 30
BUDGET_S = 2.0  # presupuesto duro por trial

NAN = float("nan")


# ══════════════════════════════════════════════════════════════════════════
# COPIA CANONICA del modulo propuesto src/reii/quality/stability.py
# (el diff del documento es este mismo texto; ver STABILITY_ENGINE.md)
# ══════════════════════════════════════════════════════════════════════════


def _thread_limits(limits: Optional[int]):
    """Contexto threadpoolctl; no-op si threadpoolctl no esta instalado."""
    if not limits:
        return contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except Exception:  # pragma: no cover - entorno sin threadpoolctl
        return contextlib.nullcontext()
    return threadpool_limits(limits=int(limits))


def _matrix(X) -> Optional[np.ndarray]:
    """Convierte a matriz 2-D float64; None si no es convertible."""
    try:
        A = np.asarray(X, dtype=np.float64)
    except Exception:
        return None
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        return None
    return A


def _labels(labels) -> Optional[np.ndarray]:
    try:
        y = np.asarray(labels).ravel().astype(int)
    except Exception:
        return None
    return y


def r2_dense(X, labels, *, ignore_noise: bool = True) -> float:
    """R2 denso: ``1 - SSE/TSS`` en las d dimensiones de ``X``. Funcion pura.

    SSE se calcula respecto de la media de cada clase; TSS respecto de la media
    global. Es el R2 del *corte*, no el de la particion final, y por eso no esta
    confundido con K (BACKEND_C.md §6).

    ``ignore_noise=True`` descarta las filas con etiqueta < 0 (misma convencion
    que ``silhouette_labels``); con ``False`` el -1 actua como una clase mas.

    Devuelve NaN (nunca lanza) si: X y labels no coinciden, n < 2, hay menos de
    2 clases, TSS = 0 (todos los puntos identicos, p.ej. vectores cero) o el
    resultado no es finito.
    """
    A, y = _matrix(X), _labels(labels)
    if A is None or y is None or A.shape[0] != y.shape[0]:
        return NAN
    if ignore_noise:
        keep = y >= 0
        A, y = A[keep], y[keep]
    n = A.shape[0]
    if n < 2:
        return NAN
    if not np.all(np.isfinite(A)):
        return NAN
    inv = np.unique(y, return_inverse=True)[1]
    counts = np.bincount(inv).astype(float)
    if counts.size < 2 or counts.min() <= 0.0:
        return NAN
    total = float(np.sum(A * A))
    mu = A.mean(axis=0)
    tss = total - n * float(mu @ mu)
    if not np.isfinite(tss) or tss <= 0.0:
        return NAN
    onehot = np.zeros((n, counts.size))
    onehot[np.arange(n), inv] = 1.0
    sums = onehot.T @ A  # K x d
    sse = total - float(np.sum(np.sum(sums * sums, axis=1) / counts))
    sse = max(sse, 0.0)
    value = 1.0 - sse / tss
    return float(value) if np.isfinite(value) else NAN


def _silhouette_precomputed(D: np.ndarray, y: np.ndarray) -> float:
    """Silhouette medio a partir de una matriz de distancias ya calculada.

    Version vectorizada de la definicion de Rousseeuw: para cada cluster j se
    calcula ``M[:, j] = mean(D[i, puntos de j])``; ``a(i)`` es la media a su
    propio cluster sin contarse (``count*own/(count-1)``) y ``b(i)`` el minimo
    de las medias a los demas. Las UCEs de un cluster de tamano 1 aportan
    ``s = 0`` (convencion de sklearn).

    Equivale a ``silhouette_score(D, y, metric="precomputed")`` pero evita los
    bucles de Python del estimador: es ~40x mas rapido en este corpus, y el
    nulo de 200 permutaciones cabe en el presupuesto.
    """
    classes, inv = np.unique(y, return_inverse=True)
    n, k = y.size, classes.size
    if k < 2 or n < 3:
        return NAN
    counts = np.bincount(inv, minlength=k).astype(float)
    M = np.empty((n, k), dtype=float)
    for j in range(k):
        M[:, j] = D[:, inv == j].mean(axis=1)
    rows = np.arange(n)
    own = M[rows, inv]
    cn = counts[inv]
    a = np.where(cn > 1.0, cn * own / np.maximum(cn - 1.0, 1.0), 0.0)
    M2 = M.copy()
    M2[rows, inv] = np.inf
    b = M2.min(axis=1)
    denom = np.maximum(a, b)
    s = np.where(denom > 0.0, (b - a) / denom, 0.0)
    s[cn <= 1.0] = 0.0
    value = float(s.mean())
    return value if np.isfinite(value) else NAN


def silhouette_precomputed(X, metric: str = "euclidean") -> Callable:
    """Fabrica un ``stat_fn(X_ignorado, labels)`` con la matriz D precalculada.

    Permutar etiquetas no cambia X, asi que la distancia se calcula UNA vez por
    llamada y no 200 veces. Sin esto, ``silhouette_labels`` recalcula O(n^2 d)
    en cada permutacion y el nulo se sale del presupuesto (§4 del documento).

    Devuelve NaN si D no es finita, si hay < 3 UCEs asignadas o < 2 clases.
    """
    A = _matrix(X)
    if A is None:
        raise ValueError("X no es una matriz 2-D")
    from scipy.spatial.distance import cdist

    D = cdist(A, A, metric=metric)
    finite_d = bool(np.all(np.isfinite(D)))

    def _stat(_X, labels) -> float:
        y = _labels(labels)
        if y is None or y.size != A.shape[0] or not finite_d:
            return NAN
        keep = y >= 0
        idx = np.flatnonzero(keep)
        yy = y[keep]
        if yy.size < 3 or np.unique(yy).size < 2:
            return NAN
        try:
            Dk = D if idx.size == A.shape[0] else D[np.ix_(idx, idx)]
            return _silhouette_precomputed(Dk, yy)
        except Exception:
            return NAN

    _stat.__name__ = "silhouette_precomputed"
    return _stat


def _ari_pair(ref: np.ndarray, pred: np.ndarray) -> Optional[float]:
    """ARI en las UCEs comunes con ambas etiquetas >= 0. None si degenerado."""
    if ref.size != pred.size:
        return None
    keep = (ref >= 0) & (pred >= 0)
    if int(keep.sum()) < 2:
        return None
    a, b = ref[keep], pred[keep]
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return None
    try:
        from sklearn.metrics import adjusted_rand_score

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = float(adjusted_rand_score(a, b))
    except Exception:
        return None
    return v if np.isfinite(v) else None


def bootstrap_ari(
    predict_fn: Callable,
    X,
    labels,
    *,
    n_boot: int = 20,
    frac: float = 0.8,
    replace: bool = False,
    seed: int = 42,
    min_n: int = 30,
    alpha: float = 0.05,
    max_n: Optional[int] = None,
    fallback_fn: Optional[Callable] = None,
    threads: int = 1,
) -> Dict:
    """Estabilidad por submuestreo de UCEs + reajuste del clusterer con el mismo K.

    Contrato de ``predict_fn``: ``predict_fn(X_sub) -> etiquetas`` de longitud
    ``len(X_sub)``. El K, el piso y el ``random_state`` los cierra el llamador,
    de modo que el motor no se acopla a ward, kmeans ni dense_cdh.

    En cada replica se submuestrea ``ceil(frac*n)`` UCEs (sin reemplazo por
    defecto; con ``replace=True`` se toman con reemplazo y se deduplican antes
    de ajustar y puntuar), se reajusta el clusterer y se mide
    ``ARI(prediccion, labels)`` sobre las UCEs comunes con etiqueta >= 0.

    ``labels`` es la particion de REFERENCIA: normalmente la del ajuste
    completo (mide estabilidad); tambien puede ser la verdad conocida (mide
    acierto bajo remuestreo).

    Guarda de memoria: si ``max_n`` no es None y la submuestra lo supera, se
    usa ``fallback_fn`` (p.ej. kmeans en lugar de ward) y se cuenta en
    ``n_fallback``. Sin fallback, la replica falla con motivo ``memory_guard``.

    Devuelve un dict serializable. ``ci_low``/``ci_high`` son el intervalo
    2.5/97.5 de la distribucion de replicas (banda de estabilidad);
    ``mean_ci_low``/``mean_ci_high`` son el IC bootstrap de la MEDIA.
    ``n_eff`` es el numero medio de UCEs distintas por replica.

    Nunca lanza: ante cualquier problema devuelve ``ok=False`` + ``reason``
    con los campos numericos a NaN.
    """
    out: Dict = {
        "ok": False,
        "reason": "",
        "mean": NAN,
        "median": NAN,
        "std": NAN,
        "ci_low": NAN,
        "ci_high": NAN,
        "mean_ci_low": NAN,
        "mean_ci_high": NAN,
        "n_eff": NAN,
        "n": 0,
        "n_boot": int(n_boot),
        "n_failed": 0,
        "n_fallback": 0,
        "fail_reasons": [],
        "ari": [],
    }
    A, y = _matrix(X), _labels(labels)
    if A is None or y is None:
        out["reason"] = "X_o_labels_invalidos"
        return out
    if A.shape[0] != y.shape[0]:
        out["reason"] = "longitudes_distintas"
        return out
    n = A.shape[0]
    out["n"] = int(n)
    if n < int(min_n):
        out["reason"] = f"n={n}_menor_que_min_n={int(min_n)}"
        return out
    if not callable(predict_fn):
        out["reason"] = "predict_fn_no_es_callable"
        return out
    n_boot = int(n_boot)
    if n_boot < 1:
        out["reason"] = "n_boot<1"
        return out
    frac = float(frac)
    if not (0.0 < frac <= 1.0):
        out["reason"] = "frac_fuera_de_(0,1]"
        return out
    n_sub = int(min(n, max(3, math.ceil(frac * n))))
    rng = np.random.default_rng(seed)
    ari_vals: List[float] = []
    n_effs: List[int] = []
    fails = 0
    falls = 0
    reasons: List[str] = []
    with _thread_limits(threads), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            try:
                idx = rng.choice(n, size=n_sub, replace=bool(replace))
                if replace:
                    idx = np.unique(idx)
            except Exception:
                fails += 1
                reasons.append("sampling_error")
                continue
            if idx.size < 3:
                fails += 1
                reasons.append("submuestra_muy_pequena")
                continue
            Xs, ys = A[idx], y[idx]
            fn = predict_fn
            if max_n is not None and Xs.shape[0] > int(max_n):
                if fallback_fn is None:
                    fails += 1
                    reasons.append("memory_guard")
                    continue
                fn = fallback_fn
                falls += 1
            try:
                pred = np.asarray(fn(Xs)).ravel().astype(int)
            except Exception as exc:
                fails += 1
                reasons.append(type(exc).__name__)
                continue
            ari = _ari_pair(ys, pred)
            if ari is None:
                fails += 1
                reasons.append("degenerado_o_ARI_NaN")
                continue
            ari_vals.append(float(ari))
            n_effs.append(int(idx.size))
    out["n_failed"] = int(fails)
    out["n_fallback"] = int(falls)
    if reasons:
        uniq: List[str] = []
        for r in reasons:
            if r not in uniq:
                uniq.append(r)
        out["fail_reasons"] = uniq
    if not ari_vals:
        out["reason"] = "todas_las_replicas_fallaron"
        return out
    arr = np.asarray(ari_vals, dtype=float)
    out["ari"] = [float(v) for v in arr]
    out["mean"] = float(arr.mean())
    out["median"] = float(np.median(arr))
    out["std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    out["ci_low"] = float(np.quantile(arr, alpha / 2.0))
    out["ci_high"] = float(np.quantile(arr, 1.0 - alpha / 2.0))
    if arr.size > 1:
        brng = np.random.default_rng(int(seed) + 1)
        means = np.array(
            [
                float(arr[brng.integers(0, arr.size, size=arr.size)].mean())
                for _ in range(200)
            ]
        )
        out["mean_ci_low"] = float(np.quantile(means, alpha / 2.0))
        out["mean_ci_high"] = float(np.quantile(means, 1.0 - alpha / 2.0))
    else:
        out["mean_ci_low"] = out["mean_ci_high"] = float(arr[0])
    out["n_eff"] = float(np.mean(n_effs)) if n_effs else NAN
    out["ok"] = True
    return out


def permutation_null(
    stat_fn: Callable,
    X,
    labels,
    *,
    n_perm: int = 200,
    seed: int = 42,
    min_n: int = 30,
    alpha: float = 0.05,
    threads: int = 1,
) -> Dict:
    """Nulo por permutacion de etiquetas. ``stat_fn(X, labels) -> float``.

    Sirve igual con ``r2_dense`` y con ``silhouette_precomputed`` (o con
    ``functools.partial(silhouette_labels, metric="euclidean")``). La etiqueta
    completa se permuta, incluidos los -1: el nulo conserva el numero de clases
    y la fraccion de ruido, y solo rompe la asociacion con X.

    Devuelve observado, media/desv del nulo, z empirico
    ``(obs-media)/desv``, ``p_value`` unilateral superior (cuantos nulos
    alcanzan el observado, con correccion +1) y cuantiles 1/5/50/95/99.

    AVISO: el estadistico observado se calcula sobre una particion AJUSTADA a X.
    Por eso el nulo responde "mejor que etiquetas aleatorias", no "hay
    estructura": en ruido puro un clusterer que optimiza SSE puede rechazar el
    nulo por sesgo de seleccion. Ver §6 del documento.

    Nunca lanza: devuelve ``ok=False`` + ``reason`` y campos a NaN.
    """
    out: Dict = {
        "ok": False,
        "reason": "",
        "observed": NAN,
        "null_mean": NAN,
        "null_std": NAN,
        "z": NAN,
        "p_value": NAN,
        "n_perm": int(n_perm),
        "n_valid": 0,
        "n_invalid": 0,
        "q01": NAN,
        "q05": NAN,
        "q50": NAN,
        "q95": NAN,
        "q99": NAN,
    }
    A, y = _matrix(X), _labels(labels)
    if A is None or y is None:
        out["reason"] = "X_o_labels_invalidos"
        return out
    if A.shape[0] != y.shape[0]:
        out["reason"] = "longitudes_distintas"
        return out
    n = A.shape[0]
    if n < int(min_n):
        out["reason"] = f"n={n}_menor_que_min_n={int(min_n)}"
        return out
    if not callable(stat_fn):
        out["reason"] = "stat_fn_no_es_callable"
        return out
    n_perm = int(n_perm)
    if n_perm < 1:
        out["reason"] = "n_perm<1"
        return out
    try:
        observed = float(stat_fn(A, y))
    except Exception:
        observed = NAN
    if not np.isfinite(observed):
        out["reason"] = "estadistico_observado_NaN"
        return out
    out["observed"] = observed
    rng = np.random.default_rng(seed)
    nulls: List[float] = []
    invalid = 0
    with _thread_limits(threads), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_perm):
            try:
                value = float(stat_fn(A, rng.permutation(y)))
            except Exception:
                invalid += 1
                continue
            if np.isfinite(value):
                nulls.append(value)
            else:
                invalid += 1
    out["n_invalid"] = int(invalid)
    if not nulls:
        out["reason"] = "todos_los_nulos_NaN"
        return out
    null = np.asarray(nulls, dtype=float)
    mean = float(null.mean())
    std = float(null.std(ddof=1)) if null.size > 1 else 0.0
    out["null_mean"] = mean
    out["null_std"] = std
    out["z"] = float((observed - mean) / std) if std > 0 else NAN
    out["p_value"] = float((1.0 + float(np.sum(null >= observed))) / (null.size + 1.0))
    qs = np.quantile(null, [0.01, 0.05, 0.50, 0.95, 0.99])
    out["q01"], out["q05"], out["q50"], out["q95"], out["q99"] = (
        float(qs[0]),
        float(qs[1]),
        float(qs[2]),
        float(qs[3]),
        float(qs[4]),
    )
    out["n_valid"] = int(null.size)
    out["ok"] = True
    return out


def stability_report(
    X,
    labels,
    *,
    predict_fn: Optional[Callable] = None,
    stat_fn: Callable = r2_dense,
    space: str = "desconocido",
    n_boot: int = 20,
    frac: float = 0.8,
    replace: bool = False,
    n_perm: int = 200,
    seed: int = 42,
    min_n: int = 30,
    alpha: float = 0.05,
    max_n: Optional[int] = None,
    fallback_fn: Optional[Callable] = None,
    threads: int = 1,
    with_bootstrap: bool = True,
    with_null: bool = True,
) -> Dict:
    """Agrega bootstrap + nulo en un dict serializable (apto para JSON).

    ``stat_name`` identifica el estadistico (``__name__`` del callable) y
    ``space`` el espacio de representacion ("pca30_euclidean", "fasttext300",
    ...). Ambos viajan al log/paper para que ninguna cifra quede huerfana de
    su metrica y de su geometria.

    El dict tiene: ``ok``, ``reason``, ``space``, ``stat_name``, ``n``, ``d``,
    ``k``, ``n_noise``, ``bootstrap`` (o None) y ``null`` (o None). ``ok`` es
    True si el bloque solicitado que existe termino bien; ``reason`` acumula
    los motivos de los bloques que fallaron.

    Nunca lanza.
    """
    A, y = _matrix(X), _labels(labels)
    rep: Dict = {
        "ok": False,
        "reason": "",
        "space": str(space),
        "stat_name": str(getattr(stat_fn, "__name__", type(stat_fn).__name__)),
        "n": 0,
        "d": 0,
        "k": 0,
        "n_noise": 0,
        "bootstrap": None,
        "null": None,
    }
    if A is None or y is None:
        rep["reason"] = "X_o_labels_invalidos"
        return rep
    if A.shape[0] != y.shape[0]:
        rep["reason"] = "longitudes_distintas"
        return rep
    rep["n"] = int(A.shape[0])
    rep["d"] = int(A.shape[1])
    classes = np.unique(y[y >= 0])
    rep["k"] = int(classes.size)
    rep["n_noise"] = int(np.sum(y < 0))
    reasons: List[str] = []
    ran = False
    if with_bootstrap and predict_fn is not None:
        boot = bootstrap_ari(
            predict_fn,
            A,
            y,
            n_boot=n_boot,
            frac=frac,
            replace=replace,
            seed=seed,
            min_n=min_n,
            alpha=alpha,
            max_n=max_n,
            fallback_fn=fallback_fn,
            threads=threads,
        )
        rep["bootstrap"] = boot
        ran = True
        if not boot["ok"]:
            reasons.append("bootstrap:" + str(boot["reason"]))
    if with_null:
        null = permutation_null(
            stat_fn,
            A,
            y,
            n_perm=n_perm,
            seed=seed,
            min_n=min_n,
            alpha=alpha,
            threads=threads,
        )
        rep["null"] = null
        ran = True
        if not null["ok"]:
            reasons.append("null:" + str(null["reason"]))
    rep["ok"] = bool(ran and not reasons)
    rep["reason"] = ";".join(reasons)
    return rep


# ══════════════════════════════════════════════════════════════════════════
# Backends (callables con el mismo K)
# ══════════════════════════════════════════════════════════════════════════


def make_predict(backend: str, k: int, floor: int) -> Callable:
    """Clausura ``predict(X_sub) -> labels`` con K y random_state fijos."""
    k = int(k)

    if backend == "ward":

        def _f(Xs: np.ndarray) -> np.ndarray:
            return arena.ward_backend(Xs, k)[0]

        return _f
    if backend == "kmeans":

        def _f(Xs: np.ndarray) -> np.ndarray:
            return arena.kmeans_spherical(Xs, k)[0]

        return _f
    if backend == "dense_cdh":

        def _f(Xs: np.ndarray) -> np.ndarray:
            return arena.dense_cdh(Xs, k, floor, min_r2_dense=0.0)[0]

        return _f
    raise ValueError(f"backend desconocido: {backend!r}")


# ══════════════════════════════════════════════════════════════════════════
# Tablas
# ══════════════════════════════════════════════════════════════════════════


def _f(x: float, nd: int = 3) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _p(x: float) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


def _row(cells: List[str], widths: List[int]) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def table_main(rows: List[Dict]) -> None:
    head = [
        "dataset",
        "backend",
        "stat",
        "obs",
        "null mean+-sd",
        "z",
        "p",
        "ARI mean",
        "IC95",
        "n_eff",
        "fail",
        "t(s)",
    ]
    widths = [14, 9, 8, 7, 20, 7, 6, 8, 15, 6, 5, 6]
    print(_row(head, widths))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        b = r["rep"]["bootstrap"] or {}
        nu = r["rep"]["null"] or {}
        ic = ""
        if b.get("ok"):
            ic = f"[{_f(b['ci_low'], 2)},{_f(b['ci_high'], 2)}]"
        print(
            _row(
                [
                    r["dataset"],
                    r["backend"],
                    r["rep"]["stat_name"][:8],
                    _f(nu.get("observed"), 3),
                    f"{_f(nu.get('null_mean'), 3)} +- {_f(nu.get('null_std'), 3)}",
                    _f(nu.get("z"), 1),
                    _p(nu.get("p_value")),
                    _f(b.get("mean"), 3),
                    ic,
                    _f(b.get("n_eff"), 0),
                    str(b.get("n_failed", "-")) + "/" + str(b.get("n_boot", "-")),
                    _f(r["t"], 2),
                ],
                widths,
            )
        )


def table_verdict(rows: List[Dict]) -> None:
    """Regla medida (ARI bootstrap, NO el p-valor del nulo):

        ESTABLE    <=> ci_low            >= 0.50
        SIN_SENAL  <=> mean              <  0.30
        FRAGIL     el resto

    El p-valor del nulo se imprime al lado como COLUMNA DE CONTROL: vale
    0.005 en ruido puro (sesgo de seleccion, §6), asi que no puede ser el
    gate. Los umbrales 0.50/0.30 no son magicos: en este banco el ruido da
    0.09-0.11 y la estructura 0.83-0.96, con el corpus real en 0.45-0.57.
    """
    head = [
        "dataset",
        "backend",
        "p (control)",
        "ARI mean",
        "IC_lo",
        "motor dice",
        "esperado",
        "acierto",
    ]
    widths = [12, 9, 11, 8, 6, 10, 14, 7]
    print(_row(head, widths))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    hit = tot = 0
    for r in rows:
        b = r["rep"]["bootstrap"] or {}
        nu = r["rep"]["null"] or {}
        p_val = nu.get("p_value", NAN)
        mean = b.get("mean", NAN)
        lo = b.get("ci_low", NAN)
        if b.get("ok") and np.isfinite(lo) and lo >= 0.50:
            says = "ESTABLE"
        elif b.get("ok") and np.isfinite(mean) and mean < 0.30:
            says = "SIN_SENAL"
        else:
            says = "FRAGIL"
        if r["dataset"].startswith("blobs"):
            ok = says == "ESTABLE"
            expected = "ESTABLE"
        elif r["dataset"] == "synth_null":
            ok = says == "SIN_SENAL"
            expected = "SIN_SENAL"
        else:
            ok = says in ("FRAGIL", "ESTABLE")
            expected = "FRAGIL|ESTABLE"
        hit += int(ok)
        tot += 1
        print(
            _row(
                [
                    r["dataset"],
                    r["backend"],
                    _p(p_val),
                    _f(mean, 2),
                    _f(lo, 2),
                    says,
                    expected,
                    "[ok]" if ok else "[X]",
                ],
                widths,
            )
        )
    print(f"acierto de la regla: {hit}/{tot}")


def table_blas(items: List[Dict]) -> None:
    head = ["backend", "n", "hilos", "iguales", "n_distintas", "ARI(1 vs 4)", "t(s)"]
    widths = [14, 5, 6, 7, 11, 11, 6]
    print(_row(head, widths))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for it in items:
        print(
            _row(
                [
                    it["backend"],
                    str(it["n"]),
                    f"{it['t1']} vs {it['t4']}",
                    "[ok]" if it["equal"] else "[X]",
                    str(it["n_diff"]),
                    _f(it["ari"], 4),
                    _f(it["t"], 2),
                ],
                widths,
            )
        )


# ══════════════════════════════════════════════════════════════════════════
# Programa
# ══════════════════════════════════════════════════════════════════════════


def build_datasets() -> List[Dict]:
    sets = [
        ("blobs_s26", arena.synth_blobs(sep=2.6), 8, True),
        ("blobs_s12", arena.synth_blobs(sep=1.2), 8, True),
        ("synth_null", arena.synth_null(), 8, False),
    ]
    if os.path.exists(arena.DATA_PATH):
        try:
            real = load_real_ctx()
            sets.append(("real_ctx", real, 7, True))
        except Exception as exc:
            print(f"[!] real_ctx no cargado: {exc}")
    else:
        print("[!] data/workflow_data.json ausente: se omite real_ctx")
    out = []
    for name, ds, k, truth in sets:
        X = np.asarray(ds["X"], dtype=np.float64)
        out.append(
            {
                "name": name,
                "X": X,
                "k": int(k),
                "truth": bool(truth),
                "note": ds.get("note", ""),
            }
        )
    return out


def pca30_numpy(X: np.ndarray, n_components: int = 30) -> np.ndarray:
    """PCA por SVD de numpy: mismo subespacio que PCA(30, random_state=42).

    Se evita ``sklearn.decomposition`` porque su PRIMER import en este entorno
    cuesta ~18 s (Windows Defender); el bench debe caber en 90 s. La proyeccion
    difiere de sklearn a lo sumo en el signo de cada componente, que no afecta
    a ninguna distancia ni a ningun cluster.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] <= n_components:
        return X
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:n_components].T


def load_real_ctx(path: str = arena.DATA_PATH) -> Dict:
    """``data/workflow_data.json`` en SOLO LECTURA -> matriz PCA(30)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    uces = data.get("uces") or []
    rows = [
        u["embedding"]
        for u in uces
        if isinstance(u.get("embedding"), list) and u["embedding"]
    ]
    if not rows:
        raise RuntimeError("workflow_data.json sin embeddings utilizables")
    X = np.asarray(rows, dtype=np.float64)
    return {
        "name": "real_ctx",
        "X": pca30_numpy(X),
        "note": (
            f"{len(rows)}/{len(uces)} UCEs con embedding "
            f"({X.shape[1]}-d SentenceTransformer; Method C usa fastText 300-d)"
        ),
    }


def floor_for(n: int, k: int) -> int:
    return max(5, int(n / (2.0 * k)))


def sanity_stats() -> None:
    """Comprueba las versiones vectorizadas contra la formula literal y sklearn."""
    X = np.asarray(arena.synth_blobs(sep=2.6)["X"], dtype=float)
    y = np.asarray(arena.ward_backend(X, 8)[0]).astype(int)
    mu = X.mean(axis=0, keepdims=True)
    tss = float(np.sum((X - mu) ** 2))
    sse = 0.0
    for c in np.unique(y):
        Ac = X[y == c]
        sse += float(np.sum((Ac - Ac.mean(axis=0, keepdims=True)) ** 2))
    ref_r2 = 1.0 - sse / tss
    got_r2 = r2_dense(X, y)
    from scipy.spatial.distance import cdist
    from sklearn.metrics import silhouette_score

    D = cdist(X, X, metric="euclidean")
    ref_sil = float(silhouette_score(D, y, metric="precomputed"))
    got_sil = _silhouette_precomputed(D, y)
    print(
        f"[check] r2_dense {got_r2:.6f} vs formula {ref_r2:.6f} -> "
        f"{'[ok]' if abs(got_r2 - ref_r2) < 1e-9 else '[X]'}"
    )
    print(
        f"[check] silhouette {got_sil:.6f} vs sklearn {ref_sil:.6f} -> "
        f"{'[ok]' if abs(got_sil - ref_sil) < 1e-9 else '[X]'}"
    )


def run_main(datasets: List[Dict], backends: List[str], stat_kind: str) -> List[Dict]:
    rows: List[Dict] = []
    for ds in datasets:
        X, k = ds["X"], ds["k"]
        floor = floor_for(X.shape[0], k)
        stat = r2_dense if stat_kind == "r2" else silhouette_precomputed(X, "euclidean")
        for backend in backends:
            pred = make_predict(backend, k, floor)
            try:
                labels = np.asarray(pred(X)).ravel().astype(int)
            except Exception as exc:
                print(f"[X] fit inicial {ds['name']}/{backend}: {type(exc).__name__}")
                continue
            t0 = time.perf_counter()
            rep = stability_report(
                X,
                labels,
                predict_fn=pred,
                stat_fn=stat,
                space="pca30_euclidean" if X.shape[1] == 30 else "synthetic_euclidean",
                n_boot=N_BOOT_MATRIX,
                n_perm=N_PERM,
                seed=SEED,
                min_n=MIN_N,
                threads=1,
            )
            dt = time.perf_counter() - t0
            rows.append(
                {
                    "dataset": ds["name"],
                    "backend": backend,
                    "rep": rep,
                    "t": dt,
                    "truth": ds["truth"],
                    "X": X,
                    "labels": labels,
                }
            )
    return rows


def run_silhouette_only(
    datasets: List[Dict], backend: str, base_rows: List[Dict]
) -> List[Dict]:
    """Nulo con silhouette; reutiliza el bootstrap ya medido con r2_dense."""
    by_key = {(r["dataset"], r["backend"]): r for r in base_rows}
    rows: List[Dict] = []
    for ds in datasets:
        base = by_key.get((ds["name"], backend))
        if base is None:
            continue
        X = ds["X"]
        t0 = time.perf_counter()
        rep = stability_report(
            X,
            base["labels"],
            predict_fn=None,
            stat_fn=silhouette_precomputed(X, "euclidean"),
            space="synthetic_euclidean",
            n_perm=N_PERM_SIL,
            seed=SEED,
            min_n=MIN_N,
            threads=1,
            with_bootstrap=False,
        )
        rep["bootstrap"] = base["rep"]["bootstrap"]
        rows.append(
            {
                "dataset": ds["name"],
                "backend": backend,
                "rep": rep,
                "t": time.perf_counter() - t0,
                "truth": ds["truth"],
            }
        )
    return rows


def run_cost_curves(ds: Dict, backend: str) -> None:
    X, k = ds["X"], ds["k"]
    floor = floor_for(X.shape[0], k)
    pred = make_predict(backend, k, floor)
    labels = np.asarray(pred(X)).ravel().astype(int)
    stat = r2_dense

    print()
    print(f"CURVA B (n_boot) -- {ds['name']}/{backend}, n_perm={N_PERM}, stat=r2_dense")
    head = [
        "B",
        "t_boot(s)",
        "t_null(s)",
        "ARI mean",
        "sd",
        "IC95 ancho",
        "|mean-mean20|",
    ]
    widths = [4, 10, 10, 9, 7, 11, 15]
    print(_row(head, widths))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    means = {}
    for B in (5, 10, 20):
        t0 = time.perf_counter()
        boot = bootstrap_ari(
            pred, X, labels, n_boot=B, seed=SEED, min_n=MIN_N, threads=1
        )
        dt = time.perf_counter() - t0
        means[B] = boot["mean"]
        print(
            _row(
                [
                    str(B),
                    _f(dt, 3),
                    "-",
                    _f(boot["mean"], 3),
                    _f(boot["std"], 3),
                    _f(boot["ci_high"] - boot["ci_low"], 3) if boot["ok"] else "nan",
                    "-",
                ],
                widths,
            )
        )
    for B in (5, 10):
        ref = means.get(20, NAN)
        print(
            f"  B={B:>2}: |mean-mean20| = {_f(abs(means[B] - ref), 4) if np.isfinite(ref) else 'nan'}"
        )

    print()
    print(f"CURVA n_perm -- {ds['name']}/{backend}, stat=r2_dense")
    head = ["n_perm", "t(s)", "p", "null mean", "null sd", "z"]
    widths = [7, 8, 8, 10, 9, 8]
    print(_row(head, widths))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for P in (50, 100, 200):
        t0 = time.perf_counter()
        nu = permutation_null(
            stat, X, labels, n_perm=P, seed=SEED, min_n=MIN_N, threads=1
        )
        dt = time.perf_counter() - t0
        print(
            _row(
                [
                    str(P),
                    _f(dt, 3),
                    _p(nu["p_value"]),
                    _f(nu["null_mean"], 4),
                    _f(nu["null_std"], 4),
                    _f(nu["z"], 1),
                ],
                widths,
            )
        )


def run_blas(datasets: List[Dict], backends: List[str]) -> List[Dict]:
    from threadpoolctl import threadpool_info, threadpool_limits

    print()
    print("DETERMINISMO BLAS (mismo X, mismo seed, distinto numero de hilos)")
    print(
        "threadpool_info antes:",
        [(t.get("internal_api"), t.get("num_threads")) for t in threadpool_info()],
    )
    items: List[Dict] = []
    for ds in datasets:
        if ds["name"] not in ("blobs_s12", "real_ctx"):
            continue
        X, k = ds["X"], ds["k"]
        floor = floor_for(X.shape[0], k)
        for backend in backends:
            pred = make_predict(backend, k, floor)
            try:
                with threadpool_limits(limits=1):
                    t0 = time.perf_counter()
                    l1 = np.asarray(pred(X)).ravel().astype(int)
                    t1 = time.perf_counter() - t0
                with threadpool_limits(limits=4):
                    t0 = time.perf_counter()
                    l4 = np.asarray(pred(X)).ravel().astype(int)
                    t4 = time.perf_counter() - t0
            except Exception as exc:
                print(f"[!] BLAS {ds['name']}/{backend}: {type(exc).__name__}")
                continue
            ari = _ari_pair(l1, l4)
            items.append(
                {
                    "backend": f"{ds['name']}/{backend}"
                    if len(backends) > 1
                    else backend,
                    "n": X.shape[0],
                    "t1": 1,
                    "t4": 4,
                    "equal": bool(np.array_equal(l1, l4)),
                    "n_diff": int(np.sum(l1 != l4)),
                    "ari": float(ari) if ari is not None else NAN,
                    "t": t1 + t4,
                }
            )
    print(
        "threadpool_info despues:",
        [(t.get("internal_api"), t.get("num_threads")) for t in threadpool_info()],
    )
    return items


def run_robustness() -> None:
    print()
    print("ROBUSTEZ (nada debe lanzar)")
    head = ["caso", "ok", "motivo", "detalle"]
    widths = [22, 5, 30, 40]
    print(_row(head, widths))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    def show(name: str, rep: Dict, detail: str = "") -> None:
        print(
            _row(
                [
                    name,
                    "[ok]" if rep.get("ok") else "no",
                    str(rep.get("reason", ""))[:30],
                    detail[:40],
                ],
                widths,
            )
        )

    rng = np.random.default_rng(SEED)

    X_small = rng.normal(size=(20, 30))
    y_small = np.array([0, 1] * 10)
    show(
        "n<30",
        stability_report(X_small, y_small, stat_fn=r2_dense, n_perm=10, min_n=30),
    )

    X = rng.normal(size=(120, 30))
    y_one = np.zeros(120, dtype=int)
    show("una sola clase", stability_report(X, y_one, stat_fn=r2_dense, n_perm=10))

    pred_big = make_predict("kmeans", 100, 5)
    y_ok = np.arange(120) % 4
    show(
        "K>n en predict_fn",
        stability_report(
            X, y_ok, predict_fn=pred_big, stat_fn=r2_dense, n_boot=3, n_perm=10
        ),
        "kmeans k=100, n_sub=96",
    )

    X_zero = np.zeros((60, 30))
    y_z = np.arange(60) % 3
    show(
        "vectores cero",
        stability_report(X_zero, y_z, stat_fn=r2_dense, n_perm=20),
        "TSS=0 -> NaN",
    )

    y_noise = np.where(np.arange(235) % 10 == 0, -1, np.arange(235) % 5)
    show(
        "etiquetas con -1",
        stability_report(
            rng.normal(size=(235, 30)),
            y_noise,
            predict_fn=make_predict("kmeans", 5, 20),
            stat_fn=r2_dense,
            n_boot=3,
            n_perm=20,
        ),
        "-1 excluido",
    )

    show(
        "guarda ward+n<max_n",
        stability_report(
            X,
            y_ok,
            predict_fn=make_predict("ward", 4, 5),
            stat_fn=r2_dense,
            n_boot=4,
            n_perm=10,
            max_n=10,
            fallback_fn=make_predict("kmeans", 4, 5),
        ),
        "n_sub=96>10 -> kmeans",
    )
    show(
        "guarda ward sin fallback",
        stability_report(
            X,
            y_ok,
            predict_fn=make_predict("ward", 4, 5),
            stat_fn=r2_dense,
            n_boot=4,
            n_perm=10,
            max_n=10,
        ),
        "memory_guard",
    )

    print()
    print("r2_dense en casos borde:")
    print("  X(3x2) identicas      ->", _f(r2_dense(np.ones((3, 2)), [0, 0, 1]), 4))
    print(
        "  labels 1-D de longitud distinta ->", _f(r2_dense(np.ones((3, 2)), [0, 1]), 4)
    )
    print(
        "  labels con -1 (2 clases) ->",
        _f(r2_dense(rng.normal(size=(60, 3)), y_noise[:60]), 4),
    )


def run_engine_determinism(ds: Dict, backend: str) -> None:
    X, k = ds["X"], ds["k"]
    floor = floor_for(X.shape[0], k)
    pred = make_predict(backend, k, floor)
    labels = np.asarray(pred(X)).ravel().astype(int)
    kw = dict(
        predict_fn=pred,
        stat_fn=r2_dense,
        n_boot=5,
        n_perm=50,
        seed=SEED,
        min_n=MIN_N,
        threads=1,
    )
    a = stability_report(X, labels, **kw)
    b = stability_report(X, labels, **kw)
    same = (
        a["bootstrap"]["ari"] == b["bootstrap"]["ari"]
        and a["null"]["p_value"] == b["null"]["p_value"]
        and a["null"]["observed"] == b["null"]["observed"]
    )
    print(
        f"repetibilidad del motor con seed=42 ({ds['name']}/{backend}): "
        f"{'[ok] identico' if same else '[X] difiere'}"
    )


def run_budget(rows: List[Dict], ds: Dict, reps: int = 1) -> None:
    """Coste del motor con la configuracion dura B=20 / n_perm=200 (min de reps)."""
    print()
    print(
        f"PRESUPUESTO B={N_BOOT} n_perm={N_PERM} r2_dense -- {ds['name']} "
        f"(n={ds['X'].shape[0]}, {reps} reps)"
    )
    head = ["backend", "rep1(s)", "rep2(s)", "min(s)", "presupuesto"]
    widths = [9, 8, 8, 8, 12]
    print(_row(head, widths))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for backend in ("ward", "kmeans", "dense_cdh"):
        base = next(
            (r for r in rows if r["dataset"] == ds["name"] and r["backend"] == backend),
            None,
        )
        if base is None:
            continue
        X, y = base["X"], base["labels"]
        pred = make_predict(backend, ds["k"], floor_for(X.shape[0], ds["k"]))
        ts: List[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            stability_report(
                X,
                y,
                predict_fn=pred,
                stat_fn=r2_dense,
                space="synthetic_euclidean",
                n_boot=N_BOOT,
                n_perm=N_PERM,
                seed=SEED,
                min_n=MIN_N,
                threads=1,
            )
            ts.append(time.perf_counter() - t0)
        cells = [_f(t, 2) for t in ts]
        while len(cells) < 2:
            cells.append("-")
        best = min(ts)
        print(
            _row(
                [
                    backend,
                    cells[0],
                    cells[1],
                    _f(best, 2),
                    "[ok]" if best <= BUDGET_S else "[X] excede",
                ],
                widths,
            )
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="bench del motor de estabilidad")
    ap.add_argument(
        "--with-reii",
        action="store_true",
        help="importar reii.quality.metrics (12.6 s)",
    )
    ap.add_argument("--quick", action="store_true", help="menos datasets y curvas")
    args = ap.parse_args()

    t_start = time.perf_counter()
    t_mark = [t_start]

    def mark(label: str) -> None:
        now = time.perf_counter()
        print(f"[t] {label:<24} {now - t_mark[0]:6.1f}s")
        t_mark[0] = now

    print("=" * 78)
    print("MOTOR DE ESTABILIDAD -- bench (sin red, sin pipeline, solo lectura)")
    print(
        f"numpy {np.__version__} | seed={SEED} | B_matriz={N_BOOT_MATRIX} | "
        f"B_presupuesto={N_BOOT} | n_perm={N_PERM} | budget={BUDGET_S}s"
    )
    try:
        import threadpoolctl

        print(f"threadpoolctl {threadpoolctl.__version__} [ok]")
    except Exception as exc:
        print(f"[!] sin threadpoolctl ({exc}): determinismo BLAS NO garantizado")

    if args.with_reii:
        t0 = time.perf_counter()
        try:
            from reii.quality.metrics import silhouette_labels

            print(f"reii.quality.metrics importado en {time.perf_counter() - t0:.1f} s")
            Xs = np.random.default_rng(SEED).normal(size=(80, 5))
            ys = np.arange(80) % 4
            D = arena.pca30(Xs)
            v_proj = silhouette_labels(
                D, ys, metric="euclidean", sample_size=None, seed=SEED
            )
            v_pre = silhouette_precomputed(D, "euclidean")(D, ys)
            print(
                f"silhouette proyecto={_f(v_proj, 6)} precomputada={_f(v_pre, 6)} "
                f"=> {'[ok] coinciden' if abs(v_proj - v_pre) < 1e-9 else '[!] difieren'}"
            )
        except Exception as exc:
            print(f"[!] --with-reii fallo: {type(exc).__name__}: {exc}")

    datasets = build_datasets()
    if args.quick:
        datasets = [d for d in datasets if d["name"] in ("blobs_s26", "synth_null")]
    for d in datasets:
        print(
            f"  dataset {d['name']}: n={d['X'].shape[0]} d={d['X'].shape[1]} K={d['k']} "
            f"verdad={'estructura' if d['truth'] else 'ruido'} | {d['note']}"
        )
    mark("datasets")

    backends = ["ward", "kmeans", "dense_cdh"]

    sanity_stats()
    mark("sanity")

    print()
    print("TABLA 1 -- r2_dense (B=5 en la matriz; B=20 en la tabla de presupuesto)")
    rows_r2 = run_main(datasets, backends, "r2")
    table_main(rows_r2)
    mark("tabla 1 r2")

    print()
    print("TABLA 1b -- silhouette precomputada (n_perm=100; reutiliza el bootstrap)")
    sub = [d for d in datasets if d["name"] in ("blobs_s26", "synth_null")]
    rows_sil = run_silhouette_only(sub, "ward", rows_r2)
    table_main(rows_sil)
    mark("tabla 1b silhouette")

    print()
    print("VEREDICTO estructura vs ruido (r2_dense)")
    table_verdict(rows_r2)

    print()
    print("TABLA 2 -- coste/precision (solo kmeans: es el backend que no cabe)")
    ds_cost = next(d for d in datasets if d["name"] == "blobs_s26")
    run_cost_curves(ds_cost, "kmeans")
    mark("tabla 2 curvas")

    print()
    print("TABLA 3 -- determinismo BLAS")
    blas = run_blas(datasets, ["ward", "kmeans", "dense_cdh"])
    table_blas(blas)
    ident = sum(1 for it in blas if it["equal"])
    print(f"etiquetas identicas con 1 vs 4 hilos: {ident}/{len(blas)}")
    mark("tabla 3 BLAS")

    run_engine_determinism(ds_cost, "ward")
    run_robustness()
    mark("robustez")

    run_budget(rows_r2, ds_cost, reps=1)

    print()
    print(f"[total] {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
