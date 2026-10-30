#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Banco de pruebas del ENDURECIMIENTO del backend denso de Method C.

Acompana a ``docs/design_v5/HARDENING.md`` (riesgos residuales de BACKEND_C.md).
Mide, sin red y sin pipeline:

  1. determinismo de ``kmeans_spherical`` con 1 hilo vs N hilos
     (``threadpoolctl``) sobre sinteticos y sobre el corpus real de solo lectura;
  2. pico de RAM de ``ward`` para n = 528 / 1000 / 2000 y extrapolacion a 5800;
  3. ``target_k`` vs ``n_clusters`` devuelto por ``dense_cdh`` en blobs y en
     ruido puro (control negativo), con el gate ``min_r2_dense=0.08``;
  4. matriz de precondiciones n x K: que backend devuelve None y por que.

QUE SE IMPORTA Y QUE NO
-----------------------
Prohibido importar ``reii.main_workflow`` (>4 min por Stanza). Este script NO
lo hace. Las matematicas de los cuatro backends se reutilizan de
``scripts/playground/c_backends_arena.py`` (misma especificacion ya medida).

La seccion "COPIA CANONICA" contiene las funciones de politica que el diff de
``src/reii/method_c_policy.py`` debe llevar VERBATIM. El script existe porque
las reglas de trabajo prohiben escribir en ``src/**``; cuando se aplique el
diff, ``tests/test_backend_hardening.py`` puede apuntar al modulo real con el
mismo import (``resolve_method_c_backend``, ``preflight``, ...) y los numeros no
cambian.

DETERMINISMO
------------
La arena fija ``OMP/OPENBLAS/MKL_NUM_THREADS=1`` al importarse (antes de numpy).
Produccion no lo hace. Por eso aqui la comparacion es explicita y no depende
del entorno: ``threadpool_limits(1)`` vs ``threadpool_limits(N_CORES)``
alrededor del fit, verificado con ``threadpool_info()`` (columna ``hilos``).
La rama "sin limite" del informe NO es "sin context manager" (el entorno del
playground ya arranca en 1 hilo), es "N hilos reales".

Uso:
    python scripts/playground/hardening_bench.py
    python scripts/playground/hardening_bench.py --reps 8 --skip-real
Salida ASCII puro (consola cp1252), exit 0.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# La arena fija los hilos a 1 ANTES de numpy (su intencion, no la nuestra).
# Se importa la primera para no romper esa secuencia.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
from c_backends_arena import (  # noqa: E402
    dense_cdh,
    hdbscan_backend,
    kmeans_spherical,
    load_real_dataset,
    synth_blobs,
    synth_null,
    ward_backend,
)

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
SEED = 42
PWD = r"[ok]"
WARN = r"[!]"
FAIL = r"[X]"

# ══════════════════════════════════════════════════════════════════════════
# COPIA CANONICA — politica de Method C (destino: src/reii/method_c_policy.py)
# ══════════════════════════════════════════════════════════════════════════

METHOD_C_BACKENDS: Tuple[str, ...] = (
    "hdbscan",
    "dense_cdh",
    "ward",
    "kmeans_spherical",
)
DEFAULT_METHOD_C_BACKEND = "hdbscan"

#: Fuentes de donde puede venir el valor. Las persistidas degradan (un
#: best_params.json viejo no debe abortar una corrida de 2 h); las internas
#: (constantes del programador) siguen lanzando ValueError.
PERSISTED_SOURCES: Tuple[str, ...] = (
    "persisted",
    "best_params",
    "config_store",
    "cli",
    "database",
)
INTERNAL_SOURCES: Tuple[str, ...] = ("internal", "constant")

K_MIN = 2  # gate duro de _evaluate_trial: n_clusters >= 2
K_MAX = 20  # gate duro de _evaluate_trial: n_clusters <= 20
K_GATE_MIN = 3  # method_selection_k_bounds[0]
K_GATE_MAX = 20  # method_selection_k_bounds[1]
ZERO_ROW_MAX_FRAC = 0.5  # espejo de main_workflow.py:5124-5131

#: Calibrado con measure_ward_memory() sobre esta maquina (ver HARDENING.md
#: §2). El presupuesto es 256 MB de pico ADICIONAL; con el ajuste cuadratico
#: medido, 4000 es el mayor multiplo de 500 que cabe con margen.
DEFAULT_WARD_MAX_N = 4000
DEFAULT_WARD_MAX_BYTES = 256 * 1024 * 1024

DEFAULT_FLOOR = 5  # hdbscan_uce_min_cluster_floor
DEFAULT_MCS_FRAC = 0.08  # hdbscan_uce_min_cluster_frac (tight)


def _as_text(value: Any) -> str:
    """Texto sin comillas extra para los avisos (``None`` -> 'None')."""
    if isinstance(value, str):
        return value
    return str(value)


def resolve_method_c_backend(
    value: Any, *, source: str = "persisted"
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Normaliza ``method_c_backend`` y decide que hacer con un valor invalido.

    Reglas (HARDENING.md §4):

    * ``None`` / cadena vacia -> ``DEFAULT_METHOD_C_BACKEND``, sin aviso
      (equivale a "no configurado", no a "configurado mal").
    * valor valido -> el valor normalizado (``strip().lower()``), sin aviso.
    * fuente PERSISTIDA (``best_params.json``, tabla ``workflow_config``, CLI)
      -> degrada a ``hdbscan`` y devuelve un aviso ``[!]``. Nunca aborta.
    * fuente INTERNA (constante del programador) -> ``ValueError`` ruidoso:
      un literal mal escrito en el codigo es un bug, no un dato.

    Devuelve ``(backend, notice|None)``. ``notice`` es un dict serializable
    (se registra en ``diagnostics["method_c"]["backend_resolution"]``).
    """
    if source not in PERSISTED_SOURCES and source not in INTERNAL_SOURCES:
        raise ValueError(
            f"source debe ser una de {PERSISTED_SOURCES + INTERNAL_SOURCES}, "
            f"no {source!r}"
        )

    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_METHOD_C_BACKEND, None

    norm = value.strip().lower() if isinstance(value, str) else value
    if norm in METHOD_C_BACKENDS:
        return str(norm), None

    if source in INTERNAL_SOURCES:
        raise ValueError(
            f"method_c_backend invalido: {value!r} "
            f"(esperado uno de {', '.join(sorted(METHOD_C_BACKENDS))})"
        )

    message = (
        f'{WARN} method_c_backend="{_as_text(value)}" desconocido '
        f'-> fallback "{DEFAULT_METHOD_C_BACKEND}" (diagnostico)'
    )
    notice = {
        "code": "unknown_backend",
        "requested": _as_text(value),
        "backend": DEFAULT_METHOD_C_BACKEND,
        "source": source,
        "message": message,
    }
    return DEFAULT_METHOD_C_BACKEND, notice


@dataclass(frozen=True)
class Precondition:
    """Resultado de ``preflight``: que va a correr y por que, sin excepciones.

    ``ok=False`` significa "el backend devolvera None" (no hay particion).
    ``ok=True`` con ``backend_effective != backend`` significa degradacion
    (hoy: guard de memoria de ward). ``k_effective != k_requested`` significa
    que el backend no puede llegar al K pedido y lo deja acotado.
    """

    ok: bool
    backend: str
    backend_effective: str
    k_requested: int
    k_effective: Optional[int]
    reason: str
    message: str
    notices: Tuple[str, ...] = ()


def ward_guard(n: int, max_n: int = DEFAULT_WARD_MAX_N) -> Tuple[bool, str]:
    """Guard de memoria de ward. ``True`` = se permite.

    Se degrada a ``dense_cdh`` en vez de abortar: la matriz de distancias
    O(n^2) depende solo de n, no de la hipotesis cientifica del trial; abortar
    por un knob de alternancia perderia la corrida completa. La degradacion se
    registra (``backend_effective``) para que el experimento siga siendo
    auditable.
    """
    if int(n) <= int(max_n):
        return True, ""
    return False, (
        f"{WARN} ward: n={int(n)} > method_c_ward_max_n={int(max_n)} "
        f'-> fallback "dense_cdh" (guard de memoria O(n^2))'
    )


def effective_target_k(
    n: int, target_k: int, floor: int = DEFAULT_FLOOR
) -> Tuple[int, bool]:
    """K viable para dense_cdh: ``min(target_k, K_MAX, n // floor)``.

    Devuelve ``(k_efectivo, acotado)``. ``k_efectivo`` puede ser 1 cuando
    ``2*floor > n`` (la raiz no se puede partir); el llamador lo trata como
    "sin particion", no como una clase.
    """
    floor = max(1, int(floor))
    k_cap = int(n) // floor
    k = int(target_k)
    bounded = k > k_cap or k > K_MAX
    k = min(k, k_cap, K_MAX)
    return max(1, k), bounded


def preflight(
    backend: str,
    n: int,
    target_k: int,
    floor: int = DEFAULT_FLOOR,
    *,
    n_zero_rows: int = 0,
    ward_max_n: int = DEFAULT_WARD_MAX_N,
    mcs_frac: float = DEFAULT_MCS_FRAC,
) -> Precondition:
    """Precondiciones de Method C, sin tocar datos.

    Nunca lanza ``IndexError``/``ValueError`` por n o K: devuelve un
    ``Precondition`` con ``ok=False`` y el mensaje exacto. ``guarded_fit`` es
    el unico punto que llama a los backends, y solo si ``ok``.
    """
    if backend not in METHOD_C_BACKENDS:
        backend, _notice = resolve_method_c_backend(backend, source="persisted")

    n = int(n)
    target_k = int(target_k)
    floor = max(1, int(floor))
    notices: List[str] = []

    def _fail(reason: str, message: str, k_eff: Optional[int] = None):
        return Precondition(
            ok=False,
            backend=backend,
            backend_effective=backend,
            k_requested=target_k,
            k_effective=k_eff,
            reason=reason,
            message=message,
            notices=tuple(notices),
        )

    if n <= 0:
        return _fail(
            "empty_corpus",
            f"{WARN} method_c: corpus vacio (n=0) -> sin particion",
        )

    if n_zero_rows > int(n * ZERO_ROW_MAX_FRAC):
        return _fail(
            "too_many_zero_rows",
            f"{WARN} method_c: {int(n_zero_rows)}/{n} zero-vectors "
            f"({100.0 * n_zero_rows / n:.0f}%) > 50% -> sin particion",
        )

    # ── dense_cdh ────────────────────────────────────────────────────────
    if backend == "dense_cdh":
        if n < 2 * floor:
            return _fail(
                "below_two_floors",
                f"{WARN} dense_cdh: n={n} < 2*floor={2 * floor} "
                f"-> sin particion (K=1 no es viable)",
                1,
            )
        k_eff, bounded = effective_target_k(n, target_k, floor)
        if bounded:
            notices.append(
                f"{WARN} dense_cdh: target_k={target_k} > n//floor="
                f"{n // floor} -> K efectivo={k_eff}"
            )
        if k_eff < K_MIN:
            return _fail(
                "k_below_gate",
                f"{WARN} dense_cdh: K efectivo={k_eff} < {K_MIN} -> sin particion",
                k_eff,
            )
        return Precondition(
            ok=True,
            backend=backend,
            backend_effective=backend,
            k_requested=target_k,
            k_effective=k_eff,
            reason="ok",
            message="",
            notices=tuple(notices),
        )

    # ── ward (con guard de memoria) ───────────────────────────────────────
    if backend == "ward":
        allowed, guard_msg = ward_guard(n, ward_max_n)
        if not allowed:
            eff = "dense_cdh"
            eff_pre = preflight(
                eff,
                n,
                target_k,
                floor,
                n_zero_rows=n_zero_rows,
                ward_max_n=ward_max_n,
                mcs_frac=mcs_frac,
            )
            return Precondition(
                ok=eff_pre.ok,
                backend=backend,
                backend_effective=eff,
                k_requested=target_k,
                k_effective=eff_pre.k_effective,
                reason="ward_memory_guard",
                message=guard_msg,
                notices=tuple(notices) + (guard_msg,) + eff_pre.notices,
            )
        if n < K_MIN:
            return _fail(
                "n_below_kmin",
                f"{WARN} ward: n={n} < n_clusters minimo ({K_MIN}) -> sin particion",
            )
        if target_k > n:
            return _fail(
                "k_above_n",
                f"{WARN} ward: n={n} < target_k={target_k} -> sin particion",
            )
        k_eff = max(K_MIN, min(target_k, n))
        return Precondition(
            ok=True,
            backend=backend,
            backend_effective=backend,
            k_requested=target_k,
            k_effective=k_eff,
            reason="ok",
            message="",
            notices=tuple(notices),
        )

    # ── kmeans_spherical ─────────────────────────────────────────────────
    if backend == "kmeans_spherical":
        if n < K_MIN:
            return _fail(
                "n_below_kmin",
                f"{WARN} kmeans_spherical: n={n} < {K_MIN} -> sin particion",
            )
        if target_k > n:
            return _fail(
                "k_above_n",
                f"{WARN} kmeans_spherical: n={n} < target_k={target_k} "
                f"-> sin particion",
            )
        return Precondition(
            ok=True,
            backend=backend,
            backend_effective=backend,
            k_requested=target_k,
            k_effective=int(target_k),
            reason="ok",
            message="",
            notices=tuple(notices),
        )

    # ── hdbscan (diagnostico) ────────────────────────────────────────────
    mcs = max(floor, int(round(float(mcs_frac) * n)))
    if n < max(2, mcs):
        return _fail(
            "n_below_mcs",
            f"{WARN} hdbscan: n={n} < min_cluster_size={mcs} -> sin particion (K=0)",
        )
    return Precondition(
        ok=True,
        backend=backend,
        backend_effective=backend,
        k_requested=target_k,
        k_effective=None,
        reason="ok",
        message=f"{WARN} hdbscan: K no es controlable (target_k={target_k} "
        f"solo fija min_cluster_size={mcs}); diagnostic_only",
        notices=tuple(notices),
    )


def format_target_k_notice(
    target_k: int, n_clusters: int, rejected_splits: int, backend: str = "dense_cdh"
) -> str:
    """Aviso canonico target_k -> n_clusters (HARDENING.md §1)."""
    if int(n_clusters) >= int(target_k):
        return ""
    return (
        f"{WARN} {backend}: target_k={int(target_k)} -> {int(n_clusters)} clases "
        f"({int(rejected_splits)} splits rechazados por r2_dense)"
    )


def guarded_fit(
    backend: str,
    X: np.ndarray,
    target_k: int,
    floor: int = DEFAULT_FLOOR,
    *,
    n_zero_rows: int = 0,
    ward_max_n: int = DEFAULT_WARD_MAX_N,
    min_r2_1d: float = 0.0,
    min_r2_dense: float = 0.0,
    mcs_frac: float = DEFAULT_MCS_FRAC,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """``preflight`` + backend + validacion de salida. Nunca propaga excepcion.

    Devuelve ``(labels|None, diag)``. ``diag`` siempre trae ``precondition``
    (serializable), ``notices`` y, si corrio, ``backend_used``/``n_clusters``.
    """
    X = np.asarray(X)
    n = int(X.shape[0]) if X.ndim == 2 else 0
    pre = preflight(
        backend,
        n,
        target_k,
        floor,
        n_zero_rows=n_zero_rows,
        ward_max_n=ward_max_n,
        mcs_frac=mcs_frac,
    )
    diag: Dict[str, Any] = {
        "backend_requested": backend,
        "backend_used": pre.backend_effective,
        "precondition": {
            "ok": pre.ok,
            "reason": pre.reason,
            "message": pre.message,
            "k_requested": pre.k_requested,
            "k_effective": pre.k_effective,
        },
        "notices": list(pre.notices),
    }
    if not pre.ok:
        return None, diag

    be = pre.backend_effective
    k_eff = int(pre.k_effective if pre.k_effective is not None else target_k)
    try:
        if be == "dense_cdh":
            labels, bdiag = dense_cdh(
                X,
                k_eff,
                int(floor),
                min_r2_1d=float(min_r2_1d),
                min_r2_dense=float(min_r2_dense),
            )
            diag["backend_diagnostics"] = bdiag
        elif be == "ward":
            labels, bdiag = ward_backend(X, k_eff)
            diag["backend_diagnostics"] = bdiag
        elif be == "kmeans_spherical":
            labels, bdiag = kmeans_spherical(X, k_eff)
            diag["backend_diagnostics"] = bdiag
        else:
            mcs = max(int(floor), int(round(float(mcs_frac) * n)))
            labels, bdiag = hdbscan_backend(X, mcs, max(2, int(round(0.25 * mcs))))
            diag["backend_diagnostics"] = bdiag
    except Exception as exc:  # degradacion explicita, no traza
        msg = f"{WARN} {be}: excepcion controlada {type(exc).__name__}: {exc}"
        diag["precondition"]["ok"] = False
        diag["precondition"]["reason"] = "exception"
        diag["precondition"]["message"] = msg
        return None, diag

    labels = np.asarray(labels).astype(int).ravel()
    if labels.shape[0] != n:
        msg = (
            f"{WARN} {be}: salida invalida (len={labels.shape[0]} != n={n}) "
            f"-> sin particion"
        )
        diag["precondition"]["ok"] = False
        diag["precondition"]["reason"] = "bad_output_length"
        diag["precondition"]["message"] = msg
        return None, diag

    real = np.unique(labels[labels >= 0])
    diag["n_clusters"] = int(len(real))
    diag["n_noise"] = int(np.sum(labels == -1))
    if be == "dense_cdh":
        rejected = int(diag["backend_diagnostics"].get("rejected_min_r2_dense", 0))
        notice = format_target_k_notice(target_k, len(real), rejected)
        if notice:
            diag["notices"].append(notice)
    return labels, diag


# ══════════════════════════════════════════════════════════════════════════
# Sondas
# ══════════════════════════════════════════════════════════════════════════


def _fingerprint(labels: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(labels, dtype=np.int32).tobytes()).hexdigest()[:12]


def _threads_seen() -> int:
    """Hilos reales del controlador BLAS/OpenMP activo (0 si no hay)."""
    try:
        from threadpoolctl import threadpool_info
    except Exception:  # pragma: no cover
        return 0
    info = threadpool_info()
    if not info:
        return 0
    return int(max(int(i.get("num_threads", 0)) for i in info))


def _kmeans_labels(X: np.ndarray, k: int, threads: int) -> np.ndarray:
    """KMeans esferico con el numero de hilos confinado al fit."""
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    Xn = X / norms
    with threadpool_limits(limits=int(threads)):
        km = KMeans(n_clusters=int(k), n_init=10, random_state=SEED)
        return km.fit_predict(Xn).astype(int)


def measure_thread_determinism(
    datasets: Sequence[Dict],
    ks: Sequence[int] = (3, 8, 14),
    reps: int = 20,
    n_threads: int = 0,
) -> List[Dict]:
    """1 hilo vs N hilos, ``reps`` repeticiones por (dataset, K, hilos)."""
    from sklearn.metrics import adjusted_rand_score

    n_threads = int(n_threads or (os.cpu_count() or 1))
    rows: List[Dict] = []
    for ds in datasets:
        X = np.asarray(ds["X"], dtype=np.float64)
        for k in ks:
            if k > X.shape[0]:
                continue
            per_mode: Dict[int, Dict[str, Any]] = {}
            for threads in (1, n_threads):
                fps, seen = [], set()
                first = None
                for _ in range(int(reps)):
                    lab = _kmeans_labels(X, k, threads)
                    seen.add(_threads_seen())
                    fps.append(_fingerprint(lab))
                    if first is None:
                        first = lab
                per_mode[threads] = {
                    "distinct": len(set(fps)),
                    "threads_seen": sorted(seen),
                    "labels": first,
                }
            a = per_mode[1]["labels"]
            b = per_mode[n_threads]["labels"]
            ari = float(adjusted_rand_score(a, b))
            n_diff = int(np.sum(a != b)) if a.shape == b.shape else -1
            rows.append(
                {
                    "dataset": ds["name"],
                    "k": int(k),
                    "reps": int(reps),
                    "distinct_1": per_mode[1]["distinct"],
                    "distinct_n": per_mode[n_threads]["distinct"],
                    "threads_1": per_mode[1]["threads_seen"],
                    "threads_n": per_mode[n_threads]["threads_seen"],
                    "ari_cross": ari,
                    "n_diff_points": n_diff,
                }
            )
    return rows


def _ward_fit(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    ac = AgglomerativeClustering(
        n_clusters=int(k), linkage="ward", compute_distances=True
    )
    return ac.fit_predict(X).astype(int)


def measure_ward_memory(
    ns: Sequence[int] = (528, 1000, 2000), k: int = 8, seed: int = 7
) -> List[Dict]:
    """Pico de RAM ADICIONAL de ward (tracemalloc + RSS como contraste)."""
    rows: List[Dict] = []
    for n in ns:
        ds = synth_blobs(n=int(n), k=k, seed=seed)
        X = np.asarray(ds["X"], dtype=np.float64)
        gc.collect()
        try:
            import psutil

            proc = psutil.Process()
            rss0 = proc.memory_info().rss
        except Exception:  # pragma: no cover
            proc, rss0 = None, 0
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        t0 = time.perf_counter()
        labels = _ward_fit(X, k)
        elapsed = time.perf_counter() - t0
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss1 = proc.memory_info().rss if proc is not None else 0
        rows.append(
            {
                "n": int(n),
                "d": int(X.shape[1]),
                "k": int(k),
                "peak_mb": (peak - base) / 2**20,
                "rss_delta_mb": (rss1 - rss0) / 2**20,
                "seconds": elapsed,
                "n_clusters": int(len(np.unique(labels))),
                "bytes_per_n2": (peak - base) / float(n * n),
            }
        )
    return rows


def extrapolate_ward(rows: Sequence[Dict], target_n: int = 5800) -> Dict[str, Any]:
    """Ajusta ``peak = b * n^a`` con los dos ultimos puntos y predice target_n."""
    usable = [r for r in rows if r["peak_mb"] > 0]
    if len(usable) < 2:
        return {"ok": False, "reason": "menos de 2 puntos medidos"}
    lo, hi = usable[-2], usable[-1]
    a = math.log(hi["peak_mb"] / lo["peak_mb"]) / math.log(hi["n"] / lo["n"])
    b = hi["peak_mb"] / (hi["n"] ** a)
    return {
        "ok": True,
        "exponent": a,
        "peak_mb_at": lambda n, _b=b, _a=a: _b * (int(n) ** _a),
        "predicted_mb": b * (int(target_n) ** a),
        "target_n": int(target_n),
    }


def calibrate_ward_max_n(
    peak_at_n: float,
    n_ref: int,
    exponent: float,
    budget_bytes: int = DEFAULT_WARD_MAX_BYTES,
    step: int = 500,
) -> int:
    """Mayor ``n`` multiplo de ``step`` cuyo pico extrapolado cabe en budget."""
    budget_mb = budget_bytes / 2**20
    b = peak_at_n / (n_ref**exponent)
    n = step
    while b * ((n + step) ** exponent) <= budget_mb:
        n += step
        if n > 200_000:  # pragma: no cover - cordura
            break
    return int(n)


def measure_target_k_drift(
    ks: Sequence[int] = (3, 5, 8, 14), gate: float = 0.08
) -> List[Dict]:
    """``target_k`` vs ``n_clusters`` de dense_cdh, blobs vs ruido puro."""
    rows: List[Dict] = []
    for ds in (synth_blobs(n=528, k=8), synth_null(n=528)):
        X = np.asarray(ds["X"], dtype=np.float64)
        n = X.shape[0]
        for k in ks:
            floor = max(5, int(round(n / (2.0 * k))))
            labels, diag = guarded_fit(
                "dense_cdh",
                X,
                int(k),
                floor,
                min_r2_dense=float(gate),
            )
            bd = diag.get("backend_diagnostics") or {}
            rows.append(
                {
                    "dataset": ds["name"],
                    "target_k": int(k),
                    "floor": int(floor),
                    "ok": labels is not None,
                    "n_clusters": int(diag.get("n_clusters", 0)),
                    "splits": int(bd.get("n_splits", 0)),
                    "rejected_r2_dense": int(bd.get("rejected_min_r2_dense", 0)),
                    "notice": next(
                        (
                            x
                            for x in diag.get("notices", [])
                            if "target_k=" in x and "clases" in x
                        ),
                        "",
                    ),
                }
            )
    return rows


def precondition_matrix(
    ns: Sequence[int] = (0, 1, 2, 7, 50, 528),
    ks: Sequence[int] = (1, 2, 3, 8, 21),
    floor: int = DEFAULT_FLOOR,
) -> List[Dict]:
    """Ejecuta ``guarded_fit`` sobre una rejilla n x K y clasifica el resultado.

    ``outcome``:
      ``ok``        -> hay particion
      ``None:<r>``  -> preflight la rechazo con motivo ``r``
      ``fb:dense_cdh`` -> ward degradado por el guard
    Nunca debe aparecer una excepcion no controlada (el test lo verifica).
    """
    rows: List[Dict] = []
    rng = np.random.default_rng(SEED)
    for n in ns:
        X = rng.normal(0.0, 1.0, size=(int(n), 8))
        for be in METHOD_C_BACKENDS:
            for k in ks:
                labels, diag = guarded_fit(be, X, int(k), floor)
                pre = diag["precondition"]
                if labels is not None:
                    outcome = (
                        "ok"
                        if diag["backend_used"] == be
                        else f"fb:{diag['backend_used']}"
                    )
                else:
                    outcome = f"None:{pre['reason']}"
                rows.append(
                    {
                        "backend": be,
                        "n": int(n),
                        "k": int(k),
                        "outcome": outcome,
                        "n_clusters": int(diag.get("n_clusters", 0)),
                        "message": pre.get("message", ""),
                        "notice": (diag["notices"][0] if diag.get("notices") else ""),
                    }
                )
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Informe
# ══════════════════════════════════════════════════════════════════════════


def _hdr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_thread_table(rows: Sequence[Dict]) -> None:
    if not rows:
        print(f"{WARN} sin filas de determinismo")
        return
    print(
        f"{'dataset':<12} {'K':>3} {'reps':>4} {'hilos1':>7} {'hilosN':>7} "
        f"{'dist1':>6} {'distN':>6} {'ARI':>6} {'pts!=}':>6}"
    )
    for r in rows:
        t1 = ",".join(str(x) for x in r["threads_1"]) or "?"
        tn = ",".join(str(x) for x in r["threads_n"]) or "?"
        flag = PWD if (r["distinct_1"] == 1 and r["distinct_n"] == 1) else WARN
        print(
            f"{r['dataset']:<12} {r['k']:>3} {r['reps']:>4} {t1:>7} {tn:>7} "
            f"{r['distinct_1']:>6} {r['distinct_n']:>6} "
            f"{r['ari_cross']:>6.3f} {r['n_diff_points']:>6}  {flag}"
        )


def print_ward_table(rows: Sequence[Dict], ex: Dict[str, Any]) -> None:
    print(
        f"{'n':>5} {'d':>3} {'K':>3} {'pico MB':>9} {'RSS MB':>8} {'s':>6} {'B/n^2':>8}"
    )
    for r in rows:
        print(
            f"{r['n']:>5} {r['d']:>3} {r['k']:>3} {r['peak_mb']:>9.1f} "
            f"{r['rss_delta_mb']:>8.1f} {r['seconds']:>6.3f} "
            f"{r['bytes_per_n2']:>8.1f}"
        )
    if ex.get("ok"):
        print(
            f"  ajuste peak ~ n^{ex['exponent']:.2f}  ->  "
            f"n={ex['target_n']}: ~{ex['predicted_mb']:.0f} MB"
        )
    else:
        print(f"  {WARN} extrapolacion no concluyente: {ex.get('reason')}")


def print_drift_table(rows: Sequence[Dict]) -> None:
    print(
        f"{'dataset':<12} {'target_k':>8} {'floor':>5} {'K real':>6} "
        f"{'splits':>6} {'rej_r2':>6}  aviso"
    )
    for r in rows:
        print(
            f"{r['dataset']:<12} {r['target_k']:>8} {r['floor']:>5} "
            f"{r['n_clusters']:>6} {r['splits']:>6} {r['rejected_r2_dense']:>6}"
            f"  {r['notice']}"
        )


def print_precondition_matrix(rows: Sequence[Dict], ns: Sequence[int]) -> None:
    ks = sorted({r["k"] for r in rows})
    for be in METHOD_C_BACKENDS:
        print(f"\n  {be}")
        print(f"  {'n':>5} " + " ".join(f"{('K=' + str(k)):>16}" for k in ks))
        for n in ns:
            cells = []
            for k in ks:
                r = next(
                    x
                    for x in rows
                    if x["backend"] == be and x["n"] == n and x["k"] == k
                )
                tag = r["outcome"]
                if tag == "ok":
                    tag = f"ok(K={r['n_clusters']})"
                cells.append(f"{tag:>16}")
            print(f"  {n:>5} " + " ".join(cells))
    print("\n  mensajes exactos (primer caso de cada motivo):")
    seen = set()
    for r in rows:
        key = r["outcome"].split(":")[-1]
        if r["outcome"].startswith("None:") and key not in seen:
            seen.add(key)
            print(f"    {r['message']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ks", default="3,8,14")
    ap.add_argument("--ward-ns", default="528,1000,2000")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-ward", action="store_true")
    args = ap.parse_args(argv)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    ward_ns = [int(x) for x in args.ward_ns.split(",") if x.strip()]
    t_start = time.perf_counter()

    _hdr("HARDENING · Method C — medicion de riesgos residuales 5-9")
    print(f"python {sys.version.split()[0]}  numpy {np.__version__}")
    try:
        import sklearn

        print(f"sklearn {sklearn.__version__}  cpus={os.cpu_count()}")
    except Exception:  # pragma: no cover
        print(f"cpus={os.cpu_count()}")
    try:
        import threadpoolctl

        print(f"threadpoolctl {threadpoolctl.__version__}")
    except Exception:  # pragma: no cover
        print(f"{WARN} sin threadpoolctl: determinismo BLAS no confinable")

    # ── 1. determinismo ──────────────────────────────────────────────────
    _hdr("1. kmeans_spherical: 1 hilo vs N hilos (threadpool_limits)")
    datasets = [synth_blobs(n=528, k=8), synth_null(n=528)]
    if not args.skip_real:
        try:
            real = load_real_dataset()
            print(
                f"real_ctx: {real['X'].shape[0]} UCEs x {real['X'].shape[1]} dims "
                f"({real['note']})"
            )
            datasets.append(real)
        except Exception as exc:
            print(f"{WARN} real_ctx no disponible: {exc}")
    det_rows = measure_thread_determinism(
        datasets, ks=ks, reps=args.reps, n_threads=os.cpu_count() or 1
    )
    print_thread_table(det_rows)
    bad = [r for r in det_rows if r["distinct_1"] > 1]
    cross = [r for r in det_rows if r["ari_cross"] < 1.0]
    print(
        f"  -> inestables con 1 hilo: {len(bad)}/{len(det_rows)} celdas; "
        f"1 vs N con ARI<1: {len(cross)}/{len(det_rows)} celdas"
    )

    # ── 2. RAM de ward ───────────────────────────────────────────────────
    _hdr("2. ward: pico de RAM (tracemalloc) y guard")
    ward_rows: List[Dict] = []
    ex: Dict[str, Any] = {"ok": False, "reason": "omitido"}
    if not args.skip_ward:
        ward_rows = measure_ward_memory(ward_ns)
        print_ward_table(ward_rows, {"ok": False, "reason": "pendiente"})
        ex = extrapolate_ward(ward_rows)
        print_ward_table(ward_rows, ex)
        if ex.get("ok"):
            cal = calibrate_ward_max_n(
                peak_at_n=ex["peak_mb_at"](ex["target_n"]),
                n_ref=ex["target_n"],
                exponent=ex["exponent"],
            )
            print(
                f"  -> method_c_ward_max_n calibrado={cal} "
                f"(pico extrapolado n={cal}: "
                f"{ex['peak_mb_at'](cal):.0f} MB, presupuesto "
                f"{DEFAULT_WARD_MAX_BYTES // 2**20} MB); "
                f"default propuesto={DEFAULT_WARD_MAX_N}"
            )
            allowed, msg = ward_guard(DEFAULT_WARD_MAX_N + 1000)
            print(f"  {msg if msg else 'sin aviso'}")

    # ── 3. target_k vs n_clusters ────────────────────────────────────────
    _hdr("3. dense_cdh: target_k vs n_clusters (gate min_r2_dense=0.08)")
    drift_rows = measure_target_k_drift()
    print_drift_table(drift_rows)
    for name in ("synth_blobs", "synth_null"):
        same = [
            r
            for r in drift_rows
            if r["dataset"] == name and r["n_clusters"] == r["target_k"]
        ]
        allr = [r for r in drift_rows if r["dataset"] == name]
        print(f"  {name}: coincide en {len(same)}/{len(allr)} celdas")

    # ── 4. precondiciones ────────────────────────────────────────────────
    _hdr("4. Precondiciones n x K (nunca IndexError)")
    ns = (0, 1, 2, 7, 50, 528)
    prec_rows = precondition_matrix(ns=ns, ks=(1, 2, 3, 8, 21))
    print_precondition_matrix(prec_rows, ns)

    total = time.perf_counter() - t_start
    _hdr(f"FIN · {total:.1f} s" + ("  " + PWD if total < 90 else "  " + WARN))
    if total >= 90:
        print(f"{WARN} presupuesto de 90 s excedido")
    return 0


if __name__ == "__main__":
    sys.exit(main())
