"""Politica de Method C: backend, precondiciones y acotado de K.

Modulo numpy-only (no importa Stanza) para que el contrato de
``BACKEND_C.md`` §1 se pueda ejercer en segundos desde
``tests/test_method_c_backends.py``. ``guarded_fit`` es el UNICO punto que
llama a los backends: el dispatcher de ``main_workflow.py`` no reimplementa
ninguna guarda (BACKEND_C.md §12.4).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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
#: §2). Presupuesto 320 MiB de pico ADICIONAL. El peor camino (cuadrado
#: completo) gasta 8 B/par (= 275 MiB a n=6000); el condensado 4 B/par (144
#: MiB), asi que 6000 respeta el peor caso con margen. Coincide con
#: _METHOD_C_WARD_MAX_N (main_workflow.py, #69): el banco y produccion no
#: deben calibrar distinto, o el guard degradaria justo donde el diseno dice
#: que se permite ward.
DEFAULT_WARD_MAX_N = 6000
DEFAULT_WARD_MAX_BYTES = 320 * 1024 * 1024

DEFAULT_FLOOR = 5  # hdbscan_uce_min_cluster_floor
DEFAULT_MCS_FRAC = 0.08  # hdbscan_uce_min_cluster_frac (tight)

WARN = r"[!]"
SEED = 42  # mismo valor que c_backends_arena.py y Config.random_state


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

    # K<2 no es una particion en NINGUN backend: rechazar en la politica, no
    # dejar que ward lo acote a 2 (fabrica el corte minimo) o que kmeans lo
    # emita tal cual. BACKEND_C.md §7.
    if target_k < K_MIN:
        return _fail(
            "k_below_kmin",
            f"{WARN} method_c: target_k={target_k} < {K_MIN} -> sin particion",
        )
    # target_k > K_MAX NO se rebana aqui: `k_requested` debe seguir siendo el
    # valor pedido (auditoria, I3 y §5). Cada rama lo acota y lo anuncia.

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
        k_eff = max(K_MIN, min(target_k, n, K_MAX))
        if k_eff != int(target_k):
            notices.append(
                f"{WARN} ward: target_k={target_k} > K_MAX={K_MAX} "
                f"-> K efectivo={k_eff}"
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
        k_eff = min(int(target_k), K_MAX)
        if k_eff != int(target_k):
            notices.append(
                f"{WARN} kmeans_spherical: target_k={target_k} > K_MAX={K_MAX} "
                f"-> K efectivo={k_eff}"
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
    min_r2_dense: float = 0.08,  # valor de produccion (BACKEND_C.md §6)
    mcs_frac: float = DEFAULT_MCS_FRAC,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """``preflight`` + backend + validacion de salida. Nunca propaga excepcion.

    Devuelve ``(labels|None, diag)``. ``diag`` siempre trae ``precondition``
    (serializable), ``notices`` y, si corrio, ``backend_used``/``n_clusters``.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return None, {
            "backend_requested": backend,
            "backend_used": backend,
            "precondition": {
                "ok": False,
                "reason": "not_convertible",
                "message": f"{WARN} method_c: matriz no convertible a float64 "
                f"({type(exc).__name__}) -> sin particion",
                "k_requested": int(target_k),
                "k_effective": None,
            },
            "notices": [],
        }
    n = int(X.shape[0]) if X.ndim == 2 else 0
    if n and not np.isfinite(X).all():
        n_nonfinite = int((~np.isfinite(X)).any(axis=1).sum())
        return None, {
            "backend_requested": backend,
            "backend_used": backend,
            "precondition": {
                "ok": False,
                "reason": "nonfinite_values",
                "message": f"{WARN} method_c: {n_nonfinite} valores no finitos "
                f"(NaN/Inf) -> sin particion",
                "k_requested": int(target_k),
                "k_effective": None,
            },
            "notices": [],
        }
    if n and not n_zero_rows:
        # Misma definicion que main_workflow.py:5125. No se confia del llamador.
        n_zero_rows = int(np.all(X == 0, axis=1).sum())
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
    if int(len(real)) < K_MIN:
        diag["precondition"]["ok"] = False
        diag["precondition"]["reason"] = "below_kmin"
        diag["precondition"]["message"] = (
            f"{WARN} {be}: n_clusters={int(len(real))} < {K_MIN} -> sin particion"
        )
        return None, diag
    if be == "dense_cdh":
        rejected = int(diag["backend_diagnostics"].get("rejected_min_r2_dense", 0))
        notice = format_target_k_notice(target_k, len(real), rejected)
        if notice:
            diag["notices"].append(notice)
    return labels, diag


# ══════════════════════════════════════════════════════════════════════════
# Cuerpos de los backends (BACKEND_C.md §12.3, hunk #72b)
# ══════════════════════════════════════════════════════════════════════════


def corte_optimo_1d(coord: np.ndarray, min_side: int = 1) -> Tuple[float, np.ndarray]:
    """Corte que maximiza SS_between restringido a k en [min_side, n-min_side].

    COPIA LITERAL de ClasificadorDescendente._corte_optimo (main_workflow).
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


def hdbscan_backend(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    *,
    epsilon: float = 0.0,
    metric: str = "euclidean",
    gen_min_span_tree: bool = True,
    prediction_data: bool = True,
):
    """HDBSCAN plano: el Method C actual. No controla K; puede emitir -1.

    Se usa gen_min_span_tree=False y prediction_data=False solo por velocidad
    (no cambian las etiquetas); produccion los deja en True.
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        cluster_selection_epsilon=float(epsilon),
        metric=str(metric),
        gen_min_span_tree=bool(gen_min_span_tree),
        prediction_data=bool(prediction_data),
    )
    labels = clusterer.fit_predict(X).astype(int)
    diag = {
        "cluster_persistence": [
            round(float(p), 4)
            for p in list(getattr(clusterer, "cluster_persistence_", []))
        ],
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "epsilon": float(epsilon),
    }
    return labels, diag


def build_method_c_result(
    labels: np.ndarray,
    diag: Dict[str, Any],
    uce_ids: Sequence[str],
    matrix: np.ndarray,
    target_k: int,
    floor: int,
    *,
    requested: Optional[str] = None,
    diagnostic_only: bool = False,
    resolution_notice: str = "",
) -> Optional[Dict[str, Any]]:
    """Ensambla el dict de resultado de Method C (BACKEND_C.md §1).

    Puro: no imprime ni toca el pipeline. Devuelve ``None`` cuando no hay
    particion (``n_clusters < 2``), mismo contrato que ``guarded_fit``.
    ``partition_sizes`` excluye ``-1``; ``n_assigned`` es el denominador
    explicito del gate de dominancia (§5).
    """
    labels = np.asarray(labels, dtype=int).ravel()
    assigned = labels >= 0
    n_assigned = int(assigned.sum())
    n_noise = int(labels.shape[0] - n_assigned)
    n_clusters = int(np.unique(labels[assigned]).size)
    if n_clusters < K_MIN:
        return None

    be = str(diag.get("backend_used", diag.get("backend_requested", "")))
    bdiag = diag.get("backend_diagnostics") or {}
    sizes = sorted(int(c) for c in np.bincount(labels[assigned]))
    pre = diag.get("precondition") or {}
    return {
        "method": "embedding",
        "min_cluster_size": int(floor),
        "uce_to_label": {
            uce_ids[i]: int(labels[i])
            for i in range(min(len(uce_ids), int(labels.shape[0])))
            if labels[i] >= 0
        },
        "labels_array": labels,
        "uce_ids": list(uce_ids),
        "embedding_matrix": matrix,
        "cluster_persistence": [float(p) for p in bdiag.get("cluster_persistence", [])],
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "backend": be,
        "target_k": int(target_k),
        "partition_sizes": sizes,
        "n_assigned": n_assigned,
        "diagnostic_only": bool(diagnostic_only),
        "diagnostics": {
            "method_c": {
                "backend_resolution": {
                    "requested": str(
                        diag.get("backend_requested", "") if requested is None else requested
                    ),
                    "effective": be,
                    "reason": str(pre.get("reason", "")),
                    "notice": str(resolution_notice),
                },
                "precondition": pre,
                "notices": list(diag.get("notices", [])),
            },
            "backend_diagnostics": bdiag,
        },
    }
