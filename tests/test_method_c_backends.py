"""Contrato de los backends de Method C (BACKEND_C.md §1-§3 y §12).

Especificacion ejecutable de los hunks #21/#22 (``main_workflow.py``) y
#72/#72b (``reii/method_c_policy.py``). Se activa solo cuando el modulo de
politica esta aplicado con la ampliacion de ``BACKEND_C.md`` §12.3:

    method_c_policy.{guarded_fit, dense_cdh, ward_backend,
                     kmeans_spherical, hdbscan_backend,
                     corte_optimo_1d, build_method_c_result}

Hasta entonces todos los tests se **saltan** (la suite queda verde) en vez de
fallar: el diff lo aplica el usuario a mano sobre ``src/**``, que es un scope de
escritura prohibido para este agente.

NO se importa ``reii.main_workflow`` (>4 min por Stanza). Todo lo que se prueba
aqui vive en ``method_c_policy``, que es numpy-only: el ensamblador del payload
(``build_method_c_result``) existe justamente para que el conjunto de claves del
contrato §1 sea verificable sin levantar el pipeline.

Cubre:

(a) los cuatro backends devuelven el MISMO conjunto de claves  -> test_conjunto
(b) dense_cdh sobre ruido isotropo => None (gate r2_dense=0.08) -> test_ruido
(c) dense_cdh sobre blobs => K pedido                          -> test_blobs
(d) ward / kmeans_spherical deterministas con la misma semilla -> test_det
(e) hdbscan marcado diagnostic_only                            -> test_hdbscan
(f) backend desconocido degrada con aviso "[!]", sin abortar    -> test_degrada
(g) partition_sizes / n_assigned / n_noise cuadran             -> test_tamanos
(h) corte_optimo_1d == corte original (oraculo por fuerza bruta) -> test_corte

Tiempo esperado: <5 s con el diff aplicado, ~0 s saltado.
"""

from functools import lru_cache

import numpy as np
import pytest

try:  # modulo nuevo (#72); ausente mientras el diff no se aplique
    from reii import method_c_policy as _pol
except Exception:  # noqa: BLE001 - import roto == diff no aplicado
    _pol = None

# Superficie exacta que estos tests consumen. ``getattr`` tolerante: con el
# modulo ausente o incompleto el fichero COLECCIONA y se salta (patron de
# tests/test_select_v4.py), nunca falla al importar.
_REQUIRED = (
    "guarded_fit",
    "resolve_method_c_backend",
    "preflight",
    "ward_guard",
    "effective_target_k",
    "format_target_k_notice",
    "dense_cdh",
    "ward_backend",
    "kmeans_spherical",
    "hdbscan_backend",
    "corte_optimo_1d",
    "build_method_c_result",
)
_MISSING = [name for name in _REQUIRED if _pol is None or not hasattr(_pol, name)]

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=(
        "method_c_policy incompleto (faltan: "
        + ", ".join(_MISSING)
        + "); aplicar HARDENING.md §8.1 y BACKEND_C.md §12.3"
    )
    if _MISSING
    else "",
)

SEED = 42
K_BLOBS = 6
N_BLOBS = 528
N_NULL = 528
D_DIMS = 30
# mcs_frac que reproduce un min_cluster_size de 14 UCEs (solo hdbscan lo usa).
MCS_FRAC = 14.0 / N_BLOBS

DENSE_BACKENDS = ("dense_cdh", "ward", "kmeans_spherical")
ALL_BACKENDS = ("hdbscan",) + DENSE_BACKENDS


# ──────────────────────────────────────────────────────────────────────
# Datasets sinteticos (mismos que la arena, n y d de BACKEND_C.md §4)
# ──────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _blobs() -> np.ndarray:
    """K=6 blobs separados en d=30 (n=528, 88 UCEs por blob).

    Separacion 8 sigma con sigma=1: estructura inequivoca, para que el test no
    dependa del gate. El control negativo es `_null`.
    """
    rng = np.random.default_rng(SEED)
    centers = rng.normal(size=(K_BLOBS, D_DIMS)) * 8.0
    parts = [
        centers[i] + rng.normal(size=(N_BLOBS // K_BLOBS, D_DIMS))
        for i in range(K_BLOBS)
    ]
    n_fill = N_BLOBS - (N_BLOBS // K_BLOBS) * K_BLOBS
    return np.vstack(parts + [rng.uniform(-20.0, 20.0, size=(n_fill, D_DIMS))])


@lru_cache(maxsize=1)
def _null() -> np.ndarray:
    """Gaussiana isotropa d=30: control negativo, sin estructura."""
    return np.random.default_rng(SEED).normal(size=(N_NULL, D_DIMS))


def _uce_ids(n: int):
    return [f"uce{i}" for i in range(int(n))]


def _fit(backend, X, target_k, floor=5, *, diagnostic_only=False, requested=None, **kw):
    """``guarded_fit`` + ensamblado del payload §1. Devuelve (result|None, diag)."""
    labels, diag = _pol.guarded_fit(backend, X, int(target_k), int(floor), **kw)
    if labels is None:
        return None, diag
    result = _pol.build_method_c_result(
        labels,
        diag,
        _uce_ids(len(labels)),
        X,
        int(target_k),
        int(floor),
        requested=(backend if requested is None else requested),
        diagnostic_only=bool(diagnostic_only),
    )
    return result, diag


# ──────────────────────────────────────────────────────────────────────
# (a) Mismo contrato de resultado para los cuatro backends
# ──────────────────────────────────────────────────────────────────────
def test_los_cuatro_backends_devuelven_el_mismo_conjunto_de_claves():
    pytest.importorskip("hdbscan")
    pytest.importorskip("sklearn.cluster")

    X = _blobs()
    reference = None
    for backend in ALL_BACKENDS:
        result, diag = _fit(backend, X, K_BLOBS, floor=5, mcs_frac=MCS_FRAC)
        assert result is not None, (
            f"{backend} no produjo particion: {diag.get('precondition')}"
        )
        keys = frozenset(result)
        if reference is None:
            reference = keys
        else:
            assert keys == reference, (
                f"{backend} difiere del contrato §1: "
                f"sobran={sorted(reference - keys)} faltan={sorted(keys - reference)}"
            )

    # Claves minimas de BACKEND_C.md §1 (heredadas + aditivas).
    for key in (
        "method",
        "uce_to_label",
        "labels_array",
        "uce_ids",
        "embedding_matrix",
        "cluster_persistence",
        "n_clusters",
        "n_noise",
        "min_cluster_size",
        "backend",
        "target_k",
        "partition_sizes",
        "n_assigned",
        "diagnostic_only",
        "diagnostics",
    ):
        assert key in reference, f"falta la clave obligatoria {key!r}"

    # `method` sigue siendo "embedding": lo espera run().
    result, _ = _fit("dense_cdh", X, K_BLOBS, floor=5)
    assert result["method"] == "embedding"


def test_backends_densos_no_emiten_ruido():
    pytest.importorskip("sklearn.cluster")
    X = _blobs()
    for backend in DENSE_BACKENDS:
        result, _ = _fit(backend, X, K_BLOBS, floor=5)
        assert result is not None
        assert result["n_noise"] == 0, f"{backend} emitio -1 (bug del backend)"
        assert np.all(result["labels_array"] >= 0)
        assert result["n_assigned"] == len(result["labels_array"])
        assert result["diagnostic_only"] is False


# ──────────────────────────────────────────────────────────────────────
# (b) Ruido isotropo: el gate r2_dense corta y no hay particion
# ──────────────────────────────────────────────────────────────────────
def test_dense_cdh_sobre_ruido_isotropo_no_devuelve_particion():
    X = _null()
    # El default de produccion de la politica es 0.08 (HARDENING §8.1);
    # se pasa explicito para no depender de la version del default.
    result, diag = _fit("dense_cdh", X, 8, floor=5, min_r2_dense=0.08)
    assert result is None, "dense_cdh fabrico una particion sobre ruido puro"

    bdiag = diag.get("backend_diagnostics") or {}
    # Medido en la arena: r2_dense de ruido 0.036-0.064 < 0.08.
    if bdiag:
        assert bdiag["mean_r2_dense"] < 0.08
        assert bdiag["rejected_min_r2_dense"] >= 1
    assert diag["precondition"]["reason"] in ("below_kmin", "ok")


def test_gate_r2_dense_apagado_si_parte_el_ruido():
    """Control del test anterior: sin gate, K=8 sobre el mismo ruido.

    Demuestra que el None de arriba lo produce el gate, no un fallo del
    backend. Es tambien la razon por la que `min_r2_threshold=0.05` del CDH
    de A no sirve aqui (BACKEND_C.md §6).
    """
    X = _null()
    labels, _diag = _pol.guarded_fit("dense_cdh", X, 8, 5, min_r2_dense=0.0)
    assert labels is not None
    assert int(np.unique(labels[labels >= 0]).size) == 8


# ──────────────────────────────────────────────────────────────────────
# (c) Estructura real: dense_cdh alcanza el K pedido
# ──────────────────────────────────────────────────────────────────────
def test_dense_cdh_sobre_blobs_alcanza_el_k_pedido():
    X = _blobs()
    result, _diag = _fit("dense_cdh", X, K_BLOBS, floor=5)
    assert result is not None
    assert result["backend"] == "dense_cdh"
    assert result["target_k"] == K_BLOBS
    assert result["n_clusters"] == K_BLOBS
    assert result["diagnostics"]["backend_diagnostics"]["n_splits"] == K_BLOBS - 1


# ──────────────────────────────────────────────────────────────────────
# (d) Determinismo de ward y kmeans_spherical
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("backend", ("ward", "kmeans_spherical"))
def test_ward_y_kmeans_son_deterministas(backend):
    pytest.importorskip("sklearn.cluster")
    X = _blobs()

    first, diag1 = _fit(backend, X, K_BLOBS, floor=5)
    second, _diag2 = _fit(backend, X, K_BLOBS, floor=5)

    assert first is not None and second is not None
    assert np.array_equal(first["labels_array"], second["labels_array"])
    assert first["uce_to_label"] == second["uce_to_label"]
    assert first["partition_sizes"] == second["partition_sizes"]

    if backend == "kmeans_spherical":
        # El RNG esta fijado y declarado, no oculto (§3).
        assert diag1["backend_diagnostics"]["random_state"] == SEED
        assert diag1["backend_diagnostics"]["n_init"] == 10


def test_ward_y_kmeans_son_tautologicos_por_construccion():
    """Sin piso, C1 == C2 en ambos: ARI=1.0 por construccion (§3).

    No es un bug que haya que arreglar en el backend: se declara y obliga a
    que la validez venga de `bootstrap_ari` (SELECT_V4).
    """
    pytest.importorskip("sklearn.cluster")
    from sklearn.metrics import adjusted_rand_score

    X = _blobs()
    for backend in ("ward", "kmeans_spherical"):
        tight, _ = _fit(backend, X, K_BLOBS, floor=5)
        loose, _ = _fit(backend, X, K_BLOBS, floor=3)
        assert tight is not None and loose is not None
        ari = adjusted_rand_score(tight["labels_array"], loose["labels_array"])
        assert ari == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────
# (e) hdbscan: se computa, no compite
# ──────────────────────────────────────────────────────────────────────
def test_hdbscan_se_marca_diagnostic_only():
    pytest.importorskip("hdbscan")
    X = _blobs()

    result, diag = _fit(
        "hdbscan", X, K_BLOBS, floor=5, diagnostic_only=True, mcs_frac=MCS_FRAC
    )
    assert result is not None, (
        f"hdbscan no produjo particion: {diag.get('precondition')}"
    )
    assert result["backend"] == "hdbscan"
    assert result["diagnostic_only"] is True
    assert "diagnostic_only" in str(diag["precondition"]["message"])

    # No controla K: `target_k` solo fija min_cluster_size (§2).
    assert result["target_k"] == K_BLOBS
    assert result["n_clusters"] >= 1

    # Escape hatch: method_c_hdbscan_selectable=True => vuelve a competir.
    result_on, _ = _fit(
        "hdbscan", X, K_BLOBS, floor=5, diagnostic_only=False, mcs_frac=MCS_FRAC
    )
    assert result_on["diagnostic_only"] is False


# ──────────────────────────────────────────────────────────────────────
# (f) Backend desconocido: degrada con "[!]", nunca aborta
# ──────────────────────────────────────────────────────────────────────
def test_backend_desconocido_degrada_con_aviso():
    backend, notice = _pol.resolve_method_c_backend("banana", source="persisted")
    assert backend == _pol.DEFAULT_METHOD_C_BACKEND
    assert notice is not None
    assert notice["message"].startswith("[!]")
    assert notice["code"] == "unknown_backend"

    # `guarded_fit` no propaga excepcion con un valor invalido.
    labels, diag = _pol.guarded_fit("banana", _blobs(), K_BLOBS, 5, mcs_frac=MCS_FRAC)
    assert diag["backend_requested"] == "banana"
    assert diag["backend_used"] == _pol.DEFAULT_METHOD_C_BACKEND
    assert labels is None or len(labels) == len(_blobs())

    # Una constante mal escrita EN EL CODIGO si es un bug: ValueError.
    with pytest.raises(ValueError):
        _pol.resolve_method_c_backend("banana", source="internal")

    # `None` no es "configurado mal": no hay aviso.
    backend_none, notice_none = _pol.resolve_method_c_backend(None, source="persisted")
    assert backend_none == _pol.DEFAULT_METHOD_C_BACKEND
    assert notice_none is None


def test_payload_degradado_registra_la_resolucion():
    """`diagnostics.method_c.backend_resolution` es lo que lee el auditor."""
    pytest.importorskip("hdbscan")
    _be, notice = _pol.resolve_method_c_backend("kmeans_esferico", source="persisted")

    labels, diag = _pol.guarded_fit(_be, _blobs(), K_BLOBS, 5, mcs_frac=MCS_FRAC)
    if labels is None:  # hdbscan sin clusters: el payload no aplica
        pytest.skip(f"hdbscan no produjo particion: {diag.get('precondition')}")

    result = _pol.build_method_c_result(
        labels,
        diag,
        _uce_ids(len(labels)),
        _blobs(),
        K_BLOBS,
        5,
        requested="kmeans_esferico",
        diagnostic_only=True,
        resolution_notice=notice["message"],
    )
    assert result is not None
    resolution = result["diagnostics"]["method_c"]["backend_resolution"]
    assert resolution["requested"] == "kmeans_esferico"
    assert resolution["effective"] == _pol.DEFAULT_METHOD_C_BACKEND
    assert resolution["notice"].startswith("[!]")
    assert resolution["reason"]


# ──────────────────────────────────────────────────────────────────────
# (g) Denominadores del gate de dominancia (§5)
# ──────────────────────────────────────────────────────────────────────
def test_tamanos_de_particion_cuadran():
    pytest.importorskip("hdbscan")
    pytest.importorskip("sklearn.cluster")
    X = _blobs()
    n_total = len(X)

    for backend in ALL_BACKENDS:
        result, _diag = _fit(backend, X, K_BLOBS, floor=5, mcs_frac=MCS_FRAC)
        assert result is not None, backend
        sizes = result["partition_sizes"]
        # `n_assigned` es el denominador explicito; `sizes` excluye -1.
        assert sum(sizes) == result["n_assigned"]
        assert result["n_assigned"] + result["n_noise"] == n_total
        assert len(sizes) == result["n_clusters"]
        assert sizes == sorted(sizes)
        assert all(s > 0 for s in sizes)
        # uce_to_label solo lleva etiquetas >= 0 (contrato §1).
        assert len(result["uce_to_label"]) == result["n_assigned"]
        assert all(v >= 0 for v in result["uce_to_label"].values())


# ──────────────────────────────────────────────────────────────────────
# (h) corte_optimo_1d == corte original (oraculo por fuerza bruta)
# ──────────────────────────────────────────────────────────────────────
def _corte_original_por_fuerza_bruta(coord, min_side):
    """SS_between maximizado enumerando TODOS los cortes factibles.

    Es el oraculo de `ClasificadorDescendente._corte_optimo`
    (main_workflow.py:3831-3871). No se importa el original para no levantar
    Stanza: se re-deriva la definicion en O(n^2), que es exactamente lo que la
    version vectorizada calcula en O(n log n).
    """
    n = len(coord)
    order = np.argsort(coord)
    s = coord[order]
    total = float(s.sum())
    lo = max(1, int(min_side))
    hi = n - lo
    ss = []
    for k in range(lo, hi + 1):
        left = float(s[:k].sum())
        right = total - left
        ss.append(left * left / k + right * right / (n - k) - total * total / n)
    best = int(np.argmax(ss))  # primer maximo, igual que np.argmax
    k_best = lo + best
    ss_total = float(np.sum((s - s.mean()) ** 2))
    r2 = float(ss[best]) / (ss_total + 1e-12)
    labels = np.zeros(n, dtype=int)
    labels[order[k_best:]] = 1
    return r2, labels


@pytest.mark.parametrize("min_side", (1, 5, 12))
def test_corte_optimo_1d_es_equivalente_al_corte_original(min_side):
    rng = np.random.default_rng(SEED)
    # Mezcla de continua y con empates (ruido redondeado) para forzar el
    # desempate por primer maximo.
    coord = np.concatenate(
        [rng.normal(size=40), np.round(rng.normal(size=40), 1), np.zeros(20)]
    )
    rng.shuffle(coord)

    r2, labels = _pol.corte_optimo_1d(coord, min_side=min_side)
    r2_ref, labels_ref = _corte_original_por_fuerza_bruta(coord, min_side)

    assert r2 == pytest.approx(r2_ref, rel=0, abs=1e-12)
    assert np.array_equal(labels, labels_ref)
    assert set(np.unique(labels)) <= {0, 1}


def test_corte_optimo_1d_respeta_el_piso():
    coord = np.arange(10, dtype=float)
    _r2, labels = _pol.corte_optimo_1d(coord, min_side=4)
    sizes = np.bincount(labels, minlength=2)
    assert int(sizes.min()) >= 4
