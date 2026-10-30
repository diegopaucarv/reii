"""Endurecimiento de Method C: guardas, precondiciones y traza de fallos.

Especificacion ejecutable de ``docs/design_v5/HARDENING.md`` (§1, §2, §4, §5,
§6). No importa ``reii.main_workflow`` (arrastra Stanza, >4 min) ni toca la BD:
solo la politica de Method C y, para un dataset sintetico, la arena.

Que se prueba y contra quien
----------------------------
La politica vive hoy en la COPIA CANONICA
``scripts/playground/hardening_bench.py:88-510`` (destino:
``src/reii/method_c_policy.py``, diff en HARDENING.md §8.1). El import es
tolerante: si el modulo de produccion existe se usa ese; si no, la copia. Si no
hay ninguno de los dos, el modulo se **salta** en vez de fallar al colectar.

Los cuatro ``xfail`` de ``TestContratoPendiente`` son la lista de trabajo del
diff §8.1: hoy el wrapper devuelve una particion donde el contrato exige
``None``. Se marcan ``strict=False`` para que, al aplicar el patch, pasen a
``XPASS`` sin romper la suite (y entonces se quita el marcador).

Tiempo: ~3 s con la API presente; ~0.7 s (solo imports) si todo se salta.
Los fits de ``ward``/``kmeans`` se omiten a proposito (ver ``_FIT_BACKENDS``).
"""

import sys
from pathlib import Path

import pytest

# ── Localizacion de la politica (produccion primero, copia canonica despues) ──
_ROOT = Path(__file__).resolve().parents[1]
_PLAYGROUND = _ROOT / "scripts" / "playground"


def _ensure_playground_path() -> None:
    p = str(_PLAYGROUND)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_policy():
    try:
        from reii import method_c_policy as mod  # type: ignore[attr-defined]

        return mod, "src/reii/method_c_policy.py"
    except Exception:
        pass
    _ensure_playground_path()
    try:
        import hardening_bench as mod

        return mod, "scripts/playground/hardening_bench.py (copia canonica)"
    except Exception as exc:  # pragma: no cover - solo entorno roto
        return None, f"no importable ({type(exc).__name__})"


_policy, _POLICY_SOURCE = _load_policy()

METHOD_C_BACKENDS = tuple(getattr(_policy, "METHOD_C_BACKENDS", ()) or ())
resolve_method_c_backend = getattr(_policy, "resolve_method_c_backend", None)
preflight = getattr(_policy, "preflight", None)
guarded_fit = getattr(_policy, "guarded_fit", None)
ward_guard = getattr(_policy, "ward_guard", None)
format_target_k_notice = getattr(_policy, "format_target_k_notice", None)
DEFAULT_WARD_MAX_N = getattr(_policy, "DEFAULT_WARD_MAX_N", None)
K_MIN = int(getattr(_policy, "K_MIN", 2) or 2)
K_MAX = int(getattr(_policy, "K_MAX", 20) or 20)
WARN = "[!]"

_MISSING = [
    name
    for name, value in (
        ("resolve_method_c_backend", resolve_method_c_backend),
        ("preflight", preflight),
        ("guarded_fit", guarded_fit),
        ("ward_guard", ward_guard),
        ("format_target_k_notice", format_target_k_notice),
        ("DEFAULT_WARD_MAX_N", DEFAULT_WARD_MAX_N),
    )
    if value is None
]
if not METHOD_C_BACKENDS:
    _MISSING.append("METHOD_C_BACKENDS")

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=(
        "API de hardening ausente en {}: {}; aplicar el diff de "
        "docs/design_v5/HARDENING.md seccion 8.1".format(
            _POLICY_SOURCE, ", ".join(_MISSING) or "-"
        )
    ),
)

np = pytest.importorskip("numpy")

FLOOR = 5  # hdbscan_uce_min_cluster_floor por defecto

# Los fits de ``ward``/``kmeans`` importan ``sklearn.cluster``, que cuesta ~15 s
# en Windows (medido: ``python -X importtime -c "import sklearn.cluster"`` =>
# 14.87 s). Por presupuesto (<5 s) aqui solo se fitea ``dense_cdh``, donde vive
# el riesgo de IndexError; ``ward``/``kmeans`` se cubren en la capa de politica
# (``preflight``) y en la arena. ``hdbscan`` ademas compila numba en el primer
# fit (~15-20 s) y es diagnostic-only.
_FIT_BACKENDS = tuple(b for b in METHOD_C_BACKENDS if b == "dense_cdh") or (
    "dense_cdh",
)


def _blobs(n=60, d=8, seed=0):
    """Nube gaussiana sin estructura: control negativo barato."""
    return np.random.default_rng(seed).normal(0.0, 1.0, size=(int(n), int(d)))


@pytest.fixture(scope="module")
def arena():
    """Arena de backends (numpy/sklearn puro; no importa Stanza)."""
    _ensure_playground_path()
    try:
        import c_backends_arena

        return c_backends_arena
    except Exception as exc:
        pytest.skip(f"c_backends_arena no importable: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# I2 - backend desconocido: degrada con aviso y nunca aborta
# ══════════════════════════════════════════════════════════════════════════
class TestBackendDesconocido:
    def test_degrada_con_aviso(self):
        backend, notice = resolve_method_c_backend("banana", source="persisted")
        assert backend in METHOD_C_BACKENDS
        assert notice is not None
        assert notice["code"] == "unknown_backend"
        assert notice["requested"] == "banana"
        assert notice["message"].startswith(WARN)
        assert "banana" in notice["message"]
        assert backend in notice["message"]

    def test_normaliza_espacios_y_mayusculas(self):
        backend, notice = resolve_method_c_backend("  Ward  ", source="persisted")
        assert backend == "ward"
        assert notice is None

    def test_valor_vacio_va_al_default_sin_aviso(self):
        backend, notice = resolve_method_c_backend(None, source="persisted")
        assert backend in METHOD_C_BACKENDS
        assert notice is None

    def test_no_aborta_el_preflight(self):
        """La resolucion degrada a hdbscan; ninguna capa lanza por el valor."""
        pre = preflight("banana", 200, 4, FLOOR)
        assert pre.ok is True
        assert pre.backend == "hdbscan"
        assert pre.backend_effective == "hdbscan"

    def test_fuente_interna_lanza(self):
        """Una constante mal escrita en el codigo es un bug, no un dato."""
        with pytest.raises(ValueError):
            resolve_method_c_backend("banana", source="internal")


# ══════════════════════════════════════════════════════════════════════════
# §2 - guarda de memoria de ward: degradar a dense_cdh, nunca reventar
# ══════════════════════════════════════════════════════════════════════════
class TestGuardaWard:
    def test_justo_en_el_limite(self):
        allowed, msg = ward_guard(int(DEFAULT_WARD_MAX_N))
        assert allowed is True
        assert msg == ""
        allowed, msg = ward_guard(int(DEFAULT_WARD_MAX_N) + 1)
        assert allowed is False
        assert msg.startswith(WARN)
        assert "method_c_ward_max_n" in msg
        assert "dense_cdh" in msg  # el fallback se anuncia en el texto

    def test_sobre_el_limite_no_ejecuta_ward(self):
        n = int(DEFAULT_WARD_MAX_N) + 1
        pre = preflight("ward", n, 8, FLOOR)
        assert pre.ok is True
        assert pre.backend == "ward"
        assert pre.backend_effective == "dense_cdh"
        assert pre.reason == "ward_memory_guard"
        assert pre.message.startswith(WARN)
        assert any("dense_cdh" in n_ for n_ in pre.notices)

    def test_fallback_produce_etiquetas_y_registra_el_cambio(self):
        """Con un tope pequeño sobre datos pequenos: ward no se ejecuta."""
        X = _blobs(n=40)
        labels, diag = guarded_fit("ward", X, 4, FLOOR, ward_max_n=10)
        assert diag["backend_requested"] == "ward"
        assert diag["backend_used"] == "dense_cdh"
        assert labels is not None
        assert len(labels) == 40
        assert any("dense_cdh" in msg for msg in diag["notices"])

    def test_el_limite_cabe_en_el_presupuesto_declarado(self):
        """8 B/par (matriz cuadrada completa) <= 320 MiB: HARDENING.md §2.2."""
        worst_bytes = 8 * int(DEFAULT_WARD_MAX_N) ** 2
        assert worst_bytes <= 320 * 2**20
        assert int(DEFAULT_WARD_MAX_N) <= 8000


# ══════════════════════════════════════════════════════════════════════════
# I3 / §5 - target_k es objetivo, no promesa: no se reporta como n_clusters
# ══════════════════════════════════════════════════════════════════════════
class TestKachVsTargetK:
    def test_target_k_no_se_reporta_como_n_clusters(self, arena):
        # Ruido isotropico: el gate r2_dense=0.08 corta el primer split.
        X = np.asarray(arena.synth_null(n=528)["X"], dtype=np.float64)
        target_k = 8
        floor = max(FLOOR, round(X.shape[0] / (2.0 * target_k)))

        labels, diag = guarded_fit("dense_cdh", X, target_k, floor, min_r2_dense=0.08)

        # La politica endurecida NO entrega particion con K<2 (BACKEND_C.md
        # §7): devuelve None pero CONSERVA el diagnostico medido, que es lo
        # que este test verifica (target_k != n_clusters).
        assert labels is None
        pre = diag["precondition"]
        assert pre["reason"] == "below_kmin"
        assert pre["k_requested"] == target_k
        assert diag["n_clusters"] == 1  # K real, medido sobre ruido puro
        assert diag["n_clusters"] != pre["k_requested"]
        # El aviso hace visible la brecha en vez de fingir que se alcanzo K.
        assert "n_clusters=1" in pre["message"]
        assert "sin particion" in pre["message"]
        assert diag.get("n_noise") == 0  # dense_cdh no emite -1

    def test_aviso_canonico_de_la_brecha(self):
        msg = format_target_k_notice(8, 5, 3)
        assert msg.startswith(WARN)
        assert "target_k=8" in msg
        assert "5 clases" in msg
        assert "3 splits" in msg
        assert format_target_k_notice(8, 8, 0) == ""  # sin brecha, sin ruido


# ══════════════════════════════════════════════════════════════════════════
# I1 / §4 - precondiciones: None con motivo, nunca IndexError
# ══════════════════════════════════════════════════════════════════════════
_GRID = [
    (0, 8),
    (1, 8),
    (2, 8),
    (7, 8),
    (50, 1),
    (50, 2),
    (50, 8),
    (50, 21),
]


@pytest.mark.parametrize("backend", METHOD_C_BACKENDS or ("dense_cdh",))
@pytest.mark.parametrize("n,k", _GRID)
def test_preflight_nunca_lanza(backend, n, k):
    """Los cuatro backends, sin tocar datos: coherencia y cero excepciones."""
    pre = preflight(backend, n, k, FLOOR)
    assert isinstance(pre.ok, bool)
    assert pre.reason
    assert pre.backend == backend
    if not pre.ok:
        assert pre.message.startswith(WARN)
    if pre.k_effective is not None:
        assert int(pre.k_effective) >= 1


@pytest.mark.parametrize("backend", _FIT_BACKENDS)
@pytest.mark.parametrize("n,k", _GRID)
def test_precondiciones_nunca_indexerror(backend, n, k):
    X = np.zeros((n, 8)) if n else np.zeros((0, 8))
    if n:
        X = _blobs(n=n, d=8)

    labels, diag = guarded_fit(backend, X, k, FLOOR)  # no debe lanzar

    pre = diag["precondition"]
    assert isinstance(pre["ok"], bool)
    assert pre["reason"]
    if labels is None:
        assert pre["ok"] is False
        assert pre["message"].startswith(WARN)
    else:
        assert pre["ok"] is True
        assert len(labels) == n
        assert np.issubdtype(np.asarray(labels).dtype, np.integer)
        assert int(diag["n_clusters"]) >= 0


@pytest.mark.parametrize("backend", _FIT_BACKENDS)
def test_matriz_1d_devuelve_none(backend):
    labels, diag = guarded_fit(backend, np.arange(50.0), 4, FLOOR)
    assert labels is None
    assert diag["precondition"]["ok"] is False
    assert diag["precondition"]["reason"] == "empty_corpus"


def test_n_bajo_el_minimo_devuelve_none():
    pre = preflight("dense_cdh", 7, 3, FLOOR)
    assert pre.ok is False
    assert pre.reason == "below_two_floors"
    labels, diag = guarded_fit("dense_cdh", _blobs(n=7), 3, FLOOR)
    assert labels is None
    assert diag["precondition"]["reason"] == "below_two_floors"


@pytest.mark.parametrize("backend", ("ward", "kmeans_spherical"))
def test_k_mayor_que_n_devuelve_none(backend):
    pre = preflight(backend, 7, 8, FLOOR)
    assert pre.ok is False
    assert pre.reason == "k_above_n"
    assert pre.message.startswith(WARN)


def test_k_menor_que_dos_no_produce_particion():
    pre = preflight("dense_cdh", 200, 1, FLOOR)
    assert pre.ok is False
    assert pre.reason in ("k_below_gate", "k_below_kmin")

    # K<2 no es particion en NINGUN backend: la politica lo rechaza en vez
    # de dejar que ward acote a K_MIN y fabrique el corte minimo
    # (BACKEND_C.md §7).
    pre_w = preflight("ward", 200, 1, FLOOR)
    assert pre_w.ok is False
    assert pre_w.reason == "k_below_kmin"


def test_ceros_por_encima_del_umbral_devuelve_none():
    X = np.zeros((60, 8))
    labels, diag = guarded_fit("dense_cdh", X, 4, FLOOR, n_zero_rows=60)
    assert labels is None
    assert diag["precondition"]["reason"] == "too_many_zero_rows"
    assert "zero-vectors" in diag["precondition"]["message"]


def test_corpus_vacio():
    labels, diag = guarded_fit("dense_cdh", np.zeros((0, 8)), 4, FLOOR)
    assert labels is None
    assert diag["precondition"]["reason"] == "empty_corpus"


def test_k_por_encima_del_maximo_se_acota():
    """dense_cdh no fabrica 21 hojas: el K efectivo <= K_MAX."""
    pre = preflight("dense_cdh", 528, 21, FLOOR)
    assert pre.ok is True
    assert int(pre.k_effective) <= K_MAX


# ══════════════════════════════════════════════════════════════════════════
# I4 / §6 - una excepcion del backend deja rastro (nunca se traga)
# ══════════════════════════════════════════════════════════════════════════
class TestTrazaDeFallos:
    def test_excepcion_del_backend_deja_rastro(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("fallo inyectado")

        monkeypatch.setattr(_policy, "dense_cdh", boom)  # guarded_fit lo resuelve aqui
        labels, diag = guarded_fit("dense_cdh", _blobs(), 4, FLOOR)

        assert labels is None
        pre = diag["precondition"]
        assert pre["ok"] is False
        assert pre["reason"] == "exception"
        # Formato exacto de HARDENING.md §6: tipo + mensaje.
        assert pre["message"].startswith(WARN)
        assert "RuntimeError" in pre["message"]
        assert "fallo inyectado" in pre["message"]


# ══════════════════════════════════════════════════════════════════════════
# §8.1 - contrato pendiente: hoy devuelve particion donde toca None
# ══════════════════════════════════════════════════════════════════════════
_XFAIL_REASON = (
    "HARDENING.md seccion 8.1: el wrapper devuelve una particion donde el "
    "contrato (BACKEND_C.md §7) exige None; lo cierra el diff de "
    "src/reii/method_c_policy.py"
)


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_nan_devuelve_none():
    labels, _diag = guarded_fit("dense_cdh", np.full((60, 8), np.nan), 4, FLOOR)
    assert labels is None


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_ceros_sin_hint_se_autodetectan():
    """n_zero_rows debe calcularse en guarded_fit, no confiarse al llamador."""
    labels, _diag = guarded_fit("dense_cdh", np.zeros((60, 8)), 4, FLOOR)
    assert labels is None


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_kmeans_k1_no_emite_una_clase():
    """La politica acepta target_k=1 para kmeans y emite una sola clase."""
    pre = preflight("kmeans_spherical", 60, 1, FLOOR)
    assert pre.ok is False
    assert pre.reason in ("k_below_gate", "k_below_kmin")


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_puntos_identicos_con_gate_activo_devuelven_none():
    """r2_dense = 0 con nada que partir: K=1 no es una particion."""
    labels, _diag = guarded_fit(
        "dense_cdh", np.ones((60, 8)), 4, FLOOR, min_r2_dense=0.08
    )
    assert labels is None
