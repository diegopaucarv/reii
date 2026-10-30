"""Contrato de estabilidad por remuestreo y nulo de etiquetas.

Especificacion ejecutable de `docs/design_v5/STABILITY_NULL.md`.

Se activa solo cuando el diff de la seccion 7 esta aplicado:

    quality.metrics.bootstrap_stability  +  null_partition_test
    method_selection.select_method(..., null_mode=...)
    method_selection.STABILITY_MEAN_MIN

Hasta entonces todos los tests se **saltan** (la suite queda verde) en vez de
fallar: los diffs los aplica el usuario a mano sobre `src/**`, que es un scope
de escritura prohibido para este agente.

Cubre:

(a) determinismo del bootstrap                  -> ``test_determinismo``
(b) el IC contiene la media                     -> ``test_ic_contiene_la_media``
(c) `coverage_stable < 1` en ruido              -> `TestRuidoYRuido`
(ii) ruido puro: mean < 0.30, SIN_SENAL         -> `test_ruido_puro_sin_senal`
(iii) estructura sintetica: ci_low >= 0.50      -> `test_estructura_sintetica_estable`
(iv) `null_p` NO gatea por defecto              -> `test_null_p_no_gatea_por_defecto`
(v) `null_mode="gate"` con `null_p` inyectado   -> ``test_gate_con_null_p_inyectado_descarta``
(vi) `nan` nunca descarta (salvo backend denso) -> ``test_nan_nunca_descarta``
(vi-b) backend denso sin bootstrap no compite  -> ``test_dense_cdh_sin_bootstrap``
(vii) piso 0.30 sobre el ARI bootstrap          -> ``test_piso_030_descarta``
(viii) precondiciones devuelven NaN/None        -> `TestPrecondiciones`

La regla es pura: no importa el pipeline, ni la BD, ni Stanza. El import de
`reii.quality.metrics` (sklearn) es **perezoso**: si el diff no esta aplicado
el fichero se salta sin pagar ese coste.
"""

import inspect

import numpy as np
import pytest

from reii import method_selection as _ms

# ``import *``-equivalente tolerante: con solo el diff de v4 aplicado (o con
# ninguno), el modulo no tiene estas funciones y el fichero debe poder
# COLECTARSE para que el ``skipif`` lo marque como saltado.
stability_mean_min = getattr(
    _ms, "STABILITY_MEAN_MIN", getattr(_ms, "stability_mean_min", float("nan"))
)
MethodMetrics = _ms.MethodMetrics
select_method = _ms.select_method

_SELECT_PARAMS = inspect.signature(select_method).parameters
_SELECT_OK = "null_mode" in _SELECT_PARAMS and np.isfinite(float(stability_mean_min))

# Solo se importa sklearn si la regla ya entiende de estabilidad: asi el salto
# cuesta ~1 s en vez de ~20 s en Windows.
_qm = None
if _SELECT_OK:
    try:
        from reii.quality import metrics as _qm
    except ImportError:  # pragma: no cover - defensivo
        _qm = None

bootstrap_stability = getattr(_qm, "bootstrap_stability", None)
null_partition_test = getattr(_qm, "null_partition_test", None)
STABILITY_MEAN_MIN = getattr(_qm, "STABILITY_MEAN_MIN", float("nan"))

_HAS_STABILITY = (
    bootstrap_stability is not None
    and null_partition_test is not None
    and np.isfinite(float(STABILITY_MEAN_MIN))
)

pytestmark = pytest.mark.skipif(
    not _HAS_STABILITY,
    reason=(
        "motor de estabilidad no aplicado; aplicar los diffs (a)-(c) de "
        "docs/design_v5/STABILITY_NULL.md seccion 7"
    ),
)

NAN = float("nan")
_CACHE = {}


# ── datos sinteticos minimos (n=90, K=3; nada pesado) ──────────────────────
def _blobs(n=90, d=3, k=3, sep=10.0, seed=0):
    """Tres nubes separadas a `sep` (centros en `I*sep`): ARI debe ser 1.0."""
    rng = np.random.default_rng(seed)
    centers = np.eye(k, d) * float(sep)
    y = np.repeat(np.arange(k), n // k)
    X = centers[y] + rng.normal(0.0, 1.0, size=(len(y), d))
    return X, y


def _noise(n=90, d=30, seed=1):
    """Gaussiana isotropica 30-d: el regimen de `synth_null` del banco.

    En 3-d las celdas de Voronoi son reproducibles bajo submuestreo y el ARI
    bootstrap sube a ~0.45; con d=30 la particion deja de reproducirse
    (STABILITY_NULL.md §2, `synth_null`: 0.09-0.11).
    """
    return np.random.default_rng(seed).normal(0.0, 1.0, size=(n, d))


def _kmeans_predict(k):
    """Closure ``predict(X_sub) -> labels``: KMeans estandar y determinista."""
    from sklearn.cluster import KMeans

    def _f(Xs):
        A = np.asarray(Xs, dtype=float)
        km = KMeans(n_clusters=int(k), n_init=10, random_state=42)
        return km.fit_predict(A)

    return _f


def _data():
    """Fixtures compartidas: se calculan una vez por proceso."""
    if "X" not in _CACHE:
        Xs, ys = _blobs()
        _CACHE["X"] = Xs
        _CACHE["y"] = ys
        _CACHE["Xn"] = _noise()
        _CACHE["pred"] = _kmeans_predict(3)
    return _CACHE


def _boot(X, y, **kw):
    kw.setdefault("n_boot", 12)
    kw.setdefault("min_n", 30)
    kw.setdefault("predict_fn", _data()["pred"])
    return bootstrap_stability(X, y, **kw)


def _m(method, k, ari, coverage, **kw):
    kw.setdefault("perturbation", f"{method}1<->{method}2")
    return MethodMetrics(method=method, k=k, ari=ari, coverage=coverage, **kw)


# ══════════════════════════════════════════════════════════════════════════
# (a)(b) determinismo y contrato del bootstrap
# ══════════════════════════════════════════════════════════════════════════
def test_determinismo():
    """Mismo X, mismas labels, misma seed => mismas cifras."""
    d = _data()
    a = _boot(d["X"], d["y"])
    b = _boot(d["X"], d["y"])
    assert a["ok"] and b["ok"]
    assert a["bootstrap_ari"] == pytest.approx(b["bootstrap_ari"], abs=0.0)
    assert a["bootstrap_ari_ci_low"] == pytest.approx(
        b["bootstrap_ari_ci_low"], abs=0.0
    )


def test_ic_contiene_la_media():
    """La banda 2.5/97.5 contiene el ARI medio (no es el IC de la media)."""
    d = _data()
    out = _boot(d["X"], d["y"])
    assert out["ok"]
    assert out["bootstrap_ari_ci_low"] <= out["bootstrap_ari"]
    assert out["bootstrap_ari"] <= out["bootstrap_ari_ci_high"]


def test_contrato_de_claves():
    """El bloque expone exactamente las claves de STABILITY_NULL.md §4."""
    d = _data()
    out = _boot(d["X"], d["y"])
    for key in (
        "bootstrap_ari",
        "bootstrap_ari_ci_low",
        "bootstrap_ari_ci_high",
        "bootstrap_n_boot",
        "coverage_stable",
        "stability_kind",
        "stability_verdict",
    ):
        assert key in out, key
    assert out["stability_kind"] == "bootstrap"
    assert int(out["bootstrap_n_boot"]) == 12


# ══════════════════════════════════════════════════════════════════════════
# (c)(ii) ruido: cobertura inestable y sin senal
# ══════════════════════════════════════════════════════════════════════════
def test_coverage_stable_menos_de_uno_en_ruido():
    """En ruido la etiqueta de una UCE cambia entre replicas: no es 1.0."""
    d = _data()
    ref = d["pred"](d["Xn"])
    out = _boot(d["Xn"], ref)
    assert out["ok"], out["reason"]
    assert np.isfinite(out["coverage_stable"])
    assert out["coverage_stable"] < 1.0


def test_ruido_puro_sin_senal():
    """Ruido gaussiano: ARI medio < 0.30 => SIN_SENAL (y no descarta solo)."""
    d = _data()
    ref = _data()["pred"](d["Xn"])
    out = _boot(d["Xn"], ref)
    assert out["ok"], out["reason"]
    assert out["bootstrap_ari"] < 0.30
    assert out["stability_verdict"] == "SIN_SENAL"


# ══════════════════════════════════════════════════════════════════════════
# (iii) estructura sintetica: ESTABLE
# ══════════════════════════════════════════════════════════════════════════
def test_estructura_sintetica_estable():
    """Tres nubes separadas: el remuestreo recupera la particion (ci_low>=0.50)."""
    d = _data()
    out = _boot(d["X"], d["y"])
    assert out["ok"], out["reason"]
    assert out["bootstrap_ari_ci_low"] >= 0.50
    assert out["stability_verdict"] == "ESTABLE"


# ══════════════════════════════════════════════════════════════════════════
# (iv)(v) el nulo es columna de control, no gate
# ══════════════════════════════════════════════════════════════════════════
def _con_nulo():
    """A debil y C fuerte; el unico ``null_p`` malo es el de C."""
    a = _m("A", 5, 0.300, 0.900, bootstrap_ari=0.30, null_p=0.010)
    c = _m(
        "C",
        4,
        0.900,
        1.000,
        backend="dense_cdh",
        bootstrap_ari=0.720,
        bootstrap_ci_lo=0.600,
        coverage_stable=0.900,
        null_p=0.990,
    )
    return a, c


def test_null_p_no_gatea_por_defecto():
    """Con el default ``diagnostic`` un p=0.99 NO descarta a C."""
    a, c = _con_nulo()
    sel = select_method([a, c])
    assert sel.winner == "C"
    assert [d["method"] for d in sel.discarded] == []


def test_gate_con_null_p_inyectado_descarta():
    """Con ``null_mode="gate"`` y el p inyectado, C cae por el nulo."""
    a, c = _con_nulo()
    sel = select_method([a, c], null_mode="gate")
    assert sel.winner == "A"
    why = [d["why"] for d in sel.discarded if d["method"] == "C"]
    assert why and "nulo" in why[0]
    assert "0.990" in why[0]


def test_nan_nunca_descarta():
    """``bootstrap_ari`` y ``null_p`` ausentes: A y B siguen siendo viables.

    La excepcion deliberada son los backends densos (``dense_cdh``, ``ward``,
    ``kmeans_spherical``): sin ``bootstrap_ari`` su ARI tight/loose no es
    conmensurable con A/B, asi que no compiten. Eso se cubre en el test
    siguiente y en ``tests/test_select_v4.py::TestBackendsNoConmensurables``.
    """
    a = _m("A", 5, 0.300, 0.500, bootstrap_ari=NAN, null_p=NAN)
    b = _m("B", 4, 0.620, 0.800, bootstrap_ari=NAN, null_p=NAN)
    sel = select_method([a, b])
    assert sel.winner == "B"
    assert not any(d["method"] in {"A", "B"} for d in sel.discarded)


def test_dense_cdh_sin_bootstrap():
    """``dense_cdh`` sin ``bootstrap_ari`` no compite (politica de patch 16)."""
    a = _m("A", 5, 0.300, 0.500, bootstrap_ari=0.30)
    c = _m(
        "C",
        4,
        0.620,
        0.800,
        backend="dense_cdh",
        bootstrap_ari=NAN,
        null_p=NAN,
    )
    sel = select_method([a, c])
    assert sel.winner == "A"
    why = [d["why"] for d in sel.discarded if d["method"] == "C"]
    assert why and "conmensurable" in why[0]
    assert "bootstrap_ari" in why[0]


# ══════════════════════════════════════════════════════════════════════════
# (vii) piso de viabilidad sobre el ARI bootstrap
# ══════════════════════════════════════════════════════════════════════════
def test_piso_030_descarta():
    """``bootstrap_ari`` finito y < 0.30 => NO VIABLE, con motivo explicito."""
    a = _m("A", 5, 0.300, 0.500, bootstrap_ari=0.300)
    c = _m(
        "C",
        4,
        0.950,
        0.900,
        backend="dense_cdh",
        bootstrap_ari=0.100,
        bootstrap_ci_lo=0.050,
    )
    sel = select_method([a, c])
    assert sel.winner == "A"
    why = [d["why"] for d in sel.discarded if d["method"] == "C"]
    assert why and "piso" in why[0]


def test_piso_coincide_entre_modulos():
    """El piso de la regla y el del motor no pueden divergir."""
    assert float(stability_mean_min) == pytest.approx(float(STABILITY_MEAN_MIN))


# ══════════════════════════════════════════════════════════════════════════
# (viii) precondiciones: NaN/None sin excepcion
# ══════════════════════════════════════════════════════════════════════════
class TestPrecondiciones:
    def test_n_menor_que_min_n(self):
        out = bootstrap_stability(
            np.zeros((10, 2)), np.zeros(10, dtype=int), predict_fn=_data()["pred"]
        )
        assert out["ok"] is False
        assert np.isnan(out["bootstrap_ari"])
        assert out["stability_verdict"] == "NO_DISPONIBLE"
        assert "min_n" in out["reason"]

    def test_x_none(self):
        out = bootstrap_stability(None, [0, 1, 0], predict_fn=_data()["pred"])
        assert out["ok"] is False
        assert np.isnan(out["bootstrap_ari"])

    def test_sin_prediccion_no_disponible(self):
        d = _data()
        out = bootstrap_stability(d["X"], d["y"], predict_fn=None)
        assert out["ok"] is False
        assert out["stability_verdict"] == "NO_DISPONIBLE"

    def test_null_una_sola_clase(self):
        out = null_partition_test(_data()["Xn"], np.zeros(90, dtype=int), n_perm=10)
        assert out["ok"] is False
        assert np.isnan(out["null_p"])
        assert "NaN" in out["reason"]

    def test_null_contrato_y_nunca_lanza(self):
        d = _data()
        out = null_partition_test(d["Xn"], np.full(90, -1, dtype=int), n_perm=5)
        assert out["ok"] is False
        for key in ("null_p", "null_stat", "null_n_perm"):
            assert key in out
        out2 = null_partition_test(d["Xn"], np.zeros(5, dtype=int), n_perm=5)
        assert out2["ok"] is False
        assert np.isnan(out2["null_p"])
