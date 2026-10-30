"""Tests del espacio condicional v5 y del guard de excepciones.

Contrato que fijan estos tests (ver `docs/design_v5/V5_SPACE.md`):

  1. El muestreo condicional NUNCA produce una combinacion que el gate
     estructural rechace (5000 draws): `tight < loose` y `frac > frac_loose`.
     Ademas no degrada la cobertura: los 21 pares `(tight, loose)` son
     alcanzables.
  2. Los dos parametros inertes (rho~0) estan congelados por defecto, y
     `optimize_freeze_inert=False` restaura el espacio exploratorio.
  3. El guard convierte CUALQUIER `Exception` del objetivo en
     `_SENTINEL_TRIAL` con `error_type`/`error_msg`/`traceback` en
     `user_attrs` (el FAIL #81 del probe v4 se perdio porque nadie guardaba
     el traceback).
  4. El guard NO captura `KeyboardInterrupt` ni `SystemExit`.
  5. Un trial con parametros factibles que explota es distinguible de un
     centinela por gate (`outcome="exception"` vs `outcome="gate"`,
     `_sentinel_breakdown` cuenta uno de cada uno).

Requiere aplicar el diff de `docs/design_v5/V5_SPACE.md` §7. Sin el diff, los
tests se SALTAN con un motivo explicito (el simulador
`scripts/playground/v5_space_sim.py` avisa igual con `[!]`), para no ensuciar
la suite mientras el cambio no esta aplicado.

Nota: este modulo importa `reii.main_workflow`, que es pesado (~min). Es el
mismo patron que `tests/test_optimizer_gate.py`.
"""

import random

import pytest

import reii.main_workflow as mw


def _need(name):
    """Devuelve un simbolo del diff v5 o salta con un motivo claro."""
    obj = getattr(mw, name, None)
    if obj is None:
        pytest.skip(
            f"falta `{name}` en main_workflow: aplicar el diff de "
            f"docs/design_v5/V5_SPACE.md §7"
        )
    return obj


# Combinacion factible por construccion (los 13 parametros del espacio).
FEASIBLE = {
    "min_forms_uc_1": 13,
    "forms_gap": 4,
    "tsj": 3,
    "cdh_target_k": 8,
    "swap_iterations": 1,
    "min_r2_threshold": 0.05,
    "similarity_threshold": 0.45,
    "coref_weight": 0.2,
    "uc_window_size": 3,
    "coref_context_units_loose": 5,
    "coref_context_units_tight": 2,
    "hdbscan_uce_min_cluster_frac_loose": 0.01,
    "hdbscan_uce_min_cluster_frac": 0.02,
}


class FakeTrial:
    """Trial minimo con la semantica de muestreo de Optuna 3.6.2.

    `suggest_*` devuelve el valor fijado en `fixed` (si existe) o un draw
    uniforme dentro del rango que le pasa el codigo: asi se ejercita la
    *aritmetica del rango dinamico* sin depender del sampler real.
    """

    def __init__(self, rng, fixed=None):
        self.number = 0
        self.values = None
        self.params = {}
        self.user_attrs = {}
        self._rng = rng
        self._fixed = dict(fixed or {})

    def suggest_int(self, name, lo, hi):
        value = int(self._fixed.get(name, self._rng.randint(int(lo), int(hi))))
        self.params[name] = value
        return value

    def suggest_float(self, name, lo, hi):
        value = float(self._fixed.get(name, self._rng.uniform(float(lo), float(hi))))
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        choices = list(choices)
        value = self._fixed.get(name, self._rng.choice(choices))
        self.params[name] = value
        return value

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


@pytest.fixture
def opt():
    """Optimizador sin __init__ (evita segmentar el corpus)."""
    optimizador_cls = _need("Optimizador")
    opt = optimizador_cls.__new__(optimizador_cls)
    opt.config_base = mw.Config()
    opt.uc_config = mw.UCBuilderConfig()
    opt.total_uces = 528
    opt.retrofitted_vectors = {f"u{i}": None for i in range(528)}
    opt.progressive_segmenter = None
    opt._last_pairwise = None
    opt._last_selection = None
    opt._current_stage_mode = "wc_only"
    opt._search_space_cached = opt.get_search_space()
    return opt


# ─────────────────────────────────────────────────────────────────────────────
# 1. Espacio condicional
# ─────────────────────────────────────────────────────────────────────────────
def test_conditional_space_never_violates_structural_gates(opt):
    space = opt._search_space_cached
    assert isinstance(space.get("coref_context_units_tight"), dict), (
        "se esperaba un spec condicional para coref_context_units_tight"
    )
    assert isinstance(space.get("hdbscan_uce_min_cluster_frac"), dict), (
        "se esperaba un spec condicional para hdbscan_uce_min_cluster_frac"
    )

    rng = random.Random(1234)
    pairs = set()
    frac_pairs = set()
    for _ in range(5000):
        trial = FakeTrial(rng)
        p = mw.Optimizador._suggest(trial, space)

        tight = p["coref_context_units_tight"]
        loose = p["coref_context_units_loose"]
        frac = p["hdbscan_uce_min_cluster_frac"]
        frac_loose = p["hdbscan_uce_min_cluster_frac_loose"]

        # los dos gates duros de _evaluate_trial (L8560-8563)
        assert tight < loose, f"tight={tight} >= loose={loose}"
        assert frac > frac_loose, f"frac={frac} <= frac_loose={frac_loose}"

        assert 1 <= tight <= 7
        assert loose in (3, 4, 5, 6, 8)
        assert frac_loose in (0.005, 0.01, 0.015, 0.02, 0.03)
        assert frac_loose < frac <= 0.04 + 1e-12

        pairs.add((tight, loose))
        frac_pairs.add((frac_loose, round(frac, 6)))

    # cobertura: no se degrada el espacio (21 pares factibles, todos vivos)
    assert len(pairs) == 21, f"pares (tight, loose) alcanzados: {len(pairs)}"
    assert {loose for _, loose in pairs} == {3, 4, 5, 6, 8}
    assert {tight for tight, _ in pairs} == set(range(1, 8))
    # 48 "voxels" en la malla de 0.0025 (ver simulador §2)
    assert len(frac_pairs) >= 40, (
        f"pares (frac_loose, frac) alcanzados: {len(frac_pairs)}"
    )


def test_conditional_spec_rejects_empty_range():
    suggest_conditional = _need("_suggest_conditional")

    class _Trap:
        def suggest_int(self, *a, **k):  # pragma: no cover
            raise AssertionError("no deberia muestrear un rango vacio")

    with pytest.raises(ValueError):
        suggest_conditional(
            _Trap(), "tight", {"kind": "lt", "other": "loose", "lo": 1}, {"loose": 1}
        )
    with pytest.raises(ValueError):
        suggest_conditional(
            _Trap(),
            "frac",
            {"kind": "gt", "other": "fl", "hi": 0.01, "eps": 0.002},
            {"fl": 0.01},
        )


def test_inert_params_are_frozen_by_default(opt):
    space = opt._search_space_cached
    assert not isinstance(space["uc_window_size"], (list, tuple))
    assert not isinstance(space["min_forms_uc_1"], (list, tuple))
    assert space["uc_window_size"] == opt.uc_config.window_size
    assert space["min_forms_uc_1"] == opt.config_base.min_forms_uc[0]

    # escape hatch: el espacio exploratorio completo sigue disponible
    opt.config_base.optimize_freeze_inert = False
    space_open = opt.get_search_space()
    assert isinstance(space_open["uc_window_size"], list)
    assert isinstance(space_open["min_forms_uc_1"], tuple)


def test_params_feasible_matches_the_conditional_space():
    """El filtro de warm start debe ser al menos tan estricto como el espacio.

    Optuna acepta un valor encolado fuera del rango dinamico con solo un
    UserWarning (medido en 3.6.2), asi que un punto infactible encolado se
    convertiria en centinela de gate.
    """
    assert not mw.Optimizador._params_feasible(
        {"coref_context_units_tight": 5, "coref_context_units_loose": 5}
    )
    assert not mw.Optimizador._params_feasible(
        {
            "hdbscan_uce_min_cluster_frac": 0.01,
            "hdbscan_uce_min_cluster_frac_loose": 0.01,
        }
    )
    assert mw.Optimizador._params_feasible(dict(FEASIBLE))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Guard de excepciones
# ─────────────────────────────────────────────────────────────────────────────
def test_guard_converts_exception_into_sentinel_with_traceback(opt, monkeypatch):
    make_guarded = _need("_make_guarded_objective")

    def boom(cfg, uc_cfg=None):
        raise RuntimeError("kaboom en hdbscan")

    monkeypatch.setattr(opt, "_evaluacion_rapida", boom)
    printed = []
    guarded = make_guarded(opt._evaluate_trial, printer=printed.append)

    trial = FakeTrial(random.Random(0), FEASIBLE)
    assert guarded(trial) == mw._SENTINEL_TRIAL

    attrs = trial.user_attrs
    assert attrs["error_type"] == "RuntimeError"
    assert "kaboom en hdbscan" in attrs["error_msg"]
    assert "RuntimeError" in attrs["traceback"]
    assert "boom" in attrs["traceback"], "el traceback debe tener el frame real"
    assert len(trial.user_attrs["traceback"].splitlines()) <= 20
    assert trial.user_attrs["outcome"] == "exception"

    # se imprime una sola vez por trial, con el formato del doc
    assert len(printed) == 1
    assert printed[0].startswith("[!] trial #0 fallo: RuntimeError:")
    assert "(traceback en user_attr)" in printed[0]


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_guard_does_not_swallow_base_exceptions(opt, monkeypatch, exc_type):
    make_guarded = _need("_make_guarded_objective")

    def boom(cfg, uc_cfg=None):
        raise exc_type()

    monkeypatch.setattr(opt, "_evaluacion_rapida", boom)
    guarded = make_guarded(opt._evaluate_trial, printer=lambda *_: None)
    trial = FakeTrial(random.Random(0), FEASIBLE)

    with pytest.raises(exc_type):
        guarded(trial)
    # Ctrl-C no deja rastro de "trial fallido"
    assert trial.user_attrs == {}


def test_exception_sentinel_is_distinguishable_from_gate_sentinel(opt, monkeypatch):
    make_guarded = _need("_make_guarded_objective")
    breakdown = _need("_sentinel_breakdown")
    guarded = make_guarded(opt._evaluate_trial, printer=lambda *_: None)

    # (a) centinela por gate: la evaluacion devuelve None -> sin primary_stable
    monkeypatch.setattr(opt, "_evaluacion_rapida", lambda cfg, uc_cfg=None: None)
    t_gate = FakeTrial(random.Random(0), FEASIBLE)
    assert guarded(t_gate) == mw._SENTINEL_TRIAL
    assert t_gate.user_attrs.get("outcome") == "gate"
    assert t_gate.user_attrs.get("gate_reason")
    assert "error_type" not in t_gate.user_attrs

    # (b) centinela por excepcion: parametros FACTIBLES, el pipeline explota
    def boom(cfg, uc_cfg=None):
        raise ValueError("fallo con parametros factibles")

    monkeypatch.setattr(opt, "_evaluacion_rapida", boom)
    t_exc = FakeTrial(random.Random(0), FEASIBLE)
    assert guarded(t_exc) == mw._SENTINEL_TRIAL
    assert t_exc.user_attrs.get("outcome") == "exception"
    assert t_exc.user_attrs["error_type"] == "ValueError"
    assert t_exc.user_attrs["traceback"]

    # el reporte final los cuenta por separado
    t_gate.values = mw._SENTINEL_TRIAL
    t_exc.values = mw._SENTINEL_TRIAL
    n_gate, n_exc, detail = breakdown([t_gate, t_exc])
    assert (n_gate, n_exc) == (1, 1)
    assert len(detail) == 1 and "ValueError" in detail[0]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pre-registro v6 (patch 19): espacio y auditor de bordes
# ─────────────────────────────────────────────────────────────────────────────
def test_swap_iterations_excludes_the_no_refinement_ablation(opt):
    """0 apaga `_intercambio` en `_partition`: es ablacion, no hiperparametro."""
    levels = list(opt._search_space_cached["swap_iterations"])
    assert levels == [1, 2, 3, 5]
    assert 0 not in levels
    assert 1 in levels


def test_tsj_ceiling_is_declared_not_widened(opt):
    """v5 censuro `tsj` al techo; NO se amplia: subirlo poda vocabulario."""
    cfg = mw.Config()
    assert opt._search_space_cached["tsj"] == (max(2, cfg.tsj - 2), cfg.tsj + 2)
    assert "tsj" in _need("_BOUNDARY_DECLINED")
    assert "tsj" not in _need("_BOUNDARY_EXPANDABLE")


def test_forms_gap_floor_is_declared_not_widened(opt):
    """El piso de `forms_gap` no se amplia: gap<2 colapsa las pasadas.

    Con `mf2 = mf1 + gap`, un gap menor hace que la pasada 2 construya casi
    las mismas UC que la pasada 1 y el ARI de 2 pasadas sube por
    construccion: misma familia de artefacto que `swap_iterations=0`.
    """
    assert opt._search_space_cached["forms_gap"] == (2, 7)
    assert "forms_gap" in _need("_BOUNDARY_DECLINED")
    assert "forms_gap" not in _need("_BOUNDARY_EXPANDABLE")


def test_boundary_audit_flags_expandable_and_structural_edges():
    audit = _need("_boundary_audit")
    space = {
        "tsj": (2, 6),
        "coref_weight": [0.0, 0.1, 0.2],
        "swap_iterations": [1, 2, 3, 5],
        "cdh_target_k": (3, 12),
        "min_forms_uc_1": (9, 17),
        "forms_gap": (2, 7),
        "similarity_threshold": [0.25, 0.35, 0.45, 0.55, 0.65],
        "min_r2_threshold": [0.03, 0.05, 0.08],
        "coref_context_units_loose": [3, 4, 5, 6, 8],
        "coref_context_units_tight": {
            "kind": "lt",
            "other": "coref_context_units_loose",
            "lo": 1,
        },
        "hdbscan_uce_min_cluster_frac_loose": [0.005, 0.01, 0.02],
        "hdbscan_uce_min_cluster_frac": {
            "kind": "gt",
            "other": "hdbscan_uce_min_cluster_frac_loose",
            "hi": 0.04,
            "eps": 0.002,
        },
    }
    chosen = {
        "tsj": 6,  # techo: declinado
        "coref_weight": 0.0,  # piso: ampliable
        "swap_iterations": 5,  # techo: estructural
        "cdh_target_k": 3,  # piso: fallback (ver abajo)
        "min_forms_uc_1": 12,  # interior
        "forms_gap": 7,  # techo: censurado-declarado
        "similarity_threshold": 0.45,  # interior
        "min_r2_threshold": 0.03,  # piso: ampliable
        "coref_context_units_loose": 5,  # interior (driver)
        "coref_context_units_tight": 4,  # hi=loose-1: estructural
        "hdbscan_uce_min_cluster_frac_loose": 0.01,  # interior (driver)
        "hdbscan_uce_min_cluster_frac": 0.04,  # techo: estructural
    }

    hits = audit(space, chosen)
    assert [h[0] for h in hits] == [
        "cdh_target_k",
        "coref_context_units_tight",
        "coref_weight",
        "forms_gap",
        "hdbscan_uce_min_cluster_frac",
        "min_r2_threshold",
        "swap_iterations",
        "tsj",
    ]

    by_name = {name: (value, rec) for name, value, rec in hits}
    # Ampliables: la recomendacion propone ensanchar el rango.
    for name, expected in (
        ("coref_weight", "por debajo del minimo"),
        ("min_r2_threshold", "por debajo del minimo"),
    ):
        rec = by_name[name][1]
        assert rec.startswith("ampliable:"), (name, rec)
        assert expected in rec, (name, rec)
    assert by_name["coref_weight"][0] == "0.0"

    # Censurados declarados: se reportan, pero la recomendacion es NO ampliar.
    rec = by_name["tsj"][1]
    assert rec.startswith("censurado-declarado:"), rec
    assert "NO se amplia" in rec, rec
    assert by_name["tsj"][0] == "6"

    # `forms_gap` en el piso: censurado declarado (gap<2 colapsa las pasadas).
    rec = by_name["forms_gap"][1]
    assert rec.startswith("censurado-declarado:"), rec
    assert "NO se amplia" in rec, rec
    assert by_name["forms_gap"][0] == "7"

    # `cdh_target_k` en el PISO ya no es "estructural": el pipeline relanza
    # la etapa 2 una vez por abajo (fallback automatico, §4.5 de
    # AHORRO_K_Y_PERSISTENCIA.md), asi que es una decision reversible.
    rec = by_name["cdh_target_k"][1]
    assert rec.startswith("fallback:"), rec
    assert "piso" in rec, rec
    assert "K=2" in rec, rec
    assert by_name["cdh_target_k"][0] == "3"

    # El TECHO sigue siendo estructural (por encima el CDH sale del gate).
    at_hi = dict(chosen)
    at_hi["cdh_target_k"] = 12
    hi_hits = {name: (value, rec) for name, value, rec in audit(space, at_hi)}
    rec = hi_hits["cdh_target_k"][1]
    assert rec.startswith("estructural:"), rec
    assert "NO se amplia" in rec, rec
    assert "TECHO" in rec, rec

    # Estructurales: el auditor dice explicitamente que NO se amplia.
    for name in (
        "swap_iterations",
        "coref_context_units_tight",
        "hdbscan_uce_min_cluster_frac",
    ):
        rec = by_name[name][1]
        assert rec.startswith("estructural:"), (name, rec)
        assert "NO se amplia" in rec, (name, rec)

    # Todos interiores -> sin bordes.
    interior = dict(chosen)
    interior.update(
        {
            "tsj": 4,
            "coref_weight": 0.1,
            "swap_iterations": 2,
            "cdh_target_k": 8,
            "forms_gap": 4,
            "min_r2_threshold": 0.05,
            "coref_context_units_tight": 3,
            "hdbscan_uce_min_cluster_frac": 0.03,
        }
    )
    assert audit(space, interior) == []


def test_boundary_audit_never_raises_on_missing_or_weird_input():
    audit = _need("_boundary_audit")
    space = {
        "tsj": (2, 6),
        "coref_weight": [0.0, 0.1],
        "coref_context_units_tight": {
            "kind": "lt",
            "other": "coref_context_units_loose",
            "lo": 5,
        },
        "hdbscan_uce_min_cluster_frac": {
            "kind": "gt",
            "other": "hdbscan_uce_min_cluster_frac_loose",
            "hi": 0.04,
        },
        "frozen_scalar": 13,
        "degenerate_range": (3, 3),
        "one_level_list": [7],
    }

    # (a) claves ausentes: se omiten, el resto sigue funcionando
    assert [h[0] for h in audit(space, {"tsj": 6})] == ["tsj"]
    # (b) valores con tipo inesperado: no lanza y no inventa bordes
    assert audit(space, {"tsj": "seis", "coref_weight": None}) == []
    assert audit(space, {"tsj": [1, 2], "coref_weight": {"a": 1}}) == []
    # (c) spec condicional sin driver: se ignora
    assert audit(space, {"coref_context_units_tight": 4}) == []
    # (d) contenedores que no son dict: lista vacia, sin excepcion
    assert audit(None, {"tsj": 6}) == []
    assert audit(space, None) == []
    assert audit([("tsj", (2, 6))], {"tsj": 6}) == []
