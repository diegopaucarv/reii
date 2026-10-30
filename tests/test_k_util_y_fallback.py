"""Tercer objetivo -K_util y fallback automatico del piso de cdh_target_k.

Pre-registro: docs/design_v5/AHORRO_K_Y_PERSISTENCIA.md (§3 y §4).

1. K_util = numero de clases con masa >= max(2, int(min_class_share*n)).
   Sustituye a K en el tercer objetivo para que el sampler no pueda mejorar
   ese slot partiendo en clases diminutas. Invariante: K_util <= K.
2. Si el ganador de la etapa 2 queda pegado al piso del espacio, se relanza
   UNA vez la etapa 2 con piso 2 en un estudio nuevo. Solo se adopta si su
   ganador elige el piso nuevo; si no, el resultado original se conserva y
   best_params.json no se toca (la caida corre con save_pareto=False).

Los tests no construyen el pipeline: usan stubs con los pocos atributos que
tocan las funciones bajo prueba.
"""

from __future__ import annotations

from types import SimpleNamespace

from reii.main_workflow import (
    Config,
    Optimizador,
    _useful_k,
    _useful_k_floor,
)
from reii.method_selection import MethodMetrics, Selection, select_method


# ── K_util ─────────────────────────────────────────────────────────────────
def test_useful_k_cuenta_solo_clases_con_masa():
    # Piso 5: {244, 52, 4, 3} -> dos clases utiles; las otras dos son polvo.
    assert _useful_k([244, 52, 4, 3], 5) == 2
    assert _useful_k([10, 20, 30], 5) == 3
    assert _useful_k([1, 1, 1], 5) == 0
    assert _useful_k([], 5) == 0


def test_useful_k_nunca_supera_k_real():
    sizes = [244, 52, 4, 3]
    assert _useful_k(sizes, 5) <= len(sizes)


def test_useful_k_piso_minimo_es_dos():
    # Piso 0 o 1 se eleva a 2: una clase de 1 UCE nunca es util.
    assert _useful_k([1, 2], 0) == 1
    assert _useful_k([1, 2], 1) == 1


def test_useful_k_tolera_valores_no_numericos():
    assert _useful_k([None, 10], 5) == 0


def test_useful_k_floor_usa_la_formula_de_tiny_frac():
    # Pre-registro §2: ceil(0.01 * 528) = 6 (no int() = 5). El piso es el
    # mismo que usa tiny_frac, para que K_util y cobertura util coincidan.
    assert _useful_k_floor(528, 0.01) == 6  # ceil(5.28)
    assert _useful_k_floor(10, 0.01) == 2
    assert _useful_k_floor(0, 0.01) == 2
    assert _useful_k_floor(528, 0.0) == 2
    # Un n exacto no se ve afectado por el redondeo.
    assert _useful_k_floor(500, 0.01) == 5


# ── La regla propaga K_util al payload ─────────────────────────────────────
def test_selection_expone_k_util_del_ganador():
    sel = select_method(
        [
            MethodMetrics(
                "A",
                k=7,
                ari=0.419,
                coverage=0.394,
                perturbation="A1<->A2",
                n_useful_classes=5,
            ),
            MethodMetrics(
                method="B", k=9, ari=0.212, coverage=0.415, perturbation="B1<->B2"
            ),
        ]
    )
    assert sel.winner == "A"
    assert sel.winner_k == 7
    assert sel.winner_k_util == 5
    assert sel.to_dict()["winner_k_util"] == 5


def test_selection_k_util_cero_si_el_productor_no_lo_mide():
    sel = select_method(
        [
            MethodMetrics("A", k=7, ari=0.419, coverage=0.394, perturbation="A1<->A2"),
            MethodMetrics(
                method="B", k=9, ari=0.212, coverage=0.415, perturbation="B1<->B2"
            ),
        ]
    )
    assert sel.winner == "A"
    assert sel.winner_k_util == 0


def test_from_dict_tolera_payload_sin_k_util():
    sel = Selection.from_dict(
        {
            "winner": "A",
            "winner_ari": 0.4,
            "winner_coverage": 0.4,
            "winner_k": 4,
            "k_bounds": [3, 20],
        }
    )
    assert sel.winner_k_util == 0


# ── Piso del espacio de busqueda ───────────────────────────────────────────
def _space_stub(k_lo=None, k_floor=3):
    cfg = Config()
    cfg.optimize_k_floor = k_floor
    stub = SimpleNamespace(
        config_base=cfg,
        uc_config=SimpleNamespace(window_size=3),
        total_uces=528,
        retrofitted_vectors=None,
    )
    if k_lo is not None:
        stub._cdh_k_lo = k_lo
    return Optimizador.get_search_space(stub)


def test_search_space_piso_por_defecto_es_tres():
    lo, hi = _space_stub()["cdh_target_k"]
    assert lo == 3
    assert hi >= lo


def test_search_space_piso_baja_a_dos_con_fallback():
    assert _space_stub(k_lo=2)["cdh_target_k"][0] == 2


def test_search_space_nunca_baja_de_dos():
    assert _space_stub(k_floor=1)["cdh_target_k"][0] == 2


# ── Fallback automatico ────────────────────────────────────────────────────
class _OptStub:
    """Stub de Optimizador con lo unico que usa _maybe_k_floor_fallback."""

    def __init__(self, fb_params, floor=3, enabled=True, raise_exc=False):
        self.config_base = SimpleNamespace(
            optimize_k_floor_fallback=enabled,
            optimize_k_floor=floor,
            optimize_k_floor_min=2,
        )
        self._cdh_k_lo = floor
        self._search_space_cached = {}
        self._last_stage_study = "study-fallback"
        self.saved = []
        self.calls = []
        self._fb = fb_params
        self._raise = raise_exc

    def get_search_space(self):
        return {"cdh_target_k": (self._cdh_k_lo, 12)}

    def _run_single_stage(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise:
            raise RuntimeError("boom")
        return dict(self._fb), []

    def save_pareto_front(self, study):
        self.saved.append(study)


def _run_fallback(stub, params):
    return Optimizador._maybe_k_floor_fallback(stub, params, [], 5)


def test_fallback_se_adopta_si_el_piso_era_vinculante():
    stub = _OptStub({"cdh_target_k": 2})
    params = {"cdh_target_k": 3}
    out = _run_fallback(stub, params)
    assert out["cdh_target_k"] == 2
    assert stub.saved == ["study-fallback"]
    assert stub._cdh_k_lo == 2
    # La caida usa un study NUEVO y no reescribe best_params.json por su cuenta.
    assert stub.calls[0]["study_suffix"] == "_stage2_select_k2"
    assert stub.calls[0]["save_pareto"] is False


def test_fallback_no_se_adopta_si_el_optimo_no_es_el_piso():
    stub = _OptStub({"cdh_target_k": 3})
    params = {"cdh_target_k": 3}
    out = _run_fallback(stub, params)
    assert out["cdh_target_k"] == 3
    assert stub.saved == []
    assert stub._cdh_k_lo == 3


def test_fallback_no_corre_si_el_ganador_no_esta_en_el_piso():
    stub = _OptStub({"cdh_target_k": 2})
    out = _run_fallback(stub, {"cdh_target_k": 7})
    assert out["cdh_target_k"] == 7
    assert stub.calls == []


def test_fallback_no_corre_si_esta_deshabilitado():
    stub = _OptStub({"cdh_target_k": 2}, enabled=False)
    out = _run_fallback(stub, {"cdh_target_k": 3})
    assert out["cdh_target_k"] == 3
    assert stub.calls == []


def test_fallback_no_corre_si_el_piso_ya_es_dos():
    stub = _OptStub({"cdh_target_k": 2}, floor=2)
    out = _run_fallback(stub, {"cdh_target_k": 2})
    assert out["cdh_target_k"] == 2
    assert stub.calls == []


def test_fallback_sobrevive_a_una_excepcion():
    stub = _OptStub({"cdh_target_k": 2}, raise_exc=True)
    out = _run_fallback(stub, {"cdh_target_k": 3})
    assert out["cdh_target_k"] == 3
    assert stub._cdh_k_lo == 3
    assert stub.saved == []


def test_fallback_ignora_params_sin_cdh_target_k():
    stub = _OptStub({"cdh_target_k": 2})
    assert _run_fallback(stub, {"tsj": 4}) == {"tsj": 4}
    assert stub.calls == []
