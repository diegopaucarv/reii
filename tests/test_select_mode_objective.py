"""Integración: objetivo de la etapa 2 del optimizador con ``mode="select"``.

El bloqueo de la etapa 2 (100 % de trials centinela, 1050/1050) tenía dos
causas, ambas eliminadas aquí:

1. ``primary_stable`` salía de la intersección triple A∩B∩C, que era vacía en
   todos los trials -> ``if not primary_stable: return _SENTINEL_TRIAL``.
   Con ``select`` el primary es el conjunto del método GANADOR.
2. El objetivo ``agreement_sym`` promediaba ARI cruzados (A↔B, B↔C, A↔C), que
   dependían de que los tres métodos coincidieran. Ahora es
   ``(coverage, ARI_intra del ganador, -K)``.

Estos tests usan los números reales de ``output_2.log`` (A: ARI=0.350,
coverage=0.606; B: ARI=0.212, coverage=0.415).
"""

import numpy as np
import pytest

from reii.main_workflow import (
    _SENTINEL_TRIAL,
    Config,
    Optimizador,
    UCBuilderConfig,
)
from reii.method_selection import MethodMetrics, select_method


class _FakeUCE:
    def __init__(self, uid, cluster_id):
        self.id = uid
        self.cluster_id = cluster_id


class _FakeTrial:
    number = 0

    def __init__(self):
        self.attrs = {}

    def suggest_int(self, name, lo, hi):
        return {"min_forms_uc_1": 12, "forms_gap": 4, "tsj": 3}[name]

    def suggest_categorical(self, name, choices):
        return {
            "min_cluster_size_cdh": 0.10,
            "swap_iterations": 1,
            "min_r2_threshold": 0.05,
            "similarity_threshold": 0.45,
            "coref_weight": 0.1,
            "uc_window_size": 2,
        }[name]

    def set_user_attr(self, key, value):
        self.attrs[key] = value


def _selection_a_wins():
    return select_method(
        [
            MethodMetrics("A", 7, 0.350, 0.606, perturbation="A1<->A2"),
            MethodMetrics("B", 9, 0.212, 0.415, perturbation="B1<->B2"),
            MethodMetrics("C", 0, 0.0, 0.0, available=False, perturbation="C1<->C2"),
        ]
    )


def _make_optimizador(selection, total_uces=100, coverage_uces=40):
    opt = Optimizador.__new__(Optimizador)  # skip __init__ (segmenta el corpus)
    opt.config_base = Config()
    opt.uc_config = UCBuilderConfig()
    opt.total_uces = total_uces
    opt.progressive_segmenter = None
    opt._last_pairwise = None
    opt._last_selection = selection
    opt._current_stage_mode = "select"
    opt._search_space_cached = {
        "min_forms_uc_1": (10, 14),
        "forms_gap": (2, 7),
        "tsj": (2, 6),
        "min_cluster_size_cdh": [0.05, 0.08, 0.10, 0.15, 0.20],
        "swap_iterations": [1, 2, 5],
        "min_r2_threshold": [0.03, 0.05, 0.08, 0.10],
        "similarity_threshold": [0.25, 0.35, 0.45, 0.55, 0.65],
        "coref_weight": [0.0, 0.1, 0.2, 0.3, 0.4],
        "uc_window_size": [1, 2, 3, 4, 5],
    }
    # 7 clusters: el K que A reporta en la seleccion (`winner_k`). La
    # fixture tiene que ser coherente con el objetivo -K, que es el K REAL
    # del ganador y no el recuento de etiquetas de este doble.
    primary = [_FakeUCE(f"u{i}", i % 7) for i in range(coverage_uces)]
    result = (primary, None, ["a", "b"], [], [], None, None, None, None, {})
    opt._evaluacion_rapida = lambda cfg, uc_cfg=None: result
    return opt


class TestSelectObjective:
    def test_no_sentinel_when_a_method_is_stable(self):
        """El caso que antes daba centinela ahora devuelve valores reales."""
        opt = _make_optimizador(_selection_a_wins())
        trial = _FakeTrial()

        values = opt._evaluate_trial(trial)

        assert values != _SENTINEL_TRIAL, "no debe ser centinela: A es viable"
        coverage, ari_winner, neg_k = values
        assert coverage == pytest.approx(0.40)
        assert ari_winner == pytest.approx(0.350), "ARI del ganador, no agreement_sym"
        assert neg_k == -7.0  # K real del ganador (A), no n_clusters
        assert trial.attrs["k_ach"] == 7
        assert trial.attrs["selected_method"] == "A"
        assert trial.attrs["ari_winner"] == pytest.approx(0.350)

    def test_sentinel_when_no_viable_method(self):
        sel = select_method([MethodMetrics("A", 1, 0.9, 0.9)])  # K fuera de rango
        assert sel.winner is None
        opt = _make_optimizador(sel)
        assert opt._evaluate_trial(_FakeTrial()) == _SENTINEL_TRIAL

    def test_sentinel_when_selection_missing(self):
        opt = _make_optimizador(None)
        assert opt._evaluate_trial(_FakeTrial()) == _SENTINEL_TRIAL

    def test_coverage_gate_from_select_field(self):
        opt = _make_optimizador(
            _selection_a_wins(), total_uces=1000, coverage_uces=40
        )  # coverage = 0.04
        opt.config_base.optimize_coverage_gate_select = 0.10
        assert opt._evaluate_trial(_FakeTrial()) == _SENTINEL_TRIAL

        # Sin gate (0.01) el mismo trial es válido: el corte es la cobertura.
        opt.config_base.optimize_coverage_gate_select = 0.01
        values = opt._evaluate_trial(_FakeTrial())
        assert values != _SENTINEL_TRIAL
        assert values[0] == pytest.approx(0.04)

    def test_active_methods_includes_three_for_select(self):
        cfg = Config(classification_mode="select")
        assert Optimizador._active_methods(cfg) == {"wc", "coref", "emb"}
        cfg_auto = Config(classification_mode="auto")
        assert Optimizador._active_methods(cfg_auto) == {"wc", "coref", "emb"}

    def test_selection_is_deterministic_across_calls(self):
        opt = _make_optimizador(_selection_a_wins())
        first = opt._evaluate_trial(_FakeTrial())
        second = opt._evaluate_trial(_FakeTrial())
        assert first == second


class TestDominanceGateWiring:
    """El gate de dominancia usa la particion CRUDA del metodo.

    Caso real (probe v3, Method C con K=3): crudo {442, 25, 12} de 479
    asignadas -> 0.92 (dispara); la interseccion tight/loose {199, 25, 12}
    daria 0.84 (no dispararia). Dividir por n_total (528) daria 0.837.
    """

    def test_partition_sizes_exclude_noise(self):
        from reii.main_workflow import _partition_sizes_from_labels

        labels = {f"u{i}": 0 for i in range(442)}
        labels.update({f"v{i}": 1 for i in range(25)})
        labels.update({f"w{i}": 2 for i in range(12)})
        labels.update({f"n{i}": -1 for i in range(49)})

        sizes = _partition_sizes_from_labels(labels)
        assert sizes == [12, 25, 442]
        assert max(sizes) / sum(sizes) == pytest.approx(442 / 479)
        assert max(sizes) / sum(sizes) > 0.90  # dispara
        assert max(sizes) / len(labels) < 0.90  # diluir con -1 no debe valer

    def test_max_class_share_winner_is_recorded(self):
        sel = select_method([MethodMetrics("A", 4, 0.250, 0.574, max_class_share=0.52)])
        opt = _make_optimizador(sel)
        trial = _FakeTrial()
        opt._evaluate_trial(trial)
        assert trial.attrs["max_class_share_winner"] == pytest.approx(0.52)
