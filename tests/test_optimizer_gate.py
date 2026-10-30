"""
Tests for the Optimizador._evaluate_trial ARI/coverage fix.

Regression: with optimize_coverage_gate=0.65, ALL trials returned the
sentinel (-2.0, -1.0, 0.0) because the corpus's real A1↔A2 stability is
23-52% — the gate was mathematically unreachable, so best_params.json was
never written. The gate is now 0.25 and the ARI objective is computed over
ALL UCEs (not just the primary-stable subset, where ARI≈1.0 constant).
"""

import numpy as np
import pytest

from reii.main_workflow import (
    _SENTINEL_TRIAL,
    Config,
    DoubleClassifier,
    Optimizador,
    UCBuilderConfig,
)


class _FakeUCE:
    """Minimal UCE stand-in with the attributes _evaluate_trial touches."""

    def __init__(self, uid, cluster_id):
        self.id = uid
        self.cluster_id = cluster_id


class _FakeUC:
    def __init__(self, uce_ids, label):
        self.uce_ids = uce_ids
        self.label = label


def _make_wc_result(uce_to_label):
    """Build a wordcount result dict shaped like _clasificar_umbral_wordcount.

    uce_to_label: {uce_id: cluster_label} — labels are per-UCE (the dict
    returned by DoubleClassifier._extract_uce_labels is built from
    res['ucs']/res['labels']).
    """
    ucs = []
    labels = []
    for uid, lbl in sorted(uce_to_label.items()):
        ucs.append(_FakeUC([uid], lbl))
        labels.append(lbl)
    return {"ucs": ucs, "labels": np.array(labels), "voc": ["a", "b"]}


def _make_result_tuple(primary_stable, all_res, total_uces):
    """Build the 10-tuple returned by DoubleClassifier.run()."""
    return (
        primary_stable,  # [0] primary_stable
        None,  # [1] DataFrame (unused)
        ["a", "b"],  # [2] voc
        [],  # [3] uces_por_doc (unused)
        all_res,  # [4] all_res
        None,  # [5] labels1_ari (legacy, unused now)
        None,  # [6] labels2_ari (legacy, unused now)
        None,
        None,
        {},  # [9] doc_metadata_map
    )


@pytest.fixture
def optimizador():
    cfg = Config(optimize_coverage_gate=0.25)
    uc_cfg = UCBuilderConfig()
    opt = Optimizador.__new__(Optimizador)  # skip __init__ (heavy)
    opt.config_base = cfg
    opt.uc_config = uc_cfg
    opt.total_uces = 100
    # __init__ normalmente fija estos atributos; al saltarlo hay que
    # replicarlos a mano para que _evaluate_trial no reviente con
    # AttributeError. Ambos tests ejercitan la etapa 1 (wc_only), que es
    # donde se calcula el ARI sobre todas las UCEs (ver docstrings).
    opt.progressive_segmenter = None
    opt._last_pairwise = None
    opt._current_stage_mode = "wc_only"
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
    return opt


class TestEvaluateTrialARI:
    def test_ari_over_all_uces_not_stable_subset(self, optimizador, monkeypatch):
        """ARI must be computed over ALL UCEs, so it discriminates trials.

        Build two passes where 40 UCEs are stable (coverage 0.40 ≥ 0.25)
        but the full-corpus ARI is meaningfully < 1.0. The old code
        computed ARI on the stable subset only (≈1.0 constant).
        """
        # 100 UCEs total. Pass A1: labels 0/1 split 50/50.
        # Pass A2: same labels for 40 UCEs (stable, spread across BOTH
        # clusters: u0-19→0, u20-39→1), flipped for the rest.
        uce_to_a1 = {f"u{i}": (0 if i < 50 else 1) for i in range(100)}
        uce_to_a2 = {}
        for i in range(100):
            if i < 20:
                uce_to_a2[f"u{i}"] = 0  # stable, cluster 0
            elif i < 40:
                uce_to_a2[f"u{i}"] = 1  # stable, cluster 1
            else:
                uce_to_a2[f"u{i}"] = 1 - uce_to_a1[f"u{i}"]  # flipped

        res_wc1 = _make_wc_result(uce_to_a1)
        res_wc2 = _make_wc_result(uce_to_a2)

        # primary_stable = the 40 stable UCEs (coverage 0.40), 2 clusters
        primary_stable = [_FakeUCE(f"u{i}", 0 if i < 20 else 1) for i in range(40)]
        result = _make_result_tuple(primary_stable, [res_wc1, res_wc2], 100)

        # Monkeypatch _evaluacion_rapida to return our fixture
        monkeypatch.setattr(
            optimizador, "_evaluacion_rapida", lambda cfg, uc_cfg=None: result
        )

        class FakeTrial:
            number = 0

            def __init__(self):
                # `_evaluate_trial` registra attrs de diagnostico en la rama
                # wc_only (k_ach, k_stable, k_rejected); sin este metodo el
                # doble de test revienta con AttributeError.
                self.user_attrs = {}

            def set_user_attr(self, key, value):
                self.user_attrs[key] = value

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

        values = optimizador._evaluate_trial(FakeTrial())

        assert values != _SENTINEL_TRIAL, "trial should NOT be sentinel"
        ari, coverage, neg_k = values
        assert coverage == pytest.approx(0.40)
        assert 0.0 < ari < 1.0, (
            f"ARI should be discriminating (<1.0), got {ari:.3f}. "
            "If ARI≈1.0 the objective is computed on the stable subset."
        )
        assert neg_k == -2  # 2 clusters

    def test_coverage_gate_uses_config_value(self, optimizador, monkeypatch):
        """With gate=0.25, a 30% coverage trial passes; with 0.65 it fails."""
        uce_to_a1 = {f"u{i}": (0 if i < 50 else 1) for i in range(100)}
        uce_to_a2 = dict(uce_to_a1)  # perfect agreement → ARI=1.0
        res_wc1 = _make_wc_result(uce_to_a1)
        res_wc2 = _make_wc_result(uce_to_a2)

        # Only 30 UCEs are primary-stable (coverage 0.30), spread across
        # 2 clusters so the K gate (2..8) passes.
        primary_stable = [_FakeUCE(f"u{i}", 0 if i < 15 else 1) for i in range(30)]
        result = _make_result_tuple(primary_stable, [res_wc1, res_wc2], 100)
        monkeypatch.setattr(
            optimizador, "_evaluacion_rapida", lambda cfg, uc_cfg=None: result
        )

        class FakeTrial:
            number = 1

            def __init__(self):
                # `_evaluate_trial` registra attrs de diagnostico en la rama
                # wc_only (k_ach, k_stable, k_rejected); sin este metodo el
                # doble de test revienta con AttributeError.
                self.user_attrs = {}

            def set_user_attr(self, key, value):
                self.user_attrs[key] = value

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

        # gate 0.25 → passes
        optimizador.config_base.optimize_coverage_gate = 0.25
        values = optimizador._evaluate_trial(FakeTrial())
        assert values != _SENTINEL_TRIAL
        assert values[1] == pytest.approx(0.30)

        # gate 0.65 → sentinel (the old unreachable gate)
        optimizador.config_base.optimize_coverage_gate = 0.65
        values = optimizador._evaluate_trial(FakeTrial())
        assert values == _SENTINEL_TRIAL

    def test_extract_uce_labels_static(self):
        """DoubleClassifier._extract_uce_labels maps UC labels to UCE ids."""
        res = _make_wc_result({"a": 0, "b": 1, "c": 0})
        out = DoubleClassifier._extract_uce_labels(res)
        assert out == {"a": 0, "b": 1, "c": 0}
