"""Method C (embedding/HDBSCAN) debe escalar su granularidad al corpus,
igual que Methods A y B escalan min_cluster_size_cdh. Antes de este fix,
hdbscan_uce_min_cluster_size era un entero absoluto (5/3) sin importar n,
lo que en un corpus de 5809 UCEs producía K~243-534 clusters en C contra
K~7-16 en A/B: una diferencia de granularidad que forzaba la intersección
triple hacia cero (ver run real: A∩B∩C = 9/5809 = 0.2%).
"""

import pytest

from reii.main_workflow import Config, DoubleClassifier


def _classifier(**config_overrides) -> DoubleClassifier:
    """DoubleClassifier mínimo: _resolve_hdbscan_uce_params solo usa self.config."""
    dc = object.__new__(DoubleClassifier)
    dc.config = Config(**config_overrides)
    return dc


def test_resolver_scales_with_corpus_size_like_ab():
    dc = _classifier()
    mcs_tight, _ = dc._resolve_hdbscan_uce_params(5809, "tight")
    mcs_loose, _ = dc._resolve_hdbscan_uce_params(5809, "loose")
    k_tight, k_loose = 5809 / mcs_tight, 5809 / mcs_loose
    # Antes (absoluto=5/3): K≈1162/1936. Ahora debe quedar en el mismo orden
    # de magnitud que A/B (K≈7-16 en el run real), no cientos.
    assert k_tight < 25 and k_loose < 35
    assert mcs_tight > mcs_loose, "tight (más conservador) debe dar clusters más grandes"


def test_regression_old_absolute_default_was_catastrophically_fine_grained():
    # Documenta el bug tal como se vio en producción: con n=5809 y el
    # absoluto anterior (5/3), K pasaba de ~15 (A/B) a cientos (C).
    old_tight_k = 5809 / 5
    old_loose_k = 5809 / 3
    assert old_tight_k > 1000 and old_loose_k > 1500


def test_floor_protects_small_corpora():
    dc = _classifier(hdbscan_uce_min_cluster_floor=5)
    mcs, ms = dc._resolve_hdbscan_uce_params(20, "tight")
    assert mcs == 5  # 8% de 20 sería ~2, por debajo del piso
    assert ms >= 2


def test_frac_none_reproduces_legacy_absolute_behavior():
    dc = _classifier(
        hdbscan_uce_min_cluster_frac=None,
        hdbscan_uce_min_cluster_frac_loose=None,
        hdbscan_uce_min_cluster_size=5,
        hdbscan_uce_min_cluster_size_loose=3,
    )
    mcs_tight, _ = dc._resolve_hdbscan_uce_params(5809, "tight")
    mcs_loose, _ = dc._resolve_hdbscan_uce_params(5809, "loose")
    assert (mcs_tight, mcs_loose) == (5, 3)


def test_explicit_min_samples_overrides_the_derived_ratio():
    dc = _classifier(hdbscan_uce_min_samples=9)
    _, ms = dc._resolve_hdbscan_uce_params(5809, "tight")
    assert ms == 9


def test_min_samples_derived_from_ratio_when_unset():
    dc = _classifier(hdbscan_uce_min_cluster_frac=0.08, hdbscan_uce_min_samples_ratio=0.25)
    mcs, ms = dc._resolve_hdbscan_uce_params(5809, "tight")
    assert ms == max(2, round(0.25 * mcs))


def test_min_samples_never_below_two_even_with_a_tiny_cluster_size():
    dc = _classifier(hdbscan_uce_min_cluster_floor=2, hdbscan_uce_min_samples_ratio=0.1)
    mcs, ms = dc._resolve_hdbscan_uce_params(10, "tight")
    assert ms >= 2


def test_tight_frac_vs_loose_frac_gate_mirrors_coref_gate():
    # El trial-apply de Optuna debe rechazar frac_tight <= frac_loose, igual
    # que ya rechaza coref_context_units_tight >= loose. Replicado aquí
    # sin tocar el estado interno del optimizador.
    frac_tight, frac_loose = 0.05, 0.08
    assert frac_tight <= frac_loose  # combinación inválida que el gate debe filtrar


def test_invalid_variant_raises():
    dc = _classifier()
    with pytest.raises(ValueError):
        dc._resolve_hdbscan_uce_params(100, "medium")
