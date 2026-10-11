"""Pruebas de reii.quality: mapeo, consenso, métricas, calibración y niveles."""

import numpy as np
import pandas as pd
import pytest
from reii.quality import calibration as cal
from reii.quality import consensus as cons
from reii.quality import mapping as mp
from reii.quality import metrics as M
from scipy.optimize import linear_sum_assignment


# ── mapeo ───────────────────────────────────────────────────────────────────
def _labels(pairs):
    return {f"u{i}": a for i, (a, _) in enumerate(pairs)}, {
        f"u{i}": b for i, (_, b) in enumerate(pairs)
    }


def test_many_to_one_recovers_ucees_lost_by_hungarian_when_k_differs():
    a = {f"u{i}": i % 4 for i in range(40)}
    b = {f"u{i}": (i % 4 if i % 4 < 3 else 2) for i in range(40)}  # B fusiona 2 y 3
    h = mp.stability(a, b, "hungarian")
    m = mp.stability(a, b, "many_to_one")
    assert h["n_stable"] == 30 and m["n_stable"] == 40
    assert m["mapping"][3] == [2]


def test_many_to_one_equals_hungarian_when_k_equal_and_matched():
    rng = np.random.default_rng(1)
    a = {f"u{i}": int(rng.integers(0, 3)) for i in range(120)}
    b = {
        u: (c if rng.random() < 0.8 else int(rng.integers(0, 3))) for u, c in a.items()
    }
    h = mp.stability(a, b, "hungarian")
    m = mp.stability(a, b, "many_to_one")
    assert set(h["stable"]) == set(m["stable"])


def test_hungarian_matches_scipy_reference():
    rng = np.random.default_rng(2)
    a = {f"u{i}": int(rng.integers(0, 4)) for i in range(200)}
    b = {f"u{i}": int(rng.integers(0, 4)) for i in range(200)}
    ov, ra, rb = mp.overlap_matrix(a, b)
    r, c = linear_sum_assignment(-ov)
    ref = {
        u for u in a if (list(ra).index(a[u]), list(rb).index(b[u])) in set(zip(r, c))
    }
    assert set(mp.stability(a, b, "hungarian")["stable"]) == ref


def test_max_merge_prevents_collapse_into_one_class():
    a = {f"u{i}": i % 5 for i in range(100)}
    b = {u: 0 for u in a}  # B colapsa todo en una clase
    m = mp.stability(a, b, "many_to_one", max_merge=2, min_share=0.1)
    absorbed = sum(1 for k, v in m["mapping"].items() if 0 in v)
    assert absorbed <= 2


def test_min_share_blocks_weak_merges():
    # la clase b2 de B se reparte 50/50 entre a0 y a1: su pertenencia a cualquiera es débil
    pairs = [(0, 0)] * 20 + [(1, 1)] * 20 + [(0, 2)] * 10 + [(1, 2)] * 10
    a, b = _labels(pairs)
    strict = mp.stability(a, b, "many_to_one", min_share=0.6)
    loose = mp.stability(a, b, "many_to_one", min_share=0.4)
    assert all(len(v) == 1 for v in strict["mapping"].values())
    assert any(len(v) == 2 for v in loose["mapping"].values())
    assert loose["n_stable"] > strict["n_stable"]


def test_refinement_is_absorbed_when_b_splits_a_cleanly():
    a = {f"u{i}": i % 2 for i in range(40)}
    b = {f"u{i}": i % 4 for i in range(40)}  # B subdivide limpiamente cada clase de A
    m = mp.stability(a, b, "many_to_one", min_share=0.6)
    assert m["n_stable"] == 40


def test_noise_labels_never_stable():
    a = {"u0": -1, "u1": 0, "u2": 0, "u3": 1}
    b = {"u0": -1, "u1": 0, "u2": 0, "u3": 1}
    assert "u0" not in mp.stability(a, b)["stable"]


def test_hungarian_compatible_signature(capsys):
    a = {f"u{i}": i % 3 for i in range(30)}
    out = mp.hungarian_compatible(a, a, "many_to_one", "t")
    stable, ari, arr_a, arr_b, overlap, mapping = out
    assert len(stable) == 30 and ari == pytest.approx(1.0) and overlap.shape == (3, 3)
    assert all(isinstance(k, int) and isinstance(v, int) for k, v in mapping.items())


# ── consenso ────────────────────────────────────────────────────────────────
def test_agreement_excludes_unavailable_comparisons():
    ids = ["a", "b", "c"]
    sets = {"wc": ["a", "b", "c"], "sim": ["a", "b"], "emb": []}
    df = cons.agreement_from_stable_sets(ids, sets)  # emb no corrió: se excluye
    assert df["n_comparisons"].iloc[0] == 2
    assert df.loc["a", "s_acuerdo"] == 1.0 and df.loc["c", "s_acuerdo"] == 0.5


def test_agreement_in_unit_interval_and_weights():
    ids = [f"u{i}" for i in range(10)]
    sets = {"wc": ids[:8], "sim": ids[:4]}
    df = cons.agreement_from_stable_sets(ids, sets, weights={"wc": 3, "sim": 1})
    assert df["s_acuerdo"].between(0, 1).all()
    assert df.loc["u0", "s_acuerdo"] == 1.0 and df.loc["u5", "s_acuerdo"] == 0.75


def test_chi2_margin_high_for_clear_rows():
    mat = np.array([[1, 1, 0, 0]] * 10 + [[0, 0, 1, 1]] * 10 + [[1, 0, 1, 0]])
    labels = np.array([0] * 10 + [1] * 10 + [0])
    m = cons.chi2_margin(mat, labels)
    assert m[0] > m[-1] and np.nanmax(m) <= 1.0


# ── métricas ────────────────────────────────────────────────────────────────
def test_vocabulary_coverage_by_hand():
    tl = [["a", "a", "b"], ["c", "a", "d"]]
    r = M.vocabulary_coverage(tl, {"a", "b"})
    assert r["n_types"] == 4 and r["n_tokens"] == 6
    assert r["cov_types"] == pytest.approx(0.5) and r["cov_tokens"] == pytest.approx(
        4 / 6
    )
    assert r["hapax_pct"] == pytest.approx(3 / 4)


def test_corpus_vocabulary_stats_tsj_and_hapax():
    # corpus: a(3), b(2), c(1), d(1) -> 4 types, 2 hápax, activos (tsj=3): {a}
    corpus = [["a", "b", "c"], ["a", "b", "d"], ["a"]]
    s = M.corpus_vocabulary_stats(corpus, tsj=3)
    assert s["n_types_total"] == 4
    assert s["n_types_active"] == 1
    assert s["n_hapax"] == 2
    assert s["hapax_pct"] == pytest.approx(0.5)
    assert s["ttr"] == pytest.approx(4 / 7)
    assert s["active_types"] == {"a"}


def test_corpus_coverage_direction_and_dual_reporting():
    # corpus: a(3), b(2), c(1), d(1); UCEs estables cubren {a, b}
    corpus = [["a", "b", "c"], ["a", "b", "d"], ["a"]]
    stable = [["a", "b"]]
    r = M.corpus_coverage(stable, corpus, tsj=3)
    # total: 2/4 = 0.5; activo: {a,b} ∩ {a} = 1/1 = 1.0
    assert r["cov_types_total"] == pytest.approx(0.5)
    assert r["cov_types_active"] == pytest.approx(1.0)
    assert r["n_uce_types"] == 2


def test_coverage_by_class_identifies_rich_vs_poor():
    corpus = [["a", "b", "c"], ["a", "b", "d"], ["a"]]
    by_class = {0: [["a", "b"]], 1: [["c"]]}
    r = M.coverage_by_class(by_class, corpus, tsj=3)
    assert r[0]["cov_types_active"] == pytest.approx(1.0)
    assert r[1]["cov_types_active"] == pytest.approx(0.0)


def test_null_coverage_zscore_high_for_rich_selection():
    # corpus con 100 types activos; UCEs estables cubren 50 de ellos
    corpus = [[f"t{i}"] for i in range(100)]
    stable = [[f"t{i}"] for i in range(50)]
    r = M.null_coverage(stable, corpus, tsj=1, n_perm=200, seed=0)
    assert r["observed"] == pytest.approx(0.5)
    # La selección rica (50 types únicos) supera claramente el azar (~39 únicos
    # esperados al muestrear 50 con reemplazo), pero el z-score no llega a 5.
    assert r["z_score"] > 0
    assert r["p_value"] < 0.05


def test_retention_funnel_is_cumulative_and_monotone():
    df = pd.DataFrame(
        {"wc": [1, 1, 1, 0], "sim": [1, 1, 0, 0], "emb": [1, 0, 0, 0]}
    ).astype(bool)
    f = M.retention_funnel(df)
    ns = [s["n"] for s in f]
    assert ns == [4, 3, 2, 1]


def test_jaccard_from_overlap_identity_is_one():
    ov = np.diag([10, 20, 30]).astype(float)
    assert all(v == pytest.approx(1.0) for v in M.jaccard_from_overlap(ov).values())


def test_risk_coverage_perfect_scores():
    s = np.linspace(1, 0, 20)
    y = (s > 0.5).astype(int)
    c = M.risk_coverage(s, y)
    assert c["precision"].iloc[0] == 1.0 and M.aurc(c) < 0.3


def test_tier_validity_and_bootstrap_ci_contains_estimate():
    y = np.array([1] * 40 + [0] * 10)
    inc = np.array([True] * 30 + [False] * 10 + [True] * 10)
    v = M.tier_validity(inc, y)
    assert v["precision_lo"] <= v["precision"] <= v["precision_hi"]


def test_smd_detects_shift():
    x = pd.Series(list(range(100)) + list(range(200, 300)))
    ret = pd.Series([False] * 100 + [True] * 100)
    assert M.smd(x, ret) > 1.0


def test_pareto_mask_drops_dominated():
    df = pd.DataFrame({"ret": [0.5, 0.6, 0.4], "ari": [0.5, 0.6, 0.3]})
    assert M.pareto_mask(df, ["ret", "ari"]).tolist() == [False, True, False]


def test_kendall_top_terms_identical_is_one():
    d = {f"t{i}": 100 - i for i in range(60)}
    assert M.kendall_top_terms(d, d, k=50) == pytest.approx(1.0)


# ── calibración y niveles ───────────────────────────────────────────────────
def _tier_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.choice([0, 1 / 3, 2 / 3, 1.0], size=n, p=[0.2, 0.2, 0.25, 0.35])
    core = np.where(s > 0, rng.integers(0, 4, n), np.nan)
    proj = np.where(
        np.isnan(core) & (rng.random(n) < 0.5), rng.integers(0, 4, n), np.nan
    )
    return pd.DataFrame(
        {
            "uce_id": [f"u{i}" for i in range(n)],
            "cluster_core": core,
            "s_acuerdo": s,
            "projected_cluster_id": proj,
        }
    )


def test_tiers_are_nested_for_any_thresholds():
    df = _tier_df()
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = cal.default_profile()
        p["tiers"][1]["fallback_s"] = float(rng.uniform(0.3, 0.9))
        p["tiers"][2]["fallback_s"] = float(rng.uniform(0.1, 0.9))
        out = cal.assign_tiers(df, p)
        assert cal.check_nesting(out, ["estricto", "equilibrado", "amplio"])


def test_strict_tier_is_full_agreement_and_projected_only_in_amplio():
    df = _tier_df()
    out = cal.assign_tiers(df, cal.default_profile())
    assert (out.loc[out["in_estricto"], "s_acuerdo"] == 1.0).all()
    ind = out["cluster_core"].notna()
    assert not (out["in_equilibrado"] & ~ind).any()
    assert (out["in_amplio"] & ~ind).any()


def test_core_cluster_never_changes_across_tiers():
    df = _tier_df()
    out = cal.assign_tiers(df, cal.default_profile())
    both = out["in_estricto"] & out["in_amplio"]
    assert (
        out.loc[both, "shown_cluster_estricto"] == out.loc[both, "shown_cluster_amplio"]
    ).all()


def test_isotonic_is_monotone_and_requires_min_n():
    rng = np.random.default_rng(4)
    s = rng.random(300)
    y = (rng.random(300) < s).astype(int)
    fit = cal.fit_isotonic(s, y, min_n=60)
    assert np.all(np.diff(fit["y"]) >= -1e-12)
    assert cal.fit_isotonic(s[:20], y[:20], min_n=60) is None
    p = cal.apply_calibration(fit, [0.1, 0.9])
    assert p[0] < p[1]


def test_calibrated_threshold_applies_to_p_cal():
    df = _tier_df()
    fit = {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0], "n": 100}
    out = cal.assign_tiers(df, cal.default_profile(), fit)
    sel = out["in_equilibrado"] & out["cluster_core"].notna()
    assert (out.loc[sel, "p_cal"] >= 0.80 - 1e-9).all()


def test_exploratory_flag_and_stable_hash():
    p = cal.default_profile()
    assert not cal.is_exploratory(p)
    p["tiers"][1]["threshold"] = 0.75
    assert cal.is_exploratory(p)
    assert cal.profile_hash(cal.default_profile()) == cal.REGISTERED_HASH


def test_threshold_for_precision():
    s = np.linspace(1, 0, 100)
    y = (s > 0.4).astype(int)
    t = cal.threshold_for_precision(s, y, 0.95, min_n=10)
    assert t is not None and t > 0.3
