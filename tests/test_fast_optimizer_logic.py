"""Fast tests for the optimizer's pure logic (no heavy imports).

Mirrors the pure functions in `Optimizador` (main_workflow.py) so the
PR-2 realignment can be validated without importing torch/stanza/spacy
(the module import alone takes ~280s locally).

Covers:
1. `_bottleneck` — which method has the lowest mean ARI against the others.
2. `_active_methods` — mode → set of active methods.
3. `_select_preference` — max_coverage→values[0], max_ari→values[1], knee→ideal.
4. `_perturb_params` — local perturbation of numeric/categorical params.
"""

import numpy as np


# ── Mirror of Optimizador._bottleneck ────────────────────────────────────────
def _bottleneck(ari_ab: float, ari_bc: float, ari_ac: float) -> str:
    mean_wc = np.mean([ari_ab, ari_ac])
    mean_coref = np.mean([ari_ab, ari_bc])
    mean_emb = np.mean([ari_bc, ari_ac])
    candidates = {"wc": mean_wc, "coref": mean_coref, "emb": mean_emb}
    return min(candidates, key=candidates.get)


# ── Mirror of Optimizador._active_methods ───────────────────────────────────
def _active_methods(mode: str) -> set:
    aliases = {
        "both": "wc_coref",
        "wc_sim": "wc_coref",
        "sim_only": "coref_only",
        "sim_emb": "coref_emb",
    }
    mode = aliases.get(mode, mode)
    return {
        "wc_only": {"wc"},
        "coref_only": {"coref"},
        "emb_only": {"emb"},
        "wc_coref": {"wc", "coref"},
        "wc_emb": {"wc", "emb"},
        "coref_emb": {"coref", "emb"},
        "all": {"wc", "coref", "emb"},
    }.get(mode, {"wc"})


# ── Mirror of Optimizador._select_preference (index selection only) ─────────
def _select_preference_index(values_list, pref: str) -> int:
    best = values_list
    if not best:
        return 0
    if pref == "max_coverage":
        return int(max(range(len(best)), key=lambda i: best[i][0]))
    if pref == "knee":
        arr = np.array(best, dtype=np.float64)
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        rng = np.where(maxs > mins, maxs - mins, 1.0)
        norm = (arr - mins) / rng
        ideal = np.array([1.0, 1.0, 1.0])
        return int(np.argmin(np.linalg.norm(norm - ideal, axis=1)))
    # default max_ari → values[1] (agreement_sym)
    return int(max(range(len(best)), key=lambda i: best[i][1]))


# ── Mirror of Optimizador._perturb_params ───────────────────────────────────
def _perturb_params(params, space, rng):
    perturbed = dict(params)
    for k, v in params.items():
        spec = space.get(k)
        if spec is None:
            continue
        if isinstance(spec, tuple) and len(spec) == 2:
            lo, hi = spec
            step = 1 if isinstance(lo, int) else (hi - lo) * 0.1
            delta = rng.choice([-step, +step])
            new = max(lo, min(hi, v + delta))
            perturbed[k] = int(new) if isinstance(lo, int) else float(new)
        elif isinstance(spec, list) and len(spec) > 1:
            try:
                idx = spec.index(v)
                new_idx = max(0, min(len(spec) - 1, idx + rng.choice([-1, +1])))
                perturbed[k] = spec[new_idx]
            except ValueError:
                continue
    return perturbed


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════


class TestBottleneck:
    def test_coref_is_bottleneck_when_lowest(self):
        # coref agrees poorly with both wc and emb
        assert _bottleneck(0.1, 0.05, 0.8) == "coref"

    def test_wc_is_bottleneck_when_lowest(self):
        assert _bottleneck(0.1, 0.9, 0.1) == "wc"

    def test_emb_is_bottleneck_when_lowest(self):
        assert _bottleneck(0.9, 0.1, 0.1) == "emb"


class TestActiveMethods:
    def test_all_has_three(self):
        assert _active_methods("all") == {"wc", "coref", "emb"}

    def test_wc_only(self):
        assert _active_methods("wc_only") == {"wc"}

    def test_alias_both_maps_to_wc_coref(self):
        assert _active_methods("both") == {"wc", "coref"}

    def test_unknown_mode_defaults_to_wc(self):
        assert _active_methods("nonsense") == {"wc"}


class TestSelectPreference:
    def test_max_coverage_picks_highest_first(self):
        vals = [[0.3, 0.5, -3], [0.8, 0.1, -2], [0.5, 0.4, -4]]
        assert _select_preference_index(vals, "max_coverage") == 1

    def test_max_ari_picks_highest_second(self):
        vals = [[0.3, 0.5, -3], [0.8, 0.1, -2], [0.5, 0.4, -4]]
        assert _select_preference_index(vals, "max_ari") == 0

    def test_knee_picks_closest_to_ideal(self):
        # ideal (1,1,1); the first is clearly closest
        vals = [[0.9, 0.9, -2], [0.1, 0.1, -8], [0.5, 0.5, -5]]
        assert _select_preference_index(vals, "knee") == 0


class TestPerturbParams:
    def test_numeric_int_perturbs_by_one_step(self):
        import random

        rng = random.Random(42)
        space = {"min_forms_uc_1": (10, 14)}
        out = _perturb_params({"min_forms_uc_1": 12}, space, rng)
        assert out["min_forms_uc_1"] in (11, 13)

    def test_categorical_perturbs_by_one_index(self):
        import random

        rng = random.Random(42)
        space = {"swap_iterations": [1, 2, 5]}
        out = _perturb_params({"swap_iterations": 2}, space, rng)
        assert out["swap_iterations"] in (1, 5)

    def test_unknown_key_passed_through(self):
        import random

        rng = random.Random(42)
        out = _perturb_params({"foo": "bar"}, {}, rng)
        assert out["foo"] == "bar"
