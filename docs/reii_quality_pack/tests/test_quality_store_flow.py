"""Pruebas de persistencia, migración, hook, servicio, parches, CLI y panel."""

import json
import os
import shutil
import sys
import types

import numpy as np
import pandas as pd
import pytest
from reii.quality import calibration as cal
from reii.quality import hooks, service
from reii.quality import migrate_legacy as ml
from reii.quality.cli import main as cli_main
from reii.quality.store import QualityStore


def _df(n=100, seed=0, offset=0):
    rng = np.random.default_rng(seed)
    s = rng.choice([0.0, 0.5, 1.0], size=n, p=[0.3, 0.2, 0.5])
    return pd.DataFrame(
        {
            "uce_id": [f"u{i}" for i in range(n)],
            "doc_id": [f"d{i % 5}" for i in range(n)],
            "cluster_core": np.where(
                s > 0, (rng.integers(0, 3, n) + offset) % 3, np.nan
            ),
            "s_acuerdo": s,
            "s_margen": rng.random(n),
            "n_comparisons": 3,
            "s_source": "t",
            "projected_cluster_id": np.where(s == 0, rng.integers(0, 3, n), np.nan),
            "n_tokens": rng.integers(5, 30, n),
            "comps": [{"wc": bool(x > 0), "sim": bool(x >= 1)} for x in s],
        }
    )


@pytest.fixture
def store():
    s = QualityStore.open(":memory:")
    yield s
    s.close()


# ── store ───────────────────────────────────────────────────────────────────
def test_roundtrip_and_idempotent_save(store):
    store.save_run({"run_id": "r1", "n_uces": 100, "config": {"a": 1}})
    d = _df()
    store.save_assignments("r1", d)
    store.save_assignments("r1", d)
    back = store.load_assignments("r1")
    assert len(back) == 100
    assert (
        back["s_acuerdo"].round(6).tolist()
        == d.sort_values("uce_id")["s_acuerdo"].round(6).tolist()
    )
    assert store.get_run("r1")["config"] == {"a": 1}


def test_saving_one_run_never_touches_another(store):
    for rid, off in (("r1", 0), ("r2", 1)):
        store.save_run({"run_id": rid, "n_uces": 100})
        store.save_assignments(rid, _df(offset=off))
    before = store.load_assignments("r2")
    store.save_assignments("r1", _df(seed=9))
    after = store.load_assignments("r2")
    assert before["cluster_core"].equals(after["cluster_core"])


def test_several_results_per_tier_coexist(store):
    store.save_run({"run_id": "r1", "n_uces": 100})
    store.save_run({"run_id": "r2", "n_uces": 100})
    store.save_assignments("r1", _df())
    store.save_assignments("r2", _df(seed=5))
    p1 = cal.default_profile()
    p2 = cal.default_profile()
    p2["tiers"][1]["fallback_s"] = 0.9
    for rid in ("r1", "r2"):
        service.evaluate_run(store, rid, p1)
        service.evaluate_run(store, rid, p2)
    res = store.list_tier_results(tier_id="equilibrado")
    assert (
        len(res) == 4
        and res["profile_hash"].nunique() == 2
        and res["run_id"].nunique() == 2
    )
    again = len(store.list_tier_results())
    service.evaluate_run(store, "r1", p1)  # re-evaluar no duplica
    assert len(store.list_tier_results()) == again


def test_active_pointer_calibration_versions_and_audit(store):
    store.save_run({"run_id": "r1", "n_uces": 1})
    store.set_active("default", "r1", "estricto", "h")
    store.set_active("default", "r1", "amplio", "h")
    assert store.get_active()["tier_id"] == "amplio"
    v1 = store.save_calibration(
        "r1", {"method": "isotonic", "x": [0, 1], "y": [0, 1], "n": 80}
    )
    v2 = store.save_calibration(
        "r1", {"method": "isotonic", "x": [0, 1], "y": [0.1, 0.9], "n": 90}
    )
    assert (v1, v2) == (1, 2) and store.load_calibration("r1")["version"] == 2
    store.log("r1", "x", {"k": 1})
    assert len(store.audit("r1")) == 1


def test_sqlite_file_dsn(tmp_path):
    path = tmp_path / "q.db"
    s = QualityStore.open(f"sqlite:///{path}")
    s.save_run({"run_id": "r", "n_uces": 0})
    s.close()
    s2 = QualityStore.open(f"sqlite:///{path}")
    assert s2.get_run("r") is not None


# ── servicio: calibración con estándar humano ───────────────────────────────
def test_recalibrate_with_gold_and_precision_by_tier(store):
    rng = np.random.default_rng(1)
    d = _df(n=600, seed=2)
    store.save_run({"run_id": "r1", "n_uces": 600})
    store.save_assignments("r1", d)
    inductive = d[d["cluster_core"].notna()]
    ok = rng.random(len(inductive)) < inductive["s_acuerdo"].to_numpy() * 0.9 + 0.05
    store.save_gold(
        "r1",
        [
            {"uce_id": u, "coder": "c1", "is_correct": bool(o)}
            for u, o in zip(inductive["uce_id"], ok)
        ],
    )
    fit = service.recalibrate(store, "r1", min_n=60)
    assert fit and fit["version"] == 1
    r = service.evaluate_run(store, "r1", persist=False)
    assert (
        r["tiers"]["estricto"]["precision"] > r["tiers"]["amplio"]["precision"] - 1e-9
    )
    assert r["curve"] is not None and r["nested_ok"]
    assert store.load_assignments("r1")["p_cal"].notna().any()


def test_gold_majority_and_stratified_sample():
    g = pd.DataFrame(
        {
            "uce_id": ["a", "a", "b", "b"],
            "coder": ["x", "y", "x", "y"],
            "is_correct": [1, 0, 1, 1],
        }
    )
    c = service.gold_consensus(g).set_index("uce_id")["is_correct"]
    assert c["a"] == 0 and c["b"] == 1  # empate = incorrecto
    df = cal.assign_tiers(_df(), cal.default_profile())
    s = service.stratified_sample(df, 30)
    assert len(s) == 30 and s["uce_id"].is_unique


# ── migración de resultados anteriores ──────────────────────────────────────
def _legacy_json(path, n=200, secret=True):
    rng = np.random.default_rng(3)
    uces = []
    for i in range(n):
        st_ = bool(rng.random() < 0.5)
        uces.append(
            {
                "id": f"u{i}",
                "doc_id": i % 4,
                "local_idx": i,
                "cluster_id": int(rng.integers(0, 3)) if st_ else None,
                "is_stable": st_,
                "stability_method": "wc",
                "stems": ["a", "b", f"t{i % 3}"],
                "lemmas": ["a", "b", f"t{i % 3}"],
                "projected_cluster_id": None if st_ else 1,
            }
        )
    cfg = (
        {"classification_mode": "all", "random_state": 7, "together_api_key": "SECRETO"}
        if secret
        else {}
    )
    json.dump(
        {
            "uces": uces,
            "vocabulario": ["a", "b"],
            "config": cfg,
            "pairwise_stability": {"overlap_aa": [[10, 1], [2, 12]], "ari_wc": 0.4},
        },
        open(path, "w"),
    )
    return uces


def test_migrate_legacy_json_is_idempotent_and_faithful(store, tmp_path):
    p = tmp_path / "wf.json"
    uces = _legacy_json(p)
    legacy = ml.load_legacy_json(str(p))
    rep1 = ml.migrate(store, legacy)
    rep2 = ml.migrate(store, legacy)
    assert rep1["run_id"] == rep2["run_id"] and rep2["already_present"]
    assert rep1["n_uces"] == 200 and rep1["n_stable"] == sum(
        u["is_stable"] for u in uces
    )
    assert (
        len(store.list_runs()) == 1
        and len(store.load_assignments(rep1["run_id"])) == 200
    )
    run = store.get_run(rep1["run_id"])
    assert run["kind"] == "legacy" and run["config"]["together_api_key"] == "***"
    assert run["summary"]["retention_estricto"] == pytest.approx(rep1["retention"])
    assert run["summary"]["extras"]["coverage"]["cov_types_total"] > 0
    assert store.get_active()["run_id"] == rep1["run_id"]


def test_migrate_dry_run_writes_nothing(store, tmp_path):
    p = tmp_path / "wf.json"
    _legacy_json(p)
    rep = ml.migrate(store, ml.load_legacy_json(str(p)), dry_run=True)
    assert rep["dry_run"] and store.list_runs().empty


def test_legacy_projected_stay_out_of_strict_and_balanced(store, tmp_path):
    p = tmp_path / "wf.json"
    _legacy_json(p)
    rep = ml.migrate(store, ml.load_legacy_json(str(p)))
    r = service.evaluate_run(store, rep["run_id"], persist=False)
    t = r["tiers"]
    assert (
        t["estricto"]["n_assigned"] == t["equilibrado"]["n_assigned"] == rep["n_stable"]
    )
    assert t["amplio"]["n_projected"] > 0


# ── hook del workflow ───────────────────────────────────────────────────────
class _U:
    def __init__(self, i, doc, stable, cid, **flags):
        self.id, self.doc_id, self.is_stable, self.cluster_id = (
            f"u{i}",
            doc,
            stable,
            cid,
        )
        self.lemmas = ["a", "b"]
        self.stems = ["a", "b"]
        self.texto = f"texto {i}"
        self.projected_cluster_id = None
        self.projection_margin = None
        for k, v in flags.items():
            setattr(self, k, v)


def _wf(mode="all"):
    cfg = types.SimpleNamespace(
        classification_mode=mode,
        random_state=5,
        together_api_key="X",
        stem_backend="snowball",
        use_bigrams=False,
        use_trigrams=False,
    )
    cfg_dc = types.SimpleNamespace(**vars(cfg))
    wf = types.SimpleNamespace(
        config=cfg_dc,
        _vocab=["a"],
        db=types.SimpleNamespace(data={}),
        double_clf=types.SimpleNamespace(
            last_pairwise_stability={"overlap_aa": [[5, 0], [1, 6]], "ari_wc": 0.5}
        ),
    )
    return wf


def test_hook_with_per_comparison_flags(store, monkeypatch):
    monkeypatch.setattr(
        hooks,
        "redact_config",
        lambda c: {"mode": c.classification_mode, "together_api_key": "***"},
    )
    docs = [
        [
            _U(0, 1, True, 0, stable_wc=True, stable_sim=True, stable_emb=True),
            _U(1, 1, False, None, stable_wc=True, stable_sim=True, stable_emb=False),
            _U(2, 1, False, None, stable_wc=True, stable_sim=False, stable_emb=False),
            _U(3, 1, False, None, stable_wc=False, stable_sim=False, stable_emb=False),
        ]
    ]
    rid = hooks.save_quality_run(_wf(), [docs[0][0]], docs, store=store)
    assert rid and rid.startswith("run-")
    a = store.load_assignments(rid).set_index("uce_id")
    assert a.loc["u0", "s_acuerdo"] == 1.0 and a.loc[
        "u1", "s_acuerdo"
    ] == pytest.approx(2 / 3)
    assert (
        a.loc["u2", "s_acuerdo"] == pytest.approx(1 / 3)
        and a.loc["u3", "s_acuerdo"] == 0.0
    )
    assert pd.isna(a.loc["u1", "cluster_core"]) and a.loc["u0", "cluster_core"] == 0
    assert store.get_run(rid)["summary"]["extras"]["comparaciones_disponibles"] == [
        "wc",
        "sim",
        "emb",
    ]


def test_hook_without_flags_falls_back_to_is_stable(store, monkeypatch):
    monkeypatch.setattr(hooks, "redact_config", lambda c: {})
    docs = [[_U(0, 1, True, 1), _U(1, 1, False, None)]]
    rid = hooks.save_quality_run(_wf("wc_only"), [docs[0][0]], docs, store=store)
    a = store.load_assignments(rid).set_index("uce_id")
    assert (
        a["s_source"].iloc[0] == "is_stable"
        and a.loc["u0", "s_acuerdo"] == 1.0
        and a.loc["u1", "s_acuerdo"] == 0.0
    )


def test_hook_never_raises_and_run_id_is_deterministic(store, monkeypatch):
    assert hooks.save_quality_run(object(), [], [], store=store) is None  # wf inválido
    monkeypatch.setattr(hooks, "redact_config", lambda c: {})
    docs = [[_U(0, 1, True, 0)]]
    r1 = hooks.save_quality_run(_wf(), docs[0], docs, store=store)
    r2 = hooks.save_quality_run(_wf(), docs[0], docs, store=store)
    assert r1 == r2 and len(store.list_runs()) == 1


def test_redact_config_masks_secrets():
    import dataclasses

    @dataclasses.dataclass
    class C:
        together_api_key: str = "abc"
        db_dsn: str = "postgresql://u:p@h/db"
        seed: int = 3

    assert hooks.redact_config(C()) == {
        "together_api_key": "***",
        "db_dsn": "***",
        "seed": 3,
    }


# ── parches ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_patch_script_idempotent_crlf_and_revert(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src/reii").mkdir(parents=True)
    for f in ("main_workflow.py", "dashboard.py"):
        shutil.copy(os.path.join(ROOT, "src/reii", f), repo / "src/reii" / f)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import apply_quality_patches as ap

    argv = sys.argv
    try:
        sys.argv = ["x", "--repo", str(repo)]
        assert ap.main() == 0
        first = (repo / "src/reii/main_workflow.py").read_bytes()
        assert ap.main() == 0
        assert (repo / "src/reii/main_workflow.py").read_bytes() == first  # idempotente
        wf = first.decode("utf-8")
        assert (
            wf.count("[quality-patch]") == 2
            and "\r\n" in wf
            and "\n" not in wf.replace("\r\n", "")
        )
        compile(wf, "main_workflow.py", "exec")
        compile(
            (repo / "src/reii/dashboard.py").read_text(encoding="utf-8"), "d", "exec"
        )
        sys.argv = ["x", "--repo", str(repo), "--revert"]
        ap.main()
        assert "[quality-patch]" not in (repo / "src/reii/main_workflow.py").read_text(
            encoding="utf-8"
        )
    finally:
        sys.argv = argv


# ── CLI y panel ─────────────────────────────────────────────────────────────
def test_cli_selftest_and_migrate(tmp_path, capsys):
    assert cli_main(["selftest"]) == 0
    p = tmp_path / "wf.json"
    _legacy_json(p)
    dsn = f"sqlite:///{tmp_path / 'q.db'}"
    assert cli_main(["--dsn", dsn, "migrate-legacy", "--json", str(p)]) == 0
    assert cli_main(["--dsn", dsn, "runs"]) == 0
    assert "legacy-" in capsys.readouterr().out


def test_dashboard_panel_renders_and_switches_tier(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    p = tmp_path / "wf.json"
    _legacy_json(p)
    dsn = f"sqlite:///{tmp_path / 'q.db'}"
    s = QualityStore.open(dsn)
    ml.migrate(s, ml.load_legacy_json(str(p)))
    s.close()
    monkeypatch.setenv("Q_DSN", dsn)

    def app():
        import os

        from reii.quality.dashboard_panel import render_quality_panel
        from reii.quality.store import QualityStore

        render_quality_panel(QualityStore.open(os.environ["Q_DSN"]))

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("Retención" in m.label for m in at.metric)
    at.radio(key="q_tier").set_value("amplio").run()
    assert not at.exception, at.exception
