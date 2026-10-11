"""Validez interna y externa: Silhouette, C_ext, kappa entre codificadores, comparación entre corridas."""

import hashlib
import json
import types

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from reii.quality import compare_modes, hooks, service
from reii.quality import metrics as M
from reii.quality.store import QualityStore


# ── fixtures ────────────────────────────────────────────────────────────────
def _corpus(n=120, seed=0):
    rng = np.random.default_rng(seed)
    cls = rng.integers(0, 3, n)
    rows = []
    for i, c in enumerate(cls):
        stems = [f"c{c}w{j}" for j in rng.integers(0, 6, 4)] + ["comun", f"x{rng.integers(0, 3)}"]
        ling = {"stems": stems, "lemmas": stems, "bigram_stems": [], "trigram_stems": [], "bigrams": [], "trigrams": []}
        rows.append((f"u{i}", f"texto {i}", ling))
    return cls, rows


def _install_uces(store, rows, texto_suffix=""):
    store.conn.execute("CREATE TABLE IF NOT EXISTS uces (uce_id TEXT, texto TEXT, linguistic_json TEXT)")
    store.conn.execute("DELETE FROM uces")
    for uid, t, ling in rows:
        store.conn.execute("INSERT INTO uces VALUES (%s,%s,%s)", (uid, t + texto_suffix, json.dumps(ling)))


def _save_run(store, rid, cls, rows, label="r", mode="all", perm=(0, 1, 2), extras=None):
    n = len(cls)
    store.save_run(
        {
            "run_id": rid, "label": label, "kind": "pipeline", "mode": mode, "seed": 1,
            "corpus_hash": service.corpus_fingerprint((u, t) for u, t, _ in rows),
            "config": {"stem_backend": "snowball", "use_bigrams": False, "use_trigrams": False, "tsj": 3},
            "summary": {"extras": extras or {}}, "n_uces": n, "n_uces_total": n,
        }
    )
    store.save_assignments(
        rid,
        pd.DataFrame(
            {
                "uce_id": [r[0] for r in rows], "doc_id": "d0",
                "cluster_core": [perm[c] for c in cls], "s_acuerdo": 1.0, "s_margen": 0.5,
                "n_comparisons": 1, "s_source": "t", "projected_cluster_id": np.nan,
                "n_tokens": 6, "comps": [{"wc": True}] * n,
            }
        ),
    )


@pytest.fixture
def store():
    s = QualityStore.open(":memory:")
    yield s
    s.close()


@pytest.fixture
def two_runs(store):
    cls, rows = _corpus()
    _install_uces(store, rows)
    _save_run(store, "run-a", cls, rows, "A · wc", "wc_only")
    _save_run(store, "run-b", cls, rows, "B · all", "all", perm=(2, 0, 1))  # mismas clases, ids permutados
    return cls, rows


# ── métricas puras ──────────────────────────────────────────────────────────
def test_silhouette_separates_structure_from_noise_in_all_spaces():
    cls, rows = _corpus(300)
    terms = [r[2]["stems"] for r in rows]
    X = M.binary_term_matrix(terms, M.prune_vocabulary(terms, 3, 2))
    shuffled = np.random.default_rng(1).permutation(cls)
    for metric in ("chi2", "cosine"):
        good, bad = M.silhouette_labels(X, cls, metric), M.silhouette_labels(X, shuffled, metric)
        assert good > 0.2 and abs(bad) < 0.05


def test_silhouette_edge_cases_and_sampling_is_deterministic():
    X = sparse.csr_matrix(np.eye(6))
    assert M.silhouette_labels(X, [0] * 6) is None  # una sola clase
    assert M.silhouette_labels(X, [-1] * 6) is None  # solo ruido
    cls, rows = _corpus(400)
    Xb = M.binary_term_matrix([r[2]["stems"] for r in rows], sorted({t for r in rows for t in r[2]["stems"]}))
    a = M.silhouette_labels(Xb, cls, "chi2", sample_size=100, seed=3)
    assert a == M.silhouette_labels(Xb, cls, "chi2", sample_size=100, seed=3)
    with pytest.raises(ValueError):
        M.silhouette_labels(Xb, cls[:-1])


def test_chi2_distance_matches_definition():
    rng = np.random.default_rng(0)
    X = (rng.random((8, 12)) < 0.4).astype(float)
    X[X.sum(axis=1) == 0, 0] = 1
    Z, ok = M.chi2_profiles(sparse.csr_matrix(X))
    P, marg = X / X.sum(1, keepdims=True), X.sum(0) / X.sum()
    d_ref = np.sqrt((((P[0] - P[1]) ** 2) / marg).sum())
    assert np.linalg.norm(Z[0].toarray() - Z[1].toarray()) == pytest.approx(d_ref)


def test_cross_partition_agreement_is_permutation_invariant():
    a = {f"u{i}": i % 3 for i in range(30)}
    b = {u: (c + 1) % 3 for u, c in a.items()}
    r = M.cross_partition_agreement(a, b)
    assert r["n_common"] == 30 and r["fmi"] == r["nmi"] == r["ari"] == pytest.approx(1.0)
    noisy = M.cross_partition_agreement(a, {u: int(np.random.default_rng(i).integers(0, 3)) for i, u in enumerate(a)})
    assert noisy["ari"] < 0.3
    assert np.isnan(M.cross_partition_agreement({"x": 0}, {"x": 0})["fmi"])
    assert M.cross_partition_agreement({"x": -1, "y": 0}, {"x": 0, "y": 0})["n_common"] == 1  # ruido fuera


def test_cohen_kappa_degenerate_and_known_value():
    assert M.cohen_kappa([0, 1, 1, 0], [0, 1, 0, 0]) == pytest.approx(0.5)
    assert np.isnan(M.cohen_kappa([1, 1, 1], [1, 1, 1]))
    assert np.isnan(M.cohen_kappa([1], [1]))
    assert M.cohen_kappa([0, 1, np.nan, 1], [0, 1, 1, 1]) == pytest.approx(M.cohen_kappa([0, 1, 1], [0, 1, 1]))
    assert M.percent_agreement([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_external_coherence_separates_id_space_from_structure():
    r = M.external_coherence([0, 0, 1, 1, 2, 2], [7, 7, 8, 8, 9, 9])  # misma estructura, otra numeración
    assert r["accuracy"] == 0.0 and r["accuracy_aligned"] == 1.0 and r["nmi"] == pytest.approx(1.0)
    assert M.external_coherence([0, 1, 2, 0], [0, 1, 2, 1])["accuracy"] == 0.75
    assert np.isnan(M.external_coherence([], [])["accuracy"])


# ── estándar humano: clase efectiva, C_ext, kappa entre codificadores ───────
def _gold_for(df, coder, n_wrong=10, shift=1):
    """Codificador que acepta todo salvo ``n_wrong`` UCEs, donde escribe otra clase."""
    rows = []
    for i, r in enumerate(df.itertuples()):
        if i < n_wrong:
            rows.append({"uce_id": r.uce_id, "coder": coder, "is_correct": False, "human_cluster": (int(r.cluster_core) + shift) % 3})
        else:
            rows.append({"uce_id": r.uce_id, "coder": coder, "is_correct": True, "human_cluster": None})
    return rows


def test_c_ext_counts_accepted_judgements_not_only_corrections(store, two_runs):
    """Con 90 «correctas» sin clase y 10 correcciones, la exactitud debe ser 0.9, no ~0."""
    res = service.evaluate_run(store, "run-a", persist=False)
    store.save_gold("run-a", _gold_for(res["df"].head(100), "c1"))
    t = service.evaluate_run(store, "run-a", persist=False)["tiers"]["estricto"]
    assert t["c_ext_n"] == 100 and t["c_ext_n_explicit"] == 10
    assert t["c_ext"]["accuracy"] == pytest.approx(0.9)
    assert t["c_ext"]["kappa"] > 0.8 and t["c_ext"]["accuracy_aligned"] >= 0.9


def test_c_ext_needs_minimum_and_never_stores_nan(store, two_runs):
    res = service.evaluate_run(store, "run-a", persist=False)
    store.save_gold("run-a", _gold_for(res["df"].head(5), "c1", n_wrong=1))
    t = service.evaluate_run(store, "run-a", persist=False)["tiers"]["estricto"]
    assert t["c_ext"] is None and t["c_ext_n"] == 5
    df_b = service.evaluate_run(store, "run-b", persist=False)["df"]
    one_class = df_b[df_b["cluster_core"] == 0].head(30)  # una sola categoría: kappa degenerado
    store.save_gold("run-b", _gold_for(one_class, "c1", n_wrong=0))
    t = service.evaluate_run(store, "run-b", persist=True)["tiers"]["estricto"]
    assert t["c_ext"]["accuracy"] == 1.0 and t["c_ext"]["kappa"] is None
    saved = store.list_tier_results(run_id="run-b").iloc[0].metrics
    json.loads(json.dumps(saved["c_ext"], allow_nan=False))  # C_ext: JSON estricto, sin NaN


def test_consensus_class_majority_and_ties_dropped(store, two_runs):
    df = service.evaluate_run(store, "run-a", persist=False)["df"].head(3)
    g = pd.DataFrame(
        [
            {"uce_id": "u0", "coder": "a", "is_correct": 0, "human_cluster": 1},
            {"uce_id": "u0", "coder": "b", "is_correct": 0, "human_cluster": 1},
            {"uce_id": "u0", "coder": "c", "is_correct": 0, "human_cluster": 2},
            {"uce_id": "u1", "coder": "a", "is_correct": 0, "human_cluster": 1},
            {"uce_id": "u1", "coder": "b", "is_correct": 0, "human_cluster": 2},
        ]
    )
    c = service.consensus_human_class(service.resolve_human_classes(g, df)).set_index("uce_id")
    assert c.loc["u0", "human_class"] == 1 and "u1" not in c.index


def test_gold_rows_from_editor_derives_correctness_from_class():
    ed = pd.DataFrame(
        {
            "uce_id": ["a", "b", "c", "d"],
            "clase_pipeline": pd.Series(["2", "2", pd.NA, "1"], dtype="string"),
            "correcta": [False, True, False, True],
            "clase_humana": pd.array([2, 0, 1, pd.NA], dtype="Int64"),
        }
    )
    r = {x["uce_id"]: x for x in service.gold_rows_from_editor(ed, "k")}
    assert r["a"]["is_correct"] is True and r["a"]["human_cluster"] == 2  # coincide con el pipeline
    assert r["b"]["is_correct"] is False  # la clase escrita manda sobre el checkbox
    assert r["c"]["is_correct"] is False and r["c"]["human_cluster"] == 1  # el pipeline no etiquetó
    assert r["d"]["is_correct"] is True and r["d"]["human_cluster"] is None


def test_coder_agreement_binary_and_class(store, two_runs):
    df = service.evaluate_run(store, "run-a", persist=False)["df"].head(60)
    g1 = _gold_for(df, "c1", n_wrong=10)
    g2 = _gold_for(df, "c2", n_wrong=10)
    g2[0]["human_cluster"] = (g2[0]["human_cluster"] + 1) % 3  # discrepan en la clase de una UCE
    gold = pd.DataFrame(g1 + g2)
    ag = service.coder_agreement(gold, df, "c1", "c2")
    assert ag["binary"]["pct_agreement"] == 1.0 and ag["binary"]["kappa"] == pytest.approx(1.0)
    assert ag["class"]["pct_agreement"] == pytest.approx(59 / 60) and ag["class"]["kappa"] < 1.0
    few = service.coder_agreement(gold[gold.uce_id.isin(df.uce_id.head(5))], df, "c1", "c2")
    assert few["binary"] is None and few["class"] is None and few["n_common"] == 5


# ── corpus, Silhouette recalculado y comparación entre corridas ─────────────
def test_corpus_fingerprint_matches_legacy_hook_formula():
    pairs = [("u2", "hola"), ("u1", None), ("u3", "mundo")]
    old = hashlib.sha1(
        "\n".join(sorted(f"{u}|{hashlib.sha1((t or '').encode()).hexdigest()[:8]}" for u, t in pairs)).encode()
    ).hexdigest()[:16]
    assert service.corpus_fingerprint(pairs) == old


def test_lazy_silhouette_persists_without_clobbering_summary(store, two_runs, monkeypatch):
    service.evaluate_run(store, "run-a", persist=True)  # deja claves en summary
    before = store.get_run("run-a")["summary"]
    rep = service.ensure_silhouette(store, "run-a")
    assert rep["status"] == "ok" and rep["origin"] == "lazy" and rep["corpus_check"] == "hash"
    assert rep["chi2"] > 0.2 and rep["ppmi_cosine"] is None and rep["vocab_source"] == "derived_tsj"
    after = store.get_run("run-a")["summary"]
    assert after["extras"]["silhouette"]["chi2"] == rep["chi2"]
    assert {k: v for k, v in after.items() if k != "extras"} == {k: v for k, v in before.items() if k != "extras"}
    # segunda llamada: sale de caché, no vuelve a leer la tabla uces
    monkeypatch.setattr(service, "load_corpus_for_run", lambda *a, **k: pytest.fail("no debía releer uces"))
    assert service.ensure_silhouette(store, "run-a")["chi2"] == rep["chi2"]
    # re-evaluar conserva el Silhouette y lo expone en el resumen
    monkeypatch.undo()
    service.evaluate_run(store, "run-a", persist=True)
    s = store.get_run("run-a")["summary"]
    assert s["extras"]["silhouette"]["chi2"] == rep["chi2"] and s["silhouette_chi2"] == rep["chi2"]


def test_lazy_silhouette_refuses_a_different_corpus_and_missing_table(store, two_runs):
    cls, rows = two_runs
    _install_uces(store, rows, texto_suffix="!")  # otra ejecución reescribió uces
    rep = service.ensure_silhouette(store, "run-a")
    assert rep["status"] == "unavailable" and "huella" in rep["reason"]
    assert "silhouette" not in store.get_run("run-a")["summary"]["extras"]
    assert service.ensure_silhouette(store, "run-a", allow_unverified=True)["status"] == "ok"
    store.conn.execute("DROP TABLE uces")
    store2 = service.ensure_silhouette(store, "run-b")
    assert store2["status"] == "unavailable" and "uces" in store2["reason"]


def test_compare_runs_permuted_ids_are_identical_partitions(store, two_runs):
    c = service.compare_runs(store, "run-a", "run-b", "estricto")
    assert c["n_common"] == 120 and c["same_corpus"] is True
    assert c["fmi"] == c["nmi"] == c["ari"] == pytest.approx(1.0)
    assert c["confusion"].shape == (3, 3) and (c["confusion"].to_numpy() > 0).sum() == 3
    assert c["class_match"] == {0: 2, 1: 0, 2: 1} and set(c["per_class_jaccard"].values()) == {1.0}
    assert c["silhouette_a"] is None  # no se calcula sin pedirlo
    c2 = service.compare_runs(store, "run-a", "run-b", "estricto", compute_silhouette=True)
    assert c2["silhouette_a"] == pytest.approx(c2["silhouette_b"]) and c2["silhouette_a"] > 0.2
    with pytest.raises(ValueError):
        service.compare_runs(store, "run-a", "run-b", "no-existe")
    with pytest.raises(KeyError):
        service.compare_runs(store, "run-a", "run-zzz")


def test_compare_runs_uses_each_runs_tier_not_all_ucees(store, two_runs):
    cls, rows = two_runs
    store.save_run({"run_id": "run-c", "label": "C", "kind": "pipeline", "n_uces": 120, "n_uces_total": 120,
                    "corpus_hash": "otro"})
    df = store.load_assignments("run-a").copy()
    df["s_acuerdo"] = 0.0  # ninguna UCE estable en C
    store.save_assignments("run-c", df)
    c = service.compare_runs(store, "run-a", "run-c")
    assert c["n_common"] == 0 and c["same_corpus"] is False and c["confusion"].empty


# ── hook: Silhouette al guardar ─────────────────────────────────────────────
class _U:
    def __init__(self, i, cid, stems):
        self.id, self.doc_id, self.is_stable, self.cluster_id = f"u{i}", 1, True, cid
        self.lemmas, self.stems = stems, stems
        self.bigram_stems = self.trigram_stems = self.bigrams = self.trigrams = []
        self.texto = f"texto {i}"
        self.projected_cluster_id = self.projection_margin = None


def _hook_wf(vocab, ppmi_builder=None):
    cfg = types.SimpleNamespace(classification_mode="wc_only", random_state=5, stem_backend="snowball",
                                use_bigrams=False, use_trigrams=False, tsj=3)
    return types.SimpleNamespace(config=cfg, _vocab=vocab, ppmi_builder=ppmi_builder, db=types.SimpleNamespace(data={}),
                                 double_clf=types.SimpleNamespace(last_pairwise_stability={}))


def test_hook_computes_silhouette_in_all_spaces(store, monkeypatch):
    monkeypatch.setattr(hooks, "redact_config", lambda c: {"mode": "x"})
    cls, rows = _corpus(150)
    uces = [_U(i, int(c), r[2]["stems"]) for i, (c, r) in enumerate(zip(cls, rows))]
    vocab = sorted({t for u in uces for t in u.stems})
    ppmi = types.SimpleNamespace(build=lambda X, v: X.astype(float).multiply(2.0).tocsr())  # PPMI de juguete
    rid = hooks.save_quality_run(_hook_wf(vocab, ppmi), uces, [uces], store=store)
    sil = store.get_run(rid)["summary"]["extras"]["silhouette"]
    assert sil["status"] == "ok" and sil["origin"] == "pipeline" and sil["vocab_source"] == "cdh"
    assert sil["chi2"] > 0.2 and sil["binary_cosine"] > 0.2 and sil["ppmi_cosine"] == pytest.approx(sil["binary_cosine"])
    assert store.get_run(rid)["summary"]["silhouette_chi2"] == sil["chi2"]
    # sin vocabulario ni PPMI: se deriva con tsj y no hay coseno PPMI
    wf2 = _hook_wf(None)
    wf2.config.random_state = 6  # otro run_id (misma config y corpus reutilizarían la corrida existente)
    rid2 = hooks.save_quality_run(wf2, uces, [uces], store=store)
    sil2 = store.get_run(rid2)["summary"]["extras"]["silhouette"]
    assert sil2["vocab_source"] == "derived_tsj" and sil2["ppmi_cosine"] is None


def test_hook_survives_silhouette_failure(store, monkeypatch):
    monkeypatch.setattr(hooks, "redact_config", lambda c: {})
    monkeypatch.setattr(service, "silhouette_report", lambda *a, **k: 1 / 0)
    uces = [_U(i, i % 2, ["a", "b"]) for i in range(12)]
    rid = hooks.save_quality_run(_hook_wf(["a", "b"]), uces, [uces], store=store)
    assert rid and "silhouette" not in store.get_run(rid)["summary"]["extras"]


# ── CLI de tablas ───────────────────────────────────────────────────────────
def test_compare_modes_tables_and_cli(tmp_path, capsys):
    dsn = f"sqlite:///{tmp_path / 'q.db'}"
    s = QualityStore.open(dsn)
    cls, rows = _corpus()
    _install_uces(s, rows)
    _save_run(s, "run-a", cls, rows, "A", "wc_only")
    _save_run(s, "run-b", cls, rows, "B", "all", perm=(1, 2, 0))
    df = service.evaluate_run(s, "run-a", persist=False)["df"]
    s.save_gold("run-a", _gold_for(df.head(40), "c1"))
    s.close()
    out = tmp_path / "t.csv"
    assert compare_modes.main(["--dsn", dsn, "--runs", "run-a,run-b", "--out", str(out), "--compute-silhouette"]) == 0
    per_run, pairs = pd.read_csv(out), pd.read_csv(tmp_path / "t_pairs.csv")
    assert list(per_run["run_id"]) == ["run-a", "run-b"] and per_run["silhouette_chi2"].notna().all()
    assert per_run.loc[0, "c_ext_accuracy"] == pytest.approx(0.75) and pd.isna(per_run.loc[1, "c_ext_accuracy"])
    assert len(pairs) == 1 and pairs.loc[0, "ari"] == pytest.approx(1.0) and pairs.loc[0, "n_common"] == 120
    with pytest.raises(SystemExit):
        compare_modes.main(["--dsn", dsn, "--runs", "run-a,nope", "--out", str(out)])


# ── panel ───────────────────────────────────────────────────────────────────
def test_panel_shows_new_sections(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    dsn = f"sqlite:///{tmp_path / 'q.db'}"
    s = QualityStore.open(dsn)
    cls, rows = _corpus()
    _install_uces(s, rows)
    _save_run(s, "run-a", cls, rows, "A", "wc_only")
    _save_run(s, "run-b", cls, rows, "B", "all", perm=(2, 0, 1))
    df = service.evaluate_run(s, "run-a", persist=False)["df"]
    s.save_gold("run-a", _gold_for(df.head(40), "c1") + _gold_for(df.head(40), "c2", n_wrong=8))
    s.close()
    monkeypatch.setenv("Q_DSN", dsn)

    def app():
        import os

        from reii.quality.dashboard_panel import render_quality_panel
        from reii.quality.store import QualityStore

        render_quality_panel(QualityStore.open(os.environ["Q_DSN"]))

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception, at.exception
    at.selectbox(key="q_run").set_value("run-a").run()
    assert not at.exception, at.exception
    labels = {m.label for m in at.metric}
    assert {"Exactitud (clase)", "Kappa (clase)", "Kappa · correcta/incorrecta", "Kappa · clase", "FMI", "NMI", "ARI"} <= labels
    assert next(m for m in at.metric if m.label == "FMI").value == "1.000"
    at.button(key="q_silbtn").click().run()
    assert not at.exception, at.exception
    assert any(m.label.startswith("χ²") for m in at.metric)
