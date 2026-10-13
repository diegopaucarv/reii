"""Pruebas de la capa de datos de reii.pipelines (sin Streamlit)."""

import json
import os
import time

import pytest

from reii.pipelines import config_sync, runtime, status
from reii.pipelines import discourse_db as ddb
from reii.quality.store import _SqliteConn

SCHEMA = """
CREATE TABLE documents(doc_id TEXT PRIMARY KEY, metadata_json TEXT);
CREATE TABLE uces(uce_id TEXT PRIMARY KEY, doc_id TEXT, local_idx INTEGER, section_id INTEGER DEFAULT 0,
                  is_stable INTEGER DEFAULT 0, created_at TEXT, cluster_id INTEGER);
CREATE TABLE clusters(cluster_id INTEGER);
CREATE TABLE annotations(annotation_id INTEGER PRIMARY KEY AUTOINCREMENT, uce_id TEXT NOT NULL, agent_name TEXT NOT NULL,
                         trait TEXT, quote TEXT, confidence TEXT, subtype TEXT, payload_json TEXT NOT NULL DEFAULT '{}');
CREATE TABLE kv_store(key TEXT PRIMARY KEY, value_json TEXT, updated_at TEXT);
"""


@pytest.fixture
def conn():
    c = _SqliteConn(":memory:")
    c.raw.executescript(SCHEMA)
    for i, (doc, sec, loc) in enumerate(
        [("1", 0, 0), ("1", 0, 1), ("2", 0, 0), ("2", 1, 0)]
    ):
        c.raw.execute(
            "INSERT INTO uces VALUES (?,?,?,?,1,'2026-10-10 10:00:00',?)",
            (f"{doc}_{sec}_{loc}", doc, loc, sec, i % 2),
        )
    c.raw.execute("INSERT INTO clusters VALUES (0)")
    return c


def ann(agent="A", trait="metáfora", quote="agua", conf="alta"):
    return {
        "agent": agent,
        "trait": trait,
        "confidence": conf,
        "spans": [{"quote": quote, "start_char": -1, "end_char": -1}],
    }


# ── resolución de ids ───────────────────────────────────────────────────────
def test_resolver_accepts_all_id_schemas(conn):
    r = ddb.build_id_resolver(ddb._fetch_uce_rows(conn))
    assert r("1_0_1") == "1_0_1"
    assert r("1_0_1__mf3") == "1_0_1"
    assert r("1_1") == "1_0_1"  # {doc}_{idx} sin ambigüedad
    assert r("9_9_9") is None


def test_resolver_ignores_ambiguous_short_ids(conn):
    r = ddb.build_id_resolver(ddb._fetch_uce_rows(conn))
    assert r("2_0") is None  # doc 2, idx 0 existe en dos secciones


# ── sincronización ──────────────────────────────────────────────────────────
def test_sync_is_idempotent_and_reports_unmatched(conn):
    state = {
        "annotations_by_uce": {
            "1_0_0__mf1": [ann()],
            "1_0_1": [ann(trait="anclaje")],
            "ghost": [ann()],
        }
    }
    rep = ddb.sync_state_to_db(conn, state)
    assert (rep["n_written"], rep["n_unmatched"], rep["unmatched_examples"]) == (
        2,
        1,
        ["ghost"],
    )
    ddb.sync_state_to_db(conn, state)
    assert conn.execute("SELECT COUNT(*) AS n FROM annotations").fetchone()["n"] == 2


def test_replace_agents_keeps_other_agents_and_replace_all_clears(conn):
    ddb.sync_state_to_db(
        conn, {"annotations_by_uce": {"1_0_0": [ann("A"), ann("B", trait="otra")]}}
    )
    ddb.sync_state_to_db(
        conn,
        {"annotations_by_uce": {"1_0_1": [ann("A", trait="nueva")]}},
        replace="agents",
    )
    agents = {
        r["agent_name"]: r["n"]
        for r in conn.execute(
            "SELECT agent_name, COUNT(*) AS n FROM annotations GROUP BY agent_name"
        ).fetchall()
    }
    assert agents == {"A": 1, "B": 1}  # A reemplazado, B intacto
    ddb.sync_state_to_db(
        conn, {"annotations_by_uce": {"1_0_0": [ann("C")]}}, replace="all"
    )
    assert [
        r["agent_name"]
        for r in conn.execute("SELECT agent_name FROM annotations").fetchall()
    ] == ["C"]


def test_load_returns_state_shape_with_spans_and_source_id(conn):
    ddb.sync_state_to_db(
        conn, {"annotations_by_uce": {"1_0_0__mf2": [ann(quote="agua")]}}
    )
    back = ddb.load_annotations_by_uce(conn)
    a = back["1_0_0"][0]
    assert (
        a["trait"] == "metáfora"
        and a["spans"][0]["quote"] == "agua"
        and a["source_uce_id"] == "1_0_0__mf2"
    )


def test_load_falls_back_to_columns_when_payload_is_empty(conn):
    conn.raw.execute(
        "INSERT INTO annotations (uce_id, agent_name, trait, quote, confidence) VALUES ('1_0_0','A','x','hola','media')"
    )
    a = ddb.load_annotations_by_uce(conn)["1_0_0"][0]
    assert a["confidence"] == "media" and a["spans"][0]["quote"] == "hola"


class FakeAdapter:
    def uces(self):
        return [
            {
                "uce_id": "u1",
                "id": "u1",
                "doc_id": 1,
                "texto": "t",
                "metadata": {"section_id": 0},
            }
        ]

    def _doc_metadata(self):
        return {"1": {"grupo": "A"}}

    def annotations(self):
        return {"u1": [ann()]}


def test_export_uses_db_ids_and_merges_doc_metadata(tmp_path):
    path = tmp_path / "in.json"
    rep = ddb.export_discourse_input(FakeAdapter(), str(path))
    d = json.load(open(path, encoding="utf-8"))
    assert rep["n_uces"] == 1 and d["uces"][0]["uce_id"] == "u1"
    assert d["uces"][0]["metadata"] == {"grupo": "A", "section_id": 0}
    assert list(d["annotations_by_uce"]) == ["u1"]


def test_build_discourse_input_returns_in_memory_payload():
    # build_discourse_input es la variante sin disco que usa el runner para
    # leer la entrada directo de PostgreSQL (sin discourse_input.json intermedio).
    payload = ddb.build_discourse_input(FakeAdapter())
    assert payload["uces"][0]["uce_id"] == "u1"
    assert payload["uces"][0]["metadata"] == {"grupo": "A", "section_id": 0}
    assert list(payload["annotations_by_uce"]) == ["u1"]


def test_runtime_prepare_discourse_input_always_generates_from_db(tmp_path):
    # El fallback source="json" (passthrough de workflow_data.json) se eliminó:
    # prepare_discourse_input ahora SIEMPRE genera la entrada desde la BD.
    wf = str(tmp_path / "workflow_data.json")
    p = runtime.prepare_discourse_input("json", wf, adapter=FakeAdapter())
    assert p.endswith(runtime.INPUT_NAME) and os.path.exists(p)
    p2 = runtime.prepare_discourse_input("db", wf, adapter=FakeAdapter())
    assert p2.endswith(runtime.INPUT_NAME) and os.path.exists(p2)


# ── configuración unificada ─────────────────────────────────────────────────
class FakeStore:
    def __init__(self):
        self.saved = {}

    def set_config_many(self, d, by):
        self.saved.update(d)
        return len(d)


def test_push_only_valid_keys_reach_the_store():
    s = FakeStore()
    rep = config_sync.push_params_to_store(
        {"use_rf_shap": True, "workflow_mode": "transformer", "min_uce_words": 3},
        store=s,
    )
    assert s.saved == {"use_rf_shap": True, "min_uce_words": 3}
    assert rep["ignored"] == ["workflow_mode"]


def test_save_unified_writes_both_channels(tmp_path):
    s, p = FakeStore(), str(tmp_path / "wc.json")
    json.dump({"viejo": 1}, open(p, "w"))
    config_sync.save_unified({"use_rf_shap": True}, p, store=s)
    assert s.saved == {"use_rf_shap": True}
    assert json.load(open(p)) == {"viejo": 1, "use_rf_shap": True}


def test_diff_channels_only_reports_real_differences():
    d = config_sync.diff_channels({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 9, "d": 4})
    assert d == [{"key": "b", "store": 2, "json": 9}]


# ── chequeos de estado ──────────────────────────────────────────────────────
def test_status_flags_stale_input_and_unsynced_annotations(conn, tmp_path):
    wf = tmp_path / "workflow_data.json"
    wf.write_text("{}")
    old = time.time() - 7 * 86400
    os.utime(wf, (old, old))
    st_path = tmp_path / "state.json"
    st_path.write_text(
        json.dumps({"annotations_by_uce": {"1_0_0": [ann()], "ghost": [ann()]}})
    )
    s = status.collect_status(
        conn, {"workflow_db": str(wf), "discourse_state": str(st_path)}
    )
    by_id = {c["id"]: c for c in s["checks"]}
    assert (
        by_id["input_freshness"]["level"] == "warn"
        and by_id["input_freshness"]["fix"] == "export_input"
    )
    assert by_id["annotations_ids"]["level"] == "warn"
    assert by_id["annotations_sync"]["fix"] == "sync_annotations"
    assert s["discourse"]["state"] == "desactualizado"
    ddb.sync_state_to_db(conn, json.load(open(st_path)))
    s2 = status.collect_status(
        conn, {"workflow_db": str(wf), "discourse_state": str(st_path)}
    )
    assert {c["id"]: c["level"] for c in s2["checks"]}["annotations_sync"] == "ok"


def test_status_config_and_rf_checks(conn, tmp_path):
    wc = tmp_path / "wc.json"
    wc.write_text(json.dumps({"use_rf_shap": False}))
    s = status.collect_status(
        conn, {"workflow_config": str(wc)}, store_overrides={"use_rf_shap": True}
    )
    by_id = {c["id"]: c for c in s["checks"]}
    assert by_id["config_channels"]["fix"] == "unify_config"
    assert by_id["rf"]["level"] == "warn"  # activo pero sin resultados guardados


def test_status_without_classification_is_an_error():
    c = _SqliteConn(":memory:")
    c.raw.executescript(SCHEMA)
    s = status.collect_status(c, {})
    assert s["workflow"]["state"] == "sin_datos"
    assert {x["id"]: x["level"] for x in s["checks"]}["classification"] == "error"


def test_parse_ts_variants():
    assert status.parse_ts("2026-10-10 10:00:00") is not None
    assert status.parse_ts("2026-10-10 10:00:00.123456+00") is not None
    assert status.parse_ts("2026-10-10T10:00:00Z") is not None
    assert status.parse_ts("basura") is None and status.parse_ts(None) is None
