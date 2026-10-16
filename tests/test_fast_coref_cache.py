"""Fast tests for the pure coref cache module (no heavy imports).

Tests the serialization/validation/atomic-write functions in
`reii.coref_cache` without importing `main_workflow` (which takes ~281s
due to torch+stanza+spacy+DB).
"""

import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from reii.coref_cache import (
    SCHEMA_VERSION,
    atomic_write_json,
    build_cache_document,
    deserialize_chains,
    file_lock,
    load_cache,
    missing_ctx_values,
    serialize_chains,
    validate_cache_document,
)


def _sample_chains():
    return {1: {"d1": [{"mentions": [{"text": "x", "start_char": 0, "end_char": 1}]}]}}


# ── serialize_chains / deserialize_chains ────────────────────────────────────


def test_serialize_deserialize_round_trip():
    chains = _sample_chains()
    raw = serialize_chains(chains)
    # claves ctx convertidas a str
    assert set(raw.keys()) == {"1"}
    back = deserialize_chains(raw)
    assert back == chains
    # claves ctx vuelven a int
    assert all(isinstance(k, int) for k in back)
    assert 1 in back


def test_serialize_does_not_mutate_input():
    chains = _sample_chains()
    serialize_chains(chains)
    assert 1 in chains  # claves originales siguen siendo int


# ── deserialize_chains rechaza estructura inválida ───────────────────────────


@pytest.mark.parametrize("bad", [[], "not-a-dict", 42, None])
def test_deserialize_rejects_non_dict(bad):
    with pytest.raises(ValueError):
        deserialize_chains(bad)


def test_deserialize_rejects_non_int_ctx():
    with pytest.raises(ValueError):
        deserialize_chains({"abc": {}})


def test_deserialize_rejects_non_dict_by_doc():
    with pytest.raises(ValueError):
        deserialize_chains({"1": []})


# ── validate_cache_document ──────────────────────────────────────────────────


def test_validate_valid_document():
    doc = build_cache_document("key", "fp", _sample_chains())
    assert validate_cache_document(doc, "key") is True


def test_validate_wrong_cache_key():
    doc = build_cache_document("key", "fp", _sample_chains())
    assert validate_cache_document(doc, "other") is False


def test_validate_unknown_schema_version():
    doc = build_cache_document("key", "fp", _sample_chains())
    doc["schema_version"] = SCHEMA_VERSION + 999
    assert validate_cache_document(doc, "key") is False


def test_validate_corrupt_chains_returns_false():
    doc = build_cache_document("key", "fp", _sample_chains())
    doc["chains"] = {"abc": {}}
    assert validate_cache_document(doc, "key") is False


def test_validate_non_dict_returns_false():
    assert validate_cache_document(None, "key") is False
    assert validate_cache_document([], "key") is False
    assert validate_cache_document("x", "key") is False


# ── atomic_write_json ────────────────────────────────────────────────────────


def test_atomic_write_and_read(tmp_path):
    path = tmp_path / "cache.json"
    obj = {"a": 1, "b": [1, 2, 3]}
    atomic_write_json(str(path), obj)
    with open(path, "r", encoding="utf-8") as f:
        assert json.load(f) == obj


def test_atomic_write_creates_parent_dir(tmp_path):
    path = tmp_path / "sub" / "nested" / "x.json"
    atomic_write_json(str(path), {"k": "v"})
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        assert json.load(f) == {"k": "v"}


def test_atomic_write_no_residual_tmp(tmp_path):
    path = tmp_path / "cache.json"
    atomic_write_json(str(path), {"k": "v"})
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


# ── load_cache ───────────────────────────────────────────────────────────────


def test_load_cache_missing_file(tmp_path):
    assert load_cache(str(tmp_path / "nope.json"), "key") is None


def test_load_cache_corrupt_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_cache(str(path), "key") is None


def test_load_cache_key_mismatch(tmp_path):
    path = tmp_path / "cache.json"
    doc = build_cache_document("key", "fp", _sample_chains())
    atomic_write_json(str(path), doc)
    assert load_cache(str(path), "other") is None


def test_load_cache_valid_round_trip(tmp_path):
    path = tmp_path / "cache.json"
    chains = _sample_chains()
    doc = build_cache_document("key", "fp", chains)
    atomic_write_json(str(path), doc)
    result = load_cache(str(path), "key")
    assert result == chains
    assert all(isinstance(k, int) for k in result)


# ── missing_ctx_values ───────────────────────────────────────────────────────


def test_missing_ctx_values_basic():
    assert missing_ctx_values([1, 2], {1: {}}) == [2]


def test_missing_ctx_values_none_missing():
    assert missing_ctx_values([1, 2], {1: {}, 2: {}}) == []


def test_missing_ctx_values_empty_requested():
    assert missing_ctx_values([], {}) == []


def test_missing_ctx_values_dedup_and_sorted():
    assert missing_ctx_values([1, 1, 2], {}) == [1, 2]


# ── file_lock ────────────────────────────────────────────────────────────────


def test_file_lock_acquire_release(tmp_path):
    lock_path = str(tmp_path / "lock.test")
    with file_lock(lock_path, timeout=1.0):
        assert os.path.exists(lock_path)
    # Tras liberar, se puede re-adquirir
    with file_lock(lock_path, timeout=1.0):
        pass


def test_file_lock_conflict_and_timeout(tmp_path):
    """Un segundo proceso no puede adquirir el lock mientras otro lo tiene."""
    lock_path = str(tmp_path / "lock.test")
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    script = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {src_dir!r})
        from reii.coref_cache import file_lock
        with file_lock({lock_path!r}, timeout=10.0):
            print("LOCKED", flush=True)
            time.sleep(3)
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    result: dict = {}

    def _read():
        result["line"] = proc.stdout.readline().strip()

    reader = threading.Thread(target=_read)
    reader.start()
    try:
        reader.join(10)
        assert "line" in result, "el hijo no imprimió LOCKED a tiempo"
        assert result["line"] == "LOCKED", f"salida inesperada: {result['line']!r}"
        with pytest.raises(TimeoutError):
            with file_lock(lock_path, timeout=0.5):
                pass
    finally:
        proc.wait(timeout=15)
        reader.join(5)
    # Tras liberar (el hijo salió), se puede adquirir
    with file_lock(lock_path, timeout=1.0):
        pass
