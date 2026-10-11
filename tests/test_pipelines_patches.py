"""Pruebas del script de parches y de la reconciliación de ids en migrate_annotations."""

import contextlib
import importlib.util
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import apply_pipeline_patches as ap  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    for rel in ("src/reii/dashboard.py", "src/reii/backend/migrate.py"):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), dst)
    return tmp_path


def _run(repo, *extra):
    argv = sys.argv
    try:
        sys.argv = ["x", "--repo", str(repo), *extra]
        return ap.main()
    finally:
        sys.argv = argv


def _text(repo, rel):
    return (repo / rel).read_bytes().decode("utf-8")


def test_patches_apply_compile_are_idempotent_and_revertible(repo):
    assert _run(repo) == 0
    first = _text(repo, "src/reii/dashboard.py")
    assert _run(repo) == 0 and _text(repo, "src/reii/dashboard.py") == first
    norm = first.replace("\r\n", "\n")
    compile(norm, "dashboard.py", "exec")
    compile(_text(repo, "src/reii/backend/migrate.py").replace("\r\n", "\n"), "migrate.py", "exec")
    assert "_render_pipeline_hub()" in norm and "E · Pipelines y configuración" in norm
    assert "render_discourse_results_from_db" in norm and "source: str = \"db\"" in norm
    assert norm.index("prepare_discourse_input(source") < norm.index("sync_after_run(DISCOURSE_STATE_PATH)")
    assert ("\r\n" in first) == ("\r\n" in open(os.path.join(ROOT, "src/reii/dashboard.py"), "rb").read().decode())
    _run(repo, "--revert")
    assert "pipeline-patch" not in _text(repo, "src/reii/dashboard.py")


def test_dry_run_changes_nothing(repo):
    before = _text(repo, "src/reii/dashboard.py")
    assert _run(repo, "--dry-run") == 0
    assert _text(repo, "src/reii/dashboard.py") == before


def test_missing_anchor_aborts_without_writing(repo):
    p = repo / "src/reii/backend/migrate.py"
    p.write_bytes(p.read_bytes().replace(b"for a in flat:", b"for b in flat:"))
    before = _text(repo, "src/reii/dashboard.py")
    assert _run(repo) == 1
    assert _text(repo, "src/reii/dashboard.py") == before


class _Conn:
    def __init__(self):
        self.inserts = []

    def execute(self, sql, params=()):
        if sql.strip().startswith("INSERT INTO annotations"):
            self.inserts.append(params)

    @contextlib.contextmanager
    def transaction(self):
        yield


def _load_migrate(path):
    spec = importlib.util.spec_from_file_location("migrate_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_patched_migrate_keeps_annotations_whose_ids_use_another_schema(repo, monkeypatch):
    _run(repo)
    mod = _load_migrate(repo / "src/reii/backend/migrate.py")
    monkeypatch.setattr(mod, "transaction", lambda conn: conn.transaction())
    data = {
        "uces": [{"id": "1_0_3", "doc_id": 1, "local_idx": 3, "metadata": {"section_id": 0}}],
        "annotations_by_uce": {
            "1_0_3__mf2": [{"agent": "A", "trait": "metáfora", "quote": "agua", "confidence": "alta"}],
            "ghost": [{"agent": "A", "trait": "x"}],
        },
    }
    conn = _Conn()
    n = mod.migrate_annotations(conn, data)
    assert n == 2 and len(conn.inserts) == 1 and conn.inserts[0][0] == "1_0_3"


def test_unpatched_migrate_drops_them(repo, monkeypatch):
    mod = _load_migrate(repo / "src/reii/backend/migrate.py")
    monkeypatch.setattr(mod, "transaction", lambda conn: conn.transaction())
    data = {"uces": [{"id": "1_0_3", "doc_id": 1, "local_idx": 3}],
            "annotations_by_uce": {"1_0_3__mf2": [{"agent": "A", "trait": "t"}]}}
    conn = _Conn()
    mod.migrate_annotations(conn, data)
    assert conn.inserts == []   # comportamiento anterior: se descartaban
