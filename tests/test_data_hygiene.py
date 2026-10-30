"""Tests de `reii.diag data-hygiene`: politica declarativa y borrado seguro.

No tocan `data/` real: todo se monta en ``tmp_path``. Se comprueban las tres
garantias (canonicos intactos, edad minima, dry-run por defecto), el orden por
mtime de ``keep_latest``, la salida JSON y que el borrado no salga de la raiz.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reii.diag import data_hygiene as dh  # noqa: E402  (tras insertar src/)


def _write(path: Path, data: bytes = b"x", mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _old(minutes: float = 60.0) -> float:
    return time.time() - minutes * 60.0


def test_canonical_never_deleted_even_if_glob_matches(tmp_path, monkeypatch):
    """`best_params.json` casa `*.json`, pero es canonico: se queda."""
    monkeypatch.setattr(
        dh,
        "POLICY",
        [dh.Rule("diag/tmp_*", 0, "tmp"), dh.Rule("*.json", 0, "test")],
    )
    residue = _write(tmp_path / "diag" / "tmp_x.txt", mtime=_old())
    canonical = _write(tmp_path / "best_params.json", mtime=_old())

    entries = dh.build_inventory(tmp_path, min_age_minutes=0)
    status = {e["path"]: e["status"] for e in entries}
    assert status["best_params.json"] == "keep"
    assert status["diag/tmp_x.txt"] == "delete"

    assert (
        dh.run(["--data-dir", str(tmp_path), "--apply", "--min-age-minutes", "0"]) == 0
    )
    assert canonical.exists()
    assert not residue.exists()


def test_canonical_dirs_are_not_inventoried(tmp_path, monkeypatch):
    monkeypatch.setattr(dh, "POLICY", [dh.Rule("*.json", 0, "test")])
    _write(tmp_path / "checkpoints" / "paso.json", mtime=_old())
    _write(tmp_path / "entrevistas" / "e.json", mtime=_old())
    suelto = _write(tmp_path / "suelto.json", mtime=_old())

    entries = dh.build_inventory(tmp_path, min_age_minutes=0)
    assert {e["path"] for e in entries} == {"suelto.json"}
    assert (
        dh.run(["--data-dir", str(tmp_path), "--apply", "--min-age-minutes", "0"]) == 0
    )
    assert (tmp_path / "checkpoints" / "paso.json").exists()
    assert (tmp_path / "entrevistas" / "e.json").exists()
    assert not suelto.exists()


def test_keep_latest_respects_mtime_order(tmp_path, monkeypatch):
    monkeypatch.setattr(dh, "POLICY", [dh.Rule("diag/*.log", 2, "test")])
    now = time.time()
    for i, name in enumerate(("a.log", "b.log", "c.log", "d.log")):
        # a es el mas viejo (100 min) y d el mas nuevo (70 min)
        _write(tmp_path / "diag" / name, mtime=now - (100 - i * 10) * 60.0)

    assert (
        dh.run(["--data-dir", str(tmp_path), "--apply", "--min-age-minutes", "0"]) == 0
    )
    remaining = sorted(p.name for p in (tmp_path / "diag").iterdir())
    assert remaining == ["c.log", "d.log"]


def test_min_age_minutes_protects_fresh_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dh, "POLICY", [dh.Rule("diag/tmp_*", 0, "test")])
    fresh = _write(tmp_path / "diag" / "tmp_fresh.txt")
    stale = _write(tmp_path / "diag" / "tmp_stale.txt", mtime=_old())

    entries = dh.build_inventory(tmp_path, min_age_minutes=10)
    status = {e["path"]: e["status"] for e in entries}
    assert status["diag/tmp_fresh.txt"] == "skip_recent"
    assert status["diag/tmp_stale.txt"] == "delete"

    assert dh.run(["--data-dir", str(tmp_path), "--apply"]) == 0
    assert fresh.exists()
    assert not stale.exists()


def test_dry_run_by_default_then_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(dh, "POLICY", [dh.Rule("diag/tmp_*", 0, "test")])
    victim = _write(tmp_path / "diag" / "tmp_z.txt", mtime=_old())

    assert dh.run(["--data-dir", str(tmp_path)]) == 0
    assert victim.exists(), "el modo por defecto no debe borrar nada"

    assert dh.run(["--data-dir", str(tmp_path), "--dry-run"]) == 0
    assert victim.exists()

    assert (
        dh.run(["--data-dir", str(tmp_path), "--apply", "--min-age-minutes", "0"]) == 0
    )
    assert not victim.exists()


def test_backup_dir_moves_instead_of_deleting(tmp_path, monkeypatch):
    monkeypatch.setattr(dh, "POLICY", [dh.Rule("diag/tmp_*", 0, "test")])
    victim = _write(tmp_path / "data" / "diag" / "tmp_b.txt", data=b"abc", mtime=_old())
    backup = tmp_path / "backup"

    rc = dh.run(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--apply",
            "--backup-dir",
            str(backup),
            "--min-age-minutes",
            "0",
        ]
    )
    assert rc == 0
    assert not victim.exists()
    moved = backup / "diag" / "tmp_b.txt"
    assert moved.read_bytes() == b"abc"


def test_json_output_is_parseable(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(dh, "POLICY", [dh.Rule("diag/*.log", 1, "test")])
    _write(tmp_path / "diag" / "viejo.log", mtime=_old(120))
    _write(tmp_path / "diag" / "nuevo.log", mtime=_old(30))

    rc = dh.run(
        ["--data-dir", str(tmp_path), "--json", "--dry-run", "--min-age-minutes", "0"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data_dir"] == str(tmp_path.resolve())
    assert payload["apply"] is False
    assert payload["totals"]["files"] == 2
    assert payload["totals"]["candidates"] == 1
    assert payload["totals"]["deleted"] == 0
    assert {f["path"] for f in payload["files"]} == {
        "diag/viejo.log",
        "diag/nuevo.log",
    }
    assert any(f["status"] == "delete" for f in payload["files"])
    assert any(rule["pattern"] == "diag/*.log" for rule in payload["policy"])


def test_json_with_apply_stays_parseable(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(dh, "POLICY", [dh.Rule("diag/tmp_*", 0, "test")])
    _write(tmp_path / "diag" / "tmp_a.txt", mtime=_old())

    rc = dh.run(
        [
            "--data-dir",
            str(tmp_path),
            "--json",
            "--apply",
            "--min-age-minutes",
            "0",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["apply"] is True
    assert payload["totals"]["deleted"] == 1
    assert payload["totals"]["errors"] == 0
    assert not (tmp_path / "diag" / "tmp_a.txt").exists()


def test_deletion_cannot_escape_data_dir(tmp_path):
    data = tmp_path / "data"
    outside = _write(tmp_path / "outside.txt", mtime=_old())
    entries = dh.build_inventory(data, min_age_minutes=0)
    entries.append({"path": "../outside.txt", "bytes": 1, "status": "delete"})

    assert not dh._within(tmp_path / "outside.txt", data)
    assert dh._within(data / "diag" / "x.log", data)

    stats = dh.apply_actions(entries, data)
    assert stats["errors"] == 1
    assert stats["deleted"] == 0
    assert outside.exists()


def test_policy_declares_the_documented_groups():
    patterns = [rule.pattern for rule in dh.POLICY]
    assert "diag/*.log" in patterns
    assert "diag/tmp_*" in patterns
    keep = {rule.pattern: rule.keep_latest for rule in dh.POLICY}
    assert keep["diag/*.log"] == 8
    assert keep["diag/tmp_*"] == 0
    assert keep["diag/best_params_seed*.json"] == 5
    assert keep["diag/*.json"] == 12
    assert keep["diag/*.png"] == 10
    assert keep["diag/*.html"] == 10
    assert keep["*.pre_v4"] == 2
    assert keep["*.ctx*.json"] == 2
    assert keep["*.round*.json"] == 2
    assert keep["best_params.v3_censurado.json"] == 1
    assert keep["comparative_results.json"] == 1

    for name in (
        "workflow_data.json",
        "workflow_data.db",
        "best_params.json",
        "method_selection.json",
        "method_selection_log.jsonl",
        "reii_optuna.db",
        "uce_calibration.json",
        "token_table.jsonl",
        "entrevistas.json",
        "SUBTLEX-ESP.xlsx",
        "query_db.py",
    ):
        assert name in dh.CANONICAL_FILES
    assert set(dh.CANONICAL_DIRS) == {"entrevistas", "txt_outputs", "checkpoints"}
