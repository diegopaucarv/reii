"""Endurecimiento del checkpoint/caché de coref (sin dependencias pesadas).

Cubre: (1) huella del contenido de entrada, (2) guardado que informa de su fallo,
(3) CheckpointGuard (doc incompleto congela el checkpoint), (4) la señal
``last_extraction_incomplete`` del segmentador y (5) que el precompute no cachee
fallos como "sin cadenas".
"""
import ast
import json
import logging
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reii.seg import coref_checkpoint as cc  # noqa: E402
from reii.seg.coref_checkpoint import (  # noqa: E402
    COREF_CHECKPOINT_VERSION,
    CheckpointGuard,
    _cargar_checkpoint_coref,
    _guardar_checkpoint_coref,
    uces_fingerprint,
)

SEG = ROOT / "src" / "reii" / "seg" / "segmentador.py"
MW = ROOT / "src" / "reii" / "main_workflow.py"


def U(doc, i, texto):
    return SimpleNamespace(doc_id=doc, id=f"d{doc}u{i}", texto=texto)


@pytest.fixture(autouse=True)
def _fixed_code_fp(monkeypatch):
    monkeypatch.setattr(cc, "_coref_code_fingerprint", lambda: "codefp")


def _state(fp, **kw):
    s = {"version": COREF_CHECKPOINT_VERSION, "context_units": 2, "min_forms": 10,
         "code_fingerprint": "codefp", "input_fingerprint": fp, "processed_doc_ids": [0],
         "ucs": [], "uce_to_uc": {}, "global_uc_counter": 0, "total_boundaries": 0,
         "coref_merges": 0, "covered_boundaries": 0, "total_chains": 0, "root_cache": {}}
    s.update(kw)
    return s


# ───────────────────────────── huella de entrada ───────────────────────────────
def test_fingerprint_is_deterministic_and_sensitive():
    base = [[U(0, 0, "hola mundo"), U(0, 1, "adiós")], [U(1, 0, "otro doc")]]
    fp = uces_fingerprint(base)
    assert fp == uces_fingerprint([[U(0, 0, "hola mundo"), U(0, 1, "adiós")], [U(1, 0, "otro doc")]])
    assert fp != uces_fingerprint([[U(0, 0, "hola mundo!"), U(0, 1, "adiós")], [U(1, 0, "otro doc")]])  # texto
    assert fp != uces_fingerprint([[U(0, 0, "hola mundo")], [U(0, 1, "adiós"), U(1, 0, "otro doc")]])    # frontera de doc
    assert fp != uces_fingerprint(base, extra={"similarity_threshold": 0.6})
    assert uces_fingerprint(base, extra={"a": 1, "b": 2}) == uces_fingerprint(base, extra={"b": 2, "a": 1})


def test_fingerprint_distinguishes_uce_segmentation():
    """Mismo texto total, distinta partición en UCEs (otro uce_target_size)."""
    a = [[U(0, 0, "uno dos tres cuatro")]]
    b = [[U(0, 0, "uno dos"), U(0, 1, "tres cuatro")]]
    assert uces_fingerprint(a) != uces_fingerprint(b)


# ───────────────────────────── carga/guardado ──────────────────────────────────
def test_stale_input_is_rejected_with_reason(tmp_path):
    p = str(tmp_path / "ck.ctx2.json")
    assert _guardar_checkpoint_coref(p, _state("AAAA")) is True
    reasons: List[str] = []
    assert _cargar_checkpoint_coref(p, 2, 10, input_fingerprint="BBBB", reason_sink=reasons) is None
    assert reasons and "OTRA entrada" in reasons[0]
    ok = _cargar_checkpoint_coref(p, 2, 10, input_fingerprint="AAAA")
    assert ok is not None and ok["processed_doc_ids"] == [0]
    # compatibilidad de API: sin huella no se valida el contenido
    assert _cargar_checkpoint_coref(p, 2, 10) is not None


def test_old_format_checkpoint_is_rejected(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps(_state("AAAA", version=2)), encoding="utf-8")
    reasons: List[str] = []
    assert _cargar_checkpoint_coref(str(p), 2, 10, input_fingerprint="AAAA", reason_sink=reasons) is None
    assert "versión" in reasons[0]


def test_save_reports_failure_instead_of_silently_swallowing(tmp_path):
    blocker = tmp_path / "es_un_archivo"
    blocker.write_text("x")
    bad = str(blocker / "sub" / "ck.json")          # el padre es un archivo → no se puede crear
    assert _guardar_checkpoint_coref(bad, _state("A")) is False
    assert _guardar_checkpoint_coref(str(tmp_path / "ok.json"), _state("A")) is True


# ───────────────────────────── CheckpointGuard ─────────────────────────────────
def test_guard_freezes_on_first_incomplete_doc():
    g = CheckpointGuard()
    assert g.can_save
    assert g.note_incomplete(7) is True
    assert not g.can_save and g.frozen_at == 7
    assert g.note_incomplete(9) is False             # ya congelado: no re-anuncia
    assert g.incomplete_docs == [7, 9] and g.frozen_at == 7
    g.write_failures, g.build_failures = 2, 1
    txt = " ".join(g.summary())
    assert "INCOMPLETA" in txt and "ESCRIBIR" in txt and "CONSTRUIR" in txt


def test_guard_clean_run_has_empty_summary():
    g = CheckpointGuard()
    assert g.can_save and g.summary() == []


# ───────────────────── señal del segmentador (AST + stubs) ─────────────────────
def _seg_func(name, extra_ns=None):
    src = SEG.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ProgressiveSegmenter")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"logging": logging, "COREF_BATCH_SIZE": 1, "STANZA_SERVER_URL": "", "gc": None,
          "STANZA_SERVER_FALLBACK_LOCAL": False}
    ns.update(extra_ns or {})
    exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns)
    return ns[name]


class _Seg:
    def __init__(self, stanza):
        self._stanza = stanza
        self.last_extraction_incomplete = "sin tocar"

    def _dprint(self, *a, **k):
        pass

    def get_stanza(self):
        return self._stanza


def test_incomplete_flag_set_when_stanza_unavailable():
    f = _seg_func("_extract_global_chains")
    s = _Seg(stanza=None)
    assert f(s, [], "") == [] and s.last_extraction_incomplete is True


def test_incomplete_flag_reset_on_a_normal_call():
    f = _seg_func("_extract_global_chains")
    s = _Seg(stanza=object())
    s.last_extraction_incomplete = True               # resto de una llamada anterior
    assert f(s, [], "") == [] and s.last_extraction_incomplete is False


def test_every_early_return_in_remote_path_flags_incomplete():
    """Los 2 `return` de abandono de _extract_coref_remote deben marcar la bandera."""
    src = SEG.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ProgressiveSegmenter")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_extract_coref_remote")
    bare_returns = 0
    for parent in ast.walk(fn):
        for field in ("body", "orelse"):
            block = getattr(parent, field, None)
            if not isinstance(block, list):
                continue
            for i, stmt in enumerate(block):
                if isinstance(stmt, ast.Return) and stmt.value is None and i > 0:
                    prev = block[i - 1]
                    seg = ast.get_source_segment(src, prev) or ""
                    if "last_extraction_incomplete = True" in seg:
                        bare_returns += 1
                    elif parent is not fn:           # un return de abandono sin marcar
                        pytest.fail(f"return sin marcar last_extraction_incomplete (línea {stmt.lineno})")
    assert bare_returns == 2


# ───────────────── el precompute no cachea fallos como "sin cadenas" ───────────
def _load_precompute():
    src = MW.read_text(encoding="utf-8")
    tree = ast.parse(src)
    ns = {"Dict": Dict, "List": List, "Iterable": Iterable, "UCE": object,
          "logger": logging.getLogger("t")}
    for name in ("_remap_uces_to_segments", "compute_coref_chains_for_ctx"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(ast.get_source_segment(src, fn), ns)
    return ns["compute_coref_chains_for_ctx"]


class _PS:
    """doc 0 → ok; doc 1 → excepción; doc 2 → incompleto (sin Stanza)."""
    def __init__(self):
        self.last_extraction_incomplete = False

    def recursive_segmentation(self, t):
        return list(t)

    def final_clustering(self, s):
        return list(s)

    def _extract_global_chains(self, segments_info, full_doc_text, context_units=2):
        self.last_extraction_incomplete = False
        if full_doc_text.startswith("DOC1"):
            raise RuntimeError("boom")
        if full_doc_text.startswith("DOC2"):
            self.last_extraction_incomplete = True
            return []
        return [{"mentions": []}]


def test_precompute_does_not_cache_failed_or_incomplete_docs(capsys):
    compute = _load_precompute()
    docs = [[U(0, 0, "DOC0 a")], [U(1, 0, "DOC1 a")], [U(2, 0, "DOC2 a")]]
    out = compute(_PS(), docs, [2, 3])
    for ctx in (2, 3):
        assert out[ctx][0] == [{"mentions": []}]       # el bueno sí
        assert 1 not in out[ctx] and 2 not in out[ctx]  # fallo/incompleto: ausentes, no []
    assert "NO se cachea" in capsys.readouterr().out
