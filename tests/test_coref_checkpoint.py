"""Checkpoint de coref: validación de parámetros y huella del algoritmo.

Verifica que ``_cargar_checkpoint_coref`` solo reanuda cuando el checkpoint
fue generado con los MISMOS parámetros (context_units, min_forms) y la MISMA
versión del algoritmo (code_fingerprint). Esto evita que B2 (context=4)
reutilice los resultados de B1 (context=2) y que un checkpoint viejo (de antes
del batching) se cargue tras un cambio de código.

Las funciones viven en ``reii.seg.coref_checkpoint`` (módulo ligero, sin
torch/spacy/stanza), así que este test corre rápido.
"""

import json

from reii.seg.coref_checkpoint import (
    COREF_CHECKPOINT_VERSION,
    _cargar_checkpoint_coref,
    _coref_code_fingerprint,
    _guardar_checkpoint_coref,
    adaptive_coref_batch_size,
    checkpoint_path_for_pass,
)


def _valid_state(**overrides):
    state = {
        "version": COREF_CHECKPOINT_VERSION,
        "context_units": 2,
        "min_forms": 13,
        "code_fingerprint": _coref_code_fingerprint(),
        "processed_doc_ids": ["doc1"],
        "ucs": [],
        "uce_to_uc": {},
        "global_uc_counter": 0,
        "total_boundaries": 0,
        "coref_merges": 0,
        "covered_boundaries": 0,
        "total_chains": 0,
        "root_cache": {},
    }
    state.update(overrides)
    return state


def _write(tmp_path, state):
    p = tmp_path / "ckpt.json"
    p.write_text(json.dumps(state), encoding="utf-8")
    return str(p)


# ── Carga válida ─────────────────────────────────────────────────────────────
def test_loads_valid_checkpoint(tmp_path):
    p = _write(tmp_path, _valid_state())
    data = _cargar_checkpoint_coref(p, context_units=2, min_forms=13)
    assert data is not None
    assert data["processed_doc_ids"] == ["doc1"]


# ── Versión vieja (p. ej. de antes del batching) se descarta ─────────────────
def test_rejects_old_version(tmp_path):
    p = _write(tmp_path, _valid_state(version=1))
    assert _cargar_checkpoint_coref(p, context_units=2, min_forms=13) is None


# ── B1 vs B2: context_units y min_forms distintos no se comparten ────────────
def test_rejects_wrong_context_units(tmp_path):
    p = _write(tmp_path, _valid_state(context_units=2))
    assert _cargar_checkpoint_coref(p, context_units=4, min_forms=13) is None


def test_rejects_wrong_min_forms(tmp_path):
    p = _write(tmp_path, _valid_state(min_forms=13))
    assert _cargar_checkpoint_coref(p, context_units=2, min_forms=19) is None


# ── Cambio de algoritmo (huella distinta) se descarta ────────────────────────
def test_rejects_wrong_fingerprint(tmp_path):
    p = _write(tmp_path, _valid_state(code_fingerprint="deadbeef"))
    assert _cargar_checkpoint_coref(p, context_units=2, min_forms=13) is None


# ── Casos límite ─────────────────────────────────────────────────────────────
def test_missing_file_returns_none(tmp_path):
    assert _cargar_checkpoint_coref(str(tmp_path / "nope.json"), 2, 13) is None


def test_corrupt_file_returns_none(tmp_path):
    p = tmp_path / "ckpt.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert _cargar_checkpoint_coref(str(p), 2, 13) is None


# ── Guardado: escribe un JSON válido con los campos nuevos ───────────────────
def test_guardar_escribe_campos_nuevos(tmp_path):
    p = str(tmp_path / "ckpt.json")
    _guardar_checkpoint_coref(p, _valid_state())
    data = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
    assert data["version"] == COREF_CHECKPOINT_VERSION
    assert data["context_units"] == 2
    assert data["min_forms"] == 13
    assert data["code_fingerprint"] == _coref_code_fingerprint()


# ── Ruta por pase: B1 y B2 no comparten archivo ──────────────────────────────
def test_checkpoint_path_for_pass_deriva_sufijo():
    assert (
        checkpoint_path_for_pass("/app/data/checkpoints/coref_all.json", 2)
        == "/app/data/checkpoints/coref_all.ctx2.json"
    )
    assert (
        checkpoint_path_for_pass("/app/data/checkpoints/coref_all.json", 4)
        == "/app/data/checkpoints/coref_all.ctx4.json"
    )


def test_checkpoint_path_for_pass_b1_b2_distintos():
    b1 = checkpoint_path_for_pass("coref_all.json", 2)
    b2 = checkpoint_path_for_pass("coref_all.json", 4)
    assert b1 != b2


def test_checkpoint_path_for_pass_none():
    assert checkpoint_path_for_pass(None, 2) is None
    assert checkpoint_path_for_pass("", 2) is None


# ── Lote adaptativo de coref (evita OOM en documentos grandes) ───────────────
def test_adaptive_batch_size_small_doc_unchanged():
    # Documento pequeño: el lote no se toca.
    assert adaptive_coref_batch_size(16, 50_000) == 16


def test_adaptive_batch_size_medium_doc_caps_at_8():
    assert adaptive_coref_batch_size(16, 150_000) == 8


def test_adaptive_batch_size_large_doc_caps_at_4():
    assert adaptive_coref_batch_size(16, 250_000) == 4


def test_adaptive_batch_size_never_raises_below_base():
    # Si el lote base ya es pequeño, no se sube.
    assert adaptive_coref_batch_size(2, 250_000) == 2
    assert adaptive_coref_batch_size(4, 150_000) == 4
