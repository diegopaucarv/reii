"""Pruebas de la gestión de transcripciones y de los editores de entrevistas (AppTest)."""

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def transcripciones_env(tmp_path, monkeypatch):
    """Apunta el gestor de transcripciones a un directorio temporal con un archivo."""
    import reii.pipelines.transcripciones as tr

    tr.TRANSCRIPTS_DIR = str(tmp_path / "transcripts")
    tr.BATCH_DB_PATH = str(tmp_path / "batch.json")
    os.makedirs(tr.TRANSCRIPTS_DIR, exist_ok=True)
    (Path(tr.TRANSCRIPTS_DIR) / "entrevista1.txt").write_text(
        "Hola mundo", encoding="utf-8"
    )
    return tr


def _transcripciones_app():
    from reii.pipelines.transcripciones import render_transcripciones

    render_transcripciones()


def test_transcripciones_renders_uploader_status_and_queue(transcripciones_env):
    at = AppTest.from_function(_transcripciones_app, default_timeout=30).run()
    assert not at.exception, at.exception
    assert at.file_uploader(key="reii_batch_uploader")
    assert at.button(key="reii_batch_start")
    assert len(at.dataframe) > 0  # tabla de estado de los documentos
    assert at.button(key="reii_batch_export")


def _guia_app():
    import streamlit as st

    from reii.pipelines.transcripciones import render_entrevistas_prompts

    st.session_state.setdefault(
        "guia_working",
        {
            "secciones": ["I. Datos", "II. Clima"],
            "preguntas": [
                {"numero": 1, "seccion": "I. Datos", "pregunta": "¿Edad?"},
                {"numero": 2, "seccion": "II. Clima", "pregunta": "¿Qué piensas?"},
            ],
        },
    )
    render_entrevistas_prompts()


def test_guia_editor_renders_a_text_input_per_question():
    at = AppTest.from_function(_guia_app, default_timeout=30).run()
    assert not at.exception, at.exception
    # Un campo de texto por pregunta (más los nombres de sección).
    assert len(at.text_input) >= 2
    assert at.button(key="guia_save")


def test_guia_save_calls_backend_and_syncs_enum(monkeypatch):
    import reii.pipelines.transcripciones as tr

    saves = []
    syncs = []
    monkeypatch.setattr(tr, "save_guia_entrevista", lambda d: saves.append(d))
    monkeypatch.setattr(
        tr,
        "sync_seccion_enum",
        lambda: (
            syncs.append(True)
            or {"added": ["III. Nuevo"], "removed": [], "unchanged": ["I. Datos"]}
        ),
    )
    at = AppTest.from_function(_guia_app, default_timeout=30).run()
    assert not at.exception, at.exception
    at.button(key="guia_save").click().run()
    assert saves and saves[0]["secciones"]  # la guía normalizada tiene secciones
    assert syncs  # sync_seccion_enum se llamó tras guardar
