# -*- coding: utf-8 -*-
"""Tests para la guía de entrevista editable (``ia/entrevistas/guia.json``).

Verifican:

- El archivo ``ia/entrevistas/guia.json`` existe con 6 secciones y 32 preguntas.
- ``guia_entrevista.SECCIONES`` / ``GUIA_ENTREVISTA`` se cargan desde el JSON.
- ``guia_como_texto()`` renderiza preguntas y secciones.
- I/O puro de prompts de entrevista (``ia_config_editor``) con round-trip
  sobre ``tmp_path`` (sin tocar archivos reales).
- Fallback a los valores por defecto si el JSON falta o es inválido.
"""

import importlib
import json
import os

import pytest

import reii.guia_entrevista as ge
from reii import ia_config_editor as ice

# Raíz del repo: tests/ está un nivel por debajo de la raíz.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIA_PATH = os.path.join(ROOT, "ia", "entrevistas", "guia.json")


@pytest.fixture(autouse=True)
def _recargar_guia_entrevista():
    """Recarga el módulo tras cada test para restaurar el estado del JSON."""
    yield
    importlib.reload(ge)


def _leer_guia_json():
    with open(GUIA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 1. Archivo ia/entrevistas/guia.json ──────────────────────────────────


def test_guia_json_existe_con_6_secciones_y_32_preguntas():
    assert os.path.exists(GUIA_PATH), f"Falta el archivo: {GUIA_PATH}"
    data = _leer_guia_json()
    assert isinstance(data, dict)
    assert "secciones" in data and "preguntas" in data
    assert len(data["secciones"]) == 6
    assert len(data["preguntas"]) == 32


# ── 2. SECCIONES / GUIA_ENTREVISTA cargados desde el JSON ────────────────


def test_secciones_y_guia_cargadas_desde_json():
    data = _leer_guia_json()
    assert ge.SECCIONES == data["secciones"]
    assert ge.GUIA_ENTREVISTA == data["preguntas"]


# ── 3. guia_como_texto ───────────────────────────────────────────────────


def test_guia_como_texto_contiene_preguntas_y_secciones():
    texto = ge.guia_como_texto()
    assert "1. ¿Cuál es tu sexo?." in texto
    assert "32. " in texto
    for seccion in ge.SECCIONES:
        assert seccion in texto


# ── 4. load_entrevista_prompt (archivo real) ─────────────────────────────


def test_load_entrevista_prompt_segmentacion():
    data = ice.load_entrevista_prompt("segmentacion")
    assert isinstance(data, dict)
    assert "instructions" in data
    assert "json_schema" in data


# ── 5. Round-trips sobre tmp_path (sin tocar archivos reales) ────────────


def test_round_trip_entrevista_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(ice, "ENTREVISTAS_DIR", tmp_path)
    data = {
        "name": "segmentacion",
        "role": "system",
        "instructions": "Instrucciones de prueba.",
        "json_schema": {"type": "object", "properties": {}},
        "few_shot_examples": [],
    }
    ice.save_entrevista_prompt("segmentacion", data)
    assert ice.load_entrevista_prompt("segmentacion") == data


def test_round_trip_guia_entrevista(tmp_path, monkeypatch):
    monkeypatch.setattr(ice, "GUIA_ENTREVISTA_PATH", str(tmp_path / "guia.json"))
    data = {
        "secciones": ["I. Sección de prueba"],
        "preguntas": [
            {
                "numero": 1,
                "seccion": "I. Sección de prueba",
                "pregunta": "¿Pregunta de prueba?.",
            }
        ],
    }
    ice.save_guia_entrevista(data)
    assert ice.load_guia_entrevista() == data


# ── 6. Fallback a defaults si falta el JSON ───────────────────────────────


def test_fallback_a_defaults_si_falta_json(tmp_path, monkeypatch):
    ruta_inexistente = str(tmp_path / "no_existe.json")
    # Se parchea la fuente (reii.config) para que el reload del módulo la
    # recoja, y también el atributo del propio módulo.
    monkeypatch.setattr("reii.config.GUIA_ENTREVISTA_PATH", ruta_inexistente)
    monkeypatch.setattr(ge, "GUIA_ENTREVISTA_PATH", ruta_inexistente)
    importlib.reload(ge)
    assert ge.SECCIONES == ge._SECCIONES_DEFAULT
    assert ge.GUIA_ENTREVISTA == ge._GUIA_DEFAULT


# ── 7. sync_seccion_enum ──────────────────────────────────────────────────


def _escribir_guia(tmp_path, secciones):
    guia = {"secciones": secciones, "preguntas": []}
    ice.save_guia_entrevista(guia)


def _escribir_segmentacion(tmp_path, enum):
    data = {
        "name": "segmentacion_estructural",
        "role": "",
        "instructions": "",
        "json_schema": {
            "type": "object",
            "properties": {
                "segmentos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "seccion_entrevista": {
                                "type": "string",
                                "enum": enum,
                            }
                        },
                        "required": ["seccion_entrevista"],
                    },
                }
            },
        },
        "few_shot_examples": [],
    }
    ice.save_entrevista_prompt("segmentacion", data)


def _enum_actual(tmp_path):
    data = ice.load_entrevista_prompt("segmentacion")
    return data["json_schema"]["properties"]["segmentos"]["items"]["properties"][
        "seccion_entrevista"
    ]["enum"]


@pytest.fixture
def _entrevistas_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(ice, "ENTREVISTAS_DIR", tmp_path)
    monkeypatch.setattr(ice, "GUIA_ENTREVISTA_PATH", str(tmp_path / "guia.json"))
    return tmp_path


def test_sync_seccion_enum_agrega_seccion(_entrevistas_tmp):
    tmp_path = _entrevistas_tmp
    _escribir_guia(tmp_path, ["I. A", "II. B"])
    _escribir_segmentacion(tmp_path, ["I. A", ice.TEXTO_NO_ASIGNADO])

    report = ice.sync_seccion_enum()
    assert report["added"] == ["II. B"]
    assert report["removed"] == []
    assert report["unchanged"] == ["I. A"]
    assert "error" not in report
    assert _enum_actual(tmp_path) == ["I. A", "II. B", ice.TEXTO_NO_ASIGNADO]


def test_sync_seccion_enum_elimina_seccion_y_conserva_no_asignado(
    _entrevistas_tmp,
):
    tmp_path = _entrevistas_tmp
    _escribir_guia(tmp_path, ["I. A"])
    _escribir_segmentacion(tmp_path, ["I. A", "II. B", "III. C", ice.TEXTO_NO_ASIGNADO])

    report = ice.sync_seccion_enum()
    assert report["added"] == []
    assert report["removed"] == ["II. B", "III. C"]
    assert report["unchanged"] == ["I. A"]
    assert _enum_actual(tmp_path) == ["I. A", ice.TEXTO_NO_ASIGNADO]


def test_sync_seccion_enum_idempotente(_entrevistas_tmp):
    tmp_path = _entrevistas_tmp
    _escribir_guia(tmp_path, ["I. A", "II. B"])
    _escribir_segmentacion(tmp_path, ["I. A", "II. B", ice.TEXTO_NO_ASIGNADO])

    ice.sync_seccion_enum()
    report = ice.sync_seccion_enum()
    assert report["added"] == []
    assert report["removed"] == []
    assert report["unchanged"] == ["I. A", "II. B"]
    assert _enum_actual(tmp_path) == ["I. A", "II. B", ice.TEXTO_NO_ASIGNADO]


def test_sync_seccion_enum_error_si_faltan_archivos(_entrevistas_tmp):
    report = ice.sync_seccion_enum()
    assert report["added"] == []
    assert report["removed"] == []
    assert report["unchanged"] == []
    assert "error" in report


# ── 8. validate_guia ─────────────────────────────────────────────────────


def test_validate_guia_acepta_guia_valida():
    data = {
        "secciones": ["I. A"],
        "preguntas": [{"numero": 1, "seccion": "I. A", "pregunta": "¿Pregunta?."}],
    }
    ok, msg = ice.validate_guia(data)
    assert ok is True
    assert msg == ""


def test_validate_guia_rechaza_malformadas():
    casos = [
        None,
        [],
        {"secciones": [], "preguntas": []},
        {"secciones": ["I. A"], "preguntas": "no"},
        {
            "secciones": ["I. A"],
            "preguntas": [{"numero": "1", "seccion": "I. A", "pregunta": "x"}],
        },
        {
            "secciones": ["I. A"],
            "preguntas": [{"numero": 1, "seccion": 5, "pregunta": "x"}],
        },
        {
            "secciones": ["I. A"],
            "preguntas": [{"numero": 1, "seccion": "I. A", "pregunta": 7}],
        },
    ]
    for caso in casos:
        ok, msg = ice.validate_guia(caso)
        assert ok is False
        assert isinstance(msg, str) and msg


# ── 9. normalize_guia ─────────────────────────────────────────────────────


def test_normalize_guia_renumera_y_recorta():
    data = {
        "secciones": ["  I. A  ", "  II. B  "],
        "preguntas": [
            {"numero": 99, "seccion": "  I. A  ", "pregunta": "  ¿Primera?  "},
            {"numero": 5, "seccion": "  II. B  ", "pregunta": "   "},
            {"numero": 3, "seccion": "  II. B  ", "pregunta": "  ¿Segunda?  "},
        ],
    }
    out = ice.normalize_guia(data)
    assert out["secciones"] == ["I. A", "II. B"]
    assert [p["numero"] for p in out["preguntas"]] == [1, 2]
    assert out["preguntas"][0]["pregunta"] == "¿Primera?"
    assert out["preguntas"][0]["seccion"] == "I. A"
    assert out["preguntas"][1]["pregunta"] == "¿Segunda?"
    # No escribe en disco: la entrada original queda intacta.
    assert data["preguntas"][0]["numero"] == 99
