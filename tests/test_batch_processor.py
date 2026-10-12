# -*- coding: utf-8 -*-
"""Tests para la modularización de prompts de ``reii.batch_processor``.

Verifican la especificación del API objetivo:

- ``load_prompt_config``: lectura de JSON de configuración de prompt.
- Constantes ``_PROMPT_SEGMENTACION`` / ``_PROMPT_EXTRACCION`` cargadas
  desde ``ia/entrevistas/*.json``.
- ``build_prompt_segmentacion`` / ``build_prompt_extraccion``: estructura de
  mensajes y bloque de localismos.
- ``BatchProcessor``: API sin cambios (db_path temporal, base vacía).

No se llama a ningún modelo LLM ni a la red.
"""

import json
import os

import pytest

from reii import batch_processor as bp

# Raíz del repo: tests/ está un nivel por debajo de la raíz.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEGMENTACION_CONFIG = os.path.join(ROOT, "ia", "entrevistas", "segmentacion.json")
EXTRACCION_CONFIG = os.path.join(ROOT, "ia", "entrevistas", "extraccion.json")

CONFIG_KEYS = {"name", "role", "instructions", "json_schema", "few_shot_examples"}


# ── 1. load_prompt_config ────────────────────────────────────────────────


def test_load_prompt_config_archivo_valido(tmp_path):
    cfg = {
        "name": "segmentacion",
        "role": "system",
        "instructions": "Instrucciones de prueba.",
        "json_schema": {"type": "object", "properties": {}},
        "few_shot_examples": [{"input": "x", "output": "y"}],
    }
    path = tmp_path / "segmentacion.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    result = bp.load_prompt_config(str(path))

    assert isinstance(result, dict)
    assert CONFIG_KEYS <= set(result.keys())


def test_load_prompt_config_archivo_inexistente(tmp_path):
    assert bp.load_prompt_config(str(tmp_path / "no_existe.json")) == {}


def test_load_prompt_config_json_invalido(tmp_path):
    path = tmp_path / "invalido.json"
    path.write_text("{esto no es json valido", encoding="utf-8")
    assert bp.load_prompt_config(str(path)) == {}


# ── 2. Archivos de configuración de prompts ──────────────────────────────


@pytest.mark.parametrize("ruta", [SEGMENTACION_CONFIG, EXTRACCION_CONFIG])
def test_archivo_config_prompt_existe_y_tiene_claves(ruta):
    assert os.path.exists(ruta), f"Falta el archivo de configuración: {ruta}"
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert "instructions" in data
    assert "json_schema" in data


# ── 3. build_prompt_segmentacion: bloque de localismos ───────────────────


def test_build_prompt_segmentacion_incluye_localismos():
    localismos = {"glosario": {"chamba": "trabajo"}, "preservar": []}
    mensajes = bp.build_prompt_segmentacion("transcripción de prueba", localismos)
    assert "chamba" in mensajes[0]["content"]


def test_build_prompt_segmentacion_sin_localismos_no_incluye_bloque():
    # Se pasa {} explícitamente: con None se cargaría ia/localismos.json,
    # cuyo glosario ya contiene "chamba" y haría el test dependiente del archivo.
    mensajes = bp.build_prompt_segmentacion("transcripción de prueba", {})
    assert "chamba" not in mensajes[0]["content"]


# ── 4. Estructura de los prompts ─────────────────────────────────────────


@pytest.mark.parametrize(
    "builder",
    [bp.build_prompt_segmentacion, bp.build_prompt_extraccion],
)
def test_prompt_estructura_dos_mensajes_y_transcripcion(builder):
    transcripcion = "Entrevista de prueba con contenido único."
    mensajes = builder(transcripcion, {})

    assert isinstance(mensajes, list)
    assert len(mensajes) == 2
    assert [m["role"] for m in mensajes] == ["system", "user"]
    assert transcripcion in mensajes[1]["content"]


# ── 5. Constantes de módulo (especificación) ─────────────────────────────


@pytest.mark.parametrize("nombre", ["_PROMPT_SEGMENTACION", "_PROMPT_EXTRACCION"])
def test_constantes_prompt_cargadas(nombre):
    data = getattr(bp, nombre)
    assert isinstance(data, dict)
    assert "instructions" in data
    assert "json_schema" in data


# ── 6. BatchProcessor ────────────────────────────────────────────────────


def test_batch_processor_db_vacia(tmp_path):
    db_path = str(tmp_path / "entrevistas.json")
    proc = bp.BatchProcessor(db_path=db_path)
    assert proc.load_all() == []
