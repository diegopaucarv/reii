# -*- coding: utf-8 -*-
"""Guía de entrevista semiestructurada (32 preguntas) para el procesamiento
por lotes de transcripciones de entrevistas.

La guía se carga desde ``ia/entrevistas/guia.json`` (editable); si el archivo
falta o es inválido, se usan los valores por defecto hardcodeados. Las
secciones y las preguntas se inyectan en los prompts de los dos agentes
(segmentación y extracción) a través de :func:`guia_como_texto`.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from reii.config import GUIA_ENTREVISTA_PATH

# ---------------------------------------------------------------------------
# Secciones (nombres exactos del enum del esquema de la Ruta 1)
# ---------------------------------------------------------------------------
_SECCIONES_DEFAULT: List[str] = [
    "I. Datos sociodemográficos y perfil profesional",
    "II. Concepciones e información general sobre el cambio climático",
    "III. Percepciones del entorno y efectos directos",
    "IV. El rol de la educación universitaria",
    "V. Acciones de mitigación y proyección docente",
    "VI. Perspectiva futura y responsabilidad social",
]

# ---------------------------------------------------------------------------
# Guía completa: 32 preguntas numeradas, agrupadas por sección.
# ---------------------------------------------------------------------------
_GUIA_DEFAULT: List[Dict] = [
    # ── I. Datos sociodemográficos y perfil profesional ──────────────
    {"numero": 1, "seccion": _SECCIONES_DEFAULT[0], "pregunta": "¿Cuál es tu sexo?."},
    {"numero": 2, "seccion": _SECCIONES_DEFAULT[0], "pregunta": "¿Edad?."},
    {
        "numero": 3,
        "seccion": _SECCIONES_DEFAULT[0],
        "pregunta": "¿Tienes dependientes económicos (personas que dependan de ti)?.",
    },
    {
        "numero": 4,
        "seccion": _SECCIONES_DEFAULT[0],
        "pregunta": "¿Cuál es tu ocupación? (si es que trabajas aparte de estudiar).",
    },
    {
        "numero": 5,
        "seccion": _SECCIONES_DEFAULT[0],
        "pregunta": "¿De qué lugar procedes (lugar de nacimiento)?.",
    },
    {
        "numero": 6,
        "seccion": _SECCIONES_DEFAULT[0],
        "pregunta": "¿Cuál es tu distrito de residencia actual?.",
    },
    {
        "numero": 7,
        "seccion": _SECCIONES_DEFAULT[0],
        "pregunta": "¿Qué carrera y especialidad estudias?.",
    },
    {
        "numero": 8,
        "seccion": _SECCIONES_DEFAULT[0],
        "pregunta": "Además de esta formación, ¿cuentas con alguna otra formación adicional?.",
    },
    {
        "numero": 9,
        "seccion": _SECCIONES_DEFAULT[0],
        "pregunta": "¿Qué te llevó a decidir por esta profesión?.",
    },
    {
        "numero": 10,
        "seccion": _SECCIONES_DEFAULT[0],
        "pregunta": "¿Cómo te ves en los próximos cinco años en tu vida profesional?.",
    },
    # ── II. Concepciones e información general sobre el cambio climático ──
    {
        "numero": 11,
        "seccion": _SECCIONES_DEFAULT[1],
        "pregunta": "Cuando piensas en cambio climático, ¿qué es lo que se te viene a la mente?.",
    },
    {
        "numero": 12,
        "seccion": _SECCIONES_DEFAULT[1],
        "pregunta": "¿Puedes profundizar algo más sobre esto? / ¿Podrías dar un ejemplo? / ¿Cuáles serían las causas?.",
    },
    {
        "numero": 13,
        "seccion": _SECCIONES_DEFAULT[1],
        "pregunta": "¿A través de qué medios has recibido noticias sobre el cambio climático?.",
    },
    {
        "numero": 14,
        "seccion": _SECCIONES_DEFAULT[1],
        "pregunta": "¿Qué medios, para ti, serían los más importantes?.",
    },
    {
        "numero": 15,
        "seccion": _SECCIONES_DEFAULT[1],
        "pregunta": "¿Has tenido oportunidad de consultar materiales con respecto al cambio climático?.",
    },
    # ── III. Percepciones del entorno y efectos directos ──────────────
    {
        "numero": 16,
        "seccion": _SECCIONES_DEFAULT[2],
        "pregunta": "¿Percibes diferencias entre cómo es hoy el clima de la región/ciudad y cómo solía ser antes?.",
    },
    {
        "numero": 17,
        "seccion": _SECCIONES_DEFAULT[2],
        "pregunta": "¿Hace cuánto tiempo estamos hablando de estos cambios?.",
    },
    {
        "numero": 18,
        "seccion": _SECCIONES_DEFAULT[2],
        "pregunta": "¿Qué significan para ti esos cambios? / ¿A qué crees que se deberían?.",
    },
    {
        "numero": 19,
        "seccion": _SECCIONES_DEFAULT[2],
        "pregunta": "Para ti, ¿cómo influye el cambio climático en la vida de las personas?.",
    },
    {
        "numero": 20,
        "seccion": _SECCIONES_DEFAULT[2],
        "pregunta": "¿Consideras que hay grupos vulnerables ante este problema?.",
    },
    {
        "numero": 21,
        "seccion": _SECCIONES_DEFAULT[2],
        "pregunta": "¿Cómo crees que el tema climático puede afectar la salud de las personas?.",
    },
    {
        "numero": 22,
        "seccion": _SECCIONES_DEFAULT[2],
        "pregunta": "En tu familia o personas cercanas, ¿de qué manera ha afectado el cambio climático? ¿Tienes alguna experiencia?.",
    },
    # ── IV. El rol de la educación universitaria ──────────────────────
    {
        "numero": 23,
        "seccion": _SECCIONES_DEFAULT[3],
        "pregunta": "¿Consideras que en la educación universitaria te ofrecen conocimientos suficientes sobre el cambio climático? ¿Por qué?.",
    },
    {
        "numero": 24,
        "seccion": _SECCIONES_DEFAULT[3],
        "pregunta": "En tu plan de estudios como estudiante de educación, ¿se han desarrollado estos temas en alguna asignatura?.",
    },
    {
        "numero": 25,
        "seccion": _SECCIONES_DEFAULT[3],
        "pregunta": "¿Consideras importante abordar los temas del cambio climático en la educación? ¿Por qué?.",
    },
    {
        "numero": 26,
        "seccion": _SECCIONES_DEFAULT[3],
        "pregunta": "Si tuvieses la oportunidad de cambiar o aportar a tu plan de estudios (o a una asignatura que se relacione con el tema), ¿qué cosas cambiarías/incrementarías?.",
    },
    # ── V. Acciones de mitigación y proyección docente ────────────────
    {
        "numero": 27,
        "seccion": _SECCIONES_DEFAULT[4],
        "pregunta": "¿Tú o alguno de tus compañeros o profesores han asumido alguna acción o actividad en particular aportando a la mitigación del cambio climático?.",
    },
    {
        "numero": 28,
        "seccion": _SECCIONES_DEFAULT[4],
        "pregunta": "¿Qué podrías hacer como acción sostenible para el problema del cambio climático? / ¿En qué aportaría?.",
    },
    {
        "numero": 29,
        "seccion": _SECCIONES_DEFAULT[4],
        "pregunta": "Como futuro docente, ¿te interesaría enseñar sobre el tema? ¿A quiénes, qué cosas enseñarías y cómo lo harías?.",
    },
    # ── VI. Perspectiva futura y responsabilidad social ───────────────
    {
        "numero": 30,
        "seccion": _SECCIONES_DEFAULT[5],
        "pregunta": "¿Crees que estos cambios climáticos con el tiempo se incrementarán, se mantendrán igual o van a mejorar? ¿Por qué?.",
    },
    {
        "numero": 31,
        "seccion": _SECCIONES_DEFAULT[5],
        "pregunta": "¿Crees que se está haciendo algo en el país o en la región por el cambio climático? ¿Será suficiente?.",
    },
    {
        "numero": 32,
        "seccion": _SECCIONES_DEFAULT[5],
        "pregunta": "¿Qué cosas crees que nosotros deberíamos hacer como sociedad para enfrentar el cambio climático?.",
    },
]


def _read_guia_json() -> Optional[Tuple[List[str], List[Dict]]]:
    """Lee el JSON de la guía y devuelve ``(secciones, preguntas)``.

    Devuelve ``None`` si el archivo falta, no es JSON válido o no tiene la
    estructura esperada (lista de secciones y lista de preguntas).
    """
    if not os.path.exists(GUIA_ENTREVISTA_PATH):
        return None
    try:
        with open(GUIA_ENTREVISTA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    secciones = data.get("secciones")
    preguntas = data.get("preguntas")
    if not isinstance(secciones, list) or not isinstance(preguntas, list):
        return None
    if not all(isinstance(s, str) for s in secciones):
        return None
    if not all(
        isinstance(p, dict) and {"numero", "seccion", "pregunta"} <= set(p)
        for p in preguntas
    ):
        return None
    return secciones, preguntas


# Carga la guía desde el JSON editable; si falta o es inválido, usa los
# valores por defecto hardcodeados.
SECCIONES, GUIA_ENTREVISTA = _read_guia_json() or (_SECCIONES_DEFAULT, _GUIA_DEFAULT)


def guia_como_texto() -> str:
    """Renderiza la guía como un bloque de texto numerado.

    Formato: encabezados de sección en números romanos seguidos de las
    preguntas numeradas (1–32). Se inyecta tal cual en los prompts de los
    agentes de segmentación y extracción.
    """
    lineas: List[str] = []
    for seccion in SECCIONES:
        lineas.append(seccion)
        for item in GUIA_ENTREVISTA:
            if item["seccion"] == seccion:
                lineas.append(f"{item['numero']}. {item['pregunta']}")
        lineas.append("")  # línea en blanco entre secciones
    return "\n".join(lineas).rstrip()
