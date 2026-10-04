# -*- coding: utf-8 -*-
"""Guía de entrevista semiestructurada (32 preguntas) para el procesamiento
por lotes de transcripciones de entrevistas.

Este módulo es la fuente única de verdad de la guía: las secciones y las
preguntas se inyectan en los prompts de los dos agentes (segmentación y
extracción) a través de :func:`guia_como_texto`.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Secciones (nombres exactos del enum del esquema de la Ruta 1)
# ---------------------------------------------------------------------------
SECCIONES: List[str] = [
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
GUIA_ENTREVISTA: List[Dict] = [
    # ── I. Datos sociodemográficos y perfil profesional ──────────────
    {"numero": 1, "seccion": SECCIONES[0], "pregunta": "¿Cuál es tu sexo?."},
    {"numero": 2, "seccion": SECCIONES[0], "pregunta": "¿Edad?."},
    {
        "numero": 3,
        "seccion": SECCIONES[0],
        "pregunta": "¿Tienes dependientes económicos (personas que dependan de ti)?.",
    },
    {
        "numero": 4,
        "seccion": SECCIONES[0],
        "pregunta": "¿Cuál es tu ocupación? (si es que trabajas aparte de estudiar).",
    },
    {
        "numero": 5,
        "seccion": SECCIONES[0],
        "pregunta": "¿De qué lugar procedes (lugar de nacimiento)?.",
    },
    {
        "numero": 6,
        "seccion": SECCIONES[0],
        "pregunta": "¿Cuál es tu distrito de residencia actual?.",
    },
    {
        "numero": 7,
        "seccion": SECCIONES[0],
        "pregunta": "¿Qué carrera y especialidad estudias?.",
    },
    {
        "numero": 8,
        "seccion": SECCIONES[0],
        "pregunta": "Además de esta formación, ¿cuentas con alguna otra formación adicional?.",
    },
    {
        "numero": 9,
        "seccion": SECCIONES[0],
        "pregunta": "¿Qué te llevó a decidir por esta profesión?.",
    },
    {
        "numero": 10,
        "seccion": SECCIONES[0],
        "pregunta": "¿Cómo te ves en los próximos cinco años en tu vida profesional?.",
    },
    # ── II. Concepciones e información general sobre el cambio climático ──
    {
        "numero": 11,
        "seccion": SECCIONES[1],
        "pregunta": "Cuando piensas en cambio climático, ¿qué es lo que se te viene a la mente?.",
    },
    {
        "numero": 12,
        "seccion": SECCIONES[1],
        "pregunta": "¿Puedes profundizar algo más sobre esto? / ¿Podrías dar un ejemplo? / ¿Cuáles serían las causas?.",
    },
    {
        "numero": 13,
        "seccion": SECCIONES[1],
        "pregunta": "¿A través de qué medios has recibido noticias sobre el cambio climático?.",
    },
    {
        "numero": 14,
        "seccion": SECCIONES[1],
        "pregunta": "¿Qué medios, para ti, serían los más importantes?.",
    },
    {
        "numero": 15,
        "seccion": SECCIONES[1],
        "pregunta": "¿Has tenido oportunidad de consultar materiales con respecto al cambio climático?.",
    },
    # ── III. Percepciones del entorno y efectos directos ──────────────
    {
        "numero": 16,
        "seccion": SECCIONES[2],
        "pregunta": "¿Percibes diferencias entre cómo es hoy el clima de la región/ciudad y cómo solía ser antes?.",
    },
    {
        "numero": 17,
        "seccion": SECCIONES[2],
        "pregunta": "¿Hace cuánto tiempo estamos hablando de estos cambios?.",
    },
    {
        "numero": 18,
        "seccion": SECCIONES[2],
        "pregunta": "¿Qué significan para ti esos cambios? / ¿A qué crees que se deberían?.",
    },
    {
        "numero": 19,
        "seccion": SECCIONES[2],
        "pregunta": "Para ti, ¿cómo influye el cambio climático en la vida de las personas?.",
    },
    {
        "numero": 20,
        "seccion": SECCIONES[2],
        "pregunta": "¿Consideras que hay grupos vulnerables ante este problema?.",
    },
    {
        "numero": 21,
        "seccion": SECCIONES[2],
        "pregunta": "¿Cómo crees que el tema climático puede afectar la salud de las personas?.",
    },
    {
        "numero": 22,
        "seccion": SECCIONES[2],
        "pregunta": "En tu familia o personas cercanas, ¿de qué manera ha afectado el cambio climático? ¿Tienes alguna experiencia?.",
    },
    # ── IV. El rol de la educación universitaria ──────────────────────
    {
        "numero": 23,
        "seccion": SECCIONES[3],
        "pregunta": "¿Consideras que en la educación universitaria te ofrecen conocimientos suficientes sobre el cambio climático? ¿Por qué?.",
    },
    {
        "numero": 24,
        "seccion": SECCIONES[3],
        "pregunta": "En tu plan de estudios como estudiante de educación, ¿se han desarrollado estos temas en alguna asignatura?.",
    },
    {
        "numero": 25,
        "seccion": SECCIONES[3],
        "pregunta": "¿Consideras importante abordar los temas del cambio climático en la educación? ¿Por qué?.",
    },
    {
        "numero": 26,
        "seccion": SECCIONES[3],
        "pregunta": "Si tuvieses la oportunidad de cambiar o aportar a tu plan de estudios (o a una asignatura que se relacione con el tema), ¿qué cosas cambiarías/incrementarías?.",
    },
    # ── V. Acciones de mitigación y proyección docente ────────────────
    {
        "numero": 27,
        "seccion": SECCIONES[4],
        "pregunta": "¿Tú o alguno de tus compañeros o profesores han asumido alguna acción o actividad en particular aportando a la mitigación del cambio climático?.",
    },
    {
        "numero": 28,
        "seccion": SECCIONES[4],
        "pregunta": "¿Qué podrías hacer como acción sostenible para el problema del cambio climático? / ¿En qué aportaría?.",
    },
    {
        "numero": 29,
        "seccion": SECCIONES[4],
        "pregunta": "Como futuro docente, ¿te interesaría enseñar sobre el tema? ¿A quiénes, qué cosas enseñarías y cómo lo harías?.",
    },
    # ── VI. Perspectiva futura y responsabilidad social ───────────────
    {
        "numero": 30,
        "seccion": SECCIONES[5],
        "pregunta": "¿Crees que estos cambios climáticos con el tiempo se incrementarán, se mantendrán igual o van a mejorar? ¿Por qué?.",
    },
    {
        "numero": 31,
        "seccion": SECCIONES[5],
        "pregunta": "¿Crees que se está haciendo algo en el país o en la región por el cambio climático? ¿Será suficiente?.",
    },
    {
        "numero": 32,
        "seccion": SECCIONES[5],
        "pregunta": "¿Qué cosas crees que nosotros deberíamos hacer como sociedad para enfrentar el cambio climático?.",
    },
]


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
