"""Agrupación de ``texto_segmentos`` por sección — fix del límite de segmento
en ``SegmentadorALCESTE.segmentar_en_uces`` (main_workflow.py).

Antes, cada entrada ``texto_segmentos`` del JSON de transcripción se
parseaba y cortaba por separado: cualquier fragmento con ≤ ideal_size+window
tokens se convertía en UNA UCE entera, sin importar qué tan corto fuera
(ver la guarda ``remaining <= ideal_size + window`` en
``_segmentar_reinert_spacy``). Eso generaba una "cola" forzada por cada
segmento de la entrevista, muy por debajo de ``uce_target_size``. ALCESTE,
en cambio, concatena Segments de Texte Calibrés libremente dentro de todo
el u.c.i. (la entrevista), sin ese límite artificial por sub-fragmento.

Este módulo agrupa el texto de todos los ``texto_segmentos`` que
comparten la misma ``seccion`` ANTES de cortar — misma regla que ya usa
``section_registry`` para numerar secciones, solo que ahora también se usa
para concatenar el texto, no solo para el id.

Sin dependencias externas (solo stdlib), para poder probarlo sin spaCy ni
el resto de REII.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Puntuación que ``_segmentar_reinert_spacy`` pondera como candidato de
# corte "fuerte" (ver los pesos en main_workflow.py); se usa aquí para no
# agregar un punto de más si el fragmento ya termina en algo equivalente.
PUNTUACION_FUERTE: Tuple[str, ...] = (".", "?", "!", ";", ":", "\u2026")


def con_punto_final(texto: str) -> str:
    """Agrega un punto al final de ``texto`` si no termina ya en
    :data:`PUNTUACION_FUERTE`.

    Al concatenar varios ``texto_segmentos``, esto preserva la frontera
    ORIGINAL entre dos fragmentos como un candidato de corte de alta
    prioridad para el cortador ponderado por puntuación — en vez de que
    desaparezca dentro del texto corrido (lo que forzaría cortes en
    lugares arbitrarios) o de que el límite entre fragmentos distintos sea
    indistinguible de media frase.

    Un texto vacío (o solo espacios) se devuelve tal cual, sin agregar
    puntuación a la nada.
    """
    t = texto.rstrip()
    if not t:
        return texto
    if t[-1] not in PUNTUACION_FUERTE:
        return t + "."
    return t


def agrupar_segmentos_por_seccion(
    segmentos: List[Dict[str, Any]],
    texto_key: str = "texto",
    seccion_default: str = "Sin_Seccion",
) -> List[Tuple[str, str]]:
    """Agrupa los textos de ``segmentos`` por el valor de
    ``otros_datos.seccion``, concatenándolos con un espacio.

    Agrupa TODAS las entradas que comparten una ``seccion``, no solo las
    consecutivas (misma regla que ``section_registry`` en
    ``segmentar_en_uces``, que numera por valor de ``seccion`` sin
    importar la posición). El orden del resultado es el de PRIMERA
    aparición de cada ``seccion`` en ``segmentos``, para que la numeración
    de ``section_id`` corriente arriba no cambie.

    Fragmentos vacíos (texto en blanco, o la clave ausente) se descartan
    silenciosamente — igual que el comportamiento anterior por-segmento,
    que también los saltaba con un ``continue``. Si TODOS los fragmentos
    de una sección quedan vacíos, esa sección no aparece en el resultado.

    Devuelve ``[(seccion, texto_concatenado), ...]``.
    """
    grupos: Dict[str, List[str]] = {}
    orden: List[str] = []
    for seg in segmentos:
        texto = (seg.get(texto_key) or "").strip()
        if not texto:
            continue
        seccion = seg.get("otros_datos", {}).get("seccion", seccion_default)
        if seccion not in grupos:
            grupos[seccion] = []
            orden.append(seccion)
        grupos[seccion].append(con_punto_final(texto))
    return [(seccion, " ".join(grupos[seccion])) for seccion in orden]
