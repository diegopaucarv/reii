"""Acumulación de UCEs en UCs por formas analizadas DISTINTAS.

``SegmentadorALCESTE.construir_ucs`` (Método A, word-count) y
``UCBuilder._merge_small_segments`` (Método C, embeddings) necesitan la
misma pregunta: "¿cuántas formas analizadas tiene este tramo de UCEs?".
Antes, ambos contaban el TOTAL de lemas con repeticiones
(``sum(len(lemmas))``); una UC con "niños niños niños" sumaba 3. ALCESTE
exige "un número entero mínimo de palabras analizadas **distintas**" — con
repeticiones, el criterio queda más laxo de lo que debería (una UC muy
repetitiva alcanza el umbral con menos UCEs reales).

Este módulo es la única implementación de ese conteo y del agrupamiento
secuencial que usa ``construir_ucs``; ``main_workflow.py`` lo importa en
vez de reimplementarlo, para que el método A y el método C compartan
exactamente la misma semántica de "formas distintas" sin divergir.

Sin dependencias externas (solo stdlib), para poder probarlo sin spaCy.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def formas_distintas(lemmas_por_item: Sequence[Sequence[str]], start: int, end: int) -> int:
    """Formas (lemas) DISTINTAS en ``lemmas_por_item[start:end+1]`` (ambos
    extremos inclusive). ``start > end`` (tramo vacío) da 0."""
    out = set()
    for i in range(start, end + 1):
        out.update(lemmas_por_item[i])
    return len(out)


def agrupar_secuencial_por_umbral(
    lemmas_por_item: Sequence[Sequence[str]],
    min_forms: int,
) -> List[Tuple[int, int]]:
    """Agrupa los índices ``[0, n)`` secuencialmente: cada grupo acumula
    items consecutivos hasta que las formas DISTINTAS acumuladas alcanzan
    ``min_forms``; ahí cierra el grupo y empieza uno nuevo vacío.

    Si al terminar queda un grupo abierto sin alcanzar el umbral (el
    "resto" al final de un documento), se incluye igual como último grupo
    — no se descarta — misma política que el ``if acum_ids:`` final de
    ``construir_ucs``.

    Devuelve ``[(start, end), ...]`` con índices inclusive que cubren TODO
    ``range(n)`` sin huecos ni solapamientos, en orden. Con ``n == 0``
    devuelve ``[]``. ``min_forms <= 0`` hace que cada item cierre su
    propio grupo de inmediato (el umbral se alcanza en cuanto hay al
    menos 1 forma, o incluso con 0 formas si ``min_forms <= 0``).
    """
    n = len(lemmas_por_item)
    grupos: List[Tuple[int, int]] = []
    if n == 0:
        return grupos

    acumuladas: set = set()
    start = 0
    for i in range(n):
        acumuladas.update(lemmas_por_item[i])
        if len(acumuladas) >= min_forms:
            grupos.append((start, i))
            start = i + 1
            acumuladas = set()
    if start < n:
        grupos.append((start, n - 1))
    return grupos
