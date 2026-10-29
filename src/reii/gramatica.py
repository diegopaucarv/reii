"""Capa de personalización gramatical para REII.

Permite ajustar, sin tocar el código, qué palabras se consideran activas,
suplementarias o excluidas, así como frases multi-palabra (expresiones) y
sinónimos que agrupan variantes bajo una raíz canónica.

La configuración vive en un JSON (por defecto ``ia/gramatica.json``) y se
fusiona sobre ``DEFAULT_GRAMATICA``. Si el archivo no existe o no se puede
leer, se devuelve una copia de los valores por defecto (cero riesgo).
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

from reii.config import GRAMATICA_PATH

DEFAULT_GRAMATICA: Dict[str, Any] = {
    # POS "núcleo" del vocabulario activo. PROPN/X NO se declaran aquí:
    # los gobierna retention_policy (ver reii/retention_policy.py y
    # docs/design_v5/RETENTION_POLICY.md).
    "content_pos": ["NOUN", "VERB", "ADJ", "ADV"],
    # Política de formas activas: pos_only | pos_plus_propn |
    # tool_adverb_filtered. Default = tool_adverb_filtered (conservadora;
    # enmienda 2026-10-28, RETENTION_POLICY.md §10).
    "retention_policy": "tool_adverb_filtered",
    # Override opcional de la lista cerrada de adverbios-herramienta
    # (vacío = usar TOOL_ADVERB_LEMMAS, que está pre-registrada).
    "tool_adverb_lemmas": [],
    "stopwords_extra": [],
    "expresiones": [],
    "palabras_activas": {},
    "palabras_suplementarias": [],
    "palabras_excluir": [],
    "sinonimos": {},
}


def load_gramatica(path: str | None = None) -> Dict[str, Any]:
    """Lee la gramática desde ``path`` (por defecto ``GRAMATICA_PATH``).

    Devuelve ``DEFAULT_GRAMATICA`` fusionada con el contenido del archivo si
    existe y es un JSON válido. Ante cualquier error (archivo inexistente,
    JSON inválido, etc.) devuelve una copia de ``DEFAULT_GRAMATICA``.
    """
    resolved = path or GRAMATICA_PATH
    merged = copy.deepcopy(DEFAULT_GRAMATICA)
    try:
        if resolved and os.path.exists(resolved):
            with open(resolved, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for key, value in data.items():
                    if key in merged:
                        merged[key] = value
    except (OSError, ValueError, TypeError):
        # Cualquier error de lectura/parseo → valores por defecto.
        pass
    return merged


def save_gramatica(data: Dict[str, Any], path: str | None = None) -> None:
    """Escribe ``data`` como JSON en ``path`` (por defecto ``GRAMATICA_PATH``).

    Crea los directorios padre si no existen.
    """
    resolved = path or GRAMATICA_PATH
    if resolved:
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
