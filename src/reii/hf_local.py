"""Resuelve modelos de HuggingFace a directorios locales ya cacheados.

``transformers`` 4.57–5.0 llama a ``huggingface_hub.model_info()`` desde
``PreTrainedTokenizerBase._patch_mistral_regex`` para *cualquier* repo id,
aunque los ficheros estén ya en la caché local. En un contenedor sin DNS esa
consulta lanza ``NameResolutionError`` y aborta ``from_pretrained`` antes de
cargar nada. Con una ruta local, transformers marca ``_is_local=True`` y se
salta la consulta.

Uso: ``resolve_model_dir(name) or name``.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _hf_hub_cache() -> Path:
    base = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    return Path(base) / "hub"


@lru_cache(maxsize=None)
def resolve_model_dir(model_name: str) -> Optional[str]:
    """Snapshot local de ``model_name``, o ``None`` si no hay copia utilizable.

    ``None`` = el llamador debe seguir usando el repo id (comportamiento
    online de siempre).
    """
    if not model_name:
        return None

    name = str(model_name)
    expanded = os.path.expanduser(name)
    if os.path.isdir(expanded):
        return expanded

    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(name, local_files_only=True)
    except Exception as exc:  # sin caché, o refs/ ilegible
        logging.debug("[HF] %s no resoluble en caché local: %s", name, exc)
    else:
        logging.info("[HF] %s -> %s (caché local)", name, snapshot)
        return snapshot

    # Cachés copiadas a mano (sin refs/main): buscar el snapshot a pelo.
    snapshots = _hf_hub_cache() / f"models--{name.replace('/', '--')}" / "snapshots"
    if snapshots.is_dir():
        for candidate in sorted(snapshots.iterdir(), reverse=True):
            if (candidate / "config.json").is_file():
                logging.info("[HF] %s -> %s (snapshot sin refs)", name, candidate)
                return str(candidate)

    return None
