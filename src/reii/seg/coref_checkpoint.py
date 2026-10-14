"""Checkpoint de la etapa COREF: validación de parámetros y huella del algoritmo.

Sin dependencias pesadas (no importa stanza/torch/spacy) para poder probarse aislado,
igual que :mod:`reii.seg.root_cache`.

El checkpoint guarda, además de los datos de progreso, tres campos de validación:

* ``version``          — versión del FORMATO del JSON (búmpala solo si cambia la
                         estructura, no el algoritmo).
* ``context_units`` / ``min_forms`` — parámetros del pase (B1 usa context=2, B2 usa
                         context=4); así B2 no reutiliza los resultados de B1.
* ``code_fingerprint`` — huella del código de extracción de cadenas de coref
                         (``segmentador.py``). Si cambia el algoritmo (p. ej. el
                         batching), la huella cambia y los checkpoints viejos se
                         invalidan automáticamente, sin bump manual.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Versión del FORMATO del checkpoint de coref. Búmpala solo cuando cambie la
# estructura del JSON (no cuando cambie el algoritmo: para eso está la huella).
COREF_CHECKPOINT_VERSION = 2

# Módulo cuyo código sella la huella del algoritmo de coref.
_COREF_SOURCE_MODULE = "reii.seg.segmentador"


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def checkpoint_path_for_pass(
    checkpoint_path: Optional[str],
    context_units: int,
) -> Optional[str]:
    """Deriva una ruta de checkpoint por pase (B1/B2) a partir de la base.

    B1 (context=2) y B2 (context=4) comparten el mismo ``--checkpoint-path``,
    pero sus resultados NO son intercambiables. Para que cada pase tenga su
    propia memoria (y no se pisen entre sí), se añade un sufijo con
    ``context_units``. Así ``coref_all.json`` se convierte en
    ``coref_all.ctx2.json`` (B1) y ``coref_all.ctx4.json`` (B2).

    Solo se usa ``context_units`` (fijo: 2 y 4) para no crear archivos
    infinitos: ``min_forms`` se valida dentro del JSON (ver
    :func:`_cargar_checkpoint_coref`), así que un cambio de ``min_forms``
    sobrescribe el mismo archivo en vez de crear uno nuevo.

    Devuelve None si ``checkpoint_path`` es None/vacío (sin checkpointing).
    """
    if not checkpoint_path:
        return None
    base, ext = os.path.splitext(checkpoint_path)
    return f"{base}.ctx{context_units}{ext}"


def adaptive_coref_batch_size(base_bs: int, total_pending_chars: int) -> int:
    """Reduce el tamaño de lote de coref cuando el texto pendiente es grande.

    ``_extract_global_chains`` procesa sub-ventanas (hasta ~4000 chars cada una)
    con el pipeline COMPLETO de coref en CPU, que es muy pesado en RAM. Un lote de
    16 sub-ventanas grandes puede provocar OOM (el proceso muere con SIGKILL, sin
    traceback). Para documentos grandes se encoge el lote:

    * > 200k chars pendientes → lote ≤ 4
    * > 100k chars pendientes → lote ≤ 8
    * en otro caso → lote sin cambios

    Es una función pura para poder probarse sin importar ``segmentador`` (que
    arrastra torch/spacy/stanza).
    """
    if total_pending_chars > 200_000:
        return min(base_bs, 4)
    if total_pending_chars > 100_000:
        return min(base_bs, 8)
    return base_bs


def _coref_code_fingerprint() -> str:
    """Huella del algoritmo de extracción de cadenas de coref.

    Hashea el archivo fuente de ``segmentador.py`` (donde vive
    ``ProgressiveSegmenter``). Se usa el archivo en vez de ``inspect.getsource``
    para no importar ``segmentador`` (que arrastra torch/spacy/stanza) y poder
    probar este módulo aislado. Si cambia el algoritmo (p. ej. el batching), la
    huella cambia y los checkpoints viejos se invalidan automáticamente.
    """
    try:
        spec = importlib.util.find_spec(_COREF_SOURCE_MODULE)
        if spec is None or spec.origin is None:
            return "unknown"
        src = Path(spec.origin).read_text(encoding="utf-8")
        return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "unknown"


def _cargar_checkpoint_coref(
    checkpoint_path: str,
    context_units: Optional[int] = None,
    min_forms: Optional[int] = None,
) -> Optional[Dict]:
    """Carga un checkpoint de ``_clasificar_umbral_coref``.

    Devuelve None si el archivo no existe, está corrupto, tiene una versión
    desconocida, fue generado con otros parámetros (context_units/min_forms)
    o con otra versión del algoritmo (code_fingerprint). En todos esos casos
    se empieza de cero, sin abortar el pipeline.
    """
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("version") != COREF_CHECKPOINT_VERSION:
            logger.warning(
                "Checkpoint con versión desconocida (%s) — ignorado.",
                data.get("version"),
            )
            return None
        if context_units is not None and data.get("context_units") != context_units:
            logger.warning(
                "Checkpoint con context_units distinto (%s vs %s) — ignorado.",
                data.get("context_units"),
                context_units,
            )
            return None
        if min_forms is not None and data.get("min_forms") != min_forms:
            logger.warning(
                "Checkpoint con min_forms distinto (%s vs %s) — ignorado.",
                data.get("min_forms"),
                min_forms,
            )
            return None
        if data.get("code_fingerprint") != _coref_code_fingerprint():
            logger.warning(
                "Checkpoint con algoritmo distinto (code_fingerprint) — ignorado."
            )
            return None
        return data
    except Exception as e:
        logger.warning("No se pudo cargar el checkpoint %s: %s", checkpoint_path, e)
        return None


def _guardar_checkpoint_coref(checkpoint_path: str, state: Dict) -> None:
    """Escribe el checkpoint de forma atómica (tmp + os.replace).

    Un fallo de escritura solo se registra en el log: nunca debe romper el
    pipeline de clasificación.
    """
    try:
        d = os.path.dirname(checkpoint_path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp_path = f"{checkpoint_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, cls=_NumpyEncoder)
        os.replace(tmp_path, checkpoint_path)
    except Exception as e:
        logger.warning("No se pudo escribir el checkpoint %s: %s", checkpoint_path, e)
