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
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Versión del FORMATO del checkpoint de coref. Búmpala solo cuando cambie la
# estructura del JSON (no cuando cambie el algoritmo: para eso está la huella).
COREF_CHECKPOINT_VERSION = 3  # v3: añade input_fingerprint

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


def uces_fingerprint(uces_por_doc: Iterable[Iterable[Any]], extra: Optional[Dict] = None) -> str:
    """Huella del CONTENIDO de entrada del pase de coref.

    El checkpoint guarda UCs ya construidos, pero hasta ahora solo validaba
    (version, context_units, min_forms, código). Un checkpoint de otro corpus, de
    otro ``uce_target_size`` o de otra segmentación se reanudaba en silencio con la
    misma ruta. Esta huella cubre (doc_id, id y texto de cada UCE) + ``extra``
    (parámetros del pre-merge). Duck-typing: cualquier objeto con
    ``doc_id``/``id``/``texto``.
    """
    h = hashlib.sha256()
    for doc in uces_por_doc:
        for u in doc:
            h.update(f"{getattr(u, 'doc_id', '')}\x1f{getattr(u, 'id', '')}\x1f".encode("utf-8"))
            h.update((getattr(u, "texto", "") or "").encode("utf-8"))
            h.update(b"\x1e")
        h.update(b"\x1d")
    if extra:
        h.update(json.dumps(extra, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:16]


class CheckpointGuard:
    """Estado de salud del checkpointing de UN pase (lógica pura, sin I/O).

    * Un doc con extracción de cadenas INCOMPLETA no se persiste como completo y
      CONGELA el checkpoint: los UCs de ese doc ya están en la lista acumulada, así
      que guardar más docs dejaría un estado inconsistente (UCs sin su doc en
      processed_doc_ids → duplicados al reanudar). Congelado, el archivo conserva
      el último prefijo consistente y la reanudación rehace desde ese doc.
    * Cuenta fallos de escritura/serialización para informar al final (antes solo
      ``logger.warning`` y el log decía "Checkpoint guardado" aunque fallara).
    """

    def __init__(self) -> None:
        self.frozen_at: Any = None
        self.write_failures = 0
        self.build_failures = 0
        self.incomplete_docs: List[Any] = []

    @property
    def can_save(self) -> bool:
        return self.frozen_at is None

    def note_incomplete(self, doc_id: Any) -> bool:
        """Registra un doc incompleto. True si este doc es el que congela."""
        self.incomplete_docs.append(doc_id)
        if self.frozen_at is None:
            self.frozen_at = doc_id
            return True
        return False

    def summary(self) -> List[str]:
        out: List[str] = []
        if self.incomplete_docs:
            out.append(
                f"{len(self.incomplete_docs)} doc(s) con extracción de coref INCOMPLETA "
                f"(primero: {self.incomplete_docs[0]}); checkpoint congelado en ese punto."
            )
        if self.write_failures:
            out.append(f"{self.write_failures} fallo(s) al ESCRIBIR el checkpoint.")
        if self.build_failures:
            out.append(f"{self.build_failures} fallo(s) al CONSTRUIR el estado del checkpoint.")
        return out


def _cargar_checkpoint_coref(
    checkpoint_path: str,
    context_units: Optional[int] = None,
    min_forms: Optional[int] = None,
    input_fingerprint: Optional[str] = None,
    reason_sink: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Carga un checkpoint de ``_clasificar_umbral_coref``.

    Devuelve None si el archivo no existe, está corrupto, tiene una versión
    desconocida, fue generado con otros parámetros (context_units/min_forms)
    o con otra versión del algoritmo (code_fingerprint). En todos esos casos
    se empieza de cero, sin abortar el pipeline.

    ``input_fingerprint`` (ver :func:`uces_fingerprint`) valida el contenido de
    entrada; ``reason_sink`` recibe el motivo del rechazo para que el llamador lo
    muestre con print (el logger puede no estar configurado).
    """
    def _reject(msg: str) -> None:
        logger.warning(msg)
        if reason_sink is not None:
            reason_sink.append(msg)

    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("version") != COREF_CHECKPOINT_VERSION:
            _reject(f"Checkpoint con versión desconocida ({data.get('version')}) — ignorado.")
            return None
        if context_units is not None and data.get("context_units") != context_units:
            _reject(
                f"Checkpoint con context_units distinto ({data.get('context_units')} "
                f"vs {context_units}) — ignorado."
            )
            return None
        if min_forms is not None and data.get("min_forms") != min_forms:
            _reject(
                f"Checkpoint con min_forms distinto ({data.get('min_forms')} "
                f"vs {min_forms}) — ignorado."
            )
            return None
        if data.get("code_fingerprint") != _coref_code_fingerprint():
            _reject("Checkpoint con algoritmo distinto (code_fingerprint) — ignorado.")
            return None
        if input_fingerprint is not None and data.get("input_fingerprint") != input_fingerprint:
            _reject(
                "Checkpoint de OTRA entrada (corpus / uce_target_size / pre-merge "
                "distintos: input_fingerprint no coincide) — ignorado."
            )
            return None
        return data
    except Exception as e:
        logger.warning("No se pudo cargar el checkpoint %s: %s", checkpoint_path, e)
        return None


def _guardar_checkpoint_coref(checkpoint_path: str, state: Dict) -> bool:
    """Escribe el checkpoint de forma atómica (tmp + os.replace).

    Un fallo de escritura no rompe el pipeline, pero ahora DEVUELVE False para que
    el llamador no anuncie "guardado" (antes siempre lo hacía, aunque fallara).
    """
    try:
        d = os.path.dirname(checkpoint_path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp_path = f"{checkpoint_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, cls=_NumpyEncoder)
        os.replace(tmp_path, checkpoint_path)
        return True
    except Exception as e:
        logger.warning("No se pudo escribir el checkpoint %s: %s", checkpoint_path, e)
        return False
