"""Carga modular del pipeline de Stanza (descarga + fallback GPU→CPU).

Extraído de ``ProgressiveSegmenter.get_stanza`` para poder reutilizarlo (p. ej.
en el servidor FastAPI de Colab) y probarlo aislado sin importar ``segmentador``
(que arrastra torch/spacy/sentence_transformers).

``stanza`` se importa de forma perezosa (la primera vez que se construye un
pipeline), así que importar este módulo es rápido y no arrastra stanza/torch.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Procesadores del pipeline COMPLETO de coref (el que usa ``_extract_global_chains``).
# El subconjunto de sujetos raíz vive en ``reii.seg.root_cache.ROOT_PROCESSORS``.
COREF_PROCESSORS = "tokenize,pos,lemma,depparse,constituency,coref"

_stanza = None


def _get_stanza_module():
    """Importa ``stanza`` de forma perezosa (la primera vez) y lo cachea.

    Mantiene el módulo ligero de importar y permite inyectar un falso en los
    tests (parcheando esta función).
    """
    global _stanza
    if _stanza is None:
        import stanza

        _stanza = stanza
    return _stanza


def build_stanza_pipeline(
    lang: str = "es",
    processors: str = COREF_PROCESSORS,
    use_gpu: bool = False,
) -> Optional[object]:
    """Descarga (si falta) y construye un pipeline de Stanza con fallback GPU→CPU.

    Devuelve None si la carga falla (p. ej. sin red para descargar el modelo).
    Los mensajes de progreso se imprimen con el prefijo ``[COREF]`` para que
    aparezcan en consola igual que antes de la extracción.
    """
    stanza = _get_stanza_module()
    try:
        stanza.download(lang, processors=processors, verbose=False)
        # GPU si hay CUDA; si falla (p. ej. VRAM compartida con SentenceTransformer)
        # se reintenta en CPU en vez de desactivar COREF.
        for gpu in [True, False] if use_gpu else [False]:
            try:
                pipe = stanza.Pipeline(
                    lang, processors=processors, use_gpu=gpu, verbose=False
                )
                print(f"[COREF] ✓ Stanza pipeline cargado ({'GPU' if gpu else 'CPU'}).")
                return pipe
            except Exception as gpu_err:
                if not gpu:
                    raise
                print(
                    f"[COREF] ✗ Stanza en GPU falló ({gpu_err}); reintentando en CPU."
                )
    except Exception as e:
        print(f"[COREF] ✗ Error cargando Stanza: {e}. Correferencias deshabilitadas.")
        logger.warning("Error cargando Stanza: %s", e)
    return None
