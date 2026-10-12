"""Sujetos raíz de la etapa COREF: qué UCEs consultar y una caché compartida.

Sin dependencias pesadas (no importa stanza/torch/spacy) para poder probarse aislado.

Contexto: ``_clasificar_umbral_coref`` fusiona grupos pre-mezclados con un recorrido
voraz que, en cada frontera i-1 -> i, consulta los sujetos raíz de la ÚLTIMA UCE del
grupo i-1 (``current_group[-1]``). Por tanto solo hacen falta las colas de todos los
grupos salvo el último, y su texto es idéntico en los pases B1 y B2.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, TypeVar

logger = logging.getLogger(__name__)

ROOT_CTX_CHARS = 500  # misma ventana que ``van_unidos`` (últimos 500 caracteres)
# Procesadores que hacen falta para los sujetos raíz (dependencias + constituyentes).
# ``coref`` NO: ejecutarlo para esto era trabajo tirado. ``Pipeline.process`` acepta un
# subconjunto de procesadores sobre el pipeline YA cargado, así que no se duplican modelos.
ROOT_PROCESSORS = "tokenize,mwt,pos,lemma,depparse,constituency"
T = TypeVar("T")


def root_context(text: str) -> str:
    """Cola del texto que se pasa a Stanza para extraer sujetos raíz."""
    return text[-ROOT_CTX_CHARS:] if len(text) > ROOT_CTX_CHARS else text


def boundary_tails(groups: Sequence[Sequence[T]]) -> List[T]:
    """Última UCE de cada grupo pre-mezclado salvo el último (los grupos vacíos se ignoran).

    Son las únicas UCEs cuyos sujetos raíz consulta el recorrido voraz: al fusionar,
    ``current_group.extend(next_group)`` deja como cola la cola del siguiente grupo, y
    al no fusionar, ``current_group = next_group``; en ambos casos
    ``current_group[-1]`` es la cola de algún grupo previo a la frontera.
    """
    return [g[-1] for g in groups[:-1] if g]


def subjects_from_doc(doc) -> List[str]:
    """Sujetos raíz (SN completos) de un documento ya procesado por Stanza.

    Es el cuerpo de ``ProgressiveSegmenter.find_subjects_for_roots`` movido tal cual,
    para poder reutilizarlo con una llamada individual o con una por lotes. Con
    desajustes de índices (MWT) o sin árbol se queda con la palabra suelta.
    """
    subjects = []
    for sentence in doc.sentences:
        tree = sentence.constituency
        root_ids = [word.id for word in sentence.words if word.head == 0]

        for word in sentence.words:
            if word.head in root_ids and "nsubj" in word.deprel:
                phrase_found = False
                try:
                    # 1. Buscar la «caja» del SN (word.id es 1-based, el árbol 0-based)
                    leaf = tree.get_leaf_for_index(word.id - 1)
                    curr = leaf
                    while curr.parent is not None:
                        if curr.label == "NP":
                            subjects.append(" ".join(curr.leaf_labels()))
                            phrase_found = True
                            break
                        curr = curr.parent
                except Exception:
                    # índice fuera de rango por MWT, árbol ausente, etc.
                    phrase_found = False

                # 2. Respaldo: si el recorrido del árbol falló, la palabra cruda
                if not phrase_found:
                    subjects.append(word.text)

    return list(set([s.strip() for s in subjects if s]))


class StanzaRootExtractor:
    """Sujetos raíz con Stanza: llamada individual o por lotes, sin ejecutar coref.

    get_pipeline : callable sin argumentos que devuelve el pipeline cargado (o None).
    skip_coref   : pasa ``processors=ROOT_PROCESSORS``. Si la versión de Stanza no
                   admite ese argumento (TypeError) vuelve, una vez, al pipeline completo.

    No importa stanza: recibe el pipeline, así que se puede probar con uno falso.
    """

    def __init__(self, get_pipeline: Callable[[], object], skip_coref: bool = True):
        self._get = get_pipeline
        self.skip_coref = bool(skip_coref)

    def _run(self, pipe, inp, bulk: bool):
        fn = pipe.bulk_process if bulk else pipe
        kw = {"processors": ROOT_PROCESSORS} if self.skip_coref else {}
        try:
            return fn(inp, **kw)
        except TypeError as e:
            if not kw:
                raise
            # Visible en consola: si esto ocurre, el coref se ejecuta de más y
            # cada raíz pasa a tardar ~7s en vez de ~0.4s.
            print(
                f"   [COREF] ⚠ Stanza no admite `processors=` ({e}); "
                "sujetos raíz con el pipeline COMPLETO (coref incluido)."
            )
            logger.warning(
                "Stanza no admite `processors=` (%s): sujetos raíz con el pipeline completo",
                e,
            )
            self.skip_coref = False
            return fn(inp)

    def single(self, text: str) -> List[str]:
        """Una llamada. Cualquier fallo = sin sujetos (igual que antes)."""
        try:
            pipe = self._get()
            if not pipe:
                return []
            return subjects_from_doc(self._run(pipe, text, bulk=False))
        except Exception as e:
            logger.error("[COREF] Subject extraction  failure: %s", e)
            return []

    def batch(self, texts: Sequence[str]) -> List[List[str]]:
        """Un solo ``bulk_process`` para todos los textos no vacíos (mismo orden).

        Lanza si algo falla: ``RootSubjectCache`` reintenta ese lote uno a uno.
        """
        out: List[List[str]] = [[] for _ in texts]
        pipe = self._get()
        if not pipe:
            return out
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if not idx:
            return out
        docs = self._run(pipe, [texts[i] for i in idx], bulk=True)
        if len(docs) != len(idx):
            raise ValueError(
                f"bulk_process devolvió {len(docs)} documentos para {len(idx)} textos"
            )
        for i, d in zip(idx, docs):
            out[i] = subjects_from_doc(d)
        return out


class RootSubjectCache:
    """Caché {contexto -> sujetos raíz} compartida entre pases y UCEs repetidas.

    extractor       : texto -> iterable de sujetos (p. ej. ``find_subjects_for_roots``).
    batch_extractor : opcional, lista de textos -> lista de iterables (mismo orden).
                      Si falla, se vuelve al extractor individual.
    """

    def __init__(
        self,
        extractor: Callable[[str], Iterable[str]],
        batch_extractor: Optional[Callable[[List[str]], List[Iterable[str]]]] = None,
        batch_size: int = 16,
        max_batch_failures: int = 3,
    ):
        self._extract = extractor
        self._batch = batch_extractor
        self._batch_size = max(1, int(batch_size))
        self._max_batch_failures = max(1, int(max_batch_failures))
        self._batch_failures = 0  # fallos de lote consecutivos
        self._store: Dict[str, frozenset] = {}
        self.hits = 0
        self.misses = 0  # extracciones reales (llamadas a Stanza)
        # Instrumentación B1: cuántos textos se resolvieron por lote vs individual.
        self.batch_texts = 0
        self.single_texts = 0

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, text: str) -> bool:
        return text in self._store

    def get(self, text: str) -> Set[str]:
        """Sujetos raíz de ``text`` (ya recortado con :func:`root_context`)."""
        if text in self._store:
            self.hits += 1
            return set(self._store[text])
        try:
            val = frozenset(self._extract(text))
        except Exception as e:  # igual que antes: un fallo = sin sujetos, sin abortar
            logger.debug("raíces: fallo extrayendo (%d chars): %s", len(text), e)
            return set()
        self.misses += 1
        self._store[text] = val
        return set(val)

    def prefetch(
        self, contexts: Iterable[str], progress_every: int = 200, label: str = ""
    ) -> int:
        """Calcula las raíces que falten (sin repetir textos). Devuelve cuántas calculó."""
        pending = [c for c in dict.fromkeys(contexts) if c not in self._store]
        if not pending:
            return 0
        t0, done = time.time(), 0

        def _report() -> None:
            if (
                progress_every
                and done
                and (done % progress_every == 0 or done == len(pending))
            ):
                el = time.time() - t0
                print(
                    f"   [COREF] raíces {label}{done}/{len(pending)} "
                    f"({el:.0f}s, {done / max(el, 1e-9):.1f}/s)"
                )

        i = 0
        while i < len(pending):
            chunk = (
                pending[i : i + self._batch_size] if self._batch else pending[i : i + 1]
            )
            if self._batch and len(chunk) > 1:
                try:
                    out = self._batch(chunk)
                    if len(out) != len(chunk):
                        raise ValueError("el lote devolvió una longitud distinta")
                    for text, subj in zip(chunk, out):
                        self._store[text] = frozenset(subj)
                        self.misses += 1
                    self.batch_texts += len(chunk)
                    self._batch_failures = 0
                    i += len(chunk)
                    done += len(chunk)
                    _report()
                    continue
                except Exception as e:
                    # Un texto problemático no debe apagar el batching: este lote va uno a
                    # uno y solo se desactiva tras ``max_batch_failures`` fallos seguidos.
                    self._batch_failures += 1
                    # Visible en consola: si esto se repite, el batching está cayendo
                    # a llamadas individuales (cada una ~7s en vez de ~0.4s).
                    print(
                        f"   [COREF] ⚠ lote de raíces falló ({e}); "
                        f"se procesa una a una (fallo {self._batch_failures}/"
                        f"{self._max_batch_failures})."
                    )
                    logger.warning("raíces: lote falló (%s); se procesa una a una", e)
                    if self._batch_failures >= self._max_batch_failures:
                        print(
                            "   [COREF] ⚠ batching de raíces DESACTIVADO tras "
                            f"{self._batch_failures} fallos seguidos."
                        )
                        logger.warning(
                            "raíces: batching desactivado tras %d fallos",
                            self._batch_failures,
                        )
                        self._batch = None
            for text in chunk:
                self.get(text)
            self.single_texts += len(chunk)
            i += len(chunk)
            done += len(chunk)
            _report()
        # Resumen B1: ratio lote vs individual, para diagnosticar la lentitud.
        if self.batch_texts or self.single_texts:
            print(
                f"   [COREF] raíces {label}resumen: {self.batch_texts} por lote, "
                f"{self.single_texts} individuales."
            )
        return len(pending)

    def stats(self) -> Dict[str, int]:
        return {
            "stanza_calls": self.misses,
            "cache_hits": self.hits,
            "cached": len(self._store),
            "batch_texts": self.batch_texts,
            "single_texts": self.single_texts,
        }
