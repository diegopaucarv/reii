# gramatical_analyzer.py
# %%viztracer
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de análisis gramatical, semántico y pragmático — v3

FALTA: ML para clasificación de Aspecto verbal, Tipo de negación, Función pragmática de pronombres,
FALTA: Ajustar umbrales (recomendación empírica). Los valores tech_ratio > 0.15, oral_ratio > 0.4, academic_ratio > 0.3 son punto de partida. Puedes calibrarlos observando los resultados en tu corpus. Si tienes anotaciones manuales, calcula las curvas ROC para optimizar.

"""

from __future__ import annotations

import bisect
import dataclasses
import gc
import json
import logging
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from gliner import GLiNER

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import spacy
import stanza
import torch
from cdlib import algorithms
from gensim.models import fasttext
from nltk.corpus import wordnet as wn
from rapidfuzz import fuzz
from scipy.spatial.distance import cdist
from scipy.stats import entropy as scipy_entropy
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from spacy.language import Language
from spacy.matcher import DependencyMatcher, PhraseMatcher
from spacy.tokens import Doc, Span, Token
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from lang.es import (
    ADVERB_BLOCKED,
    ADVERB_CATEGORIES,
    ALL_KNOWN_ADVERBS,
    AMBIGUOUS_SUBORDINATORS,
    CONECTORES_DISC_ADV,
    DOMINIO_EXTRA_MULTIPLIER,
    EXPRESIONES_IDIOMATICAS_NPI,
    INSUBORDINACION_DEFAULT_FUNCION,
    INSUBORDINACION_FUNCIONES,
    LEXICON_ADVERBS,
    LEXICON_MULTIPLIER,
    LOCUCIONES_DISCURSIVAS,
    MANUAL_TRAINING_EXAMPLES,
    MULTI_WORD_ADVERBS,
    MULTI_WORD_MULTIPLIER,
    NON_FINITE_FORMS,
    NON_REFERENTIAL,
    NPI_WORDS,
    # Datos
    PALABRAS_NEGATIVAS,
    RAREZAS_PATTERNS,
    SUBORDINATING_DEPS,
    TRAINING_ADJECTIVES,
    TRAINING_OBJECTS,
    TRAINING_SUBJECTS,
    TRAINING_TEMPLATES,
    TRAINING_VERBS,
    MorphDeriver,
    PredicateFrame,
    SpanAnnotation,
    build_adverb_phrase_matcher,
    build_discourse_matcher,
    clasificar_pronombre_explicito,
    correct_pronoun_dependency,
    corregir_lema_para_clitico,
    detect_contraction,
    es_impersonal_se,
    es_media_se,
    es_pasiva_refleja,
    extraer_enclitico,
    get_negation_scope,
    get_subordination_type,
    is_periphrastic_construction,
    is_prodrop_verb,
    tipo_cuantificador,
    tipo_negacion,
    uce_to_global_annotations,
    verificar_concordancia_participio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Stanza coref bug-fix (se mantiene igual) ─────────────────────────────────
try:
    from stanza.models.coref.config import Config as _StanzaCorefCfg

    if not hasattr(_StanzaCorefCfg, "_coref_patched_flag"):
        _StanzaCorefCfg._coref_original_init = _StanzaCorefCfg.__init__

        def _coref_patched_init(self, *args, **kwargs):
            kwargs.setdefault("plateau_epochs", 10)
            _StanzaCorefCfg._coref_original_init(self, *args, **kwargs)

        _StanzaCorefCfg.__init__ = _coref_patched_init
        _StanzaCorefCfg._coref_patched_flag = True
        logger.info("Stanza coref Config parcheado correctamente.")
except ImportError:
    pass

logger = logging.getLogger(__name__)


def lock_global_offsets(texto_completo: str, uces: list) -> None:
    cursor = 0
    for uce in uces:
        if not uce.texto.strip():
            continue

        # Cascade of attempts: tighter → looser window and block size
        attempts = [
            (max(500, len(uce.texto) * 3), 4),  # tight: original behavior
            (max(1000, len(uce.texto) * 6), 3),  # wider window, smaller block
            (max(2000, len(uce.texto) * 10), 2),  # very wide, minimal block
        ]

        aligned = False
        for window_size, min_block in attempts:
            chunk = texto_completo[cursor : cursor + window_size]
            matcher = SequenceMatcher(None, chunk.lower(), uce.texto.lower())
            valid_blocks = [
                b for b in matcher.get_matching_blocks() if b.size >= min_block
            ]

            if valid_blocks:
                start_offset = valid_blocks[0].a
                end_offset = valid_blocks[-1].a + valid_blocks[-1].size
                uce.start_char = cursor + start_offset
                uce.end_char = cursor + end_offset
                cursor = uce.end_char
                aligned = True
                break

        if not aligned:
            logger.error(
                "DESYNC: alignment failed for UCE %s after %d attempts. "
                "Cursor=%d, UCE length=%d. Falling back to cursor position.",
                uce.id,
                len(attempts),
                cursor,
                len(uce.texto),
            )
            # Log the first 80 chars of both sides to help diagnose
            logger.debug(
                "  texto_completo[cursor:cursor+200] = %r",
                texto_completo[cursor : cursor + 200],
            )
            logger.debug("  uce.texto[:80] = %r", uce.texto[:80])
            uce.start_char = cursor
            uce.end_char = cursor + len(uce.texto)
            cursor = uce.end_char


@Language.component("fix_colloquial_npi_deps")
def fix_colloquial_npi_deps(doc: Doc) -> Doc:
    """
    Custom pipeline component to fix bad dependency parsing on Spanish NPIs.
    Runs after the parser to surgically reattach floating NPIs to negated verbs.
    """
    npi_words = {"nadie", "nada", "ningún", "ninguno", "ninguna", "nunca", "jamás"}
    negation_markers = {"no", "nunca", "jamás", "tampoco"}

    for token in doc:
        if token.lower_ in npi_words:
            # If the parser attached it to something stupid (not a verb/aux)
            # or if it's just floating as the root of a fragment
            if token.head.pos_ not in ("VERB", "AUX") or token.dep_ == "ROOT":
                nearest_verb = None
                min_dist = float("inf")

                # Hunt for the nearest NEGATED verb in the same sentence
                for word in token.sent:
                    if word.pos_ in ("VERB", "AUX"):
                        # Is this verb actually negated? Look at its children.
                        is_negated = any(
                            c.lower_ in negation_markers for c in word.children
                        )

                        if is_negated:
                            dist = abs(token.i - word.i)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_verb = word

                # Perform the surgery: reassign the head and fix the dependency tag
                if nearest_verb:
                    token.head = nearest_verb
                    # If it's an adverb (nunca, jamás), it's an advmod. Otherwise, usually an object/obl.
                    token.dep_ = "advmod" if token.pos_ == "ADV" else "obj"

    return doc


# ============================================================================
# UTIL
# ============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return super().default(o)


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# str.translate translation table for blazing fast 1-to-1 character swaps.
_QUOTE_TRANS = str.maketrans("“”‘’«»", '""\'\'""')


def normalize_text(
    text: str,
    normalize_unicode: str = "NFC",
    fix_quotes: bool = True,
    strip_bom: bool = True,
    replace_control_chars: bool = True,
) -> str:
    """
    Normaliza texto para procesamiento NLP multilingüe.
    """
    if not text:
        return text

    if strip_bom:
        text = text.lstrip("\ufeff")

    if normalize_unicode:
        text = unicodedata.normalize(normalize_unicode, text)

    if fix_quotes:
        text = text.translate(_QUOTE_TRANS)

    if replace_control_chars:
        text = _CONTROL_CHARS_RE.sub(" ", text)

    return text


def to_global(local_offset: Optional[int], uce_start: int) -> Optional[int]:
    """Converts a UCE-local char offset to global document offset."""
    return (local_offset + uce_start) if local_offset is not None else None


def annotation_to_global(ann: Dict, uce_start: int, fields: List[str]) -> Dict:
    """
    Returns a copy of ann with the given offset fields shifted to global.
    fields: list of field names that are char offsets (e.g. ["char_start","char_end"])
    """
    out = dict(ann)
    for f in fields:
        if f in out and out[f] is not None:
            out[f] = out[f] + uce_start
    return out


class OffsetMapper:
    """Maps between local offsets (within a substring) and global document offsets."""

    def __init__(self, base_global: int):
        self.base = base_global

    def to_global(self, local_start: int, local_end: int) -> Tuple[int, int]:
        return self.base + local_start, self.base + local_end

    def to_local(self, global_start: int, global_end: int) -> Tuple[int, int]:
        return global_start - self.base, global_end - self.base


# ============================================================================
# 0. CONFIG
# ============================================================================


@dataclass
class Config:
    min_tokens_por_uce: int = 30
    max_tokens_por_uce: int = 200
    spacy_model: str = "es_core_news_lg"
    word_embeddings_path: Optional[str] = r"D:\cc.es.300.bin"
    gliner_model: str = "urchade/gliner_multi-v2.1"
    use_gensim_embeddings: bool = True
    gpt2_model: str = "datificate/gpt2-small-spanish"
    sentence_embedder_model: str = "BAAI/bge-m3"
    adverb_classifier_dir: str = "./adverb_model"
    stanza_lang: str = "es"
    stanza_use_gpu: bool = False
    adverb_confidence_threshold: float = 0.5
    adverb_cache_size: int = 10_000
    subtlex_path: Optional[str] = r"D:\SUBTLEX-ESP.xlsx"
    use_subtlex: bool = True
    use_surprisal: bool = False
    use_coref: bool = True
    use_wordnet_quantifiers: bool = True
    max_context_tokens: int = 512
    random_state: int = 42
    db_path: str = "./data/analisis_gramatical.json"
    use_subtlex_analytics: bool = True  # activa cálculos avanzados de SUBTLEX
    subtlex_analytics_cache: bool = True  # cachea resultados por UCE (evita recalc)
    if stanza_use_gpu:
        device = "cpu"
    else:
        device = "cuda"


# ============================================================================
# 1. ESTRUCTURAS DE DATOS
# ============================================================================


@dataclass
class UCE:
    # ---------- Core identifiers ----------
    id: str
    texto: str = ""
    doc_id: Optional[int] = None  # from ALCESTE (numeric)
    uc_id: Optional[str] = None  # ID of parent UC (ALCESTE)
    local_idx: Optional[int] = None
    seccion: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)  # ← conservar

    # ---------- Offsets (for coref & dashboard) ----------
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    # ---------- Basic linguistic data ----------
    tokens: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    pos_tags: List[str] = field(default_factory=list)
    stems: List[str] = field(default_factory=list)  # ALCESTE
    content_lemmas: List[str] = field(default_factory=list)  # grammatical analyzer
    formas_tokens: List[Dict[str, str]] = field(default_factory=list)  # ALCESTE

    # ---------- N-grams (ALCESTE) ----------
    bigrams: List[Tuple[str, str]] = field(default_factory=list)
    bigram_stems: List[Tuple[str, str]] = field(default_factory=list)
    trigrams: List[Tuple[str, str, str]] = field(default_factory=list)
    trigram_stems: List[Tuple[str, str, str]] = field(default_factory=list)

    # ---------- ALCESTE clustering results ----------
    cluster_id: Optional[int] = None
    is_stable: bool = False
    is_ambiguous: bool = False
    is_top: bool = False
    stability: Optional[float] = None
    phi_coefficients: Dict[str, float] = field(default_factory=dict)
    coordinates: Dict[str, List[float]] = field(
        default_factory=dict
    )  # e.g. {'afc_row': [...]}

    # ---------- Grammatical annotations (enriched by pipeline) ----------
    negaciones: List[Dict] = field(default_factory=list)
    pronombres: List[Dict] = field(default_factory=list)
    verbos: List[Dict] = field(default_factory=list)
    cuantificadores: List[Dict] = field(default_factory=list)
    adverbios: List[Dict] = field(default_factory=list)
    marcadores_discursivos: List[Dict] = field(default_factory=list)
    marcadores: List[Dict[str, str]] = field(
        default_factory=list
    )  # legacy ALCESTE field (kept)
    insubordinaciones: List[Dict] = field(default_factory=list)
    rarezas: List[Dict] = field(default_factory=list)
    entidades: Dict[str, List[Dict]] = field(default_factory=dict)

    # ---------- Complexity & lexical metrics ----------
    complejidad_sintactica: Dict = field(default_factory=dict)
    metricas_lexicas: Dict = field(default_factory=dict)
    diversidad_semantica: float = 0.0
    topic_shift_prev: float = 0.0
    token_surprisals: Dict[int, float] = field(default_factory=dict)

    # ---------- Coreference & predicates ----------
    coref_chains: List[Dict] = field(default_factory=list)
    predicate_frames: List[Dict] = field(
        default_factory=list
    )  # serialized PredicateFrame objects
    frame_annotations: List = field(
        default_factory=list
    )  # SpanAnnotation objects (local)
    predicate_analysis: Dict = field(
        default_factory=dict
    )  # CorefPredicateResult summary
    predicate_summary: str = ""
    verbal_frames: List[Dict] = field(default_factory=list)

    # ---------- Register & embeddings ----------
    registro: Optional[str] = None
    embedding: Optional[List[float]] = None

    # ---------- Internal temporary (not serialized) ----------
    span: Optional[Any] = field(default=None, repr=False)  # spaCy Span (grammatical)
    _coref_chains_full: Optional[List[Dict]] = field(
        default=None, repr=False
    )  # full chains
    _predicate_frames_serialized: Optional[List[Dict]] = field(default=None, repr=False)

    discourse_annotations: List[Dict] = field(default_factory=list)

    @property
    def offset_mapper(self) -> "OffsetMapper":
        return OffsetMapper(self.start_char or 0)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        """Convert to JSON‑serializable dict."""
        d = asdict(self)

        # Remove non‑serializable fields
        d.pop("span", None)
        d.pop("_coref_chains_full", None)
        d.pop("_predicate_frames_serialized", None)

        # Convert numpy arrays to lists
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()

        # Special handling for predicate_frames (already dicts)
        if (
            hasattr(self, "_predicate_frames_serialized")
            and self._predicate_frames_serialized
        ):
            d["predicate_frames"] = self._predicate_frames_serialized

        # Ensure coref_chains are plain dicts (they already are)
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "UCE":
        """Reconstruct UCE from a dict (e.g., loaded from JSON)."""
        # Get the set of known field names
        known = {f.name for f in dataclasses.fields(cls)}

        # Provide defaults for missing list fields (old data)
        for f in (
            "bigrams",
            "bigram_stems",
            "trigrams",
            "trigram_stems",
            "tokens",
            "lemmas",
            "pos_tags",
            "stems",
            "content_lemmas",
            "formas_tokens",
            "marcadores",
            "negaciones",
            "pronombres",
            "verbos",
            "cuantificadores",
            "adverbios",
            "entidades",
            "marcadores_discursivos",
            "insubordinaciones",
            "rarezas",
            "coref_chains",
            "predicate_frames",
            "frame_annotations",
            "verbal_frames",
        ):
            data.setdefault(f, [])

        # Restore integer keys in token_surprisals
        if "token_surprisals" in data and isinstance(data["token_surprisals"], dict):
            data["token_surprisals"] = {
                int(k): v for k, v in data["token_surprisals"].items()
            }

        # Filter only known fields (ignore extra keys from old schemas)
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class UC:
    id: str
    uce_ids: List[str] = field(default_factory=list)
    texto: str = ""
    lemmas: List[str] = field(default_factory=list)
    stems: List[str] = field(default_factory=list)


# ============================================================================
# 3. MÓDULOS AUXILIARES (SubtlexESP, WordEmbeddingsAnalyzer, SurprisalCalculator,
#    CoreferenceResolver) se mantienen igual, solo se corrige el bug del del self.nlp
# ============================================================================


class WordEmbeddingsAnalyzer:
    """Vectores de palabras: spaCy vectors o Gensim fastText."""

    def __init__(self, nlp: Language, config: Config):
        self.nlp = nlp
        self.use_gensim = config.use_gensim_embeddings and bool(
            config.word_embeddings_path
        )
        self.vector_size = nlp.vocab.vectors_length if not self.use_gensim else 300
        if self.use_gensim:
            try:
                self.wv = fasttext.load_facebook_vectors(config.word_embeddings_path)
                self.vector_size = self.wv.vector_size
                logger.info(
                    f"Word embeddings cargados desde {config.word_embeddings_path}"
                )
            except Exception as e:
                logger.warning(
                    f"No se pudo cargar embeddings Gensim: {e}. Usando spaCy."
                )
                self.use_gensim = False

    def vector(self, word: str) -> np.ndarray:
        word_lower = word.lower()
        if self.use_gensim:
            try:
                # fasttext wv[] usually returns a numpy array, but you can wrap this too if needed
                return np.asarray(self.wv[word_lower])
            except KeyError:
                return np.zeros(self.vector_size)

        token = self.nlp.vocab[word_lower]
        # Wrap token.vector in np.asarray() to satisfy the type checker
        return (
            np.asarray(token.vector) if token.has_vector else np.zeros(self.vector_size)
        )

    def contextual_word_embedding(
        self,
        span_or_token: Union[Span, Token],
        window: int = 2,
    ) -> np.ndarray:
        if isinstance(span_or_token, Token):
            doc = span_or_token.doc
            start, end = span_or_token.i, span_or_token.i + 1
        else:
            doc = span_or_token.doc
            start, end = span_or_token.start, span_or_token.end

        left = [self.vector(w.text) for w in doc[max(0, start - window) : start]]
        right = [self.vector(w.text) for w in doc[end : min(len(doc), end + window)]]
        center = [self.vector(w.text) for w in doc[start:end]]

        lv = np.mean(left, axis=0) if left else np.zeros(self.vector_size)
        rv = np.mean(right, axis=0) if right else np.zeros(self.vector_size)
        cv = np.mean(center, axis=0) if center else np.zeros(self.vector_size)
        return np.concatenate([lv, cv, rv])


class SurprisalCalculator:
    """Surprisal contextual con GPT-2 y alineamiento por caracteres."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def compute_surprisal(self, context: str, target: str) -> Tuple[float, float]:
        full_text = (context + " " + target) if context else target
        encoding = self.tokenizer(
            full_text, return_offsets_mapping=True, return_tensors="pt"
        )
        input_ids = encoding.input_ids.to(self.device)
        offset_mapping = encoding.offset_mapping[0].cpu().numpy()
        with torch.no_grad():
            logits = self.model(input_ids).logits[0]
        surprisals = []
        for i in range(len(input_ids[0]) - 1):
            nid = input_ids[0, i + 1].item()
            prob = torch.softmax(logits[i], dim=-1)[nid].item()
            s, e = offset_mapping[i + 1]
            surprisals.append((s, e, -np.log2(prob + 1e-10)))
        ctx_len = len(context) + 1 if context else 0
        target_surprisals = [v for (s, _, v) in surprisals if s >= ctx_len]
        if not target_surprisals:
            return 0.0, 0.0
        return target_surprisals[0], float(np.mean(target_surprisals[1:])) if len(
            target_surprisals
        ) > 1 else 0.0


class CoreferenceResolver:
    """
    Resolves coreferences using Stanza, operating on exact slices of the original document.
    No concatenation – uses character‑range windows to preserve offsets perfectly.
    """

    def __init__(self, stanza_pipeline, context_units: int = 2):
        """
        Args:
            stanza_pipeline: Stanza pipeline with coref processor.
            context_units: Number of UCEs to keep as context on each side.
        """
        self.nlp = stanza_pipeline
        self.context_units = context_units
        self._global_chains = []  # all chains across the whole document
        self._buffer = []  # list of (global_start, global_end) of buffered UCEs

    def reset(self) -> None:
        """Clear all accumulated chains and buffer – call before processing a new document."""
        self._global_chains = []
        self._buffer = []

    def _insert_mention_sorted(self, mentions: List[Dict], new_mention: Dict) -> None:
        """Insert a mention while keeping the list sorted by start_char."""
        key = (new_mention["start_char"], new_mention["end_char"])
        # Binary search for insertion point
        lo = bisect.bisect_left(
            [(m["start_char"], m["end_char"]) for m in mentions], key
        )
        # Avoid duplicates (same start and end)
        if (
            lo < len(mentions)
            and mentions[lo]["start_char"] == new_mention["start_char"]
        ):
            return
        mentions.insert(lo, new_mention)

    def resolve(
        self, full_doc_text: str, segments: List[Tuple[str, int]]
    ) -> List[Dict]:
        """
        Process all UCEs of a document and return global coreference chains.

        Args:
            full_doc_text: The complete original document text (string).
            segments: List of (uce_text, global_start_char) for each UCE in order.
                      The end char is computed as start + len(text).

        Returns:
            List of chains, each containing:
                - "representative": str (canonical mention)
                - "mentions": list of {"text", "start_char", "end_char"} (global offsets)
        """
        self.reset()
        for uce_text, uce_start in segments:
            uce_end = uce_start + len(uce_text)
            self._resolve_uce(full_doc_text, uce_text, uce_start, uce_end)
        return self._global_chains

    def _resolve_uce(
        self, full_doc_text: str, uce_text: str, uce_start: int, uce_end: int
    ) -> None:
        """
        Process a single UCE: update buffer, run Stanza on the window, and merge new chains.
        """
        self._buffer.append((uce_start, uce_end))
        if len(self._buffer) > self.context_units + 1:
            self._buffer.pop(0)

        window_start = self._buffer[0][0]
        window_end = self._buffer[-1][1]
        window_text = full_doc_text[window_start:window_end]

        try:
            doc = self.nlp(window_text)
        except Exception as e:
            print(f"Stanza error on window [{window_start}:{window_end}]: {e}")
            return

        window_mapper = OffsetMapper(window_start)
        new_chains = []

        for chain in doc.coref:
            mentions = []
            for mention in chain.mentions:
                # --- 1. Defensively extract integer for sentence index ---
                s_idx = mention.sentence
                if isinstance(s_idx, (tuple, list)):
                    s_idx = s_idx[0]
                sent = doc.sentences[s_idx]

                # --- 2. Defensively extract integers for word indices ---
                sw_idx = mention.start_word
                if isinstance(sw_idx, (tuple, list)):
                    sw_idx = sw_idx[0]

                ew_idx = mention.end_word
                if isinstance(ew_idx, (tuple, list)):
                    ew_idx = ew_idx[-1]

                # --- 3. Locate words and extract character offsets ---
                first_word = sent.words[sw_idx]
                last_word = sent.words[ew_idx - 1]

                # Stanza MWT fix: fallback to the parent Token if the Word lacks offsets
                m_start_local = first_word.start_char
                if m_start_local is None and getattr(first_word, "parent", None):
                    m_start_local = first_word.parent.start_char

                m_end_local = last_word.end_char
                if m_end_local is None and getattr(last_word, "parent", None):
                    m_end_local = last_word.parent.end_char

                # Final defensive net: if it's somehow STILL None, skip to avoid crashing
                if m_start_local is None or m_end_local is None:
                    continue

                m_start_global, m_end_global = window_mapper.to_global(
                    m_start_local, m_end_local
                )

                mentions.append(
                    {
                        "text": window_text[m_start_local:m_end_local],
                        "start_char": m_start_global,
                        "end_char": m_end_global,
                    }
                )
            if mentions:
                new_chains.append(
                    {
                        "representative": chain.representative_text,
                        "mentions": mentions,
                    }
                )

        self._merge_chains(new_chains)

    def _merge_chains(self, new_chains: List[Dict]) -> None:
        for new in new_chains:
            merged = False
            for existing in self._global_chains:
                if self._same_entity(new, existing):
                    # Insert each new mention using sorted insertion
                    for m in new["mentions"]:
                        self._insert_mention_sorted(existing["mentions"], m)
                    merged = True
                    break
            if not merged:
                # Ensure new chain's mentions are sorted (they should be from Stanza)
                new["mentions"].sort(key=lambda x: x["start_char"])
                self._global_chains.append(new)

    def _normalize_rep(self, rep: str) -> str:
        stop = {"el", "la", "los", "las", "un", "una", "unos", "unas", "y", "de", "a"}
        words = rep.lower().split()
        filtered = [w for w in words if w not in stop]
        return " ".join(filtered)

    def _same_entity(self, a, b):
        norm_a = self._normalize_rep(a["representative"])
        norm_b = self._normalize_rep(b["representative"])
        if not norm_a or not norm_b:
            return (
                a["representative"].lower().strip()
                == b["representative"].lower().strip()
            )
        return fuzz.ratio(norm_a, norm_b) >= 85

    def get_all_chains(self) -> List[Dict]:
        """Return all chains accumulated so far."""
        return self._global_chains


# ------------------------------------------------------------------------
# SubtlexAnalyzer — Extended analyses using SUBTLEX-ESP
# ------------------------------------------------------------------------


@dataclass
class SubtlexAnalyzer:
    """
    Extended lexicometric analyses using SUBTLEX-ESP.
    Wraps a SubtlexESP instance and provides corpus-level analytics.
    Also exposes compute_surprisal for use as surprisal_source.
    """

    OOV_ZIPF = 1.0
    OOV_SUBTLWF = 0.01
    OOV_CD = 0.001
    OOV_SURPRISAL = 20.0
    # Frequency bands based on Zipf scale
    FREQ_BANDS = [
        ("B1_nuclear", 6.0, 9.0),
        ("B2_alta", 5.0, 6.0),
        ("B3_media", 4.0, 5.0),
        ("B4_baja", 3.0, 4.0),
        ("B5_rara_tecnica", 0.0, 3.0),
    ]

    def __init__(self, filepath: str, sep: str = "\t"):
        """
        Args:
            subtlex: An instance of SubtlexESP (already loaded).
        """
        self._data: Dict[str, Dict[str, float]] = {}
        self._load(filepath, sep)
        logger.info(
            f"SUBTLEX-ESP cargado: {len(self._data):,} entradas desde {filepath}"
        )

    # ------------------------------------------------------------------
    #  Carga del archivo (antes en SubtlexESP)
    # ------------------------------------------------------------------
    def _load(self, filepath: str, sep: str = ","):
        logger.info(f"Cargando SUBTLEX desde {filepath}")
        if not os.path.exists(filepath):
            logger.error(f"Archivo no encontrado: {filepath}")
            return

        # 1. Handle both CSV and Excel gracefully
        try:
            if filepath.lower().endswith(".csv"):
                df_raw = pd.read_csv(filepath, dtype=str, header=None, sep=sep)
            else:
                df_raw = pd.read_excel(
                    filepath, dtype=str, engine="openpyxl", header=None
                )
        except Exception as e:
            logger.error(f"Error leyendo archivo: {e}")
            return

        # 2. Find the header row
        header_row_idx = None

        # Wrap iterrows() in enumerate() to get a strict integer position (pos)
        for pos, (idx, row) in enumerate(df_raw.iterrows()):
            for cell in row:
                if isinstance(cell, str) and cell.strip().lower() == "word":
                    header_row_idx = (
                        pos  # Use the integer position, not the Hashable index
                    )
                    break
            if header_row_idx is not None:
                break

        if header_row_idx is None:
            logger.error("No se encontró la fila de encabezado con 'Word'")
            self._data = {}
            return

        headers = (
            df_raw.iloc[header_row_idx].fillna("").astype(str).str.strip().str.lower()
        )

        # 3. Find ALL instances of the "word" column to handle the side-by-side layout
        word_col_indices = [i for i, val in enumerate(headers) if val == "word"]

        if not word_col_indices:
            logger.error("No se encontraron columnas llamadas 'Word'.")
            self._data = {}
            return

        # Now header_row_idx + 1 works perfectly because the type checker knows it's an int
        data_rows = df_raw.iloc[header_row_idx + 1 :].copy()
        # 4. Loop through each set of columns
        for word_col in word_col_indices:
            col_freqcount = word_col + 1
            col_subtlwf = word_col + 2
            col_logfreq = word_col + 3

            # Ensure we don't go out of bounds if the file is malformed
            if col_logfreq >= len(df_raw.columns):
                continue

            for _, row in data_rows.iterrows():
                word = str(row[word_col]).strip().lower()

                # Skip empty cells or pandas NaN strings
                if not word or word == "nan":
                    continue

                # Safely parse numeric values
                try:
                    logfreq = float(row[col_logfreq])
                    zipf_val = 1.0 + logfreq
                except (ValueError, TypeError):
                    zipf_val = getattr(self, "OOV_ZIPF", 1.0)  # Fallback

                try:
                    subtlwf = float(row[col_subtlwf])
                except (ValueError, TypeError):
                    subtlwf = getattr(self, "OOV_SUBTLWF", 0.0)

                try:
                    freqcount = float(row[col_freqcount])
                except (ValueError, TypeError):
                    freqcount = 0.0

                self._data[word] = {
                    "subtlwf": subtlwf,
                    "zipf": zipf_val,
                    "cd": getattr(self, "OOV_CD", 0.0),
                    "freqcount": freqcount,
                }

        logger.info(f"SUBTLEX cargado: {len(self._data)} entradas.")

    # ------------------------------------------------------------------
    #  Métodos de acceso directo (para surprisal_source)
    # ------------------------------------------------------------------
    def zipf(self, word: str) -> float:
        return self._lookup(word)["zipf"]

    def cd(self, word: str) -> float:
        return self._lookup(word)["cd"]

    def subtlwf(self, word: str) -> float:
        return self._lookup(word)["subtlwf"]

    def surprisal(self, word: str) -> float:
        freq_pm = self._lookup(word)["subtlwf"]
        prob = max(freq_pm / 1_000_000, 1e-10)
        return float(-np.log2(prob))

    def is_oov(self, word: str) -> bool:
        return word.strip().lower() not in self._data

    def compute_surprisal(self, context: str, target: str) -> Tuple[float, float]:
        """Implements the same interface as SurprisalCalculator."""
        tokens = target.lower().split()
        if not tokens:
            return self.OOV_SURPRISAL, 0.0
        vals = [self.surprisal(t) for t in tokens]
        return vals[0], float(np.mean(vals[1:])) if len(vals) > 1 else 0.0

    # ------------------------------------------------------------------
    #  Métodos analíticos de corpus
    # ------------------------------------------------------------------
    def sofisticacion_lexica(
        self, lemmas: List[str], pos_tags: List[str]
    ) -> Dict[str, float]:
        content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        content = [(l, p) for l, p in zip(lemmas, pos_tags) if p in content_pos]
        if not content:
            return {
                "mean_zipf": 0.0,
                "sofisticacion": 0.0,
                "n_content": 0,
                "pct_oov": 0.0,
                "pct_low_freq": 0.0,
            }
        words = [l for l, _ in content]
        zipfs = [self.zipf(w) for w in words]
        oov_count = sum(1 for w in words if self.is_oov(w))
        low_freq = sum(1 for z in zipfs if z < 3.0)
        mean_z = float(np.mean(zipfs))
        return {
            "mean_zipf": round(mean_z, 3),
            "sofisticacion": round(7.0 - mean_z, 3),
            "n_content": len(words),
            "pct_oov": round(oov_count / len(words) * 100, 2),
            "pct_low_freq": round(low_freq / len(words) * 100, 2),
        }

    def _lookup(self, word: str) -> Dict[str, float]:
        key = word.strip().lower()
        return self._data.get(
            key,
            {
                "subtlwf": self.OOV_SUBTLWF,
                "zipf": self.OOV_ZIPF,
                "cd": self.OOV_CD,
                "freqcount": 0.0,
            },
        )

    def perfil_frecuencias(self, lemmas: List[str]) -> pd.DataFrame:
        zipf_vals = [(w, self.zipf(w)) for w in lemmas]
        rows = []
        cumul = 0.0
        total = len(lemmas)
        unique_lemmas = set(lemmas)
        for band_name, lo, hi in self.FREQ_BANDS:
            band_tokens = [w for w, z in zipf_vals if lo <= z < hi]
            band_types = set(band_tokens)
            pct_tokens = len(band_tokens) / total * 100 if total else 0.0
            cumul += pct_tokens
            rows.append(
                {
                    "banda": band_name,
                    "rango_zipf": f"{lo}–{hi}",
                    "n_tokens": len(band_tokens),
                    "n_types": len(band_types),
                    "pct_tokens": round(pct_tokens, 2),
                    "pct_types": round(len(band_types) / len(unique_lemmas) * 100, 2)
                    if unique_lemmas
                    else 0.0,
                    "cumul_pct_tokens": round(cumul, 2),
                }
            )
        return pd.DataFrame(rows)

    def perfil_registro(self, lemmas: List[str], pos_tags: List[str]) -> Dict[str, Any]:
        content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        content = [(l, p) for l, p in zip(lemmas, pos_tags) if p in content_pos]
        if not content:
            return {}
        words = [l for l, _ in content]
        zipfs = [self.zipf(w) for w in words]
        cds = [self.cd(w) for w in words]
        n = len(words)

        oral_count = sum(1 for z, c in zip(zipfs, cds) if z >= 5.0 and c >= 0.5)
        acad_count = sum(1 for z in zipfs if z < 4.0)
        dom_count = sum(
            1 for w, z in zip(words, zipfs) if z < 3.0 and not self.is_oov(w)
        )

        rare_words = sorted(
            [(w, round(z, 2)) for w, z in zip(words, zipfs) if z < 4.0],
            key=lambda x: x[1],
        )[:10]

        return {
            "oral_ratio": round(oral_count / n, 3),
            "academic_ratio": round(acad_count / n, 3),
            "domain_specific_ratio": round(dom_count / n, 3),
            "mean_zipf": round(float(np.mean(zipfs)), 3),
            "mean_cd": round(float(np.mean(cds)), 3),
            "top_rare_content_words": rare_words,
        }

    def trayectoria_carga_cognitiva(self, uces: List) -> pd.DataFrame:
        rows = []
        for uce in uces:
            content_lemmas = [
                l
                for l, p in zip(uce.lemmas, uce.pos_tags)
                if p in ("NOUN", "VERB", "ADJ", "ADV") and l.isalpha()
            ]
            if not content_lemmas:
                rows.append(
                    {
                        "uce_id": uce.id,
                        "mean_surprisal": 0.0,
                        "max_surprisal": 0.0,
                        "std_surprisal": 0.0,
                        "n_content": 0,
                    }
                )
                continue
            vals = [self.surprisal(w) for w in content_lemmas]
            rows.append(
                {
                    "uce_id": uce.id,
                    "mean_surprisal": round(float(np.mean(vals)), 3),
                    "max_surprisal": round(float(np.max(vals)), 3),
                    "std_surprisal": round(float(np.std(vals)), 3),
                    "n_content": len(vals),
                }
            )
        return pd.DataFrame(rows).set_index("uce_id")

    def analisis_oov(self, uces: List) -> pd.DataFrame:
        oov_counter = Counter()
        oov_by_uce = defaultdict(list)
        for uce in uces:
            for l, p in zip(uce.lemmas, uce.pos_tags):
                if p in ("NOUN", "VERB", "ADJ") and self.is_oov(l):
                    oov_counter[l] += 1
                    oov_by_uce[uce.id].append(l)
        rows = []
        for word, cnt in oov_counter.most_common():
            uces_with_word = [uid for uid, wlist in oov_by_uce.items() if word in wlist]
            rows.append(
                {
                    "lemma": word,
                    "freq_abs": cnt,
                    "n_uces": len(uces_with_word),
                    "uces": ", ".join(uces_with_word),
                }
            )
        return pd.DataFrame(rows)

    def enriquecer_uces(self, uces):
        rows = []
        for uce in uces:
            # Reusar los surprisals ya calculados por _calcular_token_surprisals
            surp_vals = list(uce.token_surprisals.values())
            mean_surp = float(np.mean(surp_vals)) if surp_vals else 0.0
            std_surp = float(np.std(surp_vals)) if surp_vals else 0.0
            sof = self.sofisticacion_lexica(uce.lemmas, uce.pos_tags)
            reg = self.perfil_registro(uce.lemmas, uce.pos_tags)
            rows.append(
                {
                    "uce_id": uce.id,
                    "mean_surprisal_content": mean_surp,
                    "std_surprisal_content": std_surp,
                    **sof,
                    **reg,
                }
            )
        return rows

    def clasificar_registro(self, uce) -> str:
        """
        Clasifica una UCE en 'coloquial', 'formal', 'tecnico' o 'mixto'.
        Basado en proporciones de vocabulario oral, académico y técnico.
        """
        lemmas = uce.lemmas
        pos_tags = uce.pos_tags
        if not lemmas:
            return "desconocido"

        # Solo palabras de contenido
        content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        words = [
            l for l, p in zip(lemmas, pos_tags) if p in content_pos and l.isalpha()
        ]
        if not words:
            return "desconocido"

        zipfs = [self.zipf(w) for w in words]
        cds = [self.cd(w) for w in words]
        n = len(words)

        oral_count = sum(1 for z, c in zip(zipfs, cds) if z >= 5.0 and c >= 0.5)
        academic_count = sum(1 for z in zipfs if z < 4.0)
        tech_count = sum(
            1 for w, z in zip(words, zipfs) if z < 3.0 and not self.is_oov(w)
        )

        oral_ratio = oral_count / n
        academic_ratio = academic_count / n
        tech_ratio = tech_count / n

        # Umbrales (ajustables)
        if tech_ratio > 0.15:
            return "tecnico"
        elif oral_ratio > 0.4:
            return "coloquial"
        elif academic_ratio > 0.3:
            return "formal"
        else:
            return "mixto"


@dataclass
class GlobalCorpus:
    """
    Owns all UCEs from all documents.
    Provides the write-back mechanism after global clustering.

    Usage:
        corpus = GlobalCorpus()
        for doc_id, texto in interviews.items():
            uces = pipeline.procesar(texto, doc_id=doc_id)
            corpus.add_document(doc_id, uces)

        result = pipeline.predicate_analyzer.cluster_all(corpus)
        # All uce.predicate_frames and uce.frame_annotations now have
        # correct cluster_ids across the whole corpus.

        corpus.export_dashboard("global_dashboard.json", result, subtlex)
    """

    def __init__(self):
        # doc_id → list of UCEs (preserves insertion order)
        self._docs: Dict[str, List[UCE]] = {}
        # Flat lookup: (doc_id, uce_id) → UCE
        self._uce_index: Dict[Tuple[str, str], UCE] = {}
        # Flat lookup: (doc_id, uce_id, frame_idx) → PredicateFrame
        self._frame_index: Dict[Tuple[str, str, int], PredicateFrame] = {}

    # ── Ingestion ────────────────────────────────────────────────────────
    def add_document(self, doc_id: str, uces: List[UCE]) -> None:
        if doc_id in self._docs:
            logger.warning(
                "GlobalCorpus: doc_id '%s' already exists, overwriting.", doc_id
            )
        self._docs[doc_id] = uces
        for uce in uces:
            self._uce_index[(doc_id, uce.id)] = uce
            for raw_frame in uce.predicate_frames:
                if isinstance(raw_frame, dict):
                    frame = PredicateFrame.from_dict(raw_frame)
                else:
                    frame = raw_frame
                key = (doc_id, uce.id, frame.frame_idx)
                self._frame_index[key] = frame

    # ── Write-back ───────────────────────────────────────────────────────
    def write_back_clusters(self, global_span_index: SpanAnnotationIndex) -> None:
        """
        After cluster_all() assigns cluster_ids to frames in the global
        SpanAnnotationIndex, this method propagates those ids back into
        the UCE objects stored in this corpus.

        It also regenerates frame_annotations (local-offset SpanAnnotation
        objects) so the HTML viewer and Streamlit can use them immediately.
        """

        updated = 0
        for frame in global_span_index._frames:
            key = (frame.doc_id, frame.uce_id, frame.frame_idx)
            stored = self._frame_index.get(key)
            if stored is None:
                continue
            if isinstance(stored, dict):
                stored = PredicateFrame.from_dict(stored)
                self._frame_index[key] = stored
            if not isinstance(stored, PredicateFrame):
                logger.warning(
                    "write_back_clusters: unexpected type %s at key %s",
                    type(stored),
                    key,
                )
                continue
            stored.cluster_id = frame.cluster_id
            stored.cluster_label = frame.cluster_label
            updated += 1

        # Regenerate SpanAnnotation objects (local offsets) per UCE
        for (doc_id, uce_id), uce in self._uce_index.items():
            frames_for_uce = []
            for i, raw_f in enumerate(uce.predicate_frames):
                # 1. Cast to PredicateFrame if it's a dictionary
                if isinstance(raw_f, dict):
                    f = PredicateFrame.from_dict(raw_f)
                    uce.predicate_frames[i] = f  # Update the list in-place
                else:
                    f = raw_f

                # 2. Sync with the definitively updated object from our index
                key = (doc_id, uce_id, getattr(f, "frame_idx", i))
                if key in self._frame_index:
                    f = self._frame_index[key]
                    uce.predicate_frames[i] = (
                        f  # Ensure UCE holds the updated reference
                    )

                # 3. Filter out unclustered frames safely
                if getattr(f, "cluster_id", -1) != -1:
                    frames_for_uce.append(f)

            tmp = SpanAnnotationIndex()
            for f in frames_for_uce:
                tmp.add(f)  # f is now guaranteed to be a PredicateFrame

            uce.frame_annotations = tmp.to_uce_annotations(uce)

        logger.info(
            "write_back_clusters(): updated %d frames across %d UCEs",
            updated,
            len(self._uce_index),
        )

    # ── Cross-document chain grouping ────────────────────────────────────
    def get_chain_groups(self) -> Dict[str, List[Tuple[str, Dict]]]:
        """
        Returns a dict of normalized_chain_rep → [(doc_id, chain_dict), ...]
        Enables exact-match cross-document grouping before clustering.
        """
        groups: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
        for doc_id, uces in self._docs.items():
            for uce in uces:
                for chain in uce.coref_chains:
                    key = chain["representative"].lower().strip()
                    groups[key].append((doc_id, chain))
        return dict(groups)

    # ── Queries ──────────────────────────────────────────────────────────
    def all_uces(self) -> List[UCE]:
        return [uce for uces in self._docs.values() for uce in uces]

    def uces_for_doc(self, doc_id: str) -> List[UCE]:
        return self._docs.get(doc_id, [])

    def all_frames(self) -> List[PredicateFrame]:
        return list(self._frame_index.values())

    def doc_ids(self) -> List[str]:
        return list(self._docs.keys())

    def n_docs(self) -> int:
        return len(self._docs)

    # ── Analytics ────────────────────────────────────────────────────────
    def cross_doc_chain_summary(self) -> pd.DataFrame:
        """
        Shows which chain representatives appear in multiple documents.
        Useful for verifying cross-doc grouping before clustering.
        """
        groups = self.get_chain_groups()
        rows = []
        for rep, entries in groups.items():
            doc_ids = [e[0] for e in entries]
            n_mentions = sum(len(e[1]["mentions"]) for e in entries)
            rows.append(
                {
                    "representative": rep,
                    "n_docs": len(set(doc_ids)),
                    "n_mentions": n_mentions,
                    "docs": sorted(set(doc_ids)),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values(["n_docs", "n_mentions"], ascending=False)
            .reset_index(drop=True)
        )

    def cross_doc_cluster_summary(self) -> pd.DataFrame:
        """
        After clustering: cluster → which docs contribute, top chains per doc.
        """
        frames = [f for f in self.all_frames() if f.cluster_id != -1]
        if not frames:
            return pd.DataFrame()
        cluster_c = defaultdict(lambda: defaultdict(list))
        for f in frames:
            cluster_c[f.cluster_id][f.doc_id].append(f)
        rows = []
        for cid, doc_frames in cluster_c.items():
            all_in_cluster = [f for fl in doc_frames.values() for f in fl]
            v_top = Counter(f.verb_lemma for f in all_in_cluster).most_common(3)
            o_top = Counter(
                f.direct_object_lemma for f in all_in_cluster if f.direct_object_lemma
            ).most_common(3)
            rows.append(
                {
                    "cluster_id": cid,
                    "cluster_label": all_in_cluster[0].cluster_label
                    if all_in_cluster
                    else "",
                    "n_docs": len(doc_frames),
                    "n_frames": len(all_in_cluster),
                    "docs": sorted(doc_frames.keys()),
                    "top_verbs": [v for v, _ in v_top],
                    "top_objects": [o for o, _ in o_top],
                    "top_chains": sorted(
                        {
                            f.chain_representative
                            for f in all_in_cluster
                            if f.chain_representative
                        }
                    )[:5],
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values(["n_docs", "n_frames"], ascending=False)
            .reset_index(drop=True)
        )

    def export_json(
        self,
        path: str,
        predicate_result: "CorefPredicateResult",
        subtlex_analyzer=None,
        lex_analyzer=None,
    ) -> None:
        """
        The Master JSON export. Kills all other fragmented exports and Excel dumps.
        """
        all_uces = self.all_uces()
        corpus = CorpusAnalyzer(
            all_uces,
            subtlex_analyzer=subtlex_analyzer,
        )

        documents_export = {}
        flat_uces = []
        all_annotations = []

        # 1. Build Documents, Flat UCEs, and Annotations
        for doc_id, uces in self._docs.items():
            uces_data = [uce_to_global_annotations(u) for u in uces]

            # Inject doc_id into flat structures to prevent frontend blindness
            for u_data in uces_data:
                u_data["doc_id"] = doc_id
            flat_uces.extend(uces_data)

            doc_annotations = sorted(
                [a for u in uces_data for a in u["annotations"]],
                key=lambda a: a["start_char"],
            )
            for ann in doc_annotations:
                ann["doc_id"] = doc_id
            all_annotations.extend(doc_annotations)

            documents_export[doc_id] = {
                "n_uces": len(uces),
                "uces": uces_data,
                "annotations": doc_annotations,
            }

        # 2. Build Global Coreference Index (FIXED: Cross-Doc Collision)
        coref_index = {}
        for uce in all_uces:
            for chain in uce.coref_chains:
                rep = chain["representative"]
                if rep not in coref_index:
                    coref_index[rep] = {
                        "representative": rep,
                        "cluster_id": chain.get("cluster_id", -1),
                        "cluster_label": chain.get("cluster_label", ""),
                        "mentions": [],
                    }

                # BUG FIX: Include doc_id in the uniqueness check
                existing = {
                    (x.get("doc_id"), x["start_char"], x["end_char"])
                    for x in coref_index[rep]["mentions"]
                }
                for m in chain["mentions"]:
                    if (uce.doc_id, m["start_char"], m["end_char"]) not in existing:
                        m_copy = dict(m)
                        m_copy["doc_id"] = uce.doc_id  # Inject doc_id
                        coref_index[rep]["mentions"].append(m_copy)

        for rep in coref_index:
            # Sort by doc_id first, then char offset
            coref_index[rep]["mentions"].sort(
                key=lambda m: (str(m.get("doc_id", "")), m["start_char"])
            )

        # 3. Build Predicate Frames (Cross-doc aware)
        import numpy as np  # Ensure this is imported for the NaN fix below

        def safe_df_to_dict(df):
            """Helper to nuke NaNs so JS doesn't crash."""
            if df is None or df.empty:
                return []
            return df.replace({np.nan: None}).to_dict(orient="records")

        frame_index = {
            "frames": [],
            "chain_summary": [],
            "cluster_summary": [],
            "cross_doc_chains": safe_df_to_dict(self.cross_doc_chain_summary()),
            "cross_doc_clusters": safe_df_to_dict(self.cross_doc_cluster_summary()),
            "by_chain": {},
            "by_cluster": {},
        }

        if predicate_result and not predicate_result.chain_summary.empty:
            frame_index["frames"] = [
                f.to_dict() for f in predicate_result.span_index._frames
            ]
            frame_index["chain_summary"] = safe_df_to_dict(
                predicate_result.chain_summary
            )
            frame_index["cluster_summary"] = safe_df_to_dict(
                predicate_result.cluster_summary
            )
            frame_index["by_chain"] = {
                rep: [f.to_dict() for f in frames]
                for rep, frames in predicate_result.span_index.by_chain.items()
            }
            frame_index["by_cluster"] = {
                str(cid): [f.to_dict() for f in frames]
                for cid, frames in predicate_result.span_index.by_cluster.items()
            }

        # 4. Generate the massive stats payload
        stats = corpus._build_stats_payload(lex_analyzer=lex_analyzer)

        # 5. Assemble the ultimate payload
        payload = {
            "meta": {
                "n_docs": self.n_docs(),
                "n_uces": len(all_uces),
                "doc_ids": self.doc_ids(),
                "version": 4,
                "offset_convention": "global_per_doc",
            },
            "documents": documents_export,
            "flat_uces": flat_uces,
            "all_annotations": all_annotations,
            "coref_index": list(coref_index.values()),
            "predicate_frames": frame_index,
            "stats": stats,
        }

        # 6. The God-Tier Recursive NaN Nuke
        def nuke_nans(obj):
            if isinstance(obj, list):
                return [nuke_nans(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: nuke_nans(v) for k, v in obj.items()}
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            elif obj is pd.NA:
                return None
            return obj

        clean_payload = nuke_nans(payload)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean_payload, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        logger.info("Master JSON exported to %s. Killed the Excel export.", path)


@dataclass
class CorpusAnalyzer:
    """
    Analizador lexicométrico y gramatical de un corpus de UCEs.

    No depende de listas predefinidas de categorías; descubre los valores
    directamente de los datos contenidos en cada UCE.

    Normalización:
        - per_k: frecuencia por `norm_base` tokens (por defecto 1000)
        - log:   log10(frecuencia + 1)
        - raw:   sin normalizar

    Uso típico:
        analyzer = CorpusAnalyzer(uces, norm_method="per_k", norm_base=1000)
        df = analyzer.to_dataframe()
        analyzer.mostrar_resumen()
        analyzer.exportar_excel("resultados.xlsx")
    """

    def __init__(
        self,
        uces: List,
        norm_method: str = "per_k",
        norm_base: int = 1000,
        subtlex_analyzer: Optional[SubtlexAnalyzer] = None,
    ):
        """
        Args:
            uces:        Lista de UCEs (objetos con los campos definidos en pipeline)
            norm_method: "per_k", "log", "raw"
            norm_base:   Base para normalización "per_k" (p.ej., 1000 tokens)
        """
        self.uces = uces
        self.norm_method = norm_method
        self.norm_base = norm_base
        self.subtlex_analyzer = subtlex_analyzer
        self._cache = {}

    # ------------------------------------------------------------------------
    # Métodos auxiliares
    # ------------------------------------------------------------------------
    def _norm(self, count: int, n_tokens: int) -> float:
        """Aplica la normalización elegida a un conteo."""
        if n_tokens == 0:
            return 0.0
        if self.norm_method == "per_k":
            return (count / n_tokens) * self.norm_base
        elif self.norm_method == "log":
            return np.log10(count + 1)
        else:  # raw
            return float(count)

    @staticmethod
    def _entropy(counter: Counter) -> float:
        """Shannon entropy (nats) de una distribución."""
        total = sum(counter.values())
        if total == 0:
            return 0.0
        probs = np.array([v / total for v in counter.values()])
        return float(scipy_entropy(probs))

    # ------------------------------------------------------------------------
    # Dataframe por UCE (con métricas normalizadas)
    # ------------------------------------------------------------------------
    # -------------------- NUEVOS MÉTODOS SUBTLEX --------------------
    def perfil_frecuencias_corpus(self) -> pd.DataFrame:
        """Perfil de frecuencias Zipf de todo el corpus (agregado)."""
        if self.subtlex_analyzer is None:
            return pd.DataFrame()
        all_lemmas = []
        for uce in self.uces:
            all_lemmas.extend(uce.lemmas)
        return self.subtlex_analyzer.perfil_frecuencias(all_lemmas)

    def analisis_oov(self) -> pd.DataFrame:
        """Palabras ausentes de SUBTLEX-ESP con frecuencias y UCEs."""
        if self.subtlex_analyzer is None:
            return pd.DataFrame()
        return self.subtlex_analyzer.analisis_oov(self.uces)

    def trayectoria_carga_cognitiva(self) -> pd.Series:
        """
        Serie con mean_surprisal_content por UCE (orden de aparición).
        Si no existe en metricas_lexicas, la calcula con subtlex_analyzer.
        """
        df = self.to_dataframe()
        if "mean_surprisal_content" in df.columns:
            return df["mean_surprisal_content"]
        elif self.subtlex_analyzer is not None:
            # Calcular sobre la marcha (puede ser lento)
            cog_df = self.subtlex_analyzer.trayectoria_carga_cognitiva(self.uces)
            return cog_df["mean_surprisal"]
        else:
            return pd.Series(index=[u.id for u in self.uces], data=0.0)

    # Sobrescribir to_dataframe para incluir columnas SUBTLEX (si existen)
    def to_dataframe(self) -> pd.DataFrame:
        if "df" in self._cache:
            return self._cache["df"]
        rows = []
        for uce in self.uces:
            m = uce.metricas_lexicas
            cs = uce.complejidad_sintactica
            n_tokens = m.get("num_tokens", 1) or 1
            row = {
                "uce_id": uce.id,
                "n_tokens": n_tokens,
                "n_types": m.get("num_types", 0),
                "ttr": m.get("ttr", 0.0),
                "guiraud": m.get("guiraud", 0.0),
                "hapax_ratio": m.get("hapax_ratio", 0.0),
                "diversidad_semantica": uce.diversidad_semantica,
                "topic_shift": uce.topic_shift_prev,
                "densidad_discursiva": m.get("densidad_discursiva", 0.0),
                "prof_sint_max": cs.get("profundidad_maxima", 0),
                "recursividad": cs.get("recursividad", 0),
                "dep_dist_media": cs.get("distancia_dependencia_media", 0.0),
                "ratio_subordinacion": cs.get("ratio_subordinacion", 0.0),
                "branching_ratio": cs.get("branching_ratio", 0.5),
                # Conteos crudos
                "negaciones_raw": len(uce.negaciones),
                "pronombres_exp_raw": sum(
                    1 for p in uce.pronombres if p.get("tipo") == "EXPLICITO"
                ),
                "prodrop_raw": sum(
                    1 for p in uce.pronombres if p.get("tipo") == "NULO"
                ),
                "verbos_raw": len(uce.verbos),
                "cuantificadores_raw": len(uce.cuantificadores),
                "adverbios_raw": len(uce.adverbios),
                "marcadores_raw": len(uce.marcadores_discursivos),
                "coref_chains_raw": len(uce.coref_chains),
                # Nuevas columnas SUBTLEX
                "mean_zipf": m.get("mean_zipf", 0.0),
                "lexical_sophistication": m.get("lexical_sophistication", 0.0),
                "pct_oov": m.get("pct_oov", 0.0),
                "pct_low_freq": m.get("pct_low_freq", 0.0),
                "oral_ratio": m.get("oral_ratio", 0.0),
                "academic_ratio": m.get("academic_ratio", 0.0),
                "domain_specific_ratio": m.get("domain_specific_ratio", 0.0),
                "mean_surprisal_content": m.get("mean_surprisal_content", 0.0),
            }
            # Añadir versiones normalizadas (ya lo hacías)
            for raw_name in [
                "negaciones_raw",
                "pronombres_exp_raw",
                "prodrop_raw",
                "verbos_raw",
                "cuantificadores_raw",
                "adverbios_raw",
                "marcadores_raw",
                "coref_chains_raw",
            ]:
                col = raw_name.replace("_raw", "_norm")
                row[col] = self._norm(row[raw_name], n_tokens)
            rows.append(row)
        df = pd.DataFrame(rows).set_index("uce_id")
        self._cache["df"] = df
        return df

    # ------------------------------------------------------------------------
    # Resumen global (estadísticos descriptivos)
    # ------------------------------------------------------------------------
    def resumen_global(self) -> pd.DataFrame:
        """Estadísticos descriptivos de todas las columnas numéricas."""
        df = self.to_dataframe()
        summary = df.describe(percentiles=[0.25, 0.5, 0.75]).T
        summary.columns = ["count", "mean", "std", "min", "Q1", "median", "Q3", "max"]
        return summary

    # ------------------------------------------------------------------------
    # Perfiles gramaticales dinámicos (sin listas fijas)
    # ------------------------------------------------------------------------
    def perfil_verbal(self) -> pd.DataFrame:
        """
        Frecuencias y entropías de cada rasgo verbal presente en los datos.
        Los rasgos se detectan automáticamente de los diccionarios de verbos.
        """
        n_total_tokens = sum(u.metricas_lexicas.get("num_tokens", 0) for u in self.uces)
        total_verbs = sum(len(u.verbos) for u in self.uces)

        # Recolectar todos los campos que aparecen en los verbos
        all_fields = set()
        for u in self.uces:
            for v in u.verbos:
                all_fields.update(v.keys())
        # Campos relevantes (excluir metadatos)
        exclude = {
            "texto",
            "lema",
            "pos",
            "aux_tipo",
            "perifrasis",
            "valencia",
            "char_start",
            "char_end",
        }
        features = [f for f in all_fields if f not in exclude]

        rows = []
        for feature in features:
            counter = Counter()
            for u in self.uces:
                for v in u.verbos:
                    val = v.get(feature)
                    if val is not None:
                        counter[val] += 1
            if not counter:
                continue
            ent = self._entropy(counter)
            for val, cnt in counter.most_common():
                rows.append(
                    {
                        "categoria": feature,
                        "valor": val,
                        "freq_abs": cnt,
                        "freq_rel": cnt / total_verbs if total_verbs else 0.0,
                        "freq_norm": self._norm(cnt, n_total_tokens),
                        "entropia_categoria": ent,
                    }
                )
        return pd.DataFrame(rows)

    def perfil_negaciones(self) -> pd.DataFrame:
        """Frecuencias de los tipos de negación presentes."""
        n_total_tokens = sum(u.metricas_lexicas.get("num_tokens", 0) for u in self.uces)
        counter = Counter()
        for u in self.uces:
            for neg in u.negaciones:
                tipo = neg.get("tipo", "DESCONOCIDO")
                counter[tipo] += 1
        total = sum(counter.values())
        rows = []
        for tipo, cnt in counter.most_common():
            rows.append(
                {
                    "tipo": tipo,
                    "freq_abs": cnt,
                    "freq_rel": cnt / total if total else 0.0,
                    "freq_norm": self._norm(cnt, n_total_tokens),
                }
            )
        return pd.DataFrame(rows)

    def perfil_adverbios(self) -> pd.DataFrame:
        """Frecuencias de las categorías de adverbios presentes."""
        n_total_tokens = sum(u.metricas_lexicas.get("num_tokens", 0) for u in self.uces)
        counter = Counter()
        for u in self.uces:
            for adv in u.adverbios:
                cat = adv.get("categoria", "DESCONOCIDO")
                counter[cat] += 1
        total = sum(counter.values())
        rows = []
        for cat, cnt in counter.most_common():
            rows.append(
                {
                    "categoria": cat,
                    "freq_abs": cnt,
                    "freq_rel": cnt / total if total else 0.0,
                    "freq_norm": self._norm(cnt, n_total_tokens),
                }
            )
        return pd.DataFrame(rows)

    def perfil_pronombres(self) -> pd.DataFrame:
        """Frecuencias de tipos y subtipos de pronombres."""
        n_total_tokens = sum(u.metricas_lexicas.get("num_tokens", 0) for u in self.uces)
        tipo_counter = Counter()
        subtipo_counter = Counter()
        for u in self.uces:
            for p in u.pronombres:
                tipo = p.get("tipo", "OTRO")
                tipo_counter[tipo] += 1
                subtipo = p.get("subtipo")
                if subtipo:
                    subtipo_counter[subtipo] += 1
        total = sum(tipo_counter.values())
        rows = []
        # Por tipo
        for tipo, cnt in tipo_counter.most_common():
            rows.append(
                {
                    "nivel": "tipo",
                    "clave": tipo,
                    "freq_abs": cnt,
                    "freq_rel": cnt / total if total else 0.0,
                    "freq_norm": self._norm(cnt, n_total_tokens),
                }
            )
        # Por subtipo
        for subtipo, cnt in subtipo_counter.most_common():
            rows.append(
                {
                    "nivel": "subtipo",
                    "clave": subtipo,
                    "freq_abs": cnt,
                    "freq_rel": cnt / total if total else 0.0,
                    "freq_norm": self._norm(cnt, n_total_tokens),
                }
            )
        # Ratio pro-drop / explícito
        n_pro = tipo_counter.get("NULO", 0)
        n_exp = tipo_counter.get("EXPLICITO", 0)
        ratio = n_pro / (n_pro + n_exp) if (n_pro + n_exp) else 0.0
        rows.append(
            {
                "nivel": "ratio",
                "clave": "prodrop_vs_explicito",
                "freq_abs": n_pro,
                "freq_rel": ratio,
                "freq_norm": self._norm(n_pro, n_total_tokens),
            }
        )
        return pd.DataFrame(rows)

    def resumen_registros(self) -> pd.DataFrame:
        """Distribución de registros en el corpus."""
        registros = [uce.registro for uce in self.uces if uce.registro]
        if not registros:
            return pd.DataFrame()
        counter = Counter(registros)
        total = len(registros)
        rows = []
        for reg, cnt in counter.most_common():
            rows.append(
                {
                    "registro": reg,
                    "count": cnt,
                    "percentage": round(cnt / total * 100, 2),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------------
    # Trayectoria secuencial
    # ------------------------------------------------------------------------
    def trayectoria(self, metrica: str) -> pd.Series:
        """Devuelve Serie con valores de una métrica numérica por UCE."""
        df = self.to_dataframe()
        if metrica not in df.columns:
            raise ValueError(
                f"Métrica '{metrica}' no disponible. Opciones: {list(df.columns)}"
            )
        return df[metrica]

    def alternancia_voz(self) -> Dict[str, Any]:
        """
        Analiza la alternancia de voz a lo largo del corpus (entre UCEs consecutivas).

        Para cada UCE se determina la voz predominante entre sus verbos principales
        (VERB con modo finito). Luego se cuentan los cambios entre UCEs adyacentes.

        Returns:
            Diccionario con:
            - secuencia: lista de tuplas (uce_id, voz_predominante)
            - n_cambios: número de cambios de voz entre UCEs consecutivas
            - tasa_cambio: n_cambios / (total_UCEs - 1)
            - distribucion: Counter de frecuencias de cada voz
            - matriz_transicion: DataFrame con frecuencias de transiciones (voz_anterior → voz_actual)
        """

        import pandas as pd

        secuencia = []

        for uce in self.uces:
            # Filtrar verbos principales (no auxiliares, formas finitas)
            verbos_principales = [
                v
                for v in uce.verbos
                if v.get("pos") == "VERB" and v.get("modo") is not None
            ]

            if not verbos_principales:
                voz_pred = None
            else:
                # Extraer voces (si no tiene campo 'voz', asumir 'Act')
                voces = [
                    v.get("voz", "Act")
                    for v in verbos_principales
                    if v.get("voz") is not None
                ]
                if not voces:
                    voz_pred = None
                else:
                    # Voz más frecuente en esta UCE
                    voz_pred = Counter(voces).most_common(1)[0][0]

            secuencia.append((uce.id, voz_pred))

        # Calcular cambios y transiciones
        cambios = 0
        transiciones = []  # lista de (voz_prev, voz_curr)

        for i in range(1, len(secuencia)):
            prev_voz = secuencia[i - 1][1]
            curr_voz = secuencia[i][1]
            if prev_voz is not None and curr_voz is not None and prev_voz != curr_voz:
                cambios += 1
                transiciones.append((prev_voz, curr_voz))

        n_uces = len(secuencia)
        tasa = cambios / (n_uces - 1) if n_uces > 1 else 0.0

        # Distribución de voces (ignorando None)
        distribucion = Counter(voz for _, voz in secuencia if voz is not None)

        # Matriz de transición
        voces_unicas = sorted(set(distribucion.keys()))
        matriz = pd.DataFrame(0, index=voces_unicas, columns=voces_unicas)
        for prev, curr in transiciones:
            matriz.loc[prev, curr] += 1

        return {
            "secuencia": secuencia,
            "n_cambios": cambios,
            "tasa_cambio": tasa,
            "distribucion": dict(distribucion),
            "matriz_transicion": matriz,
        }

    # ------------------------------------------------------------------------
    # Tablas de contingencia (crosstab) dinámicas
    # ------------------------------------------------------------------------
    def crosstab_verbos(self, campo1: str, campo2: str) -> pd.DataFrame:
        """Tabla de contingencia entre dos campos de los verbos."""
        pairs = []
        for u in self.uces:
            for v in u.verbos:
                v1 = v.get(campo1)
                v2 = v.get(campo2)
                if v1 is not None and v2 is not None:
                    pairs.append((str(v1), str(v2)))
        if not pairs:
            return pd.DataFrame()
        index_vals = sorted(set(p[0] for p in pairs))
        col_vals = sorted(set(p[1] for p in pairs))
        mat = pd.DataFrame(0, index=index_vals, columns=col_vals)
        for v1, v2 in pairs:
            mat.loc[v1, v2] += 1
        mat.index.name = campo1
        mat.columns.name = campo2
        return mat

    # ------------------------------------------------------------------------
    # Entropías por categoría (dinámicas)
    # ------------------------------------------------------------------------
    def entropias(self) -> pd.DataFrame:
        """
        Entropía de Shannon para cada categoría gramatical presente.
        Las categorías se detectan automáticamente.
        """
        features = {
            "negacion_tipo": [
                neg.get("tipo") for u in self.uces for neg in u.negaciones
            ],
            "pronombre_tipo": [p.get("tipo") for u in self.uces for p in u.pronombres],
            "pronombre_subtipo": [
                p.get("subtipo")
                for u in self.uces
                for p in u.pronombres
                if p.get("subtipo")
            ],
            "adverbio_cat": [
                a.get("categoria") for u in self.uces for a in u.adverbios
            ],
            "cuantificador_tipo": [
                c.get("tipo") for u in self.uces for c in u.cuantificadores
            ],
            "marcador_cat": [
                m.get("categoria") for u in self.uces for m in u.marcadores_discursivos
            ],
        }
        # Añadir rasgos verbales dinámicamente
        verb_features = set()
        for u in self.uces:
            for v in u.verbos:
                verb_features.update(
                    k
                    for k in v.keys()
                    if k
                    not in {
                        "texto",
                        "lema",
                        "pos",
                        "aux_tipo",
                        "perifrasis",
                        "valencia",
                        "char_start",
                        "char_end",
                    }
                )
        for feat in verb_features:
            vals = [
                v.get(feat)
                for u in self.uces
                for v in u.verbos
                if v.get(feat) is not None
            ]
            if vals:
                features[f"verbo_{feat}"] = vals

        rows = []
        for fname, vals in features.items():
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            ctr = Counter(vals)
            total = sum(ctr.values())
            H = self._entropy(ctr)
            top3 = ", ".join(f"{k}:{v}" for k, v in ctr.most_common(3))
            rows.append(
                {
                    "feature": fname,
                    "n_occurrences": total,
                    "n_types": len(ctr),
                    "entropy_nat": round(H, 4),
                    "top_3": top3,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------------
    # Concordancia (KWIC)
    # ------------------------------------------------------------------------
    def concordancia(
        self, patron: str, campo: str = "texto", window: int = 40
    ) -> pd.DataFrame:
        """
        Búsqueda de concordancia en el corpus.
        campo puede ser "texto", "adverbio_cat", "verbo_lema", "verbo_modo".
        """
        rows = []
        for uce in self.uces:
            if campo == "texto":
                for m in re.finditer(patron, uce.texto, re.IGNORECASE):
                    s, e = m.start(), m.end()
                    left = uce.texto[max(0, s - window) : s]
                    right = uce.texto[e : e + window]
                    rows.append(
                        {
                            "uce_id": uce.id,
                            "left": left,
                            "match": m.group(),
                            "right": right,
                        }
                    )
            elif campo == "adverbio_cat":
                for a in uce.adverbios:
                    if a.get("categoria") == patron:
                        rows.append(
                            {
                                "uce_id": uce.id,
                                "adverbio": a.get("texto"),
                                "confianza": a.get("confianza"),
                                "contexto": a.get("contexto", ""),
                            }
                        )
            elif campo == "verbo_lema":
                for v in uce.verbos:
                    if v.get("lema") == patron:
                        rows.append(
                            {
                                "uce_id": uce.id,
                                "texto": v.get("texto"),
                                "modo": v.get("modo"),
                                "tiempo": v.get("tiempo"),
                                "aspecto": v.get("aspecto"),
                            }
                        )
            elif campo == "verbo_modo":
                for v in uce.verbos:
                    if v.get("modo") == patron:
                        rows.append(
                            {
                                "uce_id": uce.id,
                                "texto": v.get("texto"),
                                "lema": v.get("lema"),
                                "tiempo": v.get("tiempo"),
                            }
                        )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------------
    # Exportación
    # ------------------------------------------------------------------------

    def _build_stats_payload(
        self,
        lex_analyzer=None,
    ) -> Dict:
        """
        Builds the corpus stats dict shared by both export methods.
        Pure data — no I/O.
        """
        stats = {
            "global": self.resumen_global().reset_index().to_dict(orient="records"),
            "verbos": self.perfil_verbal().to_dict(orient="records"),
            "negaciones": self.perfil_negaciones().to_dict(orient="records"),
            "adverbios": self.perfil_adverbios().to_dict(orient="records"),
            "pronombres": self.perfil_pronombres().to_dict(orient="records"),
            "entropias": self.entropias().to_dict(orient="records"),
            "registros": self.resumen_registros().to_dict(orient="records"),
            "alternancia_voz": {
                k: (v.to_dict() if hasattr(v, "to_dict") else v)
                for k, v in self.alternancia_voz().items()
            },
            "crosstab_modo_aspecto": self.crosstab_verbos("modo", "aspecto")
            .reset_index()
            .to_dict(orient="records"),
            "crosstab_tiempo_voz": self.crosstab_verbos("tiempo", "voz")
            .reset_index()
            .to_dict(orient="records"),
        }

        if self.subtlex_analyzer:
            stats["perfil_frecuencias"] = self.perfil_frecuencias_corpus().to_dict(
                orient="records"
            )
            stats["analisis_oov"] = self.analisis_oov().to_dict(orient="records")
            stats["trayectoria_carga_cognitiva"] = (
                self.trayectoria_carga_cognitiva().to_dict()
            )

        if lex_analyzer is not None:
            # We grab EVERYTHING from the lexical analyzer here.
            # No leaving data behind.
            stats["lexical_analysis"] = {
                "riqueza_lexica": lex_analyzer.richness_dataframe()
                .reset_index()
                .to_dict(orient="records"),
                "zipf_profile": lex_analyzer.zipf_profile()
                .head(500)
                .to_dict(orient="records"),
                "frequency_bands": lex_analyzer.frequency_bands().to_dict(
                    orient="records"
                ),
                "corpus_vocabulary": lex_analyzer.corpus_vocabulary().to_dict(
                    orient="records"
                ),
                "semantic_network": {
                    "communities": lex_analyzer.community_summary().to_dict(
                        orient="records"
                    ),
                    "bridge_nodes": lex_analyzer.bridge_node_report().to_dict(
                        orient="records"
                    ),
                    "top_nodes": lex_analyzer.semantic_network_summary().to_dict(
                        orient="records"
                    ),
                },
            }

        return stats

    # ------------------------------------------------------------------------
    # Impresión en consola (resumen amigable)
    # ------------------------------------------------------------------------
    def mostrar_resumen(self):
        """Imprime un resumen estadístico del corpus en consola."""
        df = self.to_dataframe()
        total_tokens = df["n_tokens"].sum()
        total_uces = len(df)
        print("\n" + "=" * 70)
        print("RESUMEN DEL CORPUS")
        print("=" * 70)
        print(f"UCEs:               {total_uces}")
        print(f"Tokens totales:     {total_tokens}")
        print(f"Tipos (types):      {df['n_types'].sum()}")
        print(f"TTR medio:          {df['ttr'].mean():.3f}")
        print(f"Guiraud medio:      {df['guiraud'].mean():.3f}")
        print(f"Diversidad semántica media: {df['diversidad_semantica'].mean():.3f}")
        print(f"Topic shift medio:  {df['topic_shift'].mean():.3f}")
        print(f"\n--- Tasas normalizadas (por {self.norm_base} tokens) ---")
        for col in df.columns:
            if col.endswith("_norm"):
                mean_val = df[col].mean()
                print(f"{col:20s} : {mean_val:.3f}")
        # Entropías
        ent = self.entropias()
        if not ent.empty:
            print("\n--- Entropía de Shannon (nats) ---")
            for _, row in ent.iterrows():
                print(
                    f"{row['feature']:25s} : {row['entropy_nat']:.3f}  (tipos: {row['n_types']})"
                )
        print("=" * 70 + "\n")


@dataclass
class PredicateInstance:
    """One subject–verb pair extracted from a coref mention or semantic expansion."""

    subject: str
    verb_lemma: str
    verb_text: str
    context: str  # full sentence text — fed to BERTopic
    grammar: Dict  # from UCE verb annotation
    is_semantic_expansion: bool
    chain_representative: str = ""  # empty for expansion instances
    original_subject: str = ""  # filled for expansion instances
    topic_id: int = -1  # assigned after BERTopic
    topic_label: str = ""
    verb_char_start: int = None  # char offsets in UCE text
    verb_char_end: int = None
    subject_char_start: int = None
    subject_char_end: int = None
    subject_head_lemma: str = ""  # Añadimos esto para la expansión semántica

    # Dentro de la definición de PredicateInstance (fuera de UCE)
    def to_dict(self) -> Dict:
        return {
            "subject": self.subject,
            "verb_lemma": self.verb_lemma,
            "verb_text": self.verb_text,
            "context": self.context,
            "grammar": self.grammar,  # ya es dict
            "is_semantic_expansion": self.is_semantic_expansion,
            "chain_representative": self.chain_representative,
            "original_subject": self.original_subject,
            "topic_id": self.topic_id,
            "topic_label": self.topic_label,
            "verb_char_start": self.verb_char_start,  # nuevo campo
            "verb_char_end": self.verb_char_end,
            "subject_char_start": self.subject_char_start,
            "subject_char_end": self.subject_char_end,
            "subject_head_lemma": self.subject_head_lemma,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PredicateInstance":
        return cls(
            subject=data["subject"],
            verb_lemma=data["verb_lemma"],
            verb_text=data["verb_text"],
            context=data["context"],
            grammar=data["grammar"],
            is_semantic_expansion=data["is_semantic_expansion"],
            chain_representative=data["chain_representative"],
            original_subject=data.get("original_subject", ""),
            topic_id=data.get("topic_id", -1),
            topic_label=data.get("topic_label", ""),
            verb_char_start=data.get("verb_char_start"),
            verb_char_end=data.get("verb_char_end"),
            subject_char_start=data.get("subject_char_start"),
            subject_char_end=data.get("subject_char_end"),
            subject_head_lemma=data.get("subject_head_lemma", ""),
        )


@dataclass
class CorefPredicateResult:
    span_index: SpanAnnotationIndex
    n_base: int
    n_expansions: int
    chain_summary: pd.DataFrame
    cluster_summary: pd.DataFrame

    @classmethod
    def from_span_index(
        cls, idx: SpanAnnotationIndex, n_base: int
    ) -> "CorefPredicateResult":
        n_exp = len(idx) - n_base
        return cls(
            span_index=idx,
            n_base=n_base,
            n_expansions=max(n_exp, 0),
            chain_summary=idx.summary_by_chain(),
            cluster_summary=idx.summary_by_cluster(),
        )

    @classmethod
    def empty(cls) -> "CorefPredicateResult":
        return cls(
            span_index=SpanAnnotationIndex(),
            n_base=0,
            n_expansions=0,
            chain_summary=pd.DataFrame(),
            cluster_summary=pd.DataFrame(),
        )

    def to_dict(self) -> Dict:
        return {
            "n_base": self.n_base,
            "n_expansions": self.n_expansions,
            "frames": [f.to_dict() for f in self.span_index._frames],
            "chain_summary": self.chain_summary.to_dict(orient="records"),
            "cluster_summary": self.cluster_summary.to_dict(orient="records"),
        }

    @property
    def summary_text(self) -> str:
        if self.chain_summary.empty:
            return "No predicate frames found."
        lines = [
            f"Frames: {self.n_base} base + {self.n_expansions} expansions",
            f"Chains: {len(self.span_index.by_chain)}",
            f"Clusters: {len(self.span_index.by_cluster)}",
            "",
            self.chain_summary.to_string(index=False),
        ]
        return "\n".join(lines)


@dataclass
class CorefPredicateAnalyzer:
    """
    Extracts predicate-argument frames for all coreferenced entities,
    clusters them via paraphrase mining + Louvain, and writes results
    back to UCE objects and a SpanAnnotationIndex.

    Memory strategy:
        Phase 1 — spaCy:    one doc at a time, freed after extraction
        Phase 2 — numpy:    vectorized expansion, freed before embedding
        Phase 3 — embedder: runs on fingerprint strings only (short)
        Phase 4 — write-back: pure Python, no large allocations
    """

    def __init__(
        self,
        nlp: spacy.Language,
        we_analyzer,
        sentence_embedder,
        similarity_threshold: float = 0.72,
        top_k_similar: int = 10,
        louvain_resolution: float = 1.0,
        paraphrase_top_k: int = 15,
        embed_batch_size: int = 64,
    ):
        self.nlp = nlp
        self.we_analyzer = we_analyzer
        self.sentence_embedder = sentence_embedder
        self.similarity_threshold = similarity_threshold
        self.top_k_similar = top_k_similar
        self.louvain_resolution = louvain_resolution
        self.paraphrase_top_k = paraphrase_top_k
        self.embed_batch_size = embed_batch_size
        self._pending_frames: List[PredicateFrame] = []
        self._normalization_cache: Dict[str, str] = {}  # ← add this
        # Accumulator across documents — cleared by cluster_all()
        self._pending_frames: List[PredicateFrame] = []

    def _normalize_rep(self, rep: str) -> str:
        """Elimina artículos, palabras muy comunes y normaliza."""
        if rep in self._normalization_cache:
            return self._normalization_cache[rep]

        # Eliminar artículos determinados e indeterminados
        stop = {
            "el",
            "la",
            "los",
            "las",
            "un",
            "una",
            "unos",
            "unas",
            "y",
            "de",
            "a",
            "del",
            "al",
        }
        words = rep.lower().split()
        filtered = [w for w in words if w not in stop and len(w) > 2]
        norm = " ".join(filtered)
        self._normalization_cache[rep] = norm
        return norm

    # ================================================================
    # PHASE A — one call per document
    # ================================================================
    def extract(self, uces: List, doc_id: str = "") -> None:
        """
        Extracts predicate frames for one document.
        Writes results into uce.predicate_frames (cluster_id=-1).
        Appends frames to self._pending_frames.
        """
        verb_index = self._build_verb_index(uces)
        span_index = SpanAnnotationIndex()
        all_nouns: List[Dict] = []

        for uce in uces:
            chains = getattr(uce, "_coref_chains_full", uce.coref_chains)
            if not chains:
                continue
            doc = self.nlp(uce.texto)
            try:
                self._extract_frames_from_uce(uce, doc, verb_index, span_index)
                all_nouns.extend(self._collect_nouns_from_doc(uce, doc, verb_index))
            finally:
                del doc
                gc.collect()

        # Semantic expansion (vectorized, no spaCy)
        expansions = self._expand_vectorized(span_index._frames, all_nouns)
        del all_nouns
        gc.collect()
        for f in expansions:
            span_index.add(f)

        # Stamp provenance + sequential frame_idx
        all_frames = span_index._frames
        for i, frame in enumerate(all_frames):
            frame.doc_id = doc_id
            frame.frame_idx = i

        # Write raw frames back to UCEs (cluster_id=-1)
        for uce in uces:
            uce.predicate_frames = span_index.by_uce.get(uce.id, [])
            uce.frame_annotations = span_index.to_uce_annotations(uce)

        # Accumulate for global clustering
        self._pending_frames.extend(all_frames)
        logger.info(
            "extract() doc='%s': %d frames (%d base + %d expansions), "
            "pending total: %d",
            doc_id,
            len(all_frames),
            len(all_frames) - len(expansions),
            len(expansions),
            len(self._pending_frames),
        )

    # ================================================================
    # PHASE B — called once after all documents
    # ================================================================
    def cluster_all(self, global_corpus):
        # Work on a stable copy, do not reassign the variable
        frames: List[PredicateFrame] = []
        for f in self._pending_frames:
            if isinstance(f, dict):
                f = PredicateFrame.from_dict(f)
            f.norm_chain = self._normalize_rep(
                f.chain_representative
            )  # uses the method
            frames.append(f)

        exact_groups: Dict[str, List[PredicateFrame]] = defaultdict(list)
        for f in frames:
            exact_groups[f.norm_chain].append(f)

        fingerprints = [f.frame_fingerprint for f in frames]
        assignments, labels = self._cluster_fingerprints(
            fingerprints, frames, exact_groups
        )

        global_span_index = SpanAnnotationIndex()
        for frame in frames:
            global_span_index.add(frame)
        global_span_index.assign_clusters(assignments, labels)

        global_corpus.write_back_clusters(global_span_index)

        n_total = len(frames)
        self._pending_frames = []
        gc.collect()

        result = CorefPredicateResult.from_span_index(global_span_index, n_base=n_total)
        logger.info(
            "cluster_all(): Chains: %d | Clusters: %d | Frames: %d",
            len(global_span_index.by_chain),
            len(global_span_index.by_cluster),
            len(global_span_index),
        )
        return result

    # ================================================================
    # Clustering helper — modified to use exact groups as priors
    # ================================================================
    def _cluster_fingerprints(
        self,
        fingerprints: List[str],
        frames: List[PredicateFrame],
        exact_groups: Dict[str, List[PredicateFrame]],
    ) -> Tuple[List[int], Dict[int, str]]:

        if len(fingerprints) < 2:
            return [-1] * len(fingerprints), {}

        embeddings = self.sentence_embedder.encode(
            fingerprints,
            batch_size=self.embed_batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        paraphrases = st_util.paraphrase_mining_embeddings(
            embeddings,
            top_k=self.paraphrase_top_k,
            score_function=st_util.dot_score,
        )

        # En cluster_all(), antes de exact_groups
        frames_normalized = []
        for f in frames:
            f.norm_chain = self._normalize_rep(f.chain_representative)
            frames_normalized.append(f)

        exact_groups = defaultdict(list)
        for f in frames_normalized:
            exact_groups[f.norm_chain].append(f)

        # Build frame → index lookup
        frame_to_idx = {id(f): i for i, f in enumerate(frames)}

        G = nx.Graph()
        G.add_nodes_from(range(len(frames)))

        # Prior 1: exact chain_representative match (strongest prior)
        # Frames sharing the same normalized representative MUST be
        # in the same community — wire them with weight 2.0
        for key, group in exact_groups.items():
            indices = [frame_to_idx[id(f)] for f in group if id(f) in frame_to_idx]
            for a, b in zip(indices, indices[1:]):
                G.add_edge(a, b, weight=2.0)

        # Prior 2: coref chain membership within a document
        chain_map: Dict[str, List[int]] = defaultdict(list)
        for i, f in enumerate(frames):
            if f.chain_representative:
                chain_key = f"{f.doc_id}::{f.chain_representative}"
                chain_map[chain_key].append(i)
        for members in chain_map.values():
            for a, b in zip(members, members[1:]):
                w = G[a][b]["weight"] if G.has_edge(a, b) else 0.0
                G.add_edge(a, b, weight=max(w, 1.0))

        # Paraphrase edges
        for score, i, j in paraphrases:
            score = float(score)
            if score >= self.similarity_threshold and i != j:
                w = G[i][j]["weight"] if G.has_edge(i, j) else 0.0
                G.add_edge(i, j, weight=max(w, score))

        if G.number_of_edges() == 0:
            return [-1] * len(frames), {}

        coms = algorithms.louvain(
            G, weight="weight", resolution=self.louvain_resolution
        )

        assignments = [-1] * len(frames)
        labels: Dict[int, str] = {}

        for cid, community in enumerate(coms.communities):
            for node in community:
                assignments[node] = cid
            cf = [frames[n] for n in community]
            v_top = Counter(f.verb_lemma for f in cf).most_common(1)
            o_top = Counter(
                f.direct_object_lemma for f in cf if f.direct_object_lemma
            ).most_common(1)
            # Also show how many docs this cluster spans
            n_docs = len({f.doc_id for f in cf})
            v_str = v_top[0][0] if v_top else "?"
            o_str = o_top[0][0] if o_top else ""
            base = f"{v_str}+{o_str}" if o_str else v_str
            labels[cid] = f"{base} [{n_docs}docs]"

        return assignments, labels

    # ================================================================
    # Phase 1 helpers (unchanged from previous version)
    # ================================================================
    def _build_verb_index(self, uces):
        index = {}
        for uce in uces:
            neg_ranges = [
                (n["alcance_char_start"], n["alcance_char_end"]) for n in uce.negaciones
            ]
            for v in uce.verbos:
                key = (v["char_start"], v["char_end"])
                has_neg = any(lo <= v["char_start"] <= hi for lo, hi in neg_ranges)
                index[key] = {**v, "negacion": has_neg}
        return index

    def _extract_frames_from_uce(self, uce, doc, verb_index, span_index):
        chains_to_use = getattr(uce, "_coref_chains_full", uce.coref_chains)
        for chain in chains_to_use:
            rep = chain.get("representative", "")
            for mention in chain.get("mentions", []):
                mention_aug = {**mention, "chain_representative": rep}
                frame = extract_predicate_frame(uce, mention_aug, doc, verb_index)
                if frame is not None:
                    span_index.add(frame)

    def _collect_nouns_from_doc(self, uce, doc, verb_index):
        nouns = []
        for token in doc:
            if token.pos_ not in ("NOUN", "PROPN"):
                continue
            vec = self.we_analyzer.vector(token.text)
            if np.allclose(vec, 0):
                continue
            try:
                sent = token.sent
            except ValueError:
                continue
            verb = (
                sent.root
                if sent.root.pos_ == "VERB"
                else next((t for t in sent if t.pos_ == "VERB"), None)
            )
            if verb is None:
                continue
            v_g_start = uce.start_char + verb.idx
            v_g_end = uce.start_char + verb.idx + len(verb.text)
            grammar = verb_index.get((v_g_start, v_g_end), {})
            nouns.append(
                {
                    "text": token.text,
                    "lemma": token.lemma_.lower(),
                    "vector": vec,
                    "context": sent.text,
                    "verb_lemma": verb.lemma_,
                    "verb_text": verb.text,
                    "grammar": grammar,
                    "char_start": uce.start_char + token.idx,
                    "char_end": uce.start_char + token.idx + len(token.text),
                    "uce_id": uce.id,
                }
            )
        return nouns

    def _expand_vectorized(self, base_frames, all_nouns):
        if not base_frames or not all_nouns:
            return []
        unique_heads = list(
            {f.entity_head_lemma for f in base_frames if f.entity_head_lemma}
        )
        q_vecs = np.array([self.we_analyzer.vector(h) for h in unique_heads])
        n_vecs = np.array([n["vector"] for n in all_nouns])
        q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-8
        n_norms = np.linalg.norm(n_vecs, axis=1, keepdims=True) + 1e-8
        sim_matrix = (q_vecs / q_norms) @ (n_vecs / n_norms).T
        head_set = set(h.lower() for h in unique_heads)
        expansions = []
        for h_idx, head in enumerate(unique_heads):
            sims = sim_matrix[h_idx]
            k = min(self.top_k_similar, len(all_nouns))
            top_idx = np.argpartition(sims, -k)[-k:]
            for n_idx in top_idx:
                if sims[n_idx] < self.similarity_threshold:
                    continue
                noun = all_nouns[n_idx]
                if noun["lemma"] in head_set:
                    continue
                neg_flag = "NEG" if noun["grammar"].get("negacion") else ""
                obj_lemma = noun["grammar"].get("obj_lemma", "")
                voice = noun["grammar"].get("voz", "Act")
                fingerprint = " ".join(
                    x for x in [noun["verb_lemma"], obj_lemma, voice, neg_flag] if x
                )
                expansions.append(
                    PredicateFrame(
                        entity_text=noun["text"],
                        entity_head_lemma=noun["lemma"],
                        entity_start_char=noun["char_start"],
                        entity_end_char=noun["char_end"],
                        chain_representative="",
                        verb_lemma=noun["verb_lemma"],
                        verb_text=noun["verb_text"],
                        verb_start_char=0,
                        verb_end_char=0,
                        voice=voice,
                        tense=noun["grammar"].get("tiempo", ""),
                        mood=noun["grammar"].get("modo", ""),
                        negated=bool(noun["grammar"].get("negacion", False)),
                        frame_fingerprint=fingerprint,
                        uce_id=noun["uce_id"],
                        is_expansion=True,
                        original_entity=head,
                    )
                )
        return expansions


@dataclass
class SubordinationClassifier:
    def __init__(self, adverb_classifier=None, sentence_embedder=None):
        self.adverb_clf = (
            adverb_classifier  # puede ser el mismo que clasifica adverbios
        )
        self.embedder = sentence_embedder
        self.cache = {}  # cache por (conj_text, dep, contexto)

    def classify_verb(self, verb_token: Token) -> Dict[str, Any]:
        """
        Retorna dict con:
        - "tipo_subordinacion": str o None
        - "subtipo": str o None
        - "conjuncion": str o None
        - "confianza": float (si usó ML)
        """
        # 1. Buscar conjunción subordinante hija (SCONJ con dep='mark')
        conj = None
        for child in verb_token.children:
            if child.pos_ == "SCONJ" and child.dep_ == "mark":
                conj = child
                break
        if conj is None:
            # Buscar conjunción en ancestros? Normalmente está como hijo.
            return {"tipo_subordinacion": None, "subtipo": None, "conjuncion": None}

        conj_text = conj.text.lower()
        dep = verb_token.dep_
        # Si la dependencia no es de subordinación, no es subordinado
        if dep not in SUBORDINATING_DEPS:
            return {
                "tipo_subordinacion": None,
                "subtipo": None,
                "conjuncion": conj_text,
            }

        # 2. Obtener tipo por regla (léxico + dependencia)
        tipo_info = get_subordination_type(conj_text, dep)
        if tipo_info is not None:
            return {
                "tipo_subordinacion": tipo_info[0],
                "subtipo": tipo_info[1],
                "conjuncion": conj_text,
                "confianza": 1.0,
                "metodo": "regla",
            }

        # 3. Caso ambiguo o no cubierto: usar clasificador de adverbios
        if conj_text in AMBIGUOUS_SUBORDINATORS and self.adverb_clf is not None:
            # Obtener la oración completa donde está el verbo
            sent = verb_token.sent.text
            # Usamos el clasificador de adverbios para ver si la palabra se comporta como subordinante
            # (podríamos entrenar un clasificador específico, pero reutilizamos el de adverbios)
            cat, conf, _ = self.adverb_clf.classify(sent, target=conj_text)
            # Si la categoría es 'conjuntivo' o 'tiempo' o similar, podría ser subordinante
            # Mapeo de categorías de adverbio a tipo de subordinación
            mapping = {
                "tiempo": ("adverbial", "temporal"),
                "lugar": ("adverbial", "lugar"),
                "modo": ("adverbial", "modal"),
                "conjuntivo": ("adverbial", "otra"),
            }
            if cat in mapping and conf >= self.adverb_clf.confidence_threshold:
                tipo, subtipo = mapping[cat]
                return {
                    "tipo_subordinacion": tipo,
                    "subtipo": subtipo,
                    "conjuncion": conj_text,
                    "confianza": conf,
                    "metodo": "ml_adverb",
                }
            else:
                # Si no se reconoce, asumimos completiva por defecto (o None)
                return {
                    "tipo_subordinacion": "completiva",
                    "subtipo": None,
                    "conjuncion": conj_text,
                    "confianza": 0.5,
                    "metodo": "fallback",
                }

        # 4. Fallback final
        return {
            "tipo_subordinacion": "desconocida",
            "subtipo": None,
            "conjuncion": conj_text,
            "confianza": 0.0,
            "metodo": "none",
        }


@dataclass
class DocRecord:
    """
    Minimal per-document record for lexical analysis.
    Built from UCEs — no spaCy objects retained.
    """

    doc_id: str
    tokens: List[str]  # all surface tokens (lowercased, no punct)
    lemmas: List[str]  # all lemmas (lowercased, no punct)
    content_lemmas: List[str]  # NOUN/VERB/ADJ/ADV lemmas only
    metadata: Dict[str, Any]  # age, sex, education, etc. — caller provides


def build_doc_record(
    doc_id: str,
    uces: List,  # List[UCE]
    metadata: Optional[Dict] = None,
) -> DocRecord:
    """
    Constructs a DocRecord from a list of UCEs.
    Call this right after pipeline.procesar() returns.
    No heavy objects are retained — only string lists.
    """
    tokens: List[str] = []
    lemmas: List[str] = []
    content: List[str] = []
    content_pos = {"NOUN", "VERB", "ADJ", "ADV"}

    for uce in uces:
        # uce.tokens includes punctuation; filter it via pos_tags
        for tok, pos in zip(uce.tokens, uce.pos_tags):
            t = tok.lower().strip()
            if not t or pos == "PUNCT":
                continue
            tokens.append(t)

        for lem, pos in zip(uce.lemmas, uce.pos_tags):
            l = lem.lower().strip()
            if not l:
                continue
            lemmas.append(l)
            if pos in content_pos:
                content.append(l)

    return DocRecord(
        doc_id=doc_id,
        tokens=tokens,
        lemmas=lemmas,
        content_lemmas=content,
        metadata=metadata or {},
    )


class GlobalLexicalAnalyzer:
    """
    Accumulates lexical statistics across an entire corpus of documents.

    Map step  : .add_document(doc_record) — O(n_tokens) per call
    Reduce step: analysis methods          — called once after the loop

    Handles five tasks:
      1. Global semantic networks (co-occurrence graphs)
      2. Global Zipf / frequency profile
      3. Corpus-specific vocabulary (OOV in SUBTLEX = "jerga del corpus")
      4. Per-document lexical richness (TTR, Guiraud, MTLD-approx, D-approx)
      5. Sociodemographic variation (cross-tab metadata × lexical metrics)
    """

    # Window size for co-occurrence (tokens on each side of target word)
    COOC_WINDOW: int = 4
    # Minimum co-occurrence count to add an edge to the semantic network
    COOC_MIN_COUNT: int = 3
    # Stopwords to exclude from semantic network nodes
    # (reuses the spaCy Spanish set; caller can override)
    _STOP: Set[str] = set()

    def __init__(
        self,
        subtlex_analyzer=None,  # SubtlexAnalyzer instance (optional)
        cooc_window: int = 4,
        cooc_min_count: int = 3,
        stop_words: Optional[Set[str]] = None,
        we_analyzer=None,
    ):
        self.subtlex = subtlex_analyzer
        self.COOC_WINDOW = cooc_window
        self.COOC_MIN_COUNT = cooc_min_count
        self.we_analyzer = we_analyzer  # ← store it
        # Load Spanish stopwords lazily
        if stop_words is not None:
            self._STOP = stop_words
        else:
            try:
                from spacy.lang.es.stop_words import STOP_WORDS

                self._STOP = set(STOP_WORDS)
            except ImportError:
                self._STOP = set()

        # ── Accumulators ─────────────────────────────────────────────────
        # 1. Co-occurrence: (lemma_a, lemma_b) → count  (symmetric)
        self._cooc: Counter = Counter()
        # 1b. Term → set of doc_ids (for cross-doc edge weighting)
        self._term_docs: Dict[str, Set[str]] = defaultdict(set)

        # 2. Global lemma frequency
        self._global_freq: Counter = Counter()
        # 2b. Global token frequency (for type-token at corpus level)
        self._global_tok: Counter = Counter()

        # 3. OOV tracking: lemma → count (only lemmas absent from SUBTLEX)
        self._oov_freq: Counter = Counter()
        self._oov_docs: Dict[str, Set[str]] = defaultdict(set)

        # 4. Per-document richness (one row per doc)
        self._richness_rows: List[Dict] = []

        # 5. Metadata store (doc_id → metadata dict)
        self._metadata: Dict[str, Dict] = {}

        # Internal: doc_id → DocRecord (lightweight, just lists)
        self._records: Dict[str, DocRecord] = {}

    # ================================================================
    # MAP step — call once per document
    # ================================================================
    def add_document(self, record: DocRecord) -> None:
        """
        Processes one document record.
        All five accumulators are updated in a single pass over tokens.
        """

        doc_id = record.doc_id
        lemmas = record.lemmas
        content = record.content_lemmas
        tokens = record.tokens

        if doc_id in self._records:
            import logging

            logging.getLogger(__name__).warning(
                "GlobalLexicalAnalyzer: doc_id '%s' already added, skipping.", doc_id
            )
            return

        self._records[doc_id] = record
        self._metadata[doc_id] = record.metadata

        # ── 2. Global frequency ───────────────────────────────────────────
        self._global_freq.update(lemmas)
        self._global_tok.update(tokens)

        # ── 1. Co-occurrence over content lemmas only ────────────────────
        # (filtering stopwords keeps the graph semantically meaningful)
        content_clean = [l for l in content if l not in self._STOP and l.isalpha()]
        for i, term in enumerate(content_clean):
            self._term_docs[term].add(doc_id)
            window_start = max(0, i - self.COOC_WINDOW)
            window_end = min(len(content_clean), i + self.COOC_WINDOW + 1)
            for j in range(window_start, window_end):
                if i == j:
                    continue
                a, b = sorted((content_clean[i], content_clean[j]))
                self._cooc[(a, b)] += 1

        # ── 3. OOV ────────────────────────────────────────────────────────
        if self.subtlex is not None:
            for lem in set(content):
                if lem.isalpha() and self.subtlex.is_oov(lem):
                    self._oov_freq[lem] += 1
                    self._oov_docs[lem].add(doc_id)

        # ── 4. Per-document richness ──────────────────────────────────────
        self._richness_rows.append(
            self._compute_richness(doc_id, tokens, lemmas, record.metadata)
        )

    # ================================================================
    # REDUCE step — call once after all documents
    # ================================================================

    # ── Task 1: Global semantic network ──────────────────────────────────
    def semantic_network(
        self,
        min_count: Optional[int] = None,
        weight_by_docs: bool = True,
    ) -> nx.Graph:
        """
        Builds a weighted undirected co-occurrence graph.

        Nodes: content lemmas (not stopwords).
        Edges: weighted by co-occurrence count.
               If weight_by_docs=True, also multiplied by the number of
               documents both terms share (promotes cross-doc relevance).

        Args:
            min_count:     Minimum raw co-occurrence to add an edge.
                           Defaults to self.COOC_MIN_COUNT.
            weight_by_docs: Whether to boost edge weight by shared doc count.
        """
        threshold = min_count if min_count is not None else self.COOC_MIN_COUNT
        G = nx.Graph()

        for (a, b), count in self._cooc.items():
            if count < threshold:
                continue
            # Number of documents where both terms appear
            shared_docs = len(self._term_docs[a] & self._term_docs[b])
            weight = float(count * shared_docs) if weight_by_docs else float(count)
            G.add_edge(a, b, weight=weight, count=count, shared_docs=shared_docs)

        # Node attributes: global frequency + document frequency
        for node in G.nodes():
            G.nodes[node]["freq"] = self._global_freq.get(node, 0)
            G.nodes[node]["n_docs"] = len(self._term_docs.get(node, set()))

        return G

    def semantic_network_summary(
        self,
        top_n_nodes: int = 30,
        min_count: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of the most central nodes in the semantic network.
        Columns: lemma, degree, weighted_degree, betweenness, pagerank, freq, n_docs.
        """
        G = self.semantic_network(min_count=min_count)
        if len(G) == 0:
            return pd.DataFrame()

        degree = dict(G.degree(weight="weight"))
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        pagerank = nx.pagerank(G, weight="weight")

        rows = []
        for node in G.nodes():
            rows.append(
                {
                    "lemma": node,
                    "degree": G.degree(node),
                    "weighted_degree": round(degree.get(node, 0), 3),
                    "betweenness": round(betweenness.get(node, 0), 5),
                    "pagerank": round(pagerank.get(node, 0), 6),
                    "freq": G.nodes[node]["freq"],
                    "n_docs": G.nodes[node]["n_docs"],
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values("pagerank", ascending=False)
            .head(top_n_nodes)
            .reset_index(drop=True)
        )

    def semantic_network_clustered(
        self,
        min_count: Optional[int] = None,
        weight_by_embedding: bool = True,
        louvain_resolution: float = 1.0,
    ) -> Tuple[nx.Graph, List[List[str]]]:
        G = self.semantic_network(min_count=min_count, weight_by_docs=True)
        if len(G) < 3:
            return G, []

        if weight_by_embedding and self.we_analyzer is not None:
            for u, v, data in G.edges(data=True):
                try:
                    vec_u = self.we_analyzer.vector(u)
                    vec_v = self.we_analyzer.vector(v)
                    nu, nv = np.linalg.norm(vec_u), np.linalg.norm(vec_v)
                    if nu > 0 and nv > 0:
                        cos = float(np.dot(vec_u, vec_v) / (nu * nv))
                        data["weight"] = data["weight"] * (0.5 + 0.5 * max(0.0, cos))
                        data["cos_sim"] = round(cos, 4)
                except Exception:
                    pass

        from cdlib import algorithms as cdlib_algs

        coms = cdlib_algs.louvain(G, weight="weight", resolution=louvain_resolution)
        for cid, community in enumerate(coms.communities):
            for node in community:
                G.nodes[node]["community_id"] = cid
        return G, coms.communities

    def community_summary(
        self,
        min_count: Optional[int] = None,
        louvain_resolution: float = 1.0,
        top_n_signature: int = 10,
    ) -> pd.DataFrame:
        G, communities = self.semantic_network_clustered(
            min_count=min_count,
            louvain_resolution=louvain_resolution,
        )
        if not communities:
            return pd.DataFrame()

        pagerank = nx.pagerank(G, weight="weight")
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        global_total = sum(self._global_freq.values()) or 1

        rows = []
        for cid, community in enumerate(communities):
            nodes = set(community)
            subG = G.subgraph(nodes)

            comm_total = sum(self._global_freq.get(n, 0) for n in nodes) or 1
            scored = []
            for lemma in nodes:
                cnt = self._global_freq.get(lemma, 0)
                tf = cnt / comm_total
                idf = math.log((global_total + 1) / (cnt + 1)) + 1
                scored.append((lemma, round(tf * idf, 5)))
            scored.sort(key=lambda x: -x[1])

            top_pr = sorted(nodes, key=lambda n: -pagerank.get(n, 0))[:5]

            bridges = []
            for n in nodes:
                if betweenness.get(n, 0) < 0.01:
                    continue
                foreign = {
                    G.nodes[nb].get("community_id", 0)
                    for nb in G.neighbors(n)
                    if G.nodes[nb].get("community_id", 0) != cid
                }
                if foreign:
                    bridges.append((n, round(betweenness[n], 4)))
            bridges.sort(key=lambda x: -x[1])

            n_nodes = len(nodes)
            max_e = n_nodes * (n_nodes - 1) / 2 or 1
            int_w = sum(d["weight"] for _, _, d in subG.edges(data=True))

            rows.append(
                {
                    "community_id": cid,
                    "size": n_nodes,
                    "top_pagerank": top_pr,
                    "signature_lemmas": [l for l, _ in scored[:top_n_signature]],
                    "signature_scores": [s for _, s in scored[:top_n_signature]],
                    "bridge_nodes": [n for n, _ in bridges[:3]],
                    "internal_density": round(int_w / max_e, 4),
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values("size", ascending=False)
            .reset_index(drop=True)
        )

    def bridge_node_report(
        self,
        min_count: Optional[int] = None,
        betweenness_threshold: float = 0.02,
    ) -> pd.DataFrame:
        G, _ = self.semantic_network_clustered(min_count=min_count)
        if len(G) < 3:
            return pd.DataFrame()

        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        rows = []
        for node, btwn in betweenness.items():
            if btwn < betweenness_threshold:
                continue
            cid = G.nodes[node].get("community_id", 0)
            neighbor_comms = Counter(
                G.nodes[nb].get("community_id", 0) for nb in G.neighbors(node)
            )
            rows.append(
                {
                    "lemma": node,
                    "betweenness": round(btwn, 5),
                    "community_id": cid,
                    "n_communities_bridged": sum(1 for c in neighbor_comms if c != cid),
                    "neighbor_community_ids": sorted(neighbor_comms.keys()),
                    "freq": self._global_freq.get(node, 0),
                    "n_docs": len(self._term_docs.get(node, set())),
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values("betweenness", ascending=False)
            .reset_index(drop=True)
        )

    def export_network_gexf(self, path: str, min_count: Optional[int] = None) -> None:
        """Exports the semantic network as GEXF (readable by Gephi/Cytoscape)."""
        G = self.semantic_network(min_count=min_count)
        nx.write_gexf(G, path)

    # ── Task 2: Global Zipf / frequency profile ───────────────────────────
    def zipf_profile(self) -> pd.DataFrame:
        """
        Zipf distribution of lemmas across the whole corpus.
        Columns: rank, lemma, freq_abs, freq_rel, zipf_empirical,
                 zipf_subtlex (if subtlex available), is_oov.
        """
        total = sum(self._global_freq.values())
        if total == 0:
            return pd.DataFrame()

        rows = []
        for rank, (lemma, count) in enumerate(self._global_freq.most_common(), start=1):
            freq_rel = count / total
            # Empirical Zipf: log10(freq_per_million)
            zipf_emp = math.log10(count / total * 1_000_000 + 1)
            zipf_sub = None
            is_oov = False
            if self.subtlex:
                zipf_sub = round(self.subtlex.zipf(lemma), 3)
                is_oov = self.subtlex.is_oov(lemma)
            rows.append(
                {
                    "rank": rank,
                    "lemma": lemma,
                    "freq_abs": count,
                    "freq_rel": round(freq_rel, 6),
                    "zipf_empirical": round(zipf_emp, 3),
                    "zipf_subtlex": zipf_sub,
                    "n_docs": len(self._term_docs.get(lemma, set())),
                    "is_oov": is_oov,
                }
            )
        return pd.DataFrame(rows)

    def frequency_bands(self) -> pd.DataFrame:
        """
        Distributes lemmas into Zipf frequency bands (same bands as SubtlexAnalyzer).
        Includes empirical band (based on corpus frequency) and SUBTLEX band if available.
        """
        BANDS = [
            ("B1_nuclear", 6.0, 9.0),
            ("B2_alta", 5.0, 6.0),
            ("B3_media", 4.0, 5.0),
            ("B4_baja", 3.0, 4.0),
            ("B5_rara_tecnica", 0.0, 3.0),
        ]
        total = sum(self._global_freq.values())
        if total == 0:
            return pd.DataFrame()

        # Assign each lemma to a band based on SUBTLEX Zipf (preferred) or empirical
        band_counters: Dict[str, Counter] = {b[0]: Counter() for b in BANDS}
        band_counters["B_oov"] = Counter()  # not in any band

        for lemma, count in self._global_freq.items():
            if self.subtlex:
                z = self.subtlex.zipf(lemma)
                is_oov = self.subtlex.is_oov(lemma)
            else:
                z = math.log10(count / total * 1_000_000 + 1)
                is_oov = False

            placed = False
            for band_name, lo, hi in BANDS:
                if lo <= z < hi:
                    band_counters[band_name][lemma] += count
                    placed = True
                    break
            if not placed or is_oov:
                band_counters["B_oov"][lemma] += count

        rows = []
        for band_name, lo, hi in BANDS + [("B_oov", None, None)]:
            n_tokens = sum(band_counters[band_name].values())
            n_types = len(band_counters[band_name])
            rows.append(
                {
                    "banda": band_name,
                    "rango_zipf": f"{lo}–{hi}" if lo is not None else "OOV",
                    "n_tokens": n_tokens,
                    "n_types": n_types,
                    "pct_tokens": round(n_tokens / total * 100, 2) if total else 0.0,
                    "top_5_lemmas": [
                        l for l, _ in band_counters[band_name].most_common(5)
                    ],
                }
            )
        return pd.DataFrame(rows)

    # ── Task 3: Corpus-specific vocabulary (OOV as "jerga") ───────────────
    def corpus_vocabulary(
        self,
        min_doc_freq: int = 2,
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        Returns OOV lemmas that are frequent in THIS corpus but absent from SUBTLEX.
        These are candidates for corpus-specific jargon, neologisms, or dialectal forms.

        Args:
            min_doc_freq: Only return terms appearing in at least N documents.
            top_n:        Maximum number of rows.
        """
        if self.subtlex is None:
            # Without SUBTLEX, fall back to hapax-filtered low-frequency terms
            # (rare globally but appear in multiple docs = corpus-specific)
            global_total = sum(self._global_freq.values())
            rows = []
            for lemma, count in self._global_freq.most_common():
                n_docs = len(self._term_docs.get(lemma, set()))
                if n_docs < min_doc_freq:
                    continue
                freq_rel = count / global_total
                rows.append(
                    {
                        "lemma": lemma,
                        "freq_abs": count,
                        "freq_rel": round(freq_rel, 6),
                        "n_docs": n_docs,
                        "is_oov": False,
                        "zipf_subtlex": None,
                    }
                )
            return pd.DataFrame(rows[:top_n])

        rows = []
        for lemma, count in self._oov_freq.most_common():
            n_docs = len(self._oov_docs.get(lemma, set()))
            if n_docs < min_doc_freq:
                continue
            rows.append(
                {
                    "lemma": lemma,
                    "freq_abs": count,
                    "n_docs": n_docs,
                    "docs": sorted(self._oov_docs[lemma]),
                }
            )

        return pd.DataFrame(rows[:top_n])

    # ── Task 4: Per-document lexical richness ─────────────────────────────
    @staticmethod
    def _compute_richness(
        doc_id: str,
        tokens: List[str],
        lemmas: List[str],
        metadata: Dict,
    ) -> Dict:
        """
        Computes TTR, Guiraud, MTLD-approximation, and D-approximation
        for a single document. All are type-token based — no syntactic info needed.
        """
        n_tok = len(tokens)
        n_lem = len(lemmas)
        if n_tok == 0:
            return {
                "doc_id": doc_id,
                **metadata,
                "n_tokens": 0,
                "n_types": 0,
                "ttr": 0.0,
                "guiraud": 0.0,
                "mtld_approx": 0.0,
                "d_approx": 0.0,
                "hapax_ratio": 0.0,
                "entropy": 0.0,
            }

        type_counts = Counter(lemmas)
        n_types = len(type_counts)
        hapax = sum(1 for v in type_counts.values() if v == 1)
        ttr = n_types / n_lem if n_lem else 0.0
        guiraud = n_types / math.sqrt(n_lem) if n_lem else 0.0
        hapax_ratio = hapax / n_lem if n_lem else 0.0

        # Shannon entropy of lemma distribution
        total = sum(type_counts.values())
        probs = np.array([v / total for v in type_counts.values()])
        entropy = float(scipy_entropy(probs)) if len(probs) > 1 else 0.0

        # MTLD approximation: mean length of runs where TTR stays above 0.72
        # (simplified version without full bidirectional MTLD)
        mtld_threshold = 0.72
        run_len = 0
        run_types: Set[str] = set()
        run_lengths: List[int] = []
        for lem in lemmas:
            run_len += 1
            run_types.add(lem)
            current_ttr = len(run_types) / run_len
            if current_ttr < mtld_threshold:
                if run_len > 1:
                    run_lengths.append(run_len)
                run_len = 0
                run_types = set()
        if run_len > 0:
            run_lengths.append(run_len)
        mtld_approx = float(np.mean(run_lengths)) if run_lengths else float(n_lem)

        # D-approximation: expected TTR at sample size 35 (vocd-D proxy)
        # D ≈ TTR(35) * 35 / (2 * (1 - TTR(35)))  — Malvern & Richards formula
        if n_lem >= 35:
            sample_types = len(set(lemmas[:35]))
            ttr_35 = sample_types / 35
            d_approx = (ttr_35 * 35) / (2 * (1 - ttr_35 + 1e-8))
        else:
            d_approx = (ttr * n_lem) / (2 * (1 - ttr + 1e-8))

        return {
            "doc_id": doc_id,
            **metadata,
            "n_tokens": n_tok,
            "n_lemmas": n_lem,
            "n_types": n_types,
            "ttr": round(ttr, 4),
            "guiraud": round(guiraud, 4),
            "hapax_ratio": round(hapax_ratio, 4),
            "entropy": round(entropy, 4),
            "mtld_approx": round(mtld_approx, 2),
            "d_approx": round(d_approx, 2),
        }

    def richness_dataframe(self) -> pd.DataFrame:
        """Returns all per-document lexical richness metrics as a DataFrame."""
        if not self._richness_rows:
            return pd.DataFrame()
        df = pd.DataFrame(self._richness_rows).set_index("doc_id")
        return df

    # ── Task 5: Sociodemographic variation ────────────────────────────────
    def variation_by_metadata(
        self,
        groupby: str,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Cross-tabulates lexical richness metrics by a metadata variable.

        Args:
            groupby: Key in metadata dict (e.g. "edad", "sexo", "educacion").
            metrics: List of richness metric column names to include.
                     Defaults to all numeric columns.

        Returns:
            DataFrame: group value → mean ± std of each metric.
        """
        df = self.richness_dataframe().reset_index()
        if df.empty or groupby not in df.columns:
            return pd.DataFrame()

        numeric_cols = metrics or [
            c
            for c in [
                "ttr",
                "guiraud",
                "hapax_ratio",
                "entropy",
                "mtld_approx",
                "d_approx",
                "n_tokens",
                "n_types",
            ]
            if c in df.columns
        ]

        # Mean and std per group
        grouped = df.groupby(groupby)[numeric_cols]
        mean_df = grouped.mean().round(4).add_suffix("_mean")
        std_df = grouped.std().round(4).add_suffix("_std")
        count_df = (
            grouped.count().iloc[:, :1].rename(columns={numeric_cols[0]: "n_docs"})
        )
        result = pd.concat([count_df, mean_df, std_df], axis=1).reset_index()
        return result.sort_values(groupby)

    def variation_vocabulary_by_metadata(
        self,
        groupby: str,
        top_n_per_group: int = 20,
    ) -> Dict[str, pd.DataFrame]:
        """
        For each value of a metadata variable, returns the top N lemmas
        (by TF-IDF-like score: freq in group / freq in corpus) that
        distinguish that group from the rest.

        Returns dict: group_value → DataFrame(lemma, score, freq_in_group, freq_global)
        """
        df_rich = self.richness_dataframe().reset_index()
        if df_rich.empty or groupby not in df_rich.columns:
            return {}

        # Group doc_ids by metadata value
        groups: Dict[Any, List[str]] = defaultdict(list)
        for _, row in df_rich.iterrows():
            val = row.get(groupby)
            if val is not None:
                groups[val].append(row["doc_id"])

        corpus_total = sum(self._global_freq.values())
        result: Dict[str, pd.DataFrame] = {}

        for group_val, doc_ids in groups.items():
            # Count lemmas in this group
            group_freq: Counter = Counter()
            for doc_id in doc_ids:
                rec = self._records.get(doc_id)
                if rec:
                    group_freq.update(rec.content_lemmas)

            group_total = sum(group_freq.values())
            if group_total == 0:
                continue

            rows = []
            for lemma, count in group_freq.most_common():
                if not lemma.isalpha():
                    continue
                freq_global = self._global_freq.get(lemma, 1)
                # TF in group / IDF from corpus (log-smoothed)
                tf = count / group_total
                idf = math.log((corpus_total + 1) / (freq_global + 1)) + 1
                score = round(tf * idf, 6)
                zipf_s = None
                if self.subtlex:
                    zipf_s = round(self.subtlex.zipf(lemma), 3)
                rows.append(
                    {
                        "lemma": lemma,
                        "score": score,
                        "freq_in_group": count,
                        "freq_global": freq_global,
                        "zipf_subtlex": zipf_s,
                        "n_docs_group": sum(
                            1
                            for d in doc_ids
                            if lemma
                            in (
                                self._records[d].content_lemmas
                                if d in self._records
                                else []
                            )
                        ),
                    }
                )
            top = (
                pd.DataFrame(rows)
                .sort_values("score", ascending=False)
                .head(top_n_per_group)
                .reset_index(drop=True)
            )
            result[str(group_val)] = top

        return result


# ============================================================================
# 2. SINGLETON DE RECURSOS PESADOS
# ============================================================================


@dataclass
class NLPProvider:
    _nlp = None
    _word_vectors = None
    _sentence_embedder = None
    _adverb_clf = None
    _gpt2 = None
    _subtlex = None
    _stanza_pipeline = None
    _db = None  # Base de datos única

    @classmethod
    def get_nlp(cls, model_name: str = "es_core_news_lg") -> spacy.Language:
        if cls._nlp is None:
            cls._nlp = spacy.load(model_name)
            # Inject the surgeon
            if "fix_colloquial_npi_deps" not in cls._nlp.pipe_names:
                cls._nlp.add_pipe("fix_colloquial_npi_deps", after="parser")
        return cls._nlp

    @classmethod
    def get_word_vectors(cls, config: "Config") -> "WordEmbeddingsAnalyzer":
        if cls._word_vectors is None:
            cls._word_vectors = WordEmbeddingsAnalyzer(cls.get_nlp(), config)
        return cls._word_vectors

    @classmethod
    def get_sentence_embedder(cls, config: "Config") -> Optional[SentenceTransformer]:
        if cls._sentence_embedder is None:
            try:
                cls._sentence_embedder = SentenceTransformer(
                    config.sentence_embedder_model
                )
            except Exception as e:
                logger.warning(f"No se pudo cargar SentenceTransformer: {e}")
        return cls._sentence_embedder

    @classmethod
    def get_adverb_classifier(cls, config: "Config") -> Optional[LogisticRegression]:
        if cls._adverb_clf is None and config.adverb_classifier_dir:
            try:
                path = os.path.join(
                    config.adverb_classifier_dir, "logistic_classifier.joblib"
                )
                cls._adverb_clf = joblib.load(path)
                logger.info("Clasificador de adverbios cargado desde disco.")
            except Exception as e:
                logger.warning(f"No se pudo cargar clasificador de adverbios: {e}")
        return cls._adverb_clf

    @classmethod
    def get_gpt2(cls, config: "Config") -> Optional["SurprisalCalculator"]:
        if cls._gpt2 is None and config.use_surprisal:
            try:
                cls._gpt2 = SurprisalCalculator(
                    config.gpt2_model,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            except Exception as e:
                logger.warning(f"No se pudo cargar GPT-2: {e}")
        return cls._gpt2

    @classmethod
    def get_subtlex(cls, config: Config) -> Optional[SubtlexAnalyzer]:
        if cls._subtlex is None and config.use_subtlex and config.subtlex_path:
            try:
                cls._subtlex = SubtlexAnalyzer(config.subtlex_path)
                logger.info(
                    f"SUBTLEX-ESP cargado: {len(cls._subtlex._data):,} entradas."
                )
            except Exception as e:
                logger.warning(f"No se pudo cargar SUBTLEX-ESP: {e}")
        return cls._subtlex

    @classmethod
    def get_stanza(cls, config: "Config") -> Optional[stanza.Pipeline]:
        """Retorna pipeline Stanza con coref, sin servidor Java. Maneja fallos."""
        if not config.use_coref:
            return None
        if cls._stanza_pipeline is None:
            try:
                stanza.download(
                    config.stanza_lang,
                    processors="tokenize,pos,constituency,coref",
                    verbose=False,
                )
                cls._stanza_pipeline = stanza.Pipeline(
                    config.stanza_lang,
                    processors="tokenize,pos,constituency,coref",
                    use_gpu=config.stanza_use_gpu,
                    verbose=False,
                )
                logger.info("Stanza coref pipeline cargado.")
            except Exception as e:
                logger.warning(
                    f"Error cargando Stanza coref: {e}. Deshabilitando correferencias."
                )
                cls._stanza_pipeline = None
        return cls._stanza_pipeline

    @classmethod
    def get_database(cls, db_path: str) -> "Database":
        if cls._db is None:
            cls._db = Database(db_path)
        return cls._db


# ============================================================================
# 4. SEGMENTACIÓN (sin cambios)
# ============================================================================


def segmentar_reinert_spacy(doc: Doc, min_tokens: int, max_tokens: int) -> List[Span]:
    """Segmenta el Doc en UCEs con ventana deslizante ponderada por puntuación."""
    if len(doc) == 0:
        return []

    weights = []
    for t in doc:
        p = t.text.strip()
        if p in (".", "?", "!", "…"):
            w = 6.0
        elif p == ":":
            w = 5.0
        elif p == ";":
            w = 4.0
        elif p == ",":
            w = 1.0
        else:
            w = 0.01
        weights.append(w)

    window = int(max_tokens * 0.4)
    segments: List[Span] = []
    cursor = 0
    total = len(doc)

    while cursor < total:
        remaining = total - cursor
        if remaining <= max_tokens + window:
            segments.append(doc[cursor:])
            break
        start_s = max(0, max_tokens - window)
        end_s = min(max_tokens + window, remaining)
        if start_s >= end_s:
            segments.append(doc[cursor:])
            break
        indices = np.arange(start_s, end_s)
        pesos = np.array([weights[cursor + i] for i in indices])
        dist = np.abs(indices - max_tokens)
        scores = pesos / (dist + 1)
        cut = cursor + indices[np.argmax(scores)]
        segments.append(doc[cursor : cut + 1])
        cursor = cut + 1

    return [s for s in segments if len(s) >= min_tokens]


# ============================================================================
# 5. EXTRACTORES GRAMATICALES (modificados para usar datos de lang.es)
# ============================================================================


def extraer_negaciones(span: Span, offset_mapper: OffsetMapper) -> List[Dict]:
    resultados = []
    for token in span:
        is_neg = (
            "Neg" in token.morph.get("Polarity") or token.lower_ in PALABRAS_NEGATIVAS
        )
        if not is_neg:
            continue
        alcance = get_negation_scope(token)
        tipo = tipo_negacion(token, alcance)  # <-- nueva función
        local_start = token.idx
        local_end = token.idx + len(token.text)
        global_start, global_end = offset_mapper.to_global(local_start, local_end)
        # Fix: also convert alcance offsets
        alc_g_start, alc_g_end = offset_mapper.to_global(
            alcance.idx, alcance.idx + len(alcance.text)
        )
        npis = [
            {
                "text": t.text,
                "char_start": offset_mapper.to_global(t.idx, t.idx + len(t.text))[0],
                "char_end": offset_mapper.to_global(t.idx, t.idx + len(t.text))[1],
            }
            for t in alcance.subtree
            if t.lower_ in NPI_WORDS
        ]
        resultados.append(
            {
                "texto": token.text,
                "tipo": tipo,
                "alcance": alcance.text,
                "alcance_pos": alcance.pos_,
                "es_constituyente": "_CONSTITUYENTE" in tipo,  # o recalcular
                "npis": npis,
                "char_start": global_start,
                "char_end": global_end,
                "alcance_char_start": alc_g_start,
                "alcance_char_end": alc_g_end,
            }
        )
    return resultados


def ner(
    span: Span, 
    gliner_model, # This is now the loaded model object, not a string
    offset_mapper, 
    labels: List[str] = None, 
    threshold: float = 0.05
) -> Dict[str, List[Dict]]:
    
    if labels is None:
        labels = ["person", "organization", "date", "place"]

    # Since we are processing a single span, use predict_entities instead of batch inference
    entities = gliner_model.predict_entities(
        span.text, 
        labels, 
        threshold=threshold
    )

    r = {}
    
    for entity in entities:
        etiqueta_limpia = entity['label'].split(":")[0].strip()
        
        # Critical: Convert GLiNER's local offsets to your global document offsets
        local_start = entity['start']
        local_end = entity['end']
        global_start, global_end = offset_mapper.to_global(local_start, local_end)
        
        if etiqueta_limpia not in r:
            r[etiqueta_limpia] = []
            
        r[etiqueta_limpia].append({
            'text': entity['text'],
            'score': entity['score'],
            'char_start': global_start,
            'char_end': global_end
        })

    return r

def verificar_licencia_npi(span: Span, uce: UCE) -> List[Dict]:

    advertencias = []
    alcances_neg = [
        (neg["alcance_char_start"], neg["alcance_char_end"]) for neg in uce.negaciones
    ]

    # Detectar expresiones idiomáticas en el texto del span
    texto = span.text
    expr_spans = []
    for expr in EXPRESIONES_IDIOMATICAS_NPI:
        for m in re.finditer(re.escape(expr), texto, re.IGNORECASE):
            expr_spans.append((m.start(), m.end()))

    for token in span:
        if token.lower_ not in NPI_WORDS:
            continue
        # Verificar si el token está dentro de alguna expresión idiomática
        dentro_expr = any(s <= token.idx < e for s, e in expr_spans)
        if dentro_expr:
            continue

        licenciado = any(start <= token.idx <= end for start, end in alcances_neg)
        if not licenciado:
            advertencias.append(
                {
                    "tipo": "NPI_NO_LICENCIADO",
                    "token": token.text,
                    "char_start": token.idx,
                    "char_end": token.idx + len(token.text),
                    "mensaje": f"'{token.text}' aparece sin operador de polaridad negativo (y no está en expresión fija)",
                }
            )
    return advertencias


def extraer_pronombres_y_prodrop(span: Span, offset_mapper: OffsetMapper) -> List[Dict]:
    resultados = []
    for token in span:
        local_start = token.idx
        local_end = token.idx + len(token.text)
        global_start, global_end = offset_mapper.to_global(local_start, local_end)
        pos = token.pos_
        morph = token.morph

        def mg(feat):
            return morph.get(feat)

        # ----- PRONOMBRES EXPLÍCITOS -----
        if pos == "PRON":
            subtipo = clasificar_pronombre_explicito(token)
            dep = correct_pronoun_dependency(token.dep_, mg("Case"))
            persona = mg("Person")
            numero = mg("Number")
            resultados.append(
                {
                    "tipo": "EXPLICITO",
                    "texto": token.text,
                    "subtipo": subtipo,
                    "es_referencial": token.text.lower() not in NON_REFERENTIAL,
                    "persona": persona[0] if persona else "_",
                    "numero": numero[0] if numero else "_",
                    "genero": mg("Gender")[0] if mg("Gender") else "_",
                    "dep": dep,
                    "head": token.head.text,
                    "char_start": global_start,
                    "char_end": global_end,
                }
            )
            continue

        # ----- VERBOS: ENCLÍTICOS Y PRO-DROP -----
        if pos == "VERB":
            # Enclíticos
            encl = extraer_enclitico(token)  # debe estar definida
            if encl:
                base, cl = encl
                resultados.append(
                    {
                        "tipo": "ENCLITICO",
                        "texto": token.text,
                        "base": base,
                        "clitico": cl,
                        "subtipo": "PRONOMBRE_ENCLITICO",
                        "char_start": global_start,
                        "char_end": global_end,
                    }
                )

            # Pro-drop (sujeto nulo)
            if is_prodrop_verb(token):
                persona = mg("Person")
                numero = mg("Number")
                lema_corregido = corregir_lema_para_clitico(
                    token
                )  # función a implementar
                resultados.append(
                    {
                        "tipo": "NULO",
                        "texto": f"[PRO-{lema_corregido}]",
                        "subtipo": "PRO-DROP",
                        "verbo": token.text,
                        "lema": lema_corregido,
                        "persona": persona[0] if persona else "_",
                        "numero": numero[0] if numero else "_",
                        "char_start": global_start,
                        "char_end": global_end,
                    }
                )
            continue

        # ----- CONTRACCIONES -----
        contra = detect_contraction(token)
        if contra:
            resultados.append(
                {
                    "tipo": "CONTRACCION",
                    "texto": token.text,
                    "subtipo": "CONTRACCION_PREP_ART",
                    "preposicion": contra["preposicion"],
                    "articulo": contra["articulo"],
                    "char_start": global_start,
                    "char_end": global_end,
                }
            )

    return resultados


def extraer_verbos_enriquecido(
    span: Span,
    deriver: Optional[MorphDeriver] = None,
    sub_clf: Optional[SubordinationClassifier] = None,
    offset_mapper: Optional[OffsetMapper] = None,
) -> List[Dict]:
    """
    Extrae verbos y auxiliares, y opcionalmente enriquece la morfología
    (aspecto, voz, número, género) usando MorphDeriver.

    Args:
        span: Span de spaCy a analizar.
        deriver: Instancia de MorphDeriver (opcional). Si se provee, se aplica
                la derivación morfológica a los campos que spaCy dejó como None.
    """
    if deriver is None:
        deriver = MorphDeriver()  # Se crea uno local si no se proporciona

    resultados = []
    for token in span:
        if offset_mapper:
            local_start = token.idx
            local_end = token.idx + len(token.text)
            global_start, global_end = offset_mapper.to_global(local_start, local_end)
        if token.pos_ not in ("VERB", "AUX"):
            continue
        morph = token.morph

        def mg(feat):
            vals = morph.get(feat)
            return vals[0] if vals else None

        lema = corregir_lema_para_clitico(token)
        modo = mg("Mood")
        tiempo = mg("Tense")
        aspecto = mg("Aspect")
        voz = mg("Voice")
        persona = mg("Person")
        numero = mg("Number")
        genero = mg("Gender")
        verbform = mg("VerbForm")

        aux_tipo = None
        if token.pos_ == "AUX":
            if token.lemma_ == "ser":
                aux_tipo = (
                    "pasiva"
                    if any(c.dep_ == "advcl" for c in token.children)
                    else "copulativa"
                )
            elif token.lemma_ == "haber":
                aux_tipo = "perfectivo"
            elif token.lemma_ == "estar":
                aux_tipo = "progresivo"
            else:
                aux_tipo = "otro"

        perifrasis = is_periphrastic_construction(token)

        valencia = {
            "nsubj": sum(1 for c in token.children if "subj" in c.dep_),
            "obj": sum(1 for c in token.children if c.dep_ == "obj"),
            "iobj": sum(1 for c in token.children if c.dep_ == "iobj"),
            "obl": sum(1 for c in token.children if c.dep_ == "obl"),
            "ccomp": sum(1 for c in token.children if c.dep_ == "ccomp"),
            "xcomp": sum(1 for c in token.children if c.dep_ == "xcomp"),
        }
        if token.pos_ == "VERB" and (
            verbform is None or verbform not in NON_FINITE_FORMS
        ):
            if valencia["nsubj"] == 0:
                valencia["nsubj"] = 1

        # 1. Morphological derivation
        if deriver:
            derived = deriver.derive(token)
            if aspecto is None:
                aspecto = derived["asp"]
            if voz is None:
                voz = derived["voz"]
            if numero is None:
                numero = derived["numero"]
            if genero is None:
                genero = derived["genero"]

        # 2. 'se' voice override (must run before append so voz is final)
        if voz in (None, "Act"):
            if es_pasiva_refleja(token):
                voz = "PassRefl"
            elif es_impersonal_se(token):
                voz = "Impersonal"
            elif es_media_se(token):
                voz = "Media"

        # 3. Subordination classification (must run before append so sub_info exists)
        sub_info = {}
        if sub_clf is not None:
            sub_info = sub_clf.classify_verb(token)

        # 4. Concordance check (needs verbform, genero, numero — all final by now)
        concordancia = None
        if token.pos_ == "VERB" and verbform == "Part":
            conc = verificar_concordancia_participio(token, genero, numero)
            if conc:
                concordancia = conc

        # 5. Append — everything is final
        resultados.append(
            {
                "texto": token.text,
                "lema": lema,
                "pos": token.pos_,
                "modo": modo,
                "tiempo": tiempo,
                "aspecto": aspecto,
                "voz": voz,
                "persona": persona,
                "numero": numero,
                "genero": genero,
                "aux_tipo": aux_tipo,
                "perifrasis": perifrasis,
                "valencia": valencia,
                "concordancia": concordancia,
                "tipo_subordinacion": sub_info.get("tipo_subordinacion"),
                "subtipo_subordinacion": sub_info.get("subtipo"),
                "conjuncion_subordinante": sub_info.get("conjuncion"),
                "char_start": global_start if offset_mapper else token.idx,
                "char_end": global_end
                if offset_mapper
                else token.idx + len(token.text),
            }
        )

    return resultados


def extract_verbal_frames(
    doc: Doc, offset_mapper: Optional[OffsetMapper] = None
) -> List[Dict]:
    """
    Brute-forces verbal frames out of a text using spaCy dependency parsing.
    """

    frames = []

    for token in doc:
        # Calculate global character positions if an offset mapper is provided
        global_start, global_end = None, None
        if offset_mapper:
            local_start = token.idx
            local_end = token.idx + len(token.text)
            global_start, global_end = offset_mapper.to_global(local_start, local_end)
        # We only care about the verbs.
        if token.pos_ == "VERB":
            frame = {
                "verb_lemma": token.lemma_,
                "original_verb": token.text,
                "char_start": global_start if offset_mapper else token.idx,
                "char_end": global_end
                if offset_mapper
                else token.idx + len(token.text),
                "arguments": {},
            }

            # Look at everything syntactically attached to the verb
            for child in token.children:
                # Subjects (nominal, passive, clausal)
                if child.dep_ in ["nsubj", "nsubjpass", "csubj", "csubjpass"]:
                    frame["arguments"]["subject"] = "".join(
                        [w.text_with_ws for w in child.subtree]
                    ).strip()

                # Direct Objects and Clausal Complements
                elif child.dep_ in ["dobj", "ccomp"]:
                    frame["arguments"]["direct_object"] = "".join(
                        [w.text_with_ws for w in child.subtree]
                    ).strip()

                # Indirect Objects (Dative)
                elif child.dep_ == "dative":
                    frame["arguments"]["indirect_object"] = "".join(
                        [w.text_with_ws for w in child.subtree]
                    ).strip()

                # Prepositional Modifiers (often hold crucial frame data like instruments or locations)
                elif child.dep_ == "prep":
                    prep_phrase = "".join(
                        [w.text_with_ws for w in child.subtree]
                    ).strip()
                    if "prepositional_phrases" not in frame["arguments"]:
                        frame["arguments"]["prepositional_phrases"] = []
                    frame["arguments"]["prepositional_phrases"].append(prep_phrase)

            # Only append if we actually found arguments (ignores auxiliary verbs acting weird)
            if frame["arguments"]:
                frames.append(frame)

    return frames


def extraer_cuantificadores(
    span: Span,
    we_analyzer: WordEmbeddingsAnalyzer,
    use_wordnet: bool = True,
    marcadores_matcher: Optional[PhraseMatcher] = None,
    offset_mapper: Optional[OffsetMapper] = None,
) -> List[Dict]:
    resultados = []
    indices_ocupados: set = set()  # doc-relative, matched against token.i

    if marcadores_matcher:
        for _, doc_start, doc_end in marcadores_matcher(span):
            indices_ocupados.update(range(doc_start, doc_end))  # already doc-relative

    for token in span:
        if token.i in indices_ocupados:  # token.i is doc-relative
            continue
        if token.pos_ not in ("DET", "PRON", "ADJ", "NUM"):
            continue
        global_start, global_end = None, None
        if offset_mapper:
            local_start = token.idx
            local_end = token.idx + len(token.text)
            global_start, global_end = offset_mapper.to_global(local_start, local_end)
        tipo = tipo_cuantificador(token)
        if not tipo:
            # Si no se encontró en léxico, intentar WordNet (solo si use_wordnet)
            if use_wordnet and token.pos_ == "NOUN":
                try:
                    synsets = wn.synsets(token.lemma_, lang="spa")
                    for ss in synsets:
                        hyp_path = ss.hypernym_paths()
                        if hyp_path and "quantity" in hyp_path[0][-1].name():
                            tipo = "CUANTIFICADOR_SEMANTICO"
                            break
                except Exception:
                    pass
        if not tipo:
            continue
        cuantificado = cuantificado_start = cuantificado_end = None
        for child in token.children:
            if child.dep_ in ("det", "nmod", "obj") and child.pos_ in ("NOUN", "PRON"):
                cuantificado, cuantificado_start, cuantificado_end = (
                    child.text,
                    child.idx,
                    child.idx + len(child.text),
                )
                break
        if not cuantificado and tipo == "CUANTIFICADOR_SEMANTICO":
            for child in token.children:
                if child.lower_ == "de":
                    for gc2 in child.children:
                        if gc2.pos_ in ("NOUN", "PRON"):
                            cuantificado, cuantificado_start, cuantificado_end = (
                                gc2.text,
                                gc2.idx,
                                gc2.idx + len(gc2.text),
                            )
                            break
                    break
        ctx_emb = we_analyzer.contextual_word_embedding(token)
        densidad = (
            float(np.linalg.norm(ctx_emb)) if not np.allclose(ctx_emb, 0) else 0.0
        )
        resultados.append(
            {
                "texto": token.text,
                "tipo": tipo,
                "cuantifica_a": cuantificado,
                "pos": token.pos_,
                "context_density": densidad,
                "char_start": global_start if offset_mapper else token.idx,
                "char_end": global_end
                if offset_mapper
                else token.idx + len(token.text),
                "cuantificado_char_start": cuantificado_start,
                "cuantificado_char_end": cuantificado_end,
            }
        )
    return resultados


def extraer_adverbios_robusto(
    span: Span, matcher: PhraseMatcher, offset_mapper: Optional[OffsetMapper] = None
) -> List[Dict]:
    span_offset = span.start  # doc → span-relative conversion factor
    claimed: set = set()  # stores doc-relative indices (matches token.i)
    results: List[Dict] = []

    for _, doc_start, doc_end in matcher(span):
        rel_start = doc_start - span_offset  # span-relative
        rel_end = doc_end - span_offset
        if rel_start < 0 or rel_end > len(span):  # safety guard
            continue
        span_adv = span[rel_start:rel_end]
        if span_adv.root.dep_ in {"obj", "nmod", "nsubj", "nsubj:pass"}:
            continue
        claimed.update(
            range(span_adv.start, span_adv.end)
        )  # span_adv.start is doc-relative
        results.append(
            {
                "text": span_adv.text.lower(),
                "root": span_adv.root,
                "char_start": span_adv.start_char,
                "char_end": span_adv.end_char,
                "is_multiword": True,
            }
        )

    for token in span:
        global_start, global_end = None, None
        if offset_mapper:
            local_start = token.idx
            local_end = token.idx + len(token.text)
            global_start, global_end = offset_mapper.to_global(local_start, local_end)
        if token.pos_ != "ADV" or token.text.lower() in ADVERB_BLOCKED:
            continue
        if token.i in claimed:  # token.i is doc-relative → consistent
            continue
        results.append(
            {
                "text": token.text.lower(),
                "root": token,
                "char_start": global_start if offset_mapper else token.idx,
                "char_end": global_end
                if offset_mapper
                else token.idx + len(token.text),
                "is_multiword": False,
            }
        )
    return results


def extraer_contexto_inteligente(doc: Doc, token_adv: Token) -> str:
    # (sin cambios, igual que original)
    head = token_adv.head
    tokens_clave = [token_adv, head]
    if head.pos_ in ("ADJ", "ADV"):
        tokens_clave.append(head.head)
    else:
        nuclear = {"nsubj", "obj", "iobj", "aux", "cop", "advmod", "prt"}
        for child in head.children:
            if child.dep_ in nuclear:
                tokens_clave.append(child)
                for gc2 in child.children:
                    if gc2.dep_ in {"det", "case", "amod"}:
                        tokens_clave.append(gc2)
    indices = [t.i for t in tokens_clave]
    min_idx = min(indices)
    max_idx = max(indices)
    if (max_idx - min_idx) > 20:
        min_idx = max(0, token_adv.i - 10)
        max_idx = min(len(doc) - 1, token_adv.i + 10)
    return doc[min_idx : max_idx + 1].text


def classify_adverb(
    embedder,
    clf,
    sentence,
    target_adverb=None,
    use_lexicon_hints=True,
    confidence_threshold=0.5,
):
    # (sin cambios, igual que original, pero usa ALL_KNOWN_ADVERBS y ADVERB_CATEGORIES del módulo)
    def detect_adverb_and_context(sent, target):
        if target is not None:
            return target, sent
        sent_lower = sent.lower()
        for adv, _ in ALL_KNOWN_ADVERBS:
            if re.search(r"\b" + re.escape(adv) + r"\b", sent_lower):
                return adv, sent
        return None, sent

    def enrich_with_lexicon_hints(sent, adv):
        if adv is None:
            return sent
        for adv_lex, cat in ALL_KNOWN_ADVERBS:
            if adv == adv_lex:
                return f"{sent} [{cat}:{adv}]"
        return sent

    adverb, ctx = detect_adverb_and_context(sentence, target_adverb)
    if use_lexicon_hints and adverb:
        ctx = enrich_with_lexicon_hints(ctx, adverb)
    emb = embedder.encode([ctx])
    probs = clf.predict_proba(emb)[0]
    idx = np.argmax(probs)
    return (
        clf.classes_[idx],
        float(probs[idx]),
        probs[idx] >= confidence_threshold,
        adverb,
    )


def inicializar_matcher_marcadores(nlp: spacy.Language) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for cat, frases in LOCUCIONES_DISCURSIVAS.items():
        matcher.add(cat, [nlp.make_doc(f) for f in frases])
    return matcher


def extraer_marcadores_discursivos(
    span: Span,
    matcher: PhraseMatcher,
    surprisal_source,
    context_window: str,
    offset_mapper: Optional[OffsetMapper] = None,
) -> Tuple[List[Dict], float, List[Tuple[str, int]]]:
    marcadores: List[Dict] = []
    spans_ocupados: set = set()  # span-relative indices
    span_offset = span.start

    for match_id, doc_start, doc_end in matcher(span):
        rel_start = doc_start - span_offset
        rel_end = doc_end - span_offset
        if rel_start < 0 or rel_end > len(span):
            continue
        span_m = span[rel_start:rel_end]
        categoria = span.vocab.strings[match_id]

        if rel_start == 0 or (rel_start > 0 and span[rel_start - 1].is_punct):
            posicion = "INICIO"
        elif rel_end == len(span) or (rel_end < len(span) and span[rel_end].is_punct):
            posicion = "FINAL"
        else:
            posicion = "MEDIAL"

        entry = {
            "texto": span_m.text,
            "categoria": categoria,
            "posicion": posicion,
            "char_start": span_m.start_char,
            "char_end": span_m.end_char,
        }
        if surprisal_source and context_window:
            tr, iv = surprisal_source.compute_surprisal(context_window, span_m.text)
            entry["surprisal_transicion"] = tr
            entry["surprisal_interno"] = iv
        marcadores.append(entry)
        spans_ocupados.update(range(rel_start, rel_end))  # span-relative

    for i, token in enumerate(span):  # i is span-relative (0-based)
        # Calculate global character positions if an offset mapper is provided
        global_start, global_end = None, None
        if offset_mapper:
            local_start = token.idx
            local_end = token.idx + len(token.text)
            global_start, global_end = offset_mapper.to_global(local_start, local_end)
        if i in spans_ocupados:
            continue
        is_connector = token.pos_ in ("CCONJ", "SCONJ", "INTJ") or (
            token.pos_ == "ADV" and token.text.lower() in CONECTORES_DISC_ADV
        )
        if not is_connector:
            continue
        if i == 0 or (i > 0 and span[i - 1].is_punct):
            posicion = "INICIO"
        elif i == len(span) - 1 or (i < len(span) - 1 and span[i + 1].is_punct):
            posicion = "FINAL"
        else:
            continue
        entry = {
            "texto": token.text,
            "categoria": "PALABRA_SUELTA",
            "posicion": posicion,
            "char_start": global_start if offset_mapper else token.idx,
            "char_end": global_end if offset_mapper else token.idx + len(token.text),
        }
        if surprisal_source and context_window:
            tr, iv = surprisal_source.compute_surprisal(context_window, token.text)
            entry["surprisal_transicion"] = tr
            entry["surprisal_interno"] = iv
        marcadores.append(entry)

    total_palabras = len([t for t in span if not t.is_punct])
    densidad = (len(marcadores) / total_palabras * 100) if total_palabras else 0.0
    top = Counter(m["texto"] for m in marcadores).most_common(5)
    return marcadores, densidad, top


class InsubordinationDetector:
    def __init__(self, nlp):
        self.nlp = nlp
        self.marcadores = set(INSUBORDINACION_FUNCIONES.keys())
        # Incluir variantes con espacios como "pero si"
        self.marcadores_multipalabra = {k for k in self.marcadores if " " in k}

    def detectar_en_oracion(self, sent: Span, mapper: OffsetMapper) -> List[Dict]:
        resultados = []
        start_idx = 0
        while start_idx < len(sent) and sent[start_idx].is_punct:
            start_idx += 1
        if start_idx >= len(sent):
            return resultados

        primer_token = sent[start_idx]
        texto = primer_token.text.lower()
        es_marcador = texto in self.marcadores

        if (
            es_marcador
            and (
                primer_token.pos_ in ("SCONJ", "CCONJ")
                or primer_token.dep_ == "discourse"
            )
            and (primer_token.head.dep_ == "ROOT" or primer_token.dep_ == "ROOT")
        ):
            g_start, g_end = mapper.to_global(
                primer_token.idx, primer_token.idx + len(primer_token.text)
            )
            funcion = INSUBORDINACION_FUNCIONES.get(
                texto, INSUBORDINACION_DEFAULT_FUNCION
            )
            resultados.append(
                {
                    "tipo": "INSUBORDINACION",
                    "marcador": texto,
                    "funcion_pragmatica": funcion,
                    "texto": sent.text,
                    "char_start": g_start,
                    "char_end": g_end,
                }
            )

        for token in sent:
            if token.pos_ == "VERB" and token.dep_ == "discourse":
                g_start, g_end = mapper.to_global(
                    token.idx, token.idx + len(token.text)
                )
                resultados.append(
                    {
                        "tipo": "VERBO_LEXICALIZADO",
                        "marcador": token.text.lower(),
                        "funcion_pragmatica": "Fático/Apelativo",
                        "texto": sent.text,
                        "char_start": g_start,
                        "char_end": g_end,
                    }
                )
                return resultados

        # 2. Verbos lexicalizados como marcadores discursivos (ej. "mira", "fíjate")
        for token in sent:
            if token.pos_ == "VERB" and token.dep_ == "discourse":
                resultados.append(
                    {
                        "tipo": "VERBO_LEXICALIZADO",
                        "marcador": token.text.lower(),
                        "funcion_pragmatica": "Fático/Apelativo",
                        "texto": sent.text,
                        "char_start": token.idx,
                        "char_end": token.idx + len(token.text),
                    }
                )
                return resultados

    def detectar_en_uce(self, span: Span, mapper: OffsetMapper) -> List[Dict]:
        if span is None:
            return []
        resultados = []
        for sent in span.sents:
            resultados.extend(self.detectar_en_oracion(sent, mapper))
        return resultados

    def detectar_en_uce_con_contexto(
        self, span: Span, mapper: OffsetMapper, prev_span: Optional[Span] = None
    ) -> List[Dict]:
        """
        Detecta insubordinaciones en la UCE actual, usando la UCE anterior
        como contexto para marcadores que requieren una cláusula previa.
        """
        resultados = []

        # Primero, detectar insubordinaciones estándar (marcadores al inicio)
        resultados.extend(self._detectar_insubordinaciones_estandar(span, mapper))

        # Segundo, detectar marcadores que requieren contexto previo
        if prev_span is not None:
            resultados.extend(
                self._detectar_insubordinaciones_con_contexto(span, prev_span, mapper)
            )

        return resultados

    def _detectar_insubordinaciones_estandar(
        self, span: Span, mapper: OffsetMapper
    ) -> List[Dict]:
        """Detección básica de insubordinaciones dentro de la misma UCE."""
        resultados = []
        start_idx = 0
        while start_idx < len(span) and span[start_idx].is_punct:
            start_idx += 1
        if start_idx >= len(span):
            return resultados

        primer_token = span[start_idx]
        texto = primer_token.text.lower()
        es_marcador = texto in self.marcadores

        if (
            es_marcador
            and (
                primer_token.pos_ in ("SCONJ", "CCONJ")
                or primer_token.dep_ == "discourse"
            )
            and (primer_token.head.dep_ == "ROOT" or primer_token.dep_ == "ROOT")
        ):
            g_start, g_end = mapper.to_global(
                primer_token.idx, primer_token.idx + len(primer_token.text)
            )
            funcion = INSUBORDINACION_FUNCIONES.get(
                texto, INSUBORDINACION_DEFAULT_FUNCION
            )
            resultados.append(
                {
                    "tipo": "INSUBORDINACION",
                    "marcador": texto,
                    "funcion_pragmatica": funcion,
                    "texto": span.text,
                    "char_start": g_start,
                    "char_end": g_end,
                }
            )

        # Verbos lexicalizados
        for token in span:
            if token.pos_ == "VERB" and token.dep_ == "discourse":
                g_start, g_end = mapper.to_global(
                    token.idx, token.idx + len(token.text)
                )
                resultados.append(
                    {
                        "tipo": "VERBO_LEXICALIZADO",
                        "marcador": token.text.lower(),
                        "funcion_pragmatica": "Fático/Apelativo",
                        "texto": span.text,
                        "char_start": g_start,
                        "char_end": g_end,
                    }
                )
        return resultados

    def _detectar_insubordinaciones_con_contexto(
        self, span: Span, prev_span: Span, mapper: OffsetMapper
    ) -> List[Dict]:
        """
        Detecta insubordinaciones que requieren la UCE anterior.
        Ejemplo: "pero si" al inicio de una UCE, cuando la UCE anterior
        contenía una subordinada.
        """
        resultados = []
        # Tomamos la última oración de la UCE anterior como contexto
        prev_sent = list(prev_span.sents)[-1] if list(prev_span.sents) else prev_span
        # Verificamos si la oración anterior termina con un marcador de subordinación
        # (por ejemplo, una conjunción subordinante o un verbo en modo subjuntivo)
        tiene_subordinacion_anterior = any(
            token.pos_ == "SCONJ"
            or (token.pos_ == "VERB" and token.morph.get("Mood") == ["Subj"])
            for token in prev_sent
        )

        if not tiene_subordinacion_anterior:
            return resultados

        # Ahora buscamos marcadores insubordinados al inicio de la UCE actual
        start_idx = 0
        while start_idx < len(span) and span[start_idx].is_punct:
            start_idx += 1
        if start_idx >= len(span):
            return resultados

        primer_token = span[start_idx]
        texto = primer_token.text.lower()
        # Marcadores que solo son insubordinados cuando la cláusula anterior era subordinada
        marcadores_contextuales = {"pero si", "aunque", "si bien", "por más que"}
        if texto in marcadores_contextuales:
            g_start, g_end = mapper.to_global(
                primer_token.idx, primer_token.idx + len(primer_token.text)
            )
            funcion = INSUBORDINACION_FUNCIONES.get(texto, "Contra-argumentativo")
            resultados.append(
                {
                    "tipo": "INSUBORDINACION_CONTEXTUAL",
                    "marcador": texto,
                    "funcion_pragmatica": funcion,
                    "texto": span.text,
                    "char_start": g_start,
                    "char_end": g_end,
                    "contexto_previo": prev_sent.text[
                        :100
                    ],  # guardar contexto para depuración
                }
            )
        return resultados


# ================================================================
# 2. SPAN ANNOTATION INDEX
# ================================================================


class SpanAnnotationIndex:
    """
    Bidirectional index: PredicateFrame ↔ spans ↔ UCEs ↔ chains ↔ clusters.

    Usage pattern:
        idx = SpanAnnotationIndex()
        idx.add(frame)                    # during extraction
        idx.assign_clusters(ids, labels)  # after clustering
        idx.to_uce_annotations(uce)       # for HTML / export
        idx.summary_by_chain()            # for CorpusAnalyzer
    """

    def __init__(self):
        self._frames: List[PredicateFrame] = []
        self.by_uce: Dict[str, List[PredicateFrame]] = defaultdict(list)
        self.by_chain: Dict[str, List[PredicateFrame]] = defaultdict(list)
        self.by_cluster: Dict[int, List[PredicateFrame]] = defaultdict(list)

    # ── Mutation ─────────────────────────────────────────────────
    def add(self, frame: PredicateFrame) -> None:
        if isinstance(frame, dict):
            frame = PredicateFrame.from_dict(frame)
        self._frames.append(frame)
        self.by_uce[frame.uce_id].append(frame)
        if frame.chain_representative:
            self.by_chain[frame.chain_representative].append(frame)

    def assign_clusters(
        self,
        assignments: List[int],
        labels: Dict[int, str],
    ) -> None:
        self.by_cluster.clear()
        for frame, cid in zip(self._frames, assignments):
            frame.cluster_id = cid
            frame.cluster_label = labels.get(cid, f"cluster_{cid}")
            self.by_cluster[cid].append(frame)

    # ── Queries ──────────────────────────────────────────────────
    def query_char_range(self, start: int, end: int) -> List[PredicateFrame]:
        """All frames whose entity, verb, or object span overlaps [start, end]."""
        out = []
        for f in self._frames:
            spans = [
                (f.entity_start_char, f.entity_end_char),
                (f.verb_start_char, f.verb_end_char),
                (f.direct_object_start, f.direct_object_end),
            ]
            for s, e in spans:
                if s is not None and s < end and e > start:
                    out.append(f)
                    break
        return out

    # ── UCE-level annotation output ──────────────────────────────
    def to_uce_annotations(self, uce) -> List[SpanAnnotation]:
        """
        Returns SpanAnnotation objects with LOCAL offsets for one UCE.
        Safe to call before or after assign_clusters — cluster_id
        will be -1 if called before.
        """
        base = uce.start_char
        anns: List[SpanAnnotation] = []

        for f in self.by_uce.get(uce.id, []):
            local = f.to_local(base)

            def _ann(start_key, end_key, span_type):
                s = local.get(start_key)
                e = local.get(end_key)
                if s is None or e is None or s < 0 or e > len(uce.texto):
                    return None
                return SpanAnnotation(
                    char_start=s,
                    char_end=e,
                    span_type=span_type,
                    cluster_id=f.cluster_id,
                    cluster_label=f.cluster_label,
                    thematic_role=f.thematic_role,
                    frame_fingerprint=f.frame_fingerprint,
                    chain=f.chain_representative,
                    negated=f.negated,
                    voice=f.voice,
                )

            for args in [
                ("entity_start", "entity_end", "ENTITY"),
                ("verb_start", "verb_end", "VERB"),
                ("obj_start", "obj_end", "OBJECT"),
            ]:
                a = _ann(*args)
                if a:
                    anns.append(a)

        return sorted(anns, key=lambda a: a.char_start)

    # ── Corpus-level summaries ────────────────────────────────────
    def summary_by_chain(self) -> pd.DataFrame:
        rows = []
        for rep, frames in self.by_chain.items():
            role_c = Counter(f.thematic_role for f in frames)
            cluster_c = Counter(f.cluster_id for f in frames)
            dom_cid = cluster_c.most_common(1)[0][0] if cluster_c else -1
            dom_role = role_c.most_common(1)[0][0] if role_c else "?"
            n = len(frames)
            rows.append(
                {
                    "entity": rep,
                    "n_frames": n,
                    "dominant_cluster": dom_cid,
                    "dominant_label": frames[0].cluster_label
                    if frames and frames[0].cluster_id == dom_cid
                    else "",
                    "dominant_role": dom_role,
                    "agent_ratio": round(role_c.get("AGENT", 0) / n, 3),
                    "patient_ratio": round(role_c.get("PATIENT", 0) / n, 3),
                    "recipient_ratio": round(role_c.get("RECIPIENT", 0) / n, 3),
                    "cluster_dist": dict(cluster_c),
                    "verbs": sorted({f.verb_lemma for f in frames})[:8],
                    "objects": sorted(
                        {f.direct_object_lemma for f in frames if f.direct_object_lemma}
                    )[:8],
                    "n_uces": len({f.uce_id for f in frames}),
                    "n_expansions": sum(1 for f in frames if f.is_expansion),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("n_frames", ascending=False)
            .reset_index(drop=True)
        )

    def summary_by_cluster(self) -> pd.DataFrame:
        rows = []
        for cid, frames in self.by_cluster.items():
            n = len(frames)
            verb_c = Counter(f.verb_lemma for f in frames)
            obj_c = Counter(
                f.direct_object_lemma for f in frames if f.direct_object_lemma
            )
            role_c = Counter(f.thematic_role for f in frames)
            rows.append(
                {
                    "cluster_id": cid,
                    "cluster_label": frames[0].cluster_label if frames else "",
                    "n_frames": n,
                    "n_chains": len({f.chain_representative for f in frames}),
                    "top_verbs": [v for v, _ in verb_c.most_common(5)],
                    "top_objects": [o for o, _ in obj_c.most_common(5)],
                    "role_dist": dict(role_c),
                    "n_negated": sum(1 for f in frames if f.negated),
                    "voices": dict(Counter(f.voice for f in frames)),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("n_frames", ascending=False)
            .reset_index(drop=True)
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Flat export of every frame — one row per PredicateFrame."""
        return pd.DataFrame([f.to_dict() for f in self._frames])

    def __len__(self) -> int:
        return len(self._frames)


# ================================================================
# 3. THEMATIC ROLE ASSIGNMENT
# ================================================================


def assign_thematic_role(verb_token: Token, entity_span: Span) -> str:
    head = entity_span.root
    dep = head.dep_

    if dep in ("nsubj", "nsubj:pass"):
        morph_voice = verb_token.morph.get("Voice")
        voice = morph_voice[0] if morph_voice else "Act"
        return "PATIENT" if voice == "Pass" else "AGENT"

    if dep in ("obj", "dobj"):
        return "PATIENT"

    if dep in ("iobj", "dative"):
        return "RECIPIENT"

    if dep == "obl":
        for child in head.children:
            if child.dep_ == "case":
                if child.text.lower() in ("por",):
                    return "AGENT"
                if child.text.lower() in ("para", "a"):
                    return "RECIPIENT"
        return "OBLIQUE"

    if dep == "nmod":
        return "MODIFIER"

    return "UNSPECIFIED"


# ================================================================
# 4. FRAME EXTRACTION (one mention → one PredicateFrame)
# ================================================================


def _find_governing_verb(span: Span) -> Optional[Token]:
    """
    Returns the verb that governs the entity span.
    Search order: span root → ancestors → descendants → sent root.
    """
    head = span.root

    if head.pos_ == "VERB":
        return head

    for anc in head.ancestors:
        if anc.pos_ == "VERB":
            return anc

    for tok in span:
        if tok.pos_ == "VERB":
            return tok

    try:
        root = span.sent.root
        if root.pos_ == "VERB":
            return root
    except ValueError:
        pass

    return None


def _subtree_text(token: Token) -> Tuple[str, int, int]:
    """Returns (text, start_char, end_char) for the full subtree of a token."""
    subtree = list(token.subtree)
    return (
        token.doc[subtree[0].i : subtree[-1].i + 1].text,
        subtree[0].idx,
        subtree[-1].idx + len(subtree[-1].text),
    )


def extract_predicate_frame(
    uce,
    mention: Dict,
    doc: Doc,
    verb_index: Dict,
) -> Optional[PredicateFrame]:
    """
    Extracts one PredicateFrame from one coreference mention.
    doc is LOCAL (nlp(uce.texto)); all stored offsets are GLOBAL.

    Args:
        uce:        UCE object (provides .start_char, .id, .texto)
        mention:    dict with keys: text, start_char, end_char,
                    chain_representative (all GLOBAL)
        doc:        spaCy Doc of uce.texto (local coords)
        verb_index: {(global_start, global_end): grammar_dict}
    """
    if uce.start_char is None or uce.start_char < 0:
        logger.warning("UCE %s tiene start_char inválido: %s", uce.id, uce.start_char)
        return None

    # ── 1. Convert global mention offsets to local ───────────────
    local_s = mention["start_char"] - uce.start_char
    local_e = mention["end_char"] - uce.start_char

    if local_s < 0 or local_e > len(uce.texto) or local_s >= local_e:
        return None

    span = doc.char_span(local_s, local_e, alignment_mode="expand")
    if span is None or len(span) == 0:
        return None
    if span.root.i < span.start or span.root.i >= span.end:
        return None

    # ── 2. Find governing verb ───────────────────────────────────
    verb = _find_governing_verb(span)
    if verb is None:
        return None

    # ── 3. Global verb offsets + grammar lookup ──────────────────
    v_g_start = uce.start_char + verb.idx
    v_g_end = uce.start_char + verb.idx + len(verb.text)
    grammar = verb_index.get((v_g_start, v_g_end), {})

    # ── 4. Voice / tense / mood ──────────────────────────────────
    morph = verb.morph

    def mg(feat):
        v = morph.get(feat)
        return v[0] if v else grammar.get(feat.lower(), "")

    voice = grammar.get("voz") or mg("Voice") or "Act"
    tense = grammar.get("tiempo") or mg("Tense") or ""
    mood = grammar.get("modo") or mg("Mood") or ""
    negated = bool(grammar.get("negacion", False))

    # ── 5. Internal arguments ────────────────────────────────────
    d_obj = d_obj_lemma = None
    d_obj_g_start = d_obj_g_end = None
    i_obj = oblique = obl_lemma = None

    for child in verb.children:
        dep = child.dep_

        if dep in ("obj", "dobj") and d_obj is None:
            text, s, e = _subtree_text(child)
            d_obj = text
            d_obj_lemma = child.lemma_.lower()
            d_obj_g_start = uce.start_char + s
            d_obj_g_end = uce.start_char + e

        elif dep in ("iobj", "dative") and i_obj is None:
            i_obj = child.text

        elif dep == "obl" and oblique is None:
            obl_text, _, _ = _subtree_text(child)
            oblique = obl_text
            obl_lemma = child.lemma_.lower()

    # ── 6. Thematic role ─────────────────────────────────────────
    role = assign_thematic_role(verb, span)

    # ── 7. Frame fingerprint (the clustering unit) ───────────────
    obj_part = d_obj_lemma or obl_lemma or ""
    neg_flag = "NEG" if negated else ""
    fingerprint = " ".join(x for x in [verb.lemma_, obj_part, voice, neg_flag] if x)

    # ── 8. Global entity offsets ─────────────────────────────────
    e_g_start = uce.start_char + span.start_char
    e_g_end = uce.start_char + span.end_char

    return PredicateFrame(
        entity_text=mention["text"],
        entity_head_lemma=span.root.lemma_.lower(),
        entity_start_char=e_g_start,
        entity_end_char=e_g_end,
        chain_representative=mention.get("chain_representative", ""),
        verb_lemma=verb.lemma_,
        verb_text=verb.text,
        verb_start_char=v_g_start,
        verb_end_char=v_g_end,
        voice=voice,
        tense=tense,
        mood=mood,
        negated=negated,
        direct_object=d_obj,
        direct_object_lemma=d_obj_lemma,
        direct_object_start=d_obj_g_start,
        direct_object_end=d_obj_g_end,
        indirect_object=i_obj,
        oblique=oblique,
        oblique_lemma=obl_lemma,
        thematic_role=role,
        frame_fingerprint=fingerprint,
        uce_id=uce.id,
        is_expansion=False,
    )


# ============================================================================
# BuscadorAnalogico (usando RAREZAS_PATTERNS desde lang.es)
# ============================================================================


def _generar_huella_sintactica(frase: str, nlp: spacy.Language) -> List[Dict]:
    doc = nlp(frase)
    patron = []
    seen = set()
    for token in doc:
        if token.dep_ == "ROOT":
            patron.append({"RIGHT_ID": "root", "RIGHT_ATTRS": {"POS": token.pos_}})
            seen.add(token.i)
            break
    for token in doc:
        if token.head.dep_ == "ROOT" and token.i not in seen:
            patron.append(
                {
                    "LEFT_ID": "root",
                    "REL_OP": ">",
                    "RIGHT_ID": f"child_{token.i}",
                    "RIGHT_ATTRS": {"LOWER": token.lower_}
                    if not token.is_stop
                    else {"POS": token.pos_},
                }
            )
            seen.add(token.i)
    return patron


class BuscadorAnalogico:
    def __init__(self, nlp: spacy.Language):
        self.nlp = nlp
        self.matcher = DependencyMatcher(nlp.vocab)
        self.biblioteca: Dict[str, str] = {}
        self._cargar_patrones()

    def _cargar_patrones(self):
        for nombre, desc, patron in RAREZAS_PATTERNS:
            self.matcher.add(nombre, [patron])
            self.biblioteca[nombre] = desc

    def aprender_de_ejemplos(self, ejemplos: List[Tuple[str, str]]):
        for nombre, frase in ejemplos:
            patron = _generar_huella_sintactica(frase, self.nlp)
            try:
                self.matcher.remove(nombre)
            except ValueError:
                pass
            self.matcher.add(nombre, [patron])
            self.biblioteca[nombre] = frase

    def procesar_texto(self, texto: str) -> List[Dict]:
        doc = self.nlp(texto)
        matches = self.matcher(doc)
        resultados = []
        for match_id, token_ids in matches:
            label = doc.vocab.strings[match_id]
            s, e = min(token_ids), max(token_ids) + 1
            sp = doc[s:e]
            descrip = self.biblioteca.get(label, "Anomalía sintáctica")
            resultados.append(
                {
                    "tipo": label,
                    "texto": sp.text,
                    "funcion": descrip,
                    "char_start": sp.start_char,
                    "char_end": sp.end_char,
                }
            )
        return resultados


# ============================================================================
# 6. MÉTRICAS AGREGADAS (con corrección de random seed)
# ============================================================================


def calcular_metricas_lexicas(span: Span) -> Dict:
    lemmas = [t.lemma_.lower() for t in span if not t.is_punct]
    if not lemmas:
        return {}
    n = len(lemmas)
    freq = Counter(lemmas)
    nt = len(freq)
    hap = sum(1 for v in freq.values() if v == 1)
    return {
        "ttr": nt / n,
        "guiraud": nt / np.sqrt(n),
        "hapax_abs": hap,
        "hapax_ratio": hap / n,
        "num_tokens": n,
        "num_types": nt,
    }


def semantic_diversity(
    span: Span, we_analyzer: WordEmbeddingsAnalyzer, sample_size: int = 500
) -> float:
    rng = np.random.default_rng(42)  # ← local, no afecta global
    lemmas = [
        t.lemma_.lower()
        for t in span
        if not t.is_punct and not np.allclose(we_analyzer.vector(t.text), 0)
    ]
    if len(lemmas) < 2:
        return 0.0
    if len(lemmas) > sample_size:
        idx = rng.choice(len(lemmas), sample_size, replace=False)
        lemmas = [lemmas[i] for i in idx]
    vecs = [v for v in (we_analyzer.vector(l) for l in lemmas) if not np.allclose(v, 0)]
    if len(vecs) < 2:
        return 0.0
    pw = cdist(vecs, vecs, metric="cosine")
    triu = pw[np.triu_indices_from(pw, k=1)]
    return float(np.mean(triu))


def topic_shift_score(
    prev_span: Span,
    curr_span: Span,
    we_analyzer: WordEmbeddingsAnalyzer,
) -> float:
    def content_vecs(sp):
        return [
            we_analyzer.vector(t.lemma_.lower())
            for t in sp
            if not t.is_stop
            and t.pos_ in ("NOUN", "VERB", "ADJ")
            and not np.allclose(we_analyzer.vector(t.lemma_.lower()), 0)
        ]

    v1, v2 = content_vecs(prev_span), content_vecs(curr_span)
    if not v1 or not v2:
        return 1.0
    c1 = np.mean(v1, axis=0)
    c2 = np.mean(v2, axis=0)
    return float(1 - cosine_similarity([c1], [c2])[0][0])


def calcular_complejidad_sintactica(span: Span) -> Dict:
    doc = span.doc

    def max_depth(token, depth=0, limit=50):
        if depth >= limit or not list(token.children):
            return depth
        return max(max_depth(c, depth + 1, limit) for c in token.children)

    sents_in_span = [
        s for s in doc.sents if s.start >= span.start and s.end <= span.end
    ]
    profundidades = [max_depth(s.root) for s in sents_in_span]
    prof_max = max(profundidades) if profundidades else 0
    subord_deps = {"advcl", "ccomp", "xcomp", "relcl"}
    recursividad = sum(1 for t in span if t.dep_ in subord_deps)
    distancias = [abs(t.i - c.i) for t in span for c in t.children]
    dep_mean = float(np.mean(distancias)) if distancias else 0.0
    n_oraciones = len(sents_in_span) or 1
    ratio_sub = recursividad / n_oraciones
    right_b = sum(1 for t in span for c in t.children if c.i > t.i)
    left_b = sum(1 for t in span for c in t.children if c.i < t.i)
    total = right_b + left_b
    branch = right_b / total if total > 0 else 0.5
    return {
        "profundidad_maxima": prof_max,
        "recursividad": recursividad,
        "distancia_dependencia_media": dep_mean,
        "ratio_subordinacion": ratio_sub,
        "branching_ratio": branch,
    }


# ============================================================================
# 7. ENTRENAMIENTO DEL CLASIFICADOR DE ADVERBIOS (corregido)
# ============================================================================


def generate_training_examples() -> List[Tuple[str, int]]:
    rng = np.random.default_rng(42)  # o usa config.random_state
    examples = [
        (sent, ADVERB_CATEGORIES.index(cat)) for sent, cat in MANUAL_TRAINING_EXAMPLES
    ]

    def _fill(adv_dict, multiplier):
        for cat, adv_set in adv_dict.items():
            if cat not in TRAINING_TEMPLATES:
                continue
            templates = TRAINING_TEMPLATES[cat]
            for adv in list(adv_set)[:50]:
                for _ in range(multiplier):
                    sent = rng.choice(templates)
                    sent = (
                        sent.replace("{subject}", rng.choice(TRAINING_SUBJECTS))
                        .replace("{verb}", rng.choice(TRAINING_VERBS))
                        .replace("{adv}", adv)
                        .replace("{adj}", rng.choice(TRAINING_ADJECTIVES))
                        .replace("{obj}", rng.choice(TRAINING_OBJECTS))
                        .replace(" .", ".")
                        .replace(" ,", ",")
                    )
                    examples.append((sent, ADVERB_CATEGORIES.index(cat)))

    _fill(LEXICON_ADVERBS, LEXICON_MULTIPLIER)
    _fill(MULTI_WORD_ADVERBS, MULTI_WORD_MULTIPLIER)
    _fill({"dominio": LEXICON_ADVERBS["dominio"]}, DOMINIO_EXTRA_MULTIPLIER)

    return list({sent: lbl for sent, lbl in examples}.items())


def train_adverb_classifier(
    output_dir: str, model_name: str = "BAAI/bge-m3", use_lexicon_hints: bool = True
):
    logger.info("Generando ejemplos de entrenamiento para adverbios...")

    model_path = os.path.join(output_dir, "logistic_classifier.joblib")
    if os.path.exists(model_path):
        os.remove(model_path)
        logger.info(f"Modelo antiguo eliminado: {model_path}")

    train_data = generate_training_examples()
    texts, labels = [], []
    for sent, label_idx in train_data:
        # CORRECCIÓN: buscar la categoría correcta para el adverbio de esta oración
        adverb = None
        correct_cat = ADVERB_CATEGORIES[label_idx]
        # Buscar qué adverbio aparece en la frase
        for adv, cat in ALL_KNOWN_ADVERBS:
            if re.search(r"\b" + re.escape(adv) + r"\b", sent.lower()):
                adverb = adv
                # No rompemos el bucle, pero guardamos la categoría correcta
        if use_lexicon_hints and adverb:
            # Usar la categoría correcta (la de la etiqueta) en lugar de la última del bucle
            sent = f"{sent} [{correct_cat}:{adverb}]"
        texts.append(sent)
        labels.append(label_idx)

    embedder = SentenceTransformer(model_name)
    X = embedder.encode(texts, show_progress_bar=True)
    y = np.array(labels)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    logger.info(f"Adverb classifier accuracy: {np.mean(y_pred == y_te):.3f}")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(output_dir, "logistic_classifier.joblib"))
    return embedder, clf


# ============================================================================
# 8. PIPELINE PRINCIPAL (con Database en __init__, sin crear múltiples instancias)
# ============================================================================


class PipelineGramatical:
    def __init__(self, config: Config):
        self.config = config
        self.nlp = NLPProvider.get_nlp(config.spacy_model)
        self.we_analyzer = NLPProvider.get_word_vectors(config)
        self.sentence_embedder = NLPProvider.get_sentence_embedder(config)
        self.adverb_clf = NLPProvider.get_adverb_classifier(config)
        self.insub_detector = InsubordinationDetector(self.nlp)
        self.sub_clf = SubordinationClassifier(
            adverb_classifier=self.adverb_clf, sentence_embedder=self.sentence_embedder
        )
        # Fuente de surprisal (SUBTLex o GPT-2)
        if config.use_subtlex:
            self.surprisal_source = NLPProvider.get_subtlex(config)
        elif config.use_surprisal:
            self.surprisal_source = NLPProvider.get_gpt2(config)
        else:
            self.surprisal_source = None

        # Coreferencias
        self.stanza_pipeline = NLPProvider.get_stanza(config)
        self.coref_resolver = (
            CoreferenceResolver(self.stanza_pipeline, context_units=2)
            if self.stanza_pipeline
            else None
        )
        # Matchers y analizadores
        self.corpus_verbal_frames = []  # para almacenar frames del último procesamiento
        self.marcadores_matcher = build_discourse_matcher(self.nlp)
        self.adverb_matcher = build_adverb_phrase_matcher(self.nlp)
        self.buscador_rarezas = BuscadorAnalogico(self.nlp)
        self.adverb_cache = {}
        self.gliner_model = GLiNER.from_pretrained(config.gliner_model).to(config.device)
        self._oov_analysis = None
        self.db = NLPProvider.get_database(config.db_path)
        self._morph_deriver = MorphDeriver()
        # Analizador de predicados (coref + BERTopic)
        if self.coref_resolver and self.sentence_embedder:
            self.predicate_analyzer = CorefPredicateAnalyzer(
                nlp=self.nlp,
                we_analyzer=self.we_analyzer,
                sentence_embedder=self.sentence_embedder,
                # similarity_threshold=0.7,
                top_k_similar=10,
            )
        else:
            self.predicate_analyzer = None

    def _enriquecer_uce_desde_span(
        self, uce: UCE, span: Span, prev_span: Optional[Span] = None
    ) -> None:
        """Enriquece una UCE con todos los análisis gramaticales, semánticos y de complejidad."""
        mapper = OffsetMapper(uce.start_char or 0)

        # Token surprisals (necesario para SUBTLEX)
        uce.token_surprisals = self._calcular_token_surprisals(span, mapper)

        # Gramática básica
        uce.negaciones = extraer_negaciones(span, mapper)
        uce.entidades = ner(span, gliner_model=self.gliner_model, offset_mapper=mapper)
        uce.pronombres = extraer_pronombres_y_prodrop(span, mapper)
        uce.verbos = extraer_verbos_enriquecido(
            span,
            deriver=self._morph_deriver,
            sub_clf=self.sub_clf,
            offset_mapper=mapper,
        )
        uce.cuantificadores = extraer_cuantificadores(
            span,
            self.we_analyzer,
            self.config.use_wordnet_quantifiers,
            self.marcadores_matcher,
            offset_mapper=mapper,
        )
        uce.adverbios = []
        for adv_m in extraer_adverbios_robusto(
            span, self.adverb_matcher, offset_mapper=mapper
        ):
            ctx = extraer_contexto_inteligente(span.doc, adv_m["root"])
            cat, conf = self._clasificar_adverbio_con_cache(ctx, adv_m["text"])
            uce.adverbios.append(
                {
                    "texto": adv_m["text"],
                    "categoria": cat,
                    "confianza": conf,
                    "es_multipalabra": adv_m["is_multiword"],
                    "char_start": adv_m["char_start"],
                    "char_end": adv_m["char_end"],
                }
            )

        # Marcadores discursivos
        marcadores, densidad, top = extraer_marcadores_discursivos(
            span,
            self.marcadores_matcher,
            self.surprisal_source,
            uce.texto,
            offset_mapper=mapper,
        )
        uce.marcadores_discursivos = marcadores
        uce.metricas_lexicas["densidad_discursiva"] = densidad
        uce.metricas_lexicas["top_marcadores"] = top

        # Insubordinaciones y rarezas
        uce.insubordinaciones = self.insub_detector.detectar_en_uce_con_contexto(
            span, mapper, prev_span
        )
        uce.rarezas = self.buscador_rarezas.procesar_texto(uce.texto)

        # NPI
        advertencias_npi = verificar_licencia_npi(span, uce)
        if advertencias_npi:
            uce.rarezas.extend(advertencias_npi)

        # Métricas léxicas y semánticas
        uce.metricas_lexicas.update(calcular_metricas_lexicas(span))
        uce.diversidad_semantica = semantic_diversity(span, self.we_analyzer)
        if prev_span is not None:
            uce.topic_shift_prev = topic_shift_score(prev_span, span, self.we_analyzer)
        uce.complejidad_sintactica = calcular_complejidad_sintactica(span)

        # Registro (SUBTLEX)
        if self.surprisal_source and hasattr(
            self.surprisal_source, "clasificar_registro"
        ):
            uce.registro = self.surprisal_source.clasificar_registro(uce)

    def procesar_desde_uces(
        self,
        uces_por_doc: List[List[UCE]],
        global_corpus: Optional["GlobalCorpus"] = None,
        lex_analyzer: Optional["GlobalLexicalAnalyzer"] = None,
        corpus_raw: Optional[Dict] = None,
    ):
        """
        Enrich UCEs in-place with grammatical analysis and coreference chains.
        """
        # Build mapping from numeric doc_id to original corpus key
        idx_to_key = {}
        for key, data in corpus_raw.items():
            orden = data.get("indice_orden")
            if orden is not None:
                idx_to_key[int(orden)] = key

        for doc_uces in uces_por_doc:
            if not doc_uces:
                continue
            doc_id = doc_uces[0].doc_id  # int
            origen_key = idx_to_key.get(doc_id, str(doc_id))
            doc_data = corpus_raw.get(origen_key, {})
            full_text = doc_data.get("texto_completo_txt", "")
            metadata = doc_data.get("metadata", {})

            if not full_text:
                print(
                    f"WARNING: No full text for doc {origen_key}. Reconstructing from UCE segments..."
                )
                # Glue the segments together yourself
                full_text = " ".join([u.texto for u in doc_uces])

            if full_text:
                lock_global_offsets(full_text, doc_uces)

            # ──────────────────────────────────────────────────────────────
            # 1. Coreference resolution using the new slice‑based method
            # ──────────────────────────────────────────────────────────────
            all_chains = []
            if self.coref_resolver:
                # Prepare segments list: (text, global_start_char)
                segments = [(uce.texto, uce.start_char) for uce in doc_uces]
                # Reset resolver state for this document
                self.coref_resolver.reset()
                # Process entire document at once – returns all global chains
                all_chains = self.coref_resolver.resolve(full_text, segments)
                all_chains = [_to_dict_chain(ch) for ch in all_chains]
            # ──────────────────────────────────────────────────────────────
            # 2. Assign chains to each UCE (full and filtered)
            # ──────────────────────────────────────────────────────────────
            for uce in doc_uces:
                uce_chains_filtered = filter_chains_for_uce(
                    all_chains, uce.start_char, uce.end_char
                )
                uce.coref_chains = uce_chains_filtered
                # For full chains (all mentions), keep the original all_chains (or a deep copy)
                uce._coref_chains_full = all_chains  # all chains, not filtered

            # ──────────────────────────────────────────────────────────────
            # 3. Enrich each UCE in‑place (grammatical features)
            # ──────────────────────────────────────────────────────────────
            textos = [uce.texto for uce in doc_uces]
            docs_nlp = list(self.nlp.pipe(textos, batch_size=32))
            prev_span = None
            for uce, doc_spacy in zip(doc_uces, docs_nlp):
                span = doc_spacy[:]
                mapper = OffsetMapper(uce.start_char or 0)  # ← construct ONCE per UCE
                self._enriquecer_uce_desde_span(uce, span, prev_span)
                uce.token_surprisals = self._calcular_token_surprisals(span, mapper)
                uce.negaciones = extraer_negaciones(span, mapper)
                uce.entidades = ner(span, gliner_model=self.gliner_model, offset_mapper=mapper)
                uce.pronombres = extraer_pronombres_y_prodrop(span, mapper)
                uce.verbos = extraer_verbos_enriquecido(
                    span,
                    deriver=self._morph_deriver,
                    sub_clf=self.sub_clf,
                    offset_mapper=mapper,
                )
                uce.cuantificadores = extraer_cuantificadores(
                    span,
                    self.we_analyzer,
                    self.config.use_wordnet_quantifiers,
                    self.marcadores_matcher,
                    offset_mapper=mapper,
                )
                marcadores, densidad, _ = extraer_marcadores_discursivos(
                    span,
                    self.marcadores_matcher,
                    self.surprisal_source,
                    uce.texto,
                    offset_mapper=mapper,
                )
                uce.marcadores_discursivos = marcadores

                for adv_m in extraer_adverbios_robusto(
                    span, self.adverb_matcher, offset_mapper=mapper
                ):
                    ctx = extraer_contexto_inteligente(doc_spacy, adv_m["root"])
                    cat, conf = self._clasificar_adverbio_con_cache(ctx, adv_m["text"])
                    uce.adverbios.append(
                        {
                            "texto": adv_m["text"],
                            "categoria": cat,
                            "confianza": conf,
                            "es_multipalabra": adv_m["is_multiword"],
                            "char_start": adv_m[
                                "char_start"
                            ],  # already global from extractor
                            "char_end": adv_m["char_end"],
                        }
                    )

                # Registro (SUBTLEX)
                if self.surprisal_source and hasattr(
                    self.surprisal_source, "clasificar_registro"
                ):
                    enriched_rows = self.surprisal_source.enriquecer_uces(doc_uces)
                    for uce_e, row in zip(doc_uces, enriched_rows):
                        uce_e.metricas_lexicas.update(
                            {k: v for k, v in row.items() if k != "uce_id"}
                        )
                    self._oov_analysis = self.surprisal_source.analisis_oov(doc_uces)

                prev_span = span

            # ──────────────────────────────────────────────────────────────
            # 4. Predicate frame extraction (still no clustering)
            # ──────────────────────────────────────────────────────────────
            if self.predicate_analyzer and any(u.coref_chains for u in doc_uces):
                try:
                    self.predicate_analyzer.extract(doc_uces, doc_id=origen_key)
                except Exception as e:
                    logger.warning(
                        "Predicate extraction failed for doc %s: %s", origen_key, e
                    )

            # ──────────────────────────────────────────────────────────────
            # 5. Update global structures
            # ──────────────────────────────────────────────────────────────
            if global_corpus is not None:
                global_corpus.add_document(origen_key, doc_uces)

            record = build_doc_record(origen_key, doc_uces, metadata=metadata)
            if lex_analyzer is not None:
                lex_analyzer.add_document(record)

            # Clean up
            del docs_nlp
            gc.collect()

    # ------------------------------------------------------------------
    #  Clasificación de adverbios con caché
    # ------------------------------------------------------------------
    def _clasificar_adverbio_con_cache(
        self, contexto: str, adverbio: str
    ) -> Tuple[str, float]:
        key = (contexto, adverbio)
        if key in self.adverb_cache:
            return self.adverb_cache[key]

        if self.sentence_embedder is None or self.adverb_clf is None:
            for adv_lex, cat in ALL_KNOWN_ADVERBS:
                if adv_lex == adverbio.lower():
                    return cat, 1.0
            return "DESCONOCIDO", 0.0

        cat_idx, conf, _, _ = classify_adverb(
            self.sentence_embedder,
            self.adverb_clf,
            contexto,
            target_adverb=adverbio,
            use_lexicon_hints=True,
            confidence_threshold=self.config.adverb_confidence_threshold,
        )
        try:
            cat = ADVERB_CATEGORIES[int(cat_idx)]
        except (ValueError, IndexError):
            cat = str(cat_idx)

        # Mantener tamaño del caché
        if len(self.adverb_cache) >= self.config.adverb_cache_size:
            self.adverb_cache.pop(next(iter(self.adverb_cache)))
        self.adverb_cache[key] = (cat, conf)
        return cat, conf

    # ------------------------------------------------------------------
    #  Construcción de una UCE desde un Span
    # ------------------------------------------------------------------
    def procesar_un_doc(
        self,
        idx: int,
        span: Span,
        prev_span: Optional[Span],
        context_window: str,
        uce_chains: List[Dict],  # renamed: this is already filtered
    ) -> Tuple[UCE, str, str]:  # returns plain text, not Span
        uce = UCE(
            id=f"uce_{idx}",
            texto=span.text,
            start_char=span.start_char,
            end_char=span.end_char,
        )

        # Construct once, pass everywhere
        mapper = OffsetMapper(span.start_char)
        valid_tokens = [t for t in span if not t.is_punct]

        uce.tokens = [t.text for t in valid_tokens]
        uce.lemmas = [corregir_lema_para_clitico(t).lower() for t in valid_tokens]
        uce.pos_tags = [t.pos_ for t in valid_tokens]

        uce.content_lemmas = [
            t.lemma_.lower()
            for t in valid_tokens
            if not t.is_stop and t.pos_ in ("NOUN", "VERB", "ADJ", "ADV")
        ]

        ls = uce.lemmas
        uce.bigrams = [(ls[i], ls[i + 1]) for i in range(len(ls) - 1)]
        uce.trigrams = [(ls[i], ls[i + 1], ls[i + 2]) for i in range(len(ls) - 2)]
        # bigram_stems / trigram_stems require stemming — left to ALCESTE layer
        uce.bigram_stems = []
        uce.trigram_stems = []

        uce.token_surprisals = self._calcular_token_surprisals(span, mapper)

        # Grammatical extraction — mapper passed everywhere
        uce.negaciones = extraer_negaciones(span, mapper)
        uce.pronombres = extraer_pronombres_y_prodrop(span, mapper)
        uce.verbos = extraer_verbos_enriquecido(
            span,
            deriver=self._morph_deriver,
            sub_clf=self.sub_clf,
            offset_mapper=mapper,
        )
        uce.cuantificadores = extraer_cuantificadores(
            span,
            self.we_analyzer,
            self.config.use_wordnet_quantifiers,
            self.marcadores_matcher,
            offset_mapper=mapper,
        )

        # Adverbios
        uce.adverbios = []
        for adv_m in extraer_adverbios_robusto(
            span, self.adverb_matcher, offset_mapper=mapper
        ):
            ctx = extraer_contexto_inteligente(span.doc, adv_m["root"])
            cat, conf = self._clasificar_adverbio_con_cache(ctx, adv_m["text"])
            uce.adverbios.append(
                {
                    "texto": adv_m["text"],
                    "categoria": cat,
                    "confianza": conf,
                    "origen": "ML_MULTIWORD" if adv_m["is_multiword"] else "ML",
                    "char_start": adv_m["char_start"],
                    "char_end": adv_m["char_end"],
                    "es_multipalabra": adv_m["is_multiword"],
                    "contexto": ctx,
                }
            )

        # Marcadores discursivos
        marcadores, densidad, top = extraer_marcadores_discursivos(
            span,
            self.marcadores_matcher,
            self.surprisal_source,
            context_window,
            offset_mapper=mapper,
        )
        uce.marcadores_discursivos = marcadores
        uce.metricas_lexicas["densidad_discursiva"] = densidad
        uce.metricas_lexicas["top_marcadores"] = top

        # Insubordinaciones — pass mapper so offsets are global
        uce.insubordinaciones = self.insub_detector.detectar_en_uce(span, mapper)
        uce.rarezas = self.buscador_rarezas.procesar_texto(span.text)

        # NPI license check
        advertencias_npi = verificar_licencia_npi(span, uce)
        if advertencias_npi:
            uce.rarezas.extend(advertencias_npi)

        # Metrics
        uce.metricas_lexicas.update(calcular_metricas_lexicas(span))
        uce.diversidad_semantica = semantic_diversity(span, self.we_analyzer)
        if prev_span is not None:
            uce.topic_shift_prev = topic_shift_score(prev_span, span, self.we_analyzer)
        uce.complejidad_sintactica = calcular_complejidad_sintactica(span)

        # Coreference: store both filtered and full
        # uce_chains is already filtered to this UCE's mentions
        uce.coref_chains = uce_chains
        uce._coref_chains_full = (
            uce_chains  # same here; procesar() filtered at build time
        )

        # SUBTLEX register
        if self.surprisal_source and hasattr(
            self.surprisal_source, "clasificar_registro"
        ):
            uce.registro = self.surprisal_source.clasificar_registro(uce)

        # Context for next UCE — use char count, not token count
        max_chars = (
            self.config.max_context_tokens * 5
        )  # rough chars-per-token for Spanish
        new_context = span.text
        if len(new_context) > max_chars:
            new_context = new_context[-max_chars:]

        # Return plain string for prev_span tracking to avoid holding Doc in memory
        # Caller must hold the actual Span only as long as needed for topic_shift
        return uce, new_context, span

    # ------------------------------------------------------------------
    #  Procesa en Reii
    # ------------------------------------------------------------------
    def procesar_uces(
        self,
        uces_por_doc: List[List],
        doc_id_attr: str = "doc_id",
        global_corpus=None,
        lex_analyzer=None,
        doc_metadata_map: Optional[Dict] = None,
        corpus_raw=None,
    ) -> None:
        """
        Enriquece in-place las UCEs de ALCESTE con análisis gramatical.
        """
        # Build mapping from numeric doc_id (ALCESTE) to original corpus key
        idx_to_origen = {}
        if doc_metadata_map:
            for doc_idx, meta in doc_metadata_map.items():
                origen = meta.get("origen", str(doc_idx))
                idx_to_origen[int(doc_idx)] = origen
                idx_to_origen[str(doc_idx)] = origen

        # Process each document group separately
        for doc_uces in uces_por_doc:
            if not doc_uces:
                continue

            # Get numeric doc_id from the first UCE
            first_uce = doc_uces[0]
            int_doc_id = getattr(first_uce, doc_id_attr, None)
            if int_doc_id is None:
                logger.warning("UCE sin doc_id, se omite.")
                continue

            # Resolve original corpus key and fetch full document text
            origen_key = idx_to_origen.get(int_doc_id, str(int_doc_id))
            doc_data = corpus_raw.get(origen_key, {}) if corpus_raw else {}
            texto_completo = doc_data.get("texto_completo_txt", "")

            # ----- 1. Coreference resolution (only if full text exists) -----
            all_chains = []
            if self.coref_resolver and texto_completo:
                segmentos_txt = [(u.texto, u.start_char or 0) for u in doc_uces]
                try:
                    all_chains = self.coref_resolver.resolve(
                        normalize_text(texto_completo), segmentos_txt
                    )
                except Exception as e:
                    logger.warning("Coref falló para '%s': %s", origen_key, e)
            else:
                if not texto_completo:
                    logger.warning(
                        "Doc '%s': sin texto_completo. Coref desactivado.", origen_key
                    )

            gc.collect()

            # ----- 2. Parse all UCE texts with spaCy -----
            textos = [uce.texto for uce in doc_uces]
            docs_spacy = list(self.nlp.pipe(textos, batch_size=32))

            # ----- 3. Enrich each UCE (grammatical features + chains) -----
            prev_span = None
            for uce, doc_spacy in zip(doc_uces, docs_spacy):
                span = doc_spacy[:]
                uce_start = uce.start_char if uce.start_char is not None else 0
                uce_end = (
                    uce.end_char
                    if uce.end_char is not None
                    else (uce_start + len(uce.texto))
                )

                # Separate chains: local (for export) vs full (for predicate analysis)
                chains_local = []
                chains_full = []
                for ch in all_chains:
                    mentions_in_uce = [
                        m
                        for m in ch.get("mentions", [])
                        if uce_start <= m["start_char"] < uce_end
                    ]
                    if not mentions_in_uce:
                        continue
                    chains_full.append(ch)
                    chains_local.append({**ch, "mentions": mentions_in_uce})

                uce.coref_chains = chains_local
                uce._coref_chains_full = chains_full  # temporary, not serialized

                # Extract grammatical features
                uce.negaciones = extraer_negaciones(span, uce.offset_mapper)
                uce.pronombres = extraer_pronombres_y_prodrop(span, uce.offset_mapper)
                uce.verbos = extraer_verbos_enriquecido(
                    span, deriver=self._morph_deriver, sub_clf=self.sub_clf
                )
                uce.cuantificadores = extraer_cuantificadores(
                    span,
                    self.we_analyzer,
                    self.config.use_wordnet_quantifiers,
                    self.marcadores_matcher,
                )
                uce.marcadores_discursivos, _, _ = extraer_marcadores_discursivos(
                    span, self.marcadores_matcher, self.surprisal_source, uce.texto
                )
                uce.insubordinaciones = self.insub_detector.detectar_en_uce(
                    span, uce.offset_mapper
                )
                uce.rarezas = self.buscador_rarezas.procesar_texto(uce.texto)
                uce.complejidad_sintactica = calcular_complejidad_sintactica(span)
                uce.metricas_lexicas = calcular_metricas_lexicas(span)
                uce.diversidad_semantica = semantic_diversity(span, self.we_analyzer)

                if prev_span is not None:
                    uce.topic_shift_prev = topic_shift_score(
                        prev_span, span, self.we_analyzer
                    )

                # Adverbios con caché
                uce.adverbios = []
                for adv_m in extraer_adverbios_robusto(span, self.adverb_matcher):
                    ctx = extraer_contexto_inteligente(doc_spacy, adv_m["root"])
                    cat, conf = self._clasificar_adverbio_con_cache(ctx, adv_m["text"])
                    uce.adverbios.append(
                        {
                            "texto": adv_m["text"],
                            "categoria": cat,
                            "confianza": conf,
                            "es_multipalabra": adv_m["is_multiword"],
                            "char_start": adv_m["char_start"],
                            "char_end": adv_m["char_end"],
                        }
                    )

                # Registro (SUBTLEX)
                if self.surprisal_source and hasattr(
                    self.surprisal_source, "clasificar_registro"
                ):
                    uce.registro = self.surprisal_source.clasificar_registro(uce)

                prev_span = span

            # ----- 4. Predicate frame extraction (uses full chains) -----
            if self.predicate_analyzer:
                try:
                    self.predicate_analyzer.extract(doc_uces, doc_id=origen_key)
                except Exception as e:
                    logger.warning(
                        "Predicate extraction falló para doc %s: %s", origen_key, e
                    )

            # ----- 5. Feed GlobalCorpus and GlobalLexicalAnalyzer (once per document) -----
            if global_corpus is not None:
                global_corpus.add_document(origen_key, doc_uces)

            if lex_analyzer is not None:
                shared_meta = (
                    doc_metadata_map.get(int_doc_id, {}) if doc_metadata_map else {}
                )
                record = build_doc_record(origen_key, doc_uces, metadata=shared_meta)
                lex_analyzer.add_document(record)

    # ------------------------------------------------------------------
    #  Análisis de predicados (CorefPredicateAnalyzer)
    # ------------------------------------------------------------------
    def analizar_predicados(self, uces: List, all_chains: List[Dict]) -> None:
        """
        Drop-in replacement for the method in PipelineGramatical.
        Stores CorefPredicateResult on self for downstream access.
        """
        if not self.predicate_analyzer:
            return
        if not any(uce.coref_chains for uce in uces):
            logger.info("No coref chains — skipping predicate analysis.")
            return
        try:
            result = self.predicate_analyzer.analyze_uces(uces)
            self._last_predicate_result = result  # store on pipeline

            # Attach summary to first UCE (legacy compat)
            if uces:
                uces[0].predicate_summary = result.summary_text

            logger.info(
                "Predicate analysis complete. Chains: %d | Clusters: %d | Frames: %d",
                len(result.span_index.by_chain),
                len(result.span_index.by_cluster),
                len(result.span_index),
            )
        except Exception as e:
            logger.warning("Predicate analysis failed: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    #  Análisis SUBTLEX (enriquecimiento y registro)
    # ------------------------------------------------------------------
    def _analizar_subtlex(self, uces: List[UCE]) -> None:
        if not self.config.use_subtlex_analytics or not self.surprisal_source:
            return
        # Enriquecer métricas léxicas
        enriched = self.surprisal_source.enriquecer_uces(uces)
        for uce, data in zip(uces, enriched):
            uce.metricas_lexicas.update(data)
            uce.registro = self.surprisal_source.clasificar_registro(uce)

        self._oov_analysis = self.surprisal_source.analisis_oov(uces)

        # Asignar registro (coloquial / formal / técnico)
        if isinstance(self.surprisal_source, SubtlexAnalyzer):
            for uce in uces:
                uce.registro = self.surprisal_source.clasificar_registro(uce)

    def _calcular_token_surprisals(
        self, span: Span, offset_mapper: OffsetMapper = None
    ) -> Dict[int, float]:
        """
        Calcula surprisal (basado en SUBTLEX) para cada token de contenido.
        Retorna un diccionario {char_start: surprisal}
        """
        result = {}
        if not self.surprisal_source or not hasattr(self.surprisal_source, "surprisal"):
            return result

        for tok in span:
            if tok.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and not tok.is_punct:
                lemma = tok.lemma_.lower()
                try:
                    surp = self.surprisal_source.surprisal(lemma)
                except Exception:
                    surp = 0.0

                # Use the mapper if provided so keys align with global text
                char_start = (
                    offset_mapper.to_global(tok.idx, tok.idx + len(tok.text))[0]
                    if offset_mapper
                    else tok.idx
                )
                result[char_start] = surp

        return result

    def _build_mention_index(
        self,
        all_chains: List[Dict],
        segmentos: list,  # List[Span]
    ) -> Dict[int, List[Dict]]:
        """
        Para cada UCE (índice de segmento) devuelve las cadenas COMPLETAS
        cuya representante aparece al menos una vez en esa UCE.
        Las menciones fuera de la UCE se conservan para que el predicado
        analyzer pueda acceder al contexto completo de la cadena.
        """
        # Mapa start_char_global → índice de segmento (lookup O(1))
        seg_ranges = [
            (s.start_char, s.end_char, idx) for idx, s in enumerate(segmentos)
        ]

        def seg_of(char: int) -> Optional[int]:
            for s_start, s_end, s_idx in seg_ranges:
                if s_start <= char < s_end:
                    return s_idx
            return None

        index: Dict[int, List[Dict]] = defaultdict(list)
        seen_per_seg: Dict[int, set] = defaultdict(
            set
        )  # evita duplicar la misma cadena

        for chain in all_chains:
            rep = chain["representative"]
            for mention in chain["mentions"]:
                s_idx = seg_of(mention["start_char"])
                if s_idx is None:
                    continue
                if rep not in seen_per_seg[s_idx]:
                    # Añadir la cadena COMPLETA (todas sus menciones)
                    index[s_idx].append(chain)
                    seen_per_seg[s_idx].add(rep)

        return index

    # ------------------------------------------------------------------
    #  Método principal
    # ------------------------------------------------------------------
    def procesar(self, texto: str, doc_id: str = "") -> List[UCE]:
        """
        Extracts all linguistic features for ONE document.
        Does NOT cluster predicates — clustering is deferred to GlobalCorpus.

        Args:
            texto:  Raw document text.
            doc_id: Caller-assigned identifier (e.g. interview filename).
                    Used for cross-document write-back after clustering.
        """
        logger.info("Iniciando pipeline para doc '%s'...", doc_id)
        texto_normalizado = normalize_text(texto)
        doc = self.nlp(texto_normalizado)
        segmentos = segmentar_reinert_spacy(
            doc, self.config.min_tokens_por_uce, self.config.max_tokens_por_uce
        )
        logger.info("doc='%s' → %d UCEs", doc_id, len(segmentos))

        all_chains: List[Dict] = []
        if self.coref_resolver:
            all_chains = self.coref_resolver.resolve(
                texto_normalizado,
                segmentos_txt=[(s.text, s.start_char) for s in segmentos],
            )
        logger.info("doc='%s' → %d cadenas coref", doc_id, len(all_chains))

        mention_index = self._build_mention_index(all_chains, segmentos)

        uces: List[UCE] = []
        prev_span = None
        context_window = ""

        for idx, span in enumerate(segmentos):
            uce_chains = mention_index.get(idx, [])
            uce, new_context, prev_span = self.procesar_un_doc(
                idx, span, prev_span, context_window, uce_chains
            )
            uce.doc_id = doc_id  # stamp provenance
            uces.append(uce)
            context_window = new_context

        # Verbal frames (batch, no per-uce re-parse)
        for uce_doc in self.nlp.pipe([u.texto for u in uces], batch_size=20):
            self.corpus_verbal_frames.extend(extract_verbal_frames(uce_doc))

        # ── Extract predicate FRAMES (no clustering yet) ──────────────────
        if self.predicate_analyzer and any(u.coref_chains for u in uces):
            self.predicate_analyzer.extract(uces, doc_id=doc_id)
            # uce.predicate_frames is now populated with raw PredicateFrames
            # cluster_id = -1 on all of them until GlobalCorpus.cluster_all() runs

        self._analizar_subtlex(uces)
        self.db.save_uces(uces)  # save raw (cluster_id=-1 for now)

        del segmentos
        gc.collect()
        return uces


# ============================================================================
# 9. PERSISTENCIA (sin cambios, pero ahora se instancia una sola vez)
# ============================================================================


class Database:
    SCHEMA_VERSION = 1

    def __init__(self, db_path: str):
        self.db_path = db_path
        dirpart = os.path.dirname(db_path)
        if dirpart:
            os.makedirs(dirpart, exist_ok=True)
        self._load()

    def _load(self):
        if not os.path.exists(self.db_path):
            self.data = {"schema_version": self.SCHEMA_VERSION, "uces": []}
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            if "uces" not in self.data:
                raise ValueError("Missing 'uces' key")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Base de datos corrupta ({e}), reiniciando.")
            import shutil

            shutil.copy2(self.db_path, self.db_path + ".corrupt")
            self.data = {"schema_version": self.SCHEMA_VERSION, "uces": []}

    def _save(self):
        tmp = self.db_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        os.replace(tmp, self.db_path)

    def save_uces(self, uces: List[UCE]):
        self.data["uces"] = [u.to_dict() for u in uces]
        self._save()

    def load_uces(self) -> List[UCE]:

        return [UCE.from_dict(d) for d in self.data.get("uces", [])]


# ============================================================================
# 10. VISUALIZACIÓN (sin cambios)
# ============================================================================


def inyectar_etiquetas_html(
    texto_original: str,
    anotaciones: List[Dict],
    clase_default: str = "resaltado",
) -> str:
    for ann in sorted(anotaciones, key=lambda x: x["char_start"], reverse=True):
        s, e = ann["char_start"], ann["char_end"]
        clase = ann.get("clase", clase_default)
        texto_original = (
            texto_original[:s]
            + f'<span class="{clase}">'
            + texto_original[s:e]
            + "</span>"
            + texto_original[e:]
        )
    return texto_original


def filter_chains_for_uce(
    all_chains: List[Dict], uce_start: int, uce_end: int
) -> List[Dict]:
    """
    Return a list of chains where at least one mention falls inside the UCE.
    Mentions are filtered to only those inside the UCE.
    """
    filtered = []
    for ch in all_chains:
        mentions_in_uce = [
            m for m in ch["mentions"] if uce_start <= m["start_char"] < uce_end
        ]
        if mentions_in_uce:
            filtered.append(
                {
                    "representative": ch["representative"],
                    "mentions": mentions_in_uce,
                }
            )
    return filtered


def _to_dict_chain(chain):
    """Convert a Stanza CorefChain or dict to a plain dict with 'representative' and 'mentions'."""
    if hasattr(chain, "representative_text"):  # Stanza object
        return {
            "representative": chain.representative_text,
            "mentions": [
                {
                    "text": m.text,
                    "start_char": m.start_char,
                    "end_char": m.end_char,
                }
                for m in chain.mentions
            ],
        }
    elif isinstance(chain, dict):  # Already a dict – ensure mentions are dicts
        if "mentions" in chain:
            new_mentions = []
            for m in chain["mentions"]:
                if hasattr(m, "start_char"):  # Stanza mention inside dict
                    new_mentions.append(
                        {
                            "text": m.text,
                            "start_char": m.start_char,
                            "end_char": m.end_char,
                        }
                    )
                else:
                    new_mentions.append(m)  # Already dict
            chain["mentions"] = new_mentions
        return chain
    else:
        return {"representative": "", "mentions": []}
