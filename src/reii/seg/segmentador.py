import gc
import hashlib
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import hnswlib
import numpy as np
import requests
import spacy
import stanza
import torch
from rapidfuzz import fuzz
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.language import Language
from transformers import AutoTokenizer, BertModel, BertTokenizer, pipeline

from reii.config import (
    COREF_BATCH_SIZE,
    NLI_MODEL,
    ROOTS_BATCH_SIZE,
    ROOTS_SKIP_COREF,
    SEGMENTER_EMBEDDING_MODEL,
    SPACY_MODEL,
    STANZA_SERVER_COREF_REMOTE_BATCH,
    STANZA_SERVER_CONCURRENCY,
    STANZA_SERVER_FALLBACK_LOCAL,
    STANZA_SERVER_TIMEOUT,
    STANZA_SERVER_TOKEN,
    STANZA_SERVER_URL,
    STANZA_USE_GPU,
)
from reii.seg.coref_checkpoint import adaptive_coref_batch_size
from reii.seg.root_cache import RootSubjectCache, StanzaRootExtractor
from reii.seg.stanza_pipeline import COREF_PROCESSORS, build_stanza_pipeline

# ── Stanza coref bug-fix ──────────────────────────────────────────────────────
try:
    from stanza.models.coref.config import Config as _StanzaCorefCfg

    if not hasattr(_StanzaCorefCfg, "_coref_patched_flag"):
        _StanzaCorefCfg._coref_original_init = _StanzaCorefCfg.__init__

        def _coref_patched_init(self, *args, **kwargs):
            kwargs.setdefault("plateau_epochs", 10)
            _StanzaCorefCfg._coref_original_init(self, *args, **kwargs)

        _StanzaCorefCfg.__init__ = _coref_patched_init
        _StanzaCorefCfg._coref_patched_flag = True
        print("[PATCH] Stanza coref Config parcheado correctamente.")
except ImportError:
    pass


@Language.component("conversational_sbd")
def conversational_sbd(doc):
    # Common conversational pivots in Spanish discourse
    pivots = {
        "entonces",
        "bueno",
        "o sea",
        "además",
        "pero",
        "porque",
        "luego",
        "así que",
    }

    # We iterate through the tokens, looking for specific syntactic patterns
    for i in range(len(doc) - 2):
        token = doc[i]
        if (
            token.lower_ in pivots
            and doc[i + 1].text == ","
            and doc[i + 2].pos_ == "VERB"
        ):
            doc[i].is_sent_start = True
        elif token.lower_ in {"bueno", "o sea", "entonces"}:
            doc[i].is_sent_start = True
    return doc


device = 0 if torch.cuda.is_available() else -1


# ── Filtro rápido de candidatos a correferencia (B2) ─────────────────────────
# Si la ventana no contiene pronombres personales ni nombres propios con
# mayúscula inicial, se salta la llamada a Stanza coref. Chequeo conservador:
# ante la duda (falso positivo) se procesa la ventana.
_COREF_CANDIDATE_RE = re.compile(
    r"(?i:\b(?:él|ella|ellos|ellas|lo|la|le|les|los|las|se|su|sus|me|te|nos|os|mí|mi|ti|sí|conmigo|contigo|consigo)\b)"
    r"|\b[A-ZÁÉÍÓÚÑ][a-záéíóúñü]{1,}\b"
    r"|\b[A-ZÁÉÍÓÚÑ]{2,}\b",
    re.UNICODE,
)


# ─────────────────────────────────────────────────────────────────────────────
# AttentionShiftDetector  (FIXED: self.device now assigned in __init__)
# ─────────────────────────────────────────────────────────────────────────────
class AttentionShiftDetector:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, attn_implementation="eager")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # FIX: was missing
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_length = 512

    def get_attention_weights(self, text):
        if self.device == "cuda" and not torch.cuda.is_available():
            logging.warning("[ASD] CUDA not available, falling back to CPU")
            self.device = "cpu"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        attentions = []
        num_chunks = (len(inputs["input_ids"][0]) // self.max_length) + 1
        for i in range(num_chunks):
            chunk_input = inputs["input_ids"][
                :, i * self.max_length : (i + 1) * self.max_length
            ]
            if chunk_input.size(1) == 0:
                continue
            chunk_attention = self.model(chunk_input, output_attentions=True)
            attentions.append(chunk_attention.attentions)
            del chunk_attention
            torch.cuda.empty_cache()
        attentions = [torch.cat(att, dim=1) for att in zip(*attentions)]
        return attentions

    def compare_attention(self, attentions1, attentions2):
        attention_diff = 0
        min_len = min(len(attentions1), len(attentions2))
        for layer1, layer2 in zip(attentions1[:min_len], attentions2[:min_len]):
            avg_attention1 = layer1.mean(dim=1).squeeze().detach().cpu().numpy()
            avg_attention2 = layer2.mean(dim=1).squeeze().detach().cpu().numpy()
            if avg_attention1.shape != avg_attention2.shape:
                max_len = max(avg_attention1.shape[0], avg_attention2.shape[0])
                avg_attention1 = np.pad(
                    avg_attention1, (0, max_len - avg_attention1.shape[0])
                )
                avg_attention2 = np.pad(
                    avg_attention2, (0, max_len - avg_attention2.shape[0])
                )
            attention_diff += np.abs(avg_attention1 - avg_attention2).mean()
        return attention_diff

    def detect_topic_shift(self, segment1, segment2, threshold=0.5):
        attentions1 = self.get_attention_weights(segment1)
        attentions2 = self.get_attention_weights(segment2)
        attention_diff = self.compare_attention(attentions1, attentions2)

        return attention_diff > threshold


# ─────────────────────────────────────────────────────────────────────────────
# ClassicSegmenter  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
class ClassicSegmenter:
    def __init__(self, embedding_model, spacy_model):
        self.nlp = spacy.load(spacy_model)
        self.embedding_model = embedding_model
        self.embedding_cache = {}
        self.nli_model = pipeline(
            "zero-shot-classification", model=NLI_MODEL, device=device
        )

    def semantic_cohesion_score(self, segment1, segment2):
        doc1 = self.nlp(segment1)
        doc2 = self.nlp(segment2)
        ents1 = {ent.text for ent in doc1.ents}
        ents2 = {ent.text for ent in doc2.ents}
        if not (ents1 or ents2):
            return 0.0
        intersection = ents1 & ents2
        union = ents1 | ents2
        return len(intersection) / len(union)

    def compute_semantic_shift(self, segment1, segment2):
        try:
            if not segment1 or not segment2:
                return 1.0
            seg1_tail = segment1[-1000:] if len(segment1) > 1000 else segment1
            result = self.nli_model(seg1_tail, candidate_labels=[segment2])
            return result["scores"][0] if "scores" in result else 1.0
        except Exception as e:
            print(f"[ClassicSeg] Error computing semantic shift: {e}")
            return 1.0

    def compute_boundary_score(self, segment1, segment2):
        # ── NEW: Substance Filter for Overflow Cuts ──
        doc2 = self.nlp(segment2)
        content_pos = {"VERB", "NOUN", "ADJ", "ADV"}

        # If the sentence is just conversational filler, force an immediate merge
        if sum(1 for t in doc2 if t.pos_ in content_pos) < 4:
            return -1.0

        # ── Standard Math ──
        emb1 = self._get_cached_embedding(segment1)
        emb2 = self._get_cached_embedding(segment2)

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        # Handle zero division gracefully
        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)

        cohesion = self.semantic_cohesion_score(segment1, segment2)

        return 0.5 * (1 - similarity) + 0.5 * (1 - cohesion)

    def _get_cached_embedding(self, text):
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.embedding_model.encode(
                [text], convert_to_numpy=True
            )[0]
        return self.embedding_cache[text]

    def robust_sentence_split(self, text):
        doc = self.nlp(text)
        return [
            {"text": sent.text.strip(), "start": sent.start_char, "end": sent.end_char}
            for sent in doc.sents
        ]

    def segment_sentences(self, sentences, threshold=0.8):
        segments = [[sentences[0]]]
        for i in range(1, len(sentences)):
            current_sentence = sentences[i]["text"]
            last_segment_text = " ".join([s["text"] for s in segments[-1]])
            if (
                self.compute_boundary_score(last_segment_text, current_sentence)
                < threshold
            ):
                segments[-1].append(sentences[i])
            else:
                segments.append([sentences[i]])
        return segments

    def segment_text(self, text, max_segments=10, threshold=0.5):
        sentences = self.robust_sentence_split(text)
        if not sentences:
            return [text]
        grouped = self.segment_sentences(sentences, threshold)

        # Precompute and cache boundary scores
        boundary_scores = [
            self.compute_boundary_score(
                " ".join(s["text"] for s in grouped[i]),
                " ".join(s["text"] for s in grouped[i + 1]),
            )
            for i in range(len(grouped) - 1)
        ]

        while len(grouped) > max_segments:
            merge_idx = int(np.argmin(boundary_scores))
            # Merge and update only the affected neighbors
            grouped[merge_idx] += grouped.pop(merge_idx + 1)
            boundary_scores.pop(merge_idx)
            if merge_idx < len(boundary_scores):
                boundary_scores[merge_idx] = (
                    self.compute_boundary_score(
                        " ".join(s["text"] for s in grouped[merge_idx]),
                        " ".join(s["text"] for s in grouped[merge_idx + 1]),
                    )
                    if merge_idx + 1 < len(grouped)
                    else float("inf")
                )

        return [" ".join(s["text"] for s in seg) for seg in grouped]


# ─────────────────────────────────────────────────────────────────────────────
# ProgressiveSegmenter  — with full debug instrumentation on coref pipeline
# ─────────────────────────────────────────────────────────────────────────────
_PREMERGE_CACHE_MAXSIZE = 2000


class ProgressiveSegmenter:
    def __init__(
        self,
        model_name=SEGMENTER_EMBEDDING_MODEL,
        stanza_lang="es",
        spacy_model=SPACY_MODEL,
        similarity_threshold=0.6,
        max_depth=3,
        window_size=3,
        hnsw_ef=200,
        hnsw_m=16,
        device=None,
        # ── new: control debug verbosity ────────────────────────────
        debug_coref=True,
        # sujetos raíz de COREF: None -> valores de config (REII_ROOTS_*)
        roots_skip_coref=None,
        roots_batch_size=None,
        # piso de sustancia del pre-merge (antes 4 hardcodeado en detect_topic_shift)
        min_content_tokens: int = 4,
        min_segment_chars: int = 15,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_content_tokens = int(min_content_tokens)
        self.min_segment_chars = int(min_segment_chars)
        if self.min_content_tokens != 4:
            import warnings

            warnings.warn(
                "min_content_tokens != 4 cambia el pre-merge, y por tanto los "
                "grupos/ventanas del coref. La clave del caché de cadenas "
                "(COREF_CHAINS_CACHE_PATH) NO incluye este parámetro: bórralo "
                "antes de re-correr o se reutilizarán cadenas calculadas con "
                "otra segmentación.",
                stacklevel=2,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_depth = max_depth
        self.index = None
        self.embeddings = None
        self.window_size = window_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hnsw_ef = hnsw_ef
        self.hnsw_m = hnsw_m
        self.stanza_lang = stanza_lang
        # Stanza se mantiene en CPU por defecto (config/env REII_STANZA_USE_GPU,
        # default false) para no desplazar presión de RAM a VRAM. Antes se derivaba
        # de self.device, que ahora es CUDA cuando hay GPU, lo que forzaba Stanza a
        # GPU casi siempre y podía provocar OOM de VRAM en inferencia.
        self.stanza_use_gpu = STANZA_USE_GPU
        self.debug_coref = debug_coref  # toggle for coref debug prints
        # True si la ÚLTIMA llamada a _extract_global_chains terminó antes de
        # procesar todas las sub-ventanas (Stanza/servidor no disponible). Permite a
        # los llamadores NO persistir ese resultado parcial como si fuera completo.
        self.last_extraction_incomplete = False

        self.model = SentenceTransformer(model_name).to(self.device)
        self.classicseg = ClassicSegmenter(self.model, spacy_model=spacy_model)
        self.nlp = spacy.load(spacy_model)
        if "conversational_sbd" not in self.nlp.pipe_names:
            # We insert it BEFORE the parser so the dependency parser respects these cuts
            self.nlp.add_pipe("conversational_sbd", before="parser")
        if "conversational_sbd" not in self.classicseg.nlp.pipe_names:
            self.classicseg.nlp.add_pipe("conversational_sbd", before="parser")
        # self.attention_detector = AttentionShiftDetector()  # now safe (device fixed)
        self.tfidf_vectorizer = TfidfVectorizer()
        self._stanza_pipeline = None
        # Sujetos raíz: sin coref (subconjunto de procesadores del pipeline ya cargado)
        # y, opcionalmente, por lotes. La caché (últimos 500 chars) la comparten B1 y B2.
        self._roots = StanzaRootExtractor(
            self.get_stanza,
            skip_coref=ROOTS_SKIP_COREF
            if roots_skip_coref is None
            else roots_skip_coref,
            remote_url=STANZA_SERVER_URL,
            remote_token=STANZA_SERVER_TOKEN,
            remote_timeout=STANZA_SERVER_TIMEOUT,
            remote_fallback_local=STANZA_SERVER_FALLBACK_LOCAL,
            lang=self.stanza_lang,
        )
        _bs = ROOTS_BATCH_SIZE if roots_batch_size is None else int(roots_batch_size)
        self.root_cache = RootSubjectCache(
            self.find_subjects_for_roots,
            batch_extractor=self._roots.batch if _bs > 1 else None,
            batch_size=max(1, _bs),
        )
        # Caché de pre-merge (recursive_segmentation + final_clustering)
        # por documento. Vive toda la vida del ProgressiveSegmenter, que es
        # única por corrida del pipeline (main + Optimizador comparten la
        # misma instancia). Ver premerge_cached() para el análisis de pureza.
        self._premerge_cache: dict = {}
        logging.info(f"[Init] Device: {self.device}")

    # ── debug helper ──────────────────────────────────────────────────────────
    def _dprint(self, msg):
        """Print only when debug_coref is True."""
        if self.debug_coref:
            print(f"[COREF] {msg}")

    def _empty_cuda_cache(self):
        """Free fragmented VRAM after a Stanza window failure (guarded)."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        # CPU-only: empty_cache() es un no-op; gc.collect() libera los tensores
        # de PyTorch retenidos por los Document de Stanza.
        gc.collect()

    # ─────────────────────────────────────────────────────────────────────────
    # Stanza lazy-load  FIX: was writing self.stanza_pipeline (no underscore)
    # which meant _stanza_pipeline was never set → pipeline rebuilt every call.
    # ─────────────────────────────────────────────────────────────────────────
    def get_stanza(self) -> Optional[stanza.Pipeline]:
        if self._stanza_pipeline is None:
            print("[COREF] Stanza pipeline not loaded — iniciando carga...")
            self._stanza_pipeline = build_stanza_pipeline(
                self.stanza_lang, COREF_PROCESSORS, self.stanza_use_gpu
            )
        else:
            self._dprint("Pipeline ya en memoria — reutilizando.")
        return self._stanza_pipeline

    def preprocess_text(self, text: str, min_chars: int = 3) -> list:
        if not text:
            return []

        # Standardize and deep clean
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[“”«»]", '"', text)
        text = re.sub(r"[‘’`]", "'", text)
        text = text.replace(',"', "'").replace('""', "'").replace('"', "'")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text).strip()

        # Temporarily increase limits for massive run-on blocks
        original_max = self.nlp.max_length
        self.nlp.max_length = max(original_max, len(text) + 100)

        try:
            doc = self.nlp(text)
            sentences = []

            for sent in doc.sents:
                clean_sent = sent.text.strip()
                if len(clean_sent) >= min_chars:
                    sentences.append(clean_sent)

        finally:
            self.nlp.max_length = original_max

        logging.info(
            f"[Preprocess] {len(sentences)} oraciones extraídas de {len(text)} caracteres."
        )
        return sentences

    def generate_embeddings(self, sentences):
        embeddings = self.model.encode(
            sentences,
            batch_size=min(64, len(sentences)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings

    def compute_similarities(self, embeddings):
        return cosine_similarity(embeddings)

    def contextual_coherence(self, embeddings, sentences):
        if len(sentences) <= 1:
            return np.array([])
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix.diagonal(offset=1)

    def hierarchical_clustering(self, similarities):
        num_elements = similarities.shape[0]
        distance_matrix = 1 - similarities
        if num_elements > 1000:
            condensed_distance = squareform(distance_matrix, checks=False)
        else:
            condensed_distance = distance_matrix
        linkage_matrix = linkage(condensed_distance, method="ward")
        clusters = fcluster(
            linkage_matrix, t=self.similarity_threshold, criterion="distance"
        )
        boundaries = np.where(np.diff(clusters) != 0)[0] + 1
        logging.info(f"[HierClust] {len(boundaries)} límites detectados.")
        return boundaries

    def recursive_segmentation(self, sentences, depth=0):
        if depth > self.max_depth or len(sentences) <= 1:
            return [" ".join(sentences)]

        # OOM guard: switch to windowed segmentation for large inputs
        if len(sentences) > 500:
            self._dprint(
                f"[RecSeg] {len(sentences)} sentences at depth {depth} — switching to windowed segmentation."
            )
            return self._windowed_segmentation(sentences)

        embeddings = self.generate_embeddings(sentences)

        # OOM guard: full N×N matrix is O(N²) — cap it
        if len(sentences) > 200:
            # Windowed cosine: only compare each sentence to its neighbors
            sim_matrix = self._windowed_similarity(embeddings, window=10)
        else:
            sim_matrix = cosine_similarity(embeddings)

        dist_matrix = 1.0 - np.clip(sim_matrix, 0, 1)

        from scipy.spatial.distance import squareform

        condensed = squareform(dist_matrix, checks=False)

        from scipy.cluster.hierarchy import fcluster, linkage

        Z = linkage(condensed, method="ward")

        cut_height = np.percentile(Z[:, 2], 60)
        labels = fcluster(Z, t=cut_height, criterion="distance")

        boundaries = np.where(np.diff(labels) != 0)[0] + 1

        segments, start = [], 0
        for b in boundaries:
            segments.append(self.recursive_segmentation(sentences[start:b], depth + 1))
            start = b
        segments.append(self.recursive_segmentation(sentences[start:], depth + 1))
        return [item for sub in segments for item in sub]

    def _windowed_similarity(
        self, embeddings: np.ndarray, window: int = 10
    ) -> np.ndarray:
        """
        Builds an approximate similarity matrix by only computing cosine similarity
        within a local window. O(N×window) instead of O(N²).
        Off-window entries default to 0 (maximum distance).
        """
        n = len(embeddings)
        sim = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            start = max(0, i - window)
            end = min(n, i + window + 1)
            chunk = embeddings[start:end]
            sims = cosine_similarity(embeddings[i : i + 1], chunk)[0]
            sim[i, start:end] = sims
            sim[start:end, i] = sims  # keep symmetric
        np.fill_diagonal(sim, 1.0)
        return sim

    def _windowed_segmentation(self, sentences: list, window: int = 20) -> list:
        """
        Linear-pass segmentation fallback for very large sentence lists.
        Compares each sentence only to the mean embedding of the previous window.
        O(N) memory, O(N×window) compute.
        """
        if not sentences:
            return []

        segments = [[sentences[0]]]
        for i in range(1, len(sentences)):
            window_sentences = segments[-1][-window:]
            window_emb = self.generate_embeddings(window_sentences)
            current_emb = self.generate_embeddings([sentences[i]])

            mean_window = window_emb.mean(axis=0, keepdims=True)
            sim = cosine_similarity(mean_window, current_emb)[0][0]

            if sim >= self.similarity_threshold:
                segments[-1].append(sentences[i])
            else:
                segments.append([sentences[i]])

        return [" ".join(seg) for seg in segments]

    def detect_topic_shift(
        self, segment1, segment2, last_n=3, min_content_tokens=None
    ):
        import math
        from collections import Counter

        if min_content_tokens is None:
            min_content_tokens = getattr(self, "min_content_tokens", 4)

        if not hasattr(self, "similarity_history"):
            self.similarity_history = []
            self.syntactic_diff_history = []
            self.lex_sim_history = []

        seg1_tail = segment1[-500:] if len(segment1) > 500 else segment1
        first_period = seg1_tail.find(".")
        if first_period != -1 and len(seg1_tail) > 300:
            seg1_tail = seg1_tail[first_period + 1 :].strip()

        # ── 1. SUBSTANCE FILTER FIRST (Speed Optimization) ──
        doc1 = self.nlp(seg1_tail)
        doc2 = self.nlp(segment2)

        content_pos = {"VERB", "NOUN", "ADJ", "ADV"}
        content_count1 = sum(1 for token in doc1 if token.pos_ in content_pos)
        content_count2 = sum(1 for token in doc2 if token.pos_ in content_pos)

        # If it lacks semantic density, abort the heavy math and force a merge (return False)
        if content_count1 < min_content_tokens or content_count2 < min_content_tokens:
            self.similarity_history.append(1.0)
            self.syntactic_diff_history.append(0.0)
            self.lex_sim_history.append(1.0)
            return False

        # ── 2. HEAVY MATH ONLY IF IT PASSES THE FILTER ──
        embs = self.model.encode(
            [seg1_tail, segment2],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        emb1, emb2 = embs[0], embs[1]
        similarity = float(cosine_similarity(emb1[None, :], emb2[None, :])[0][0])
        try:
            tfidf1 = self.tfidf_vectorizer.transform([seg1_tail])
            tfidf2 = self.tfidf_vectorizer.transform([segment2])
            lex_sim = cosine_similarity(tfidf1, tfidf2)[0][0]
        except Exception:
            lex_sim = 0.0

        dep_counts1 = Counter([token.dep_ for token in doc1])
        dep_counts2 = Counter([token.dep_ for token in doc2])
        all_labels = set(dep_counts1.keys()).union(set(dep_counts2.keys()))
        vec1 = [
            dep_counts1[label] / len(doc1) if len(doc1) > 0 else 0
            for label in all_labels
        ]
        vec2 = [
            dep_counts2[label] / len(doc2) if len(doc2) > 0 else 0
            for label in all_labels
        ]
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        mag1 = math.sqrt(sum(v**2 for v in vec1))
        mag2 = math.sqrt(sum(v**2 for v in vec2))
        syntactic_diff = (
            1.0 if mag1 == 0 or mag2 == 0 else 1.0 - (dot_product / (mag1 * mag2))
        )

        MAX_HISTORY = 200
        if len(self.similarity_history) > MAX_HISTORY:
            self.similarity_history = self.similarity_history[-MAX_HISTORY:]
            self.syntactic_diff_history = self.syntactic_diff_history[-MAX_HISTORY:]
            self.lex_sim_history = self.lex_sim_history[-MAX_HISTORY:]

        self.similarity_history.append(similarity)
        self.syntactic_diff_history.append(syntactic_diff)
        self.lex_sim_history.append(lex_sim)

        return (similarity < 0.6) or (syntactic_diff > 0.4)

    # ── Sustancia mínima de un segmento (mismo filtro que detect_topic_shift) ──
    def _content_token_count(self, text: str) -> int:
        """Cuenta tokens VERB/NOUN/ADJ/ADV."""
        if not text or not text.strip():
            return 0
        return sum(
            1 for t in self.nlp(text) if t.pos_ in {"VERB", "NOUN", "ADJ", "ADV"}
        )

    def compact_segments(
        self,
        segments: list,
        min_content_tokens: int = None,
        min_chars: int = None,
        debug_tag: str = "",
    ) -> list:
        """Fusiona todo segmento sin sustancia con su vecino más similar.

        Cubre lo que ``progressive_clustering`` no valida (el primer segmento,
        documentos de una sola oración). Solo fusiona segmentos ADYACENTES, así
        que el resultado sigue siendo una concatenación en orden del input
        (requisito de ``_remap_uces_to_segments``). Idempotente.

        Coste: el conteo de content-tokens se calcula UNA vez por segmento y
        solo se recalcula el segmento fusionado → n + F llamadas a spaCy
        (F = nº de fusiones), no n·(F+1).
        """
        if not segments or len(segments) == 1:
            return list(segments)
        min_ct = self.min_content_tokens if min_content_tokens is None else min_content_tokens
        min_ch = self.min_segment_chars if min_chars is None else min_chars

        work = list(segments)
        counts = [self._content_token_count(s) for s in work]

        while len(work) > 1:
            failing = [
                i
                for i in range(len(work))
                if counts[i] < min_ct or len(work[i].strip()) < min_ch
            ]
            if not failing:
                break
            i = min(failing, key=lambda j: (counts[j], len(work[j])))

            if i == 0:
                j = 1
            elif i == len(work) - 1:
                j = i - 1
            else:
                embs = self.model.encode(
                    [work[i - 1], work[i], work[i + 1]],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                sim_left = float(cosine_similarity([embs[1]], [embs[0]])[0][0])
                sim_right = float(cosine_similarity([embs[1]], [embs[2]])[0][0])
                j = i - 1 if sim_left >= sim_right else i + 1

            lo, hi = min(i, j), max(i, j)
            self._dprint(
                f"  compact{debug_tag}: fusionados segs {lo + 1}+{hi + 1} "
                f"(content={counts[i]})"
            )
            work[lo] = work[lo].rstrip() + " " + work[hi].lstrip()
            counts[lo] = self._content_token_count(work[lo])
            del work[hi]
            del counts[hi]
        return work

    # ── Control duro de tamaño ────────────────────────────────────────────
    def _special_tokens(self) -> int:
        try:
            return int(self.tokenizer.num_special_tokens_to_add(pair=False))
        except Exception:
            return 2

    def _token_len(self, text: str) -> int:
        """Longitud en tokens CON especiales, sin truncar (misma vara que el
        chequeo de overflow de segment_text)."""
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def _balanced_split(
        self,
        text: str,
        max_tokens: int,
        min_content_tokens: int = None,
        max_overflow_ratio: float = 1.0,
    ) -> list:
        """Parte ``text`` en piezas con ``_token_len(pieza) <= max_tokens``.

        Rellena por límites de oración (greedy); una oración mayor que el
        presupuesto se parte por tokens. Una cola sin sustancia se absorbe en la
        pieza previa solo si el resultado cabe en ``max_tokens * max_overflow_ratio``
        (1.0 = cota estricta: preferimos una cola corta a violar el límite).
        """
        min_ct = self.min_content_tokens if min_content_tokens is None else min_content_tokens
        budget = max(1, max_tokens - self._special_tokens())
        sentences = self.classicseg.robust_sentence_split(text)
        if not sentences:
            return [text]

        expanded = []
        for s in sentences:
            ids = self.tokenizer.encode(s["text"], add_special_tokens=False)
            if len(ids) <= budget:
                expanded.append((s["text"], len(ids)))
            else:  # oración desbordada: caso raro, corte por tokens
                for k in range(0, len(ids), budget):
                    chunk = ids[k : k + budget]
                    expanded.append(
                        (self.tokenizer.decode(chunk, skip_special_tokens=True), len(chunk))
                    )

        pieces, current, cur_tok = [], [], 0
        for txt, n in expanded:
            if current and cur_tok + n > budget:
                pieces.append(" ".join(current))
                current, cur_tok = [], 0
            current.append(txt)
            cur_tok += n
        if current:
            pieces.append(" ".join(current))

        if len(pieces) >= 2:
            tail = pieces[-1]
            weak = (
                self._content_token_count(tail) < min_ct
                or len(tail.strip()) < self.min_segment_chars
            )
            merged = pieces[-2] + " " + tail
            if weak and self._token_len(merged) <= int(max_tokens * max_overflow_ratio):
                pieces[-2] = merged
                pieces.pop()
        return pieces

    def _enforce_max_tokens(self, pieces: list, max_tokens: int) -> list:
        """Garantía dura: toda pieza que siga excediendo max_tokens se re-parte."""
        out = []
        for piece in pieces:
            if self._token_len(piece) > max_tokens:
                out.extend(self._balanced_split(piece, max_tokens))
            else:
                out.append(piece)
        return out

    def progressive_clustering(self, segments):
        if not segments:
            return segments

        merged_segments = [segments[0]]

        for i in range(1, len(segments)):
            last_segment = merged_segments[-1]
            current_segment = segments[i]

            if not self.detect_topic_shift(last_segment, current_segment, last_n=3):
                merged_segments[-1] += " " + current_segment
            else:
                merged_segments.append(current_segment)

            # Cap history lists (esto sí se mantiene, es barato)
            MAX_HISTORY = 200
            for attr in (
                "similarity_history",
                "syntactic_diff_history",
                "lex_sim_history",
            ):
                hist = getattr(self, attr, None)
                if hist is not None and len(hist) > MAX_HISTORY:
                    setattr(self, attr, hist[-MAX_HISTORY:])

        return merged_segments

    def compute_syntactic_difference(self, doc1, doc2):
        dep_diff = sum(
            1 for token1, token2 in zip(doc1, doc2) if token1.dep_ != token2.dep_
        )
        return dep_diff / max(len(doc1), len(doc2))

    def premerge_cached(self, doc_uces) -> list:
        """Recursive + final clustering por documento, con caché por contenido.

        Devuelve la lista de segmentos mergeados que produce
        ``final_clustering(recursive_segmentation([u.texto for u in doc_uces]))``.

        ── Por qué es seguro cachear ────────────────────────────────────────
        Ambas sub-operaciones son puras respecto a su input:

          · ``recursive_segmentation`` lee ``self.model`` (persistente, sin
            dropout), ``self.similarity_threshold`` y ``self.max_depth``
            —los tres inmutables tras ``__init__``— y usa scipy (determinista).

          · ``final_clustering`` → ``progressive_clustering`` →
            ``detect_topic_shift``. Este último MUTA self.similarity_history, self.syntactic_diff_history y self.lex_sim_history, pero NINGUNO de esos atributos se lee para decidir si un par de segmentos se mergea. Son estado muerto: se escriben y nadie los consulta. (Nota: self.embeddings y self.index ya NO se mutan aquí tras el Fix A; solo los escribe segment_text al final del path no-coref.)

        Por tanto, saltarse la recomputación cuando el input no cambió no
        altera el resultado. Los side-effects que se pierden en un cache hit
        (historiales, ``self.embeddings``, ``self.index``) no afectan a
        ninguna decisión posterior del pipeline.

        ── Clave de caché ───────────────────────────────────────────────────
        SHA-1 sobre los textos de los UCEs unidos con NUL. Inmune a:
          · sufijos de ``uce.id`` (``__coref{ctx}``) añadidos por el caller
            antes de invocar este método — el id no entra en la clave;
          · el ``copy.deepcopy`` del caller — la clave usa los textos, no la
            identidad de los objetos;
          · reordenamientos dentro del doc (el hash es sensible al orden).

        ── Contrato con el caller ───────────────────────────────────────────
        El caller NO debe mutar la lista devuelta. Se devuelve una copia
        superficial (los strings son inmutables, el contenedor es nuevo) para
        que un caller futuro que mute la lista no envenene el caché.

        ── Cota de memoria ──────────────────────────────────────────────────
        Un entry típico pesa ~5 KB (≈15 strings de ~300 chars). El caché se
        acota a ``_PREMERGE_CACHE_MAXSIZE`` entries con evicción FIFO. Con 2000
        entries el pico es ~10 MB, independiente del tamaño del corpus.

        ── Determinismo verificado ──────────────────────────────────────────
        Ver ``tests/test_premerge_cache.py`` (si no existe, el bloque de
        verificación del PR lo incluye) — compara bit-exacto el output con y
        sin caché sobre docs reales del corpus.
        """
        if not doc_uces:
            return []

        key = hashlib.sha1(
            b"\x00".join(u.texto.encode("utf-8") for u in doc_uces)
        ).digest()

        cached = self._premerge_cache.get(key)
        if cached is not None:
            # Copia superficial: el caller puede mutar la lista sin dañar el
            # caché; los strings son inmutables y se comparten.
            if self.debug_coref:
                self._dprint(
                    f"premerge cache HIT ({len(cached)} segmentos) key={key.hex()[:8]}"
                )
            return list(cached)

        if self.debug_coref:
            self._dprint(
                f"premerge cache MISS ({len(doc_uces)} UCEs) key={key.hex()[:8]}"
            )

        uce_texts = [u.texto for u in doc_uces]
        rec_seg = self.recursive_segmentation(uce_texts)
        prog_seg = self.final_clustering(rec_seg)

        # Evicción FIFO antes de insertar. Los dict de Python 3.7+ preservan
        # el orden de inserción, así que next(iter(...)) es el más antiguo.
        if len(self._premerge_cache) >= _PREMERGE_CACHE_MAXSIZE:
            oldest = next(iter(self._premerge_cache))
            del self._premerge_cache[oldest]

        self._premerge_cache[key] = prog_seg
        return list(prog_seg)

    def final_clustering(self, segments):
        clustered_segments = self.progressive_clustering(segments)
        logging.info(
            f"[FinalClust] {len(clustered_segments)} segmentos tras clustering."
        )
        return clustered_segments

    class OffsetMapper:
        def __init__(self, global_offset):
            self.global_offset = global_offset

        def to_global(self, local_start, local_end):
            return local_start + self.global_offset, local_end + self.global_offset

    def _fuzzy_match(self, text1, text2):
        stop = self.nlp.Defaults.stop_words

        def clean(t):
            return " ".join([w for w in t.lower().split() if w not in stop])

        norm1, norm2 = clean(text1), clean(text2)
        if not norm1 or not norm2:
            return text1.lower().strip() == text2.lower().strip()

        # token_set_ratio is better for phrasal matching (e.g., "médicos" inside "los médicos residentes")
        return fuzz.token_set_ratio(norm1, norm2) >= 85

    def find_subjects_for_roots(self, text: str) -> list:
        """
        Extracts full Noun Phrases (NP) as subjects.
        Includes fallback logic for MWT index mismatches.

        Runs only tokenize/mwt/pos/lemma/depparse/constituency (no coref) unless
        ``roots_skip_coref=False``; the extraction itself lives in
        ``reii.seg.root_cache.subjects_from_doc``.
        """
        return self._roots.single(text)

    # ─────────────────────────────────────────────────────────────────────────
    # _extract_global_chains — with full debug prints + chain deduplication FIX
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_global_chains(
        self, segments_info, full_doc_text, context_units=2, coref_batch_size=None
    ):
        """
        Sliding window coreference extraction with:
          - Debug prints at every stage
          - Chain deduplication by mention-span fingerprint (FIX for duplicate merges)

        NOTA (batching): ``bulk_process`` NO es válido para coref. Stanza's
        ``CorefProcessor`` hereda ``UDProcessor.bulk_process``, que combina todas
        las oraciones de todos los documentos en un único ``combined_doc``, resuelve
        coref a través de TODAS las sub-ventanas (rompe la ventana limitada) y
        devuelve los documentos originales con ``coref`` VACÍO. Por eso el default
        es ``coref_batch_size=1`` (una sub-ventana por llamada). Si se fuerza
        ``>1`` se emite un warning y se procesa igualmente (bajo tu propio riesgo).

        Modo remoto: si ``STANZA_SERVER_URL`` está definido, las sub-ventanas se
        envían a ``/v1/extract_coref`` y el servidor las procesa UNA A UNA (nunca
        ``bulk_process``), respetando la ventana limitada. El pipeline local no se
        carga salvo que el fallback a local lo requiera.
        """
        self.last_extraction_incomplete = False
        global_chains = []
        # FIX: track chain fingerprints to avoid adding the same chain from overlapping windows
        seen_chain_keys = set()

        buffer = []

        total_uces = len(segments_info)
        self._dprint(
            f"Iniciando extracción de cadenas globales sobre {total_uces} UCEs "
            f"(ventana deslizante de {context_units} UCEs)."
        )

        _bs = COREF_BATCH_SIZE if coref_batch_size is None else int(coref_batch_size)

        if _bs > 1 and not STANZA_SERVER_URL:
            logging.warning(
                "REII_COREF_BATCH_SIZE=%d > 1: Stanza coref bulk_process combina "
                "sub-ventanas en un único documento (rompe la ventana limitada y "
                "pierde cadenas). Se recomienda REII_COREF_BATCH_SIZE=1.",
                _bs,
            )

        if not STANZA_SERVER_URL:
            stanza_pipe = self.get_stanza()
            if stanza_pipe is None:
                self._dprint("✗ Stanza no disponible — saltando extracción de cadenas.")
                self.last_extraction_incomplete = True
                return global_chains

        # ── Pass 1: collect candidate sub-windows (windowing + candidate filter) ──
        pending = []  # list of (sub_text, sub_start)
        for uce_idx, uce in enumerate(segments_info):
            buffer.append((uce["start"], uce["end"]))
            if len(buffer) > context_units + 1:
                buffer.pop(0)

            window_start = buffer[0][0]
            window_end = buffer[-1][1]
            window_text = full_doc_text[window_start:window_end]
            window_char_len = window_end - window_start

            self._dprint(
                f"  Ventana UCE {uce_idx + 1}/{total_uces} | "
                f"chars [{window_start}:{window_end}] ({window_char_len} chars) | "
                f"buffer size={len(buffer)}"
            )

            # Coref es lento (~11s por ~100 chars medido en T4). El túnel de
            # Cloudflare corta a ~100s y el timeout del cliente es 90s, así que
            # cada sub-ventana debe resolverse en <60s. 1000 chars lo garantiza
            # (~20-25s por llamada) incluso con la GPU degradada.
            MAX_WINDOW_CHARS = 1000
            CHUNK_OVERLAP = 400

            # ── Troceo con solape (reemplaza el truncado duro) ──
            if window_char_len > MAX_WINDOW_CHARS:
                step = MAX_WINDOW_CHARS - CHUNK_OVERLAP
                sub_windows = []
                pos = 0
                while pos < window_char_len:
                    end = min(pos + MAX_WINDOW_CHARS, window_char_len)
                    sub_windows.append((window_text[pos:end], window_start + pos))
                    if end == window_char_len:
                        break
                    pos += step
                self._dprint(
                    f"    Ventana grande ({window_char_len} chars) troceada en "
                    f"{len(sub_windows)} sub-ventanas de ~{MAX_WINDOW_CHARS} chars "
                    f"con solape de ~{CHUNK_OVERLAP} chars."
                )
            else:
                sub_windows = [(window_text, window_start)]

            for sub_idx, (sub_text, sub_start) in enumerate(sub_windows):
                # ── Filtro rápido de candidatos a correferencia (B2) ──
                if not _COREF_CANDIDATE_RE.search(sub_text):
                    self._dprint(
                        f"    Sub-ventana {sub_idx + 1}/{len(sub_windows)} sin "
                        f"candidatos de correferencia — saltando Stanza."
                    )
                    continue
                pending.append((sub_text, sub_start))

        # ── Pass 2: batch process candidate sub-windows ──
        if STANZA_SERVER_URL:
            # Modo remoto: el servidor procesa cada sub-ventana individualmente
            # (nunca bulk_process), respetando la ventana limitada.
            self._extract_coref_remote(pending, seen_chain_keys, global_chains)
        elif _bs > 1:
            # Lote adaptativo: el coref completo en CPU es pesado en RAM. Un lote
            # de 16 sub-ventanas grandes puede OOM (SIGKILL, sin traceback). Para
            # documentos grandes se encoge el lote (ver adaptive_coref_batch_size).
            _bs = adaptive_coref_batch_size(_bs, sum(len(t) for t, _ in pending))
            for i in range(0, len(pending), _bs):
                batch = pending[i : i + _bs]
                texts = [t for t, _ in batch]
                try:
                    docs = stanza_pipe.bulk_process(texts)
                    if len(docs) != len(texts):
                        raise ValueError(
                            f"bulk_process devolvió {len(docs)} docs para {len(texts)} textos"
                        )
                except Exception as e:
                    logging.warning(
                        "coref bulk_process falló (%s); procesando una a una", e
                    )
                    for sub_text, sub_start in batch:
                        self._process_coref_window(
                            stanza_pipe,
                            sub_text,
                            sub_start,
                            seen_chain_keys,
                            global_chains,
                        )
                    continue
                for (sub_text, sub_start), doc in zip(batch, docs):
                    self._extract_chains_from_doc(
                        doc, sub_text, sub_start, seen_chain_keys, global_chains
                    )
                    del doc
                gc.collect()
        else:
            for sub_text, sub_start in pending:
                self._process_coref_window(
                    stanza_pipe, sub_text, sub_start, seen_chain_keys, global_chains
                )

        self._dprint(
            f"Extracción global completada: {len(global_chains)} cadenas únicas "
            f"sobre {total_uces} UCEs."
        )
        return global_chains

    def _extract_coref_remote(self, pending, seen_chain_keys, global_chains):
        """Extrae cadenas coref vía el servidor remoto (/v1/extract_coref).

        Cada sub-ventana se envía como texto independiente: el servidor llama a
        ``nlp(text)`` por texto (nunca ``bulk_process``), así que la ventana
        limitada se respeta. Los offsets devueltos (relativos a la sub-ventana)
        se mapean a globales con ``OffsetMapper`` y las cadenas se deduplican por
        huella de menciones, igual que en ``_extract_chains_from_doc``.

        Si una llamada falla y ``STANZA_SERVER_FALLBACK_LOCAL`` es True, se cae a
        Stanza local (una sub-ventana por llamada); si es False, se propaga el
        fallo (sin cadenas para ese lote).
        """
        if not pending:
            return

        # Lotes por petición: el servidor sigue llamando nlp() UNA vez por
        # texto dentro de cada batch (nunca los mezcla entre sí — ver nota en
        # stanza_server.py), así que REMOTE_BATCH>1 solo ahorra overhead de
        # red/túnel por texto, no tiempo de GPU. Default=1: comportamiento
        # original. Subir tras migrar de cloudflared quick-tunnel (~100s de
        # timeout) a un túnel propio — vigilar que
        # REMOTE_BATCH × tiempo_por_texto < STANZA_SERVER_TIMEOUT.
        REMOTE_BATCH = STANZA_SERVER_COREF_REMOTE_BATCH
        batches = [pending[i : i + REMOTE_BATCH] for i in range(0, len(pending), REMOTE_BATCH)]

        def _fetch_batch(batch):
            """Solo la llamada de red — SIN tocar seen_chain_keys/global_chains.

            Esas dos estructuras se mutan después, secuencialmente, en el
            hilo principal: son sets/listas comunes, no thread-safe, y
            mutarlas desde varios hilos (ThreadPoolExecutor) a la vez
            reintroduciría exactamente el tipo de carrera que la dedupe por
            fingerprint está pensada para evitar. Esta función solo decide
            éxito/fracaso; el merge pasa por acá abajo, en orden, como antes.
            """
            texts = [t for t, _ in batch]
            try:
                resp = requests.post(
                    f"{STANZA_SERVER_URL}/v1/extract_coref",
                    json={"texts": texts, "lang": self.stanza_lang},
                    headers={"Authorization": f"Bearer {STANZA_SERVER_TOKEN}"},
                    timeout=STANZA_SERVER_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                if len(results) != len(texts):
                    raise ValueError(
                        f"el servidor devolvió {len(results)} resultados para "
                        f"{len(texts)} textos"
                    )
                return (batch, results, None)
            except Exception as e:
                return (batch, None, e)

        concurrency = max(1, STANZA_SERVER_CONCURRENCY)
        # Solo rinde si el servidor tiene >1 réplica (REII_STANZA_N_REPLICAS);
        # si no, las peticiones concurrentes se encolan igual detrás de la
        # única réplica remota — no hace daño, pero tampoco acelera nada.
        if concurrency <= 1 or len(batches) <= 1:
            fetched = [_fetch_batch(b) for b in batches]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # executor.map preserva el orden de batches, igual que el
                # for secuencial original — el merge de abajo no cambia.
                fetched = list(executor.map(_fetch_batch, batches))

        # ── Merge: SIEMPRE secuencial, en el orden original de `batches` ──
        for batch, results, err in fetched:
            if err is not None:
                self._dprint(f"    ✗ Stanza remoto falló ({err!r}); usando Stanza local.")
                logging.warning("Stanza remoto falló (%s); usando Stanza local", err)
                if not STANZA_SERVER_FALLBACK_LOCAL:
                    # Idéntico al comportamiento original: un fallo duro
                    # aborta el resto de `pending`, no solo este batch.
                    self.last_extraction_incomplete = True
                    return
                stanza_pipe = self.get_stanza()
                if stanza_pipe is None:
                    self._dprint(
                        "✗ Stanza local no disponible — saltando sub-ventanas."
                    )
                    self.last_extraction_incomplete = True
                    return
                for sub_text, sub_start in batch:
                    self._process_coref_window(
                        stanza_pipe, sub_text, sub_start, seen_chain_keys, global_chains
                    )
                continue

            for (sub_text, sub_start), r in zip(batch, results):
                mapper = self.OffsetMapper(sub_start)
                for chain in r.get("chains", []):
                    mentions = []
                    for m in chain.get("mentions", []):
                        m_start_local = m.get("start_char")
                        m_end_local = m.get("end_char")
                        if m_start_local is None or m_end_local is None:
                            continue
                        m_start_global, m_end_global = mapper.to_global(
                            m_start_local, m_end_local
                        )
                        mentions.append(
                            {
                                "text": sub_text[m_start_local:m_end_local],
                                "start_char": m_start_global,
                                "end_char": m_end_global,
                            }
                        )
                    if not mentions:
                        continue
                    # ── FIX: deduplicate chains by their global mention span fingerprint ──
                    chain_key = tuple(
                        sorted((m["start_char"], m["end_char"]) for m in mentions)
                    )
                    if chain_key in seen_chain_keys:
                        continue
                    seen_chain_keys.add(chain_key)
                    global_chains.append({"mentions": mentions})
                    if self.debug_coref:
                        mention_texts = [m["text"] for m in mentions]
                        self._dprint(
                            f"      ✓ Cadena #{len(global_chains)} añadida (remota) | "
                            f"{len(mentions)} menciones: {mention_texts}"
                        )

    def _extract_chains_from_doc(
        self, doc, sub_text, sub_start, seen_chain_keys, global_chains
    ):
        """Extract coref chains from a processed doc, map offsets, dedup, append."""
        mapper = self.OffsetMapper(sub_start)

        for chain_idx, chain in enumerate(doc.coref):
            mentions = []

            for mention in chain.mentions:
                # Defensive index extraction (MWT / Spanish contraction guard)
                s_idx = (
                    mention.sentence[0]
                    if isinstance(mention.sentence, (tuple, list))
                    else mention.sentence
                )
                sw_idx = (
                    mention.start_word[0]
                    if isinstance(mention.start_word, (tuple, list))
                    else mention.start_word
                )
                ew_idx = (
                    mention.end_word[-1]
                    if isinstance(mention.end_word, (tuple, list))
                    else mention.end_word
                )

                sent = doc.sentences[s_idx]
                first_word = sent.words[sw_idx]
                last_word = sent.words[ew_idx - 1]

                m_start_local = first_word.start_char
                if m_start_local is None and getattr(first_word, "parent", None):
                    m_start_local = first_word.parent.start_char

                m_end_local = last_word.end_char
                if m_end_local is None and getattr(last_word, "parent", None):
                    m_end_local = last_word.parent.end_char

                if m_start_local is None or m_end_local is None:
                    self._dprint(
                        f"      Mención saltada (offset None): "
                        f"cadena {chain_idx}, palabra '{first_word.text}'"
                    )
                    continue

                m_start_global, m_end_global = mapper.to_global(
                    m_start_local, m_end_local
                )
                mentions.append(
                    {
                        "text": sub_text[m_start_local:m_end_local],
                        "start_char": m_start_global,
                        "end_char": m_end_global,
                    }
                )

            if not mentions:
                continue

            # ── FIX: deduplicate chains by their global mention span fingerprint ──
            chain_key = tuple(
                sorted((m["start_char"], m["end_char"]) for m in mentions)
            )
            if chain_key in seen_chain_keys:
                continue
            seen_chain_keys.add(chain_key)

            global_chains.append({"mentions": mentions})

            if self.debug_coref:
                mention_texts = [m["text"] for m in mentions]
                self._dprint(
                    f"      ✓ Cadena #{len(global_chains)} añadida | "
                    f"{len(mentions)} menciones: {mention_texts}"
                )

    def _process_coref_window(
        self, stanza_pipe, sub_text, sub_start, seen_chain_keys, global_chains
    ):
        """Process a single sub-window (fallback path) with per-window error handling."""
        try:
            doc = stanza_pipe(sub_text)
        except MemoryError:
            self._dprint("    MemoryError en sub-ventana — saltando.")
            gc.collect()
            self._empty_cuda_cache()
            return
        except Exception as e:
            logging.error(
                f"[COREF] Stanza window error "
                f"[{sub_start}:{sub_start + len(sub_text)}]: {e}"
            )
            self._dprint(f"    ✗ Stanza falló en sub-ventana: {e}")
            self._empty_cuda_cache()
            return

        self._extract_chains_from_doc(
            doc, sub_text, sub_start, seen_chain_keys, global_chains
        )
        del doc
        gc.collect()

    # ─────────────────────────────────────────────────────────────────────────
    # van_unidos — with debug prints
    # ─────────────────────────────────────────────────────────────────────────
    def van_unidos(self, seg1_info, seg2_info, global_chains, precomputed_roots=None):
        """
        Decides whether a coreference chain bridges the boundary between two segments.

        Parameters
        ----------
        precomputed_roots : set or None
            If provided (pre-computed root NP subjects for seg1's tail), Stanza is
            skipped.  When the set is empty the method still reports spanning chains
            for quality tracking — unlike the None (backward-compat) case which
            short-circuits immediately.
        Returns
        -------
        (should_merge, has_spanning)
            should_merge  : True iff a chain has a root-matching mention in seg1
                            AND any mention in seg2.
            has_spanning  : True iff any chain (regardless of root match) has
                            mentions in BOTH segments (quality metric).
        """
        if precomputed_roots is not None:
            los_roots = precomputed_roots
        else:
            text1 = seg1_info["text"]
            boundary_context = text1[-500:] if len(text1) > 500 else text1
            los_roots = set(self.find_subjects_for_roots(boundary_context))

        self._dprint(
            f"  van_unidos | UCE [{seg1_info['start']}:{seg1_info['end']}] → "
            f"[{seg2_info['start']}:{seg2_info['end']}] | "
            f"sujetos raíz detectados: {los_roots if los_roots else '∅'}"
        )

        if not los_roots:
            self._dprint("    No se encontraron sujetos raíz — no merge.")
            # When precomputed roots were explicitly supplied (even empty), still
            # report any chain that spans both segments for quality tracking.
            if precomputed_roots is not None:
                has_spanning = any(
                    any(
                        seg1_info["start"] <= m["start_char"] < seg1_info["end"]
                        for m in chain["mentions"]
                    )
                    and any(
                        seg2_info["start"] <= m["start_char"] < seg2_info["end"]
                        for m in chain["mentions"]
                    )
                    for chain in global_chains
                )
                return False, has_spanning
            return False, False

        has_any_spanning = False

        for chain_idx, chain in enumerate(global_chains):
            has_seg1_root = False
            has_seg2_mention = False
            has_mention_in_1 = False

            for mention in chain["mentions"]:
                if seg1_info["start"] <= mention["start_char"] < seg1_info["end"]:
                    has_mention_in_1 = True
                    if any(
                        self._fuzzy_match(mention["text"], root) for root in los_roots
                    ):
                        has_seg1_root = True
                        self._dprint(
                            f"    Cadena #{chain_idx + 1}: mención raíz hallada en UCE1 "
                            f"→ '{mention['text']}'"
                        )

                elif seg2_info["start"] <= mention["start_char"] < seg2_info["end"]:
                    has_seg2_mention = True
                    self._dprint(
                        f"    Cadena #{chain_idx + 1}: mención hallada en UCE2 "
                        f"→ '{mention['text']}'"
                    )

                if has_seg1_root and has_seg2_mention:
                    self._dprint(
                        f"    ✓ MERGE: cadena #{chain_idx + 1} cruza el límite — "
                        f"fusionando UCEs."
                    )
                    return True, True

            # Track chains that span both segments without a root match (quality metric)
            if has_mention_in_1 and has_seg2_mention:
                has_any_spanning = True

        self._dprint("    No se encontró puente correferencial — manteniendo límite.")
        return False, has_any_spanning

    # ─────────────────────────────────────────────────────────────────────────
    # resolve_coreferences — with debug prints
    # ─────────────────────────────────────────────────────────────────────────
    def resolve_coreferences(self, segments):
        self._dprint(
            f"\n{'=' * 60}\n"
            f"resolve_coreferences() iniciado con {len(segments)} segmentos.\n"
            f"{'=' * 60}"
        )

        if (
            not segments
            or not isinstance(segments, list)
            or (not STANZA_SERVER_URL and not self.get_stanza())
        ):
            self._dprint(
                "Condición de salida temprana: lista vacía o Stanza no disponible."
            )
            return segments

        # 1. Build offset map
        full_doc_text = ""
        segments_info = []
        current_offset = 0
        for i, seg in enumerate(segments):
            start = current_offset
            end = start + len(seg)
            segments_info.append({"text": seg, "start": start, "end": end})
            full_doc_text += seg + " "
            current_offset = end + 1
            self._dprint(
                f"  Seg {i + 1:03d}: chars [{start}:{end}] "
                f"({end - start} chars) — '{seg[:40].strip()}...'"
            )

        self._dprint(
            f"\nTexto completo reconstruido: {len(full_doc_text)} chars, "
            f"{len(segments_info)} UCEs mapeadas.\n"
        )

        # 2. Extract global chains (sliding window)
        self._dprint("Fase 1: extrayendo cadenas coref globales...")
        global_chains = self._extract_global_chains(segments_info, full_doc_text)
        self._dprint(
            f"Fase 1 completada: {len(global_chains)} cadenas únicas disponibles "
            f"para resolución.\n"
        )

        # 3. Greedy merge pass
        self._dprint("Fase 2: pasada de merge greedy con van_unidos()...")
        merged_segments = []
        current_seg_info = segments_info[0]
        merges_total = 0

        for i in range(1, len(segments_info)):
            next_seg_info = segments_info[i]
            self._dprint(
                f"\n  Evaluando límite {i}/{len(segments_info) - 1}: "
                f"UCE {i} → UCE {i + 1}"
            )
            should_merge, _ = self.van_unidos(
                current_seg_info, next_seg_info, global_chains
            )
            if should_merge:
                current_seg_info["text"] += " " + next_seg_info["text"]
                current_seg_info["end"] = next_seg_info["end"]
                merges_total += 1
                self._dprint(
                    f"  → MERGE aplicado. UCE actual extendida a "
                    f"[{current_seg_info['start']}:{current_seg_info['end']}] "
                    f"({len(current_seg_info['text'])} chars)"
                )
            else:
                merged_segments.append(current_seg_info["text"])
                current_seg_info = next_seg_info
                self._dprint(f"  → Límite conservado.")

        merged_segments.append(current_seg_info["text"])

        self._dprint(
            f"\nFase 2 completada: {merges_total} merges aplicados. "
            f"{len(merged_segments)} UCEs finales (de {len(segments)} originales).\n"
            f"{'=' * 60}\n"
        )

        return merged_segments

    def process_segment(self, i, max_tokens):
        token_ids = self.tokenizer.encode(
            i,
            add_special_tokens=True,
            max_length=max_tokens,
            truncation=True,
            padding=True,
        )
        if len(token_ids) > max_tokens:
            print(
                f"[SegText] Segmento muy largo ({len(token_ids)} tokens) — enviando a ClassicSegmenter."
            )
            return self.classicseg.segment_text(i)
        return [i]

    def segment_text(self, text, max_tokens=1024):
        text = text or ""  # None → "" (contrato: entrada vacía devuelve [])
        print(f"\n[SegText] Iniciando segmentación. Texto: {len(text)} chars.")
        sentences = self.preprocess_text(text)
        if not sentences:
            # Entrada vacía / solo puntuación: contrato explícito de salida
            stripped = text.strip()
            print("[SegText] Sin oraciones tras preprocesado.")
            return [stripped] if stripped else []
        print(f"[SegText] {len(sentences)} oraciones tras preprocesado.")

        all_segments = self.recursive_segmentation(sentences)
        print(f"[SegText] {len(all_segments)} segmentos tras segmentación recursiva.")

        # FIX: fit TF-IDF on raw sentences, not intermediate segments (avoids circularity)
        self.tfidf_vectorizer.fit(sentences)
        print(f"[SegText] TF-IDF ajustado sobre {len(sentences)} oraciones crudas.")

        clustered_segments = self.final_clustering(all_segments)
        print(f"[SegText] {len(clustered_segments)} segmentos tras clustering final.")

        # Barrido de sustancia: valida lo que progressive_clustering no ve
        # (primer segmento, docs de 1 oración). Solo afecta a segment_text; el
        # pre-merge del pipeline REII (premerge_cached) NO pasa por aquí.
        clustered_segments = self.compact_segments(
            clustered_segments, debug_tag=" pre-coref"
        )

        self.embeddings = self.generate_embeddings(clustered_segments)
        if len(self.embeddings) > 0:
            self.index = hnswlib.Index(space="cosine", dim=self.embeddings.shape[1])
            self.index.init_index(
                max_elements=len(self.embeddings),
                ef_construction=self.hnsw_ef,
                M=self.hnsw_m,
            )
            self.index.add_items(self.embeddings)
            self.index.set_ef(self.hnsw_ef)

        print(f"[SegText] Iniciando resolución de correferencias...")
        clustered_segments = self.resolve_coreferences(clustered_segments)
        print(
            f"[SegText] {len(clustered_segments)} UCEs tras resolución de correferencias."
        )

        segmentos = []
        overflow_count = 0
        for i in clustered_segments:
            token_count = len(
                self.tokenizer.encode(
                    i,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_tokens + 1,
                )
            )

            if token_count > max_tokens:
                overflow_count += 1

                # token_count está TRUNCADO a max_tokens+1 (ver encode arriba), así
                # que no sirve para dimensionar el corte: antes daba siempre 2
                # piezas, aunque la UCE tuviera 3000 tokens. Se mide sin truncar.
                real_tokens = self._token_len(i)
                num_required_splits = max(2, -(-real_tokens // max_tokens))

                print(
                    f"[SegText] Overflow UCE ({real_tokens} tokens) → Corte quirúrgico en {num_required_splits} fragmento(s)."
                )

                # Force the ClassicSegmenter to make ONLY the required cuts
                safe_segments = self.classicseg.segment_text(
                    i,
                    max_segments=num_required_splits,
                    threshold=-1.0,  # -1.0 bypasses the strict score check, forcing a clean merge
                )
                # El corte semántico no controla tamaño: se re-mide y, si alguna
                # pieza aún excede max_tokens, se re-parte por oraciones.
                safe_segments = self._enforce_max_tokens(safe_segments, max_tokens)
                segmentos.extend(safe_segments)
            else:
                segmentos.append(i)

        if self.debug_coref and segmentos:
            _cc = [self._content_token_count(x) for x in segmentos]
            print(
                f"[SegText] Distribución: content-tokens min={min(_cc)} "
                f"med={sorted(_cc)[len(_cc) // 2]} max={max(_cc)}"
            )
        print(
            f"[SegText] Segmentación completada: {len(segmentos)} UCEs finales "
            f"({overflow_count} desbordamientos resueltos con cortes quirúrgicos)."
        )
        return segmentos


# transcript_text = """
# Tengo actualmente veinte años no tengo dependientes económicos por vacaciones trabajo de manera ocasional pero no tengo trabajo estable normalmente trabajo como docente en particular o si no en fábricas depende el tiempo soy de lima del distrito de Carabayllo estudio educación primaria y también estudio inglés. Más que todo tuve influencia de mis padres de mi madre que es docente.
# En cinco años espero estar ejerciendo y con una mentalidad más fuerte supongo con una confianza mayor.
# Cuando pienso en cambio climático como siempre a veces tocamos cositas en primaria con los niños podemos asemejarlo a la contaminación o sea las basuras los desperdicios las plantas ese tipo de cosas creo que se produce más que todo por no tener un control absoluto del daño que hacemos con los beneficios.
# Normalmente recibo información sobre cambio climático en las noticias así en el apartado de google o si no en las noticias de la televisión el internet es lo que me brinda mayor información porque te permite explorar más de lo que te da considero que es importante esta información porque nos permite en este caso como docentes hacer un cambio desde la infancia y que no se vuelva un hábito de contaminar escuché charlas hace un tiempo pero más allá de eso no he explorado mucho.
# En mi zona se ha vuelto más fácil respirar porque antes era un montón de tierra puro tierra ahorita hay un poquito más de árboles las veredas están naciendo se respira mejor en mi infancia he estado en el campo se siente ese aire tranquilo respirar hondo más que todo en el centro y en gamarra es difícil respirar yo lo siento yo siento que me sofoco.
# El cambio climático influye en la vida de las personas por los hábitos te vuelves una persona más ordenada más limpia en el buen sentido pero de la otra manera podría convertirnos en alguien que no tiene un poco del control de sus cosas afortunadamente no he tenido la oportunidad de ver a alguien enfermo a causa del cambio climático pero espero no ver una vez en una chocolatada vimos a una chica en la cual se desmayó por el calor otro fue el resfrío común pero el que más impactó fue ese desmayo por insolación en mis niños por esta temporada pero más allá de eso son niños fuertes.
# En mi familia nos pone más alerta sobre la prevención abrígate no estés con tanta ropa cuando hace mucho calor eso el calor a mí me afecta porque siento que me mareo mucho me produce dolor de cabeza y a veces mi visión se pone borrosa antes mi familia tenía una chacra en chincha y ahí nos afectaba un poco la sequía de que no llegaba el agua.
# Quizás soy un poco ajeno a esos temas pero la universidad nos ofrece lo justo y necesario creo pero si quieres explorar más es siempre por tu cuenta y ahí sí a veces discrepo porque debería ser un poco más accesible más que todo ahorita lo siento por el tema político y cosas así en la universidad ha pasado eso de las tomas y esas cosas a veces se tiene la información en un grupo nomás no se socializa quizás para todos.
# Aprendí un poco más sobre cambio climático en didáctica de las ciencias sociales y didáctica de ciencia y tecnología creo que es insuficiente porque más que todo nos enseñan hasta cómo tratar el tema más no el tema en sí considero importante abordar el cambio climático dentro de la educación porque genera buenos hábitos y aparte que beneficia nuestra salud la universidad podría designar talleres donde ciertos días estén disponibles para que el tipo de ponencias cada cierto tiempo se establezca y los estudiantes tengan libre acceso.
# Si tuviera la oportunidad de cambiar o aportar algo en el plan de estudios sería ampliar nuestro conocimiento con un curso más que sea honestamente abordando esos temas porque como señalé en ciencia y tecnología tocamos diversos temas los cuales nos hacen tocar pequeñas partes nomás de lo que es el cambio climático para cinco años yo siento que es corto todavía la información que se nos da.
# Creo que más allá de marchas a favor de la prevención del cambio climático o por diversos temas no he visto ninguna acción más allá de eso ningún docente hace algo sobre esto más allá de tocarlo como un tema general no creo que no estén interesados por prioridad supongo y tiempo no lo toman como un tema prioritario.
# Como acción sostenible para el cambio climático podría ser el voluntariado para comprender un poco más la labor de las personas que barren o sea y nos ayudan a que nuestra universidad esté limpia.
# Como futuro docente me interesaría enseñar sobre este tema porque es un tema que se comprende a profundidad nos ayuda a mejorar nuestra calidad de enseñanza lo cual permite mejores estudiantes yo me especializo en niños de siete a once quince años les enseñaría la prevención cómo combatirla y más que todo desearía centrarme en la prevención cómo evitar contaminar la población a veces es inevitable pero hacerlo de manera medida por ejemplo con diversas actividades lúdicas en las cuales comprendamos la importancia de ello primero abordar por qué es importante para nosotros prevenir el cambio climático luego cómo debemos enfrentar entre esas cositas.
# Yo soy una persona con mucha esperanza y siempre espero que el país pueda afrontar diversas adversidades en este caso del cambio climático por ejemplo ya hemos pasado esto del incendio forestal que pasó hace poco ha habido pérdidas obviamente pero se está intentando mejorar o prevenir estos tipos de situaciones con ayuda de la tecnología quizás de nosotros mismos un cambio es posible empezaríamos con un cambio de mentalidad sobre todo en la limpieza yo creo empieza ahí empieza de por ejemplo cuando comes algo y lo tiras en el suelo ahí empieza todo el problema desde ahí se arraiga más tirar un montón de cosas a la esquina o no respetar los horarios cuando viene el basurero esas cosas son mínimas pero yo siento que podrían influir mucho en cómo se ve uno.
# No sabría decirlo con exactitud pero hace poco escuché algo de añadir más contenedores de basura lo cual facilita la recolección de ello pero no sabría decirlo siempre es insuficiente por más que uno piense que está aportando algo o que siempre se va a necesitar más pero se tiene esperanza con que se mejoren las cosas hay que tener esperanza y fe hay que también poner nuestros granitos de arena y seguir.

# """

# # 2. Initialize the pipeline
# print("Cargando modelos... (Esto puede tomar unos segundos la primera vez)")
# segmenter = ProgressiveSegmenter(
#     model_name="ibm-granite/granite-embedding-107m-multilingual",
#     similarity_threshold=0.6,
#     max_depth=3,
#     device="cpu",  # Set to "cpu" if running on a machine without a dedicated GPU
# )

# # 3. Execute the pipeline
# print("\nIniciando segmentación progresiva...")

# # We pass the text and set a max_tokens limit.
# # If a segment naturally exceeds this, it falls back to your ClassicSegmenter to force a cut.
# final_segments = segmenter.segment_text(transcript_text, max_tokens=500)

# # 4. Review the Output
# print(
#     f"\n✅ Pipeline completado. Total de segmentos resultantes: {len(final_segments)}\n"
# )

# for i, segment in enumerate(final_segments, 1):
#     print(f"--- Segmento {i} ---")
#     print(segment)
#     print("-" * 50)
