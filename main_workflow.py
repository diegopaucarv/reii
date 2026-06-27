# %%viztracer
# main_pipeline.py
from __future__ import annotations

import copy
import dataclasses
import gc
import glob
import json
import logging
import os
import re
import time
import traceback
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import prince
import spacy
from nltk.stem import SnowballStemmer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.special import softmax
from scipy.stats import chi2_contingency, f_oneway, fisher_exact
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.utils.extmath import randomized_svd
from statsmodels.stats.multitest import multipletests
from transformers import utils

from gram.gramatical_analyzer import (
    UCE,
    GlobalCorpus,
    GlobalLexicalAnalyzer,
    NLPProvider,
    PipelineGramatical,
    SubtlexAnalyzer,
    WordEmbeddingsAnalyzer,
    train_adverb_classifier,
)
from gram.gramatical_analyzer import (
    Config as GramConfig,
)
from seg.segmentador import ProgressiveSegmenter

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from huggingface_hub import logging as hub_logging
from transformers import logging as transformers_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

utils.logging.set_verbosity_error()
# Set the verbosity levels to ERROR
transformers_logging.set_verbosity_error()
hub_logging.set_verbosity_error()

nltk.download("punkt", quiet=True)

# Opcionales
try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    import networkx as nx

    _NETWORKX_AVAILABLE = True
except ImportError:
    _NETWORKX_AVAILABLE = False

try:
    from community import community_louvain

    _LOUVAIN_AVAILABLE = True
except ImportError:
    _LOUVAIN_AVAILABLE = False

try:
    from joblib import Parallel, delayed

    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

# Robust RF + SHAP imports
try:
    import scipy.stats as stats
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import (
        RandomizedSearchCV,
        StratifiedKFold,
        cross_val_predict,
        cross_val_score,
    )
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    _RF_SHAP_AVAILABLE = True
except ImportError:
    _RF_SHAP_AVAILABLE = False

try:
    import hdbscan
    import umap

    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False


gc.collect()


# Función a prueba de balas para limpiar extensiones dobles y simples
# Convierte "doc.txt.json" o "doc.txt" a una llave maestra limpia: "doc"
def obtener_llave_maestra(nombre):
    return str(nombre).replace(".txt.json", "").replace(".json", "").replace(".txt", "")


# 0. Cargamos tu metadata usando la llave maestra
df_meta = pd.read_csv("C:/Users/Julia/Desktop/Refined_Database.csv", sep=";")
meta_dict = {}

for _, row in df_meta.iterrows():
    llave = obtener_llave_maestra(row["Documento Fuente"])
    meta_dict[llave] = row[
        ["Edad_Cat", "Sexo", "Dependientes_Cat", "Ocupacion_Cat", "Procedencia_Cat"]
    ].to_dict()

# 1. Ubicamos los JSONs y aseguramos el orden alfabético
archivos_json = sorted(glob.glob("C:/Users/Julia/Desktop/txt_outputs/tmp/*.json"))
dir_txt = "C:/Users/Julia/Desktop/txt_outputs/"

# 2. EL DICCIONARIO PRINCIPAL
uwu = {}
secciones_ignoradas = {
    "TEXTO_NO_ASIGNADO",
    "I. Datos sociodemográficos y perfil profesional",
}

for indice_orden, ruta_json in enumerate(archivos_json):
    nombre_json = os.path.basename(ruta_json)  # Ej: "entrevista_01.txt.json"
    llave_maestra = obtener_llave_maestra(nombre_json)  # Ej: "entrevista_01"

    # C) Buscamos el TXT equivalente (usando la llave maestra)
    ruta_txt = os.path.join(dir_txt, f"{llave_maestra}.txt")
    texto_completo = ""
    if os.path.exists(ruta_txt):
        with open(ruta_txt, "r", encoding="utf-8") as f_txt:
            texto_completo = f_txt.read()

    # A y B) Capa 1: Origen como llave maestra con Metadata UNA sola vez
    uwu[nombre_json] = {
        "indice_orden": indice_orden,
        "metadata": meta_dict.get(llave_maestra, {}),  # ¡Ahora sí se llenará!
        "texto_completo_txt": texto_completo,
        "texto_segmentos": [],  # Nueva estructura que pediste
    }

    # Leemos el JSON correspondiente
    with open(ruta_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segmentos = data.get("output", {}).get("segmentos", [])

    # Capa 2: Llenamos texto_segmentos
    indice_interno = 0
    for seg in segmentos:
        sec = seg.get("seccion_entrevista")
        txt = seg.get("texto_literal")

        if sec and txt and sec not in secciones_ignoradas:
            uwu[nombre_json]["texto_segmentos"].append(
                {
                    "indice_interno": indice_interno,
                    "texto": txt,
                    "otros_datos": {
                        "seccion": sec
                        # Si luego quieres meter más datos del JSON, los agregas aquí
                    },
                }
            )
            indice_interno += 1  # Aumentamos el contador solo para segmentos válidos

# Listo. 'uwu' ahora tiene exactamente la arquitectura que definiste.


# ══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Config:
    spacy_model: str = "es_core_news_lg"
    use_bigrams: bool = True
    use_trigrams: bool = True
    min_bigram_freq: int = 2
    bigram_pos_patterns: List[Tuple[str, str]] = field(
        default_factory=lambda: [("NOUN", "NOUN"), ("ADJ", "NOUN"), ("NOUN", "ADJ")]
    )
    trigram_pos_patterns: List[Tuple[str, str, str]] = field(
        default_factory=lambda: [("ADV", "ADJ", "NOUN"), ("ADV", "VERB", "NOUN")]
    )
    stem_backend: str = "snowball"  # 'snowball' | 'none'

    min_uce_words: int = 3
    min_forms_uc: List[int] = field(default_factory=lambda: [13, 17])

    tsj: int = 3
    use_poisson_tsj: bool = False
    poisson_alpha: float = 0.05
    min_term_abs_freq: int = 2

    use_ctest: bool = False
    ctest_threshold: float = 0.3
    adaptive_ctest: bool = False
    ctest_alpha: float = 0.05
    n_permutations_ctest: int = 100

    use_cdh: bool = True

    pseudocount: float = 0.01
    swap_iterations: int = 2
    n_perm_cdh: int = 100
    perm_min_uc_size: int = 50
    min_r2_threshold: float = 0.05  # Minimum R² for a split to be accepted
    chi2_threshold_small: float = 3.84
    max_depth_cdh: int = 10
    min_cluster_size_cdh: float = 0.10

    clustering_method: str = "hdbscan"
    n_clusters: Optional[int] = None
    linkage_method: str = "ward"
    distance_metric: str = "jaccard"
    bootstrap_n_iter: int = 30
    min_cluster_overlap: float = 0.0
    hdbscan_min_cluster_size: int = 8
    hdbscan_min_samples: Optional[int] = 1
    bootstrap_sample_frac: float = 0.8
    hdbscan_cluster_selection_epsilon: float = 0.01
    cluster_selection_method: str = "silhouette"
    gap_n_references: int = 10

    fdr_alpha: float = 0.05

    use_projection: bool = True
    projection_method: str = "afc"

    use_cah_per_class: bool = True
    cah_per_class_top_terms: int = 50

    analyze_metadata: bool = True
    glm_method: str = "chi2"
    glm_alpha: float = 0.05
    n_permutations: int = 1000

    use_llm_synthesis: bool = True
    deepseek_api_key: str = ""
    llm_model: str = "deepseek-chat"
    synthesis_similarity_threshold: float = 0.6

    use_embeddings: bool = True  # Flag maestra
    embedding_model_name: str = "ibm-granite/granite-embedding-107m-multilingual"

    use_multivariate_analysis: bool = False
    multivariate_metadata: List[str] = field(default_factory=list)

    use_network_analysis: bool = True
    network_cooccurrence_threshold: int = 2
    network_weight_method: str = "ppmi"
    network_npmi_positive_only: bool = True
    network_significance_filter: bool = False
    network_significance_alpha: float = 0.05

    use_term_stability: bool = True
    term_stability_n_iter: int = 50

    # Robust RF + SHAP
    use_rf_shap: bool = False
    rf_n_estimators: int = 200
    rf_max_depth: int = 8
    rf_scale_features: bool = False
    rf_feature_selection_threshold: float = 0.01
    rf_outlier_method: str = "iqr"  # 'iqr' or 'zscore'
    rf_impute_strategy: str = "median"  # 'median', 'mean', 'constant'
    rf_cat_encoding: str = "frequency"  # 'onehot', 'frequency', 'target'
    rf_min_samples_for_tuning: int = 30

    optimize: bool = False
    optimize_trials: int = 50
    prune_small_clusters: bool = False
    min_class_size: int = 19
    db_local_path: str = "./data/workflow_data.json"
    subtlex_df_path: Optional[str] = None
    random_state: int = 42
    corpus_name: str = "Corpus ALCESTE"
    n_entrevistas: Optional[int] = None  # None → auto-counted from unique doc_ids
    fecha_analisis: Optional[str] = None  # None → auto-filled at runtime

    ppmi_k: float = 1.0
    analytic_temperature: float = 0.1

    # Method B (coref) comparison parameters
    coref_context_units_tight: int = 2  # B1: narrow window → conservative merges
    coref_context_units_loose: int = 4  # B2: wide window  → more chains, more merges

    # Updated mode table:
    #   "wc_only"    Method A only
    #   "coref_only" Method B only  (was "sim_only")
    #   "emb_only"   Method C only
    #   "wc_coref"   A + B          (was "wc_sim" / "both")
    #   "wc_emb"     A + C
    #   "coref_emb"  B + C          (was "sim_emb")
    #   "all"        A + B + C
    #   Aliases: "both"→"wc_coref", "sim_only"→"coref_only", "wc_sim"→"wc_coref", "sim_emb"→"coref_emb"
    classification_mode: str = "wc_only"

    # Method C — HDBSCAN on retrofitted UCE embedding space
    hdbscan_uce_min_cluster_size: int = 5  # tight pass
    hdbscan_uce_min_cluster_size_loose: int = 3  # loose pass (stability comparison)
    hdbscan_uce_epsilon: float = 0.0
    hdbscan_uce_min_samples: Optional[int] = 1
    hdbscan_uce_metric: str = "euclidean"  # "euclidean" | "cosine"


# ══════════════════════════════════════════════════════════════════════
# (Skipping boilerplate UCE, UC, CDHNode, Database, Segmentador for brevity - assume unchanged from base code except POS filtering)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class RetrofittingConfig:
    """Controls the three-way interpolation in retrofitting."""

    alpha: float = 0.8  # weight for original fastText vector
    beta: float = 0.5  # weight for Reinert class neighbors
    gamma: float = 0.3  # weight for PPMI-SVD corpus embedding
    n_iter: int = 10  # retrofitting iterations
    svd_components: int = 100
    chi2_threshold: float = 3.84  # significance threshold for constraint graph edges


@dataclass
class UCBuilderConfig:
    """
    All hyperparameters that govern cohesion-based UC construction.

    similarity_threshold : float  [0.0–1.0]
        Cohesion score below which a boundary is declared.
        Higher → more, smaller UCs (more boundaries).
        Lower  → fewer, larger UCs (fewer boundaries).
        Typical range searched by EnhancedOptimizador: [0.25, 0.65].

    coref_weight : float  [0.0–1.0]
        Relative weight of the coreference overlap signal vs. embedding
        similarity.  0.0 = embedding only; 1.0 = coref only.
        Typical range: [0.0, 0.4].

    similarity_weight : float
        Complement weight; auto-computed as (1 − coref_weight) unless set.

    window_size : int  [1–5]
        Number of neighbors used for moving-average smoothing of cohesion
        scores.  1 = no smoothing (raw adjacent pairs).

    min_drop : float  [0.0–0.3]
        Minimum gap below threshold for a boundary to be accepted.
        Prevents spurious boundaries caused by small noise fluctuations.

    section_hard_boundary : bool
        Always insert a UC boundary when the UCE's section (seccion) changes.
        Strongly recommended: True.

    min_gap : int
        Minimum number of UCEs between two boundaries.
        Prevents micro-UCs of 1–2 UCEs that are too small for CHD.
    """

    similarity_threshold: float = 0.45
    coref_weight: float = 0.20
    window_size: int = 3
    min_drop: float = 0.05
    section_hard_boundary: bool = True
    min_gap: int = 2

    @property
    def similarity_weight(self) -> float:
        return 1.0 - self.coref_weight


@dataclass
class UC:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    texto: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    uce_ids: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    stems: List[str] = field(default_factory=list)
    bigrams: List[Tuple[str, str]] = field(default_factory=list)
    bigram_stems: List[Tuple[str, str]] = field(default_factory=list)
    trigrams: List[Tuple[str, str, str]] = field(default_factory=list)
    trigram_stems: List[Tuple[str, str, str]] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    cluster_id: Optional[int] = None
    is_stable: bool = False
    coordinates: Dict[str, List[float]] = field(default_factory=dict)
    stability: Optional[float] = None

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, data):
        # Drop unknown keys so old DB records don't crash new code
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class CDHNode:
    label: int = -1
    depth: int = 0
    n_ucs: int = 0
    is_leaf: bool = False
    indices: List[int] = field(default_factory=list)
    children: List["CDHNode"] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "depth": self.depth,
            "n_ucs": self.n_ucs,
            "is_leaf": self.is_leaf,
            "indices": self.indices,
            "children": [c.to_dict() for c in self.children],
        }


# ══════════════════════════════════════════════════════════════════════
# BASE DE DATOS
# ══════════════════════════════════════════════════════════════════════


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


def normalizar_id_uce(uce: UCE) -> str:
    if uce.id and not _es_uuid(uce.id):
        return uce.id

    doc_id = uce.doc_id
    local_idx = uce.local_idx
    # Stop using uce.seccion. Grab the integer from metadata.
    section_id = uce.metadata.get("section_id", "0") if uce.metadata else "0"

    if doc_id is not None and local_idx is not None:
        nuevo_id = f"{doc_id}_{section_id}_{local_idx}"
        print(f"UCE con ID '{uce.id}' se renombrará a '{nuevo_id}'")
        return nuevo_id
    else:
        print(f"No se puede reparar ID de UCE {uce.id}: faltan doc_id o local_idx")
        return uce.id


def _es_uuid(id_str: str) -> bool:

    return bool(re.match(r"^[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$", id_str, re.I))


class UCEVectorizer:
    """
    Computes a single dense vector per UCE using SUBTLEX-CD inverse weighting.

    Rare words (low contextual diversity) contribute more to the centroid
    because they are topically specific.  Common words like *entonces* or
    *ser* are structurally important but thematically neutral — downweighting
    them improves boundary detection precision.

    Optional: if retrofitted_vectors is supplied (post-CHD second pass),
    those are used instead of raw fastText vectors.
    """

    def __init__(
        self,
        we_analyzer: WordEmbeddingsAnalyzer,
        subtlex_analyzer: SubtlexAnalyzer,
    ):
        self.we = we_analyzer
        self.subtlex = subtlex_analyzer
        self._cache: Dict[str, np.ndarray] = {}

    def _token_weight(self, lemma: str) -> float:
        """1 / (SUBTLEX_CD + ε) — rare tokens get higher weight."""
        cd = self.subtlex.cd(lemma)
        if cd is not None and cd > 0:
            return 1.0 / (cd + 1e-6)
        return 1.0  # neutral fallback for OOV

    def vectorize(
        self,
        uce: UCE,
        retrofitted_vectors: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Returns the SUBTLEX-CD weighted mean of token vectors for a UCE.
        Result is L2-normalized (unit norm) for cosine efficiency.
        Cached by UCE id.
        """
        cache_key = f"{uce.id}_{'retro' if retrofitted_vectors else 'base'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Use content lemmas if available; fall back to all lemmas
        tokens = uce.lemmas if uce.lemmas else uce.tokens
        if not tokens:
            zero = np.zeros(self.we.vector_size)
            self._cache[cache_key] = zero
            return zero

        vecs, weights = [], []
        for lemma in tokens:
            if retrofitted_vectors and lemma in retrofitted_vectors:
                vec = retrofitted_vectors[lemma]
            else:
                vec = self.we.vector(lemma)

            if np.linalg.norm(vec) < 1e-8:
                continue  # skip zero vectors (OOV with no subword)

            w = self._token_weight(lemma)
            vecs.append(vec)
            weights.append(w)

        if not vecs:
            zero = np.zeros(self.we.vector_size)
            self._cache[cache_key] = zero
            return zero

        w_arr = np.array(weights, dtype=np.float64)
        w_arr /= w_arr.sum()
        result = np.average(vecs, weights=w_arr, axis=0)

        # L2-normalize for cosine
        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result /= norm

        self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        self._cache.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  COHESION SCORER
# ══════════════════════════════════════════════════════════════════════════════


class CohesionScorer:
    """
    Computes scalar cohesion between two adjacent UCEs.

    cohesion(a, b) = sim_weight · cos(vec_a, vec_b)
                   + coref_weight · coref_overlap(a, b)

    Coreference overlap
    ────────────────────
    Uses uce.coref_chains, which contains chains FILTERED to mentions within
    that UCE (populated by PipelineGramatical.procesar_desde_uces).
    A chain is considered "spanning" if its representative appears in both UCEs.

    Why not use global chain offsets?
    ──────────────────────────────────
    The local chain list is sufficient: if chain C has at least one mention in
    UCE[i] AND at least one in UCE[i+1], the same representative appears in
    both local chain lists.  No need to access the full document chain.
    """

    def __init__(self, vectorizer: UCEVectorizer, cfg: UCBuilderConfig):
        self.vec = vectorizer
        self.cfg = cfg

    def _coref_overlap(self, uce_a: UCE, uce_b: UCE) -> float:
        """
        Jaccard-like score: |chains_a ∩ chains_b| / |chains_a ∪ chains_b|
        Returns 0.0 if neither UCE has any chains.
        """
        chains_a = {c["representative"] for c in (uce_a.coref_chains or [])}
        chains_b = {c["representative"] for c in (uce_b.coref_chains or [])}

        union = chains_a | chains_b
        if not union:
            return 0.0

        intersection = chains_a & chains_b
        return len(intersection) / len(union)

    def score(
        self,
        uce_a: UCE,
        uce_b: UCE,
        retrofitted_vectors: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        """Returns cohesion score in [0, 1] between two adjacent UCEs."""
        vec_a = self.vec.vectorize(uce_a, retrofitted_vectors)
        vec_b = self.vec.vectorize(uce_b, retrofitted_vectors)

        # Cosine similarity (vectors are already L2-normalized)
        cos_sim = float(np.dot(vec_a, vec_b))
        cos_sim = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))  # remap [-1,1] → [0,1]

        coref_sim = self._coref_overlap(uce_a, uce_b)
        print(self.cfg.similarity_weight, cos_sim, self.cfg.coref_weight, coref_sim)
        return self.cfg.similarity_weight * cos_sim + self.cfg.coref_weight * coref_sim

    def score_sequence(
        self,
        doc_uces: List[UCE],
        retrofitted_vectors: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[float]:
        """
        Returns n-1 cohesion scores for a doc of n UCEs.
        Score[i] = cohesion between UCE[i] and UCE[i+1].
        """
        n = len(doc_uces)
        if n < 2:
            return []
        return [
            self.score(doc_uces[i], doc_uces[i + 1], retrofitted_vectors)
            for i in range(n - 1)
        ]

    @staticmethod
    def smooth(scores: List[float], window: int) -> List[float]:
        """
        Moving average smoothing over cohesion score sequence.
        Uses symmetric window; pads edges with edge values.
        """
        if window <= 1 or len(scores) <= 1:
            return list(scores)

        half = window // 2
        padded = [scores[0]] * half + list(scores) + [scores[-1]] * half
        smoothed = []
        for i in range(len(scores)):
            chunk = padded[i : i + window]
            smoothed.append(sum(chunk) / len(chunk))
        return smoothed


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BOUNDARY DETECTOR
# ══════════════════════════════════════════════════════════════════════════════


class BoundaryDetector:
    """
    Finds UC boundaries from smoothed cohesion scores.

    A boundary is placed AFTER position i (i.e., UCE[i] ends one UC,
    UCE[i+1] starts the next) when ALL of:
      1. smoothed_cohesion[i] < similarity_threshold
      2. smoothed_cohesion[i] is a local minimum (≤ both neighbours)
      3. The drop below threshold ≥ min_drop
      4. At least min_gap UCEs have passed since the last boundary

    Additionally, section changes (seccion field) always produce a boundary
    when section_hard_boundary=True.
    """

    def __init__(self, cfg: UCBuilderConfig):
        self.cfg = cfg

    def detect(
        self,
        doc_uces: List[UCE],
        smoothed_scores: List[float],
    ) -> List[int]:
        """
        Returns a list of boundary positions.
        Position b means: UCEs[0..b] form one UC; UCEs[b+1..] start the next.
        Positions are 0-indexed into the gaps BETWEEN UCEs.
        Last position is always n-1 (end of document).
        """
        n = len(doc_uces)
        if n < 2:
            return [n - 1] if n == 1 else []

        boundaries: List[int] = []
        last_boundary = -1

        for i, score in enumerate(smoothed_scores):
            # ── Section hard boundary ───────────────────────────────────────
            if (
                self.cfg.section_hard_boundary
                and i + 1 < n
                and doc_uces[i].seccion != doc_uces[i + 1].seccion
            ):
                if i - last_boundary >= self.cfg.min_gap:
                    boundaries.append(i)
                    last_boundary = i
                continue

            # ── Gap constraint ──────────────────────────────────────────────
            if i - last_boundary < self.cfg.min_gap:
                continue

            # ── Threshold check ─────────────────────────────────────────────
            drop = self.cfg.similarity_threshold - score
            if drop < self.cfg.min_drop:
                continue

            # ── Local minimum check ─────────────────────────────────────────
            left_ok = (i == 0) or (smoothed_scores[i] <= smoothed_scores[i - 1])
            right_ok = (i == len(smoothed_scores) - 1) or (
                smoothed_scores[i] <= smoothed_scores[i + 1]
            )
            if not (left_ok and right_ok):
                continue

            boundaries.append(i)
            last_boundary = i

        # Always close at end of document
        boundaries.append(n - 1)
        return boundaries

    @staticmethod
    def boundaries_to_segments(
        boundaries: List[int],
        n_uces: int,
    ) -> List[Tuple[int, int]]:
        """
        Converts boundary list to (start, end) index pairs (inclusive).
        """
        segments = []
        start = 0
        for b in sorted(set(boundaries)):
            segments.append((start, b))
            start = b + 1
        if start < n_uces:
            segments.append((start, n_uces - 1))
        return segments


# ══════════════════════════════════════════════════════════════════════════════
# 5.  UC BUILDER  (main class)
# ══════════════════════════════════════════════════════════════════════════════


class UCBuilder:
    """
    Drop-in replacement for SegmentadorALCESTE.construir_ucs().

    Public interface (identical to the old method):

        ucs, uce_to_uc = uc_builder.build(
            uces_por_doc, min_forms, doc_metadata_map,
            retrofitted_vectors=None
        )

    Where:
        uces_por_doc     : List[List[UCE]]  (output of segmentar_en_uces)
        min_forms        : int              (minimum content lemmas per UC)
        doc_metadata_map : Dict[int, Dict]  (doc_id → metadata)
        retrofitted_vectors : Optional dict for second-pass refinement

    Returns:
        ucs       : List[UC]
        uce_to_uc : Dict[str, str]  (uce_id → uc_id)
    """

    def __init__(
        self,
        we_analyzer: WordEmbeddingsAnalyzer,
        subtlex_analyzer: SubtlexAnalyzer,
        cfg: UCBuilderConfig,
        alc_config: Config,
    ):
        self.cfg = cfg
        self.alc_config = alc_config
        self.vectorizer = UCEVectorizer(we_analyzer, subtlex_analyzer)
        self.scorer = CohesionScorer(self.vectorizer, cfg)
        self.detector = BoundaryDetector(cfg)

    # ── Segment a single document ────────────────────────────────────────────
    def _segment_doc(
        self,
        doc_uces: List[UCE],
        retrofitted_vectors: Optional[Dict[str, np.ndarray]],
    ) -> List[Tuple[int, int]]:
        """Returns (start, end) index pairs for UCs within one document."""
        if len(doc_uces) == 1:
            return [(0, 0)]

        raw_scores = self.scorer.score_sequence(doc_uces, retrofitted_vectors)
        smoothed = CohesionScorer.smooth(raw_scores, self.cfg.window_size)
        boundaries = self.detector.detect(doc_uces, smoothed)
        return BoundaryDetector.boundaries_to_segments(boundaries, len(doc_uces))

    # ── Merge under-sized segments ───────────────────────────────────────────
    def _merge_small_segments(
        self,
        doc_uces: List[UCE],
        segments: List[Tuple[int, int]],
        min_forms: int,
        retrofitted_vectors: Optional[Dict[str, np.ndarray]],
    ) -> List[Tuple[int, int]]:
        """
        Enforces min_forms as a FLOOR by merging segments that are too small.
        A segment's size is measured by its content lemma count (same criterion
        as the old construir_ucs).

        Merging strategy:
          For each small segment, merge it with the neighbour that produces
          the highest mean internal cohesion after merging (greedy).
          This is O(n_segments²) in the worst case but n_segments is small.
        """

        def content_forms(start: int, end: int) -> int:
            return sum(len(doc_uces[i].lemmas) for i in range(start, end + 1))

        def seg_vec(start: int, end: int) -> np.ndarray:
            vecs = [
                self.vectorizer.vectorize(doc_uces[i], retrofitted_vectors)
                for i in range(start, end + 1)
            ]
            result = np.mean(vecs, axis=0)
            norm = np.linalg.norm(result)
            return result / norm if norm > 1e-8 else result

        changed = True
        while changed and len(segments) > 1:
            changed = False
            for idx, (s, e) in enumerate(segments):
                if content_forms(s, e) >= min_forms:
                    continue

                # Find best merge neighbour
                best_score = -1.0
                best_nb = -1

                for nb in [idx - 1, idx + 1]:
                    if not (0 <= nb < len(segments)):
                        continue
                    ns, ne = segments[nb]
                    # Cohesion of merged segment
                    merged_vec = seg_vec(min(s, ns), max(e, ne))
                    nb_vec = seg_vec(ns, ne)
                    cur_vec = seg_vec(s, e)
                    score = float(np.dot(merged_vec, cur_vec)) + float(
                        np.dot(merged_vec, nb_vec)
                    )
                    if score > best_score:
                        best_score = score
                        best_nb = nb

                if best_nb == -1:
                    continue  # isolated; can't merge

                ns, ne = segments[best_nb]
                merged = (min(s, ns), max(e, ne))
                lo, hi = sorted([idx, best_nb])
                segments = segments[:lo] + [merged] + segments[hi + 1 :]
                changed = True
                break  # restart scan after any merge

        return segments

    # ── Build UC objects from a segment ─────────────────────────────────────
    def _make_uc(
        self,
        doc_uces: List[UCE],
        start: int,
        end: int,
        uc_local_idx: int,
        doc_metadata_map: Optional[Dict],
    ) -> UC:
        """Assembles a UC dataclass from a contiguous slice of UCEs."""
        segment = doc_uces[start : end + 1]
        doc_id = segment[0].doc_id
        shared_meta = (doc_metadata_map or {}).get(doc_id, {})

        uce_ids = [u.id for u in segment]
        lemmas, stems, bigrams, bigram_stems = [], [], [], []
        trigrams, trigram_stems = [], []
        texts = []

        for u in segment:
            texts.append(u.texto)
            lemmas.extend(u.lemmas)
            stems.extend(u.stems)
            bigrams.extend(u.bigrams)
            bigram_stems.extend(u.bigram_stems)
            trigrams.extend(getattr(u, "trigrams", []))
            trigram_stems.extend(getattr(u, "trigram_stems", []))

        uc_id = f"uc_{doc_id}_{uc_local_idx}"

        uc = UC(
            id=uc_id,
            texto=" ".join(texts),
            metadata={**shared_meta, "uce_local_idx": uce_ids[0]},
            uce_ids=uce_ids,
            lemmas=lemmas,
            stems=stems,
            bigrams=bigrams,
            bigram_stems=bigram_stems,
            trigrams=trigrams,
            trigram_stems=trigram_stems,
        )
        return uc

    # ── Main public method ───────────────────────────────────────────────────
    def build(
        self,
        uces_por_doc: List[List[UCE]],
        min_forms: int,
        doc_metadata_map: Optional[Dict] = None,
        retrofitted_vectors: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[List[UC], Dict[str, str]]:
        """
        Cohesion-based UC construction.

        Parameters
        ──────────
        uces_por_doc         : one inner list per document
        min_forms            : minimum content lemmas per UC (floor)
        doc_metadata_map     : doc_id → metadata dict (same as old API)
        retrofitted_vectors  : optional; use for second-pass refinement

        Returns
        ───────
        ucs        : List[UC]
        uce_to_uc  : Dict[uce_id, uc_id]
        """
        ucs: List[UC] = []
        uce_to_uc: Dict[str, str] = {}
        global_uc_counter = 0

        for doc_uces in uces_por_doc:
            if not doc_uces:
                continue

            # 1. Detect natural discourse segments
            segments = self._segment_doc(doc_uces, retrofitted_vectors)

            # 2. Enforce min_forms floor via merging
            segments = self._merge_small_segments(
                doc_uces, segments, min_forms, retrofitted_vectors
            )

            # 3. Build UC objects
            for seg_idx, (start, end) in enumerate(segments):
                uc = self._make_uc(
                    doc_uces, start, end, global_uc_counter, doc_metadata_map
                )
                ucs.append(uc)

                # Back-propagate uc_id to UCE objects and build mapping
                for uid in uc.uce_ids:
                    uce_to_uc[uid] = uc.id
                    for u in doc_uces[start : end + 1]:
                        if u.id == uid:
                            u.uc_id = uc.id

                global_uc_counter += 1

        # Integrity check (same guarantee as the original)
        all_ids = [uid for uc in ucs for uid in uc.uce_ids]
        assert len(all_ids) == len(set(all_ids)), "UCE assigned to multiple UCs"

        print(
            f"   [UCBuilder] {len(ucs)} UCs (min_forms={min_forms}, "
            f"threshold={self.cfg.similarity_threshold:.2f})"
        )
        return ucs, uce_to_uc


class Database:
    def __init__(self, config: Config):
        self.config = config
        self.path = config.db_local_path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._load()
        self.doc_metadata = {}  # new

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {
                "uces": [],
                "ucs": [],
                "sintesis_por_clase": [],
                "terminos": [],
                "vocabulario": [],
                "multivariate": {},
                "network": {},
                "term_stability": [],
                "forma_index": {},
                "cah_terminos": {},
                "afc_result": {},
                "cdh_tree_umbral1": {},
                "cdh_tree_umbral2": {},
                "shap_analysis": {},
            }
        # Load doc_metadata if present
        self.doc_metadata = self.data.get("doc_metadata", {})
        # For backward compatibility, if old UCEs have full metadata, we can extract them

    def _save(self):
        self.data["doc_metadata"] = self.doc_metadata  # include in save
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    def _upsert(self, collection_name: str, items: List, key: str = "id"):
        existing = {x[key] for x in self.data.get(collection_name, [])}
        for item in items:
            d = item.to_dict()
            if d[key] not in existing:
                self.data.setdefault(collection_name, []).append(d)
            else:
                self.data[collection_name] = [
                    x if x[key] != d[key] else d for x in self.data[collection_name]
                ]

    def save_uces(self, uces):
        self.data["uces"] = [u.to_dict() for u in uces]

    def save_ucs(self, ucs):
        self._upsert("ucs", ucs)

    def save_terminos(self, df):
        self.data["terminos"] = df.to_dict(orient="records")

    def save_sintesis(self, s):
        self.data["sintesis_por_clase"].append(s)

    def save_multivariate(self, r):
        self.data["multivariate"] = r

    def save_network(self, r):
        self.data["network"] = r

    def save_term_stability(self, df):
        self.data["term_stability"] = df.to_dict(orient="records")

    def load_ucs(self):
        return [UC.from_dict(u) for u in self.data.get("ucs", [])]

    def save_doc_metadata(self, doc_id: int, meta: Dict):
        self.doc_metadata[str(doc_id)] = meta

    def get_doc_metadata(self, doc_id: int) -> Dict:
        return self.doc_metadata.get(str(doc_id), {})

    def load_uces(self):
        uce_dicts = self.data.get("uces", [])
        uces = []
        for d in uce_dicts:
            doc_id = d.get("doc_id")
            local_idx = d.get("local_idx")
            # Same thing here: grab section_id from the metadata dict
            section_id = d.get("metadata", {}).get("section_id", "0")

            if doc_id is not None and local_idx is not None:
                d["id"] = f"{doc_id}_{section_id}_{local_idx}"
            else:
                import uuid

                d["id"] = str(uuid.uuid4())
                print(f"UCE sin doc_id/local_idx, se asignó UUID: {d['id']}")
            uces.append(UCE.from_dict(d))
        return uces


# ══════════════════════════════════════════════════════════════════════
# BERTOPIC COMPLEMENTARIO (Placeholder)
# ══════════════════════════════════════════════════════════════════════


class BERTopicAnalyzer:
    def __init__(self, config: Config):
        self.config = config

    def run(self, uces: List[UCE]) -> Dict[str, Any]:
        """
        Placeholder para la integración de BERTopic (Roadmap V.3).
        Devuelve la estructura vacía esperada por el dashboard.
        """
        return {
            "is_active": False,
            "topics": [],
            "document_info": [],
            "mapping_alceste": {},
        }


# ══════════════════════════════════════════════════════════════════════
# SEGMENTACIÓN ALCESTE
# ══════════════════════════════════════════════════════════════════════


class SegmentadorALCESTE:
    def __init__(self, config: Config):
        self.config = config
        self.nlp = spacy.load(config.spacy_model)
        self.content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        self.stemmer = (
            SnowballStemmer("spanish") if config.stem_backend == "snowball" else None
        )
        self.doc_metadata_map = {}

    def _segmentar_reinert_spacy(self, doc) -> List:
        """Segment a full document Doc into UCE spans with global char offsets."""
        if len(doc) == 0:
            return []

        ideal_size = getattr(self.config, "uce_target_size", 40)
        window = int(ideal_size * 0.4)

        weights = np.array(
            [
                6.0
                if t.text.strip() in [".", "?", "!", "…"]
                else 5.0
                if t.text.strip() == ":"
                else 4.0
                if t.text.strip() == ";"
                else 1.0
                if t.text.strip() == ","
                else 0.01
                for t in doc
            ]
        )

        total_tokens = len(doc)
        segments = []
        cursor = 0

        while cursor < total_tokens:
            remaining = total_tokens - cursor
            if remaining <= ideal_size + window:
                span = doc[cursor:]
                segments.append(span)
                break

            start_search = max(0, ideal_size - window)
            end_search = min(ideal_size + window, remaining)

            if start_search >= end_search:
                span = doc[cursor:]
                segments.append(span)
                break

            indices_ventana = np.arange(start_search, end_search)
            pesos_ventana = weights[cursor + indices_ventana]
            distancias = np.abs(indices_ventana - ideal_size)
            puntuaciones_corte = pesos_ventana / (distancias + 1)

            mejor_punto_local = np.argmax(puntuaciones_corte)
            punto_corte_global = cursor + indices_ventana[mejor_punto_local]

            span = doc[cursor : punto_corte_global + 1]
            segments.append(span)
            cursor = punto_corte_global + 1

        return segments

    def segmentar_en_uces(self, corpus_raw):
        uces_por_doc = []
        self.doc_metadata_map = {}
        self.section_registry = {}

        for doc_idx, (origen_key, doc_data) in enumerate(corpus_raw.items()):
            metadata_global = doc_data.get("metadata", {}).copy()
            shared_meta = {
                **metadata_global,
                "origen": origen_key,
                "doc_idx": doc_idx,
                "indice_orden": doc_data.get("indice_orden"),
                "texto_completo_txt": doc_data.get("texto_completo_txt", ""),
            }
            self.doc_metadata_map[doc_idx] = shared_meta

            segmentos = doc_data.get("texto_segmentos", [])
            if not segmentos:
                continue

            doc_uces = []
            local_idx = 0
            section_registry: Dict[str, int] = {}
            local_section_counter = 0

            # Stop guessing with character offsets. Read the damn JSON segments directly.
            for seg in segmentos:
                seg_text = seg.get("texto", "").strip()
                seccion = seg.get("otros_datos", {}).get("seccion", "Sin_Seccion")

                if not seg_text:
                    continue

                if seccion not in section_registry:
                    section_registry[seccion] = local_section_counter
                    local_section_counter += 1
                section_id = section_registry[seccion]

                self.section_registry[f"{doc_idx}_{section_id}"] = {
                    "origen": origen_key,
                    "seccion": seccion,
                }

                # Parse just this perfectly-sectioned segment with Spacy
                doc_spacy = self.nlp(seg_text)
                spans = self._segmentar_reinert_spacy(doc_spacy)

                for span in spans:
                    if len(span.text.split()) < self.config.min_uce_words:
                        continue

                    uce_id = f"{doc_idx}_{section_id}_{local_idx}"

                    uce = UCE(
                        id=uce_id,
                        texto=span.text,
                        metadata={
                            "uce_local_idx": local_idx,
                            "seccion": seccion,
                            "section_id": section_id,
                        },
                        doc_id=doc_idx,
                        local_idx=local_idx,
                        seccion=seccion,
                        start_char=span.start_char,
                        end_char=span.end_char,
                        verbos=[],
                        adverbios=[],
                    )
                    doc_uces.append(uce)
                    local_idx += 1

            if doc_uces:
                uces_por_doc.append(doc_uces)

        total = sum(len(d) for d in uces_por_doc)
        print(f"   {total} UCEs en {len(uces_por_doc)} documentos.")
        return uces_por_doc, self.doc_metadata_map

    def lematizar_uces(self, uces: List[UCE]) -> List[UCE]:
        stemmer = self.stemmer
        for uce in uces:
            doc = self.nlp(uce.texto)
            tokens_raw = [
                (i, t.text, t.lemma_.lower(), t.pos_, t.is_stop, t.is_alpha)
                for i, t in enumerate(doc)
            ]
            lemmas, stems, pos_tags, formas_tokens = [], [], [], []
            marcadores = []
            bigrams, bigram_stems = [], []
            trigrams, trigram_stems = [], []
            all_tokens = []  # ← ADD THIS

            for i, text, lemma, pos, is_stop, is_alpha in tokens_raw:
                if is_alpha and pos != "PUNCT":  # ← ADD THIS
                    all_tokens.append(text)  # ← ADD THIS
                if pos in self.content_pos and not is_stop and is_alpha:
                    stem = stemmer.stem(lemma) if stemmer else lemma
                    lemmas.append(lemma)
                    stems.append(stem)
                    pos_tags.append(pos)
                    formas_tokens.append(
                        {
                            "forma": text,
                            "lemma": lemma,
                            "stem": stem,
                            "pos": pos,
                            "pos_idx": i,
                        }
                    )
                elif is_alpha and (
                    is_stop
                    or pos in {"PRON", "SCONJ", "CCONJ", "DET", "AUX", "ADP", "INTJ"}
                ):
                    marcadores.append(
                        {"forma": text.lower(), "lemma": lemma, "pos": pos}
                    )
            if self.config.use_bigrams:
                for j in range(len(tokens_raw) - 1):
                    t1, t2 = tokens_raw[j], tokens_raw[j + 1]
                    if (t1[3], t2[3]) in self.config.bigram_pos_patterns:
                        l1, l2 = t1[2], t2[2]
                        bigrams.append((l1, l2))
                        bigram_stems.append(
                            (
                                stemmer.stem(l1) if stemmer else l1,
                                stemmer.stem(l2) if stemmer else l2,
                            )
                        )

            if self.config.use_trigrams:
                for j in range(len(tokens_raw) - 2):
                    t1, t2, t3 = tokens_raw[j], tokens_raw[j + 1], tokens_raw[j + 2]
                    if (t1[3], t2[3], t3[3]) in self.config.trigram_pos_patterns:
                        l1, l2, l3 = t1[2], t2[2], t3[2]
                        trigrams.append((l1, l2, l3))
                        trigram_stems.append(
                            (
                                stemmer.stem(l1) if stemmer else l1,
                                stemmer.stem(l2) if stemmer else l2,
                                stemmer.stem(l3) if stemmer else l3,
                            )
                        )

            uce.lemmas = lemmas
            uce.stems = stems
            uce.pos_tags = pos_tags
            uce.formas_tokens = formas_tokens
            uce.bigrams = bigrams
            uce.bigram_stems = bigram_stems
            uce.trigrams = trigrams
            uce.trigram_stems = trigram_stems
            uce.marcadores = marcadores
            uce.tokens = all_tokens

        return uces

    def construir_ucs(self, uces_por_doc, min_forms, doc_metadata_map):
        ucs: List[UC] = []
        uce_to_uc: Dict[str, str] = {}

        for doc_uces in uces_por_doc:
            acum_ids: List[str] = []
            acum_text: List[str] = []
            acum_lemmas: List[str] = []
            acum_stems: List[str] = []
            acum_bigrams: List[Tuple] = []
            acum_bigram_stems: List[Tuple] = []
            acum_trigrams: List[Tuple] = []
            acum_trigram_stems: List[Tuple] = []
            acum_forms = 0

            for uce in doc_uces:
                acum_ids.append(uce.id)
                acum_text.append(uce.texto)
                acum_lemmas.extend(uce.lemmas)
                acum_stems.extend(uce.stems)
                acum_bigrams.extend(uce.bigrams)
                acum_bigram_stems.extend(uce.bigram_stems)
                acum_trigrams.extend(uce.trigrams)
                acum_trigram_stems.extend(uce.trigram_stems)
                acum_forms += len(uce.lemmas)

                if acum_forms >= min_forms:
                    doc_id = doc_uces[0].doc_id
                    shared = (doc_metadata_map or {}).get(doc_id, {})
                    full_meta = {**shared, "uce_local_idx": acum_ids[0]}
                    uc = UC(
                        texto=" ".join(acum_text),
                        metadata=doc_uces[0].metadata.copy(),
                        uce_ids=list(acum_ids),
                        lemmas=list(acum_lemmas),
                        stems=list(acum_stems),
                        bigrams=list(acum_bigrams),
                        bigram_stems=list(acum_bigram_stems),
                        trigrams=list(acum_trigrams),
                        trigram_stems=list(acum_trigram_stems),
                    )
                    ucs.append(uc)
                    for uid in acum_ids:
                        uce_to_uc[uid] = uc.id
                        for u in doc_uces:
                            if u.id == uid:
                                u.uc_id = uc.id
                    # Reset
                    acum_ids = []
                    acum_text = []
                    acum_lemmas = []
                    acum_stems = []
                    acum_bigrams = []
                    acum_bigram_stems = []
                    acum_trigrams = []
                    acum_trigram_stems = []
                    acum_forms = 0

            if acum_ids:
                doc_id = doc_uces[0].doc_id
                shared = (doc_metadata_map or {}).get(doc_id, {})
                full_meta = {**shared, "uce_local_idx": acum_ids[0]}
                uc = UC(
                    texto=" ".join(acum_text),
                    metadata=doc_uces[0].metadata.copy(),
                    uce_ids=list(acum_ids),
                    lemmas=list(acum_lemmas),
                    stems=list(acum_stems),
                    bigrams=list(acum_bigrams),
                    bigram_stems=list(acum_bigram_stems),
                    trigrams=list(acum_trigrams),
                    trigram_stems=list(acum_trigram_stems),
                )
                ucs.append(uc)
                for uid in acum_ids:
                    uce_to_uc[uid] = uc.id
                    for u in doc_uces:
                        if u.id == uid:
                            u.uc_id = uc.id

        print(f"   {len(ucs)} UCs with ≥{min_forms} forms.")
        all_ids = [uid for uc in ucs for uid in uc.uce_ids]
        assert len(all_ids) == len(set(all_ids)), "UCE in multiple UCs"
        return ucs, uce_to_uc

    def construir_forma_index(
        self,
        uces_estables: List[UCE],
        labels_uces: np.ndarray,
    ) -> Dict[int, Dict[str, Dict[str, Dict[str, int]]]]:
        index: Dict[int, Dict] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        )

        for uce, label in zip(uces_estables, labels_uces):
            c = int(label)

            # 1. Unigrams
            for tok in uce.formas_tokens:
                index[c][tok["stem"]][tok["lemma"]][tok["forma"]] += 1

            # 2. Bigrams (Store as forma = lemma)
            if hasattr(uce, "bigrams") and hasattr(uce, "bigram_stems"):
                for (l1, l2), (s1, s2) in zip(uce.bigrams, uce.bigram_stems):
                    lem = f"{l1}_{l2}"
                    stm = f"{s1}_{s2}"
                    index[c][stm][lem][lem] += 1

            # 3. Trigrams
            if hasattr(uce, "trigrams") and hasattr(uce, "trigram_stems"):
                for (l1, l2, l3), (s1, s2, s3) in zip(uce.trigrams, uce.trigram_stems):
                    lem = f"{l1}_{l2}_{l3}"
                    stm = f"{s1}_{s2}_{s3}"
                    index[c][stm][lem][lem] += 1

        return {
            clase: {
                stem: {lemma: dict(formas) for lemma, formas in lemmas.items()}
                for stem, lemmas in stems.items()
            }
            for clase, stems in index.items()
        }

    def generar_embeddings_ucs(self, ucs: List[UC]) -> List[UC]:
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            print("   SentenceTransformer no disponible.")
            return ucs
        model = SentenceTransformer(self.config.embedding_model_name)
        embs = model.encode([uc.texto for uc in ucs], show_progress_bar=True)
        for uc, emb in zip(ucs, embs):
            uc.embedding = emb.tolist()
        return ucs

    def generar_embeddings_uces(self, uces: List[UCE]) -> List[UCE]:
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            print("   SentenceTransformer no disponible.")
            return uces
        model = SentenceTransformer(self.config.embedding_model_name)
        embs = model.encode(
            [u.texto for u in uces],
            show_progress_bar=True,
            batch_size=min(64, len(uces)),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        for uce, emb in zip(uces, embs):
            uce.embedding = emb.tolist()
        return uces


# ══════════════════════════════════════════════════════════════════════
# MATRIZ BUILDER (ALCESTE: STRICT BINARY & DOC-FREQ TSJ)
# ══════════════════════════════════════════════════════════════════════


class EmbeddingSpecializationPipeline:
    """
    Produces retrofitted word vectors that are simultaneously:
      • Grounded in general Spanish semantics (fastText cc.es.300.bin)
      • Bent toward the corpus's thematic geometry (Reinert CHD classes)
      • Frequency-normalized (SUBTLEX-CD background distribution)

    Stages
    ──────
    8a  SVD factorization of PPMI matrix → corpus-specific term embeddings
    8b  Procrustes alignment of corpus space → fastText space
    8c  Build Reinert constraint graph from CHD class assignments + chi² scores
    8d  Retrofitting: closed-form iterative update (Faruqui et al. 2015,
        extended to three-way interpolation)
    """

    def __init__(
        self,
        we_analyzer: WordEmbeddingsAnalyzer,
        retro_config: RetrofittingConfig = None,
    ):
        self.we_analyzer = we_analyzer
        self.cfg = retro_config or RetrofittingConfig()
        self.corpus_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.retrofitted_vectors: Optional[Dict[str, np.ndarray]] = None
        self.class_centroids: Optional[Dict[int, np.ndarray]] = None

    # ── Stage 8a ─────────────────────────────────────────────────────────────
    def _svd_corpus_embeddings(
        self,
        ppmi_matrix: csr_matrix,
        vocab: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Factorize the PPMI matrix via truncated SVD.
        Result encodes the Reinert thematic structure geometrically.
        """
        if ppmi_matrix.shape[0] < 2 or ppmi_matrix.shape[1] < 2:
            logger.warning(
                "SVD: matrix too small (%s). Returning empty corpus embeddings.",
                ppmi_matrix.shape,
            )
            return {}
        n_components = min(self.cfg.svd_components, min(ppmi_matrix.shape) - 1)
        if n_components < 1:
            logger.warning(
                "SVD: n_components=%d (matrix shape %s). Returning empty corpus embeddings.",
                n_components,
                ppmi_matrix.shape,
            )
            return {}
        U, Sigma, Vt = randomized_svd(
            ppmi_matrix,
            n_components=n_components,
            n_iter=10,
            random_state=42,
        )
        # Weight by sqrt(Sigma) — standard for word vector tasks
        term_embeddings = U * np.sqrt(Sigma)  # (n_terms, n_components)
        return {w: term_embeddings[i] for i, w in enumerate(vocab)}

    # ── Stage 8b ─────────────────────────────────────────────────────────────
    def _procrustes_align(
        self,
        corpus_emb: Dict[str, np.ndarray],
        vocab: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Least-squares projection from corpus SVD space → fastText space.
        This is necessary because SVD and fastText live in incommensurable
        vector spaces despite having similar dimensionality.
        """
        # Only align terms that exist in both spaces
        aligned_vocab = [w for w in vocab if w in corpus_emb]
        if not aligned_vocab:
            logger.warning(
                "Procrustes: no overlapping vocab. Returning corpus_emb as-is."
            )
            return corpus_emb

        corpus_matrix = np.stack([corpus_emb[w] for w in aligned_vocab])  # (n, d_svd)
        ft_matrix = np.stack(
            [self.we_analyzer.vector(w) for w in aligned_vocab]
        )  # (n, d_ft)

        # Solve corpus_matrix @ W ≈ ft_matrix
        W, _, _, _ = np.linalg.lstsq(corpus_matrix, ft_matrix, rcond=None)

        aligned = {}
        for w, vec in corpus_emb.items():
            aligned[w] = vec @ W  # project into fastText dimensionality

        return aligned

    # ── Stage 8c ─────────────────────────────────────────────────────────────
    def _build_constraint_graph(
        self,
        df_terms: pd.DataFrame,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Build a weighted constraint graph from CHD class membership.
        Nodes: vocabulary terms
        Edges: (term_a, term_b, weight) where both are significantly associated
               with the SAME class, and weight = min(chi²_a, chi²_b).

        Only positive-phi terms are included (class-characteristic, not absent).
        """
        graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        sig_df = df_terms[
            (df_terms["significativo"])
            & (df_terms["phi"] > 0)
            & (df_terms["chi2_yates"] > self.cfg.chi2_threshold)
        ]

        # Group by class
        for class_id, grp in sig_df.groupby("cluster"):
            terms_chi2 = dict(zip(grp["termino"], grp["chi2_yates"]))
            terms = list(terms_chi2.keys())
            # All pairs within the same class are neighbors
            for i, t_a in enumerate(terms):
                for t_b in terms[i + 1 :]:
                    w = min(terms_chi2[t_a], terms_chi2[t_b])
                    graph[t_a].append((t_b, w))
                    graph[t_b].append((t_a, w))

        logger.info(
            "Constraint graph: %d nodes, %d edge pairs.",
            len(graph),
            sum(len(v) for v in graph.values()) // 2,
        )
        return dict(graph)

    # ── Stage 8d ─────────────────────────────────────────────────────────────
    def _retrofit(
        self,
        vocab: List[str],
        corpus_aligned: Dict[str, np.ndarray],
        constraint_graph: Dict[str, List[Tuple[str, float]]],
    ) -> Dict[str, np.ndarray]:
        """
        Three-way retrofitting update (closed-form per word per iteration):

            q(w) = α·ft(w) + β·Σ_{v∈N(w)} w_vq(v) + γ·corpus(w)
                   ─────────────────────────────────────────────────
                         α + β·Σw_v + γ

        where w_v is the edge weight (min chi² of both terms).

        Notes:
        ──────
        • fastText subword architecture handles OOV terms (morphological fallback).
        • Neighbors are updated in-place per iteration (Gauss-Seidel style),
          which converges faster than the pure Jacobi (batch) update.
        • For words with NO neighbors in the constraint graph (e.g. generic
          function words that survived the vocabulary filter), the result is
          a simple α/γ interpolation between fastText and corpus_aligned.
        """
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        gamma = self.cfg.gamma

        # Initialize with fastText vectors
        retrofitted: Dict[str, np.ndarray] = {
            w: self.we_analyzer.vector(w).copy() for w in vocab
        }
        ft_vectors: Dict[str, np.ndarray] = {
            w: self.we_analyzer.vector(w) for w in vocab
        }

        for iteration in range(self.cfg.n_iter):
            for w in vocab:
                neighbors = constraint_graph.get(w, [])
                neighbors_in_vocab = [
                    (v, wt) for v, wt in neighbors if v in retrofitted
                ]

                numerator = alpha * ft_vectors[w] + gamma * corpus_aligned.get(
                    w, ft_vectors[w]
                )
                denominator = alpha + gamma

                if neighbors_in_vocab:
                    weighted_neighbor_sum = sum(
                        wt * retrofitted[v] for v, wt in neighbors_in_vocab
                    )
                    total_weight = sum(wt for _, wt in neighbors_in_vocab)
                    numerator += beta * weighted_neighbor_sum
                    denominator += beta * total_weight

                retrofitted[w] = numerator / denominator

            logger.debug(
                "Retrofitting iteration %d/%d complete.", iteration + 1, self.cfg.n_iter
            )

        return retrofitted

    # ── Public entry point ────────────────────────────────────────────────────
    def run(
        self,
        ppmi_matrix: csr_matrix,
        vocab: List[str],
        df_terms: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        Full specialization pipeline (8a → 8b → 8c → 8d).

        Parameters
        ──────────
        ppmi_matrix : PPMI-weighted UC × terms matrix from SubtlexPPMIBuilder
        vocab       : vocabulary aligned with ppmi_matrix columns
        df_terms    : output of TermAnalyzer (chi², phi, significativo per term×class)

        Returns
        ───────
        retrofitted_vectors : Dict[str, ndarray]
            One vector per vocabulary term, in fastText's dimensional space.
        """
        logger.info("Stage 8a: SVD factorization...")
        corpus_emb = self._svd_corpus_embeddings(ppmi_matrix, vocab)

        logger.info("Stage 8b: Procrustes alignment to fastText space...")
        corpus_aligned = self._procrustes_align(corpus_emb, vocab)

        logger.info("Stage 8c: Building Reinert constraint graph...")
        constraint_graph = self._build_constraint_graph(df_terms)

        logger.info("Stage 8d: Retrofitting (%d iterations)...", self.cfg.n_iter)
        self.retrofitted_vectors = self._retrofit(
            vocab, corpus_aligned, constraint_graph
        )

        # Store corpus embeddings for downstream use
        self.corpus_embeddings = corpus_emb
        return self.retrofitted_vectors

    def compute_class_centroids(
        self,
        df_terms: pd.DataFrame,
        retrofitted_vectors: Dict[str, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """
        Build χ²-weighted class centroids in retrofitted space.
        Only significant, positive-phi terms contribute.
        Used by all five analytic products (P1–P5).
        """
        centroids: Dict[int, np.ndarray] = {}
        sig_df = df_terms[(df_terms["significativo"]) & (df_terms["phi"] > 0)]

        for class_id, grp in sig_df.groupby("cluster"):
            terms_chi2 = {
                row["termino"]: row["chi2_yates"]
                for _, row in grp.iterrows()
                if row["termino"] in retrofitted_vectors
            }
            if not terms_chi2:
                continue
            total_chi2 = sum(terms_chi2.values())
            centroid = sum(
                retrofitted_vectors[t] * (chi2_val / total_chi2)
                for t, chi2_val in terms_chi2.items()
            )
            centroids[int(class_id)] = centroid

        self.class_centroids = centroids
        return centroids


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ANALYTIC PRODUCTS GENERATOR  (P1–P5)
# ══════════════════════════════════════════════════════════════════════════════


class AnalyticProductsGenerator:
    """
    Generates the five new analytic products that are only possible because
    class centroids now exist in a SUBTLEX-anchored, retrofitted embedding space.

    These products are IMPOSSIBLE with standard CHD alone:
    ────────────────────────────────────────────────────────
    P1  Inter-class semantic distance matrix
        (CHD dendrogram distance ≠ semantic distance)
    P2  Soft class membership + per-UCE entropy
        (CHD is a hard partition by construction)
    P3  Latent vocabulary profiles
        (CHD is closed over corpus vocabulary)
    P4  Narrative trajectories within each interview
        (CHD ignores sequential order of UCEs entirely)
    P5  Cross-corpus class alignment metadata
        (CHD classes are corpus-relative; centroids here are not)
    """

    def __init__(
        self,
        we_analyzer: WordEmbeddingsAnalyzer,
        subtlex_analyzer: SubtlexAnalyzer,
        temperature: float = 0.1,
    ):
        self.we = we_analyzer
        self.subtlex = subtlex_analyzer
        self.temperature = temperature

    # ── P1: Inter-class semantic distance matrix ───────────────────────────
    def p1_inter_class_distances(
        self,
        class_centroids: Dict[int, np.ndarray],
    ) -> pd.DataFrame:
        """
        Pairwise cosine distances between class centroids in retrofitted space.

        Interpretation note:
        ─────────────────────
        Two classes that split EARLY in the CHD dendrogram (topologically distant)
        may be semantically CLOSE here — meaning they represent two rhetorical
        registers for the same underlying theme.  This is a genuinely new finding
        that the dendrogram alone cannot reveal.
        """
        classes = sorted(class_centroids.keys())
        n = len(classes)
        dist_matrix = np.zeros((n, n))

        for i, ci in enumerate(classes):
            vi = class_centroids[ci]
            for j, cj in enumerate(classes):
                vj = class_centroids[cj]
                norm_i = np.linalg.norm(vi)
                norm_j = np.linalg.norm(vj)
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cos_sim = np.dot(vi, vj) / (norm_i * norm_j)
                    dist_matrix[i, j] = 1.0 - float(cos_sim)
                else:
                    dist_matrix[i, j] = 1.0

        return pd.DataFrame(dist_matrix, index=classes, columns=classes)

    # ── P2: Soft membership + UCE entropy ─────────────────────────────────
    def p2_soft_membership(
        self,
        uces: List[UCE],
        class_centroids: Dict[int, np.ndarray],
        retrofitted_vectors: Dict[str, np.ndarray],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns
        ───────
        membership_df : one row per UCE, one column per class → P(class|UCE)
        entropy_df    : one row per UCE → {uce_id, entropy, dominant_class, is_liminal}

        Liminal UCEs (high entropy across classes) are analytically important:
        they are moments where a speaker's discourse crosses thematic boundaries.
        A UCE is flagged as liminal if its entropy exceeds the median + 1 MAD.
        """
        classes = sorted(class_centroids.keys())

        def uce_centroid(uce: UCE) -> np.ndarray:
            """SUBTLEX-CD inverse-weighted mean of retrofitted lemma vectors."""
            vecs, weights = [], []
            for lemma in uce.lemmas:
                if lemma in retrofitted_vectors:
                    vec = retrofitted_vectors[lemma]
                else:
                    vec = self.we.vector(lemma)  # fastText subword fallback
                cd = self.subtlex.cd(lemma) or 0.5
                weight = 1.0 / (cd + 1e-6)
                vecs.append(vec)
                weights.append(weight)
            if not vecs:
                return np.zeros(list(class_centroids.values())[0].shape)
            w_arr = np.array(weights)
            w_arr /= w_arr.sum()
            return np.average(vecs, weights=w_arr, axis=0)

        records_membership, records_entropy = [], []

        for uce in uces:
            vec = uce_centroid(uce)
            sims = np.array(
                [
                    float(
                        np.dot(vec, class_centroids[c])
                        / (
                            np.linalg.norm(vec) * np.linalg.norm(class_centroids[c])
                            + 1e-8
                        )
                    )
                    for c in classes
                ]
            )
            probs = softmax(sims / self.temperature)
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            dominant = classes[int(np.argmax(probs))]

            records_membership.append(
                {
                    "uce_id": uce.id,
                    **{f"p_class_{c}": float(probs[i]) for i, c in enumerate(classes)},
                }
            )
            records_entropy.append(
                {
                    "uce_id": uce.id,
                    "entropy": entropy,
                    "dominant_class": dominant,
                    "hard_class": uce.cluster_id,
                }
            )

        membership_df = pd.DataFrame(records_membership).set_index("uce_id")
        entropy_df = pd.DataFrame(records_entropy).set_index("uce_id")

        # Flag liminal UCEs: entropy > median + 1 MAD
        med = entropy_df["entropy"].median()
        mad = (entropy_df["entropy"] - med).abs().median()
        entropy_df["is_liminal"] = entropy_df["entropy"] > (med + mad)

        return membership_df, entropy_df

    # ── P3: Latent vocabulary profiles ────────────────────────────────────
    def p3_latent_vocabulary(
        self,
        class_centroids: Dict[int, np.ndarray],
        corpus_vocab: set,
        subtlex_df: pd.DataFrame,
        top_n: int = 20,
    ) -> Dict[int, List[Dict]]:
        """
        For each class, retrieves SUBTLEX words NOT present in the corpus
        but semantically proximate to the class centroid in retrofitted space.

        These are the class's 'latent vocabulary': concepts the participants
        COULD have used but didn't.  Useful for:
          • Interview guide refinement in future studies
          • Identifying conceptual territory the class occupies in general Spanish
          • Comparing what participants avoided against what the theme predicts

        Scoring:
        ─────────
        score(w) = cosine(ft(w), centroid(c)) × (1 − |log(CD(w) + ε)|)
        The second factor rewards mid-frequency words (neither too rare — likely
        proper nouns or errors — nor too common — likely function words).
        """
        subtlex_words_lower = set(subtlex_df["Word"].str.lower().tolist())
        candidate_words = subtlex_words_lower - corpus_vocab
        subtlex_cd_map = dict(
            zip(
                subtlex_df["Word"].str.lower(),
                subtlex_df["SUBTLCD"],
            )
        )

        results: Dict[int, List[Dict]] = {}

        for class_id, centroid in class_centroids.items():
            candidates = []
            for word in candidate_words:
                vec = self.we.vector(word)
                norm_c = np.linalg.norm(centroid)
                norm_v = np.linalg.norm(vec)
                if norm_c < 1e-8 or norm_v < 1e-8:
                    continue
                cos_sim = float(np.dot(centroid, vec) / (norm_c * norm_v))
                cd = subtlex_cd_map.get(word, 0.5)
                # Penalize extremes: very rare OR very common words
                freq_score = 1.0 - abs(np.log(cd + 1e-6))
                score = cos_sim * max(0.0, freq_score)
                candidates.append(
                    {
                        "word": word,
                        "cosine_similarity": round(cos_sim, 4),
                        "subtlex_cd": round(float(cd), 4),
                        "score": round(score, 4),
                    }
                )

            candidates.sort(key=lambda x: x["score"], reverse=True)
            results[class_id] = candidates[:top_n]

        return results

    # ── P4: Narrative trajectories ─────────────────────────────────────────
    def p4_narrative_trajectories(
        self,
        uces_by_doc: List[List[UCE]],
        class_centroids: Dict[int, np.ndarray],
        retrofitted_vectors: Dict[str, np.ndarray],
        membership_df: pd.DataFrame,
        entropy_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        For each document (interview), produces a sequential record of how the
        speaker's discourse moves through the class space.

        This is the first DIACHRONIC view available in this pipeline.
        Standard CHD is fundamentally synchronic: it treats the corpus as a bag
        of UCs and discards all sequential information.

        Trajectory features per UCE position:
        ───────────────────────────────────────
        • dominant_class     : argmax of soft membership probabilities
        • entropy            : from P2 (thematic ambiguity at that moment)
        • class_shift        : 1 if dominant class changed from previous UCE
        • sim_to_each_class  : full soft membership vector
        """
        trajectories: Dict[str, pd.DataFrame] = {}
        classes = sorted(class_centroids.keys())

        for doc_uces in uces_by_doc:
            if not doc_uces:
                continue
            doc_id = str(doc_uces[0].doc_id)
            records = []
            prev_dominant = None

            for position, uce in enumerate(doc_uces):
                uid = uce.id
                if uid in membership_df.index:
                    probs = membership_df.loc[uid, :].values
                    entropy = (
                        float(entropy_df.loc[uid, "entropy"])
                        if uid in entropy_df.index
                        else 0.0
                    )
                    dominant = classes[int(np.argmax(probs))]
                else:
                    # UCE not in stable set: use its hard cluster_id if available
                    probs = np.zeros(len(classes))
                    dominant = uce.cluster_id
                    entropy = 0.0

                class_shift = (
                    int(dominant != prev_dominant) if prev_dominant is not None else 0
                )
                prev_dominant = dominant

                row = {
                    "position": position,
                    "uce_id": uid,
                    "dominant_class": dominant,
                    "entropy": entropy,
                    "class_shift": class_shift,
                    **{f"p_class_{c}": float(probs[i]) for i, c in enumerate(classes)},
                }
                records.append(row)

            trajectories[doc_id] = pd.DataFrame(records)

        return trajectories

    # ── P5: Cross-corpus alignment metadata ───────────────────────────────
    def p5_centroid_export(
        self,
        class_centroids: Dict[int, np.ndarray],
        df_terms: pd.DataFrame,
        top_terms_per_class: int = 10,
    ) -> Dict[str, Any]:
        """
        Exports class centroids in a format that can be compared against
        centroids from OTHER studies that used the same retrofitting space.

        Because both live in a SUBTLEX-CD-anchored, fastText-grounded space,
        cross-study comparison is semantically meaningful — unlike standard CHD
        where classes are entirely corpus-relative.

        The export includes the centroid vector + top characteristic terms
        so that human analysts can verify alignment plausibility.
        """
        sig_df = df_terms[
            (df_terms["significativo"]) & (df_terms["phi"] > 0)
        ].sort_values("chi2_yates", ascending=False)

        export = {
            "embedding_space": "fastText_cc.es.300_retrofitted_SUBTLEX",
            "vector_dim": next(iter(class_centroids.values())).shape[0],
            "classes": {},
        }

        for class_id, centroid in class_centroids.items():
            top_terms = (
                sig_df[sig_df["cluster"] == class_id]["termino"]
                .head(top_terms_per_class)
                .tolist()
            )
            export["classes"][str(class_id)] = {
                "centroid": centroid.tolist(),
                "top_characteristic_terms": top_terms,
                "centroid_norm": float(np.linalg.norm(centroid)),
            }

        return export


class SubtlexPPMIBuilder:
    """
    Transforms the binary UC × terms CSR matrix produced by MatrizBuilder
    into a PPMI-weighted sparse matrix using SUBTLEX-ESP contextual diversity
    as the background distribution.

    Why contextual diversity (CD) over raw frequency (WF)?
    ───────────────────────────────────────────────────────
    SUBTLEX_CD(w) = proportion of subtitle files containing w.
    A word appearing 1000× in one film scores lower than a word appearing
    100× in 50 films.  CD is more representative of genuine lexical
    availability and is less sensitive to corpus-specific outliers.
    This aligns with Reinert's interest in *distributed* co-presence rather
    than raw frequency dominance.

    Parameters
    ──────────
    subtlex_analyzer : SubtlexAnalyzer
        Must expose .cd(word: str) -> float in [0, 1].
    k : float
        Negative PPMI shift.  k=1 → standard PPMI.  k=5 → shifted PPMI,
        recommended for small corpora (< 50k tokens) because it downweights
        low-frequency co-occurrences that are likely spurious.
    oov_strategy : str
        How to handle words absent from SUBTLEX.
        "corpus_marginal"  → fall back to P(w) estimated from the corpus.
        "midpoint"         → assign SUBTLEX_CD = 0.5 (neutral).
    """

    def __init__(
        self,
        subtlex_analyzer: SubtlexAnalyzer,
        k: float = 1.0,
        oov_strategy: str = "corpus_marginal",
    ):
        self.subtlex = subtlex_analyzer
        self.k = k
        self.oov_strategy = oov_strategy

    def build(
        self,
        binary_matrix: csr_matrix,
        vocab: List[str],
    ) -> csr_matrix:
        """
        Parameters
        ──────────
        binary_matrix : csr_matrix  shape (n_ucs, n_terms)
            Binary presence/absence from MatrizBuilder.construir_matriz_dispersa()
        vocab : List[str]
            Vocabulary aligned with matrix columns (same order as MatrizBuilder).

        Returns
        ───────
        ppmi_matrix : csr_matrix  shape (n_ucs, n_terms)
            PPMI-weighted, non-negative, sparse.
        """
        mat = binary_matrix.astype(np.float64)
        n_ucs, n_terms = mat.shape
        total = mat.sum()
        if total == 0:
            logger.warning("SubtlexPPMIBuilder: empty matrix, returning zeros.")
            return mat

        # ── Row (UC) marginal probabilities ───────────────────────────────
        row_sums = np.asarray(mat.sum(axis=1)).flatten()  # (n_ucs,)
        uc_prob = row_sums / total  # P(UC_i)

        # ── Background (term) probability from SUBTLEX-CD ─────────────────
        corpus_marginals = np.asarray(mat.sum(axis=0)).flatten()  # (n_terms,)
        corpus_marginal_prob = corpus_marginals / corpus_marginals.sum()

        bg_prob = np.zeros(n_terms, dtype=np.float64)
        for j, term in enumerate(vocab):
            cd = self.subtlex.cd(term)  # float in [0,1] or None
            if cd is not None and cd > 0:
                bg_prob[j] = cd
            else:
                # OOV handling
                if self.oov_strategy == "corpus_marginal":
                    bg_prob[j] = corpus_marginal_prob[j]
                else:  # midpoint
                    bg_prob[j] = 0.5

        # Renormalize background to sum to 1
        bg_sum = bg_prob.sum()
        if bg_sum > 0:
            bg_prob /= bg_sum

        # ── PPMI computation over non-zero entries (sparse-friendly) ───────
        cx = mat.tocoo()
        rows_idx = cx.row
        cols_idx = cx.col
        data = cx.data

        joint_vals = data / total
        expected = uc_prob[rows_idx] * bg_prob[cols_idx]

        # Guard against zero denominator
        safe_expected = np.where(expected > 1e-12, expected, 1e-12)
        ppmi_vals = np.log(joint_vals / safe_expected) - np.log(self.k)
        ppmi_vals = np.maximum(0.0, ppmi_vals)  # positive PPMI only

        ppmi_matrix = csr_matrix(
            (ppmi_vals, (rows_idx, cols_idx)),
            shape=(n_ucs, n_terms),
        )
        logger.info(
            "SubtlexPPMIBuilder: %.1f%% non-zero entries retained after PPMI thresholding.",
            100.0 * ppmi_matrix.nnz / (n_ucs * n_terms),
        )
        return ppmi_matrix

    def subtlex_cd_weights(self, vocab: List[str]) -> np.ndarray:
        """
        Returns per-token SUBTLEX-CD inverse weights for UCE similarity
        computation in Phase 3 (UC construction).
        Rare words (low CD) → higher weight → contribute more to boundary detection.
        """
        weights = np.zeros(len(vocab), dtype=np.float64)
        for j, term in enumerate(vocab):
            cd = self.subtlex.cd(term)
            if cd is not None and cd > 0:
                weights[j] = 1.0 / (cd + 1e-6)
            else:
                weights[j] = 1.0  # neutral fallback
        return weights


class MatrizBuilder:
    def __init__(self, config: Config):
        self.config = config

    def _iter_terms(self, obj):
        use_stems = self.config.stem_backend != "none"
        for s in obj.stems if use_stems else obj.lemmas:
            yield s
        if self.config.use_bigrams:
            for b in obj.bigram_stems if use_stems else obj.bigrams:
                yield "_".join(b)
        if self.config.use_trigrams:
            for t in obj.trigram_stems if use_stems else obj.trigrams:
                yield "_".join(t)

    def construir_vocabulario(self, objects: List) -> List[str]:
        counter = Counter()
        for obj in objects:
            # Document frequency logic
            counter.update(set(self._iter_terms(obj)))
        return sorted(t for t, cnt in counter.items() if cnt >= self.config.tsj)

    def construir_matriz(self, objects: List, vocabulario: List[str]) -> np.ndarray:
        idx = {t: i for i, t in enumerate(vocabulario)}
        mat = np.zeros((len(objects), len(vocabulario)), dtype=int)
        for i, obj in enumerate(objects):
            for term in set(self._iter_terms(obj)):  # Binary presence/absence
                if term in idx:
                    mat[i, idx[term]] = 1
        return mat

    def construir_matriz_dispersa(self, objects: List, vocabulario: List[str]):
        from scipy.sparse import csr_matrix, dok_matrix

        idx = {t: i for i, t in enumerate(vocabulario)}
        mat = dok_matrix((len(objects), len(vocabulario)), dtype=np.int32)
        for i, obj in enumerate(objects):
            for term in set(self._iter_terms(obj)):  # Binary presence/absence
                if term in idx:
                    mat[i, idx[term]] = 1
        return csr_matrix(mat)

    def agregar_por_clase(
        self, mat_uces: np.ndarray, labels_uces: np.ndarray, vocabulario: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        class_ids = np.unique(labels_uces)
        mat = np.zeros((len(class_ids), len(vocabulario)), dtype=int)
        for i, cid in enumerate(class_ids):
            mat[i] = mat_uces[labels_uces == cid].sum(axis=0)
        return mat, class_ids


# ══════════════════════════════════════════════════════════════════════
# CLASIFICADOR DESCENDENTE JERÁRQUICO (CDH)
# ══════════════════════════════════════════════════════════════════════


def _compute_uce_centroids_from_retro(
    uces: List["UCE"],
    retrofitted_vectors: Dict[str, np.ndarray],
    we_analyzer,
    subtlex_analyzer,
) -> np.ndarray:
    """
    Vectorize every UCE for HDBSCAN (Method C).
    Mirrors _compute_uc_centroids but operates on UCE.lemmas directly.

    Returns
    -------
    np.ndarray  shape (n_uces, d_ft), float64, L2-normalised rows.
    All-zero row = UCE with no embedding coverage (OOV / empty lemmas).
    """
    if not retrofitted_vectors:
        return np.zeros((len(uces), 1), dtype=np.float64)

    d = next(iter(retrofitted_vectors.values())).shape[0]
    result = np.zeros((len(uces), d), dtype=np.float64)

    for i, uce in enumerate(uces):
        tokens = uce.lemmas or []
        if not tokens:
            continue
        vecs: List[np.ndarray] = []
        weights: List[float] = []
        for lemma in tokens:
            vec = retrofitted_vectors.get(lemma)
            if vec is None:
                vec = we_analyzer.vector(lemma)  # fastText subword fallback
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                continue
            cd = subtlex_analyzer.cd(lemma) or 0.5
            vecs.append(vec)
            weights.append(1.0 / (cd + 1e-6))  # rare - higher weight

        if not vecs:
            continue
        w = np.array(weights, dtype=np.float64)
        w /= w.sum()
        centroid = np.average(vecs, weights=w, axis=0)
        n = np.linalg.norm(centroid)
        if n > 1e-8:
            centroid /= n
        result[i] = centroid

    return result


def _compute_uc_centroids(
    ucs: List[UC],
    retrofitted_vectors: Dict[str, np.ndarray],
    we_analyzer: WordEmbeddingsAnalyzer,
    subtlex_analyzer: SubtlexAnalyzer,
) -> np.ndarray:
    """
    Precomputes one L2-normalized dense vector per UC in retrofitted space.

    Vector = SUBTLEX-CD inverse-weighted mean of retrofitted lemma vectors.
    Mirrors UCEVectorizer.vectorize() but operates on UC.lemmas directly.

    Returns
    ───────
    np.ndarray  shape (n_ucs, d_ft), float64
        Row i is aligned with ucs[i] and the CHD labels array.
        All-zero row means the UC had no coverage in retrofitted_vectors
        and will simply not contribute to warm-start or hybrid swap.
    """
    if not retrofitted_vectors:
        return np.zeros((len(ucs), 1), dtype=np.float64)  # sentinel

    d = next(iter(retrofitted_vectors.values())).shape[0]
    result = np.zeros((len(ucs), d), dtype=np.float64)

    for i, uc in enumerate(ucs):
        tokens = uc.lemmas or []
        if not tokens:
            continue

        vecs: List[np.ndarray] = []
        weights: List[float] = []

        for lemma in tokens:
            vec = retrofitted_vectors.get(lemma)
            if vec is None:
                vec = we_analyzer.vector(lemma)  # fastText subword fallback
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                continue
            cd = subtlex_analyzer.cd(lemma) or 0.5
            vecs.append(vec)
            weights.append(1.0 / (cd + 1e-6))  # rare → higher weight

        if not vecs:
            continue

        w = np.array(weights, dtype=np.float64)
        w /= w.sum()
        centroid = np.average(vecs, weights=w, axis=0)
        n = np.linalg.norm(centroid)
        if n > 1e-8:
            centroid /= n
        result[i] = centroid

    return result


class ClasificadorDescendente:
    def __init__(self, config: Config):
        self.config = config
        self._leaf_counter = 0
        self._rng = np.random.default_rng(config.random_state)
        self._min_cluster_size: int = 5  # resolved in clasificar()

    def _next_leaf(self) -> int:
        label = self._leaf_counter
        self._leaf_counter += 1
        return label

    def _fill_labels(self, node: CDHNode, labels: np.ndarray):
        if node.is_leaf:
            for idx in node.indices:
                labels[idx] = node.label
        else:
            for child in node.children:
                self._fill_labels(child, labels)

    def clasificar(
        self,
        mat_sparse,
        uc_vectors: Optional[np.ndarray] = None,  # ← NEW: (n_ucs, d), or None
    ) -> Tuple[np.ndarray, CDHNode]:
        self._leaf_counter = 0
        n_total = mat_sparse.shape[0]
        self._min_cluster_size = max(5, int(n_total * self.config.min_cluster_size_cdh))
        print(
            f"   [CDH] n={n_total} · min_cluster_size={self._min_cluster_size} "
            f"({self.config.min_cluster_size_cdh:.0%} of corpus)"
        )
        # Validate uc_vectors shape; discard silently if mismatched
        if uc_vectors is not None and uc_vectors.shape[0] != n_total:
            logger.warning(
                "CDH: uc_vectors shape %s mismatches n_total=%d — ignoring.",
                uc_vectors.shape,
                n_total,
            )
            uc_vectors = None

        indices = np.arange(n_total)
        arbol = self._partition(indices, mat_sparse, depth=0, uc_vectors=uc_vectors)
        labels = np.full(n_total, -1, dtype=int)
        self._fill_labels(arbol, labels)
        return labels, arbol

    def _primer_factor(
        self,
        sub_mat,
        uc_vectors: Optional[np.ndarray] = None,  # ← NEW: (n, d) subset
    ) -> np.ndarray:
        """
        Reciprocal averaging with optional warm-start from retrofitted centroids.

        When uc_vectors is supplied (Pass 2 only), the initial axis x is seeded
        from the first principal component of the UC centroid matrix instead of
        a random vector.  The power iteration converges to the same fixed point
        regardless of initialization — the warm-start only reduces iteration count
        and variance across random seeds on ambiguous splits.

        Falls back to random initialization if:
          • uc_vectors is None (Pass 1 / optimizer trials)
          • n < 5 (too few UCs for a meaningful PCA seed)
          • all UC vectors are zero (no retrofitting coverage)
        """
        n, m = sub_mat.shape
        if n < 2:
            return np.zeros(n)

        mat = sub_mat.astype(float)
        pc = self.config.pseudocount
        row_sums = np.asarray(mat.sum(axis=1)).flatten().reshape(-1) + pc * m
        col_sums = np.asarray(mat.sum(axis=0)).flatten().reshape(-1) + pc * n
        total = row_sums.sum()
        if total <= 0:
            return np.zeros(n)

        row_sums[row_sums == 0] = 1
        col_sums[col_sums == 0] = 1

        rng = np.random.default_rng(self.config.random_state)

        # ── Warm-start from retrofitted centroid PCA ──────────────────────────
        x = None
        if uc_vectors is not None and n >= 5:
            try:
                centered = uc_vectors - uc_vectors.mean(axis=0)  # (n, d)
                norms = np.linalg.norm(centered, axis=1)
                if norms.max() > 1e-8:  # at least some coverage
                    # Power iteration for first PC (20 steps is enough)
                    v = rng.standard_normal(centered.shape[1]).astype(np.float64)
                    for _ in range(20):
                        v = centered.T @ (centered @ v)
                        vn = np.linalg.norm(v)
                        if vn < 1e-12:
                            break
                        v /= vn
                    x_init = centered @ v  # projection scores (n,)
                    x_init -= np.average(x_init, weights=row_sums)
                    init_norm = np.linalg.norm(x_init)
                    if init_norm > 1e-12:
                        x = x_init / init_norm
            except Exception:
                pass  # any linear algebra edge case → fall through to random

        if x is None:
            x = rng.standard_normal(n)
            x -= np.average(x, weights=row_sums)
            x /= max(np.linalg.norm(x), 1e-12)

        # ── Reciprocal averaging (unchanged) ──────────────────────────────────
        for _ in range(100):
            y = mat.T.dot(x)
            z = y / col_sums
            w = mat.dot(z)
            x_new = w / row_sums
            x_new -= np.average(x_new, weights=row_sums)
            norm = np.linalg.norm(x_new)
            if norm < 1e-12:
                break
            x_new /= norm
            flip = np.sign(np.dot(x_new, x))
            if np.linalg.norm(x_new - flip * x) < 1e-8:
                break
            x = x_new

        return x

    def _corte_optimo(self, coord: np.ndarray) -> Tuple[float, np.ndarray]:
        n = len(coord)
        if n < 2:
            return 0.0, np.zeros(n, dtype=int)

        order = np.argsort(coord)
        sorted_c = coord[order]
        total = sorted_c.sum()
        mean = total / n

        # Between-class sum of squares (unnormalized)
        cum = np.cumsum(sorted_c)
        left_sum = cum[:-1]
        right_sum = total - left_sum
        left_n = np.arange(1, n)
        right_n = n - left_n

        ss_between = left_sum**2 / left_n + right_sum**2 / right_n - total**2 / n
        k_best = int(np.argmax(ss_between)) + 1

        # Normalize: var_inter = SS_between / SS_total (= R²)
        ss_total = float(np.sum((sorted_c - mean) ** 2))
        var_inter = float(ss_between[k_best - 1]) / (ss_total + 1e-12)

        labels = np.zeros(n, dtype=int)
        labels[order[k_best:]] = 1
        return var_inter, labels

    def _intercambio(
        self,
        sub_mat,
        labels: np.ndarray,
        max_iter: int,
        uc_vectors: Optional[np.ndarray] = None,  # ← NEW: (n, d) or None
    ) -> np.ndarray:
        """
        Chi-square swap refinement with optional hybrid centroid tiebreaker.

        Hybrid criterion
        ────────────────
        When uc_vectors is supplied, UCs whose chi-square improvement delta
        falls within the ambiguity band [-tol, +tol] are decided by centroid
        proximity instead.  A UC is moved if it is closer (cosine) to the
        centroid of the destination class.

        The chi-square signal is always dominant:
          • delta > +tol  → move unconditionally (clear improvement)
          • delta < -tol  → do not move (clear regression)
          • |delta| ≤ tol → tiebreaker via centroid cosine (if available)

        Class centroids in retrofitted space are updated incrementally after
        each accepted move, keeping the O(n × nnz) complexity unchanged.
        tol is set to 0.5, corresponding to roughly one unit of chi-square
        noise from a sparse UC.  It applies only when uc_vectors is supplied.
        """
        from scipy.sparse import issparse

        binary = (sub_mat > 0).astype(np.float32)
        n = binary.shape[0]
        labels = labels.copy()

        cc = [
            np.asarray(binary[labels == 0].sum(axis=0)).flatten().reshape(-1),
            np.asarray(binary[labels == 1].sum(axis=0)).flatten().reshape(-1),
        ]
        ct = [int((labels == 0).sum()), int((labels == 1).sum())]
        global_freq = cc[0] + cc[1]

        # ── Initialise centroid tracking for hybrid tiebreaker ────────────────
        _use_hybrid = uc_vectors is not None and uc_vectors.shape[0] == n
        _tol = 0.5  # ambiguity band half-width (chi-square units)
        c_vecs = None  # will hold (2, d) class centroid matrix
        _d = 0

        if _use_hybrid:
            _d = uc_vectors.shape[1]
            c_vecs = np.zeros((2, _d), dtype=np.float64)
            for cls in (0, 1):
                mask = labels == cls
                if mask.any():
                    cv = uc_vectors[mask].mean(axis=0)
                    cn = np.linalg.norm(cv)
                    c_vecs[cls] = cv / cn if cn > 1e-8 else cv

        for _ in range(max_iter):
            moved = False
            for i in range(n):
                old = int(labels[i])
                new = 1 - old
                if ct[old] <= 1:
                    continue

                row = (
                    np.asarray(binary[i].todense()).flatten()
                    if issparse(binary)
                    else binary[i]
                )
                nonzero_idx = np.where(row > 0)[0]
                if len(nonzero_idx) == 0:
                    continue

                # ── Chi-square delta (unchanged logic) ───────────────────────
                delta = 0.0
                total_ucs = n
                new_ct_old = ct[old] - 1
                new_ct_new = ct[new] + 1

                for t in nonzero_idx:
                    gf = global_freq[t]
                    if gf == 0:
                        continue
                    E_old = ct[old] * gf / total_ucs
                    E_new = ct[new] * gf / total_ucs
                    E_old2 = new_ct_old * gf / total_ucs
                    E_new2 = new_ct_new * gf / total_ucs
                    chi_now = (cc[old][t] - E_old) ** 2 / max(E_old, 1e-9) + (
                        cc[new][t] - E_new
                    ) ** 2 / max(E_new, 1e-9)
                    chi_after = (cc[old][t] - 1 - E_old2) ** 2 / max(E_old2, 1e-9) + (
                        cc[new][t] + 1 - E_new2
                    ) ** 2 / max(E_new2, 1e-9)
                    delta += chi_after - chi_now

                # ── Decision logic ────────────────────────────────────────────
                do_swap = False

                if delta > _tol:
                    # Clear chi-square improvement — always swap
                    do_swap = True

                elif _use_hybrid and abs(delta) <= _tol and c_vecs is not None:
                    # Ambiguous — use centroid proximity as tiebreaker
                    uv = uc_vectors[i]  # (d,) already normalised
                    cos_old = float(np.dot(uv, c_vecs[old]))
                    cos_new = float(np.dot(uv, c_vecs[new]))
                    do_swap = cos_new > cos_old  # move if new class is closer

                elif delta > 0:
                    # Falls through only when _use_hybrid=False and delta ∈ (0, tol]
                    do_swap = True

                if do_swap:
                    labels[i] = new
                    cc[old] -= row
                    cc[new] += row
                    ct[old] -= 1
                    ct[new] += 1
                    moved = True

                    # ── Incremental centroid update ───────────────────────────
                    if (
                        _use_hybrid
                        and c_vecs is not None
                        and ct[old] > 0
                        and ct[new] > 0
                    ):
                        uv = uc_vectors[i]
                        # old class loses UC i
                        cv_old = (c_vecs[old] * (ct[old] + 1) - uv) / ct[old]
                        n_old = np.linalg.norm(cv_old)
                        c_vecs[old] = cv_old / n_old if n_old > 1e-8 else cv_old
                        # new class gains UC i
                        cv_new = (c_vecs[new] * (ct[new] - 1) + uv) / ct[new]
                        n_new = np.linalg.norm(cv_new)
                        c_vecs[new] = cv_new / n_new if n_new > 1e-8 else cv_new

            if not moved:
                break

        return labels

    # ──────────────────────────────────────────────
    # Prueba de significancia
    # ──────────────────────────────────────────────

    def _es_significativo(self, coord: np.ndarray, var_obs: float) -> bool:
        # var_obs is now R² (normalized), so chi2_approx = n * R²
        n = len(coord)
        chi2_approx = n * var_obs
        if chi2_approx < self.config.chi2_threshold_small:
            return False
        if var_obs < self.config.min_r2_threshold:
            return False
        return True

    # ──────────────────────────────────────────────
    # Recursión principal
    # ──────────────────────────────────────────────

    def _partition(
        self,
        indices,
        mat_sparse,
        depth,
        uc_vectors: Optional[np.ndarray] = None,  # ← NEW: full array, sliced here
    ) -> CDHNode:
        n = len(indices)
        if n < self._min_cluster_size or depth >= self.config.max_depth_cdh:
            node = CDHNode(depth=depth, n_ucs=n, is_leaf=True, indices=indices.tolist())
            node.label = self._next_leaf()
            return node
        sub_mat = mat_sparse[indices]

        # Slice uc_vectors to this partition's rows — keeps indices aligned
        sub_vecs = uc_vectors[indices] if uc_vectors is not None else None

        coord = self._primer_factor(sub_mat, uc_vectors=sub_vecs)  # ← warm-start
        var_obs, labels = self._corte_optimo(coord)

        if not self._es_significativo(coord, var_obs):
            node = CDHNode(depth=depth, n_ucs=n, is_leaf=True, indices=indices.tolist())
            node.label = self._next_leaf()
            return node

        if self.config.swap_iterations > 0:
            labels = self._intercambio(
                sub_mat,
                labels,
                max_iter=self.config.swap_iterations,
                uc_vectors=sub_vecs,  # ← hybrid swap
            )

        if len(np.unique(labels)) < 2:
            node = CDHNode(depth=depth, n_ucs=n, is_leaf=True, indices=indices.tolist())
            node.label = self._next_leaf()
            return node

        idx0 = indices[labels == 0]
        idx1 = indices[labels == 1]

        child0 = self._partition(idx0, mat_sparse, depth + 1, uc_vectors=uc_vectors)
        child1 = self._partition(idx1, mat_sparse, depth + 1, uc_vectors=uc_vectors)

        return CDHNode(
            depth=depth,
            n_ucs=n,
            is_leaf=False,
            indices=indices.tolist(),
            children=[child0, child1],
        )


# ─────────────────────────────────────────────────────────────────────────────
# min_forms floor for coref-built UCE groups
# ─────────────────────────────────────────────────────────────────────────────


def _merge_small_uc_groups(
    uc_groups: List[List],  # List[List[UCE]]
    min_forms: int,
) -> List[List]:
    """
    Merges UCE groups that have fewer than min_forms content lemmas with their
    neighbour that has the fewest forms (greedy, iterative).

    Invariant: the returned list covers all original UCEs in their original order.
    A single-group document is returned unchanged even if below the floor.

    Analogous to UCBuilder._merge_small_segments but works on UCE-group lists
    (no embeddings required).
    """

    def n_forms(group: List) -> int:
        return sum(len(getattr(u, "lemmas", [])) for u in group)

    changed = True
    while changed and len(uc_groups) > 1:
        changed = False
        for i, group in enumerate(uc_groups):
            if n_forms(group) >= min_forms:
                continue

            # Collect available neighbours and their form counts
            neighbours = []
            if i > 0:
                neighbours.append((i - 1, n_forms(uc_groups[i - 1])))
            if i < len(uc_groups) - 1:
                neighbours.append((i + 1, n_forms(uc_groups[i + 1])))
            if not neighbours:
                continue  # isolated single group — leave as-is

            # Merge with the smallest neighbour to minimise disruption
            best_nb = min(neighbours, key=lambda x: x[1])[0]
            lo, hi = sorted([i, best_nb])
            merged = uc_groups[lo] + uc_groups[hi]  # preserves UCE order
            uc_groups = uc_groups[:lo] + [merged] + uc_groups[hi + 1 :]
            changed = True
            break  # restart scan after any merge

    return uc_groups


def _remap_uces_to_segments(
    uces: List[UCE],
    merged_texts: List[str],
) -> List[List[UCE]]:
    """
    Maps merged text segments (output of recursive_segmentation + final_clustering)
    back to groups of original UCE objects.

    The segmentation pipeline only merges adjacent items and preserves order.
    This function uses greedy string accumulation with exact matching since both
    the pipeline joining and this function use the same " ".join() separator.
    """
    groups: List[List[UCE]] = []
    uce_ptr = 0
    for merged in merged_texts:
        group: List[UCE] = []
        merged_norm = " ".join(merged.split())
        acc_parts: List[str] = []
        while uce_ptr < len(uces):
            acc_parts.append(uces[uce_ptr].texto)
            group.append(uces[uce_ptr])
            uce_ptr += 1
            acc_norm = " ".join(acc_parts)
            if acc_norm == merged_norm:
                break
        groups.append(group)

    # Safety check — all UCEs must be accounted for
    mapped = sum(len(g) for g in groups)
    if mapped != len(uces):
        logger.warning(
            "_remap_uces_to_segments: mapped %d / %d UCEs — "
            "falling back to no-op grouping",
            mapped,
            len(uces),
        )
        return [[u] for u in uces]

    return groups


# ══════════════════════════════════════════════════════════════════════
# T3 — DOBLE CLASIFICACIÓN (refactorizado para CDH)
# ══════════════════════════════════════════════════════════════════════

# ── Import needed for Hungarian matching ──────────────────────────────
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


class DoubleClassifier:
    """
    Double-classification with configurable UC construction mode.

    Modes
    -----
    wc_only   : Classic word-count UC construction (segmentador.construir_ucs).
                Double classification at two min_forms thresholds → Hungarian alignment.
    sim_only  : Cohesion-based UC construction (UCBuilder).
                Double classification at two similarity thresholds → Hungarian alignment.
    both      : Hierarchical — word-count first establishes the baseline,
                cohesion-based second confirms or rejects.  Only UCEs that are
                stable within *both* methods AND agree across methods survive.

    Parameters
    ----------
    config         : Config with classification_mode among other settings.
    segmentador    : SegmentadorALCESTE instance (needed for word-count path).
    uc_builder     : UCBuilder instance (needed for similarity path).
    """

    def __init__(
        self,
        config: Config,
        segmentador: SegmentadorALCESTE,
        uc_builder: UCBuilder,
        progressive_segmenter=None,  # Optional[ProgressiveSegmenter] — injected externally
    ):
        self.config = config
        self.segmentador = segmentador
        self.threshold_delta = 0.05
        self.uc_builder = uc_builder
        self.ppmi_builder: Optional["SubtlexPPMIBuilder"] = None
        # Method B: ProgressiveSegmenter instance (None = Method B unavailable)
        self.progressive_segmenter = progressive_segmenter

    # ═══════════════════════════════════════════════════════════════════
    # METHOD 1 — word-count UC construction (classic Reinert)
    # ═══════════════════════════════════════════════════════════════════
    def _clasificar_umbral_wordcount(
        self,
        uces_por_doc: List[List[UCE]],
        min_forms: int,
    ) -> Optional[Dict]:
        suffix = f"__wc{min_forms}"

        uces_por_doc_local = copy.deepcopy(uces_por_doc)
        for doc_uces in uces_por_doc_local:
            for uce in doc_uces:
                uce.id = normalizar_id_uce(uce)
                uce.id = f"{uce.id}{suffix}"

        ucs, uce_to_uc = self.segmentador.construir_ucs(
            uces_por_doc_local,
            min_forms,
            self.segmentador.doc_metadata_map,
        )

        # Strip suffix back to bare ids
        for uc in ucs:
            uc.uce_ids = [
                uid[: -len(suffix)] if uid.endswith(suffix) else uid
                for uid in uc.uce_ids
            ]
        uce_to_uc = {
            (uid[: -len(suffix)] if uid.endswith(suffix) else uid): uc_id
            for uid, uc_id in uce_to_uc.items()
        }

        if not ucs:
            return None

        if self.config.use_embeddings:
            ucs = self.segmentador.generar_embeddings_ucs(ucs)

        builder = MatrizBuilder(self.config)
        voc = builder.construir_vocabulario(ucs)
        if not voc:
            return None

        mat_sparse = builder.construir_matriz_dispersa(ucs, voc)
        if self.ppmi_builder is not None:
            mat_sparse = self.ppmi_builder.build(mat_sparse, voc)

        cdh = ClasificadorDescendente(self.config)
        labels, tree = cdh.clasificar(mat_sparse)

        for i, uc in enumerate(ucs):
            uc.cluster_id = int(labels[i]) if labels[i] != -1 else None

        real = np.unique(labels[labels >= 0])
        print(
            f"   [wordcount umbral={min_forms}] UCs={len(ucs)}, "
            f"clusters={len(real)}, ruido={np.sum(labels == -1)}"
        )
        return {
            "method": "wordcount",
            "min_forms": min_forms,
            "ucs": ucs,
            "labels": labels,
            "voc": voc,
            "uce_to_uc": uce_to_uc,
            "tree": tree,
        }

    # ═══════════════════════════════════════════════════════════════════
    # METHOD 2 — cohesion-based UC construction (UCBuilder)
    # ═══════════════════════════════════════════════════════════════════
    def _clasificar_umbral_similarity(
        self,
        uces_por_doc: List[List[UCE]],
        min_forms: int,
        similarity_threshold: float,
        retrofitted_vectors: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[Dict]:
        suffix = f"__sim{min_forms}"

        uces_por_doc_local = copy.deepcopy(uces_por_doc)
        for doc_uces in uces_por_doc_local:
            for uce in doc_uces:
                uce.id = normalizar_id_uce(uce)
                uce.id = f"{uce.id}{suffix}"

        # Temporarily adjust threshold for this pass, then restore
        original_threshold = self.uc_builder.cfg.similarity_threshold
        self.uc_builder.cfg.similarity_threshold = similarity_threshold
        self.uc_builder.vectorizer.clear_cache()

        try:
            ucs, uce_to_uc = self.uc_builder.build(
                uces_por_doc_local,
                min_forms,
                self.segmentador.doc_metadata_map,
                retrofitted_vectors=retrofitted_vectors,
            )
        finally:
            self.uc_builder.cfg.similarity_threshold = original_threshold

        # Strip suffix
        for uc in ucs:
            uc.uce_ids = [
                uid[: -len(suffix)] if uid.endswith(suffix) else uid
                for uid in uc.uce_ids
            ]
        uce_to_uc = {
            (uid[: -len(suffix)] if uid.endswith(suffix) else uid): uc_id
            for uid, uc_id in uce_to_uc.items()
        }
        if not ucs:
            return None

        if self.config.use_embeddings:
            ucs = self.segmentador.generar_embeddings_ucs(ucs)

        builder = MatrizBuilder(self.config)
        voc = builder.construir_vocabulario(ucs)
        if not voc:
            return None

        mat_sparse = builder.construir_matriz_dispersa(ucs, voc)
        if self.ppmi_builder is not None:
            mat_sparse = self.ppmi_builder.build(mat_sparse, voc)

        # Precompute UC centroid vectors for warm-start (only when retrofitted)
        uc_vectors: Optional[np.ndarray] = None
        if retrofitted_vectors is not None:
            try:
                uc_vectors = _compute_uc_centroids(
                    ucs,
                    retrofitted_vectors,
                    self.uc_builder.vectorizer.we,
                    self.uc_builder.vectorizer.subtlex,
                )
            except Exception as e:
                logger.warning("_compute_uc_centroids failed: %s", e)

        cdh = ClasificadorDescendente(self.config)
        labels, tree = cdh.clasificar(mat_sparse, uc_vectors=uc_vectors)

        for i, uc in enumerate(ucs):
            uc.cluster_id = int(labels[i]) if labels[i] != -1 else None

        real = np.unique(labels[labels >= 0])
        print(
            f"   [similarity threshold={similarity_threshold:.2f} min_forms={min_forms}]"
            f" UCs={len(ucs)}, clusters={len(real)}, ruido={np.sum(labels == -1)}"
        )
        return {
            "method": "similarity",
            "similarity_threshold": similarity_threshold,
            "min_forms": min_forms,
            "ucs": ucs,
            "labels": labels,
            "voc": voc,
            "uce_to_uc": uce_to_uc,
            "tree": tree,
            "uc_vectors": uc_vectors,
        }

    # ═══════════════════════════════════════════════════════════════════
    # METHOD B — coreference-driven UC construction (ProgressiveSegmenter)
    # ═══════════════════════════════════════════════════════════════════
    def _clasificar_umbral_coref(
        self,
        uces_por_doc: List[List[UCE]],
        min_forms: int,
        context_units: int,
        retrofitted_vectors: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[Dict]:
        """
        Builds UCs by dissolving UCE boundaries that are bridged by a
        coreference chain whose antecedent is a root-subject NP.

        UC formation algorithm
        ──────────────────────
        For each document:
          1. Pre-compute root NP subjects for every UCE (single Stanza call each,
             cached — avoids re-running Stanza during the merge pass).
          2. Pre-merge UCEs into coarser discourse groups via the same
             recursive_segmentation() + final_clustering() pipeline that
             ProgressiveSegmenter.segment_text() uses.
          3. Build segments_info from the pre-merged groups (sequential char offsets).
          4. Run _extract_global_chains(segments_info, full_doc_text, context_units)
             → deduplicated coref chains over a sliding window of context_units groups.
          5. Greedy forward merge: for group[i]→group[i+1], call van_unidos()
             with pre-computed roots (Stanza-free). Dissolve boundary if a chain's
             antecedent in group[i]'s root subjects has a coreferent mention in group[i+1].
          6. Apply _merge_small_uc_groups() to enforce the min_forms floor.
          7. Build UC objects in the standard format.

        Comparison parameter
        ────────────────────
        context_units : sliding-window width for chain extraction.
          Tight pass (B1): small value → fewer chains → conservative merges.
          Loose pass (B2): larger value → more chains → more merges.

        Quality signal (in returned dict under "coref_quality")
        ──────────────────────────────────────────────────────
        merge_rate      : coref_merges / total_boundaries  ← stability input
        chain_coverage  : boundaries where any chain spanned / total_boundaries
        n_chains_total  : total deduplicated chains across the corpus

        Fallback policy
        ───────────────
        Returns None (never silently degrades to a different method) when:
          • ProgressiveSegmenter not injected
          • Stanza not available
          • Zero UCs produced after min_forms floor
          • Empty vocabulary after UC construction
        Callers (run()) treat None as "Method B unavailable for this pass."
        """
        # ── 0. Guard: hard fail, no silent fallback ───────────────────────────
        ps = self.progressive_segmenter
        if ps is None:
            print(
                "   [COREF] ProgressiveSegmenter not injected — Method B unavailable."
            )
            return None
        if ps.get_stanza() is None:
            print("   [COREF] Stanza unavailable — Method B unavailable.")
            return None

        # ── 1. Deep-copy + suffix ─────────────────────────────────────────────
        suffix = f"__coref{context_units}"
        uces_por_doc_local = copy.deepcopy(uces_por_doc)
        for doc_uces in uces_por_doc_local:
            for uce in doc_uces:
                uce.id = normalizar_id_uce(uce)
                uce.id = f"{uce.id}{suffix}"

        # ── 2. Pre-compute root subjects for every UCE (one Stanza call each) ─
        #       Cached by suffixed UCE id.  Executed BEFORE the merge loop so
        #       van_unidos never needs to call Stanza again.
        total_uces = sum(len(d) for d in uces_por_doc_local)
        print(f"   [COREF] Pre-computing root subjects for {total_uces} UCEs...")
        root_cache: Dict[str, set] = {}
        for doc_uces in uces_por_doc_local:
            for uce in doc_uces:
                ctx = uce.texto[-500:] if len(uce.texto) > 500 else uce.texto
                try:
                    roots = set(ps.find_subjects_for_roots(ctx))
                except Exception as e:
                    logger.debug("root_cache miss for %s: %s", uce.id, e)
                    roots = set()
                root_cache[uce.id] = roots

        # ── 3. Per-document UC construction ──────────────────────────────────
        ucs: List[UC] = []
        uce_to_uc: Dict[str, str] = {}
        global_uc_counter = 0

        # Quality accumulators
        total_boundaries = 0
        coref_merges = 0
        covered_boundaries = 0
        total_chains = 0

        doc_metadata_map = self.segmentador.doc_metadata_map

        for doc_uces in uces_por_doc_local:
            if not doc_uces:
                continue

            # ── 3a. Pre-merge UCEs via recursive + progressive segmentation ──
            #       This mirrors segmentador.py's pipeline: sentences →
            #       recursive_segmentation → final_clustering → coref.
            #       Each resulting group becomes the atomic segment for coref.
            if len(doc_uces) > 1:
                uce_texts = [u.texto for u in doc_uces]
                try:
                    rec_seg = ps.recursive_segmentation(uce_texts)
                    prog_seg = ps.final_clustering(rec_seg)
                    premerged_groups = _remap_uces_to_segments(doc_uces, prog_seg)
                    n_pre = len(premerged_groups)
                    print(
                        f"   [COREF] Pre-merge: {len(doc_uces)} UCEs → "
                        f"{n_pre} grupos (recursive+progressive)"
                    )
                except Exception as e:
                    logger.warning(
                        "Pre-merge failed for doc %s: %s",
                        doc_uces[0].doc_id,
                        e,
                    )
                    premerged_groups = [[u] for u in doc_uces]
            else:
                premerged_groups = [doc_uces]

            # ── 3b. Rebuild segments_info from PRE-MERGED groups ──
            full_doc_text = ""
            segments_info: List[Dict] = []
            current_offset = 0
            for group in premerged_groups:
                group_text = " ".join(u.texto for u in group)
                start = current_offset
                end = start + len(group_text)
                segments_info.append(
                    {
                        "text": group_text,
                        "start": start,
                        "end": end,
                        "uce_id": group[-1].id,  # last UCE in group (root lookup)
                    }
                )
                full_doc_text += group_text + " "
                current_offset = end + 1

            # ── 3c. Extract global coref chains (sliding window, deduped) ──
            try:
                global_chains = ps._extract_global_chains(
                    segments_info,
                    full_doc_text,
                    context_units=context_units,
                )
            except Exception as e:
                logger.warning(
                    "COREF chain extraction failed for doc %s: %s",
                    doc_uces[0].doc_id,
                    e,
                )
                global_chains = []
            total_chains += len(global_chains)

            # ── 3d. Greedy forward merge pass (group-level) ──
            #     Iterates over pre-merged groups.  van_unidos may still
            #     dissolve boundaries bridged by a coref chain.
            n_segs = len(premerged_groups)
            uc_groups: List[List[UCE]] = []
            current_group: List[UCE] = list(premerged_groups[0])  # mutable copy
            current_seg: Dict = dict(segments_info[0])

            for i in range(1, n_segs):
                next_group = premerged_groups[i]
                next_seg = segments_info[i]
                total_boundaries += 1

                # Root lookup: last UCE in the current merged group
                last_uce_id = current_group[-1].id
                seg1_roots = root_cache.get(last_uce_id, set())

                should_merge, has_spanning = ps.van_unidos(
                    current_seg,
                    next_seg,
                    global_chains,
                    precomputed_roots=seg1_roots,
                )

                if has_spanning:
                    covered_boundaries += 1

                if should_merge:
                    coref_merges += 1
                    current_group.extend(next_group)
                    current_seg = {
                        "text": current_seg["text"] + " " + next_seg["text"],
                        "start": current_seg["start"],
                        "end": next_seg["end"],
                        "uce_id": current_seg["uce_id"],
                    }
                else:
                    uc_groups.append(current_group)
                    current_group = list(next_group)
                    current_seg = dict(next_seg)

            uc_groups.append(current_group)

            # 3e. Apply min_forms floor (merges groups too small for CHD)
            uc_groups = _merge_small_uc_groups(uc_groups, min_forms)

            # 3f. Build UC objects
            doc_id = doc_uces[0].doc_id
            shared_meta = (doc_metadata_map or {}).get(doc_id, {})

            for group_uces in uc_groups:
                if not group_uces:
                    continue

                uce_ids: List[str] = [u.id for u in group_uces]
                texts: List[str] = []
                lemmas: List[str] = []
                stems: List[str] = []
                bigrams: List = []
                bigram_stems: List = []
                trigrams: List = []
                trigram_stems: List = []

                for u in group_uces:
                    texts.append(u.texto)
                    lemmas.extend(u.lemmas)
                    stems.extend(u.stems)
                    bigrams.extend(u.bigrams)
                    bigram_stems.extend(u.bigram_stems)
                    trigrams.extend(getattr(u, "trigrams", []))
                    trigram_stems.extend(getattr(u, "trigram_stems", []))

                uc = UC(
                    id=f"uc_coref_{doc_id}_{global_uc_counter}",
                    texto=" ".join(texts),
                    metadata={**shared_meta, "uce_local_idx": uce_ids[0]},
                    uce_ids=uce_ids,
                    lemmas=lemmas,
                    stems=stems,
                    bigrams=bigrams,
                    bigram_stems=bigram_stems,
                    trigrams=trigrams,
                    trigram_stems=trigram_stems,
                )
                ucs.append(uc)
                for uid in uce_ids:
                    uce_to_uc[uid] = uc.id
                    for u in group_uces:
                        if u.id == uid:
                            u.uc_id = uc.id
                global_uc_counter += 1

        # ── 4. Strip suffix from all uce_ids (restore bare IDs for Hungarian) ─
        for uc in ucs:
            uc.uce_ids = [
                uid[: -len(suffix)] if uid.endswith(suffix) else uid
                for uid in uc.uce_ids
            ]
        uce_to_uc = {
            (uid[: -len(suffix)] if uid.endswith(suffix) else uid): uc_id
            for uid, uc_id in uce_to_uc.items()
        }

        # ── 5. Hard failure: no UCs produced ────────────────────────────────
        if not ucs:
            print(
                f"   [COREF context={context_units}] Zero UCs produced — "
                "check min_forms or coref coverage."
            )
            return None

        # Integrity check (same guarantee as construir_ucs)
        all_ids = [uid for uc in ucs for uid in uc.uce_ids]
        assert len(all_ids) == len(set(all_ids)), (
            "[COREF] UCE assigned to multiple UCs — bug in merge logic"
        )

        # ── 6. Embeddings, matrix, PPMI, CHD (identical to other methods) ────
        if self.config.use_embeddings:
            ucs = self.segmentador.generar_embeddings_ucs(ucs)

        builder = MatrizBuilder(self.config)
        voc = builder.construir_vocabulario(ucs)
        if not voc:
            print(f"   [COREF context={context_units}] Empty vocabulary — skipping.")
            return None

        mat_sparse = builder.construir_matriz_dispersa(ucs, voc)
        if self.ppmi_builder is not None:
            mat_sparse = self.ppmi_builder.build(mat_sparse, voc)

        uc_vectors: Optional[np.ndarray] = None
        if retrofitted_vectors is not None:
            try:
                uc_vectors = _compute_uc_centroids(
                    ucs,
                    retrofitted_vectors,
                    self.uc_builder.vectorizer.we,
                    self.uc_builder.vectorizer.subtlex,
                )
            except Exception as e:
                logger.warning("_compute_uc_centroids failed (coref): %s", e)

        cdh = ClasificadorDescendente(self.config)
        labels, tree = cdh.clasificar(mat_sparse, uc_vectors=uc_vectors)

        for i, uc in enumerate(ucs):
            uc.cluster_id = int(labels[i]) if labels[i] != -1 else None

        real = np.unique(labels[labels >= 0])
        merge_rate = coref_merges / max(1, total_boundaries)
        chain_cov = covered_boundaries / max(1, total_boundaries)

        print(
            f"   [COREF context={context_units} min_forms={min_forms}] "
            f"UCs={len(ucs)}, clusters={len(real)}, noise={np.sum(labels == -1)} | "
            f"merges={coref_merges}/{total_boundaries} ({merge_rate:.1%}), "
            f"chain_cov={chain_cov:.1%}, chains={total_chains}"
        )

        return {
            "method": "coref",
            "context_units": context_units,
            "min_forms": min_forms,
            "ucs": ucs,
            "labels": labels,
            "voc": voc,
            "uce_to_uc": uce_to_uc,
            "tree": tree,
            "coref_quality": {
                "total_boundaries": total_boundaries,
                "coref_merges": coref_merges,
                "covered_boundaries": covered_boundaries,
                "merge_rate": round(merge_rate, 4),
                "chain_coverage": round(chain_cov, 4),
                "n_chains_total": total_chains,
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # METHOD 3 — HDBSCAN on retrofitted UCE embedding space
    # ═══════════════════════════════════════════════════════════════════
    def _clasificar_umbral_embedding(
        self,
        uces_por_doc: List[List[UCE]],
        min_cluster_size: int,
        retrofitted_vectors: Dict[str, np.ndarray],
    ) -> Optional[Dict]:
        """
        Method C: direct HDBSCAN clustering of all UCEs in retrofitted space.

        No UC objects are created here; cluster labels are assigned per-UCE.
        The result dict carries 'uce_to_label' instead of 'ucs'/'labels',
        making it structurally distinct from Methods A and B — handled
        explicitly in _extract_uce_labels and _hungarian_stability_direct.

        Quality signal: HDBSCAN cluster_persistence_ (higher = more stable).
        """
        if not _HDBSCAN_AVAILABLE:
            print("   [EMB] hdbscan not installed — skipping Method C.")
            return None

        we = self.uc_builder.vectorizer.we
        sub = self.uc_builder.vectorizer.subtlex

        all_uces = [uce for doc in uces_por_doc for uce in doc]
        if not all_uces:
            return None

        # Normalised bare UCE IDs (no suffix needed — we never build UCs here)
        uce_ids = [normalizar_id_uce(uce) for uce in all_uces]

        mat = _compute_uce_centroids_from_retro(all_uces, retrofitted_vectors, we, sub)

        # Reject if >50 % of UCEs are unembeddable
        zero_rows = int(np.all(mat == 0, axis=1).sum())
        if zero_rows > len(all_uces) * 0.5:
            print(
                f"   [EMB] {zero_rows}/{len(all_uces)} zero-vectors — "
                "too many OOV UCEs, skipping Method C."
            )
            return None

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=getattr(self.config, "hdbscan_uce_min_samples", 1),
            cluster_selection_epsilon=getattr(self.config, "hdbscan_uce_epsilon", 0.0),
            metric=getattr(self.config, "hdbscan_uce_metric", "euclidean"),
            gen_min_span_tree=True,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(mat)

        real = np.unique(labels[labels >= 0])
        n_noise = int(np.sum(labels == -1))
        persist = list(getattr(clusterer, "cluster_persistence_", []))

        print(
            f"   [EMB min_cluster_size={min_cluster_size}] "
            f"UCEs={len(all_uces)}, clusters={len(real)}, "
            f"noise={n_noise}, persistence={[round(p, 3) for p in persist]}"
        )

        if len(real) < 2:
            return None

        # uce_to_label: only non-noise UCEs
        uce_to_label: Dict[str, int] = {
            uce_ids[i]: int(labels[i]) for i in range(len(uce_ids)) if labels[i] >= 0
        }

        return {
            "method": "embedding",
            "min_cluster_size": min_cluster_size,
            "uce_to_label": uce_to_label,  # ← per-UCE, no UC wrapper
            "labels_array": labels,  # full array aligned with uce_ids
            "uce_ids": uce_ids,
            "embedding_matrix": mat,
            "cluster_persistence": persist,
            "n_clusters": int(len(real)),
            "n_noise": n_noise,
        }

    # ─────────────────────────────────────────────────────────────────────
    # HELPER: extract {uce_id: label} from WC/SIM result dict (A or B)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_uce_labels(res: Dict) -> Dict[str, int]:
        """
        Maps every UCE id to its cluster label in a WC or SIM result dict.
        Method C result dicts carry 'uce_to_label' directly and skip this.
        """
        out: Dict[str, int] = {}
        for i, uc in enumerate(res["ucs"]):
            lbl = int(res["labels"][i])
            for uid in uc.uce_ids:
                out[uid] = lbl
        return out

    # ─────────────────────────────────────────────────────────────────────
    # UNIFIED HUNGARIAN — works on plain {uce_id: label} dicts (all pairs)
    # ─────────────────────────────────────────────────────────────────────
    def _hungarian_stability_direct(
        self,
        uce_to_ca: Dict[str, int],
        uce_to_cb: Dict[str, int],
        label: str = "",
    ) -> Tuple[
        Dict[str, int], float, np.ndarray, np.ndarray, np.ndarray, Dict[int, int]
    ]:
        """
        Hungarian alignment between two uce_id→cluster mappings.

        Parameters
        ----------
        uce_to_ca  : {uce_id: cluster_label}  — primary (A)
        uce_to_cb  : {uce_id: cluster_label}  — secondary (B)
        label      : display label for print output

        Returns
        -------
        stable      : {uce_id: cluster_from_a}  for stably matched UCEs
        ari         : ARI over common valid-label UCEs
        arr_a/arr_b : aligned label arrays for further ARI calls
        overlap     : raw overlap matrix (n_real_a × n_real_b)
        mapping_a_b : {class_a → class_b}  (Hungarian assignment)
        """
        if linear_sum_assignment is None:
            logger.error("scipy.optimize.linear_sum_assignment not available.")
            return {}, 0.0, np.array([]), np.array([]), np.zeros((0, 0)), {}

        common = set(uce_to_ca) & set(uce_to_cb)
        if not common:
            return {}, 0.0, np.array([]), np.array([]), np.zeros((0, 0)), {}

        real_a = np.unique([v for v in uce_to_ca.values() if v >= 0])
        real_b = np.unique([v for v in uce_to_cb.values() if v >= 0])
        if len(real_a) == 0 or len(real_b) == 0:
            return {}, 0.0, np.array([]), np.array([]), np.zeros((0, 0)), {}

        r_ai = {int(v): i for i, v in enumerate(real_a)}
        r_bi = {int(v): i for i, v in enumerate(real_b)}

        overlap = np.zeros((len(real_a), len(real_b)), dtype=float)
        for uid in common:
            la = int(uce_to_ca[uid])
            lb = int(uce_to_cb[uid])
            if la >= 0 and lb >= 0:
                overlap[r_ai[la], r_bi[lb]] += 1

        row_ind, col_ind = linear_sum_assignment(-overlap)
        mapping: Dict[int, int] = {
            int(real_a[r]): int(real_b[c]) for r, c in zip(row_ind, col_ind)
        }

        stable: Dict[str, int] = {
            uid: int(uce_to_ca[uid])
            for uid in common
            if uce_to_ca[uid] >= 0
            and uce_to_cb[uid] >= 0
            and mapping.get(int(uce_to_ca[uid])) == int(uce_to_cb[uid])
        }

        common_sorted = sorted(common)
        arr_a = np.array([uce_to_ca[u] for u in common_sorted])
        arr_b = np.array([uce_to_cb[u] for u in common_sorted])
        valid = (arr_a >= 0) & (arr_b >= 0)
        ari = (
            float(adjusted_rand_score(arr_a[valid], arr_b[valid]))
            if valid.sum() >= 2
            else 0.0
        )

        tag = f" [{label}]" if label else ""
        print(
            f"   Hungarian{tag}: {len(stable)}/{len(common)} stable "
            f"({100.0 * len(stable) / max(1, len(common)):.1f}%)  ARI={ari:.3f}"
        )
        print(f"   Overlap matrix ({len(real_a)}×{len(real_b)}):\n{overlap}")

        return stable, ari, arr_a[valid], arr_b[valid], overlap, mapping

    # ═══════════════════════════════════════════════════════════════════
    # HUNGARIAN — align two classification results over common UCEs
    # ═══════════════════════════════════════════════════════════════════
    # VERDICT ASSIGNMENT — triple-method stability
    # ═══════════════════════════════════════════════════════════════════
    def _triple_stability_verdicts(
        self,
        uces_por_doc: List[List[UCE]],
        mode: str,
        # within-method stable sets  (uce_id → primary cluster)
        stable_wc: Dict[str, int],
        stable_sim: Dict[str, int],
        stable_emb: Dict[str, int],
        # cross-method stable sets   (uce_id → cluster in primary reference = A)
        stable_cross_ab: Dict[str, int],
        stable_cross_bc: Dict[str, int],
        stable_cross_ac: Dict[str, int],
        # quality scores
        ari_wc: float,
        ari_sim: float,
        ari_emb: float,
        ari_cross_ab: float,
        ari_cross_bc: float,
        ari_cross_ac: float,
    ) -> Dict:
        """
        Assign per-UCE stability verdicts and build the primary stable list.

        Triple-intersection logic ("all" mode):
        ────────────────────────────────────────
        A UCE is primary-stable if and only if it is:
          1. stable within A  (WC1 ≈ WC2)
          2. stable within B  (SIM1 ≈ SIM2)
          3. stable within C  (EMB1 ≈ EMB2)
          4. A and B cross-agree  (uce ∈ stable_cross_ab)
          5. B and C cross-agree  (uce ∈ stable_cross_bc)
          6. A and C cross-agree  (uce ∈ stable_cross_ac)
        Conditions 4–6 are all required; no two-of-three shortcuts.

        Graceful degradation:
          If EMB is unavailable (empty stable_emb), "all" collapses to "wc_sim".
        """
        all_uids = {uce.id for doc in uces_por_doc for uce in doc}
        n_total = len(all_uids)

        # ── Resolve primary set and cluster reference ─────────────────────
        emb_available = bool(stable_emb)

        _mode = mode
        if _mode == "both":
            _mode = "wc_sim"
        if _mode == "all" and not emb_available:
            print("   [Verdict] EMB unavailable — degrading 'all' to 'wc_sim'.")
            _mode = "wc_sim"

        if _mode == "wc_only":
            primary_ids = set(stable_wc)
            primary_cluster = stable_wc

        elif _mode == "sim_only":
            primary_ids = set(stable_sim)
            primary_cluster = stable_sim

        elif _mode == "emb_only":
            primary_ids = set(stable_emb)
            primary_cluster = stable_emb

        elif _mode == "wc_sim":
            primary_ids = set(stable_cross_ab)
            primary_cluster = stable_wc  # WC1 as cluster reference

        elif _mode == "wc_emb":
            primary_ids = set(stable_cross_ac)
            primary_cluster = stable_wc

        elif _mode == "sim_emb":
            primary_ids = set(stable_cross_bc)
            primary_cluster = stable_sim

        else:  # "all" — full triple intersection
            wc_s = set(stable_wc)
            sim_s = set(stable_sim)
            emb_s = set(stable_emb)
            ab_s = set(stable_cross_ab)
            bc_s = set(stable_cross_bc)
            ac_s = set(stable_cross_ac)
            primary_ids = wc_s & sim_s & emb_s & ab_s & bc_s & ac_s
            primary_cluster = stable_wc  # WC1 as cluster reference

        # ── Back-propagate to UCE objects ─────────────────────────────────
        primary_stable_uces: List[UCE] = []

        for doc_uces in uces_por_doc:
            for uce in doc_uces:
                uid = uce.id

                if _mode == "all":
                    uce.stable_wc = uid in stable_wc
                    uce.stable_sim = uid in stable_sim
                    uce.stable_emb = uid in stable_emb
                    uce.stable_cross_ab = uid in stable_cross_ab
                    uce.stable_cross_bc = uid in stable_cross_bc
                    uce.stable_cross_ac = uid in stable_cross_ac

                if uid in primary_ids and uid in primary_cluster:
                    uce.cluster_id = primary_cluster[uid]
                    uce.is_stable = True
                    uce.stability_method = _mode
                    primary_stable_uces.append(uce)
                else:
                    # Best available partial tag for debugging
                    in_wc = uid in stable_wc
                    in_sim = uid in stable_sim
                    in_emb = uid in stable_emb
                    methods = (
                        ("wc" if in_wc else "")
                        + ("_sim" if in_sim else "")
                        + ("_emb" if in_emb else "")
                    ).strip("_") or "unstable"
                    # Assign best available cluster_id for downstream reference
                    if in_wc:
                        uce.cluster_id = stable_wc[uid]
                    elif in_sim:
                        uce.cluster_id = stable_sim[uid]
                    elif in_emb:
                        uce.cluster_id = stable_emb[uid]
                    else:
                        uce.cluster_id = None
                    uce.is_stable = False
                    uce.stability_method = methods

        # ── Build summary report ──────────────────────────────────────────
        n_primary = len(primary_stable_uces)
        n_unstable = n_total - n_primary

        def _pct(n):
            return f"{100 * n / max(1, n_total):5.1f}%"

        if _mode == "all":
            n_wc = len(stable_wc)
            n_sim = len(stable_sim)
            n_emb = len(stable_emb)
            n_ab = len(stable_cross_ab)
            n_bc = len(stable_cross_bc)
            n_ac = len(stable_cross_ac)
            summary = (
                f"\n╔{'═' * 70}╗\n"
                f"║{'STABILITY REPORT — TRIPLE METHOD (WC + SIM + EMB)':^70}║\n"
                f"╠{'═' * 70}╣\n"
                f"║  Total UCEs:                   {n_total:>6}  (100.0%)                    ║\n"
                f"╠{'═' * 70}╣\n"
                f"║  WITHIN-METHOD STABILITY                                             ║\n"
                f"║  A  Word-count   (A1↔A2):      {n_wc:>6}  ({_pct(n_wc)})  ARI={ari_wc:.3f}     ║\n"
                f"║  B  Similarity   (B1↔B2):      {n_sim:>6}  ({_pct(n_sim)})  ARI={ari_sim:.3f}     ║\n"
                f"║  C  Embedding    (C1↔C2):      {n_emb:>6}  ({_pct(n_emb)})  ARI={ari_emb:.3f}     ║\n"
                f"╠{'═' * 70}╣\n"
                f"║  CROSS-METHOD PAIRWISE                                               ║\n"
                f"║  A↔B cross-stable:             {n_ab:>6}  ({_pct(n_ab)})  ARI={ari_cross_ab:.3f}     ║\n"
                f"║  B↔C cross-stable:             {n_bc:>6}  ({_pct(n_bc)})  ARI={ari_cross_bc:.3f}     ║\n"
                f"║  A↔C cross-stable:             {n_ac:>6}  ({_pct(n_ac)})  ARI={ari_cross_ac:.3f}     ║\n"
                f"╠{'═' * 70}╣\n"
                f"║  TRIPLE INTERSECTION  (A∩B∩C):  {n_primary:>6}  ({_pct(n_primary)})                   ║\n"
                f"║  Unstable in any method:        {n_unstable:>6}  ({_pct(n_unstable)})                   ║\n"
                f"╚{'═' * 70}╝\n"
            )
        elif _mode in ("wc_sim", "both"):
            n_wc = len(stable_wc)
            n_sim = len(stable_sim)
            n_ab = len(stable_cross_ab)
            summary = (
                f"\n╔{'═' * 70}╗\n"
                f"║{'STABILITY REPORT — DUAL METHOD (WC + SIM)':^70}║\n"
                f"╠{'═' * 70}╣\n"
                f"║  Total UCEs:         {n_total:>6}  (100.0%)                              ║\n"
                f"║  WC stable    A1↔A2: {n_wc:>6}  ({_pct(n_wc)})  ARI={ari_wc:.3f}            ║\n"
                f"║  SIM stable   B1↔B2: {n_sim:>6}  ({_pct(n_sim)})  ARI={ari_sim:.3f}            ║\n"
                f"║  Cross A↔B:          {n_ab:>6}  ({_pct(n_ab)})  ARI={ari_cross_ab:.3f}            ║\n"
                f"║  Primary output:     {n_primary:>6}  ({_pct(n_primary)})                          ║\n"
                f"╚{'═' * 70}╝\n"
            )
        else:
            summary = (
                f"\n╔{'═' * 70}╗\n"
                f"║  STABILITY REPORT — {_mode.upper():<50}║\n"
                f"╠{'═' * 70}╣\n"
                f"║  Total: {n_total:>6}   Stable: {n_primary:>6} ({_pct(n_primary)})   "
                f"Unstable: {n_unstable:>6} ({_pct(n_unstable)})  ║\n"
                f"╚{'═' * 70}╝\n"
            )

        return {
            "primary_stable_uces": primary_stable_uces,
            "summary": summary,
            "effective_mode": _mode,
        }

    # ═══════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════════════
    def run(
        self,
        corpus_raw: Dict,
        retrofitted_vectors: Optional[Dict[str, np.ndarray]] = None,
        ppmi_builder: Optional["SubtlexPPMIBuilder"] = None,
        cached_uces_por_doc: Optional[List[List[UCE]]] = None,
        cached_doc_metadata_map: Optional[Dict] = None,
    ) -> Tuple:
        self.ppmi_builder = ppmi_builder
        mode = getattr(self.config, "classification_mode", "wc_only")
        # normalise alias
        # Normalise legacy aliases so downstream code only sees canonical names
        _ALIASES = {
            "both": "wc_coref",
            "wc_sim": "wc_coref",
            "sim_only": "coref_only",
            "sim_emb": "coref_emb",
        }
        mode = _ALIASES.get(mode, mode)

        # ── Step 0: obtain / re-use UCEs ──────────────────────────────
        if cached_uces_por_doc is not None:
            print("   Reutilizando UCEs cacheadas...")
            uces_por_doc = copy.deepcopy(cached_uces_por_doc)
            doc_metadata_map = cached_doc_metadata_map or {}
            for doc_uces in uces_por_doc:
                for uce in doc_uces:
                    uce.id = normalizar_id_uce(uce)
            self.segmentador.doc_metadata_map = doc_metadata_map
        else:
            print("Segmentando en UCEs...")
            uces_por_doc, doc_metadata_map = self.segmentador.segmentar_en_uces(
                corpus_raw
            )
            for doc_uces in uces_por_doc:
                self.segmentador.lematizar_uces(doc_uces)
            for doc_uces in uces_por_doc:
                for uce in doc_uces:
                    uce.id = normalizar_id_uce(uce)

        mf1, mf2 = self.config.min_forms_uc
        thr = self.uc_builder.cfg.similarity_threshold
        thr_delta = getattr(self, "threshold_delta", 0.05)

        # ── Per-method result containers ──────────────────────────────
        res_wc1 = res_wc2 = res_coref1 = res_coref2 = res_emb1 = res_emb2 = None
        # within-method stable sets
        stable_wc: Dict[str, int] = {}
        stable_coref: Dict[str, int] = {}
        stable_emb: Dict[str, int] = {}
        ari_wc = ari_coref = ari_emb = 0.0
        # within-method overlap matrices
        overlap_aa = overlap_bb = overlap_cc = np.zeros((0, 0))

        # ── Step 1: Method A — word-count double-pass ─────────────────
        _do_wc = mode in ("wc_only", "wc_coref", "wc_emb", "all")
        if _do_wc:
            print(f"\n--- Classification A1: word-count tight (mf={mf1}) ---")
            res_wc1 = self._clasificar_umbral_wordcount(
                copy.deepcopy(uces_por_doc), mf1
            )
            print(f"--- Classification A2: word-count loose (mf={mf2}) ---")
            res_wc2 = self._clasificar_umbral_wordcount(
                copy.deepcopy(uces_por_doc), mf2
            )

            if res_wc1 and res_wc2:
                uce_to_a1 = self._extract_uce_labels(res_wc1)
                uce_to_a2 = self._extract_uce_labels(res_wc2)
                stable_wc, ari_wc, _, _, overlap_aa, _ = (
                    self._hungarian_stability_direct(uce_to_a1, uce_to_a2, "A1↔A2")
                )
            else:
                print("   [!] Method A failed — skipping.")

        # ── Step 2: Method B — coref-driven double-pass ───────────────
        _do_coref = mode in ("coref_only", "wc_coref", "coref_emb", "all")
        res_coref1 = res_coref2 = None
        stable_coref: Dict[str, int] = {}
        ari_coref = 0.0
        overlap_bb = np.zeros((0, 0))

        cu_tight = getattr(self.config, "coref_context_units_tight", 2)
        cu_loose = getattr(self.config, "coref_context_units_loose", 4)

        if _do_coref:
            print(
                f"\n--- Classification B1: coref tight (context_units={cu_tight}) ---"
            )
            res_coref1 = self._clasificar_umbral_coref(
                copy.deepcopy(uces_por_doc),
                mf1,
                cu_tight,
                retrofitted_vectors=retrofitted_vectors,
            )
            print(f"--- Classification B2: coref loose (context_units={cu_loose}) ---")
            res_coref2 = self._clasificar_umbral_coref(
                copy.deepcopy(uces_por_doc),
                mf2,
                cu_loose,
                retrofitted_vectors=retrofitted_vectors,
            )

            if res_coref1 and res_coref2:
                uce_to_b1 = self._extract_uce_labels(res_coref1)
                uce_to_b2 = self._extract_uce_labels(res_coref2)
                stable_coref, ari_coref, _, _, overlap_bb, _ = (
                    self._hungarian_stability_direct(uce_to_b1, uce_to_b2, "B1↔B2")
                )
            else:
                print("   [!] Method B (coref) failed — skipping.")

        # ── Step 3: Method C — HDBSCAN on retrofitted embedding space ─
        _do_emb = (
            mode in ("emb_only", "wc_emb", "coref_emb", "all")
            and retrofitted_vectors is not None
        )
        if _do_emb:
            mcs_tight = getattr(self.config, "hdbscan_uce_min_cluster_size", 5)
            mcs_loose = getattr(self.config, "hdbscan_uce_min_cluster_size_loose", 3)
            print(
                f"\n--- Classification C1: EMB tight (min_cluster_size={mcs_tight}) ---"
            )
            res_emb1 = self._clasificar_umbral_embedding(
                uces_por_doc, mcs_tight, retrofitted_vectors
            )
            print(
                f"--- Classification C2: EMB loose (min_cluster_size={mcs_loose}) ---"
            )
            res_emb2 = self._clasificar_umbral_embedding(
                uces_por_doc, mcs_loose, retrofitted_vectors
            )

            if res_emb1 and res_emb2:
                uce_to_c1 = res_emb1["uce_to_label"]
                uce_to_c2 = res_emb2["uce_to_label"]
                stable_emb, ari_emb, _, _, overlap_cc, _ = (
                    self._hungarian_stability_direct(uce_to_c1, uce_to_c2, "C1↔C2")
                )
            else:
                print("   [!] Method C failed — skipping.")

        # ── Step 4: cross-method pairwise comparisons ─────────────────
        stable_cross_ab: Dict[str, int] = {}
        stable_cross_bc: Dict[str, int] = {}
        stable_cross_ac: Dict[str, int] = {}
        ari_cross_ab = ari_cross_bc = ari_cross_ac = 0.0
        overlap_ab = overlap_bc = overlap_ac = np.zeros((0, 0))
        mapping_ab: Dict[int, int] = {}
        mapping_bc: Dict[int, int] = {}
        mapping_ac: Dict[int, int] = {}

        if _do_wc and _do_coref and res_wc1 and res_coref1:
            print("\n--- Cross A↔B (wc ↔ coref) ---")
            stable_cross_ab, ari_cross_ab, _, _, overlap_ab, mapping_ab = (
                self._hungarian_stability_direct(
                    self._extract_uce_labels(res_wc1),
                    self._extract_uce_labels(res_coref1),
                    "A↔B cross",
                )
            )

        if _do_coref and _do_emb and res_coref1 and res_emb1:
            print("\n--- Cross B↔C (coref ↔ emb) ---")
            stable_cross_bc, ari_cross_bc, _, _, overlap_bc, mapping_bc = (
                self._hungarian_stability_direct(
                    self._extract_uce_labels(res_coref1),
                    res_emb1["uce_to_label"],
                    "B↔C cross",
                )
            )

        if _do_wc and _do_emb and res_wc1 and res_emb1:
            print("\n--- Cross A↔C (wc ↔ emb) ---")
            stable_cross_ac, ari_cross_ac, _, _, overlap_ac, mapping_ac = (
                self._hungarian_stability_direct(
                    self._extract_uce_labels(res_wc1),
                    res_emb1["uce_to_label"],
                    "A↔C cross",
                )
            )

        # ── Step 5: triple verdict ─────────────────────────────────────
        report = self._triple_stability_verdicts(
            uces_por_doc=uces_por_doc,
            mode=mode,
            stable_wc=stable_wc,
            stable_sim=stable_coref,
            stable_emb=stable_emb,
            stable_cross_ab=stable_cross_ab,
            stable_cross_bc=stable_cross_bc,
            stable_cross_ac=stable_cross_ac,
            ari_wc=ari_wc,
            ari_sim=ari_coref,
            ari_emb=ari_emb,
            ari_cross_ab=ari_cross_ab,
            ari_cross_bc=ari_cross_bc,
            ari_cross_ac=ari_cross_ac,
        )

        primary_stable = report["primary_stable_uces"]
        print(report["summary"])

        # ── Store pairwise matrices for orchestrator ──────────────────
        def _arr(m):
            return m.tolist() if isinstance(m, np.ndarray) and m.size else []

        self.last_pairwise_stability = {
            "overlap_aa": _arr(overlap_aa),
            "ari_wc": ari_wc,
            "overlap_bb": _arr(overlap_bb),
            "ari_coref": ari_coref,
            "overlap_cc": _arr(overlap_cc),
            "ari_emb": ari_emb,
            "coref_quality_b1": res_coref1["coref_quality"] if res_coref1 else {},
            "coref_quality_b2": res_coref2["coref_quality"] if res_coref2 else {},
            "emb_persistence_c1": res_emb1["cluster_persistence"] if res_emb1 else [],
            "emb_persistence_c2": res_emb2["cluster_persistence"] if res_emb2 else [],
            "overlap_ab": _arr(overlap_ab),
            "ari_cross_ab": ari_cross_ab,
            "overlap_bc": _arr(overlap_bc),
            "ari_cross_bc": ari_cross_bc,
            "overlap_ac": _arr(overlap_ac),
            "ari_cross_ac": ari_cross_ac,
        }

        # ── Early return guard ────────────────────────────────────────
        if not primary_stable:
            print("   [!] Zero primary-stable UCEs — returning empty result.")
            all_res = [r for r in [res_wc1, res_wc2, res_coref1, res_coref2] if r]
            return (
                [],
                pd.DataFrame(),
                [],
                uces_por_doc,
                all_res,
                None,
                None,
                None,
                None,
                doc_metadata_map,
            )

        # Choose primary result dict (for vocabulary extraction)
        effective_mode = report["effective_mode"]
        if effective_mode in ("wc_only", "wc_coref", "wc_emb", "all"):
            primary_res = res_wc1
        elif effective_mode in ("coref_only", "coref_emb"):
            primary_res = res_coref1
        else:  # emb_only
            primary_res = None

        if primary_res is None and effective_mode != "emb_only":
            print("   [!] Fatal: primary_res unavailable.")
            return (
                [],
                pd.DataFrame(),
                [],
                uces_por_doc,
                [],
                None,
                None,
                None,
                None,
                doc_metadata_map,
            )

        if primary_res is not None:
            voc_uc = primary_res["voc"]
        else:
            builder_tmp = MatrizBuilder(self.config)
            voc_uc = builder_tmp.construir_vocabulario(primary_stable)

        all_res = [r for r in [res_wc1, res_wc2, res_coref1, res_coref2] if r]

        # Provide ARI arrays for optimizer compatibility (use best available pair)
        if _do_wc and res_wc1 and res_wc2:
            labels1_ari = np.array(
                [
                    self._extract_uce_labels(res_wc1).get(u.id, -1)
                    for u in primary_stable
                ]
            )
            labels2_ari = np.array(
                [
                    self._extract_uce_labels(res_wc2).get(u.id, -1)
                    for u in primary_stable
                ]
            )
        elif _do_coref and res_coref1 and res_coref2:
            labels1_ari = np.array(
                [
                    self._extract_uce_labels(res_coref1).get(u.id, -1)
                    for u in primary_stable
                ]
            )
            labels2_ari = np.array(
                [
                    self._extract_uce_labels(res_coref2).get(u.id, -1)
                    for u in primary_stable
                ]
            )
        else:
            labels1_ari = labels2_ari = None

        return (
            primary_stable,
            pd.DataFrame(),
            voc_uc,
            uces_por_doc,
            all_res,
            labels1_ari,
            labels2_ari,
            None,
            None,
            doc_metadata_map,
        )

    # ═══════════════════════════════════════════════════════════════════
    # LEGACY METHODS — kept for reference, no longer called from run()
    # ═══════════════════════════════════════════════════════════════════

    def _uce_to_indices(self, uce_to_uc, ucs):
        """Legacy — kept for external callers."""
        uid_to_idx = {uc.id: i for i, uc in enumerate(ucs)}
        return {
            uid: uid_to_idx[uc_id]
            for uid, uc_id in uce_to_uc.items()
            if uc_id in uid_to_idx
        }

    def _clasificar_umbral(
        self,
        uces_por_doc: List[List[UCE]],
        min_forms: int,
        retrofitted_vectors: Optional[Dict[str, np.ndarray]] = None,
        ppmi_builder: Optional["SubtlexPPMIBuilder"] = None,
    ) -> Optional[Dict]:
        """Legacy single-threshold classification — delegate to similarity path."""
        return self._clasificar_umbral_similarity(
            uces_por_doc,
            min_forms,
            self.uc_builder.cfg.similarity_threshold,
            retrofitted_vectors=retrofitted_vectors,
        )

    def _finalize(
        self, uces_por_doc, resultados, doc_metadata_map, retrofitted_vectors
    ):
        """Legacy alignment — no longer used by run()."""
        if len(resultados) < 2:
            if not resultados:
                return (
                    [],
                    pd.DataFrame(),
                    [],
                    uces_por_doc,
                    [],
                    None,
                    None,
                    None,
                    None,
                    doc_metadata_map,
                )
            res = resultados[0]
            uces_est_list = []
            for doc_uces in uces_por_doc:
                for uce in doc_uces:
                    uid = uce.id
                    if uid in res["uce_to_uc"]:
                        uc_idx = res["uce_to_uc"][uid]
                        c = (
                            int(res["labels"][uc_idx])
                            if uc_idx < len(res["labels"])
                            else -1
                        )
                        if c >= 0:
                            uce.cluster_id = c
                            uce.is_stable = True
                            uces_est_list.append(uce)
                            continue
                    uce.cluster_id = None
                    uce.is_stable = False
            return (
                uces_est_list,
                pd.DataFrame(),
                res["voc"],
                uces_por_doc,
                resultados,
                None,
                None,
                None,
                None,
                doc_metadata_map,
            )

        res1, res2 = resultados[0], resultados[1]
        stable, ari, _, _, ari_arr1, ari_arr2 = self._hungarian_stability(res1, res2)
        uces_est_list = []
        for doc_uces in uces_por_doc:
            for uce in doc_uces:
                if uce.id in stable:
                    uce.cluster_id = stable[uce.id]
                    uce.is_stable = True
                    uces_est_list.append(uce)
                else:
                    uce.cluster_id = None
                    uce.is_stable = False
        return (
            uces_est_list,
            pd.DataFrame(),
            res1["voc"],
            uces_por_doc,
            resultados,
            ari_arr1,
            ari_arr2,
            None,
            None,
            doc_metadata_map,
        )


# ══════════════════════════════════════════════════════════════════════
# FALLBACK (solo cuando use_cdh=False)
# ══════════════════════════════════════════════════════════════════════


class _FallbackClusterizador:
    """HDBSCAN / jerárquico — se mantiene como opción legacy."""

    def __init__(self, config: Config):
        self.config = config
        np.random.seed(config.random_state)

    def _distancia_chi2(self, X):
        rs = X.sum(axis=1, keepdims=True).astype(float)
        rs[rs == 0] = 1
        prof = X / rs
        cm = X.mean(axis=0).astype(float)
        cm[cm == 0] = 1
        dists = []
        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                d = X[i] - X[j]
                dists.append(np.sqrt(np.sum(d**2 / cm)))
        return np.array(dists)

    def clustering(self, matriz, ucs):
        if self.config.clustering_method == "hierarchical":
            dv = (
                self._distancia_chi2(matriz)
                if self.config.distance_metric == "chi2"
                else pdist(matriz > 0, metric=self.config.distance_metric)
            )
            Z = linkage(dv, method=self.config.linkage_method)
            k = self.config.n_clusters or self._sel_k(Z, dv, len(ucs))
            labels = fcluster(Z, t=k, criterion="maxclust") - 1
            return labels, Z, None
        elif self.config.clustering_method == "hdbscan" and _HDBSCAN_AVAILABLE:
            if hasattr(ucs[0], "embedding") and ucs[0].embedding is not None:
                data = np.array([uc.embedding for uc in ucs])
            else:
                dm = squareform(pdist(matriz > 0, metric="jaccard"))
                data = umap.UMAP(
                    n_components=2,
                    random_state=self.config.random_state,
                    metric="precomputed",
                ).fit_transform(dm)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                min_samples=self.config.hdbscan_min_samples,
                cluster_selection_epsilon=self.config.hdbscan_cluster_selection_epsilon,
                gen_min_span_tree=True,
                prediction_data=True,
            )
            labels = clusterer.fit_predict(data)
            return labels, None, None
        raise ValueError(f"Clustering no disponible: {self.config.clustering_method}")

    def _sel_k(self, Z, dv, n):
        if n < 4:
            return 2
        k_min, k_max = 2, min(8, n // 4)
        jumps = {
            k: Z[n - k, 2] - Z[n - k - 1, 2]
            for k in range(k_min, k_max + 1)
            if n - k >= 0 and n - k - 1 >= 0
        }
        return max(jumps, key=jumps.__getitem__) if jumps else k_min


# ══════════════════════════════════════════════════════════════════════
# TÉRMINOS CARACTERÍSTICOS
# ══════════════════════════════════════════════════════════════════════


class TermAnalyzer:
    def __init__(self, config: Config):
        self.config = config

    def compute(
        self, matriz: np.ndarray, labels: np.ndarray, vocabulario: List[str]
    ) -> pd.DataFrame:
        n_ucs, n_terms = matriz.shape
        unique_labels = np.unique(labels)
        if -1 in unique_labels:
            mask = labels != -1
            if not np.any(mask):
                return pd.DataFrame()
            matriz = matriz[mask]
            labels = labels[mask]
            unique_labels = unique_labels[unique_labels != -1]
        if len(unique_labels) < 2:
            return pd.DataFrame()
        n_clusters = len(unique_labels)

        thresholds = (
            self._compute_ctest_thresholds(matriz, labels, n_clusters)
            if self.config.adaptive_ctest
            else np.full((n_terms, n_clusters), self.config.ctest_threshold)
        )

        records = []
        presencia_global = (matriz > 0).sum(axis=0)

        for j in range(n_terms):
            presente = matriz[:, j] > 0
            for kidx, k in enumerate(unique_labels):
                in_cl = labels == k
                a = int(np.sum(presente & in_cl))
                b = int(np.sum(presente & ~in_cl))
                c = int(np.sum(~presente & in_cl))
                d = int(np.sum(~presente & ~in_cl))
                total = a + b + c + d
                if (
                    total == 0
                    or (a + c) == 0
                    or (b + d) == 0
                    or (a + b) == 0
                    or (c + d) == 0
                ):
                    continue
                tabla = np.array([[a, b], [c, d]])
                chi2y, pv, _, _ = chi2_contingency(tabla, correction=True)
                phi = np.sqrt(chi2y / total) if chi2y > 0 else 0.0
                signo = 1 if (a / (a + c)) > (presencia_global[j] / n_ucs) else -1
                cv = np.sqrt(chi2y / (total * (min(tabla.shape) - 1)))

                # Correct Reinert CTEST
                C = np.sqrt(chi2y / (chi2y + presencia_global[j])) if chi2y > 0 else 0.0

                records.append(
                    {
                        "termino": vocabulario[j],
                        "cluster": int(k),
                        "frecuencia_global": int(presencia_global[j]),
                        "frecuencia_cluster": a,
                        "chi2_yates": round(chi2y, 4),
                        "p_valor": pv,
                        "phi": round(signo * phi, 4),
                        "cramer_v": round(cv, 4),
                        "C": round(C, 4),
                        "asignado_estricto": (
                            C > thresholds[j, kidx]
                            if (self.config.use_ctest or self.config.adaptive_ctest)
                            else False
                        ),
                    }
                )
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["p_adj"] = np.nan
        df["significativo"] = False
        for k in unique_labels:
            mask = df["cluster"] == k
            if not mask.any():
                continue
            _, p_adj, _, _ = multipletests(
                df.loc[mask, "p_valor"].values,
                alpha=self.config.fdr_alpha,
                method="fdr_bh",
            )
            df.loc[mask, "p_adj"] = p_adj
            df.loc[mask, "significativo"] = (
                df.loc[mask, "p_valor"] < self.config.fdr_alpha
            )
        return df

    def _compute_ctest_thresholds(self, matriz, labels, n_clusters):
        n_ucs, n_terms = matriz.shape
        rng = np.random.RandomState(self.config.random_state)
        null_C = np.zeros((self.config.n_permutations_ctest, n_terms, n_clusters))
        for pi in range(self.config.n_permutations_ctest):
            lp = rng.permutation(labels)
            for k in range(n_clusters):
                in_k = (lp == k).astype(int)
                pres = (matriz > 0).astype(int)
                a = pres.T @ in_k
                ti = in_k.sum()
                b = matriz.sum(axis=0) - a
                c = ti - a
                exp_a = (a + b) * (a + c) / n_ucs
                chi2 = np.where(exp_a > 0, (a - exp_a) ** 2 / exp_a, 0.0)
                freq_global = a + b
                null_C[pi, :, k] = np.sqrt(chi2 / (chi2 + freq_global))
        return np.percentile(null_C, 100 * (1 - self.config.ctest_alpha), axis=0)


# ══════════════════════════════════════════════════════════════════════
# AFC PROPIO (Math corrections included)
# ══════════════════════════════════════════════════════════════════════


class AFC:
    def __init__(self, config: Config):
        self.config = config

    def fit(
        self,
        mat_clases: np.ndarray,
        class_ids: np.ndarray,
        voc: List[str],
        n_components: int = 2,
    ) -> Dict[str, Any]:
        mat = mat_clases.astype(float) + self.config.pseudocount
        n_rows, n_cols = mat.shape
        total = mat.sum()
        if total <= 0 or n_rows < 2 or n_cols < 2:
            return {}

        row_sums = mat.sum(axis=1)
        col_sums = mat.sum(axis=0)
        r = row_sums / total
        c = col_sums / total

        E = np.outer(row_sums, col_sums) / total
        S = (mat - E) / np.sqrt(np.maximum(E, 1e-12))

        from scipy.linalg import svd as scipy_svd

        U, s, Vt = scipy_svd(S, full_matrices=False)

        # Drop trivial component
        if len(s) <= 1:
            return {}
        k = min(n_components, len(s) - 1)
        U_k = U[:, 1 : k + 1]
        s_k = s[1 : k + 1]
        Vt_k = Vt[1 : k + 1, :]

        # Proper Chi-square coordinates
        row_coords = (U_k * s_k) / np.sqrt(r[:, np.newaxis])
        col_coords = (Vt_k.T * s_k) / np.sqrt(c[:, np.newaxis])
        row_std = U_k / np.sqrt(r[:, np.newaxis])
        col_std = Vt_k.T / np.sqrt(c[:, np.newaxis])

        col_contrib = (Vt_k.T**2).tolist()
        total_inertia = float(np.sum(s**2))
        explained = (
            (s_k**2 / total_inertia).tolist() if total_inertia > 0 else [0.0] * k
        )

        return {
            "class_ids": class_ids.tolist(),
            "row_coords": row_coords.tolist(),
            "col_coords": col_coords.tolist(),
            "row_std": row_std.tolist(),
            "col_std": col_std.tolist(),
            "col_contrib": col_contrib,
            "explained_inertia": explained,
            "singular_values": s_k.tolist(),
            "total_inertia": total_inertia,
        }


# ══════════════════════════════════════════════════════════════════════
# CAH DE TÉRMINOS
# ══════════════════════════════════════════════════════════════════════


class CAHTerminos:
    def __init__(self, config: Config):
        self.config = config

    def run_global(self, afc_result: Dict[str, Any], voc: List[str]) -> Dict[str, Any]:
        if not afc_result or "col_std" not in afc_result:
            return {}
        col_std = np.array(afc_result["col_std"])
        if len(col_std) < 2:
            return {}
        dist_vector = pdist(col_std, metric="euclidean")
        Z = linkage(dist_vector, method="ward")
        return {"Z": Z.tolist(), "labels": voc, "n_terms": len(voc)}

    def run_per_class(
        self,
        class_id: int,
        uces_estables: List[UCE],
        df_terms: pd.DataFrame,
        voc: List[str],
        top_n: int = None,
    ) -> Dict[str, Any]:
        if top_n is None:
            top_n = getattr(self.config, "cah_per_class_top_terms", 50)
        class_uces = [uce for uce in uces_estables if uce.cluster_id == class_id]
        if len(class_uces) < 2:
            return {}

        df_clase = df_terms[df_terms["cluster"] == class_id]
        if df_clase.empty:
            return {}

        df_clase = df_clase.sort_values("phi", key=abs, ascending=False)
        top_terms = df_clase.head(top_n)["termino"].tolist()
        if len(top_terms) < 2:
            return {}

        term_to_idx = {t: i for i, t in enumerate(top_terms)}
        n_uces = len(class_uces)
        n_terms = len(top_terms)
        M = np.zeros((n_uces, n_terms), dtype=int)

        for i, uce in enumerate(class_uces):
            tokens = uce.stems if self.config.stem_backend != "none" else uce.lemmas
            if self.config.use_bigrams:
                tokens.extend(
                    [
                        "_".join(b)
                        for b in (
                            uce.bigram_stems
                            if self.config.stem_backend != "none"
                            else uce.bigrams
                        )
                    ]
                )
            if self.config.use_trigrams:
                tokens.extend(
                    [
                        "_".join(t)
                        for t in (
                            uce.trigram_stems
                            if self.config.stem_backend != "none"
                            else uce.trigrams
                        )
                    ]
                )
            tokens_set = set(tokens)
            for term in top_terms:
                if term in tokens_set:
                    M[i, term_to_idx[term]] = 1

        col_sums = M.sum(axis=0)
        row_sums = M.sum(axis=1)
        total_occ = row_sums.sum()
        if total_occ == 0:
            return {}

        col_mass = row_sums / total_occ
        valid_terms = [i for i, f in enumerate(col_sums) if f > 0]
        if len(valid_terms) < 2:
            return {}

        n_valid = len(valid_terms)
        M_valid = M[:, valid_terms]
        col_sums_valid = col_sums[valid_terms]
        profiles = M_valid / col_sums_valid

        d2_mat = np.zeros((n_valid, n_valid))
        for i_idx in range(n_valid):
            for j_idx in range(i_idx + 1, n_valid):
                diff = profiles[:, i_idx] - profiles[:, j_idx]
                d2 = np.sum((diff**2) * col_mass)
                d2_mat[i_idx, j_idx] = d2_mat[j_idx, i_idx] = d2

        Z = linkage(squareform(d2_mat), method="ward")
        phi_dict = df_clase.set_index("termino")["phi"].to_dict()
        labels_with_phi = [
            f"{top_terms[idx]} (φ={phi_dict.get(top_terms[idx], 0.0):.3f})"
            for idx in valid_terms
        ]

        return {
            "class_id": class_id,
            "Z": Z.tolist(),
            "labels": labels_with_phi,
            "n_terms": len(labels_with_phi),
        }


# ══════════════════════════════════════════════════════════════════════
# ROBUST RANDOM FOREST + SHAP
# ══════════════════════════════════════════════════════════════════════
class RobustRFShapAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.random_state = config.random_state
        self.n_estimators = config.rf_n_estimators
        self.max_depth = config.rf_max_depth
        self.n_iter_tune = 20
        self.feature_selection_threshold = config.rf_feature_selection_threshold
        self.outlier_method = config.rf_outlier_method
        self.impute_strategy = config.rf_impute_strategy
        self.cat_encoding = config.rf_cat_encoding
        self.min_samples_for_tuning = config.rf_min_samples_for_tuning

    def _preprocess(
        self, X_raw: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        X = X_raw.copy()
        n_samples_orig = len(X)
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=np.number).columns.tolist()

        # Fix 1: collect cols to convert BEFORE mutating cat_cols
        cols_to_freq = []
        for col in cat_cols:
            if X[col].isnull().any():
                mode = X[col].mode()[0] if not X[col].mode().empty else "missing"
                X[col] = X[col].fillna(mode)  # Fix 4: no inplace
            if len(X[col].unique()) > 20:
                cols_to_freq.append(col)

        for col in cols_to_freq:
            freq = X[col].value_counts(normalize=True)
            X[col] = X[col].map(freq)
            cat_cols.remove(col)
            num_cols.append(col)

        for col in num_cols:
            if X[col].isnull().any():
                if self.impute_strategy == "median":
                    fill = X[col].median()
                elif self.impute_strategy == "mean":
                    fill = X[col].mean()
                else:
                    fill = 0
                X[col] = X[col].fillna(fill)  # Fix 4: no inplace

        # Fix 2: flag outlier rows with OR logic, invert once at the end
        outlier_flag = np.zeros(len(X), dtype=bool)
        if len(num_cols) > 0 and len(X) >= 10:
            for col in num_cols:
                if col not in X.columns:  # Fix 3: guard stale num_cols
                    continue
                if self.outlier_method == "iqr":
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    col_outliers = (X[col] < Q1 - 1.5 * IQR) | (X[col] > Q3 + 1.5 * IQR)
                else:
                    col_outliers = np.abs(stats.zscore(X[col])) > 3
                outlier_flag |= col_outliers  # Fix 2: OR, not AND
            clean_mask = ~outlier_flag
            if clean_mask.sum() >= max(10, len(X) * 0.5):
                X = X[clean_mask].reset_index(drop=True)  # Fix 3: reset index
                y = np.asarray(y)[
                    clean_mask
                ]  # ← must be plain ndarray before boolean indexing

        cat_cols = [c for c in cat_cols if c in X.columns]
        if self.cat_encoding == "onehot":
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        elif self.cat_encoding == "frequency":
            for col in cat_cols:
                n_unique = X[col].nunique()
                if n_unique <= 10:
                    # Too few levels — frequency encoding is meaningless, use ordinal instead
                    le = {v: i for i, v in enumerate(X[col].unique())}
                    X[col] = X[col].map(le)
                else:
                    X[col] = X[col].map(X[col].value_counts(normalize=True))
        if getattr(self.config, "rf_scale_features", False):
            scaler = StandardScaler()
            current_num_cols = X.select_dtypes(include=np.number).columns.tolist()
            X[current_num_cols] = scaler.fit_transform(X[current_num_cols])

        return X, y, {"n_samples_orig": n_samples_orig, "n_samples_after": len(X)}

    def _tune_hyperparams(self, X, y) -> Dict:
        if len(X) < self.min_samples_for_tuning:
            return {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            }

        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        cv = StratifiedKFold(
            n_splits=min(5, max(2, len(X) // 10)),
            shuffle=True,
            random_state=self.random_state,
        )
        search = RandomizedSearchCV(
            RandomForestClassifier(
                random_state=self.random_state, class_weight="balanced"
            ),
            param_dist,
            n_iter=self.n_iter_tune,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
            random_state=self.random_state,
        )
        search.fit(X, y)
        return search.best_params_

    def _select_features(self, X, y, clf) -> pd.DataFrame:
        selector = SelectFromModel(
            clf, threshold=self.feature_selection_threshold, prefit=True
        )
        return X[X.columns[selector.get_support()].tolist()]

    def analyze(self, uces_estables: List[UCE], labels_uces: np.ndarray) -> Dict:
        # Build raw DataFrame
        meta_rows = [uce.metadata.copy() for uce in uces_estables]
        X_raw = pd.DataFrame(meta_rows)
        y = np.asarray(labels_uces)

        print(f"[RF] All metadata columns: {X_raw.columns.tolist()}")

        # Filter to only the desired metadata columns
        keep_cols = getattr(self.config, "multivariate_metadata", [])
        if keep_cols:
            available = [c for c in keep_cols if c in X_raw.columns]
            if not available:
                print("   [WARNING] No metadata columns found for RF. Skipping.")
                return {"error": "No suitable metadata columns"}
            X_raw = X_raw[available]
            print(f"[RF] Keeping: {available}")
            print(f"[RF] Final columns: {X_raw.columns.tolist()}")

        if X_raw.empty or len(np.unique(y)) < 2:
            return {"error": "Datos insuficientes"}

        # Preprocess (outlier removal, encoding, etc.)
        X, y, preproc_info = self._preprocess(X_raw, y)
        X = X.reset_index(
            drop=True
        )  # guarantee positional alignment for all downstream ops
        y = np.asarray(y)
        print("Columns used in RF:", X.columns.tolist())
        print("Unique value counts per column:")
        for col in X.columns:
            print(f"{col}: {X[col].nunique()} unique values")

        if len(X) < 10 or len(np.unique(y)) < 2:
            return {"error": "Después de preprocesamiento, datos insuficientes"}

        # Tune hyperparameters
        best_params = self._tune_hyperparams(X, y)
        clf = RandomForestClassifier(
            **best_params, random_state=self.random_state, class_weight="balanced"
        )
        clf.fit(X, y)

        # Optional feature selection
        if X.shape[1] > 20 and len(X) > 30:
            X = self._select_features(X, y, clf)
            clf.fit(X, y)

        # Cross‑validated predictions (out‑of‑fold)
        cv = StratifiedKFold(
            n_splits=min(5, max(3, len(X) // 10)),
            shuffle=True,
            random_state=self.random_state,
        )
        scores = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
        oof_preds = cross_val_predict(clf, X, y, cv=cv)

        # Feature importance with bootstrapping
        imp_bootstrap = []
        rng = np.random.RandomState(self.random_state)
        n_bootstrap = min(20, max(5, len(X) // 5))
        for _ in range(n_bootstrap):
            idx = rng.choice(len(X), size=len(X), replace=True)
            clf_b = RandomForestClassifier(
                **best_params, random_state=self.random_state, class_weight="balanced"
            )
            clf_b.fit(X.iloc[idx], y[idx])
            imp_bootstrap.append(clf_b.feature_importances_)

        imp_mean = clf.feature_importances_
        imp_std = np.array(imp_bootstrap).std(axis=0)
        importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": imp_mean,
                "std": imp_std,
                "lower_ci": imp_mean - 1.96 * imp_std,
                "upper_ci": imp_mean + 1.96 * imp_std,
            }
        ).sort_values("importance", ascending=False)

        # Permutation importance
        perm_imp = permutation_importance(
            clf, X, y, n_repeats=10, random_state=self.random_state, n_jobs=-1
        )
        perm_imp_df = pd.DataFrame(
            {
                "feature": X.columns,
                "perm_importance": perm_imp.importances_mean,
                "perm_std": perm_imp.importances_std,
            }
        ).sort_values("perm_importance", ascending=False)

        # ================== SHAP ANALYSIS ==================
        explainer = shap.TreeExplainer(clf)
        shap_out = explainer.shap_values(X)

        # Step 1: unwrap Explanation object if needed
        if hasattr(shap_out, "values"):
            shap_out = shap_out.values

        # Step 2: ensure numpy array
        shap_out = (
            np.array(shap_out) if not isinstance(shap_out, np.ndarray) else shap_out
        )

        # Step 3: normalize shape to list of (n_samples, n_features) arrays, one per class
        if shap_out.ndim == 3:
            if shap_out.shape[0] == len(clf.classes_):
                # Old SHAP format: (n_classes, n_samples, n_features)
                shap_out = [shap_out[i] for i in range(shap_out.shape[0])]
            elif shap_out.shape[2] == len(clf.classes_):
                # New SHAP format: (n_samples, n_features, n_classes)
                shap_out = [shap_out[:, :, i] for i in range(shap_out.shape[2])]
            else:
                print(
                    f"[WARNING] Cannot interpret SHAP shape {shap_out.shape} "
                    f"with {len(clf.classes_)} classes. Falling back to zeros."
                )
                shap_out = [np.zeros((len(X), len(X.columns))) for _ in clf.classes_]
        elif shap_out.ndim == 2:
            # Binary: (n_samples, n_features) — wrap for uniform handling
            shap_out = [shap_out]

        # Step 4: from here shap_out is always a list of arrays
        n_features = len(X.columns)
        n_classes = len(shap_out)
        clf_classes = clf.classes_[:n_classes]

        per_class_signed_mean = [sv.mean(axis=0) for sv in shap_out]
        per_class_abs_mean = [np.abs(sv).mean(axis=0) for sv in shap_out]
        stacked = np.stack(per_class_abs_mean, axis=0)  # (n_classes, n_features)
        mean_abs_shap = stacked.mean(axis=0)  # guaranteed (n_features,)
        assert mean_abs_shap.shape == (n_features,), (
            f"Shape mismatch: {mean_abs_shap.shape}"
        )
        if n_classes == 1:
            # Binary — report for positive class only
            shap_per_class_mean = [
                {
                    "class": int(clf.classes_[-1]),
                    "features": [
                        {
                            "feature": X.columns[i],
                            "mean_shap": float(per_class_signed_mean[0][i]),
                        }
                        for i in range(n_features)
                    ],
                }
            ]
            shap_per_class_mean_abs = [
                {
                    "class": int(clf.classes_[-1]),
                    "features": [
                        {
                            "feature": X.columns[i],
                            "mean_abs_shap": float(per_class_abs_mean[0][i]),
                        }
                        for i in range(n_features)
                    ],
                }
            ]
        else:
            shap_per_class_mean = [
                {
                    "class": int(clf_classes[c_idx]),
                    "features": [
                        {
                            "feature": X.columns[i],
                            "mean_shap": float(per_class_signed_mean[c_idx][i]),
                        }
                        for i in range(n_features)
                    ],
                }
                for c_idx in range(n_classes)
            ]
            shap_per_class_mean_abs = [
                {
                    "class": int(clf_classes[c_idx]),
                    "features": [
                        {
                            "feature": X.columns[i],
                            "mean_abs_shap": float(per_class_abs_mean[c_idx][i]),
                        }
                        for i in range(n_features)
                    ],
                }
                for c_idx in range(n_classes)
            ]

        shap_export = [sv.tolist() for sv in shap_out]

        # Ensure mean_abs_shap is 1D and same length as X.columns
        if isinstance(mean_abs_shap, (int, float)):
            mean_abs_shap = np.array([mean_abs_shap])
        if len(mean_abs_shap) != n_features:
            # Fallback: create zero array
            print(
                f"[WARNING] mean_abs_shap length {len(mean_abs_shap)} != {n_features}. Using zeros."
            )
            mean_abs_shap = np.zeros(n_features)

        # Build overall SHAP importance DataFrame
        shap_df = pd.DataFrame(
            {"feature": X.columns, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False)

        # ========== END SHAP ANALYSIS ==========

        # Confusion matrix
        classes_list = np.unique(y).tolist()
        conf_mat = sk_confusion_matrix(y, oof_preds, labels=classes_list).tolist()

        # Misclassified examples (first few per pair)
        misclassified_texts = {}
        X_dict = X.to_dict(orient="records")
        assert len(y) == len(oof_preds) == len(X_dict), (
            f"[RF] Alignment error: y={len(y)}, oof={len(oof_preds)}, X={len(X_dict)}"
        )
        for true_c, pred_c, uce_row in zip(y, oof_preds, X_dict):
            if true_c != pred_c:
                key = f"{int(true_c)}_to_{int(pred_c)}"
                if key not in misclassified_texts:
                    misclassified_texts[key] = []
                if len(misclassified_texts[key]) < 5:
                    snippet = " | ".join(
                        f"{k}={v}" for k, v in list(uce_row.items())[:4]
                    )
                    misclassified_texts[key].append(snippet)

        return {
            "preprocessing": preproc_info,
            "best_params": best_params,
            "cv_balanced_accuracy": {"mean": scores.mean(), "std": scores.std()},
            "feature_importance": importance_df.to_dict(orient="records"),
            "permutation_importance": perm_imp_df.to_dict(orient="records"),
            "shap_importance": shap_df.to_dict(orient="records"),
            "shap_per_class_mean": shap_per_class_mean,  # NEW
            "shap_per_class_mean_abs": shap_per_class_mean_abs,  # NEW
            "shap_values": shap_export,
            "raw_X": X_dict,
            "oof_predictions": oof_preds.tolist(),
            "y_true": y.tolist(),
            "confusion_matrix": conf_mat,
            "misclassified_texts": misclassified_texts,
            "feature_names": X.columns.tolist(),
            "n_samples": len(X),
            "n_features": n_features,
            "classes": classes_list,
        }


# ══════════════════════════════════════════════════════════════════════
# ANÁLISIS DE METADATOS (nivel UCE)
# ══════════════════════════════════════════════════════════════════════


class MetaAnalyzer:
    def __init__(self, config: Config):
        self.config = config

    def analyze(self, uces_estables, labels_uces):
        if not uces_estables:
            return pd.DataFrame()
        data = [
            dict(**uce.metadata, cluster=int(lbl))
            for uce, lbl in zip(uces_estables, labels_uces)
            if isinstance(uce.metadata, dict)
        ]
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "cluster" not in df.columns:
            return pd.DataFrame()

        # Filter to only the declared metadata variables
        meta_vars = getattr(self.config, "multivariate_metadata", [])
        if meta_vars:
            available = [c for c in meta_vars if c in df.columns]
            if not available:
                print(
                    f"   [MetaAnalyzer] WARNING: none of {meta_vars} found in UCE metadata. "
                    f"Available columns: {[c for c in df.columns if c != 'cluster']}"
                )
                return pd.DataFrame()
            cols_to_analyze = available
        else:
            # Fallback: analyze everything except internal bookkeeping fields
            _skip = {
                "cluster",
                "uce_local_idx",
                "doc_idx",
                "origen_full",
                "origen",
                "seccion",
            }
            cols_to_analyze = [c for c in df.columns if c not in _skip]

        print(f"   MetaAnalyzer: {len(df)} UCEs | variables: {cols_to_analyze}")
        resultados = []
        for col in cols_to_analyze:
            if col == "cluster":
                continue
            if self.config.glm_method == "permutation":
                res = (
                    self._perm_anova(df[col], df["cluster"])
                    if pd.api.types.is_numeric_dtype(df[col])
                    else self._perm_chi2(df[col], df["cluster"])
                )
                if res:
                    resultados.append(res)
            else:
                if (
                    df[col].dtype in ("object", "category")
                    or df[col].dtype.name == "category"
                ):
                    _col_clean = df[col].dropna()
                    _cluster_clean = df.loc[_col_clean.index, "cluster"]
                    tabla = pd.crosstab(_col_clean.astype(str), _cluster_clean)
                    if tabla.size == 0:
                        continue
                    chi2, p, _, _ = chi2_contingency(tabla, correction=False)
                    n = tabla.sum().sum()
                    md = min(tabla.shape) - 1
                    cv = np.sqrt(chi2 / (n * md)) if md > 0 else np.nan
                    resultados.append(
                        {
                            "variable": col,
                            "tipo": "categorica",
                            "chi2": round(chi2, 4),
                            "p_valor": p,
                            "cramer_v": round(cv, 4),
                            "eta_squared": np.nan,
                            "significativo": p < self.config.glm_alpha,
                        }
                    )
                elif pd.api.types.is_numeric_dtype(df[col]):
                    gs = [
                        df[df["cluster"] == g][col].dropna().values
                        for g in df["cluster"].unique()
                    ]
                    if len(gs) < 2 or any(len(g) < 2 for g in gs):
                        continue
                    f, p = f_oneway(*gs)
                    all_v = np.concatenate(gs)
                    ss_t = np.sum((all_v - all_v.mean()) ** 2)
                    ss_b = sum(len(g) * (g.mean() - all_v.mean()) ** 2 for g in gs)
                    resultados.append(
                        {
                            "variable": col,
                            "tipo": "numerica",
                            "estadistico": round(float(f), 4),
                            "p_valor": round(float(p), 4),
                            "cramer_v": np.nan,
                            "eta_squared": round(ss_b / ss_t, 4)
                            if ss_t > 0
                            else np.nan,
                            "significativo": p < self.config.glm_alpha,
                        }
                    )
        return pd.DataFrame(resultados)

    def _perm_anova(self, y, groups):
        ug = groups.unique()
        if len(ug) < 2:
            return None
        mask = y.notna()
        y_c, g_c = y[mask], groups[mask]
        if len(y_c) < len(ug) * 2:
            return None
        f_obs, _ = f_oneway(*(y_c[g_c == g] for g in ug))
        rng = np.random.RandomState(self.config.random_state)
        f_p = [
            f_oneway(*(rng.permutation(y_c.values)[g_c == g] for g in ug))[0]
            for _ in range(self.config.n_permutations)
        ]
        p_v = (np.sum(np.array(f_p) >= f_obs) + 1) / (self.config.n_permutations + 1)
        all_v = y_c.values
        ss_t = np.sum((all_v - all_v.mean()) ** 2)
        ss_b = sum(
            len(y_c[g_c == g]) * (y_c[g_c == g].mean() - all_v.mean()) ** 2 for g in ug
        )
        return {
            "variable": y.name,
            "tipo": "numerica",
            "estadistico": round(float(f_obs), 4),
            "p_valor": round(p_v, 4),
            "eta_squared": round(ss_b / ss_t, 4) if ss_t > 0 else np.nan,
            "cramer_v": np.nan,
            "significativo": p_v < self.config.glm_alpha,
        }

    def _perm_chi2(self, cat, groups):
        tabla = pd.crosstab(cat, groups)
        if tabla.size == 0:
            return None
        chi2_obs, _, _, _ = chi2_contingency(tabla, correction=False)
        n = tabla.sum().sum()
        md = min(tabla.shape) - 1
        cv = np.sqrt(chi2_obs / (n * md)) if (md > 0 and n > 0) else np.nan
        rng = np.random.RandomState(self.config.random_state)
        chi_p = [
            chi2_contingency(
                pd.crosstab(rng.permutation(cat.values), groups), correction=False
            )[0]
            for _ in range(self.config.n_permutations)
        ]
        p_v = (np.sum(np.array(chi_p) >= chi2_obs) + 1) / (
            self.config.n_permutations + 1
        )
        return {
            "variable": cat.name,
            "tipo": "categorica",
            "estadistico": round(float(chi2_obs), 4),
            "p_valor": round(p_v, 4),
            "cramer_v": round(float(cv), 4) if not np.isnan(cv) else np.nan,
            "eta_squared": np.nan,
            "significativo": p_v < self.config.glm_alpha,
        }


# ══════════════════════════════════════════════════════════════════════
# ANÁLISIS DE REDES
# ══════════════════════════════════════════════════════════════════════


class NetworkAnalyzer:
    def __init__(self, config: Config):
        self.config = config

    def build_cooccurrence_graph(self, voc: List[str], matriz: np.ndarray):
        if not _NETWORKX_AVAILABLE:
            return None, {}
        bin_mat = matriz > 0
        N, n_t = bin_mat.shape[0], len(voc)
        cooc = bin_mat.T @ bin_mat
        np.fill_diagonal(cooc, 0)
        freqs = bin_mat.sum(axis=0)
        edges = []
        thr = self.config.network_cooccurrence_threshold
        for i in range(n_t):
            for j in range(i + 1, n_t):
                cij = cooc[i, j]
                if cij < thr:
                    continue
                if self.config.network_significance_filter:
                    a = cij
                    b = freqs[i] - cij
                    c = freqs[j] - cij
                    d = N - (a + b + c)
                    _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
                    if p > self.config.network_significance_alpha:
                        continue
                m = self.config.network_weight_method
                if m == "raw":
                    w = float(cij)
                elif m in ("pmi", "ppmi", "npmi"):
                    p_ij = (cij + 1) / N
                    p_i = (freqs[i] + 1) / N
                    p_j = (freqs[j] + 1) / N
                    if m in ("pmi", "ppmi"):
                        pmi = np.log2(p_ij / (p_i * p_j))
                        w = pmi if m == "pmi" else max(0, pmi)
                    else:
                        pmi = np.log(p_ij / (p_i * p_j))
                        d = -np.log(p_ij)
                        npmi = (pmi / d) if d != 0 else 0.0
                        w = (
                            max(0, npmi)
                            if self.config.network_npmi_positive_only
                            else npmi
                        )
                else:
                    raise ValueError(f"Método desconocido: {m}")
                if w > 0:
                    edges.append((voc[i], voc[j], w))
        G = nx.Graph()
        G.add_nodes_from(voc)
        G.add_weighted_edges_from(edges)
        if G.number_of_edges() == 0:
            return G, {}
        partition = {}
        if _LOUVAIN_AVAILABLE:
            partition = community_louvain.best_partition(G, weight="weight")
            nx.set_node_attributes(G, partition, "community")
        else:
            from networkx.algorithms.community import greedy_modularity_communities

            for ci, comm in enumerate(
                greedy_modularity_communities(G, weight="weight")
            ):
                for nd in comm:
                    partition[nd] = ci
            nx.set_node_attributes(G, partition, "community")
        print(f"   Redes: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas.")
        return G, partition

    def get_community_terms(self, G, top_n=10):
        if G is None or G.number_of_nodes() == 0:
            return {}
        comms: Dict[int, List] = {}
        for nd, d in G.nodes(data=True):
            c = d.get("community", -1)
            if c != -1:
                comms.setdefault(c, []).append((nd, G.degree(nd, weight="weight")))
        return {
            c: [n for n, _ in sorted(ns, key=lambda x: x[1], reverse=True)[:top_n]]
            for c, ns in comms.items()
        }


# ══════════════════════════════════════════════════════════════════════
# ESTABILIDAD DE TÉRMINOS (bootstrap UCEs)
# ══════════════════════════════════════════════════════════════════════


class TermStabilityAnalyzer:
    def __init__(self, config: Config):
        self.config = config

    def bootstrap_terms(
        self,
        uces_estables: List[UCE],
        mat_uces: np.ndarray,
        labels_uces: np.ndarray,
        voc: List[str],
    ) -> pd.DataFrame:
        stemmer_fb = SnowballStemmer("spanish")
        stems_per = []
        sc = Counter()
        for uce in uces_estables:
            c = (
                Counter(uce.stems)
                if uce.stems
                else Counter(stemmer_fb.stem(l) for l in uce.lemmas)
            )
            stems_per.append(c)
            sc.update(c)
        sv = [s for s, cnt in sc.items() if cnt >= self.config.tsj]
        if not sv:
            return pd.DataFrame()
        si = {s: i for i, s in enumerate(sv)}
        n, ns = len(uces_estables), len(sv)
        sm = np.zeros((n, ns), dtype=int)
        for i, c in enumerate(stems_per):
            for s, f in c.items():
                if s in si:
                    sm[i, si[s]] += f
        ul = np.unique(labels_uces[labels_uces != -1])
        if len(ul) < 2:
            return pd.DataFrame()
        sel = np.zeros((ns, len(ul)))
        ta = TermAnalyzer(self.config)
        for _ in range(self.config.term_stability_n_iter):
            idx = np.random.choice(n, size=n, replace=True)
            sl, sb = labels_uces[idx], sm[idx]
            mask = sl != -1
            if not np.any(mask) or len(np.unique(sl[mask])) < 2:
                continue
            df_b = ta.compute(sb[mask], sl[mask], sv)
            if df_b.empty:
                continue
            for _, row in df_b[df_b["significativo"]].iterrows():
                s = row["termino"]
                if s not in si:
                    continue
                ci = np.where(ul == row["cluster"])[0]
                if len(ci) > 0:
                    sel[si[s], ci[0]] += 1
        sel /= self.config.term_stability_n_iter
        records = [
            {"termino": s, "cluster": int(ul[ci]), "selection_freq": sel[si2, ci]}
            for si2, s in enumerate(sv)
            for ci in range(len(ul))
            if sel[si2, ci] > 0
        ]
        df = pd.DataFrame(records)
        if not df.empty:
            print(
                f"   Bootstrap: {len(df)} pares, media={df['selection_freq'].mean():.3f}"
            )
        return df


# ══════════════════════════════════════════════════════════════════════
# SÍNTESIS LLM
# ══════════════════════════════════════════════════════════════════════


class SynthesisGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.client = (
            OpenAI(api_key=config.deepseek_api_key, base_url="https://api.deepseek.com")
            if config.use_llm_synthesis
            and config.deepseek_api_key
            and _OPENAI_AVAILABLE
            else None
        )
        self.embedder = (
            SentenceTransformer(config.embedding_model_name)
            if config.use_embeddings and _SENTENCE_TRANSFORMERS_AVAILABLE
            else None
        )

    def _formato_raices(self, fi_clase: Dict, top_n=15) -> str:
        if not fi_clase:
            return ""
        sf = {
            s: sum(sum(f.values()) for f in ls.values()) for s, ls in fi_clase.items()
        }
        lines = ["**Raíces más frecuentes (lema → formas originales):**"]
        for stem in sorted(sf, key=sf.__getitem__, reverse=True)[:top_n]:
            ld = fi_clase[stem]
            lt = {l: sum(f.values()) for l, f in ld.items()}
            top_l = sorted(lt, key=lt.__getitem__, reverse=True)[:2]
            partes = []
            for l in top_l:
                tf = ld[l]
                top_f = sorted(tf, key=tf.__getitem__, reverse=True)[:3]
                partes.append(
                    f"'{l}' → " + ", ".join(f"'{f}' ({tf[f]})" for f in top_f)
                )
            lines.append(f"- {stem}: {' | '.join(partes)}")
        return "\n".join(lines)

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        if not self.client:
            return "LLM no disponible."
        try:
            r = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un asistente experto en análisis lexicométrico.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"

    def _validate(self, sintesis: str, ucs: List[UC]) -> Tuple[bool, float]:
        if not self.embedder:
            return True, 1.0
        sample = ucs[:10]
        es = self.embedder.encode([sintesis])[0]
        eu = np.array([self.embedder.encode([u.texto])[0] for u in sample])
        sims = np.dot(eu, es) / (np.linalg.norm(eu, axis=1) * np.linalg.norm(es) + 1e-8)
        m = float(np.mean(sims))
        return m >= self.config.synthesis_similarity_threshold, m

    def generate_for_class(self, class_id, df_terms, ucs, forma_index=None):
        df_k = df_terms[df_terms["cluster"] == class_id]
        prompt = (
            f"Eres experto en análisis de discurso ALCESTE.\n"
            f"Clase {class_id} — genera síntesis ≤300 palabras.\n\n"
        )
        if forma_index and class_id in forma_index:
            prompt += self._formato_raices(forma_index[str(class_id)])
        if not df_k.empty:
            pos = (
                df_k[df_k["phi"] > 0]
                .nsmallest(10, "p_adj")[["termino", "phi"]]
                .to_dict("records")
            )
            neg = (
                df_k[df_k["phi"] < 0]
                .nsmallest(10, "p_adj")[["termino", "phi"]]
                .to_dict("records")
            )
            if pos:
                prompt += (
                    "φ+: "
                    + ", ".join(f"{t['termino']}({t['phi']:.2f})" for t in pos)
                    + "\n"
                )
            if neg:
                prompt += (
                    "φ−: "
                    + ", ".join(f"{t['termino']}({t['phi']:.2f})" for t in neg)
                    + "\n"
                )
        cls_ucs = [u for u in ucs if u.cluster_id == class_id]
        if cls_ucs:
            prompt += "\nFragmentos:\n" + "".join(
                f"  * {u.texto[:200]}\n" for u in cls_ucs[:3]
            )
        sint = self._call_llm(prompt, 500)
        v, s = self._validate(sint, cls_ucs)
        if not v:
            sint = f"[similitud baja {s:.2f}] " + sint
        return {
            "class_id": class_id,
            "sintesis_por_clase": sint,
            "timestamp": datetime.now().isoformat(),
            "validacion": {"valido": v, "similitud": s},
        }

    def generate(self, ucs, df_terms, forma_index=None):
        prompt = "Síntesis global ALCESTE (≤400 palabras) describiendo clases y diferencias.\n\n"
        for k in sorted(df_terms["cluster"].unique()):
            prompt += f"## Clase {k + 1}\n"
            if forma_index and str(int(k)) in forma_index:
                prompt += self._formato_raices(forma_index[str(int(k))]) + "\n"
            df_k = df_terms[df_terms["cluster"] == k]
            pos = (
                df_k[df_k["phi"] > 0]
                .nsmallest(10, "p_adj")[["termino", "phi"]]
                .to_dict("records")
            )
            neg = (
                df_k[df_k["phi"] < 0]
                .nsmallest(10, "p_adj")[["termino", "phi"]]
                .to_dict("records")
            )
            if pos:
                prompt += (
                    "φ+: "
                    + ", ".join(f"{t['termino']}({t['phi']:.2f})" for t in pos)
                    + "\n"
                )
            if neg:
                prompt += (
                    "φ−: "
                    + ", ".join(f"{t['termino']}({t['phi']:.2f})" for t in neg)
                    + "\n"
                )
            cls_ucs = [u for u in ucs if u.cluster_id == k]
            if cls_ucs:
                prompt += "".join(f"  * {u.texto[:200]}\n" for u in cls_ucs[:2])
            prompt += "\n"
        sint = self._call_llm(prompt, 600)
        v, s = self._validate(sint, ucs)
        if not v:
            sint = f"[similitud baja {s:.2f}] " + sint
        return {
            "sintesis_por_clase": sint,
            "timestamp": datetime.now().isoformat(),
            "validacion": {"valido": v, "similitud": s},
        }


# ══════════════════════════════════════════════════════════════════════
# MCA COMPLEMENTARIO
# ══════════════════════════════════════════════════════════════════════


class MultivariateAnalyzer:
    def __init__(self, config: Config):
        self.config = config

    def run_mca(self, ucs, matriz, voc, meta_df):
        df_bin = (pd.DataFrame(matriz, columns=voc) > 0).astype(int).astype(str)
        df_full = pd.concat(
            [df_bin, meta_df.reset_index(drop=True).astype(str)], axis=1
        )
        mca = prince.MCA(n_components=2, random_state=self.config.random_state)
        mca.fit(df_full[voc])
        rc = mca.row_coordinates(df_full[voc])
        cc = mca.column_coordinates(df_full[voc])
        eigen = mca.eigenvalues_
        exp = eigen / eigen.sum()
        print(f"   MCA: eje1={exp[0]:.2%}, eje2={exp[1]:.2%}")
        for i, uc in enumerate(ucs):
            uc.coordinates["mca_row"] = rc.iloc[i].tolist()

        return {
            "row_coords": rc.values.tolist(),
            "col_coords": cc.values.tolist(),
            "explained_inertia": exp.tolist(),
            "eigenvalues": eigen.tolist(),
            "cluster_labels": [
                uc.cluster_id if uc.cluster_id is not None else -1 for uc in ucs
            ],
            "terms": voc,  # <-- ADD THIS
            "doc_ids": [
                uc.id for uc in ucs
            ],  # <-- ADD THIS (adjust attribute name as needed)
        }

    # ══════════════════════════════════════════════════════════════════════


# T7 — OPTIMIZADOR ROBUSTO (Con caché y consciencia de Guardrails)
# ══════════════════════════════════════════════════════════════════════

try:
    from gradient_free_optimizers import PatternSearch as _PatternSearch

    _GFO_AVAILABLE = True
except ImportError:
    _GFO_AVAILABLE = False


class Optimizador:
    """
    Optimizador de hiperparámetros ALCESTE.
    Incluye caché de segmentación para acelerar las iteraciones y
    penalizaciones estrictas para asegurar la viabilidad de modelos downstream (RF+SHAP).
    """

    def __init__(
        self,
        config_base: Config,
        uc_config: UCBuilderConfig,
        corpus_raw: List[Dict],
        we_analyzer,
        subtlex_analyzer,
        total_uces: int = None,
    ):
        self.config_base = config_base
        self.corpus_raw = corpus_raw
        self.uc_config = uc_config
        self.we_analyzer = we_analyzer
        self.subtlex_analyzer = subtlex_analyzer
        self.segmentador = SegmentadorALCESTE(config_base)
        self.uces_por_doc, self.doc_metadata_map = self.segmentador.segmentar_en_uces(
            corpus_raw
        )
        start_time = time.time()
        for doc_uces in self.uces_por_doc:
            self.segmentador.lematizar_uces(doc_uces)

        self.total_uces = total_uces or sum(len(d) for d in self.uces_por_doc)
        print(
            f"   [Optimizador] Caché listo en {time.time() - start_time:.2f}s. Total UCEs: {self.total_uces}"
        )

    def save_best_params(self, params: Dict, score: float):
        """Save best parameters to a JSON file."""
        best_data = {
            "params": params,
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "total_uces": self.total_uces,
        }
        best_path = os.path.join(
            os.path.dirname(self.config_base.db_local_path), "best_params.json"
        )
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_data, f, indent=2, ensure_ascii=False)
        print(f"   [Optimizador] Saved best params to {best_path}")

    def load_best_params(self) -> Optional[Dict]:
        """Load best parameters from JSON file if it exists."""
        best_path = os.path.join(
            os.path.dirname(self.config_base.db_local_path), "best_params.json"
        )
        if not os.path.exists(best_path):
            return None
        with open(best_path, "r", encoding="utf-8") as f:
            best_data = json.load(f)
        print(
            f"   [Optimizador] Loaded best params from {best_path} (score={best_data['score']:.4f})"
        )
        return best_data["params"]

    def get_search_space(self) -> Dict[str, List]:
        f1, f2 = self.config_base.min_forms_uc
        tsj = self.config_base.tsj
        return {
            "min_forms_uc_1": np.arange(max(2, f1 - 4), f1 + 5, 1).tolist(),
            "forms_gap": np.arange(2, 8, 1).tolist(),  # ← replaces min_forms_uc_2
            "tsj": np.arange(max(2, tsj - 2), tsj + 3, 1).tolist(),
            "pseudocount": [0.0001, 0.001, 0.005],
            "min_cluster_size_cdh": [0.05, 0.08, 0.10, 0.15, 0.20],  # ← % of UCs
            "swap_iterations": [1, 2, 5],
            "use_poisson_tsj": [True, False],
            "min_r2_threshold": [0.03, 0.05, 0.08, 0.10],  # ← new
            "similarity_threshold": [0.25, 0.35, 0.45, 0.55, 0.65],
            "coref_weight": [0.0, 0.1, 0.2, 0.3, 0.4],
            "uc_window_size": [1, 2, 3, 4, 5],
        }

    def _evaluacion_rapida(self, cfg: Config, uc_cfg: UCBuilderConfig = None):
        uc_config_actual = uc_cfg or self.uc_config
        uc_builder = UCBuilder(
            self.we_analyzer, self.subtlex_analyzer, uc_config_actual, self.config_base
        )
        uc_builder.vectorizer.clear_cache()  # ← prevents retro-key contamination across trials
        double_clf = DoubleClassifier(cfg, self.segmentador, uc_builder)
        # Inject lemmatization cache (same as parent)
        cache_ref = self.uces_por_doc
        for doc_uces in cache_ref:
            for uce in doc_uces:
                uce.id = normalizar_id_uce(uce)

        original_seg = self.segmentador.segmentar_en_uces
        original_lem = self.segmentador.lematizar_uces
        self.segmentador.segmentar_en_uces = lambda x: (
            copy.deepcopy(cache_ref),
            self.doc_metadata_map,
        )
        self.segmentador.lematizar_uces = lambda x: x

        try:
            resultados = double_clf.run(self.corpus_raw)
        finally:
            self.segmentador.segmentar_en_uces = original_seg
            self.segmentador.lematizar_uces = original_lem

        return resultados

    def objetivo(self, params: Dict) -> float:
        mf1 = params["min_forms_uc_1"]
        mf2 = mf1 + params["forms_gap"]

        cfg = copy.deepcopy(self.config_base)
        cfg.min_forms_uc = [mf1, mf2]
        cfg.tsj = params["tsj"]
        cfg.pseudocount = params["pseudocount"]
        cfg.min_cluster_size_cdh = params["min_cluster_size_cdh"]
        cfg.swap_iterations = params["swap_iterations"]
        cfg.min_r2_threshold = params["min_r2_threshold"]
        cfg.optimize = cfg.use_projection = cfg.use_network_analysis = False

        # Apply UCBuilderConfig hyperparameters (Issue 8 fix)
        uc_cfg = copy.deepcopy(self.uc_config)
        uc_cfg.similarity_threshold = params.get(
            "similarity_threshold", uc_cfg.similarity_threshold
        )
        uc_cfg.coref_weight = params.get("coref_weight", uc_cfg.coref_weight)
        uc_cfg.window_size = params.get("uc_window_size", uc_cfg.window_size)

        try:
            (
                ucs_est,
                df_terms,
                voc,
                uces_por_doc,
                resultados_list,
                labels1_uce,
                labels2_uce,
                _,
                _,
                _doc_meta,
            ) = self._evaluacion_rapida(cfg, uc_cfg=uc_cfg)
        except Exception:
            return -1e6

        uces_list = ucs_est
        labels_uces = np.array([uce.cluster_id for uce in uces_list])
        n_clusters = len(np.unique(labels_uces))

        # Hard gates — return immediately if outside useful range
        if n_clusters < 2:
            return -1e6
        if n_clusters > 8:
            # Graded penalty so the optimizer can still navigate toward fewer clusters
            # -1 per extra cluster so 9 clusters = -1, 20 clusters = -11, etc.
            return -(n_clusters - 8) * 1.5

        # We're in the target range (2-8 clusters) — now optimize quality
        ari = (
            adjusted_rand_score(labels1_uce, labels2_uce)
            if labels1_uce is not None and labels2_uce is not None
            else 0.0
        )

        if df_terms is not None and hasattr(df_terms, "empty") and not df_terms.empty:
            n_sig = df_terms["significativo"].sum()
        else:
            n_sig = 0

        sig_ratio = n_sig / max(1, len(voc))
        coverage = len(ucs_est) / max(1, self.total_uces)
        min_class_count = int(np.min(np.bincount(labels_uces)))
        balance_penalty = (
            -2.0 if min_class_count < max(10, len(uces_list) * 0.08) else 0.0
        )

        score = (
            2.5 * ari
            + 0.5 * np.log1p(n_sig)  # log term is essential — penalizes marginal gains
            + 0.3 * sig_ratio  # original weight
            + 0.8 * coverage
            + balance_penalty
        )
        return float(score)

    def optimizar(self, n_trials: int = 50) -> Dict:
        if not _GFO_AVAILABLE:
            print(
                "   [!] gradient_free_optimizers no instalado. Saltando optimización."
            )
            return {}

        ss = self.get_search_space()
        print(f"\n=== Iniciando Búsqueda de Patrones ({n_trials} iteraciones) ===")
        print(f"   Optimizando para {self.total_uces} UCEs...")

        opt = _PatternSearch(ss)
        opt.search(self.objetivo, n_iter=n_trials)

        print(f"   >>> Mejor Score Alcanzado: {opt.best_score:.4f}")
        print(f"   >>> Parámetros Óptimos: {opt.best_para}")

        return opt.best_para


# ══════════════════════════════════════════════════════════════════════
# WORKFLOW ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

DEBUG = True  # Cambia a False para silenciar


def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


class WorkflowOrchestrator:
    def __init__(
        self,
        config: Config,
        uc_config: UCBuilderConfig,
        retro_config: RetrofittingConfig,
        we_analyzer,
        subtlex_analizer,
        uc_builder_config: UCBuilderConfig = None,
        progressive_segmenter=None,  # Optional[ProgressiveSegmenter]
    ):
        self.config = config
        self.uc_config = uc_config
        self.we_analyzer = we_analyzer
        self.subtlex_analyzer = subtlex_analizer
        self.db = Database(config)
        self.segmentador = SegmentadorALCESTE(config)
        self.uc_builder = UCBuilder(we_analyzer, subtlex_analizer, uc_config, config)
        self.double_clf = DoubleClassifier(
            config,
            self.segmentador,
            self.uc_builder,
            progressive_segmenter=progressive_segmenter,
        )
        self.afc = AFC(config)
        self.cah_terminos = CAHTerminos(config)
        self.meta_analyzer = MetaAnalyzer(config)
        self.synthesis = SynthesisGenerator(config)
        self.multivariate = MultivariateAnalyzer(config)
        self.network = NetworkAnalyzer(config)
        self.term_stability = TermStabilityAnalyzer(config)
        # Internal state populated during ejecutar()
        self._ppmi_matrix: Optional[csr_matrix] = None
        self._vocab: Optional[List[str]] = None
        self._df_terms: Optional[pd.DataFrame] = None
        self._uces_est_list: Optional[List[UCE]] = None
        self._uces_por_doc: Optional[List[List[UCE]]] = None
        self._retrofitted_vectors: Optional[Dict[str, np.ndarray]] = None
        self._class_centroids: Optional[Dict[int, np.ndarray]] = None

        if _RF_SHAP_AVAILABLE:  # guard import availability
            self.rf_shap = RobustRFShapAnalyzer(config)
        else:
            self.rf_shap = None

        self.ppmi_builder = SubtlexPPMIBuilder(subtlex_analizer, k=config.ppmi_k)
        self.embedding_pipeline = EmbeddingSpecializationPipeline(
            we_analyzer, retro_config or RetrofittingConfig()
        )
        self.products_generator = AnalyticProductsGenerator(
            we_analyzer, subtlex_analizer, temperature=config.analytic_temperature
        )
        # SUBTLEX DataFrame for P3 (needs raw Word column)
        self.subtlex_df = None
        if config.subtlex_df_path and os.path.exists(config.subtlex_df_path):
            self.subtlex_df = pd.read_excel(config.subtlex_df_path)
            print(f"SUBTLEX-ESP loaded for P3: {len(self.subtlex_df)} entries.")

    def _enrich_uces_with_doc_metadata(
        self, uces_por_doc: List[List[UCE]], doc_metadata_map: Dict[int, Dict]
    ) -> None:
        for doc_uces in uces_por_doc:
            for uce in doc_uces:
                shared = doc_metadata_map.get(uce.doc_id, {})
                uce.metadata.update(shared)

    def _save_all_uces(self, uces_est_list, uces_por_doc):
        stable_ids = {uce.id for uce in uces_est_list}
        all_uces = list(uces_est_list)
        for doc_uces in uces_por_doc:
            for uce in doc_uces:
                if uce.id not in stable_ids:
                    all_uces.append(uce)

        texto_map = self.db.data.get("texto_completo_por_doc_id", {})

        # Serialize manually so texto_completo_doc survives to_dict()
        serialized = []
        for uce in all_uces:
            d = uce.to_dict()
            d["texto_completo_doc"] = texto_map.get(str(uce.doc_id), "")
            serialized.append(d)

        self.db.data["uces"] = serialized
        print(
            f"   Saved {len(uces_est_list)} stable + "
            f"{len(all_uces) - len(uces_est_list)} unstable UCEs to DB."
        )

    def _poblar_phi_coefficients(
        self,
        uces_est_list: List[UCE],
        df_terms: pd.DataFrame,
    ) -> None:
        """
        Populates uce.phi_coefficients = {term_stem: phi_float}
        for every term present in that UCE and significant in its assigned cluster.
        Only positive-phi (characteristic) terms are included.
        """
        if df_terms.empty:
            return

        use_stems = self.config.stem_backend != "none"

        # Pre-build a lookup: {cluster_id: {term: phi}} for significant terms only
        phi_lookup: Dict[int, Dict[str, float]] = {}
        for cid, grp in df_terms[df_terms["significativo"]].groupby("cluster"):
            phi_lookup[int(cid)] = dict(zip(grp["termino"], grp["phi"]))

        for uce in uces_est_list:
            if uce.cluster_id is None:
                continue
            cluster_phi = phi_lookup.get(int(uce.cluster_id), {})
            if not cluster_phi:
                continue

            # Collect all term representations present in this UCE
            uce_terms: set = set(uce.stems if use_stems else uce.lemmas)
            if self.config.use_bigrams:
                src = uce.bigram_stems if use_stems else uce.bigrams
                uce_terms.update("_".join(b) for b in src)
            if self.config.use_trigrams:
                src = uce.trigram_stems if use_stems else uce.trigrams
                uce_terms.update("_".join(t) for t in src)

            uce.phi_coefficients = {
                term: round(float(phi), 4)
                for term, phi in cluster_phi.items()
                if term in uce_terms
            }

    def _rf_shap_analysis(self, uces_estables, labels_uces):
        if self.rf_shap is None:
            return None
        return self.rf_shap.analyze(uces_estables, labels_uces)

    def _build_dashboard_keys(
        self,
        uces_est_list,
        labels_uces,
        df_terms,
        forma_index,
        uces_por_doc,  # ← NEW parameter
    ):
        """
        Writes every key the dashboard reads that ejecutar() never saved.
        Called at the end of ejecutar(), before db._save().
        """
        from datetime import datetime

        from scipy.stats import chi2_contingency

        origen_index = {}
        for doc_uces in uces_por_doc:
            for uce in doc_uces:
                origen = uce.metadata.get("origen", f"doc{uce.doc_id}")
                if origen not in origen_index:
                    origen_index[origen] = []
                origen_index[origen].append(uce.id)
        self.db.data["origen_index"] = origen_index

        # ── FIX 1 · stem_summary ──────────────────────────────────────────────────
        # Dashboard:  stem_summary[str(class_id)][stem][lemma] → int
        # Pipeline:   forma_index[int(class_id)][stem][lemma][forma] → int
        stem_summary = {}
        for class_id, stems in forma_index.items():
            key = str(int(class_id))
            stem_summary[key] = {}
            for stem, lemmas in stems.items():
                stem_summary[key][stem] = {}
                for lemma, formas in lemmas.items():
                    if isinstance(formas, dict):
                        stem_summary[key][stem][lemma] = int(sum(formas.values()))
                    else:
                        stem_summary[key][stem][lemma] = int(formas)
        self.db.data["stem_summary"] = stem_summary

        # ── FIX 3 · uce_phi ───────────────────────────────────────────────────────
        # Dashboard:  uce_phi_dict[uce_id] → float phi score shown next to each UCE
        # Pipeline:   never written; all scores default to 0.0
        # Proxy: mean phi of significant positive terms in the UCE's class
        # DESPUÉS:
        uce_phi = []
        if not df_terms.empty:
            # Lookup: {cluster_id: {term: phi}} — solo términos significativos positivos
            phi_lookup: Dict[int, Dict[str, float]] = {}
            for cid in df_terms["cluster"].unique():
                df_c = df_terms[
                    (df_terms["cluster"] == cid)
                    & (df_terms["significativo"])
                    & (df_terms["phi"] > 0)
                ]
                phi_lookup[int(cid)] = dict(zip(df_c["termino"], df_c["phi"]))

            use_stems = self.config.stem_backend != "none"
            for uce, label in zip(uces_est_list, labels_uces):
                cluster_phi = phi_lookup.get(int(label), {})
                # Términos presentes en la UCE
                uce_terms: set = set(uce.stems if use_stems else uce.lemmas)
                if self.config.use_bigrams:
                    src = uce.bigram_stems if use_stems else uce.bigrams
                    uce_terms.update("_".join(b) for b in src)
                if self.config.use_trigrams:
                    src = uce.trigram_stems if use_stems else uce.trigrams
                    uce_terms.update("_".join(t) for t in src)
                present_phis = [
                    phi for term, phi in cluster_phi.items() if term in uce_terms
                ]
                phi_score = (
                    round(float(np.mean(present_phis)), 4) if present_phis else 0.0
                )
                uce_phi.append({"uce_id": uce.id, "phi_score": phi_score})
        self.db.data["uce_phi"] = uce_phi

        # ── FIX 4 · words_per_cluster ─────────────────────────────────────────────
        # Dashboard tries both str and int keys; write str keys (most robust)
        words_per_cluster = {}
        for uce, label in zip(uces_est_list, labels_uces):
            k = str(int(label))
            words_per_cluster[k] = words_per_cluster.get(k, 0) + len(uce.lemmas)
        self.db.data["words_per_cluster"] = words_per_cluster

        # ── FIX 5 · pos_by_cluster ────────────────────────────────────────────────
        # Dashboard:  pos_by_cluster[str(class_id)][pos_tag] → count
        # Pipeline:   UCEs have pos_tags list but it's never aggregated
        pos_by_cluster = {}
        for uce, label in zip(uces_est_list, labels_uces):
            k = str(int(label))
            if k not in pos_by_cluster:
                pos_by_cluster[k] = {}
            for tag in uce.pos_tags:
                pos_by_cluster[k][tag] = pos_by_cluster[k].get(tag, 0) + 1
        self.db.data["pos_by_cluster"] = pos_by_cluster

        # ── FIX 6 · clustering_method at top level ────────────────────────────────
        # Dashboard:  data.get("clustering_method", "unknown")
        # Pipeline:   only inside data['config']['clustering_method']
        self.db.data["clustering_method"] = (
            "cdh" if self.config.use_cdh else self.config.clustering_method
        )

        # ── FIX 7 · sintesis_por_clase ────────────────────────────────────────────
        # Dashboard:  sintesis_por_clase[str(class_id)] → {sintesis, validacion, ...}
        # Pipeline:   data["sintesis_por_clase"] is a flat list [{class_id, sintesis, ...}]
        sintesis_por_clase = {}
        for s in self.db.data.get("sintesis_por_clase", []):
            cid = s.get("class_id")
            if cid is not None:
                sintesis_por_clase[str(int(cid))] = s
        self.db.data["sintesis_por_clase"] = sintesis_por_clase

        # ── FIX 8 · metadata_residuals ────────────────────────────────────────────
        # Dashboard heatmap expects:
        #   metadata_residuals[var]['residuals'][category_label][str(class_id)] → float
        # These are Pearson standardized residuals: (O − E) / √E
        metadata_residuals = {}
        meta_rows = []
        for uce, lbl in zip(uces_est_list, labels_uces):
            if isinstance(uce.metadata, dict) and uce.metadata:
                row = dict(**uce.metadata, cluster=int(lbl))
                meta_rows.append(row)
        if meta_rows:
            meta_df = pd.DataFrame(meta_rows)
            for col in meta_df.columns:
                if col == "cluster":
                    continue
                is_cat = (
                    meta_df[col].dtype == "object"
                    or str(meta_df[col].dtype) == "category"
                    or pd.api.types.is_string_dtype(meta_df[col])
                )
                if not is_cat:
                    continue
                tabla = pd.crosstab(meta_df[col].astype(str), meta_df["cluster"])
                if tabla.empty or tabla.shape[0] < 2 or tabla.shape[1] < 2:
                    continue
                chi2_val, _, _, expected = chi2_contingency(tabla, correction=False)
                observed = tabla.values.astype(float)
                residuals = (observed - expected) / np.sqrt(np.maximum(expected, 1e-9))
                res_dict = {
                    str(cat): {
                        str(c): round(float(residuals[i, j]), 3)
                        for j, c in enumerate(tabla.columns)
                    }
                    for i, cat in enumerate(tabla.index)
                }
                metadata_residuals[str(col)] = {
                    "chi2": round(float(chi2_val), 4),
                    "residuals": res_dict,
                }

        self.db.data["metadata_residuals"] = metadata_residuals
        # ── condensed_tree_plot_data desde CDH ───────────────────────────────────
        cdh_tree = self.db.data.get("cdh_tree_umbral1", {})
        if cdh_tree:
            bar_centers, bar_bottoms, bar_tops, bar_widths = [], [], [], []

            # def _traverse(node, depth=0):
            #     if not node:
            #         return
            #     n = node.get('n_ucs', 0)
            #     if n > 0:
            #         bar_centers.append(depth)
            #         bar_bottoms.append(0)
            #         bar_tops.append(float(n) / max(cdh_tree.get('n_ucs', 1), 1))
            #         bar_widths.append(n)
            #     for child in node.get('children', []):
            #         _traverse(child, depth + 1)

            # _traverse(cdh_tree)
            if bar_centers:
                self.db.data["condensed_tree_plot_data"] = {
                    "bar_centers": bar_centers,
                    "bar_bottoms": bar_bottoms,
                    "bar_tops": bar_tops,
                    "bar_widths": bar_widths,
                }

        # ── FIX 10 · corpus_name / n_entrevistas / fecha_analisis ────────────────
        # These fields exist in Config but were not present before; backfill safely.
        cfg = self.db.data.get("config", {})
        if not cfg.get("corpus_name"):
            cfg["corpus_name"] = "Corpus ALCESTE"
        if not cfg.get("n_entrevistas"):
            unique_docs = len(
                {uce.doc_id for uce in uces_est_list if uce.doc_id is not None}
            )
            cfg["n_entrevistas"] = unique_docs
        if not cfg.get("fecha_analisis"):
            cfg["fecha_analisis"] = datetime.now().strftime("%d %B %Y")
        self.db.data["config"] = cfg

        # ── Grammatical summary by class ──────────────────────────────────────
        # Single-pass accumulation (no outer/inner loop bug)
        gram_summary_by_class = {}
        n_uces_por_clase = defaultdict(int)

        for uce, label in zip(uces_est_list, labels_uces):
            k = str(int(label))
            n_uces_por_clase[k] += 1

            if k not in gram_summary_by_class:
                gram_summary_by_class[k] = {
                    # Conteos absolutos
                    "n_negaciones": 0,
                    "n_verbos": 0,
                    "n_pronombres_exp": 0,
                    "n_prodrop": 0,
                    "n_marcadores": 0,
                    "n_frames": 0,
                    "n_cuantificadores": 0,
                    "n_adverbios": 0,
                    "n_insubordinaciones": 0,
                    "n_rarezas": 0,
                    "n_subj": 0,
                    # Sumas para medias
                    "sum_ttr": 0.0,
                    "sum_guiraud": 0.0,
                    "sum_diversidad_semantica": 0.0,
                    "sum_topic_shift": 0.0,
                    "sum_profundidad": 0.0,
                    "sum_recursividad": 0.0,
                    "sum_distancia_dependencia": 0.0,
                    "sum_ratio_subordinacion": 0.0,
                    "sum_branching_ratio": 0.0,
                    "sum_mean_zipf": 0.0,
                    "sum_pct_oov": 0.0,
                    "sum_oral_ratio": 0.0,
                    "sum_academic_ratio": 0.0,
                    "sum_domain_specific_ratio": 0.0,
                    "sum_mean_surprisal": 0.0,
                    "voice_Act": 0,
                    "voice_Pass": 0,
                    "voice_PassRefl": 0,
                    "voice_Impersonal": 0,
                    "voice_Media": 0,
                    "registros": [],
                }

            s = gram_summary_by_class[k]
            s["n_negaciones"] += len(getattr(uce, "negaciones", []))
            s["n_verbos"] += len(getattr(uce, "verbos", []))
            s["n_pronombres_exp"] += sum(
                1
                for p in getattr(uce, "pronombres", [])
                if p.get("tipo") == "EXPLICITO"
            )
            s["n_prodrop"] += sum(
                1 for p in getattr(uce, "pronombres", []) if p.get("tipo") == "NULO"
            )
            s["n_marcadores"] += len(getattr(uce, "marcadores_discursivos", []))
            s["n_frames"] += len(getattr(uce, "predicate_frames", []))
            s["n_cuantificadores"] += len(getattr(uce, "cuantificadores", []))
            s["n_adverbios"] += len(getattr(uce, "adverbios", []))
            s["n_insubordinaciones"] += len(getattr(uce, "insubordinaciones", []))
            s["n_rarezas"] += len(getattr(uce, "rarezas", []))
            s["n_subj"] += sum(
                1 for v in getattr(uce, "verbos", []) if v.get("modo") == "Subj"
            )

            s["sum_ttr"] += uce.metricas_lexicas.get("ttr", 0.0)
            s["sum_guiraud"] += uce.metricas_lexicas.get("guiraud", 0.0)
            s["sum_diversidad_semantica"] += getattr(uce, "diversidad_semantica", 0.0)
            s["sum_topic_shift"] += getattr(uce, "topic_shift_prev", 0.0)

            cs = getattr(uce, "complejidad_sintactica", {})
            s["sum_profundidad"] += cs.get("profundidad_maxima", 0)
            s["sum_recursividad"] += cs.get("recursividad", 0)
            s["sum_distancia_dependencia"] += cs.get("distancia_dependencia_media", 0.0)
            s["sum_ratio_subordinacion"] += cs.get("ratio_subordinacion", 0.0)
            s["sum_branching_ratio"] += cs.get("branching_ratio", 0.5)

            s["sum_mean_zipf"] += uce.metricas_lexicas.get("mean_zipf", 0.0)
            s["sum_pct_oov"] += uce.metricas_lexicas.get("pct_oov", 0.0)
            s["sum_oral_ratio"] += uce.metricas_lexicas.get("oral_ratio", 0.0)
            s["sum_academic_ratio"] += uce.metricas_lexicas.get("academic_ratio", 0.0)
            s["sum_domain_specific_ratio"] += uce.metricas_lexicas.get(
                "domain_specific_ratio", 0.0
            )
            s["sum_mean_surprisal"] += uce.metricas_lexicas.get(
                "mean_surprisal_content", 0.0
            )

            for v in getattr(uce, "verbos", []):
                voz = v.get("voz", "Act")
                if voz == "Act":
                    s["voice_Act"] += 1
                elif voz == "Pass":
                    s["voice_Pass"] += 1
                elif voz == "PassRefl":
                    s["voice_PassRefl"] += 1
                elif voz == "Impersonal":
                    s["voice_Impersonal"] += 1
                elif voz == "Media":
                    s["voice_Media"] += 1

            if getattr(uce, "registro", None):
                s["registros"].append(uce.registro)

        # ── Convertir sumas en medias ──────────────────────────────────────────
        for k, s in gram_summary_by_class.items():
            n = n_uces_por_clase[k]
            if n > 0:
                s["mean_ttr"] = s.pop("sum_ttr") / n
                s["mean_guiraud"] = s.pop("sum_guiraud") / n
                s["mean_diversidad_semantica"] = s.pop("sum_diversidad_semantica") / n
                s["mean_topic_shift"] = s.pop("sum_topic_shift") / n
                s["mean_profundidad"] = s.pop("sum_profundidad") / n
                s["mean_recursividad"] = s.pop("sum_recursividad") / n
                s["mean_distancia_dependencia"] = s.pop("sum_distancia_dependencia") / n
                s["mean_ratio_subordinacion"] = s.pop("sum_ratio_subordinacion") / n
                s["mean_branching_ratio"] = s.pop("sum_branching_ratio") / n
                s["mean_zipf"] = s.pop("sum_mean_zipf") / n
                s["mean_pct_oov"] = s.pop("sum_pct_oov") / n
                s["mean_oral_ratio"] = s.pop("sum_oral_ratio") / n
                s["mean_academic_ratio"] = s.pop("sum_academic_ratio") / n
                s["mean_domain_specific_ratio"] = s.pop("sum_domain_specific_ratio") / n
                s["mean_surprisal"] = s.pop("sum_mean_surprisal") / n
            else:
                for key in list(s.keys()):
                    if key.startswith("sum_"):
                        del s[key]

        self.db.data["grammatical_summary_by_class"] = gram_summary_by_class

    def _agrupar_por_doc(self, uces: List) -> List[List]:
        """Agrupa UCEs por doc_id preservando el orden."""
        grupos = defaultdict(list)
        for uce in uces:
            grupos[uce.doc_id].append(uce)
        return list(grupos.values())

    def ejecutar(
        self,
        corpus_raw: Dict[str, Any],
        global_corpus=None,
        grammatical_pipeline: Optional[PipelineGramatical] = None,
        grammatical_dashboard_path: str = "",
        lex_analyzer=None,  # GlobalLexicalAnalyzer instance, optional
    ) -> None:
        print("=== Iniciando Pipeline ALCESTE v5 ===")

        # ── 1. Optimización de hiperparámetros (El núcleo robusto) ─────────────
        cached_uces_por_doc = None
        cached_doc_metadata_map = None

        best_params_path = os.path.join("data/", "best_params.json")
        if os.path.exists(best_params_path):
            print(
                "   Found existing best_params.json. Loading parameters and skipping optimizer."
            )
            try:
                with open(best_params_path, "r", encoding="utf-8") as f:
                    best_data = json.load(f)
                best_params = best_data["params"]
                print(
                    f"   Loaded params: {best_params} (score={best_data['score']:.4f})"
                )
                # Apply loaded parameters to config
                mf1 = best_params.get("min_forms_uc_1", self.config.min_forms_uc[0])
                gap = best_params.get("forms_gap", 4)
                self.config.min_forms_uc = [mf1, mf1 + gap]
                self.config.tsj = best_params.get("tsj", self.config.tsj)
                self.config.pseudocount = best_params.get(
                    "pseudocount", self.config.pseudocount
                )
                self.config.min_cluster_size_cdh = best_params.get(
                    "min_cluster_size_cdh", self.config.min_cluster_size_cdh
                )
                self.config.swap_iterations = best_params.get(
                    "swap_iterations", self.config.swap_iterations
                )
                self.config.min_r2_threshold = best_params.get(
                    "min_r2_threshold", self.config.min_r2_threshold
                )
                # (Note: use_poisson_tsj is not used in the pipeline; ignore)
            except Exception as e:
                print(
                    f"   [WARNING] Could not load best_params.json: {e}. Running optimizer if enabled."
                )
                best_params = None
        else:
            print(
                "   [WARNING] Could not load best_params.json. Running optimizer if enabled."
            )
            best_params = None

        if best_params is None and self.config.optimize:
            # Run optimizer
            self.config.use_embeddings = False

            optimizador = Optimizador(
                self.config,
                self.uc_config,
                corpus_raw,
                self.we_analyzer,
                self.subtlex_analyzer,
            )
            mejores_params = optimizador.optimizar(n_trials=self.config.optimize_trials)
            if mejores_params:
                # Compute score for the best parameters (reuse objetivo)
                best_score = optimizador.objetivo(mejores_params)
                optimizador.save_best_params(mejores_params, best_score)
                # Apply to config
                mf1 = mejores_params.get("min_forms_uc_1", self.config.min_forms_uc[0])
                gap = mejores_params.get("forms_gap", 4)
                self.config.min_forms_uc = [mf1, mf1 + gap]
                self.config.tsj = mejores_params.get("tsj", self.config.tsj)
                self.config.pseudocount = mejores_params.get(
                    "pseudocount", self.config.pseudocount
                )
                self.config.min_cluster_size_cdh = mejores_params.get(
                    "min_cluster_size_cdh", self.config.min_cluster_size_cdh
                )
                self.config.swap_iterations = mejores_params.get(
                    "swap_iterations", self.config.swap_iterations
                )
                self.config.min_r2_threshold = mejores_params.get(
                    "min_r2_threshold", self.config.min_r2_threshold
                )
            else:
                print("   [!] Optimización no viable. Usando config base.")
            cached_uces_por_doc = optimizador.uces_por_doc
            cached_doc_metadata_map = optimizador.doc_metadata_map

        # Set use_embeddings back to True after possible optimizer run
        self.config.use_embeddings = True
        # ── Main run — skip re-lemmatization if we have the cache ──────────
        if cached_uces_por_doc is not None:
            print("   Reutilizando cache de lematización del optimizador...")
            # Patch the double classifier's segmenter
            original_seg = self.double_clf.segmentador.segmentar_en_uces
            original_lem = self.double_clf.segmentador.lematizar_uces
            self.double_clf.segmentador.segmentar_en_uces = lambda x: (
                copy.deepcopy(cached_uces_por_doc),
                cached_doc_metadata_map,
            )
            self.double_clf.segmentador.lematizar_uces = lambda x: x

        print("=== Pass 1: CHD with generic vectors ===")
        (
            uces_est_p1,
            _,
            voc_p1,
            uces_por_doc_p1,
            resultados_p1,
            _,
            _,
            _,
            _,
            doc_metadata_map,
        ) = self.double_clf.run(
            corpus_raw,
            ppmi_builder=self.ppmi_builder,
            cached_uces_por_doc=cached_uces_por_doc,  # from optimizer, or None
            cached_doc_metadata_map=cached_doc_metadata_map,
        )

        # ── Retrofitting on pass-1 output ─────────────────────────────────────────
        retro_vecs_p1: Optional[Dict[str, np.ndarray]] = None
        if resultados_p1 and voc_p1:
            _all_ucs_p1 = resultados_p1[0]["ucs"]
            _labels_p1 = resultados_p1[0]["labels"]
            _builder_p1 = MatrizBuilder(self.config)
            _mat_p1 = _builder_p1.construir_matriz(_all_ucs_p1, voc_p1)
            _df_terms_p1 = TermAnalyzer(self.config).compute(
                _mat_p1, _labels_p1, voc_p1
            )
            if not _df_terms_p1.empty and _df_terms_p1["significativo"].any():
                _binary_p1 = _builder_p1.construir_matriz_dispersa(_all_ucs_p1, voc_p1)
                _ppmi_p1 = self.ppmi_builder.build(_binary_p1, voc_p1)
                retro_vecs_p1 = self.embedding_pipeline.run(
                    _ppmi_p1, voc_p1, _df_terms_p1
                )
                self.uc_builder.vectorizer.clear_cache()  # invalidate before pass 2
                print(f"   Pass-1 retrofitting complete: {len(retro_vecs_p1)} vectors.")
            else:
                print("   Pass-1 had no significant terms; skipping retrofitting.")

        # ── Pass 2 (or fallback to pass-1 results) ───────────────────────────────
        if retro_vecs_p1 is not None:
            print("=== Pass 2: CHD with retrofitted vectors ===")
            (
                uces_est_list,
                _,
                voc_uc,
                uces_por_doc,
                resultados_por_umbral,
                labels1_uce,
                labels2_uce,
                _,
                _,
                doc_metadata_map,
            ) = self.double_clf.run(
                corpus_raw,
                retrofitted_vectors=retro_vecs_p1,
                ppmi_builder=self.ppmi_builder,
                cached_uces_por_doc=uces_por_doc_p1,  # always reuse pass-1 UCEs
                cached_doc_metadata_map=doc_metadata_map,
            )

            # ── Persist pairwise stability matrices ───────────────────────────────
            if hasattr(self.double_clf, "last_pairwise_stability"):
                self.db.data["pairwise_stability"] = (
                    self.double_clf.last_pairwise_stability
                )

        else:
            print("   Single-pass only (no retrofitting available).")
            uces_est_list = uces_est_p1
            voc_uc = voc_p1
            uces_por_doc = uces_por_doc_p1
            resultados_por_umbral = resultados_p1

        self._enrich_uces_with_doc_metadata(uces_por_doc, doc_metadata_map)
        self.db.data["section_registry"] = getattr(
            self.double_clf.segmentador, "section_registry", {}
        )
        self.db.data["texto_completo_por_origen"] = {
            origen_key: doc_data.get("texto_completo_txt", "")
            for origen_key, doc_data in corpus_raw.items()
        }
        self.db.data["texto_completo_por_doc_id"] = {
            str(doc_data.get("indice_orden", i)): doc_data.get("texto_completo_txt", "")
            for i, (origen_key, doc_data) in enumerate(corpus_raw.items())
        }
        if not uces_est_list:
            print(
                "   [FATAL] Cero UCs estables. Tu corpus es un desastre o los parámetros son muy restrictivos. Terminando."
            )
            return

        # ── Phase 8: Class centroids + P1–P5 from Pass-2 retrofitted vectors ──
        #     (was: full PPMI rebuild + retrofitting — redundant with Step 2)
        self._vocab = voc_uc
        self._uces_est_list = uces_est_list
        self._uces_por_doc = uces_por_doc
        self._retrofitted_vectors = retro_vecs_p1
        self._ppmi_matrix = None  # rebuilt lazily by _rebuild_ppmi_matrix if needed

        if retro_vecs_p1 is not None and voc_uc:
            uce_labels_for_centroids = np.array(
                [uce.cluster_id for uce in uces_est_list if uce.cluster_id is not None]
            )
            if len(np.unique(uce_labels_for_centroids)) >= 2:
                # Build df_terms from the same stable UCEs (needed by compute_class_centroids)
                _builder_ph8 = MatrizBuilder(self.config)
                _mat_ph8 = _builder_ph8.construir_matriz(uces_est_list, voc_uc)
                _labels_ph8 = np.array([uce.cluster_id for uce in uces_est_list])
                _df_ph8 = TermAnalyzer(self.config).compute(
                    _mat_ph8, _labels_ph8, voc_uc
                )
                self._class_centroids = self.embedding_pipeline.compute_class_centroids(
                    df_terms=_df_ph8,
                    retrofitted_vectors=retro_vecs_p1,
                )
                print(
                    f"   [Phase 8] Centroids from Pass-2 retrofitted vectors: "
                    f"{sorted(self._class_centroids.keys())}"
                )

                if self._class_centroids and len(self._class_centroids) >= 2:
                    print("   [Phase 8] Generating analytic products P1–P5...")
                    # P1
                    p1_distances = self.products_generator.p1_inter_class_distances(
                        self._class_centroids
                    )
                    # P2
                    p2_membership, p2_entropy = (
                        self.products_generator.p2_soft_membership(
                            uces=uces_est_list or [],
                            class_centroids=self._class_centroids,
                            retrofitted_vectors=retro_vecs_p1,
                        )
                    )
                    # P3
                    p3_latent: Dict[int, List[Dict]] = {}
                    if self.subtlex_df is not None:
                        corpus_vocab_set = set(voc_uc)
                        p3_latent = self.products_generator.p3_latent_vocabulary(
                            class_centroids=self._class_centroids,
                            corpus_vocab=corpus_vocab_set,
                            subtlex_df=self.subtlex_df,
                            top_n=20,
                        )
                    # P4
                    p4_trajectories = self.products_generator.p4_narrative_trajectories(
                        uces_by_doc=uces_por_doc or [],
                        class_centroids=self._class_centroids,
                        retrofitted_vectors=retro_vecs_p1,
                        membership_df=p2_membership,
                        entropy_df=p2_entropy,
                    )
                    # P5
                    p5_export = self.products_generator.p5_centroid_export(
                        class_centroids=self._class_centroids,
                        df_terms=_df_ph8,
                    )
                    # Persist
                    self._save_new_products(
                        db=self.db,
                        p1_distances=p1_distances,
                        p2_membership=p2_membership,
                        p2_entropy=p2_entropy,
                        p3_latent=p3_latent,
                        p4_trajectories=p4_trajectories,
                        p5_export=p5_export,
                    )
                    print("   [Phase 8] P1–P5 complete.")
            else:
                print("   [Phase 8] Fewer than 2 classes — skipping centroids + P1–P5.")
        else:
            print(
                "   [Phase 8] No retrofitted vectors available — skipping centroids + P1–P5."
            )
            self._class_centroids = {}

        # Store document metadata
        for doc_id, meta in doc_metadata_map.items():
            self.db.save_doc_metadata(doc_id, meta)

        grammar_index = {}  # (doc_id, start_char) -> dict de UCE gramatical
        if grammatical_dashboard_path and os.path.exists(grammatical_dashboard_path):
            with open(grammatical_dashboard_path, "r", encoding="utf-8") as f:
                grammar_full = json.load(f)
            for doc_id, doc_info in grammar_full.get("documents", {}).items():
                for uce_gram in doc_info.get("uces", []):
                    # USE THE ID, NOT THE OFFSETS.
                    # Fallback to start_char only if you have legacy data without IDs, but fix your data.
                    uce_id = uce_gram.get("id", uce_gram.get("uce_id"))
                    if uce_id:
                        key = (str(doc_id), uce_id)
                        grammar_index[key] = uce_gram
            print(f"   [Gramática] Indexadas {len(grammar_index)} UCEs.")

        # ── NUEVO: Enriquecimiento gramatical ──────────────────────────
        if grammatical_pipeline is not None:
            print("   Enriqueciendo UCEs con análisis gramatical...")
            try:
                # STANDALONE FUNCTION CALL. Pass the pipeline as a dependency.
                grammatical_pipeline.procesar_desde_uces(
                    uces_por_doc=uces_por_doc,
                    global_corpus=global_corpus,
                    lex_analyzer=lex_analyzer,
                    corpus_raw=corpus_raw,
                )
                print("   Enriquecimiento gramatical completado.")
            except Exception as e:
                print(f"   [WARNING] Enriquecimiento gramatical falló: {e}")
                traceback.print_exc()  # <--- THIS WILL SHOW YOU THE EXACT LINE IF IT FAILS AGAIN
        # ── NOW extract stable UCEs (already enriched, same object references) ──
        if not uces_est_list:
            return

        labels_uces = np.array([uce.cluster_id for uce in uces_est_list])

        self.segmentador.generar_embeddings_uces(uces_est_list)
        for uce, label in zip(uces_est_list, labels_uces):
            uce.cluster_id = int(label)

        builder = MatrizBuilder(self.config)
        voc = builder.construir_vocabulario(uces_est_list)
        mat_uces = builder.construir_matriz(uces_est_list, voc)
        mat_clases, class_ids = builder.agregar_por_clase(mat_uces, labels_uces, voc)
        df_terms = TermAnalyzer(self.config).compute(mat_uces, labels_uces, voc)

        forma_index_raw = self.segmentador.construir_forma_index(
            uces_est_list, labels_uces
        )
        forma_index = {str(int(k)): v for k, v in forma_index_raw.items()}

        resultados = {}
        afc_result = {}

        if self.config.use_projection:
            afc_result = self.afc.fit(mat_clases, class_ids, voc)
            if afc_result:
                afc_result["voc"] = voc
                resultados["proyeccion"] = afc_result

                # ── Task 2: Map AFC coordinates directly onto each UCE ────────────
                col_coords = np.array(afc_result["col_coords"])  # principal coords
                col_sums_vec = mat_clases.sum(axis=0).astype(float)
                col_sums_vec = np.where(col_sums_vec == 0, 1.0, col_sums_vec)
                col_mass = col_sums_vec / col_sums_vec.sum()  # column marginals

                row_sums = mat_uces.sum(axis=1, keepdims=True).astype(float)
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                row_profile = mat_uces.astype(float) / row_sums  # (n_uce, n_terms)

                # Supplementary-row projection: (profile - column_mass) / sqrt(column_mass) @ V / singular_values
                sv = np.array(afc_result["singular_values"])
                # col_coords = Vt.T * s / sqrt(c)  →  Vt.T = col_coords * sqrt(c) / s
                sqrt_c = np.sqrt(col_mass)
                V_unit = (
                    col_coords * sqrt_c[:, np.newaxis]
                ) / sv  # (n_terms, n_factors)
                diff = row_profile - col_mass  # deviation from marginal
                uc_afc = (diff / sqrt_c) @ V_unit  # (n_uce, n_factors)

                for uce, coord in zip(uces_est_list, uc_afc):
                    uce.coordinates["afc_row"] = coord.tolist()

                cah_global = self.cah_terminos.run_global(afc_result, voc)
                if cah_global:
                    resultados["cah_terminos_global"] = cah_global

        if self.config.use_cah_per_class:
            per_class_cah = {}
            for class_id in sorted({int(l) for l in labels_uces}):
                res = self.cah_terminos.run_per_class(
                    class_id,
                    uces_est_list,
                    df_terms,
                    voc,
                    top_n=self.config.cah_per_class_top_terms,
                )
                if res:
                    per_class_cah[str(class_id)] = res
            self.db.data["cah_por_clase"] = per_class_cah

        # Random Forest + SHAP
        # ── Task 3: Hook up orphaned analyzers ───────────────────────────────────
        if self.config.use_network_analysis:
            G, partition = self.network.build_cooccurrence_graph(voc, mat_uces)
            if G is not None and G.number_of_nodes() > 0:
                # NetworkX edge data must be serialized: (u, v, {weight: float})
                edges_serial = [
                    (
                        u,
                        v,
                        {
                            k: float(val)
                            if isinstance(val, (np.floating, float))
                            else val
                            for k, val in d.items()
                        },
                    )
                    for u, v, d in G.edges(data=True)
                ]
                resultados["network"] = {
                    "nodes": list(G.nodes()),
                    "edges": edges_serial,
                    "partition": partition,
                }

        if self.config.analyze_metadata:
            meta_df = self.meta_analyzer.analyze(uces_est_list, labels_uces)
            if not meta_df.empty:
                resultados["metadata_analysis"] = meta_df.to_dict(orient="records")
                if self.config.use_multivariate_analysis:
                    mca_res = self.multivariate.run_mca(
                        uces_est_list,  # ← fix NameError
                        mat_uces,
                        voc,
                        meta_df,
                    )
                    if mca_res:
                        resultados["multivariate"] = mca_res
                        self.db.data["multivariate"] = mca_res  # ← persist to DB

        if self.config.use_term_stability:
            stab_df = self.term_stability.bootstrap_terms(
                uces_est_list, mat_uces, labels_uces, voc
            )
            if not stab_df.empty:
                self.db.save_term_stability(stab_df)
                resultados["term_stability"] = stab_df.to_dict(
                    orient="records"
                )  # ← add to resultados

        # ── Task 3: Persist CDH trees ─────────────────────────────────────────────
        if resultados_por_umbral:
            if resultados_por_umbral[0].get("tree"):
                self.db.data["cdh_tree_umbral1"] = resultados_por_umbral[0][
                    "tree"
                ].to_dict()
            if len(resultados_por_umbral) > 1 and resultados_por_umbral[1].get("tree"):
                self.db.data["cdh_tree_umbral2"] = resultados_por_umbral[1][
                    "tree"
                ].to_dict()

        # Random Forest + SHAP

        if self.config.use_rf_shap:
            rf_result = self._rf_shap_analysis(uces_est_list, labels_uces)
            if rf_result and "error" not in rf_result:
                resultados["shap_analysis"] = rf_result
                self.db.data["shap_analysis"] = rf_result  # ← persist to DB

        # Llamada BERTOPIC basado en uces_est_list (no implementado aún)

        # Guardado a disco
        self.db.data["vocabulario"] = voc
        self.db.data["forma_index"] = forma_index
        self._poblar_phi_coefficients(uces_est_list, df_terms)
        self.db.data["config"] = dataclasses.asdict(self.config)
        self.db.data.update(resultados)
        self.db.save_terminos(df_terms)

        for uce in uces_est_list:
            uce._predicate_frames_serialized = [
                f.to_dict() if hasattr(f, "to_dict") else f
                for f in getattr(uce, "predicate_frames", [])
            ]

        self._save_all_uces(uces_est_list, uces_por_doc)  # ← replaces save_uces

        # Build every key the dashboard expects
        self._build_dashboard_keys(
            uces_est_list=uces_est_list,
            labels_uces=labels_uces,
            df_terms=df_terms,
            #            voc=voc,
            forma_index=forma_index,
            uces_por_doc=uces_por_doc,
        )

        self.db._save()
        print("\\n=== Pipeline completado. Dashboard keys escritas. ===")

    # ── Helper: rebuild PPMI matrix from stable UCEs ─────────────────────────
    def _rebuild_ppmi_matrix(self) -> Optional[csr_matrix]:
        """
        Rebuilds the PPMI-weighted matrix from stable UCEs.
        Necessary because the patched matrix was used inside CHD but not stored.
        """
        if not self._uces_est_list or not self._vocab:
            return None
        try:
            builder = MatrizBuilder(self.config)
            binary = builder.construir_matriz_dispersa(self._uces_est_list, self._vocab)
            ppmi = self.ppmi_builder.build(binary, self._vocab)
            logger.info(
                "Rebuilt PPMI matrix: %d×%d, nnz=%d",
                ppmi.shape[0],
                ppmi.shape[1],
                ppmi.nnz,
            )
            return ppmi
        except Exception as e:
            logger.error("Failed to rebuild PPMI matrix: %s", e)
            return None

    # ── Helper: reconstruct df_terms from DB ─────────────────────────────────
    def _reconstruct_df_terms(self, db) -> Optional[pd.DataFrame]:
        """Recover TermAnalyzer output from the database."""
        terminos = db.data.get("terminos", [])
        if terminos:
            return pd.DataFrame(terminos)
        return None

    def _recover_stable_uces(self, db) -> List[UCE]:
        """Reconstruct stable UCE objects from DB serialization."""
        uces_data = db.data.get("uces", [])
        stable = []
        for d in uces_data:
            if d.get("is_stable", False):
                try:
                    stable.append(UCE.from_dict(d))
                except Exception:
                    pass
        return stable

    def _recover_uces_por_doc(self, db) -> List[List[UCE]]:
        """
        Reconstruct uces_por_doc (all UCEs, grouped by doc_id) from DB.
        Needed for P4 (narrative trajectories require sequential UCE order).
        """
        uces_data = db.data.get("uces", [])
        by_doc: Dict[int, List[UCE]] = defaultdict(list)
        for d in uces_data:
            try:
                uce = UCE.from_dict(d)
                doc_id = uce.doc_id if uce.doc_id is not None else -1
                by_doc[doc_id].append(uce)
            except Exception:
                pass
        # Sort each doc's UCEs by local_idx to restore sequential order
        result = []
        for doc_id in sorted(by_doc.keys()):
            doc_uces = sorted(
                by_doc[doc_id],
                key=lambda u: u.local_idx if u.local_idx is not None else 0,
            )
            result.append(doc_uces)
        return result

    # ── Helper: persist new products ─────────────────────────────────────────
    def _save_new_products(
        self,
        db,
        p1_distances: pd.DataFrame,
        p2_membership: pd.DataFrame,
        p2_entropy: pd.DataFrame,
        p3_latent: Dict,
        p4_trajectories: Dict[str, pd.DataFrame],
        p5_export: Dict,
    ) -> None:
        """Write P1–P5 outputs to the database under new keys."""
        # P1
        db.data["inter_class_semantic_distances"] = p1_distances.to_dict()

        # P2 — store as records for JSON serializability
        db.data["soft_class_membership"] = p2_membership.reset_index().to_dict(
            orient="records"
        )
        db.data["uce_entropy"] = (
            p2_entropy.reset_index()
            .assign(is_liminal=p2_entropy["is_liminal"].astype(int))
            .to_dict(orient="records")
        )

        # P3
        db.data["latent_vocabulary_profiles"] = {
            str(k): v for k, v in p3_latent.items()
        }

        # P4 — store per-document trajectories
        db.data["narrative_trajectories"] = {
            doc_id: traj_df.to_dict(orient="records")
            for doc_id, traj_df in p4_trajectories.items()
        }

        # P5
        # Convert centroid ndarray lists to plain Python lists
        export_serializable = dict(p5_export)
        export_serializable["classes"] = {
            cid: {
                **info,
                "centroid": [float(x) for x in info["centroid"]],
            }
            for cid, info in p5_export["classes"].items()
        }
        db.data["centroid_export"] = export_serializable

        # Store top-N retrofitted vectors (full export would be too large)
        # Keep only terms present in at least one class's top characteristic terms
        sig_terms: set = set()
        df_terms = self._df_terms
        if df_terms is not None and not df_terms.empty:
            sig_terms = set(df_terms[df_terms["significativo"]]["termino"].tolist())
        if self._retrofitted_vectors:
            db.data["retrofitted_vectors_significant"] = {
                term: [float(x) for x in vec]
                for term, vec in self._retrofitted_vectors.items()
                if term in sig_terms
            }


if __name__ == "__main__":
    os.environ["HF_TOKEN"] = "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"

    # ── Configs  ────────────────────────────────────────────────
    alceste_config = Config(
        spacy_model="es_core_news_lg",
        use_bigrams=True,
        stem_backend="snowball",
        min_uce_words=3,
        min_forms_uc=[10, 14],
        tsj=3,
        use_cdh=True,
        pseudocount=0.1,
        use_cah_per_class=True,
        cah_per_class_top_terms=20,
        use_projection=True,
        analyze_metadata=True,
        use_network_analysis=True,
        use_term_stability=True,
        optimize=True,
        optimize_trials=100,
        random_state=42,
        use_rf_shap=True,
        rf_n_estimators=100,
        rf_max_depth=5,
        rf_outlier_method="iqr",
        rf_cat_encoding="onehot",
        glm_method="chi2",
        use_multivariate_analysis=True,
        multivariate_metadata=["Edad_Cat", "Sexo", "Ocupacion_Cat", "Procedencia_Cat"],
        ppmi_k=1.0,  # set to 5.0 for small corpora (< 50k tokens)
        analytic_temperature=0.1,
        use_llm_synthesis=False,
        classification_mode="all",
        coref_context_units_tight=2,
        coref_context_units_loose=4,
        hdbscan_uce_min_cluster_size=5,
        hdbscan_uce_min_cluster_size_loose=3,
        hdbscan_uce_metric="euclidean",
    )

    gram_config = GramConfig(
        min_tokens_por_uce=30,
        max_tokens_por_uce=200,
        adverb_classifier_dir="./adverb_model",
        use_coref=True,
        use_subtlex=True,
        subtlex_path=r"D:\SUBTLEX-ESP.xlsx",
        use_wordnet_quantifiers=True,
        stanza_use_gpu=False,
        word_embeddings_path=r"D:\cc.es.300.bin",
        use_gensim_embeddings=True,
    )

    retro_config = RetrofittingConfig(
        alpha=0.8,  # strong anchor to general Spanish semantics
        beta=0.5,  # moderate pull from Reinert class neighbors
        gamma=0.3,  # weaker pull from PPMI-SVD (corpus-specific)
        n_iter=10,
        svd_components=100,
        chi2_threshold=3.84,  # α=0.05 for 1 degree of freedom (χ²=3.84)
    )
    cfg = UCBuilderConfig()

    # Entrenar clasificador de adverbios si no existe
    clf_path = os.path.join(
        gram_config.adverb_classifier_dir, "logistic_classifier.joblib"
    )
    if not os.path.exists(clf_path):
        train_adverb_classifier(
            gram_config.adverb_classifier_dir,
            gram_config.sentence_embedder_model,
        )

    # ── Inicializar pipelines ─────────────────────────────────────────
    gram_pipeline = PipelineGramatical(gram_config)
    subtlex = NLPProvider.get_subtlex(gram_config)
    we_analyzer = NLPProvider.get_word_vectors(gram_config)

    global_corpus = GlobalCorpus()
    lex_analyzer = GlobalLexicalAnalyzer(
        subtlex_analyzer=subtlex,
        cooc_window=4,
        cooc_min_count=3,
        we_analyzer=we_analyzer,
    )

    # ── ProgressiveSegmenter (Method B — coref-driven UC construction) ──
    # Only instantiate when the mode actually needs it (Stanza load is expensive)
    _coref_modes = {
        "coref_only",
        "wc_coref",
        "coref_emb",
        "all",
        "both",
        "sim_only",
        "wc_sim",
        "sim_emb",
    }  # aliases too
    progressive_seg = None
    if alceste_config.classification_mode in _coref_modes:
        progressive_seg = ProgressiveSegmenter(
            model_name=alceste_config.embedding_model_name,
            stanza_lang="es",
            spacy_model=alceste_config.spacy_model,
            debug_coref=True,  # silence chain-level prints in production
            similarity_threshold=0.6,
            max_depth=3,
        )
        # Eagerly verify Stanza loads; fail fast rather than mid-pipeline
        if progressive_seg.get_stanza() is None:
            print("[WARN] Stanza failed to load — Method B will be skipped.")
            progressive_seg = None

    # ── PHASE 1: ALCESTE — segmentación + clasificación ──────────────
    # WorkflowOrchestrator.ejecutar() ahora acepta gram_pipeline
    orchestrator = WorkflowOrchestrator(
        config=alceste_config,
        uc_config=cfg,
        retro_config=retro_config,
        we_analyzer=we_analyzer,
        subtlex_analizer=subtlex,
        progressive_segmenter=progressive_seg,
    )

    # Necesitamos uces_por_doc ANTES de que ejecutar() termine.
    # La forma más limpia: exponer uces_por_doc como atributo del orchestrator.
    # Por ahora, pasamos gram_pipeline directamente para que ejecutar()
    # llame a procesar_desde_uces() en el momento correcto.
    orchestrator.ejecutar(
        corpus_raw=uwu,  # <--- Changed from uwu
        grammatical_pipeline=gram_pipeline,
        global_corpus=global_corpus,
        lex_analyzer=lex_analyzer,
        grammatical_dashboard_path="global_dashboard.json",
    )
    # ── PHASE 2: Clustering de predicados (post-ALCESTE) ─────────────
    global_predicate_result = None

    if (
        gram_pipeline is not None
        and global_corpus is not None
        and gram_pipeline.predicate_analyzer is not None
    ):
        # This is where cluster_all actually belongs!
        global_predicate_result = gram_pipeline.predicate_analyzer.cluster_all(
            global_corpus
        )

    # ── PHASE 3: Red semántica global ────────────────────────────────
    G = lex_analyzer.semantic_network()
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    lex_analyzer.export_network_gexf("red_semantica.gexf")

    # ── PHASE 4: Guardar todo ─────────────────────────────────────────
    # Fixed: Removed the overlapping positional argument
    global_corpus.export_json(
        "global_dashboard.json",
        predicate_result=global_predicate_result,
        subtlex_analyzer=subtlex,
        lex_analyzer=lex_analyzer,  # ← now wired through
    )
