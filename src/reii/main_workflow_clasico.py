# %%viztracer
# main_pipeline.py
from __future__ import annotations

import copy
import dataclasses
import gc
import json
import os
import re
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
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency, f_oneway, fisher_exact
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from statsmodels.stats.multitest import multipletests

from reii.config import (
    BEST_PARAMS_PATH,
    CORPUS_NAME,
    DEEPSEEK_BASE_URL,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL,
    OUTPUT_DASHBOARD,
    OUTPUT_LEXICAL_JSON,
    OUTPUT_LEXICAL_XLSX,
    OUTPUT_NETWORK_GEXF,
    SPACY_MODEL,
    WORKFLOW_DB_PATH,
)
from reii.config import (
    DATA_DIR as REII_DATA_DIR,
)
from reii.gram.gramatical_analyzer import (
    UCE,
    GlobalCorpus,
    GlobalLexicalAnalyzer,
    NLPProvider,
    PipelineGramatical,
    train_adverb_classifier,
)
from reii.gram.gramatical_analyzer import (
    Config as GramConfig,
)

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

import glob
import json
import os

import pandas as pd

gc.collect()


# Función a prueba de balas para limpiar extensiones dobles y simples
# Convierte "doc.txt.json" o "doc.txt" a una llave maestra limpia: "doc"
def obtener_llave_maestra(nombre):
    return str(nombre).replace(".txt.json", "").replace(".json", "").replace(".txt", "")


# 0. Cargamos tu metadata usando la llave maestra
metadata_csv = os.path.join(REII_DATA_DIR, "Refined_Database.csv")
df_meta = pd.read_csv(metadata_csv, sep=";")
meta_dict = {}

for _, row in df_meta.iterrows():
    llave = obtener_llave_maestra(row["Documento Fuente"])
    meta_dict[llave] = row[
        ["Edad_Cat", "Sexo", "Dependientes_Cat", "Ocupacion_Cat", "Procedencia_Cat"]
    ].to_dict()

# 1. Ubicamos los JSONs y aseguramos el orden alfabético
json_input_dir = os.path.join(REII_DATA_DIR, "txt_outputs", "tmp")
archivos_json = sorted(glob.glob(os.path.join(json_input_dir, "*.json")))
dir_txt = os.path.join(REII_DATA_DIR, "txt_outputs")

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
    spacy_model: str = SPACY_MODEL
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
    llm_model: str = LLM_MODEL
    synthesis_similarity_threshold: float = 0.6

    use_embeddings: bool = True  # Flag maestra
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"

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
    db_local_path: str = WORKFLOW_DB_PATH
    random_state: int = 42
    corpus_name: str = CORPUS_NAME
    n_entrevistas: Optional[int] = None  # None → auto-counted from unique doc_ids
    fecha_analisis: Optional[str] = None  # None → auto-filled at runtime


# ══════════════════════════════════════════════════════════════════════
# (Skipping boilerplate UCE, UC, CDHNode, Database, Segmentador for brevity - assume unchanged from base code except POS filtering)
# ══════════════════════════════════════════════════════════════════════


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
    import re

    return bool(re.match(r"^[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$", id_str, re.I))


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
        self._upsert("uces", uces)

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
        self.section_registry = {}  # Build directly in the loop

        for doc_idx, (origen_key, doc_data) in enumerate(corpus_raw.items()):
            metadata_global = doc_data.get("metadata", {}).copy()
            shared_meta = {
                **metadata_global,
                "origen": origen_key,
                "doc_idx": doc_idx,
                "indice_orden": doc_data.get("indice_orden"),
            }
            self.doc_metadata_map[doc_idx] = shared_meta

            full_text = doc_data.get("texto_completo_txt", "")
            if not full_text:
                continue
            section_map: List[Tuple[int, int, str]] = []
            search_pos = 0
            for seg in doc_data.get("texto_segmentos", []):
                seg_text = seg.get("texto", "")
                sec_name = seg.get("otros_datos", {}).get("seccion", "")
                if seg_text and sec_name:
                    found = full_text.find(seg_text, search_pos)
                    if found >= 0:
                        section_map.append((found, found + len(seg_text), sec_name))
                        search_pos = found + len(seg_text)

            doc_spacy = self.nlp(full_text)
            spans = self._segmentar_reinert_spacy(doc_spacy)

            local_idx = 0
            doc_uces = []
            section_registry: Dict[str, int] = {}  # ← Reset PER DOCUMENT
            local_section_counter = 0

            for span in spans:
                if len(span.text.split()) < self.config.min_uce_words:
                    continue

                seccion = ""
                for s_start, s_end, s_name in section_map:
                    if s_start <= span.start_char < s_end:
                        seccion = s_name
                        break

                key = seccion  # ← Just the section name
                if key not in section_registry:
                    section_registry[key] = local_section_counter
                    local_section_counter += 1
                section_id = section_registry[key]

                uce_id = f"{doc_idx}_{section_id}_{local_idx}"

                # Register globally for downstream uses
                self.section_registry[f"{doc_idx}_{section_id}"] = {
                    "origen": origen_key,
                    "seccion": seccion,
                }

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
        embs = model.encode([u.texto for u in uces], show_progress_bar=True)
        for uce, emb in zip(uces, embs):
            uce.embedding = emb.tolist()
        return uces


# ══════════════════════════════════════════════════════════════════════
# MATRIZ BUILDER (ALCESTE: STRICT BINARY & DOC-FREQ TSJ)
# ══════════════════════════════════════════════════════════════════════


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


class ClasificadorDescendente:
    def __init__(self, config: Config):
        self.config = config
        self._leaf_counter = 0
        self._rng = np.random.default_rng(config.random_state)
        self._min_cluster_size: int = 5  # resolved in clasificar()

    def clasificar(self, mat_sparse) -> Tuple[np.ndarray, CDHNode]:
        self._leaf_counter = 0
        n_total = mat_sparse.shape[0]

        # Resolve percentage → absolute count, with a hard floor of 5
        self._min_cluster_size = max(5, int(n_total * self.config.min_cluster_size_cdh))
        print(
            f"   [CDH] n={n_total} · min_cluster_size={self._min_cluster_size} "
            f"({self.config.min_cluster_size_cdh:.0%} of corpus)"
        )

        indices = np.arange(n_total)
        arbol = self._partition(indices, mat_sparse, depth=0)
        labels = np.full(n_total, -1, dtype=int)
        self._fill_labels(arbol, labels)
        return labels, arbol

    def _primer_factor(self, sub_mat) -> np.ndarray:
        """
        Calcula las coordenadas del primer eje factorial mediante
        el algoritmo de promedios recíprocos (reciprocal averaging).
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

        # Evitar divisiones por cero
        row_sums[row_sums == 0] = 1
        col_sums[col_sums == 0] = 1

        # Inicialización aleatoria
        rng = np.random.default_rng(self.config.random_state)
        x = rng.standard_normal(n)
        # Centrado inicial (eliminar componente trivial)
        x -= np.average(x, weights=row_sums)
        x /= max(np.linalg.norm(x), 1e-12)

        for _ in range(100):
            # y = N^T x
            y = mat.T.dot(x)
            # z = y / col_sums
            z = y / col_sums
            # w = N z
            w = mat.dot(z)
            # x_new = w / row_sums
            x_new = w / row_sums
            # Centrado: quitar la media ponderada
            x_new -= np.average(x_new, weights=row_sums)
            norm = np.linalg.norm(x_new)
            if norm < 1e-12:
                break
            x_new /= norm
            # Convergencia considerando cambio de signo
            flip = np.sign(np.dot(x_new, x))  # +1 or -1
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
        sub_mat,  # csr_matrix (n × m) — presencia/ausencia
        labels: np.ndarray,
        max_iter: int,
    ) -> np.ndarray:
        """
        Mueve cada UC a la clase contraria si el Δχ² es positivo.
        Actualización incremental: solo recalcula los términos no-cero de la fila.

        Complejidad por pasada: O(n × nnz_promedio)
        """
        from scipy.sparse import issparse

        binary = (sub_mat > 0).astype(np.float32)
        n = binary.shape[0]
        labels = labels.copy()

        # Conteos por clase y totales
        cc = [
            np.asarray(binary[labels == 0].sum(axis=0)).flatten().reshape(-1),
            np.asarray(binary[labels == 1].sum(axis=0)).flatten().reshape(-1),
        ]
        ct = [int((labels == 0).sum()), int((labels == 1).sum())]
        global_freq = cc[0] + cc[1]

        for _ in range(max_iter):
            moved = False
            for i in range(n):
                old = int(labels[i])
                new = 1 - old
                if ct[old] <= 1:  # no vaciar una clase
                    continue

                row = (
                    np.asarray(binary[i].todense()).flatten()
                    if issparse(binary)
                    else binary[i]
                )
                nonzero_idx = np.where(row > 0)[0]
                if len(nonzero_idx) == 0:
                    continue

                # Δχ² incremental sobre los términos no-cero de la fila
                delta = 0.0
                total_ucs = n
                new_ct_old = ct[old] - 1
                new_ct_new = ct[new] + 1

                for t in nonzero_idx:
                    gf = global_freq[t]
                    if gf == 0:
                        continue
                    # Contribuciones actuales
                    E_old = ct[old] * gf / total_ucs
                    E_new = ct[new] * gf / total_ucs
                    chi_now = (cc[old][t] - E_old) ** 2 / max(E_old, 1e-9) + (
                        cc[new][t] - E_new
                    ) ** 2 / max(E_new, 1e-9)
                    # Contribuciones después del movimiento
                    E_old2 = new_ct_old * gf / total_ucs
                    E_new2 = new_ct_new * gf / total_ucs
                    chi_after = (cc[old][t] - 1 - E_old2) ** 2 / max(E_old2, 1e-9) + (
                        cc[new][t] + 1 - E_new2
                    ) ** 2 / max(E_new2, 1e-9)
                    delta += chi_after - chi_now

                if delta > 0:
                    labels[i] = new
                    cc[old] -= row
                    cc[new] += row
                    ct[old] -= 1
                    ct[new] += 1
                    moved = True

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
    ) -> CDHNode:
        n = len(indices)
        # Criterios de parada
        # Use the resolved absolute value, not the percentage
        if n < self._min_cluster_size or depth >= self.config.max_depth_cdh:
            node = CDHNode(depth=depth, n_ucs=n, is_leaf=True, indices=indices.tolist())
            node.label = self._next_leaf()
            return node
        sub_mat = mat_sparse[indices]

        # 1. Primer factor
        coord = self._primer_factor(sub_mat)

        # 2. Corte óptimo
        var_obs, labels = self._corte_optimo(coord)

        # 3. Significancia (antes del intercambio para evitar sesgo)
        if not self._es_significativo(coord, var_obs):
            node = CDHNode(depth=depth, n_ucs=n, is_leaf=True, indices=indices.tolist())
            node.label = self._next_leaf()
            return node

        # 4. Refinamiento por intercambio
        if self.config.swap_iterations > 0:
            labels = self._intercambio(
                sub_mat, labels, max_iter=self.config.swap_iterations
            )

        # Verificar que el intercambio no colapsó una clase
        if len(np.unique(labels)) < 2:
            node = CDHNode(depth=depth, n_ucs=n, is_leaf=True, indices=indices.tolist())
            node.label = self._next_leaf()
            return node

        idx0 = indices[labels == 0]
        idx1 = indices[labels == 1]

        # 5. Recursión
        child0 = self._partition(idx0, mat_sparse, depth + 1)
        child1 = self._partition(idx1, mat_sparse, depth + 1)

        return CDHNode(
            depth=depth,
            n_ucs=n,
            is_leaf=False,
            indices=indices.tolist(),
            children=[child0, child1],
        )

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


# ══════════════════════════════════════════════════════════════════════
# T3 — DOBLE CLASIFICACIÓN (refactorizado para CDH)
# ══════════════════════════════════════════════════════════════════════


class DoubleClassifier:
    def __init__(self, config: Config, segmentador: SegmentadorALCESTE):
        self.config = config
        self.segmentador = segmentador

    def _uce_to_indices(self, uce_to_uc, ucs):
        uid_to_idx = {uc.id: i for i, uc in enumerate(ucs)}
        return {
            uid: uid_to_idx[uc_id]
            for uid, uc_id in uce_to_uc.items()
            if uc_id in uid_to_idx
        }

    def _clasificar_umbral(self, uces_por_doc, min_forms):
        suffix = f"__mf{min_forms}"  # ← double-underscore won't collide with
        #   the single _ separating the int triple

        uces_por_doc_local = copy.deepcopy(uces_por_doc)
        for doc_uces in uces_por_doc_local:
            for uce in doc_uces:
                uce.id = normalizar_id_uce(uce)
        for doc_uces in uces_por_doc_local:
            for uce in doc_uces:
                uce.id = f"{uce.id}{suffix}"
        # then use uces_por_doc_local everywhere below in this method
        ucs, uce_to_uc = self.segmentador.construir_ucs(
            uces_por_doc_local, min_forms, self.segmentador.doc_metadata_map
        )
        # Strip suffix back to bare int-triple id
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

        if self.config.use_cdh:
            mat_sparse = builder.construir_matriz_dispersa(ucs, voc)
            cdh = ClasificadorDescendente(self.config)
            labels, tree = cdh.clasificar(mat_sparse)
        else:
            mat_dense = builder.construir_matriz(ucs, voc)
            clust = _FallbackClusterizador(self.config)
            labels, Z, persistence = clust.clustering(mat_dense, ucs)
            tree = None

        for i, uc in enumerate(ucs):
            uc.cluster_id = int(labels[i]) if labels[i] != -1 else None

        real = np.unique(labels[labels >= 0])
        print(
            f"   [umbral={min_forms}] UCs={len(ucs)}, "
            f"clusters={len(real)}, ruido={np.sum(labels == -1)}"
        )
        debug_print(f"  Built {len(ucs)} UCs with {len(voc)} terms")

        return {
            "min_forms": min_forms,
            "ucs": ucs,
            "labels": labels,
            "voc": voc,
            "uce_to_uc": uce_to_uc,
            "tree": tree,
        }

    def run(self, corpus_raw):
        print("Segmentando en UCEs...")
        uces_por_doc, doc_metadata_map = self.segmentador.segmentar_en_uces(corpus_raw)
        for doc_uces in uces_por_doc:
            self.segmentador.lematizar_uces(doc_uces)
        for doc_uces in uces_por_doc:
            for uce in doc_uces:
                uce.id = normalizar_id_uce(uce)

        # DIAGNOSTIC: check what segmentar_en_uces actually returned
        all_ids = [uce.id for doc in uces_por_doc for uce in doc]
        id_counts = Counter(all_ids)
        dupes = {k: v for k, v in id_counts.items() if v > 1}
        print(f"   [run() AFTER segmentar] total={len(all_ids)}, dupes={len(dupes)}")
        if dupes:
            for uid, c in list(dupes.items())[:3]:
                print(f"      '{uid}' × {c}  ← ALREADY POLLUTED BEFORE deepcopy")

        # Each threshold gets its own deepcopy so suffix mutation is isolated
        if _JOBLIB_AVAILABLE:
            resultados = Parallel(n_jobs=1)(
                delayed(self._clasificar_umbral)(copy.deepcopy(uces_por_doc), mf)
                for mf in self.config.min_forms_uc
            )
        else:
            resultados = [
                self._clasificar_umbral(copy.deepcopy(uces_por_doc), mf)
                for mf in self.config.min_forms_uc
            ]
        resultados = [r for r in resultados if r is not None]

        # --------------------------------------------------------------
        # Less than 2 thresholds → only one classification is possible.
        # Return a 10‑tuple (the last element is doc_metadata_map).
        # --------------------------------------------------------------
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

            # Map UCs back to UCEs directly
            uces_est_list = []
            for doc_uces in uces_por_doc:
                for uce in doc_uces:
                    uid = uce.id
                    if uid in res["uce_to_uc"]:
                        uc_idx = res["uce_to_uc"][uid]
                        c = int(res["labels"][uc_idx])
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

        uce_to_uc1 = self._uce_to_indices(res1["uce_to_uc"], res1["ucs"])
        uce_to_uc2 = self._uce_to_indices(res2["uce_to_uc"], res2["ucs"])
        uces_comunes = set(uce_to_uc1) & set(uce_to_uc2)

        if not uces_comunes:
            return (
                [],
                pd.DataFrame(),
                [],
                uces_por_doc,
                resultados,
                None,
                None,
                None,
                None,
                doc_metadata_map,
            )

        labels1, labels2 = res1["labels"], res2["labels"]
        real1 = np.unique(labels1[labels1 >= 0])
        real2 = np.unique(labels2[labels2 >= 0])

        # T3.2: cruce estrictamente a nivel de UCE
        r1i = {int(v): i for i, v in enumerate(real1)}
        r2i = {int(v): i for i, v in enumerate(real2)}
        sim = np.zeros((len(real1), len(real2)), dtype=float)
        for uid in uces_comunes:
            l1 = int(labels1[uce_to_uc1[uid]])
            l2 = int(labels2[uce_to_uc2[uid]])
            if l1 >= 0 and l2 >= 0:
                sim[r1i[l1], r2i[l2]] += 1

        print(f"   umbral1 clusters: {len(real1)}")
        print(f"   umbral2 clusters: {len(real2)}")
        print(f"   Matriz solapamiento:\n{sim}")

        row_ind, col_ind = linear_sum_assignment(-sim)
        mapping = {int(real1[r]): int(real2[c]) for r, c in zip(row_ind, col_ind)}
        print(f"   Mapeo húngaro: {len(mapping)}")

        # T3.3: determinar UCEs estables
        uce_to_cluster1: Dict[str, int] = {}
        for uid, uc_idx in uce_to_uc1.items():
            uce_to_cluster1[uid] = int(labels1[uc_idx])

        uce_to_cluster2: Dict[str, int] = {}
        for uid, uc_idx in uce_to_uc2.items():
            uce_to_cluster2[uid] = int(labels2[uc_idx])

        # Apply Hungarian mapping and mark stable UCEs directly
        stable_uces: Dict[str, int] = {}  # uce_id -> cluster_id
        for uid in uces_comunes:
            c1 = uce_to_cluster1[uid]
            c2 = uce_to_cluster2[uid]
            if c1 >= 0 and c2 >= 0 and c1 in mapping and mapping[c1] == c2:
                stable_uces[uid] = c1  # assign cluster from threshold 1

        # Back-propagate to UCE objects
        uces_est_list = []
        for doc_uces in uces_por_doc:
            for uce in doc_uces:
                if uce.id in stable_uces:
                    uce.cluster_id = stable_uces[uce.id]
                    uce.is_stable = True
                    uces_est_list.append(uce)
                else:
                    uce.cluster_id = None
                    uce.is_stable = False

        unicos = np.unique([u.cluster_id for u in uces_est_list])
        print(f"   UCEs estables: {len(uces_est_list)} | clases: {len(unicos)}")

        if len(unicos) < 2:
            print("   ⚠️  Solo 1 clase estable.")

        if not uces_est_list:
            return (
                [],
                pd.DataFrame(),
                [],
                uces_por_doc,
                resultados,
                None,
                None,
                None,
                None,
                doc_metadata_map,
            )

        # ARI a nivel UCE
        uce_comunes_ari = sorted(set(uce_to_uc1) & set(uce_to_uc2))
        labels1_uc = np.array([labels1[uce_to_uc1[u]] for u in uce_comunes_ari])
        labels2_uc = np.array([labels2[uce_to_uc2[u]] for u in uce_comunes_ari])

        # Final return: 10‑tuple (last is doc_metadata_map)
        return (
            uces_est_list,
            pd.DataFrame(),
            res1["voc"],
            uces_por_doc,
            resultados,
            labels1_uc,
            labels2_uc,
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
                y = np.asarray(y)[clean_mask]  # Fix 3: plain ndarray

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
        mean_abs_shap = np.mean(per_class_abs_mean, axis=0)

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
            OpenAI(api_key=config.deepseek_api_key, base_url=DEEPSEEK_BASE_URL)
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

import copy
import time

import numpy as np


class Optimizador:
    """
    Optimizador de hiperparámetros ALCESTE.
    Incluye caché de segmentación para acelerar las iteraciones y
    penalizaciones estrictas para asegurar la viabilidad de modelos downstream (RF+SHAP).
    """

    def __init__(
        self, config_base: Config, corpus_raw: List[Dict], total_uces: int = None
    ):
        self.config_base = config_base
        self.corpus_raw = corpus_raw
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
        }

    def _evaluacion_rapida(self, cfg: Config) -> Tuple:
        double_clf = DoubleClassifier(cfg, self.segmentador)
        original_seg = self.segmentador.segmentar_en_uces
        original_lem = self.segmentador.lematizar_uces

        # Each call to segmentar_en_uces gets a FRESH deepcopy from the cache
        # This means _clasificar_umbral gets independent copies per threshold
        cache_ref = self.uces_por_doc
        # --- NUEVO: normalizar IDs de todas las UCEs en caché ---
        for doc_uces in cache_ref:
            for uce in doc_uces:
                uce.id = normalizar_id_uce(uce)
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
            ) = self._evaluacion_rapida(cfg)
        except Exception:
            return -1e6

        if not ucs_est or len(ucs_est) < 5:
            return -1e6

        uces_list = uces_est
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

        n_sig = df_terms["significativo"].sum() if not df_terms.empty else 0
        sig_ratio = n_sig / max(1, len(voc))
        coverage = len(ucs_est) / max(1, self.total_uces)

        # Penalize tiny classes
        min_class_count = np.min(np.bincount(labels_uces))
        balance_penalty = 0.0
        if min_class_count < max(10, len(uces_list) * 0.08):
            balance_penalty = -2.0

        score = (
            2.5 * ari
            + 0.5 * np.log1p(n_sig)
            + 0.3 * sig_ratio
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


import traceback


class WorkflowOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config)
        self.segmentador = SegmentadorALCESTE(config)
        self.double_clf = DoubleClassifier(config, self.segmentador)
        self.afc = AFC(config)
        self.cah_terminos = CAHTerminos(config)
        self.meta_analyzer = MetaAnalyzer(config)
        self.synthesis = SynthesisGenerator(config)
        self.multivariate = MultivariateAnalyzer(config)
        self.network = NetworkAnalyzer(config)
        self.term_stability = TermStabilityAnalyzer(config)
        if _RF_SHAP_AVAILABLE:  # guard import availability
            self.rf_shap = RobustRFShapAnalyzer(config)
        else:
            self.rf_shap = None

    def _enrich_uces_with_doc_metadata(
        self, uces_por_doc: List[List[UCE]], doc_metadata_map: Dict[int, Dict]
    ) -> None:
        for doc_uces in uces_por_doc:
            for uce in doc_uces:
                shared = doc_metadata_map.get(uce.doc_id, {})
                uce.metadata.update(shared)

    def _save_all_uces(self, uces_est_list, uces_por_doc):
        """
        Saves AtLL UCEs to the DB:
          • Stable ones  → already in uces_est_list with cluster_id set.
          • Unstable ones → cluster_id stays None, uc_id untouched.

        Call this instead of  self.db.save_uces(uces_est_list)  in ejecutar().
        """
        stable_ids = {uce.id for uce in uces_est_list}

        all_uces = list(uces_est_list)  # stable first (already processed)

        for doc_uces in uces_por_doc:
            for uce in doc_uces:
                if uce.id not in stable_ids:
                    # Unstable: cluster_id and uc_id are left as they are
                    # (cluster_id=None, uc_id=None — set by the pipeline).
                    all_uces.append(uce)

        self.db.save_uces(all_uces)
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
                    & (df_terms["significativo"] == True)
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
            cfg["corpus_name"] = CORPUS_NAME
        if not cfg.get("n_entrevistas"):
            unique_docs = len(
                {uce.doc_id for uce in uces_est_list if uce.doc_id is not None}
            )
            cfg["n_entrevistas"] = unique_docs
        if not cfg.get("fecha_analisis"):
            cfg["fecha_analisis"] = datetime.now().strftime("%d %B %Y")
        self.db.data["config"] = cfg

        gram_summary_by_class = {}
        for uce, label in zip(uces_est_list, labels_uces):
            k = str(int(label))
            if k not in gram_summary_by_class:
                gram_summary_by_class = {}
                n_uces_por_clase = defaultdict(int)  # para calcular medias después

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
                            "n_subj": 0,  # verbos en subjuntivo
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
                            # Sumas para métricas SUBTLEX
                            "sum_mean_zipf": 0.0,
                            "sum_pct_oov": 0.0,
                            "sum_oral_ratio": 0.0,
                            "sum_academic_ratio": 0.0,
                            "sum_domain_specific_ratio": 0.0,
                            "sum_mean_surprisal": 0.0,
                            # Distribución de voces (conteos)
                            "voice_Act": 0,
                            "voice_Pass": 0,
                            "voice_PassRefl": 0,
                            "voice_Impersonal": 0,
                            "voice_Media": 0,
                            # Lista de registros (para modo)
                            "registros": [],
                        }

                    s = gram_summary_by_class[k]

                    # Conteos absolutos
                    s["n_negaciones"] += len(getattr(uce, "negaciones", []))
                    s["n_verbos"] += len(getattr(uce, "verbos", []))
                    s["n_pronombres_exp"] += sum(
                        1
                        for p in getattr(uce, "pronombres", [])
                        if p.get("tipo") == "EXPLICITO"
                    )
                    s["n_prodrop"] += sum(
                        1
                        for p in getattr(uce, "pronombres", [])
                        if p.get("tipo") == "NULO"
                    )
                    s["n_marcadores"] += len(getattr(uce, "marcadores_discursivos", []))
                    s["n_frames"] += len(getattr(uce, "predicate_frames", []))
                    s["n_cuantificadores"] += len(getattr(uce, "cuantificadores", []))
                    s["n_adverbios"] += len(getattr(uce, "adverbios", []))
                    s["n_insubordinaciones"] += len(
                        getattr(uce, "insubordinaciones", [])
                    )
                    s["n_rarezas"] += len(getattr(uce, "rarezas", []))
                    s["n_subj"] += sum(
                        1
                        for v in getattr(uce, "verbos", [])
                        if v.get("modo") in {"Sub", "Subj"}
                    )

                    # Métricas léxicas y sintácticas (sumas)
                    s["sum_ttr"] += uce.metricas_lexicas.get("ttr", 0.0)
                    s["sum_guiraud"] += uce.metricas_lexicas.get("guiraud", 0.0)
                    s["sum_diversidad_semantica"] += getattr(
                        uce, "diversidad_semantica", 0.0
                    )
                    s["sum_topic_shift"] += getattr(uce, "topic_shift_prev", 0.0)

                    cs = getattr(uce, "complejidad_sintactica", {})
                    s["sum_profundidad"] += cs.get("profundidad_maxima", 0)
                    s["sum_recursividad"] += cs.get("recursividad", 0)
                    s["sum_distancia_dependencia"] += cs.get(
                        "distancia_dependencia_media", 0.0
                    )
                    s["sum_ratio_subordinacion"] += cs.get("ratio_subordinacion", 0.0)
                    s["sum_branching_ratio"] += cs.get("branching_ratio", 0.5)

                    # Métricas SUBTLEX (si existen)
                    s["sum_mean_zipf"] += uce.metricas_lexicas.get("mean_zipf", 0.0)
                    s["sum_pct_oov"] += uce.metricas_lexicas.get("pct_oov", 0.0)
                    s["sum_oral_ratio"] += uce.metricas_lexicas.get("oral_ratio", 0.0)
                    s["sum_academic_ratio"] += uce.metricas_lexicas.get(
                        "academic_ratio", 0.0
                    )
                    s["sum_domain_specific_ratio"] += uce.metricas_lexicas.get(
                        "domain_specific_ratio", 0.0
                    )
                    s["sum_mean_surprisal"] += uce.metricas_lexicas.get(
                        "mean_surprisal_content", 0.0
                    )

                    # Voces verbales
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

                    # Registro
                    if getattr(uce, "registro", None):
                        s["registros"].append(uce.registro)

                # ── Convertir sumas en medias ──────────────────────────────────────────────
                for k, s in gram_summary_by_class.items():
                    n = n_uces_por_clase[k]
                    if n > 0:
                        s["mean_ttr"] = s.pop("sum_ttr") / n
                        s["mean_guiraud"] = s.pop("sum_guiraud") / n
                        s["mean_diversidad_semantica"] = (
                            s.pop("sum_diversidad_semantica") / n
                        )
                        s["mean_topic_shift"] = s.pop("sum_topic_shift") / n
                        s["mean_profundidad"] = s.pop("sum_profundidad") / n
                        s["mean_recursividad"] = s.pop("sum_recursividad") / n
                        s["mean_distancia_dependencia"] = (
                            s.pop("sum_distancia_dependencia") / n
                        )
                        s["mean_ratio_subordinacion"] = (
                            s.pop("sum_ratio_subordinacion") / n
                        )
                        s["mean_branching_ratio"] = s.pop("sum_branching_ratio") / n

                        # SUBTLEX medias
                        s["mean_zipf"] = s.pop("sum_mean_zipf") / n
                        s["mean_pct_oov"] = s.pop("sum_pct_oov") / n
                        s["mean_oral_ratio"] = s.pop("sum_oral_ratio") / n
                        s["mean_academic_ratio"] = s.pop("sum_academic_ratio") / n
                        s["mean_domain_specific_ratio"] = (
                            s.pop("sum_domain_specific_ratio") / n
                        )
                        s["mean_surprisal"] = s.pop("sum_mean_surprisal") / n
                    else:
                        # Eliminar sumas residuales (no debería pasar)
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
        grammatical_pipeline=None,
        grammatical_dashboard_path: str = "",
        global_corpus=None,  # GlobalCorpus instance, optional
        lex_analyzer=None,  # GlobalLexicalAnalyzer instance, optional
    ):
        print("=== Iniciando Pipeline ALCESTE v4 ===")

        # ── 1. Optimización de hiperparámetros (El núcleo robusto) ─────────────
        cached_uces_por_doc = None
        cached_doc_metadata_map = None

        best_params_path = BEST_PARAMS_PATH
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
                f"   [WARNING] Could not load best_params.json. Running optimizer if enabled."
            )
            best_params = None

        if best_params is None and self.config.optimize:
            # Run optimizer
            self.config.use_embeddings = False
            optimizador = Optimizador(self.config, corpus_raw)
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

        try:
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
            ) = self.double_clf.run(corpus_raw)
            self._enrich_uces_with_doc_metadata(uces_por_doc, doc_metadata_map)
            self.db.data["section_registry"] = getattr(
                self.double_clf.segmentador, "section_registry", {}
            )
        finally:
            if cached_uces_por_doc is not None:
                self.double_clf.segmentador.segmentar_en_uces = original_seg
                self.double_clf.segmentador.lematizar_uces = original_lem

        if not uces_est_list:
            print(
                "   [FATAL] Cero UCs estables. Tu corpus es un desastre o los parámetros son muy restrictivos. Terminando."
            )
            return

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


config = Config(
    # NLP
    use_bigrams=True,
    stem_backend="snowball",
    # Parámetros base (El optimizador los va a mutar de todos modos)
    min_uce_words=3,
    min_forms_uc=[10, 14],
    tsj=3,
    # CDH & AFC
    use_cdh=True,
    pseudocount=0.1,
    use_projection=True,
    # CAH
    use_cah_per_class=True,
    cah_per_class_top_terms=20,
    # Análisis Extra
    analyze_metadata=True,
    glm_method="chi2",
    use_network_analysis=True,
    use_term_stability=True,
    use_multivariate_analysis=True,
    multivariate_metadata=["Edad_Cat", "Sexo", "Ocupacion_Cat", "Procedencia_Cat"],
    # LLM apagado
    use_llm_synthesis=False,
    # deepseek_api_key="",
    # Random Forest + SHAP
    use_rf_shap=True,
    rf_n_estimators=100,
    rf_max_depth=5,
    rf_outlier_method="iqr",
    rf_cat_encoding="onehot",
    # OPTIMIZADOR ENCENDIDO
    optimize=True,
    optimize_trials=100,  # 15 iteraciones es suficiente para ver si tu código sirve
    random_state=42,
)

if __name__ == "__main__":
    # ── Config ALCESTE ────────────────────────────────────────────────
    alceste_config = Config(
        use_bigrams=True,
        stem_backend="snowball",
        min_uce_words=3,
        min_forms_uc=[10, 14],
        tsj=3,
        use_cdh=True,
        use_projection=True,
        analyze_metadata=True,
        use_network_analysis=True,
        use_term_stability=True,
        optimize=True,
        optimize_trials=100,
        random_state=42,
        multivariate_metadata=["Edad_Cat", "Sexo", "Ocupacion_Cat", "Procedencia_Cat"],
    )

    # ── Config gramatical ─────────────────────────────────────────────

    gram_config = GramConfig(
        min_tokens_por_uce=30,
        max_tokens_por_uce=200,
    )

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
    global_corpus = GlobalCorpus()
    lex_analyzer = GlobalLexicalAnalyzer(
        subtlex_analyzer=subtlex,
        cooc_window=4,
        cooc_min_count=3,
    )

    # ── PHASE 1: ALCESTE — segmentación + clasificación ──────────────
    # WorkflowOrchestrator.ejecutar() ahora acepta gram_pipeline
    orchestrator = WorkflowOrchestrator(alceste_config)

    # Necesitamos uces_por_doc ANTES de que ejecutar() termine.
    # La forma más limpia: exponer uces_por_doc como atributo del orchestrator.
    # Por ahora, pasamos gram_pipeline directamente para que ejecutar()
    # llame a procesar_desde_uces() en el momento correcto.
    orchestrator.ejecutar(
        corpus_raw=uwu,  # <--- Changed from uwu
        grammatical_pipeline=gram_pipeline,
        global_corpus=global_corpus,
        lex_analyzer=lex_analyzer,
        grammatical_dashboard_path=OUTPUT_DASHBOARD,
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
    lex_analyzer.export_network_gexf(OUTPUT_NETWORK_GEXF)

    # ── PHASE 4: Guardar todo ─────────────────────────────────────────
    # Fixed: Removed the overlapping positional argument
    global_corpus.export_dashboard(
        OUTPUT_DASHBOARD,
        predicate_result=global_predicate_result,
        subtlex_analyzer=subtlex,
    )
    lex_analyzer.export_excel(OUTPUT_LEXICAL_XLSX)
    lex_analyzer.export_json(OUTPUT_LEXICAL_JSON)
