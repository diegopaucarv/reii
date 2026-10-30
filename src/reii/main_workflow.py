# %%viztracer
# main_pipeline.py
from __future__ import annotations

import copy
import dataclasses
import gc
import glob
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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

from reii.config import (
    BEST_PARAMS_PATH,
    CLASSIC_DATA_DIR,
    CLASSIC_OUTPUT_DIR,
    COREF_CHAINS_CACHE_PATH,
    CORPUS_NAME,
    DATABASE_URL,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL,
    OUTPUT_DASHBOARD,
    OUTPUT_NETWORK_GEXF,
    RF_N_JOBS,
    SPACY_MODEL,
    STANZA_SERVER_URL,
    TOGETHER_API_KEY,
    TOGETHER_BASE_URL,
    UCE_CALIBRATION_PATH,
    WORKFLOW_DB_PATH,
)
from reii.config import (
    DATA_DIR as REII_DATA_DIR,
)
from reii.coref_cache import (
    atomic_write_json,
    build_cache_document,
    file_lock,
    load_cache,
    missing_ctx_values,
)
from reii.gram.gramatical_analyzer import (
    UCE,
    GlobalCorpus,
    GlobalLexicalAnalyzer,
    NLPProvider,
    PipelineGramatical,
    SubtlexAnalyzer,
    WordEmbeddingsAnalyzer,
    train_adverb_classifier,
)
from reii.gramatica import load_gramatica
from reii.hf_local import resolve_model_dir
from reii.method_c_policy import (
    build_method_c_result,
    format_target_k_notice,
    guarded_fit,
    resolve_method_c_backend,
)
from reii.method_c_policy import (
    corte_optimo_1d as _corte_optimo_1d,
)
from reii.method_selection import (
    MethodMetrics,
    Selection,
    persist_selection,
    select_method,
)
from reii.retention_policy import (
    TOOL_ADVERB_LEMMAS,
    TokenTableWriter,
    normalize_lemmas,
    policy_content_pos,
    resolve_retention_policy,
    token_decision,
    token_rows,
)

# Contrato dinámico de UCE para la fase abductiva.
# NUNCA modificar uce.cluster_id desde _project_liminal_uces().
#   uce.projected_cluster_id : Optional[int]   → clase más cercana
#   uce.projection_distance  : Optional[float] → d²_χ² mínima
#   uce.projection_margin    : Optional[float] → d²_χ²(2ª) − d²_χ²(1ª)
#   uce.projection_ratio     : Optional[float] → d²_χ²(2ª) / d²_χ²(1ª)
_UCE_PROJECTION_FIELDS = (
    "projected_cluster_id",
    "projection_distance",
    "projection_margin",
    "projection_ratio",
)

from reii.gram.gramatical_analyzer import (
    Config as GramConfig,
)
from reii.seg.coref_checkpoint import (
    COREF_CHECKPOINT_VERSION,
    CheckpointGuard,
    _cargar_checkpoint_coref,
    _coref_code_fingerprint,
    _guardar_checkpoint_coref,
    checkpoint_path_for_pass,
    uces_fingerprint,
)
from reii.seg.forms_merging import agrupar_secuencial_por_umbral, formas_distintas
from reii.seg.root_cache import RootSubjectCache, boundary_tails, root_context
from reii.seg.segment_grouping import agrupar_segmentos_por_seccion
from reii.seg.segmentador import ProgressiveSegmenter
from reii.seg.uce_calibration import calibrate as _calibrar_bayesiano
from reii.seg.uce_calibration import desired_avg_for_corpus

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


@lru_cache(maxsize=4)
def _get_sentence_embedder(model_name: str) -> SentenceTransformer:
    """Instancia única (cacheada) del SentenceTransformer.

    Evita re-validar el cache de HuggingFace y recargar pesos en cada
    llamada (generar_embeddings_uces/ucs, SynthesisGenerator, ...).
    """
    return SentenceTransformer(resolve_model_dir(model_name) or model_name)


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

    _HDBSCAN_AVAILABLE = True
except ImportError as e:
    _HDBSCAN_AVAILABLE = False
    print(f"   [EMB] hdbscan no disponible: {e}")

try:
    import umap

    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


gc.collect()


# Función a prueba de balas para limpiar extensiones dobles y simples
# Convierte "doc.txt.json" o "doc.txt" a una llave maestra limpia: "doc"
def obtener_llave_maestra(nombre):
    return str(nombre).replace(".txt.json", "").replace(".json", "").replace(".txt", "")


# 0. Cargamos tu metadata usando la llave maestra
# batch_processor.exportar_para_workflow() escribe Refined_Database.csv en
# data/txt_outputs/ (no en data/).
metadata_csv = os.path.join(REII_DATA_DIR, "txt_outputs", "Refined_Database.csv")
df_meta = (
    pd.read_csv(metadata_csv, sep=";")
    if os.path.exists(metadata_csv)
    else pd.DataFrame()
)
meta_dict = {}

for _, row in df_meta.iterrows():
    llave = obtener_llave_maestra(row["id"])
    meta_dict[llave] = row[
        ["Edad_Cat", "Sexo", "Ocupacion_Cat", "Procedencia_Cat"]
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
# SKIP DE DOCUMENTOS YA PROCESADOS
# ══════════════════════════════════════════════════════════════════════
# Por defecto, los documentos que ya tienen todas sus UCEs en la base de
# datos NO se vuelven a procesar (evita re-correr 4h+ de pipeline por
# documentos sin cambios). Para forzar el reprocesamiento completo:
#   python src/reii/main_workflow.py --force
# (o --reprocess, alias).
# Para elegir el modo de clasificación (A/B/C):
#   python src/reii/main_workflow.py --mode wc_only|emb_only|all
# (ver _parse_cli_mode más abajo).
_FORCE_REPROCESS = "--force" in sys.argv or "--reprocess" in sys.argv
if not _FORCE_REPROCESS:
    try:
        from reii.backend.sql.db import connect as _pg_connect

        _pg = _pg_connect(DATABASE_URL)
        try:
            _procesados = {
                r[0]
                for r in _pg.execute(
                    "SELECT DISTINCT origen FROM documents "
                    "WHERE origen IS NOT NULL AND n_uces > 0"
                ).fetchall()
            }
        finally:
            _pg.close()
        _a_omitir = [k for k in uwu if k in _procesados]
        if _a_omitir:
            print(
                f"   [SKIP] {len(_a_omitir)}/{len(uwu)} documentos ya procesados "
                f"(usa --force para reprocesar): {_a_omitir}"
            )
            uwu = {k: v for k, v in uwu.items() if k not in _procesados}
    except Exception as _e:
        print(f"   [WARNING] No se pudo verificar docs procesados: {_e}")


# ══════════════════════════════════════════════════════════════════════
# MODO DE CLASIFICACIÓN VÍA CLI (--mode)
# ══════════════════════════════════════════════════════════════════════
#   python src/reii/main_workflow.py --mode wc_only
#   python src/reii/main_workflow.py --mode emb_only
#   python src/reii/main_workflow.py --mode all
# El flag --mode gana sobre el override de ConfigStore (dashboard/PostgreSQL).
_CLI_MODES_VALIDOS = {
    "wc_only",
    "coref_only",
    "emb_only",
    "wc_coref",
    "wc_emb",
    "coref_emb",
    "all",
    "select",
    "classic",  # preset ALCESTE 2012 (NO es un classification_mode)
    # aliases legacy
    "auto",
    "both",
    "sim_only",
    "wc_sim",
    "sim_emb",
}


def _parse_cli_mode() -> Optional[str]:
    """Devuelve el modo de --mode si está presente; valida y sale con error si es inválido."""
    if "--mode" not in sys.argv:
        return None
    i = sys.argv.index("--mode")
    if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("--"):
        print(
            "   [ERROR] --mode requiere un valor. Válidos: "
            + ", ".join(sorted(_CLI_MODES_VALIDOS))
        )
        sys.exit(2)
    mode = sys.argv[i + 1].strip().lower()
    if mode not in _CLI_MODES_VALIDOS:
        print(
            f"   [ERROR] --mode inválido: {mode!r}. Válidos: "
            + ", ".join(sorted(_CLI_MODES_VALIDOS))
        )
        sys.exit(2)
    return mode


def _parse_cli_uce_target_size() -> Optional[int]:
    """Devuelve el valor de --uce-target-size si está presente; valida y sale con error si es inválido."""
    if "--uce-target-size" not in sys.argv:
        return None
    i = sys.argv.index("--uce-target-size")
    if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("--"):
        print(
            "   [ERROR] --uce-target-size requiere un valor entero "
            "(p.ej. --uce-target-size 30)"
        )
        sys.exit(2)
    raw = sys.argv[i + 1].strip()
    try:
        value = int(raw)
    except ValueError:
        print(f"   [ERROR] --uce-target-size inválido: {raw!r}. Debe ser un entero.")
        sys.exit(2)
    if value <= 0:
        print(f"   [ERROR] --uce-target-size debe ser > 0 (recibido: {value}).")
        sys.exit(2)
    return value


# ══════════════════════════════════════════════════════════════════════
# FLAGS DEL OPTIMIZADOR / SONDAS
# ══════════════════════════════════════════════════════════════════════
# Este módulo escanea sys.argv a mano (no usa argparse): un flag
# desconocido se ignora EN SILENCIO. Eso convertía a los scripts de
# diagnóstico en no-ops silenciosos. Los overrides se aplican al final de
# main(), DESPUÉS de ConfigStore, porque ConfigStore puede pisar los
# valores de Config(...).
_CLI_FLAGS_CON_VALOR = {
    "--trials",
    "--seed",
    "--study-name",
    "--verdict-mode",
    "--stage1-fraction",
    "--coverage-gate",
    "--coverage-gate-all",
    "--method-c-backend",
    "--max-class-share",
}
_CLI_FLAGS_SIN_VALOR = {
    "--optimize-only",
    "--no-optuna-warm-start",
    "--single-stage",
    "--stability-bootstrap",
    "--allow-unknown-flags",
}

#: Backends de Method C expuestos por CLI. Espejo de `Config.method_c_backend`;
#: la autoridad final sigue siendo `method_c_policy.resolve_method_c_backend`.
_METHOD_C_BACKENDS_CLI = {"dense_cdh", "ward", "kmeans_spherical", "hdbscan"}

#: Ayuda en ASCII puro: en consola Windows con cp1252 los acentos y las
#: flechas rompen el print justo cuando el usuario pide ayuda.
_CLI_USAGE = """
REII - main_workflow.py [flags]

  --force | --reprocess          reprocesa documentos ya completos
  --uce-target-size N            fija el target de UCE (desactiva la calibracion)

--mode (modo de clasificacion):
  select                         A | B | C: elige UNO con select-v4 y lo
                                 persiste en data/method_selection.json
  wc_only | coref_only | emb_only             un solo metodo (A | B | C)
  wc_coref | wc_emb | coref_emb | all         interseccion de metodos
  classic                        escenario A: ALCESTE 2012 aislado en data/classic/
  alias: auto->select, both->wc_coref, sim_only->coref_only,
         wc_sim->wc_coref, sim_emb->coref_emb

Optimizador:
  --optimize-only                ignora best_params.json y TERMINA tras optimizar
  --trials N                     presupuesto (default operativo 300, >= 1)
  --stage1-fraction f            reparto etapa 1 / etapa 2 (0 < f < 1, default 0.3)
  --single-stage                 una sola etapa (mode=select), sin etapa 1
  --study-name X                 prefijo del study; el codigo anade
                                 _stage1_wc_<fp> y _stage2_select_<fp>
  --seed N                       random_state (>= 0)
  --no-optuna-warm-start         sin warm start
  --verdict-mode all|two_of_three
  --coverage-gate f              gate de cobertura del select (en [0, 1])
  --coverage-gate-all f          gate de cobertura del modo all (en [0, 1])

Method C y gate de seleccion:
  --method-c-backend B           dense_cdh (default) | ward | kmeans_spherical
                                 | hdbscan (diagnostico: no controla K, emite -1)
  --max-class-share f            gate de dominancia sobre el CORPUS
                                 (pre-registrado en el paper: 0.70)

Estabilidad:
  --stability-bootstrap          bootstrap real; solo en el select final
                                 (incompatible con --optimize-only)

Otros:
  -h, --help                     esta ayuda
  --allow-unknown-flags          ignora flags desconocidos en vez de abortar

Notas:
  * Este modulo escanea sys.argv a mano: un flag desconocido ABORTA la corrida
    (antes se ignoraba en silencio y la corrida parecia valida).
  * El default del entrypoint es `select`, pero ConfigStore (PostgreSQL) puede
    pisarlo; --mode gana sobre ambos.
  * Sin flag a proposito: optimize_preference (knee esta pre-registrado).
"""


def _print_cli_help() -> None:
    print(_CLI_USAGE.strip("\n"))


_OPTIMIZE_ONLY = "--optimize-only" in sys.argv
_NO_OPTUNA_WARM_START = "--no-optuna-warm-start" in sys.argv
_SINGLE_STAGE = "--single-stage" in sys.argv

# La ayuda se resuelve antes de construir Config, pero DESPUES de los imports
# del modulo: no se puede evitar el coste de importar spacy/nltk.
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    _print_cli_help()
    sys.exit(0)


def _cli_val(flag: str) -> Optional[str]:
    """Valor de `flag` en argv, o None si el flag no está presente."""
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("--"):
        print(f"   [ERROR] {flag} requiere un valor.")
        sys.exit(2)
    return sys.argv[i + 1].strip()


def _cli_cast(flag: str, cast):
    raw = _cli_val(flag)
    if raw is None:
        return None
    try:
        return cast(raw)
    except ValueError:
        print(f"   [ERROR] {flag} inválido: {raw!r}")
        sys.exit(2)


def _warn_unknown_cli_flags() -> None:
    """Valida argv contra la lista de flags que este modulo conoce.

    El escaneo de argv es manual: lo que no se reconocia se ignoraba en
    silencio, de modo que un typo (`--stabilyty-bootstrap`) corria sin
    bootstrap y el resultado parecia valido. Ahora aborta con exit 2, salvo
    que se pida `--allow-unknown-flags`.
    """
    known = {"--force", "--reprocess", "--mode", "--uce-target-size"}
    known |= _CLI_FLAGS_CON_VALOR | _CLI_FLAGS_SIN_VALOR
    known |= {"--help", "-h"}
    unknown = [a for a in sys.argv[1:] if a.startswith("-") and a not in known]
    if not unknown:
        return
    if "--allow-unknown-flags" in sys.argv:
        print(
            "   [WARNING] flags desconocidos (ignorados por "
            f"--allow-unknown-flags): {unknown}"
        )
        return
    print(f"   [ERROR] flags desconocidos: {unknown}")
    print("           se aborta en vez de ignorarlos en silencio.")
    print("           usa --help para la lista de flags validos, o")
    print("           --allow-unknown-flags para ignorarlos a proposito.")
    sys.exit(2)


def _check_cli_ranges(upd: dict) -> None:
    """Valida rangos de los overrides antes de pasarlos a Config.

    El cast solo comprobaba el tipo: `--stage1-fraction 1.5` llegaba al
    optimizador y el reparto de trials se "autocorregia" con un mensaje que
    no describia el problema.
    """
    rules = (
        ("optimize_trials", "--trials", lambda v: v >= 1, ">= 1"),
        ("random_state", "--seed", lambda v: v >= 0, ">= 0"),
        (
            "optimize_stage1_fraction",
            "--stage1-fraction",
            lambda v: 0.0 < v < 1.0,
            "0 < f < 1",
        ),
        (
            "optimize_coverage_gate",
            "--coverage-gate",
            lambda v: 0.0 <= v <= 1.0,
            "en [0, 1]",
        ),
        (
            "optimize_coverage_gate_all",
            "--coverage-gate-all",
            lambda v: 0.0 <= v <= 1.0,
            "en [0, 1]",
        ),
    )
    for field, flag, ok, txt in rules:
        if field in upd and not ok(upd[field]):
            print(f"   [ERROR] {flag} debe ser {txt} (recibido: {upd[field]!r})")
            sys.exit(2)


def _cli_overrides(cfg):
    """Overrides del optimizador. Llamar SOLO tras aplicar ConfigStore."""
    import dataclasses as _dc

    upd = {}
    for flag, field, cast in (
        ("--trials", "optimize_trials", int),
        ("--seed", "random_state", int),
        ("--stage1-fraction", "optimize_stage1_fraction", float),
        ("--coverage-gate", "optimize_coverage_gate", float),
        ("--coverage-gate-all", "optimize_coverage_gate_all", float),
        ("--study-name", "optimize_study_name", str),
    ):
        value = _cli_cast(flag, cast)
        if value is not None:
            upd[field] = value
    _check_cli_ranges(upd)

    verdict = _cli_val("--verdict-mode")
    if verdict is not None:
        if verdict not in ("all", "two_of_three"):
            print(f"   [ERROR] --verdict-mode inválido: {verdict!r} (all|two_of_three)")
            sys.exit(2)
        upd["verdict_mode"] = verdict
    if _SINGLE_STAGE:
        upd["optimize_two_stage"] = False
        print("   [CLI] --single-stage: una sola etapa (mode=select)")
    if "--stability-bootstrap" in sys.argv:
        if _OPTIMIZE_ONLY:
            print(
                "   [ERROR] --stability-bootstrap no se combina con "
                "--optimize-only: el bootstrap real cuesta 1-10 s por select, el"
                " optimizador corre cientos, y ademas mueve el objetivo de la"
                " etapa 2. Reservado para el select final."
            )
            sys.exit(2)
        upd["stability_bootstrap"] = True

    _backend = _cli_val("--method-c-backend")
    if _backend is not None:
        if _backend not in _METHOD_C_BACKENDS_CLI:
            print(
                f"   [ERROR] --method-c-backend invalido: {_backend!r}. "
                f"Validos: {', '.join(sorted(_METHOD_C_BACKENDS_CLI))}"
            )
            sys.exit(2)
        upd["method_c_backend"] = _backend
        if _backend == "hdbscan":
            print(
                "   [!] --method-c-backend=hdbscan: no controla K y emite -1;"
                " `select` lo descarta salvo method_c_hdbscan_selectable=True."
            )

    _max_share = _cli_cast("--max-class-share", float)
    if _max_share is not None:
        if not 0.0 < _max_share < 1.0:
            print(
                f"   [ERROR] --max-class-share debe cumplir 0 < share < 1 "
                f"(recibido: {_max_share!r})"
            )
            sys.exit(2)
        upd["method_selection_max_class_share"] = _max_share
        print(
            f"   [!] --max-class-share={_max_share}: el valor pre-registrado del"
            " paper es 0.70 sobre el CORPUS. Cambiarlo rompe la comparacion con"
            " la corrida registrada."
        )

    if _OPTIMIZE_ONLY:
        _m = _parse_cli_mode()
        if _m not in (None, "classic"):
            print(
                f"   [i] --mode {_m} es inerte con --optimize-only: la corrida"
                " termina tras el optimizador y la etapa 2 optimiza siempre en"
                " mode='select'."
            )
        print(
            "   [CLI] --optimize-only: se ignora best_params.json "
            "y se sale tras el optimizador"
        )
    for field, value in upd.items():
        print(f"   [CLI] {field}={value}")
    return _dc.replace(cfg, **upd) if upd else cfg


# ══════════════════════════════════════════════════════════════════════
# FINGERPRINT DEL CORPUS (best_params dataset-specific)
# ══════════════════════════════════════════════════════════════════════
# best_params.json se optimiza para UN corpus concreto. Para no reutilizar
# parámetros de otro dataset (o de una versión editada de este), sellamos el
# archivo con un hash de los archivos de entrada: nombres + mtime + tamaño.
# Cualquier cambio (añadir/quitar/editar un archivo) invalida el cache y
# fuerza re-optimización.


def _corpus_fingerprint() -> str:
    """Hash del corpus: nombres + mtime + tamaño de los archivos de entrada.

    Incluye los JSONs segmentados (txt_outputs/tmp), los TXTs crudos
    (txt_outputs) y la metadata sociodemográfica (Refined_Database.csv).
    """
    h = hashlib.sha256()
    _paths: List[str] = []
    _paths += sorted(
        glob.glob(os.path.join(REII_DATA_DIR, "txt_outputs", "tmp", "*.json"))
    )
    _paths += sorted(glob.glob(os.path.join(REII_DATA_DIR, "txt_outputs", "*.txt")))
    _meta = os.path.join(REII_DATA_DIR, "txt_outputs", "Refined_Database.csv")
    if os.path.exists(_meta):
        _paths.append(_meta)
    for _p in _paths:
        h.update(os.path.basename(_p).encode("utf-8"))
        h.update(str(os.path.getmtime(_p)).encode("utf-8"))
        h.update(str(os.path.getsize(_p)).encode("utf-8"))
    return h.hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════
# Cache de cadenas coref: helpers de módulo para que el orquestador
# (no solo el Optimizador) pueda leer/escribir el mismo cache en disco.
# ══════════════════════════════════════════════════════════════════════
from reii.coref_cache import (
    atomic_write_json,
    build_cache_document,
    get_doc_chains,
    load_cache,
    missing_ctx_values,
)


def _partition_sizes_from_labels(labels: Optional[Dict[str, int]]) -> List[int]:
    """Tamanos de clase ordenados de una particion CRUDA (uce_id -> label).

    Excluye los no asignados (``label < 0``, p.ej. HDBSCAN ``-1``): el
    denominador del gate de dominancia son las UCEs ASIGNADAS. Sin esta
    exclusion, un metodo con 55% de ruido y una clase mayoritaria entre los
    asignados pasaria el gate por pura dilution del denominador.
    """
    if not labels:
        return []
    counts: Dict[int, int] = {}
    for v in labels.values():
        iv = int(v)
        if iv < 0:
            continue
        counts[iv] = counts.get(iv, 0) + 1
    return sorted(counts.values())


# Tope de n para `ward` (memoria O(n²) de la matriz de distancias). A n=6000
# son ~144 MiB condensados / ~275 MiB cuadrados. Por encima, la política
# degrada a `dense_cdh` con aviso "[!]" en vez de abortar (HARDENING.md §2,
# hunk #69). Se pasa a `guarded_fit(ward_max_n=...)`; NO es un campo de Config.
_METHOD_C_WARD_MAX_N = 6000


def _coref_cache_key_standalone(config_base: "Config", fp: str) -> str:
    """Debe coincidir EXACTAMENTE con Optimizador._coref_cache_key."""
    uce_target = int(getattr(config_base, "uce_target_size", 26))
    min_words = int(getattr(config_base, "min_uce_words", 2))
    return f"{fp}:{uce_target}:{min_words}"


def load_coref_chains_from_disk(
    config_base: "Config",
) -> Dict[int, Dict[str, List]]:
    """Carga el cache de cadenas coref del disco si su clave coincide con
    el corpus + segmentación actuales. Devuelve {} si no hay cache válido."""
    fp = _corpus_fingerprint()
    cache_key = _coref_cache_key_standalone(config_base, fp)
    result = load_cache(COREF_CHAINS_CACHE_PATH, cache_key)
    return result or {}


def save_coref_chains_to_disk(
    config_base: "Config", chains: Dict[int, Dict[str, List]]
) -> None:
    """Persiste el cache completo de forma atómica. Nunca lanza."""
    fp = _corpus_fingerprint()
    cache_key = _coref_cache_key_standalone(config_base, fp)
    doc = build_cache_document(cache_key, fp, chains)
    try:
        atomic_write_json(COREF_CHAINS_CACHE_PATH, doc)
    except OSError as e:
        logger.warning("No se pudo guardar el cache de cadenas coref: %s", e)


def compute_coref_chains_for_ctx(
    progressive_segmenter,
    uces_por_doc: List[List["UCE"]],
    ctx_values: Iterable[int],
) -> Dict[int, Dict[str, List]]:
    """
    Réplica exacta del paso 3a–3c de DoubleClassifier._clasificar_umbral_coref,
    pero standalone: extrae cadenas para cada ctx sin clasificar nada.

    Se usa para precomputar el cache ANTES de Pass 1, replicando lo que el
    Optimizador ya hace en _precompute_coref_chains (no podemos reusar el
    método porque es instance method del Optimizador y aquí no lo tenemos).
    """
    ctx_values = {int(c) for c in ctx_values if int(c) > 0}
    result: Dict[int, Dict[str, List]] = {c: {} for c in ctx_values}
    if not ctx_values:
        return result

    ps = progressive_segmenter
    for doc_uces in uces_por_doc:
        if not doc_uces:
            continue
        doc_id = doc_uces[0].doc_id

        # Pre-merge (recursive_segmentation + final_clustering), idéntico a
        # _clasificar_umbral_coref paso 3a.
        if len(doc_uces) > 1:
            try:
                rec_seg = ps.recursive_segmentation([u.texto for u in doc_uces])
                prog_seg = ps.final_clustering(rec_seg)
                premerged_groups = _remap_uces_to_segments(doc_uces, prog_seg)
            except Exception as e:
                logger.warning(
                    "Pre-merge falló para doc %s en precompute: %s", doc_id, e
                )
                premerged_groups = [[u] for u in doc_uces]
        else:
            premerged_groups = [doc_uces]

        # Guard: mismo colapso que en runtime
        if len(premerged_groups) == 1 and len(doc_uces) > 50:
            premerged_groups = [[u] for u in doc_uces]

        # segments_info con offsets secuenciales
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
                    "uce_id": group[-1].id,
                }
            )
            full_doc_text += group_text + " "
            current_offset = end + 1

        for ctx in ctx_values:
            try:
                chains = ps._extract_global_chains(
                    segments_info, full_doc_text, context_units=ctx
                )
                if getattr(ps, "last_extraction_incomplete", False):
                    raise RuntimeError(
                        "extracción incompleta (Stanza/servidor no disponible)"
                    )
            except Exception as e:
                logger.warning("Extracción falló doc %s ctx %d: %s", doc_id, ctx, e)
                # NO se guarda [] (se persistiría como "sin cadenas" para siempre):
                # el doc queda ausente y se re-extrae bajo demanda.
                print(f"   [Coref] ⚠ doc {doc_id} ctx={ctx} NO se cachea: {e}")
                continue
            result[ctx][doc_id] = chains

    for ctx, by_doc in result.items():
        total = sum(len(v) for v in by_doc.values())
        print(
            f"   [Coref] ctx={ctx}: {total} cadenas en {len(by_doc)} docs (precompute)"
        )
    return result


def _corpus_total_words(corpus_raw) -> int:
    """Palabras totales del corpus (conteo crudo, .split()) — usado para
    elegir el promedio deseado por tamaño de corpus (desired_avg_for_corpus),
    la misma tabla que ALCESTE usa para auto-calibrar su UCE."""
    total = 0
    for doc_data in corpus_raw.values():
        for seg in doc_data.get("texto_segmentos", []):
            total += len((seg.get("texto") or "").split())
    return total


def calibrar_uce_target_size(
    segmentador: "SegmentadorALCESTE",
    corpus_raw,
    desired_avg: Optional[float] = None,
    bounds: Optional[Tuple[int, int]] = None,
    max_evals: Optional[int] = None,
    cache_path: Optional[str] = None,
    force: bool = False,
) -> Optional[int]:
    """Calibra uce_target_size contra ESTE corpus y devuelve el valor a usar,
    o None si la calibración falla o está desactivada (el llamador debe
    conservar el uce_target_size que ya tenía en ese caso — nunca lo pisa
    con un valor inventado).

    Cachea el resultado en `cache_path` (por defecto UCE_CALIBRATION_PATH)
    bajo la huella del corpus (_corpus_fingerprint), mismo mecanismo de
    invalidación que best_params.json: si el corpus cambia, el cache se
    descarta solo y se recalibra. `force=True` ignora el cache existente.

    Nunca lanza: cualquier fallo (spaCy no disponible, corpus vacío, lo que
    sea) se registra con logger.warning y devuelve None — la calibración es
    una mejora opcional, no debe poder tumbar una corrida.
    """
    cfg = segmentador.config
    bounds = bounds or getattr(cfg, "uce_calibration_bounds", (14, 30))
    max_evals = max_evals or getattr(cfg, "uce_calibration_max_evals", 5)
    desired_avg = (
        desired_avg
        if desired_avg is not None
        else getattr(cfg, "uce_calibration_desired_avg", None)
    )
    cache_path = cache_path or UCE_CALIBRATION_PATH

    try:
        if desired_avg is None:
            desired_avg = desired_avg_for_corpus(_corpus_total_words(corpus_raw))

        fp = _corpus_fingerprint()
        cache_key = f"{fp}:{desired_avg}:{bounds[0]}-{bounds[1]}"

        if not force and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("cache_key") == cache_key:
                    print(
                        f"   [UCE-calib] cache válido ({cache_path}): "
                        f"uce_target_size={cached['target']} "
                        f"(promedio real medido: {cached['measured_avg']:.1f})"
                    )
                    return int(cached["target"])
            except (OSError, json.JSONDecodeError, KeyError) as e:
                logger.warning("   [UCE-calib] cache ilegible, recalibrando: %s", e)

        docs_muestra = segmentador.muestra_para_calibracion(corpus_raw)
        if not docs_muestra:
            logger.warning("   [UCE-calib] corpus vacío, no se calibra.")
            return None

        def measure_fn(target: int):
            return segmentador.medir_promedio_real(docs_muestra, target)

        resultado = _calibrar_bayesiano(
            measure_fn, desired_avg=desired_avg, bounds=bounds, max_evals=max_evals
        )
        print(f"   [UCE-calib] {resultado.summary()}")

        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cache_key": cache_key,
                        "target": resultado.target,
                        "desired_avg": resultado.desired_avg,
                        "measured_avg": resultado.measured_avg,
                        "converged": resultado.converged,
                        "n_mediciones": len(resultado.observations),
                    },
                    f,
                    indent=2,
                )
        except OSError as e:
            logger.warning("   [UCE-calib] no se pudo guardar el cache: %s", e)

        return resultado.target
    except Exception as e:
        logger.warning(
            "   [UCE-calib] calibración falló, se conserva el target actual: %s", e
        )
        return None


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

    # Escenario del paper (docs/design_v5/CLASSIC_MODE.md, §1):
    #   "modern"  = B (ALCESTE + innovaciones)
    #   "classic" = A (ALCESTE 2012, fidelidad; se activa con --mode classic)
    # NO está en ConfigStore a propósito: el dashboard no debe poder convertir
    # una corrida en clásica por accidente.
    run_mode: str = "modern"

    min_uce_words: int = 2  # era 3; alineado con C (UCEs cortas)
    uce_target_size: int = 26  # objetivo de entrada; ver auto_calibrate_uce_size
    min_forms_uc: List[int] = field(default_factory=lambda: [13, 17])

    # Auto-calibración bayesiana de uce_target_size (ver calibrar_uce_target_size
    # y reii.seg.uce_calibration): corrige, ANTES de segmentar el corpus completo,
    # el sesgo entre el objetivo de entrada y el promedio real de palabras/UCE
    # que produce _segmentar_reinert_spacy en ESTE corpus — el sesgo depende de
    # la densidad de puntuación del texto, no es una constante universal.
    # uce_target_size de arriba queda como PUNTO DE PARTIDA si la calibración
    # está desactivada o falla; con auto_calibrate_uce_size=True, se reemplaza
    # por el valor calibrado antes de la segmentación real.
    auto_calibrate_uce_size: bool = True
    uce_calibration_bounds: Tuple[int, int] = (
        14,
        30,
    )  # rango Reinert, nunca se sale de aquí
    uce_calibration_desired_avg: Optional[float] = (
        None  # None → tabla ALCESTE por tamaño de corpus
    )
    uce_calibration_max_evals: int = 5

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

    pseudocount: float = 0.00
    swap_iterations: int = 2
    n_perm_cdh: int = 100
    perm_min_uc_size: int = 50
    min_r2_threshold: float = 0.05  # Minimum R² for a split to be accepted
    chi2_threshold_small: float = 3.84
    max_depth_cdh: int = 10
    min_cluster_size_cdh: float = 0.10
    # min_cluster_size_cdh se interpreta como PISO DURO del tamaño de cada
    # clase terminal (no solo como regla de parada de la recursión). Con
    # False se recupera el comportamiento anterior, donde una clase podía
    # terminar con 1 UC y K dependía de cortes 1-vs-(n-1).
    cdh_enforce_min_leaf: bool = True

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
    # ── Proyección liminal (operación ABDUCTIVA, no inductiva) ──────────
    # Las UCEs que no sobreviven la triple intersección NO son parte del
    # corpus analítico. Este flag permite proyectarlas sobre los centroides
    # de clase ya consolidados, escribiendo en campos SEPARADOS
    # (projected_cluster_id, projection_distance, ...) que ningún cálculo
    # de φ, AFC, inercia o estabilidad de términos consume.
    # Default False: ALCESTE puro.
    use_liminal_projection: bool = True
    # Umbral opcional de confianza: si d2_second / d2_best < este valor,
    # la UCE queda sin proyección (ambigua entre clases).
    liminal_projection_min_ratio: float = 1.10

    use_projection: bool = True
    projection_method: str = "afc"

    use_cah_per_class: bool = True
    cah_per_class_top_terms: int = 50

    analyze_metadata: bool = True
    glm_method: str = "chi2"
    glm_alpha: float = 0.05
    n_permutations: int = 1000

    use_llm_synthesis: bool = True
    together_api_key: str = TOGETHER_API_KEY
    llm_model: str = LLM_MODEL
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
    # Workers de sklearn (-1 = todos los núcleos). Ver RF_N_JOBS en config.py:
    # con -1 en el contenedor puede agotar la RAM y matar la daemon de Docker.
    rf_n_jobs: int = RF_N_JOBS

    # ── Optimizer (Optuna) ──────────────────────────────────────────────
    optimize: bool = True
    optimize_trials: int = 200
    optimize_stage1_fraction: float = 0.3  # n_stage1 = max(20, int(n * frac))
    optimize_two_stage: bool = True  # False => una sola etapa (mode="select")
    optimize_sampler: str = "tpe"  # "tpe" | "random" | "cmaes"
    optimize_pruner: str = "median"  # "median" | "hyperband" | "none"
    optimize_storage: Optional[str] = None  # ej. "sqlite:///reii_optuna.db"
    optimize_study_name: str = "reii_search"
    # v5: 25 startup con 40 trials dejaba a TPE con n_below=2 (Parzen
    # degenerado). 12 siembra el modelo y deja presupuesto para aprender.
    optimize_n_startup_trials: int = 12
    optimize_multivariate: bool = True
    # v5: resolucion del argmax de EI por trial (default de Optuna: 24).
    # Solo cuesta CPU, no invoca el pipeline.
    optimize_n_ei_candidates: int = 48
    # v5: tope de puntos del frente de etapa 1 que se encolan en etapa 2
    # (x3 copias cada uno); sin tope, el warm start puede comerse el
    # presupuesto entero de la etapa 2.
    optimize_warm_start_max: int = 10
    # v5: congelar los parametros inertes (rho~0) en get_search_space().
    optimize_freeze_inert: bool = True
    optimize_prune_n_startup: int = 15
    # Multi-objective: (ARI↑, coverage↑, −K↑  ≡ K↓)
    optimize_directions: Tuple[str, ...] = ("maximize", "maximize", "minimize")
    # Preferencia sobre el frente de Pareto al seleccionar la solución final
    optimize_preference: str = "max_ari"  # "max_ari" | "max_coverage" | "knee"
    # Gate duro de cobertura (Recomendación A) — prune si cae por debajo.
    # 0.65 era inalcanzable: la estabilidad A1↔A2 real del corpus es
    # 23-52%, así que todos los trials caían en centinela y nunca se
    # escribía best_params.json. 0.25 filtra degenerados sin matar realistas.
    optimize_coverage_gate: float = 0.25
    # Gate duro de cobertura para el modo all (optimizador de etapa 2).
    # Evidencia empírica (mockup edge case C): la cobertura estructural
    # de la triple intersección se estaciona en ~26% bajo ruido realista,
    # con mínimo observado ~20% en corpus de 300 UCEs. Un gate de 0.25
    # sentineliza trials legítimos. 0.10 filtra degenerados extremos
    # (K=1, todos en un cluster) sin matar soluciones reales.
    optimize_coverage_gate_all: float = 0.10

    # Modo del veredicto final de estabilidad:
    #   "all"           → triple intersección (dentro + cross). Default, ALCESTE puro.
    #   "two_of_three"  → 2-de-3 dentro + 2-de-3 cross. SÓLO diagnóstico;
    #                     vulnerable cuando dos métodos comparten sesgo (edge case B2).
    verdict_mode: str = "all"

    # Fuente del cluster_id en el veredicto final:
    #   "wc"        → ancla en word-count tight (comportamiento actual). Default.
    #   "consensus" → por UCE, etiqueta del método con mayor ARI medio contra
    #                 los otros dos (heterogéneo dentro del run). NO ACTIVO aún.
    primary_cluster_source: str = "wc"

    prune_small_clusters: bool = False
    min_class_size: int = 19

    db_local_path: str = WORKFLOW_DB_PATH
    subtlex_df_path: Optional[str] = None
    random_state: int = 42
    corpus_name: str = CORPUS_NAME
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
    #   "all"        A + B + C  (intersección triple: NO es el modo por defecto)
    #   "select"     A | B | C  — elige UN método con la regla `select-v3`
    #                (reii/method_selection.py) y lo persiste en
    #                data/method_selection.json. Sin intersecciones.
    #   Aliases: "auto"→"select", "both"→"wc_coref", "sim_only"→"coref_only",
    #            "wc_sim"→"wc_coref", "sim_emb"→"coref_emb"
    classification_mode: str = "wc_only"

    # ── Regla de selección de método (`select`, select-v3) ──────────────
    # Pre-registrada: viabilidad (K en rango, coverage UTIL ≥ floor) → max ARI
    # intra-método → desempate por coverage útil → aviso de casi-empate.
    # coverage útil = coverage (estables) − tiny (estables en clases menores
    # que min_class_share·n), con denominador n_total para todos los métodos.
    method_selection_margin: float = 0.02
    method_selection_coverage_floor: float = 0.15

    #: Modo del contraste contra el nulo: "diagnostic" (default) lo imprime
    #: como control y no filtra. Ver docs/design_v5/STABILITY_NULL.md.
    method_selection_null_mode: str = "diagnostic"
    #: Presupuesto del motor de estabilidad (STABILITY_NULL.md §6).
    method_selection_stability_boot: int = 10
    method_selection_stability_perm: int = 50
    method_selection_stability_min_n: int = 30
    #: Opt-in del bootstrap real de estabilidad (submuestra + reajuste)
    #: que alimenta ``DoubleClassifier._stability_spaces``. Con False no
    #: se construye ningun espacio y ``_stability_blocks`` devuelve
    #: ``{}``: comportamiento identico al anterior (la corrida congelada
    #: no cambia). Ver docs/design_v5/BOOTSTRAP_WIRING.md.
    stability_bootstrap: bool = False
    method_selection_k_bounds: Tuple[int, int] = (3, 20)
    method_selection_min_class_share: float = 0.01
    method_selection_noise_max: float = 0.50
    # Gate de no dominancia: si la clase mayor ocupa mas de esta fraccion del
    # CORPUS (c = max_class_share * (1 - noise_frac), particion cruda), la
    # particion es degenerada (una clase + satelites) y el metodo se descarta.
    # Medir sobre el corpus y no sobre lo asignado evita que el ruido de
    # HDBSCAN decida el veredicto (ver docs/design_v5/GATE_NOISE.md).
    method_selection_max_class_share: float = 0.70
    optimize_coverage_gate_select: float = 0.10

    # Method C — HDBSCAN on retrofitted UCE embedding space
    #
    # min_cluster_size solía ser un entero absoluto (5 / 3), igual en
    # cualquier corpus. Methods A y B lo escalan al corpus vía
    # min_cluster_size_cdh (fracción de n) y por eso producen K~7-16 sin
    # importar n. Method C con un absoluto fijo sobre n=5809 producía
    # K~243-534 clusters (>50% de ruido, persistence=1.0 en todos): una
    # granularidad ~20-70x más fina que A/B, así que casi nada podía
    # coincidir entre métodos (intersección triple ≈0.2%). Ahora se
    # escala igual que A/B: min_cluster_size = frac * n_embeddable,
    # con un piso absoluto para corpus chicos. Dejar *_frac en None
    # restaura el comportamiento absoluto anterior (compatibilidad con
    # configuraciones o best_params.json guardados antes de este cambio).
    # ([hdbscan-scale-patch])
    hdbscan_uce_min_cluster_frac: Optional[float] = 0.08  # tight pass
    hdbscan_uce_min_cluster_frac_loose: Optional[float] = 0.05  # loose pass
    hdbscan_uce_min_cluster_floor: int = 5  # piso absoluto (corpus chicos)
    hdbscan_uce_min_cluster_size: int = 5  # tight pass — override absoluto (frac=None)
    hdbscan_uce_min_cluster_size_loose: int = (
        3  # loose pass — override absoluto (frac=None)
    )
    hdbscan_uce_epsilon: float = 0.0
    # min_samples=1 (el default anterior) es muy permisivo: con min_cluster_size
    # chico produce clusters triviales con persistence≈1.0. None → se deriva de
    # min_samples_ratio; un int explícito sigue siendo un override manual.
    hdbscan_uce_min_samples: Optional[int] = None
    hdbscan_uce_min_samples_ratio: float = (
        0.25  # min_samples ≈ ratio · min_cluster_size
    )
    hdbscan_uce_metric: str = "euclidean"  # "euclidean" | "cosine"

    # ── Method C — backend de clasificación densa (BACKEND_C.md §2, hunk #21)
    # Valores y rol:
    #   "dense_cdh"        bisección best-first por PC1 local con gate r2_dense.
    #                      Controla K, no emite -1, O(n·d²) con d fijo. DEFAULT.
    #   "ward"             AgglomerativeClustering(n_clusters=K, linkage=ward).
    #                      K exacto, sin RNG, O(n²) en memoria: 144 MiB en el
    #                      camino condensado y 275 MiB en el cuadrado a n=6000
    #                      (HARDENING.md §2.2). Guarda: _METHOD_C_WARD_MAX_N.
    #   "kmeans_spherical" L2 + KMeans(n_init=10, random_state=42). K exacto.
    #   "hdbscan"          comportamiento histórico. NO controla K (target_k sólo
    #                      fija min_cluster_size) y emite -1. Diagnostic only
    #                      salvo que method_c_hdbscan_selectable sea True.
    # Valor desconocido: NO aborta la corrida. Se resuelve con
    # method_c_policy.resolve_method_c_backend(source="persisted"), que degrada
    # al DEFAULT_METHOD_C_BACKEND de la política y emite un aviso "[!]".
    # La procedencia (persistida vs interna) y el resto de la regla viven en
    # CLASSIC_MODE.md §6; aquí sólo se declara el campo, no se duplica.
    method_c_backend: str = "dense_cdh"
    # Escape hatch: True devuelve a hdbscan el derecho a competir en `select`.
    # False => diagnostic_only=True y `select-v4` lo descarta por
    # backend == "hdbscan" (SELECT_V4.md §2.1).
    method_c_hdbscan_selectable: bool = False
    # K objetivo de dense_cdh. None => se deriva del plumbing existente:
    # min_cluster_size_cdh = frac = 1/(2K) => K = round(1/(2·frac)), acotado a
    # [2, n//floor] por method_c_policy.effective_target_k. Ver §3.
    method_c_target_k: Optional[int] = None
    # Gates medidos en §6: r2_1d ≈ 2/π sobre ruido => inútil (0.0, apagado);
    # r2_dense discrimina (null 0.036-0.064, real 0.103-0.142) => 0.08.
    dense_cdh_min_r2_1d: float = 0.0
    dense_cdh_min_r2_dense: float = 0.08
    # Pase loose: piso × 0.75, nunca por debajo de
    # hdbscan_uce_min_cluster_floor (§3). En los densos no cambia K.
    dense_cdh_loose_factor: float = 0.75

    # DSN de la base de datos (PostgreSQL). Si None, se deriva de db_local_path.
    db_dsn: Optional[str] = DATABASE_URL
    # Escribir también el JSON de compatibilidad durante la transición.
    dual_write_json: bool = False

    # ── Gramática personalizable (capa de personalización léxica) ──────────
    # content_pos: POS "núcleo" del vocabulario activo. PROPN/X NO se
    # declaran aquí: los gobierna retention_policy (más abajo).
    content_pos: List[str] = field(
        default_factory=lambda: ["NOUN", "VERB", "ADJ", "ADV"]
    )
    # retention_policy: política de formas activas (fidelidad ALCESTE).
    #   "pos_only"             POS núcleo y not is_stop (baseline pre-11,
    #                          10.72 lemas/UCE en este corpus).
    #   "pos_plus_propn"       núcleo + PROPN/X, sin is_stop (13.16).
    #   "tool_adverb_filtered" idem, menos adverbios-herramienta (12.24).
    # None = sin elección programática: manda ia/gramatica.json y, si
    # tampoco hay valor, "tool_adverb_filtered" (default adoptado en la
    # enmienda 2026-10-28). A diferencia de content_pos, este
    # knob NO se pisa en silencio (ver reii.retention_policy).
    retention_policy: Optional[str] = None
    # Caché de tokens por UCE ({forma, lemma, pos, is_stop, is_alpha} de
    # TODOS los tokens), invalidada por corpus_fingerprint. Permite
    # reevaluar políticas sin volver a correr Stanza (~13 s/UCE).
    token_table_path: str = os.path.join(REII_DATA_DIR, "token_table.jsonl")
    use_token_table_cache: bool = True
    # stopwords_extra: palabras tratadas como suplementarias (además de is_stop).
    stopwords_extra: List[str] = field(default_factory=list)
    # expresiones: frases multi-palabra (separadas por espacio) unidas con '_'.
    expresiones: List[str] = field(default_factory=list)
    # palabras_activas: palabra -> POS forzado (fuerza a activa).
    palabras_activas: Dict[str, str] = field(default_factory=dict)
    # palabras_suplementarias: palabras forzadas a suplementarias/marcadores.
    palabras_suplementarias: List[str] = field(default_factory=list)
    # palabras_excluir: palabras eliminadas por completo.
    palabras_excluir: List[str] = field(default_factory=list)
    # sinonimos: variante -> raíz canónica (agrupa variantes bajo una raíz).
    sinonimos: Dict[str, str] = field(default_factory=dict)


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
    doc_id: Optional[int] = None
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
    is_terminal_consolidated: bool = False  # NEW: hoja por falta de significancia
    indices: List[int] = field(default_factory=list)
    children: List["CDHNode"] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "depth": self.depth,
            "n_ucs": self.n_ucs,
            "is_leaf": self.is_leaf,
            "is_terminal_consolidated": self.is_terminal_consolidated,
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
            # Formas analizadas DISTINTAS (semántica de ALCESTE), no el
            # total con repeticiones — misma función que usa construir_ucs
            # (método A), ver reii.seg.forms_merging.
            lemmas_por_item = [u.lemmas for u in doc_uces]
            return formas_distintas(lemmas_por_item, start, end)

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
        # FIX: Convert doc_id to int to match the keys in doc_metadata_map
        # (int(doc_data.get("indice_orden", ...))), causing UCs without doc_id.
        shared_meta = (doc_metadata_map or {}).get(int(doc_id), {})

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


def _as_int(v: Any) -> Any:
    """Convierte a int si es posible; si no, devuelve el valor original.

    En PostgreSQL doc_id se guarda como TEXT, pero en el JSON histórico era
    numérico. Al reconstruir desde la BD intentamos recuperar el int para
    mantener la misma estructura que el JSON.
    """
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


class Database:
    """
    Persistencia híbrida: SQLite (primaria) + JSON (export opcional).

    Durante la transición:
      · `dual_write=True`  → escribe a SQL y a JSON (default)
      · `dual_write=False` → solo SQL

    El contrato público (save_uces, save_ucs, etc.) NO cambia — todo el
    pipeline sigue funcionando sin tocar el orquestador.
    """

    def __init__(self, config: Config, dual_write: bool = True):
        self.config = config
        self.path = config.db_local_path  # JSON (compat)
        self.dsn = config.db_dsn  # PostgreSQL (primaria)
        self.dual_write = dual_write
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # Buffer en memoria: replicamos la interfaz dict anterior
        if os.path.exists(self.path):
            self.data = self._load_json_dict()
        else:
            self.data = self._load_from_db()
        self.doc_metadata = self.data.get("doc_metadata", {})

    # ── Carga inicial ─────────────────────────────────────────────────────
    def _empty_defaults(self) -> Dict[str, Any]:
        """Dict de defaults vacíos (misma forma que el JSON sin datos)."""
        return {
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

    def _load_json_dict(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._empty_defaults()

    def _load_from_db(self) -> Dict[str, Any]:
        """Reconstruye self.data desde PostgreSQL cuando no existe el JSON.

        Devuelve el mismo dict de defaults vacíos que _load_json_dict si la
        BD no está accesible (para no romper el arranque sin conexión).
        """
        defaults = self._empty_defaults()
        try:
            from reii.backend.sql.db import connect

            conn = connect(self.dsn)
            try:
                data: Dict[str, Any] = {}

                # ── documents → doc_metadata + doc_info ────────────────
                doc_rows = conn.execute(
                    "SELECT doc_id, doc_idx, orden, origen, metadata_json "
                    "FROM documents"
                ).fetchall()
                doc_meta: Dict[str, Dict] = {}
                doc_info: Dict[str, Dict] = {}
                for r in doc_rows:
                    did = str(r["doc_id"])
                    try:
                        meta = (
                            json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                        )
                    except (TypeError, ValueError):
                        meta = {}
                    if not isinstance(meta, dict):
                        meta = {}
                    doc_meta[did] = meta
                    doc_info[did] = {
                        "doc_idx": r["doc_idx"],
                        "orden": r["orden"],
                        "origen": r["origen"],
                    }

                # ── uces ───────────────────────────────────────────────
                uce_rows = conn.execute("SELECT * FROM uces").fetchall()
                uces: List[Dict] = []
                for r in uce_rows:
                    did = str(r["doc_id"])
                    dinfo = doc_info.get(did, {})
                    uce: Dict[str, Any] = {
                        "id": r["uce_id"],
                        "doc_id": _as_int(r["doc_id"]),
                        "local_idx": r["local_idx"],
                        "section_id": r["section_id"],
                        "seccion": r["seccion"],
                        "texto": r["texto"],
                        "n_tokens": r["n_tokens"],
                        "cluster_id": r["cluster_id"],
                        "is_stable": bool(r["is_stable"]),
                        "stability_method": r["stability_method"],
                        "is_terminal_consolidated": bool(r["is_terminal_consolidated"]),
                        "projected_cluster_id": r["projected_cluster_id"],
                        "projection_distance": r["projection_distance"],
                        "projection_margin": r["projection_margin"],
                        "projection_ratio": r["projection_ratio"],
                        "phi_score": r["phi_score"],
                    }
                    # Campos lingüísticos (linguistic_json)
                    try:
                        ling = (
                            json.loads(r["linguistic_json"])
                            if r["linguistic_json"]
                            else {}
                        )
                    except (TypeError, ValueError):
                        ling = {}
                    if isinstance(ling, dict):
                        uce.update(ling)
                    # Métricas (metrics_json)
                    try:
                        metrics = (
                            json.loads(r["metrics_json"]) if r["metrics_json"] else {}
                        )
                    except (TypeError, ValueError):
                        metrics = {}
                    if isinstance(metrics, dict):
                        uce.update(metrics)
                    # metadata reconstruida desde documents + la propia UCE
                    uce["metadata"] = {
                        "section_id": r["section_id"],
                        "seccion": r["seccion"],
                        "origen": dinfo.get("origen"),
                        "doc_idx": dinfo.get("doc_idx"),
                        "indice_orden": dinfo.get("orden"),
                    }
                    uces.append(uce)
                data["uces"] = uces

                # ── ucs ───────────────────────────────────────────────
                uc_rows = conn.execute("SELECT * FROM ucs").fetchall()
                ucs: List[Dict] = []
                for r in uc_rows:
                    try:
                        uce_ids = (
                            json.loads(r["uce_ids_json"]) if r["uce_ids_json"] else []
                        )
                    except (TypeError, ValueError):
                        uce_ids = []
                    try:
                        lemmas = (
                            json.loads(r["lemmas_json"]) if r["lemmas_json"] else []
                        )
                    except (TypeError, ValueError):
                        lemmas = []
                    ucs.append(
                        {
                            "id": r["uc_id"],
                            "doc_id": _as_int(r["doc_id"]),
                            "texto": r["texto"],
                            "cluster_id": r["cluster_id"],
                            "uce_ids": uce_ids,
                            "lemmas": lemmas,
                        }
                    )
                data["ucs"] = ucs

                # ── doc_metadata ───────────────────────────────────────
                data["doc_metadata"] = doc_meta

                # ── terminos ───────────────────────────────────────────
                term_rows = conn.execute("SELECT * FROM terms").fetchall()
                terminos: List[Dict] = []
                for r in term_rows:
                    terminos.append(
                        {
                            "termino": r["term"],
                            "cluster": r["cluster_id"],
                            "frecuencia_global": r["frecuencia_global"],
                            "frecuencia_cluster": r["frecuencia_cluster"],
                            "chi2_yates": r["chi2_yates"],
                            "p_valor": r["p_valor"],
                            "p_adj": r["p_adj"],
                            "phi": r["phi"],
                            "cramer_v": r["cramer_v"],
                            "C": r["c_value"],
                            "significativo": bool(r["significativo"]),
                            "asignado_estricto": bool(r["asignado_estricto"]),
                        }
                    )
                data["terminos"] = terminos

                # ── network ───────────────────────────────────────────
                edge_rows = conn.execute(
                    "SELECT src, dst, weight FROM network_edges"
                ).fetchall()
                nodes: List[str] = []
                seen: Set[str] = set()
                edges: List[list] = []
                for r in edge_rows:
                    src, dst = r["src"], r["dst"]
                    if src not in seen:
                        seen.add(src)
                        nodes.append(src)
                    if dst not in seen:
                        seen.add(dst)
                        nodes.append(dst)
                    edges.append([src, dst, {"weight": r["weight"]}])
                data["network"] = {"nodes": nodes, "edges": edges, "partition": []}

                # ── kv_store (config y resto de claves) ────────────────
                kv_rows = conn.execute(
                    "SELECT key, value_json FROM kv_store"
                ).fetchall()
                for r in kv_rows:
                    try:
                        data[r["key"]] = json.loads(r["value_json"])
                    except (TypeError, ValueError):
                        data[r["key"]] = r["value_json"]

                # Claves de defaults ausentes en la BD
                for k, v in defaults.items():
                    data.setdefault(k, v)
                return data
            finally:
                conn.close()
        except Exception as e:
            logger.warning("No se pudo cargar desde PostgreSQL: %s", e)
            return defaults

    # ── Guarda anti-borrado accidental ───────────────────────────────────
    def _guard_against_ucs_network_wipe(self) -> None:
        """Evita que un _save() borre ucs/network_edges en PostgreSQL.

        Si el buffer en memoria (JSON) tiene 0 ucs (o red vacía) pero
        PostgreSQL ya tiene filas, un _save() escribiría el JSON con 0 ucs y
        migrate_network haría DELETE FROM network_edges → borrado accidental.
        Esto ocurre típicamente con --skip-classify sobre un JSON que ya
        perdió sus ucs. Lanzamos RuntimeError antes de escribir nada; el
        usuario debe re-ejecutar con clasificación o limpiar las tablas
        explícitamente.
        """
        ucs_count = len(self.data.get("ucs", []) or [])
        net = self.data.get("network") or {}
        net_edges = len(net.get("edges", [])) if isinstance(net, dict) else 0
        if ucs_count > 0 and net_edges > 0:
            return
        try:
            from reii.backend.sql.db import connect

            conn = connect(self.dsn)
            try:
                pg_ucs = conn.execute("SELECT count(*) AS count FROM ucs").fetchone()[
                    "count"
                ]
                pg_edges = conn.execute(
                    "SELECT count(*) AS count FROM network_edges"
                ).fetchone()["count"]
            finally:
                conn.close()
        except Exception:
            # BD inaccesible: no bloqueamos el save; la migración
            # reportará el error real.
            return
        problems = []
        if ucs_count == 0 and pg_ucs > 0:
            problems.append(f"ucs: JSON=0 pero PostgreSQL={pg_ucs}")
        if net_edges == 0 and pg_edges > 0:
            problems.append(f"network_edges: JSON=0 pero PostgreSQL={pg_edges}")
        if problems:
            raise RuntimeError(
                "Refusing to save: posible borrado accidental de "
                + "; ".join(problems)
                + ". Re-ejecuta con clasificación (sin --skip-classify) o "
                "limpia explícitamente las tablas antes de continuar."
            )

    # ── Persistencia ──────────────────────────────────────────────────────
    def _save(self):
        self._guard_against_ucs_network_wipe()
        if self.dual_write:
            self.data["doc_metadata"] = self.doc_metadata
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
            os.replace(tmp, self.path)
        # SQL: reconstrucción total a partir de self.data (idempotente)
        try:
            from reii.backend.migrate import (
                migrate_annotations,
                migrate_clusters,
                migrate_documents,
                migrate_kv,
                migrate_network,
                migrate_terms,
                migrate_uces,
                migrate_ucs,
            )
            from reii.backend.sql.db import connect

            conn = connect(self.dsn)
            try:
                migrate_documents(conn, self.data)
                migrate_uces(conn, self.data)
                migrate_ucs(conn, self.data)
                migrate_clusters(conn, self.data)
                migrate_terms(conn, self.data)
                migrate_annotations(conn, self.data)
                migrate_network(conn, self.data)
                migrate_kv(conn, self.data)
            finally:
                conn.close()
        except Exception as e:
            logger.error("SQLite save failed: %s", e)
            raise

    # ── API pública (contrato sin cambios) ────────────────────────────────
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
        # Reemplazo total (no upsert): save_ucs siempre recibe el set completo
        # de UCs del run. El upsert dejaba UCs obsoletas en el JSON cuando la
        # clasificación producía menos UCs que el run anterior.
        self.data["ucs"] = [u.to_dict() for u in ucs]

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
            section_id = d.get("metadata", {}).get("section_id", "0")
            if doc_id is not None and local_idx is not None:
                d["id"] = f"{doc_id}_{section_id}_{local_idx}"
            else:
                import uuid

                d["id"] = str(uuid.uuid4())
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

        # ── Gramática personalizable (capa de personalización léxica) ──────
        # Carga el JSON (por defecto ia/gramatica.json) y sobreescribe los
        # campos de Config con los valores resueltos. Los valores también se
        # guardan como atributos de instancia para que lematizar_uces los use.
        gramatica = load_gramatica()
        # Precedencia del knob (RETENTION_POLICY.md §2.0):
        #   Config.retention_policy > ia/gramatica.json > "tool_adverb_filtered"
        # A diferencia de content_pos (que load_gramatica pisa SIEMPRE),
        # una elección programática explícita aquí no se pierde.
        self.retention_policy = resolve_retention_policy(
            getattr(config, "retention_policy", None),
            gramatica.get("retention_policy"),
        )
        self.config.retention_policy = self.retention_policy
        self.config.content_pos = policy_content_pos(
            self.retention_policy, gramatica.get("content_pos", config.content_pos)
        )
        self.config.stopwords_extra = list(gramatica.get("stopwords_extra", []))
        self.config.expresiones = list(gramatica.get("expresiones", []))
        self.config.palabras_activas = dict(gramatica.get("palabras_activas", {}))
        self.config.palabras_suplementarias = list(
            gramatica.get("palabras_suplementarias", [])
        )
        self.config.palabras_excluir = list(gramatica.get("palabras_excluir", []))
        self.config.sinonimos = dict(gramatica.get("sinonimos", {}))

        # Valores resueltos (sets/dicts) para uso directo en lematizar_uces.
        self.content_pos = set(self.config.content_pos)
        self.stopwords_extra = set(self.config.stopwords_extra)
        self.expresiones = list(self.config.expresiones)
        self.palabras_activas = dict(self.config.palabras_activas)
        self.palabras_suplementarias = set(self.config.palabras_suplementarias)
        self.palabras_excluir = set(self.config.palabras_excluir)
        self.sinonimos = dict(self.config.sinonimos)
        self.tool_adverb_lemmas = normalize_lemmas(
            gramatica.get("tool_adverb_lemmas") or TOOL_ADVERB_LEMMAS
        )
        self.token_table = TokenTableWriter(
            getattr(config, "token_table_path", None),
            _corpus_fingerprint(),
            enabled=bool(getattr(config, "use_token_table_cache", True)),
        )

        self.stemmer = (
            SnowballStemmer("spanish") if config.stem_backend == "snowball" else None
        )
        self.doc_metadata_map = {}

    def _segmentar_reinert_spacy(self, doc, ideal_size: Optional[int] = None) -> List:
        """Segment a full document Doc into UCE spans with global char offsets.

        ideal_size : si se pasa, anula ``self.config.uce_target_size`` SOLO para
          esta llamada (usado por la auto-calibración para probar varios
          targets sobre el mismo ``doc`` ya parseado, sin repetir spaCy).
        """
        if len(doc) == 0:
            return []

        if ideal_size is None:
            ideal_size = getattr(self.config, "uce_target_size", 12)
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

    def muestra_para_calibracion(
        self, corpus_raw, max_textos: int = 80, seed: int = 42
    ) -> List:
        """Parsea con spaCy una muestra de los bloques por-sección del corpus
        (ya agrupados como los verá segmentar_en_uces), UNA sola vez, para
        reutilizarla en todas las mediciones de medir_promedio_real — cada
        target de uce_target_size que se prueba solo repite el corte (barato,
        sin spaCy), nunca el parseo (caro).

        Si el corpus tiene <= max_textos bloques, se usan todos (sin
        muestreo). Con más, se toma una muestra aleatoria con semilla fija
        para que la calibración sea reproducible.
        """
        textos: List[str] = []
        for doc_data in corpus_raw.values():
            for _seccion, texto in agrupar_segmentos_por_seccion(
                doc_data.get("texto_segmentos", [])
            ):
                if texto.strip():
                    textos.append(texto)

        if len(textos) > max_textos:
            textos = random.Random(seed).sample(textos, max_textos)

        return [self.nlp(t) for t in textos]

    def medir_promedio_real(
        self, docs_muestra: List, target: int
    ) -> Tuple[float, float, int]:
        """Promedio, desvío estándar y tamaño de muestra de palabras/UCE que
        produce `target` como uce_target_size, sobre los documentos YA
        parseados en docs_muestra (ver muestra_para_calibracion).

        Usado como measure_fn de reii.seg.uce_calibration.calibrate: la
        firma (target) -> (media, std, n) es exactamente la que esa función
        espera. Reutiliza el mismo filtro min_uce_words que segmentar_en_uces,
        para que la medición sea fiel a lo que de verdad va a producir el
        corte real.
        """
        largos: List[int] = []
        for doc in docs_muestra:
            for span in self._segmentar_reinert_spacy(doc, ideal_size=target):
                n = len(span.text.split())
                if n >= self.config.min_uce_words:
                    largos.append(n)

        if not largos:
            return (0.0, 0.0, 0)
        arr = np.array(largos, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 1.0
        return (float(arr.mean()), std, len(arr))

    def segmentar_en_uces(self, corpus_raw):
        uces_por_doc = []
        self.doc_metadata_map = {}
        self.section_registry = {}

        for doc_idx, (origen_key, doc_data) in enumerate(corpus_raw.items()):
            # Usar indice_orden (índice original del archivo) como doc_idx:
            # si el run filtra documentos ya procesados, el enumerate()
            # cambiaría los índices y rompería la correspondencia con la DB.
            doc_idx = int(doc_data.get("indice_orden", doc_idx))
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

            # Antes: se parseaba y cortaba CADA seg por separado, así que
            # cualquier seg de <=ideal_size+window tokens se volvía UNA UCE
            # entera sin importar qué tan corto fuera (ver la guarda
            # `remaining <= ideal_size+window` en _segmentar_reinert_spacy),
            # generando una "cola" forzada por cada segmento de la entrevista.
            # ALCESTE concatena STC libremente dentro de todo el u.c.i., sin
            # ese límite artificial. Ahora se concatena todo el texto de una
            # misma sección (misma semántica que ya usaba section_registry
            # para agrupar, solo que antes NO se usaba para concatenar el
            # texto, solo para el id) y se corta una sola vez por sección.
            for seccion, seg_text in agrupar_segmentos_por_seccion(segmentos):
                if seccion not in section_registry:
                    section_registry[seccion] = local_section_counter
                    local_section_counter += 1
                section_id = section_registry[seccion]

                self.section_registry[f"{doc_idx}_{section_id}"] = {
                    "origen": origen_key,
                    "seccion": seccion,
                }

                # NOTA: start_char/end_char quedan relativos a este texto
                # concatenado por sección (como antes quedaban relativos al
                # seg_text individual) — no son offsets dentro de
                # texto_completo_txt ni antes ni ahora.
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
        largos = [len(u.texto.split()) for d in uces_por_doc for u in d]
        promedio = (sum(largos) / len(largos)) if largos else 0.0
        objetivo = getattr(self.config, "uce_target_size", None)
        print(
            f"   {total} UCEs en {len(uces_por_doc)} documentos. "
            f"Promedio real: {promedio:.1f} palabras/UCE "
            f"(objetivo uce_target_size={objetivo})."
        )
        return uces_por_doc, self.doc_metadata_map

    def lematizar_uces(self, uces: List[UCE]) -> List[UCE]:
        stemmer = self.stemmer
        for uce in uces:
            doc = self.nlp(uce.texto)
            # Tabla completa de tokens (TODOS, también PUNCT): es la caché
            # que permite reevaluar políticas sin volver a correr Stanza.
            self.token_table.add(
                getattr(uce, "id", None),
                token_rows(doc),
                words=len(uce.texto.split()),
            )
            tokens_raw = [
                (i, t.text, t.lemma_.lower(), t.pos_, t.is_stop, t.is_alpha)
                for i, t in enumerate(doc)
            ]
            lemmas, stems, pos_tags, formas_tokens = [], [], [], []
            marcadores = []
            bigrams, bigram_stems = [], []
            trigrams, trigram_stems = [], []
            all_tokens = []

            # ── 1. Expresiones (frases multi-palabra) ─────────────────────
            # Detecta frases del usuario en el texto y las une con '_' para
            # que entren al vocabulario como una sola unidad (misma convención
            # que bigramas/trigramas). Los tokens consumidos se excluyen del
            # bucle individual para no duplicarlos. Se permiten nombres propios.
            consumed_indices = set()
            for expr in self.expresiones:
                expr_words = expr.lower().split()
                if not expr_words:
                    continue
                n = len(expr_words)
                for start in range(len(tokens_raw) - n + 1):
                    window = tokens_raw[start : start + n]
                    if all(
                        (w[2] == ew) or (w[1].lower() == ew)
                        for w, ew in zip(window, expr_words)
                    ):
                        joined_lemma = "_".join(w[2] for w in window)
                        joined_stem = "_".join(
                            (stemmer.stem(w[2]) if stemmer else w[2]) for w in window
                        )
                        lemmas.append(joined_lemma)
                        stems.append(joined_stem)
                        pos_tags.append("PROPN")
                        formas_tokens.append(
                            {
                                "forma": " ".join(w[1] for w in window),
                                "lemma": joined_lemma,
                                "stem": joined_stem,
                                "pos": "PROPN",
                                "pos_idx": window[0][0],
                            }
                        )
                        for w in window:
                            consumed_indices.add(w[0])
                        break

            # ── 2-5. Clasificación individual + reasignación + sinonimia ──
            for i, text, lemma, pos, is_stop, is_alpha in tokens_raw:
                if i in consumed_indices:
                    continue

                text_lower = text.lower()
                lemma_lower = lemma.lower()

                # 2. palabras_excluir: eliminar por completo.
                if (
                    text_lower in self.palabras_excluir
                    or lemma_lower in self.palabras_excluir
                ):
                    continue

                if is_alpha and pos != "PUNCT":
                    all_tokens.append(text)

                # 2. Reasignación individual.
                forced_pos = self.palabras_activas.get(
                    text_lower, self.palabras_activas.get(lemma_lower)
                )
                is_forced_supp = (
                    text_lower in self.palabras_suplementarias
                    or lemma_lower in self.palabras_suplementarias
                )

                # 3. stopwords_extra: tratar como suplementaria.
                is_extra_stop = (
                    text_lower in self.stopwords_extra
                    or lemma_lower in self.stopwords_extra
                )

                # 5. Sinonimia: variante -> raíz canónica.
                resolved_lemma = self.sinonimos.get(
                    lemma_lower, self.sinonimos.get(text_lower, lemma)
                )

                # 4. Clasificación activa/suplementaria.
                if forced_pos is not None:
                    stem = stemmer.stem(resolved_lemma) if stemmer else resolved_lemma
                    lemmas.append(resolved_lemma)
                    stems.append(stem)
                    pos_tags.append(forced_pos)
                    formas_tokens.append(
                        {
                            "forma": text,
                            "lemma": resolved_lemma,
                            "stem": stem,
                            "pos": forced_pos,
                            "pos_idx": i,
                        }
                    )
                else:
                    # Una sola fuente de verdad para la clasificación:
                    # la política decide is_stop, PROPN/X y la lista de
                    # adverbios-herramienta (ver retention_policy.py).
                    _decision = token_decision(
                        policy=self.retention_policy,
                        pos=pos,
                        lemma=resolved_lemma,
                        is_stop=is_stop,
                        is_alpha=is_alpha,
                        content_pos=self.content_pos,
                        is_forced_supp=is_forced_supp,
                        is_extra_stop=is_extra_stop,
                        tool_adverb_lemmas=self.tool_adverb_lemmas,
                    )
                    if _decision == "marker":
                        marcadores.append(
                            {"forma": text.lower(), "lemma": resolved_lemma, "pos": pos}
                        )
                    elif _decision == "active":
                        stem = (
                            stemmer.stem(resolved_lemma) if stemmer else resolved_lemma
                        )
                        lemmas.append(resolved_lemma)
                        stems.append(stem)
                        pos_tags.append(pos)
                        formas_tokens.append(
                            {
                                "forma": text,
                                "lemma": resolved_lemma,
                                "stem": stem,
                                "pos": pos,
                                "pos_idx": i,
                            }
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
        """Concatena UCEs consecutivas de un mismo documento hasta acumular
        >= min_forms formas (lemas) analizadas DISTINTAS — semántica de
        ALCESTE ("mínimo de formas analizadas distintas por UC"), no el
        total con repeticiones. El agrupamiento en sí (qué UCEs van en cada
        UC) lo decide `agrupar_secuencial_por_umbral`, probado aparte en
        tests/test_forms_merging.py; aquí solo se arma el objeto UC por
        cada grupo que devuelve.
        """
        ucs: List[UC] = []
        uce_to_uc: Dict[str, str] = {}

        for doc_uces in uces_por_doc:
            if not doc_uces:
                continue
            doc_id = doc_uces[0].doc_id
            shared_meta = (
                (doc_metadata_map or {}).get(int(doc_id), {})
                if doc_id is not None
                else {}
            )
            lemmas_por_uce = [uce.lemmas for uce in doc_uces]
            for start, end in agrupar_secuencial_por_umbral(lemmas_por_uce, min_forms):
                grupo = doc_uces[start : end + 1]
                uc = UC(
                    texto=" ".join(u.texto for u in grupo),
                    doc_id=doc_id,
                    metadata={**shared_meta, **doc_uces[0].metadata.copy()},
                    uce_ids=[u.id for u in grupo],
                    lemmas=[lema for u in grupo for lema in u.lemmas],
                    stems=[s for u in grupo for s in u.stems],
                    bigrams=[b for u in grupo for b in u.bigrams],
                    bigram_stems=[b for u in grupo for b in u.bigram_stems],
                    trigrams=[t for u in grupo for t in u.trigrams],
                    trigram_stems=[t for u in grupo for t in u.trigram_stems],
                )
                ucs.append(uc)
                for u in grupo:
                    uce_to_uc[u.id] = uc.id
                    u.uc_id = uc.id

        print(f"   {len(ucs)} UCs with ≥{min_forms} forms (distintas).")
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
        model = _get_sentence_embedder(self.config.embedding_model_name)
        embs = model.encode([uc.texto for uc in ucs], show_progress_bar=True)
        for uc, emb in zip(ucs, embs):
            uc.embedding = emb.tolist()
        return ucs

    def generar_embeddings_uces(self, uces: List[UCE]) -> List[UCE]:
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            print("   SentenceTransformer no disponible.")
            return uces
        model = _get_sentence_embedder(self.config.embedding_model_name)
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
        term_embeddings = Vt.T * np.sqrt(Sigma)  # (n_terms, n_components)
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
        """
        Poda explícita en dos pasadas:
        1. Frecuencia documental  (f_j^doc >= tsj)
        2. Frecuencia absoluta    (f_j      >= min_term_abs_freq)

        Sustituye la suma de un pseudoconteo global. Elimina hapax y términos
        ultra-raros en lugar de rellenarlos con ε (que densificaba la matriz y
        deformaba las masas marginales del AFC).
        """
        doc_freq = Counter()
        raw_freq = Counter()
        for obj in objects:
            terms = list(self._iter_terms(obj))
            doc_freq.update(set(terms))
            raw_freq.update(terms)

        min_doc = self.config.tsj
        min_abs = max(1, getattr(self.config, "min_term_abs_freq", 2))

        vocab = [
            t
            for t, cnt in doc_freq.items()
            if cnt >= min_doc and raw_freq[t] >= min_abs
        ]
        return sorted(vocab)

    def construir_matriz(self, objects: List, vocabulario: List[str]) -> np.ndarray:
        idx = {t: i for i, t in enumerate(vocabulario)}
        mat = np.zeros((len(objects), len(vocabulario)), dtype=int)
        for i, obj in enumerate(objects):
            for term in set(self._iter_terms(obj)):  # Binary presence/absence
                if term in idx:
                    mat[i, idx[term]] = 1
        return mat

    def construir_matriz_dispersa(self, objects: List, vocabulario: List[str]):
        """
        Construye la matriz binaria UC × términos directamente en formato CSR.

        Construimos vía COO (una pasada, sin mutación) y convertimos a CSR en
        una sola operación O(nnz log nnz). El antiguo DOK iteraba celda por
        celda, lo que saturaba memoria y era órdenes de magnitud más lento.
        """
        from scipy.sparse import coo_matrix

        idx = {t: i for i, t in enumerate(vocabulario)}
        rows: List[int] = []
        cols: List[int] = []
        for i, obj in enumerate(objects):
            for term in set(self._iter_terms(obj)):
                j = idx.get(term)
                if j is not None:
                    rows.append(i)
                    cols.append(j)

        data = np.ones(len(rows), dtype=np.int32)
        mat = coo_matrix(
            (data, (rows, cols)),
            shape=(len(objects), len(vocabulario)),
            dtype=np.int32,
        )
        return mat.tocsr()

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


def _debug_emb_matrix(
    mat: np.ndarray, zero_rows: int, n_total: int, label: str = "pre-PCA"
) -> None:
    """Diagnóstico de REII_EMB_DEBUG=1: ¿la matriz tiene estructura real?

    No cambia ningún comportamiento — solo imprime. Pensado para responder
    una pregunta concreta cuando Method C da clusters=0 en toda la grilla:
    ¿es que no hay estructura de densidad en estos vectores, o se rompió
    algo antes de llegar a HDBSCAN (duplicados, varianza colapsada, etc)?
    """
    import hdbscan as _hdbscan_mod

    n, d = mat.shape
    print(f"   [EMB-DEBUG {label}] shape={n}x{d}, zero_rows={zero_rows}/{n_total}")

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    unit = mat / norms
    sim = unit @ unit.T
    np.fill_diagonal(sim, 0.0)
    near_dup_pairs = int((sim > 0.999).sum() // 2)
    print(f"   [EMB-DEBUG {label}] pares casi-duplicados (cos>0.999): {near_dup_pairs}")

    var_per_dim = mat.var(axis=0)
    dead_dims = int((var_per_dim < 1e-6).sum())
    print(
        f"   [EMB-DEBUG {label}] dims con varianza~0: {dead_dims}/{d} · "
        f"varianza total={var_per_dim.sum():.4f}"
    )

    from scipy.spatial.distance import pdist

    d_sample = (
        mat
        if n <= 2000
        else mat[np.random.default_rng(42).choice(n, 2000, replace=False)]
    )
    dists = pdist(d_sample, metric="euclidean")
    p5, p50, p95 = np.percentile(dists, [5, 50, 95])
    print(
        f"   [EMB-DEBUG {label}] distancias euclid. p5={p5:.3f} "
        f"p50={p50:.3f} p95={p95:.3f} (ratio p95/p5={p95 / max(p5, 1e-9):.2f})"
    )

    probe = _hdbscan_mod.HDBSCAN(min_cluster_size=5, min_samples=3, metric="euclidean")
    labels = probe.fit_predict(mat)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    max_pers = (
        max(probe.cluster_persistence_) if len(probe.cluster_persistence_) else 0.0
    )
    print(
        f"   [EMB-DEBUG {label}] sonda min_cluster_size=5 → clusters={n_clusters}, "
        f"noise={(labels == -1).sum()}/{n}, max_persistence={max_pers:.4f}"
    )


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
        # Los vectores retrofitted están indexados por stems (stem_backend !=
        # "none"), no por lemmas; usar stems con fallback a lemmas.
        tokens = uce.stems or uce.lemmas or []
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
        tokens = uc.stems or uc.lemmas or []
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
        _k_cap = max(1, n_total // max(1, self._min_cluster_size))
        print(
            f"   [CDH] n={n_total} · min_cluster_size={self._min_cluster_size} "
            f"({self.config.min_cluster_size_cdh:.0%} of corpus) · K_max~={_k_cap}"
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

        # Invariante del piso: ninguna clase terminal debe quedar por debajo
        # de min_cluster_size (salvo que una rama no alcance 2*floor UCs).
        sizes = np.bincount(labels[labels >= 0])
        if sizes.size:
            floor = self._min_cluster_size
            small = int((sizes < floor).sum())
            print(
                f"   [CDH] clases={sizes.size} · tamano min={int(sizes.min())} "
                f"max={int(sizes.max())} mediana={int(np.median(sizes))}"
            )
            if small and getattr(self.config, "cdh_enforce_min_leaf", True):
                print(
                    f"   [!] {small}/{sizes.size} clases por debajo del piso "
                    f"({floor}): la rama no llega a 2*floor UCs"
                )
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
        # Sin pseudoconteo: preservar masas marginales reales.
        # Las filas/columnas sin masa se clavan a 1.0 SOLO para evitar 0/0 en
        # la iteración de promediado recíproco; no contaminan las no-cero.
        row_sums = np.asarray(mat.sum(axis=1)).flatten().reshape(-1)
        col_sums = np.asarray(mat.sum(axis=0)).flatten().reshape(-1)
        total = row_sums.sum()
        if total <= 0:
            return np.zeros(n)

        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        col_sums = np.where(col_sums > 0, col_sums, 1.0)

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

    def _corte_optimo(
        self, coord: np.ndarray, min_side: int = 1
    ) -> Tuple[float, np.ndarray]:
        """Corte optimo 1-D con piso. Delegado a la funcion pura.

        El cuerpo se extrajo a `method_c_policy.corte_optimo_1d` (misma
        matematica, misma reversion, mismo orden de desempate) para que el
        backend `dense_cdh` use EXACTAMENTE el mismo corte que el CDH de A.
        `min_side > 1` evita cortes 1-vs-(n-1).
        """
        return _corte_optimo_1d(coord, min_side=min_side)

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

    def _permutation_pvalue(
        self,
        coord: np.ndarray,
        labels: np.ndarray,
        n_perm: int = 1000,
        alpha: float = 0.05,
    ) -> Tuple[float, bool]:
        """
        Test de permutación empírico sobre el split observado.

        H0: las etiquetas son intercambiables; el R² observado no es mayor
            que el que produciría una asignación aleatoria de las mismas
            masas a los dos grupos.

        Devuelve (p_valor, significativo).
        """
        n = len(coord)
        n1 = int((labels == 1).sum())
        n0 = n - n1
        if n < 4 or n0 == 0 or n1 == 0:
            return 1.0, False

        grand = coord.mean()
        ss_total = float(np.sum((coord - grand) ** 2))
        if ss_total <= 0:
            return 1.0, False

        m0 = coord[labels == 0].mean()
        m1 = coord[labels == 1].mean()
        ss_b_obs = n0 * (m0 - grand) ** 2 + n1 * (m1 - grand) ** 2
        r2_obs = ss_b_obs / ss_total

        rng = np.random.default_rng(self.config.random_state)
        n_ge = 0
        for _ in range(n_perm):
            perm_idx = rng.permutation(n)
            perm_in1 = np.zeros(n, dtype=bool)
            perm_in1[perm_idx[:n1]] = True
            pm0 = coord[~perm_in1].mean()
            pm1 = coord[perm_in1].mean()
            ss_b_perm = n0 * (pm0 - grand) ** 2 + n1 * (pm1 - grand) ** 2
            if ss_b_perm / ss_total >= r2_obs:
                n_ge += 1

        p_value = (n_ge + 1) / (n_perm + 1)
        return p_value, p_value < alpha

    def _es_significativo(
        self,
        coord: np.ndarray,
        labels: np.ndarray,
        var_obs: float,
    ) -> bool:
        """
        Gate de significancia:
          1. Piso de R² (descarta splits triviales baratos)
          2. Test de permutación (reemplaza la aproximación chi² determinista)
        """
        if var_obs < self.config.min_r2_threshold:
            return False

        n_perm = int(getattr(self.config, "n_perm_cdh", 1000))
        if n_perm <= 0:
            # Fallback determinista si el usuario desactiva permutaciones
            n = len(coord)
            chi2_approx = n * var_obs
            return chi2_approx >= self.config.chi2_threshold_small

        p_value, sig = self._permutation_pvalue(coord, labels, n_perm=n_perm)
        return sig

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
        floor = self._min_cluster_size
        enf_floor = bool(getattr(self.config, "cdh_enforce_min_leaf", True))
        # Con piso duro hacen falta 2*floor UCs para que AMBOS hijos lo cumplan.
        n_min_split = 2 * floor if enf_floor else floor
        if n < n_min_split or depth >= self.config.max_depth_cdh:
            node = CDHNode(
                depth=depth,
                n_ucs=n,
                is_leaf=True,
                is_terminal_consolidated=True,
                indices=indices.tolist(),
            )
            node.label = self._next_leaf()
            return node
        sub_mat = mat_sparse[indices]

        # Slice uc_vectors to this partition's rows — keeps indices aligned
        sub_vecs = uc_vectors[indices] if uc_vectors is not None else None

        coord = self._primer_factor(sub_mat, uc_vectors=sub_vecs)  # ← warm-start
        var_obs, labels = self._corte_optimo(coord, min_side=floor if enf_floor else 1)

        if not self._es_significativo(coord, labels, var_obs):
            node = CDHNode(
                depth=depth,
                n_ucs=n,
                is_leaf=True,
                is_terminal_consolidated=True,  # BLOQUE 2: Clase Terminal Consolidada
                indices=indices.tolist(),
            )
            node.label = self._next_leaf()
            return node

        labels_pre_swap = labels
        if self.config.swap_iterations > 0:
            labels = self._intercambio(
                sub_mat,
                labels,
                max_iter=self.config.swap_iterations,
                uc_vectors=sub_vecs,  # ← hybrid swap
            )
            if enf_floor and int(np.bincount(labels, minlength=2).min()) < floor:
                # El refinamiento chi² desbalanceó el corte por debajo del
                # piso: se conserva la partición del promediado recíproco.
                labels = labels_pre_swap

        if len(np.unique(labels)) < 2:
            node = CDHNode(
                depth=depth,
                n_ucs=n,
                is_leaf=True,
                is_terminal_consolidated=True,
                indices=indices.tolist(),
            )
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
        # Última selección de método (modo "select"), para el optimizador y
        # para el hook de calidad.
        self.last_selection: Optional[Selection] = None
        # Pass intermedio del orquestador (pre-retrofitting): no imprime ni
        # persiste la selección, que aún no ve Method C.
        self.quiet_selection = False
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
        uces_flat = [uce for doc_uces in uces_por_doc_local for uce in doc_uces]
        voc = builder.construir_vocabulario(uces_flat)
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
        uces_flat = [uce for doc_uces in uces_por_doc_local for uce in doc_uces]
        voc = builder.construir_vocabulario(uces_flat)
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
        checkpoint_path: Optional[str] = None,
        force: bool = False,
        precomputed_chains: Optional[
            Dict[str, List]
        ] = None,  # ← NUEVO: {doc_id: chains}
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

        Checkpointing (opcional)
        ────────────────────────
        checkpoint_path : si se pasa, tras cada documento se escribe un JSON con
          los UCs ya construidos, el mapeo uce→uc, los acumuladores de calidad y
          el root_cache. Si el proceso muere (OOM, corte de energía), una
          re-ejecución con el mismo checkpoint_path reanuda saltando los
          documentos ya completados. Con None (default) no hay checkpointing.

          La ruta se deriva por pase: ``checkpoint_path_for_pass`` añade un
          sufijo ``.ctx{context_units}`` para que B1 (context=2) y B2
          (context=4) no se pisen entre sí y cada uno conserve su memoria.

        force : si es True, ignora cualquier checkpoint existente y rehace el
          pase desde cero (sobrescribiendo el archivo).
        """
        # ── 0. Guard: hard fail, no silent fallback ───────────────────────────
        ps = self.progressive_segmenter
        if ps is None:
            print(
                "   [COREF] ProgressiveSegmenter not injected — Method B unavailable."
            )
            return None
        if not STANZA_SERVER_URL and ps.get_stanza() is None:
            print("   [COREF] Stanza unavailable — Method B unavailable.")
            return None

        if STANZA_SERVER_URL:
            print(
                f"   [COREF] Modo remoto (Colab): {STANZA_SERVER_URL} — "
                "coref por HTTP, sin pipeline local."
            )
        else:
            print("   [COREF] Modo local: coref con pipeline Stanza local.")

        # ── 0'. Checkpoint por pase: B1 y B2 no comparten archivo ────────────
        checkpoint_path = checkpoint_path_for_pass(checkpoint_path, context_units)

        # ── 1. Deep-copy + suffix ─────────────────────────────────────────────
        suffix = f"__coref{context_units}"
        uces_por_doc_local = copy.deepcopy(uces_por_doc)
        for doc_uces in uces_por_doc_local:
            for uce in doc_uces:
                uce.id = normalizar_id_uce(uce)
                uce.id = f"{uce.id}{suffix}"

        # ── 2. Root subjects (lazy, per document) ─────────────────────────────
        #       The greedy merge (3d) only looks up the TAIL UCE of each pre-merged
        #       group, so roots are computed right after the pre-merge (3a') for
        #       those UCEs only, through a cache shared by the B1 and B2 passes.
        #       (Before: one Stanza call per UCE per pass, keyed by suffixed id.)
        root_cache: Dict[str, set] = {}
        # OJO: NO usar ``or`` aquí. ``RootSubjectCache`` define ``__len__``
        # (devuelve len(self._store)), así que una caché VACÍA es falsy y el
        # ``or`` crearía una caché nueva SIN el extractor por lotes, matando el
        # batching (0 por lote, N individuales). Hay que comprobar None explícito.
        rcache = getattr(ps, "root_cache", None)
        if rcache is None:
            rcache = RootSubjectCache(ps.find_subjects_for_roots)
        _calls0 = rcache.misses

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

        # ── 3'. Checkpointing: reanudar desde un checkpoint previo ──────────
        processed_doc_ids: List[str] = []
        processed_doc_set: Set[str] = set()
        # Huella del CONTENIDO de entrada (corpus, uce_target_size, pre-merge):
        # sin ella, otro corpus/segmentación con la misma ruta se reanudaba en silencio.
        input_fp = uces_fingerprint(
            uces_por_doc_local,
            extra={
                k: getattr(ps, k, None)
                for k in (
                    "similarity_threshold",
                    "max_depth",
                    "window_size",
                    "min_content_tokens",
                    "min_segment_chars",
                    "stanza_lang",
                )
            },
        )
        guard = CheckpointGuard()
        _coref_hits = 0  # docs servidos desde el caché de cadenas
        _coref_misses = 0  # docs extraídos con Stanza en ESTA llamada
        _ckpt_reasons: List[str] = []
        if checkpoint_path and not force:
            ckpt = _cargar_checkpoint_coref(
                checkpoint_path,
                context_units=context_units,
                min_forms=min_forms,
                input_fingerprint=input_fp,
                reason_sink=_ckpt_reasons,
            )
            if ckpt is not None:
                processed_doc_ids = ckpt["processed_doc_ids"]
                processed_doc_set = set(processed_doc_ids)
                ucs = [UC.from_dict(d) for d in ckpt["ucs"]]
                uce_to_uc = dict(ckpt["uce_to_uc"])
                global_uc_counter = ckpt["global_uc_counter"]
                total_boundaries = ckpt["total_boundaries"]
                coref_merges = ckpt["coref_merges"]
                covered_boundaries = ckpt["covered_boundaries"]
                total_chains = ckpt["total_chains"]
                for ctx, subjects in ckpt["root_cache"].items():
                    rcache._store[ctx] = frozenset(subjects)
                print(
                    f"   [COREF] Checkpoint cargado ({checkpoint_path}): "
                    f"{len(processed_doc_ids)} docs ya procesados, "
                    f"{len(ucs)} UCs — reanudando."
                )
            else:
                for _r in _ckpt_reasons:
                    print(f"   [COREF] ⚠ {_r}")
                print(
                    f"   [COREF] No hay checkpoint válido en {checkpoint_path} — "
                    "empezando de cero."
                )
        elif checkpoint_path and force:
            print(
                f"   [COREF] --force: ignorando checkpoint en {checkpoint_path} — "
                "rehaciendo desde cero."
            )

        for doc_uces in uces_por_doc_local:
            if not doc_uces:
                continue
            doc_id = doc_uces[0].doc_id

            # Reanudar: saltar documentos ya completados (checkpointing)
            if doc_id in processed_doc_set:
                continue

            # ── 3a. Pre-merge UCEs via recursive + progressive segmentation ──
            #       This mirrors segmentador.py's pipeline: sentences →
            #       recursive_segmentation → final_clustering → coref.
            #       Each resulting group becomes the atomic segment for coref.
            if len(doc_uces) > 1:
                try:
                    # premerge_cached es bit-equivalente a
                    # final_clustering(recursive_segmentation(texts)); el
                    # caché por contenido evita recomputar en B2, Pass 2 y
                    # trials del optimizador. Ver ProgressiveSegmenter.
                    prog_seg = ps.premerge_cached(doc_uces)
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

            # ── 3a''. Guard: pre-merge colapsado a 1 grupo ──
            #       Un documento entero en un solo grupo hace que
            #       _extract_global_chains procese todo el documento de una vez
            #       (OOM de RAM). Con muchas UCEs se cae a grupos individuales.
            if len(premerged_groups) == 1 and len(doc_uces) > 50:
                print(
                    f"   [COREF] Pre-merge colapsó a 1 grupo ({len(doc_uces)} UCEs) — "
                    "usando UCEs individuales."
                )
                premerged_groups = [[u] for u in doc_uces]

            # ── 3a'. Root subjects for the tails of the pre-merged groups only ──
            _tails = boundary_tails(premerged_groups)
            rcache.prefetch(
                [root_context(u.texto) for u in _tails],
                label=f"ctx={context_units} ",
            )
            for u in _tails:
                root_cache[u.id] = rcache.get(root_context(u.texto))

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
            #       (print, no _dprint: debug_coref=False en rerun_classification
            #       silenciaba este paso y el crash quedaba invisible)
            print(
                f"   [COREF] Extrayendo cadenas globales para doc {doc_id} "
                f"({len(premerged_groups)} grupos)..."
            )
            doc_incomplete = False
            try:
                # get_doc_chains tolera int/str: un cache recargado de disco trae
                # claves str y `doc_id in precomputed_chains` fallaba siempre,
                # re-extrayendo el coref (Stanza) en cada trial.
                _pre = (
                    get_doc_chains(precomputed_chains, doc_id)
                    if precomputed_chains is not None
                    else None
                )
                if _pre is not None:
                    global_chains = _pre
                    _coref_hits += 1
                else:
                    _coref_misses += 1
                    global_chains = ps._extract_global_chains(
                        segments_info,
                        full_doc_text,
                        context_units=context_units,
                    )
                    if getattr(ps, "last_extraction_incomplete", False):
                        doc_incomplete = True
            except Exception as e:
                logger.warning(
                    "COREF chain extraction failed for doc %s: %s",
                    doc_uces[0].doc_id,
                    e,
                )
                print(f"   [COREF] ⚠ extracción falló en doc {doc_id}: {e}")
                global_chains = []
                doc_incomplete = True
            print(
                f"   [COREF] Cadenas globales extraídas: {len(global_chains)} "
                f"para doc {doc_id}."
            )
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
            shared_meta = (doc_metadata_map or {}).get(int(doc_id), {})

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

            # ── 3g. Checkpoint: persistir progreso tras cada documento ──
            #       El dict de estado se construye DENTRO de un try/except:
            #       antes se evaluaba como argumento de _guardar_checkpoint_coref
            #       (fuera de su try/except), así que un fallo de serialización
            #       (p. ej. un campo no serializable en uc.to_dict()) rompía el
            #       pipeline sin mensaje.
            if doc_incomplete:
                _first = guard.note_incomplete(doc_id)
                print(
                    f"   [COREF] ⚠ doc {doc_id}: extracción de cadenas INCOMPLETA "
                    "(Stanza/servidor no disponible); sus UCs usan cadenas parciales."
                )
                if checkpoint_path and _first:
                    print(
                        "   [COREF] ⚠ checkpoint CONGELADO en este doc: no se persiste "
                        "como completo; al reanudar se rehace desde aquí."
                    )
            if checkpoint_path and guard.can_save:
                processed_doc_ids.append(doc_id)
                processed_doc_set.add(doc_id)
                try:
                    state = {
                        "version": COREF_CHECKPOINT_VERSION,
                        "context_units": context_units,
                        "min_forms": min_forms,
                        "code_fingerprint": _coref_code_fingerprint(),
                        "input_fingerprint": input_fp,
                        "processed_doc_ids": processed_doc_ids,
                        "ucs": [uc.to_dict() for uc in ucs],
                        "uce_to_uc": uce_to_uc,
                        "global_uc_counter": global_uc_counter,
                        "total_boundaries": total_boundaries,
                        "coref_merges": coref_merges,
                        "covered_boundaries": covered_boundaries,
                        "total_chains": total_chains,
                        "root_cache": {
                            ctx: list(subjects)
                            for ctx, subjects in rcache._store.items()
                        },
                    }
                except Exception as e:
                    logger.warning(
                        "No se pudo construir el checkpoint %s: %s",
                        checkpoint_path,
                        e,
                    )
                    guard.build_failures += 1
                    if guard.build_failures == 1:
                        print(
                            f"   [COREF] ⚠ No se pudo CONSTRUIR el checkpoint: {e} "
                            "(no se guardará progreso hasta que se corrija)."
                        )
                else:
                    if _guardar_checkpoint_coref(checkpoint_path, state):
                        print(
                            f"   [COREF] Checkpoint guardado ({checkpoint_path}): "
                            f"{len(processed_doc_ids)} docs, {len(ucs)} UCs."
                        )
                    else:
                        guard.write_failures += 1
                        if guard.write_failures == 1:
                            print(
                                f"   [COREF] ⚠ NO se pudo escribir el checkpoint "
                                f"({checkpoint_path}); el progreso de este pase no está a salvo."
                            )

        print(
            f"   [COREF context={context_units}] roots: "
            f"{rcache.misses - _calls0} Stanza calls for {total_boundaries} boundaries "
            f"({len(rcache)} contexts cached)"
        )

        print(
            f"   [COREF context={context_units}] cadenas: {_coref_hits} docs desde "
            f"caché, {_coref_misses} extraídos con Stanza en esta llamada."
        )
        for _line in guard.summary():
            print(f"   [COREF] ⚠ {_line}")

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
        uces_flat = [uce for doc_uces in uces_por_doc_local for uce in doc_uces]
        voc = builder.construir_vocabulario(uces_flat)
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
    def _resolve_hdbscan_uce_params(
        self, n_embeddable: int, variant: str
    ) -> Tuple[int, int]:
        """Deriva (min_cluster_size, min_samples) para Method C, escalados a
        ``n_embeddable`` igual que ``min_cluster_size_cdh`` escala A/B. ([hdbscan-scale-patch])

        variant: "tight" o "loose". Con *_frac en None, reproduce el
        comportamiento absoluto anterior (compatibilidad hacia atrás).
        """
        if variant not in ("tight", "loose"):
            raise ValueError(f"variant debe ser 'tight' o 'loose', no {variant!r}")
        floor = max(2, int(getattr(self.config, "hdbscan_uce_min_cluster_floor", 5)))
        if variant == "tight":
            frac = getattr(self.config, "hdbscan_uce_min_cluster_frac", None)
            absolute = getattr(self.config, "hdbscan_uce_min_cluster_size", 5)
        else:
            frac = getattr(self.config, "hdbscan_uce_min_cluster_frac_loose", None)
            absolute = getattr(self.config, "hdbscan_uce_min_cluster_size_loose", 3)
        if frac is not None:
            min_cluster_size = max(floor, round(float(frac) * n_embeddable))
        else:
            # frac=None → modo absoluto legacy: NO se aplica el piso, para
            # reproducir exactamente el comportamiento anterior al patch de
            # escalado (hdbscan_uce_min_cluster_size / _loose tal cual).
            min_cluster_size = int(absolute)

        min_samples = getattr(self.config, "hdbscan_uce_min_samples", None)
        if min_samples is None:
            ratio = float(getattr(self.config, "hdbscan_uce_min_samples_ratio", 0.25))
            min_samples = max(2, round(ratio * min_cluster_size))
        else:
            min_samples = max(1, int(min_samples))
        return min_cluster_size, min_samples

    def _clasificar_umbral_embedding(
        self,
        uces_por_doc: List[List[UCE]],
        min_cluster_size: int,
        retrofitted_vectors: Dict[str, np.ndarray],
        min_samples: Optional[
            int
        ] = None,  # ([hdbscan-scale-patch]) resuelto por el caller
        *,
        variant: str = "tight",  # "tight" | "loose": solo cambia el piso
    ) -> Optional[Dict]:
        """
        Method C: clasificacion densa de las UCEs en espacio retrofitted.

        El backend lo elige `Config.method_c_backend` (hunk #21) y lo resuelve
        `method_c_policy` (hunk #72): PREFLIGHT, guarda de memoria de ward,
        acotado de K y validacion de salida NO se reimplementan aqui.

        No UC objects are created here; cluster labels are assigned per-UCE.
        The result dict carries 'uce_to_label' instead of 'ucs'/'labels'
        (BACKEND_C.md §1).
        """
        requested = getattr(self.config, "method_c_backend", "dense_cdh")
        backend, _notice = resolve_method_c_backend(requested, source="persisted")
        if _notice:
            print(f"   {_notice['message']}")
        if backend == "hdbscan" and not _HDBSCAN_AVAILABLE:
            print("   [EMB] hdbscan not installed - skipping Method C.")
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
        embeddable_mask = ~np.all(mat == 0, axis=1)
        zero_rows = int((~embeddable_mask).sum())
        if zero_rows > len(all_uces) * 0.5:
            print(
                f"   [EMB] {zero_rows}/{len(all_uces)} zero-vectors — "
                "too many OOV UCEs, skipping Method C."
            )
            return None

        # ── DEBUG: diagnóstico de la matriz antes de PCA/HDBSCAN ──────────
        # Actívalo con REII_EMB_DEBUG=1. No cambia nada del comportamiento
        # normal — solo imprime estadísticas para decidir si clusters=0 es
        # "no hay estructura real" o "algo rompió la matriz antes de llegar
        # aquí" (duplicados, varianza colapsada, etc).
        if os.environ.get("REII_EMB_DEBUG") == "1":
            _debug_emb_matrix(mat, zero_rows, len(all_uces))

        # ── PCA antes de HDBSCAN ──────────────────────────────────────────
        # mat vive en el espacio de fastText (~300 dims). Con corpus chicos
        # (cientos de UCEs, no miles) eso es maldición de la dimensionalidad
        # pura para un clusterer de densidad: las distancias se vuelven casi
        # uniformes y HDBSCAN no encuentra estructura sin importar
        # min_cluster_size (clusters=0, noise=todo, cualquier frac).
        # Reducimos a un número de componentes acorde al tamaño real de la
        # muestra antes de clusterizar.
        n_samples, n_dims = mat.shape
        target_dims = max(2, min(30, n_samples // 10, n_dims))
        if target_dims < n_dims:
            from sklearn.decomposition import PCA

            _pca = PCA(n_components=target_dims, random_state=42)
            mat = _pca.fit_transform(mat)
            print(
                f"   [EMB] PCA {n_dims}→{target_dims} dims "
                f"({n_samples} UCEs) antes de HDBSCAN."
            )
            if os.environ.get("REII_EMB_DEBUG") == "1":
                var_pct = 100 * _pca.explained_variance_ratio_.sum()
                print(f"   [EMB-DEBUG] PCA explica {var_pct:.1f}% de la varianza.")
                _debug_emb_matrix(mat, zero_rows, n_samples, label="post-PCA")

        cfg = self.config

        # Piso de hoja: el mcs del pase (ya escalado al corpus) es el piso de
        # los backends densos; el pase loose lo reduce por
        # dense_cdh_loose_factor, nunca por debajo del floor absoluto (§3).
        _floor_min = int(getattr(cfg, "hdbscan_uce_min_cluster_floor", 5))
        floor = max(_floor_min, int(round(float(min_cluster_size))))
        if variant == "loose":
            _factor = float(getattr(cfg, "dense_cdh_loose_factor", 0.75))
            floor = max(_floor_min, int(round(floor * _factor)))

        # K pedido: explícito o recuperado del plumbing existente
        # (min_cluster_size_cdh = 1/(2K) => K = round(1/(2*frac))).
        target_k = getattr(cfg, "method_c_target_k", None)
        if target_k is None:
            _frac = getattr(cfg, "min_cluster_size_cdh", None)
            _frac = 0.10 if _frac is None else float(_frac)
            target_k = int(round(1.0 / (2.0 * max(1e-6, _frac))))

        # Unica llamada a los backends: las guardas viven en la politica.
        labels, _diag = guarded_fit(
            backend,
            mat,
            int(target_k),
            floor,
            n_zero_rows=zero_rows,
            ward_max_n=_METHOD_C_WARD_MAX_N,
            min_r2_1d=float(getattr(cfg, "dense_cdh_min_r2_1d", 0.0)),
            min_r2_dense=float(getattr(cfg, "dense_cdh_min_r2_dense", 0.08)),
            # mcs_frac reproduce el min_cluster_size del pase para hdbscan.
            mcs_frac=float(min_cluster_size) / float(max(1, len(all_uces))),
        )
        for _msg in _diag.get("notices", []):
            print(f"   {_msg}")

        if labels is None:
            _be = str(_diag.get("backend_used", backend))
            # I3 (HARDENING §9): target_k -> n_clusters tambien cuando no hay
            # particion. guarded_fit ya dejo n_clusters en diag.
            _ncls = _diag.get("n_clusters")
            if _be == "dense_cdh" and _ncls is not None:
                _rej = int(
                    (_diag.get("backend_diagnostics") or {}).get(
                        "rejected_min_r2_dense", 0
                    )
                )
                _k_notice = format_target_k_notice(int(target_k), int(_ncls), _rej)
                if _k_notice:
                    print(f"   {_k_notice}")
            print(
                f"   [!] Method C backend={_be} sin particion "
                f"({_diag['precondition']['reason']})"
            )
            return None

        # hdbscan no controla K y emite -1: se computa, no compite, salvo
        # escape hatch (BACKEND_C.md §2, hunk #22.5).
        _be = str(_diag["backend_used"])
        diagnostic_only = bool(
            _be == "hdbscan"
            and not bool(getattr(cfg, "method_c_hdbscan_selectable", False))
        )

        result = build_method_c_result(
            labels,
            _diag,
            uce_ids,
            mat,
            int(target_k),
            int(floor),
            requested=str(requested),
            diagnostic_only=diagnostic_only,
            resolution_notice=(_notice["message"] if _notice else ""),
        )
        if result is None:
            return None
        # [stability] UCEs con embedding real del pase tight: el bootstrap
        # de C remuestrea solo esta subpoblacion (las OOV colapsan en un
        # unico punto tras el PCA y falsearian el ARI).
        result["embeddable_mask"] = embeddable_mask
        print(
            f"   [EMB backend={result['backend']}] UCEs={len(all_uces)}, "
            f"clusters={result['n_clusters']}, noise={result['n_noise']}, "
            f"K_pedido={result['target_k']}"
            + (", diagnostic_only" if diagnostic_only else "")
        )
        return result

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
        _q_strategy = os.environ.get(
            "REII_STABILITY_MAPPING", "hungarian"
        )  # [quality-patch] mapping
        if _q_strategy != "hungarian":
            from reii.quality.mapping import hungarian_compatible

            return hungarian_compatible(
                uce_to_ca, uce_to_cb, strategy=_q_strategy, label=label
            )
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
    # ═══════════════════════════════════════════════════════════════════
    # Proyección liminal — OPERACIÓN ABDUCTIVA (post-inductiva)
    # ═══════════════════════════════════════════════════════════════════
    # Filosofía:
    #   El pipeline inductivo (CHD + triple intersección) define las clases.
    #   Las UCEs que no sobreviven NO son parte del corpus analítico. Esta
    #   función las PROYECTA sobre centroides ya consolidados (nunca
    #   recalcula φ, AFC, inercia ni estabilidad). Escribe en campos
    #   separados, dejando uce.cluster_id = None.
    #
    #   Invariante respetado:
    #     · df_terms         → calculado solo sobre uces_est_list
    #     · afc_result       → calculado solo sobre uces_est_list
    #     · forma_index      → calculado solo sobre uces_est_list
    #     · centroides       → calculados solo sobre uces_est_list
    #     · uce_phi          → solo para uces_est_list
    #
    #   Ninguna línea de código inductivo consulta projected_cluster_id.
    # ═══════════════════════════════════════════════════════════════════
    def _project_liminal_uces(
        self,
        uces_por_doc: List[List[UCE]],
        primary_stable: List[UCE],
        voc: List[str],
    ) -> Dict[str, Any]:
        """
        Proyecta UCEs huérfanas sobre los centroides de las clases
        consolidadas, por distancia χ² en el espacio de perfiles fila.

        Definiciones:
            perfil_uce_i   = v_i / Σ_j v_ij              (row-normalized)
            centroide_k    = mean_i∈k (perfil_uce_i)     (media de perfiles)
            c_j            = masa marginal global del término j
            d²_χ²(i, k)    = Σ_j (perfil_ij − centroide_kj)² / c_j

        Salida (por UCE huérfana, vía setattr):
            projected_cluster_id : int    — argmin_k d²_χ²
            projection_distance  : float  — d²_χ² al ganador
            projection_margin    : float  — d²_χ²(2º) − d²_χ²(1º)
            projection_ratio     : float  — d²_χ²(2º) / d²_χ²(1º)

        Si projection_ratio < liminal_projection_min_ratio, la UCE queda
        SIN proyección (ambigua entre dos o más clases; no forzamos).

        Retorna un dict de métricas para logging y persistencia.
        """
        from scipy.sparse import issparse

        summary: Dict[str, Any] = {
            "n_candidates": 0,
            "n_projected": 0,
            "n_ambiguous": 0,
            "n_empty_profile": 0,
            "class_histogram": {},
        }
        if not primary_stable or not voc:
            return summary

        stable_ids = {u.id for u in primary_stable}
        labels_stable = np.array(
            [u.cluster_id for u in primary_stable if u.cluster_id is not None]
        )
        classes = np.unique(labels_stable[labels_stable >= 0])
        if len(classes) < 2:
            print(
                "   [Projection] <2 clases consolidadas — no se puede "
                "proyectar huérfanas."
            )
            return summary

        # ── Centroides en espacio de perfiles fila ─────────────────────────
        builder = MatrizBuilder(self.config)
        mat_stable = builder.construir_matriz_dispersa(primary_stable, voc)
        mat_dense = (
            mat_stable.toarray().astype(np.float64)
            if issparse(mat_stable)
            else np.asarray(mat_stable, dtype=np.float64)
        )

        n_terms = len(voc)
        class_centroids: Dict[int, np.ndarray] = {}
        # Solo UCEs con cluster_id válido entran al centroide
        valid_mask = np.array([u.cluster_id is not None for u in primary_stable])
        for k in classes:
            class_mask = valid_mask & (
                np.array([u.cluster_id for u in primary_stable]) == k
            )
            if not class_mask.any():
                continue
            sub = mat_dense[class_mask]
            row_sums = sub.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            profiles = sub / row_sums
            class_centroids[int(k)] = profiles.mean(axis=0)

        if len(class_centroids) < 2:
            print("   [Projection] Centroides insuficientes — abortando.")
            return summary

        # ── Marginal global por término (denominador χ²) ────────────────────
        col_mass = mat_dense.sum(axis=0)
        col_mass = np.where(col_mass > 0, col_mass, 1e-12)
        global_marginal = col_mass / col_mass.sum()

        voc_idx = {t: j for j, t in enumerate(voc)}
        min_ratio = float(getattr(self.config, "liminal_projection_min_ratio", 1.10))
        class_ids_sorted = sorted(class_centroids.keys())

        for doc_uces in uces_por_doc:
            for uce in doc_uces:
                # Solo huérfanas: no estables, sin cluster_id asignado
                if uce.id in stable_ids:
                    continue
                if uce.cluster_id is not None:
                    continue

                summary["n_candidates"] += 1

                # Vector binario alineado al vocabulario
                vec = np.zeros(n_terms, dtype=np.float64)
                for term in set(self._iter_terms_for_uce(uce)):
                    j = voc_idx.get(term)
                    if j is not None:
                        vec[j] = 1.0

                s = vec.sum()
                if s == 0:
                    summary["n_empty_profile"] += 1
                    continue
                profile = vec / s

                # ── Distancias a cada centroide ─────────────────────────────
                d2 = np.empty(len(class_ids_sorted), dtype=np.float64)
                for idx_k, k in enumerate(class_ids_sorted):
                    diff = profile - class_centroids[k]
                    d2[idx_k] = float(np.sum(diff * diff / global_marginal))

                order = np.argsort(d2)
                best_k = class_ids_sorted[int(order[0])]
                d2_best = float(d2[order[0]])

                # ── Ambigüedad: si solo hay 1 clase no aplica ───────────────
                if len(class_ids_sorted) >= 2:
                    d2_second = float(d2[order[1]])
                    # d2_second siempre ≥ d2_best por construcción
                    ratio = d2_second / d2_best if d2_best > 1e-12 else float("inf")
                    margin = d2_second - d2_best
                else:
                    ratio = float("inf")
                    margin = float("inf")

                if ratio < min_ratio:
                    summary["n_ambiguous"] += 1
                    # Marcar el intento, pero SIN proyección definitiva
                    setattr(uce, "projected_cluster_id", None)
                    setattr(uce, "projection_distance", None)
                    setattr(uce, "projection_margin", None)
                    setattr(uce, "projection_ratio", float(ratio))
                    continue

                # ── Proyección aceptada ─────────────────────────────────────
                setattr(uce, "projected_cluster_id", int(best_k))
                setattr(uce, "projection_distance", d2_best)
                setattr(uce, "projection_margin", float(margin))
                setattr(uce, "projection_ratio", float(ratio))

                summary["n_projected"] += 1
                k_str = str(int(best_k))
                summary["class_histogram"][k_str] = (
                    summary["class_histogram"].get(k_str, 0) + 1
                )

        # ── Guardarraíl: invariante inductivo intacto ───────────────────────
        # Ninguna UCE con projected_cluster_id puede tener también
        # cluster_id. Si esto se rompe, la fase inductiva quedó contaminada.
        contaminated = [
            uce.id
            for doc in uces_por_doc
            for uce in doc
            if getattr(uce, "projected_cluster_id", None) is not None
            and uce.cluster_id is not None
            and uce.id not in stable_ids
        ]
        if contaminated:
            raise RuntimeError(
                f"[Projection] INVARIANTE ROTO: {len(contaminated)} UCEs "
                f"tienen cluster_id Y projected_cluster_id. Ejemplo: "
                f"{contaminated[:3]}. Abortando para evitar contaminación."
            )

        print(
            f"   [Projection] {summary['n_projected']}/"
            f"{summary['n_candidates']} UCEs proyectadas "
            f"({summary['n_ambiguous']} ambiguas, "
            f"{summary['n_empty_profile']} sin perfil léxico). "
            f"Nota: NO son parte del corpus inductivo."
        )

        return summary

    def _iter_terms_for_uce(self, uce):
        """Réplica local de MatrizBuilder._iter_terms para una UCE suelta."""
        use_stems = self.config.stem_backend != "none"
        for s in uce.stems if use_stems else uce.lemmas:
            yield s
        if self.config.use_bigrams:
            for b in uce.bigram_stems if use_stems else uce.bigrams:
                yield "_".join(b)
        if self.config.use_trigrams:
            for t in uce.trigram_stems if use_stems else uce.trigrams:
                yield "_".join(t)

    #: Techo de celdas de la matriz densa que usa el bootstrap de A/B. Con
    #: n=528 UCEs y ~5k terminos son ~2.6M celdas (~21 MiB); el techo solo
    #: evita que un vocabulario anomalo densifique cientos de MiB.
    _STABILITY_MAX_DENSE_CELLS = 20_000_000

    def _build_stability_spaces(
        self,
        res_wc: Optional[Dict],
        res_coref: Optional[Dict],
        res_emb: Optional[Dict],
        *,
        min_n: int = 30,
    ) -> Optional[Dict[str, Tuple[Any, List[str], Dict[str, int], Any]]]:
        """Espacios ``(X, ids, cluster_of, predict_fn)`` del bootstrap.

        Opt-in: con ``config.stability_bootstrap`` en False devuelve ``None``
        y ``_stability_blocks`` recibe ``None`` -> ``{}``, que es el
        comportamiento previo byte a byte. Con True se intenta cada metodo por
        separado: el que no se pueda construir se OMITE (su ``bootstrap_ari``
        queda ``nan`` = "no disponible" y la regla no descarta por eso) y se
        imprime ``[stability] metodo X sin espacio: motivo``. Nunca lanza.

        Las unidades son las de la particion PRIMARIA del metodo; NO se
        re-segmenta el corpus:

          * A/B: las UCs del pase tight, con la MISMA matriz UC x terminos
            (binaria -> PPMI) que consumio el CDH; ``predict_fn`` reajusta el
            CDH sobre la submuestra de UCs conservando el piso absoluto.
          * C: las UCEs con embedding del pase tight, en las coordenadas PCA
            del ajuste; ``predict_fn`` reajusta el backend denso con el mismo
            ``target_k`` y el piso re-escalado a esa subpoblacion.
        """
        if not bool(getattr(self.config, "stability_bootstrap", False)):
            return None

        import contextlib
        import dataclasses
        import io

        try:
            from scipy.sparse import csr_matrix
        except Exception as exc:  # pragma: no cover - scipy es dependencia
            print(f"   [stability] sin espacios: {exc}")
            return {}

        spaces: Dict[str, Tuple[Any, List[str], Dict[str, int], Any]] = {}

        def _omit(method: str, reason: object) -> None:
            print(f"   [stability] metodo {method} sin espacio: {reason}")

        def _term_space(res: Optional[Dict]):
            if not res:
                raise RuntimeError("sin resultado del pase tight")
            ucs = res.get("ucs") or []
            voc = res.get("voc") or []
            labels = res.get("labels")
            if not ucs or not voc or labels is None:
                raise RuntimeError("resultado sin ucs/voc/labels")
            if len(ucs) != len(labels):
                raise RuntimeError(f"ucs={len(ucs)} != labels={len(labels)}")
            mat = MatrizBuilder(self.config).construir_matriz_dispersa(ucs, voc)
            if self.ppmi_builder is not None:
                mat = self.ppmi_builder.build(mat, voc)
            n_cells = int(mat.shape[0]) * int(mat.shape[1])
            if n_cells > self._STABILITY_MAX_DENSE_CELLS:
                raise RuntimeError(
                    f"matriz {mat.shape[0]}x{mat.shape[1]}={n_cells} celdas "
                    f"> {self._STABILITY_MAX_DENSE_CELLS}"
                )
            X = np.asarray(
                mat.toarray() if hasattr(mat, "toarray") else mat, dtype=float
            )
            ids = [str(uc.id) for uc in ucs]
            cluster_of = {uid: int(labels[i]) for i, uid in enumerate(ids)}
            n_assigned = sum(1 for v in cluster_of.values() if v >= 0)
            if n_assigned < int(min_n):
                raise RuntimeError(f"asignadas={n_assigned} < min_n={int(min_n)}")
            # Piso ABSOLUTO del ajuste completo: se conserva en cada replica
            # (mismo K efectivo), no la fraccion del submuestreo.
            floor = max(
                5,
                int(len(ids) * float(self.config.min_cluster_size_cdh)),
            )

            def predict(X_sub):
                n_sub = int(np.asarray(X_sub).shape[0])
                # +0.5 para que int(n_sub * frac) == floor pese al redondeo
                # en coma flotante (`clasificar` trunca, no redondea).
                frac_sub = (float(floor) + 0.5) / float(max(1, n_sub))
                cfg_sub = dataclasses.replace(
                    self.config, min_cluster_size_cdh=frac_sub
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    labels_sub, _tree = ClasificadorDescendente(cfg_sub).clasificar(
                        csr_matrix(X_sub)
                    )
                return labels_sub

            return (X, ids, cluster_of, predict)

        for method, res in (("A", res_wc), ("B", res_coref)):
            try:
                spaces[method] = _term_space(res)
            except Exception as exc:
                _omit(method, exc)
        try:
            if not res_emb:
                raise RuntimeError("sin resultado del pase tight")
            X_emb = res_emb.get("embedding_matrix")
            labels_arr = res_emb.get("labels_array")
            ids_all = list(res_emb.get("uce_ids") or [])
            backend = str(res_emb.get("backend") or "")
            if X_emb is None or labels_arr is None:
                raise RuntimeError("resultado sin embedding_matrix/labels_array")
            if backend not in ("dense_cdh", "ward", "kmeans_spherical"):
                raise RuntimeError(f"backend={backend!r} no controla K")
            A = np.asarray(X_emb, dtype=float)
            y = np.asarray(labels_arr, dtype=int).ravel()
            if A.ndim != 2 or A.shape[0] != len(ids_all):
                raise RuntimeError(f"matriz {A.shape} != uce_ids={len(ids_all)}")
            if y.size != len(ids_all):
                raise RuntimeError(f"labels={y.size} != uce_ids={len(ids_all)}")
            mask = res_emb.get("embeddable_mask")
            if mask is None:
                raise RuntimeError("resultado sin embeddable_mask")
            mask = np.asarray(mask, dtype=bool).ravel()
            if mask.size != len(ids_all):
                raise RuntimeError(
                    f"embeddable_mask={mask.size} != uce_ids={len(ids_all)}"
                )
            ids = [ids_all[i] for i in np.flatnonzero(mask)]
            A = A[mask]
            y = y[mask]
            n_assigned = int((y >= 0).sum())
            if n_assigned < int(min_n):
                raise RuntimeError(
                    f"embeddable={len(ids)} asignadas={n_assigned} < min_n={int(min_n)}"
                )
            target_k = int(res_emb.get("target_k") or 0)
            floor_full = int(res_emb.get("min_cluster_size") or 0)
            floor_min = int(getattr(self.config, "hdbscan_uce_min_cluster_floor", 5))
            # Misma FRACCION de piso que el ajuste completo, aplicada a las
            # UCEs informativas: el K efectivo se conserva y el piso no
            # depende del tamano de la replica.
            frac_floor = float(floor_full) / float(max(1, len(ids_all)))
            floor = max(floor_min, int(round(frac_floor * len(ids))))
            min_r2_1d = float(getattr(self.config, "dense_cdh_min_r2_1d", 0.0))
            min_r2_dense = float(getattr(self.config, "dense_cdh_min_r2_dense", 0.08))
            mcs_frac = float(floor) / float(max(1, len(ids)))

            def predict_c(X_sub):
                with contextlib.redirect_stdout(io.StringIO()):
                    labels_sub, diag = guarded_fit(
                        backend,
                        np.asarray(X_sub, dtype=float),
                        target_k,
                        floor,
                        n_zero_rows=0,
                        ward_max_n=_METHOD_C_WARD_MAX_N,
                        min_r2_1d=min_r2_1d,
                        min_r2_dense=min_r2_dense,
                        mcs_frac=mcs_frac,
                    )
                if labels_sub is None:
                    reason = (diag.get("precondition") or {}).get("reason", "?")
                    raise RuntimeError(f"guarded_fit sin particion ({reason})")
                return labels_sub

            spaces["C"] = (
                A,
                ids,
                {uid: int(y[i]) for i, uid in enumerate(ids)},
                predict_c,
            )
        except Exception as exc:
            _omit("C", exc)

        return spaces

    def _stability_blocks(
        self,
        spaces: Optional[Dict[str, Tuple[Any, List[str], Dict[str, int], Any]]],
        *,
        n_boot: int = 10,
        n_perm: int = 50,
        seed: int = 42,
        min_n: int = 30,
    ) -> Dict[str, Dict]:
        """``{metodo: bloque}`` con el ARI bootstrap y el nulo de cada metodo.

        ``spaces[m] = (X, uce_ids, cluster_of_uce, predict_fn)``. Un metodo sin
        espacio produce ``{}``: sus claves caen a ``nan`` y la regla no lo
        descarta por falta de dato (STABILITY_NULL.md §4). Nunca lanza.
        """
        from reii.quality.metrics import bootstrap_stability, null_partition_test

        out: Dict[str, Dict] = {}
        for method, spec in (spaces or {}).items():
            try:
                X, uce_ids, cluster_of, predict_fn = spec
                A = np.asarray(X, dtype=float)
                if A.ndim != 2 or A.shape[0] != len(uce_ids):
                    out[method] = {}
                    continue
                y = np.asarray([int(cluster_of.get(u, -1)) for u in uce_ids], dtype=int)
                if int((y >= 0).sum()) < int(min_n) or not callable(predict_fn):
                    out[method] = {}
                    continue
                boot = bootstrap_stability(
                    A,
                    y,
                    predict_fn=predict_fn,
                    n_boot=int(n_boot),
                    seed=int(seed),
                    min_n=int(min_n),
                )
                nul = null_partition_test(
                    A, y, n_perm=int(n_perm), seed=int(seed), min_n=int(min_n)
                )
                block = dict(boot)
                block["ids"] = list(uce_ids)
                block.update(
                    null_p=nul.get("null_p", float("nan")),
                    null_stat=nul.get("null_stat", float("nan")),
                    null_n_perm=int(nul.get("null_n_perm", n_perm)),
                    null_z=nul.get("null_z", float("nan")),
                )
                out[method] = block
            except Exception:
                out[method] = {}
        return out

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
        # fracción de UCEs sin asignar en el pase tight de cada método
        noise_wc: float = 0.0,
        noise_coref: float = 0.0,
        noise_emb: float = 0.0,
        # tamanos de clase de la particion CRUDA de cada metodo (sin -1).
        # Solo los usa el modo 'select' para el gate de no dominancia.
        partition_sizes: Optional[Dict[str, List[int]]] = None,
        # Salida de ``_stability_blocks`` (STABILITY_NULL.md §5). ``None`` o
        # ``{}`` => todo cae a nan y la seleccion se comporta como antes.
        stability_blocks: Optional[Dict[str, Dict]] = None,
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
          If EMB is unavailable (empty stable_emb), "all" collapses to "wc_coref".

        Selected mode ("select"):
        ──────────────────────────
        No hay intersección. Se mide cada método por separado (K, ARI
        intra-método, coverage, ruido) y ``select_method`` elige UNO con la
        regla pre-registrada ``select-v3``. El ganador es el que decide el
        ``primary_cluster``; los demás solo aportan métricas comparativas.
        """
        all_uids = {uce.id for doc in uces_por_doc for uce in doc}
        n_total = len(all_uids)
        _stab = stability_blocks or {}
        selection: Optional[Selection] = None

        # ── Resolve primary set and cluster reference ─────────────────────
        emb_available = bool(stable_emb)

        _mode = mode
        if _mode in ("both", "wc_sim"):
            # Alias histórico: el nombre canónico que run() espera más abajo
            # (en la selección de primary_res, ~línea 5612) es "wc_coref",
            # no "wc_sim". run() ya hace mode = _ALIASES.get(mode, mode)
            # ANTES de llamar a este método, así que normalmente nunca
            # llegamos acá con "wc_sim"/"both" — pero la degradación de
            # abajo (EMB unavailable) reintroducía "wc_sim" como su propio
            # fallback interno, inconsistente con esa canonicalización. El
            # mismatch hacía que run() cayera en el branch
            # `else: primary_res = None` (ninguna de las dos tuplas de
            # effective_mode incluye "wc_sim") y disparara
            # "[!] Fatal: primary_res unavailable" — aun con UCEs
            # primary-stable reales y no vacías (ver "Primary output: 510"
            # en el summary, impreso ANTES de este fallo).
            _mode = "wc_coref"
        if _mode == "all" and not emb_available:
            print("   [Verdict] EMB unavailable — degrading 'all' to 'wc_coref'.")
            _mode = "wc_coref"

        if _mode == "wc_only":
            primary_ids = set(stable_wc)
            primary_cluster = stable_wc

        elif _mode in ("coref_only", "sim_only"):
            primary_ids = set(stable_sim)
            primary_cluster = stable_sim

        elif _mode == "emb_only":
            primary_ids = set(stable_emb)
            primary_cluster = stable_emb

        elif _mode == "select":
            # ── Un solo método, elegido por regla pre-registrada ─────────
            _stable_by_method = {
                "A": stable_wc,
                "B": stable_sim,
                "C": stable_emb,
            }
            _share = float(
                getattr(self.config, "method_selection_min_class_share", 0.01)
            )
            _min_class = max(2, int(n_total * _share))

            def _class_stats(stable: Dict[str, int]):
                """(min_size, mediana, tiny_frac) de las clases del método.

                `tiny_frac` = UCEs estables en clases menores que
                `min_class_share * n_total`, sobre `n_total`. Es el ruido real
                de A/B, que el CDH nunca reporta como -1.
                """
                if not stable:
                    return 0, 0.0, 0.0
                counts: Dict[int, int] = {}
                for cid in stable.values():
                    counts[int(cid)] = counts.get(int(cid), 0) + 1
                sizes = sorted(counts.values())
                tiny = sum(s for s in sizes if s < _min_class)
                return sizes[0], float(sizes[len(sizes) // 2]), tiny / max(1, n_total)

            _stats = {m: _class_stats(_stable_by_method[m]) for m in ("A", "B", "C")}
            _noise = {"A": noise_wc, "B": noise_coref, "C": noise_emb}
            _ari = {"A": ari_wc, "B": ari_sim, "C": ari_emb}
            _pert = {"A": "A1<->A2", "B": "B1<->B2", "C": "C1<->C2"}
            _raw_sizes = partition_sizes or {}

            def _max_share(_m: str) -> float:
                """Clase mayor / asignadas en la particion CRUDA del metodo.

                Se usa la particion cruda (no la estable) a proposito: una
                particion C con {442, 25, 12} es degenerada aunque la
                interseccion tight/loose solo retenga {199, 25, 12}.
                """
                sizes = [int(s) for s in (_raw_sizes.get(_m) or []) if int(s) > 0]
                tot = sum(sizes)
                return (max(sizes) / tot) if tot > 0 else 0.0

            def _raw_stats(_m: str) -> Dict[str, int]:
                sizes = [int(s) for s in (_raw_sizes.get(_m) or []) if int(s) > 0]
                assigned = sum(sizes)
                return {
                    "n_total": int(n_total),
                    "n_assigned": assigned,
                    "n_noise": int(n_total) - assigned,
                    "max_class_size": max(sizes) if sizes else 0,
                }

            def _sb(_m: str) -> Dict:
                """Bloque de estabilidad del metodo, o {} si no se midio."""
                return _stab.get(_m) or {}

            metrics = [
                MethodMetrics(
                    method=_m,
                    k=len(set(_stable_by_method[_m].values())),
                    ari=float(_ari[_m]),
                    coverage=len(_stable_by_method[_m]) / max(1, n_total),
                    noise_frac=float(_noise[_m]),
                    tiny_frac=float(_stats[_m][2]),
                    perturbation=_pert[_m],
                    available=bool(_stable_by_method[_m]),
                    assigned_frac=max(0.0, 1.0 - float(_noise[_m])),
                    min_class_size=int(_stats[_m][0]),
                    median_class_size=float(_stats[_m][1]),
                    max_class_share=_max_share(_m),
                    **_raw_stats(_m),
                    backend=(
                        str(getattr(self.config, "method_c_backend", "hdbscan"))
                        if _m == "C"
                        else ""
                    ),
                    diagnostic_only=bool(
                        _m == "C"
                        and getattr(self.config, "method_c_backend", "hdbscan")
                        == "hdbscan"
                        and not getattr(
                            self.config, "method_c_hdbscan_selectable", False
                        )
                    ),
                    bootstrap_ari=float(_sb(_m).get("bootstrap_ari", float("nan"))),
                    bootstrap_ci_lo=float(
                        _sb(_m).get("bootstrap_ari_ci_low", float("nan"))
                    ),
                    bootstrap_ci_hi=float(
                        _sb(_m).get("bootstrap_ari_ci_high", float("nan"))
                    ),
                    coverage_stable=float(_sb(_m).get("coverage_stable", float("nan"))),
                    null_p=float(_sb(_m).get("null_p", float("nan"))),
                    null_z=float(_sb(_m).get("null_z", float("nan"))),
                )
                for _m in ("A", "B", "C")
            ]
            selection = select_method(
                metrics,
                margin=float(getattr(self.config, "method_selection_margin", 0.02)),
                coverage_floor=float(
                    getattr(self.config, "method_selection_coverage_floor", 0.15)
                ),
                k_bounds=tuple(
                    getattr(self.config, "method_selection_k_bounds", (3, 20))
                ),
                min_class_share=_share,
                noise_max=float(
                    getattr(self.config, "method_selection_noise_max", 0.50)
                ),
                max_class_share_max=float(
                    getattr(self.config, "method_selection_max_class_share", 0.70)
                ),
                null_mode=str(
                    getattr(self.config, "method_selection_null_mode", "diagnostic")
                ),
            )
            self.last_selection = selection
            if selection.winner is None:
                primary_ids = set()
                primary_cluster = {}
            else:
                primary_cluster = _stable_by_method[selection.winner]
                primary_ids = set(primary_cluster)
                if abs(float(_ari[selection.winner]) - 1.0) <= 1e-9:
                    _blk = _stab.get(selection.winner) or {}
                    _ids = list(_blk.get("ids") or [])
                    _mask = _blk.get("stable_mask")
                    if _ids and _mask is not None and len(_ids) == len(_mask):
                        _m_ref = _stable_by_method[selection.winner]
                        _boot_ok = {
                            uid: _m_ref[uid]
                            for uid, ok in zip(_ids, _mask)
                            if ok and uid in _m_ref
                        }
                        if _boot_ok:
                            primary_cluster = _boot_ok
                            primary_ids = set(primary_cluster)
                            print(
                                f"   [select] {selection.winner}: pase tight/loose "
                                f"tautologico (ARI=1.0); is_stable por bootstrap "
                                f"({len(primary_ids)}/{n_total})"
                            )

            # La persistencia NO se hace aqui: este metodo corre una vez por
            # trial del optimizador (1050+ veces). La corrida principal la
            # escribe via WorkflowOrchestrator -> persist_method_selection().

        elif _mode == "wc_coref":
            primary_ids = set(stable_cross_ab)
            primary_cluster = stable_wc  # WC1 as cluster reference

        elif _mode == "wc_emb":
            primary_ids = set(stable_cross_ac)
            primary_cluster = stable_wc

        elif _mode in ("coref_emb", "sim_emb"):
            primary_ids = set(stable_cross_bc)
            primary_cluster = stable_sim

        elif _mode == "two_of_three":
            wc_s = set(stable_wc)
            sim_s = set(stable_sim)
            emb_s = set(stable_emb)
            ab_s = set(stable_cross_ab)
            bc_s = set(stable_cross_bc)
            ac_s = set(stable_cross_ac)
            within_2of3 = (wc_s & sim_s) | (sim_s & emb_s) | (wc_s & emb_s)
            cross_2of3 = (ab_s & bc_s) | (bc_s & ac_s) | (ab_s & ac_s)
            primary_ids = within_2of3 & cross_2of3
            primary_cluster = stable_wc  # ancla WC preservada

        else:  # "all" — full triple intersection
            wc_s = set(stable_wc)
            sim_s = set(stable_sim)
            emb_s = set(stable_emb)
            ab_s = set(stable_cross_ab)
            bc_s = set(stable_cross_bc)
            ac_s = set(stable_cross_ac)
            primary_ids = wc_s & sim_s & emb_s & ab_s & bc_s & ac_s
            primary_cluster = stable_wc  # WC1 as cluster reference

        if getattr(self.config, "primary_cluster_source", "wc") == "consensus":
            raise NotImplementedError(
                "primary_cluster_source='consensus' no está activo en esta versión. "
                "El análisis aguas abajo (vocabulario, φ, AFC, CAH, forma_index, "
                "P1–P5) asume primary_cluster = stable_wc. Activar 'consensus' "
                "requiere refactorizar esos consumidores."
            )

        # ── Back-propagate to UCE objects ─────────────────────────────────
        primary_stable_uces: List[UCE] = []
        _stability_tag = _mode
        if _mode == "select":
            _winner = selection.winner if selection is not None else None
            _stability_tag = f"select:{_winner}" if _winner else "select:none"

        for doc_uces in uces_por_doc:
            for uce in doc_uces:
                uid = uce.id

                if _mode in ("all", "two_of_three", "select"):
                    uce.stable_wc = uid in stable_wc
                    uce.stable_sim = uid in stable_sim
                    uce.stable_emb = uid in stable_emb
                    uce.stable_cross_ab = uid in stable_cross_ab
                    uce.stable_cross_bc = uid in stable_cross_bc
                    uce.stable_cross_ac = uid in stable_cross_ac

                if uid in primary_ids and uid in primary_cluster:
                    uce.cluster_id = primary_cluster[uid]
                    uce.is_stable = True
                    uce.stability_method = _stability_tag
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
                    # Assign best available cluster_id for downstream reference.
                    # En 'select' NO se mezclan espacios de etiquetas: el ganador
                    # tiene su propio K y las etiquetas de stable_wc/stable_sim
                    # pertenecen a OTRO metodo (K distinto). Una UCE inestable
                    # queda sin clase, como en ALCESTE ("no clasificada"), y no
                    # infla member_count de la tabla clusters.
                    if _mode == "select":
                        uce.cluster_id = None
                    elif in_wc:
                        uce.cluster_id = stable_wc[uid]
                    elif in_sim:
                        uce.cluster_id = stable_sim[uid]
                    elif in_emb:
                        uce.cluster_id = stable_emb[uid]
                    else:
                        uce.cluster_id = None
                    uce.is_stable = False
                    uce.stability_method = (
                        f"{_stability_tag}|{methods}" if _mode == "select" else methods
                    )

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
        elif _mode == "two_of_three":
            n_wc = len(stable_wc)
            n_sim = len(stable_sim)
            n_emb = len(stable_emb)
            n_ab = len(stable_cross_ab)
            n_bc = len(stable_cross_bc)
            n_ac = len(stable_cross_ac)
            summary = (
                f"\n╔{'═' * 70}╗\n"
                f"║{'STABILITY REPORT — TWO-OF-THREE VERDICT':^70}║\n"
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
                f"║  TWO-OF-THREE INTERSECTION:     {n_primary:>6}  ({_pct(n_primary)})                   ║\n"
                f"║  Unstable in any method:        {n_unstable:>6}  ({_pct(n_unstable)})                   ║\n"
                f"╚{'═' * 70}╝\n"
            )
        elif _mode == "wc_coref":
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
        elif _mode == "select":
            _sel_lines = (
                selection.explain_lines()
                if selection is not None
                else ["METHOD SELECTION (sin metricas disponibles)"]
            )
            _size_lines = []
            for _row in selection.all_metrics if selection is not None else []:
                if not _row.get("available"):
                    continue
                _size_lines.append(
                    f"  [i] {_row['method']}: clase min="
                    f"{int(_row.get('min_class_size') or 0)} mediana="
                    f"{float(_row.get('median_class_size') or 0):.0f} "
                    f"mayor={float(_row.get('max_class_share') or 0):.2f} "
                    f"asignadas={float(_row.get('assigned_frac') or 0):.3f}"
                )
            summary = (
                f"\n{'=' * 70}\n"
                f"  STABILITY REPORT - SELECT (un metodo, sin interseccion)\n"
                f"{'=' * 70}\n"
                f"  Total UCEs: {n_total}\n"
                f"  Primary output (metodo elegido): {n_primary} ({_pct(n_primary)})\n"
                f"  Unstable: {n_unstable} ({_pct(n_unstable)})\n"
                f"{'-' * 70}\n"
                + "\n".join(_sel_lines)
                + ("\n" + "\n".join(_size_lines) if _size_lines else "")
                + f"\n{'=' * 70}\n"
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

        # Nota de interpretacion (una vez por corrida, no por trial): con el
        # piso del CDH aplicado, la granularidad esta acotada por
        # construccion (K <= n/piso) y el arbol equilibrado da K ~ 2^d. Subir
        # el piso baja K y SUBE el ARI entre perturbaciones por construccion.
        _piso = max(
            5,
            int(
                n_total
                * float(getattr(self.config, "min_cluster_size_cdh", 0.0) or 0.0)
            ),
        )
        _k_cap = max(1, int(n_total) // max(1, _piso))
        summary += (
            f"  [i] Piso CDH={_piso} UCEs -> K_max~={_k_cap}. La estabilidad (ARI\n"
            f"      entre perturbaciones) crece de forma MONOTONA con la coarseza:\n"
            f"      mas piso => K menor => ARI mayor por construccion. No comparar\n"
            f"      ARI entre K distintos como si midieran lo mismo.\n"
        )

        return {
            "primary_stable_uces": primary_stable_uces,
            "summary": summary,
            "effective_mode": _mode,
            "selection": selection,
            "stability_blocks": _stab,
            "selected_method": selection.winner if selection is not None else None,
        }

    def persist_method_selection(self, selection: Selection, n_total: int) -> None:
        """Persiste la eleccion junto al corpus al que se aplico.

        Snapshot (sobrescrito) + historial JSONL (append-only) para que el
        auditor ``9_selection_audit.py`` pueda detectar si el ganador cambia
        entre targets vecinos. Un fallo de escritura nunca aborta el run.
        """
        if str(getattr(self.config, "run_mode", "modern")) == "classic":
            print(
                "   [i] classic: sin method_selection.json (no hay elección de método)."
            )
            return
        target = int(getattr(self.config, "uce_target_size", 0) or 0)
        try:
            persist_selection(
                selection,
                os.path.join(REII_DATA_DIR, "method_selection.json"),
                history_path=os.path.join(REII_DATA_DIR, "method_selection_log.jsonl"),
                context={
                    "corpus_fingerprint": _corpus_fingerprint(),
                    "uce_target_size": target,
                    "mode": "select",
                    "n_uces": int(n_total),
                    "seed": getattr(self.config, "random_state", None),
                    "method_c_backend": str(
                        getattr(self.config, "method_c_backend", "hdbscan")
                    ),
                    "spec_hash": hashlib.sha256(
                        json.dumps(
                            dataclasses.asdict(self.config),
                            sort_keys=True,
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest()[:16],
                },
            )
        except Exception as exc:  # defensivo: la seleccion ya esta en memoria
            print(f"   [!] no se pudo persistir la seleccion de metodo: {exc}")

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
        checkpoint_path: Optional[str] = None,
        force: bool = False,
        precomputed_coref_chains: Optional[Dict[int, Dict[str, List]]] = None,
    ) -> Tuple:
        self.ppmi_builder = ppmi_builder
        mode = getattr(self.config, "classification_mode", "wc_only")
        # normalise alias
        # Normalise legacy aliases so downstream code only sees canonical names
        _ALIASES = {
            "auto": "select",
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
        _do_wc = mode in ("wc_only", "wc_coref", "wc_emb", "all", "select")
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
        _do_coref = mode in ("coref_only", "wc_coref", "coref_emb", "all", "select")
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
                checkpoint_path=checkpoint_path,
                force=force,
                precomputed_chains=(
                    precomputed_coref_chains.get(cu_tight)
                    if precomputed_coref_chains
                    else None
                ),
            )
            print(f"--- Classification B2: coref loose (context_units={cu_loose}) ---")
            res_coref2 = self._clasificar_umbral_coref(
                copy.deepcopy(uces_por_doc),
                mf2,
                cu_loose,
                retrofitted_vectors=retrofitted_vectors,
                checkpoint_path=checkpoint_path,
                force=force,
                precomputed_chains=(
                    precomputed_coref_chains.get(cu_loose)
                    if precomputed_coref_chains
                    else None
                ),
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
            mode in ("emb_only", "wc_emb", "coref_emb", "all", "select")
            and retrofitted_vectors is not None
        )
        if _do_emb:
            # Escalado al corpus (mismo criterio que min_cluster_size_cdh en
            # A/B), no un absoluto fijo — ver _resolve_hdbscan_uce_params.
            # ([hdbscan-scale-patch])
            _n_emb = sum(len(d) for d in uces_por_doc)
            mcs_tight, ms_tight = self._resolve_hdbscan_uce_params(_n_emb, "tight")
            mcs_loose, ms_loose = self._resolve_hdbscan_uce_params(_n_emb, "loose")
            print(
                f"\n--- Classification C1: EMB tight (min_cluster_size={mcs_tight}, "
                f"min_samples={ms_tight}) ---"
            )
            res_emb1 = self._clasificar_umbral_embedding(
                uces_por_doc,
                mcs_tight,
                retrofitted_vectors,
                min_samples=ms_tight,
                variant="tight",
            )
            print(
                f"--- Classification C2: EMB loose (min_cluster_size={mcs_loose}, "
                f"min_samples={ms_loose}) ---"
            )
            res_emb2 = self._clasificar_umbral_embedding(
                uces_por_doc,
                mcs_loose,
                retrofitted_vectors,
                min_samples=ms_loose,
                variant="loose",
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
        effective_verdict_mode = mode
        if (
            mode == "all"
            and getattr(self.config, "verdict_mode", "all") == "two_of_three"
        ):
            effective_verdict_mode = "two_of_three"

        def _noise_frac(labels: Optional[Dict[str, int]], n_total: int) -> float:
            """UCEs SIN ASIGNAR (label < 0, o fuera de `labels`) / `n_total`.

            Denominador COMUN (n_total) para que A, B y C sean comparables:
            antes se dividia por `len(labels)`, asi que el CDH (que asigna
            etiqueta a toda UC, nunca -1) daba 0.00 por construccion y HDBSCAN
            parecia el unico metodo con ruido.
            """
            if not labels:
                return 1.0
            assigned = sum(1 for v in labels.values() if int(v) >= 0)
            return max(0.0, (int(n_total) - assigned) / max(1, int(n_total)))

        _n_total = len({uce.id for doc in uces_por_doc for uce in doc})
        noise_wc = _noise_frac(
            self._extract_uce_labels(res_wc1) if res_wc1 else None, _n_total
        )
        noise_coref = _noise_frac(
            self._extract_uce_labels(res_coref1) if res_coref1 else None, _n_total
        )
        noise_emb = _noise_frac(
            res_emb1["uce_to_label"] if res_emb1 else None, _n_total
        )

        def _partition_sizes(labels: Optional[Dict[str, int]]) -> List[int]:
            """Tamanos de clase de la particion CRUDA (excluye no asignados <0).

            Es el denominador correcto del gate de dominancia: la particion
            cruda del metodo, no la interseccion tight/loose.
            """
            return _partition_sizes_from_labels(labels)

        _raw_sizes = {
            "A": _partition_sizes(
                self._extract_uce_labels(res_wc1) if res_wc1 else None
            ),
            "B": _partition_sizes(
                self._extract_uce_labels(res_coref1) if res_coref1 else None
            ),
            "C": _partition_sizes(res_emb1["uce_to_label"] if res_emb1 else None),
        }

        self._stability_spaces = self._build_stability_spaces(
            res_wc1,
            res_coref1,
            res_emb1,
            min_n=int(getattr(self.config, "method_selection_stability_min_n", 30)),
        )
        stability_blocks = self._stability_blocks(
            getattr(self, "_stability_spaces", None),
            n_boot=int(getattr(self.config, "method_selection_stability_boot", 10)),
            n_perm=int(getattr(self.config, "method_selection_stability_perm", 50)),
            min_n=int(getattr(self.config, "method_selection_stability_min_n", 30)),
            seed=int(getattr(self.config, "random_state", 42) or 42),
        )
        report = self._triple_stability_verdicts(
            uces_por_doc=uces_por_doc,
            mode=effective_verdict_mode,
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
            noise_wc=noise_wc,
            noise_coref=noise_coref,
            noise_emb=noise_emb,
            partition_sizes=_raw_sizes,
            stability_blocks=stability_blocks,
        )

        primary_stable = report["primary_stable_uces"]
        if self.quiet_selection:
            print(
                "   [select] pass intermedio: seleccion no persistida "
                "(Method C aun no disponible)"
            )
        else:
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
        _selected = report.get("selected_method")
        if effective_mode == "select":
            # El vocabulario sale del pase tight del metodo GANADOR (C no tiene
            # vocabulario propio: se construye desde primary_stable).
            primary_res = {"A": res_wc1, "B": res_coref1, "C": None}.get(_selected)
        elif effective_mode in ("wc_only", "wc_coref", "wc_emb", "all", "two_of_three"):
            primary_res = res_wc1
        elif effective_mode in ("coref_only", "coref_emb"):
            primary_res = res_coref1
        else:  # emb_only
            primary_res = None

        if primary_res is None and effective_mode not in ("emb_only", "select"):
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
        # En "select" el par es el del metodo ganador: el ARI que ve el
        # optimizador debe ser el mismo que decidio la seleccion.
        _ari_pair: Optional[Tuple[Optional[Dict], Optional[Dict]]] = None
        if effective_mode == "select":
            _ari_pair = {
                "A": (res_wc1, res_wc2),
                "B": (res_coref1, res_coref2),
                "C": (res_emb1, res_emb2),
            }.get(_selected)
        elif _do_wc and res_wc1 and res_wc2:
            _ari_pair = (res_wc1, res_wc2)
        elif _do_coref and res_coref1 and res_coref2:
            _ari_pair = (res_coref1, res_coref2)

        def _labels_of(res: Optional[Dict]) -> Optional[Dict[str, int]]:
            if res is None:
                return None
            if "uce_to_label" in res:  # Method C
                return res["uce_to_label"]
            return self._extract_uce_labels(res)

        if _ari_pair is not None:
            _l1 = _labels_of(_ari_pair[0])
            _l2 = _labels_of(_ari_pair[1])
        else:
            _l1 = _l2 = None

        if _l1 is not None and _l2 is not None:
            labels1_ari = np.array([_l1.get(u.id, -1) for u in primary_stable])
            labels2_ari = np.array([_l2.get(u.id, -1) for u in primary_stable])
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
        elif (
            self.config.clustering_method == "hdbscan"
            and _HDBSCAN_AVAILABLE
            and _UMAP_AVAILABLE
        ):
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
        """
        AFC sobre matriz clase × términos SIN pseudoconteo.

        Reglas (Bloque 1 del plan):
          · No sumar ε. Las masas marginales r_i y c_j deben reflejar la
            estructura real, no una matriz densificada artificialmente.
          · Poda previa: eliminar columnas con masa < 2 y filas sin masa.
          · Truncated SVD vía randomized_svd (Krylov/Lanczos) sobre la
            matriz de residuos estandarizados.
        """
        from sklearn.utils.extmath import randomized_svd

        mat = np.asarray(mat_clases, dtype=np.float64)
        if mat.ndim != 2:
            return {}
        n_rows, n_cols = mat.shape
        if n_rows < 2 or n_cols < 2:
            return {}

        # ── Poda de columnas sin masa y filas vacías (reemplaza al ε) ────────
        col_mass = mat.sum(axis=0)
        keep_cols = np.where(col_mass >= 2.0)[0]
        if len(keep_cols) < 2:
            return {}
        mat = mat[:, keep_cols]
        voc_kept = [voc[j] for j in keep_cols] if voc else None

        row_mass = mat.sum(axis=1)
        keep_rows = np.where(row_mass > 0)[0]
        if len(keep_rows) < 2:
            return {}
        mat = mat[keep_rows, :]
        class_ids_kept = np.asarray(class_ids)[keep_rows]

        total = mat.sum()
        if total <= 0:
            return {}

        row_sums = mat.sum(axis=1)
        col_sums = mat.sum(axis=0)
        r = row_sums / total
        c = col_sums / total

        # Residuos estandarizados (Pearson) — sin ε
        E = np.outer(row_sums, col_sums) / total
        S = (mat - E) / np.sqrt(np.maximum(E, 1e-12))

        # ── SVD truncada vía Lanczos/randomized ──────────────────────────────
        k_max = min(n_components + 1, min(S.shape) - 1)
        if k_max < 2:
            return {}
        U, s, Vt = randomized_svd(
            S, n_components=k_max, n_iter=7, random_state=self.config.random_state
        )

        # Descartar componente trivial (autovalor 1 por construcción)
        U_k = U[:, 1 : n_components + 1]
        s_k = s[1 : n_components + 1]
        Vt_k = Vt[1 : n_components + 1, :]
        k = s_k.shape[0]
        if k < 1:
            return {}

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
            "class_ids": class_ids_kept.tolist(),
            "row_coords": row_coords.tolist(),
            "col_coords": col_coords.tolist(),
            "row_std": row_std.tolist(),
            "col_std": col_std.tolist(),
            "col_contrib": col_contrib,
            "explained_inertia": explained,
            "singular_values": s_k.tolist(),
            "total_inertia": total_inertia,
            # Metadatos de la poda: el caller debe usar estos, NO voc completo
            "voc": voc_kept,
            "kept_cols_idx": keep_cols.tolist(),
            "kept_rows_idx": keep_rows.tolist(),
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
        labels = afc_result.get("voc") or voc
        dist_vector = pdist(col_std, metric="euclidean")
        Z = linkage(dist_vector, method="ward")
        return {"Z": Z.tolist(), "labels": labels, "n_terms": len(labels)}

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
        self.n_jobs = config.rf_n_jobs

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
            n_jobs=self.n_jobs,
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

        # Tune hyperparameters. Si el tuning falla (p. ej. joblib no puede
        # crear workers por falta de memoria), seguimos con los valores por
        # defecto: perder el tuning es preferible a abortar aquí la corrida
        # completa, con todo el pipeline ya calculado.
        try:
            best_params = self._tune_hyperparams(X, y)
        except Exception as exc:  # noqa: BLE001 - degradar, no abortar
            print(
                f"[RF] WARNING: tuning falló ({type(exc).__name__}: {exc}); "
                "usando hiperparámetros por defecto."
            )
            best_params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            }
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
            clf, X, y, n_repeats=10, random_state=self.random_state, n_jobs=self.n_jobs
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
        # Cast to int BEFORE matmul: a boolean matmul yields True/False
        # co-occurrence counts, so `cij < thr` skips every pair and the
        # network ends up empty.
        bin_mat = (matriz > 0).astype(int)
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
            OpenAI(api_key=config.together_api_key, base_url=TOGETHER_BASE_URL)
            if config.use_llm_synthesis
            and config.together_api_key
            and _OPENAI_AVAILABLE
            else None
        )
        self.embedder = (
            _get_sentence_embedder(config.embedding_model_name)
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
                extra_body={"reasoning_effort": "none"},
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
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

    _OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    _OPTUNA_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
# MODO CLÁSICO (ALCESTE 2012) — preset + identidad de especificación
# ══════════════════════════════════════════════════════════════════════
# Escenario A del paper (docs/design_v5/CLASSIC_MODE.md). Un override, no un
# fork: el mismo pipeline con la fidelidad de 2012. El fork
# main_workflow_clasico.py queda retirado (shim, §9.5).

CLASSIC_OVERRIDES: Dict[str, Any] = {
    "run_mode": "classic",
    "classification_mode": "wc_only",  # §2 Paso 2: un solo CDH
    "min_forms_uc": [10, 14],  # §2 Paso 1: UC 10 vs 14 palabras
    "cdh_enforce_min_leaf": False,  # §2: el clásico no tiene piso de hoja
    "optimize": False,  # §2: parámetros fijos, sin búsqueda
    "use_embeddings": False,  # §2/§3: sin retrofitting ni Method C
    "use_liminal_projection": False,  # §3 Paso 6: proyección es de REII
    "auto_calibrate_uce_size": False,  # §2: tamaño de UCE fijo
}

_CLASSIC_SPEC_FIELDS = (
    "classification_mode",
    "min_forms_uc",
    "tsj",
    "pseudocount",
    "cdh_enforce_min_leaf",
    "min_cluster_size_cdh",
    "min_r2_threshold",
    "uce_target_size",
    "min_uce_words",
    "stem_backend",
    "use_bigrams",
    "use_trigrams",
    "use_embeddings",
    "random_state",
)


def run_spec_id(config) -> str:
    """Identidad de la especificación (§4.1). Moderno => "modern" literal,
    para no invalidar ninguna caché previa."""
    if str(getattr(config, "run_mode", "modern")) != "classic":
        return "modern"
    payload = {k: getattr(config, k, None) for k in _CLASSIC_SPEC_FIELDS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "classic:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def apply_classic_mode(config: Config, uc_config: "UCBuilderConfig"):
    """Devuelve (Config, UCBuilderConfig) del escenario A. Idempotente.

    No toca `pseudocount`: el paper pide 0.1 pero la CDH no lo consume
    (docs/design_v5/CLASSIC_MODE.md §3). Queda 0.0 y se declara.
    """
    # `_optimizer_will_run = bool(_OPTIMIZE_ONLY) or (...)` (main()) fuerza el
    # optimizador aunque classic fije optimize=False; la etapa 2 corre en
    # mode="select", asi que optimizaria el objetivo moderno sobre una config
    # clasica y escribiria best_params en data/classic/ con exit 0.
    if _OPTIMIZE_ONLY:
        print(
            "[X] --optimize-only no es compatible con --mode classic: el escenario A "
            "fija los hiperparámetros del paper (optimize=False) y el optimizador los "
            "buscaría con el objetivo de 'select'."
        )
        sys.exit(2)
    base = os.path.join(CLASSIC_DATA_DIR)
    os.makedirs(base, exist_ok=True)
    os.makedirs(CLASSIC_OUTPUT_DIR, exist_ok=True)
    classic = dataclasses.replace(
        config,
        **CLASSIC_OVERRIDES,
        # ── Aislamiento (§4.2): una sola palanca arrastra best_params,
        #    comparative_results y el SQLite de Optuna.
        db_local_path=os.path.join(base, "workflow_data.json"),
        optimize_storage=(
            f"sqlite:///{os.path.join(base, 'reii_optuna.db').replace(os.sep, '/')}"
            if getattr(config, "optimize_storage", None)
            else None
        ),
        optimize_study_name=f"{getattr(config, 'optimize_study_name', 'reii_search')}"
        f":{run_spec_id(dataclasses.replace(config, **CLASSIC_OVERRIDES))}",
    )
    spec = run_spec_id(classic)
    print("   [CLI] --mode classic -> ALCESTE 2012 (escenario A)")
    print(f"         spec={spec}  data_dir={CLASSIC_DATA_DIR}")
    print(
        "         overrides: classification_mode=wc_only min_forms_uc=[10, 14]\n"
        "                    cdh_enforce_min_leaf=False optimize=False\n"
        "                    use_embeddings=False use_liminal_projection=False\n"
        "                    auto_calibrate_uce_size=False"
    )
    print(
        "         [!] pseudocount=0.0 (paper pide 0.1; epsilon no implementado "
        "en la CDH)"
    )
    return classic, uc_config


# ══════════════════════════════════════════════════════════════════════
# LECTOR DE PARAMS ROBUSTO (claves desconocidas) — §6
# ══════════════════════════════════════════════════════════════════════
_NON_CONFIG_PARAM_KEYS = {
    # espacio de búsqueda / parámetros que se aplican fuera de Config
    "cdh_target_k",
    "min_forms_uc_1",
    "forms_gap",
    "similarity_threshold",  # UCBuilderConfig
    "coref_weight",  # UCBuilderConfig
    "uc_window_size",  # UCBuilderConfig
}


def resolve_best_params(
    params: Dict[str, Any],
    *,
    where: str = "best_params.json",
) -> Dict[str, Any]:
    """Filtra claves no aplicables; falla SÓLO si no queda ninguna válida.

    Un archivo con una clave de un diseño futuro debe degradar, no abortar:
    el resto de los parámetros sigue siendo el resultado del optimizador.
    """
    if not params:
        return {}
    known = set(Config.__dataclass_fields__) | _NON_CONFIG_PARAM_KEYS
    valid: Dict[str, Any] = {}
    ignored: List[str] = []
    for key, value in params.items():
        if key in known:
            valid[key] = value
        else:
            ignored.append(str(key))
    if ignored:
        print(
            f"   [!] {where}: {len(ignored)} clave(s) desconocida(s) ignorada(s): "
            f"{', '.join(sorted(ignored))}"
        )
        logger.warning(
            "%s: claves ignoradas (no aplicables a esta versión de Config): %s",
            where,
            sorted(ignored),
        )
        for _k in sorted(ignored):
            print(f"       - {_k} = {params[_k]!r}")
    if not valid:
        raise ValueError(
            f"{where}: ninguna de las {len(params)} claves es aplicable "
            f"(ignoradas: {', '.join(sorted(ignored))}). "
            "No se puede continuar sin parámetros."
        )
    return valid


# Centinela dominado para trials inválidos (gates duros). Optuna 3.6.2
# (pin <4.0.0) crashea en TPE multi-objetivo cuando un trial PRUNED
# (values=None) cae en el conjunto "below" del Parzen estimator
# (_calculate_weights_below_for_multi_objective itera trial.values).
# Devolver un punto estrictamente dominado en vez de TrialPruned evita
# el crash y no contamina el frente de Pareto.
#
# Valores FUERA de los rangos válidos para que el centinela sea
# estrictamente dominado por CUALQUIER trial real, sin depender de los
# gates: ARI ∈ [-1,1] → -2.0; coverage ∈ [0,1] → -1.0; −K ∈ [-8,-2]
# (minimizar) → 0.0. Ningún trial válido puede igualarlo ni empeorarlo
# en ningún objetivo. Ver scripts/_repro_tpe_pruned.py.
_SENTINEL_TRIAL = (-2.0, -1.0, 0.0)


def _is_sentinel(values) -> bool:
    """True si `values` es el centinela de trial inválido."""
    return tuple(values) == _SENTINEL_TRIAL


_MAX_TB_CHARS = 2000
_MAX_TB_LINES = 20


def _sentinel(trial: "optuna.Trial", reason: str):
    """Centinela dominado + motivo, para separar gates de excepciones."""
    try:
        trial.set_user_attr("outcome", "gate")
        trial.set_user_attr("gate_reason", str(reason)[:120])
    except Exception:  # nunca enmascarar el centinela por un fallo de atributos
        logger.debug("No se pudo registrar gate_reason=%s", reason)
    return _SENTINEL_TRIAL


def _trial_exc_attrs(exc: BaseException) -> Dict[str, str]:
    """user_attrs de un trial que explotó: tipo, mensaje y ultimas ~20 lineas."""
    tb = traceback.format_exc().splitlines()[-_MAX_TB_LINES:]
    return {
        "error_type": type(exc).__name__,
        "error_msg": str(exc)[:500],
        "traceback": "\n".join(tb)[-_MAX_TB_CHARS:],
    }


def _make_guarded_objective(objective, printer=print):
    """Envuelve el objetivo de Optuna para que NINGUNA excepcion se pierda.

    - `Exception` -> `_SENTINEL_TRIAL` + error_type/error_msg/traceback en
      user_attrs (el FAIL #81 del probe v4 se perdio justo por no tener esto);
    - `KeyboardInterrupt`/`SystemExit` NO se capturan (BaseException): se
      re-lanzan para que Ctrl-C corte de verdad;
    - imprime una sola vez por trial.
    """

    def _guarded(trial):
        try:
            return objective(trial)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 — ver docstring
            attrs = _trial_exc_attrs(exc)
            try:
                trial.set_user_attr("outcome", "exception")
                trial.set_user_attr("trial_failed", True)
                for k, v in attrs.items():
                    trial.set_user_attr(k, v)
            except Exception:  # registrar el fallo no puede tumbar el trial
                logger.warning(
                    "No se pudieron guardar los attrs del trial %s",
                    getattr(trial, "number", "?"),
                )
            printer(
                f"[!] trial #{getattr(trial, 'number', '?')} fallo: "
                f"{attrs['error_type']}: {attrs['error_msg']} "
                f"(traceback en user_attr)"
            )
            return _SENTINEL_TRIAL

    return _guarded


def _sentinel_breakdown(trials) -> Tuple[int, int, List[str]]:
    """(por_gate, por_excepcion, detalle) sobre los trials centinela."""
    n_gate = n_exc = 0
    detail: List[str] = []
    for t in trials:
        try:
            values = t.values
        except Exception:
            continue
        if values is None or not _is_sentinel(values):
            continue
        etype = (getattr(t, "user_attrs", {}) or {}).get("error_type")
        if etype:
            n_exc += 1
            detail.append(
                f"#{getattr(t, 'number', '?')} {etype}: "
                f"{(getattr(t, 'user_attrs', {}) or {}).get('error_msg', '')}"
            )
        else:
            n_gate += 1
    return n_gate, n_exc, detail


#: Techo de `cdh_target_k` (piso del CDH expresado en clases objetivo).
#: No es un objetivo: se convierte en `min_cluster_size_cdh = 1/(2K)`, así
#: que subirlo REDUCE el piso de hoja y el CDH particiona más. Medido en
#: `reii_probe_v5_stage1_wc` (n=528): kT<=12 -> K<=18 (viable); kT>=16 ->
#: K=25..34, fuera del gate `k_bounds=(3,20)`, centinela garantizado. Con
#: el techo anterior (20) se gastaba el 30 % del presupuesto de la etapa 1
#: en trials infactibles por construcción.
_CDH_TARGET_K_HI = 12


def _cdh_frac_from_params(params: Dict[str, Any], default: float = 0.10) -> float:
    """Piso del CDH como fracción de n, resuelto desde los params de un trial.

    Espacio actual: ``cdh_target_k`` (K objetivo) -> ``piso = n / (2K)``, es
    decir ``frac = 1 / (2K)``. El árbol CDH parte de forma equilibrada, así
    que K ≈ 2^profundidad y el piso necesario para K hojas es n/(2K).

    Prioridad: si ``min_cluster_size_cdh`` está presente se respeta tal cual
    (fracción explícita). Eso cubre los estudios y ``best_params.json``
    previos y, sobre todo, permite que las sondas que editan el archivo a
    mano (``8_probe_b_permissive.py``) sigan teniendo la última palabra:
    ``save_pareto_front`` escribe las dos claves y la explícita gana.
    """
    legacy = params.get("min_cluster_size_cdh")
    if legacy is not None:
        try:
            return float(min(0.5, max(0.01, float(legacy))))
        except (TypeError, ValueError):
            pass
    k = params.get("cdh_target_k")
    if k is not None:
        try:
            return float(min(0.5, max(0.01, 1.0 / (2.0 * max(1.0, float(k))))))
        except (TypeError, ValueError):
            pass
    return float(default)


# ── Espacio condicional v5 ─────────────────────────────────────────────
def _suggest_conditional(
    trial: "optuna.Trial", name: str, spec: Dict[str, Any], params: Dict[str, Any]
):
    """Muestrea un parametro cuyo rango depende de otro ya muestreado.

    ``spec = {"kind": "lt"|"gt", "other": <driver>, ...}``

      lt → entero en ``[lo, other-1]``   (p.ej. tight < loose)
      gt → float  en ``(other, hi]``     (p.ej. frac > frac_loose)

    Rangos dinamicos: Optuna 3.6.2 los soporta para int/float (solo exige
    que ``step`` y ``log`` no cambien entre trials) pero NO para categoricos
    (``ValueError: CategoricalDistribution does not support dynamic value
    space``). Por eso el dependiente nunca es categorico.
    """
    driver = spec.get("other")
    base = params.get(driver)
    if base is None:
        raise ValueError(f"{name}: falta el parametro driver {driver!r}")

    kind = spec.get("kind")
    if kind == "lt":
        lo = int(spec.get("lo", 1))
        hi = int(base) - 1
        if hi < lo:
            raise ValueError(f"{name}: rango vacio [{lo}, {hi}] con {driver}={base}")
        return trial.suggest_int(name, lo, hi)

    if kind == "gt":
        lo = float(base) + float(spec.get("eps", 0.002))
        hi = float(spec.get("hi", 0.06))
        if hi < lo:
            raise ValueError(f"{name}: rango vacio [{lo}, {hi}] con {driver}={base}")
        return trial.suggest_float(name, lo, hi)

    raise ValueError(f"{name}: tipo condicional desconocido {kind!r}")


def _expand_levels(spec: Any) -> List[Any]:
    """Valores alcanzables de un spec (para precargar caches y contar)."""
    if isinstance(spec, dict):
        return []  # dependiente: lo cubre el driver + _suggest_conditional
    if isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = spec
        if isinstance(lo, int) and isinstance(hi, int):
            return list(range(lo, hi + 1))
        return [lo, hi]
    if isinstance(spec, list):
        return list(spec)
    return [spec]


# ── Auditor de bordes del espacio (v6, WARN-only) ──────────────────────
#: Parametros AMPLIABLES: si el ganador cae en un extremo, el rango puede
#: estar recortando el optimo. La recomendacion es ensanchar UN paso la
#: proxima campana con un study NUEVO (Optuna no debe mezclar trials
#: muestreados de espacios distintos).
_BOUNDARY_EXPANDABLE = (
    "min_forms_uc_1",
    "similarity_threshold",
    "min_r2_threshold",
    "coref_weight",
)

#: Parametros ESTRUCTURALES: su extremo describe un limite del metodo (o el
#: rango lo fija otro parametro). Ampliarlo no compra busqueda: cambia el
#: metodo o produce trials infactibles. El auditor NO los recomienda ampliar.
_BOUNDARY_STRUCTURAL = (
    "swap_iterations",
    "cdh_target_k",
    "coref_context_units_tight",
    "hdbscan_uce_min_cluster_frac",
    "hdbscan_uce_min_cluster_frac_loose",
)

#: Motivo (una frase) de por que cada estructural NO se amplia.
_BOUNDARY_STRUCTURAL_WHY = {
    "swap_iterations": (
        "0 apaga _intercambio (ablacion del refinamiento, no un "
        "hiperparametro) y por encima de 5 el swap ya satura"
    ),
    "cdh_target_k": (
        "no es un objetivo: fija min_cluster_size_cdh=1/(2K); pasado "
        "_CDH_TARGET_K_HI el CDH sale del gate k_bounds y el trial es "
        "centinela por construccion"
    ),
    "coref_context_units_tight": (
        "es el dependiente de coref_context_units_loose (tight < loose); "
        "el borde lo fija el driver, no el presupuesto"
    ),
    "hdbscan_uce_min_cluster_frac": (
        "es el dependiente de frac_loose (frac > frac_loose); su techo "
        "0.04 codifica el regimen mcs~5-20 medido"
    ),
    "hdbscan_uce_min_cluster_frac_loose": (
        "el driver fija la escala de Method C (mcs), no un rango explorable"
    ),
}

#: Censura DECLARADA: el optimo queda en el borde y el rango se deja como esta.
#: No es un descuido: es una decision tomada antes del run v6, y el auditor la
#: imprime para que quede en el log, no para que se "arregle" despues.
_BOUNDARY_DECLINED = {
    "tsj": (
        "el techo se mantiene: `tsj` es umbral de frecuencia documental, "
        "subirlo poda vocabulario y baja lemas/UCE (retencion), que es "
        "endpoint registrado (0.451 frente a 0.537 de ALCESTE). Ganar "
        "estabilidad podando retencion es el intercambio que este auditor "
        "existe para hacer visible, no para firmar"
    ),
    "forms_gap": (
        "el piso se mantiene: `gap` separa los pisos de la pasada 1 y la "
        "pasada 2 (mf2 = mf1 + gap). Con gap<2 las dos pasadas producen casi "
        "la misma particion y el ARI de 2 pasadas sube por construccion: es "
        "la misma familia de artefacto que swap_iterations=0 (patch 19), no "
        "un hiperparametro. gap=2 es el minimo que las mantiene distintas. "
        "Desviacion declarada: ALCESTE usa 10/14 (gap=4) y el default del "
        "codigo [13, 17] (gap=4); el optimo de v6 corre con (10, 12). La "
        "sensibilidad gap=2 vs gap=4 se mide en "
        "docs/design_v5/EXPERIMENTO_SENSIBILIDAD_FORMS_GAP.md"
    ),
}


def _edge_equal(a: Any, b: Any) -> bool:
    """Igualdad tolerante a floats y a tipos raros (nunca lanza)."""
    try:
        if isinstance(a, bool) or isinstance(b, bool):
            return bool(a == b)
        if isinstance(a, float) or isinstance(b, float):
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _edge_bounds(spec: Any, chosen: Dict[str, Any]):
    """(lo, hi) del spec, o None si es de una sola opcion / no clasificable."""
    if isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = spec
        if isinstance(lo, bool) or isinstance(hi, bool):
            return None
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return None if lo == hi else (lo, hi)
        return None
    if isinstance(spec, list):
        if len(spec) < 2:
            return None
        return spec[0], spec[-1]
    if isinstance(spec, dict):
        kind = spec.get("kind")
        base = chosen.get(spec.get("other"))
        if base is None:
            return None
        try:
            if kind == "lt":
                lo, hi = int(spec.get("lo", 1)), int(base) - 1
            elif kind == "gt":
                lo = float(base) + float(spec.get("eps", 0.002))
                hi = float(spec.get("hi", 0.06))
            else:
                return None
        except (TypeError, ValueError):
            return None
        return (lo, hi) if hi >= lo else None
    return None


def _edge_num(x: Any) -> str:
    """Numero compacto: 0.012000000000000002 -> 0.012."""
    return f"{x:g}" if isinstance(x, float) else str(x)


def _edge_label(spec: Any, lo: Any = None, hi: Any = None) -> str:
    """Rango legible del spec para la linea del auditor."""
    if isinstance(spec, list):
        return "[" + ", ".join(_edge_num(x) for x in spec) + "]"
    if isinstance(spec, tuple) and len(spec) == 2:
        return f"({_edge_num(spec[0])}, {_edge_num(spec[1])})"
    if isinstance(spec, dict):
        other = spec.get("other")
        return f"(dependiente de {other}: [{_edge_num(lo)}, {_edge_num(hi)}])"
    return str(spec)


def _edge_recommendation(
    name: str, spec: Any, lo: Any, hi: Any, at_lo: bool, at_hi: bool
) -> str:
    """Recomendacion; el prefijo declara la politica (4 categorias)."""
    if name in _BOUNDARY_STRUCTURAL:
        why = _BOUNDARY_STRUCTURAL_WHY.get(name, "el borde es un limite del metodo")
        return f"estructural: rango {_edge_label(spec, lo, hi)}; NO se amplia ({why})"
    if name in _BOUNDARY_DECLINED:
        return (
            f"censurado-declarado: rango {_edge_label(spec, lo, hi)}; NO se "
            f"amplia ({_BOUNDARY_DECLINED[name]})"
        )
    if name not in _BOUNDARY_EXPANDABLE:
        return (
            f"sin-clasificar: rango {_edge_label(spec, lo, hi)}; revisar a mano antes "
            "de ampliar (no hay politica pre-registrada)"
        )

    tail = "la proxima pasada con un study nuevo"
    if isinstance(spec, list):
        if at_hi:
            step = spec[-1] - spec[-2]
            what = f"anadir el nivel {round(spec[-1] + step, 10)} por arriba"
        else:
            step = spec[1] - spec[0]
            what = f"anadir un nivel por debajo del minimo (paso {round(step, 10)})"
        return f"ampliable: lista {_edge_label(spec, lo, hi)} -> {what}; {tail}"

    if isinstance(lo, int) and isinstance(hi, int):
        step = 1
    else:
        step = float(hi) - float(lo)
        step = step if step > 0 else 1.0
    new_lo, new_hi = (lo, hi + step) if at_hi else (lo - step, hi)
    widened = f"({_edge_num(new_lo)}, {_edge_num(new_hi)})"
    return f"ampliable: rango {_edge_label(spec, lo, hi)} -> {widened}; {tail}"


def _boundary_audit(space: Any, chosen_params: Any) -> List[Tuple[str, str, str]]:
    """Bordes del espacio que toca el trial ganador: solo diagnostico.

    Devuelve ``(param, valor, recomendacion)`` por cada parametro cuyo valor
    cae en un extremo. ``recomendacion`` empieza por ``ampliable:``,
    ``estructural:``, ``censurado-declarado:`` o ``sin-clasificar:`` para
    que el llamador (y los tests) distingan la politica sin parsear texto
    libre.

    Reglas: tupla ``(lo, hi)`` -> extremo si el valor es ``lo`` o ``hi``;
    lista categorica -> primer/ultimo nivel; spec condicional -> el extremo
    se resuelve con el driver; escalar -> una sola opcion, se ignora.

    NUNCA lanza y NUNCA altera la optimizacion: un ``space``/``chosen``
    inesperado devuelve una lista (posiblemente vacia).
    """
    hits: List[Tuple[str, str, str]] = []
    try:
        if not isinstance(space, dict) or not isinstance(chosen_params, dict):
            return hits
        for name in sorted(space):
            try:
                spec = space[name]
                if name not in chosen_params:
                    continue
                value = chosen_params[name]
                if value is None or isinstance(value, bool):
                    continue
                bounds = _edge_bounds(spec, chosen_params)
                if bounds is None:
                    continue
                lo, hi = bounds
                at_lo = _edge_equal(value, lo)
                at_hi = _edge_equal(value, hi)
                if not (at_lo or at_hi):
                    continue
                hits.append(
                    (
                        name,
                        str(value),
                        _edge_recommendation(name, spec, lo, hi, at_lo, at_hi),
                    )
                )
            except Exception:  # un spec raro no tumba el auditor
                continue
    except Exception:  # ni un `space`/`chosen` raro
        return hits
    return hits


class Optimizador:
    """
    Optimizador de hiperparámetros REII basado en Optuna.

    Sustituye a PatternSearch (gradient_free_optimizers) con:

      · TPE (Tree-structured Parzen Estimator) con soporte multivariado
        → modela cada parámetro según su tipo, no asume orden métrico
          donde no lo hay (categóricos como similarity_threshold).
      · MedianPruner para early stopping
        → corta trials que ya mostraron cobertura por debajo del running
          median, liberando presupuesto para la exploración útil.
      · NSGA-II multi-objective (ARI, coverage, −K)
        → elimina la suma ponderada arbitraria del objective antiguo.
          Devuelve un frente de Pareto de soluciones no-dominadas.
      · Storage SQLite opcional
        → persistencia entre sesiones y paralelización de trials.
      · Reproducibilidad vía sampler seed.

    El espacio de búsqueda y la caché de segmentación/lematización son
    idénticos al optimizador anterior. Los cambios son en el motor de
    búsqueda y en la forma del objetivo (escalar → vector de 3 objetivos).
    """

    def __init__(
        self,
        config_base: Config,
        uc_config: UCBuilderConfig,
        corpus_raw: List[Dict],
        we_analyzer,
        subtlex_analyzer,
        total_uces: int = None,
        progressive_segmenter=None,  # ← NUEVO: instancia de ProgressiveSegmenter o None
        retrofitted_vectors: Optional[Dict[str, np.ndarray]] = None,  # ← NUEVO
    ):
        self.config_base = config_base
        self.corpus_raw = corpus_raw
        self.uc_config = uc_config
        self.we_analyzer = we_analyzer
        self.subtlex_analyzer = subtlex_analyzer
        self.progressive_segmenter = progressive_segmenter
        self.retrofitted_vectors = retrofitted_vectors
        self.segmentador = SegmentadorALCESTE(config_base)
        self.uces_por_doc, self.doc_metadata_map = self.segmentador.segmentar_en_uces(
            corpus_raw
        )
        start_time = time.time()
        for doc_uces in self.uces_por_doc:
            self.segmentador.lematizar_uces(doc_uces)

        self.total_uces = total_uces or sum(len(d) for d in self.uces_por_doc)
        # Cache del espacio de búsqueda — se llena en optimizar()
        self._search_space_cached: Dict[str, Any] = {}
        self._last_pairwise: Optional[Dict[str, Any]] = None
        self._last_selection: Optional[Selection] = None

        # ── Precomputación de cadenas coref por contexto ──────────────────
        # Sólo si hay progressive_segmenter y el modo lo requiere. Con None,
        # la evaluación degrada a wc_emb (Method B devuelve None dentro de
        # DoubleClassifier.run, y el veredicto degrada como ya hace).
        self._precomputed_chains: Optional[Dict[int, Dict[str, List]]] = None
        if progressive_segmenter is not None:
            try:
                self._precomputed_chains = self._load_coref_cache() or {}
            except Exception as e:
                logger.warning(
                    "Carga del cache de cadenas coref falló (%s). "
                    "Se extraerán lazy por trial.",
                    e,
                )
                self._precomputed_chains = {}

        print(
            f"   [Optimizador] Caché listo en {time.time() - start_time:.2f}s. "
            f"Total UCEs: {self.total_uces}"
            + (
                f" · cadenas coref cargadas para ctx={list(self._precomputed_chains.keys())}"
                if self._precomputed_chains is not None
                else " · sin coref (modo degradado)"
            )
        )

    def _precompute_coref_chains(
        self, ctx_values: Optional[Iterable[int]] = None
    ) -> Dict[int, Dict[str, List]]:
        """
        Extrae cadenas de correferencia por documento para cada valor de
        context_units que el optimizador puede pedir (TODO el espacio de
        búsqueda de coref_context_units_tight/loose, no solo los defaults).
        Determinista entre trials: la pre-merge sólo depende de los textos
        de las UCEs, no de los hiperparámetros del trial.

        Devuelve: {context_units: {doc_id: [chain, ...]}}
        """
        ps = self.progressive_segmenter
        # Derivar TODOS los valores posibles del espacio de búsqueda (no solo
        # los defaults del config). Si solo precomputáramos {2, 4}, cualquier
        # trial con context_units ∈ {1, 3, 5} caería al fallback de
        # re-extracción de cadenas con Stanza remoto (la parte cara).
        if ctx_values is None:
            try:
                space = self.get_search_space()
                loose_vals = _expand_levels(space.get("coref_context_units_loose", [4]))
                tight_spec = space.get("coref_context_units_tight", [2])
                if isinstance(tight_spec, dict):
                    # v5: el dependiente es un rango dinamico [lo, max(loose)-1]
                    _lo = int(tight_spec.get("lo", 1))
                    _hi = max(int(c) for c in loose_vals) - 1 if loose_vals else _lo
                    tight_vals = list(range(_lo, max(_lo, _hi) + 1))
                else:
                    tight_vals = _expand_levels(tight_spec)
                ctx_values = set(tight_vals) | set(loose_vals)
            except Exception:
                ctx_values = {
                    int(getattr(self.config_base, "coref_context_units_tight", 2)),
                    int(getattr(self.config_base, "coref_context_units_loose", 4)),
                }
        # Excluir el valor 0 o negativos por seguridad
        ctx_values = {int(c) for c in ctx_values if int(c) > 0}
        if not ctx_values:
            return {}

        result: Dict[int, Dict[str, List]] = {c: {} for c in ctx_values}

        for doc_uces in self.uces_por_doc:
            if not doc_uces:
                continue
            doc_id = doc_uces[0].doc_id

            # Replicar la pre-merge de _clasificar_umbral_coref (paso 3a)
            if len(doc_uces) > 1:
                uce_texts = [u.texto for u in doc_uces]
                try:
                    rec_seg = ps.recursive_segmentation(uce_texts)
                    prog_seg = ps.final_clustering(rec_seg)
                    premerged_groups = _remap_uces_to_segments(doc_uces, prog_seg)
                except Exception as e:
                    logger.warning(
                        "Pre-merge falló para doc %s en precomputación coref: %s",
                        doc_id,
                        e,
                    )
                    premerged_groups = [[u] for u in doc_uces]
            else:
                premerged_groups = [doc_uces]

            # Guard: pre-merge colapsado a 1 grupo en docs largos (mismo que en runtime)
            if len(premerged_groups) == 1 and len(doc_uces) > 50:
                premerged_groups = [[u] for u in doc_uces]

            # Reconstruir segments_info (paso 3b)
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
                        "uce_id": group[-1].id,
                    }
                )
                full_doc_text += group_text + " "
                current_offset = end + 1

            # Extraer cadenas para cada context_units
            for ctx in ctx_values:
                try:
                    chains = ps._extract_global_chains(
                        segments_info, full_doc_text, context_units=ctx
                    )
                    if getattr(ps, "last_extraction_incomplete", False):
                        raise RuntimeError(
                            "extracción incompleta (Stanza/servidor no disponible)"
                        )
                    result[ctx][doc_id] = chains
                except Exception as e:
                    logger.warning(
                        "Extracción de cadenas falló para doc %s ctx=%d: %s",
                        doc_id,
                        ctx,
                        e,
                    )
                    # NO se guarda [] (quedaría persistido como "sin cadenas"):
                    # el doc queda ausente y se re-extrae bajo demanda.
                    print(
                        f"   [Optimizador] ⚠ doc {doc_id} ctx={ctx} NO se cachea: {e}"
                    )

        for ctx, by_doc in result.items():
            total_chains = sum(len(v) for v in by_doc.values())
            print(
                f"   [Optimizador] ctx={ctx}: {total_chains} cadenas "
                f"en {len(by_doc)} documentos precomputadas."
            )
        return result

    def _coref_cache_key(self, fp: str) -> str:
        """Clave compuesta de invalidación: corpus + segmentación."""
        uce_target = int(getattr(self.config_base, "uce_target_size", 26))
        min_words = int(getattr(self.config_base, "min_uce_words", 2))
        return f"{fp}:{uce_target}:{min_words}"

    def _load_coref_cache(self) -> Optional[Dict[int, Dict[str, List]]]:
        """Carga el cache en disco si su cache_key coincide con el corpus +
        segmentación actuales. Devuelve None si no hay cache válido."""
        fp = _corpus_fingerprint()
        cache_key = self._coref_cache_key(fp)
        return load_cache(COREF_CHAINS_CACHE_PATH, cache_key)

    def _save_coref_cache(self, chains: Dict[int, Dict[str, List]]) -> None:
        """Persiste el dict completo de cadenas de forma atómica, sellado con
        cache_key. Nunca lanza (fallo de escritura se loguea y se ignora)."""
        fp = _corpus_fingerprint()
        cache_key = self._coref_cache_key(fp)
        doc = build_cache_document(cache_key, fp, chains)
        try:
            atomic_write_json(COREF_CHAINS_CACHE_PATH, doc)
        except OSError as e:
            logger.warning("No se pudo guardar el cache de cadenas coref: %s", e)

    def _ensure_coref_chains(self, ctx_values: Iterable[int]) -> None:
        """Garantiza que cada ctx en ctx_values esté en self._precomputed_chains.
        Extrae solo los faltantes (lazy) y persiste incrementalmente. Idempotente.
        Nunca lanza: un fallo deja el dict como estaba y el fallback de
        _clasificar_umbral_coref cubre la ausencia.

        Con n_jobs>1 (multiprocessing) cada worker tiene su propia copia del
        dict; el lock de archivo serializa load+compute+save entre workers
        para evitar trabajo duplicado. Si el lock no se puede adquirir en
        `timeout`, se procede sin lock (la escritura atómica evita corrupción;
        el peor caso es trabajo duplicado).
        """
        if self.progressive_segmenter is None:
            return
        if self._precomputed_chains is None:
            self._precomputed_chains = {}
        missing = missing_ctx_values(ctx_values, self._precomputed_chains)
        if not missing:
            return
        lock_path = COREF_CHAINS_CACHE_PATH + ".lock"
        try:
            with file_lock(lock_path, timeout=300.0):
                # Re-check bajo el lock: otro worker pudo computar/guardar
                # mientras esperábamos.
                missing = missing_ctx_values(ctx_values, self._precomputed_chains)
                if missing:
                    disk = self._load_coref_cache() or {}
                    for c in list(missing):
                        if c in disk:
                            self._precomputed_chains[c] = disk[c]
                            missing.remove(c)
                if missing:
                    self._compute_and_save_coref(missing)
        except OSError as e:
            logger.warning(
                "No se pudo adquirir el lock del cache coref (%s). "
                "Se procede sin lock (escritura atómica; puede duplicar trabajo).",
                e,
            )
            self._compute_and_save_coref(missing)

    def _compute_and_save_coref(self, ctx_values: Iterable[int]) -> None:
        """Computa los ctx faltantes y persiste el cache. Nunca lanza."""
        try:
            new_chains = self._precompute_coref_chains(ctx_values)
        except Exception as e:
            logger.warning("Extracción lazy de cadenas coref falló (%s).", e)
            return
        self._precomputed_chains.update(new_chains)
        self._save_coref_cache(self._precomputed_chains)

    # ══════════════════════════════════════════════════════════════════════
    # Persistencia del frente de Pareto
    # ══════════════════════════════════════════════════════════════════════
    def _best_params_path(self) -> str:
        return os.path.join(
            os.path.dirname(self.config_base.db_local_path),
            "best_params.json",
        )

    def save_pareto_front(self, study: "optuna.Study") -> None:
        """
        Guarda el frente de Pareto completo + la solución seleccionada
        según `optimize_preference`.

        Compatibilidad: incluye alias "params"/"score" para que el lector
        legacy en WorkflowOrchestrator.ejecutar() siga funcionando sin
        cambios.
        """
        if not study.best_trials:
            print("   [!] save_pareto_front: frente vacío, nada que guardar.")
            return

        # Excluir trials centinela (todos inválidos → nada que guardar)
        best_trials = [t for t in study.best_trials if not _is_sentinel(t.values)]
        if not best_trials:
            print("   [!] save_pareto_front: sin trials válidos, nada que guardar.")
            return

        best_idx = self._select_preference(best_trials)
        best_trial = best_trials[best_idx]

        def _bp_params(t) -> Dict[str, Any]:
            """params del trial + la fracción CDH derivada.

            Se emite también ``min_cluster_size_cdh`` para que los lectores
            legacy (``rerun_classification``, sondas) no tengan que conocer
            la parametrización por K.
            """
            out = dict(t.params)
            if "cdh_target_k" in out:
                out["min_cluster_size_cdh"] = round(_cdh_frac_from_params(out), 4)
            # v5: los parametros congelados no pasan por suggest_*, asi que no
            # estan en trial.params. Se emiten explicitos para que
            # _apply_best_params replique exactamente lo que corrio el trial.
            for _k, _v in self._search_space_cached.items():
                if not isinstance(_v, (list, tuple, dict)):
                    out.setdefault(_k, _v)
            return out

        payload = {
            # ── Compatibilidad con lector legacy ────────────────────────
            "params": _bp_params(best_trial),
            # values[0] es coverage_primary en stage 2 ("all") y ari en stage 1 ("wc_only").
            "score": float(best_trial.values[0]),
            # ── Nuevo formato Pareto ────────────────────────────────────
            "best_params": _bp_params(best_trial),
            "best_values": list(best_trial.values),
            "preference": self.config_base.optimize_preference,
            "directions": list(self.config_base.optimize_directions),
            "objectives": (
                ["ari", "coverage", "neg_k"]
                if getattr(self, "_current_stage_mode", "select") == "wc_only"
                else ["coverage_primary", "ari_winner", "neg_k"]
                if getattr(self, "_current_stage_mode", "select") == "select"
                else ["coverage_primary", "agreement_sym", "neg_k"]
            ),
            "pareto_front": [
                {
                    "params": _bp_params(t),
                    "values": list(t.values),
                    "trial": int(t.number),
                }
                for t in best_trials
            ],
            "n_trials_total": len(study.trials),
            "n_trials_completed": sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ),
            "n_trials_pruned": sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
            ),
            "timestamp": datetime.now().isoformat(),
            "total_uces": self.total_uces,
            "corpus_fingerprint": _corpus_fingerprint(),
        }
        path = self._best_params_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"   [Optimizador] Pareto front saved to {path}")
        try:
            self.save_comparative_results(study)
        except Exception as e:
            logger.warning("save_comparative_results falló: %s", e)

    def save_comparative_results(self, study: "optuna.Study") -> None:
        """Escribe comparative_results.json con el desglose por trial."""
        path = os.path.join(
            os.path.dirname(self.config_base.db_local_path),
            "comparative_results.json",
        )
        trials_payload = []
        for t in study.trials:
            trials_payload.append(
                {
                    "trial": int(t.number),
                    "state": t.state.name,
                    "params": dict(t.params),
                    "values": list(t.values) if t.values else None,
                    "user_attrs": dict(t.user_attrs),
                }
            )
        payload = {
            "schema_version": 2,
            "corpus_fingerprint": _corpus_fingerprint(),
            "timestamp": datetime.now().isoformat(),
            "study_name": study.study_name,
            "directions": list(study.directions),
            "objective_names": ["coverage_primary", "agreement_sym", "neg_k"],
            "n_trials": len(study.trials),
            "n_completed": sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ),
            "n_pruned": sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
            ),
            "n_failed": sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL
            ),
            "trials": trials_payload,
            "pareto_front": [
                {
                    "trial": int(t.number),
                    "params": dict(t.params),
                    "values": list(t.values),
                }
                for t in study.best_trials
                if not _is_sentinel(t.values)
            ],
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, path)
        print(
            f"   [Optimizador] comparative_results.json escrito ({len(trials_payload)} trials)."
        )
        try:
            self._persist_trials_to_postgres(study)
        except Exception as e:
            logger.warning("Persistencia Postgres de trials falló: %s", e)

    def _persist_trials_to_postgres(self, study: "optuna.Study") -> None:
        from reii.backend.sql.db import connect

        conn = connect(self.config_base.db_dsn)
        try:
            fp = _corpus_fingerprint()
            for t in study.trials:
                conn.execute(
                    """
                    INSERT INTO optimizer_trials
                        (study_name, corpus_fingerprint, trial_number, state,
                         params_json, values_json, user_attrs_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (study_name, corpus_fingerprint, trial_number)
                    DO UPDATE SET
                        state = EXCLUDED.state,
                        params_json = EXCLUDED.params_json,
                        values_json = EXCLUDED.values_json,
                        user_attrs_json = EXCLUDED.user_attrs_json
                    """,
                    (
                        study.study_name,
                        fp,
                        int(t.number),
                        t.state.name,
                        json.dumps(dict(t.params), default=str),
                        json.dumps(list(t.values) if t.values else None, default=str),
                        json.dumps(dict(t.user_attrs), default=str),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def load_best_params(self) -> Optional[Dict]:
        """
        Lee best_params.json en cualquiera de sus dos formatos (legacy
        escalar o nuevo Pareto) y devuelve el dict de parámetros elegido.
        """
        path = self._best_params_path()
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("best_params") or data.get("params")
        if params:
            params = resolve_best_params(params, where=os.path.basename(path))
            print(f"   [Optimizador] Loaded params from {path}: {params}")
        return params

    def _select_preference(self, best_trials: List) -> int:
        """
        Elige el índice del trial del frente de Pareto que se aplicará
        al pipeline definitivo.

        max_ari       → mayor agreement_sym (segundo objetivo, calidad)
        max_coverage  → mayor coverage_primary (primer objetivo)
        knee          → punto más cercano al ideal normalizado (1, 1, 1)
        """
        pref = getattr(self.config_base, "optimize_preference", "max_ari")
        best = best_trials
        if not best:
            return 0

        if pref == "max_coverage":
            return int(max(range(len(best)), key=lambda i: best[i].values[0]))

        if pref == "knee":
            arr = np.array([t.values for t in best], dtype=np.float64)
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
            rng = np.where(maxs > mins, maxs - mins, 1.0)
            norm = (arr - mins) / rng
            # (coverage↑, agreement_sym↑, −K↑): ideal normalizado es (1, 1, 1)
            ideal = np.array([1.0, 1.0, 1.0])
            return int(np.argmin(np.linalg.norm(norm - ideal, axis=1)))

        # default max_ari (agreement_sym)
        return int(max(range(len(best)), key=lambda i: best[i].values[1]))

    # ══════════════════════════════════════════════════════════════════════
    # Espacio de búsqueda (idéntico al optimizador anterior)
    # ══════════════════════════════════════════════════════════════════════
    def get_search_space(self) -> Dict[str, Any]:
        """
        Devuelve el espacio de búsqueda como dict tipado.

        Convención:
          · (lo, hi) con ints → suggest_int
          · (lo, hi) con floats → suggest_float
          · [a, b, c] → suggest_categorical
        """
        f1, _ = self.config_base.min_forms_uc
        tsj = self.config_base.tsj
        # ── Piso del CDH: se barre en unidades de K, no de fracción ───────
        # El árbol CDH parte equilibradamente (K ~ 2^profundidad), así que la
        # fracción solo se distingue en escalones: en la sonda reii_probe_v2
        # 0.20 y 0.15 daban AMBAS K=4, y 0.10 y 0.08 AMBAS K=8. Una rejilla
        # de fracciones duplica niveles efectivos (§2 de cada 5 valores) y
        # deja el óptimo pegado a un techo arbitrario.
        #
        # Ahora el sampler elige el K objetivo y el piso se DERIVA:
        #     min_cluster_size_cdh = 1 / (2K)      (piso ~ n/(2K))
        # El rango de K coincide con el gate de select-v2 (3..20) y se
        # recorta para que el piso absoluto (5 UCEs) sea alcanzable; el techo
        # lo acota además _CDH_TARGET_K_HI (kT no es un objetivo, ver allí).
        n_total = int(getattr(self, "total_uces", 0) or 0)
        k_hi = (
            _CDH_TARGET_K_HI
            if n_total <= 0
            else max(3, min(_CDH_TARGET_K_HI, n_total // 10))
        )

        # ── v5: separacion minima del par de fracciones = 1 UCE ──────────
        # mcs = max(floor, round(frac*n_emb)). Sumar exactamente 1/n_emb
        # garantiza round(frac_t*n) >= round(frac_l*n) + 1 (round(x+1) =
        # round(x)+1), es decir una UCE de separacion real entre C1 y C2.
        _rv = getattr(self, "retrofitted_vectors", None)
        n_emb = int(len(_rv)) if _rv else n_total
        frac_eps = round(1.0 / n_emb, 5) if n_emb > 0 else 0.002
        frac_eps = max(0.0005, frac_eps)

        space: Dict[str, Any] = {
            "min_forms_uc_1": (max(2, f1 - 4), f1 + 4),
            # v6b (patch 19c): el piso NO se baja a 1. `mf2 = mf1 + gap`;
            # con gap<2 las dos pasadas son casi la misma particion y el ARI
            # de 2 pasadas sube por construccion (misma familia que
            # swap_iterations=0). El optimo de v6 cayo en 2 (=piso): se
            # declara en _BOUNDARY_DECLINED, no se amplia.
            "forms_gap": (2, 7),
            # v6 (PRE-REGISTRADO antes del run v6): en v5 el optimo de `tsj`
            # quedo CENSURADO al techo en AMBAS etapas (tsj=5, rango [2, 5]).
            # NO se amplia: `tsj` es umbral de frecuencia documental y subirlo
            # poda vocabulario, lo que baja lemas/UCE (retencion), endpoint
            # registrado del paper (0.451 frente a 0.537 de ALCESTE). La
            # censura se declara en _BOUNDARY_DECLINED y el auditor la deja
            # en el log de cada corrida; el techo se queda en 5.
            "tsj": (max(2, tsj - 2), tsj + 2),
            # pseudocount eliminado (BLOQUE 1: siempre 0.0)
            "cdh_target_k": (3, k_hi),
            # v6 (PRE-REGISTRADO antes del run v6): 0 fuera del espacio.
            # Con 0, `_partition` no llama a `_intercambio` (`if > 0`): la
            # particion es el corte k-means crudo. Eso quita varianza entre
            # pases y SUBE el ARI de estabilidad por construccion, no por
            # mejor estructura: es una ablacion del refinamiento, no un
            # hiperparametro (los ganadores de v5, trials 18/19, caian ahi).
            "swap_iterations": [1, 2, 3, 5],
            # v5 (§3.3): el ganador esta en el maximo (0.10).
            "min_r2_threshold": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
            "similarity_threshold": [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
            # v5 (§3.1): combinacion convexa con similarity_weight=1-w; no
            # hay cota estructural en 0.4.
            "coref_weight": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "uc_window_size": [1, 2, 3, 4, 5],
            # v5 (§3.2): driver primero, dependiente derivado. El ganador del
            # probe esta en (loose=5, tight=3), ambos bordes superiores.
            # 6 y 8 duplican el paso para cubrir el regimen saturado
            # (ctx >= len(doc) es un no-op).
            "coref_context_units_loose": [3, 4, 5, 6, 8],
            "coref_context_units_tight": {
                "kind": "lt",
                "other": "coref_context_units_loose",
                "lo": 1,
            },
            # Fracción de n_embeddable, no absoluto — ver _resolve_hdbscan_uce_params.
            # ([hdbscan-scale-patch])
            # Grilla angostada: REII_EMB_DEBUG confirmó que con corpus chicos
            # (cientos de UCEs) la estructura real de Method C solo aparece
            # en min_cluster_size≈5-20 (sonda a mcs=5: post-PCA 3 clusters,
            # 13% ruido, max_persistence=0.054). El grid anterior
            # (0.03-0.15 → mcs≈16-79 en un corpus de 528) nunca entraba en
            # ese rango — de ahí el clusters=0 sistemático en toda la grilla
            # vieja pese a que la matriz sí tenía densidad real.
            # v5: frac_loose NO se mueve (§3.5): con floor=5, 0.002 == 0.005
            # para el corpus del probe (n_emb=528, ver §3.5 para el limite).
            "hdbscan_uce_min_cluster_frac": {
                "kind": "gt",
                "other": "hdbscan_uce_min_cluster_frac_loose",
                "hi": 0.04,
                "eps": frac_eps,
            },
            "hdbscan_uce_min_cluster_frac_loose": [0.005, 0.01, 0.015, 0.02, 0.03],
        }

        # ── v5 (§5.4): congelar los inertes (rho~0, n=11). El dict admite
        # escalares: _suggest los devuelve tal cual, sin suggest_*.
        if bool(getattr(self.config_base, "optimize_freeze_inert", True)):
            space["min_forms_uc_1"] = int(f1)
            space["uc_window_size"] = int(getattr(self.uc_config, "window_size", 3))
        return space

    @staticmethod
    def _suggest(trial: "optuna.Trial", space: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        pending: Dict[str, Any] = {}
        for k, v in space.items():
            # v5: condicional -> segunda pasada, cuando el driver ya esta
            # muestreado (no depende del orden de insercion del dict).
            if isinstance(v, dict):
                pending[k] = v
                continue
            if isinstance(v, tuple) and len(v) == 2:
                lo, hi = v
                if isinstance(lo, int) and isinstance(hi, int):
                    params[k] = trial.suggest_int(k, lo, hi)
                else:
                    params[k] = trial.suggest_float(k, float(lo), float(hi))
            elif isinstance(v, list):
                params[k] = trial.suggest_categorical(k, v)
            else:
                params[k] = v
        for k, spec in pending.items():
            params[k] = _suggest_conditional(trial, k, spec, params)
        return params

    # ══════════════════════════════════════════════════════════════════════
    # Evaluación con caché (idéntica a la anterior)
    # ══════════════════════════════════════════════════════════════════════
    def _evaluacion_rapida(self, cfg: Config, uc_cfg: UCBuilderConfig = None):
        uc_config_actual = uc_cfg or self.uc_config
        uc_builder = UCBuilder(
            self.we_analyzer, self.subtlex_analyzer, uc_config_actual, self.config_base
        )
        uc_builder.vectorizer.clear_cache()  # evita contaminación entre trials
        # ← Inyectar progressive_segmenter para Method B
        double_clf = DoubleClassifier(
            cfg,
            self.segmentador,
            uc_builder,
            progressive_segmenter=self.progressive_segmenter,
        )

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
            resultados = double_clf.run(
                self.corpus_raw,
                retrofitted_vectors=self.retrofitted_vectors,  # ← NUEVO
                precomputed_coref_chains=self._precomputed_chains,  # ← NUEVO
            )
        finally:
            self.segmentador.segmentar_en_uces = original_seg
            self.segmentador.lematizar_uces = original_lem

        # Persistir last_pairwise_stability para que _evaluate_trial lo consuma
        self._last_pairwise = getattr(double_clf, "last_pairwise_stability", None)
        # Última selección de método (modo "select"): el objetivo de la
        # etapa 2 es (coverage, ARI del ganador, -K), no agreement_sym.
        self._last_selection = getattr(double_clf, "last_selection", None)
        return resultados

    # ══════════════════════════════════════════════════════════════════════
    # Objetivo multi-objective
    # ══════════════════════════════════════════════════════════════════════
    def _evaluate_trial(self, trial: "optuna.Trial"):
        """
        Devuelve la tupla objetivo o un centinela dominado.

        Etapa 1 (wc_only): (ARI, coverage, −K) — prior para la etapa 2.
        Etapa 2 (all):     (coverage, agreement_sym, −K) — veredicto final.

        Los gates duros (K fuera de [2,8], coverage < umbral, excepción
        en la evaluación) devuelven _SENTINEL_TRIAL en vez de lanzar
        TrialPruned. Motivo: Optuna 3.6.2 (pin <4.0.0) crashea en TPE
        multi-objetivo cuando un trial PRUNED (values=None) cae en el
        conjunto "below" del Parzen estimator. El centinela es
        estrictamente dominado por cualquier trial real, así que no
        contamina el frente de Pareto ni el conjunto "below" del sampler.

        Nota sobre pruning: al no lanzar TrialPruned, el MedianPruner
        queda inerte; el filtrado real lo hacen los gates duros de
        coverage y de K.
        """
        params = self._suggest(trial, self._search_space_cached)

        mode = getattr(self, "_current_stage_mode", "all")

        mf1 = params["min_forms_uc_1"]
        mf2 = mf1 + params["forms_gap"]

        cfg = copy.deepcopy(self.config_base)
        cfg.classification_mode = mode
        cfg.min_forms_uc = [mf1, mf2]
        cfg.tsj = params["tsj"]
        # BLOQUE 1: nunca sumar ε. En clásico se conserva el valor declarado
        # para poder REPORTARLO (sigue siendo 0.0 mientras ε no exista en la
        # CDH — ver CLASSIC_MODE.md §3). En moderno, invariante 0.0.
        if str(getattr(cfg, "run_mode", "modern")) != "classic":
            cfg.pseudocount = 0.0
        cfg.min_cluster_size_cdh = _cdh_frac_from_params(
            params, default=cfg.min_cluster_size_cdh
        )
        cfg.swap_iterations = params["swap_iterations"]
        cfg.min_r2_threshold = params["min_r2_threshold"]
        cfg.use_liminal_projection = False
        cfg.optimize = cfg.use_projection = cfg.use_network_analysis = False

        # Parámetros nuevos del espacio de búsqueda (coref + HDBSCAN UCE)
        cfg.coref_context_units_tight = params.get(
            "coref_context_units_tight",
            getattr(cfg, "coref_context_units_tight", 2),
        )
        cfg.coref_context_units_loose = params.get(
            "coref_context_units_loose",
            getattr(cfg, "coref_context_units_loose", 4),
        )
        cfg.hdbscan_uce_min_cluster_frac = params.get(
            "hdbscan_uce_min_cluster_frac",
            getattr(cfg, "hdbscan_uce_min_cluster_frac", 0.08),
        )
        cfg.hdbscan_uce_min_cluster_frac_loose = params.get(
            "hdbscan_uce_min_cluster_frac_loose",
            getattr(cfg, "hdbscan_uce_min_cluster_frac_loose", 0.05),
        )

        # ── Gate: tight debe ser estrictamente menor que loose (coref), y
        # la fracción tight debe ser estrictamente MAYOR que la loose (más
        # grande = clusters más grandes/pocos = "tight"). ([hdbscan-scale-patch]) ─────────
        if cfg.coref_context_units_tight >= cfg.coref_context_units_loose:
            return _sentinel(trial, "espacio: tight >= loose")
        if cfg.hdbscan_uce_min_cluster_frac <= cfg.hdbscan_uce_min_cluster_frac_loose:
            return _sentinel(trial, "espacio: frac <= frac_loose")

        # ── Lazy: asegurar cadenas coref para los ctx de ESTE trial ──────
        # Solo si el modo del trial realmente usa Method B — en wc_only
        # (etapa 1) _do_coref es False más abajo en _evaluacion_rapida, así
        # que precomputar aquí es trabajo puro desperdiciado (premerge CPU +
        # Stanza remoto por cada (tight, loose) nuevo que Optuna muestree).
        _mode_needs_coref = mode in (
            "coref_only",
            "wc_coref",
            "coref_emb",
            "all",
            "select",
            "auto",
        )
        if _mode_needs_coref and self.progressive_segmenter is not None:
            self._ensure_coref_chains(
                [cfg.coref_context_units_tight, cfg.coref_context_units_loose]
            )

        uc_cfg = copy.deepcopy(self.uc_config)
        uc_cfg.similarity_threshold = params.get(
            "similarity_threshold", uc_cfg.similarity_threshold
        )
        uc_cfg.coref_weight = params.get("coref_weight", uc_cfg.coref_weight)
        uc_cfg.window_size = params.get("uc_window_size", uc_cfg.window_size)

        # v5: sin try/except local. La excepcion sube al guard de
        # study.optimize (_make_guarded_objective), que la convierte en
        # centinela CON traceback. Antes se descartaba aqui y el trial
        # quedaba indistinguible de un gate (o FAIL sin rastro, #81).
        result = self._evaluacion_rapida(cfg, uc_cfg=uc_cfg)

        primary_stable = result[0] if result else None
        if not primary_stable:
            return _sentinel(trial, "sin primary_stable")

        labels_primary = [
            u.cluster_id for u in primary_stable if u.cluster_id is not None
        ]
        if len(labels_primary) == 0:
            return _sentinel(trial, "sin labels primarios")

        n_clusters = int(len(np.unique(labels_primary)))

        # ── K real vs K pedido (CLASSIC_MODE.md §7, HARDENING.md §5) ──────
        # `n_clusters` cuenta solo UCEs ESTABLES. El K que el sampler cree
        # haber evaluado es `cdh_target_k`; el gate se aplica al K REAL.
        _k_target = params.get("cdh_target_k")
        _k_stable = int(n_clusters)
        _k_ach = _k_stable  # se refina por rama (winner_k / A1 crudo)

        # ── Gate duro: K fuera de rango util ─────────────────────────────
        # Se aplica al K REAL: un ganador con K=25 no debe llegar al frente
        # como -25 (HARDENING.md §8.3).
        if _k_ach < 2 or _k_ach > 20:
            trial.set_user_attr("k_rejected", int(_k_ach))
            return _sentinel(trial, "K fuera de [2, 20]")

        coverage_primary = len(primary_stable) / max(1, self.total_uces)

        # ── Gate duro de cobertura, dependiente del modo ─────────────────
        if mode == "select":
            coverage_gate = float(
                getattr(self.config_base, "optimize_coverage_gate_select", 0.10)
            )
        elif mode == "all":
            coverage_gate = float(
                getattr(self.config_base, "optimize_coverage_gate_all", 0.10)
            )
        else:
            coverage_gate = float(
                getattr(self.config_base, "optimize_coverage_gate", 0.25)
            )
        if coverage_primary < coverage_gate:
            return _sentinel(trial, "coverage < gate")

        # ── Etapa 2 (select): un solo método, sin agreement_sym ───────────
        # El sentinel de la etapa 2 anterior ("all") venía de
        # `if not primary_stable` sobre una intersección triple que era vacía
        # en el 100% de los trials. Con "select" el primary_stable es el del
        # método ganador: si algún método es estable, hay resultado.
        if mode == "select":
            sel = self._last_selection
            if sel is None or sel.winner is None:
                # Abstención (ningun metodo viable). Se registra el motivo: si
                # el 100% de los trials son centinela, 3_diagnose_optuna_db.py
                # puede distinguir "gates infactibles" de "abstencion".
                if sel is not None:
                    trial.set_user_attr("abstained", True)
                    trial.set_user_attr("selection_reason", str(sel.reason)[:200])
                    trial.set_user_attr(
                        "discarded",
                        "; ".join(f"{d['method']}: {d['why']}" for d in sel.discarded)[
                            :300
                        ],
                    )
                return _sentinel(trial, "abstencion (sin metodo viable)")
            if not math.isfinite(float(sel.winner_ari)):
                return _sentinel(trial, "ARI del ganador no finito")
            _win_row = next(
                (r for r in (sel.all_metrics or []) if r.get("method") == sel.winner),
                {},
            )
            trial.set_user_attr("selected_method", sel.winner)
            trial.set_user_attr("ari_winner", float(sel.winner_ari))
            trial.set_user_attr("coverage_primary", float(coverage_primary))
            trial.set_user_attr(
                "usable_coverage_winner",
                float(sel.winner_usable_coverage),
            )
            trial.set_user_attr(
                "tiny_frac_winner", float(_win_row.get("tiny_frac") or 0.0)
            )
            trial.set_user_attr(
                "noise_frac_winner", float(_win_row.get("noise_frac") or 0.0)
            )
            trial.set_user_attr(
                "max_class_share_winner",
                float(_win_row.get("max_class_share") or 0.0),
            )
            trial.set_user_attr(
                "dominance_corpus_winner",
                float(_win_row.get("dominance_corpus_eff") or 0.0),
            )
            trial.set_user_attr("abstained", False)
            trial.set_user_attr("n_clusters", n_clusters)
            trial.set_user_attr("selection_ambiguous", bool(sel.ambiguous))
            trial.set_user_attr(
                "selection_criteria_conflict", bool(sel.criteria_conflict)
            )
            # ── Kach vs target_k (CLASSIC_MODE.md §7) ────────────────────
            # El objetivo −K debe ser el K REAL de la partición del ganador.
            # `n_clusters` cuenta sólo UCEs estables: un trial puede haber
            # pedido K=8 y producir 5 hojas. Se reporta la brecha en vez de
            # fingir que se alcanzó el objetivo del sampler.
            _k_ach = int(getattr(sel, "winner_k", 0) or n_clusters)
            # HARDENING.md §8.3: la revalidación 2..20 va DESPUÉS del
            # refinamiento (winner_k=25 no debe llegar al frente como -25).
            if _k_ach < 2 or _k_ach > 20:
                trial.set_user_attr("k_rejected", int(_k_ach))
                return _sentinel(trial, "select: K real fuera de [2, 20]")
            trial.set_user_attr("k_ach", _k_ach)
            trial.set_user_attr(
                "k_target", int(_k_target) if _k_target is not None else -1
            )
            trial.set_user_attr(
                "k_shortfall",
                int(_k_target) - _k_ach if _k_target is not None else 0,
            )
            trial.set_user_attr("k_stable", _k_stable)
            return (
                float(coverage_primary),
                float(sel.winner_ari),
                -float(_k_ach),
            )

        # ── Etapa 1 (wc_only): ARI como prior ─────────────────────────────
        if mode == "wc_only":
            all_res = result[4] if len(result) > 4 else []
            if len(all_res) < 2 or all_res[0] is None or all_res[1] is None:
                return _sentinel(trial, "wc_only: faltan los dos pases")
            uce_to_a1 = DoubleClassifier._extract_uce_labels(all_res[0])
            uce_to_a2 = DoubleClassifier._extract_uce_labels(all_res[1])
            common = sorted(set(uce_to_a1) & set(uce_to_a2))
            if len(common) < 2:
                return _sentinel(trial, "wc_only: <2 UCEs comunes")
            arr_a = np.array([uce_to_a1[u] for u in common])
            arr_b = np.array([uce_to_a2[u] for u in common])
            valid = (arr_a >= 0) & (arr_b >= 0)
            if valid.sum() < 2:
                return _sentinel(trial, "wc_only: <2 UCEs validas")
            ari = float(adjusted_rand_score(arr_a[valid], arr_b[valid]))
            # K de la partición CRUDA de A1 (no el de la estable).
            _k_ach = int(len(np.unique(arr_a[valid])))
            # HARDENING.md §8.3: la revalidación 2..20 va DESPUÉS del
            # refinamiento (el K crudo de A1 manda sobre n_clusters).
            if _k_ach < 2 or _k_ach > 20:
                trial.set_user_attr("k_rejected", int(_k_ach))
                return _sentinel(trial, "wc_only: K real fuera de [2, 20]")
            trial.set_user_attr("k_ach", _k_ach)
            trial.set_user_attr(
                "k_target", int(_k_target) if _k_target is not None else -1
            )
            trial.set_user_attr(
                "k_shortfall",
                int(_k_target) - _k_ach if _k_target is not None else 0,
            )
            trial.set_user_attr("k_stable", _k_stable)
            return ari, float(coverage_primary), -float(_k_ach)

        # ── Etapa 2 (all): veredicto final vía agreement simétrico ───────
        pairwise = self._last_pairwise or {}

        # ── Gate: Method C (embeddings) debe producir ≥2 clusters en AMBOS
        # pases (C1 tight, C2 loose). Sin este gate, cuando Method C falla
        # (p.ej. hdbscan_uce_min_cluster_frac demasiado grande para el
        # corpus), _triple_stability_verdicts degrada 'all' a 'wc_coref' en
        # silencio: coverage_primary y n_clusters quedan basados en A↔B
        # (ninguno de los dos objetivos "ve" el fallo de C). Solo
        # agreement_sym se resiente (ari_bc=ari_ac=0.0), y promediado con
        # ari_ab eso no basta para dominar el trial de forma consistente.
        if not pairwise.get("emb_persistence_c1") or not pairwise.get(
            "emb_persistence_c2"
        ):
            return _sentinel(trial, "Method C sin persistencia en C1/C2")

        ari_ab = float(pairwise.get("ari_cross_ab", 0.0))
        ari_bc = float(pairwise.get("ari_cross_bc", 0.0))
        ari_ac = float(pairwise.get("ari_cross_ac", 0.0))
        agreement_sym = float(np.mean([ari_ab, ari_bc, ari_ac]))

        trial.set_user_attr("coverage_primary", float(coverage_primary))
        trial.set_user_attr("agreement_sym", agreement_sym)
        trial.set_user_attr("n_clusters", n_clusters)
        trial.set_user_attr(
            "bottleneck_method", self._bottleneck(ari_ab, ari_bc, ari_ac)
        )

        return float(coverage_primary), agreement_sym, -float(n_clusters)

    @staticmethod
    def _active_methods(cfg) -> Set[str]:
        mode = getattr(cfg, "classification_mode", "wc_only")
        aliases = {
            "auto": "select",
            "both": "wc_coref",
            "wc_sim": "wc_coref",
            "sim_only": "coref_only",
            "sim_emb": "coref_emb",
        }
        mode = aliases.get(mode, mode)
        return {
            "wc_only": {"wc"},
            "coref_only": {"coref"},
            "emb_only": {"emb"},
            "wc_coref": {"wc", "coref"},
            "wc_emb": {"wc", "emb"},
            "coref_emb": {"coref", "emb"},
            "all": {"wc", "coref", "emb"},
            "select": {"wc", "coref", "emb"},
        }.get(mode, {"wc"})

    @staticmethod
    def _bottleneck(ari_ab: float, ari_bc: float, ari_ac: float) -> str:
        mean_wc = np.mean([ari_ab, ari_ac])
        mean_coref = np.mean([ari_ab, ari_bc])
        mean_emb = np.mean([ari_bc, ari_ac])
        candidates = {"wc": mean_wc, "coref": mean_coref, "emb": mean_emb}
        return min(candidates, key=candidates.get)

    # ══════════════════════════════════════════════════════════════════════
    # Orquestación Optuna
    # ══════════════════════════════════════════════════════════════════════
    def _make_sampler(self):
        name = getattr(self.config_base, "optimize_sampler", "tpe")
        seed = self.config_base.random_state
        n_startup = int(getattr(self.config_base, "optimize_n_startup_trials", 12))
        multivariate = bool(getattr(self.config_base, "optimize_multivariate", True))

        if name == "tpe":
            return TPESampler(
                seed=seed,
                n_startup_trials=n_startup,
                multivariate=multivariate,
                n_ei_candidates=int(
                    getattr(self.config_base, "optimize_n_ei_candidates", 48)
                ),
                # v5: tight/frac_loose se muestrean con rango dinamico, asi
                # que Optuna los excluye del TPE multivariado y avisa una vez
                # por trial. Es esperado (el par es un triangulo, no una
                # gaussiana); se silencia para no inundar el log.
                warn_independent_sampling=False,
            )
        if name == "random":
            return RandomSampler(seed=seed)
        if name == "cmaes":
            return CmaEsSampler(seed=seed)
        raise ValueError(f"Sampler desconocido: {name}")

    def _make_pruner(self, n_trials: int):
        name = getattr(self.config_base, "optimize_pruner", "median")
        startup = int(getattr(self.config_base, "optimize_prune_n_startup", 15))
        if name == "median":
            return MedianPruner(n_startup_trials=startup, n_warmup_steps=0)
        if name == "hyperband":
            return HyperbandPruner(min_resource=1, max_resource=n_trials)
        return NopPruner()

    def _normalize_storage(self, storage: Optional[str], study_name: str):
        """Remapea el storage sqlite a la carpeta de datos montada y aísla
        el estudio por fingerprint del corpus.

        - storage falsy → estudio en memoria (None, study_name).
        - sqlite relativo (sqlite:///reii_optuna.db) → se reescribe dentro
          de la carpeta de datos (montada en el contenedor; la misma de
          best_params.json).
        - El nombre del estudio se sella con el fingerprint del corpus para
          que un cambio de dataset no reutilice trials viejos.
        """
        if not storage:
            return None, study_name
        if storage.startswith("sqlite:///"):
            db_path = storage[len("sqlite:///") :]
            # isabs() en Windows no reconoce rutas POSIX (/app/...);
            # comprobamos también el prefijo '/' explícitamente.
            if not (os.path.isabs(db_path) or db_path.startswith("/")):
                data_dir = os.path.dirname(self.config_base.db_local_path)
                db_path = os.path.join(data_dir, os.path.basename(db_path)).replace(
                    "\\", "/"
                )
                storage = f"sqlite:///{db_path}"
        study_name = f"{study_name}_{_corpus_fingerprint()}"
        return storage, study_name

    def optimizar(
        self,
        n_trials: int = 200,
        two_stage: bool = True,
        stage1_fraction: float = 0.3,
    ) -> Dict[str, Any]:
        if not _OPTUNA_AVAILABLE:
            print("   [!] optuna no instalado. Ejecuta: pip install optuna")
            return {}

        # Cache del espacio de búsqueda — se consulta en cada trial
        self._search_space_cached = self.get_search_space()

        if not two_stage:
            params, _ = self._run_single_stage(
                n_trials=n_trials, mode="select", warm_start_trials=[]
            )
            return params

        # El piso de 20 trials de la etapa 1 puede agotar (o superar) el
        # presupuesto total: con --trials 12 y frac=0.5 daba n_stage1=20 y
        # n_stage2=-8, de modo que la etapa 2 no ejecutaba NINGUN trial y el
        # resumen lo atribuia a centinelas. Se garantiza 1 trial por etapa.
        n_stage1 = max(20, int(n_trials * stage1_fraction))
        if n_stage1 >= n_trials:
            n_stage2_prev = max(1, int(n_trials * (1.0 - stage1_fraction)))
            n_stage1 = max(1, n_trials - n_stage2_prev)
            print(
                f"   [!] Presupuesto de {n_trials} trials < suelo de 20 de la"
                f" etapa 1: reparto ajustado a {n_stage1} (etapa 1) +"
                f" {n_trials - n_stage1} (etapa 2)."
            )
        _, stage1_pareto = self._run_single_stage(
            n_trials=n_stage1,
            mode="wc_only",
            warm_start_trials=[],
            study_suffix="_stage1_wc",
        )

        n_stage2 = max(1, n_trials - n_stage1)
        stage2_params, _ = self._run_single_stage(
            n_trials=n_stage2,
            mode="select",
            warm_start_trials=stage1_pareto,
            study_suffix="_stage2_select",
            perturb_warm_start=True,
        )
        return stage2_params

    def _run_single_stage(
        self,
        n_trials: int,
        mode: str,
        warm_start_trials: List,
        study_suffix: str = "",
        perturb_warm_start: bool = False,
    ) -> Tuple[Dict[str, Any], List]:
        self._current_stage_mode = mode

        if n_trials <= 0:
            print(
                f"   [!] Etapa '{mode}' sin presupuesto de trials "
                f"(n_trials={n_trials}); no se ejecuta."
            )
            return {}, []

        sampler = self._make_sampler()
        pruner = self._make_pruner(n_trials)

        directions = list(
            getattr(
                self.config_base,
                "optimize_directions",
                ("maximize", "maximize", "minimize"),
            )
        )
        storage = getattr(self.config_base, "optimize_storage", None)
        study_name = getattr(self.config_base, "optimize_study_name", "reii_search")
        study_name = f"{study_name}{study_suffix}"

        # ── Storage: el CWD del contenedor es /app (NO montado) ─────────
        # sqlite:///reii_optuna.db se crearía en /app y se perdería al
        # salir del contenedor; además load_if_exists=True cargaría trials
        # stale o un DB corrupto. Remapeamos el DB a la carpeta de datos
        # montada (la misma de best_params.json) y aislamos el estudio por
        # fingerprint del corpus. Si el DB no es utilizable, caemos a
        # estudio en memoria.
        storage, study_name = self._normalize_storage(storage, study_name)

        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                load_if_exists=bool(storage),
                sampler=sampler,
                pruner=pruner,
                directions=directions,
            )
        except Exception as e:
            logger.warning(
                "Storage Optuna no utilizable (%s: %s). Usando estudio en memoria.",
                type(e).__name__,
                e,
            )
            study = optuna.create_study(
                study_name=study_name,
                storage=None,
                load_if_exists=False,
                sampler=sampler,
                pruner=pruner,
                directions=directions,
            )

        # ── Warm start: enqueue de trials previos (y copias perturbadas) ─
        # Sin duplicados ni puntos infactibles (gates tight<loose / frac_t>frac_l):
        # cada uno consumiría un trial del presupuesto para devolver el centinela.
        _seen_enq = set()

        def _enqueue(pr):
            key = tuple(sorted(pr.items()))
            if key in _seen_enq or not self._params_feasible(pr):
                return
            _seen_enq.add(key)
            study.enqueue_trial(pr)

        if _NO_OPTUNA_WARM_START:
            print("   [CLI] --no-optuna-warm-start: etapa 2 sin warm start.")
            warm_start_trials = []

        # v5 (§5.5): el frente de etapa 1 puede traer decenas de puntos y cada
        # uno se encola x3; sin tope, el warm start consume el presupuesto
        # entero de la etapa 2 y TPE nunca muestrea.
        _ws_max = int(getattr(self.config_base, "optimize_warm_start_max", 10))
        for prev_trial in list(warm_start_trials)[:_ws_max]:
            _enqueue(dict(prev_trial.params))
            if perturb_warm_start:
                for _salt in range(2):
                    _enqueue(self._perturb_params(prev_trial.params, salt=_salt))

        print(f"\n=== Optuna · {n_trials} trials (mode={mode}) ===")
        print(f"   Sampler    : {self.config_base.optimize_sampler}")
        print(f"   Pruner     : {self.config_base.optimize_pruner}")
        _obj_names = (
            ["ari", "coverage", "n_clusters"]
            if mode == "wc_only"
            else ["coverage", "ari_winner", "n_clusters"]
            if mode == "select"
            else ["coverage", "agreement", "n_clusters"]
        )
        print(f"   Objectives : {list(zip(_obj_names, directions))}")
        print(f"   UCEs       : {self.total_uces}")
        _cdh_space = self._search_space_cached.get("cdh_target_k")
        if isinstance(_cdh_space, tuple) and len(_cdh_space) == 2:
            _lo, _hi = _cdh_space
            print(
                f"   CDH floor  : K_obj en [{_lo}, {_hi}] -> "
                f"frac=1/(2K) en [{1.0 / (2.0 * _hi):.3f}, {1.0 / (2.0 * _lo):.3f}]"
            )

        study.optimize(
            _make_guarded_objective(self._evaluate_trial),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        # ── Resumen final ────────────────────────────────────────────────
        n_completed = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        n_pruned = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        )
        n_failed = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL
        )
        print(
            f"\n   >>> Estado: {n_completed} completados, "
            f"{n_pruned} pruned, {n_failed} failed"
        )
        # v5: separar los dos centinelas (antes se confundian).
        _n_gate, _n_exc, _exc_detail = _sentinel_breakdown(study.trials)
        print(
            f"=> Centinelas: {_n_gate} por gates de factibilidad · "
            f"{_n_exc} por excepción"
        )
        if _n_exc:
            print(
                f"[!] {_n_exc} trial(s) con excepcion; el detalle esta en "
                f"user_attr 'traceback' (comparative_results.json)"
            )
            for _line in _exc_detail[:5]:
                print(f"[X] {_line[:160]}")

        best_trials = [t for t in study.best_trials if not _is_sentinel(t.values)]
        if not best_trials:
            if n_completed == 0:
                print(
                    "   [!] Sin soluciones válidas: la etapa no ejecutó ningún trial."
                )
            else:
                print(
                    "   [!] Sin soluciones válidas (todos los trials fueron centinela)."
                )
            return {}, []

        # Persistir frente completo
        self.save_pareto_front(study)

        # Selección según preferencia
        best_idx = self._select_preference(best_trials)
        best_trial = best_trials[best_idx]

        # ── Auditor de bordes (v6, WARN-only): diagnostico, no decision ──
        # Corre DESPUES de elegir el ganador y no toca ni `best_trial` ni el
        # study: si un parametro quedo censurado, se reporta aqui para
        # planificar la proxima campana sin reescribir esta corrida.
        try:
            _audit = _boundary_audit(
                getattr(self, "_search_space_cached", {}) or {},
                dict(getattr(best_trial, "params", {}) or {}),
            )
            print("   ── BOUNDARY AUDIT (WARN-only; no altera el resultado) ──")
            if not _audit:
                print("   [OK] ningun parametro elegido esta en un borde.")
            for _name, _val, _rec in _audit:
                print(f"   [BORDE] {_name} = {_val} | {_rec}")
        except Exception as _exc:  # un diagnostico nunca tumba la corrida
            logger.debug("BOUNDARY AUDIT omitido: %s", _exc)

        print(
            f"\n   >>> Pareto front: {len(study.best_trials)} soluciones no-dominadas"
        )
        print(
            f"   >>> Seleccionada (preference="
            f"{self.config_base.optimize_preference}): trial #{best_trial.number}"
        )
        print(
            f"       coverage={best_trial.values[0]:.4f}  "
            f"agreement={best_trial.values[1]:.4f}  "
            f"K={-best_trial.values[2]:.0f}"
        )
        print(f"       params: {best_trial.params}")

        return dict(best_trial.params), best_trials

    @staticmethod
    def _params_feasible(p: Dict[str, Any]) -> bool:
        """Mismos gates estructurales que _evaluate_trial (sin coste de evaluación)."""
        t, l = p.get("coref_context_units_tight"), p.get("coref_context_units_loose")
        if t is not None and l is not None and t >= l:
            return False
        a, b = (
            p.get("hdbscan_uce_min_cluster_frac"),
            p.get("hdbscan_uce_min_cluster_frac_loose"),
        )
        if a is not None and b is not None and a <= b:
            return False
        return True

    def _perturb_params(self, params: Dict[str, Any], salt: int = 0) -> Dict[str, Any]:
        import random

        # Semilla derivada de (random_state, salt, params): reproducible pero
        # distinta por copia. Antes: Random(random_state) idéntico en cada
        # llamada → las 2 copias "perturbadas" de un trial eran duplicados.
        rng = random.Random(
            f"{self.config_base.random_state}:{salt}:{sorted(params.items())}"
        )
        perturbed = dict(params)
        space = self._search_space_cached
        for k, v in params.items():
            spec = space.get(k)
            if spec is None:
                continue
            if isinstance(spec, tuple) and len(spec) == 2:
                lo, hi = spec
                step = 1 if isinstance(lo, int) else (hi - lo) * 0.1
                delta = rng.choice([-step, +step])
                new = max(lo, min(hi, v + delta))
                perturbed[k] = int(new) if isinstance(lo, int) else float(new)
            elif isinstance(spec, list) and len(spec) > 1:
                try:
                    idx = spec.index(v)
                    new_idx = max(0, min(len(spec) - 1, idx + rng.choice([-1, +1])))
                    perturbed[k] = spec[new_idx]
                except ValueError:
                    continue
        return perturbed


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
        self.db = Database(config, dual_write=config.dual_write_json)
        # Publicar defaults de config en PostgreSQL (ON CONFLICT DO NOTHING: no pisa ediciones del dashboard)
        try:
            from reii.backend.config_store import ConfigStore

            # v5: en clásico el seed NO se aplica (el modo fija sus overrides
            # después); en moderno se conserva el comportamiento previo.
            if str(getattr(config, "run_mode", "modern")) != "classic":
                ConfigStore(config.db_dsn).seed_from_config(config)
        except Exception as e:
            print(f"   [WARNING] ConfigStore.seed_from_config falló: {e}")
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
                shared = doc_metadata_map.get(int(uce.doc_id), {})
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
        # Serialize manually so texto_completo_doc survives to_dict(),
        # y para persistir los campos dinámicos de la fase abductiva
        # (projected_cluster_id, projection_distance, ...). Estos campos
        # NO son declarados en la dataclass UCE y por tanto asdict() los
        # ignoraría por defecto.
        serialized = []
        for uce in all_uces:
            d = uce.to_dict()
            d["texto_completo_doc"] = texto_map.get(str(uce.doc_id), "")

            # Campos de proyección liminal (siempre presentes, None si no
            # hubo proyección o si la UCE no era huérfana)
            for field_name in _UCE_PROJECTION_FIELDS:
                d[field_name] = getattr(uce, field_name, None)

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
            cfg["corpus_name"] = CORPUS_NAME
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
                1
                for v in getattr(uce, "verbos", [])
                if v.get("modo") in {"Sub", "Subj"}
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

    def _compute_retrofitted_vectors(
        self,
        resultados: Optional[List[Dict]],
        voc: Optional[List[str]],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Build retrofitted vectors from a pass-1 result (resultados[0] + voc).

        Returns None if there are no significant terms or inputs are empty.
        Mirrors the inline retrofitting block that used to live in ejecutar().
        """
        if not resultados or not voc:
            return None
        _all_ucs = resultados[0]["ucs"]
        _labels = resultados[0]["labels"]
        _builder = MatrizBuilder(self.config)
        _mat = _builder.construir_matriz(_all_ucs, voc)
        _df_terms = TermAnalyzer(self.config).compute(_mat, _labels, voc)
        if not _df_terms.empty and _df_terms["significativo"].any():
            _binary = _builder.construir_matriz_dispersa(_all_ucs, voc)
            _ppmi = self.ppmi_builder.build(_binary, voc)
            retro_vecs = self.embedding_pipeline.run(_ppmi, voc, _df_terms)
            self.uc_builder.vectorizer.clear_cache()  # invalidate before pass 2
            print(f"   Pass-1 retrofitting complete: {len(retro_vecs)} vectors.")
            return retro_vecs
        print("   Pass-1 had no significant terms; skipping retrofitting.")
        return None

    def ejecutar(
        self,
        corpus_raw: Dict[str, Any],
        global_corpus=None,
        grammatical_pipeline: Optional[PipelineGramatical] = None,
        grammatical_dashboard_path: str = "",
        lex_analyzer=None,  # GlobalLexicalAnalyzer instance, optional
        checkpoint_path: Optional[str] = None,
        force: bool = False,
    ) -> None:
        print("=== Iniciando Pipeline ALCESTE v5 ===")

        # ── 0. Auto-calibración de uce_target_size (antes de TODO lo demás:
        #      best_params, el Optimizador y la segmentación real comparten
        #      self.config por referencia, así que corregirlo acá se propaga
        #      a todo el resto de esta ejecución) ───────────────────────────
        if getattr(self.config, "auto_calibrate_uce_size", True):
            target_calibrado = calibrar_uce_target_size(self.segmentador, corpus_raw)
            if target_calibrado is not None:
                self.config.uce_target_size = target_calibrado

        # ── 1. Optimización de hiperparámetros (El núcleo robusto) ─────────────
        cached_uces_por_doc = None
        cached_doc_metadata_map = None

        def _apply_best_params(bp: Dict[str, Any]) -> None:
            """Aplica los hiperparámetros ganadores al Config activo."""
            bp = resolve_best_params(bp, where="best_params (apply)")
            mf1 = bp.get("min_forms_uc_1", self.config.min_forms_uc[0])
            gap = bp.get("forms_gap", self.config.min_forms_uc[1] - mf1)
            self.config.min_forms_uc = [mf1, mf1 + gap]
            self.config.tsj = bp.get("tsj", self.config.tsj)
            if str(getattr(self.config, "run_mode", "modern")) != "classic":
                self.config.pseudocount = 0.0  # BLOQUE 1: invariante
            # El piso llega como K objetivo (cdh_target_k) o, en archivos
            # legacy, como fracción directa (min_cluster_size_cdh).
            self.config.min_cluster_size_cdh = _cdh_frac_from_params(
                bp, default=self.config.min_cluster_size_cdh
            )
            self.config.swap_iterations = bp.get(
                "swap_iterations", self.config.swap_iterations
            )
            self.config.min_r2_threshold = bp.get(
                "min_r2_threshold", self.config.min_r2_threshold
            )
            # Hiperparámetros de UCBuilderConfig se aplican vía self.uc_config
            if "similarity_threshold" in bp:
                self.uc_config.similarity_threshold = bp["similarity_threshold"]
            if "coref_weight" in bp:
                self.uc_config.coref_weight = bp["coref_weight"]
            if "uc_window_size" in bp:
                self.uc_config.window_size = bp["uc_window_size"]
            # Parámetros nuevos del espacio ampliado (coref + HDBSCAN UCE)
            self.config.coref_context_units_tight = bp.get(
                "coref_context_units_tight", self.config.coref_context_units_tight
            )
            self.config.coref_context_units_loose = bp.get(
                "coref_context_units_loose", self.config.coref_context_units_loose
            )
            # [hdbscan-scale-patch]: best_params.json nuevos traen las fracciones
            # (hdbscan_uce_min_cluster_frac*); un archivo viejo, guardado
            # antes de este cambio, trae el absoluto — si aparece SOLO el
            # absoluto, se respeta tal cual (frac se deja en None) para no
            # reinterpretar silenciosamente un best_params.json ya validado.
            if "hdbscan_uce_min_cluster_frac" in bp:
                self.config.hdbscan_uce_min_cluster_frac = bp[
                    "hdbscan_uce_min_cluster_frac"
                ]
            elif "hdbscan_uce_min_cluster_size" in bp:
                self.config.hdbscan_uce_min_cluster_frac = None
                self.config.hdbscan_uce_min_cluster_size = bp[
                    "hdbscan_uce_min_cluster_size"
                ]
            if "hdbscan_uce_min_cluster_frac_loose" in bp:
                self.config.hdbscan_uce_min_cluster_frac_loose = bp[
                    "hdbscan_uce_min_cluster_frac_loose"
                ]
            elif "hdbscan_uce_min_cluster_size_loose" in bp:
                self.config.hdbscan_uce_min_cluster_frac_loose = None
                self.config.hdbscan_uce_min_cluster_size_loose = bp[
                    "hdbscan_uce_min_cluster_size_loose"
                ]

        cached_uces_por_doc = None
        cached_doc_metadata_map = None

        best_params_path = BEST_PARAMS_PATH
        best_params: Optional[Dict[str, Any]] = None

        if os.path.exists(best_params_path) and not _OPTIMIZE_ONLY:
            print(
                "   Found existing best_params.json. Loading parameters and skipping optimizer."
            )
            try:
                with open(best_params_path, "r", encoding="utf-8") as f:
                    best_data = json.load(f)
                if str(getattr(self.config, "run_mode", "modern")) == "classic":
                    raise RuntimeError(
                        "classic: no se lee best_params.json del modo moderno "
                        "(aislamiento de artefactos, CLASSIC_MODE.md §4)"
                    )
                _stored_fp = best_data.get("corpus_fingerprint")
                _current_fp = _corpus_fingerprint()
                if _stored_fp and _stored_fp != _current_fp:
                    if self.config.optimize:
                        print(
                            "   [WARNING] best_params.json es de otro dataset. Re-optimizando..."
                        )
                        best_params = None
                    else:
                        print(
                            "   [WARNING] best_params.json es de otro dataset y optimize=False. "
                            "Se usará la config base."
                        )
                        best_params = None

                else:
                    # Nuevo formato → 'best_params'; legacy → 'params'
                    best_params = best_data.get("best_params") or best_data.get(
                        "params"
                    )
                    if best_params:
                        print(f"   Loaded params: {best_params}")
                        _apply_best_params(best_params)
                    if _stored_fp is None:
                        # Archivo legacy (sin fingerprint): asumimos que es del
                        # dataset actual (el usuario lo validó) y lo sellamos
                        # para que futuros cambios de dataset se detecten.
                        try:
                            best_data["corpus_fingerprint"] = _current_fp
                            with open(best_params_path, "w", encoding="utf-8") as f:
                                json.dump(best_data, f, indent=2, ensure_ascii=False)
                            print(
                                "   [Optimizador] best_params.json sellado con "
                                "fingerprint del dataset actual."
                            )
                        except Exception as _se:
                            print(
                                f"   [WARNING] No se pudo sellar best_params.json: {_se}"
                            )
            except RuntimeError:
                # Guardas de contrato (p.ej. aislamiento del modo clasico,
                # #44): NO deben degradar a 'optimizador si habilitado'.
                raise
            except Exception as e:
                print(
                    f"   [WARNING] Could not load best_params.json: {e}. "
                    "Running optimizer if enabled."
                )
                best_params = None
        else:
            print(
                "   [WARNING] Could not load best_params.json. "
                "Running optimizer if enabled."
            )

        if best_params is None and self.config.optimize:
            # ── Mini pass-1 (retrofitting) ANTES del optimizador ─────────────
            # Solo necesita el resultado word-count (resultados[0]) para
            # computar los vectores retrofitted. Forzamos wc_only para NO
            # correr coref/embeddings aquí (la extracción coref es muy lenta
            # y no aporta nada al retrofitting). El optimizador ya precomputa
            # las cadenas coref por separado en __init__.
            retro_vecs_for_optimizer: Optional[Dict[str, np.ndarray]] = None
            _saved_mode = self.config.classification_mode
            try:
                self.config.classification_mode = "wc_only"
                (
                    _,
                    _,
                    mini_voc,
                    _,
                    mini_resultados,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self.double_clf.run(
                    corpus_raw,
                    ppmi_builder=self.ppmi_builder,
                )
                print("== Mini-pass! No tomar estos resultados en serio == ")
                retro_vecs_for_optimizer = self._compute_retrofitted_vectors(
                    mini_resultados, mini_voc
                )
            except Exception as e:
                logger.warning(
                    "   [Optimizador] Mini pass-1 (retrofitting) falló; "
                    "el optimizador degradará a 'wc_coref': %s",
                    e,
                    exc_info=True,
                )
                retro_vecs_for_optimizer = None
            finally:
                self.config.classification_mode = _saved_mode

            # Desactivar embeddings durante la optimización (coste)
            self.config.use_embeddings = False

            optimizador = Optimizador(
                self.config,
                self.uc_config,
                corpus_raw,
                self.we_analyzer,
                self.subtlex_analyzer,
                progressive_segmenter=self.double_clf.progressive_segmenter,
                # retrofitted_vectors precomputados en el mini pass-1 de arriba:
                # permite evaluar 'all' completo (Method C / embeddings) durante
                # la optimización. Si el mini pass-1 falló, es None y 'all'
                # degrada a 'wc_coref' como antes.
                retrofitted_vectors=retro_vecs_for_optimizer,
            )
            mejores_params = optimizador.optimizar(
                n_trials=self.config.optimize_trials,
                two_stage=getattr(self.config, "optimize_two_stage", True),
                stage1_fraction=getattr(self.config, "optimize_stage1_fraction", 0.3),
            )
            if mejores_params:
                # El frente de Pareto ya quedó persistido dentro de optimizar()
                _apply_best_params(mejores_params)
            else:
                print("   [!] Optimización no viable. Usando config base.")
            cached_uces_por_doc = optimizador.uces_por_doc
            cached_doc_metadata_map = optimizador.doc_metadata_map

            if _OPTIMIZE_ONLY:
                print(
                    "   [CLI] --optimize-only: fin tras el optimizador "
                    f"({os.path.basename(best_params_path)})."
                )
                sys.exit(0)

        # Restaurar embeddings para la corrida principal
        self.config.use_embeddings = True

        # ── Precomputar / cargar cache de cadenas coref ────────────────────
        # Sin esto, Pass 1 y Pass 2 re-extraen TODAS las cadenas vía HTTP,
        # que es lo que se come 3–4 h del pipeline. El Optimizador ya guardó
        # su cache en disco; aquí lo cargamos y solo computamos lo que falte.
        precomputed_coref_chains: Dict[int, Dict[str, List]] = {}
        if self.double_clf.progressive_segmenter is not None:
            precomputed_coref_chains = load_coref_chains_from_disk(self.config)
            if precomputed_coref_chains:
                print(
                    f"   [Coref] Cache en disco cargado: "
                    f"ctx={sorted(precomputed_coref_chains.keys())}, "
                    f"{sum(len(v) for v in precomputed_coref_chains.values())} cadenas."
                )

            ctx_needed = {
                int(getattr(self.config, "coref_context_units_tight", 2)),
                int(getattr(self.config, "coref_context_units_loose", 4)),
            }
            missing = ctx_needed - set(precomputed_coref_chains.keys())
            if missing:
                print(
                    f"   [Coref] Cache incompleto. Computando ctx={sorted(missing)}..."
                )
                # Necesitamos uces_por_doc: del optimizador si lo tenemos,
                # si no segmentamos aquí una vez (Pass 1 lo reutilizará).
                if cached_uces_por_doc is not None:
                    uces_for_coref = cached_uces_por_doc
                else:
                    print(
                        "   [Coref] Segmentando para precomputar cadenas "
                        "(Pass 1 reutilizará este resultado)..."
                    )
                    uces_for_coref, cached_doc_metadata_map = (
                        self.segmentador.segmentar_en_uces(corpus_raw)
                    )
                    for doc_uces in uces_for_coref:
                        self.segmentador.lematizar_uces(doc_uces)
                    for doc_uces in uces_for_coref:
                        for uce in doc_uces:
                            uce.id = normalizar_id_uce(uce)
                    cached_uces_por_doc = uces_for_coref

                new_chains = compute_coref_chains_for_ctx(
                    self.double_clf.progressive_segmenter,
                    uces_for_coref,
                    missing,
                )
                precomputed_coref_chains.update(new_chains)
                save_coref_chains_to_disk(self.config, precomputed_coref_chains)

        # ── Main run — skip re-lemmatization if we have the cache ──────────
        if cached_uces_por_doc is not None:
            print("   Reutilizando cache de lematización del optimizador...")
            original_seg = self.double_clf.segmentador.segmentar_en_uces
            original_lem = self.double_clf.segmentador.lematizar_uces
            self.double_clf.segmentador.segmentar_en_uces = lambda x: (
                copy.deepcopy(cached_uces_por_doc),
                cached_doc_metadata_map,
            )
            self.double_clf.segmentador.lematizar_uces = lambda x: x

        print("=== Pass 1: CHD with generic vectors ===")

        # En modo "select" el pass 1 no puede ver Method C (aun no hay
        # vectores retrofitted): su seleccion es provisional. No se imprime
        # como veredicto ni se persiste; el pass 2 es el definitivo.
        self.double_clf.quiet_selection = True
        try:
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
                checkpoint_path=checkpoint_path,
                force=force,
                precomputed_coref_chains=precomputed_coref_chains,  # ← NUEVO
            )
        finally:
            self.double_clf.quiet_selection = False

        # ── Retrofitting on pass-1 output ─────────────────────────────────────────
        if self.config.use_embeddings:
            retro_vecs_p1: Optional[Dict[str, np.ndarray]] = (
                self._compute_retrofitted_vectors(resultados_p1, voc_p1)
            )
        else:
            # Escenario A: sin retrofitting (no hay espacio denso que poblar).
            retro_vecs_p1 = {}
            print("   [i] use_embeddings=False: sin retrofitting (modo clasico).")

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
                checkpoint_path=checkpoint_path,
                force=force,
                precomputed_coref_chains=precomputed_coref_chains,  # ← NUEVO
            )

            # ── Persist pairwise stability matrices ───────────────────────────────
            if hasattr(self.double_clf, "last_pairwise_stability"):
                self.db.data["pairwise_stability"] = (
                    self.double_clf.last_pairwise_stability
                )
            # ── Persist the selected method (mode "select") ──────────────────────
            _sel = getattr(self.double_clf, "last_selection", None)
            if _sel is not None:
                self.db.data["method_selection"] = _sel.to_dict()
                # Decision persistida (snapshot + historial append-only).
                # La tabla ya se imprimio en DoubleClassifier.run().
                self.double_clf.persist_method_selection(
                    _sel, len({uce.id for doc in uces_por_doc for uce in doc})
                )

            # ── Estabilidad ALCESTE clásica (CR + ARI 10↔14) — §5 ────────
            if str(getattr(self.config, "run_mode", "modern")) == "classic":
                _pairwise = (
                    getattr(self.double_clf, "last_pairwise_stability", {}) or {}
                )
                _cr = self._classic_stability(uces_est_list, _pairwise)
                self.db.data["classic_stability"] = _cr
                print(_classic_stability_text(_cr))

        else:
            print("   Single-pass only (no retrofitting available).")
            uces_est_list = uces_est_p1
            voc_uc = voc_p1
            uces_por_doc = uces_por_doc_p1
            resultados_por_umbral = resultados_p1

        # ── Persist UCs ───────────────────────────────────────────────────────
        # Las UCs se construyen dentro de double_clf.run() pero nunca se
        # serializaban → la tabla `ucs` quedaba vacía (ucs=0). Guardamos las
        # UCs del resultado primario (word-count tight en wc_only).
        if resultados_por_umbral and resultados_por_umbral[0].get("ucs"):
            self.db.save_ucs(resultados_por_umbral[0]["ucs"])

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

        # ── FASE ABDUCTIVA (opt-in): proyección liminal de huérfanas ──────
        # Si use_liminal_projection=True, las UCEs que fallaron la triple
        # intersección se PROYECTAN sobre los centroides consolidados.
        # NO se modifica uce.cluster_id, ni df_terms, ni AFC, ni inercia.
        # Los resultados van a campos separados y se registran bajo
        # `db.data["liminal_projection"]` para trazabilidad.
        if self.config.use_liminal_projection and uces_est_list and voc_uc:
            try:
                projection_summary = self.double_clf._project_liminal_uces(
                    uces_por_doc=uces_por_doc,
                    primary_stable=uces_est_list,
                    voc=voc_uc,
                )
                self.db.data["liminal_projection"] = {
                    **projection_summary,
                    "enabled": True,
                    "min_ratio": float(
                        getattr(
                            self.config,
                            "liminal_projection_min_ratio",
                            1.10,
                        )
                    ),
                    "note": (
                        "Operación abductiva post-inductiva. Las UCEs "
                        "proyectadas NO forman parte del corpus analítico: "
                        "df_terms, AFC, inercia y estabilidad de términos "
                        "se calcularon exclusivamente sobre uces_est_list."
                    ),
                }
            except Exception as e:
                print(f"   [WARNING] Proyección liminal falló: {e}")
                self.db.data["liminal_projection"] = {
                    "enabled": True,
                    "error": str(e),
                }
        else:
            self.db.data["liminal_projection"] = {
                "enabled": False,
                "note": "use_liminal_projection=False (default)",
            }

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
                # El AFC devuelve su propio vocabulario podado. Alinear mat_uces
                # a las columnas conservadas para la proyección suplementaria.
                kept_cols = afc_result.pop("kept_cols_idx", None)
                afc_result.pop("kept_rows_idx", None)
                mat_uces_afc = (
                    mat_uces[:, kept_cols] if kept_cols is not None else mat_uces
                )

                resultados["proyeccion"] = afc_result

                # ── Task 2: Map AFC coordinates directly onto each UCE ────────────
                col_coords = np.array(afc_result["col_coords"])  # principal coords
                col_sums_vec = mat_clases.sum(axis=0).astype(float)
                col_sums_vec = np.where(col_sums_vec == 0, 1.0, col_sums_vec)
                col_mass = col_sums_vec / col_sums_vec.sum()  # column marginals

                row_sums = mat_uces_afc.sum(axis=1, keepdims=True).astype(float)
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                row_profile = mat_uces_afc.astype(float) / row_sums  # (n_uce, n_terms)

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
        try:  # [quality-patch] hook
            from reii.quality.hooks import save_quality_run

            save_quality_run(self, uces_est_list, uces_por_doc)
        except Exception as _qe:  # nunca rompe el pipeline
            logger.warning("quality hook falló: %s", _qe)

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

    def _classic_stability(self, uces_por_doc, pairwise) -> Dict[str, Any]:
        """CR y ARI de las dos corridas (UC 10 vs UC 14). Sin select-v4."""
        total = sum(len(d) for d in uces_por_doc)
        labels = [
            u.cluster_id
            for d in uces_por_doc
            for u in d
            if getattr(u, "cluster_id", None) is not None
        ]
        k = len(set(labels))
        sizes = Counter(labels)
        return {
            "scenario": "A",
            "run_mode": "classic",
            "spec_id": run_spec_id(self.config),
            "n_uces": int(total),
            "n_stable": int(len(labels)),
            "cr": float(len(labels) / max(1, total)),
            "ari_a1_a2": float(pairwise.get("ari_wc", float("nan"))),
            "k_partition": int(k),
            "min_class_size": int(min(sizes.values())) if sizes else 0,
            "max_class_share": float(max(sizes.values()) / max(1, len(labels)))
            if sizes
            else 0.0,
            "max_class_share": float(max(sizes.values()) / max(1, len(labels)))
            if sizes
            else 0.0,
            "pseudocount": float(getattr(self.config, "pseudocount", 0.0)),
            "epsilon_status": "not_implemented",
        }


def _classic_stability_text(rep: Dict[str, Any]) -> str:
    return (
        "\n  ALCESTE 2012 - ESTABILIDAD (UC 10 vs 14)\n"
        f"    UCEs totales       : {rep['n_uces']}\n"
        f"    Estables (misma clase en A1 y A2) : {rep['n_stable']}"
        f"  CR = {rep['cr']:.3f}\n"
        f"    ARI(A1,A2)         : {rep['ari_a1_a2']:.3f}\n"
        f"    K (particion)      : {rep['k_partition']}"
        f"  clase min={rep['min_class_size']}"
        f"  mayor={rep['max_class_share']:.2f}\n"
        "    [!] pseudocount=0.0 (paper: 0.1; epsilon no implementado en la CDH)\n"
    )


if __name__ == "__main__":
    # ── Configs  ────────────────────────────────────────────────
    alceste_config = Config(
        use_bigrams=True,
        stem_backend="snowball",
        min_uce_words=3,
        min_forms_uc=[10, 14],
        tsj=3,
        use_cdh=True,
        uce_target_size=26,
        pseudocount=0.0,
        use_cah_per_class=True,
        cah_per_class_top_terms=20,
        use_projection=True,
        analyze_metadata=True,
        use_network_analysis=True,
        use_term_stability=True,
        random_state=42,
        use_rf_shap=True,
        rf_n_estimators=100,
        rf_max_depth=5,
        rf_outlier_method="iqr",
        rf_cat_encoding="onehot",
        glm_method="chi2",
        use_multivariate_analysis=True,
        multivariate_metadata=["Edad_Cat", "Sexo", "Ocupacion_Cat", "Procedencia_Cat"],
        ppmi_k=1.0,
        analytic_temperature=0.1,
        use_llm_synthesis=False,
        # Modo por defecto: un solo método, elegido por regla pre-registrada y
        # persistido en data/method_selection.json. Los modos de intersección
        # ("all", "two_of_three", ...) siguen disponibles pero ya no son el
        # default: no son identificables y no son ALCESTE 2012.
        classification_mode="select",
        coref_context_units_tight=2,
        coref_context_units_loose=4,
        hdbscan_uce_min_cluster_size=5,
        hdbscan_uce_min_cluster_size_loose=3,
        hdbscan_uce_metric="euclidean",
        # ── Optimizer (Optuna) ─────────────────────────────────────────
        optimize=True,
        optimize_trials=300,  # subir de 100 a 300 — TPE rinde más
        optimize_sampler="tpe",
        optimize_pruner="median",
        optimize_storage="sqlite:///reii_optuna.db",  # persistencia entre sesiones
        optimize_study_name="reii_search_v2",
        optimize_n_startup_trials=12,
        optimize_multivariate=True,
        optimize_prune_n_startup=15,
        optimize_directions=("maximize", "maximize", "minimize"),
        optimize_preference="knee",  # o "max_ari" / "max_coverage"
        optimize_coverage_gate=0.25,
    )

    # ── Overrides del dashboard (PostgreSQL vía ConfigStore) ─────────────
    try:
        from reii.backend.config_store import ConfigStore

        _store = ConfigStore(alceste_config.db_dsn)
        _store.seed_from_config(
            alceste_config
        )  # publica defaults (no-op si ya sembrado)
        _stored = _store.get_all()
        _valid = {
            k: v for k, v in _stored.items() if k in alceste_config.__dataclass_fields__
        }
        if _valid:
            alceste_config = dataclasses.replace(alceste_config, **_valid)
    except Exception as e:
        print(f"   [WARNING] ConfigStore apply falló: {e}")

    # ── Override CLI: --mode gana sobre ConfigStore ─────────────────────
    _cli_mode = _parse_cli_mode()
    if _cli_mode is not None and _cli_mode != "classic":
        alceste_config = dataclasses.replace(
            alceste_config, classification_mode=_cli_mode
        )
        print(f"   [CLI] classification_mode={_cli_mode} (--mode override)")

    _mode_final = str(getattr(alceste_config, "classification_mode", "") or "")
    _mode_final = {"auto": "select"}.get(_mode_final, _mode_final)
    print(f"   [i] modo de clasificacion efectivo: {_mode_final}")
    if _mode_final in (
        "all",
        "two_of_three",
        "wc_coref",
        "wc_emb",
        "coref_emb",
        "both",
        "wc_sim",
        "sim_emb",
    ):
        print(
            f"   [WARNING] classification_mode={_mode_final!r} es una INTERSECCION "
            "de metodos (filtro propio, no ALCESTE). El modo por defecto es "
            "'select' (un metodo: A, B o C, persistido)."
        )

    # ── Override CLI: --uce-target-size gana sobre ConfigStore y desactiva
    #    la auto-calibración (el valor explícito manda) ──────────────────
    _cli_uce = _parse_cli_uce_target_size()
    if _cli_uce is not None:
        alceste_config = dataclasses.replace(
            alceste_config, uce_target_size=_cli_uce, auto_calibrate_uce_size=False
        )
        print(
            f"   [CLI] uce_target_size={_cli_uce} "
            "(--uce-target-size override; auto-calibración desactivada)"
        )

    # ── Overrides CLI del optimizador (DESPUÉS de ConfigStore) ──────────
    alceste_config = _cli_overrides(alceste_config)
    _warn_unknown_cli_flags()

    gram_config = GramConfig(
        min_tokens_por_uce=30,
        max_tokens_por_uce=200,
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
    if _cli_mode == "classic":
        alceste_config, cfg = apply_classic_mode(alceste_config, cfg)

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
        "select",
        "both",
        "sim_only",
        "wc_sim",
        "sim_emb",
        "auto",
    }  # aliases too
    # El optimizador de dos etapas corre la etapa 2 en mode="select", que
    # incluye Method B. Con el modo del run en wc_only (p.ej. --optimize-only
    # sin --mode) el segmenter NO se construía y Method B quedaba
    # ESTRUCTURALMENTE fuera de la optimización: select-v2 solo podía elegir
    # entre A y C, así que la decisión persistida no era «A|B|C».
    _optimizer_will_run = bool(_OPTIMIZE_ONLY) or (
        bool(getattr(alceste_config, "optimize", True))
        and not os.path.exists(BEST_PARAMS_PATH)
    )
    _needs_progressive = (
        alceste_config.classification_mode in _coref_modes or _optimizer_will_run
    )
    progressive_seg = None
    if _needs_progressive:
        progressive_seg = ProgressiveSegmenter(
            model_name=alceste_config.embedding_model_name,
            stanza_lang="es",
            spacy_model=alceste_config.spacy_model,
            debug_coref=True,  # silence chain-level prints in production
            similarity_threshold=0.6,
            max_depth=3,
        )
        # Eagerly verify Stanza loads; fail fast rather than mid-pipeline.
        # Con servidor remoto (Colab) el pipeline local no hace falta: el coref
        # completo se resuelve por HTTP (/v1/extract_coref).
        if not STANZA_SERVER_URL and progressive_seg.get_stanza() is None:
            print("[WARN] Stanza failed to load — Method B will be skipped.")
            progressive_seg = None
        elif _optimizer_will_run and (
            alceste_config.classification_mode not in _coref_modes
        ):
            print("   [i] Method B habilitado para el optimizador (etapa 2 = select).")

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
    # ── Checkpoint de coref: permite reanudar tras OOM o corte ─────────────
    # _clasificar_umbral_coref deriva un archivo por ctx (B1, B2) y guarda
    # tras cada documento. Si la corrida muere, la siguiente reanuda.
    _coref_ckpt_path = os.path.join(
        os.path.dirname(COREF_CHAINS_CACHE_PATH) or ".",
        "coref_seg_checkpoint.json",
    )
    _force_from_cli = "--force" in sys.argv or "--reprocess" in sys.argv

    orchestrator.ejecutar(
        corpus_raw=uwu,
        grammatical_pipeline=gram_pipeline,
        global_corpus=global_corpus,
        lex_analyzer=lex_analyzer,
        grammatical_dashboard_path=(
            os.path.join(CLASSIC_OUTPUT_DIR, "global_dashboard.json")
            if alceste_config.run_mode == "classic"
            else OUTPUT_DASHBOARD
        ),
        checkpoint_path=_coref_ckpt_path,
        force=_force_from_cli,
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
    lex_analyzer.export_network_gexf(
        os.path.join(CLASSIC_OUTPUT_DIR, "red_semantica.gexf")
        if alceste_config.run_mode == "classic"
        else OUTPUT_NETWORK_GEXF
    )

    # ── PHASE 4: Guardar todo ─────────────────────────────────────────
    # Fixed: Removed the overlapping positional argument
    global_corpus.export_json(
        OUTPUT_DASHBOARD,
        predicate_result=global_predicate_result,
        subtlex_analyzer=subtlex,
        lex_analyzer=lex_analyzer,  # ← now wired through
    )
