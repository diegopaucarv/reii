"""
Centralized configuration for REII.
All hardcoded values live here, sourced from environment variables.
Loads a ``.env`` file from the project root automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env (optional dependency; skips silently if not installed)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    _DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
    if _DOTENV_PATH.exists():
        load_dotenv(_DOTENV_PATH)
except ImportError:
    pass  # dotenv not installed — using os.environ only

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(
    os.environ.get("REII_ROOT_DIR", Path(__file__).resolve().parents[2])
).resolve()
DATA_DIR = ROOT_DIR / os.environ.get("REII_DATA_DIR", "data")
OUTPUT_DIR = ROOT_DIR / os.environ.get("REII_OUTPUT_DIR", "output")
MODELS_DIR = ROOT_DIR / os.environ.get("REII_MODELS_DIR", "models")

# ---------------------------------------------------------------------------
# spaCy / NLP models
# ---------------------------------------------------------------------------
SPACY_MODEL: str = os.environ.get("REII_SPACY_MODEL", "es_core_news_lg")
SENTENCE_EMBEDDER_MODEL: str = os.environ.get(
    "REII_SENTENCE_EMBEDDER_MODEL", "BAAI/bge-m3"
)
GLINER_MODEL: str = os.environ.get("REII_GLINER_MODEL", "urchade/gliner_multi-v2.1")
GPT2_MODEL: str = os.environ.get("REII_GPT2_MODEL", "datificate/gpt2-small-spanish")

# ---------------------------------------------------------------------------
# Embedding / segmenter models (separate from the above)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = os.environ.get(
    "REII_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)
SEGMENTER_EMBEDDING_MODEL: str = os.environ.get(
    "REII_SEGMENTER_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
NLI_MODEL: str = os.environ.get("REII_NLI_MODEL", "facebook/bart-large-mnli")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_MODEL: str = os.environ.get("REII_LLM_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
LLM_FALLBACK_MODEL: str = os.environ.get(
    "REII_LLM_FALLBACK_MODEL", "Prism-ML/Ternary-Bonsai-27B"
)
LLM_MAX_RETRIES: int = int(os.environ.get("REII_LLM_MAX_RETRIES", "5"))
# Timeout de "silencio" en segundos: con streaming, es el tiempo máximo sin
# recibir datos del modelo (no el tiempo total de generación).
LLM_TIMEOUT: int = int(os.environ.get("REII_LLM_TIMEOUT", "300"))

# ---------------------------------------------------------------------------
# Stanza
# ---------------------------------------------------------------------------
STANZA_LANG: str = os.environ.get("REII_STANZA_LANG", "es")
STANZA_USE_GPU: bool = os.environ.get("REII_STANZA_USE_GPU", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Adverb classifier
# ---------------------------------------------------------------------------
ADVERB_CLASSIFIER_DIR: str = str(
    ROOT_DIR / os.environ.get("REII_ADVERB_CLASSIFIER_DIR", "models/adverb_model")
)

# ---------------------------------------------------------------------------
# Database / persistence
# ---------------------------------------------------------------------------
DB_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_DB_PATH", "data/analisis_gramatical.json")
)
WORKFLOW_DB_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_WORKFLOW_DB_PATH", "data/workflow_data.json")
)
DATABASE_URL: str = os.environ.get(
    "REII_DATABASE_URL", "postgresql://reii:reii@reii-db:5432/reii"
)
WORKFLOW_CONFIG_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_WORKFLOW_CONFIG_PATH", "data/workflow_config.json")
)
BATCH_DB_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_BATCH_DB_PATH", "data/entrevistas.json")
)
TRANSCRIPTS_DIR: str = str(
    ROOT_DIR / os.environ.get("REII_TRANSCRIPTS_DIR", "data/entrevistas")
)
BEST_PARAMS_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_BEST_PARAMS_PATH", "data/best_params.json")
)
# Cache de la auto-calibración bayesiana de uce_target_size (ver
# calibrar_uce_target_size en main_workflow.py). Mismo patrón de invalidación
# por huella del corpus que BEST_PARAMS_PATH.
UCE_CALIBRATION_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_UCE_CALIBRATION_PATH", "data/uce_calibration.json")
)
# Cache en disco de las cadenas coref precomputadas por context_units
# (ver Optimizador._ensure_coref_chains en main_workflow.py). Mismo patrón
# de invalidación por huella del corpus + segmentación que UCE_CALIBRATION_PATH.
COREF_CHAINS_CACHE_PATH: str = str(
    ROOT_DIR
    / os.environ.get("REII_COREF_CHAINS_CACHE_PATH", "data/coref_chains_cache.json")
)
DISCOURSE_STATE_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_DISCOURSE_STATE_PATH", "data/discourse_state.json")
)

# ---------------------------------------------------------------------------
# IA agent config paths
# ---------------------------------------------------------------------------
IA_DIR = ROOT_DIR / "ia"
DISCOURSE_DIR = IA_DIR / "analisis_discursivo"
GRAMMAR_CONFIG_PATH: str = str(
    ROOT_DIR
    / os.environ.get(
        "REII_GRAMMAR_CONFIG_PATH", "ia/analisis_discursivo/resumidor_gramatical.json"
    )
)
GRAMATICA_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_GRAMATICA_PATH", "ia/gramatica.json")
)
DISCOURSE_CONFIG_PATH: str = str(
    ROOT_DIR
    / os.environ.get(
        "REII_DISCOURSE_CONFIG_PATH", "ia/analisis_discursivo/discurso_lexico.json"
    )
)
LOCALISMOS_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_LOCALISMOS_PATH", "ia/localismos.json")
)
ENTREVISTAS_DIR = IA_DIR / "entrevistas"
GUIA_ENTREVISTA_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_GUIA_ENTREVISTA_PATH", "ia/entrevistas/guia.json")
)
SEGMENTACION_CONFIG_PATH: str = str(
    ROOT_DIR
    / os.environ.get(
        "REII_SEGMENTACION_CONFIG_PATH", "ia/entrevistas/segmentacion.json"
    )
)
EXTRACCION_CONFIG_PATH: str = str(
    ROOT_DIR
    / os.environ.get("REII_EXTRACCION_CONFIG_PATH", "ia/entrevistas/extraccion.json")
)

# ---------------------------------------------------------------------------
# SUPTLEX-ESP
# ---------------------------------------------------------------------------
_SUBTLEX_ENV = os.environ.get("REII_SUBTLEX_PATH", "")
SUBTLEX_PATH: str | None = str(ROOT_DIR / _SUBTLEX_ENV) if _SUBTLEX_ENV else None

# ---------------------------------------------------------------------------
# Word embeddings (FastText .bin)
# ---------------------------------------------------------------------------
_WE_ENV = os.environ.get("REII_WORD_EMBEDDINGS_PATH", "")
WORD_EMBEDDINGS_PATH: str | None = str(ROOT_DIR / _WE_ENV) if _WE_ENV else None

# ---------------------------------------------------------------------------
# Pipeline toggles
# ---------------------------------------------------------------------------
USE_COREF: bool = os.environ.get("REII_USE_COREF", "true").lower() == "true"
USE_SUBTLEX: bool = os.environ.get("REII_USE_SUBTLEX", "true").lower() == "true"
USE_WORDNET_QUANTIFIERS: bool = (
    os.environ.get("REII_USE_WORDNET_QUANTIFIERS", "true").lower() == "true"
)
USE_GENSIM_EMBEDDINGS: bool = (
    os.environ.get("REII_USE_GENSIM_EMBEDDINGS", "true").lower() == "true"
)
# COREF: sujetos raíz por UCE. Sin coref (basta depparse+constituency) y por lotes.
# Para volver al comportamiento anterior: REII_ROOTS_SKIP_COREF=false REII_ROOTS_BATCH_SIZE=0
ROOTS_SKIP_COREF: bool = (
    os.environ.get("REII_ROOTS_SKIP_COREF", "true").lower() == "true"
)
ROOTS_BATCH_SIZE: int = int(
    os.environ.get("REII_ROOTS_BATCH_SIZE", "32")
)  # <=1: sin lotes
# COREF: ventanas de coref por lote en ``_extract_global_chains`` (bulk_process).
#
# ⚠ NO usar lotes (>1). Stanza's ``CorefProcessor`` hereda ``UDProcessor.bulk_process``,
# que COMBINA todas las oraciones de todos los documentos en un único ``combined_doc``,
# resuelve coref a través de TODAS las sub-ventanas (rompe la ventana limitada de
# ``context_units`` UCEs) y devuelve los documentos originales con ``coref`` VACÍO
# (las cadenas quedan en ``combined_doc``, que se descarta). Resultado: cadenas
# perdidas + coref cruzando límites de ventana. Por eso el default es 1.
#
# <=1: sin lotes (una ventana por llamada, comportamiento correcto).
COREF_BATCH_SIZE: int = int(os.environ.get("REII_COREF_BATCH_SIZE", "1"))

# ---------------------------------------------------------------------------
# Stanza remoto (servidor FastAPI en Colab): si STANZA_SERVER_URL está definido,
# los sujetos raíz se piden por HTTP en vez de ejecutar Stanza localmente.
# ---------------------------------------------------------------------------
STANZA_SERVER_URL: str = os.environ.get("STANZA_SERVER_URL", "").rstrip("/")
STANZA_SERVER_TOKEN: str = os.environ.get("STANZA_SERVER_TOKEN", "")
STANZA_SERVER_TIMEOUT: float = float(os.environ.get("STANZA_SERVER_TIMEOUT", "90"))
STANZA_SERVER_FALLBACK_LOCAL: bool = (
    os.environ.get("STANZA_SERVER_FALLBACK_LOCAL", "true").lower() == "true"
)
# Cuántos textos van en CADA petición a /v1/extract_coref. El servidor sigue
# llamando nlp() UNA vez por texto (nunca los batchea entre sí — ver nota en
# stanza_server.py), así que esto solo ahorra overhead de red/túnel por
# texto, no tiempo de GPU. Default=1: comportamiento actual. Solo subir tras
# migrar de cloudflared quick-tunnel (~100s de timeout) a un túnel propio.
STANZA_SERVER_COREF_REMOTE_BATCH: int = int(
    os.environ.get("STANZA_SERVER_COREF_REMOTE_BATCH", "1")
)
# Cuántas peticiones a /v1/extract_coref puede tener el cliente en vuelo al
# mismo tiempo. Solo rinde si el servidor tiene REII_STANZA_N_REPLICAS>1 (si
# no, todas las peticiones concurrentes se van a encolar igual detrás de la
# única réplica del servidor — no hace daño, pero tampoco acelera nada).
# Default=1: comportamiento actual (totalmente secuencial).
STANZA_SERVER_CONCURRENCY: int = int(os.environ.get("STANZA_SERVER_CONCURRENCY", "1"))

# ---------------------------------------------------------------------------
# API keys / tokens
# ---------------------------------------------------------------------------
TOGETHER_API_KEY: str = os.environ.get("TOGETHER_API_KEY", "")
TOGETHER_API_URL: str = os.environ.get(
    "TOGETHER_API_URL", "https://api.together.ai/v1/chat/completions"
)
# Base URL for OpenAI-compatible client (strips the /v1/chat/completions path)
TOGETHER_BASE_URL: str = TOGETHER_API_URL.rsplit("/v1/", 1)[0]

# Backward-compatible aliases (deprecated — prefer TOGETHER_*)
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", TOGETHER_API_KEY)
DEEPSEEK_API_URL: str = os.environ.get("DEEPSEEK_API_URL", TOGETHER_API_URL)
DEEPSEEK_BASE_URL: str = DEEPSEEK_API_URL.rsplit("/v1/", 1)[0]

HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Corpus metadata
# ---------------------------------------------------------------------------
CORPUS_NAME: str = os.environ.get("REII_CORPUS_NAME", "Corpus ALCESTE")

# ---------------------------------------------------------------------------
# Output file names (used by workflows)
# ---------------------------------------------------------------------------
OUTPUT_DASHBOARD: str = str(OUTPUT_DIR / "global_dashboard.json")
OUTPUT_NETWORK_GEXF: str = str(OUTPUT_DIR / "red_semantica.gexf")
OUTPUT_LEXICAL_JSON: str = str(OUTPUT_DIR / "global_lexical.json")
OUTPUT_LEXICAL_XLSX: str = str(OUTPUT_DIR / "global_lexical.xlsx")
