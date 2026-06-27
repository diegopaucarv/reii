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
ROOT_DIR = Path(os.environ.get("REII_ROOT_DIR", ".")).resolve()
DATA_DIR = ROOT_DIR / os.environ.get("REII_DATA_DIR", "data/input")
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
LLM_MODEL: str = os.environ.get("REII_LLM_MODEL", "deepseek-chat")

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
BEST_PARAMS_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_BEST_PARAMS_PATH", "data/best_params.json")
)
DISCOURSE_STATE_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_DISCOURSE_STATE_PATH", "data/discourse_state.json")
)

# ---------------------------------------------------------------------------
# IA agent config paths
# ---------------------------------------------------------------------------
IA_DIR = ROOT_DIR / "ia"
GRAMMAR_CONFIG_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_GRAMMAR_CONFIG_PATH", "ia/0.json")
)
DISCOURSE_CONFIG_PATH: str = str(
    ROOT_DIR / os.environ.get("REII_DISCOURSE_CONFIG_PATH", "ia/1.json")
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

# ---------------------------------------------------------------------------
# API keys / tokens
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL: str = os.environ.get(
    "DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"
)
# Base URL for OpenAI-compatible client (strips the /v1/chat/completions path)
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
