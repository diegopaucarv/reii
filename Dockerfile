# ============================================================
# REII — Docker build (CPU)
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    REII_ROOT_DIR=/app

WORKDIR /app

# System deps for compiling C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (cached layer) ────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# ── spaCy model ───────────────────────────────────────────────────────
RUN python -m spacy download es_core_news_lg

# ── NLTK data ─────────────────────────────────────────────────────────
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# ── Application source ────────────────────────────────────────────────
COPY pyproject.toml .
COPY src/ src/
RUN pip install -e .

# ── Runtime directories ───────────────────────────────────────────────
RUN mkdir -p /app/data /app/output /app/models /app/ia

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import reii" || exit 1

CMD ["streamlit", "run", "src/reii/dashboard.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
