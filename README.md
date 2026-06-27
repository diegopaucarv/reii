# REII — ALCESTE Spanish Discourse Analysis Pipeline

A computational linguistics pipeline for Spanish discourse analysis, implementing ALCESTE-style segmentation, classification, and semantic network construction.

## Architecture

```
src/reii/
├── gram/          # Grammar analysis, UCE extraction, NLP pipeline
├── lang/          # Spanish language rules & lexicons
├── seg/           # Progressive text segmentation
├── dashboard.py   # Streamlit interactive dashboard
├── main_workflow.py          # Pipeline with Transformers (heavy)
├── main_workflow_clasico.py  # Pipeline without Transformers (light)
└── ia_discursiva.py          # AI discourse agent (DeepSeek API)
```

## Quick Start (Docker)

```bash
# Start the dashboard
docker compose up dashboard

# Run the batch pipeline
docker compose run workflow
```

## Quick Start (Local)

```bash
# Install
pip install -e .
python -m spacy download es_core_news_lg

# Run dashboard
streamlit run src/reii/dashboard.py

# Run workflow
export REII_DATA_DIR=./data/input
python src/reii/main_workflow_clasico.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REII_DATA_DIR` | `./data/input` | Path to input documents |
| `REII_ROOT_DIR` | `.` | Project root for output resolution |
| `DEEPSEEK_API_KEY` | — | API key for the discourse agent |
