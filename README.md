# REII – Spanish Discourse Analysis Pipeline

**ALCESTE‑style computational linguistics for Spanish qualitative data. Presented at IV International Congress of Human Sciences (Sep 2026) **

REII is a production‑grade NLP pipeline for Spanish discourse analysis, implementing a progressive segmentation algorithm inspired by the ALCESTE method. It combines traditional statistical approaches with transformer‑based models to classify text segments, extract semantic networks, and generate interpretable discourse structures. The pipeline can be theoretically adapted for different 

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![spaCy](https://img.shields.io/badge/spaCy-3.x-green)
![Stanza](https://img.shields.io/badge/Stanza-1.x-purple)

## 🧠 Core NLP Capabilities

### Advanced Text Segmentation with Sliding Window

REII's most distinctive feature is its **progressive text segmentation** with a **sliding window** for coreference resolution. Unlike traditional fixed‑window or sentence‑based segmentation, this approach:

- **Preserves discourse continuity** – Segments overlap, ensuring that cross‑sentential references are not lost.
- **Resolves coreferences** – Using **Stanza** (Stanford NLP), the system identifies anaphoric references across the sliding window, maintaining referential coherence even when entities span multiple sentences.
- **Adapts to text length** – The window size is configurable based on document length and genre, optimising for both short interview excerpts and long transcripts.
- **Handles Spanish linguistic complexity** – The pipeline is specifically tuned for Spanish, with custom lexicons and grammar rules for the language's rich morphology and flexible syntax.

#### How the Sliding Window Works

```text
Document: [s1] [s2] [s3] [s4] [s5] [s6] [s7] [s8] [s9]

Window 1: [s1] [s2] [s3] [s4] [s5] → extract features
Window 2: [s2] [s3] [s4] [s5] [s6] → extract features
Window 3: [s3] [s4] [s5] [s6] [s7] → extract features
```

Each window is processed for:
- **Coreference chains** – Identifying which entities (people, organisations, concepts) are being referred to across the segment.
- **Lexical cohesion** – Measuring vocabulary overlap and semantic relatedness between adjacent windows.
- **Discourse markers** – Detecting transitional phrases that signal shifts in topic or argument.

This sliding‑window approach ensures that no discourse‑level meaning is lost, while still producing granular, analysable units for classification and network construction.

### Classification Pipeline

REII offers two parallel classification workflows:

| Pipeline | Backend | Use Case |
|----------|---------|----------|
| **Classic** (`main_workflow_clasico.py`) | Statistical NLP (spaCy, custom lexicons) | Lightweight, fast, no GPU required |
| **Transformer** (`main_workflow.py`) | Transformer models (fine‑tuned for Spanish) | Higher accuracy, requires GPU |

Both pipelines produce:
- **UCE (Unités de Contexte Élémentaires)** – The atomic units of discourse, similar to elementary context units in ALCESTE.
- **Discourse classifications** – Each segment is assigned to a discourse type (e.g., narrative, argumentative, descriptive).
- **Semantic networks** – Co‑occurrence graphs of key terms, visualising conceptual structures.

### AI Discourse Agent

The `ia_discursiva.py` module provides an optional **DeepSeek‑powered** discourse agent that can:

- Generate natural‑language summaries of discourse patterns.
- Propose interpretive labels for emerging themes.
- Suggest relationships between discourse units.

This agent is invoked only when the researcher chooses, preserving the inductive integrity of the pipeline.

## 🏗️ Architecture

```mermaid
flowchart TD
    %% Style Definitions
    classDef dash fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000;
    classDef pipeline fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000;
    classDef lang fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000;
    classDef optional fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000;

    subgraph Layer1_Dashboard ["Streamlit Dashboard"]
        D1[Interactive visualisation of results]
    end

    subgraph Layer2_Pipeline ["Core Pipeline"]
        P1[Progressive Segmentation] --> P2[Grammar & UCE Analysis] --> P3[Semantic Network Construction]
    end

    subgraph Layer3_Resources ["Language Resources"]
        R1[Spanish Grammar Rules] --> R2[Custom Lexicons] --> R3[spaCy & Stanza Models]
    end

    subgraph Layer4_Optional ["Optional Modules"]
        O1[AI Discourse Agent]
        O2[Summary Generation]
        O3[Thematic Proposal]
        O4[Relationship Suggestion]
        O1 --- O2
        O1 --- O3
        O1 --- O4
    end

    %% Vertical Connections between Layers
    D1 --> P1
    P3 --> R1
    R3 --> O1

    %% Dashed optional connection from Core Pipeline to Optional Modules
    P3 -.-> O1

    %% Apply Styles
    class D1 dash

```

## 📦 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/diegopaucarv/reii.git
cd reii

# Start the dashboard
docker-compose up dashboard

# Run the batch pipeline (classic, lightweight)
docker-compose run workflow

Local Installation
bash

# Install the package
pip install -e .

# Download Spanish spaCy model
python -m spacy download es_core_news_lg

# Run the dashboard
streamlit run src/reii/dashboard.py

# Run the classic pipeline
export REII_DATA_DIR=./data/input
python src/reii/main_workflow_clasico.py

Environment Variables
Variable	Default	Description
REII_DATA_DIR	./data/input	Path to input documents
REII_ROOT_DIR	.	Project root for output resolution
DEEPSEEK_API_KEY	—	API key for the AI discourse agent (optional)
```

## 📁 Project Structure

```text
reii/
├── src/reii/
│   ├── gram/               # Grammar analysis, UCE extraction, NLP pipeline
│   ├── lang/               # Spanish language rules & custom lexicons
│   ├── seg/                # Progressive text segmentation (sliding window)
│   ├── dashboard.py        # Streamlit interactive dashboard
│   ├── main_workflow.py    # Transformer‑based pipeline (heavy)
│   ├── main_workflow_clasico.py  # Statistical pipeline (light)
│   └── ia_discursiva.py    # AI discourse agent (DeepSeek API)
├── data/                   # Input and output data
├── docker-compose.yml      # Container orchestration
└── setup.py                # Package installation
```

## 🔬 Methodology

REII is inspired by the ALCESTE (Analyse des Lexèmes Co‑occurrents dans un Ensemble de Segments de Texte) method, adapted for Spanish discourse analysis. Key principles:

    Progressive segmentation – Text is segmented iteratively, with each pass refining the boundaries based on lexical and syntactic cues.

    Contextual classification – Segments are classified not in isolation, but within their discourse context, using the sliding window to maintain coherence.

    Semantic network construction – Co‑occurrence graphs reveal the latent conceptual structure of the corpus, supporting both exploratory and confirmatory analysis.

The sliding‑window coreference resolution, powered by Stanza, is particularly valuable for Spanish, where anaphoric references are frequent and pronouns are often dropped (pro‑drop language) – making coreference chains harder to detect without broader context.
