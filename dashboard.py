import ast
import hashlib
import html as _html
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations, groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder

from reii.config import DISCOURSE_STATE_PATH, WORKFLOW_DB_PATH

st.set_page_config(
    page_title="ALCESTE · Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# PUENTE JAVASCRIPT -> PYTHON (Decláralo a nivel global)
# ─────────────────────────────────────────────────────────────
SUBCAT_FILTER_KEY = "subcat_filters"  # dict: {category_key: value_or_None}


def _set_multiple_filters(tab_name: str, filters_dict: dict):
    """
    Reemplaza los filtros actuales por un grupo nuevo de filtros.
    """
    clean_filters = {k: str(v) for k, v in filters_dict.items() if v}
    st.session_state[SUBCAT_FILTER_KEY] = clean_filters

    # ESTO ES CRÍTICO: Debe ser el nombre exacto de la 'key' del st.radio
    st.session_state.radio_tabs_a = tab_name

    st.rerun()


st.markdown(
    """
<style>
div[data-testid="stTextInput"]:has(input[aria-label="js_bridge"]) {
    position: absolute;
    width: 0px;
    height: 0px;
    opacity: 0;
    z-index: -9999;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# 2. El Input
js_bridge_val = st.text_input(
    "js_bridge", key="js_bridge", label_visibility="collapsed"
)

# 3. La Lógica
if js_bridge_val:
    try:
        payload = json.loads(js_bridge_val)
        current_ts = payload.get("ts")

        if st.session_state.get("last_processed_js_ts") != current_ts:
            st.session_state.last_processed_js_ts = current_ts
            _set_multiple_filters(payload["tab"], payload["filters"])

    except json.JSONDecodeError:
        pass


# ─────────────────────────────────────────────────────────────
# MODO DÍA / NOCHE
# ─────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

_dm = st.session_state.dark_mode

st.session_state.setdefault("coref_selected_entity", None)
st.session_state.setdefault("pred_selected_lemma", None)
st.session_state.setdefault("pred_filter_voice", "Todas")
st.session_state.setdefault("pred_filter_role", "Todos")
st.session_state.setdefault("pred_filter_negated", "Ambos")
if "coref_selected_entity" not in st.session_state:
    st.session_state.coref_selected_entity = None

if _dm:
    T = dict(
        bg_page="#0C0E14",
        bg_panel="#13161F",
        bg_card="#1C2030",
        bg_hover="#242840",
        surface="#161929",
        border="#1C2030",
        border2="#2C3050",
        text_hi="#E2E4F0",
        text_mid="#8B90B0",
        text_low="#555A78",
        text_dim="#3E4360",
        accent="#7c8cff",  # electric indigo
        accent2="#f97b6b",  # coral – regression / highlight
        accent3="#50fa7b",  # mint – positive
        accent4="#ffb86c",  # amber – warning / OOV        card_shadow="none",
        input_bg="#1C2030",
        scrollbar="#2C3050",
        gradient_lo="#1e2240",
        gradient_hi="#4a5270",
    )
else:
    T = dict(
        bg_page="#EDEEF2",
        bg_panel="#F5F6F9",
        bg_card="#ECEDF2",
        bg_hover="#E2E4EC",
        border="#D4D6E0",
        border2="#C0C3D4",
        text_hi="#111827",
        text_mid="#4B5563",
        text_low="#7B80A0",
        text_dim="#B0B5CC",
        accent="#7c8cff",  # electric indigo
        accent2="#f97b6b",  # coral – regression / highlight
        accent3="#50fa7b",  # mint – positive
        accent4="#ffb86c",  # amber – warning / OOV        card_shadow="0 1px 4px rgba(0,0,0,0.08)",
        input_bg="#F5F6F9",
        scrollbar="#C0C3D4",
        gradient_lo="#1e2240",
        gradient_hi="#4a5270",
    )

_CAT = [
    "#7c8cff",
    "#f97b6b",
    "#50fa7b",
    "#ffb86c",
    "#bd93f9",
    "#8be9fd",
    "#ff79c6",
    "#f1fa8c",
    "#6272a4",
    "#ff5555",
]

_CONFIDENCE_RANK = {"alta": 3, "media": 2, "baja": 1}


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_workflow_data():
    path = WORKFLOW_DB_PATH
    if not os.path.exists(path):
        st.warning(
            f"⚠️ No se encontró la base de datos del workflow en `{path}`. "
            "Ejecuta el workflow primero o monta el archivo correcto."
        )
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_discourse_state():
    path = DISCOURSE_STATE_PATH
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_merged_data():
    wf = load_workflow_data()
    if not wf:
        return {}

    disc = load_discourse_state()
    annotations_by_uce = disc.get("annotations_by_uce", {})

    if annotations_by_uce:
        # Normalize legacy format: some annotations use flat fields instead of spans[]
        for uce in wf.get("uces", []):
            raw_anns = annotations_by_uce.get(uce["id"], [])
            normalized = []
            for ann in raw_anns:
                # Legacy: single uce_id + quote → wrap in spans[]
                if "spans" not in ann and ann.get("uce_id") and ann.get("quote"):
                    ann = {
                        **ann,
                        "spans": [
                            {
                                "uce_id": ann["uce_id"],
                                "quote": ann["quote"],
                                "start_char": ann.get("start_char", -1),
                                "end_char": ann.get("end_char", -1),
                            }
                        ],
                    }
                normalized.append(ann)
            uce["discourse_annotations"] = normalized

    return wf


data = load_merged_data()
uces = data.get("uces", [])

if not uces:
    st.title("ALCESTE · Dashboard")
    st.info(
        "No hay datos del workflow disponibles todavía. "
        "Ejecuta primero el workflow o monta un `data/workflow_data.json` válido."
    )
    st.caption(f"Ruta esperada: `{WORKFLOW_DB_PATH}`")
    st.stop()


@st.cache_data(show_spinner=False)
def _prepare_annotation_frame(
    annotations_by_uce_json: str,  # pass as JSON string so Streamlit can hash it
    uces_json: str,
) -> pd.DataFrame:
    """
    Flattens annotations_by_uce into a tidy DataFrame, one row per SPAN.
    Joins with UCE-level data (cluster_id, phi_score, n_verbos, n_negaciones, etc.)

    Returns columns:
      uce_id, span_idx, trait, agent, subtype, confidence, conf_rank,
      quote, start_char, end_char,
      quant_valency, qual_valency (dict→str), omitted_args (list→str),
      reasoning,
      cluster_id, texto, phi_score,
      n_verbos, n_negaciones, n_pronombres, n_marcadores
    """
    import json

    ann_raw: Dict = json.loads(annotations_by_uce_json)
    uces_raw: List[Dict] = json.loads(uces_json)

    # Build fast UCE lookup
    uce_lookup: Dict[str, Dict] = {}
    for u in uces_raw:
        uid = re.sub(r"__mf\d+$", "", u.get("id", ""))
        uce_lookup[uid] = u

    rows = []
    for uce_id, anns in ann_raw.items():
        uid_clean = re.sub(r"__mf\d+$", "", uce_id)
        uce = uce_lookup.get(uid_clean, {})

        for ann_idx, ann in enumerate(anns):
            meta = ann.get("metadata", {})
            qv = meta.get("qualitative_valency", {})
            oa = meta.get("omitted_arguments", [])

            spans = ann.get("spans", [])
            if not spans:
                # Legacy flat annotation — wrap it
                spans = [
                    {
                        "uce_id": uce_id,
                        "quote": ann.get("quote", ""),
                        "start_char": ann.get("start_char", -1),
                        "end_char": ann.get("end_char", -1),
                    }
                ]

            for sp_idx, span in enumerate(spans):
                rows.append(
                    {
                        # Identity
                        "uce_id": uid_clean,
                        "ann_idx": ann_idx,
                        "span_idx": sp_idx,
                        # Annotation fields
                        "trait": ann.get("trait", "?"),
                        "agent": ann.get("agent", ""),
                        "subtype": ann.get("subtype", ""),
                        "confidence": ann.get("confidence", "baja"),
                        "conf_rank": _CONFIDENCE_RANK.get(
                            ann.get("confidence", "baja"), 1
                        ),
                        # Span
                        "quote": span.get("quote", ""),
                        "start_char": span.get("start_char", -1),
                        "end_char": span.get("end_char", -1),
                        # Rich metadata
                        "quant_valency": meta.get("quantitative_valency", ""),
                        "qual_valency": str(qv) if qv else "",
                        "qual_roles": list(qv.keys()) if isinstance(qv, dict) else [],
                        "omitted_args": oa,
                        "n_omitted": len(oa),
                        "reasoning": meta.get("reasoning", ""),
                        # UCE context
                        "cluster_id": uce.get("cluster_id"),
                        "texto": uce.get("texto", ""),
                        "phi_score": uce.get("phi_score", uce.get("stability", 0.0)),
                        # Pipeline cross-links (counts)
                        "n_verbos": len(uce.get("verbos", [])),
                        "n_negaciones": len(uce.get("negaciones", [])),
                        "n_pronombres": len(uce.get("pronombres", [])),
                        "n_marcadores": len(uce.get("marcadores_discursivos", [])),
                        "n_coref": len(uce.get("coref_chains", [])),
                        "registro": uce.get("registro", ""),
                        "mean_zipf": uce.get("metricas_lexicas", {}).get(
                            "mean_zipf", 0.0
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    return df


# FIX #3: normalise uce_map at load time — strip _mfN suffixes so lookups never miss
uce_map = {re.sub(r"__mf\d+$", "", u["id"]): u for u in uces}

ucs = data.get("ucs", [])
terminos_df = pd.DataFrame(data.get("terminos", []))
proyeccion = data.get("proyeccion", {})
config = data.get("config", {})
metadata_residuals = data.get("metadata_residuals", {})
pos_by_cluster = data.get("pos_by_cluster", {})
words_per_cluster = data.get("words_per_cluster", {})
uce_phi = data.get("uce_phi", [])
sintesis_por_clase = data.get("sintesis_por_clase", {})
sintesis_estructurada = data.get("sintesis_estructurada", {})
modalizacion_by_cluster = data.get("modalizacion_by_cluster", {})
cah_por_clase = data.get("cah_por_clase", {})
cah_terminos_global = data.get("cah_terminos_global", {})
# AUG #2: prefer umbral2 (longer UCs → more stable classification)
cdh_tree = data.get("cdh_tree_umbral2", data.get("cdh_tree_umbral1", {}))
multivariate = data.get("multivariate", {})
term_stability_raw = data.get("term_stability", [])
# AUG #1: shared document metadata — avoids per-UCE repetition
doc_meta_map = data.get("doc_metadata", {})
shap_data = data.get("shap_analysis", {})

# FIX 1: stem_summary — flatten forma_index if pipeline fix hasn't run
stem_summary = data.get("stem_summary", {})
if not stem_summary:
    for _cid, _stems in data.get("forma_index", {}).items():
        stem_summary[str(_cid)] = {}
        for _stem, _lemmas in _stems.items():
            stem_summary[str(_cid)][_stem] = {}
            for _lemma, _formas in _lemmas.items():
                stem_summary[str(_cid)][_stem][_lemma] = (
                    int(sum(_formas.values()))
                    if isinstance(_formas, dict)
                    else int(_formas)
                )

# FIX 2: vocabulario — derive from terminos if not saved
vocabulario = data.get("vocabulario", [])
if not vocabulario and not terminos_df.empty:
    vocabulario = sorted(terminos_df["termino"].unique().tolist())

# FIX 3: clustering_method — check config if not at top level
clustering_method = data.get("clustering_method") or config.get(
    "clustering_method", "cdh"
)

# FIX 5: uce_phi_dict
uce_phi_dict = {item["uce_id"]: item["phi_score"] for item in uce_phi}

# term_stability as a dict: {(str(termino), int(cluster)): selection_freq}
term_stability_dict = {}
for _ts in term_stability_raw:
    _key = (str(_ts.get("termino", "")), int(_ts.get("cluster", -1)))
    term_stability_dict[_key] = float(_ts.get("selection_freq", 0.0))

# Singular values and total inertia from AFC
_proj = data.get("proyeccion", {})
singular_values = _proj.get("singular_values", [])
total_inertia = float(_proj.get("total_inertia", 0.0))

# ─────────────────────────────────────────────────────────────
# DERIVED DATA
# ─────────────────────────────────────────────────────────────
clusters_unicos = sorted(
    {
        uc.get("cluster_label_double")
        for uc in ucs
        if uc.get("cluster_label_double") is not None
    }
)
if not clusters_unicos:
    clusters_unicos = sorted(
        {
            uce.get("cluster_id")
            for uce in uces
            if uce.get("cluster_id") is not None and uce.get("cluster_id") >= 0
        }
    )

if not sintesis_estructurada and clusters_unicos:
    _mt = {
        0: (
            "Esta clase articula un discurso centrado en la **experiencia cotidiana** "
            "y las relaciones interpersonales próximas."
        ),
        1: (
            "La Clase 1 organiza un campo semántico marcado por el **vocabulario de la "
            "institución y el deber**."
        ),
        2: ("Esta clase construye un **discurso de proyección y futuro**."),
    }
    _ml = {
        0: "• Mundo de la vida cotidiana\n• Discurso de la proximidad",
        1: "• Discurso institucional\n• Habla normativa",
        2: "• Discurso prospectivo\n• Agencia y proyecto",
    }
    sintesis_estructurada = {
        "per_class": {
            str(c): _mt.get(i, f"Clase {c}") for i, c in enumerate(clusters_unicos)
        },
        "label_proposals": {
            str(c): _ml.get(i, "• Pendiente") for i, c in enumerate(clusters_unicos)
        },
        "opposition_analysis": "Oposición principal a lo largo del Factor 1.",
        "global_synthesis": "Estructura de representación social heterogénea.",
        "_is_mockup": True,
    }

CLASS_COLORS_DARK = {0: "#E8A838", 1: "#5BA8DC", 2: "#5DC88A"}
CLASS_COLORS_LIGHT = {0: "#C8800A", 1: "#1B6FAD", 2: "#1D8A50"}


def _class_color(c):
    base = CLASS_COLORS_DARK if _dm else CLASS_COLORS_LIGHT
    if c in base:
        return base[c]
    from plotly.colors import qualitative

    return qualitative.Plotly[c % len(qualitative.Plotly)]


class_colors = {c: _class_color(c) for c in clusters_unicos}

class_sizes = Counter()
for uce in uces:
    cid = uce.get("cluster_id")
    if cid is not None and cid >= 0:
        class_sizes[cid] += 1

# FIX 6: words_per_cluster — normalise to int keys
if not words_per_cluster:
    _wpc = {}
    for uce in uces:
        cid = uce.get("cluster_id")
        if cid is not None and cid >= 0:
            _wpc[cid] = _wpc.get(cid, 0) + len(uce.get("lemmas", []))
    words_per_cluster = _wpc
else:
    _wpc = {}
    for k, v in words_per_cluster.items():
        try:
            _wpc[int(k)] = v
        except (ValueError, TypeError):
            pass
    words_per_cluster = _wpc

# Normalise pos_by_cluster to int keys (FIX type inconsistency)
_pbc_norm = {}
for k, v in pos_by_cluster.items():
    try:
        _pbc_norm[int(k)] = v
    except (ValueError, TypeError):
        pass
pos_by_cluster = _pbc_norm

# FIX 11: Modalization by Cluster (Safe Dict Parsing)
if not modalizacion_by_cluster:
    for uce in uces:
        cid = uce.get("cluster_id")
        if cid is not None and cid >= 0:
            modalizacion_by_cluster[int(cid)] = modalizacion_by_cluster.get(
                int(cid), {}
            )
            for m in uce.get("marcadores", []):
                lema = m.get("lemma", "")
                if lema:
                    modalizacion_by_cluster[int(cid)][lema] = (
                        modalizacion_by_cluster[int(cid)].get(lema, 0) + 1
                    )

# FIX 12: Proper Lemma-to-Forms Mapping
lemma_map = defaultdict(lambda: {"stems": set(), "formas": set(), "total_freq": 0})
_forma_idx = data.get("forma_index", {})
for cid, stems in _forma_idx.items():
    for stem, lemmas in stems.items():
        for lemma, formas in lemmas.items():
            if lemma not in lemma_map:
                lemma_map[lemma] = {"stems": set(), "formas": set(), "total_freq": 0}
            lemma_map[lemma]["stems"].add(stem)
            if isinstance(formas, dict):
                for f_exact, count in formas.items():
                    lemma_map[lemma]["formas"].add(f_exact)
                    lemma_map[lemma]["total_freq"] += count
            else:
                lemma_map[lemma]["formas"].add(lemma)
lemma_map = dict(lemma_map)


# AUG #1: helper to get metadata for a UCE, checking doc_meta_map first
def _uce_meta(uce: dict) -> dict:
    doc_id = str(uce.get("doc_id", ""))
    if doc_id and doc_id in doc_meta_map:
        return doc_meta_map[doc_id]
    # Last resort: whatever is in metadata minus the internal bookkeeping key
    raw = uce.get("metadata", {})
    return {k: v for k, v in raw.items() if k != "uce_local_idx"}


# ── FIX 13: Índice secuencial de documentos (origen_index) ──
origen_index = data.get("origen_index", {})
if not origen_index:
    for _u in uces:
        _m = _uce_meta(_u)
        _key = str(_m.get("origen", _u.get("doc_id", "Desconocido")))
        origen_index.setdefault(_key, []).append(_u["id"])

# ── INICIALIZACIÓN DE ESTADOS PARA LA PESTAÑA B ──
if "tab_b_view" not in st.session_state:
    st.session_state.tab_b_view = "search"
if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None
if "target_uce_id" not in st.session_state:
    st.session_state.target_uce_id = None

total_uces = len(uces)
total_forms = sum(len(u.get("lemmas", [])) for u in uces)
all_lemmas = [l for u in uces for l in u.get("lemmas", [])]
unique_forms = len(set(all_lemmas))
hapax = sum(1 for _, n in Counter(all_lemmas).items() if n == 1)
analyzed_forms = len(vocabulario)
classified_uces = sum(
    1 for u in uces if u.get("cluster_id") is not None and u.get("cluster_id") >= 0
)
classification_rate = (classified_uces / total_uces * 100) if total_uces else 0
vocabulary_richness = (analyzed_forms / unique_forms * 100) if unique_forms else 0

expl_inertia = proyeccion.get("explained_inertia", [0, 0])
axis1 = expl_inertia[0] * 100 if expl_inertia else 0
axis2 = expl_inertia[1] * 100 if len(expl_inertia) > 1 else 0

min_forms_uc = config.get("min_forms_uc", [13, 17])
tsj = config.get("tsj", 4)
corpus_name = config.get("corpus_name", "Corpus ALCESTE")
fecha = config.get("fecha_analisis", datetime.now().strftime("%d %B %Y"))
n_documentos = len(origen_index)
n_ucs = len(ucs)
# ─────────────────────────────────────────────────────────────
# MOTOR DE BÚSQUEDA
# ─────────────────────────────────────────────────────────────
PROXIMITY_THRESHOLD = 2


def _word_count(s: str) -> int:
    return len(s.split())


def _build_part_pattern(part: str, is_first: bool) -> re.Pattern:
    escaped = re.escape(part)
    prefix = r"(?<![a-záéíóúüñA-ZÁÉÍÓÚÜÑ])" if is_first else ""
    return re.compile(prefix + escaped, re.IGNORECASE)


def _find_spans(text: str, pattern: re.Pattern) -> list:
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def match_term(term: str, text: str) -> list:
    _WB = re.compile(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]+")

    def _word_end(pos):
        m = _WB.match(text, pos)
        return m.end() if m else pos

    parts = term.split("_")
    if len(parts) == 1:
        return _find_spans(text, _build_part_pattern(parts[0], True))

    part_spans = []
    for i, p in enumerate(parts):
        spans = _find_spans(text, _build_part_pattern(p, i == 0))
        if not spans:
            return []
        part_spans.append(spans)

    results = []

    def _chain(idx, prev_end):
        if idx == len(parts):
            return [prev_end]
        anchor = max(prev_end, _word_end(prev_end - 1))
        out = []
        for s, e in part_spans[idx]:
            if s < anchor:
                continue
            if _word_count(text[anchor:s]) <= PROXIMITY_THRESHOLD:
                for fe in _chain(idx + 1, e):
                    out.append(fe)
        return out

    for start, end0 in part_spans[0]:
        for fe in _chain(1, end0):
            results.append((start, fe))

    results.sort(key=lambda x: x[0])
    merged = []
    for sp in results:
        if merged and sp[0] < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], sp[1]))
        else:
            merged.append(list(sp))
    return [tuple(s) for s in merged]


@st.cache_data(show_spinner=False)
def _cached_corpus_search(terms_key: tuple, target_cls, logic: str) -> list:
    terms = list(terms_key)
    n_terms = len(terms)
    results = []
    for uce in uces:
        cid = uce.get("cluster_id")
        if target_cls is not None and cid != target_cls:
            continue
        if not terms:
            results.append(
                {
                    "uce": uce,
                    "matched_terms": [],
                    "match_count": 0,
                    "positions": {},
                    "phi": uce_phi_dict.get(uce.get("id", ""), 0.0),
                }
            )
            continue
        text = uce.get("texto", "")
        positions = {}
        matched = []
        for t in terms:
            spans = match_term(t, text)
            if spans:
                positions[t] = spans
                matched.append(t)
        if not matched:
            continue
        mc = len(matched)
        if logic == "AND" and mc < n_terms:
            continue
        results.append(
            {
                "uce": uce,
                "matched_terms": matched,
                "match_count": mc,
                "positions": positions,
                "phi": uce_phi_dict.get(uce.get("id", ""), 0.0),
            }
        )
    if not terms:
        results.sort(key=lambda r: -r["phi"])
    else:
        results.sort(key=lambda r: (-r["match_count"], -r["phi"]))
    return results


def corpus_search(terms, target_cls, logic):
    return _cached_corpus_search(tuple(terms), target_cls, logic)


def _build_term_color_map_cached(terms_key: tuple, classes_key: tuple) -> dict:
    terms = list(terms_key)
    selected_classes = list(classes_key)
    cmap = {}
    for term in terms:
        root = term.split("_")[0]
        best_phi = -999
        best_col = T["text_mid"]
        if not terminos_df.empty:
            for c in selected_classes:
                rows = terminos_df[
                    (terminos_df["cluster"] == c)
                    & (terminos_df["termino"].str.startswith(root))
                ]
                if not rows.empty:
                    pv = rows["phi"].max()
                    if pv > best_phi:
                        best_phi = pv
                        best_col = class_colors.get(c, T["text_mid"])
        cmap[term] = best_col
    return cmap


def build_term_color_map(terms, selected_classes):
    return _build_term_color_map_cached(tuple(terms), tuple(sorted(selected_classes)))


def render_highlighted_text(text: str, positions: dict, term_color_map: dict) -> str:
    flat = []
    for term, spans in positions.items():
        col = term_color_map.get(term, T["text_mid"])
        for s, e in spans:
            flat.append((s, e, col))
    if not flat:
        return _html.escape(text)
    flat.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    merged = []
    for sp in flat:
        if merged and sp[0] < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], sp[1]), merged[-1][2])
        else:
            merged.append(list(sp))
    out, cursor = "", 0
    for s, e, col in merged:
        out += _html.escape(text[cursor:s])
        word = text[s:e]
        out += (
            f'<mark style="background:{col}33;color:{col};'
            f'border-radius:2px;padding:0 2px;font-style:normal;font-weight:500">'
            f"{_html.escape(word)}</mark>"
        )
        cursor = e
    out += _html.escape(text[cursor:])
    return out


def sh(s: str) -> str:
    return _html.escape(str(s))


# FIX 7: resolve_stems_to_terms — added try/except guards
def resolve_stems_to_terms(stems: list) -> tuple:
    seen: set = set()
    terms: list = []
    label_map: dict = {}

    if not stems:
        return [], {}

    for s in stems:
        found_exact = False

        for class_data in stem_summary.values():
            if s in class_data:
                found_exact = True
                for lemma in class_data[s]:
                    if lemma not in seen:
                        seen.add(lemma)
                        terms.append(lemma)
                        label_map[lemma] = s

        if not found_exact and not terminos_df.empty:
            try:
                if s in terminos_df["termino"].values:
                    found_exact = True
                    if s not in seen:
                        seen.add(s)
                        terms.append(s)
                        label_map[s] = s
            except Exception:
                pass

        if "_" in s and not found_exact:
            if s not in seen:
                seen.add(s)
                terms.append(s)
                label_map[s] = s
            for part in s.split("_"):
                for class_data in stem_summary.values():
                    for lemma in class_data.get(part, {}):
                        if lemma not in seen:
                            seen.add(lemma)
                            terms.append(lemma)
                            label_map[lemma] = s

        elif not found_exact and s not in seen:
            seen.add(s)
            terms.append(s)
            label_map[s] = s

    return sorted(terms), label_map


def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def adapt_to_dashboard_format(raw: Dict) -> Dict:
    """
    Converts the raw Database format  {"schema_version":1, "uces":[...]}
    into the full dashboard format    {"meta":{...}, "documents":{...}, "stats":{...}}
    so the dashboard can load either file transparently.
    """
    uces: List[Dict] = raw.get("uces", [])
    if not uces:
        return raw  # nothing to adapt

    # Group UCEs by doc_id (fall back to "doc_0" if absent)
    doc_groups: Dict[str, List[Dict]] = defaultdict(list)
    for u in uces:
        doc_groups[u.get("doc_id", "doc_0")].append(u)

    documents: Dict[str, Dict] = {}
    for doc_id, doc_uces in doc_groups.items():
        documents[doc_id] = {
            "n_uces": len(doc_uces),
            "uces": doc_uces,
            "annotations": [],  # flat index not needed for rendering
        }

    stats = _build_stats_from_uces(uces)

    return {
        "meta": {
            "n_docs": len(doc_groups),
            "n_uces": len(uces),
            "doc_ids": list(doc_groups.keys()),
            "version": raw.get("schema_version", 1),
            "offset_convention": "global_per_doc",
            "_adapted": True,  # flag so we know it was converted
        },
        "documents": documents,
        "predicate_frames": {},
        "stats": stats,
    }


def is_raw_db_format(data: Dict) -> bool:
    """True if the file is the raw Database format, not the full dashboard format."""
    return "uces" in data and "documents" not in data


# ─────────────────────────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;1,6..72,300;1,6..72,400&display=swap');
:root {{
  --bg-page:    {{T['bg_page']}};  --bg-panel:   {{T['bg_panel']}};
  --bg-card:    {{T['bg_card']}};  --bg-hover:   {{T['bg_hover']}};
  --border:     {{T['border']}};   --border2:    {{T['border2']}};
  --text-hi:    {{T['text_hi']}};  --text-mid:   {{T['text_mid']}};
  --text-low:   {{T['text_low']}}; --text-dim:   {{T['text_dim']}};
  --accent:     {{T['accent']}};   --card-shadow:{{T['card_shadow']}};
  --input-bg:   {{T['input_bg']}}; --scrollbar:  {{T['scrollbar']}};
  --font-mono: 'IBM Plex Mono', 'Courier New', monospace;
  --font-serif:'Newsreader', Georgia, serif;
  --r-sm: 6px; --r-md: 10px; --r-lg: 14px;
  --neg:       #e05c5c;
  --neg-scope: rgba(224,92,92,0.08);
  --pron:      #3ec9c9;
  --verb-ind:  #5b9cf6;
  --verb-sub:  #a78bfa;
  --verb-imp:  #f97b6b;
  --verb-cond: #fbbf24;
  --quant-uni: #4ade80;
  --quant-neg: #f87171;
  --quant-num: #94a3b8;
  --quant-pro: #fbbf24;
  --quant-exi: #2dd4bf;
  --adv:       #c084fc;
  --disc:      #818cf8;
  --coref0:    #2dd4bf;
  --coref1:    #f97b6b;
  --coref2:    #a78bfa;
  --coref3:    #fbbf24;
  --reg-col:   #fb923c;
  --reg-for:   #60a5fa;
  --reg-tec:   #c084fc;
  --reg-mix:   #64748b;

}}

.stApp {{ background: var(--bg-page) !important; color: var(--text-hi); }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
[data-testid="stSidebar"] {{ display: none !important; }}
.stMarkdown p {{ margin: 0; }}
footer {{ display: none !important; }}
#MainMenu {{ display: none !important; }}
[data-testid="metric-container"] {{
    background: var(--bg-panel); border-right: 1px solid var(--border);
    padding: 12px 16px 10px !important; margin: 0 !important; transition: background .2s;
}}
[data-testid="stMetricLabel"] p {{
    font-family: var(--font-mono) !important; font-size: 9px !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    color: var(--text-low) !important;
}}
[data-testid="stMetricValue"] {{
    font-family: var(--font-mono) !important; font-size: 20px !important;
    font-weight: 500 !important; color: var(--text-hi) !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    background: var(--bg-panel); border-bottom: 1px solid var(--border);
    padding: 0 24px; gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.10em;
    text-transform: uppercase; color: var(--text-low); background: transparent;
    border-bottom: 2px solid transparent; padding: 10px 20px;
    margin-right: 2px; transition: color .15s;
}}
.stTabs [aria-selected="true"] {{
    color: var(--text-hi) !important; border-bottom-color: var(--accent) !important;
}}
.stTabs [data-baseweb="tab-panel"] {{ background: var(--bg-page); padding: 0 !important; }}
.sec-rule {{
    display: flex; align-items: center; gap: 12px; padding: 28px 28px 14px;
}}
.sec-rule::after {{
    content: ''; flex: 1; height: 1px; background: var(--border);
}}
.sec-rule span {{
    font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--text-low); white-space: nowrap;
}}
.panel-hdr {{
    font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--text-low);
    padding: 8px 0 6px; border-bottom: 1px solid var(--border); margin-bottom: 10px;
}}
.uce-card {{
    background: var(--bg-card); border: 1px solid var(--border2);
    border-radius: var(--r-sm); padding: 10px 14px;
    margin-bottom: 8px; box-shadow: var(--card-shadow); transition: border-color .15s;
}}
.uce-card:hover {{ border-color: var(--accent); }}
.uce-meta {{
    font-family: var(--font-mono); font-size: 9px; color: var(--text-low);
    margin-bottom: 6px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
}}
.uce-text {{
    font-family: var(--font-serif); font-size: 13px;
    font-style: italic; color: var(--text-mid); line-height: 1.75;
}}
.match-badge {{
    display: inline-flex; align-items: center; padding: 1px 7px;
    border-radius: 10px; font-family: var(--font-mono);
    font-size: 9px; font-weight: 500; letter-spacing: .04em;
}}
.phi-row {{
    display: grid; grid-template-columns: 140px 1fr 48px;
    align-items: center; gap: 8px; padding: 4px 0;
    border-bottom: 1px solid var(--border);
}}
.phi-row:last-child {{ border-bottom: none; }}
.phi-word {{
    font-family: var(--font-mono); font-size: 10.5px; color: var(--text-hi);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.phi-bar-bg {{ background: var(--bg-hover); height: 4px; border-radius: 2px; overflow: hidden; }}
.phi-fill   {{ height: 100%; border-radius: 2px; }}
.phi-val    {{ font-family: var(--font-mono); font-size: 10px; color: var(--text-low); text-align: right; }}
.synth-card {{
    background: var(--bg-panel); border: 1px solid var(--border2);
    border-radius: var(--r-md); overflow: hidden; margin: 4px 0 16px;
    box-shadow: var(--card-shadow);
}}
.synth-header {{
    display: flex; align-items: center; gap: 12px;
    padding: 14px 20px 12px; border-bottom: 1px solid var(--border);
}}
.synth-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}
.synth-class-label {{
    font-family: var(--font-mono); font-size: 10px;
    letter-spacing: .16em; text-transform: uppercase; color: var(--text-mid);
}}
.synth-body {{
    padding: 18px 22px 20px; font-family: var(--font-serif);
    font-size: 14px; line-height: 1.85; color: var(--text-mid); font-style: italic;
}}
.synth-body strong {{ font-style: normal; font-weight: 600; color: var(--text-hi); }}
.synth-footer {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 22px 10px; border-top: 1px solid var(--border);
    font-family: var(--font-mono); font-size: 9px; color: var(--text-dim);
}}
.retention-gauge {{
    padding: 18px 16px; background: var(--bg-panel);
    border-left: 3px solid var(--ret-color, #5DC88A);
    border-radius: 0 var(--r-sm) var(--r-sm) 0; margin: 4px 0;
    box-shadow: var(--card-shadow);
}}
.retention-pct {{
    font-family: var(--font-mono); font-size: 34px; font-weight: 600;
    color: var(--ret-color, #5DC88A); line-height: 1;
}}
.retention-label {{
    font-family: var(--font-mono); font-size: 9px; letter-spacing: .14em;
    text-transform: uppercase; color: var(--text-low); margin-top: 4px;
}}
.retention-msg {{
    font-family: var(--font-mono); font-size: 10px;
    color: var(--ret-color, #5DC88A); margin-top: 6px;
}}
.aux-table {{ width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 10.5px; }}
.aux-table th {{
    text-align: left; padding: 5px 10px; font-size: 9px;
    letter-spacing: .1em; text-transform: uppercase; color: var(--text-low);
    border-bottom: 1px solid var(--border);
}}
.aux-table td {{ padding: 5px 10px; border-bottom: 1px solid var(--border); color: var(--text-hi); }}
.aux-table tr:last-child td {{ border-bottom: none; }}
.info-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid var(--border);
}}
.info-row:last-child {{ border-bottom: none; }}
.info-key {{ font-family: var(--font-mono); font-size: 10px; color: var(--text-low); }}
.info-val {{ font-family: var(--font-mono); font-size: 10px; color: var(--text-hi); font-weight: 500; }}
.stTextInput > div > div > input {{
    background: var(--input-bg) !important; border: 1px solid var(--border2) !important;
    color: var(--text-hi) !important; font-family: var(--font-mono) !important;
    font-size: 11px !important; border-radius: var(--r-sm) !important;
}}
.stTextInput label {{ color: var(--text-low) !important; font-family: var(--font-mono) !important; font-size: 9px !important; }}
.stSelectbox > div > div {{ background: var(--input-bg) !important; border: 1px solid var(--border2) !important; }}
.stRadio label {{ font-family: var(--font-mono) !important; font-size: 10px !important; color: var(--text-mid) !important; }}
.stButton > button {{
    font-family: var(--font-mono) !important; font-size: 9.5px !important;
    letter-spacing: .08em !important; border-radius: 20px !important;
    transition: all 0.18s ease !important; border-width: 1px !important;
    border-style: solid !important; min-height: 28px !important; padding: 2px 14px !important;
}}
.stButton > button[kind="secondary"] {{
    background: var(--bg-panel) !important; border-color: var(--border2) !important;
    color: var(--text-low) !important;
}}
.stButton > button[kind="secondary"]:hover {{
    background: var(--bg-hover) !important; border-color: var(--text-low) !important;
    color: var(--text-mid) !important; transform: translateY(-1px);
}}
.stButton > button[kind="primary"] {{
    background: var(--bg-hover) !important; border-color: var(--accent) !important;
    color: var(--text-hi) !important;
    box-shadow: 0 0 8px color-mix(in srgb, var(--accent) 25%, transparent) !important;
}}
.stButton > button[kind="primary"]:hover {{ filter: brightness(1.1); transform: translateY(-1px); }}
.corpus-scroll {{ max-height: 460px; overflow-y: auto; padding-right: 6px; }}
.corpus-scroll::-webkit-scrollbar {{ width: 3px; }}
.corpus-scroll::-webkit-scrollbar-track {{ background: transparent; }}
.corpus-scroll::-webkit-scrollbar-thumb {{ background: var(--scrollbar); border-radius: 3px; }}
.residual-scroll {{ max-height: 220px; overflow-y: auto; padding-right: 4px; }}
.residual-scroll::-webkit-scrollbar {{ width: 3px; }}
.residual-scroll::-webkit-scrollbar-track {{ background: transparent; }}
.residual-scroll::-webkit-scrollbar-thumb {{ background: var(--scrollbar); border-radius: 3px; }}
.footer {{
    background: var(--bg-panel); border-top: 1px solid var(--border);
    padding: 9px 28px; font-family: var(--font-mono); font-size: 9px;
    color: var(--text-low); display: flex; justify-content: space-between; margin-top: 32px;
}}
.stPlotlyChart {{ background: transparent !important; }}
hr {{ border-color: var(--border) !important; margin: 0 !important; }}



/* ── top bar ── */
.topbar {{
  display: flex; align-items: center; gap: 1.5rem;
  padding: 0.6rem 1rem; margin-bottom: 0.8rem;
  background: var(--bg1); border: 1px solid var(--border);
  border-radius: 8px;
}}
.topbar-title {{ font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.15rem; color: var(--text-hi); white-space: nowrap; }}
.topbar-diamond {{ color: var(--verb-ind); margin: 0 0.25rem; }}

/* ── layer pills ── */
.layer-bar {{
  display: flex; flex-wrap: wrap; gap: 0.4rem;
  padding: 0.4rem 0; margin-bottom: 0.6rem;
}}
.layer-pill {{
  font-family: 'DM Mono', monospace; font-size: 0.7rem;
  padding: 0.2rem 0.65rem; border-radius: 20px;
  border: 1px solid; cursor: pointer; transition: all 0.15s;
  user-select: none; white-space: nowrap;
}}

/* ── stat cards ── */
.kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-bottom: 0.8rem; }}
.kpi-card {{
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.6rem 0.75rem;
}}
.kpi-value {{ font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: var(--text-hi); line-height: 1; }}
.kpi-label {{ font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem; }}

/* ── plotly container ── */
.stPlotlyChart {{ border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }}

/* ── UCE block ── */
.uce-block {{
  background: var(--bg1); border: 1px solid var(--border);
  border-radius: 8px; margin-bottom: 0.75rem; overflow: hidden;
}}
.uce-header {{
  display: flex; align-items: center; gap: 0.75rem;
  padding: 0.5rem 0.8rem;
  background: var(--bg2); border-bottom: 1px solid var(--border);
  font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--text-dim);
  cursor: pointer;
}}
.uce-id {{ font-weight: 500; color: var(--text-hi); }}
.reg-pill {{
  padding: 0.1rem 0.5rem; border-radius: 12px; font-size: 0.65rem; font-weight: 600;
  font-family: 'DM Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em;
}}
.reg-coloquial {{ background: rgba(251,146,60,0.2); color: var(--reg-col); border: 1px solid rgba(251,146,60,0.35); }}
.reg-formal    {{ background: rgba(96,165,250,0.2); color: var(--reg-for); border: 1px solid rgba(96,165,250,0.35); }}
.reg-tecnico   {{ background: rgba(192,132,252,0.2); color: var(--reg-tec); border: 1px solid rgba(192,132,252,0.35); }}
.reg-mixto     {{ background: rgba(100,116,139,0.2); color: var(--reg-mix); border: 1px solid rgba(100,116,139,0.35); }}

.uce-separator-high {{ height: 3px; background: linear-gradient(90deg, var(--neg) 0%, transparent 100%); margin: 0.4rem 0; border-radius: 2px; }}
.uce-separator-low  {{ height: 1px; background: var(--border); margin: 0.4rem 0; }}

/* ── annotated text ── */
.annotated-text {{
line-height: 1.85; display: inline;
  font-family: 'Spectral', Georgia, serif; font-size: 0.92rem;
  color: var(--text);
}}
/* annotation spans */
.ann-neg     {{ text-decoration: underline 2px #e05c5c; text-underline-offset: 3px;  }}
.ann-neg-scope {{ background: var(--neg-scope); border-radius: 3px; padding: 0 2px; }}
.ann-pron    {{ background: rgba(62,201,201,0.18); border-radius: 3px; color: var(--pron); }}
.ann-verb-ind  {{ color: var(--verb-ind); font-weight: 600; }}
.ann-verb-sub  {{ color: var(--verb-sub); font-weight: 600; }}
.ann-verb-imp  {{ color: var(--verb-imp); font-weight: 600; }}
.ann-verb-cond {{ color: var(--verb-cond); font-weight: 600; }}
.ann-adv     {{ color: var(--adv); font-style: italic; }}
.ann-disc    {{ background: rgba(129,140,248,0.2); color: var(--disc); border-radius: 3px; padding: 0 3px; font-family: 'DM Mono', monospace; font-size: 0.8em; }}
.ann-prodrop {{ color: var(--pron); opacity: 0.6; font-family: 'DM Mono', monospace; font-size: 0.78em; }}
.ann-quant-uni {{ color: var(--quant-uni); }}
.ann-quant-neg {{ color: var(--quant-neg); }}
.ann-quant-num {{ color: var(--quant-num); }}
.ann-quant-pro {{ color: var(--quant-pro); }}
.ann-oov       {{ border-left: 2px solid var(--reg-col); padding-left: 2px; }}
.ann-high-surp {{ background: rgba(251,191,36,0.1); border-radius: 2px; }}
.ann-entity    {{ border-bottom: 2px solid var(--coref0); padding-bottom: 1px; }}
.ann-coref     {{ border-bottom: 2px solid; padding-bottom: 1px; cursor: pointer; }}
.ann-insub     {{ background: #fbbf2444; border-radius: 3px; padding: 0 2px; }}

 /* Capas discursivas */
 .ann-discourse-valencia      {{ background: #ffb3ba44; border-radius: 3px; }}
 .ann-discourse-telicidad     {{ background: #baffc944; border-radius: 3px; }}
 .ann-discourse-ideacion      {{ background: #bae1ff44; border-radius: 3px; }}
 .ann-discourse-eufemismo     {{ background: #ffffba44; border-radius: 3px; }}
 .ann-discourse-metafora      {{ background: #ffdfba44; border-radius: 3px; }}
 .ann-discourse-juicios-valor {{ background: #d4b8d944; border-radius: 3px; }}
 .ann-discourse-oposiciones   {{ background: #b0e0e644; border-radius: 3px; }}
 .ann-discourse-mitopoetica   {{ background: #f5c6a044; border-radius: 3px; }}

/* ── detail panel ── */
.detail-panel {{
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.5rem;
  font-size: 0.85rem;
}}
.detail-label {{font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.07em; }}
.detail-value {{ color: var(--text-hi); margin-bottom: 0.35rem; }}
.morph-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.4rem; margin-top: 0.4rem; }}
.morph-cell {{ background: var(--bg3); border-radius: 4px; padding: 0.3rem 0.5rem; }}

/* ── tabs override ── */
.stTabs [data-baseweb="tab"] {{
  font-family: 'DM Mono', monospace; font-size: 0.72rem;
  color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em;
  padding: 0.35rem 0.8rem;
}}
.stTabs [aria-selected="true"] {{ color: var(--text-hi) !important; }}
.stTabs [data-baseweb="tab-border"] {{ background: var(--verb-ind) !important; }}
.stTabs [data-baseweb="tab-list"] {{ background: var(--bg2) !important; border-radius: 6px 6px 0 0; }}

/* ── tables ── */
.stDataFrame {{ font-family: 'DM Mono', monospace; font-size: 0.75rem; }}
table {{ font-family: 'DM Mono', monospace; font-size: 0.77rem; width: 100%; border-collapse: collapse; }}
th {{ color: var(--text-dim); border-bottom: 1px solid var(--border); padding: 0.3rem 0.5rem; text-align: left; }}
td {{ padding: 0.25rem 0.5rem; border-bottom: 1px solid rgba(42,47,66,0.5); }}

/* scrollable UCE panel */
.uce-scroll {{ max-height: 80vh; overflow-y: auto; padding-right: 0.4rem; }}
.uce-scroll::-webkit-scrollbar {{ width: 4px; }}
.uce-scroll::-webkit-scrollbar-track {{ background: var(--bg); }}
.uce-scroll::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}

/* divider */
hr.thin {{ border: none; border-top: 1px solid var(--border); margin: 0.6rem 0; }}

/* selectbox / multiselect */
[data-baseweb="select"] {{ background: var(--bg2) !important; border-color: var(--border) !important; }}
[data-baseweb="tag"]    {{ background: var(--bg3) !important; }}

/* info / warning boxes */
.info-box {{
  background: rgba(91,156,246,0.1); border: 1px solid rgba(91,156,246,0.3);
  border-radius: 6px; padding: 0.5rem 0.75rem; font-size: 0.82rem;
  font-family: 'DM Mono', monospace; color: var(--verb-ind);
}}
/* Colores para los tags del multiselect de capas */
div[data-baseweb="select"] span[data-baseweb="tag"]:has(span[title="Verbos"]) {{ background-color: var(--verb-ind) !important; color: #fff !important; border: none !important;}}
div[data-baseweb="select"] span[data-baseweb="tag"]:has(span[title="Negación"]) {{ background-color: var(--neg) !important; color: #fff !important; border: none !important;}}
div[data-baseweb="select"] span[data-baseweb="tag"]:has(span[title="Pronombres"]) {{ background-color: var(--pron) !important; color: #111 !important; border: none !important;}}
div[data-baseweb="select"] span[data-baseweb="tag"]:has(span[title="Adverbios"]) {{ background-color: var(--adv) !important; color: #fff !important; border: none !important;}}
div[data-baseweb="select"] span[data-baseweb="tag"]:has(span[title="Discurso"]) {{ background-color: var(--disc) !important; color: #fff !important; border: none !important;}}
div[data-baseweb="select"] span[data-baseweb="tag"]:has(span[title="Cuantificadores"]) {{ background-color: var(--quant-uni) !important; color: #111 !important; border: none !important;}}
div[data-baseweb="select"] span[data-baseweb="tag"]:has(span[title="Correferencia"]) {{ background-color: var(--coref0) !important; color: #111 !important; border: none !important;}}

.clickable-token {{
    cursor: pointer;
    transition: filter 0.15s;
}}
.clickable-token:hover {{
    filter: brightness(1.35) drop-shadow(0 0 2px rgba(255,255,255,0.2));
}}
            </style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# CABECERA
# ─────────────────────────────────────────────────────────────
hdr_col, toggle_col = st.columns([10, 1])
with hdr_col:
    st.markdown(
        f"""
    <div style="background:var(--bg-panel);border-bottom:1px solid var(--border2);
                padding:14px 28px 12px;">
      <div style="font-family:var(--font-mono);font-size:10px;letter-spacing:.18em;
                  color:var(--accent);text-transform:uppercase;margin-bottom:3px">
        ALCESTE · Analyse de Données Textuelles
      </div>
      <div style="font-family:var(--font-serif);font-size:22px;font-weight:300;
                  color:var(--text-hi);letter-spacing:-.01em;margin-bottom:4px">
        {corpus_name} · {n_documentos} documentos
      </div>
      <div style="font-family:var(--font-mono);font-size:10px;color:var(--text-low);
                  letter-spacing:.04em">
        Clasificación doble (121) · {clustering_method} ·
        min_forms={min_forms_uc} · tsj={tsj} · {fecha}
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with toggle_col:
    st.markdown('<div style="padding-top:16px">', unsafe_allow_html=True)
    mode_label = "☀ Día" if _dm else "☾ Noche"
    if st.button(mode_label, key="toggle_mode", width="stretch"):
        st.session_state.dark_mode = not _dm
        st.cache_data.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────
cols_stat = st.columns(10)
STATS = [
    (str(len(class_sizes)), "clases"),
    (f"{classification_rate:.1f}%", "tasa clasif."),
    (str(len(uces)), "UCE generadas"),
    (str(classified_uces), "UCE clasificadas"),
    (str(n_ucs), "UC generadas"),
    (str(total_forms), "total formas"),
    (str(analyzed_forms), "formas analizadas"),
    (str(unique_forms), "formas distintas"),
    (str(hapax), "hapax"),
    (f"{vocabulary_richness:.1f}%", "riqueza vocab."),
]
for col, (val, lbl) in zip(cols_stat, STATS):
    with col:
        st.metric(label=lbl, value=val)


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT ADAPTER  (raw DB → dashboard format)
# ─────────────────────────────────────────────────────────────────────────────
def _build_stats_from_uces(uces: List[Dict]) -> Dict:
    """Derives all stats sub-dicts from a flat UCE list."""
    from collections import Counter, defaultdict

    import numpy as np

    n_tokens_total = sum(
        u.get("metricas_lexicas", {}).get("num_tokens", 1) or 1 for u in uces
    )

    def norm(count):
        return (count / n_tokens_total * 1000) if n_tokens_total else 0

    # ── negaciones ────────────────────────────────────────────────────────
    neg_ctr = Counter(n.get("tipo", "?") for u in uces for n in u.get("negaciones", []))
    neg_total = sum(neg_ctr.values()) or 1
    negaciones = [
        {"tipo": t, "freq_abs": c, "freq_rel": c / neg_total, "freq_norm": norm(c)}
        for t, c in neg_ctr.most_common()
    ]

    # ── verbos ────────────────────────────────────────────────────────────
    verb_features = ["modo", "tiempo", "aspecto", "voz", "tipo_subordinacion"]
    v_rows = []
    all_verbs = [v for u in uces for v in u.get("verbos", [])]
    total_verbs = len(all_verbs) or 1
    for feat in verb_features:
        ctr = Counter(v.get(feat) for v in all_verbs if v.get(feat) is not None)
        if not ctr:
            continue
        probs = np.array(list(ctr.values())) / sum(ctr.values())
        ent = float(-np.sum(probs * np.log(probs + 1e-12)))
        for val, cnt in ctr.most_common():
            v_rows.append(
                {
                    "categoria": feat,
                    "valor": val,
                    "freq_abs": cnt,
                    "freq_rel": cnt / total_verbs,
                    "freq_norm": norm(cnt),
                    "entropia_categoria": ent,
                }
            )
    verbos = v_rows

    # ── adverbios ─────────────────────────────────────────────────────────
    adv_ctr = Counter(
        a.get("categoria", "?") for u in uces for a in u.get("adverbios", [])
    )
    adv_total = sum(adv_ctr.values()) or 1
    adverbios = [
        {"categoria": c, "freq_abs": n, "freq_rel": n / adv_total, "freq_norm": norm(n)}
        for c, n in adv_ctr.most_common()
    ]

    # ── pronombres ────────────────────────────────────────────────────────
    tipo_ctr = Counter(
        p.get("tipo", "?") for u in uces for p in u.get("pronombres", [])
    )
    sub_ctr = Counter(
        p.get("subtipo", "?")
        for u in uces
        for p in u.get("pronombres", [])
        if p.get("subtipo")
    )
    pron_total = sum(tipo_ctr.values()) or 1
    n_pro = tipo_ctr.get("NULO", 0)
    n_exp = tipo_ctr.get("EXPLICITO", 0)
    pronombres = (
        [
            {
                "nivel": "tipo",
                "clave": k,
                "freq_abs": v,
                "freq_rel": v / pron_total,
                "freq_norm": norm(v),
            }
            for k, v in tipo_ctr.most_common()
        ]
        + [
            {
                "nivel": "subtipo",
                "clave": k,
                "freq_abs": v,
                "freq_rel": v / pron_total,
                "freq_norm": norm(v),
            }
            for k, v in sub_ctr.most_common()
        ]
        + [
            {
                "nivel": "ratio",
                "clave": "prodrop_vs_explicito",
                "freq_abs": n_pro,
                "freq_rel": n_pro / (n_pro + n_exp) if (n_pro + n_exp) else 0,
                "freq_norm": norm(n_pro),
            }
        ]
    )

    # ── entropías ─────────────────────────────────────────────────────────
    ent_sources = {
        "negacion_tipo": [n.get("tipo") for u in uces for n in u.get("negaciones", [])],
        "pronombre_tipo": [
            p.get("tipo") for u in uces for p in u.get("pronombres", [])
        ],
        "adverbio_cat": [
            a.get("categoria") for u in uces for a in u.get("adverbios", [])
        ],
        "verbo_modo": [v.get("modo") for u in uces for v in u.get("verbos", [])],
        "verbo_voz": [v.get("voz") for u in uces for v in u.get("verbos", [])],
        "verbo_aspecto": [v.get("aspecto") for u in uces for v in u.get("verbos", [])],
    }
    entropias = []
    for fname, vals in ent_sources.items():
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        ctr = Counter(vals)
        total = sum(ctr.values())
        probs = np.array(list(ctr.values())) / total
        H = float(-np.sum(probs * np.log(probs + 1e-12)))
        entropias.append(
            {
                "feature": fname,
                "n_occurrences": total,
                "n_types": len(ctr),
                "entropy_nat": round(H, 4),
                "top_3": ", ".join(f"{k}:{v}" for k, v in ctr.most_common(3)),
            }
        )

    # ── registros ─────────────────────────────────────────────────────────
    reg_ctr = Counter(u.get("registro", "mixto") for u in uces)
    reg_total = len(uces) or 1
    registros = [
        {"registro": r, "count": c, "percentage": round(c / reg_total * 100, 2)}
        for r, c in reg_ctr.most_common()
    ]

    # ── Zipf bands ────────────────────────────────────────────────────────
    BANDS = [
        ("B1_nuclear", 6, 9),
        ("B2_alta", 5, 6),
        ("B3_media", 4, 5),
        ("B4_baja", 3, 4),
        ("B5_rara_tecnica", 0, 3),
    ]
    all_lemmas_with_zipf = []
    for u in uces:
        m = u.get("metricas_lexicas", {})
        mean_z = m.get("mean_zipf", 5.0)
        for l in u.get("lemmas", []):
            all_lemmas_with_zipf.append(mean_z)  # approximate per-token with UCE mean
    freq_bands = []
    for bname, lo, hi in BANDS:
        n = sum(1 for z in all_lemmas_with_zipf if lo <= z < hi)
        pct = (
            round(n / len(all_lemmas_with_zipf) * 100, 2) if all_lemmas_with_zipf else 0
        )
        freq_bands.append(
            {
                "banda": bname,
                "rango_zipf": f"{lo}–{hi}",
                "n_tokens": n,
                "n_types": 0,
                "pct_tokens": pct,
            }
        )

    # ── cognitive load ─────────────────────────────────────────────────────
    carga_cognitiva = {
        u["id"]: u.get("metricas_lexicas", {}).get("mean_surprisal_content", 0)
        for u in uces
    }

    # ── crosstabs ─────────────────────────────────────────────────────────
    def crosstab(f1, f2):
        pairs = [
            (v.get(f1), v.get(f2))
            for u in uces
            for v in u.get("verbos", [])
            if v.get(f1) and v.get(f2)
        ]
        if not pairs:
            return []
        idx = sorted(set(p[0] for p in pairs))
        cols = sorted(set(p[1] for p in pairs))
        mat = defaultdict(lambda: defaultdict(int))
        for a, b in pairs:
            mat[a][b] += 1
        rows = [{f1: i, **{c: mat[i].get(c, 0) for c in cols}} for i in idx]
        return rows

    # ── global summary ────────────────────────────────────────────────────
    numeric_fields = [
        ("n_tokens", lambda u: u.get("metricas_lexicas", {}).get("num_tokens", 0)),
        ("ttr", lambda u: u.get("metricas_lexicas", {}).get("ttr", 0)),
        ("guiraud", lambda u: u.get("metricas_lexicas", {}).get("guiraud", 0)),
        ("topic_shift", lambda u: u.get("topic_shift_prev", 0)),
        ("div_semantica", lambda u: u.get("diversidad_semantica", 0)),
        ("mean_zipf", lambda u: u.get("metricas_lexicas", {}).get("mean_zipf", 0)),
        (
            "mean_surprisal",
            lambda u: u.get("metricas_lexicas", {}).get("mean_surprisal_content", 0),
        ),
    ]
    global_rows = []
    for name, fn in numeric_fields:
        vals = [fn(u) for u in uces]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        a = np.array(vals)
        global_rows.append(
            {
                "index": name,
                "count": len(a),
                "mean": round(float(a.mean()), 4),
                "std": round(float(a.std()), 4),
                "min": round(float(a.min()), 4),
                "Q1": round(float(np.percentile(a, 25)), 4),
                "median": round(float(np.median(a)), 4),
                "Q3": round(float(np.percentile(a, 75)), 4),
                "max": round(float(a.max()), 4),
            }
        )

    # ── OOV ──────────────────────────────────────────────────────────────
    oov_rows = []
    for u in uces:
        rare = u.get("metricas_lexicas", {}).get("top_rare_content_words", [])
        for item in rare:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                oov_rows.append({"lemma": item[0], "zipf": item[1], "uce_id": u["id"]})
    oov_df_rows = []
    if oov_rows:
        from collections import defaultdict as dd2

        grouped = dd2(list)
        for r in oov_rows:
            grouped[r["lemma"]].append(r)
        for lemma, items in sorted(grouped.items(), key=lambda x: x[1][0]["zipf"]):
            oov_df_rows.append(
                {
                    "lemma": lemma,
                    "freq_abs": len(items),
                    "n_uces": len(set(i["uce_id"] for i in items)),
                    "zipf": items[0]["zipf"],
                }
            )

    return {
        "global": global_rows,
        "verbos": verbos,
        "negaciones": negaciones,
        "adverbios": adverbios,
        "pronombres": pronombres,
        "entropias": entropias,
        "registros": registros,
        "frecuencias_zipf": freq_bands,
        "carga_cognitiva": carga_cognitiva,
        "oov": oov_df_rows,
        "crosstab_modo_aspecto": crosstab("modo", "aspecto"),
        "crosstab_tiempo_voz": crosstab("tiempo", "voz"),
        "predicate_frames": {},
    }


# ─────────────────────────────────────────────────────────────
# ESTADO GLOBAL
# ─────────────────────────────────────────────────────────────
class_list = sorted(class_sizes.keys())

if not class_list:
    st.warning(
        "⚠️ No se encontraron clases clasificadas. "
        "Verifica que el pipeline haya corrido correctamente."
    )
    st.markdown(
        f"""
    <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-low);
                padding:16px;background:var(--bg-card);border-radius:6px;margin:12px 0">
        <b>Diagnóstico:</b><br>
        • Total UCEs: {total_uces}<br>
        • UCEs clasificadas (cluster_id ≥ 0): {classified_uces}<br>
        • Clases únicas en `ucs`: {len(clusters_unicos)}<br>
        • Vocabulario: {len(vocabulario)} términos<br>
        • terminos_df vacío: {terminos_df.empty}
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.stop()

if "selected_classes" not in st.session_state:
    st.session_state.selected_classes = class_list.copy()
if "selected_class_single" not in st.session_state:
    st.session_state.selected_class_single = class_list[0]

selected_classes = list(st.session_state.selected_classes)
if not selected_classes:
    selected_classes = class_list.copy()
    st.session_state.selected_classes = selected_classes


# ─────────────────────────────────────────────────────────────
# VIZ HELPERS
# ─────────────────────────────────────────────────────────────
def _plot_defaults():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text_mid"], family="IBM Plex Mono, monospace", size=9),
    )


def _layout(**overrides) -> dict:
    """Merge caller overrides into the base layout."""
    base = dict(_plot_defaults)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


def _title_style(text: str) -> dict:
    return dict(
        text=text, font=dict(size=12, color=T["text_mid"]), x=0.0, xanchor="left"
    )


def make_cdh_dendrogram(_dm_key: bool, root_at_top: bool = True):
    """Draw the CDH decision tree. If root_at_top=True, root is placed at top."""
    if not cdh_tree:
        return None

    stable_counts = {}
    for uc in ucs:
        if hasattr(uc, "cluster_label_double") and uc.cluster_label_double is not None:
            label = uc.cluster_label_double
            stable_counts[label] = stable_counts.get(label, 0) + 1

    def get_total(node):
        if isinstance(node, dict):
            return node.get("n_ucs", 0)
        elif isinstance(node, list):
            return sum(get_total(child) for child in node)
        else:
            return 0

    total_ucs = get_total(cdh_tree)

    # Build graph
    G = nx.DiGraph()
    node_counter = 0
    nodes = {}

    def add_node(node, parent_id=None):
        nonlocal node_counter
        node_id = node_counter
        node_counter += 1
        nodes[node_id] = node
        G.add_node(node_id, label=node.get("label", -1), n_ucs=node.get("n_ucs", 0))
        if parent_id is not None:
            G.add_edge(parent_id, node_id)
        for child in node.get("children", []):
            add_node(child, node_id)

    if isinstance(cdh_tree, list):
        virtual_root = {
            "n_ucs": sum(n.get("n_ucs", 0) for n in cdh_tree),
            "children": cdh_tree,
        }
        add_node(virtual_root)
    else:
        add_node(cdh_tree)

    # Compute depths
    depths = {}

    def compute_depth(node_id, depth=0):
        depths[node_id] = depth
        for child in G.successors(node_id):
            compute_depth(child, depth + 1)

    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    for r in roots:
        compute_depth(r, 0)

    # Find max depth for orientation
    max_depth = max(depths.values()) if depths else 0

    # Assign x positions via inorder traversal
    leaf_positions = {}
    leaf_counter = 0

    def assign_x(node_id):
        nonlocal leaf_counter
        children = list(G.successors(node_id))
        if not children:
            leaf_positions[node_id] = leaf_counter
            leaf_counter += 1
            return leaf_positions[node_id]
        else:
            child_xs = [assign_x(child) for child in children]
            x = sum(child_xs) / len(child_xs)
            leaf_positions[node_id] = x
            return x

    for r in roots:
        assign_x(r)

    if not leaf_positions:
        return None

    # Build edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = leaf_positions[u], depths[u]
        x1, y1 = leaf_positions[v], depths[v]
        if root_at_top:
            y0 = max_depth - y0
            y1 = max_depth - y1
        else:
            y0 = -y0
            y1 = -y1
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color=T["text_low"]),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Nodes
    node_x, node_y, node_text, node_color = [], [], [], []
    for node_id, attrs in G.nodes(data=True):
        x = leaf_positions[node_id]
        y = depths[node_id]
        if root_at_top:
            y = max_depth - y
        else:
            y = -y
        node_x.append(x)
        node_y.append(y)

        n = attrs["n_ucs"]  # total UCs in this node
        label = attrs["label"]  # leaf cluster label (or -1)

        # Determine the text to display
        if stable_counts and label != -1 and G.out_degree(node_id) == 0:
            # Leaf with stable data: use stable count
            n_stable = stable_counts.get(label, n)
            text = f"Clase {label}\n{n_stable} UCs"
        else:
            # Internal node or leaf without stable data: show total + percentage
            pct = (n / total_ucs) * 100 if total_ucs else 0
            text = f"{pct:.1f}%"
            if label != -1:
                text = f"Clase {label}\n" + text

        node_text.append(text)

        # Color assignment (unchanged)
        if G.out_degree(node_id) == 0:
            col = class_colors.get(label, T["accent"]) if label != -1 else T["text_mid"]
        else:
            col = T["text_mid"]
        node_color.append(col)

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=12, color=node_color, line=dict(width=1, color=T["bg_page"])
            ),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    # Adjust y-axis range to have some padding
    y_min = min(node_y) - 0.5 if node_y else -0.5
    y_max = max(node_y) + 0.5 if node_y else 0.5

    fig.update_layout(
        height=400,
        margin=dict(l=8, r=8, t=36, b=40),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[min(node_x) - 0.5, max(node_x) + 0.5] if node_x else [-0.5, 0.5],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[y_min, y_max]
        ),
        title=dict(
            text="Árbol de decisión CDH", font=dict(size=9, color=T["text_low"]), x=0.5
        ),
        **_plot_defaults(),
    )
    return fig


def make_shap_per_class_importance(_dm_key: bool):
    """Grouped bar: mean |SHAP| per feature, one bar per class."""
    shap_pcm = shap_data.get("shap_per_class_mean_abs", [])
    feature_names = shap_data.get("feature_names", [])
    if not shap_pcm or not feature_names:
        return None
    fig = go.Figure()
    for entry in shap_pcm:
        c = entry["class"]
        col = class_colors.get(int(c), "#AAA")
        feats = entry["features"]
        fig.add_trace(
            go.Bar(
                name=f"Clase {c}",
                x=[f["mean_abs_shap"] for f in feats],
                y=[f["feature"] for f in feats],
                orientation="h",
                marker=dict(color=col, opacity=0.85),
                hovertemplate=f"Clase {c} · %{{y}}: %{{x:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        height=max(280, len(feature_names) * 40 + 80),
        margin=dict(l=8, r=8, t=36, b=40),
        xaxis=dict(
            title="Mean |SHAP|",
            showgrid=True,
            gridcolor=T["border"],
            tickfont=dict(size=9),
        ),
        yaxis=dict(autorange="reversed", automargin=True, tickfont=dict(size=9)),
        legend=dict(
            orientation="h",
            y=1.04,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        title=dict(
            text="Importancia SHAP por clase · |SHAP| medio por variable",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def make_global_cah_dendrogram(_dm_key: bool):
    from scipy.cluster.hierarchy import dendrogram as _dendrogram

    if not cah_terminos_global:
        return None
    Z_raw = cah_terminos_global.get("Z", [])
    leaf_labels = cah_terminos_global.get("labels", [])
    if not Z_raw or not leaf_labels or len(leaf_labels) < 2:
        return None
    Z = np.array(Z_raw, dtype=float)
    try:
        ddata = _dendrogram(
            Z, labels=leaf_labels, no_plot=True, orientation="left", color_threshold=0
        )
    except Exception:
        return None
    n = len(ddata["ivl"])
    fig = go.Figure()
    for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
        fig.add_trace(
            go.Scatter(
                x=ys,
                y=xs,
                mode="lines",
                line=dict(color=T["accent"], width=1),
                hoverinfo="none",
                showlegend=False,
            )
        )
    leaf_positions = {}
    for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
        for x, y in zip(xs, ys):
            if y == 0.0:
                leaf_positions[x] = x
    for label, y in zip(ddata["ivl"], sorted(leaf_positions.keys())):
        col = T["text_mid"]
        if not terminos_df.empty:
            rows_t = terminos_df[terminos_df["termino"] == label]
            if not rows_t.empty:
                best = int(rows_t.loc[rows_t["phi"].abs().idxmax(), "cluster"])
                col = class_colors.get(best, T["text_mid"])
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[y],
                mode="markers+text",
                text=[label],
                textposition="middle right",
                textfont=dict(size=7, color=col),
                marker=dict(size=3, color=col),
                showlegend=False,
                hovertemplate=f"<b>{label}</b><extra></extra>",
            )
        )
    fig.update_layout(
        height=max(400, n * 13 + 60),
        margin=dict(l=8, r=120, t=32, b=8),
        xaxis=dict(
            title="Distancia Ward",
            showgrid=True,
            gridcolor=T["border"],
            autorange="reversed",
            tickfont=dict(size=8),
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        title=dict(
            text=f"CAH Global · {n} términos · color = clase dominante",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def metadata_quality_html(
    _selected_classes_key: tuple, _dm_key: bool, show_residuals: bool = False
) -> str:
    if show_residuals:
        return heatmap_html(_selected_classes_key, _dm_key)
    meta_analysis = data.get("metadata_analysis", [])
    if not meta_analysis:
        return "<div style='padding:12px;color:var(--text-low)'>No hay análisis de metadatos (metadata_analysis vacío).</div>"
    df_meta = pd.DataFrame(meta_analysis)
    if df_meta.empty:
        return "<div style='padding:12px;color:var(--text-low)'>Sin datos.</div>"
    df_meta = df_meta.sort_values("p_valor")
    rows_html = ""
    for _, row in df_meta.iterrows():
        var = str(row.get("variable", ""))
        tipo = str(row.get("tipo", ""))
        p = float(row.get("p_valor", 1.0) or 1.0)
        sig = bool(row.get("significativo", False))
        if tipo == "categorica":
            effect = float(row.get("cramer_v", 0) or 0)
            effect = 0 if pd.isna(effect) else float(effect)
            label = "V Cramér"
            color = "#5BA8DC"
        else:
            effect = float(row.get("eta_squared", 0) or 0)
            effect = 0 if pd.isna(effect) else float(effect)
            label = "η²"
            color = "#E8A838"
        bar_w = int(min(effect * 200, 100))
        sig_icon = (
            f'<span style="color:#5DC88A;font-size:10px">●</span>'
            if sig
            else f'<span style="color:var(--text-dim);font-size:10px">○</span>'
        )
        rows_html += f"""
        <tr>
          <td style="padding:6px 10px;color:var(--text-mid)">{sh(var)}</td>
          <td style="padding:6px 10px;color:var(--text-dim);font-size:9px">{sh(tipo)}</td>
          <td style="padding:6px 14px">
            <div style="display:flex;align-items:center;gap:7px">
              <div style="width:80px;height:5px;background:var(--bg-hover);border-radius:3px;overflow:hidden">
                <div style="width:{bar_w}%;height:100%;background:{color};border-radius:3px"></div>
              </div>
              <span style="font-family:var(--font-mono);font-size:10px;color:{color};font-weight:500">{effect:.3f}</span>
              <span style="font-family:var(--font-mono);font-size:8px;color:var(--text-dim)">{label}</span>
            </div>
          </td>
          <td style="padding:6px 10px;font-family:var(--font-mono);font-size:10px;color:var(--text-low)">p={p:.4f}</td>
          <td style="padding:6px 10px;text-align:center">{sig_icon}</td>
        </tr>"""
    return f"""
    <table class="aux-table">
      <thead><tr>
        <th>Variable</th><th>Tipo</th><th>Tamaño del efecto</th><th>p-valor</th><th>Sig</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""


# @st.cache_data(show_spinner=False)
# def compute_condensed_tree_plot_data(tree):
#     """Traverse CDH tree and produce bar_centers, bar_bottoms, bar_tops, bar_widths."""
#     if not tree:
#         return None
#     # If tree is a list, treat it as children of an implicit root
#     if isinstance(tree, list):
#         total = sum(node.get('n_ucs', 0) for node in tree)
#         root = {"n_ucs": total, "children": tree}
#     else:
#         root = tree

#     bar_centers, bar_bottoms, bar_tops, bar_widths = [], [], [], []

#     def _traverse(node, depth=0):
#         if not node:
#             return
#         n = node.get('n_ucs', 0)
#         if n > 0:
#             bar_centers.append(depth)
#             bar_bottoms.append(0)
#             bar_tops.append(float(n) / max(root.get('n_ucs', 1), 1))
#             bar_widths.append(n)
#         for child in node.get('children', []):
#             _traverse(child, depth + 1)

#     _traverse(root)
#     if bar_centers:
#         return {
#             "bar_centers": bar_centers,
#             "bar_bottoms": bar_bottoms,
#             "bar_tops": bar_tops,
#             "bar_widths": bar_widths,
#         }
#     return None

# def make_hdbscan_condensed_tree(_dm_key: bool):
#     plot_data = data.get("condensed_tree_plot_data")
#     if not plot_data and cdh_tree:
#         plot_data = compute_condensed_tree_plot_data(cdh_tree)
#         if plot_data:
#             data["condensed_tree_plot_data"] = plot_data
#     if not plot_data:
#         return None
#     cx = plot_data.get("bar_centers") or plot_data.get("x")
#     y0 = plot_data.get("bar_bottoms") or plot_data.get("y_bottom")
#     yt = plot_data.get("bar_tops")    or plot_data.get("y_top")
#     w  = plot_data.get("bar_widths")  or plot_data.get("width")
#     if not cx or not y0 or not yt:
#         return None
#     n     = len(cx)
#     w_arr = np.array(w, dtype=float) if w else np.ones(n)
#     wmin, wmax = w_arr.min(), w_arr.max()
#     def gc(val):
#         nr = (val-wmin)/(wmax-wmin) if wmax>wmin else 0
#         r,g,b = int(nr*255), int(240*(1-nr)), int(255-105*nr)
#         return f"rgba({r},{g},{b},0.8)", f"rgb({r},{g},{b})"
#     fig = go.Figure()
#     for i in range(n):
#         ys,ye,hw = y0[i],y0[i]+yt[i],w_arr[i]/2
#         fr,lr = gc(w_arr[i])
#         fig.add_trace(go.Scatter(
#             x=[cx[i]-hw,cx[i]-hw,cx[i]+hw,cx[i]+hw,cx[i]-hw],
#             y=[ys,ye,ye,ys,ys], mode="lines", fill="toself",
#             fillcolor=fr, line=dict(color=lr,width=0.5),
#             customdata=[[w_arr[i],yt[i]]]*5,
#             hovertemplate="Puntos: %{customdata[0]:.0f}<br>Δλ: %{customdata[1]:.3f}<extra></extra>",
#             showlegend=False,
#         ))
#     ye_arr = np.round(np.array(y0)+np.array(yt),5)
#     ys_arr = np.round(np.array(y0),5)
#     for sy in set(ys_arr[ys_arr>0]):
#         xs = [cx[i] for i in range(n) if ys_arr[i]==sy or ye_arr[i]==sy]
#         if len(xs)>1:
#             fig.add_trace(go.Scatter(x=[min(xs),max(xs)],y=[sy,sy],mode="lines",
#                           line=dict(color=T["text_low"],width=1.5),hoverinfo="skip",showlegend=False))
#     fig.update_layout(height=280,margin=dict(l=8,r=8,t=24,b=8),
#                       xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
#                       yaxis=dict(showgrid=True,gridcolor=T["border"],zeroline=False,autorange="reversed"),
#                       showlegend=False, **_plot_defaults())
#     return fig


def make_donut(_selected_classes_key: tuple, _dm_key: bool):
    selected_classes = list(_selected_classes_key)
    if not selected_classes or not class_sizes:
        return None
    lbl = [f"Clase {c}" for c in selected_classes if c in class_sizes]
    val = [class_sizes[c] for c in selected_classes if c in class_sizes]
    col = [class_colors.get(c, "#AAA") for c in selected_classes if c in class_sizes]
    if not val:
        return None
    fig = go.Figure(
        go.Pie(
            labels=lbl,
            values=val,
            hole=0.62,
            marker=dict(colors=col, line=dict(color=T["bg_page"], width=2)),
            textinfo="percent",
            textfont=dict(size=9),
            hovertemplate="<b>%{label}</b><br>UCEs: %{value}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=8, b=8),
        showlegend=False,
        **_plot_defaults(),
    )
    return fig


def make_word_bars(_selected_classes_key: tuple, _dm_key: bool):
    selected_classes = list(_selected_classes_key)
    if not selected_classes or not words_per_cluster:
        return None
    rows = []
    for c in selected_classes:
        if c not in class_sizes:
            continue
        count = words_per_cluster.get(c, 0)  # int keys after normalisation
        rows.append({"clase": str(c), "palabras": count})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    cmap = {str(c): class_colors.get(c, "#AAA") for c in selected_classes}
    fig = px.bar(
        df,
        x="palabras",
        y="clase",
        orientation="h",
        text="palabras",
        color="clase",
        color_discrete_map=cmap,
        category_orders={"clase": [str(c) for c in selected_classes]},
    )
    fig.update_yaxes(automargin=True)
    fig.update_layout(
        height=240,
        margin=dict(l=8, r=40, t=8, b=8),
        xaxis_title="Palabras analizadas",
        yaxis_title="",
        showlegend=False,
        **_plot_defaults(),
    )
    fig.update_traces(textposition="outside", textfont=dict(size=10))
    return fig


def make_mca_scatter(_selected_classes_key: tuple, _dm_key: bool):
    selected_classes = list(_selected_classes_key)
    if not multivariate:
        return None
    row_coords = multivariate.get("row_coords", [])
    col_coords = multivariate.get("col_coords", [])
    cluster_lbls = multivariate.get("cluster_labels", [])
    expl = multivariate.get("explained_inertia", [0, 0])
    if not row_coords or not col_coords:
        return None
    row_arr = pd.DataFrame(row_coords)
    col_arr = pd.DataFrame(col_coords)
    if row_arr.shape[1] < 2 or col_arr.shape[1] < 2:
        return None
    ax1_pct = round(expl[0] * 100, 1) if expl else 0
    ax2_pct = round(expl[1] * 100, 1) if len(expl) > 1 else 0
    fig = go.Figure()
    fig.add_hline(y=0, line_width=0.8, line_color=T["border2"])
    fig.add_vline(x=0, line_width=0.8, line_color=T["border2"])
    doc_ids = multivariate.get("doc_ids", [])
    phi_by_pos = {}
    for i, uid in enumerate(doc_ids):
        u = uce_map.get(str(uid), uce_map.get(uid, {}))
        phi_coefs = (u.get("phi_coefficients", {}) or {}) if isinstance(u, dict) else {}
        top = sorted(phi_coefs.items(), key=lambda x: x[1], reverse=True)[:4]
        phi_by_pos[i] = "<br>".join(f"{t}: {v:+.2f}" for t, v in top)

    for c in sorted(set(int(lbl) for lbl in cluster_lbls if int(lbl) >= 0)):
        if c not in selected_classes:
            continue
        idxs = [i for i, lbl in enumerate(cluster_lbls) if int(lbl) == c]
        xs = [row_arr.iloc[i, 0] for i in idxs]
        ys = [row_arr.iloc[i, 1] for i in idxs]
        phi_texts = [phi_by_pos.get(i, "") for i in idxs]
        col = class_colors.get(c, "#AAA")
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=f"UC Clase {c}",
                marker=dict(
                    size=4,
                    color=col,
                    opacity=0.4,
                    line=dict(width=0, color=T["bg_page"]),
                ),
                customdata=phi_texts,
                hovertemplate=(
                    f"Clase {c}<br>MCA1: %{{x:.3f}}<br>MCA2: %{{y:.3f}}"
                    "<br><span style='color:var(--text-low)'>%{customdata}</span><extra></extra>"
                ),
                showlegend=True,
            )
        )

    n_vars = len(col_arr)
    var_labels = multivariate.get("column_labels", [f"Var {i}" for i in range(n_vars)])
    if len(var_labels) != n_vars:
        var_labels = [f"Var {i}" for i in range(n_vars)]
    fig.add_trace(
        go.Scatter(
            x=col_arr.iloc[:, 0].tolist(),
            y=col_arr.iloc[:, 1].tolist(),
            mode="markers+text",
            text=var_labels,
            textposition="top center",
            textfont=dict(size=8, color=T["text_mid"]),
            name="Variables",
            marker=dict(
                size=10,
                color=T["accent"],
                symbol="diamond",
                opacity=0.9,
                line=dict(width=1.5, color=T["bg_page"]),
            ),
            hovertemplate="<b>%{text}</b><br>MCA1: %{x:.3f}<br>MCA2: %{y:.3f}<extra></extra>",
            showlegend=True,
        )
    )
    fig.update_layout(
        height=460,
        margin=dict(l=8, r=8, t=36, b=50),
        xaxis=dict(
            title=f"MCA1 ({ax1_pct}%)",
            showgrid=True,
            gridcolor=T["border"],
            zeroline=False,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title=f"MCA2 ({ax2_pct}%)",
            showgrid=True,
            gridcolor=T["border"],
            zeroline=False,
            tickfont=dict(size=9),
        ),
        legend=dict(
            orientation="h",
            y=-0.1,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        title=dict(
            text=f"MCA · variables + UCs  MCA1={ax1_pct}%  ·  MCA2={ax2_pct}%",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def make_scree_plot(_dm_key: bool):
    if not singular_values or len(singular_values) < 2:
        return None
    sv = [float(v) for v in singular_values]
    inertias = [
        round(v**2 / total_inertia * 100, 1) if total_inertia > 0 else 0.0 for v in sv
    ]
    cumulative = []
    acc = 0.0
    for v in inertias:
        acc += v
        cumulative.append(round(acc, 1))
    factor_labels = [f"F{i + 1}" for i in range(len(sv))]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=factor_labels,
            y=inertias,
            name="Inercia por factor",
            marker_color=[
                T["accent"] if i < 2 else T["border2"] for i in range(len(sv))
            ],
            hovertemplate="<b>%{x}</b><br>Inercia: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=factor_labels,
            y=cumulative,
            name="Inercia acumulada",
            mode="lines+markers",
            line=dict(color=T["text_mid"], width=1.5, dash="dot"),
            marker=dict(size=5, color=T["text_mid"]),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Acumulado: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=8, r=8, t=32, b=32),
        xaxis=dict(tickfont=dict(size=9), showgrid=False),
        yaxis=dict(
            title="% inercia",
            tickfont=dict(size=9),
            showgrid=True,
            gridcolor=T["border"],
        ),
        yaxis2=dict(
            title="% acumulado",
            tickfont=dict(size=9),
            overlaying="y",
            side="right",
            range=[0, 110],
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            y=1.06,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        title=dict(
            text=f"Scree plot · inercia total = {total_inertia:.3f}",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def make_afc_biplot(
    _selected_classes_key: tuple, view_mode: str, _dm_key: bool, show_ucs: bool = False
):
    selected_classes = list(_selected_classes_key)
    if not proyeccion:
        return None
    expl = proyeccion.get("explained_inertia", [0, 0])
    ax1_pct = (expl[0] * 100) if expl else 0
    ax2_pct = (expl[1] * 100) if len(expl) > 1 else 0
    class_ids_afc = proyeccion.get("class_ids", [])
    print("[make_afc_biplot] proyeccion keys:", proyeccion.keys())
    print("[make_afc_biplot] view_mode:", view_mode)
    voc_list = proyeccion.get("voc", vocabulario)
    voc_list = proyeccion.get("voc") or vocabulario
    top_n_terms = min(len(voc_list), 300)  # or whatever cap you want
    print(f"[make_afc_biplot] len(voc_list) = {len(voc_list)}")

    if view_mode == "contributions":
        col_contrib = proyeccion.get("col_contrib", [])
        if not col_contrib or not voc_list:
            return None
        arr = np.array(col_contrib)
        if arr.ndim < 2 or arr.shape[1] < 2:
            return None
        f1 = arr[:, 0]
        f2 = arr[:, 1]
        f1_pct = f1 / (f1.sum() + 1e-12) * 100
        f2_pct = f2 / (f2.sum() + 1e-12) * 100
        top_idx = np.argsort(f1_pct + f2_pct)[::-1][:top_n_terms]
        top_terms = [voc_list[i] for i in top_idx if i < len(voc_list)]
        f1_vals = [f1_pct[i] for i in top_idx if i < len(voc_list)]
        f2_vals = [f2_pct[i] for i in top_idx if i < len(voc_list)]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name=f"F1 ({ax1_pct:.1f}%)",
                x=f1_vals,
                y=top_terms,
                orientation="h",
                marker_color="#5BA8DC",
                opacity=0.85,
                hovertemplate="<b>%{y}</b><br>Contribución F1: %{x:.2f}%<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                name=f"F2 ({ax2_pct:.1f}%)",
                x=f2_vals,
                y=top_terms,
                orientation="h",
                marker_color="#E8A838",
                opacity=0.85,
                hovertemplate="<b>%{y}</b><br>Contribución F2: %{x:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            barmode="group",
            height=max(320, len(top_terms) * 18 + 80),
            margin=dict(l=8, r=24, t=36, b=40),
            yaxis=dict(autorange="reversed", tickfont=dict(size=8), automargin=True),
            xaxis=dict(
                title="% inercia del factor", showgrid=True, gridcolor=T["border"]
            ),
            legend=dict(
                orientation="h",
                y=1.04,
                x=0.5,
                xanchor="center",
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=9),
            ),
            title=dict(
                text=f"En contributions · F1={ax1_pct:.1f}% · F2={ax2_pct:.1f}%",
                font=dict(size=9, color=T["text_low"]),
                x=0.5,
            ),
            **_plot_defaults(),
        )
        return fig

    if view_mode == "coordonnees":
        row_key, col_key = "row_coords", "col_coords"
        x_title = f"F1 — coordenada χ²  ({ax1_pct:.1f}%)"
        y_title = f"F2 — coordenada χ²  ({ax2_pct:.1f}%)"
        show_circle = False
        view_title = f"En coordonnées · F1={ax1_pct:.1f}% · F2={ax2_pct:.1f}%"
    else:
        row_key, col_key = "row_std", "col_std"
        x_title = f"F1 — correlación  ({ax1_pct:.1f}%)"
        y_title = f"F2 — correlación  ({ax2_pct:.1f}%)"
        show_circle = True
        view_title = f"En corrélations · F1={ax1_pct:.1f}% · F2={ax2_pct:.1f}%"

    row_data = proyeccion.get(row_key, [])
    col_data = proyeccion.get(col_key, [])
    if not row_data or not col_data:
        return None
    row_arr = np.array(row_data)
    col_arr = np.array(col_data)
    if row_arr.ndim < 2 or col_arr.ndim < 2:
        return None
    if row_arr.shape[1] < 2 or col_arr.shape[1] < 2:
        return None
    print(f"[make_afc_biplot] len(col_coords) = {len(col_data)}")
    print(f"[make_afc_biplot] col_coords sample (first 3): {col_data[:3]}")
    fig = go.Figure()
    if show_circle:
        theta = np.linspace(0, 2 * np.pi, 120)
        fig.add_trace(
            go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode="lines",
                line=dict(color=T["border2"], width=1),
                hoverinfo="none",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=0.5 * np.cos(theta),
                y=0.5 * np.sin(theta),
                mode="lines",
                line=dict(color=T["border"], width=0.5, dash="dot"),
                hoverinfo="none",
                showlegend=False,
            )
        )
    fig.add_hline(y=0, line_width=0.8, line_color=T["border2"])
    fig.add_vline(x=0, line_width=0.8, line_color=T["border2"])

    n_avail = min(len(col_arr), len(voc_list))
    quality = col_arr[:n_avail, 0] ** 2 + col_arr[:n_avail, 1] ** 2
    top_idx = (
        np.argsort(quality)[::-1][:top_n_terms]
        if n_avail > top_n_terms
        else np.arange(n_avail)
    )
    term_x = col_arr[top_idx, 0]
    term_y = col_arr[top_idx, 1]
    t_labels = [voc_list[i] for i in top_idx]
    t_colors = []
    for t in t_labels:
        if not terminos_df.empty:
            rows_t = terminos_df[terminos_df["termino"] == t]
            if not rows_t.empty:
                best = int(rows_t.loc[rows_t["phi"].abs().idxmax(), "cluster"])
                t_colors.append(class_colors.get(best, T["text_low"]))
                continue
        t_colors.append(T["text_low"])

    fig.add_trace(
        go.Scatter(
            x=term_x,
            y=term_y,
            mode="markers+text",
            text=t_labels,
            textposition="top center",
            textfont=dict(size=7, color=T["text_mid"]),
            marker=dict(
                size=5,
                color=t_colors,
                opacity=0.75,
                line=dict(width=0.5, color=T["bg_page"]),
            ),
            name="Términos",
            hovertemplate="<b>%{text}</b><br>F1: %{x:.3f}<br>F2: %{y:.3f}<extra></extra>",
            showlegend=True,
        )
    )
    for i, raw_cid in enumerate(class_ids_afc):
        cid = int(raw_cid)
        if cid not in selected_classes or i >= len(row_arr):
            continue
        cx, cy = float(row_arr[i, 0]), float(row_arr[i, 1])
        col = class_colors.get(cid, "#AAA")
        fig.add_trace(
            go.Scatter(
                x=[0, cx],
                y=[0, cy],
                mode="lines",
                line=dict(color=col, width=1.2, dash="dash"),
                hoverinfo="none",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers+text",
                text=[f"Clase {cid}"],
                textposition="top center",
                textfont=dict(size=11, color=col, family="IBM Plex Mono, monospace"),
                marker=dict(
                    size=20,
                    color=col,
                    opacity=0.95,
                    symbol="diamond",
                    line=dict(width=2, color=T["bg_page"]),
                ),
                name=f"Clase {cid}",
                hovertemplate=f"<b>Clase {cid}</b><br>F1: {cx:.3f}<br>F2: {cy:.3f}<extra></extra>",
                showlegend=True,
            )
        )
    if show_ucs:
        for uc in ucs:
            cid = uc.get("cluster_label_double")
            if cid is None or cid not in selected_classes:
                continue
            coord = (uc.get("coordinates") or {}).get("afc_row")
            if not coord or len(coord) < 2:
                continue
            col = class_colors.get(cid, T["text_low"])
            fig.add_trace(
                go.Scatter(
                    x=[coord[0]],
                    y=[coord[1]],
                    mode="markers",
                    marker=dict(size=3, color=col, opacity=0.2),
                    hovertemplate=f"Clase {cid}  F1:{coord[0]:.3f} F2:{coord[1]:.3f}<extra></extra>",
                    showlegend=False,
                )
            )
    if show_circle:
        all_vals = np.concatenate(
            [np.abs(col_arr[:n_avail, :2]).ravel(), np.abs(row_arr[:, :2]).ravel()]
        )
        mx = float(all_vals.max()) * 1.15 if len(all_vals) else 1.2
        axis_range = [-mx, mx]
        scale_anchor = "x"
    else:
        axis_range = scale_anchor = None
    fig.update_layout(
        height=500,
        margin=dict(l=8, r=8, t=36, b=50),
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor=T["border"],
            zeroline=False,
            range=axis_range,
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor=T["border"],
            zeroline=False,
            range=axis_range,
            scaleanchor=scale_anchor,
        ),
        legend=dict(
            orientation="h",
            y=-0.1,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        title=dict(text=view_title, font=dict(size=9, color=T["text_low"]), x=0.5),
        hoverlabel=dict(
            bgcolor=T["bg_card"],
            bordercolor=T["border2"],
            font=dict(size=11, color=T["text_hi"]),
        ),
        **_plot_defaults(),
    )
    return fig


def make_stability_map(_selected_classes_key: tuple, _dm_key: bool):
    selected_classes = list(_selected_classes_key)
    if not term_stability_dict or terminos_df.empty or not selected_classes:
        return None
    classes = sorted([c for c in selected_classes if c in class_sizes])
    if not classes:
        return None
    rows = []
    for (term, cluster), sf in term_stability_dict.items():
        if int(cluster) not in classes:
            continue
        row_phi = terminos_df[
            (terminos_df["termino"] == term) & (terminos_df["cluster"] == int(cluster))
        ]
        if row_phi.empty:
            continue
        phi_val = float(row_phi["phi"].values[0])
        freq = int(row_phi["frecuencia_global"].values[0])
        rows.append(
            {
                "term": term,
                "cluster": int(cluster),
                "phi": phi_val,
                "stability": sf,
                "freq": freq,
            }
        )
    if not rows:
        return None
    df_stab = pd.DataFrame(rows)
    med_phi = float(df_stab["phi"].median())
    med_stab = float(df_stab["stability"].median())
    fig = go.Figure()
    for c in classes:
        sub = df_stab[df_stab["cluster"] == c]
        if sub.empty:
            continue
        col = class_colors.get(c, "#AAA")
        fig.add_trace(
            go.Scatter(
                x=sub["phi"],
                y=sub["stability"],
                mode="markers",
                name=f"Clase {c}",
                marker=dict(
                    size=(sub["freq"] ** 0.4).clip(4, 16).tolist(),
                    color=col,
                    opacity=0.75,
                    line=dict(width=0.5, color=T["bg_page"]),
                ),
                customdata=list(
                    zip(
                        sub["term"],
                        sub["phi"].round(3),
                        sub["stability"].round(3),
                        sub["freq"],
                    )
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "φ: %{customdata[1]}<br>"
                    "Estabilidad: %{customdata[2]}<br>"
                    "Frec: %{customdata[3]}<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=med_phi, line_width=1, line_dash="dot", line_color=T["border2"])
    fig.add_hline(y=med_stab, line_width=1, line_dash="dot", line_color=T["border2"])
    phi_max = float(df_stab["phi"].max())
    phi_min = float(df_stab["phi"].min())
    for anchor_x, anchor_y, label, xanchor, yanchor in [
        (phi_max, 1.0, "Hallazgo robusto", "right", "top"),
        (phi_max, 0.0, "Hallazgo frágil", "right", "bottom"),
        (phi_min, 1.0, "Estable inespecífico", "left", "top"),
        (phi_min, 0.0, "Ruido", "left", "bottom"),
    ]:
        fig.add_annotation(
            x=anchor_x,
            y=anchor_y,
            text=label,
            showarrow=False,
            font=dict(size=9, color=T["text_dim"], family="IBM Plex Mono"),
            xanchor=xanchor,
            yanchor=yanchor,
        )
    fig.update_layout(
        height=380,
        margin=dict(l=8, r=8, t=36, b=50),
        xaxis=dict(
            title="Coeficiente φ",
            showgrid=True,
            gridcolor=T["border"],
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title="Estabilidad bootstrap (0–1)",
            showgrid=True,
            gridcolor=T["border"],
            tickfont=dict(size=9),
            range=[-0.05, 1.05],
        ),
        legend=dict(
            orientation="h",
            y=-0.12,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        title=dict(
            text="Mapa de estabilidad · tamaño = frecuencia global",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def phi_bars_html(cluster_id: int, _dm_key: bool) -> str:
    if terminos_df.empty:
        return "<div>No hay términos.</div>"
    df_c = terminos_df[terminos_df["cluster"] == cluster_id].copy()
    if df_c.empty:
        return "<div>No hay términos para esta clase.</div>"
    pos = df_c[df_c["phi"] > 0].sort_values("phi", ascending=False).head(20)
    neg = df_c[df_c["phi"] < 0].sort_values("phi", ascending=True).head(20)
    mx_pos = pos["phi"].max() if not pos.empty else 1
    mx_neg = abs(neg["phi"].min()) if not neg.empty else 1
    col = class_colors.get(cluster_id, "#AAA")
    has_stability = bool(term_stability_dict)

    def _stability_dot(term: str, cluster: int) -> str:
        if not has_stability:
            return ""
        sf = term_stability_dict.get((str(term), int(cluster)), None)
        if sf is None:
            return '<span title="Sin datos de estabilidad" style="color:var(--text-dim)">·</span>'
        if sf >= 0.7:
            dot_col, title = "#5DC88A", f"Estable ({sf:.2f})"
        elif sf >= 0.4:
            dot_col, title = "#E8A838", f"Moderado ({sf:.2f})"
        else:
            dot_col, title = "#E86450", f"Frágil ({sf:.2f})"
        return (
            f'<span title="{title}" style="color:{dot_col};font-size:10px;'
            f'margin-left:3px;cursor:default">●</span>'
        )

    def rows_html(df, is_pos):
        if df.empty:
            return (
                f"<div style='padding:8px 0;color:var(--text-dim)'>"
                f"{'Sin presencias' if is_pos else 'Sin ausencias'}.</div>"
            )
        out = ""
        for _, row in df.iterrows():
            mx = mx_pos if is_pos else mx_neg
            w = int(abs(row["phi"]) / mx * 100)
            fill = col if is_pos else "rgba(232,100,80,0.55)"
            vc = "var(--text-mid)" if is_pos else "rgba(232,100,80,0.7)"
            dot = _stability_dot(row["termino"], cluster_id)
            out += f"""<div class="phi-row">
              <span class="phi-word" title="{sh(row["termino"])}">{sh(row["termino"])}{dot}</span>
              <div class="phi-bar-bg"><div class="phi-fill" style="width:{w}%;background:{fill}"></div></div>
              <span class="phi-val" style="color:{vc}">{row["phi"]:.2f}</span>
            </div>"""
        return out

    stability_legend = ""
    if has_stability:
        stability_legend = (
            '<div style="font-family:var(--font-mono);font-size:9px;color:var(--text-dim);'
            'padding:6px 0 0;display:flex;gap:12px">'
            '<span style="color:#5DC88A">● estable</span>'
            '<span style="color:#E8A838">● moderado</span>'
            '<span style="color:#E86450">● frágil</span></div>'
        )

    head_pos = (
        f'<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:.12em;'
        f'text-transform:uppercase;color:{col};margin-bottom:8px">▲ Presencias significativas</div>'
    )
    head_neg = (
        '<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:.12em;'
        'text-transform:uppercase;color:rgba(232,100,80,0.8);margin-bottom:8px">'
        "▼ Ausencias significativas</div>"
    )
    return f"""<div style="display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);margin-top:2px">
      <div style="background:var(--bg-panel);padding:14px 16px">{head_pos}{rows_html(pos, True)}{stability_legend}</div>
      <div style="background:var(--bg-panel);padding:14px 16px">{head_neg}{rows_html(neg, False)}</div>
    </div>"""


def _render_tab_b_highlight(
    text: str, uce_lemmas: list, filter_lemmas: list, search_query: str, target_col: str
) -> str:
    is_searching = bool(search_query or filter_lemmas)
    spans = []
    text_lower = text.lower()
    if is_searching:
        if search_query:
            sq = search_query.lower()
            idx = 0
            while True:
                idx = text_lower.find(sq, idx)
                if idx == -1:
                    break
                spans.append((idx, idx + len(sq)))
                idx += len(sq)
        if filter_lemmas:
            for l in filter_lemmas:
                l_lower = l.lower()
                if (
                    l_lower.replace("_", " ") not in text_lower
                    and l_lower.replace("_", "") not in text_lower
                ):
                    continue
                m_spans = match_term(l, text)
                if m_spans:
                    parts = set(p.lower() for p in l.split("_"))
                    for bs, be in m_spans:
                        chunk = text[bs:be]
                        for m in re.finditer(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]+", chunk):
                            if any(m.group().lower().startswith(p) for p in parts):
                                spans.append((bs + m.start(), bs + m.end()))
    if not spans:
        return sh(text)
    spans.sort(key=lambda x: x[0])
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    out = ""
    cursor = 0
    for s, e in merged:
        out += sh(text[cursor:s])
        out += f'<span style="color:{target_col}; background: {target_col}22; padding: 0 3px; border-radius: 3px;">{sh(text[s:e])}</span>'
        cursor = e
    out += sh(text[cursor:])
    return out


def make_single_doc_trajectory(
    doc_id: str, ordered_ids: dict, uce_lookup: dict, _dm_key: bool
):
    if not ordered_ids:
        return None
    ys, xs, colors, hover = [], [], [], []
    uc_counter = 0
    row = 0
    for uc_idx, uce_ids in ordered_ids.items():
        uc_counter += 1
        uc_str = f"UC {uc_counter}"
        for i, uid in enumerate(uce_ids):
            u = uce_lookup.get(uid, {})
            cid = u.get("cluster_id")
            ys.append(row)
            xs.append(1)
            if cid is not None and cid >= 0:
                colors.append(class_colors.get(cid, T["accent"]))
                hover.append(f"<b>Clase {cid} · {uc_str} · UCE {i + 1}</b>")
            else:
                colors.append("rgba(0,0,0,0)")
                hover.append(f"<b>No clasificado · {uc_str}</b>")
            row += 1
    fig = go.Figure(
        go.Bar(
            y=ys,
            x=xs,
            orientation="h",
            marker=dict(color=colors, line=0),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="stack",
        bargap=0.0,
        height=600,
        margin=dict(l=40, r=8, t=30, b=40),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(
            title="Secuencia Temporal (UCEs)",
            autorange="reversed",
            tickfont=dict(size=9),
        ),
        title=dict(
            text=f"Secuencia Discursiva · Doc {doc_id}",
            font=dict(size=10, color=T["text_low"]),
        ),
        hoverlabel=dict(
            bgcolor=T["bg_card"],
            bordercolor=T["border2"],
            font=dict(size=11, color=T["text_hi"]),
        ),
        **_plot_defaults(),
    )
    return fig


# FIX 8: make_butterfly_chart — NaN guard + empty num_cols guard
def make_butterfly_chart(
    _selected_classes_key: tuple,
    top_n=28,
    sig_only=True,
    _term_filter_key=None,
    _dm_key: bool = True,
):
    selected_classes = list(_selected_classes_key)
    term_filter = list(_term_filter_key) if _term_filter_key else None
    if terminos_df.empty or not selected_classes:
        return None
    df = terminos_df.copy()
    if sig_only and "significativo" in df.columns:
        df = df[df["significativo"] == True]
    classes = sorted([c for c in selected_classes if c in class_sizes])
    if not classes:
        return None
    pivot = (
        df[df["cluster"].isin(classes)]
        .pivot_table(index="termino", columns="cluster", values="phi", aggfunc="first")
        .reindex(columns=classes)
        .fillna(0.0)
    )
    if pivot.empty:
        return None
    if term_filter:
        pivot = pivot[pivot.index.isin(set(term_filter))]
        if pivot.empty:
            return None
        terms = pivot.index.tolist()
    else:
        pivot = pivot[(pivot.abs() > 0).sum(axis=1) >= 1]
        if pivot.empty:
            return None
        pivot["_s"] = pivot.max(axis=1) - pivot.min(axis=1)
        pivot = pivot.sort_values("_s", ascending=True).tail(top_n).drop(columns=["_s"])
        terms = pivot.index.tolist()
    if not terms:
        return None
    fh = max(350, len(terms) * 16 + 100)
    fig = go.Figure()
    for c in classes:
        phi_vals = pivot[c].tolist()
        col = class_colors.get(c, "#AAA")
        txt = [f"{v:+.2f}" if abs(v) > 0.05 else "" for v in phi_vals]
        cdata = []
        for term in terms:
            row = terminos_df[
                (terminos_df["termino"] == term) & (terminos_df["cluster"] == c)
            ]
            if not row.empty:
                fg = int(row["frecuencia_global"].values[0])
                fc = int(row["frecuencia_cluster"].values[0])
                pa = (
                    float(row["p_adj"].values[0])
                    if "p_adj" in row.columns
                    else float("nan")
                )
                phi = float(row["phi"].values[0])
                cdata.append(
                    f"<b>{term}</b> · Clase {c}<br>φ={phi:+.3f}<br>frec global:{fg} · en clase:{fc}<br>p_adj:{pa:.4f}"
                )
            else:
                cdata.append(f"<b>{term}</b> · Clase {c}<br>φ=0 (ausente)")
        fig.add_trace(
            go.Bar(
                name=f"Clase {c}",
                x=phi_vals,
                y=terms,
                orientation="h",
                marker=dict(color=col, opacity=0.88, line=dict(color=col, width=0.4)),
                text=txt,
                textposition="inside",
                textfont=dict(size=8, color="rgba(255,255,255,0.6)"),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=cdata,
            )
        )
    fig.add_vline(x=0, line_width=1.5, line_color=T["border2"])
    # FIX: guard against all-NaN or empty numeric columns
    num_cols = pivot.select_dtypes(include="number")
    if num_cols.empty:
        pmin, pmax = -0.05, 0.05
    else:
        pmin = num_cols.min().min()
        pmax = num_cols.max().max()
    if pd.isna(pmin):
        pmin = -0.05
    if pd.isna(pmax):
        pmax = 0.05
    fig.update_layout(
        barmode="group",
        height=fh,
        margin=dict(l=8, r=24, t=14, b=44),
        legend=dict(
            orientation="h",
            y=1.025,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        xaxis=dict(
            title=dict(
                text="← ausencia (φ < 0)     φ = 0     presencia (φ > 0) →",
                font=dict(size=9, color=T["text_low"]),
            ),
            showgrid=True,
            gridcolor=T["border"],
            zeroline=False,
            tickfont=dict(size=9),
            range=[min(-0.05, pmin * 1.15), max(0.05, pmax * 1.15)],
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=9.5), automargin=True),
        hoverlabel=dict(
            bgcolor=T["bg_card"],
            bordercolor=T["border2"],
            font=dict(size=11, color=T["text_hi"]),
        ),
        **_plot_defaults(),
    )
    return fig


def make_cross_class_radar_lemmas(
    chosen_lemmas: list, selected_classes: list, _dm_key: bool
):
    if terminos_df.empty or not chosen_lemmas or not selected_classes:
        return None
    classes = sorted([c for c in selected_classes if c in class_sizes])
    if not classes:
        return None
    fig = go.Figure()
    for c in classes:
        vals = []
        for lemma in chosen_lemmas:
            entry = lemma_map.get(lemma, {})
            stems = list(entry.get("stems", [])) or [lemma]
            # Also try the lemma itself as a direct term match (for unstemmed vocabularies)
            rows = terminos_df[
                (terminos_df["cluster"] == c)
                & (terminos_df["termino"].isin(stems + [lemma]))
            ]
            cum_phi = float(rows["phi"].sum()) if not rows.empty else 0.0
            vals.append(cum_phi)
        vc = vals + [vals[0]]
        tc = chosen_lemmas + [chosen_lemmas[0]]
        col = class_colors.get(c, "#AAA")
        fig.add_trace(
            go.Scatterpolar(
                r=vc,
                theta=tc,
                fill="toself",
                name=f"Clase {c}",
                line_color=col,
                fillcolor=hex_to_rgba(col, 0.15),
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor=T["bg_card"],
            radialaxis=dict(visible=True, gridcolor=T["border2"], color=T["text_low"]),
            angularaxis=dict(gridcolor=T["border2"], color=T["text_mid"]),
        ),
        height=340,
        margin=dict(l=28, r=28, t=18, b=64),
        legend=dict(
            orientation="h",
            y=-0.22,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        **_plot_defaults(),
    )
    return fig


def make_term_freq_bar_lemmas(
    chosen_lemmas: list, selected_classes: list, _dm_key: bool
):
    if terminos_df.empty or not selected_classes or not chosen_lemmas:
        return None
    classes = sorted([c for c in selected_classes if c in class_sizes])
    if not classes:
        return None
    fig = go.Figure()
    for c in classes:
        vals = []
        for lemma in chosen_lemmas:
            stems = list(lemma_map.get(lemma, {}).get("stems", [lemma]))
            rows = terminos_df[
                (terminos_df["cluster"] == c) & (terminos_df["termino"].isin(stems))
            ]
            cum_phi = float(rows["phi"].clip(lower=0).sum()) if not rows.empty else 0.0
            vals.append(cum_phi)
        fig.add_trace(
            go.Bar(
                name=f"Clase {c}",
                x=vals,
                y=chosen_lemmas,
                orientation="h",
                marker_color=class_colors.get(c, "#AAA"),
                hovertemplate=f"Clase {c}: Σφ=%{{x:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=340,
        margin=dict(l=8, r=16, t=8, b=100),
        legend=dict(
            orientation="h",
            y=-0.32,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        xaxis=dict(
            title="Σφ acumulado por clase", showgrid=True, gridcolor=T["border"]
        ),
        yaxis=dict(autorange="reversed"),
        **_plot_defaults(),
    )
    return fig


def make_lemma_sunburst(chosen_lemmas: list, forma_index: dict, selected_classes: list):
    if not chosen_lemmas or not forma_index:
        return None
    ids = ["ROOT"]
    labels = [" "]
    parents = [""]
    values = [0]
    stem_to_lemmas = defaultdict(list)
    for lemma in chosen_lemmas:
        stems = lemma_map.get(lemma, {}).get("stems", [lemma])
        for stem in stems:
            stem_to_lemmas[stem].append(lemma)
    for stem, lemmas_for_stem in stem_to_lemmas.items():
        stem_id = f"S_{stem}"
        stem_total = 0
        for lemma in lemmas_for_stem:
            lemma_id = f"L_{stem}_{lemma}"
            lemma_total = 0
            forma_counts = defaultdict(int)
            for c in selected_classes:
                c_str = str(c)
                if (
                    c_str in forma_index
                    and stem in forma_index[c_str]
                    and lemma in forma_index[c_str][stem]
                ):
                    for f_exact, freq in forma_index[c_str][stem][lemma].items():
                        forma_counts[f_exact] += freq
            if not forma_counts:
                continue
            for forma, freq in forma_counts.items():
                forma_id = f"F_{stem}_{lemma}_{forma}"
                ids.append(forma_id)
                labels.append(forma)
                parents.append(lemma_id)
                values.append(freq)
                lemma_total += freq
            if lemma_total > 0:
                ids.append(lemma_id)
                labels.append(lemma)
                parents.append(stem_id)
                values.append(lemma_total)
                stem_total += lemma_total
        if stem_total > 0:
            ids.append(stem_id)
            labels.append(stem)
            parents.append("ROOT")
            values.append(stem_total)
    values[0] = sum(v for p, v in zip(parents, values) if p == "ROOT")
    if values[0] == 0:
        return None
    df_sun = pd.DataFrame(
        {"id": ids, "label": labels, "parent": parents, "value": values}
    )
    fig = px.sunburst(
        df_sun,
        ids="id",
        parents="parent",
        values="value",
        names="label",
        branchvalues="total",
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Apariciones: %{value}<extra></extra>",
        insidetextorientation="radial",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=450, **_plot_defaults()
    )
    return fig


def heatmap_html(_selected_classes_key: tuple, _dm_key: bool) -> str:
    selected_classes = list(_selected_classes_key)
    if not metadata_residuals or not selected_classes:
        return "<div style='padding:12px;color:var(--text-low)'>No hay residuos de metadatos.</div>"
    clusters = sorted([c for c in selected_classes if c in class_sizes])
    hdr = "".join(
        f'<th style="text-align:center;padding:5px 10px">Clase {c}</th>'
        for c in clusters
    )
    rows_html = ""
    for var, var_data in metadata_residuals.items():
        rd = var_data.get("residuals", {})
        cells = ""
        for c in clusters:
            best_val = None
            best_cat = None
            for cat_label, class_dict in rd.items():
                v = class_dict.get(str(c))
                if v is not None and (best_val is None or abs(v) > abs(best_val)):
                    best_val = v
                    best_cat = cat_label
            if best_val is None:
                cells += '<td style="text-align:center;color:var(--text-dim)">—</td>'
            else:
                col = "#5BA8DC" if best_val > 0 else "#E86450"
                rgb = "91,168,220" if best_val > 0 else "232,100,80"
                op = min(0.08 + abs(best_val) / 4.0, 0.55)
                cells += (
                    f'<td style="text-align:center;background:rgba({rgb},{op:.2f})" '
                    f'title="{sh(str(best_cat))} · r={best_val:.2f}">'
                    f'<span style="color:{col};font-weight:500">{best_val:.2f}</span>'
                    f'<div style="font-size:8px;color:{col};opacity:0.7">{sh(str(best_cat))}</div></td>'
                )
        rows_html += f'<tr><td style="padding:5px 10px;color:var(--text-mid)">{sh(var)}</td>{cells}</tr>'
    return (
        f'<table class="aux-table"><thead><tr><th>Variable</th>{hdr}</tr></thead>'
        f"<tbody>{rows_html}</tbody></table>"
    )


# FIX 9: gramcat_html — uses normalised int keys for pos_by_cluster
def gramcat_html(_selected_classes_key: tuple, _dm_key: bool) -> str:
    selected_classes = list(_selected_classes_key)
    if not pos_by_cluster or not selected_classes:
        return "<div style='padding:12px;color:var(--text-low)'>No hay datos POS.</div>"
    all_tags = set()
    for td in pos_by_cluster.values():
        all_tags.update(td.keys())
    if not all_tags:
        return "<div style='padding:12px;color:var(--text-low)'>No hay etiquetas POS.</div>"
    tcnt = defaultdict(int)
    for td in pos_by_cluster.values():
        for tag, cnt in td.items():
            tcnt[tag] += cnt
    tags = sorted(all_tags, key=lambda t: tcnt[t], reverse=True)[:15]
    clusters = sorted([c for c in selected_classes if c in class_sizes])
    hdr = "".join(f'<th style="text-align:center">Clase {c}</th>' for c in clusters)
    rows = ""
    for tag in tags:
        means = []
        for c in clusters:
            td = pos_by_cluster.get(c, {})  # int keys after normalisation
            tot = sum(td.values())
            means.append(td.get(tag, 0) / tot * 100 if tot else 0.0)
        mp = sum(means) / len(means) if means else 0
        cells = ""
        for c, pct in zip(clusters, means):
            col = (
                class_colors.get(c, "#AAA")
                if pct > mp + 10
                else ("#E86450" if pct < mp - 10 else T["text_mid"])
            )
            cells += f'<td style="text-align:center;color:{col}">{pct:.1f}%</td>'
        rows += f'<tr><td style="padding:5px 10px;color:var(--text-mid)">{sh(tag)}</td>{cells}</tr>'
    return (
        f'<table class="aux-table"><thead><tr><th>Categoría</th>{hdr}</tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )


def build_lemma_network(
    chosen_lemmas: list,
    _uces: list,
    _lemma_map: dict,
    min_cooc: int = 2,
    top_n: int = 50,
    min_degree: int = 1,
):
    if not chosen_lemmas or not _lemma_map:
        return None, None
    seed_set = set(chosen_lemmas)
    expanded_seeds = set()
    for lem in seed_set:
        expanded_seeds.add(lem)
        expanded_seeds.update(_lemma_map.get(lem, {}).get("stems", []))
        expanded_seeds.update(_lemma_map.get(lem, {}).get("formas", []))
    relevant_uces = []
    for uce in _uces:
        lms = uce.get("lemmas", []).copy()
        txt = uce.get("texto", "").lower()
        matched = False
        for s in expanded_seeds:
            if "_" in s:
                spaced = s.replace("_", " ")
                if s in txt or spaced in txt or all(p in lms for p in s.split("_")):
                    lms.append(s)
                    matched = True
            elif s in lms or s in txt:
                matched = True
        if matched:
            relevant_uces.append(lms)
    if not relevant_uces:
        return None, None
    cooc_counter = Counter()
    term_freq = Counter()
    for lemmas in relevant_uces:
        ls = set(lemmas)
        for t in ls:
            term_freq[t] += 1
        lst = list(ls)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                cooc_counter[tuple(sorted((lst[i], lst[j])))] += 1
    candidate_terms = set(seed_set)
    for (a, b), cnt in cooc_counter.items():
        if cnt >= min_cooc and (a in seed_set or b in seed_set):
            candidate_terms.update([a, b])
    if len(candidate_terms) > top_n + len(seed_set):
        others = candidate_terms - seed_set
        top_others = {
            t for t, _ in Counter({t: term_freq[t] for t in others}).most_common(top_n)
        }
        candidate_terms = seed_set | top_others
    G = nx.Graph()
    for t in candidate_terms:
        G.add_node(t, freq=term_freq[t])
    for (a, b), cnt in cooc_counter.items():
        if a in candidate_terms and b in candidate_terms and cnt >= min_cooc:
            G.add_edge(a, b, weight=cnt)
    nodes_to_remove = [
        n for n in G.nodes() if G.degree(n) < min_degree and n not in seed_set
    ]
    G.remove_nodes_from(nodes_to_remove)
    if G.number_of_nodes() == 0:
        return None, None
    neighbor_set = set()
    for s in seed_set:
        if s in G:
            neighbor_set.update(G.neighbors(s))
    neighbor_set -= seed_set
    isolated = [n for n in G.nodes() if G.degree(n) == 0 and n not in seed_set]
    G.remove_nodes_from(isolated)
    if G.number_of_nodes() == 0:
        return None, None
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)
    edge_x_hi, edge_y_hi, edge_x_lo, edge_y_lo = [], [], [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        seg = [x0, x1, None], [y0, y1, None]
        if u in seed_set or v in seed_set:
            edge_x_hi += seg[0]
            edge_y_hi += seg[1]
        else:
            edge_x_lo += seg[0]
            edge_y_lo += seg[1]
    traces = []
    if edge_x_lo:
        traces.append(
            go.Scatter(
                x=edge_x_lo,
                y=edge_y_lo,
                mode="lines",
                line=dict(width=0.5, color=T["border2"]),
                hoverinfo="none",
                showlegend=False,
            )
        )
    if edge_x_hi:
        traces.append(
            go.Scatter(
                x=edge_x_hi,
                y=edge_y_hi,
                mode="lines",
                line=dict(width=1.2, color=T["accent"]),
                hoverinfo="none",
                showlegend=False,
            )
        )

    def _node_group(nodes, color, size_scale, marker_extra=None, name=""):
        nx_list = list(nodes)
        if not nx_list:
            return None
        xs = [pos[n][0] for n in nx_list]
        ys = [pos[n][1] for n in nx_list]
        sizes = [max(6, G.nodes[n]["freq"] ** 0.5 * size_scale) for n in nx_list]
        marker = dict(size=sizes, color=color, line=dict(width=1.2, color=T["bg_page"]))
        if marker_extra:
            marker.update(marker_extra)
        hover_data = []
        for n in nx_list:
            top_stems = ", ".join(list(_lemma_map.get(n, {}).get("stems", [n]))[:2])
            hover_data.append([G.nodes[n]["freq"], G.degree(n), top_stems])
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=nx_list,
            textposition="top center",
            marker=marker,
            customdata=hover_data,
            hovertemplate="<b>%{text}</b><br>Frec: %{customdata[0]}<br>Grado: %{customdata[1]}<br>Raíces: %{customdata[2]}<extra></extra>",
            showlegend=bool(name),
            name=name,
        )

    seed_nodes = [n for n in G.nodes() if n in seed_set]
    t_seed = _node_group(
        seed_nodes,
        T["accent"],
        2.2,
        dict(line=dict(width=2, color=T["bg_page"])),
        "Lema semilla",
    )
    t_neighbors = _node_group(
        [n for n in G.nodes() if n in neighbor_set],
        hex_to_rgba("#EF9F27", 0.85),
        1.6,
        dict(symbol="circle-open-dot", line=dict(width=2.5, color="#EF9F27")),
        "Vecino directo",
    )
    t_other = _node_group(
        [n for n in G.nodes() if n not in seed_set and n not in neighbor_set],
        T["text_low"],
        1.4,
        name="",
    )
    traces += [t for t in [t_other, t_neighbors, t_seed] if t is not None]
    fig = go.Figure(data=traces)
    fig.update_layout(
        showlegend=True,
        hovermode="closest",
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.08,
            font=dict(size=9, family="IBM Plex Mono, monospace"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=460,
        margin=dict(l=8, r=8, t=8, b=40),
        **_plot_defaults(),
    )
    return fig, G


def make_term_explorer(cluster_id: int, _dm_key: bool):
    if terminos_df.empty:
        return None
    df_c = terminos_df[terminos_df["cluster"] == cluster_id].copy()
    if df_c.empty:
        return None
    df_c["sign"] = df_c["phi"].apply(lambda v: "positivo" if v > 0 else "negativo")
    cmap = {"positivo": class_colors.get(cluster_id, "#5BA8DC"), "negativo": "#E86450"}
    fig = px.scatter(
        df_c,
        x="frecuencia_global",
        y="phi",
        size=df_c["frecuencia_cluster"].clip(lower=1),
        color="sign",
        color_discrete_map=cmap,
        hover_name="termino",
        hover_data={
            "frecuencia_global": True,
            "frecuencia_cluster": True,
            "phi": ":.3f",
            "p_adj": ":.4f",
            "sign": False,
        },
        size_max=26,
        labels={
            "frecuencia_global": "Frec. global",
            "phi": "Coeficiente φ",
            "frecuencia_cluster": "Frec. en clase",
        },
    )
    fig.add_hline(y=0, line_dash="dot", line_color=T["border2"], line_width=1)
    fig.update_layout(
        height=320,
        margin=dict(l=8, r=8, t=8, b=24),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        xaxis=dict(showgrid=True, gridcolor=T["border"]),
        yaxis=dict(showgrid=True, gridcolor=T["border"], zeroline=False),
        **_plot_defaults(),
    )
    return fig


def make_transition_matrix(_dm_key: bool):
    if not uces:
        return None
    rows = []
    for uce in uces:
        cid = uce.get("cluster_id")
        did = uce.get("doc_id")
        local_idx = uce.get("metadata", {}).get("uce_local_idx", 0)
        if cid is not None and cid >= 0 and did is not None:
            rows.append(
                {"doc_id": did, "cluster_id": int(cid), "local_idx": int(local_idx)}
            )
    if not rows:
        return None
    df_seq = pd.DataFrame(rows).sort_values(["doc_id", "local_idx"])
    df_seq["next_cluster"] = df_seq.groupby("doc_id")["cluster_id"].shift(-1)
    df_seq = df_seq.dropna(subset=["next_cluster"])
    df_seq["next_cluster"] = df_seq["next_cluster"].astype(int)
    trans = pd.crosstab(df_seq["cluster_id"], df_seq["next_cluster"])
    row_sums = trans.sum(axis=1)
    trans_norm = trans.div(row_sums, axis=0).fillna(0.0)
    classes = sorted(trans_norm.index.tolist())
    z = [
        [
            round(trans_norm.loc[r, c], 3) if c in trans_norm.columns else 0.0
            for c in classes
        ]
        for r in classes
    ]
    axis_labels = [f"Clase {c}" for c in classes]
    hover = [
        [
            f"Desde Clase {r} → Clase {c}<br>P = {z[ri][ci]:.2f}"
            for ci, c in enumerate(classes)
        ]
        for ri, r in enumerate(classes)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=axis_labels,
            y=axis_labels,
            colorscale=[[0, "rgba(0,0,0,0)"], [0.5, "#5BA8DC"], [1.0, "#5BD9FF"]],
            zmin=0,
            zmax=1,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title="P",
                thickness=10,
                len=0.7,
                tickfont=dict(size=9, family="IBM Plex Mono"),
            ),
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=8, r=8, t=36, b=40),
        xaxis=dict(title="Clase destino", tickfont=dict(size=10), side="bottom"),
        yaxis=dict(title="Clase origen", tickfont=dict(size=10), autorange="reversed"),
        title=dict(
            text="Probabilidades de transición entre clases (normalizado por fila)",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def make_rf_triangulation(_dm_key: bool):
    if not shap_data or "error" in shap_data:
        return None
    fi = {
        r["feature"]: r["importance"] for r in shap_data.get("feature_importance", [])
    }
    perm = {
        r["feature"]: r["perm_importance"]
        for r in shap_data.get("permutation_importance", [])
    }
    shap_imp = {
        r["feature"]: r["mean_abs_shap"] for r in shap_data.get("shap_importance", [])
    }
    features = list(shap_imp.keys())
    if not features:
        return None

    def _minmax(vals):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) for v in vals]

    shap_vals = [shap_imp.get(f, 0) for f in features]
    perm_vals = [perm.get(f, 0) for f in features]
    gini_vals = [fi.get(f, 0) for f in features]
    shap_n = _minmax(shap_vals)
    perm_n = _minmax(perm_vals)
    gini_n = _minmax(gini_vals)
    order = sorted(range(len(features)), key=lambda i: shap_n[i])
    feat_ord = [features[i] for i in order]
    col_shap = "#7C6BF8"
    col_perm = "#5DC88A"
    col_gini = "#E8A838"
    fig = go.Figure()
    for yi, oi in enumerate(order):
        vals = [shap_n[oi], perm_n[oi], gini_n[oi]]
        fig.add_trace(
            go.Scatter(
                x=vals,
                y=[yi, yi, yi],
                mode="lines",
                line=dict(color=T["border2"], width=1),
                hoverinfo="none",
                showlegend=False,
            )
        )
    for label, norm_vals, color, symbol in [
        ("SHAP", [shap_n[i] for i in order], col_shap, "circle"),
        ("Permutación", [perm_n[i] for i in order], col_perm, "square"),
        ("Gini", [gini_n[i] for i in order], col_gini, "diamond"),
    ]:
        orig_vals = (
            shap_vals
            if label == "SHAP"
            else perm_vals
            if label == "Permutación"
            else gini_vals
        )
        fig.add_trace(
            go.Scatter(
                x=norm_vals,
                y=list(range(len(order))),
                mode="markers",
                name=label,
                marker=dict(
                    size=10,
                    color=color,
                    symbol=symbol,
                    line=dict(width=1.5, color=T["bg_page"]),
                ),
                customdata=[
                    [feat_ord[yi], orig_vals[order[yi]]] for yi in range(len(order))
                ],
                hovertemplate="<b>%{customdata[0]}</b><br>"
                + label
                + ": %{customdata[1]:.4f}<extra></extra>",
            )
        )
    cv = shap_data.get("cv_balanced_accuracy", {})
    ba_mean = cv.get("mean", 0)
    ba_std = cv.get("std", 0)
    validity_col = (
        "#5DC88A" if ba_mean > 0.65 else ("#E8A838" if ba_mean > 0.45 else "#E86450")
    )
    fig.update_layout(
        height=max(300, len(features) * 36 + 80),
        margin=dict(l=8, r=8, t=60, b=40),
        xaxis=dict(
            title="Importancia normalizada (0–1)",
            range=[-0.05, 1.05],
            showgrid=True,
            gridcolor=T["border"],
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            tickvals=list(range(len(feat_ord))),
            ticktext=feat_ord,
            tickfont=dict(size=9.5),
            automargin=True,
            showgrid=True,
            gridcolor=T["border"],
        ),
        legend=dict(
            orientation="h",
            y=1.06,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        title=dict(
            text=(
                f"Triangulación RF · Balanced Accuracy = "
                f"<span style='color:{validity_col}'>{ba_mean:.3f} ± {ba_std:.3f}</span>"
            ),
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def make_doc_composition(_dm_key: bool):
    if not uces:
        return None
    rows = []
    for uce in uces:
        cid = uce.get("cluster_id")
        did = uce.get("doc_id")
        if cid is not None and cid >= 0 and did is not None:
            rows.append({"doc_id": str(did), "cluster_id": int(cid)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    counts = df.groupby(["doc_id", "cluster_id"]).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)
    if props.empty:
        return None
    classes = sorted(props.columns.tolist())
    dom_class = props.idxmax(axis=1)
    props["_dom"] = dom_class
    props["_dom_val"] = props.max(axis=1)
    props = props.sort_values(["_dom", "_dom_val"], ascending=[True, False])
    props = props.drop(columns=["_dom", "_dom_val"])
    doc_labels = props.index.tolist()
    bar_h = max(280, len(doc_labels) * 22 + 80)
    fig = go.Figure()
    for c in classes:
        col = class_colors.get(c, "#AAA")
        vals = [
            round(props.loc[d, c], 3) if c in props.columns else 0.0 for d in doc_labels
        ]
        fig.add_trace(
            go.Bar(
                name=f"Clase {c}",
                x=vals,
                y=doc_labels,
                orientation="h",
                marker=dict(color=col, line=dict(width=0)),
                hovertemplate=f"Clase {c}: %{{x:.0%}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=bar_h,
        margin=dict(l=8, r=8, t=36, b=40),
        xaxis=dict(
            title="Composición discursiva",
            tickformat=".0%",
            range=[0, 1],
            showgrid=True,
            gridcolor=T["border"],
            tickfont=dict(size=9),
        ),
        yaxis=dict(tickfont=dict(size=8.5), automargin=True, showgrid=False),
        legend=dict(
            orientation="h",
            y=1.04,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        bargap=0.05,
        title=dict(
            text="Perfil discursivo por documento · ordenado por clase dominante",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def make_graphe_specificites(_selected_classes_key: tuple, top_n: int, _dm_key: bool):
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    selected_classes = list(_selected_classes_key)
    if terminos_df.empty or not selected_classes:
        return None
    classes = sorted([c for c in selected_classes if c in class_sizes])
    if not classes:
        return None
    pivot = (
        terminos_df[terminos_df["cluster"].isin(classes)]
        .pivot_table(index="termino", columns="cluster", values="phi", aggfunc="first")
        .reindex(columns=classes)
        .fillna(0.0)
    )
    if pivot.empty:
        return None
    pivot["_max"] = pivot.abs().max(axis=1)
    pivot = (
        pivot.sort_values("_max", ascending=False).head(top_n).drop(columns=["_max"])
    )
    if len(pivot) < 2:
        return None
    try:
        jitter = np.random.default_rng(42).uniform(0, 1e-9, pivot.values.shape)
        dist = pdist(pivot.values + jitter, metric="euclidean")
        Z = linkage(dist, method="ward")
        order = leaves_list(Z)
        pivot = pivot.iloc[order]
    except Exception as e:
        print(f"[make_graphe_specificites] CAH ordering failed: {e}")
    terms = pivot.index.tolist()
    z_vals = pivot.values.tolist()
    max_abs = max(abs(pivot.values.max()), abs(pivot.values.min()), 0.01)
    fig = go.Figure(
        go.Heatmap(
            z=z_vals,
            x=[f"Clase {c}" for c in classes],
            y=terms,
            colorscale=[[0.0, "#E86450"], [0.5, T["bg_card"]], [1.0, "#5BA8DC"]],
            zmid=0,
            zmin=-max_abs,
            zmax=max_abs,
            hovertemplate="<b>%{y}</b> · %{x}<br>φ = %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title="φ",
                thickness=10,
                len=0.6,
                tickfont=dict(size=9, family="IBM Plex Mono"),
                tickvals=[-max_abs, 0, max_abs],
                ticktext=[f"−{max_abs:.2f}", "0", f"+{max_abs:.2f}"],
            ),
        )
    )
    fig.update_layout(
        height=max(350, len(terms) * 16 + 100),
        margin=dict(l=8, r=8, t=36, b=40),
        xaxis=dict(side="top", tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=8), automargin=True, autorange="reversed"),
        title=dict(
            text="Graphe des spécificités · termes ordonnés por CAH",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def make_tensions_lexicales(_selected_classes_key: tuple, _dm_key: bool):
    selected_classes = list(_selected_classes_key)
    if terminos_df.empty or not selected_classes:
        return None
    classes = sorted([c for c in selected_classes if c in class_sizes])
    if not classes:
        return None
    agg = (
        terminos_df[terminos_df["cluster"].isin(classes)]
        .groupby("termino")
        .agg(
            phi_max=("phi", lambda x: x.abs().max()),
            phi_var=("phi", "var"),
            freq=("frecuencia_global", "max"),
        )
        .reset_index()
        .dropna()
    )
    pivot_dom = (
        terminos_df[terminos_df["cluster"].isin(classes)]
        .pivot_table(index="termino", columns="cluster", values="phi", aggfunc="first")
        .reindex(columns=classes)
        .fillna(0.0)
    )
    dom_map = pivot_dom.abs().idxmax(axis=1).to_dict()
    agg["dom_class"] = agg["termino"].map(dom_map).fillna(classes[0]).astype(int)
    med_x = float(agg["phi_max"].median())
    med_y = float(agg["phi_var"].median())
    fig = go.Figure()
    for c in classes:
        col = class_colors.get(c, "#AAA")
        sub = agg[agg["dom_class"] == c]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["phi_max"],
                y=sub["phi_var"],
                mode="markers",
                name=f"Clase {c}",
                marker=dict(
                    size=(sub["freq"] ** 0.45).clip(4, 20).tolist(),
                    color=col,
                    opacity=0.75,
                    line=dict(width=0.5, color=T["bg_page"]),
                ),
                customdata=list(
                    zip(
                        sub["termino"],
                        sub["freq"].round(0),
                        sub["phi_max"].round(3),
                        sub["phi_var"].round(3),
                    )
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Max φ: %{customdata[2]}<br>"
                    "Var φ: %{customdata[3]}<br>"
                    "Frec: %{customdata[1]}<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=med_x, line_width=1, line_dash="dot", line_color=T["border2"])
    fig.add_hline(y=med_y, line_width=1, line_dash="dot", line_color=T["border2"])
    for tx, ty, label in [
        (0.97, 0.97, "Polisémicos"),
        (0.97, 0.03, "Especializados"),
        (0.03, 0.97, "Disputados"),
        (0.03, 0.03, "Neutros"),
    ]:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=tx,
            y=ty,
            text=label,
            showarrow=False,
            font=dict(size=9, color=T["text_dim"], family="IBM Plex Mono"),
            xanchor="right" if tx > 0.5 else "left",
            yanchor="top" if ty > 0.5 else "bottom",
        )
    fig.update_layout(
        height=400,
        margin=dict(l=8, r=8, t=36, b=50),
        xaxis=dict(
            title="Max |φ| across classes",
            showgrid=True,
            gridcolor=T["border"],
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title="Variance of φ across classes",
            showgrid=True,
            gridcolor=T["border"],
            tickfont=dict(size=9),
        ),
        legend=dict(
            orientation="h",
            y=-0.12,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        title=dict(
            text="Carta de tensiones lexicales · tamaño = frecuencia global",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def make_semantic_confusion(_dm_key: bool):
    if not shap_data or "error" in shap_data:
        return None

    # Intentar obtener clases desde múltiples fuentes
    classes_rf = shap_data.get("classes", [])
    if not classes_rf:
        # Si no hay classes, podemos extraer de y_true (si existe)
        y_true = shap_data.get("y_true", [])
        if y_true:
            classes_rf = sorted(set(int(v) for v in y_true))
        else:
            return None

    if len(classes_rf) < 2:
        return None

    n = len(classes_rf)
    misclass_texts = shap_data.get("misclassified_texts", {})

    conf_raw = shap_data.get("confusion_matrix")
    if conf_raw:
        # Use provided confusion matrix
        raw = [[int(v) for v in row] for row in conf_raw]
        # Validate shape
        if len(raw) != n or any(len(row) != n for row in raw):
            return None
        row_totals = [max(sum(row), 1) for row in raw]
        z_conf = [
            [round(raw[ri][ci] / row_totals[ri], 3) for ci in range(n)]
            for ri in range(n)
        ]
        # Build hover with counts and rates
        hover = []
        for ri in range(n):
            row_hover = []
            for ci in range(n):
                rate = z_conf[ri][ci]
                count = raw[ri][ci]
                if ri != ci:
                    key = f"{classes_rf[ri]}_to_{classes_rf[ci]}"
                    snippets = misclass_texts.get(key, [])
                    tip = (
                        "<br>".join(f"• {s[:60]}…" for s in snippets[:3])
                        if snippets
                        else "Sin ejemplos"
                    )
                    row_hover.append(
                        f"True {classes_rf[ri]} → Pred {classes_rf[ci]}<br>n={count} ({rate:.0%})<br>{tip}"
                    )
                else:
                    row_hover.append(
                        f"Correcto: Clase {classes_rf[ri]}<br>n={count} ({rate:.0%})"
                    )
            hover.append(row_hover)
    else:
        # Compute confusion matrix from y_true and oof_predictions
        y_true = shap_data.get("y_true", [])
        oof_preds = shap_data.get("oof_predictions", [])
        if not y_true or not oof_preds or len(y_true) != len(oof_preds):
            return None
        try:
            from sklearn.metrics import confusion_matrix as sk_cm

            cm = sk_cm(y_true, oof_preds, labels=classes_rf)
            # Convert to list of lists and normalize?
            # For hover, we might want raw counts only, but can also show rates
            raw = cm.tolist()  # raw counts
            row_totals = [max(sum(row), 1) for row in raw]
            z_conf = [
                [round(raw[ri][ci] / row_totals[ri], 3) for ci in range(n)]
                for ri in range(n)
            ]
        except Exception as e:
            return None

        # Build hover (simple, without snippets because we don't have misclassified_texts keys? Actually we do have misclass_texts but it's based on classes, not on the confusion matrix indices)
        hover = []
        for ri in range(n):
            row_hover = []
            for ci in range(n):
                count = raw[ri][ci]
                if ri != ci:
                    key = f"{classes_rf[ri]}_to_{classes_rf[ci]}"
                    snippets = misclass_texts.get(key, [])
                    tip = (
                        "<br>".join(f"• {s[:60]}…" for s in snippets[:3])
                        if snippets
                        else "Sin ejemplos"
                    )
                    row_hover.append(
                        f"True Clase {classes_rf[ri]} → Pred Clase {classes_rf[ci]}<br>n={count}<br>{tip}"
                    )
                else:
                    row_hover.append(f"Correcto: Clase {classes_rf[ri]}<br>n={count}")
            hover.append(row_hover)

    # Now build figure
    axis_labels = [f"Clase {c}" for c in classes_rf]

    fig = go.Figure(
        go.Heatmap(
            z=z_conf,
            x=axis_labels,
            y=axis_labels,
            colorscale=[[0, T["bg_card"]], [1, "#E86450"]],
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title="n",
                thickness=10,
                len=0.7,
                tickfont=dict(size=9, family="IBM Plex Mono"),
            ),
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=8, r=8, t=36, b=40),
        xaxis=dict(title="Predicha", side="bottom", tickfont=dict(size=10)),
        yaxis=dict(title="Real", autorange="reversed", tickfont=dict(size=10)),
        title=dict(
            text="Matriz de confusión semántica · hover = UCEs mal clasificadas",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def make_mca_plot(
    _dm_key: bool,
    outlier_filter: bool = True,
    jitter: float = 0.02,
    term_threshold: float = 0.0,
):
    """
    MCA biplot with correct scaling: both rows and columns in principal coordinates.
    """
    global multivariate, shap_data, class_colors, T, _plot_defaults
    if not multivariate or "row_coords" not in multivariate:
        return None
    if not shap_data or "error" in shap_data:
        return None

    # ----- Row coordinates (already principal) -----
    row_coords = np.array(multivariate["row_coords"], dtype=float)
    if row_coords.ndim == 1:
        row_coords = np.vstack([row_coords, np.zeros_like(row_coords)]).T
    if row_coords.shape[1] < 2:
        row_coords = np.hstack([row_coords, np.zeros((row_coords.shape[0], 1))])

    # ----- Document IDs -----
    doc_ids = multivariate.get(
        "doc_ids", [f"Doc_{i}" for i in range(row_coords.shape[0])]
    )
    if len(doc_ids) != row_coords.shape[0]:
        doc_ids = [f"Doc_{i}" for i in range(row_coords.shape[0])]

    # ----- True labels and OOF predictions -----
    y_true = [int(float(v)) for v in shap_data.get("y_true", [])]
    oof_preds = [
        int(float(v)) if v is not None else -1
        for v in shap_data.get("oof_predictions", [])
    ]

    if not y_true or not oof_preds or len(y_true) != len(oof_preds):
        return None
    if len(y_true) != row_coords.shape[0]:
        print(
            f"[make_mca_plot] row_coords ({row_coords.shape[0]}) vs y_true ({len(y_true)}) mismatch"
        )
        return None

    # ----- Outlier filter -----
    if outlier_filter and row_coords.shape[0] > 3:
        std0 = np.std(row_coords[:, 0]) or 1.0
        std1 = np.std(row_coords[:, 1]) or 1.0
        z1 = np.abs((row_coords[:, 0] - np.mean(row_coords[:, 0])) / std0)
        z2 = np.abs((row_coords[:, 1] - np.mean(row_coords[:, 1])) / std1)
        keep = np.where((z1 < 3) & (z2 < 3))[0]
        row_coords = row_coords[keep]
        y_true = [y_true[i] for i in keep]
        oof_preds = [oof_preds[i] for i in keep]
        doc_ids = [doc_ids[i] for i in keep]

    # ----- Jitter -----
    if jitter > 0:
        rng = np.random.default_rng(42)
        row_coords = row_coords + jitter * (rng.random(row_coords.shape) - 0.5)

    # ----- Masks -----
    mask_mis = [t != p for t, p in zip(y_true, oof_preds)]
    classes = sorted(set(y_true))
    masks_ok = {
        c: np.array([t == c and t == p for t, p in zip(y_true, oof_preds)])
        for c in classes
    }

    # ----- Term data -----
    terms = multivariate.get("terms", [])
    col_coords_raw = multivariate.get("col_coords", [])  # standard coordinates
    term_cluster = multivariate.get("term_cluster", [])
    term_phi = multivariate.get("term_phi", [])
    eigenvalues = multivariate.get("eigenvalues", [])  # for scaling

    has_terms = bool(col_coords_raw) and bool(terms)

    if has_terms:
        # Convert to numpy
        col_coords = np.array(col_coords_raw, dtype=float)
        if col_coords.ndim == 1:
            col_coords = np.vstack([col_coords, np.zeros_like(col_coords)]).T
        if col_coords.shape[1] < 2:
            col_coords = np.hstack([col_coords, np.zeros((col_coords.shape[0], 1))])

        # Fix indicator matrix duplication (prince.MCA returns 2×n_terms)
        if len(col_coords) == 2 * len(terms):
            col_coords = col_coords[::2]
            if term_cluster and len(term_cluster) == 2 * len(terms):
                term_cluster = term_cluster[::2]
            if term_phi and len(term_phi) == 2 * len(terms):
                term_phi = term_phi[::2]

        # Align lengths
        if len(col_coords) != len(terms):
            print(
                f"[MCA] terms ({len(terms)}) / col_coords ({len(col_coords)}) mismatch – truncating"
            )
            n = min(len(col_coords), len(terms))
            col_coords = col_coords[:n]
            terms = terms[:n]
            if term_cluster:
                term_cluster = term_cluster[:n] if len(term_cluster) >= n else []
            if term_phi:
                term_phi = term_phi[:n] if len(term_phi) >= n else []

        # ----- Scale column coordinates to principal coordinates -----
        # Standard coordinates (V) -> principal coordinates (V * sqrt(λ))
        if len(eigenvalues) >= 2:
            col_coords_principal = col_coords.copy()
            col_coords_principal[:, 0] *= np.sqrt(eigenvalues[0])
            col_coords_principal[:, 1] *= np.sqrt(eigenvalues[1])
        else:
            col_coords_principal = col_coords

        # Now col_coords_principal is on the same scale as row_coords
        contrib = col_coords_principal[:, 0] ** 2 + col_coords_principal[:, 1] ** 2
        max_contrib = contrib.max() if len(contrib) else 1.0
        keep_terms = (
            contrib >= term_threshold * max_contrib
            if term_threshold > 0
            else np.ones(len(contrib), dtype=bool)
        )

        use_cluster_colors = bool(term_cluster) and len(term_cluster) == len(terms)

        if keep_terms.any():
            col_coords_keep = col_coords_principal[keep_terms]
            terms_keep = [terms[i] for i in range(len(terms)) if keep_terms[i]]
            contrib_keep = contrib[keep_terms]
            if use_cluster_colors:
                term_cluster_keep = [
                    term_cluster[i] for i in range(len(terms)) if keep_terms[i]
                ]
                term_phi_keep = (
                    [term_phi[i] for i in range(len(terms)) if keep_terms[i]]
                    if term_phi
                    else []
                )
            else:
                term_cluster_keep = []
                term_phi_keep = []
        else:
            has_terms = False
            col_coords_keep = np.empty((0, 2))
            terms_keep = []
            contrib_keep = []

    # ----- Figure -----
    fig = go.Figure()
    fig.add_hline(y=0, line_width=0.6, line_color=T["border2"])
    fig.add_vline(x=0, line_width=0.6, line_color=T["border2"])

    # ----- Reference circle (mean distance of terms in principal space) -----
    if has_terms and len(contrib_keep) > 0:
        mean_dist = np.sqrt(contrib_keep).mean()
        theta = np.linspace(0, 2 * np.pi, 120)
        fig.add_trace(
            go.Scatter(
                x=(mean_dist * np.cos(theta)).tolist(),
                y=(mean_dist * np.sin(theta)).tolist(),
                mode="lines",
                line=dict(color=T["border2"], width=1, dash="dot"),
                name="Radio medio (referencia)",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    # ----- UCE points: misclassified (grey, solid) -----
    mis_idx = [i for i, m in enumerate(mask_mis) if m]
    if mis_idx:
        fig.add_trace(
            go.Scatter(
                x=row_coords[mis_idx, 0].tolist(),
                y=row_coords[mis_idx, 1].tolist(),
                mode="markers",
                name="Mal clasificadas",
                marker=dict(
                    size=6,
                    color="#AAAAAA",
                    symbol="circle",
                    opacity=0.8,
                    line=dict(width=1, color="#666"),
                ),
                customdata=[[doc_ids[i], y_true[i], oof_preds[i]] for i in mis_idx],
                hovertemplate=(
                    "Documento: %{customdata[0]}<br>"
                    "Real: %{customdata[1]} · Pred: %{customdata[2]}<extra></extra>"
                ),
                showlegend=True,
            )
        )

    # ----- UCE points: correctly classified per class, centroids, ellipses -----
    class_centroids = {}
    for c in classes:
        col = class_colors.get(int(c), "#AAA")
        ok_idx = np.where(masks_ok[c])[0]
        if len(ok_idx) == 0:
            continue

        xs = row_coords[ok_idx, 0].tolist()
        ys = row_coords[ok_idx, 1].tolist()

        # Correct points
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=f"Clase {c} (correctas)",
                marker=dict(
                    size=6,
                    color=col,
                    symbol="circle",
                    opacity=0.7,
                    line=dict(width=0.5, color=T["bg_page"]),
                ),
                customdata=[[doc_ids[i], c, c] for i in ok_idx],
                hovertemplate=(
                    "Documento: %{customdata[0]}<br>"
                    "Real: %{customdata[1]} · Pred: %{customdata[2]}<extra></extra>"
                ),
                showlegend=True,
            )
        )

        # Centroid
        cx, cy = (
            float(np.mean(row_coords[ok_idx, 0])),
            float(np.mean(row_coords[ok_idx, 1])),
        )
        class_centroids[c] = (cx, cy)
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers",
                name=f"Centroide {c}",
                marker=dict(
                    size=13,
                    color=col,
                    symbol="x-thin",
                    line=dict(width=2.5, color="black"),
                ),
                hoverinfo="text",
                text=f"Centroide clase {c}<br>({cx:.3f}, {cy:.3f})",
                showlegend=False,
            )
        )

        # 95% confidence ellipse
        coords_c = row_coords[ok_idx]
        if len(coords_c) >= 3:
            cov = np.cov(coords_c.T)
            if cov.shape == (2, 2):
                try:
                    from scipy.stats import chi2

                    chi2_val = chi2.ppf(0.95, 2)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    eigvals = np.maximum(eigvals, 1e-12)
                    width = 2 * np.sqrt(chi2_val * eigvals[1])
                    height = 2 * np.sqrt(chi2_val * eigvals[0])
                    angle = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1])))
                    t = np.linspace(0, 2 * np.pi, 80)
                    ang_r = np.radians(angle)
                    ex = (
                        cx
                        + (width / 2) * np.cos(t) * np.cos(ang_r)
                        - (height / 2) * np.sin(t) * np.sin(ang_r)
                    )
                    ey = (
                        cy
                        + (width / 2) * np.cos(t) * np.sin(ang_r)
                        + (height / 2) * np.sin(t) * np.cos(ang_r)
                    )
                    path = (
                        f"M {ex[0]} {ey[0]} "
                        + " ".join(f"L {ex[i]} {ey[i]}" for i in range(1, len(ex)))
                        + " Z"
                    )
                    fig.add_shape(
                        type="path",
                        path=path,
                        line=dict(color=col, width=1.5, dash="dash"),
                        fillcolor="rgba(0,0,0,0)",
                        layer="below",
                        xref="x",
                        yref="y",
                    )
                except Exception as e:
                    print(f"[MCA] Ellipse failed for class {c}: {e}")

    # ----- Terms (single trace, small dots, now on same scale) -----
    if has_terms and col_coords_keep.shape[0] > 0:
        # Assign colors
        if use_cluster_colors and term_cluster_keep:
            term_colors = []
            for i, cid in enumerate(term_cluster_keep):
                if cid is not None and cid >= 0 and cid in class_colors:
                    base = class_colors[cid]
                    if base.startswith("rgb"):
                        rgb = base[base.find("(") + 1 : base.find(")")].split(",")
                        r, g, b = [int(x.strip()) for x in rgb[:3]]
                        phi_val = term_phi_keep[i] if i < len(term_phi_keep) else 1.0
                        phi_val = max(0.0, min(1.0, phi_val))
                        opacity = 0.3 + 0.6 * phi_val
                        term_colors.append(f"rgba({r},{g},{b},{opacity:.2f})")
                    else:
                        term_colors.append(base)
                else:
                    term_colors.append(T["text_mid"])
        else:
            avg_contrib = contrib_keep.mean()
            active = contrib_keep >= avg_contrib
            term_colors = [
                "rgba(80,120,200,0.7)" if a else "rgba(160,160,160,0.5)" for a in active
            ]

        fig.add_trace(
            go.Scatter(
                x=col_coords_keep[:, 0].tolist(),
                y=col_coords_keep[:, 1].tolist(),
                mode="markers",
                name="Términos",
                marker=dict(
                    size=4, color=term_colors, symbol="circle", line=dict(width=0)
                ),
                hoverinfo="text",
                text=terms_keep,
                showlegend=True,
            )
        )

    # ----- Axis titles -----
    expl = multivariate.get("explained_inertia", [0, 0])
    x_title = f"Dimensión 1 ({expl[0] * 100:.1f}%)" if len(expl) > 0 else "Dimensión 1"
    y_title = f"Dimensión 2 ({expl[1] * 100:.1f}%)" if len(expl) > 1 else "Dimensión 2"

    # ----- Layout -----
    fig.update_layout(
        height=560,
        margin=dict(l=40, r=40, t=48, b=48),
        xaxis=dict(title=x_title, showgrid=True, gridcolor=T["border"], zeroline=False),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor=T["border"],
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        title=dict(
            text=(
                "Análisis de Correspondencias Múltiples · "
                "Documentos y términos · ✗ = mal clasificadas por el RF"
            ),
            font=dict(size=10, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def make_error_by_class(_dm_key: bool):
    """
    Create a horizontal bar chart showing out‑of‑fold error rates per class.

    Parameters
    ----------
    _dm_key : bool
        Unused, kept for callback compatibility.
    """
    if not shap_data or "error" in shap_data:
        return None

    # 1. Normalise data to integers (handles JSON deserialization)
    y_true = [int(v) for v in shap_data.get("y_true", [])]
    oof_preds = shap_data.get("oof_predictions", [])
    if not y_true or not oof_preds or len(y_true) != len(oof_preds):
        return None
    oof_preds = [int(v) if v is not None else None for v in oof_preds]

    # 2. Get sorted list of class labels (integers)
    classes_rf = sorted({int(c) for c in shap_data.get("classes", y_true)})

    rows = []
    for c in classes_rf:
        total = 0
        errors = 0
        for i, true_val in enumerate(y_true):
            if true_val == c:
                total += 1
                pred_val = oof_preds[i]
                if pred_val is not None and pred_val != c:
                    errors += 1
        if total == 0:
            continue
        rows.append(
            {
                "clase": f"Clase {c}",
                "error_rate": errors / total,
                "total": total,
                "errors": errors,
                "c": c,
            }
        )

    if not rows:
        return None

    # 3. Sort by error rate (highest first) for display
    rows.sort(key=lambda x: x["error_rate"], reverse=True)

    # 4. Build single‑trace bar chart
    fig = go.Figure(
        go.Bar(
            x=[row["error_rate"] for row in rows],
            y=[row["clase"] for row in rows],
            orientation="h",
            marker=dict(
                color=[class_colors.get(row["c"], "#AAA") for row in rows],
                opacity=0.85,
            ),
            customdata=[[row["errors"], row["total"], row["c"]] for row in rows],
            hovertemplate=(
                "Clase %{customdata[2]}: "
                "%{customdata[0]}/%{customdata[1]} errores "
                "(%{x:.1%})<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # 5. Add a vertical line at random‑chance baseline
    n_classes = len(classes_rf)
    if n_classes > 0:
        random_baseline = 1 - (1 / n_classes)
        fig.add_vline(
            x=random_baseline,
            line_dash="dash",
            line_color=T["text_low"],
            annotation_text="azar",
            annotation_font_size=8,
        )

    # 6. Layout adjustments
    fig.update_layout(
        height=max(160, len(rows) * 50 + 60),
        margin=dict(l=8, r=8, t=36, b=30),
        xaxis=dict(
            title="Tasa de error OOF",
            tickformat=".0%",
            showgrid=True,
            gridcolor=T["border"],
            range=[0, 1],
        ),
        yaxis=dict(autorange="reversed", automargin=True),
        title=dict(
            text="Tasa de error por clase (Out‑of‑Fold)",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def make_shap_beeswarm(_dm_key: bool, class_idx: int = None):
    """
    Creates a beeswarm plot of SHAP values.

    Parameters
    ----------
    _dm_key : bool
        Unused (kept for callback compatibility).
    class_idx : int, optional
        If given, show SHAP values for that class only.
        If None, show the mean across all classes.
    """
    shap_values = shap_data.get("shap_values")
    feature_names = shap_data.get("feature_names", [])
    if not feature_names:
        return None

    raw_X_dicts = shap_data.get("raw_X", [])
    if not raw_X_dicts:
        return None
    raw_X = pd.DataFrame(raw_X_dicts)

    # Check column alignment
    if set(raw_X.columns) != set(feature_names):
        print("Column mismatch between raw_X and feature_names")
        return None
    raw_X = raw_X[feature_names]

    # Convert SHAP values to numpy, handling possible list-of-arrays serialisation
    sv = np.array(shap_values)
    # Shape may be (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
    if sv.ndim == 3:
        # Try to determine correct axis order
        # Assume that features are the last dimension (most common)
        if sv.shape[2] != len(feature_names):
            # Maybe the second dimension is features
            if sv.shape[1] == len(feature_names):
                # Transpose to (n_classes, n_samples, n_features)
                sv = sv.transpose(0, 2, 1)
            else:
                print(
                    f"Cannot reconcile SHAP shape {sv.shape} with {len(feature_names)} features"
                )
                return None
        n_classes, n_samples, n_features = sv.shape

        # Choose which class to use for the signed direction
        if class_idx is not None and 0 <= class_idx < n_classes:
            sv_signed = sv[class_idx, :, :]  # SHAP values for that class
        else:
            sv_signed = np.abs(sv).mean(axis=0)  # mean across classes

        # Mean absolute SHAP across all samples and classes
        mean_abs_shap = np.abs(sv).mean(axis=(0, 1))  # (n_features,)
    else:
        # Binary case: shape (n_samples, n_features)
        n_samples, n_features = sv.shape
        sv_signed = sv
        mean_abs_shap = np.abs(sv).mean(axis=0)

    # Safety: ensure n_samples is defined (it will be from the branch above)
    if n_samples is None:
        n_samples = sv_signed.shape[0]

    # Feature importance (mean |SHAP|) for sorting
    shap_imp = shap_data.get("shap_importance", [])
    if shap_imp:
        imp_dict = {item["feature"]: item["mean_abs_shap"] for item in shap_imp}
        importance = [
            imp_dict.get(f, mean_abs_shap[i]) for i, f in enumerate(feature_names)
        ]
    else:
        importance = mean_abs_shap

    # Sort features by importance (descending)
    indexed = sorted(
        [(i, f, importance[i]) for i, f in enumerate(feature_names)],
        key=lambda x: x[2],
        reverse=False,
    )
    indices, ordered_features, ordered_imp = zip(*indexed) if indexed else ([], [], [])

    # Build the plot
    rng = np.random.default_rng(42)
    fig = go.Figure()
    fig.add_vline(x=0, line_width=1, line_color=T["border2"])

    for rank, (idx, fname, _) in enumerate(zip(indices, ordered_features, ordered_imp)):
        sv_col = sv_signed[:, idx]  # SHAP values for this feature
        feat_val_col = raw_X[fname].values

        # Normalise feature values to [0,1] for colour mapping
        # Use numpy type checking to avoid boolean subtraction issues
        if np.issubdtype(feat_val_col.dtype, np.number):
            fv_min, fv_max = feat_val_col.min(), feat_val_col.max()
            if fv_max == fv_min:
                fv_norm = np.zeros_like(feat_val_col)
            else:
                fv_norm = (feat_val_col - fv_min) / (fv_max - fv_min)
        else:
            # For non‑numeric, use the position in the sorted order
            _, fv_norm = np.unique(feat_val_col, return_inverse=True)
            if fv_norm.max() > 0:
                fv_norm = fv_norm / fv_norm.max()

        colors = [
            f"rgba(226,75,74,{0.6 + 0.3 * v})"
            if v > 0.5
            else f"rgba(55,138,221,{0.6 + 0.3 * (1 - v)})"
            for v in fv_norm
        ]

        jitter_y = rank + rng.uniform(-0.35, 0.35, size=n_samples)

        # Build hover text: try to show the original category if available
        customdata = []
        for v, orig_val in zip(feat_val_col, raw_X[fname]):
            # If the column was frequency‑encoded, the original value is not in raw_X.
            # We could look up the original via metadata, but for simplicity we mark it as encoded.
            if pd.api.types.is_float(v) and (v == 0.0 or v == 1.0 or 0 <= v <= 1):
                customdata.append(f"{v:.3f} (codificado)")
            else:
                customdata.append(v)

        fig.add_trace(
            go.Scatter(
                x=sv_col.tolist(),
                y=jitter_y.tolist(),
                mode="markers",
                marker=dict(size=4, color=colors, opacity=0.7),
                name=fname,
                customdata=customdata,
                hovertemplate=(
                    f"<b>{fname}</b><br>"
                    f"Valor: %{{customdata}}<br>"
                    f"SHAP: %{{x:.3f}}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Plot title: indicate which class is shown
    class_text = ""
    if sv.ndim == 3 and class_idx is not None:
        class_text = f" · Clase {class_idx}"
    elif sv.ndim == 3:
        class_text = " · Promedio entre clases"

    fig.update_layout(
        height=max(300, len(ordered_features) * 30 + 80),
        margin=dict(l=8, r=8, t=36, b=40),
        xaxis=dict(
            title="Valor SHAP (impacto en la predicción)",
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            tickvals=list(range(len(ordered_features))),
            ticktext=ordered_features,
            tickfont=dict(size=9.5),
            automargin=True,
            showgrid=True,
            gridcolor=T["border"],
        ),
        title=dict(
            text=f"SHAP Beeswarm{class_text} · azul = valor bajo · rojo = valor alto",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRAJECTORY  –  smoothed, dual-trace
# ─────────────────────────────────────────────────────────────────────────────


def plot_trajectory_smooth(
    series: pd.Series,
    metric: str = "",
    use_log: bool = False,
    title: str = "",
    color: str = T["accent"],
    height: int = 220,
) -> go.Figure:
    """
    Dual-trace trajectory: raw (dim) + SavGol-smoothed (bright coral).

    Args:
        series:   pd.Series indexed by UCE order (0…n).
        metric:   y-axis label.
        use_log:  Apply log scale on y.
        title:    Chart title.
        color:    Base colour for raw trace.
        height:   Figure height in px.
    """
    y = series.fillna(0).values
    x = np.arange(len(y))
    y_smooth = smooth_trajectory(tuple(y))

    fig = go.Figure()

    # Raw trace – semi-transparent dots
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            opacity=0.25,
            marker=dict(size=3, color=color),
            line=dict(width=1, color=color),
            name="bruto",
            hovertemplate="UCE %{x}: %{y:.4f}<extra></extra>",
        )
    )

    # Smoothed trace
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_smooth,
            mode="lines",
            line=dict(width=2.5, color=T["accent2"]),
            name="suavizado",
            hovertemplate="suav %{x}: %{y:.4f}<extra></extra>",
        )
    )

    # Zero / mean reference line
    mean_val = float(np.nanmean(y))
    fig.add_hline(
        y=mean_val,
        line_dash="dot",
        line_color=T["text_dim"],
        line_width=1,
        annotation_text=f"μ={mean_val:.3f}",
        annotation_font_color=T["text_dim"],
        annotation_font_size=9,
    )

    fig.update_layout(
        **_layout(height=height, title=_title_style(title or metric)),
        xaxis_title="Orden UCE",
        yaxis_title=metric,
        yaxis_type="log" if use_log else "linear",
        showlegend=True,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BIVARIATE DOTPLOT WITH MARGINAL HISTOGRAMS
# ─────────────────────────────────────────────────────────────────────────────


def bivariate_dotplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    height: int = 480,
) -> go.Figure:
    """
    Scatter + OLS regression line + marginal histograms (top / right).
    Colour by a categorical column when provided.

    Uses a 2×2 subplot grid: [hist_x, empty] / [scatter, hist_y].
    """
    df = df.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        return go.Figure(layout=_layout(height=height, title=_title_style(title)))

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        column_widths=[0.82, 0.18],
        row_heights=[0.18, 0.82],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # ── marginal X histogram (top-left) ──────────────────────────────────────
    fig.add_trace(
        go.Histogram(
            x=df[x_col],
            nbinsx=30,
            marker_color=T["accent"],
            opacity=0.7,
            showlegend=False,
            hovertemplate="%{x:.3f}: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # ── scatter (bottom-left) ─────────────────────────────────────────────────
    if color_col and color_col in df.columns:
        cats = df[color_col].astype(str).unique()
        for i, cat in enumerate(sorted(cats)):
            sub = df[df[color_col].astype(str) == cat]
            fig.add_trace(
                go.Scatter(
                    x=sub[x_col],
                    y=sub[y_col],
                    mode="markers",
                    marker=dict(size=5, color=_CAT[i % len(_CAT)], opacity=0.65),
                    name=f"Cl.{cat}",
                    hovertemplate=f"Cl.{cat}<br>{x_col}=%{{x:.3f}}<br>{y_col}=%{{y:.3f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=dict(size=5, color=T["accent"], opacity=0.55),
                name="UCE",
                hovertemplate=f"{x_col}=%{{x:.3f}}<br>{y_col}=%{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # ── OLS regression line ───────────────────────────────────────────────────
    r = _safe_pearsonr(df[x_col].values, df[y_col].values)
    if not math.isnan(r) and len(df) >= 3:
        slope, intercept = np.polyfit(df[x_col], df[y_col], 1)
        xr = np.array([df[x_col].min(), df[x_col].max()])
        fig.add_trace(
            go.Scatter(
                x=xr,
                y=slope * xr + intercept,
                mode="lines",
                line=dict(dash="dash", width=1.8, color=T["accent2"]),
                name=f"OLS  r={r:.2f}",
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    # ── marginal Y histogram (bottom-right) ────────────────────────────────
    fig.add_trace(
        go.Histogram(
            y=df[y_col],
            nbinsy=30,
            marker_color=T["accent"],
            opacity=0.7,
            showlegend=False,
            hovertemplate="%{y:.3f}: %{x}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        **_layout(height=height, title=_title_style(title)),
        bargap=0.05,
    )
    fig.update_xaxes(title_text=x_col, row=2, col=1)
    fig.update_yaxes(title_text=y_col, row=2, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STACKED ZIPF AREAS  –  frequency band evolution per UCE
# ─────────────────────────────────────────────────────────────────────────────

_ZIPF_PALETTE = {
    "B1_nuclear": "#7c8cff",
    "B2_alta": "#bd93f9",
    "B3_media": "#50fa7b",
    "B4_baja": "#ffb86c",
    "B5_rara_tecnica": "#f97b6b",
    "B_oov": "#ff5555",
}

_ZIPF_ORDER = [
    "B5_rara_tecnica",
    "B4_baja",
    "B3_media",
    "B2_alta",
    "B1_nuclear",
    "B_oov",
]


def stacked_zipf_areas(
    df_bands_por_uce: pd.DataFrame,
    band_cols: Optional[List[str]] = None,
    height: int = 260,
) -> go.Figure:
    """
    Stacked area chart: x = UCE order, y = % tokens per Zipf band.

    df_bands_por_uce: rows = UCEs (in order), columns = band names with % values.
    band_cols: which columns to stack. If None, auto-detect from _ZIPF_ORDER.
    """
    if band_cols is None:
        band_cols = [c for c in _ZIPF_ORDER if c in df_bands_por_uce.columns]
    if not band_cols:
        return go.Figure(layout=_layout(height=height))

    fig = go.Figure()
    x = np.arange(len(df_bands_por_uce))

    for band in band_cols:
        if band not in df_bands_por_uce.columns:
            continue
        y = df_bands_por_uce[band].fillna(0).values
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=band,
                stackgroup="one",
                mode="lines",
                line=dict(width=0.5, color=_ZIPF_PALETTE.get(band, T["text_dim"])),
                fillcolor=_ZIPF_PALETTE.get(band, T["text_dim"])
                .replace("#", "rgba(")
                .replace(")", ",0.65)")
                if "#" in _ZIPF_PALETTE.get(band, "")
                else T["text_dim"],
                hovertemplate=f"{band}: %{{y:.1f}}%  UCE %{{x}}<extra></extra>",
            )
        )

    fig.update_layout(
        **_layout(
            height=height, title=_title_style("Evolución de bandas de frecuencia Zipf")
        ),
        xaxis_title="Orden de UCE",
        yaxis_title="% tokens",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OOV TREEMAP
# ─────────────────────────────────────────────────────────────────────────────


def oov_treemap(df_oov: pd.DataFrame, height: int = 420) -> go.Figure:
    """
    Treemap of OOV lemmas.  Expects columns: lemma, freq_abs, n_docs.
    """
    if df_oov.empty:
        return go.Figure(layout=_layout(height=height))

    df_oov = df_oov.copy()
    df_oov["label"] = df_oov["lemma"] + "<br>" + df_oov["freq_abs"].astype(str) + "×"

    fig = px.treemap(
        df_oov,
        path=["lemma"],
        values="freq_abs",
        color="n_docs",
        color_continuous_scale=["#161929", "#7c8cff"],
        hover_data={"freq_abs": True, "n_docs": True},
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{value} occ.",
        hovertemplate="<b>%{label}</b><br>frecuencia: %{value}<br>documentos: %{color:.0f}<extra></extra>",
        marker_line_color=T["border"],
        marker_line_width=1.5,
        textfont=dict(family="IBM Plex Mono", size=11),
    )
    fig.update_coloraxes(
        colorbar=dict(
            title="docs",
            tickfont=dict(family="IBM Plex Mono", size=9, color=T["text_dim"]),
            outlinecolor=T["border"],
        )
    )
    fig.update_layout(
        **_layout(height=height, title=_title_style("Palabras OOV – jerga del corpus"))
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VOICE TRANSITION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────


def voice_transition_heatmap(
    alternancia: Dict[str, Any],
    height: int = 340,
) -> go.Figure:
    """
    Renders the alternancia_voz['matriz_transicion'] as an annotated heatmap.

    alternancia: dict returned by CorpusAnalyzer.alternancia_voz()
      Keys used: 'matriz_transicion' (DataFrame), 'n_cambios', 'tasa_cambio'
    """
    mat = alternancia.get("matriz_transicion")
    if mat is None or (isinstance(mat, pd.DataFrame) and mat.empty):
        return go.Figure(layout=_layout(height=height))

    if isinstance(mat, dict):
        mat = pd.DataFrame(mat)

    # Normalise rows → conditional probabilities
    row_sums = mat.sum(axis=1).replace(0, 1)
    prob = mat.div(row_sums, axis=0)

    z = prob.values
    x_labels = list(prob.columns)
    y_labels = list(prob.index)

    # Annotation text = raw count (mat) on top of probability colour
    text = [
        [f"{mat.iloc[r, c]}<br>{z[r, c]:.2f}" for c in range(len(x_labels))]
        for r in range(len(y_labels))
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, T["surface"]], [0.5, "#4a5270"], [1, T["accent"]]],
            zmin=0,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono", size=10),
            hovertemplate="<b>%{y} → %{x}</b><br>prob: %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title="P(transición)",
                tickfont=dict(family="IBM Plex Mono", size=9, color=T["text_dim"]),
                outlinecolor=T["border"],
            ),
        )
    )

    n_cambios = alternancia.get("n_cambios", "?")
    tasa = alternancia.get("tasa_cambio", 0)
    subtitle = f"  n_cambios={n_cambios}  |  tasa={tasa:.3f}"

    fig.update_layout(
        **_layout(
            height=height,
            title=_title_style(f"Transición entre voces verbales{subtitle}"),
        ),
        xaxis=dict(title="Voz siguiente", **_plot_defaults["xaxis"]),
        yaxis=dict(title="Voz anterior", **_plot_defaults["yaxis"]),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CROSSTAB HEATMAP  (modo×aspecto, tiempo×voz)
# ─────────────────────────────────────────────────────────────────────────────


def crosstab_heatmap(
    cross_df: pd.DataFrame,
    title: str = "",
    normalize: bool = True,
    height: int = 360,
) -> go.Figure:
    """
    Annotated heatmap from a contingency DataFrame.
    cross_df: index = cat1, columns = cat2, values = counts.
    normalize: show row-conditional probabilities as colours, raw counts as text.
    """
    if cross_df.empty:
        return go.Figure(layout=_layout(height=height))

    if normalize:
        row_sums = cross_df.sum(axis=1).replace(0, 1)
        z_df = cross_df.div(row_sums, axis=0)
    else:
        z_df = cross_df.astype(float)

    z = z_df.values
    text = [
        [str(int(cross_df.iloc[r, c])) for c in range(cross_df.shape[1])]
        for r in range(cross_df.shape[0])
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(cross_df.columns),
            y=list(cross_df.index),
            colorscale=[[0, T["surface"]], [0.4, "#3d4875"], [1, T["accent"]]],
            text=text,
            texttemplate="<b>%{text}</b>",
            textfont=dict(family="IBM Plex Mono", size=11),
            hovertemplate="<b>%{y} · %{x}</b><br>n=%{text}  p=%{z:.3f}<extra></extra>",
            colorbar=dict(
                tickfont=dict(family="IBM Plex Mono", size=9, color=T["text_dim"]),
                outlinecolor=T["border"],
            ),
        )
    )
    fig.update_layout(**_layout(height=height, title=_title_style(title)))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PREDICATE CLUSTER BAR
# ─────────────────────────────────────────────────────────────────────────────


def predicate_cluster_bar(
    df_frames: pd.DataFrame,
    height: int = 320,
) -> go.Figure:
    """
    Horizontal bar chart: one bar per predicate cluster, sized by frame count.
    Expects columns: cluster_id, cluster_label (optional), n_frames or will count rows.
    """
    if df_frames.empty:
        return go.Figure(layout=_layout(height=height))

    if "n_frames" in df_frames.columns:
        agg = df_frames[["cluster_id", "cluster_label", "n_frames"]].copy()
    else:
        agg = df_frames.groupby("cluster_id").size().reset_index(name="n_frames")
        if "cluster_label" in df_frames.columns:
            labels = df_frames.drop_duplicates("cluster_id")[
                ["cluster_id", "cluster_label"]
            ]
            agg = agg.merge(labels, on="cluster_id", how="left")
        else:
            agg["cluster_label"] = "Cl." + agg["cluster_id"].astype(str)

    agg = agg.sort_values("n_frames", ascending=True)
    colors = [_CAT[i % len(_CAT)] for i in range(len(agg))]

    label_col = "cluster_label" if "cluster_label" in agg.columns else "cluster_id"
    fig = go.Figure(
        go.Bar(
            x=agg["n_frames"],
            y=agg[label_col].astype(str),
            orientation="h",
            marker=dict(color=colors, line=dict(color=T["border"], width=0.5)),
            text=agg["n_frames"],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10, color=T["text_dim"]),
            hovertemplate="<b>%{y}</b><br>frames: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout(height=height, title=_title_style("Marcos predicativos por clúster")),
        xaxis_title="n frames",
        yaxis_title="",
        bargap=0.28,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PREDICATE SUNBURST  (cluster → voz → rol temático)
# ─────────────────────────────────────────────────────────────────────────────


def predicate_sunburst(
    df_frames: pd.DataFrame,
    height: int = 460,
) -> go.Figure:
    """
    Three-level sunburst: cluster_label → voice → thematic_role.
    Expects these columns in df_frames (flat frame list).
    """
    req = {"cluster_label", "voice", "thematic_role"}
    missing = req - set(df_frames.columns)
    if missing or df_frames.empty:
        return go.Figure(layout=_layout(height=height))

    agg = (
        df_frames.groupby(["cluster_label", "voice", "thematic_role"])
        .size()
        .reset_index(name="count")
    )

    fig = px.sunburst(
        agg,
        path=["cluster_label", "voice", "thematic_role"],
        values="count",
        color="voice",
        color_discrete_sequence=_CAT,
    )
    fig.update_traces(
        textfont=dict(family="IBM Plex Mono", size=10),
        insidetextorientation="horizontal",
        hovertemplate="<b>%{label}</b><br>%{value} frames<br>%{percentParent:.1%} del padre<extra></extra>",
        marker=dict(line=dict(color=T["bg_page"], width=1.5)),
    )
    fig.update_layout(
        **_layout(height=height, title=_title_style("Jerarquía de marcos predicativos"))
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10.  PREDICATE PARALLEL COORDINATES
# ─────────────────────────────────────────────────────────────────────────────


def predicate_parallel_coordinates(
    df_frames: pd.DataFrame,
    height: int = 500,
) -> go.Figure:
    """
    Parallel coordinates for predicate frames.
    Dimensions: cluster_id, verb_lemma, voice, thematic_role, negated.
    """
    needed = {"cluster_id", "verb_lemma", "voice", "thematic_role", "negated"}
    if not needed.issubset(df_frames.columns) or df_frames.empty:
        return go.Figure(layout=_layout(height=height))

    df_p = df_frames[list(needed)].dropna().copy()
    le_verb = LabelEncoder()
    le_voice = LabelEncoder()
    le_role = LabelEncoder()

    df_p["verb_code"] = le_verb.fit_transform(df_p["verb_lemma"].astype(str))
    df_p["voice_code"] = le_voice.fit_transform(df_p["voice"].astype(str))
    df_p["role_code"] = le_role.fit_transform(df_p["thematic_role"].astype(str))
    df_p["neg_code"] = df_p["negated"].astype(int)

    cid = df_p["cluster_id"].astype(int)

    dims = [
        dict(
            label="Clúster",
            values=cid,
            range=[cid.min(), cid.max()],
        ),
        dict(
            label="Verbo",
            values=df_p["verb_code"],
            tickvals=list(range(len(le_verb.classes_))),
            ticktext=list(le_verb.classes_),
        ),
        dict(
            label="Voz",
            values=df_p["voice_code"],
            tickvals=list(range(len(le_voice.classes_))),
            ticktext=list(le_voice.classes_),
        ),
        dict(
            label="Rol temático",
            values=df_p["role_code"],
            tickvals=list(range(len(le_role.classes_))),
            ticktext=list(le_role.classes_),
        ),
        dict(
            label="Negado",
            values=df_p["neg_code"],
            tickvals=[0, 1],
            ticktext=["No", "Sí"],
            range=[-0.1, 1.1],
        ),
    ]

    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=cid,
                colorscale=[[i / (len(_CAT) - 1), c] for i, c in enumerate(_CAT)],
                showscale=True,
                colorbar=dict(
                    title="Clúster",
                    tickfont=dict(family="IBM Plex Mono", size=9, color=T["text_dim"]),
                    outlinecolor=T["border"],
                ),
            ),
            dimensions=dims,
            labelfont=dict(family="IBM Plex Mono", size=10, color=T["text_mid"]),
            tickfont=dict(family="IBM Plex Mono", size=9, color=T["text_dim"]),
        )
    )
    fig.update_layout(
        **_layout(
            height=height,
            title=_title_style("Coordenadas paralelas – marcos predicativos"),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 11.  CHAIN SUMMARY TABLE  (HTML inline bars)
# ─────────────────────────────────────────────────────────────────────────────


def render_chain_summary_table(
    chain_summary_df: pd.DataFrame,
    top_n: int = 12,
) -> None:
    """
    Renders an HTML table with inline bar sparklines directly into Streamlit.
    Columns shown: entity, n_frames, agent_ratio, patient_ratio, dominant_role, verbs.
    """
    if chain_summary_df is None or (
        isinstance(chain_summary_df, list) and not chain_summary_df
    ):
        st.caption("Sin datos de cadenas.")
        return
    if isinstance(chain_summary_df, list):
        df = pd.DataFrame(chain_summary_df)
    else:
        df = chain_summary_df.copy()

    if df.empty:
        st.caption("Sin datos de cadenas.")
        return

    needed = {"entity", "n_frames"}
    if not needed.issubset(df.columns):
        st.dataframe(df.head(top_n))
        return

    df = df.nlargest(top_n, "n_frames")
    max_frames = df["n_frames"].max() or 1

    rows_html = []
    for _, row in df.iterrows():
        pct = int(row["n_frames"] / max_frames * 100)
        agent = row.get("agent_ratio", 0)
        patient = row.get("patient_ratio", 0)
        role = row.get("dominant_role", "?")
        verbs = row.get("verbs", [])
        verbs_str = ", ".join(verbs[:4]) if isinstance(verbs, list) else str(verbs)
        entity = str(row["entity"])[:38]

        bar_html = (
            f'<div style="background:{T["border"]};border-radius:2px;height:6px;width:100%;">'
            f'<div style="background:{T["accent"]};border-radius:2px;height:6px;width:{pct}%;"></div>'
            f"</div>"
        )
        rows_html.append(f"""
        <tr>
          <td style="font-family:IBM Plex Mono,monospace;font-size:11px;color:{T["text_mid"]};
                     white-space:nowrap;padding:5px 8px;" title="{row["entity"]}">{entity}</td>
          <td style="text-align:right;padding:5px 8px;font-size:11px;
                     color:{T["accent"]};font-family:IBM Plex Mono,monospace;">{row["n_frames"]}</td>
          <td style="padding:5px 12px;width:90px;">{bar_html}</td>
          <td style="text-align:center;font-size:10px;color:{T["text_dim"]};
                     padding:5px 6px;font-family:IBM Plex Mono,monospace;">{agent:.2f}</td>
          <td style="text-align:center;font-size:10px;color:{T["text_dim"]};
                     padding:5px 6px;font-family:IBM Plex Mono,monospace;">{patient:.2f}</td>
          <td style="font-size:10px;color:{T["accent3"]};
                     padding:5px 6px;font-family:IBM Plex Mono,monospace;">{role}</td>
          <td style="font-size:9px;color:{T["text_dim"]};
                     padding:5px 6px;font-family:IBM Plex Mono,monospace;max-width:160px;
                     white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{verbs_str}</td>
        </tr>
        """)

    header_style = (
        f"background:{T['surface']};color:{T['muted']};font-size:9px;"
        f"text-transform:uppercase;letter-spacing:1px;padding:6px 8px;"
        f"font-family:IBM Plex Mono,monospace;border-bottom:1px solid {T['border']};"
    )
    html = f"""
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;
                  background:{T["bg_page"]};border:1px solid {T["border"]};border-radius:6px;
                  overflow:hidden;">
      <thead>
        <tr>
          <th style="{header_style}">Entidad</th>
          <th style="{header_style}text-align:right;">Frames</th>
          <th style="{header_style}"></th>
          <th style="{header_style}text-align:center;">Agente</th>
          <th style="{header_style}text-align:center;">Paciente</th>
          <th style="{header_style}">Rol dom.</th>
          <th style="{header_style}">Verbos</th>
        </tr>
      </thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 12.  DISCOURSE COOCCURRENCE NETWORK
# ─────────────────────────────────────────────────────────────────────────────


def discourse_cooccurrence_network(
    annotations_by_uce: Dict[str, List[Dict]],
    trait_field: str = "label",
    min_cooc: int = 2,
    height: int = 440,
) -> go.Figure:
    """
    Builds a co-occurrence graph of discourse traits across UCEs.

    annotations_by_uce: {uce_id: [annotation_dict, ...]}
    trait_field: which field in the annotation dict to use as the node label.
    min_cooc: minimum co-occurrence count to draw an edge.
    """
    # Build co-occurrence counter
    cooc: Counter = Counter()
    trait_freq: Counter = Counter()

    for uce_id, anns in annotations_by_uce.items():
        traits = list(
            {str(a.get(trait_field, "?")) for a in anns if a.get(trait_field)}
        )
        trait_freq.update(traits)
        for a, b in combinations(sorted(traits), 2):
            cooc[(a, b)] += 1

    edges = [(a, b, w) for (a, b), w in cooc.items() if w >= min_cooc]
    if not edges:
        fig = go.Figure(
            layout=_layout(height=height, title=_title_style("Red de co-ocurrencia"))
        )
        fig.add_annotation(
            text="Datos insuficientes (min_cooc no alcanzado)",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=T["text_dim"]),
        )
        return fig

    G = nx.Graph()
    for a, b, w in edges:
        G.add_edge(a, b, weight=w)
    for t, freq in trait_freq.items():
        if t in G.nodes:
            G.nodes[t]["freq"] = freq

    pos = nx.spring_layout(G, seed=42, k=0.5, weight="weight")

    # Edge traces – width ∝ co-occurrence
    max_w = max(w for _, _, w in edges)
    edge_traces = []
    for a, b, w in edges:
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=0.8 + 3.5 * (w / max_w), color=T["gradient_hi"]),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Node trace
    nodes = list(G.nodes())
    nx_x = [pos[n][0] for n in nodes]
    nx_y = [pos[n][1] for n in nodes]
    freq_vals = [G.nodes[n].get("freq", 1) for n in nodes]
    size_vals = [8 + 20 * (f / max(freq_vals)) for f in freq_vals]
    degree_vals = [G.degree(n) for n in nodes]

    node_trace = go.Scatter(
        x=nx_x,
        y=nx_y,
        mode="markers+text",
        text=nodes,
        textposition="top center",
        textfont=dict(family="IBM Plex Mono", size=9, color=T["text_mid"]),
        marker=dict(
            size=size_vals,
            color=degree_vals,
            colorscale=[[0, T["gradient_lo"]], [1, T["accent"]]],
            line=dict(width=1, color=T["border"]),
            showscale=True,
            colorbar=dict(
                title="grado",
                tickfont=dict(family="IBM Plex Mono", size=8, color=T["text_dim"]),
                outlinecolor=T["border"],
                thickness=10,
            ),
        ),
        hovertemplate="<b>%{text}</b><br>grado: %{marker.color}<extra></extra>",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        **_layout(
            height=height, title=_title_style("Co-ocurrencia de rasgos discursivos")
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 13.  DISCOURSE SANKEY  (trait transitions between consecutive UCEs)
# ─────────────────────────────────────────────────────────────────────────────


def discourse_sankey(
    annotations_by_uce: Dict[str, List[Dict]],
    uces_ordered: List[Dict],
    trait_field: str = "label",
    top_n_traits: int = 8,
    height: int = 400,
) -> go.Figure:
    """
    Sankey: flow of dominant discourse trait from UCE_i → UCE_{i+1}.

    annotations_by_uce: {uce_id: [annotation_dict, ...]}
    uces_ordered: list of UCE dicts in document order (needs 'id' key).
    trait_field: field in annotation to treat as the trait label.
    top_n_traits: collapse minor traits into "otros".
    """
    # Determine dominant trait per UCE
    uce_trait: List[str] = []
    for uce in uces_ordered:
        anns = annotations_by_uce.get(uce.get("id", ""), [])
        traits = [str(a.get(trait_field, "")) for a in anns if a.get(trait_field)]
        if traits:
            dominant = Counter(traits).most_common(1)[0][0]
        else:
            dominant = "ninguno"
        uce_trait.append(dominant)

    # Collapse rare traits
    freq = Counter(uce_trait)
    top_traits = {t for t, _ in freq.most_common(top_n_traits)}
    uce_trait = [t if t in top_traits else "otros" for t in uce_trait]

    # Build transition counts
    trans: Counter = Counter()
    for i in range(len(uce_trait) - 1):
        trans[(uce_trait[i], uce_trait[i + 1])] += 1

    if not trans:
        return go.Figure(
            layout=_layout(height=height, title=_title_style("Flujo de estrategias"))
        )

    labels = sorted({t for pair in trans for t in pair})
    label_idx = {l: i for i, l in enumerate(labels)}

    source = [label_idx[p[0]] for p in trans]
    target = [label_idx[p[1]] for p in trans]
    value = list(trans.values())
    link_colors = [
        f"rgba({int(_CAT[label_idx[p[0]] % len(_CAT)].lstrip('#')[0:2], 16)},"
        f"{int(_CAT[label_idx[p[0]] % len(_CAT)].lstrip('#')[2:4], 16)},"
        f"{int(_CAT[label_idx[p[0]] % len(_CAT)].lstrip('#')[4:6], 16)},0.45)"
        for p in trans
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                color=[_CAT[i % len(_CAT)] for i in range(len(labels))],
                line=dict(color=T["border"], width=0.5),
                pad=18,
                thickness=16,
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors,
                hovertemplate="<b>%{source.label} → %{target.label}</b><br>n=%{value}<extra></extra>",
            ),
            textfont=dict(family="IBM Plex Mono", size=10, color=T["text_mid"]),
        )
    )
    fig.update_layout(
        **_layout(
            height=height,
            title=_title_style("Flujo de estrategias discursivas entre UCEs"),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 14.  SEMANTIC NETWORK PLOT  (community-coloured, bridge nodes highlighted)
# ─────────────────────────────────────────────────────────────────────────────


def semantic_network_plot(
    network_data: Dict[str, Any],
    height: int = 560,
    max_nodes: int = 120,
) -> go.Figure:
    """
    Renders the GlobalLexicalAnalyzer semantic network stored in stats.

    network_data: dict with keys:
      'top_nodes'   → list of {lemma, degree, pagerank, freq, n_docs}
      'communities' → list of {community_id, top_pagerank: [lemma,...], size}
      'bridge_nodes'→ list of {lemma, betweenness, community_id}

    Reconstructs a lightweight graph from the top_nodes co-occurrence data.
    For a full render, pass the nx.Graph directly via `network_data['_graph']`.
    """
    G_raw = network_data.get("_graph")
    if G_raw is not None and isinstance(G_raw, nx.Graph):
        G = G_raw
    else:
        # Reconstruct minimal graph from top_nodes AND edges
        top_nodes_raw = network_data.get("top_nodes", [])
        edges_raw = network_data.get("edges", [])

        if not top_nodes_raw:
            return go.Figure()

        df_nodes = pd.DataFrame(top_nodes_raw)
        G = nx.Graph()

        # 1. Add Nodes
        for _, row in df_nodes.head(max_nodes).iterrows():
            G.add_node(
                row["lemma"],
                freq=row.get("freq", 1),
                n_docs=row.get("n_docs", 1),
                pagerank=row.get("pagerank", 0.0),
                community_id=row.get("community_id", 0),  # Grab it directly
            )

        # 2. Add Edges (Only keep edges where BOTH nodes are in our top_nodes subset)
        valid_nodes = set(G.nodes())
        for edge in edges_raw:
            if edge["source"] in valid_nodes and edge["target"] in valid_nodes:
                G.add_edge(
                    edge["source"], edge["target"], weight=edge.get("weight", 1.0)
                )

    # Community assignments
    communities_raw = network_data.get("communities", [])
    node_community: Dict[str, int] = {}
    if communities_raw:
        for comm in communities_raw:
            cid = comm.get("community_id", 0)
            for lemma in comm.get("top_pagerank", []):
                node_community[lemma] = cid

    bridge_set = {b["lemma"] for b in network_data.get("bridge_nodes", [])}

    # Limit to top max_nodes by pagerank / degree
    nodes_to_show = list(G.nodes())
    if len(nodes_to_show) > max_nodes:
        pr = (
            nx.pagerank(G, weight="weight")
            if G.number_of_edges() > 0
            else {n: 1 for n in G.nodes()}
        )
        nodes_to_show = sorted(pr, key=pr.get, reverse=True)[:max_nodes]
        G = G.subgraph(nodes_to_show).copy()

    pos = nx.spring_layout(
        G, seed=42, k=0.35, weight="weight" if G.number_of_edges() > 0 else None
    )

    # Edge traces
    edge_traces = []
    if G.number_of_edges() > 0:
        max_ew = max((d.get("weight", 1) for _, _, d in G.edges(data=True)), default=1)
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = d.get("weight", 1) / max_ew
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=0.4 + 2.5 * w, color=T["gradient_hi"]),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

    # Node trace
    nodes = list(G.nodes())
    if not nodes:
        return go.Figure(
            layout=_layout(
                height=height, title=_title_style("Red semántica (sin datos)")
            )
        )

    nx_x = [pos[n][0] for n in nodes]
    nx_y = [pos[n][1] for n in nodes]
    colors = [_CAT[G.nodes[n].get("community_id", 0) % len(_CAT)] for n in nodes]
    border_colors = [T["accent2"] if n in bridge_set else T["border"] for n in nodes]
    border_widths = [2.5 if n in bridge_set else 0.8 for n in nodes]

    deg = {n: G.degree(n) for n in nodes}
    max_deg = max(deg.values(), default=1)
    sizes = [7 + 18 * (deg[n] / max_deg) for n in nodes]

    node_trace = go.Scatter(
        x=nx_x,
        y=nx_y,
        mode="markers+text",
        text=nodes,
        textposition="top center",
        textfont=dict(family="IBM Plex Mono", size=8, color=T["text_dim"]),
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(color=border_colors, width=border_widths),
        ),
        hovertemplate="<b>%{text}</b><br>grado: %{marker.size:.0f}<extra></extra>",
        showlegend=False,
    )

    # Bridge node annotations
    annotations = [
        dict(
            x=pos[n][0],
            y=pos[n][1],
            text=f"◆ {n}",
            font=dict(family="IBM Plex Mono", size=9, color=T["accent2"]),
            showarrow=False,
            yshift=14,
        )
        for n in bridge_set
        if n in pos
    ]

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        **_layout(
            height=height,
            title=_title_style("Red semántica – comunidades y nodos puente"),
            annotations=annotations,
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest",
    )
    return fig


@st.cache_data(show_spinner=False)
def _compute_modalization(_clusters_key: tuple, _n_uces: int):
    MARKER_CATEGORIES = {
        "deóntica": [
            "hay que",
            "se debe",
            "tenemos que",
            "es necesario",
            "debe",
            "deben",
        ],
        "epist. alta": [
            "es evidente",
            "está claro",
            "sin duda",
            "evidentemente",
            "por supuesto",
        ],
        "epist. baja": [
            "creo",
            "me parece",
            "quizás",
            "a lo mejor",
            "supongo",
            "tal vez",
        ],
        "hedges": ["más o menos", "como que", "digamos", "tipo", "algo así"],
        "amplif.": ["muy", "totalmente", "absolutamente", "jamás", "siempre"],
        "atenua.": ["un poco", "relativamente", "bastante", "algo"],
        "polifonía": ["dijo que", "dicen que", "según", "me dijeron", "se dice"],
    }
    class_counts = {
        str(c): {cat: 0 for cat in MARKER_CATEGORIES} for c in _clusters_key
    }
    class_totals = {str(c): 0 for c in _clusters_key}
    for uce in uces:
        cid = uce.get("cluster_id")
        if cid is None or cid < 0:
            continue
        k = str(int(cid))
        if k not in class_totals:
            continue
        txt = uce.get("texto", "").lower()
        class_totals[k] += len(txt.split())
        for cat, markers in MARKER_CATEGORIES.items():
            for m in markers:
                class_counts[k][cat] += txt.count(m)
    return class_counts, class_totals


def make_modalization_radar(cluster_id: int, _dm_key: bool):
    MARKER_CATEGORIES = {
        "deóntica": [
            "hay que",
            "se debe",
            "tenemos que",
            "es necesario",
            "debe",
            "deben",
        ],
        "epist. alta": [
            "es evidente",
            "está claro",
            "sin duda",
            "evidentemente",
            "por supuesto",
        ],
        "epist. baja": [
            "creo",
            "me parece",
            "quizás",
            "a lo mejor",
            "supongo",
            "tal vez",
        ],
        "hedges": ["más o menos", "como que", "digamos", "tipo", "algo así"],
        "amplif.": ["muy", "totalmente", "absolutamente", "jamás", "siempre"],
        "atenua.": ["un poco", "relativamente", "bastante", "algo"],
        "polifonía": ["dijo que", "dicen que", "según", "me dijeron", "se dice"],
    }
    categories = list(MARKER_CATEGORIES.keys())
    class_counts, class_totals = _compute_modalization(
        tuple(clusters_unicos), len(uces)
    )
    fig = go.Figure()
    for c in clusters_unicos:
        col = class_colors.get(c, "#AAA")
        k = str(c)
        total_words = max(class_totals.get(k, 1), 1)
        rates = [
            class_counts.get(k, {}).get(cat, 0) / total_words * 1000
            for cat in categories
        ]
        max_rate = max(rates) if max(rates) > 0 else 1
        rates_norm = [r / max_rate for r in rates]
        vc = rates_norm + [rates_norm[0]]
        tc = categories + [categories[0]]
        is_active = c == cluster_id
        fig.add_trace(
            go.Scatterpolar(
                r=vc,
                theta=tc,
                fill="toself",
                name=f"Clase {c}",
                line=dict(color=col, width=2.5 if is_active else 1),
                fillcolor=hex_to_rgba(col, 0.35 if is_active else 0.08),
                opacity=1.0 if is_active else 0.5,
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor=T["bg_card"],
            radialaxis=dict(
                visible=True,
                gridcolor=T["border2"],
                color=T["text_low"],
                tickfont=dict(size=8),
                range=[0, 1],
            ),
            angularaxis=dict(
                gridcolor=T["border2"], color=T["text_mid"], tickfont=dict(size=10)
            ),
        ),
        height=340,
        margin=dict(l=28, r=28, t=18, b=64),
        legend=dict(
            orientation="h",
            y=-0.22,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        **_plot_defaults(),
    )
    return fig


def make_cah_dendrogram(cluster_id: int, _dm_key: bool):
    from scipy.cluster.hierarchy import dendrogram as _dendrogram

    key = str(cluster_id)
    if key not in cah_por_clase:
        return None
    cah = cah_por_clase[key]
    Z_raw = cah.get("Z", [])
    leaf_labels = cah.get("labels", [])
    if not Z_raw or not leaf_labels or len(leaf_labels) < 2:
        return None
    Z = np.array(Z_raw, dtype=float)
    try:
        ddata = _dendrogram(
            Z, labels=leaf_labels, no_plot=True, orientation="left", color_threshold=0
        )
    except Exception:
        return None
    n = len(ddata["ivl"])
    col_main = class_colors.get(cluster_id, T["accent"])

    def _leaf_color(label: str) -> str:
        m = re.search(r"φ=([−\-]?\d+\.\d+)", label)
        if m:
            try:
                phi_val = float(m.group(1).replace("−", "-"))
                return col_main if phi_val >= 0 else "#E86450"
            except ValueError:
                pass
        return T["text_mid"]

    fig = go.Figure()
    for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
        fig.add_trace(
            go.Scatter(
                x=ys,
                y=xs,
                mode="lines",
                line=dict(color=col_main, width=1),
                hoverinfo="none",
                showlegend=False,
            )
        )
    leaf_positions = {}
    for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
        for x, y in zip(xs, ys):
            if y == 0.0:  # y==0 in dcoord means it's a leaf connection
                leaf_positions[x] = x  # x in icoord IS the y-axis position
    # Map ivl labels to their actual y positions
    for label, y in zip(ddata["ivl"], sorted(leaf_positions.keys())):
        c = _leaf_color(label)
        display = re.sub(r"\s*\(φ=[^)]+\)", "", label)
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[y],
                mode="markers+text",
                text=[display],
                textposition="middle right",
                textfont=dict(size=8.5, color=c),
                marker=dict(size=4, color=c),
                hovertext=[label],
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        height=max(250, n * 16 + 60),
        margin=dict(l=8, r=8, t=32, b=8),
        xaxis=dict(
            title="Distancia Ward",
            showgrid=True,
            gridcolor=T["border"],
            tickfont=dict(size=8),
            autorange="reversed",
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        title=dict(
            text=f"CAH intra-clase · Clase {cluster_id} · {n} términos",
            font=dict(size=9, color=T["text_low"]),
            x=0.5,
        ),
        **_plot_defaults(),
    )
    return fig


def _md_inline(s: str) -> str:
    s = s.replace("\n", "<br>")
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return s


@st.cache_data(show_spinner=False)
def build_isotopy_data(
    _terminos_df,
    _uces,
    _uce_phi_dict,
    _clusters_unicos,
    _class_colors,
    _sintesis_estructurada,
    top_terms=8,
    top_uces=5,
):
    per_class = _sintesis_estructurada.get("per_class", {})
    lbl_props = _sintesis_estructurada.get("label_proposals", {})
    out = []
    for c in _clusters_unicos:
        col = _class_colors.get(c, "#888")
        df_c = (
            _terminos_df[_terminos_df["cluster"] == c].copy()
            if not _terminos_df.empty
            else pd.DataFrame()
        )
        pos = (
            df_c[df_c["phi"] > 0].sort_values("phi", ascending=False)
            if not df_c.empty and "phi" in df_c.columns
            else pd.DataFrame()
        )
        neg = (
            df_c[df_c["phi"] < 0].sort_values("phi", ascending=True)
            if not df_c.empty and "phi" in df_c.columns
            else pd.DataFrame()
        )
        all_pos = pos["termino"].tolist() if not pos.empty else []
        chunk = max(1, len(all_pos) // 3)
        slices = [all_pos[:chunk], all_pos[chunk : chunk * 2], all_pos[chunk * 2 :]]
        fb_names = [
            "Campo semántico primario",
            "Campo semántico secundario",
            "Campo semántico terciario",
        ]
        named = _sintesis_estructurada.get("isotopias", {}).get(
            str(c), []
        ) or _sintesis_estructurada.get("isotopias", {}).get(c, [])
        iso_g = []
        for idx, (it, fn) in enumerate(zip(slices, fb_names)):
            if not it:
                continue
            nm = named[idx].get("nombre", fn) if idx < len(named) else fn
            it = (
                named[idx].get("terminos", it[:top_terms])
                if idx < len(named)
                else it[:top_terms]
            )
            ft = named[idx].get("funcion_discursiva", "") if idx < len(named) else ""
            mc = []
            for u in _uces:
                if u.get("cluster_id") != c:
                    continue
                ll = {l.lower() for l in u.get("lemmas", [])}
                tl = u.get("texto", "").lower()
                hits = [t for t in it if t.lower() in ll or t.lower() in tl]
                if hits:
                    mc.append(
                        {
                            "id": u["id"][:12],
                            "phi": round(_uce_phi_dict.get(u["id"], 0.0), 3),
                            "txt": u.get("texto", ""),
                            "hits": hits,
                        }
                    )
            mc.sort(key=lambda x: x["phi"], reverse=True)
            iso_g.append({"name": nm, "terms": it, "fn": ft, "uces": mc[:top_uces]})
        abs_t = neg["termino"].tolist()[:12] if not neg.empty else []
        abs_d = (
            _sintesis_estructurada.get("ausencias", {})
            .get(str(c), {})
            .get("descripcion", "")
            or "Términos con φ negativo."
        )
        mets_r = _sintesis_estructurada.get("metaforas", {}).get(
            str(c), []
        ) or _sintesis_estructurada.get("metaforas", {}).get(c, [])
        met_g = []
        for m in mets_r:
            formula = m.get("formato_lakoffiano", m.get("formula", ""))
            desc = m.get("evidencia_uce_o_termino", m.get("descripcion", ""))
            ev_t = [t.strip() for t in re.split(r"[,;]", desc) if len(t.strip()) > 3][
                :4
            ]
            mu = []
            for u in _uces:
                if u.get("cluster_id") != c:
                    continue
                hits = [t for t in ev_t if t.lower() in u.get("texto", "").lower()]
                if hits:
                    mu.append(
                        {
                            "id": u["id"][:12],
                            "phi": round(_uce_phi_dict.get(u["id"], 0.0), 3),
                            "txt": u.get("texto", ""),
                            "hits": hits,
                        }
                    )
            mu.sort(key=lambda x: x["phi"], reverse=True)
            met_g.append({"formula": formula, "desc": desc, "uces": mu[:top_uces]})
        obj_r = _sintesis_estructurada.get("objetivacion", {}).get(
            str(c), {}
        ) or _sintesis_estructurada.get("objetivacion", {}).get(c, {})
        anc_r = _sintesis_estructurada.get("anclaje", {}).get(
            str(c), {}
        ) or _sintesis_estructurada.get("anclaje", {}).get(c, {})
        lbls = _sintesis_estructurada.get("etiquetas", {}).get(
            str(c), []
        ) or _sintesis_estructurada.get("etiquetas", {}).get(c, [])
        if not lbls:
            for line in lbl_props.get(str(c), lbl_props.get(c, "")).split("\n"):
                line = line.strip().lstrip("•·-").strip()
                if line:
                    lbls.append(
                        {
                            "nombre_propuesto": line,
                            "tipo_enfasis": "",
                            "justificacion": "",
                        }
                    )
        out.append(
            {
                "id": c,
                "color": col,
                "n_uces": sum(1 for u in _uces if u.get("cluster_id") == c),
                "tone": _sintesis_estructurada.get("tono", {}).get(str(c), "")
                or _sintesis_estructurada.get("tono", {}).get(c, ""),
                "hyp": per_class.get(str(c), per_class.get(c, "")),
                "iso": iso_g,
                "abs_terms": abs_t,
                "abs_desc": abs_d,
                "met": met_g,
                "obj": obj_r,
                "anc": anc_r,
                "lbls": lbls,
                "tensions": _sintesis_estructurada.get("tensiones", {}).get(str(c), [])
                or _sintesis_estructurada.get("tensiones", {}).get(c, []),
                "limits": _sintesis_estructurada.get("limites", {}).get(str(c), [])
                or _sintesis_estructurada.get("limites", {}).get(c, []),
            }
        )
    return out


@st.cache_data(show_spinner=False)
def _build_isotopy_cached(
    terminos_json: str,
    uces_json: str,
    phi_json: str,
    clusters_key: tuple,
    colors_key: tuple,
    sintesis_json: str,
    top_terms=8,
    top_uces=5,
):
    _terminos_df = pd.DataFrame(json.loads(terminos_json))
    _uces = json.loads(uces_json)
    _uce_phi_dict = json.loads(phi_json)
    _clusters_unicos = list(clusters_key)
    _class_colors = dict(zip(clusters_key, colors_key))
    _sintesis_estructurada = json.loads(sintesis_json)
    return build_isotopy_data(
        _terminos_df,
        _uces,
        _uce_phi_dict,
        _clusters_unicos,
        _class_colors,
        _sintesis_estructurada,
        top_terms,
        top_uces,
    )


def build_global_data(_se, _cu, _cc):
    opp1 = _se.get("oposicion_principal", {})
    opp2 = _se.get("oposicion_secundaria", {})
    op = _se.get("opposition_analysis", "")
    if not opp1 and op:
        opp1 = {"formulacion": "Oposición principal (AFC)", "justificacion": op[:600]}
    mapa = _se.get("mapa_posiciones_discursivas", []) or [
        {"clase": str(c), "posicion_y_distincion": "", "clase_en_tension": ""}
        for c in _cu
    ]
    return {
        "opp1": opp1,
        "opp2": opp2,
        "mapa": mapa,
        "hyp": _se.get("hipotesis_estructura_global", "")
        or _se.get("global_synthesis", ""),
        "agenda": _se.get("agenda_validacion", []),
        "valid": _se.get("validation", {}),
    }


iso_classes = _build_isotopy_cached(
    terminos_df.to_json(),
    json.dumps(uces),
    json.dumps(uce_phi_dict),
    tuple(clusters_unicos),
    tuple(class_colors[c] for c in clusters_unicos),
    json.dumps(sintesis_estructurada),
)
global_data = build_global_data(sintesis_estructurada, clusters_unicos, class_colors)


def _build_imap(cd):
    col = cd["color"]
    imap = {}
    shades = [col, col + "DD", col + "AA"]
    for i, iso in enumerate(cd.get("iso", [])):
        for t in iso.get("terms", []):
            imap[t.lower()] = shades[i % 3]
    return imap


def _highlight_text(txt, hits, imap, defcol):
    segs = []
    for h in sorted(set(h.lower() for h in hits), key=len, reverse=True):
        for m in re.finditer(re.escape(h), txt.lower()):
            if not any(m.start() < e and m.end() > s for s, e, _, _ in segs):
                segs.append((m.start(), m.end(), txt[m.start() : m.end()], h))
    segs.sort(key=lambda x: x[0])
    out, cur = "", 0
    for s, e, word, kh in segs:
        out += sh(txt[cur:s])
        col = imap.get(kh, defcol)
        out += (
            f'<span style="background:{col}22;color:{col};border-bottom:1.5px solid {col};'
            f'border-radius:2px;padding:0 2px;font-style:normal;font-weight:500">{sh(word)}</span>'
        )
        cur = e
    out += sh(txt[cur:])
    return out


def _uce_cards_html(ul, imap, col):
    if not ul:
        return f'<div style="padding:14px 0;font-family:var(--font-mono);font-size:11px;color:var(--text-low)">Sin UCEs vinculadas.</div>'
    html = ""
    for u in ul:
        txt = u.get("txt", u.get("texto", ""))
        hits = u.get("hits", [])
        lc = imap.get(hits[0].lower(), col) if hits else col
        body = _highlight_text(txt, hits, imap, col)
        html += (
            f'<div style="border:.5px solid var(--border2);border-radius:var(--r-sm);overflow:hidden;'
            f'border-left:2px solid {lc};margin-bottom:8px;box-shadow:var(--card-shadow)">'
            f'<div style="display:flex;align-items:center;gap:8px;padding:4px 10px;'
            f'background:var(--bg-card);border-bottom:.5px solid var(--border)">'
            f'<span style="font-family:var(--font-mono);font-size:11px;color:var(--text-low)">{sh(u.get("id", "?"))}</span>'
            f'<span style="font-family:var(--font-mono);font-size:11px;padding:1px 7px;border-radius:12px;'
            f'border:.5px solid {lc};color:var(--text-hi)">φ = {u.get("phi", 0):.2f}</span></div>'
            f'<div style="padding:10px 12px;font-family:var(--font-serif);font-size:13px;font-style:italic;'
            f'line-height:1.85;color:var(--text-mid)">"{body}"</div></div>'
        )
    return html


@st.cache_data(show_spinner=False)
def build_lemma_network_cached(
    chosen_lemmas_key: tuple, min_cooc: int, top_n: int, min_degree: int
):
    return build_lemma_network(
        list(chosen_lemmas_key), uces, lemma_map, min_cooc, top_n, min_degree
    )


def unified_kwic_search(uces_list, filter_terms, cluster_id, color):
    if not filter_terms:
        return ""
    rows = []
    for u in uces_list:
        if u.get("cluster_id") != cluster_id:
            continue
        txt = u.get("texto", "")
        uid = u.get("id", "?")[:8]
        phi = uce_phi_dict.get(u.get("id", ""), 0.0)
        for term in filter_terms:
            for m in re.finditer(re.escape(term), txt, re.IGNORECASE):
                left_ctx = txt[max(0, m.start() - 80) : m.start()]
                kw = txt[m.start() : m.end()]
                right_ctx = txt[m.end() : m.end() + 80]
                rows.append((phi, left_ctx, kw, right_ctx, uid, txt))
    if not rows:
        return "<div style='color:var(--text-low); padding: 20px;'>Sin coincidencias KWIC.</div>"
    rows.sort(key=lambda x: x[0], reverse=True)
    html = f"""
    <div style="max-height: 400px; overflow-y: scroll; border: 1px solid var(--border); border-radius: var(--r-sm);">
        <table style="width: 100%; border-collapse: collapse; font-family: var(--font-serif); font-size: 13px;">
            <thead style="position: sticky; top: 0; background: var(--bg-panel); font-family: var(--font-mono); font-size: 10px; text-transform: uppercase; color: var(--text-low); z-index: 1; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                <tr>
                    <th style="padding: 10px; text-align: right; width: 42%;">Contexto Izquierdo</th>
                    <th style="padding: 10px; text-align: center; width: 16%;">Término</th>
                    <th style="padding: 10px; text-align: left; width: 42%;">Contexto Derecho</th>
                </tr>
            </thead>
            <tbody>
    """
    for phi, l, kw, r, uid, full_txt in rows[:100]:
        html += f"""
        <tr style="border-bottom: 1px solid var(--border2); background: var(--bg-card);" title="{sh(full_txt)}">
            <td style="padding: 8px 10px; text-align: right; color: var(--text-mid); direction: rtl; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">...{sh(l)}</td>
            <td style="padding: 8px 10px; text-align: center;"><span style="background:{color}33; color:{color}; padding: 2px 6px; border-radius: 4px; font-weight: bold;">{sh(kw)}</span><br><span style="font-family: var(--font-mono); font-size: 9px; color: var(--text-dim); margin-top:4px; display:inline-block;">φ={phi:.2f}</span></td>
            <td style="padding: 8px 10px; text-align: left; color: var(--text-mid); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{sh(r)}...</td>
        </tr>"""
    html += "</tbody></table></div>"
    return html


def _sec_label(txt, top=False):
    pt = "28px 28px 10px" if top else "18px 28px 10px"
    st.markdown(
        f'<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:.14em;'
        f"text-transform:uppercase;color:var(--text-low);padding:{pt};"
        f'border-bottom:1px solid var(--border)">{sh(txt)}</div>',
        unsafe_allow_html=True,
    )


def _chips(terms, col, neg=False):
    bg = "#2C1014" if neg else "transparent"
    fg = "#E86450" if neg else "var(--text-hi)"
    brd = "#E8645044" if neg else f"{col}55"
    return "".join(
        f'<span style="padding:2px 9px;border-radius:12px;background:{bg};color:{fg};'
        f'border:.5px solid {brd};font-family:var(--font-mono);font-size:11px;margin:2px 3px">'
        f"{sh(t)}</span>"
        for t in terms
    )


def _hyp_block(text, col):
    st.markdown(
        f'<div style="border-left:2px solid {col};padding:12px 16px;background:var(--bg-card);'
        f"border-radius:0 var(--r-sm) var(--r-sm) 0;font-family:var(--font-serif);font-size:13.5px;"
        f'font-style:italic;line-height:1.85;color:var(--text-mid)">{_md_inline(text)}</div>',
        unsafe_allow_html=True,
    )


def _sec_rule(txt):
    st.markdown(
        f'<div class="sec-rule"><span>{sh(txt)}</span></div>', unsafe_allow_html=True
    )


def render_gram_summary_section(filter_class: int = None):
    """
    Renders grammatical summary stats from workflow_data.
    If filter_class is None → shows all classes side by side (for tab B).
    If filter_class is int → shows only that class (for tab C).
    """
    gram = data.get("grammatical_summary_by_class", {})
    if not gram:
        st.info(
            "Sin resumen gramatical. Verifica que el enriquecimiento gramatical haya corrido."
        )
        return

    if filter_class is not None:
        classes_to_show = [str(filter_class)]
    else:
        classes_to_show = [str(c) for c in sorted(class_sizes.keys())]

    # Build comparison DataFrame
    count_fields = [
        "n_negaciones",
        "n_verbos",
        "n_pronombres_exp",
        "n_prodrop",
        "n_marcadores",
        "n_frames",
        "n_cuantificadores",
        "n_adverbios",
        "n_insubordinaciones",
        "n_rarezas",
        "n_subj",
    ]
    mean_fields = [
        "mean_ttr",
        "mean_guiraud",
        "mean_diversidad_semantica",
        "mean_topic_shift",
        "mean_profundidad",
        "mean_recursividad",
        "mean_distancia_dependencia",
        "mean_ratio_subordinacion",
        "mean_branching_ratio",
        "mean_zipf",
        "mean_pct_oov",
        "mean_oral_ratio",
        "mean_academic_ratio",
        "mean_domain_specific_ratio",
        "mean_surprisal",
    ]
    voice_fields = [
        "voice_Act",
        "voice_Pass",
        "voice_PassRefl",
        "voice_Impersonal",
        "voice_Media",
    ]

    rows_count, rows_mean, rows_voice = [], [], []
    for field in count_fields:
        row = {"Métrica": field}
        for k in classes_to_show:
            row[f"Clase {k}"] = gram.get(k, {}).get(field, 0)
        rows_count.append(row)

    for field in mean_fields:
        row = {"Métrica": field}
        for k in classes_to_show:
            v = gram.get(k, {}).get(field, 0.0)
            row[f"Clase {k}"] = round(float(v), 3) if v else 0.0
        rows_mean.append(row)

    for field in voice_fields:
        row = {"Voz": field.replace("voice_", "")}
        for k in classes_to_show:
            row[f"Clase {k}"] = gram.get(k, {}).get(field, 0)
        rows_voice.append(row)

    gc1, gc2 = st.columns(2)
    with gc1:
        st.markdown(
            '<p class="panel-hdr">Conteos gramaticales por clase</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(rows_count).set_index("Métrica"), width="stretch")

        st.markdown(
            '<p class="panel-hdr" style="margin-top:12px">Distribución de voces verbales</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(rows_voice).set_index("Voz"), width="stretch")

    with gc2:
        st.markdown(
            '<p class="panel-hdr">Medias léxicas y sintácticas por clase</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(rows_mean).set_index("Métrica"), width="stretch")

        # Register distribution per class
        reg_rows = []
        for k in classes_to_show:
            regs = gram.get(k, {}).get("registros", [])
            if regs:
                ctr = Counter(regs)
                reg_rows.append({"Clase": f"Clase {k}", **dict(ctr)})
        if reg_rows:
            st.markdown(
                '<p class="panel-hdr" style="margin-top:12px">Distribución de registros</p>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame(reg_rows).set_index("Clase").fillna(0).astype(int),
                width="stretch",
            )


# ─────────────────────────────────────────────────────────────
# ANÁLISIS GRAMATICAL
# ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#13161e",
    margin=dict(l=8, r=8, t=28, b=8),
    hoverlabel=dict(
        bgcolor=T["surface"],
        bordercolor=T["border"],
        font=dict(family="IBM Plex Mono, monospace", size=11, color=T["text_mid"]),
    ),
    xaxis=dict(
        gridcolor=T["border"],
        linecolor=T["border"],
        tickcolor=T["border"],
        zerolinecolor=T["border"],
    ),
    yaxis=dict(
        gridcolor=T["border"],
        linecolor=T["border"],
        tickcolor=T["border"],
        zerolinecolor=T["border"],
    ),
    font=dict(family="DM Mono, monospace", size=11, color="#c8cfe0"),
    legend=dict(
        bgcolor=T["bg_page"], bordercolor=T["border"], font=dict(size=10), borderwidth=1
    ),
    colorway=[
        "#5b9cf6",
        "#2dd4bf",
        "#a78bfa",
        "#fbbf24",
        "#f97b6b",
        "#4ade80",
        "#c084fc",
        "#f472b6",
    ],
)
COLORS = {
    "neg": "#e05c5c",
    "pron": "#3ec9c9",
    "verb_ind": "#5b9cf6",
    "verb_sub": "#a78bfa",
    "verb_imp": "#f97b6b",
    "verb_cond": "#fbbf24",
    "adv": "#c084fc",
    "disc": "#818cf8",
    "quant_uni": "#4ade80",
    "quant_neg": "#f87171",
    "quant_num": "#94a3b8",
    "quant_pro": "#fbbf24",
    "quant_exi": "#2dd4bf",
    "coref": ["#2dd4bf", "#f97b6b", "#a78bfa", "#fbbf24", "#5b9cf6", "#4ade80"],
    "reg": {
        "coloquial": "#fb923c",
        "formal": "#60a5fa",
        "tecnico": "#c084fc",
        "mixto": "#64748b",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading corpus data…")
def load_dashboard(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if is_raw_db_format(data):
        data = adapt_to_dashboard_format(data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def smooth_trajectory(values: tuple, window: int = 7, order: int = 2) -> np.ndarray:
    """Savitzky–Golay smoothing. Accepts a tuple so Streamlit can hash it."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 5:
        return arr
    w = min(window, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w += 1
    w = max(w, 5)
    return savgol_filter(arr, w, min(order, w - 1))


def _cluster_color(cluster_id) -> str:
    try:
        return _CAT[int(cluster_id) % len(_CAT)]
    except (TypeError, ValueError):
        return T["text_dim"]


def _safe_pearsonr(x, y) -> float:
    try:
        if len(x) < 3:
            return float("nan")
        return pearsonr(x, y)[0]
    except Exception:
        return float("nan")


def safe_df(records: Any) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    try:
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


def fmt(val: Any, decimals: int = 2) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return str(val)


def mini_bar(fig: go.Figure) -> go.Figure:
    fig.update_layout(PLOTLY_LAYOUT, height=220)
    return fig


def sparkline(values: List[float], color: str = "#5b9cf6") -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            y=values,
            mode="lines",
            fill="tozeroy",
            line=dict(color=color, width=1.5),
            fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba"),
        )
    )
    fig.update_layout(
        PLOTLY_LAYOUT,
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def horizontal_bar(
    df: pd.DataFrame, x: str, y: str, color: str = "#5b9cf6", title: str = ""
) -> go.Figure:
    df = df.sort_values(x)
    fig = go.Figure(
        go.Bar(
            x=df[x],
            y=df[y],
            orientation="h",
            marker_color=color,
            marker_line_width=0,
        )
    )
    fig.update_layout(PLOTLY_LAYOUT, height=max(160, len(df) * 26 + 40), title=title)
    return fig


def donut(labels: List, values: List, colors: List, title: str = "") -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hole=0.55,
            textinfo="label+percent",
            textfont_size=10,
        )
    )
    fig.update_layout(
        PLOTLY_LAYOUT,
        height=220,
        title=title,
        showlegend=False,
        margin=dict(l=0, r=0, t=28, b=0),
    )
    return fig


def heatmap(df: pd.DataFrame, title: str = "") -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=df.values,
            x=list(df.columns),
            y=list(df.index),
            colorscale=[[0, "#13161e"], [0.5, "#2d5a9e"], [1, "#5b9cf6"]],
            showscale=False,
            texttemplate="%{z}",
        )
    )
    fig.update_layout(PLOTLY_LAYOUT, height=max(160, len(df) * 30 + 60), title=title)
    return fig


# ══════════════════════════════════════════════════════════════
# TRAJECTORY CHARTS (register, TTR, CX) — replaces single topic-shift chart
# ══════════════════════════════════════════════════════════════


def _trajectory_charts(uces_all: List[Dict]) -> go.Figure:
    """4-panel small-multiples trajectory: topic shift, TTR, register, CX index."""
    n = len(uces_all)
    if not n:
        return None

    ts_vals = [u.get("topic_shift_prev", 0) for u in uces_all]
    ttr_vals = [u.get("metricas_lexicas", {}).get("ttr", 0) for u in uces_all]

    # CX composite: same formula as the old card badge
    def _cx(u):
        cx = u.get("complejidad_sintactica", {})
        return round(
            cx.get("profundidad_maxima", 0) * 0.3
            + cx.get("recursividad", 0) * 0.3
            + cx.get("ratio_subordinacion", 0) * 0.4,
            3,
        )

    cx_vals = [_cx(u) for u in uces_all]

    # Register as numeric: coloquial=1, mixto=2, formal=3, tecnico=4
    reg_map = {"coloquial": 1, "mixto": 2, "desconocido": 2, "formal": 3, "tecnico": 4}
    reg_vals = [
        reg_map.get((u.get("registro") or "mixto").lower(), 2) for u in uces_all
    ]
    reg_labels = {1: "coloquial", 2: "mixto", 3: "formal", 4: "técnico"}

    xs = list(range(n))

    # FIX 1: Added the make_subplots call
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "Cambio temático",
            "Riqueza léxica (TTR)",
            "Índice CX",
            "Registro",
        ],
    )

    common = dict(mode="lines", line=dict(width=1.5))

    # Topic shift
    fig.add_hline(
        y=0.4,
        line_dash="dot",
        line_color="#f97b6b",
        annotation_text="umbral",
        annotation_font_size=8,
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ts_vals,
            fill="tozeroy",
            fillcolor="rgba(249,123,107,0.07)",
            line=dict(color="#f97b6b", width=1.5),
            hovertemplate="UCE %{x}<br>shift: %{y:.3f}<extra></extra>",
            **{k: v for k, v in common.items() if k not in ["line"]},
        ),
        row=1,
        col=1,
    )

    # TTR
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ttr_vals,
            fill="tozeroy",
            fillcolor="rgba(91,156,246,0.07)",
            line=dict(color="#5b9cf6", width=1.5),
            hovertemplate="UCE %{x}<br>TTR: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # CX index
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=cx_vals,
            fill="tozeroy",
            fillcolor="rgba(192,132,252,0.07)",
            line=dict(color="#c084fc", width=1.5),
            hovertemplate="UCE %{x}<br>CX: %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Register (step chart)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=reg_vals,
            mode="lines",
            line=dict(color="#4ade80", width=1.5, shape="hv"),
            hovertemplate=[
                f"UCE {i}<br>{reg_labels.get(v, '')}<extra></extra>"
                for i, v in enumerate(reg_vals)
            ],
        ),
        row=4,
        col=1,
    )

    fig.update_yaxes(
        tickvals=[1, 2, 3, 4],
        ticktext=["coloquial", "mixto", "formal", "técnico"],
        row=4,
        col=1,
    )

    # FIX 2: PLOTLY_LAYOUT passed positionally so 'margin' overrides it cleanly
    fig.update_layout(
        PLOTLY_LAYOUT,
        height=400,
        showlegend=False,
        margin=dict(t=40, b=20, l=60, r=10),
        xaxis4_title="UCE",
    )

    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor="#2a2f42", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#2a2f42", row=i, col=1)

    return fig


def _human_label(key: str, val) -> str:
    """Convert cryptic field names to readable Spanish labels."""
    MAP = {
        "ttr": "Riqueza léxica (TTR)",
        "guiraud": "Índice Guiraud",
        "hapax_ratio": "Palabras únicas (%)",
        "diversidad_semantica": "Diversidad semántica",
        "topic_shift": "Cambio temático",
        "prof_sint_max": "Profundidad sintáctica",
        "recursividad": "Subordinación recursiva",
        "dep_dist_media": "Distancia de dependencia",
        "ratio_subordinacion": "Ratio de subordinación",
        "branching_ratio": "Ramificación derecha",
        "negaciones_norm": "Negaciones (x1000 tokens)",
        "pronombres_exp_norm": "Pronombres explícitos",
        "prodrop_norm": "Sujeto nulo (pro-drop)",
        "verbos_norm": "Verbos",
        "cuantificadores_norm": "Cuantificadores",
        "adverbios_norm": "Adverbios",
        "marcadores_norm": "Marcadores discursivos",
        "mean_zipf": "Frecuencia media (Zipf)",
        "pct_oov": "Palabras fuera de vocabulario (%)",
        "oral_ratio": "Léxico oral (%)",
        "academic_ratio": "Léxico académico (%)",
        "mean_surprisal_content": "Carga cognitiva media",
    }
    return MAP.get(key, key.replace("_", " ").title())


def _complexity_radar(uces_all):
    LABELS = [
        "Profundidad",
        "Subordinación",
        "Dist. dependencia",
        "Ramificación",
        "Recursividad",
    ]
    KEYS = [
        "profundidad_maxima",
        "ratio_subordinacion",
        "distancia_dependencia_media",
        "branching_ratio",
        "recursividad",
    ]
    means = []
    for k in KEYS:
        vals = [u.get("complejidad_sintactica", {}).get(k, 0) for u in uces_all]
        means.append(float(np.mean(vals)) if vals else 0)
    # Normalize 0-1 per axis for readability
    maxes = [5, 1, 8, 1, 5]
    normed = [min(v / m, 1.0) if m else 0 for v, m in zip(means, maxes)]
    fig = go.Figure(
        go.Scatterpolar(
            r=normed + [normed[0]],
            theta=LABELS + [LABELS[0]],
            fill="toself",
            fillcolor="rgba(91,156,246,0.12)",
            line=dict(color="#5b9cf6", width=2),
            hovertemplate=[
                f"<b>{l}</b><br>Media: {v:.2f}<extra></extra>"
                for l, v in zip(LABELS, means)
            ]
            + [""],
        )
    )
    fig.update_layout(
        PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="#13161e",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["25%", "50%", "75%", "100%"],
                gridcolor="#2a2f42",
                linecolor="#2a2f42",
            ),
            angularaxis=dict(gridcolor="#2a2f42", linecolor="#2a2f42"),
        ),
        height=260,
        title=dict(text="Complejidad sintáctica", font=dict(size=11)),
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# SUBCATEGORY FILTER STATE
# ══════════════════════════════════════════════════════════════


def _get_subcat_filter() -> Dict[str, str]:
    """Obtiene el diccionario actual de filtros activos."""
    if SUBCAT_FILTER_KEY not in st.session_state:
        st.session_state[SUBCAT_FILTER_KEY] = {}
    return st.session_state[SUBCAT_FILTER_KEY]


def _set_subcat_filter(cat: str, val: Optional[str]):
    """
    Activa o desactiva un filtro individual.
    Es utilizado por los gráficos de barras (_filter_bar) para hacer 'toggle'.
    """
    current = _get_subcat_filter()

    # Si hacemos clic en la misma barra que ya está activa, se desactiva
    if current.get(cat) == str(val):
        current.pop(cat, None)
    elif val is not None:
        current[cat] = str(val)
    else:
        current.pop(cat, None)

    st.session_state[SUBCAT_FILTER_KEY] = current
    st.rerun()


def _set_multiple_filters(tab_name: str, filters_dict: dict):
    """
    Reemplaza los filtros actuales por un grupo nuevo de filtros.
    """
    clean_filters = {k: str(v) for k, v in filters_dict.items() if v}
    st.session_state[SUBCAT_FILTER_KEY] = clean_filters

    # ESTO ES CRÍTICO: Debe ser el nombre exacto de la 'key' del st.radio
    st.session_state.radio_tabs_a = tab_name

    st.rerun()


def _clear_subcat_filters():
    """Clear all subcategory filters and related UI selections."""
    st.session_state[SUBCAT_FILTER_KEY] = {}
    # Clear coreference selection
    st.session_state.coref_selected_entity = None
    # Clear predicate selections
    st.session_state.pred_selected_lemma = None
    st.session_state.pred_filter_voice = "Todas"
    st.session_state.pred_filter_role = "Todos"
    st.session_state.pred_filter_negated = "Ambos"
    st.rerun()


# ══════════════════════════════════════════════════════════════
# HELPER: clickable horizontal barchart that writes to session_state
# ══════════════════════════════════════════════════════════════


def _filter_bar(
    df: pd.DataFrame,
    count_col: str,
    label_col: str,
    filter_cat: str,
    raw_val_col: str,
    color: str,
    title: str,
    zero_labels: List[str] = None,
):
    """
    Renders a horizontal barchart. Clicking a bar sets subcat_filter[filter_cat] = raw_val.
    zero_labels: list of category names to show even if count = 0.
    """
    active = _get_subcat_filter().get(filter_cat)

    # Ensure zero-count categories appear
    if zero_labels:
        existing = set(df[raw_val_col].tolist())
        missing = [z for z in zero_labels if z not in existing]
        if missing:
            pad = pd.DataFrame({raw_val_col: missing, label_col: missing, count_col: 0})
            df = pd.concat([df, pad], ignore_index=True)

    df = df.sort_values(count_col, ascending=True)
    colors = [
        f"rgba({_hex_to_rgb(color)},0.9)"
        if (active is None or str(row[raw_val_col]) == active)
        else f"rgba({_hex_to_rgb(color)},0.25)"
        for _, row in df.iterrows()
    ]

    fig = go.Figure(
        go.Bar(
            x=df[count_col],
            y=df[label_col],
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=df[count_col].astype(int),
            textposition="outside",
            customdata=df[raw_val_col],
            hovertemplate="%{y}: %{x}<br><i>clic para filtrar</i><extra></extra>",
        )
    )
    fig.update_layout(
        PLOTLY_LAYOUT,
        height=max(120, len(df) * 32 + 40),
        title=dict(text=title, font=dict(size=11)),
        margin=dict(t=30, b=10, l=10, r=40),
        xaxis=dict(visible=False),
        clickmode="event",
    )

    event = st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False},
        on_select="rerun",
        key=f"filterbar_{filter_cat}",
    )
    if event and event.get("selection", {}).get("points"):
        pt = event["selection"]["points"][0]
        raw = df.iloc[pt["point_index"]][raw_val_col]
        _set_subcat_filter(filter_cat, str(raw))

    # Active filter badge
    if active:
        col1, col2 = st.columns([8, 2])
        with col1:
            st.caption(f"Filtrando: **{active}**")
        with col2:
            if st.button("✕", key=f"clear_{filter_cat}", help="Quitar filtro"):
                _set_subcat_filter(filter_cat, None)


def _hex_to_rgb(hex_color: str) -> str:
    """'#5b9cf6' → '91,156,246'"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


# ══════════════════════════════════════════════════════════════
# UPDATED TAB FUNCTIONS (subcategory barcharts + filter)
# ══════════════════════════════════════════════════════════════


def tab_global(stats: Dict, uces_all: List[Dict]):
    g = safe_df(stats.get("global", []))
    n_uces = len(uces_all)
    n_tokens = sum(u.get("metricas_lexicas", {}).get("num_tokens", 0) for u in uces_all)

    def _mean(idx_name):
        if g.empty or "index" not in g.columns or "mean" not in g.columns:
            return 0.0
        row = g.loc[g["index"] == idx_name, "mean"]
        return float(row.values[0]) if len(row) else 0.0

    ttr_mean = _mean("ttr")
    ts_mean = _mean("topic_shift")

    st.markdown(
        f"""
    <div class="kpi-row">
      <div class="kpi-card" title="Unidades de Contexto Elemental analizadas">
        <div class="kpi-value">{n_uces}</div><div class="kpi-label">UCEs</div></div>
      <div class="kpi-card" title="Total de palabras en el corpus filtrado">
        <div class="kpi-value">{n_tokens:,}</div><div class="kpi-label">Tokens</div></div>
      <div class="kpi-card" title="Type-Token Ratio: proporción de palabras únicas.">
        <div class="kpi-value">{fmt(ttr_mean)}</div><div class="kpi-label">Riqueza léxica</div></div>
      <div class="kpi-card" title="Cambio temático promedio entre UCEs consecutivas">
        <div class="kpi-value">{fmt(ts_mean)}</div><div class="kpi-label">Cambio temático</div></div>
    </div>""",
        unsafe_allow_html=True,
    )

    # ── Trajectory multi-panel (replaces single topic-shift chart) ──
    traj_fig = _trajectory_charts(uces_all)
    if traj_fig:
        st.plotly_chart(traj_fig, width="stretch", config={"displayModeBar": False})
        st.caption(
            "Arriba→abajo: cambio temático (picos = ruptura discursiva) · "
            "riqueza léxica (TTR) · índice de complejidad sintáctica · "
            "registro (escalón = cambio de registro)."
        )

    st.divider()
    st.plotly_chart(
        _complexity_radar(uces_all), width="stretch", config={"displayModeBar": False}
    )
    st.caption(
        "Radar normalizado. Los valores representan la media del corpus filtrado."
    )

    norm_keys = [
        "negaciones_norm",
        "pronombres_exp_norm",
        "prodrop_norm",
        "verbos_norm",
        "cuantificadores_norm",
        "adverbios_norm",
        "marcadores_norm",
    ]
    if not g.empty and "index" in g.columns:
        rate_rows = g[g["index"].isin(norm_keys)][["index", "mean"]].copy()
        rate_rows["label"] = rate_rows["index"].map(_human_label)
        rate_rows = rate_rows.dropna(subset=["mean"])
        if not rate_rows.empty:
            fig = horizontal_bar(
                rate_rows,
                "mean",
                "label",
                "#5b9cf6",
                "Densidad gramatical (media por 1000 tokens)",
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Frecuencia normalizada de cada categoría gramatical.")


def tab_verbos(stats: Dict, uces_all: List[Dict]):
    v_df = safe_df(stats.get("verbos", []))
    if v_df.empty or "categoria" not in v_df.columns:
        st.info("Sin datos verbales.")
        return

    # ── FACET FILTERS: mood, tense, voice, person×number ──────────
    st.markdown("**Filtros morfológicos** — clic en una barra para filtrar el texto")

    voice_labels = {
        "Act": "Activa",
        "Pass": "Pasiva",
        "PassRefl": "Pasiva refleja",
        "Media": "Media",
        "Impersonal": "Impersonal",
    }
    mood_labels = {
        "Ind": "Indicativo",
        "Sub": "Subjuntivo",
        "Imp": "Imperativo",
        "Cnd": "Condicional",
    }
    tense_labels = {
        "Pres": "Presente",
        "Past": "Pasado",
        "Fut": "Futuro",
        "Imp": "Imperfecto",
        "Pqp": "Pluscuamperfecto",
    }

    voice_rows = v_df[v_df["categoria"] == "voz"].copy()
    mood_rows = v_df[v_df["categoria"] == "modo"].copy()
    tense_rows = v_df[v_df["categoria"] == "tiempo"].copy()

    # Person × number grid from raw verb data
    pn_data = []
    for uce in uces_all:
        for vb in uce.get("verbos", []):
            p = vb.get("persona")
            n = vb.get("numero")
            if p and n:
                pn_data.append(f"{p}{n}")
    pn_series = pd.Series(pn_data).value_counts().reset_index()
    pn_series.columns = ["pn", "n"]
    pn_labels_map = {
        "1Sing": "1ª sing.",
        "2Sing": "2ª sing.",
        "3Sing": "3ª sing.",
        "1Plur": "1ª plur.",
        "2Plur": "2ª plur.",
        "3Plur": "3ª plur.",
    }
    pn_series["label"] = pn_series["pn"].map(lambda x: pn_labels_map.get(x, x))

    col1, col2 = st.columns(2)
    with col1:
        if not mood_rows.empty:
            mood_rows["label"] = mood_rows["valor"].map(
                lambda x: mood_labels.get(str(x), str(x))
            )
            _filter_bar(
                mood_rows,
                "freq_abs",
                "label",
                "verb_mood",
                "valor",
                "#5b9cf6",
                "Modo",
                zero_labels=list(mood_labels.keys()),
            )
        if not tense_rows.empty:
            tense_rows["label"] = tense_rows["valor"].map(
                lambda x: tense_labels.get(str(x), str(x))
            )
            _filter_bar(
                tense_rows,
                "freq_abs",
                "label",
                "verb_tense",
                "valor",
                "#5b9cf6",
                "Tiempo",
                zero_labels=list(tense_labels.keys()),
            )
    with col2:
        if not voice_rows.empty:
            voice_rows["label"] = voice_rows["valor"].map(
                lambda x: voice_labels.get(str(x), str(x))
            )
            _filter_bar(
                voice_rows,
                "freq_abs",
                "label",
                "verb_voice",
                "valor",
                "#5b9cf6",
                "Voz",
                zero_labels=list(voice_labels.keys()),
            )
        if not pn_series.empty:
            _filter_bar(
                pn_series, "n", "label", "verb_pn", "pn", "#5b9cf6", "Persona y número"
            )

    # ── Subordination ──────────────────────────────────────────────
    sub_rows = v_df[v_df["categoria"] == "tipo_subordinacion"].copy()
    if not sub_rows.empty:
        st.divider()
        sub_labels = {
            "adverbial": "Adverbial",
            "completiva": "Completiva",
            "relativa": "Relativa",
            "desconocida": "Sin clasificar",
        }
        sub_rows["label"] = sub_rows["valor"].map(
            lambda x: sub_labels.get(str(x), str(x))
        )
        _filter_bar(
            sub_rows,
            "freq_abs",
            "label",
            "verb_sub",
            "valor",
            "#fbbf24",
            "Tipo de subordinación",
        )
        st.caption(
            "Las completivas dominantes indican discurso de opinión/reporte. "
            "Las adverbiales causales y concesivas = argumentación compleja."
        )


def tab_negacion(stats: Dict, uces_all: List[Dict]):
    neg_df = safe_df(stats.get("negaciones", []))

    # Grouped into scope types vs pragmatic subtypes
    scope_types = [
        "NEGACION_VERBAL",
        "NEGACION_ADVERBIAL",
        "NEGACION_NOMINAL",
        "NEGACION_CONJUNTIVA",
        "NEGACION_OTRA",
    ]
    pragma_types = ["NEGACION_DE_GRADO", "NPI", "DOBLE", "METALINGUISTICA"]

    neg_type_labels = {
        "NEGACION_VERBAL": "Verbal (no + verbo)",
        "NEGACION_ADVERBIAL": "Adverbial (nunca, jamás)",
        "NEGACION_NOMINAL": "Nominal (ningún, nadie)",
        "NEGACION_CONJUNTIVA": "Conjuntiva (ni)",
        "NEGACION_OTRA": "Otra",
        "NEGACION_DE_GRADO": "De grado (no muy)",
        "NPI": "Polaridad negativa (NPI)",
        "DOBLE": "Doble negación",
        "METALINGUISTICA": "Metalingüística",
        # Legacy type names from tipo_negacion
        "STANDARD": "Verbal (no + verbo)",
        "CONSTITUYENTE": "De constituyente",
    }

    if not neg_df.empty and "tipo" in neg_df.columns:
        neg_df = neg_df.copy()
        neg_df["label"] = neg_df["tipo"].map(
            lambda x: neg_type_labels.get(str(x), str(x))
        )

        col1, col2 = st.columns(2)
        with col1:
            scope_df = neg_df[
                neg_df["tipo"].isin(scope_types + ["STANDARD", "CONSTITUYENTE"])
            ]
            if not scope_df.empty:
                _filter_bar(
                    scope_df,
                    "freq_abs",
                    "label",
                    "neg_scope",
                    "tipo",
                    COLORS["neg"],
                    "Tipo de alcance",
                    zero_labels=scope_types,
                )
                st.caption(
                    "El alcance determina qué elemento niega la palabra negativa."
                )
        with col2:
            pragma_df = neg_df[neg_df["tipo"].isin(pragma_types)]
            # DOBLE negación: normal en español (no sé nada)
            if not pragma_df.empty:
                _filter_bar(
                    pragma_df,
                    "freq_abs",
                    "label",
                    "neg_pragma",
                    "tipo",
                    "#fbbf24",
                    "Tipo pragmático",
                    zero_labels=pragma_types,
                )
                st.caption(
                    "Doble negación = normal en español estándar. "
                    "NPIs sin licenciar = uso enfático o error gramatical."
                )

    # NPI tables (unchanged)
    unlicensed = []
    licensed = []
    for uce in uces_all:
        for r in uce.get("rarezas", []):
            if r.get("tipo") == "NPI_NO_LICENCIADO":
                unlicensed.append(
                    {
                        "UCE": uce["id"],
                        "Expresión": r.get("token", ""),
                        "Nota": "NPI sin operador negativo licenciador",
                    }
                )
        for neg in uce.get("negaciones", []):
            for npi in neg.get("npis", []):
                licensed.append(
                    {
                        "UCE": uce["id"],
                        "NPI": npi.get("text", ""),
                        "Licenciado por": neg.get("texto", ""),
                    }
                )
    if unlicensed:
        st.markdown("**⚠ NPIs sin licenciar**")
        st.dataframe(pd.DataFrame(unlicensed), width="stretch", hide_index=True)
    if licensed:
        with st.expander(f"NPIs licenciados ({len(licensed)})"):
            st.dataframe(pd.DataFrame(licensed), width="stretch", hide_index=True)


def tab_pronombres(stats: Dict, uces_all: List[Dict]):
    pr_df = safe_df(stats.get("pronombres", []))
    if pr_df.empty:
        st.info("Sin datos de pronombres.")
        return

    tipo_df = (
        pr_df[pr_df["nivel"] == "tipo"] if "nivel" in pr_df.columns else pd.DataFrame()
    )
    sub_df = (
        pr_df[pr_df["nivel"] == "subtipo"]
        if "nivel" in pr_df.columns
        else pd.DataFrame()
    )

    n_null = (
        float(tipo_df.loc[tipo_df["clave"] == "NULO", "freq_abs"].sum())
        if not tipo_df.empty and "clave" in tipo_df.columns
        else 0
    )
    n_exp = (
        float(tipo_df.loc[tipo_df["clave"] == "EXPLICITO", "freq_abs"].sum())
        if not tipo_df.empty
        else 0
    )
    total_p = n_null + n_exp
    ratio = n_null / total_p if total_p else 0

    st.markdown(
        f"""
    <div class="kpi-row">
      <div class="kpi-card" title="Verbos sin pronombre sujeto explícito">
        <div class="kpi-value">{fmt(ratio, decimals=1)}%</div>
        <div class="kpi-label">Sujeto nulo</div></div>
      <div class="kpi-card" title="Pronombres sujeto expresados explícitamente">
        <div class="kpi-value">{int(n_exp)}</div>
        <div class="kpi-label">Pron. explícitos</div></div>
    </div>""",
        unsafe_allow_html=True,
    )
    st.caption(
        "El español es lengua de sujeto nulo: 70-80% nulo es esperable. "
        "Valores más altos de explícito → énfasis, contraste o contexto L2."
    )

    # Pro-drop vs explicit as primary filter
    if not tipo_df.empty and "clave" in tipo_df.columns:
        tipo_labels = {
            "NULO": "Sujeto nulo (pro-drop)",
            "EXPLICITO": "Pronombre explícito",
            "ENCLITICO": "Enclítico",
            "CONTRACCION": "Contracción",
        }
        tipo_df = tipo_df.copy()
        tipo_df["label"] = tipo_df["clave"].map(
            lambda x: tipo_labels.get(str(x), str(x))
        )
        _filter_bar(
            tipo_df,
            "freq_abs",
            "label",
            "pron_tipo",
            "clave",
            COLORS["pron"],
            "Tipo de sujeto pronominal",
            zero_labels=list(tipo_labels.keys()),
        )

    # Explicit subtypes
    if not sub_df.empty:
        st.divider()
        subtipo_labels = {
            "PRONOMBRE_REFLEXIVO": "Reflexivo (se, me, te)",
            "PRONOMBRE_ACUSATIVO": "Acusativo (lo, la, los)",
            "PRONOMBRE_DATIVO": "Dativo (le, les)",
            "PRONOMBRE_PERSONAL": "Personal (yo, tú, él)",
            "PRONOMBRE_DEMOSTRATIVO": "Demostrativo (este, ese)",
            "PRONOMBRE_RELATIVO": "Relativo (que, quien)",
            "PRONOMBRE_INTERROGATIVO": "Interrogativo (qué, quién)",
            "PRONOMBRE_INDEFINIDO": "Indefinido (alguien, nadie)",
            "PRONOMBRE_OTRO": "Otro",
        }
        sub_df = sub_df.copy()
        sub_df["label"] = sub_df["clave"].map(
            lambda x: subtipo_labels.get(str(x), str(x))
        )
        _filter_bar(
            sub_df,
            "freq_abs",
            "label",
            "pron_subtipo",
            "clave",
            COLORS["pron"],
            "Subtipos de pronombre",
            zero_labels=list(subtipo_labels.keys()),
        )
        st.caption(
            "Reflexivos altos = verbos de cambio de estado (se fue, se rió). "
            "Dativos altos = presencia de objetos indirectos frecuentes."
        )

    # Person breakdown from raw data
    pn_data = []
    for uce in uces_all:
        for pr in uce.get("pronombres", []):
            p = pr.get("persona")
            n = pr.get("numero")
            if p and n and pr.get("es_referencial", True):
                pn_data.append({"persona": str(p), "numero": str(n)})
    if pn_data:
        pn_df = pd.DataFrame(pn_data)
        pn_counts = pn_df.groupby(["persona", "numero"]).size().reset_index(name="n")
        pn_counts["label"] = pn_counts.apply(
            lambda r: (
                f"{r['persona']}ª {'sing.' if r['numero'] == 'Sing' else 'plur.'}"
            ),
            axis=1,
        )
        _filter_bar(
            pn_counts,
            "n",
            "label",
            "pron_pn",
            "persona",
            COLORS["pron"],
            "Persona gramatical",
        )
        st.caption(
            "1ª persona singular alta = narración autobiográfica o posicionamiento subjetivo. "
            "2ª persona alta = interpelación directa al oyente."
        )


def tab_adverbios(stats: Dict, uces_all: List[Dict]):
    adv_df = safe_df(stats.get("adverbios", []))
    if adv_df.empty:
        st.info("Sin datos de adverbios.")
        return

    adv_colors_map = {
        "modo": "#94a3b8",
        "epistemico": "#a78bfa",
        "conjuntivo": "#2dd4bf",
        "dominio": "#5b9cf6",
        "grado": "#fbbf24",
        "orientado_hablante": "#f97b6b",
        "tiempo": "#4ade80",
        "lugar": "#fb923c",
        "frecuencia": "#38bdf8",
        "foco": "#e879f9",
        "comparativo": "#a3e635",
        "orientado_sujeto": "#f59e0b",
        "DESCONOCIDO": "#374151",
    }
    cat_labels = {
        "modo": "Modo (bien, mal)",
        "epistemico": "Epistémico (quizás, probablemente)",
        "conjuntivo": "Conectivo (además, sin embargo)",
        "dominio": "Dominio (técnicamente, legalmente)",
        "grado": "Grado (muy, bastante, apenas)",
        "orientado_hablante": "Orient. hablante (francamente)",
        "tiempo": "Temporal (ayer, siempre)",
        "lugar": "Espacial (aquí, lejos)",
        "frecuencia": "Frecuencia (a menudo, rara vez)",
        "foco": "Foco (solo, únicamente)",
        "comparativo": "Comparativo (más, menos, mejor)",
        "orientado_sujeto": "Orient. sujeto (deliberadamente)",
        "DESCONOCIDO": "Sin clasificar",
    }
    # Semantic group labeling (for expert users)
    group_info = {
        "epistemico": "Expresan la actitud del hablante hacia la verdad del enunciado.",
        "orientado_hablante": "Evalúan el acto de habla completo (≈ 'comment adverbs').",
        "conjuntivo": "Conectan proposiciones; son marcadores discursivos léxicos.",
        "dominio": "Restringen el ámbito o dominio de aplicación del predicado.",
        "foco": "Presuponen alternativas y seleccionan una (partículas de foco).",
    }

    if "categoria" in adv_df.columns:
        adv_df = adv_df.copy()
        adv_df["label"] = adv_df["categoria"].map(
            lambda x: cat_labels.get(str(x), str(x))
        )
        adv_df["color"] = adv_df["categoria"].map(
            lambda x: adv_colors_map.get(str(x), "#64748b")
        )

        # Show ALL possible categories including zero-count ones
        _filter_bar(
            adv_df,
            "freq_abs",
            "label",
            "adv_cat",
            "categoria",
            "#a78bfa",
            "Categorías de adverbio",
            zero_labels=list(cat_labels.keys()),
        )
        st.caption(
            "Las categorías vacías se muestran explícitamente: su ausencia también es un dato."
        )

        # Explain present non-obvious categories
        present_cats = set(adv_df[adv_df["freq_abs"] > 0]["categoria"].tolist())
        for cat, info in group_info.items():
            if cat in present_cats:
                st.caption(
                    f"**{cat_labels.get(cat, cat).split('(')[0].strip()}**: {info}"
                )

    # Confidence warning (unchanged logic)
    conf_data = []
    for uce in uces_all:
        for a in uce.get("adverbios", []):
            conf_data.append(
                {
                    "cat": cat_labels.get(
                        a.get("categoria", ""), a.get("categoria", "")
                    ),
                    "conf": a.get("confianza", 0),
                }
            )
    if conf_data:
        cf = pd.DataFrame(conf_data).groupby("cat")["conf"].mean().reset_index()
        low_conf = cf[cf["conf"] < 0.6]
        if not low_conf.empty:
            st.markdown("**⚠ Categorías con clasificación incierta** (confianza < 60%)")
            for _, row in low_conf.iterrows():
                st.caption(f"· {row['cat']}: {row['conf']:.0%} — revisar manualmente")

    # Stacked area trajectory (unchanged)
    if uces_all:
        cats = (
            list(adv_df["categoria"].unique()) if "categoria" in adv_df.columns else []
        )
        traj = {c: [] for c in cats}
        for uce in uces_all:
            cnt = defaultdict(int)
            for a in uce.get("adverbios", []):
                cnt[a.get("categoria", "")] += 1
            for c in cats:
                traj[c].append(cnt.get(c, 0))
        if traj:
            fig = go.Figure()
            for c in cats:
                fig.add_trace(
                    go.Scatter(
                        y=traj[c],
                        mode="lines",
                        name=cat_labels.get(c, c),
                        stackgroup="one",
                        line=dict(width=0.5),
                        fillcolor=adv_colors_map.get(c, "#374151"),
                        hovertemplate=f"{cat_labels.get(c, c)}: %{{y}}<extra></extra>",
                    )
                )
            fig.update_layout(
                PLOTLY_LAYOUT,
                height=180,
                title="Densidad de adverbios a lo largo del discurso",
                xaxis_title="UCE",
                margin=dict(t=30, b=20),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


@st.cache_data(show_spinner=False)
def _compute_radar_data(
    df_json: str,  # JSON of the *full* df_all (unfiltered by class)
    traits_key: tuple,  # sorted tuple of active traits
    classes_key: tuple,  # sorted tuple of active classes
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame:
        clase | trait | pct
    where pct = (annotations of this trait in this clase) /
                (total annotations of this trait across ALL classes) * 100

    We use df_all (not df_filt) as denominator so percentages always sum to 100
    across classes for each trait axis.
    """
    df = pd.read_json(df_json, orient="records")
    if df.empty:
        return pd.DataFrame(columns=["clase", "trait", "pct", "n", "total"])

    traits = list(traits_key)
    classes = list(classes_key)

    df_t = df[df["trait"].isin(traits)].copy()

    # Total per trait (denominator) — across ALL classes in df_all
    trait_totals = df_t.groupby("trait").size().rename("total")

    # Count per (class, trait)
    ct = (
        df_t[df_t["cluster_id"].isin(classes)]
        .groupby(["cluster_id", "trait"])
        .size()
        .reset_index(name="n")
    )
    ct = ct.merge(trait_totals, on="trait", how="left")
    ct["pct"] = (ct["n"] / ct["total"] * 100).round(2)
    ct.rename(columns={"cluster_id": "clase"}, inplace=True)
    return ct


def _chart_radar(
    df_all: pd.DataFrame,
    active_traits: Set[str],
    active_classes: List[Any],
    class_colors: Dict[Any, str],
) -> go.Figure:
    """
    Superimposed radar (scatterpolar) — one trace per class.
    Axes = discourse traits.
    Value = % of total annotations for that trait that fall in this class.
    """
    import json

    traits = sorted(active_traits)
    if len(traits) < 3:
        fig = go.Figure(layout=PLOTLY_LAYOUT)
        fig.add_annotation(
            text="Selecciona ≥3 rasgos para el radar",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=T["text_dim"], size=12),
        )
        return fig

    radar_df = _compute_radar_data(
        df_all.to_json(orient="records"),
        tuple(sorted(active_traits)),
        tuple(sorted(str(c) for c in active_classes)),
    )

    fig = go.Figure()

    for cls in active_classes:
        col = class_colors.get(cls, T["accent"])
        sub = radar_df[radar_df["clase"] == str(cls)].set_index("trait")
        vals = [float(sub.loc[t, "pct"]) if t in sub.index else 0.0 for t in traits]
        ns = [int(sub.loc[t, "n"]) if t in sub.index else 0 for t in traits]
        # Close the polygon
        vals_closed = vals + [vals[0]]
        traits_closed = traits + [traits[0]]
        ns_closed = ns + [ns[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=vals_closed,
                theta=traits_closed,
                fill="toself",
                fillcolor=_hex_to_rgba(col, 0.08),
                line=dict(color=col, width=1.8),
                name=f"Clase {cls}",
                hovertemplate=(
                    f"<b>%{{theta}}</b><br>Clase {cls}: %{{r:.1f}}%<br><extra></extra>"
                ),
                customdata=ns_closed,
            )
        )

    # Overlay a reference circle at 100/n_classes (equal distribution baseline)
    n_cl = len(active_classes)
    if n_cl > 0:
        base = round(100 / n_cl, 1)
        base_vals = [base] * len(traits) + [base]
        fig.add_trace(
            go.Scatterpolar(
                r=base_vals,
                theta=traits_closed,
                mode="lines",
                line=dict(color=T["text_dim"], width=1, dash="dot"),
                name=f"referencia ({base}%)",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        polar=dict(
            bgcolor=T["bg_panel"],
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
                tickfont=dict(size=9, color=T["text_dim"]),
                gridcolor=T["border"],
                linecolor=T["border"],
                dtick=25,
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color=T["text_hi"]),
                gridcolor=T["border"],
                linecolor=T["border"],
            ),
        ),
        title=dict(
            text="Peso discursivo por clase · % del total de cada rasgo",
            font=dict(size=10, color=T["text_dim"]),
            x=0,
            xanchor="left",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.08, font=dict(size=9)),
        margin=dict(t=44, b=56, l=20, r=20),
    )
    return fig


def _radar_summary_table(
    df_all: pd.DataFrame,
    active_traits: Set[str],
    active_classes: List[Any],
) -> pd.DataFrame:
    """
    Pivot table: clase × trait → pct, plus a 'dominancia' column
    (the trait with the highest % for each class).
    """
    radar_df = _compute_radar_data(
        df_all.to_json(orient="records"),
        tuple(sorted(active_traits)),
        tuple(sorted(str(c) for c in active_classes)),
    )
    if radar_df.empty:
        return pd.DataFrame()

    pivot = radar_df.pivot_table(
        index="clase", columns="trait", values="pct", fill_value=0.0
    ).reset_index()
    trait_cols = [c for c in pivot.columns if c != "clase"]
    pivot["rasgo_dominante"] = pivot[trait_cols].idxmax(axis=1)
    pivot["pct_max"] = pivot[trait_cols].max(axis=1).round(1)
    return pivot


def tab_discurso(stats: Dict, uces_all: List[Dict]):
    disc_data = []
    for uce in uces_all:
        for m in uce.get("marcadores_discursivos", []):
            disc_data.append(
                {
                    "cat": m.get("categoria", ""),
                    "pos": m.get("posicion", ""),
                    "text": m.get("texto", ""),
                    "surp": m.get("surprisal_transicion", 0),
                }
            )

    if not disc_data:
        st.info("Sin marcadores discursivos detectados.")
    else:
        dd = pd.DataFrame(disc_data)
        cnt = dd.groupby("cat").size().reset_index(name="n").nlargest(12, "n")

        disc_cat_labels = {
            "REFORMULADORES": "Reformuladores (o sea, es decir)",
            "ESTRUCTURADORES": "Estructuradores (en primer lugar, por otro lado)",
            "ARGUMENTATIVOS": "Argumentativos (sin embargo, por tanto)",
            "CONVERSACIONALES": "Conversacionales (bueno, mira, claro)",
            "PALABRA_SUELTA": "Conectores simples",
        }
        cnt["label"] = cnt["cat"].map(lambda x: disc_cat_labels.get(str(x), str(x)))
        _filter_bar(
            cnt,
            "n",
            "label",
            "disc_cat",
            "cat",
            COLORS["disc"],
            "Marcadores por función discursiva",
            zero_labels=list(disc_cat_labels.keys()),
        )
        st.caption(
            "Los reformuladores indican reelaboración del mensaje. "
            "Los argumentativos son el núcleo de la estructura retórica."
        )

        # Position distribution
        pos_cnt = dd.groupby("pos").size().reset_index(name="n")
        pos_labels = {
            "INICIO": "Inicio de UCE",
            "MEDIAL": "Posición medial",
            "FINAL": "Final de UCE",
        }
        pos_cnt["label"] = pos_cnt["pos"].map(lambda x: pos_labels.get(x, x))
        _filter_bar(
            pos_cnt,
            "n",
            "label",
            "disc_pos",
            "pos",
            COLORS["disc"],
            "Posición en la UCE",
            zero_labels=list(pos_labels.keys()),
        )
        st.caption("Los marcadores en posición inicial estructuran el turno de habla.")

    # Insubordinations and rarezas unchanged
    insub_data = [
        {
            "UCE": u["id"],
            "Marcador": i.get("marcador", ""),
            "Función pragmática": i.get("funcion_pragmatica", ""),
            "Tipo": i.get("tipo", "INSUBORDINACION"),
            "Fragmento": u.get("texto", "")[:60],
        }
        for u in uces_all
        for i in u.get("insubordinaciones", [])
    ]
    if insub_data:
        st.divider()
        st.markdown(
            "**Insubordinaciones** — subordinadas usadas como enunciados independientes"
        )

        insub_df = pd.DataFrame(insub_data)

        # Filter bar by pragmatic function
        func_cnt = insub_df.groupby("Función pragmática").size().reset_index(name="n")
        func_cnt["raw"] = func_cnt["Función pragmática"]
        _filter_bar(
            func_cnt,
            "n",
            "Función pragmática",
            "insub_func",
            "raw",
            "#818cf8",
            "Función pragmática",
        )

        # Filter the table by active insub_func filter
        active_insub = _get_subcat_filter().get("insub_func")
        if active_insub:
            insub_df = insub_df[insub_df["Función pragmática"] == active_insub]

        st.dataframe(insub_df, width="stretch", hide_index=True)
        st.caption(
            "Ej.: '¡Que venga!' · 'Si hubiera sabido…' Frecuentes en habla espontánea y argumentación emocional."
        )
    rarezas = [
        {
            "UCE": u["id"],
            "Tipo": r.get("tipo", ""),
            "Fragmento": r.get("texto", "")[:60],
        }
        for u in uces_all
        for r in u.get("rarezas", [])
        if r.get("tipo") != "NPI_NO_LICENCIADO"
    ]
    if rarezas:
        st.divider()
        st.markdown("**Anomalías sintácticas detectadas**")
        st.dataframe(pd.DataFrame(rarezas), width="stretch", hide_index=True)


def tab_cuantificadores(stats: Dict, uces_all: List[Dict]):
    quant_data = []
    for uce in uces_all:
        for q in uce.get("cuantificadores", []):
            quant_data.append(
                {
                    "tipo": q.get("tipo", ""),
                    "texto": q.get("texto", ""),
                    "cuantifica": q.get("cuantifica_a", ""),
                    "uce": uce["id"],
                }
            )
    if not quant_data:
        st.info("Sin cuantificadores detectados.")
        return

    qd = pd.DataFrame(quant_data)
    cnt = qd.groupby("tipo").size().reset_index(name="n")
    quant_labels = {
        "UNIVERSAL": "Universal (todo, cada)",
        "EXISTENCIAL": "Existencial (algún, varios)",
        "NEGATIVO": "Negativo (ningún, nadie)",
        "NUMERICO": "Numérico (dos, tres…)",
        "PROPORCIONAL": "Proporcional (mitad, mayoría)",
        "CUANTIFICADOR_SEMANTICO": "Semántico (cantidad, número…)",
        "NUMERAL_CARDINAL": "Numeral cardinal",
        "NUMERAL_ORDINAL": "Numeral ordinal",
    }
    cnt["label"] = cnt["tipo"].map(lambda x: quant_labels.get(str(x), str(x)))

    # Flag universals (argumentative absolutism risk)
    n_universal = (
        int(cnt.loc[cnt["tipo"] == "UNIVERSAL", "n"].sum())
        if "UNIVERSAL" in cnt["tipo"].values
        else 0
    )
    if n_universal > 5:
        st.warning(
            f"⚠ {n_universal} cuantificadores universales detectados. "
            "Alta frecuencia puede indicar generalización argumentativa absoluta."
        )

    _filter_bar(
        cnt.nlargest(8, "n"),
        "n",
        "label",
        "quant_tipo",
        "tipo",
        "#4ade80",
        "Tipos de cuantificador",
        zero_labels=list(quant_labels.keys()),
    )
    st.caption(
        "Universales en afirmaciones = generalización. En contexto político = absolutismo argumentativo."
    )

    top_quant = (
        qd.groupby(["tipo", "texto"]).size().reset_index(name="n").nlargest(15, "n")
    )
    top_quant["label"] = top_quant["tipo"].map(
        lambda x: quant_labels.get(str(x), str(x))
    )
    st.markdown("**Formas más frecuentes**")
    st.dataframe(
        top_quant[["label", "texto", "n"]].rename(
            columns={"label": "Tipo", "texto": "Forma", "n": "Frecuencia"}
        ),
        width="stretch",
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════
# SUBCAT FILTER AWARE PARAGRAPH RENDERER
# ══════════════════════════════════════════════════════════════


def _uce_matches_subcat_filters(uce: Dict) -> bool:
    """
    Returns True if the UCE contains at least one annotation matching ALL active subcat filters.
    Filters are ANDed across categories, ORed within (i.e., any verb with mood=Ind counts).
    """
    filters = _get_subcat_filter()
    if not filters:
        return True  # no filter active

    for fkey, fval in filters.items():
        if fkey == "verb_mood":
            if not any(str(v.get("modo", "")) == fval for v in uce.get("verbos", [])):
                return False
        elif fkey == "verb_tense":
            if not any(str(v.get("tiempo", "")) == fval for v in uce.get("verbos", [])):
                return False
        elif fkey == "verb_voice":
            if not any(str(v.get("voz", "")) == fval for v in uce.get("verbos", [])):
                return False
        elif fkey == "verb_pn":
            if not any(
                str(v.get("persona", "")) == fval for v in uce.get("verbos", [])
            ):
                return False
        elif fkey == "verb_sub":
            if not any(
                str(v.get("tipo_subordinacion", "")) == fval
                for v in uce.get("verbos", [])
            ):
                return False
        elif fkey == "neg_scope":
            if not any(
                str(n.get("tipo", "")) == fval for n in uce.get("negaciones", [])
            ):
                return False
        elif fkey == "neg_pragma":
            if not any(
                str(n.get("tipo", "")) == fval for n in uce.get("negaciones", [])
            ):
                return False
        elif fkey == "pron_tipo":
            if not any(
                str(p.get("tipo", "")) == fval for p in uce.get("pronombres", [])
            ):
                return False
        elif fkey == "pron_subtipo":
            if not any(
                str(p.get("subtipo", "")) == fval for p in uce.get("pronombres", [])
            ):
                return False
        elif fkey == "pron_pn":
            if not any(
                str(p.get("persona", "")) == fval for p in uce.get("pronombres", [])
            ):
                return False
        elif fkey == "adv_cat":
            if not any(
                str(a.get("categoria", "")) == fval for a in uce.get("adverbios", [])
            ):
                return False
        elif fkey == "disc_cat":
            if not any(
                str(m.get("categoria", "")) == fval
                for m in uce.get("marcadores_discursivos", [])
            ):
                return False
        elif fkey == "disc_pos":
            if not any(
                str(m.get("posicion", "")) == fval
                for m in uce.get("marcadores_discursivos", [])
            ):
                return False
        elif fkey == "quant_tipo":
            if not any(
                str(q.get("tipo", "")) == fval for q in uce.get("cuantificadores", [])
            ):
                return False
        elif fkey == "predicate_cluster":
            if not any(
                str(pf.get("cluster_id", "")) == fval
                for pf in uce.get("predicate_frames", [])
            ):
                return False
        elif fkey == "insub_func":
            if not any(
                str(i.get("funcion_pragmatica", "")) == fval
                for i in uce.get("insubordinaciones", [])
            ):
                return False
        elif fkey == "coref_entity":
            if not any(
                str(chain.get("representative", "")) == fval
                for chain in uce.get("coref_chains", [])
            ):
                return False
        elif fkey == "predicate_role":
            if not any(
                str(pf.get("thematic_role", "")) == fval
                for pf in uce.get("predicate_frames", [])
            ):
                return False
        elif fkey == "predicate_lemma":
            if not any(
                str(pf.get("verb_lemma", "")) == fval
                for pf in uce.get("predicate_frames", [])
            ):
                return False
    return True


# ══════════════════════════════════════════════════════════════
# PARAGRAPH CARD RENDERER — e-Sword style, no UCE sub-cards
# ══════════════════════════════════════════════════════════════


def verb_class(modo: Optional[str]) -> str:
    m = (modo or "").lower()
    if "ind" in m:
        return "ann-verb-ind"
    if "sub" in m:
        return "ann-verb-sub"
    if "imp" in m:
        return "ann-verb-imp"
    if "cnd" in m or "cond" in m:
        return "ann-verb-cond"
    return "ann-verb-ind"


def reg_pill_html(registro: Optional[str]) -> str:
    r = (registro or "mixto").lower()
    return f'<span class="reg-pill reg-{r}">{r}</span>'


def _entity_color(rep: str) -> str:
    """Deterministic color per coreference entity."""
    palette = ["#2dd4bf", "#f97b6b", "#a78bfa", "#fbbf24", "#5b9cf6", "#4ade80"]
    return palette[hash(rep) % len(palette)]


def _predicate_cluster_color(cluster_id) -> str:
    """Deterministic color per predicate cluster id."""
    palette = [
        "#5b9cf6",
        "#2dd4bf",
        "#a78bfa",
        "#fbbf24",
        "#f97b6b",
        "#4ade80",
        "#c084fc",
        "#f472b6",
        "#fb923c",
        "#38bdf8",
    ]
    if cluster_id is None:
        return "#64748b"
    return palette[int(cluster_id) % len(palette)]


def html_escape(s: str) -> str:
    """Escapa caracteres HTML especiales."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_onclick(tab_name: str, filters: Dict[str, Any]) -> str:
    if not filters:
        return ""
    filters_json = json.dumps(filters, ensure_ascii=False)
    filters_escaped = filters_json.replace('"', "&quot;").replace("'", "\\'")
    return f"onclick=\"sendToPython('{tab_name}', '{filters_escaped}')\" style=\"cursor:pointer;\""


def _clear_specific_filter(filter_key: str):
    """
    Removes a specific filter key from the active subcategory filters
    and triggers a UI rerun to apply the changes immediately.
    """
    # Retrieve the current active filters
    current_filters = _get_subcat_filter()

    # Check if the key exists before trying to pop it
    if filter_key in current_filters:
        current_filters.pop(filter_key)

        # Update the session state
        # (Make sure SUBCAT_FILTER_KEY matches the exact string variable used in your app)
        st.session_state[SUBCAT_FILTER_KEY] = current_filters

        # Force the app to refresh so the UI and charts update without the filter
        st.rerun()


def render_annotated_uce(
    uce: Dict,
    active_layers: set,
    colors: Dict = COLORS,
    theme: Dict = T,
    cls_colors: List = class_colors,
) -> str:
    texto = uce.get("texto", "")
    texto_len = len(texto)
    if not texto:
        return ""

    uce_start = uce.get("start_char")  # may be None

    def _local(char_abs: Optional[int]) -> Optional[int]:
        """Convert absolute offset to local (0..texto_len)."""
        if char_abs is None or uce_start is None:
            return char_abs
        local = char_abs - uce_start
        return local if 0 <= local <= texto_len else None

    def _find_quote(quote: str) -> Optional[int]:
        """Locate quote string inside texto. Returns local start or None."""
        if not quote:
            return None
        pos = texto.find(quote)
        if pos != -1:
            return pos
        pattern = re.escape(quote).replace(r"\ ", r"\s+")
        m = re.search(pattern, texto, re.IGNORECASE)
        return m.start() if m else None

    spans: List[tuple] = []  # (start, end, open_tag, close_tag, priority)

    # --- 1. Negaciones ---
    if "neg" in active_layers:
        for neg in uce.get("negaciones", []):
            s = _local(neg.get("char_start"))
            e = _local(neg.get("char_end"))
            if s is None or e is None or s >= e:
                continue
            tipo = neg.get("tipo", "")
            onclick = _build_onclick(
                "Negación", {"neg_scope": tipo, "neg_pragma": tipo}
            )
            spans.append(
                (
                    s,
                    e,
                    f'<span class="ann-neg clickable-token" title="Negación: {tipo}" {onclick}>',
                    "</span>",
                    10,
                )
            )
            ss = _local(neg.get("alcance_char_start"))
            se = _local(neg.get("alcance_char_end"))
            if ss is not None and se is not None and ss < se and ss != s:
                spans.append((ss, se, '<span class="ann-neg-scope">', "</span>", 5))

    # --- 2. Verbos ---
    if "verb" in active_layers:
        for v in uce.get("verbos", []):
            s = _local(v.get("char_start"))
            e = _local(v.get("char_end"))
            if s is None or e is None or s >= e:
                continue
            cls = verb_class(v.get("modo"))
            pn = (
                f"{v.get('persona', '')}{v.get('numero', '')}"
                if v.get("persona") and v.get("numero")
                else None
            )
            tip = f"{v.get('lema', '')} | {v.get('modo', '')} {v.get('tiempo', '')} {v.get('voz', '')}"
            onclick = _build_onclick(
                "Verbos",
                {
                    "verb_mood": v.get("modo"),
                    "verb_tense": v.get("tiempo"),
                    "verb_voice": v.get("voz"),
                    "verb_pn": pn,
                    "verb_sub": v.get("tipo_subordinacion"),
                },
            )
            spans.append(
                (
                    s,
                    e,
                    f'<span class="{cls} clickable-token" title="{tip}" {onclick}>',
                    "</span>",
                    8,
                )
            )

    # --- 3. Pronombres ---
    if "pron" in active_layers:
        for p in uce.get("pronombres", []):
            tipo = p.get("tipo", "")
            if tipo == "NULO":
                s = _local(p.get("char_start"))
                if s is not None and 0 <= s <= texto_len:
                    onclick = _build_onclick(
                        "Pronombres", {"pron_tipo": "NULO", "pron_pn": p.get("persona")}
                    )
                    spans.append(
                        (
                            s,
                            s,
                            f'<span class="ann-prodrop clickable-token" title="Pro-drop" {onclick}>[∅]</span>',
                            "",
                            9,
                        )
                    )
            elif tipo == "EXPLICITO":
                s = _local(p.get("char_start"))
                e = _local(p.get("char_end"))
                if s is not None and e is not None and s < e:
                    subtipo = p.get("subtipo", "")
                    tip = f"{subtipo}·{p.get('persona', '')}·{p.get('numero', '')}"
                    onclick = _build_onclick(
                        "Pronombres",
                        {
                            "pron_tipo": "EXPLICITO",
                            "pron_subtipo": subtipo,
                            "pron_pn": p.get("persona"),
                        },
                    )
                    spans.append(
                        (
                            s,
                            e,
                            f'<span class="ann-pron clickable-token" title="{tip}" {onclick}>',
                            "</span>",
                            9,
                        )
                    )

    # --- 4. Cuantificadores ---
    if "quant" in active_layers:
        for q in uce.get("cuantificadores", []):
            s = _local(q.get("char_start"))
            e = _local(q.get("char_end"))
            if s is None or e is None or s >= e:
                continue
            onclick = _build_onclick("Global", {"quant_tipo": q.get("tipo")})
            tip = f"Cuantificador: {q.get('tipo', '')}"
            spans.append(
                (
                    s,
                    e,
                    f'<span class="ann-quant-uni clickable-token" title="{tip}" {onclick}>',
                    "</span>",
                    7,
                )
            )

    # --- 5. Adverbios ---
    if "adv" in active_layers:
        for a in uce.get("adverbios", []):
            s = _local(a.get("char_start"))
            e = _local(a.get("char_end"))
            if s is None or e is None or s >= e:
                continue
            onclick = _build_onclick("Adverbios", {"adv_cat": a.get("categoria")})
            tip = f"{a.get('categoria', '')} (conf: {a.get('confianza', 0):.2f})"
            spans.append(
                (
                    s,
                    e,
                    f'<span class="ann-adv clickable-token" title="{tip}" {onclick}>',
                    "</span>",
                    7,
                )
            )

    # --- 6. Marcadores discursivos ---
    if "disc" in active_layers:
        for m in uce.get("marcadores_discursivos", []):
            s = _local(m.get("char_start"))
            e = _local(m.get("char_end"))
            if s is None or e is None or s >= e:
                continue
            onclick = _build_onclick(
                "Discurso",
                {
                    "disc_cat": m.get("categoria"),
                    "disc_pos": m.get("posicion"),
                },
            )
            tip = f"{m.get('categoria', '')} · {m.get('posicion', '')}"
            spans.append(
                (
                    s,
                    e,
                    f'<span class="ann-disc clickable-token" title="{tip}" {onclick}>',
                    "</span>",
                    8,
                )
            )

    # --- 7. Insubordinaciones ---
    if "insub" in active_layers:
        for ins in uce.get("insubordinaciones", []):
            s = _local(ins.get("char_start"))
            e = _local(ins.get("char_end"))
            if s is None or e is None or s >= e:
                continue
            onclick = _build_onclick(
                "Discurso", {"insub_func": ins.get("funcion_pragmatica")}
            )
            tip = f"Insubordinación: {ins.get('funcion_pragmatica', '')}"
            spans.append(
                (
                    s,
                    e,
                    f'<span class="ann-insub clickable-token" title="{tip}" {onclick}>',
                    "</span>",
                    7,
                )
            )

    # --- 8. Discourse agent annotations (ideacion, valencia, etc.) ---
    if "discourse" in active_layers:
        for ann in uce.get("discourse_annotations", []):
            agent = ann.get("trait", "")
            subtype = ann.get("subtype", "")
            confidence = ann.get("confidence", "")
            for span_info in ann.get("spans", []):
                quote = span_info.get("quote", "")
                s = _find_quote(quote)
                if s is None:
                    continue
                e = s + len(quote)
                cls = f"ann-discourse-{agent.lower().replace('_', '-')}"
                tip = f"{agent}: {subtype} (conf: {confidence})"
                onclick = _build_onclick(
                    "Discurso",
                    {
                        "discourse_agent": agent,
                        "discourse_subtype": subtype,
                    },
                )
                spans.append(
                    (
                        s,
                        e,
                        f'<span class="{cls} clickable-token" title="{tip}" {onclick}>',
                        "</span>",
                        8,
                    )
                )

    # --- 9. Correferencias ---
    if "coref" in active_layers:
        highlight_entity = st.session_state.get("highlight_entity", None)
        for chain in uce.get("coref_chains", []):
            rep = chain.get("representative", "")
            color = (
                _entity_color(rep)
                if highlight_entity is None
                else ("#2dd4bf" if rep == highlight_entity else "#4a5270")
            )
            for mention in chain.get("mentions", []):
                s = _local(mention.get("start_char"))
                e = _local(mention.get("end_char"))
                if s is None or e is None or s >= e:
                    continue
                onclick = _build_onclick("Coref", {"entity": rep})
                tip = f"Entidad: {rep}"
                spans.append(
                    (
                        s,
                        e,
                        f'<span class="ann-coref" style="border-bottom:2px solid {color};" title="{tip}" {onclick}>',
                        "</span>",
                        6,
                    )
                )

    # --- 10. Marcos predicativos ---
    if "predicate" in active_layers:
        for pf in uce.get("predicate_frames", []):
            if not isinstance(pf, dict):
                pf = pf.__dict__
            cluster_id = pf.get("cluster_id")
            col = _predicate_cluster_color(cluster_id)
            verb_lemma = pf.get("verb_lemma", "")
            role = pf.get("thematic_role", "")
            voice = pf.get("voice", "")
            tip_base = f"{verb_lemma} | {role} | {voice}"
            if pf.get("negated", False):
                tip_base += " | NEG"
            onclick = _build_onclick(
                "Predicados", {"predicate_cluster": str(cluster_id)}
            )

            s = _local(pf.get("entity_start_char"))
            e = _local(pf.get("entity_end_char"))
            if s is not None and e is not None and 0 <= s < e <= texto_len:
                spans.append(
                    (
                        s,
                        e,
                        f'<span class="ann-pred-subj clickable-token" '
                        f'style="border-bottom:2.5px solid {col};padding-bottom:1px;" '
                        f'title="nsubj · {tip_base}" {onclick}>',
                        "</span>",
                        9,
                    )
                )

            vs = _local(pf.get("verb_start_char"))
            ve = _local(pf.get("verb_end_char"))
            if vs is not None and ve is not None and 0 <= vs < ve <= texto_len:
                spans.append(
                    (
                        vs,
                        ve,
                        f'<span class="ann-pred-verb clickable-token" '
                        f'style="background:{col}22;border-radius:3px;font-weight:600;" '
                        f'title="verbo · {tip_base}" {onclick}>',
                        "</span>",
                        8,
                    )
                )

            os_ = _local(pf.get("direct_object_start_char"))
            oe = _local(pf.get("direct_object_end_char"))
            if os_ is not None and oe is not None and 0 <= os_ < oe <= texto_len:
                spans.append(
                    (
                        os_,
                        oe,
                        f'<span class="ann-pred-obj clickable-token" '
                        f'style="border-bottom:2px dotted {col};padding-bottom:1px;" '
                        f'title="dobj · {tip_base}" {onclick}>',
                        "</span>",
                        7,
                    )
                )

    # ── Assemble HTML ──────────────────────────────────────────────────
    if not spans:
        return f'<span class="annotated-text">{html_escape(texto)}</span>'

    spans.sort(key=lambda x: (x[0], -(x[1] - x[0]), -x[4]))

    events: List[Tuple[int, int, str]] = []
    for s, e, otag, ctag, _ in spans:
        if s == e:
            events.append((s, 0, otag + ctag))
        else:
            events.append((s, 0, otag))
            events.append((e, 1, ctag))
    events.sort(key=lambda x: (x[0], x[1]))

    html_parts = []
    cursor = 0
    for pos, typ, tag in events:
        if pos > cursor:
            html_parts.append(html_escape(texto[cursor:pos]))
            cursor = pos
        html_parts.append(tag)
    if cursor < texto_len:
        html_parts.append(html_escape(texto[cursor:]))

    return f'<span class="annotated-text">{"".join(html_parts)}</span>'


# ══════════════════════════════════════════════════════════════
# PARAGRAPH CARD RENDERER
# ══════════════════════════════════════════════════════════════


def render_paragraph_card(
    par_key: str,
    par_uces: List[Dict],
    inspect_col: str,
    active_layers: set,
    CAT_MAP: Dict,
):
    is_stable = any(u.get("is_stable", False) for u in par_uces)
    matching = [u for u in par_uces if _uce_matches_subcat_filters(u)]
    n_match = len(matching)
    n_total = len(par_uces)
    dim_par = n_match == 0

    ts = par_uces[0].get("topic_shift_prev", 0)
    if ts > 0.4:
        st.html(
            '<div style="height:2px;background:linear-gradient('
            "to right,transparent,#f97b6b44,transparent);"
            'margin:4px 0;border-radius:1px;"></div>'
        )

    stable_badge = (
        ""
        if is_stable
        else '<span style="font-size:9px;color:#4a5270;border:1px solid #2a2f42;'
        'border-radius:8px;padding:1px 5px;margin-left:6px;">inestable</span>'
    )
    filter_badge = ""
    if _get_subcat_filter():
        filter_badge = (
            f'<span style="font-size:9px;color:#4ade80;border:1px solid #166534;'
            f'border-radius:8px;padding:1px 5px;margin-left:6px;">{n_match}/{n_total} UCEs</span>'
            if n_match > 0
            else '<span style="font-size:9px;color:#4a5270;border:1px solid #2a2f42;'
            'border-radius:8px;padding:1px 5px;margin-left:6px;">sin coincidencias</span>'
        )

    with st.expander(f"❡ {par_key} · {n_total} UCEs", expanded=True):
        if stable_badge or filter_badge:
            st.html(
                f"<div style='margin-bottom: 8px;'>{stable_badge}{filter_badge}</div>"
            )
        if dim_par:
            st.caption("Sin UCEs que coincidan con el filtro activo.")
            return

        html_parts = []
        for uce in par_uces:
            is_match = _uce_matches_subcat_filters(uce)
            cid = uce.get("cluster_id")
            classified = cid is not None and cid >= 0
            opacity = "1" if is_match else "0.28"
            uid_short = uce.get("id", "").split("_")[-1]

            if not classified:
                siglum = (
                    f'<sup style="font-family:monospace;font-size:0.6rem;margin-right:3px;'
                    f'color:var(--text-dim);opacity:0.5;" title="UCE no clasificada">∅{uid_short}</sup>'
                )
                border_style = "border-left:2px dashed var(--text-dim);padding-left:6px;opacity:0.45;"
            else:
                cls_col = class_colors.get(cid, T["text_low"])
                siglum = (
                    f'<sup style="font-family:monospace;font-size:0.65rem;margin-right:4px;'
                    f'color:{cls_col};opacity:0.7;">{uid_short}</sup>'
                )
                border_style = ""

            annotated_html = render_annotated_uce(uce, active_layers)
            html_parts.append(
                f'<span style="{border_style}opacity:{opacity};">'
                f"{siglum}{annotated_html}</span> "
            )

        iframe_styles = f"""
        <style>
        :root {{
          --neg:        #e05c5c;
          --neg-scope:  rgba(224,92,92,0.08);
          --pron:       #3ec9c9;
          --verb-ind:   #5b9cf6;
          --verb-sub:   #a78bfa;
          --verb-imp:   #f97b6b;
          --verb-cond:  #fbbf24;
          --quant-uni:  #4ade80;
          --adv:        #c084fc;
          --disc:       #818cf8;
          --coref0:     #2dd4bf;
          --text-hi:    {T["text_hi"]};
        }}
        body {{ color: var(--text-hi); font-family: 'Newsreader', Georgia, serif; margin: 0; }}
        .annotated-text {{ font-size: 0.92rem; }}
        .ann-neg        {{ text-decoration: underline 2px var(--neg); text-underline-offset: 3px; }}
        .ann-neg-scope  {{ background: var(--neg-scope); border-radius: 3px; padding: 0 2px; }}
        .ann-pron       {{ background: rgba(62,201,201,0.18); border-radius: 3px; color: var(--pron); }}
        .ann-verb-ind   {{ color: var(--verb-ind); font-weight: 600; }}
        .ann-verb-sub   {{ color: var(--verb-sub); font-weight: 600; }}
        .ann-verb-imp   {{ color: var(--verb-imp); font-weight: 600; }}
        .ann-verb-cond  {{ color: var(--verb-cond); font-weight: 600; }}
        .ann-adv        {{ color: var(--adv); font-style: italic; }}
        .ann-disc       {{ background: rgba(129,140,248,0.2); color: var(--disc); border-radius: 3px; padding: 0 3px; font-family: 'IBM Plex Mono', monospace; font-size: 0.8em; }}
        .ann-prodrop    {{ color: var(--pron); opacity: 0.6; font-family: 'IBM Plex Mono', monospace; font-size: 0.78em; }}
        .ann-quant-uni  {{ color: var(--quant-uni); }}
        .ann-coref      {{ padding-bottom: 1px; }}
        .ann-insub      {{ background: rgba(129,140,248,0.15); border-radius: 3px; padding: 0 2px; font-style: italic; }}
        .ann-pred-subj  {{ padding-bottom: 1px; }}
        .ann-pred-verb  {{ padding: 0 2px; }}
        .ann-pred-obj   {{ padding-bottom: 1px; }}
        .clickable-token {{ cursor: pointer; transition: filter 0.15s; }}
        .clickable-token:hover {{ filter: brightness(1.35) drop-shadow(0px 0px 2px rgba(255,255,255,0.2)); }}
        </style>
        """

        js_script = """
        <script>
        function sendToPython(tabName, filtersJsonStr) {
            try {
                const payloadObj = {
                    "tab": tabName,
                    "filters": JSON.parse(filtersJsonStr.replace(/&quot;/g, '"')),
                    "ts": Date.now()
                };
                const payload = JSON.stringify(payloadObj);
                const input = window.parent.document.querySelector('input[aria-label="js_bridge"]');
                if (!input) { console.error("JS bridge input not found."); return; }
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLInputElement.prototype, "value"
                ).set;
                nativeInputValueSetter.call(input, payload);
                input.dispatchEvent(new Event('input',  { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
                input.dispatchEvent(new window.parent.KeyboardEvent('keydown',
                    { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }));
                input.dispatchEvent(new window.parent.KeyboardEvent('keyup',
                    { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }));
                input.blur();
            } catch (e) { console.error("sendToPython error:", e); }
        }
        </script>
        """

        final_html = (
            f"{iframe_styles}"
            f'<div style="line-height:2.1;font-size:0.88rem;padding:8px 12px;">'
            f"{''.join(html_parts)}</div>"
            f"{js_script}"
        )
        st.iframe(
            final_html,
            height=max(300, len(par_uces) * 110),
            scrolling=True,
        )


TRAIT_COLORS: Dict[str, str] = {
    "ideacion": "#7F77DD",
    "metafora": "#EF9F27",
    "eufemismo": "#E24B4A",
    "juicios_valor": "#1D9E75",
    "oposiciones": "#D4537E",
    "telicidad": "#5b9cf6",
    "valencia": "#f97b6b",
    "transformacion_semantica": "#c084fc",
    "mitopoetica": "#2dd4bf",
}

# Default colour for unknown traits
_DEFAULT_COLOR = "#888780"

# ── Schema dependency graph (source → targets it enables) ─────
# Read from your schema's disc_cats fields.
DISC_DEPS: Dict[str, List[str]] = {
    "ideacion": [
        "eufemismo",
        "metafora",
        "transformacion_semantica",
        "juicios_valor",
        "oposiciones",
    ],
    "telicidad": [
        "eufemismo",
        "metafora",
        "transformacion_semantica",
        "juicios_valor",
        "oposiciones",
    ],
    "valencia": ["ideacion"],
    "eufemismo": [
        "ideacion",
        "metafora",
        "transformacion_semantica",
        "juicios_valor",
        "oposiciones",
    ],
    "metafora": ["ideacion", "telicidad", "valencia"],
    "transformacion_semantica": ["ideacion", "telicidad", "valencia"],
    "juicios_valor": [
        "eufemismo",
        "metafora",
        "transformacion_semantica",
        "ideacion",
        "telicidad",
        "valencia",
    ],
    "oposiciones": [
        "eufemismo",
        "metafora",
        "transformacion_semantica",
        "ideacion",
        "telicidad",
        "valencia",
    ],
    "mitopoetica": [
        "eufemismo",
        "metafora",
        "transformacion_semantica",
        "ideacion",
        "telicidad",
        "valencia",
        "juicios_valor",
        "oposiciones",
    ],
}

# ── Plotly layout base (copy from your dashboard globals) ─────
PLOTLY_LAYOUT: dict = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", size=11, color="#c2bfff"),
    margin=dict(t=30, b=20, l=10, r=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════


def _trait_color(trait: str) -> str:
    return TRAIT_COLORS.get(trait, _DEFAULT_COLOR)


def _badge_html(confidence: str) -> str:
    conf = (confidence or "").lower()
    colors = {
        "alta": ("rgba(29,158,117,0.15)", "#1D9E75"),
        "media": ("rgba(239,159,39,0.15)", "#EF9F27"),
        "baja": ("rgba(226,75,74,0.15)", "#E24B4A"),
    }
    bg, fg = colors.get(conf, ("rgba(136,135,128,0.15)", "#888780"))
    return (
        f'<span style="padding:1px 7px;border-radius:9px;font-size:9px;'
        f'background:{bg};color:{fg};border:.5px solid {fg}55">{conf}</span>'
    )


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _hex_to_rgba(h: str, alpha: float = 1.0) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _section_label(text: str) -> None:
    st.markdown(
        f'<p style="font-size:9px;letter-spacing:.12em;color:{T["text_dim"]};'
        f'text-transform:uppercase;margin:10px 0 4px;">{text}</p>',
        unsafe_allow_html=True,
    )


def _parse_annotations(
    annotations_by_uce: Dict[str, List[dict]],
    uces: List[dict],
) -> Dict[str, List[dict]]:
    """
    Normalise annotations_by_uce so every annotation is a flat dict
    with guaranteed keys.  Returns a new dict keyed by uce_id.
    """
    result: Dict[str, List[dict]] = {}
    for uce_id, ann_list in annotations_by_uce.items():
        if not isinstance(ann_list, list):
            continue
        normalised = []
        for ann in ann_list:
            if not isinstance(ann, dict):
                continue
            normalised.append(
                {
                    "uce_id": ann.get("uce_id", uce_id),
                    "trait": ann.get("trait") or ann.get("agent", ""),
                    "subtype": ann.get("subtype", ""),
                    "quote": ann.get("quote", ""),
                    "start_char": ann.get("start_char"),
                    "end_char": ann.get("end_char"),
                    "confidence": ann.get("confidence", ""),
                    "reasoning": (ann.get("metadata") or {}).get("reasoning", ""),
                    "meta": ann.get("metadata", {}),
                }
            )
        result[uce_id] = normalised
    return result


# ══════════════════════════════════════════════════════════════
# ANNOTATED TEXT RENDERER (inline HTML)
# ══════════════════════════════════════════════════════════════


def _render_annotated_text(texto: str, annotations: list) -> str:
    """
    Returns an HTML string with annotation spans underlined by trait colour.
    Handles multiple annotations on the same quote and overlapping spans.
    """
    if not texto:
        return ""

    # ── 1. Locate every annotation in the text ────────────────
    # resolved: list of (start, end, trait, subtype, confidence)
    resolved: list = []
    for ann in annotations:
        quote = (ann.get("quote") or "").strip()
        if not quote:
            continue
        trait = ann.get("trait", "")
        subtype = ann.get("subtype", "")
        confidence = ann.get("confidence", "")

        s = texto.find(quote)
        if s == -1:
            import re as _re

            pattern = _re.escape(quote).replace(r"\ ", r"\s+")
            m = _re.search(pattern, texto, _re.IGNORECASE)
            if not m:
                continue
            s, e = m.start(), m.end()
        else:
            e = s + len(quote)

        resolved.append((s, e, trait, subtype, confidence))

    if not resolved:
        return (
            f'<span style="font-size:0.88rem;line-height:1.9">'
            f"{_html_escape(texto)}</span>"
        )

    # ── 2. Group by (start, end) so same-span annotations merge ──
    # merged: { (s,e): [ (trait, subtype, confidence), ... ] }

    span_groups: dict = defaultdict(list)
    for s, e, trait, subtype, confidence in resolved:
        span_groups[(s, e)].append((trait, subtype, confidence))

    # ── 3. Build one open/close tag per unique (s, e) ─────────
    # For merged spans we stack underlines via box-shadow trick.
    spans: list = []  # (start, end, open_tag, close_tag)
    for (s, e), entries in span_groups.items():
        cols = [_trait_color(t) for t, _, _ in entries]
        tips = " | ".join(f"{t}·{st_}·{c}" for t, st_, c in entries)
        # Primary underline (border-bottom)
        primary_color = cols[0]
        # Extra underlines via box-shadow (offset downward by 3px each)
        shadow_parts = []
        for i, col in enumerate(cols[1:], start=1):
            shadow_parts.append(f"0 {3 * i}px 0 {col}")
        shadow_css = f"box-shadow:{','.join(shadow_parts)};" if shadow_parts else ""
        open_tag = (
            f'<span style="border-bottom:2px solid {primary_color};'
            f"padding-bottom:{1 + 3 * (len(cols) - 1)}px;"
            f"{shadow_css}"
            f'border-radius:2px;" title="{_html_escape(tips)}">'
        )
        spans.append((s, e, open_tag, "</span>"))

    # ── 4. Sort spans: by start asc, then by length desc (outer first) ──
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # ── 5. Build event list ────────────────────────────────────
    # Each event: (position, type, tag)
    #   type 0 = open, type 1 = close
    events: list = []
    for s, e, otag, ctag in spans:
        events.append((s, 0, otag))
        events.append((e, 1, ctag))
    # Sort: position first; at same position, closes before opens
    # so we don't nest incorrectly
    events.sort(key=lambda x: (x[0], x[1]))

    # ── 6. Assemble HTML ──────────────────────────────────────
    parts = []
    cursor = 0
    texto_len = len(texto)
    for pos, typ, tag in events:
        pos = max(0, min(pos, texto_len))
        if pos > cursor:
            parts.append(_html_escape(texto[cursor:pos]))
            cursor = pos
        parts.append(tag)
    if cursor < texto_len:
        parts.append(_html_escape(texto[cursor:]))

    return f'<span style="font-size:0.88rem;line-height:1.9">{"".join(parts)}</span>'


def _build_cooc(
    uces_clase: List[dict],
    annotations_by_uce: Dict[str, List[dict]],
    traits: List[str],
) -> pd.DataFrame:
    cooc = pd.DataFrame(0, index=traits, columns=traits, dtype=int)
    for u in uces_clase:
        uid = u.get("id", "")
        anns = annotations_by_uce.get(uid, [])
        present = {a["trait"] for a in anns if a.get("trait")}
        unique = [t for t in traits if t in present]
        for i, t1 in enumerate(unique):
            for t2 in unique[i:]:
                cooc.loc[t1, t2] += 1
                if t1 != t2:
                    cooc.loc[t2, t1] += 1
    return cooc


# ══════════════════════════════════════════════════════════════
# DEPENDENCY GRAPH SVG
# ══════════════════════════════════════════════════════════════


def _dep_graph_html(active_trait: Optional[str], present_traits: Set[str]) -> str:
    """
    Renders the schema dependency graph as static HTML+SVG.
    Nodes present in the corpus are bright; absent ones are dimmed.
    The active_trait node is highlighted.
    """
    # Fixed positions (x,y) inside a 320×220 canvas
    NODE_POS: Dict[str, Tuple[int, int]] = {
        "ideacion": (130, 30),
        "eufemismo": (50, 90),
        "metafora": (210, 90),
        "valencia": (50, 155),
        "telicidad": (130, 155),
        "transformacion_semantica": (210, 155),
        "juicios_valor": (85, 210),
        "oposiciones": (175, 210),
        "mitopoetica": (130, 265),
    }
    NODE_LABELS: Dict[str, str] = {
        "ideacion": "ideacion",
        "eufemismo": "eufemismo",
        "metafora": "metafora",
        "valencia": "valencia",
        "telicidad": "telicidad",
        "transformacion_semantica": "trans.sem.",
        "juicios_valor": "juicios",
        "oposiciones": "oposiciones",
        "mitopoetica": "mitopoetica",
    }

    # Draw edges
    edge_lines = []
    for src, targets in DISC_DEPS.items():
        if src not in NODE_POS:
            continue
        sx, sy = NODE_POS[src]
        for tgt in targets:
            if tgt not in NODE_POS:
                continue
            tx, ty = NODE_POS[tgt]
            edge_lines.append(
                f'<line x1="{sx}" y1="{sy}" x2="{tx}" y2="{ty}" '
                f'stroke="var(--border2,#2a2f42)" stroke-width="0.6" opacity="0.5"/>'
            )

    # Draw nodes
    node_els = []
    for trait, (mx, my) in NODE_POS.items():
        col = _trait_color(trait)
        label = NODE_LABELS.get(trait, trait)
        present = trait in present_traits
        is_active = trait == active_trait

        if is_active:
            fill = col
            text_col = "#ffffff"
            stroke = col
            opacity = "1"
        elif present:
            fill = col + "22"
            text_col = col
            stroke = col
            opacity = "1"
        else:
            fill = "transparent"
            text_col = "#4a5270"
            stroke = "#2a2f42"
            opacity = "0.5"

        node_els.append(
            f'<g style="cursor:pointer" opacity="{opacity}">'
            f'<rect x="{mx - 34}" y="{my - 11}" width="68" height="22" rx="11" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>'
            f'<text x="{mx}" y="{my + 1}" text-anchor="middle" dominant-baseline="central" '
            f'style="font-family:IBM Plex Mono,monospace;font-size:9px;fill:{text_col}">'
            f"{label}</text></g>"
        )

    viewbox_h = 300
    svg = (
        f'<svg width="100%" viewBox="0 0 320 {viewbox_h}" '
        f'xmlns="http://www.w3.org/2000/svg" style="overflow:visible">'
        + "".join(edge_lines)
        + "".join(node_els)
        + "</svg>"
    )
    return svg


# ══════════════════════════════════════════════════════════════
# UCE CARD HTML
# ══════════════════════════════════════════════════════════════


def _uce_card_html(
    uce: dict,
    annotations: List[dict],
    active_traits: Set[str],
    class_color: str,
) -> str:
    uid = uce.get("id", "?")
    uid_short = uid.split("_")[-1]
    texto = uce.get("texto", "")

    # Filter annotations to active traits only
    visible_anns = (
        [a for a in annotations if a.get("trait") in active_traits]
        if active_traits
        else annotations
    )
    n_strat = len({a.get("trait") for a in visible_anns})

    # Strategy count badge
    if n_strat >= 3:
        badge_bg, badge_fg = "rgba(29,158,117,0.15)", "#1D9E75"
    elif n_strat >= 2:
        badge_bg, badge_fg = "rgba(239,159,39,0.15)", "#EF9F27"
    elif n_strat == 1:
        badge_bg, badge_fg = "rgba(91,156,246,0.15)", "#5b9cf6"
    else:
        badge_bg, badge_fg = "rgba(74,82,112,0.1)", "#4a5270"

    badge = (
        f'<span style="padding:1px 8px;border-radius:9px;font-size:9px;'
        f'background:{badge_bg};color:{badge_fg};border:.5px solid {badge_fg}55">'
        f"{n_strat} estrategia{'s' if n_strat != 1 else ''}</span>"
        if n_strat > 0
        else '<span style="font-size:9px;color:#4a5270">sin anotaciones</span>'
    )

    opacity = "1" if (not active_traits or n_strat > 0) else "0.3"

    annotated = _render_annotated_text(texto, visible_anns)

    # Annotation chips
    chip_html = ""
    for ann in visible_anns:
        trait = ann.get("trait", "")
        subtype = ann.get("subtype", "")
        conf = ann.get("confidence", "")
        col = _trait_color(trait)
        chip_html += (
            f'<span style="padding:1px 8px;border-radius:9px;font-size:9px;'
            f"background:{col}18;color:{col};border:.5px solid {col}44;"
            f'margin:2px;display:inline-block">'
            f"{_html_escape(trait)} · {_html_escape(subtype)} {_badge_html(conf)}</span>"
        )

    reasoning_html = ""
    for ann in visible_anns:
        r = ann.get("reasoning", "")
        if r:
            col = _trait_color(ann.get("trait", ""))
            reasoning_html += (
                f'<div style="font-size:10px;color:#8892b0;font-style:italic;'
                f'border-left:2px solid {col}44;padding-left:6px;margin-top:4px;">'
                f"{_html_escape(r)}</div>"
            )

    return f"""
<div style="border:0.5px solid #1e2235;border-left:3px solid {class_color};
            border-radius:6px;padding:10px 12px;margin-bottom:8px;
            background:#0d1117;opacity:{opacity}">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;flex-wrap:wrap">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#4a5270">
      {_html_escape(uid_short)}</span>
    {badge}
    <span style="margin-left:auto;font-size:9px;color:#4a5270">{_html_escape(str(uce.get("doc_id", "")))}</span>
  </div>
  <div style="line-height:1.9;margin-bottom:6px">{annotated}</div>
  {f'<div style="margin-top:4px;flex-wrap:wrap;display:flex;gap:2px">{chip_html}</div>' if chip_html else ""}
  {reasoning_html}
</div>"""


# ══════════════════════════════════════════════════════════════
# SUBTYPE BAR CHART
# ══════════════════════════════════════════════════════════════


def _subtype_chart(trait: str, anns: List[dict]) -> Optional[go.Figure]:
    counts = Counter(a.get("subtype", "") for a in anns if a.get("trait") == trait)
    if not counts:
        return None
    df = pd.DataFrame(counts.most_common(), columns=["Subtipo", "n"])
    col = _trait_color(trait)
    fig = go.Figure(
        go.Bar(
            x=df["n"],
            y=df["Subtipo"],
            orientation="h",
            marker_color=col,
            text=df["n"],
            textposition="outside",
        )
    )
    fig.update_layout(
        PLOTLY_LAYOUT,
        height=max(140, len(df) * 28 + 60),
        title=dict(text=f"Subtipos · {trait}", font=dict(size=11)),
        xaxis=dict(visible=False),
        margin=dict(t=36, b=10, l=10, r=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# CONFIDENCE SUMMARY METRICS
# ══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — HEATMAP TRAYECTORIA
# ─────────────────────────────────────────────────────────────────────────────


def _compute_trayectoria_matrix(
    df: pd.DataFrame,
    uce_order: List[str],
    active_traits: List[str],
    active_classes: List[Any],
    window: int = 5,
) -> Dict[Any, pd.DataFrame]:
    """
    For each class, returns a DataFrame indexed by window-position (int)
    and columns = traits, values = annotation count normalised by
    n_uces in that class-window (so counts are per-UCE density).

    window: number of UCEs to group per cell (reduces sparsity).
    """
    uce_pos = {uid: i for i, uid in enumerate(uce_order)}
    df_w = df.copy()
    df_w["uce_pos"] = df_w["uce_id"].map(uce_pos)
    df_w = df_w.dropna(subset=["uce_pos"])
    df_w["uce_pos"] = df_w["uce_pos"].astype(int)
    n_pos = len(uce_order)
    n_windows = max(1, (n_pos + window - 1) // window)
    df_w["window"] = df_w["uce_pos"] // window

    result = {}
    for cls in active_classes:
        sub = df_w[df_w["cluster_id"] == cls] if cls is not None else df_w
        # n_uces per window (denominator)
        uces_in_class = (
            [
                u
                for u, p in uce_pos.items()
                if df_w[(df_w["uce_id"] == u)]["cluster_id"].eq(cls).any()
            ]
            if cls is not None
            else list(uce_pos.keys())
        )

        # Count annotations per (window, trait)
        ct = (
            sub[sub["trait"].isin(active_traits)]
            .groupby(["window", "trait"])
            .size()
            .reset_index(name="n")
        )
        if ct.empty:
            result[cls] = pd.DataFrame(
                0.0,
                index=range(n_windows),
                columns=active_traits,
            )
            continue

        pivot = ct.pivot_table(
            index="window", columns="trait", values="n", fill_value=0
        )
        # Reindex to full window range
        pivot = pivot.reindex(
            index=range(n_windows), columns=active_traits, fill_value=0
        )

        # Normalise: divide each window row by n_uces in that window for this class
        uces_per_window = (
            sub.drop_duplicates("uce_id")
            .assign(window=lambda d: d["uce_pos"] // window)
            .groupby("window")["uce_id"]
            .nunique()
            .reindex(range(n_windows), fill_value=1)
        )
        pivot = pivot.div(uces_per_window, axis=0).fillna(0)
        result[cls] = pivot.astype(float)

    return result


def _chart_heatmap_trayectoria(
    df: pd.DataFrame,
    uce_order: List[str],
    active_traits: List[str],
    active_classes: List[Any],
    class_colors: Dict[Any, str],
    window: int = 5,
) -> go.Figure:
    """
    Subplot grid: one row per active class.
    Each subplot is a heatmap: x=UCE-window, y=trait, z=density.
    """
    if not active_classes or not active_traits:
        fig = go.Figure(layout=PLOTLY_LAYOUT)
        fig.add_annotation(
            text="Sin datos",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=T["text_dim"]),
        )
        return fig

    matrices = _compute_trayectoria_matrix(
        df, uce_order, active_traits, active_classes, window=window
    )

    n_cls = len(active_classes)
    # Dynamic height: 40px per trait × n_classes + margins
    cell_h = max(28, 200 // max(len(active_traits), 1))
    fig_h = max(280, (len(active_traits) * cell_h + 60) * n_cls + 20)

    subplot_titles = [f"Clase {c}" for c in active_classes]
    fig = make_subplots(
        rows=n_cls,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06 / max(n_cls, 1),
        shared_xaxes=True,
    )

    for row_i, cls in enumerate(active_classes, start=1):
        col = class_colors.get(cls, T["accent"])
        mat = matrices.get(cls, pd.DataFrame())
        if mat.empty:
            continue

        z = mat.values.T  # shape: (n_traits, n_windows)
        x_labs = [f"w{i * window}–{(i + 1) * window - 1}" for i in range(mat.shape[0])]
        y_labs = list(mat.columns)

        # Custom colorscale anchored to class colour
        cs = [
            [0.0, T["bg_page"]],
            [0.4, _hex_to_rgba(col, 0.25)],
            [1.0, col],
        ]

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x_labs,
                y=y_labs,
                colorscale=cs,
                zmin=0,
                showscale=(row_i == 1),
                colorbar=dict(
                    len=0.3,
                    y=1.0,
                    yanchor="top",
                    thickness=8,
                    tickfont=dict(size=8, color=T["text_dim"]),
                    title=dict(
                        text="ann/UCE",
                        side="right",
                        font=dict(size=8, color=T["text_dim"]),
                    ),
                ),
                hovertemplate=(
                    f"<b>Clase {cls}</b><br>"
                    "Ventana: %{x}<br>"
                    "Rasgo: %{y}<br>"
                    "Densidad: %{z:.3f} ann/UCE<extra></extra>"
                ),
                xgap=1,
                ygap=1,
            ),
            row=row_i,
            col=1,
        )

    # Style axes
    for row_i in range(1, n_cls + 1):
        xa = f"xaxis{row_i if row_i > 1 else ''}"
        ya = f"yaxis{row_i if row_i > 1 else ''}"
        fig.update_layout(
            **{
                xa: dict(
                    showticklabels=(row_i == n_cls),
                    tickfont=dict(size=8, color=T["text_dim"]),
                    gridcolor=T["border"],
                    linecolor=T["border"],
                ),
                ya: dict(
                    tickfont=dict(size=9, color=T["text_hi"]),
                    gridcolor=T["border"],
                    linecolor=T["border"],
                    autorange="reversed",
                ),
            }
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=fig_h,
        title=dict(
            text=f"Trayectoria: densidad rasgo × posición UCE  (ventana={window})",
            font=dict(size=10, color=T["text_dim"]),
            x=0,
            xanchor="left",
        ),
        margin=dict(t=52, b=30, l=8, r=50),
        showlegend=False,
    )
    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=9, color=T["text_dim"], family="IBM Plex Mono, monospace")

    return fig


@st.cache_data(show_spinner=False)
def _compute_subtype_stats(df_json: str) -> pd.DataFrame:
    """
    From the annotation DataFrame, compute per-(trait, subtype, cluster_id):
      n          count of spans
      mean_conf  mean conf_rank
      mean_phi   mean phi_score
      pct_biv    % with quant_valency in bivalent/trivalent
      pct_omit   % with n_omitted > 0
    """
    df = pd.read_json(df_json, orient="records")
    if df.empty or "subtype" not in df.columns:
        return pd.DataFrame()

    df = df[df["subtype"].notna() & (df["subtype"] != "")].copy()
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby(["trait", "subtype", "cluster_id"])
        .agg(
            n=("span_idx", "count"),
            mean_conf=("conf_rank", "mean"),
            mean_phi=("phi_score", "mean"),
            pct_biv=(
                "quant_valency",
                lambda x: (x.isin(["bivalent", "trivalent"])).mean() * 100,
            ),
            pct_omit=("n_omitted", lambda x: (x > 0).mean() * 100),
        )
        .reset_index()
    )
    agg["mean_conf"] = agg["mean_conf"].round(2)
    agg["mean_phi"] = agg["mean_phi"].round(3)
    agg["pct_biv"] = agg["pct_biv"].round(1)
    agg["pct_omit"] = agg["pct_omit"].round(1)
    return agg


def _chart_subtype_bars(
    stats: pd.DataFrame,
    trait: str,
    active_classes: List[Any],
    class_colors: Dict[Any, str],
) -> go.Figure:
    """
    Grouped horizontal bars: for the selected trait,
    subtypes on Y axis, bars grouped by class.
    """
    sub = stats[
        (stats["trait"] == trait) & (stats["cluster_id"].isin(active_classes))
    ].copy()

    if sub.empty:
        fig = go.Figure(layout=PLOTLY_LAYOUT)
        fig.add_annotation(
            text=f"Sin subtipos para '{trait}'",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=T["text_dim"], size=11),
        )
        return fig

    subtypes = (
        sub.groupby("subtype")["n"].sum().sort_values(ascending=True).index.tolist()
    )

    fig = go.Figure()
    for cls in active_classes:
        col = class_colors.get(cls, T["accent"])
        csub = sub[sub["cluster_id"] == cls].set_index("subtype")
        vals = [int(csub.loc[st, "n"]) if st in csub.index else 0 for st in subtypes]
        fig.add_trace(
            go.Bar(
                y=subtypes,
                x=vals,
                orientation="h",
                name=f"Cl.{cls}",
                marker=dict(
                    color=col, opacity=0.82, line=dict(color=T["bg_hover"], width=0.5)
                ),
                hovertemplate=(
                    f"<b>%{{y}}</b> · Clase {cls}<br>n: %{{x}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        height=max(200, 28 * len(subtypes) + 80),
        xaxis=dict(
            gridcolor=T["border"], linecolor=T["border"], zerolinecolor=T["border"]
        ),
        yaxis=dict(gridcolor=T["border"], linecolor=T["border"]),
        showlegend=True,
        legend=dict(orientation="h", y=1.04, font=dict(size=9)),
        margin=dict(t=32, b=16, l=4, r=4),
    )
    return fig


def _chart_subtype_scatter(
    stats: pd.DataFrame,
    trait: str,
    active_classes: List[Any],
    class_colors: Dict[Any, str],
    x_col: str = "mean_phi",
    y_col: str = "mean_conf",
) -> go.Figure:
    """
    Scatter: mean_phi × mean_conf for each (subtype, class) pair.
    Size = n. Reveals which subtypes cluster in high-certainty / high-φ zones.
    """
    sub = stats[
        (stats["trait"] == trait) & (stats["cluster_id"].isin(active_classes))
    ].copy()

    if sub.empty:
        return go.Figure(layout=PLOTLY_LAYOUT)

    fig = go.Figure()
    for cls in active_classes:
        col = class_colors.get(cls, T["accent"])
        csub = sub[sub["cluster_id"] == cls]
        if csub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=csub[x_col],
                y=csub[y_col],
                mode="markers+text",
                marker=dict(
                    size=6 + csub["n"] * 2,
                    color=col,
                    opacity=0.75,
                    line=dict(color=T["bg_hover"], width=0.5),
                    sizemode="area",
                    sizeref=max(csub["n"]) / (18**2) if csub["n"].max() > 0 else 1,
                ),
                text=csub["subtype"].str[:14],
                textfont=dict(size=8, color=T["text_hi"]),
                textposition="top center",
                name=f"Cl.{cls}",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"Clase {cls}<br>"
                    f"{x_col}: %{{x:.3f}}<br>"
                    f"{y_col}: %{{y:.2f}}<br>"
                    "n: %{marker.size:.0f}<extra></extra>"
                ),
                customdata=csub["n"].values,
            )
        )

    x_label = {
        "mean_phi": "φ medio de UCE",
        "mean_conf": "confianza media",
        "pct_biv": "% bivalente+",
        "pct_omit": "% arg. omitido",
    }.get(x_col, x_col)
    y_label = {
        "mean_phi": "φ medio de UCE",
        "mean_conf": "confianza media",
        "pct_biv": "% bivalente+",
        "pct_omit": "% arg. omitido",
    }.get(y_col, y_col)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        xaxis=dict(
            title=x_label,
            gridcolor=T["border"],
            linecolor=T["border"],
            zerolinecolor=T["border"],
        ),
        yaxis=dict(title=y_label, gridcolor=T["border"], linecolor=T["border"]),
        showlegend=True,
        legend=dict(orientation="h", y=1.04, font=dict(size=9)),
        margin=dict(t=32, b=24, l=8, r=8),
    )
    return fig


def _render_subtype_tab(
    df_filt: pd.DataFrame,
    active_traits: Set[str],
    active_classes: List[Any],
    class_colors: Dict[Any, str],
) -> None:
    """Renders the full subtype analysis UI."""
    if df_filt.empty:
        st.info("Sin datos de subtipos.")
        return

    stats = _compute_subtype_stats(df_filt.to_json(orient="records"))
    if stats.empty:
        st.info("Las anotaciones actuales no tienen campo 'subtype' poblado.")
        return

    traits_with_subtypes = sorted(
        stats[stats["cluster_id"].isin(active_classes)]["trait"].unique()
    )
    if not traits_with_subtypes:
        st.info("Sin subtipos para las clases seleccionadas.")
        return

    # Trait selector pill row
    if (
        "te_sub_trait" not in st.session_state
        or st.session_state["te_sub_trait"] not in traits_with_subtypes
    ):
        st.session_state["te_sub_trait"] = traits_with_subtypes[0]

    pill_cols = st.columns(len(traits_with_subtypes))
    for i, tr in enumerate(traits_with_subtypes):
        with pill_cols[i]:
            active = st.session_state["te_sub_trait"] == tr
            if st.button(
                tr,
                key=f"te_sub_trait_{tr}",
                type="primary" if active else "secondary",
                use_container_width=True,
            ):
                st.session_state["te_sub_trait"] = tr
                st.rerun()

    sel_trait = st.session_state["te_sub_trait"]

    # Axis selectors for scatter
    s1, s2 = st.columns(2)
    scatter_opts = {
        "φ medio de UCE": "mean_phi",
        "Confianza media": "mean_conf",
        "% bivalente+": "pct_biv",
        "% arg. omitido": "pct_omit",
    }
    with s1:
        x_lbl = st.selectbox(
            "Eje X · scatter",
            list(scatter_opts),
            index=0,
            key="te_sub_x",
            label_visibility="collapsed",
        )
    with s2:
        y_lbl = st.selectbox(
            "Eje Y · scatter",
            list(scatter_opts),
            index=1,
            key="te_sub_y",
            label_visibility="collapsed",
        )

    # Charts
    _section_label(f"Distribución de subtipos · {sel_trait}")
    st.plotly_chart(
        _chart_subtype_bars(stats, sel_trait, active_classes, class_colors),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    _section_label("Correlación subtipo × métricas UCE")
    st.plotly_chart(
        _chart_subtype_scatter(
            stats,
            sel_trait,
            active_classes,
            class_colors,
            x_col=scatter_opts[x_lbl],
            y_col=scatter_opts[y_lbl],
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Detailed table
    with st.expander("Tabla completa de subtipos", expanded=False):
        view = (
            stats[
                (stats["trait"] == sel_trait)
                & (stats["cluster_id"].isin(active_classes))
            ]
            .sort_values("n", ascending=False)
            .reset_index(drop=True)
        )
        view.columns = [
            "Rasgo",
            "Subtipo",
            "Clase",
            "n",
            "Conf.media",
            "φ medio",
            "% bivalente",
            "% omitido",
        ]
        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Conf.media": st.column_config.ProgressColumn(
                    min_value=0, max_value=3, format="%.2f"
                ),
                "% bivalente": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.1f%%"
                ),
                "% omitido": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )


def _confidence_metrics_html(anns: List[dict]) -> str:
    total = len(anns)
    if total == 0:
        return ""
    counts = Counter(a.get("confidence", "").lower() for a in anns)
    metrics = [
        ("alta", "#1D9E75", "rgba(29,158,117,0.12)"),
        ("media", "#EF9F27", "rgba(239,159,39,0.12)"),
        ("baja", "#E24B4A", "rgba(226,75,74,0.12)"),
    ]
    cells = ""
    for label, fg, bg in metrics:
        pct = round(counts.get(label, 0) / total * 100)
        cells += (
            f'<div style="flex:1;text-align:center;padding:8px 4px;'
            f'background:{bg};border-radius:6px">'
            f'<div style="font-size:18px;font-weight:500;color:{fg}">{pct}%</div>'
            f'<div style="font-size:9px;color:{fg}">{label}</div></div>'
        )
    return f'<div style="display:flex;gap:6px;margin-top:4px">{cells}</div>'


# ─────────────────────────────────────────────────────────────
# PESTAÑAS PRINCIPALES
# ─────────────────────────────────────────────────────────────
tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs(
    [
        "A · Análisis gramatical",
        "B · Resultados de clasificación",
        "C · Análisis por clase",
        "D· Correlaciones y calidad",
        "E · Discurso por clase",
    ]
)


# ══════════════════════════════════════════════════════════════
# PESTAÑA A
# ══════════════════════════════════════════════════════════════

with tab_a:
    all_doc_ids_a = sorted(origen_index.keys())

    # ── Barra superior ─────────────────────────────────────────
    top_cols = st.columns([3, 5, 2])
    with top_cols[0]:
        selected_docs = st.multiselect(
            "Documents",
            options=all_doc_ids_a,
            default=None,
            label_visibility="collapsed",
            placeholder="Filter by document…",
        )
        if not selected_docs:
            selected_docs = all_doc_ids_a

    with top_cols[1]:
        LAYERS = {
            "neg": "Negación",
            "pron": "Pronombres",
            "verb": "Verbos",
            "quant": "Cuantificadores",
            "adv": "Adverbios",
            "disc": "Marcadores",
            "coref": "Correferencia",
            "lex": "Léxico",
            "discourse": "Discurso",
            "insub": "Insubordinaciones",
            "predicate": "Marcos predicativos",
        }
        active_layers = set(
            st.multiselect(
                "Layers",
                options=list(LAYERS.keys()),
                format_func=lambda x: LAYERS[x],
                default=["neg", "verb", "pron"],
                label_visibility="collapsed",
                placeholder="Annotation layers…",
            )
        )

    # Build UCE list filtered by selected documents
    uce_lookup_a = {u["id"]: u for u in uces}
    all_uces_filtered: List[Dict] = []
    for doc_key in selected_docs:
        for uid in origen_index.get(doc_key, []):
            u = uce_lookup_a.get(uid)
            if u:
                all_uces_filtered.append(u)

    # ── PRECOMPUTE SHARED DATA STRUCTURES ──────────────────────────
    # Chain index for coreference (used by both columns)
    chain_index: Dict[str, Dict] = {}
    for u in all_uces_filtered:
        for chain in u.get("coref_chains", []):
            rep = chain.get("representative", "").strip()
            if not rep:
                continue
            if rep not in chain_index:
                chain_index[rep] = {"mentions": set(), "uce_ids": []}
            chain_index[rep]["uce_ids"].append(u["id"])
            for m in chain.get("mentions", []):
                txt = m.get("text", "").strip()
                if txt:
                    chain_index[rep]["mentions"].add(txt)

    # All predicate frames (used by both columns)
    all_frames: List[Dict] = []
    for u in all_uces_filtered:
        for pf in u.get("predicate_frames", []):
            frame = dict(pf) if not isinstance(pf, dict) else dict(pf)
            frame["_uce_id"] = u.get("id", "")
            all_frames.append(frame)

    # Now proceed with stats and column layout...
    stats_a = _build_stats_from_uces(all_uces_filtered)
    stats_a["predicate_frames"] = data.get("predicate_frames", {})

    def _pred_matches(f: dict) -> bool:
        """
        Returns True if frame f passes the current predicate sub-filters.
        Reads directly from st.session_state so it's always current.
        Safe to call from both columns.
        """
        voice_filter = st.session_state.get("pred_filter_voice", "Todas")
        role_filter = st.session_state.get("pred_filter_role", "Todos")
        negated_filter = st.session_state.get("pred_filter_negated", "Ambos")

        if voice_filter != "Todas" and f.get("voice") != voice_filter:
            return False
        if role_filter != "Todos" and f.get("thematic_role") != role_filter:
            return False
        neg_val = f.get("negated", False)
        if negated_filter == "Sí" and not neg_val:
            return False
        if negated_filter == "No" and neg_val:
            return False
        return True

    # ══════════════════════════════════════════════════════════
    # COLUMNAS PRINCIPALES  izq=stats/filtros  der=corpus
    # ══════════════════════════════════════════════════════════
    col_a, col_b = st.columns([35, 65])

    # ── Columna izquierda: pestañas de análisis ────────────────
    with col_a:
        opciones_tabs = [
            "Global",
            "Verbos",
            "Negación",
            "Pronombres",
            "Adverbios",
            "Discurso",
            "Coref",
            "Predicados",
        ]

        if "radio_tabs_a" not in st.session_state:
            st.session_state.radio_tabs_a = "Global"

        active_tab = st.radio(
            "Pestañas A",
            options=opciones_tabs,
            horizontal=True,
            label_visibility="collapsed",
            key="radio_tabs_a",
        )

        # ── Pestañas existentes (sin cambios) ─────────────────
        if active_tab == "Global":
            tab_global(stats_a, all_uces_filtered)
        elif active_tab == "Verbos":
            tab_verbos(stats_a, all_uces_filtered)
        elif active_tab == "Negación":
            tab_negacion(stats_a, all_uces_filtered)
        elif active_tab == "Pronombres":
            tab_pronombres(stats_a, all_uces_filtered)
        elif active_tab == "Adverbios":
            tab_adverbios(stats_a, all_uces_filtered)
        elif active_tab == "Discurso":
            tab_discurso(stats_a, all_uces_filtered)

        # ══════════════════════════════════════════════════════
        # PESTAÑA COREF — rediseñada
        # ══════════════════════════════════════════════════════
        elif active_tab == "Coref":
            # 1. Construir índice de cadenas agrupadas por representative
            # ─────────────────────────────────────────────────────────────
            #  chain_index: { representative: { mentions:[str], uce_ids:[str] } }
            chain_index: Dict[str, Dict] = {}
            for u in all_uces_filtered:
                for chain in u.get("coref_chains", []):
                    rep = chain.get("representative", "").strip()
                    if not rep:
                        continue
                    if rep not in chain_index:
                        chain_index[rep] = {"mentions": set(), "uce_ids": []}
                    chain_index[rep]["uce_ids"].append(u["id"])
                    for m in chain.get("mentions", []):
                        txt = m.get("text", "").strip()
                        if txt:
                            chain_index[rep]["mentions"].add(txt)

            # Convertir a lista ordenada por nº de menciones desc
            sorted_chains = sorted(
                chain_index.items(),
                key=lambda kv: len(kv[1]["uce_ids"]),
                reverse=True,
            )

            if not sorted_chains:
                st.info(
                    "No hay entidades correferentes en los documentos seleccionados."
                )
            else:
                # 2. Selector de entidad
                # ─────────────────────────────────────────────
                st.markdown(
                    f'<p style="font-family:var(--font-mono);font-size:9px;'
                    f"text-transform:uppercase;letter-spacing:.1em;"
                    f'color:var(--text-low);margin-bottom:6px">'
                    f"{len(sorted_chains)} entidades correferentes — ordenadas por frecuencia</p>",
                    unsafe_allow_html=True,
                )

                # Columnas rep/count para cada cadena
                max_uces = len(sorted_chains[0][1]["uce_ids"]) if sorted_chains else 1

                for rep, info in sorted_chains:
                    n_uces_chain = len(info["uce_ids"])
                    n_mentions = len(info["mentions"])
                    bar_pct = int(n_uces_chain / max_uces * 100)
                    is_selected = st.session_state.get("coref_selected_entity") == rep
                    entity_color = _entity_color(rep)

                    # Card clicable por entidad
                    label_html = (
                        f'<div style="display:flex;align-items:center;gap:8px;'
                        f"padding:6px 8px;border-radius:6px;margin-bottom:3px;cursor:pointer;"
                        f"border:0.5px solid {'' + entity_color + '88' if is_selected else 'var(--border)'};"
                        f'background:{"" + entity_color + "14" if is_selected else "transparent"};">'
                        f'<div style="width:8px;height:8px;border-radius:50%;'
                        f'background:{entity_color};flex-shrink:0"></div>'
                        f'<span style="font-family:var(--font-mono);font-size:11px;'
                        f"font-weight:{'600' if is_selected else '400'};"
                        f'color:{"var(--text-hi)" if is_selected else "var(--text-mid)"};flex:1">'
                        f"{sh(rep)}</span>"
                        f'<div style="height:3px;width:60px;background:var(--bg-hover);'
                        f'border-radius:2px;overflow:hidden">'
                        f'<div style="height:100%;width:{bar_pct}%;background:{entity_color};'
                        f'border-radius:2px"></div></div>'
                        f'<span style="font-family:var(--font-mono);font-size:9px;'
                        f'color:var(--text-low);min-width:32px;text-align:right">'
                        f"{n_uces_chain} UCE</span>"
                        f"</div>"
                    )
                    st.markdown(label_html, unsafe_allow_html=True)

                    # Botón invisible de selección superpuesto
                    btn_key = f"coref_btn_{rep}"
                    if st.button(
                        "✓" if is_selected else rep,
                        key=btn_key,
                        help=f"{n_mentions} tokens distintos · {n_uces_chain} UCEs",
                    ):
                        if is_selected:
                            st.session_state.coref_selected_entity = None
                        else:
                            st.session_state.coref_selected_entity = rep
                        st.rerun()

                # Detalle de menciones de la entidad seleccionada
                sel_entity = st.session_state.get("coref_selected_entity")
                if sel_entity and sel_entity in chain_index:
                    info = chain_index[sel_entity]
                    mentions = sorted(info["mentions"])
                    color = _entity_color(sel_entity)
                    pills = "".join(
                        f'<span style="display:inline-block;padding:1px 8px;'
                        f"border-radius:10px;font-family:var(--font-mono);font-size:9px;"
                        f"margin:2px;background:{color}18;color:{color};"
                        f'border:0.5px solid {color}44">{sh(m)}</span>'
                        for m in mentions
                    )
                    st.markdown(
                        f'<div style="margin:8px 0 4px;padding:8px 10px;'
                        f"border-radius:6px;border:0.5px solid {color}44;"
                        f'background:{color}0a">'
                        f'<div style="font-family:var(--font-mono);font-size:9px;'
                        f"text-transform:uppercase;letter-spacing:.1em;color:{color};"
                        f'margin-bottom:6px">{sh(sel_entity)} · tokens</div>'
                        f"{pills}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Aparece en {len(info['uce_ids'])} UCEs: "
                        f"{', '.join(info['uce_ids'][:6])}"
                        + (" …" if len(info["uce_ids"]) > 6 else "")
                    )

        # ══════════════════════════════════════════════════════
        # PESTAÑA PREDICADOS — rediseñada
        # ══════════════════════════════════════════════════════
        elif active_tab == "Predicados":
            # 1. Recopilar todos los frames de las UCEs visibles
            # ─────────────────────────────────────────────────
            all_frames: List[Dict] = []
            for u in all_uces_filtered:
                for pf in u.get("predicate_frames", []):
                    frame = dict(pf) if not isinstance(pf, dict) else pf
                    frame["_uce_id"] = u.get("id", "")
                    all_frames.append(frame)

            if not all_frames:
                st.info("No hay marcos predicativos en los documentos seleccionados.")
            else:
                # 2. Índice por lema verbal
                # ─────────────────────────

                lemma_counts = Counter(f.get("verb_lemma", "?") for f in all_frames)
                sorted_lemmas = [lm for lm, _ in lemma_counts.most_common()]

                # Estado de filtros
                if "pred_selected_lemma" not in st.session_state:
                    st.session_state.pred_selected_lemma = (
                        sorted_lemmas[0] if sorted_lemmas else None
                    )
                if "pred_filter_voice" not in st.session_state:
                    st.session_state.pred_filter_voice = "Todas"
                if "pred_filter_role" not in st.session_state:
                    st.session_state.pred_filter_role = "Todos"
                if "pred_filter_negated" not in st.session_state:
                    st.session_state.pred_filter_negated = "Ambos"

                # 3. Selector de lema (chips clicables)
                # ─────────────────────────────────────
                st.markdown(
                    '<p style="font-family:var(--font-mono);font-size:9px;'
                    "text-transform:uppercase;letter-spacing:.1em;"
                    'color:var(--text-low);margin-bottom:6px">Verbo lema</p>',
                    unsafe_allow_html=True,
                )

                # Mostramos hasta 12 lemas; el resto se colapsa
                max_show = 12
                chips_html_parts = []
                for lm in sorted_lemmas[:max_show]:
                    cnt = lemma_counts[lm]
                    is_s = st.session_state.pred_selected_lemma == lm
                    chips_html_parts.append(
                        f'<span style="display:inline-block;padding:2px 10px;'
                        f"border-radius:10px;font-family:var(--font-mono);font-size:10px;"
                        f"margin:2px;cursor:pointer;"
                        f"background:{'var(--accent)' if is_s else 'var(--bg-card)'};"
                        f"color:{'#000' if is_s else 'var(--text-mid)'};"
                        f'border:0.5px solid {"var(--accent)" if is_s else "var(--border2)"}">'
                        f'{sh(lm)} <span style="opacity:.6;font-size:9px">{cnt}</span></span>'
                    )
                st.markdown(
                    '<div style="line-height:2">'
                    + "".join(chips_html_parts)
                    + "</div>",
                    unsafe_allow_html=True,
                )

                sel_lemma = st.selectbox(
                    "Seleccionar lema:",
                    options=sorted_lemmas,
                    index=sorted_lemmas.index(st.session_state.pred_selected_lemma)
                    if st.session_state.pred_selected_lemma in sorted_lemmas
                    else 0,
                    key="pred_lemma_select",
                    label_visibility="collapsed",
                )
                st.session_state.pred_selected_lemma = sel_lemma

                # Frames del lema seleccionado
                lemma_frames = [
                    f for f in all_frames if f.get("verb_lemma") == sel_lemma
                ]

                # 4. Filtros secundarios
                # ─────────────────────
                voices = ["Todas"] + sorted(
                    {f.get("voice", "") for f in lemma_frames if f.get("voice")}
                )
                roles = ["Todos"] + sorted(
                    {
                        f.get("thematic_role", "")
                        for f in lemma_frames
                        if f.get("thematic_role")
                    }
                )

                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    st.session_state.pred_filter_voice = st.selectbox(
                        "Voz",
                        voices,
                        key="pf_voice",
                        index=voices.index(st.session_state.pred_filter_voice)
                        if st.session_state.pred_filter_voice in voices
                        else 0,
                    )
                with fc2:
                    st.session_state.pred_filter_role = st.selectbox(
                        "Rol temático",
                        roles,
                        key="pf_role",
                        index=roles.index(st.session_state.pred_filter_role)
                        if st.session_state.pred_filter_role in roles
                        else 0,
                    )
                with fc3:
                    st.session_state.pred_filter_negated = st.selectbox(
                        "Negado",
                        ["Ambos", "Sí", "No"],
                        key="pf_neg",
                        index=["Ambos", "Sí", "No"].index(
                            st.session_state.pred_filter_negated
                        ),
                    )

                # 5. Aplicar filtros
                # ─────────────────

                filtered_frames = [f for f in lemma_frames if _pred_matches(f)]

                # 6. Estadísticas del lema bajo filtro
                # ─────────────────────────────────────
                if filtered_frames:
                    agent_counts = Counter(
                        f.get("entity_text", f.get("chain_representative", "?"))
                        for f in filtered_frames
                    )
                    obj_counts = Counter(
                        f.get("direct_object", "—")
                        for f in filtered_frames
                        if f.get("direct_object")
                    )
                    tense_counts = Counter(f.get("tense", "?") for f in filtered_frames)
                    neg_count = sum(1 for f in filtered_frames if f.get("negated"))

                    top_agent = (
                        agent_counts.most_common(1)[0][0] if agent_counts else "—"
                    )
                    top_obj = obj_counts.most_common(1)[0][0] if obj_counts else "—"

                    stat_rows = [
                        ("Agente más frecuente", sh(top_agent)),
                        ("OD más frecuente", sh(top_obj)),
                        ("Frames totales", str(len(filtered_frames))),
                        (
                            "Negados",
                            f"{neg_count} ({int(neg_count / len(filtered_frames) * 100)}%)",
                        ),
                        (
                            "Tiempos",
                            " · ".join(
                                f"{t}:{n}" for t, n in tense_counts.most_common(3)
                            ),
                        ),
                    ]
                    rows_html = "".join(
                        f'<div style="display:flex;justify-content:space-between;'
                        f"align-items:center;padding:4px 0;"
                        f'border-bottom:0.5px solid var(--border);font-size:11px">'
                        f'<span style="color:var(--text-low)">{k}</span>'
                        f'<span style="font-family:var(--font-mono);color:var(--text-mid)">'
                        f"{v}</span></div>"
                        for k, v in stat_rows
                    )
                    st.markdown(
                        f'<div style="padding:6px 8px;border-radius:6px;'
                        f"border:0.5px solid var(--border2);margin:8px 0;"
                        f'background:var(--bg-panel)">{rows_html}</div>',
                        unsafe_allow_html=True,
                    )

                    # Sync filtro de subcat para que el corpus resalte
                    current_pred_filter = _get_subcat_filter().get("predicate_lemma")
                    if current_pred_filter != sel_lemma:
                        _set_subcat_filter("predicate_lemma", sel_lemma)
                        st.rerun()
                st.caption(f"{len(filtered_frames)} frames bajo filtro activo")

    # ══════════════════════════════════════════════════════════
    # COLUMNA DERECHA: corpus anotado
    # ══════════════════════════════════════════════════════════
    with col_b:
        with st.container(height=1000, border=False):
            CAT_MAP = {
                "verbos": "Verbos",
                "negaciones": "Negaciones",
                "pronombres": "Pronombres",
                "adverbios": "Adverbios",
                "cuantificadores": "Cuantificadores",
                "marcadores_discursivos": "Marcadores",
                "insubordinaciones": "Insubordinaciones",
                "rarezas": "Rarezas",
                "coref_chains": "Correferencia",
                "discourse_annotations": "Discurso",
            }
            inspect_col = st.selectbox(
                "Inspeccionar categoría",
                list(CAT_MAP.keys()),
                format_func=lambda x: CAT_MAP[x],
                label_visibility="collapsed",
            )

            # Filtros activos + botón de limpieza
            active_filters = _get_subcat_filter()
            if active_filters:
                filt_str = " · ".join(f"{k}={v}" for k, v in active_filters.items())
                c1, c2 = st.columns([8, 2])
                with c1:
                    st.caption(f"Filtros activos: {filt_str}")
                with c2:
                    if st.button("Limpiar filtros", key="clear_all_filters"):
                        _clear_subcat_filters()
                        st.stop()  # Prevent further execution after rerun request

            # ── DEFAULT CORPUS: all filtered UCEs ────────────────────────
            corpus_uces = all_uces_filtered

            # ── Coref: highlight selected entity and restrict UCEs ───────
            sel_coref_entity = st.session_state.get("coref_selected_entity")
            if active_tab == "Coref":
                if sel_coref_entity:
                    st.session_state["highlight_entity"] = sel_coref_entity
                    coref_uce_ids = set(
                        chain_index.get(sel_coref_entity, {}).get("uce_ids", [])
                    )
                    corpus_uces = [
                        u for u in all_uces_filtered if u["id"] in coref_uce_ids
                    ]
                    if not corpus_uces:
                        st.info(
                            f"Ninguna UCE visible contiene la entidad '{sel_coref_entity}'."
                        )
                else:
                    # No entity selected → show all but clear highlight
                    st.session_state.pop("highlight_entity", None)

            # ── Predicados: restrict to UCEs containing selected lemma ───
            elif active_tab == "Predicados":
                sel_lm = st.session_state.get("pred_selected_lemma")
                if sel_lm:
                    lm_uce_ids = {
                        f["_uce_id"]
                        for f in all_frames
                        if f.get("verb_lemma") == sel_lm and _pred_matches(f)
                    }
                    # mini_html = _render_pred_mini_table(lm_uce_ids)
                    # st.markdown(mini_html, unsafe_allow_html=True)

                    corpus_uces = [
                        u for u in all_uces_filtered if u["id"] in lm_uce_ids
                    ]
                # else: keep default (all UCEs)

            # ── For any other tab, ensure highlight is cleared ───────────
            else:
                st.session_state.pop("highlight_entity", None)

            # ── Agrupación por párrafo y renderizado ─────────────────────
            def _par_key(uid: str) -> str:
                parts = uid.split("_")
                return "_".join(parts[:2]) if len(parts) >= 3 else uid

            # (groupby should be imported at top: from itertools import groupby)
            for pk, group in groupby(
                corpus_uces, key=lambda u: _par_key(u.get("id", ""))
            ):
                par_uces = list(group)
                render_paragraph_card(pk, par_uces, inspect_col, active_layers, CAT_MAP)


with tab_b:
    st.markdown(
        '<div style="padding:16px 28px 4px;font-family:var(--font-mono);font-size:9px;'
        'letter-spacing:.12em;color:var(--text-low);text-transform:uppercase">Clases activas:</div>',
        unsafe_allow_html=True,
    )
    cls_cols = st.columns(len(class_list))
    for i, c in enumerate(class_list):
        with cls_cols[i]:
            sel = c in st.session_state.selected_classes
            if st.button(
                f"Clase {c}",
                key=f"ta_cls_{c}",
                width="stretch",
                type="primary" if sel else "secondary",
            ):
                new_list = list(st.session_state.selected_classes)
                if sel:
                    new_list.remove(c)
                else:
                    new_list.append(c)
                if not new_list:
                    new_list = class_list.copy()
                st.session_state.selected_classes = new_list
                st.rerun()

    _sec_rule("Estructura · Persistencia y distribución")
    col_l, col_r = st.columns(2)
    with col_l:
        cl1, cl2 = st.columns(2)
        with cl1:
            st.markdown('<div style="padding:0 0 4px">', unsafe_allow_html=True)
            st.markdown(
                '<p class="panel-hdr">Distribución por clase</p>',
                unsafe_allow_html=True,
            )
            dn = make_donut(tuple(selected_classes), _dm)
            if dn:
                st.plotly_chart(dn, width="stretch", config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        with cl2:
            st.markdown(
                '<p class="panel-hdr">Palabras analizadas</p>', unsafe_allow_html=True
            )
            wb = make_word_bars(tuple(selected_classes), _dm)
            if wb:
                st.plotly_chart(wb, width="stretch", config={"displayModeBar": False})
        for c in sorted(selected_classes):
            pct = class_sizes[c] / classified_uces * 100 if classified_uces else 0
            col = class_colors.get(c, "#AAA")
            st.markdown(
                f"""
            <div style="display:grid;grid-template-columns:60px 1fr 36px 60px;align-items:center;
                        gap:8px;padding:5px 0;border-bottom:1px solid var(--border)">
              <span style="font-family:var(--font-mono);font-size:11px;color:{col};font-weight:500">Clase {c}</span>
              <div style="height:4px;background:var(--bg-hover);border-radius:2px;overflow:hidden">
                <div style="height:100%;width:{pct:.0f}%;background:{col};border-radius:2px"></div>
              </div>
              <span style="font-family:var(--font-mono);font-size:10px;color:{col}">{pct:.0f}%</span>
              <span style="font-family:var(--font-mono);font-size:10px;color:var(--text-low)">{class_sizes[c]} UCE</span>
            </div>""",
                unsafe_allow_html=True,
            )

    with col_r:
        _sec_rule("AFC · Analyse Factorielle des Correspondances")
        _afc_c1, _afc_c2 = st.columns([2, 2])
        with _afc_c1:
            afc_view = st.radio(
                "Vista AFC:",
                options=["coordonnees", "correlations", "contributions"],
                format_func=lambda x: {
                    "coordonnees": "📐 En coordonnées",
                    "correlations": "⭕ En corrélations",
                    "contributions": "📊 En contributions",
                }[x],
                key="afc_view_mode",
                horizontal=True,
            )
        with _afc_c2:
            show_uc_proj = st.checkbox(
                "Proyectar UCs individuales", False, key="afc_show_ucs"
            )
        afc_fig = make_afc_biplot(
            tuple(selected_classes),
            view_mode=afc_view,
            _dm_key=_dm,
            show_ucs=show_uc_proj,
        )
        if afc_fig:
            st.plotly_chart(afc_fig, width="stretch", config={"displayModeBar": True})
        else:
            st.info("Sin datos AFC. Asegúrate de use_projection=True en el pipeline.")
        _AFC_HELP = {
            "coordonnees": "Distancias χ² reales. Puntos alejados del origen = mayor especificidad.",
            "correlations": "Proyección sobre el círculo unitario. Puntos en el borde = bien representados.",
            "contributions": "Peso de cada término en la formación de los ejes factoriales.",
        }
        st.markdown(
            f'<div style="font-family:var(--font-serif);font-size:12px;font-style:italic;'
            f'color:var(--text-low);padding:4px 0 16px;line-height:1.6">'
            f"{_AFC_HELP.get(afc_view, '')}</div>",
            unsafe_allow_html=True,
        )

    bn = st.slider("Términos", 10, 60, 28, 2, key="ta_bn")
    st.markdown(
        '<div style="font-family:var(--font-mono);font-size:9px;color:var(--text-dim);padding:8px 0 0">'
        "Ordenados por spread máx−mín entre clases. Arriba = mayor polarización.</div>",
        unsafe_allow_html=True,
    )
    bs = st.checkbox("Solo significativos", True, key="ta_bs")

    bc1, bc3 = st.columns([1, 1])
    with bc1:
        _sec_rule("Butterfly Chart · φ por término y clase")
        bff = make_butterfly_chart(tuple(selected_classes), bn, bs, _dm_key=_dm)
        if bff:
            st.plotly_chart(bff, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Sin datos para el butterfly chart.")

    with bc3:
        _sec_rule("Graphe des spécificités · φ × CAH")
        gs_fig = make_graphe_specificites(tuple(selected_classes), bn, _dm)
        if gs_fig:
            st.plotly_chart(gs_fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Sin datos para el graphe des spécificités.")

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        _sec_rule("Estabilidad léxica · robustez de los hallazgos")
        stab_fig = make_stability_map(tuple(sorted(class_sizes.keys())), _dm)
        if stab_fig:
            st.plotly_chart(stab_fig, width="stretch", config={"displayModeBar": False})
            st.markdown(
                '<div style="font-family:var(--font-serif);font-size:12px;font-style:italic;'
                'color:var(--text-low);padding:4px 0 16px;line-height:1.6">'
                "Arriba-derecha = hallazgos robustos (alto φ, alta estabilidad bootstrap). "
                "Abajo-derecha = hallazgos frágiles: interpretar con cautela. "
                "Tamaño = frecuencia global.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "Sin datos de estabilidad. Activa use_term_stability=True en el pipeline."
            )

    with cc2:
        _sec_rule("Tensiones lexicales · campo de batalla semántico")
        tl_fig = make_tensions_lexicales(tuple(selected_classes), _dm)
        if tl_fig:
            st.plotly_chart(tl_fig, width="stretch", config={"displayModeBar": False})

    _sec_rule("Comparación inter‑clases · Buscador de corpus")

    _lemma_options = sorted(
        lemma_map.keys(), key=lambda x: lemma_map[x]["total_freq"], reverse=True
    )
    _def_lemmas = _lemma_options[:6] if len(_lemma_options) >= 6 else _lemma_options

    st.markdown(
        '<div style="padding:4px 0 2px;font-family:var(--font-mono);font-size:9px;'
        'letter-spacing:.1em;color:var(--text-low);text-transform:uppercase">'
        "Lemas a explorar · el sistema resuelve raíces y formas exactas automáticamente</div>",
        unsafe_allow_html=True,
    )

    chosen_lemmas = st.multiselect(
        "Lemas:",
        options=_lemma_options,
        default=_def_lemmas,
        key="cross_lemmas",
        label_visibility="collapsed",
    )
    resolved_terms, _stem_label_map = resolve_stems_to_terms(chosen_lemmas)

    if chosen_lemmas:
        expanded_terms = set()
        for lem in chosen_lemmas:
            expanded_terms.add(lem)
            expanded_terms.update(lemma_map.get(lem, {}).get("stems", []))
            expanded_terms.update(lemma_map.get(lem, {}).get("formas", []))
        chips_html = "".join(
            f'<span style="padding:1px 8px;border-radius:10px;background:var(--bg-card);'
            f"color:var(--text-mid);font-family:var(--font-mono);font-size:9px;"
            f'margin:2px 3px;border:.5px solid var(--border2)">{sh(t)}</span>'
            for t in chosen_lemmas
        )
        st.markdown(
            f'<div style="padding:3px 0 8px;font-family:var(--font-mono);font-size:9px;'
            f'color:var(--text-low)">Lemas seleccionados ({len(chosen_lemmas)}): {chips_html}<br>'
            f"Expandido a {len(expanded_terms)} raíces y formas para el motor de búsqueda.</div>",
            unsafe_allow_html=True,
        )

    if len(selected_classes) >= 2:
        cr, cb = st.columns(2)
        with cr:
            st.markdown(
                '<p class="panel-hdr">Radar Σφ por clase</p>', unsafe_allow_html=True
            )
            if len(chosen_lemmas) >= 3:
                rf = make_cross_class_radar_lemmas(
                    chosen_lemmas, tuple(selected_classes), _dm
                )
                if rf:
                    st.plotly_chart(
                        rf, width="stretch", config={"displayModeBar": False}
                    )
            else:
                st.markdown(
                    '<div style="padding:16px;font-family:var(--font-mono);font-size:10px;'
                    'color:var(--text-low)">Necesitas al menos 3 lemas resueltos para el radar.</div>',
                    unsafe_allow_html=True,
                )
        with cb:
            sf = make_term_freq_bar_lemmas(chosen_lemmas, tuple(selected_classes), _dm)
            if sf:
                st.markdown(
                    f'<p class="panel-hdr">Σφ acumulado por clase</p>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(sf, width="stretch", config={"displayModeBar": False})

    cr1, cb1 = st.columns(2)
    with cr1:
        st.markdown(
            '<p class="panel-hdr">Formas exactas por lema · sunburst</p>',
            unsafe_allow_html=True,
        )
        if chosen_lemmas:
            sun = make_lemma_sunburst(
                chosen_lemmas, data.get("forma_index", {}), selected_classes
            )
            if sun:
                st.plotly_chart(sun, width="stretch", config={"displayModeBar": False})
            else:
                st.markdown(
                    '<div style="padding:12px;font-family:var(--font-mono);font-size:10px;'
                    'color:var(--text-low)">Sin variantes para los lemas seleccionados.</div>',
                    unsafe_allow_html=True,
                )

    with cb1:
        st.markdown(
            '<p class="panel-hdr">Red de co‑ocurrencia · lemas resueltos</p>',
            unsafe_allow_html=True,
        )
        min_deg = st.slider(
            "Grado mínimo:", min_value=1, max_value=20, value=1, key="net_min_degree"
        )
        if chosen_lemmas:
            net_fig, net_G = build_lemma_network_cached(
                tuple(chosen_lemmas), 2, 50, min_deg
            )
            if net_fig:
                if net_G:
                    actual_max = max((d for _, d in net_G.degree()), default=0)
                    st.markdown(
                        f'<div style="font-family:var(--font-mono);font-size:9px;color:var(--text-low);'
                        f'margin:-8px 0 6px">{net_G.number_of_nodes()} nodos · '
                        f"{net_G.number_of_edges()} aristas · grado máx. {actual_max}</div>",
                        unsafe_allow_html=True,
                    )
                st.plotly_chart(
                    net_fig, width="stretch", config={"displayModeBar": False}
                )
            else:
                st.info(
                    "Sin red para los lemas resueltos con el grado mínimo seleccionado."
                )
        else:
            st.info("Selecciona lemas para construir la red.")

    _sec_rule("Perfil gramatical · Clase seleccionada")
    render_gram_summary_section(filter_class=None)

    _sec_rule("Buscador de corpus")
    sc1, sc2, sc3 = st.columns([2, 1, 1])
    with sc1:
        cls_opts = ["Todas las clases"] + [
            f"Clase {c}" for c in sorted(selected_classes)
        ]
        fcls = st.selectbox(
            "Clase:", cls_opts, key="ta_fcls", label_visibility="collapsed"
        )
    with sc2:
        logic = st.radio(
            "Lógica:",
            ["OR", "AND"],
            horizontal=True,
            key="ta_logic",
            help="OR: al menos un término · AND: todos los términos",
        )
    with sc3:
        show_all = st.checkbox("Mostrar todas", False, key="ta_showall")

    tcls = None if fcls == "Todas las clases" else int(fcls.split()[-1])
    sr = corpus_search(chosen_lemmas, tcls, logic)
    tcm = build_term_color_map(chosen_lemmas, selected_classes) if chosen_lemmas else {}
    ndsp = len(sr) if show_all else min(20, len(sr))
    mc_counts = Counter(r["match_count"] for r in sr) if chosen_lemmas and sr else {}
    breakdown_html = (
        "".join(
            f'<span style="padding:1px 8px;border-radius:10px;background:var(--bg-card);'
            f'color:var(--text-mid);font-family:var(--font-mono);font-size:9px;margin-left:6px">'
            f"{mc_counts[mc]} UCEs · {mc}/{len(chosen_lemmas)} términos</span>"
            for mc in sorted(mc_counts, reverse=True)
        )
        if mc_counts
        else ""
    )
    lc = "#5BA8DC" if logic == "OR" else "#E8A838"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0 8px;'
        f'font-family:var(--font-mono);font-size:9px">'
        f'<span style="color:var(--text-low)">{len(sr)} UCEs encontradas</span>'
        f'<span style="padding:1px 8px;border-radius:10px;background:{lc}22;'
        f'color:{lc};border:.5px solid {lc}55">{logic}</span>'
        f"{breakdown_html}"
        f'<span style="color:var(--text-dim);margin-left:auto">mostrando {ndsp}</span></div>',
        unsafe_allow_html=True,
    )

    scroll_html = '<div class="corpus-scroll">'
    if not sr:
        scroll_html += (
            '<div style="padding:20px 0;font-family:var(--font-mono);'
            'font-size:10px;color:var(--text-dim)">Sin UCEs que coincidan.</div>'
        )
    else:
        for res in sr[:ndsp]:
            uce = res["uce"]
            c = uce.get("cluster_id")
            cc = (
                class_colors.get(c, T["text_low"])
                if c is not None and c >= 0
                else T["text_low"]
            )
            phi_coefs = uce.get("phi_coefficients", {}) or {}
            phi_html = ""
            if phi_coefs:
                top_phi = sorted(phi_coefs.items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]
                phi_html = (
                    '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-top:6px">'
                    + "".join(
                        f'<span style="padding:1px 7px;border-radius:10px;font-family:var(--font-mono);'
                        f'font-size:9px;background:{cc}18;color:{cc};border:.5px solid {cc}44">'
                        f"{sh(t)} {v:+.2f}</span>"
                        for t, v in top_phi
                    )
                    + "</div>"
                )
            phi = res["phi"]
            mc = res["match_count"]
            matched = res["matched_terms"]
            texto_html = (
                render_highlighted_text(uce.get("texto", ""), res["positions"], tcm)
                if res["positions"]
                else sh(uce.get("texto", ""))
            )
            ctag = f"Clase {c}" if c is not None and c >= 0 else "no clasificada"
            if mc == 0:
                bbg, bfg, btxt = "var(--bg-card)", T["text_low"], "sin coincidencias"
            elif chosen_lemmas and mc == len(chosen_lemmas):
                bbg, bfg, btxt = "#0F2A18", "#5DC88A", f"✓ {mc}/{len(chosen_lemmas)}"
            else:
                bbg, bfg, btxt = (
                    "var(--bg-card)",
                    "#E8A838",
                    f"{mc}/{len(chosen_lemmas)}" if chosen_lemmas else "",
                )
            badge = (
                f'<span class="match-badge" style="background:{bbg};color:{bfg};border:.5px solid {bfg}55">{btxt}</span>'
                if btxt
                else ""
            )
            chips = "".join(
                f'<span style="padding:0 6px;border-radius:8px;font-size:9px;'
                f"background:{tcm.get(mt, T['text_mid'])}22;color:{tcm.get(mt, T['text_mid'])};"
                f'border:.5px solid {tcm.get(mt, T["text_mid"])}44;font-family:var(--font-mono)">{sh(mt)}</span> '
                for mt in matched
            )
            doc_id = uce.get("doc_id", "—")
            # AUG #1: use doc_meta_map for metadata display
            meta_dict = _uce_meta(uce)
            if meta_dict:
                meta_items = [
                    f"<span style='color:var(--text-dim)'>{k}:</span> <span style='color:var(--text-mid)'>{v}</span>"
                    for k, v in meta_dict.items()
                ]
                meta_str = (
                    "<span style='color:var(--border2); margin:0 8px;'> </span>".join(
                        meta_items
                    )
                )
            else:
                meta_str = "<span style='color:var(--text-dim)'>Sin metadatos</span>"
            scroll_html += f"""
            <div class="uce-card" style="border-left:3px solid {cc}; padding:14px; display:flex; flex-direction:column; gap:12px;">
              <div>
                <div class="uce-meta" style="margin-bottom:10px;">
                  <span style="font-family:var(--font-mono); font-size:11px; font-weight:600; color:var(--text-hi);">{sh(uce.get("id", "?")[:12])}</span>
                  <span style="color:{cc}">{ctag} · φ={phi:.2f}</span>
                  {badge}<span style="margin-left:4px">{chips}</span>
                </div>
                <div class="uce-text">{texto_html}</div>{phi_html}              </div>
              <div style="background:var(--bg-panel); border:1px solid var(--border2); border-radius:var(--r-sm); padding:6px 12px; font-family:var(--font-mono); font-size:10px; display:flex; align-items:center; flex-wrap:wrap;">
                <span style="color:var(--text-hi); font-weight:600; margin-right:8px;">Doc: {sh(doc_id)}</span>
                <span style="color:var(--border2); margin-right:8px;"> </span>
                <span>{meta_str}</span>
              </div>
            </div>"""
    scroll_html += "</div>"
    st.html(scroll_html)


# ══════════════════════════════════════════════════════════════
# PESTAÑA C
# ══════════════════════════════════════════════════════════════
with tab_c:
    st.markdown(
        '<div style="padding:16px 28px 4px;font-family:var(--font-mono);font-size:9px;'
        'letter-spacing:.12em;color:var(--text-low);text-transform:uppercase">Clase en análisis:</div>',
        unsafe_allow_html=True,
    )
    b_cols = st.columns(len(class_list))
    for i, c in enumerate(class_list):
        with b_cols[i]:
            is_cur = st.session_state.selected_class_single == c
            if st.button(
                f"Clase {c}",
                key=f"tb_cls_{c}",
                width="stretch",
                type="primary" if is_cur else "secondary",
            ):
                st.session_state.selected_class_single = c
                st.rerun()

    cur_c = st.session_state.selected_class_single
    if st.session_state.get("_last_iso_class") != cur_c:
        st.session_state.iso_view = "hyp"
        st.session_state.iso_idx = 0
        st.session_state["_last_iso_class"] = cur_c
    cur_col = class_colors.get(cur_c, T["accent"]) if cur_c is not None else T["accent"]

    if cur_c is None:
        st.info("Selecciona una clase para comenzar el análisis.")
    else:
        st.markdown(
            f"""
        <div style="display:flex;align-items:center;gap:10px;padding:12px 28px;
                    border-left:3px solid {cur_col};background:var(--bg-panel);margin:8px 0 0">
          <div style="width:8px;height:8px;border-radius:50%;background:{cur_col}"></div>
          <span style="font-family:var(--font-mono);font-size:10px;letter-spacing:.12em;
                       text-transform:uppercase;color:var(--text-hi)">Clase {cur_c}</span>
          <span style="color:var(--border2)">·</span>
          <span style="font-family:var(--font-mono);font-size:10px;color:var(--text-low)">
            {class_sizes.get(cur_c, 0)} UCEs clasificadas
          </span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        _sec_rule("Análisis léxico · Barras φ y explorador de términos")
        lc1, lc2 = st.columns(2)

        with lc1:
            _sec_rule("Dendrograma léxico intra-clase · CAH de términos")
            cah_key = str(cur_c)
            if cah_key in cah_por_clase:
                cah_fig = make_cah_dendrogram(cur_c, _dm)
                if cah_fig:
                    st.plotly_chart(
                        cah_fig, width="stretch", config={"displayModeBar": False}
                    )
                    st.markdown(
                        '<div style="font-family:var(--font-serif);font-size:12px;font-style:italic;'
                        'color:var(--text-low);padding:4px 0 12px;line-height:1.6">'
                        'Color: <span style="color:#5BA8DC">■</span> φ positivo  '
                        '· <span style="color:#E86450">■</span> φ negativo. '
                        "Las ramas más largas separan campos semánticos más distintos.</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info(f"Sin CAH disponible para Clase {cur_c}.")

            st.markdown(
                '<p class="panel-hdr" style="margin-bottom:8px">Explorador de términos · φ vs frecuencia '
                '<span style="color:var(--text-low);float:right;font-size:9px">burbuja = frec. en clase</span></p>',
                unsafe_allow_html=True,
            )
            ef = make_term_explorer(cur_c, _dm)
            if ef:
                st.plotly_chart(ef, width="stretch", config={"displayModeBar": False})
            else:
                st.info("Sin datos.")

        with lc2:
            st.markdown(
                '<p class="panel-hdr">Presencias y ausencias significativas</p>',
                unsafe_allow_html=True,
            )
            st.html(phi_bars_html(cur_c, _dm))

        _sec_rule("Buscador de Contexto y KWIC")

        iframe_css = f"""
        <style>
        :root {{
          --bg-page:    {T["bg_page"]};  --bg-panel:   {T["bg_panel"]};
          --bg-card:    {T["bg_card"]};  --border:     {T["border"]};
          --border2:    {T["border2"]};  --text-hi:    {T["text_hi"]};
          --text-mid:   {T["text_mid"]}; --text-low:   {T["text_low"]};
          --text-dim:   {T["text_dim"]}; --font-mono: 'IBM Plex Mono', monospace;
          --font-serif: 'Newsreader', serif;
        }}
        body {{ background-color: transparent; color: var(--text-hi); margin: 0; font-family: var(--font-serif); }}
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 3px; }}
        .clickable-uce {{ cursor: pointer; transition: transform 0.1s, border-color 0.1s; }}
        .clickable-uce:hover {{ transform: translateX(2px); border-color: {cur_col} !important; }}
        </style>
        <script>
        function jumpToDoc(uid) {{
            const btns = window.parent.document.querySelectorAll('button');
            for(let b of btns) {{
                if(b.innerText.includes('JMP_' + uid)) {{ b.click(); break; }}
            }}
        }}
        </script>
        """

        if st.session_state.tab_b_view == "search":
            _sec_rule("Buscador de Contextos y Lectura de Tarjetas")

            if not terminos_df.empty:
                class_stems = terminos_df[
                    (terminos_df["cluster"] == cur_c) & (terminos_df["phi"] > 0)
                ]["termino"].tolist()
                class_lemmas = sorted(
                    [
                        lem
                        for lem, d in lemma_map.items()
                        if any(s in class_stems for s in d.get("stems", []))
                    ]
                )
            else:
                class_lemmas = []

            class_expanded_terms = set()
            for lem in class_lemmas:
                class_expanded_terms.add(lem)
                class_expanded_terms.update(lemma_map.get(lem, {}).get("stems", []))
                class_expanded_terms.update(lemma_map.get(lem, {}).get("formas", []))
            class_expanded_terms = list(class_expanded_terms)

            uc_lemmas_selected = st.multiselect(
                "Filtrar por Lemas (expande a raíces y formas exactas):",
                options=class_lemmas,
                key=f"uc_lemmas_{cur_c}",
            )

            expanded_search_terms = set()
            for lem in uc_lemmas_selected:
                expanded_search_terms.add(lem)
                expanded_search_terms.update(lemma_map.get(lem, {}).get("stems", []))
                expanded_search_terms.update(lemma_map.get(lem, {}).get("formas", []))

            uc_search = st.text_input(
                "Buscar texto libre en contextos:", key=f"uc_search_{cur_c}"
            )

            if expanded_search_terms or uc_search:
                st.html(
                    f'<div style="margin-top:10px;">{unified_kwic_search(uces, list(expanded_search_terms), cur_c, cur_col)}</div>'
                )

            matching_ucs = []

            # Group UCEs by their document and section
            def _get_section_key(uid: str) -> str:
                parts = uid.split("_")
                return "_".join(parts[:2]) if len(parts) >= 3 else uid

            grouped_uces = defaultdict(list)
            for u in uces:
                grouped_uces[_get_section_key(u.get("id", ""))].append(u)

            for sec_key, uc_uces in grouped_uces.items():
                if not uc_uces or not any(
                    u.get("cluster_id") == cur_c for u in uc_uces
                ):
                    continue

                uc_full_text = " ".join(u.get("texto", "") for u in uc_uces)

                if uc_search and uc_search.lower() not in uc_full_text.lower():
                    continue
                if expanded_search_terms and not any(
                    match_term(l, uc_full_text) for l in expanded_search_terms
                ):
                    continue

                max_phi = max(
                    [
                        uce_phi_dict.get(u["id"], 0.0)
                        for u in uc_uces
                        if u.get("cluster_id") == cur_c
                    ],
                    default=0.0,
                )

                _m = _uce_meta(uc_uces[0])
                doc_id = str(_m.get("origen", uc_uces[0].get("doc_id", "—")))
                try:
                    sort_val = int("".join(filter(str.isdigit, doc_id)))
                except ValueError:
                    sort_val = 999999

                # We pass None for the old 'uc' dict since it no longer exists
                matching_ucs.append((sort_val, doc_id, max_phi, None, uc_uces))

            matching_ucs.sort(key=lambda x: x[0])

            if not matching_ucs:
                st.info(f"No hay contextos que coincidan para la Clase {cur_c}.")
            else:
                cards_html = iframe_css + '<div style="padding:4px;">'
                for i, (_, doc_id, max_phi, uc, uc_uces) in enumerate(matching_ucs):
                    cards_html += f"""
                    <div style="border-left:3px solid {cur_col}; margin-bottom:14px; background:var(--bg-panel); border-radius: 4px; border: 1px solid var(--border2);">
                        <div style="padding:10px 16px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between;">
                            <span style="font-family:var(--font-mono); font-size:12px; font-weight:500; color:var(--text-hi);">Doc {sh(doc_id)} (UC {i + 1}/{len(matching_ucs)})</span>
                            <span style="font-family:var(--font-mono); font-size:10px; color:{cur_col};">Max φ = {max_phi:.2f}</span>
                        </div>
                        <div style="padding: 8px;">
                    """
                    for u in uc_uces:
                        c = u.get("cluster_id")
                        phi = uce_phi_dict.get(u["id"], 0.0)
                        raw_text = u.get("texto", "")
                        if c == cur_c:
                            hl_terms = (
                                list(expanded_search_terms)
                                if (expanded_search_terms or uc_search)
                                else class_expanded_terms
                            )
                            display_text = _render_tab_b_highlight(
                                raw_text, [], hl_terms, uc_search, cur_col
                            )
                            content = f'<span style="color:var(--text-hi); line-height:1.65;">{display_text} <span style="font-size:9px; font-weight:bold; color:{cur_col};">φ={phi:.2f}</span></span>'
                            b_left = f"2px solid {cur_col}"
                        elif c is not None and c >= 0:
                            content = f'<span style="color:var(--text-mid);">{sh(raw_text)}</span>'
                            b_left = "2px solid transparent"
                        else:
                            content = f'<span style="color:var(--text-dim);">{sh(raw_text)}</span>'
                            b_left = "2px solid transparent"
                        cards_html += f"""
                        <div class="clickable-uce" onclick="jumpToDoc('{u["id"]}')" style="background:var(--bg-card); padding: 10px 16px; border-radius: 4px; margin-bottom: 4px; border: 1px solid var(--border2); border-left: {b_left};">
                            <div style="font-family:var(--font-mono); font-size:9px; color:var(--text-dim); margin-bottom:4px;"> </div>
                            <div style="font-size:13.5px;">{content}</div>
                        </div>"""
                    cards_html += "</div></div>"
                cards_html += "</div>"
                st.iframe(cards_html, height=400, scrolling=True)

                with st.sidebar:
                    if st.button("🔄 Recargar datos discursivos"):
                        st.cache_resource.clear()
                        st.rerun()
                    for uc_idx, (_, doc_id, _, _, uc_uces) in enumerate(matching_ucs):
                        for u in uc_uces:
                            if st.button(
                                f"JMP_{u['id']}", key=f"btn_{uc_idx}{u['id']}"
                            ):
                                st.session_state.tab_b_view = "document"
                                st.session_state.selected_doc_id = doc_id
                                st.session_state.target_uce_id = u["id"]
                                st.session_state.doc_search_query = uc_search
                                st.session_state.doc_filter_lemmas = list(
                                    expanded_search_terms
                                )
                                st.rerun()

        elif st.session_state.tab_b_view == "document":
            # .---------------------------

            target_doc = st.session_state.selected_doc_id
            st.session_state.selected_doc_id = doc_id
            _sec_rule(f"Visor de Documento Completo: {target_doc}")

            if st.button("← Volver a los resultados de búsqueda", type="primary"):
                st.session_state.tab_b_view = "search"
                st.rerun()

            uce_to_uc_id = {}
            uc_cluster_map = {}
            for uc in ucs:
                uc_cluster_map[uc.get("id")] = uc.get("cluster_label_double")
                for uid in uc.get("uce_ids", []):
                    base_uid = re.sub(r"_mf\d+$", "", uid)  # remove threshold suffix
                    uce_to_uc_id[base_uid] = uc_cluster_map[uc.get("id")]

            # After building doc_uce_ids (now includes all UCEs)
            def _get_section_key(uid: str) -> str:
                parts = uid.split("_")
                return "_".join(parts[:2]) if len(parts) >= 3 else uid

            uc_to_uces = {}
            uc_order = []
            doc_uce_ids = origen_index.get(target_doc, [])

            for uid in doc_uce_ids:
                base_uid = re.sub(r"__mf\d+$", "", uid)
                group_key = _get_section_key(base_uid)
                if group_key not in uc_order:
                    uc_order.append(group_key)
                uc_to_uces.setdefault(group_key, []).append(base_uid)

            uce_lookup = {u["id"]: u for u in uces}

            # Determine highlight terms
            sq = st.session_state.get("doc_search_query", "")
            fl = st.session_state.get("doc_filter_lemmas", [])
            if not fl and not sq:
                class_stems = (
                    terminos_df[
                        (terminos_df["cluster"] == cur_c) & (terminos_df["phi"] > 0)
                    ]["termino"].tolist()
                    if not terminos_df.empty
                    else []
                )
                class_lemmas_doc = sorted(
                    [
                        lem
                        for lem, d in lemma_map.items()
                        if any(s in class_stems for s in d.get("stems", []))
                    ]
                )
                hl_terms_doc = set()
                for lem in class_lemmas_doc:
                    hl_terms_doc.add(lem)
                    hl_terms_doc.update(lemma_map.get(lem, {}).get("stems", []))
                    hl_terms_doc.update(lemma_map.get(lem, {}).get("formas", []))
                hl_terms_doc = list(hl_terms_doc)
            else:
                hl_terms_doc = fl

            # ------------------------------------------------------------------
            # 2. Left column: Bar chart (Sequence)
            # ------------------------------------------------------------------
            doc_c1, doc_c2 = st.columns([1, 3])

            with doc_c1:
                st.markdown(
                    '<p class="panel-hdr">Secuencia Discursiva</p>',
                    unsafe_allow_html=True,
                )

                # Build flat list in document order
                flat_colors = []
                flat_hover = []

                for idx, uid in enumerate(doc_uce_ids):
                    u = uce_lookup.get(uid, {})
                    cid = u.get("cluster_id")
                    phi = uce_phi_dict.get(uid, 0.0)

                    if cid is not None and cid >= 0:
                        flat_colors.append(class_colors.get(cid, T["accent"]))
                        flat_hover.append(
                            f"<b>Clase {cid}</b><br>"
                            f"UCE {idx + 1}/{len(doc_uce_ids)} · φ = {phi:.2f}"
                        )
                    else:
                        flat_colors.append(T["text_low"])
                        flat_hover.append(
                            f"<b>No clasificado</b><br>UCE {idx + 1}/{len(doc_uce_ids)}"
                        )

                # Plot bar chart: each UCE gets a horizontal bar of height 1
                fig_traj = go.Figure(
                    go.Bar(
                        y=list(range(len(doc_uce_ids))),  # 0,1,2,...
                        x=[1] * len(doc_uce_ids),
                        orientation="h",
                        width=1,
                        marker=dict(color=flat_colors, line=dict(width=0)),
                        text=flat_hover,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

                fig_traj.update_layout(
                    height=600,
                    bargap=0,
                    bargroupgap=0,
                    margin=dict(l=40, r=8, t=30, b=40),
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(
                        title=None,
                        autorange="reversed",  # first UCE at top
                        showticklabels=False,  # hide numbers
                        showgrid=False,
                        zeroline=False,
                    ),
                    title=dict(
                        text=f"Secuencia Discursiva · Doc {target_doc}",
                        font=dict(size=10, color=T["text_low"]),
                    ),
                    hoverlabel=dict(
                        bgcolor=T["bg_card"],
                        bordercolor=T["border2"],
                        font=dict(size=11, color=T["text_hi"]),
                    ),
                    **_plot_defaults(),
                )

                st.plotly_chart(
                    fig_traj, width="content", config={"displayModeBar": False}
                )
            # ------------------------------------------------------------------
            # 3. Right column: Text reading (same HTML as before)
            # ------------------------------------------------------------------
            with doc_c2:
                st.markdown(
                    '<p class="panel-hdr">Lectura Continua de UCEs</p>',
                    unsafe_allow_html=True,
                )

                doc_html = '<div style="height: 600px; overflow-y: auto; padding-right: 10px; padding-bottom: 200px;">'

                # Iterate over UCE IDs in document order (preserved by origen_index)
                for uid in doc_uce_ids:
                    u = uce_lookup.get(uid, {})
                    if not u:
                        continue

                    cid = u.get("cluster_id")
                    phi = uce_phi_dict.get(uid, 0.0)
                    txt = u.get("texto", "")

                    # Choose border and text color based on cluster (or neutral for unclassified)
                    if cid is not None and cid >= 0:
                        border_color = class_colors.get(cid, T["border2"])
                        text_color = T["text_hi"] if cid == cur_c else T["text_mid"]
                    else:
                        border_color = T["border2"]
                        text_color = T["text_mid"]

                    # Apply highlighting if search terms are active
                    if hl_terms_doc or sq:
                        display_text = _render_tab_b_highlight(
                            txt, [], hl_terms_doc, sq, border_color
                        )
                    else:
                        display_text = sh(txt)

                    doc_html += f"""
                        <div style="margin-bottom: 12px; border-left: 4px solid {border_color}; background: {T["bg_card"]}; border-radius: 4px; padding: 8px 12px;">
                            <div>
                                <span style="font-family: var(--font-serif); font-size: 13px; color: {text_color}; line-height: 1.5;">{display_text}</span>
                                <span style="font-family: var(--font-mono); font-size: 9px; color: {T["text_dim"]}; margin-left: 8px;">φ={phi:.2f}</span>
                            </div>
                        </div>
                        """

                doc_html += "</div>"

                # Scroll to target UCE if needed
                if st.session_state.target_uce_id:
                    doc_html += f'''
                        <script>
                            setTimeout(function() {{
                                var target = document.getElementById("{st.session_state.target_uce_id}");
                                if (target) {{
                                    var container = target.closest("div[style*='overflow-y']");
                                    if (container) {{
                                        var targetTop = target.offsetTop;
                                        var targetHeight = target.offsetHeight;
                                        var containerHeight = container.clientHeight;
                                        container.scrollTop = targetTop - (containerHeight / 2) + (targetHeight / 2);
                                    }} else {{
                                        target.scrollIntoView({{behavior: "smooth", block: "center"}});
                                    }}
                                }}
                            }}, 300);
                        </script>
                        '''
                st.iframe(doc_html, height=620)

        _sec_rule("Síntesis discursiva")
        if str(cur_c) in sintesis_por_clase:
            sy = sintesis_por_clase[str(cur_c)]
            sv = sy.get("validacion", {}).get("similitud", None)
            sok = sy.get("validacion", {}).get("valido", True)
            scol = "#5DC88A" if sok else "#E8A838"
            raw = sy["sintesis"].replace("\n", "<br>")
            raw = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", raw)
            sbdg = (
                f'<span style="background:{scol}22;color:{scol};padding:2px 8px;border-radius:12px;'
                f'font-family:var(--font-mono);font-size:9px">{"✓" if sok else "△"} similitud {sv:.3f}</span>'
                if sv is not None
                else ""
            )
            st.markdown(
                f"""
            <div class="synth-card" style="border-color:{cur_col}44">
              <div class="synth-header" style="background:linear-gradient(90deg,{cur_col}14 0%,transparent 100%)">
                <div class="synth-dot" style="background:{cur_col}"></div>
                <span class="synth-class-label">Clase {cur_c} · síntesis discursiva</span>
              </div>
              <div class="synth-body">{raw}</div>
              <div class="synth-footer">
                <span>Generado {sy.get("timestamp", "")[:19].replace("T", " ")}</span>
                {sbdg}
              </div>
            </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("Sin síntesis disponible para esta clase.")

        _sec_rule("Perfil gramatical · Clase seleccionada")
        render_gram_summary_section(filter_class=cur_c)

        _sec_rule("Perfil de modalización · agencia enunciativa")
        mod_fig = make_modalization_radar(cur_c, _dm)
        if mod_fig:
            st.plotly_chart(mod_fig, width="stretch", config={"displayModeBar": False})
            st.markdown(
                '<div style="font-family:var(--font-serif);font-size:12px;font-style:italic;'
                'color:var(--text-low);padding:4px 0 12px;line-height:1.6">'
                "Tasas normalizadas por 1000 palabras. La clase activa está resaltada. "
                "Alta deóntica = discurso normativo. Alta polifonía = discurso referido.</div>",
                unsafe_allow_html=True,
            )

        _sec_rule("Análisis isotópico")
        if "iso_view" not in st.session_state:
            st.session_state.iso_view = "hyp"
        if "iso_idx" not in st.session_state:
            st.session_state.iso_idx = 0

        cur_iso_cd = next((d for d in iso_classes if d["id"] == cur_c), None)
        n_isos = len(cur_iso_cd.get("iso", [])) if cur_iso_cd else 0

        sub_nav = (
            [("hyp", "Hipótesis")]
            + [(f"iso_{i}", f"Isotopía {i + 1}") for i in range(n_isos)]
            + [
                ("abs", "Ausencias"),
                ("met", "Metáforas"),
                ("anc", "Anclaje"),
                ("lbl", "Etiquetas"),
                ("val", "Validación"),
                ("global", "Oposiciones AFC"),
            ]
        )

        if sub_nav:
            sn_cols = st.columns(len(sub_nav), gap="small")
            for i, (vk, vl) in enumerate(sub_nav):
                with sn_cols[i]:
                    if st.button(
                        vl,
                        key=f"tb_iso_{vk}",
                        width="stretch",
                        type="primary"
                        if st.session_state.iso_view == vk
                        else "secondary",
                    ):
                        st.session_state.iso_view = vk
                        if vk.startswith("iso_"):
                            st.session_state.iso_idx = int(vk.split("_")[1])
                        st.rerun()

        iso_view = st.session_state.iso_view

        if iso_view == "global":
            gd = global_data
            oc1, oc2 = st.columns(2, gap="medium")
            for ocol, okey, olbl in [
                (oc1, "opp1", "Oposición principal · Factor 1"),
                (oc2, "opp2", "Oposición secundaria · Factor 2"),
            ]:
                opp = gd.get(okey, {})
                if not opp:
                    continue
                with ocol:
                    st.markdown(
                        f'<div style="border:.5px solid var(--border2);border-radius:var(--r-sm);overflow:hidden;box-shadow:var(--card-shadow)">'
                        f'<div style="padding:6px 12px;background:var(--bg-card);font-family:var(--font-mono);font-size:9px;'
                        f'letter-spacing:.1em;text-transform:uppercase;color:var(--text-low);border-bottom:.5px solid var(--border)">{olbl}</div>'
                        f'<div style="padding:10px 12px;font-family:var(--font-mono);font-size:12px;font-weight:500;color:var(--text-hi);line-height:1.4;border-bottom:.5px solid var(--border)">{sh(opp.get("formulacion", ""))}</div>'
                        f'<div style="padding:10px 12px;font-size:12px;color:var(--text-mid);font-style:italic;line-height:1.6">{_md_inline(opp.get("justificacion", ""))}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            if gd.get("hyp"):
                _sec_label("Hipótesis global", top=True)
                _hyp_block(gd["hyp"], "#7C6BF8")

        elif cur_iso_cd is None:
            st.info("Datos isotópicos no disponibles para esta clase.")
        else:
            _imap = _build_imap(cur_iso_cd)
            _c_id = cur_iso_cd["id"]

            if iso_view == "hyp":
                hyp = cur_iso_cd.get("hyp", "")
                if hyp:
                    _hyp_block(hyp, cur_col)
                else:
                    st.info("Hipótesis no disponible.")
                _sec_label("Isotopías activas", top=True)
                chips_h = "".join(
                    f'<span style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;'
                    f"border-radius:12px;border:.5px solid {cur_col}55;font-family:var(--font-mono);"
                    f'font-size:11px;color:var(--text-hi);margin:3px">{sh(iso["name"])}'
                    f'<span style="color:var(--text-low)">· {len(iso["uces"])} UCE{"s" if len(iso["uces"]) != 1 else ""}</span></span>'
                    for iso in cur_iso_cd.get("iso", [])
                )
                st.markdown(
                    f'<div style="padding:4px 0 8px">{chips_h}</div>',
                    unsafe_allow_html=True,
                )
                if cur_iso_cd.get("abs_terms"):
                    _sec_label("Ausencias significativas")
                    st.markdown(
                        f'<div style="padding:2px 0 10px">{_chips(cur_iso_cd["abs_terms"], cur_col, neg=True)}</div>',
                        unsafe_allow_html=True,
                    )

            elif iso_view.startswith("iso_"):
                idx = st.session_state.iso_idx
                isos = cur_iso_cd.get("iso", [])
                if idx >= len(isos):
                    st.info("Isotopía no encontrada.")
                else:
                    iso = isos[idx]
                    _sec_label("Términos constitutivos", top=True)
                    st.markdown(
                        f'<div style="padding:2px 0 8px">{_chips(iso.get("terms", []), cur_col)}</div>',
                        unsafe_allow_html=True,
                    )
                    if iso.get("fn"):
                        st.markdown(
                            f'<div style="font-size:12px;color:var(--text-mid);font-style:italic;line-height:1.55;padding-bottom:12px">{sh(iso["fn"])}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        unified_kwic_search(uces, iso.get("terms", []), cur_c, cur_col),
                        unsafe_allow_html=True,
                    )

            elif iso_view == "abs":
                at = cur_iso_cd.get("abs_terms", [])
                ad = cur_iso_cd.get("abs_desc", "")
                if at:
                    st.markdown(
                        f'<div style="border:.5px solid #E8645044;border-radius:var(--r-sm);overflow:hidden;margin-bottom:14px">'
                        f'<div style="padding:7px 12px;background:#2C1014;font-family:var(--font-mono);font-size:11px;color:#E86450">Términos con φ negativo significativo</div>'
                        f'<div style="padding:10px 12px">{_chips(at, cur_col, neg=True)}'
                        f'<div style="font-size:12px;color:var(--text-mid);font-style:italic;line-height:1.6;margin-top:10px;padding-top:8px;border-top:.5px solid var(--border2)">{sh(ad)}</div>'
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Sin términos negativos.")

            elif iso_view == "met":
                mets = cur_iso_cd.get("met", [])
                if not mets:
                    st.info("Sin metáforas disponibles.")
                else:
                    for mi, m in enumerate(mets):
                        _sec_label(f"Metáfora {mi + 1}", top=(mi == 0))
                        st.markdown(
                            f'<div style="font-family:var(--font-mono);font-size:12px;font-weight:500;color:var(--text-hi);padding-bottom:4px">{sh(m.get("formula", ""))}</div>'
                            f'<div style="font-size:12px;color:var(--text-mid);font-style:italic;padding-bottom:10px;line-height:1.55">{sh(m.get("desc", ""))}</div>',
                            unsafe_allow_html=True,
                        )
                        e2, k2 = st.tabs([f"Evidencia {mi + 1}", f"KWIC {mi + 1}"])
                        with e2:
                            st.markdown(
                                _uce_cards_html(m.get("uces", []), _imap, cur_col),
                                unsafe_allow_html=True,
                            )
                        with k2:
                            st.markdown(
                                unified_kwic_search(
                                    uces, m.get("terms", []) or [], cur_c, cur_col
                                ),
                                unsafe_allow_html=True,
                            )

            elif iso_view == "anc":
                obj = cur_iso_cd.get("obj", {})
                anc = cur_iso_cd.get("anc", {})
                ao1, ao2 = st.columns(2, gap="medium")
                for ocol_a, lbl_a, dat_a, fields_a in [
                    (
                        ao1,
                        "Objetivación",
                        obj,
                        [
                            ("Núcleo figurativo", "nucleo_figurativo"),
                            ("Términos de naturalización", "terminos_naturalizacion"),
                        ],
                    ),
                    (
                        ao2,
                        "Anclaje",
                        anc,
                        [
                            ("Categorías previas", "sistema_categorias_previas"),
                            ("Grupo social", "grupo_social_asociado"),
                        ],
                    ),
                ]:
                    with ocol_a:
                        st.markdown(
                            f'<div style="border:.5px solid var(--border2);border-radius:var(--r-sm);overflow:hidden;box-shadow:var(--card-shadow)">'
                            f'<div style="padding:6px 10px;background:var(--bg-card);font-family:var(--font-mono);font-size:9px;letter-spacing:.09em;'
                            f'text-transform:uppercase;color:var(--text-low);border-bottom:.5px solid var(--border)">{lbl_a}</div>'
                            f'<div style="padding:12px">',
                            unsafe_allow_html=True,
                        )
                        for fl, fk in fields_a:
                            val = dat_a.get(fk, dat_a.get(fk.split("_")[0], ""))
                            if val:
                                if isinstance(val, list):
                                    st.markdown(
                                        f'<div style="font-family:var(--font-mono);font-size:9px;text-transform:uppercase;color:var(--text-low);margin-bottom:4px">{fl}</div>{_chips(val, cur_col)}',
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(
                                        f'<div style="font-family:var(--font-mono);font-size:9px;text-transform:uppercase;color:var(--text-low);margin-bottom:3px">{fl}</div>'
                                        f'<div style="font-size:12px;color:var(--text-mid);font-style:italic;line-height:1.5;margin-bottom:10px">{sh(val)}</div>',
                                        unsafe_allow_html=True,
                                    )
                        st.markdown("</div></div>", unsafe_allow_html=True)

            elif iso_view == "lbl":
                lbls = cur_iso_cd.get("lbls", [])
                lkey = f"tb_lbl_{_c_id}"
                if lkey not in st.session_state:
                    st.session_state[lkey] = None
                _sec_label("Propuestas de etiqueta", top=True)
                if not lbls:
                    st.info("Sin propuestas.")
                else:
                    for li, lbl in enumerate(lbls):
                        nm = lbl.get("nombre_propuesto", lbl.get("nombre", ""))
                        tp = lbl.get("tipo_enfasis", "")
                        jt = lbl.get("justificacion", "")
                        is_ch = st.session_state.get(lkey) == li
                        brd = (
                            f"1.5px solid {cur_col}"
                            if is_ch
                            else ".5px solid var(--border2)"
                        )
                        chk = (
                            f'<span style="color:{cur_col};margin-left:auto">✓ seleccionada</span>'
                            if is_ch
                            else ""
                        )
                        st.markdown(
                            f'<div style="border:{brd};border-radius:var(--r-sm);padding:12px 14px;margin-bottom:8px;box-shadow:var(--card-shadow)">'
                            f'<div style="display:flex;align-items:center;margin-bottom:6px">'
                            f'<span style="padding:2px 8px;border-radius:12px;border:.5px solid var(--border2);color:var(--text-mid);font-family:var(--font-mono);font-size:11px">{sh(tp)}</span>{chk}</div>'
                            f'<div style="font-size:13px;font-weight:500;color:var(--text-hi);margin-bottom:5px">{sh(nm)}</div>'
                            f'<div style="font-size:12px;color:var(--text-mid);line-height:1.5">{sh(jt)}</div></div>',
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            "✓ Seleccionada" if is_ch else "Validar esta etiqueta",
                            key=f"tb_lbl_{_c_id}_{li}",
                            type="primary" if is_ch else "secondary",
                        ):
                            st.session_state[lkey] = li if not is_ch else None
                            st.rerun()
                st.text_input(
                    "O escribe tu propia etiqueta:",
                    placeholder="ej: discurso de la urgencia…",
                    key=f"tb_lcustom_{_c_id}",
                )
                if st.session_state.get(f"tb_lcustom_{_c_id}", ""):
                    st.success(
                        f"✓ Etiqueta registrada: **{st.session_state[f'tb_lcustom_{_c_id}']}**"
                    )

            elif iso_view == "val":
                tens = cur_iso_cd.get("tensions", [])
                lims = cur_iso_cd.get("limits", [])
                vt, vl = st.columns(2, gap="medium")
                with vt:
                    _sec_label("Tensiones interpretativas", top=True)
                    if not tens:
                        st.markdown(
                            '<div style="font-size:12px;color:var(--text-low);font-style:italic">Sin tensiones registradas.</div>',
                            unsafe_allow_html=True,
                        )
                    for ti, t in enumerate(tens):
                        tk = f"tb_tens_{_c_id}_{ti}"
                        done = st.session_state.get(tk, False)
                        st.markdown(
                            f'<div style="display:flex;gap:8px;padding:8px 10px;border:.5px solid {"var(--border)" if done else "#E8A83844"};'
                            f'border-radius:var(--r-sm);margin-bottom:6px;background:{"transparent" if done else "#1C1400"};opacity:{"0.4" if done else "1"}">'
                            f'<span style="font-family:var(--font-mono);font-size:11px;color:{"var(--text-low)" if done else "#E8A838"};flex-shrink:0">▲</span>'
                            f'<span style="font-size:12px;color:var(--text-mid);line-height:1.5">{sh(t)}</span></div>',
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            "Desmarcar" if done else "Marcar como atendida",
                            key=f"tb_tbtn_{_c_id}_{ti}",
                            type="secondary",
                            width="stretch",
                        ):
                            st.session_state[tk] = not done
                            st.rerun()
                with vl:
                    _sec_label("Límites de interpretación", top=True)
                    if not lims:
                        st.markdown(
                            '<div style="font-size:12px;color:var(--text-low);font-style:italic">Sin límites registrados.</div>',
                            unsafe_allow_html=True,
                        )
                    for li2, lim in enumerate(lims):
                        lk = f"tb_lim_{_c_id}_{li2}"
                        done2 = st.session_state.get(lk, False)
                        brd2 = "#5DC88A44" if done2 else "var(--border2)"
                        bg2 = "#0F2A18" if done2 else "transparent"
                        tc2 = "var(--text-low)" if done2 else "var(--text-mid)"
                        st.markdown(
                            f'<div style="display:flex;gap:8px;padding:7px 0;border-bottom:.5px solid var(--border);align-items:flex-start">'
                            f'<div style="width:13px;height:13px;border-radius:3px;border:.5px solid {brd2};background:{bg2};flex-shrink:0;margin-top:2px;'
                            f'display:flex;align-items:center;justify-content:center;font-size:9px;color:#5DC88A">{"✓" if done2 else ""}</div>'
                            f'<span style="font-size:12px;color:{tc2};line-height:1.5;{"text-decoration:line-through;" if done2 else ""}">{sh(lim)}</span></div>',
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            "Desmarcar" if done2 else "Marcar como verificado",
                            key=f"tb_lbtn_{_c_id}_{li2}",
                            type="secondary",
                            width="stretch",
                        ):
                            st.session_state[lk] = not done2
                            st.rerun()


# ══════════════════════════════════════════════════════════════
# PESTAÑA D
# ══════════════════════════════════════════════════════════════
with tab_d:
    # Primera fila: árbol de persistencia y scree plot
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown(
            '<p class="panel-hdr">Árbol de decisión CDH</p>', unsafe_allow_html=True
        )
        cf = make_cdh_dendrogram(_dm, root_at_top=True)
        if cf:
            st.plotly_chart(cf, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Sin datos de árbol CDH.")
    with ec2:
        st.markdown(
            '<p class="panel-hdr">Scree plot · valores propios (AFC)</p>',
            unsafe_allow_html=True,
        )
        scree_fig = make_scree_plot(_dm)
        if scree_fig:
            st.plotly_chart(
                scree_fig, width="stretch", config={"displayModeBar": False}
            )
        else:
            st.info("Sin datos AFC. Activa use_projection=True en el pipeline.")

    # Segunda fila: residuos de metadatos y MCA
    ec3, ec4 = st.columns(2)
    with ec3:
        _sec_rule("Variables ilustrativas · asociación con clases")
        _show_res = st.checkbox(
            "Ver residuos estandarizados", False, key="tc_meta_residuals"
        )
        st.markdown(
            metadata_quality_html(
                tuple(sorted(class_sizes.keys())), _dm, show_residuals=_show_res
            ),
            unsafe_allow_html=True,
        )
    with ec4:
        _sec_rule("CAH Global · estructura léxica AFC")
        gcah_fig = make_global_cah_dendrogram(_dm)
        if gcah_fig:
            st.plotly_chart(gcah_fig, width="stretch", config={"displayModeBar": False})
            st.markdown(
                '<div style="font-family:var(--font-serif);font-size:12px;font-style:italic;'
                'color:var(--text-low);padding:4px 0 16px;line-height:1.6">'
                "Agrupamiento jerárquico de todos los términos del vocabulario AFC. "
                "Color = clase con mayor |φ| para ese término.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Sin datos CAH global. Activa use_projection=True en el pipeline.")
    # Transiciones discursivas
    _sec_rule("Transiciones discursivas · gramática del documento")
    tm_fig = make_transition_matrix(_dm)
    if tm_fig:
        st.plotly_chart(tm_fig, width="stretch", config={"displayModeBar": False})
        st.markdown(
            '<div style="font-family:var(--font-serif);font-size:12px;font-style:italic;'
            'color:var(--text-low);padding:4px 0 16px;line-height:1.6">'
            "Los loops en la diagonal = un tipo de discurso tiende a seguirse a sí mismo. "
            "Los puentes fuera de la diagonal = transiciones frecuentes entre mundos discursivos.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Datos insuficientes para la matriz de transición.")

    # Categorías gramaticales
    _sec_rule("Categorías gramaticales · Porcentaje por clase")
    st.markdown(
        gramcat_html(tuple(sorted(class_sizes.keys())), _dm), unsafe_allow_html=True
    )

    # Random Forest y SHAP
    rf1, rf2 = st.columns(2)
    with rf1:
        st.markdown(
            '<p class="panel-hdr">Triangulación de importancias</p>',
            unsafe_allow_html=True,
        )
        rf_fig = make_rf_triangulation(_dm)
        if rf_fig:
            st.plotly_chart(rf_fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Sin datos RF. Activa use_rf_shap=True en el pipeline.")
    with rf2:
        st.markdown(
            '<p class="panel-hdr">Importancia SHAP por clase</p>',
            unsafe_allow_html=True,
        )
        pc_fig = make_shap_per_class_importance(_dm)
        if pc_fig:
            st.plotly_chart(pc_fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Sin datos shap_per_class_mean_abs.")

    _sec_rule("SHAP Beeswarm · dirección del efecto por variable")
    _shap_classes_avail = [int(c) for c in shap_data.get("classes", [])]
    _shap_opts = ["Promedio entre clases"] + [f"Clase {c}" for c in _shap_classes_avail]
    _shap_sel = st.radio(
        "Vista SHAP:", _shap_opts, horizontal=True, key="tc_shap_class"
    )
    if _shap_sel == "Promedio entre clases":
        _shap_idx = None
    else:
        _chosen_c = int(_shap_sel.split()[-1])
        _shap_idx = (
            _shap_classes_avail.index(_chosen_c)
            if _chosen_c in _shap_classes_avail
            else None
        )
    bs_fig = make_shap_beeswarm(_dm, class_idx=_shap_idx)
    if bs_fig:
        st.plotly_chart(bs_fig, width="stretch", config={"displayModeBar": False})
    else:
        st.info("Sin datos SHAP. Activa use_rf_shap=True en el pipeline.")

    # Matriz de confusión semántica y métricas adicionales
    _sec_rule("Confusión semántica · errores del modelo = hallazgos")
    sc_fig = make_semantic_confusion(_dm)
    if sc_fig:
        st.plotly_chart(sc_fig, width="stretch", config={"displayModeBar": False})
    else:
        st.info(
            "Sin datos de confusión. El pipeline necesita guardar 'confusion_matrix' en shap_analysis, o bien 'y_true' y 'oof_predictions'."
        )

    # Nuevos gráficos: PCA de metadatos y tasa de error por clase
    _sec_rule("Diagnóstico del clasificador Random Forest")
    pca1, err1 = st.columns(2)
    with pca1:
        st.markdown(
            '<p class="panel-hdr">PCA · proyección de metadatos (✗ = mal clasificadas)</p>',
            unsafe_allow_html=True,
        )
        bn = st.slider("Términos", 10, 60, 28, 2, key="ta_bn2")
        pca_fig = make_mca_plot(_dm)
        if pca_fig:
            st.plotly_chart(pca_fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info(
                "Sin datos para PCA de metadatos. Requiere shap_analysis con raw_X, y_true y oof_predictions."
            )
    with err1:
        st.markdown(
            '<p class="panel-hdr">Tasa de error por clase (Out-of-Fold)</p>',
            unsafe_allow_html=True,
        )
        err_fig = make_error_by_class(_dm)
        if err_fig:
            st.plotly_chart(err_fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info(
                "Sin datos para tasa de error. Requiere shap_analysis con y_true y oof_predictions."
            )

    # Tasa de retención y residuo del corpus
    _sec_rule("Tasa de retención y residuo del corpus")
    retention_rate = (classified_uces / total_uces * 100) if total_uces else 0
    residual_uces = [
        u for u in uces if u.get("cluster_id") is None or u.get("cluster_id", -1) < 0
    ]

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        if retention_rate >= 75:
            rcol, rmsg = "#5DC88A", "Retención aceptable"
        elif retention_rate >= 55:
            rcol, rmsg = "#E8A838", "Retención marginal — revisar parámetros"
        else:
            rcol, rmsg = "#E86450", "⚠ Insuficiente — ajusta min_forms_uc o tsj"
        st.markdown(
            f"""
        <div class="retention-gauge" style="--ret-color:{rcol}">
          <div class="retention-pct">{retention_rate:.1f}%</div>
          <div class="retention-label">UCEs clasificadas</div>
          <div class="retention-msg">{rmsg}</div>
          <div style="font-family:var(--font-mono);font-size:9px;color:var(--text-low);margin-top:5px">
            {classified_uces} / {total_uces} UCEs
          </div>
        </div>
        <div style="margin:10px 0 0">
          <div style="height:5px;background:var(--bg-card);border-radius:3px;overflow:hidden">
            <div style="height:100%;width:{int(retention_rate)}%;background:{rcol};border-radius:3px"></div>
          </div>
        </div>
        <div style="margin-top:12px">
          <div class="info-row"><span class="info-key">min_forms_uc</span><span class="info-val">{min_forms_uc}</span></div>
          <div class="info-row"><span class="info-key">tsj</span><span class="info-val">{tsj}</span></div>
          <div class="info-row"><span class="info-key">método</span><span class="info-val">{sh(clustering_method)}</span></div>
          <div class="info-row"><span class="info-key">vocabulario</span><span class="info-val">{analyzed_forms} términos</span></div>
          <div class="info-row"><span class="info-key">hapax</span><span class="info-val">{hapax}</span></div>
          <div class="info-row"><span class="info-key">riqueza léxica</span><span class="info-val">{vocabulary_richness:.1f}%</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with rc2:
        st.markdown(
            f'<p class="panel-hdr">Muestra del residuo · '
            f'<span style="color:var(--text-low)">{len(residual_uces)} UCEs no clasificadas</span></p>',
            unsafe_allow_html=True,
        )
        filt = st.text_input(
            "Filtrar por texto:", placeholder="ej: educación, familia…", key="tc_rfilt"
        )
        filt_res = (
            [u for u in residual_uces if filt.lower() in u.get("texto", "").lower()]
            if filt
            else residual_uces
        )
        if filt_res:
            scr = '<div class="residual-scroll">'
            for u in filt_res:
                scr += f"""
                <div class="uce-card" style="border-left:3px solid var(--border2);opacity:.8">
                  <div class="uce-meta">
                    <span>{sh(u.get("id", "?")[:8])}</span>
                    <span>doc {u.get("doc_id", "?")}</span>
                    <span style="color:var(--text-dim)">no clasificada</span>
                  </div>
                  <div class="uce-text">{sh(u.get("texto", ""))}</div>
                </div>"""
            scr += "</div>"
            st.markdown(scr, unsafe_allow_html=True)
        else:
            st.info("Sin UCEs residuales que coincidan.")

# ─────────────────────────────────────────────────────────────
# PESTAÑA E
# ─────────────────────────────────────────────────────────────
with tab_e:
    import hashlib as _hlib
    from collections import defaultdict as _ddict
    from itertools import combinations as _combs

    # ════════════════════════════════════════════════════════════════════════
    # DESIGN TOKENS
    # ════════════════════════════════════════════════════════════════════════
    _E_BG = T["bg_page"]
    _E_SURF = T["bg_panel"]
    _E_BORD = T["border"]
    _E_TEXT = T["text_hi"]
    _E_DIM = T["text_dim"]
    _E_ACC = T["accent"]
    _E_MINT = "#50fa7b"
    _E_AMBER = "#ffb86c"
    _E_CORAL = "#f97b6b"
    _E_VIO = "#bd93f9"
    _E_PAL = [
        "#7c8cff",
        "#f97b6b",
        "#50fa7b",
        "#ffb86c",
        "#bd93f9",
        "#8be9fd",
        "#ff79c6",
        "#f1fa8c",
        "#6272a4",
        "#ff5555",
    ]

    # _E_PLY as plain dict — never unpacked with **, passed via update_layout()
    _E_PLY_BASE = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": dict(family="IBM Plex Mono, monospace", size=11, color=_E_TEXT),
        "legend": dict(
            bgcolor="rgba(22,25,41,0.9)",
            bordercolor=_E_BORD,
            borderwidth=1,
            font=dict(size=10),
        ),
        "hoverlabel": dict(
            bgcolor=_E_SURF,
            bordercolor=_E_BORD,
            font=dict(family="IBM Plex Mono, monospace", size=11),
        ),
    }

    def _ply(**extra):
        """Return a merged layout dict safe to pass to update_layout()."""
        d = dict(_E_PLY_BASE)
        d.update(extra)
        return d

    _E_CONF_RANK = {"alta": 3, "media": 2, "baja": 1}

    def _ec(s):
        return _E_PAL[int(_hlib.md5(str(s).encode()).hexdigest(), 16) % len(_E_PAL)]

    def _rgba(h, a=1.0):
        h = h.lstrip("#")
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"

    def _conf_color(conf):
        return {"alta": _E_MINT, "media": _E_AMBER, "baja": _E_CORAL}.get(conf, _E_DIM)

    # ════════════════════════════════════════════════════════════════════════
    # METADATA FIELD CLASSIFIER
    # ════════════════════════════════════════════════════════════════════════
    _META_SKIP = {
        "reasoning",
        "quote",
        "text",
        "texto",
        "description",
        "descripcion",
        "justificacion",
        "evidencia",
        "evidencia_uce_o_termino",
    }

    def _classify_meta_fields(ann_list):
        fv = _ddict(set)
        for ann in ann_list:
            for k, v in ann.get("metadata", {}).items():
                if isinstance(v, str) and v:
                    fv[k].add(v)
        cat, txt = {}, set()
        for k, vals in fv.items():
            if k in _META_SKIP or len(vals) > 7:
                txt.add(k)
            else:
                cat[k] = sorted(vals)
        return cat, txt

    # ════════════════════════════════════════════════════════════════════════
    # DATA PREPARATION
    # ════════════════════════════════════════════════════════════════════════

    @st.cache_data(show_spinner=False)
    def _e_prepare(ann_json: str, uces_json: str):
        ann_raw = json.loads(ann_json)
        uces_raw = json.loads(uces_json)

        ulookup = {}
        for u in uces_raw:
            uid = re.sub(r"__mf\d+$", "", u.get("id", ""))
            ulookup[uid] = u

        all_anns = [a for anns in ann_raw.values() for a in anns]
        cat_fields, txt_fields = _classify_meta_fields(all_anns)

        seen = set()
        rows = []
        for uce_id, anns in ann_raw.items():
            uid = re.sub(r"__mf\d+$", "", uce_id)
            uce = ulookup.get(uid, {})
            for ann in anns:
                meta = ann.get("metadata", {})
                trait = ann.get("trait", "?")
                spans = ann.get("spans", [])
                if not spans:
                    spans = [
                        {
                            "uce_id": uce_id,
                            "quote": ann.get("quote", ""),
                            "start_char": ann.get("start_char", -1),
                            "end_char": ann.get("end_char", -1),
                        }
                    ]
                for span in spans:
                    s = span.get("start_char", -1)
                    e = span.get("end_char", -1)
                    q = span.get("quote", "")
                    key = (uid, trait, s, e, q)
                    if key in seen:
                        continue
                    seen.add(key)
                    row = {
                        "uce_id": uid,
                        "trait": trait,
                        "confidence": ann.get("confidence", "baja"),
                        "conf_rank": _E_CONF_RANK.get(ann.get("confidence", "baja"), 1),
                        "subtype": ann.get("subtype", ""),
                        "agent": ann.get("agent", ""),
                        "quote": q,
                        "start_char": s,
                        "end_char": e,
                        "reasoning": meta.get("reasoning", ""),
                        "cluster_id": uce.get("cluster_id"),
                        "texto": uce.get("texto", ""),
                        "n_verbos": len(uce.get("verbos", [])),
                        "n_neg": len(uce.get("negaciones", [])),
                        "n_pron": len(uce.get("pronombres", [])),
                        "n_coref": len(uce.get("coref_chains", [])),
                        "registro": uce.get("registro", ""),
                    }
                    for f in cat_fields:
                        row[f"meta_{f}"] = meta.get(f, "")
                    # store all non-reasoning metadata for hover tooltip
                    row["meta_full"] = json.dumps(
                        {k: v for k, v in meta.items() if k != "reasoning"},
                        ensure_ascii=False,
                    )
                    rows.append(row)

        df = pd.DataFrame(rows)
        vidx = {
            re.sub(r"__mf\d+$", "", u.get("id", "")): u.get("verbos", [])
            for u in uces_raw
        }
        return df, cat_fields, txt_fields, vidx

    # ── serialise once ───────────────────────────────────────────────────────
    _e_disc = load_discourse_state()
    _e_ann_raw = _e_disc.get("annotations_by_uce", {})
    _e_ann_json = json.dumps(_e_ann_raw, ensure_ascii=False, default=str)
    _e_uces_json = json.dumps(
        [
            {
                k: v
                for k, v in u.items()
                if k
                not in ("span", "_coref_chains_full", "_predicate_frames_serialized")
            }
            for u in uces
        ],
        ensure_ascii=False,
        default=str,
    )

    _df_all, _cat_fields, _txt_fields, _verb_idx = _e_prepare(_e_ann_json, _e_uces_json)

    if _df_all.empty:
        st.info("No hay anotaciones discursivas en este corpus.")
        st.stop()

    _all_traits = sorted(_df_all["trait"].unique())
    _meta_cols = [f"meta_{f}" for f in _cat_fields]  # promoted columns in df

    # ════════════════════════════════════════════════════════════════════════
    # CLASS SELECTOR
    # ════════════════════════════════════════════════════════════════════════
    if "te_active_cls" not in st.session_state:
        st.session_state.te_active_cls = class_list[0] if class_list else None

    _cls_cols = st.columns(len(class_list))
    for _i, _c in enumerate(class_list):
        with _cls_cols[_i]:
            _is_act_cls = st.session_state.te_active_cls == _c
            if st.button(
                f"Clase {_c}",
                key=f"te_cls_{_c}",
                use_container_width=True,
                type="primary" if _is_act_cls else "secondary",
            ):
                st.session_state.te_active_cls = _c
                st.rerun()
    _active_cls = st.session_state.te_active_cls
    _acc_col = class_colors.get(_active_cls, _E_ACC)

    # ════════════════════════════════════════════════════════════════════════
    # TRAIT SELECTOR
    # ════════════════════════════════════════════════════════════════════════
    if (
        "te_trait" not in st.session_state
        or st.session_state.te_trait not in _all_traits
    ):
        st.session_state.te_trait = _all_traits[-1] if _all_traits else None

    _tr_cols = st.columns(len(_all_traits))
    for _i, _tr in enumerate(_all_traits):
        with _tr_cols[_i]:
            _is_act_tr = st.session_state.te_trait == _tr
            if st.button(
                _tr,
                key=f"te_tr_{_tr}",
                use_container_width=True,
                type="primary" if _is_act_tr else "secondary",
            ):
                st.session_state.te_trait = _tr
                st.rerun()
    _sel_trait = st.session_state.te_trait

    # 1. NEW BIG HEADER FOR THE TRAIT
    st.markdown(
        f'<h1 style="margin: 1.2rem 0 0.5rem; font-size: 2.2rem; font-weight: bold; '
        f'color: {_ec(_sel_trait)};">{_sel_trait}</h1>',
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════════════
    # CONFIDENCE SELECTOR
    # ════════════════════════════════════════════════════════════════════════
    if "te_confs" not in st.session_state:
        st.session_state.te_confs = ["alta"]

    _cf1, _cf2, _cf3 = st.columns(3)
    for _cw, _cv in zip([_cf1, _cf2, _cf3], ["alta", "media", "baja"]):
        with _cw:
            _conf_in = _cv in st.session_state.te_confs
            if st.button(
                _cv,
                key=f"te_conf_{_cv}",
                use_container_width=True,
                type="primary" if _conf_in else "secondary",
            ):
                _cur_c = list(st.session_state.te_confs)
                if _conf_in and len(_cur_c) > 1:
                    _cur_c.remove(_cv)
                elif not _conf_in:
                    _cur_c.append(_cv)
                st.session_state.te_confs = _cur_c
                st.rerun()
    _sel_confs = st.session_state.te_confs

    # ════════════════════════════════════════════════════════════════════════
    # DERIVED DATAFRAMES
    # ════════════════════════════════════════════════════════════════════════
    _df_cls = _df_all[
        (_df_all["cluster_id"] == _active_cls)
        & (_df_all["confidence"].isin(_sel_confs))
    ].copy()

    _df_filt = _df_cls[_df_cls["trait"] == _sel_trait].copy()

    _uces_active = [u for u in uces if u.get("cluster_id") == _active_cls]
    _uce_order = [re.sub(r"__mf\d+$", "", u["id"]) for u in _uces_active]
    _uce_pos_map = {uid: i for i, uid in enumerate(_uce_order)}

    # ════════════════════════════════════════════════════════════════════════
    # BEST SUBTYPE COLUMN HELPER
    # ════════════════════════════════════════════════════════════════════════
    def _best_sub_col(df_tr):
        """
        Pick the best categorical metadata column for this trait's df.
        Prefers columns that are non-empty and have ≤7 unique values.
        Falls back to raw 'subtype' column.
        """
        if df_tr.empty:
            return "subtype"
        for mc in _meta_cols:
            if mc in df_tr.columns:
                filled = df_tr[mc][df_tr[mc] != ""]
                if len(filled) > 0 and filled.nunique() <= 7:
                    return mc
        return "subtype"

    # ════════════════════════════════════════════════════════════════════════
    # STATS BANNER
    # ════════════════════════════════════════════════════════════════════════
    def _badge(lbl, val, col=_E_ACC):
        # width adapts to content via min-width + padding only
        return (
            f'<div style="display:inline-flex;flex-direction:column;'
            f"align-items:center;padding:6px 10px;background:{_E_SURF};"
            f'border:1px solid {_E_BORD};border-radius:6px;white-space:nowrap;">'
            f'<span style="font-size:15px;font-weight:700;color:{col};'
            f'font-family:IBM Plex Mono,monospace;">{val}</span>'
            f'<span style="font-size:8px;color:{_E_DIM};text-transform:uppercase;'
            f'letter-spacing:.1em;margin-top:1px;">{lbl}</span></div>'
        )

    _n_cls_total = len(_df_cls)
    _n_trait = len(_df_filt)
    _n_uces_tr = _df_filt["uce_id"].nunique() if not _df_filt.empty else 0
    _pct_trait = round(_n_trait / _n_cls_total * 100, 1) if _n_cls_total else 0
    _n_alta = int((_df_filt["confidence"] == "alta").sum()) if not _df_filt.empty else 0

    # tipo · subtipo top: show trait name as "tipo", best meta value as "subtipo"
    _top_meta_str = ""
    if _sel_trait and not _df_filt.empty:
        for _mf in _meta_cols:
            if _mf in _df_filt.columns:
                _vv = _df_filt[_mf][_df_filt[_mf] != ""].value_counts()
                if not _vv.empty:
                    _top_meta_str = f"{_vv.index[0]}"  # Removed the trait prefix
                    break

    st.html(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 10px;">'
        + _badge("spans cls", _n_cls_total, _acc_col)
        + _badge(_sel_trait or "trait", _n_trait, _ec(_sel_trait or ""))
        + _badge("UCEs", _n_uces_tr, _acc_col)
        + _badge("% total", f"{_pct_trait}%", _E_VIO)
        + _badge("alta conf.", _n_alta, _E_MINT)
        + (
            _badge("subtipo top", _top_meta_str, _E_AMBER)  # Updated label
            if _top_meta_str
            else ""
        )
        + "</div>"
    )
    # ════════════════════════════════════════════════════════════════════════
    # TRAJECTORY HELPER  (cached)
    # ════════════════════════════════════════════════════════════════════════

    def _doc_from_uid(uid):
        """First segment of doc_sec_uce → document number string."""
        return uid.split("_")[0]

    @st.cache_data(show_spinner=False)
    def _build_traj(df_ann: pd.DataFrame, uce_order_key: tuple, n_bins: int = 20):
        # 1. Safely extract UIDs as strings
        ann_uids = (
            set(df_ann["uce_id"].astype(str).unique()) if not df_ann.empty else set()
        )

        uce_order = list(uce_order_key)

        # Group UCEs by document, preserving corpus order
        doc_uces = _ddict(list)  # doc_str → [uid, uid, ...]
        for uid in uce_order:
            doc_uces[_doc_from_uid(uid)].append(uid)

        rows = []
        for doc, uids in doc_uces.items():
            total = len(uids)
            if total == 0:
                continue
            # assign bin for each uce position
            bin_map = _ddict(lambda: {"total": 0, "ann": 0, "uids": []})
            for rank, uid in enumerate(uids):
                b = min(int(rank / total * n_bins), n_bins - 1)
                bin_map[b]["total"] += 1
                bin_map[b]["uids"].append(uid)
                if uid in ann_uids:
                    bin_map[b]["ann"] += 1
            for b, info in bin_map.items():
                prob = info["ann"] / info["total"] if info["total"] > 0 else 0.0
                rows.append(
                    {
                        "doc": doc,
                        "bin_idx": b,
                        "bin_label": f"{int(b / n_bins * 100)}–{int((b + 1) / n_bins * 100)}%",
                        "probability": round(prob, 4),
                        "first_uid": info["uids"][0] if info["uids"] else "",
                    }
                )

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ════════════════════════════════════════════════════════════════════════
    # THREE-PANEL LAYOUT  [1 · 1.5 · 1]
    # ════════════════════════════════════════════════════════════════════════
    _col_l, _col_c, _col_r = st.columns([1, 1.5, 1])

    # ════════════════════════════════════════════════════════════════════════
    # LEFT PANEL
    # ════════════════════════════════════════════════════════════════════════
    with _col_l:
        # ── Radar ────────────────────────────────────────────────────────────
        st.markdown(
            f'<p style="font-size:9px;letter-spacing:.12em;color:{_E_DIM};'
            f'text-transform:uppercase;margin-bottom:4px;">'
            f"Peso por rasgo · Clase {_active_cls}</p>",
            unsafe_allow_html=True,
        )
        _radar_traits = sorted(_df_cls["trait"].unique()) if not _df_cls.empty else []

        if len(_radar_traits) >= 3:
            _rt_counts = _df_cls.groupby("trait").size()
            _rt_pct = (_rt_counts / _rt_counts.sum() * 100).round(1)
            _rv = [float(_rt_pct.get(t, 0.0)) for t in _radar_traits]
            _r_max_ceil = max(10, round(max(_rv) * 1.15))

            _fig_r = go.Figure()
            _fig_r.add_trace(
                go.Scatterpolar(
                    r=_rv + [_rv[0]],
                    theta=_radar_traits + [_radar_traits[0]],
                    fill="toself",
                    fillcolor=_rgba(_acc_col, 0.12),
                    line=dict(color=_acc_col, width=2),
                    name=f"Clase {_active_cls}",
                    hovertemplate="<b>%{theta}</b>: %{r:.1f}%<extra></extra>",
                )
            )
            _ref = round(100 / len(_radar_traits), 1)
            _fig_r.add_trace(
                go.Scatterpolar(
                    r=[_ref] * len(_radar_traits) + [_ref],
                    theta=_radar_traits + [_radar_traits[0]],
                    mode="lines",
                    line=dict(color=_E_DIM, width=1, dash="dot"),
                    name=f"ref. ({_ref}%)",
                    hoverinfo="skip",
                )
            )
            _fig_r.update_layout(
                _ply(
                    height=300,
                    polar=dict(
                        bgcolor=_E_SURF,
                        radialaxis=dict(
                            visible=True,
                            range=[0, _r_max_ceil],
                            ticksuffix="%",
                            tickfont=dict(size=8, color=_E_DIM),
                            gridcolor=_E_BORD,
                            linecolor=_E_BORD,
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=9, color=_E_TEXT),
                            gridcolor=_E_BORD,
                            linecolor=_E_BORD,
                        ),
                    ),
                    showlegend=False,
                    margin=dict(t=30, b=30, l=20, r=20),
                )
            )
            st.plotly_chart(
                _fig_r,
                width="stretch",
                key="te_radar",
                config={"displayModeBar": False},
            )
        else:
            st.caption("Se necesitan ≥3 rasgos para el radar.")

        st.divider()

        # ── Subtipos por rasgo — horizontal stacked bars, no legend ──────────
        st.markdown(
            f'<p style="font-size:9px;letter-spacing:.12em;color:{_E_DIM};'
            f'text-transform:uppercase;margin-bottom:4px;">Subtipos por rasgo</p>',
            unsafe_allow_html=True,
        )
        if not _df_cls.empty:
            _bar_rows = []
            for _tr in sorted(_df_cls["trait"].unique()):
                _dtr = _df_cls[_df_cls["trait"] == _tr]
                _sc = _best_sub_col(_dtr)
                for _sv, _n in _dtr[_sc][_dtr[_sc] != ""].value_counts().items():
                    _bar_rows.append({"trait": _tr, "sub": str(_sv), "n": int(_n)})

            if _bar_rows:
                _bdf = pd.DataFrame(_bar_rows)
                _fig_b = go.Figure()
                for _sv in list(_bdf["sub"].unique()):
                    _bsub = _bdf[_bdf["sub"] == _sv]
                    _fig_b.add_trace(
                        go.Bar(
                            y=_bsub["trait"],
                            x=_bsub["n"],
                            name=_sv,
                            orientation="h",
                            marker=dict(
                                color=_ec(_sv), line=dict(color=_E_BG, width=0.4)
                            ),
                            hovertemplate=f"{_sv}: %{{x}}<extra></extra>",
                        )
                    )
                _fig_b.update_layout(
                    _ply(
                        barmode="stack",
                        height=max(180, 30 * _df_cls["trait"].nunique() + 50),
                        showlegend=False,
                        xaxis=dict(
                            gridcolor=_E_BORD,
                            linecolor=_E_BORD,
                            zerolinecolor=_E_BORD,
                            tickfont=dict(size=9),
                        ),
                        yaxis=dict(
                            gridcolor=_E_BORD, linecolor=_E_BORD, tickfont=dict(size=9)
                        ),
                        margin=dict(t=10, b=16, l=4, r=4),
                    )
                )
                st.plotly_chart(
                    _fig_b,
                    width="stretch",
                    key="te_sub_bars",
                    config={"displayModeBar": False},
                )

        st.divider()

        # ── Co-occurrence matrix ──────────────────────────────────────────────
        st.markdown(
            f'<p style="font-size:9px;letter-spacing:.12em;color:{_E_DIM};'
            f'text-transform:uppercase;margin-bottom:4px;">Co-ocurrencia P(a∩b)</p>',
            unsafe_allow_html=True,
        )

        if not _df_cls.empty:
            _traits_u = sorted(_df_cls["trait"].unique())
            _n_uces_cl = _df_cls["uce_id"].nunique()
            _cooc = pd.DataFrame(0.0, index=_traits_u, columns=_traits_u)

            for _, _grp in _df_cls.groupby("uce_id"):
                _present = _grp["trait"].unique()
                for _a, _b in _combs(sorted(_present), 2):
                    _cooc.loc[_a, _b] += 1
                    _cooc.loc[_b, _a] += 1

            if _n_uces_cl > 0:
                _cooc = (_cooc / _n_uces_cl).round(3)

            # 1. ISOLATE THE LOWER TRIANGLE
            # Create a boolean mask of the upper triangle.
            # k=0 includes the diagonal in the mask (since combinations don't self-intersect, the diagonal is 0 anyway).
            _mask = np.triu(np.ones_like(_cooc, dtype=bool), k=0)

            # .mask() replaces the True values in _mask with NaN. Plotly renders NaN as invisible.
            _cooc_half = _cooc.mask(_mask)

            # 2. DYNAMIC COLOR NORMALIZATION
            # Find the absolute maximum value in the active half to anchor the top of the colorscale.
            # Fallback to 1.0 if the matrix is entirely empty/zeros to avoid rendering errors.
            _actual_max = _cooc_half.max().max()
            _zmax = _actual_max if pd.notna(_actual_max) and _actual_max > 0 else 1.0

            _fig_cooc = go.Figure(
                go.Heatmap(
                    z=_cooc_half.values,
                    x=_traits_u,
                    y=_traits_u,
                    colorscale=[[0, _E_SURF], [0.5, _E_BORD], [1, _acc_col]],
                    zmin=0,
                    zmax=_zmax,  # Gradient is now perfectly stretched to your data's reality
                    showscale=False,
                    xgap=1,  # Optional: Adds a 1px gap between cells. Makes solid colors look much cleaner.
                    ygap=1,
                    hovertemplate="P(%{y} ∩ %{x}) = %{z:.3f}<extra></extra>",
                )
            )

            _fig_cooc.update_layout(
                _ply(
                    height=max(180, 36 * len(_traits_u)),
                    margin=dict(t=10, b=10, l=4, r=4),
                    xaxis=dict(tickfont=dict(size=8, color=_E_DIM), linecolor=_E_BORD),
                    yaxis=dict(
                        tickfont=dict(size=8, color=_E_DIM),
                        linecolor=_E_BORD,
                        autorange="reversed",
                    ),
                    plot_bgcolor="rgba(0,0,0,0)",  # Ensures the NaN cells match your app background perfectly
                    paper_bgcolor="rgba(0,0,0,0)",
                )
            )

            st.plotly_chart(
                _fig_cooc,
                width="stretch",
                key="te_cooc_left",
                config={"displayModeBar": False},
            )

    # ════════════════════════════════════════════════════════════════════════
    # CENTRE PANEL — UCE cards
    # ════════════════════════════════════════════════════════════════════════
    with _col_c:
        # 1. Dynamically extract unique document IDs from the filtered dataframe
        if not _df_filt.empty:
            _doc_ids = sorted(
                list(set(uid.split("_")[0] for uid in _df_filt["uce_id"])),
                key=lambda x: int(x) if x.isdigit() else x,
            )
        else:
            _doc_ids = []
        _doc_opts = ["Todos"] + _doc_ids

        # 2. Layout the sort and document selectors side-by-side
        _c_ctrl1, _c_ctrl2 = st.columns(2)

        with _c_ctrl1:
            _sort_opts = {
                "Menos anotaciones": ("n_anns", True),
                "Más anotaciones": ("n_anns", False),
                "Orden de UCE": ("pos", True),
            }
            _sort_sel = st.selectbox(
                "Ordenar",
                list(_sort_opts),
                index=0,
                key="te_sort",
                label_visibility="collapsed",
            )

        with _c_ctrl2:
            _doc_sel = st.selectbox(
                "Documento",
                _doc_opts,
                index=0,
                key="te_doc_filter",
                label_visibility="collapsed",
            )

        _scroll_uid = st.session_state.get("te_scroll_uid", None)

        # build span lists per uce
        _spans_by_uce = _ddict(list)
        for _, _r in _df_filt.iterrows():
            _spans_by_uce[_r["uce_id"]].append(_r.to_dict())

        _uid_to_uce = {re.sub(r"__mf\d+$", "", u["id"]): u for u in _uces_active}
        _ann_count = (
            _df_filt.groupby("uce_id").size().to_dict() if not _df_filt.empty else {}
        )

        _cards = []
        for _uid, _uce in _uid_to_uce.items():
            _spns = _spans_by_uce.get(_uid, [])
            if not _spns:
                continue

            # 3. Apply the document filter here
            _doc_of_uid = _uid.split("_")[0]
            if _doc_sel != "Todos" and _doc_of_uid != str(_doc_sel):
                continue

            _cards.append(
                {
                    "uid": _uid,
                    "uce": _uce,
                    "spans": _spns,
                    "n_anns": _ann_count.get(_uid, 0),
                    "pos": _uce_pos_map.get(_uid, 9999),
                }
            )

        _sk, _sr = _sort_opts[_sort_sel]
        _cards.sort(key=lambda x: x[_sk], reverse=not _sr)

        # Update the header string to reflect the active document filter
        _doc_lbl = f" · Doc {_doc_sel}" if _doc_sel != "Todos" else ""
        st.markdown(
            f'<div style="font-size:9px;color:{_E_DIM};margin-bottom:6px;">'
            f"{len(_cards)} UCEs · {_sel_trait}{_doc_lbl} · conf: {', '.join(_sel_confs)}</div>",
            unsafe_allow_html=True,
        )

        # ── highlight (underline only, no background) ─────────────────────────
        def _hl(texto, spans):
            if not texto or not spans:
                return texto or ""
            evts = []
            for sp in spans:
                s, e = sp.get("start_char", -1), sp.get("end_char", -1)
                if s < 0 or e <= s or e > len(texto):
                    continue
                col_ = _ec(sp.get("trait", ""))
                parts = []
                rsn = sp.get("reasoning", "")
                if rsn:
                    parts.append(rsn[:160])
                try:
                    mf = json.loads(sp.get("meta_full", "{}"))
                    for mk, mv in mf.items():
                        if mv:
                            parts.append(f"{mk}: {str(mv)[:60]}")
                except Exception:
                    pass
                tip = " | ".join(parts)[:220].replace('"', "'")
                evts.append((s, e, col_, tip))
            evts.sort(key=lambda x: (x[0], -(x[1] - x[0])))
            res, cur = [], 0
            for s, e, col_, tip in evts:
                if s < cur:
                    continue
                res.append(texto[cur:s])
                res.append(
                    f'<span style="border-bottom:2px solid {col_};color:{col_};'
                    f'cursor:help;" title="{tip}">{texto[s:e]}</span>'
                )
                cur = e
            res.append(texto[cur:])
            return "".join(res)

        # ── card renderer ─────────────────────────────────────────────────────
        def _card_html(item):
            _uid = item["uid"]
            _uce = item["uce"]
            _cl = class_colors.get(_uce.get("cluster_id"), _E_ACC)
            _hl_t = _hl(_uce.get("texto", ""), item["spans"])
            _border = _E_MINT if _uid == _scroll_uid else _cl

            _pills = ""
            for sp in item["spans"]:
                _conf = sp.get("confidence", "")
                _cc = _conf_color(_conf)
                _cat_p = ""
                for _mf in _meta_cols:
                    _v = sp.get(_mf, "")
                    if _v:
                        _fn = _mf.replace("meta_", "")
                        _cat_p += (
                            f'<span style="font-size:8px;padding:1px 6px;'
                            f"border-radius:3px;background:{_ec(_v)}22;"
                            f'color:{_ec(_v)};margin-right:3px;white-space:nowrap;">'
                            f"{_fn}: {_v}</span>"
                        )
                _rsn = sp.get("reasoning", "")
                _rsn_ic = (
                    (
                        f' <span style="font-size:9px;color:{_E_DIM};cursor:help;" '
                        f'title="{_rsn[:200].replace(chr(34), chr(39))}">ℹ</span>'
                    )
                    if _rsn
                    else ""
                )
                _pills += (
                    f'<div style="display:flex;flex-wrap:wrap;align-items:center;'
                    f'gap:3px;margin:2px 0;">'
                    f'<span style="font-size:8px;padding:0 5px;border-radius:2px;'
                    f'background:{_cc}22;color:{_cc};white-space:nowrap;">{_conf}</span>'
                    f"{_cat_p}{_rsn_ic}"
                    f'<span style="font-size:9px;color:{_E_DIM};font-style:italic;">'
                    f"&ldquo;{sp.get('quote', '')[:80]}&rdquo;</span>"
                    f"</div>"
                )

            return (
                f'<div id="uce-{_uid}" style="margin-bottom:9px;padding:9px 11px;'
                f"background:{_E_SURF};border:1px solid {_E_BORD};"
                f'border-left:3px solid {_border};border-radius:0 6px 6px 0;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                f'<span style="font-size:8px;color:{_E_DIM};'
                f'font-family:IBM Plex Mono,monospace;">{_uid}</span>'
                f'<span style="font-size:8px;color:{_E_ACC};">{item["n_anns"]} ann.</span>'
                f"</div>"
                f'<div style="font-family:Georgia,serif;font-size:12px;'
                f'line-height:1.65;color:{_E_TEXT};margin-bottom:5px;">{_hl_t}</div>'
                f"{_pills}"
                f"</div>"
            )

        _scroll_js = (
            (
                f'<script>window.addEventListener("load",function(){{'
                f'var el=document.getElementById("uce-{_scroll_uid}");'
                f'if(el)el.scrollIntoView({{behavior:"smooth",block:"start"}});}});</script>'
            )
            if _scroll_uid
            else ""
        )

        with st.container(height=980, border=False):
            if not _cards:
                st.info(f"Sin UCEs con anotaciones de '{_sel_trait}'.")
            else:
                st.html(_scroll_js + "".join(_card_html(it) for it in _cards))

    # ════════════════════════════════════════════════════════════════════════
    # RIGHT PANEL — trajectory heatmap
    # ════════════════════════════════════════════════════════════════════════
    with _col_r:
        st.markdown(
            f'<p style="font-size:9px;letter-spacing:.12em;color:{_E_DIM};'
            f'text-transform:uppercase;margin-bottom:4px;">'
            f"Trayectoria · {_sel_trait} · Clase {_active_cls}</p>",
            unsafe_allow_html=True,
        )

        _N_BINS = 20
        _traj_df = _build_traj(
            _df_filt,  # Pass the DataFrame natively
            tuple(_uce_order),
            n_bins=_N_BINS,
        )

        if _traj_df.empty:
            st.info(f"Sin datos de trayectoria para '{_sel_trait}'.")
        else:
            _docs = sorted(
                _traj_df["doc"].unique(), key=lambda x: int(x) if x.isdigit() else x
            )
            _bin_lbs = [
                f"{int(b / _N_BINS * 100)}–{int((b + 1) / _N_BINS * 100)}%"
                for b in range(_N_BINS)
            ]

            _z_mat = np.zeros((len(_docs), _N_BINS))
            _uid_mat = [[""] * _N_BINS for _ in _docs]
            _hover_mat = [[""] * _N_BINS for _ in _docs]

            for _di, _doc in enumerate(_docs):
                _sub_d = _traj_df[_traj_df["doc"] == _doc]
                for _, _trow in _sub_d.iterrows():
                    _bi = int(_trow["bin_idx"])
                    _z_mat[_di, _bi] = float(_trow["probability"])
                    _uid_mat[_di][_bi] = str(_trow.get("first_uid", ""))
                    _hover_mat[_di][_bi] = (
                        f"Doc {_doc} · {_trow['bin_label']}<br>"
                        f"P = {_trow['probability']:.3f}"
                    )

            # 3. TRANSPOSE MATRICES
            _z_mat_T = _z_mat.T
            _uid_mat_T = list(map(list, zip(*_uid_mat)))
            _hover_mat_T = list(map(list, zip(*_hover_mat)))

            # 4. BINARY PRESENCE COLORSCALE (using trait color)
            _tr_col = _ec(_sel_trait)
            # Anything > 0 immediately snaps to the trait color
            _cs_binary = [
                [0.0, _E_SURF],
                [0.0001, _E_SURF],
                [0.0001, _tr_col],
                [1.0, _tr_col],
            ]

            _fig_traj = go.Figure(
                go.Heatmap(
                    z=_z_mat_T,
                    x=[f"doc {d}" for d in _docs],  # Docs are now X
                    y=_bin_lbs,  # Bins are now Y
                    colorscale=_cs_binary,
                    zmin=0,
                    zmax=1,
                    xgap=2,
                    ygap=2,
                    text=_hover_mat_T,
                    hovertemplate="%{text}<extra></extra>",
                    showscale=False,  # Hid colorbar since it's just binary presence now
                )
            )

            _fig_traj.update_layout(
                _ply(
                    height=max(400, 15 * _N_BINS + 80),  # Height scales by bins now
                    xaxis=dict(
                        tickfont=dict(size=8, color=_E_DIM),
                        tickangle=45,
                        gridcolor=_E_BORD,
                        linecolor=_E_BORD,
                        title="Documento",
                    ),
                    yaxis=dict(
                        tickfont=dict(size=9, color=_E_TEXT),
                        linecolor=_E_BORD,
                        autorange="reversed",  # Puts 0-5% at the top
                        title="Posición (%)",
                    ),
                    margin=dict(t=20, b=70, l=60, r=20),
                    showlegend=False,
                )
            )

            _traj_ev = st.plotly_chart(
                _fig_traj,
                width="stretch",
                config={"displayModeBar": False},
                on_select="rerun",
                key="te_traj_chart",
            )

            # 5. UPDATED CLICK LOGIC FOR TRANSPOSED AXES
            if isinstance(_traj_ev, dict) and _traj_ev.get("selection", {}).get(
                "points"
            ):
                _pt = _traj_ev["selection"]["points"][0]
                # X is now Doc, Y is now Bin
                _x_str = str(_pt.get("x", ""))
                _y_str = str(_pt.get("y", ""))
                try:
                    _di_idx = [f"doc {d}" for d in _docs].index(_x_str)
                    _bi_idx = _bin_lbs.index(_y_str)
                    # The original uid_mat was generated as [doc][bin], so we use di_idx first
                    _tgt = _uid_mat[_di_idx][_bi_idx]
                    if _tgt:
                        st.session_state["te_scroll_uid"] = _tgt
                        st.rerun()
                except (ValueError, IndexError):
                    pass

            st.caption(f"Presencia de rasgo. {_N_BINS} divisiones. Clic → ir al texto.")

st.markdown(
    f"""
<div class="footer">
  <span>ALCESTE · Dashboard v4 · {len(clusters_unicos)} clases · {len(uces)} UCEs</span>
  <span>workflow_data.json · {fecha}</span>
</div>
""",
    unsafe_allow_html=True,
)
