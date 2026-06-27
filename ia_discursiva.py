#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
from json_repair import repair_json
from jsonschema import ValidationError, validate  # new import

try:
    from rapidfuzz import fuzz

    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("⚠️  rapidfuzz no instalado. El fallback de fuzzy-match estará desactivado.")

from gram.gramatical_analyzer import UCE, PredicateFrame

# ─────────────────────────────────────────────
# Rutas de persistencia
# ─────────────────────────────────────────────
DATA_DIR = Path("data")
STATE_FILE = DATA_DIR / "discourse_state.json"

# ─────────────────────────────────────────────
# Configuración API
# ─────────────────────────────────────────────
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-717344c1653040459602049505783c92")
if not DEEPSEEK_API_KEY:
    print("⚠️  DEEPSEEK_API_KEY no definida.")


# ─────────────────────────────────────────────
# Modelos de datos
# ─────────────────────────────────────────────
class Confidence(str, Enum):
    ALTA = "alta"
    MEDIA = "media"
    BAJA = "baja"


@dataclass
class SpanInfo:
    uce_id: str
    quote: str
    start_char: int = -1
    end_char: int = -1

    def to_dict(self) -> Dict:
        return {
            "uce_id": self.uce_id,
            "quote": self.quote,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SpanInfo":
        return cls(
            uce_id=d.get("uce_id", ""),
            quote=d.get("quote", ""),
            start_char=d.get("start_char", -1),
            end_char=d.get("end_char", -1),
        )


@dataclass
class DiscourseAnnotation:
    trait: str
    agent: Optional[str] = None
    subtype: Optional[str] = None
    spans: List[SpanInfo] = field(default_factory=list)
    confidence: Confidence = Confidence.BAJA
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Legacy fields kept for compatibility
    uce_id: Optional[str] = None
    quote: Optional[str] = None

    def __post_init__(self):
        if self.uce_id and self.quote and not self.spans:
            self.spans = [SpanInfo(uce_id=self.uce_id, quote=self.quote)]

    def to_dict(self) -> Dict:
        return {
            "trait": self.trait,
            "agent": self.agent,
            "subtype": self.subtype,
            "spans": [s.to_dict() for s in self.spans],
            "confidence": self.confidence.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DiscourseAnnotation":
        conf = d.get("confidence", "baja")
        if conf not in ("alta", "media", "baja"):
            conf = "baja"
        spans = [SpanInfo.from_dict(s) for s in d.get("spans", [])]
        if not spans and d.get("uce_id") and d.get("quote"):
            spans = [SpanInfo(uce_id=d["uce_id"], quote=d["quote"])]
        return cls(
            trait=d.get("trait", ""),
            agent=d.get("agent"),
            subtype=d.get("subtype"),
            spans=spans,
            confidence=Confidence(conf),
            metadata=d.get("metadata", {}),
        )


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _safe_truncate_dict(d: Any, max_list_items: int = 5, max_str_len: int = 200) -> Any:
    if isinstance(d, dict):
        return {
            k: _safe_truncate_dict(v, max_list_items, max_str_len) for k, v in d.items()
        }
    if isinstance(d, list):
        return [
            _safe_truncate_dict(i, max_list_items, max_str_len)
            for i in d[:max_list_items]
        ]
    if isinstance(d, str) and len(d) > max_str_len:
        return d[:max_str_len] + "…"
    return d


def _counter_to_serialisable(c: Any) -> Any:
    """Recursively convert Counter / set objects so they survive json.dumps."""
    if isinstance(c, Counter):
        return dict(c)
    if isinstance(c, set):
        return sorted(c)
    if isinstance(c, dict):
        return {k: _counter_to_serialisable(v) for k, v in c.items()}
    if isinstance(c, list):
        return [_counter_to_serialisable(i) for i in c]
    return c


def _grammar_summary_block(summary: str, max_chars: int = 3000) -> str:
    if len(summary) <= max_chars:
        return summary
    lines = summary.splitlines()
    out, total = [], 0
    for line in lines:
        if total + len(line) + 1 > max_chars:
            break
        out.append(line)
        total += len(line) + 1
    return "\n".join(out) + "\n[…resumen truncado…]"


# ─────────────────────────────────────────────
# gram_cats → structured_grammar bridge
# ─────────────────────────────────────────────

# Canonical gram_cat names (case-insensitive lookup keys)
_GRAM_CAT_ALIASES: Dict[str, str] = {
    "verbos": "verbos",
    "verbo": "verbos",
    "verb": "verbos",
    "sustantivos": "nouns",
    "sustantivo": "nouns",
    "noun": "nouns",
    "nouns": "nouns",
    "adjetivos": "adjectives",
    "adjetivo": "adjectives",
    "adjective": "adjectives",
    "adjectives": "adjectives",
    "preposiciones": "prepositions",
    "preposicion": "prepositions",
    "preposition": "prepositions",
    "prepositions": "prepositions",
    "pronombres": "pronombres",
    "pronombre": "pronombres",
    "pronoun": "pronombres",
    "adverbios": "adverbios",
    "adverbio": "adverbios",
    "adverb": "adverbios",
    "marcadores": "marcadores",
    "marcadores/conjunciones": "marcadores",
    "conjunciones": "marcadores",
    "connectives": "marcadores",
    "negaciones": "negaciones",
    "negacion": "negaciones",
    "negation": "negaciones",
    "cuantificadores": "cuantificadores",
    "cuantificador": "cuantificadores",
    "quantifier": "cuantificadores",
    "insubordinaciones": "insubordinaciones",
    "coref": "coref",
    "coreferencias": "coref",
    "coreference": "coref",
    "predicados": "predicados",
    "predicate": "predicados",
    "predicado": "predicados",
    "complejidad": "complejidad_sintactica",
    "sintaxis": "complejidad_sintactica",
    "complexity": "complejidad_sintactica",
    "metricas": "metricas_lexicas",
    "metricas_lexicas": "metricas_lexicas",
    "lexical": "metricas_lexicas",
    "subtlex": "subtlex",
    "frecuencia": "subtlex",
    "registros": "registros",
    "register": "registros",
    "verbos_lexicos": "verbos_lexicos",
    "verbos lexicos": "verbos_lexicos",
    "lexical verbs": "verbos_lexicos",
}


# How to render each canonical key as readable text
def _render_section(key: str, data: Any) -> str:
    """Convert a structured_grammar sub-dict into a human-readable string."""
    if data is None:
        return f"  [{key}: sin datos]"

    # ── VERBOS ──────────────────────────────────────────────────────────
    if key == "verbos":
        lines = ["  Verbos:"]
        for sub in (
            "modo",
            "tiempo",
            "aspecto",
            "voz",
            "persona",
            "subordinacion_tipo",
            "perifrasis",
        ):
            ctr = data.get(sub, {})
            if ctr:
                top = sorted(ctr.items(), key=lambda x: -x[1])[:5]
                lines.append(f"    {sub}: {top}")
        return "\n".join(lines)

    # ── SUSTANTIVOS ──────────────────────────────────────────────────────
    if key == "nouns":
        lines = [
            f"  Sustantivos: {data.get('total_tokens', 0)} tokens, "
            f"{data.get('unique_count', 0)} lemas distintos."
        ]
        pbc = data.get("pos_by_cluster")
        if pbc:
            lines.append("  Por cluster (top 5 lemas):")
            lines.append(
                _render_pos_cluster_table("nouns", pbc, data.get("_active_cluster_ids"))
            )
        else:
            top = sorted(data.get("top_lemmas", {}).items(), key=lambda x: -x[1])[:5]
            lines.append(f"    Top global: {top}")
        return "\n".join(lines)

    # ── ADJETIVOS ────────────────────────────────────────────────────────
    if key == "adjectives":
        lines = [
            f"  Adjetivos: {data.get('total_tokens', 0)} tokens, "
            f"{data.get('unique_count', 0)} lemas distintos."
        ]
        pbc = data.get("pos_by_cluster")
        if pbc:
            lines.append("  Por cluster (top 5 lemas):")
            lines.append(
                _render_pos_cluster_table(
                    "adjectives", pbc, data.get("_active_cluster_ids")
                )
            )
        else:
            top = sorted(data.get("top_lemmas", {}).items(), key=lambda x: -x[1])[:5]
            lines.append(f"    Top global: {top}")
        return "\n".join(lines)

    # ── PREPOSICIONES ────────────────────────────────────────────────────
    if key == "prepositions":
        lines = [f"  Preposiciones: {data.get('total_tokens', 0)} tokens."]
        pbc = data.get("pos_by_cluster")
        if pbc:
            lines.append("  Por cluster (top 5 formas):")
            lines.append(
                _render_pos_cluster_table(
                    "prepositions", pbc, data.get("_active_cluster_ids")
                )
            )
        else:
            top = sorted(data.get("top_lemmas", {}).items(), key=lambda x: -x[1])[:5]
            lines.append(f"    Top global: {top}")
        return "\n".join(lines)

    # ── VERBOS LÉXICOS (de formas_tokens) ───────────────────────────────
    if key == "verbos_lexicos":
        lines = [
            f"  Verbos léxicos (formas_tokens): {data.get('total_tokens', 0)} tokens, "
            f"{data.get('unique_count', 0)} lemas distintos."
        ]
        pbc = data.get("pos_by_cluster")
        if pbc:
            lines.append("  Por cluster (top 5 lemas):")
            lines.append(
                _render_pos_cluster_table(
                    "verbos_lexicos", pbc, data.get("_active_cluster_ids")
                )
            )
        else:
            top = sorted(data.get("top_lemmas", {}).items(), key=lambda x: -x[1])[:5]
            lines.append(f"    Top global: {top}")
        return "\n".join(lines)

    # ── PRONOMBRES ───────────────────────────────────────────────────────
    if key == "pronombres":
        pro_drop = data.get("tipo", {}).get("NULO", 0)
        explicit = data.get("tipo", {}).get("EXPLICITO", 0)
        top_tipo = sorted(data.get("tipo", {}).items(), key=lambda x: -x[1])[:5]
        return (
            f"  Pronombres — tipo: {top_tipo}, "
            f"pro-drop: {pro_drop}, explícito: {explicit}"
        )

    # ── ADVERBIOS ────────────────────────────────────────────────────────
    if key == "adverbios":
        top_cat = sorted(data.get("categoria", {}).items(), key=lambda x: -x[1])[:5]
        return (
            f"  Adverbios — categorías: {top_cat}, "
            f"confianza baja: {data.get('confianza_baja', 0)}"
        )

    # ── MARCADORES ───────────────────────────────────────────────────────
    if key == "marcadores":
        top_cat = sorted(data.get("categoria", {}).items(), key=lambda x: -x[1])[:5]
        top_pos = sorted(data.get("posicion", {}).items(), key=lambda x: -x[1])[:3]
        return (
            f"  Marcadores discursivos — categorías: {top_cat}, "
            f"posición: {top_pos}, "
            f"surprisal transición: {data.get('mean_surprisal_transicion', 0):.2f}"
        )

    # ── NEGACIONES ───────────────────────────────────────────────────────
    if key == "negaciones":
        top = sorted(data.get("tipo", {}).items(), key=lambda x: -x[1])[:5]
        npi = data.get("con_npi", {}).get(True, data.get("con_npi", {}).get("True", 0))
        return f"  Negaciones — tipo: {top}, con NPI: {npi}"

    # ── CUANTIFICADORES ──────────────────────────────────────────────────
    if key == "cuantificadores":
        top = sorted(data.get("tipo", {}).items(), key=lambda x: -x[1])[:5]
        return f"  Cuantificadores — tipo: {top}"

    # ── INSUBORDINACIONES ────────────────────────────────────────────────
    if key == "insubordinaciones":
        fn = sorted(data.get("funcion", {}).items(), key=lambda x: -x[1])[:5]
        return f"  Insubordinaciones — función: {fn}"

    # ── COREFERENCIAS ────────────────────────────────────────────────────
    if key == "coref":
        return (
            f"  Coreferencias — cadenas: {data.get('total_chains', 0)}, "
            f"menciones: {data.get('total_mentions', 0)}, "
            f"media/UCE: {data.get('mean_mentions_per_uce', 0):.2f}, "
            f"entidades únicas: {len(data.get('unique_entities', []))}"
        )

    # ── PREDICADOS ───────────────────────────────────────────────────────
    if key == "predicados":
        top_v = sorted(data.get("top_verbs", {}).items(), key=lambda x: -x[1])[:5]
        top_o = sorted(data.get("top_objects", {}).items(), key=lambda x: -x[1])[:5]
        roles = sorted(data.get("thematic_roles", {}).items(), key=lambda x: -x[1])[:5]
        return (
            f"  Predicados — frames: {data.get('total_frames', 0)} "
            f"(base: {data.get('base_frames', 0)}, exp: {data.get('expansions', 0)}), "
            f"voz: {dict(sorted(data.get('voice_dist', {}).items(), key=lambda x: -x[1])[:3])}, "
            f"roles: {roles}, verbos top: {top_v}, objetos top: {top_o}"
        )

    # ── COMPLEJIDAD SINTÁCTICA ───────────────────────────────────────────
    if key == "complejidad_sintactica":
        return (
            f"  Complejidad sintáctica — "
            f"profundidad max media: {data.get('profundidad_maxima', 0):.2f}, "
            f"recursividad: {data.get('recursividad', 0):.2f}, "
            f"dist dependencia: {data.get('distancia_dependencia_media', 0):.2f}, "
            f"ratio sub: {data.get('ratio_subordinacion', 0):.3f}, "
            f"branching: {data.get('branching_ratio', 0):.3f}"
        )

    # ── MÉTRICAS LÉXICAS ─────────────────────────────────────────────────
    if key == "metricas_lexicas":
        return (
            f"  Métricas léxicas — TTR: {data.get('ttr', 0):.3f}, "
            f"Guiraud: {data.get('guiraud', 0):.3f}, "
            f"hapax ratio: {data.get('hapax_ratio', 0):.3f}, "
            f"div. semántica: {data.get('diversidad_semantica', 0):.3f}, "
            f"topic shift: {data.get('topic_shift', 0):.3f}"
        )

    # ── SUBTLEX ──────────────────────────────────────────────────────────
    if key == "subtlex":
        return (
            f"  SUBTLEX — Zipf: {data.get('mean_zipf', 0):.2f}, "
            f"sofisticación: {data.get('lexical_sophistication', 0):.2f}, "
            f"OOV: {data.get('pct_oov', 0):.1f}%, "
            f"baja freq: {data.get('pct_low_freq', 0):.1f}%, "
            f"oralidad: {data.get('oral_ratio', 0):.2f}, "
            f"academicismo: {data.get('academic_ratio', 0):.2f}, "
            f"tecnicismo: {data.get('domain_ratio', 0):.2f}"
        )

    # ── REGISTROS ────────────────────────────────────────────────────────
    if key == "registros":
        top = (
            sorted(data.items(), key=lambda x: -x[1])[:5]
            if isinstance(data, dict)
            else []
        )
        return f"  Registros — distribución: {top}"

    # Fallback: just dump it
    return f"  {key}: {str(data)[:200]}"


def build_agent_grammar_summary(
    gram_cats: List[str],
    structured_grammar: Dict,
    active_cluster_ids: Optional[Set] = None,
) -> str:
    """
    Return a compact, agent-specific grammar summary containing only the
    sections listed in gram_cats.

    Parameters
    ----------
    gram_cats : list[str]
        Category names declared in the agent config.
    structured_grammar : dict
        Full structured grammar dict from _summarize_global().
    active_cluster_ids : set, optional
        Cluster IDs present in the current document.  When given, the
        per-cluster POS tables are restricted to these IDs only.

    Returns
    -------
    str  — multi-line string ready for LLM injection.
    """
    if not gram_cats or not structured_grammar:
        return "  [Sin categorías gramaticales especificadas]"

    seen_keys: Set[str] = set()
    lines = ["=== RESUMEN GRAMATICAL (categorías relevantes para este agente) ==="]

    for f in ("n_uces", "n_tokens_total"):
        if f in structured_grammar:
            lines.append(f"  {f}: {structured_grammar[f]}")

    pos_by_cluster = structured_grammar.get("pos_by_cluster", {})

    for cat in gram_cats:
        canonical = _GRAM_CAT_ALIASES.get(cat.lower().strip())
        if canonical is None:
            lines.append(f"  [{cat}: categoría no reconocida]")
            continue
        if canonical in seen_keys:
            continue
        seen_keys.add(canonical)

        section_data = structured_grammar.get(canonical)

        # For POS sections, inject the cluster table and active filter
        # as transient keys so _render_section can use them without
        # mutating the canonical structured_grammar dict.
        if canonical in ("nouns", "adjectives", "prepositions", "verbos_lexicos"):
            section_data = dict(section_data or {})
            section_data["pos_by_cluster"] = pos_by_cluster
            if active_cluster_ids is not None:
                section_data["_active_cluster_ids"] = active_cluster_ids

        lines.append(_render_section(canonical, section_data))

    return "\n".join(lines)


# ─────────────────────────────────────────────
# POS token accumulator
# ─────────────────────────────────────────────
# Data sources (hard partition, no overlap):
#   uce.formas_tokens  → list of {forma, lemma, stem, pos, pos_idx}
#                        contains NOUN, PROPN, ADJ, VERB (lexical tokens only)
#   uce.marcadores     → list of {forma, lemma, pos}
#                        contains ADP, DET, CCONJ, SCONJ, PRON, AUX, ADV …
#
# Both are read unconditionally and funnelled into the same POS buckets.
# Defensive fallback strategies are kept for forward-compatibility in case
# the UCE model shape changes.


def _empty_pos_bucket() -> Dict:
    return {
        "total_tokens": 0,
        "unique_lemmas": set(),  # finalised to sorted list after aggregation
        "top_lemmas": Counter(),  # also used as top_forms for ADP
    }


def _accumulate_pos_from_uce(uce: UCE, agg: Dict) -> None:
    """
    Read uce.formas_tokens and uce.marcadores and accumulate counts into
    the global buckets in *agg* (nouns, adjectives, prepositions,
    verbos_lexicos) AND into the per-cluster bucket for this UCE's cluster_id.

    Global buckets:
        agg["nouns"], agg["adjectives"], agg["prepositions"],
        agg["verbos_lexicos"]

    Per-cluster structure (created on demand):
        agg["pos_by_cluster"][cluster_id][pos_bucket_name]  →  same shape as
        a global bucket, plus a running "n_uces" counter used to compute rates.
    """
    cluster_id = getattr(uce, "cluster_id", None)

    # Ensure the cluster entry exists
    if cluster_id is not None:
        if cluster_id not in agg["pos_by_cluster"]:
            agg["pos_by_cluster"][cluster_id] = {
                "n_uces": 0,
                "nouns": _empty_pos_bucket(),
                "adjectives": _empty_pos_bucket(),
                "prepositions": _empty_pos_bucket(),
                "verbos_lexicos": _empty_pos_bucket(),
            }
        agg["pos_by_cluster"][cluster_id]["n_uces"] += 1

    def _push(pos: str, lemma: str, forma: str) -> None:
        """Route one token into the correct global + cluster bucket."""
        pos_upper = (pos or "").upper()
        if pos_upper in ("NOUN", "PROPN"):
            _add_to_bucket(agg["nouns"], lemma)
            if cluster_id is not None:
                _add_to_bucket(agg["pos_by_cluster"][cluster_id]["nouns"], lemma)
        elif pos_upper == "ADJ":
            _add_to_bucket(agg["adjectives"], lemma)
            if cluster_id is not None:
                _add_to_bucket(agg["pos_by_cluster"][cluster_id]["adjectives"], lemma)
        elif pos_upper == "ADP":
            _add_to_bucket(agg["prepositions"], forma.lower())
            if cluster_id is not None:
                _add_to_bucket(
                    agg["pos_by_cluster"][cluster_id]["prepositions"], forma.lower()
                )
        elif pos_upper == "VERB":
            _add_to_bucket(agg["verbos_lexicos"], lemma)
            if cluster_id is not None:
                _add_to_bucket(
                    agg["pos_by_cluster"][cluster_id]["verbos_lexicos"], lemma
                )

    # ── Source 1: formas_tokens (NOUN, PROPN, ADJ, VERB) ─────────────────
    for tok in getattr(uce, "formas_tokens", []) or []:
        if not isinstance(tok, dict):
            continue
        _push(
            tok.get("pos", ""),
            tok.get("lemma", tok.get("forma", "")).lower(),
            tok.get("forma", ""),
        )

    # ── Source 2: marcadores raw list (ADP, DET, SCONJ, CCONJ, …) ────────
    # These items have {forma, lemma, pos} — no 'categoria' key.
    # The discourse-marker objects (with 'categoria') live in
    # uce.marcadores_discursivos and are handled separately by the main loop.
    for tok in getattr(uce, "marcadores", []) or []:
        if not isinstance(tok, dict):
            continue
        # Skip items that look like discourse-marker objects, not raw tokens
        if tok.get("categoria"):
            continue
        _push(
            tok.get("pos", ""),
            tok.get("lemma", tok.get("forma", "")).lower(),
            tok.get("forma", ""),
        )

    # ── Fallback: heuristic mining when both sources are absent ───────────
    formas_present = bool(getattr(uce, "formas_tokens", None))
    marcadores_present = bool(getattr(uce, "marcadores", None))
    if formas_present or marcadores_present:
        return  # at least one real source was available

    # Nouns from coreference mentions
    for ch in getattr(uce, "coref_chains", []):
        for mention in ch.get("mentions", []):
            text = mention.get("text", "").lower().strip()
            if text:
                _add_to_bucket(agg["nouns"], text)
                if cluster_id is not None:
                    _add_to_bucket(agg["pos_by_cluster"][cluster_id]["nouns"], text)

    # Prepositions from subordination conjunction fields
    for v in getattr(uce, "verbos", []):
        conj = v.get("conjuncion_subordinante", "")
        if conj and conj not in ("NINGUNA", ""):
            _add_to_bucket(agg["prepositions"], conj.lower())
            if cluster_id is not None:
                _add_to_bucket(
                    agg["pos_by_cluster"][cluster_id]["prepositions"], conj.lower()
                )


def _add_to_bucket(bucket: Dict, key: str) -> None:
    """Increment counts in a pos bucket for *key* (lemma or surface form)."""
    bucket["total_tokens"] += 1
    bucket["unique_lemmas"].add(key)
    bucket["top_lemmas"][key] += 1


def _finalise_pos_bucket(bucket: Dict) -> None:
    """Convert mutable sets/Counters to JSON-safe forms in-place."""
    bucket["unique_count"] = len(bucket["unique_lemmas"])
    bucket["unique_lemmas"] = sorted(bucket["unique_lemmas"])
    # top_lemmas stays as Counter so .most_common() works during rendering;
    # _counter_to_serialisable handles it at save time.


def _render_pos_cluster_table(
    pos_key: str,
    pos_by_cluster: Dict,
    active_cluster_ids: Optional[Set] = None,
    top_n: int = 5,
) -> str:
    """
    Render a compact per-cluster table for one POS bucket (nouns /
    adjectives / prepositions / verbos_lexicos).

    active_cluster_ids — if given, only those clusters are shown.
    """
    if not pos_by_cluster:
        return f"    [sin datos por cluster para {pos_key}]"

    cluster_ids = sorted(pos_by_cluster.keys())
    if active_cluster_ids is not None:
        cluster_ids = [c for c in cluster_ids if c in active_cluster_ids]
    if not cluster_ids:
        return f"    [ningún cluster activo para {pos_key}]"

    lines = []
    for cid in cluster_ids:
        cdata = pos_by_cluster[cid]
        bucket = cdata.get(pos_key, {})
        n_uces = cdata.get("n_uces", 1) or 1
        total = bucket.get("total_tokens", 0)
        rate = total / n_uces
        top = sorted(bucket.get("top_lemmas", {}).items(), key=lambda x: -x[1])[:top_n]
        lines.append(
            f"    cluster {cid} ({n_uces} UCEs): "
            f"{total} tokens ({rate:.1f}/UCE) — top: {top}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Cliente DeepSeek
# ─────────────────────────────────────────────


def call_deepseek_structured(
    messages: List[Dict[str, str]],
    schema: Dict,
    max_retries: int = 3,
    temperature: float = 0.2,
) -> Dict:
    if not isinstance(schema, dict) or "type" not in schema:
        print("❌ Esquema JSON inválido. Usando fallback vacío.")
        schema = {
            "type": "object",
            "properties": {
                "annotations": {"type": "array", "items": {"type": "object"}}
            },
            "required": ["annotations"],
        }

    tool_def = {
        "type": "function",
        "function": {
            "name": "emit_annotations",
            "description": "Devuelve anotaciones estrictamente bajo este esquema.",
            "parameters": schema,
        },
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    original_messages = messages.copy()

    for attempt in range(max_retries + 1):
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "tools": [tool_def],
            "tool_choice": {
                "type": "function",
                "function": {"name": "emit_annotations"},
            },
            #  "response_format": {"type": "json_object"},
            "temperature": temperature,
        }

        try:
            resp = requests.post(
                DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120
            )
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error de red: {e}. Reintentando en {2**attempt}s...")
            time.sleep(2**attempt)
            continue

        if resp.status_code == 429:
            wait = 2**attempt
            print(f"⏳ Rate limit. Esperando {wait}s...")
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            print(f"⚠️ Error HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code == 400 and "tool_choice" in resp.text.lower():
                print("   → Cambiando tool_choice a 'auto'...")
                payload["tool_choice"] = "auto"
                continue
            time.sleep(2)
            continue

        data = resp.json()
        msg = data["choices"][0].get("message", {})
        tool_calls = msg.get("tool_calls")

        if tool_calls:
            json_str = tool_calls[0]["function"]["arguments"]
        else:
            json_str = msg.get("content", "{}")

        # ─── Parse & Repair ─────────────────────────────────────────────
        parsed = None
        error_message = None

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON malformado (intento {attempt + 1}): {e}")
            error_message = f"JSON syntax error: {e}"
            try:
                repaired = repair_json(json_str)
                parsed = json.loads(repaired)
                print("   ✅ JSON reparado exitosamente.")
                error_message = None  # Clear error if repair succeeded
            except Exception as repair_err:
                print(f"   ❌ La reparación falló: {repair_err}")
                error_message = f"JSON repair failed: {repair_err}"

        if parsed is None:
            if attempt < max_retries:
                # Build tool response messages
                tool_msgs = []
                for tc in tool_calls or []:
                    tool_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"Error: {error_message}. PPlease correct the JSON output to match the schema.",
                        }
                    )
                messages = (
                    original_messages
                    + [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        },
                    ]
                    + tool_msgs
                )
                print("   🔁 Solicitando corrección al modelo (vía tool message)...")
                continue
            else:
                print("❌ Máximo de reintentos alcanzado. Devolviendo vacío.")
                return {"annotations": []}

        if "annotations" in parsed:
            for ann in parsed["annotations"]:
                ann.setdefault("confidence", "baja")
                if not ann.get("spans"):
                    ann["spans"] = []

        # ─── Schema Validation ─────────────────────────────────────────
        try:
            validate(instance=parsed, schema=schema)
            return parsed  # Success!
        except ValidationError as val_err:
            print(
                f"⚠️ JSON válido pero no cumple el esquema (intento {attempt + 1}): {val_err.message}"
            )
            if attempt < max_retries:
                tool_msgs = []
                for tc in tool_calls or []:
                    tool_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"Schema validation error: {val_err.message}. Please correct the output to match the schema.",
                        }
                    )
                messages = (
                    original_messages
                    + [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        },
                    ]
                    + tool_msgs
                )
                print(
                    "   🔁 Solicitando corrección de esquema al modelo (vía tool message)..."
                )
                continue
            else:
                print("❌ Máximo de reintentos alcanzado. Devolviendo vacío.")
                return {"annotations": []}

    return {"annotations": []}


# ─────────────────────────────────────────────
# Agente Base
# ─────────────────────────────────────────────


class BaseAgent:
    def __init__(self, config: Dict):
        self.name = config.get("name")
        self.role = config["role"]
        self.instructions = config["instructions"]
        self.json_schema = config["json_schema"]
        self.few_shot_examples = config.get("few_shot_examples", [])
        self.gram_cats = config.get("gram_cats", [])
        self.disc_cats = config.get("disc_cats", [])
        self._clean_schema_required()

    def _clean_schema_required(self):
        """
        Modifica self.json_schema para que solo se exijan 'spans' y 'confidence'
        en los objetos de anotación. Elimina cualquier otro campo del array 'required'.
        """
        try:
            # Navegar hasta el nivel de items de annotations
            annotations_props = (
                self.json_schema.get("properties", {})
                .get("annotations", {})
                .get("items", {})
            )
            if "required" in annotations_props:
                # Conservar solo spans y confidence si están presentes
                original_required = annotations_props["required"]
                new_required = [
                    f for f in original_required if f in ("spans", "confidence")
                ]
                annotations_props["required"] = new_required
        except Exception:
            # Si la estructura no es la esperada, no hacemos nada
            pass

    def _build_prompt(
        self,
        tagged_text: str,
        grammatical_summary: str,  # now always agent-specific
        previous_results: List[DiscourseAnnotation],
    ) -> str:
        few_shot_text = "".join(
            f"\nEntrada:\n{json.dumps(ex['input'], ensure_ascii=False)}"
            f"\nSalida:\n{json.dumps(ex['output'], ensure_ascii=False)}\n"
            for ex in self.few_shot_examples
        )
        prev_text = "## Anotaciones previas\nNinguna.\n"
        if previous_results:
            prev_text = "## Anotaciones discursivas previas (referencia cruzada)\n"
            for ann in previous_results[:30]:
                agent_name = ann.agent or ann.trait

                # Extract data from the new spans list, fallback to legacy if empty
                if ann.spans:
                    locs = ", ".join([f"[{s.uce_id}]" for s in ann.spans if s.uce_id])
                    main_quote = ann.spans[0].quote or ""
                else:
                    locs = f"[{ann.uce_id}]" if ann.uce_id else "[?]"
                    main_quote = ann.quote or ""

                locs = locs or "[?]"

                # Strip newlines so we don't break the markdown list formatting
                clean_quote = main_quote.replace("\n", " ").strip()
                short_quote = clean_quote[:80] + (
                    "..." if len(clean_quote) > 80 else ""
                )

                prev_text += (
                    f"- {locs} **{agent_name}**: '{short_quote}' → {ann.subtype}\n"
                )

        return f"""## Rol
{self.role}

## Resumen gramatical del corpus (categorías relevantes para este agente)
{grammatical_summary}

## Texto del corpus (con identificadores de UCE)
Cada segmento empieza con [uce_id]. Usa ese id en el campo `uce_id` de cada anotación.
```
{tagged_text}
```

{prev_text}

## Instrucciones
{self.instructions}
- Devuelve SIEMPRE el campo `uce_id` indicando a qué segmento pertenece la anotación.
- El campo `quote` debe ser una subcadena EXACTA del texto del segmento correspondiente. Añade un contexto al texto extraido (una palabra antes y una después)
- El campo `quote` debe ser una subcadena **EXACTA** del texto, sin añadir puntos suspensivos (...) al inicio o al final. Si la cita es incompleta, corta palabra exacta, no uses "...". Ejemplo correcto: "situación era" en lugar de "...situación era".
- Si una misma estrategia discursiva aparece en varias UCEs, crea UNA sola anotación con múltiples `spans`. Cada span debe tener `uce_id` y `quote`.
- NO agrupes varias ocurrencias independientes en una sola anotación.
- Si la estrategia se manifiesta en una sola UCE, usa un único `span`.

## Ejemplos
{few_shot_text}

Devuelve ÚNICAMENTE JSON válido según el esquema. Si no hay nada, array vacío.
"""

    def analyze_corpus(
        self,
        tagged_text: str,
        grammatical_summary: str,
        previous_results: List[DiscourseAnnotation] = None,
    ) -> List[DiscourseAnnotation]:
        previous_results = previous_results or []
        messages = [
            {
                "role": "system",
                "content": """
                Eres un lingüista computacional estricto.
                You must output strictly valid JSON.
                - No trailing commas.
                - Escape all double quotes inside strings with \".
                - Do not include any text outside the JSON object.
                - If you need to write a literal backslash, double it \\\\.
                """,
            },
            {
                "role": "user",
                "content": self._build_prompt(
                    tagged_text, grammatical_summary, previous_results
                ),
            },
        ]
        try:
            response = call_deepseek_structured(messages, self.json_schema)
            response = self._parse_response(response.get("annotations", []))
            print(response)
            return response

        except Exception as e:
            print(f"❌ Agente {self.name} falló: {e}")
            return []

    def _parse_response(self, data: List[Dict]) -> List[DiscourseAnnotation]:
        if not isinstance(data, list):
            print(f"⚠️ Respuesta no es una lista: {type(data)}. Se devuelve vacío.")
            return []

        annos = []
        for item in data:
            if not isinstance(item, dict):
                continue

            # Subtype flexible: primero "subtype", luego "type", sino "otro"
            subtype = item.get("subtype") or item.get("type") or "otro"

            conf_str = item.get("confidence", "baja").lower()
            if conf_str not in ("alta", "media", "baja"):
                conf_str = "baja"

            # Metadatos: todo lo que no sea 'spans', 'confidence', 'subtype', 'type', 'trait'
            metadata = {
                k: v
                for k, v in item.items()
                if k not in ("spans", "confidence", "subtype", "type", "trait")
            }

            spans = []
            for s in item.get("spans", []):
                if isinstance(s, dict):
                    spans.append(
                        SpanInfo(uce_id=s.get("uce_id", ""), quote=s.get("quote", ""))
                    )

            annos.append(
                DiscourseAnnotation(
                    trait=self.name,
                    agent=self.name,
                    subtype=subtype,
                    spans=spans,
                    confidence=Confidence(conf_str),
                    metadata=metadata,
                )
            )
        return annos


class GlobalGrammaticalSummarizer:
    def __init__(self, config: Dict):
        self.config = config

    def summarize(self, uces: List[UCE], by_cluster: bool = False) -> Tuple[str, Dict]:
        if by_cluster and any(uce.cluster_id is not None for uce in uces):
            return self._summarize_by_cluster(uces)
        return self._summarize_global(uces)

    def _summarize_global(self, uces: List[UCE]) -> Tuple[str, Dict]:
        agg: Dict[str, Any] = {
            "n_uces": len(uces),
            "n_tokens_total": 0,
            "verbos": {
                "modo": Counter(),
                "tiempo": Counter(),
                "aspecto": Counter(),
                "voz": Counter(),
                "persona": Counter(),
                "numero": Counter(),
                "genero": Counter(),
                "aux_tipo": Counter(),
                "perifrasis": Counter(),
                "subordinacion_tipo": Counter(),
                "subordinacion_conjuncion": Counter(),
                "concordancia": Counter(),
            },
            "pronombres": {
                "tipo": Counter(),
                "subtipo": Counter(),
                "persona": Counter(),
                "numero": Counter(),
                "genero": Counter(),
                "es_referencial": Counter(),
            },
            "negaciones": {
                "tipo": Counter(),
                "con_npi": Counter(),
                "es_constituyente": Counter(),
            },
            "cuantificadores": {"tipo": Counter(), "pos": Counter()},
            "adverbios": {
                "categoria": Counter(),
                "confianza_baja": 0,
                "multipalabra": Counter(),
            },
            "marcadores": {
                "categoria": Counter(),
                "posicion": Counter(),
                "surprisal_transicion": [],
                "surprisal_interno": [],
            },
            "insubordinaciones": {"funcion": Counter(), "tipo": Counter()},
            "rarezas": {"tipo": Counter()},
            "coref": {
                "total_chains": 0,
                "total_mentions": 0,
                "mentions_per_uce": [],
                "unique_entities": set(),
            },
            "predicados": {
                "total_frames": 0,
                "base_frames": 0,
                "expansions": 0,
                "cluster_dist": Counter(),
                "thematic_roles": Counter(),
                "negated_frames": 0,
                "voice_dist": Counter(),
                "top_verbs": Counter(),
                "top_objects": Counter(),
            },
            "complejidad_sintactica": {
                k: []
                for k in [
                    "profundidad_maxima",
                    "recursividad",
                    "distancia_dependencia_media",
                    "ratio_subordinacion",
                    "branching_ratio",
                ]
            },
            "metricas_lexicas": {
                k: []
                for k in [
                    "ttr",
                    "guiraud",
                    "hapax_ratio",
                    "diversidad_semantica",
                    "topic_shift",
                ]
            },
            "subtlex": {
                k: []
                for k in [
                    "mean_zipf",
                    "lexical_sophistication",
                    "pct_oov",
                    "pct_low_freq",
                    "oral_ratio",
                    "academic_ratio",
                    "domain_ratio",
                    "mean_surprisal",
                ]
            },
            "registros": Counter(),
            # ── NEW: noun / adj / prep / verb-lex buckets ────────────────
            "nouns": _empty_pos_bucket(),
            "adjectives": _empty_pos_bucket(),
            "prepositions": _empty_pos_bucket(),
            "verbos_lexicos": _empty_pos_bucket(),
            # per-cluster POS breakdown (populated by _accumulate_pos_from_uce)
            "pos_by_cluster": {},
        }

        for uce in uces:
            agg["n_tokens_total"] += uce.metricas_lexicas.get("num_tokens", 0)

            for v in uce.verbos:
                agg["verbos"]["modo"][v.get("modo", "DESCONOCIDO")] += 1
                agg["verbos"]["tiempo"][v.get("tiempo", "DESCONOCIDO")] += 1
                agg["verbos"]["aspecto"][v.get("aspecto", "DESCONOCIDO")] += 1
                agg["verbos"]["voz"][v.get("voz", "DESCONOCIDO")] += 1
                agg["verbos"]["persona"][v.get("persona", "DESCONOCIDO")] += 1
                agg["verbos"]["numero"][v.get("numero", "DESCONOCIDO")] += 1
                agg["verbos"]["genero"][v.get("genero", "DESCONOCIDO")] += 1
                agg["verbos"]["aux_tipo"][v.get("aux_tipo", "NINGUNO")] += 1
                agg["verbos"]["perifrasis"][str(v.get("perifrasis", False))] += 1
                agg["verbos"]["subordinacion_tipo"][
                    v.get("tipo_subordinacion", "NINGUNA")
                ] += 1
                agg["verbos"]["subordinacion_conjuncion"][
                    v.get("conjuncion_subordinante", "NINGUNA")
                ] += 1
                if v.get("concordancia"):
                    agg["verbos"]["concordancia"]["correcta"] += 1
                else:
                    agg["verbos"]["concordancia"]["incorrecta"] += 1

            for p in uce.pronombres:
                agg["pronombres"]["tipo"][p.get("tipo", "OTRO")] += 1
                subtipo = p.get("subtipo")
                if subtipo:
                    agg["pronombres"]["subtipo"][subtipo] += 1
                agg["pronombres"]["persona"][p.get("persona", "_")] += 1
                agg["pronombres"]["numero"][p.get("numero", "_")] += 1
                agg["pronombres"]["genero"][p.get("genero", "_")] += 1
                agg["pronombres"]["es_referencial"][
                    str(p.get("es_referencial", False))
                ] += 1

            for n in uce.negaciones:
                agg["negaciones"]["tipo"][n.get("tipo", "DESCONOCIDO")] += 1
                if n.get("npis"):
                    agg["negaciones"]["con_npi"][True] += 1
                else:
                    agg["negaciones"]["con_npi"][False] += 1
                agg["negaciones"]["es_constituyente"][
                    str(n.get("es_constituyente", False))
                ] += 1

            for q in uce.cuantificadores:
                agg["cuantificadores"]["tipo"][q.get("tipo", "DESCONOCIDO")] += 1
                agg["cuantificadores"]["pos"][q.get("pos", "DESCONOCIDO")] += 1

            for a in uce.adverbios:
                agg["adverbios"]["categoria"][a.get("categoria", "DESCONOCIDO")] += 1
                if a.get("confianza", 1.0) < 0.6:
                    agg["adverbios"]["confianza_baja"] += 1
                agg["adverbios"]["multipalabra"][
                    str(a.get("es_multipalabra", False))
                ] += 1

            for m in uce.marcadores_discursivos:
                agg["marcadores"]["categoria"][m.get("categoria", "DESCONOCIDO")] += 1
                agg["marcadores"]["posicion"][m.get("posicion", "DESCONOCIDO")] += 1
                if "surprisal_transicion" in m:
                    agg["marcadores"]["surprisal_transicion"].append(
                        m["surprisal_transicion"]
                    )
                if "surprisal_interno" in m:
                    agg["marcadores"]["surprisal_interno"].append(
                        m["surprisal_interno"]
                    )

            for i in uce.insubordinaciones:
                agg["insubordinaciones"]["funcion"][
                    i.get("funcion_pragmatica", "DESCONOCIDA")
                ] += 1
                agg["insubordinaciones"]["tipo"][i.get("tipo", "INSUBORDINACION")] += 1

            for r in uce.rarezas:
                agg["rarezas"]["tipo"][r.get("tipo", "DESCONOCIDO")] += 1

            agg["coref"]["total_chains"] += len(uce.coref_chains)
            mentions_in_uce = sum(
                len(ch.get("mentions", [])) for ch in uce.coref_chains
            )
            agg["coref"]["total_mentions"] += mentions_in_uce
            agg["coref"]["mentions_per_uce"].append(mentions_in_uce)
            for ch in uce.coref_chains:
                agg["coref"]["unique_entities"].add(ch.get("representative", ""))

            if hasattr(uce, "predicate_frames") and uce.predicate_frames:
                agg["predicados"]["total_frames"] += len(uce.predicate_frames)
                for f in uce.predicate_frames:
                    if isinstance(f, dict):
                        f = PredicateFrame.from_dict(f)
                    if getattr(f, "is_expansion", False):
                        agg["predicados"]["expansions"] += 1
                    else:
                        agg["predicados"]["base_frames"] += 1
                    agg["predicados"]["cluster_dist"][getattr(f, "cluster_id", -1)] += 1
                    agg["predicados"]["thematic_roles"][
                        getattr(f, "thematic_role", "UNKNOWN")
                    ] += 1
                    if getattr(f, "negated", False):
                        agg["predicados"]["negated_frames"] += 1
                    agg["predicados"]["voice_dist"][getattr(f, "voice", "Act")] += 1
                    agg["predicados"]["top_verbs"][getattr(f, "verb_lemma", "")] += 1
                    if getattr(f, "direct_object_lemma", None):
                        agg["predicados"]["top_objects"][f.direct_object_lemma] += 1

            cs = uce.complejidad_sintactica
            for k in agg["complejidad_sintactica"]:
                agg["complejidad_sintactica"][k].append(cs.get(k, 0))

            ml = uce.metricas_lexicas
            agg["metricas_lexicas"]["ttr"].append(ml.get("ttr", 0))
            agg["metricas_lexicas"]["guiraud"].append(ml.get("guiraud", 0))
            agg["metricas_lexicas"]["hapax_ratio"].append(ml.get("hapax_ratio", 0))
            agg["metricas_lexicas"]["diversidad_semantica"].append(
                uce.diversidad_semantica
            )
            agg["metricas_lexicas"]["topic_shift"].append(uce.topic_shift_prev)

            for k, f in [
                ("mean_zipf", "mean_zipf"),
                ("lexical_sophistication", "lexical_sophistication"),
                ("pct_oov", "pct_oov"),
                ("pct_low_freq", "pct_low_freq"),
                ("oral_ratio", "oral_ratio"),
                ("academic_ratio", "academic_ratio"),
                ("domain_ratio", "domain_specific_ratio"),
                ("mean_surprisal", "mean_surprisal_content"),
            ]:
                agg["subtlex"][k].append(ml.get(f, 0.0))

            if uce.registro:
                agg["registros"][uce.registro] += 1

            # ── NEW: POS-based token accumulation ────────────────────────
            _accumulate_pos_from_uce(uce, agg)

        # ── Finalise numeric aggregates ───────────────────────────────────
        for section in ("complejidad_sintactica", "metricas_lexicas", "subtlex"):
            for k, vals in agg[section].items():
                agg[section][k] = float(np.mean(vals)) if vals else 0.0

        agg["coref"]["unique_entities"] = list(agg["coref"]["unique_entities"])
        if agg["coref"]["mentions_per_uce"]:
            agg["coref"]["mean_mentions_per_uce"] = float(
                np.mean(agg["coref"]["mentions_per_uce"])
            )
            agg["coref"]["std_mentions_per_uce"] = float(
                np.std(agg["coref"]["mentions_per_uce"])
            )
        else:
            agg["coref"]["mean_mentions_per_uce"] = 0.0
            agg["coref"]["std_mentions_per_uce"] = 0.0

        for key in ["surprisal_transicion", "surprisal_interno"]:
            vals = agg["marcadores"][key]
            agg["marcadores"][f"mean_{key}"] = float(np.mean(vals)) if vals else 0.0
            agg["marcadores"][f"std_{key}"] = float(np.std(vals)) if vals else 0.0
            del agg["marcadores"][key]

        # ── Finalise POS buckets ──────────────────────────────────────────
        for bkey in ("nouns", "adjectives", "prepositions", "verbos_lexicos"):
            _finalise_pos_bucket(agg[bkey])
        for cid, cdata in agg["pos_by_cluster"].items():
            for bkey in ("nouns", "adjectives", "prepositions", "verbos_lexicos"):
                _finalise_pos_bucket(cdata[bkey])

        return self._build_summary_text(agg), agg

    def _summarize_by_cluster(self, uces: List[UCE]) -> Tuple[str, Dict]:
        clusters = defaultdict(list)
        for uce in uces:
            if uce.cluster_id is not None:
                clusters[uce.cluster_id].append(uce)

        per_cluster = {}
        for cid, cluster_uces in clusters.items():
            _, agg = self._summarize_global(cluster_uces)
            per_cluster[str(cid)] = agg

        lines = [f"Resumen por cluster (total clusters: {len(clusters)})"]
        for cid, agg in per_cluster.items():
            lines.append(f"\n--- Cluster {cid} ---")
            lines.append(f"UCEs: {agg['n_uces']} | Tokens: {agg['n_tokens_total']}")
            lines.append(
                f"Registro predominante: "
                f"{agg['registros'].most_common(1)[0][0] if hasattr(agg['registros'], 'most_common') and agg['registros'] else 'N/A'}"
            )
            lines.append(
                f"TTR medio: {agg['metricas_lexicas']['ttr']:.3f} | "
                f"Guiraud: {agg['metricas_lexicas']['guiraud']:.3f}"
            )
        return "\n".join(lines), per_cluster

    def _build_summary_text(self, agg: Dict) -> str:
        lines = [
            "=== RESUMEN GRAMATICAL GLOBAL ===",
            f"UCEs: {agg['n_uces']} | Tokens totales: {agg['n_tokens_total']}",
            "",
            "--- VERBOS ---",
            f"  Modo: {dict(agg['verbos']['modo'].most_common(5))}",
            f"  Tiempo: {dict(agg['verbos']['tiempo'].most_common(5))}",
            f"  Aspecto: {dict(agg['verbos']['aspecto'].most_common())}",
            f"  Voz: {dict(agg['verbos']['voz'].most_common())}",
            f"  Subordinación: {dict(agg['verbos']['subordinacion_tipo'].most_common())}",
            "",
            "--- PRONOMBRES ---",
            f"  Tipo: {dict(agg['pronombres']['tipo'].most_common())}",
            f"  Subtipo: {dict(agg['pronombres']['subtipo'].most_common(5))}",
            f"  Pro-drop vs explícito: {agg['pronombres']['tipo'].get('NULO', 0)} / {agg['pronombres']['tipo'].get('EXPLICITO', 0)}",
            "",
            "--- NEGACIONES ---",
            f"  Tipo: {dict(agg['negaciones']['tipo'].most_common())}",
            f"  Con NPI: {agg['negaciones']['con_npi'].get(True, 0)}",
            "",
            "--- ADVERBIOS ---",
            f"  Categorías: {dict(agg['adverbios']['categoria'].most_common(5))}",
            f"  Confianza baja (<0.6): {agg['adverbios']['confianza_baja']}",
            "",
            "--- MARCADORES DISCURSIVOS ---",
            f"  Categorías: {dict(agg['marcadores']['categoria'].most_common(5))}",
            f"  Posición: {dict(agg['marcadores']['posicion'].most_common())}",
            f"  Surprisal transición medio: {agg['marcadores']['mean_surprisal_transicion']:.2f}",
            "",
            "--- INSUBORDINACIONES ---",
            f"  Funciones: {dict(agg['insubordinaciones']['funcion'].most_common())}",
            "",
            "--- RAREZAS ---",
            f"  Tipos: {dict(agg['rarezas']['tipo'].most_common(5))}",
            "",
            "--- SUSTANTIVOS ---",
            f"  Tokens: {agg['nouns']['total_tokens']} | Lemas distintos: {agg['nouns']['unique_count']}",
            f"  Top lemas: {sorted(agg['nouns']['top_lemmas'].items(), key=lambda x: -x[1])[:10]}",
            "",
            "--- ADJETIVOS ---",
            f"  Tokens: {agg['adjectives']['total_tokens']} | Lemas distintos: {agg['adjectives']['unique_count']}",
            f"  Top lemas: {sorted(agg['adjectives']['top_lemmas'].items(), key=lambda x: -x[1])[:10]}",
            "",
            "--- PREPOSICIONES ---",
            f"  Tokens: {agg['prepositions']['total_tokens']}",
            f"  Top formas: {sorted(agg['prepositions']['top_lemmas'].items(), key=lambda x: -x[1])[:10]}",
            "",
            "--- VERBOS LÉXICOS (formas_tokens) ---",
            f"  Tokens: {agg['verbos_lexicos']['total_tokens']} | Lemas distintos: {agg['verbos_lexicos']['unique_count']}",
            f"  Top lemas: {sorted(agg['verbos_lexicos']['top_lemmas'].items(), key=lambda x: -x[1])[:10]}",
            "",
            f"  Clusters con datos POS: {sorted(agg['pos_by_cluster'].keys())}",
            "",
            "--- COREFERENCIAS ---",
            f"  Cadenas totales: {agg['coref']['total_chains']}",
            f"  Menciones totales: {agg['coref']['total_mentions']}",
            f"  Menciones media por UCE: {agg['coref']['mean_mentions_per_uce']:.2f}",
            f"  Entidades únicas: {len(agg['coref']['unique_entities'])}",
            "",
            "--- PREDICADOS (sujeto-verbo-objeto) ---",
            f"  Total frames: {agg['predicados']['total_frames']} (base: {agg['predicados']['base_frames']}, expansiones: {agg['predicados']['expansions']})",
            f"  Roles temáticos: {dict(agg['predicados']['thematic_roles'].most_common())}",
            f"  Verbos más frecuentes: {agg['predicados']['top_verbs'].most_common(5)}",
            f"  Objetos directos más frecuentes: {agg['predicados']['top_objects'].most_common(5)}",
            "",
            "--- COMPLEJIDAD SINTÁCTICA ---",
            f"  Profundidad máxima media: {agg['complejidad_sintactica']['profundidad_maxima']:.2f}",
            f"  Recursividad media: {agg['complejidad_sintactica']['recursividad']:.2f}",
            f"  Distancia dependencia media: {agg['complejidad_sintactica']['distancia_dependencia_media']:.2f}",
            f"  Ratio subordinación: {agg['complejidad_sintactica']['ratio_subordinacion']:.3f}",
            f"  Branching ratio: {agg['complejidad_sintactica']['branching_ratio']:.3f}",
            "",
            "--- MÉTRICAS LÉXICAS ---",
            f"  TTR medio: {agg['metricas_lexicas']['ttr']:.3f}",
            f"  Guiraud medio: {agg['metricas_lexicas']['guiraud']:.3f}",
            f"  Hapax ratio medio: {agg['metricas_lexicas']['hapax_ratio']:.3f}",
            f"  Diversidad semántica media: {agg['metricas_lexicas']['diversidad_semantica']:.3f}",
            f"  Topic shift medio: {agg['metricas_lexicas']['topic_shift']:.3f}",
            "",
            "--- PERFIL SUBTLEX ---",
            f"  Zipf medio: {agg['subtlex']['mean_zipf']:.2f}",
            f"  Sofisticación léxica: {agg['subtlex']['lexical_sophistication']:.2f}",
            f"  % OOV: {agg['subtlex']['pct_oov']:.1f}%",
            f"  % baja frecuencia: {agg['subtlex']['pct_low_freq']:.1f}%",
            f"  Oralidad: {agg['subtlex']['oral_ratio']:.2f}",
            f"  Academicismo: {agg['subtlex']['academic_ratio']:.2f}",
            f"  Tecnicismo: {agg['subtlex']['domain_ratio']:.2f}",
            f"  Surprisal medio: {agg['subtlex']['mean_surprisal']:.2f}",
            "",
            "--- REGISTROS ---",
            f"  Distribución: {dict(agg['registros'].most_common()) if hasattr(agg['registros'], 'most_common') else agg['registros']}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# DiscourseMultiAgent
# ─────────────────────────────────────────────
class DiscourseMultiAgent:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            configs = json.load(f)
        self.agents: Dict[str, BaseAgent] = {}
        self.disc_dependencies: Dict[str, Set[str]] = {}
        for key, cfg in configs.items():
            if "name" not in cfg:
                cfg["name"] = key
            if cfg["name"] != "grammatical_summarizer":
                self.agents[cfg["name"]] = BaseAgent(cfg)
                self.disc_dependencies[cfg["name"]] = set(cfg.get("disc_cats", []))
        self.disc_execution_order = self._topological_sort()

    def _topological_sort(self) -> List[str]:
        graph = {n: set(d) for n, d in self.disc_dependencies.items()}
        in_degree = {n: len(d) for n, d in graph.items()}
        queue = [n for n, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for other, deps in graph.items():
                if node in deps:
                    deps.remove(node)
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)
        if len(order) != len(self.disc_dependencies):
            raise ValueError("Bucle en dependencias discursivas. Revisa tu config.")
        return order

    def analyze_global(
        self,
        tagged_text: str,
        grammar_summary: str,
        existing_annotations: List[DiscourseAnnotation] = None,
    ) -> List[DiscourseAnnotation]:
        all_annos = list(existing_annotations or [])
        agent_results: Dict[str, List[DiscourseAnnotation]] = {}
        for agent_name in self.disc_execution_order:
            agent = self.agents[agent_name]
            prev = []
            for dep in agent.disc_cats:
                prev.extend(agent_results.get(dep, []))
            annos = agent.analyze_corpus(tagged_text, grammar_summary, prev)
            agent_results[agent_name] = annos
            all_annos.extend(annos)
        return all_annos


# ─────────────────────────────────────────────
# Mapeo y fuzzy-matching de quotes
# ─────────────────────────────────────────────
def _clean_quote_for_matching(quote: str) -> str:
    """Elimina puntos suspensivos y espacios sobrantes al inicio/final de una cita."""
    if not quote:
        return ""
    cleaned = quote.strip().strip('"“”‘’')
    # Eliminar ... o … al inicio
    cleaned = re.sub(r"^\.{3,}\s*", "", cleaned)
    cleaned = re.sub(r"^\s*…\s*", "", cleaned)
    # Eliminar ... o … al final
    cleaned = re.sub(r"\s*\.{3,}$", "", cleaned)
    cleaned = re.sub(r"\s*…\s*$", "", cleaned)
    return cleaned.strip()


def _resolve_quote_in_text(
    quote: str, text: str, threshold: int = 82
) -> Tuple[int, int]:
    if not quote or not text:
        return -1, -1

    # Clean the quote first
    cleaned_quote = _clean_quote_for_matching(quote)

    # Exact match with cleaned quote
    pos = text.find(cleaned_quote)
    if pos != -1:
        return pos, pos + len(cleaned_quote)

    # Fallback to original quote
    pos = text.find(quote)
    if pos != -1:
        return pos, pos + len(quote)

    # Regex with cleaned quote (allow variable whitespace)
    pattern = re.escape(cleaned_quote).replace(r"\ ", r"\s+")
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.start(), m.end()

    # Fuzzy matching (using cleaned quote)
    if HAS_RAPIDFUZZ and len(cleaned_quote) >= 8:
        q_len = len(quote)
        step = max(1, q_len // 4)
        best_score, best_start, best_end = 0, -1, -1
        for window in (q_len, int(q_len * 1.15), int(q_len * 0.85)):
            for i in range(0, len(text) - window + 1, step):
                candidate = text[i : i + window]
                score = fuzz.ratio(cleaned_quote, candidate)
                if score > best_score:
                    best_score, best_start, best_end = score, i, i + window
        if best_score >= threshold:
            return best_start, best_end
    return -1, -1


# ─────────────────────────────────────────────
# Persistencia de estado
# ─────────────────────────────────────────────
def _load_state(path: Path) -> Dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(path: Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    # structured_grammar may still contain Counter / set — serialise first
    serialisable = _counter_to_serialisable(state)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)
    print(f"💾 Estado guardado en {path}")


# ─────────────────────────────────────────────
# DebugOrchestrator
# ─────────────────────────────────────────────
class DebugOrchestrator:
    MAX_TAGGED_CHARS = 300_000

    def __init__(
        self,
        grammar_config_path: str,
        discourse_config_path: str,
        workflow_data_path: str,
        state_path: str = str(STATE_FILE),
    ):
        self.state_path = Path(state_path)
        print("📚 Cargando configuraciones…")

        with open(grammar_config_path, "r", encoding="utf-8") as f:
            self.grammar_summarizer = GlobalGrammaticalSummarizer(json.load(f))

        with open(workflow_data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        raw_uces = raw.get("uces", [])
        for d in raw_uces:
            if "uce_id" not in d:
                doc_id = d.get("doc_id")
                local_idx = d.get("local_idx")
                section_id = d.get("metadata", {}).get("section_id", "0")
                if doc_id is not None and local_idx is not None:
                    d["uce_id"] = f"{doc_id}_{section_id}_{local_idx}"

        self.uces = [UCE.from_dict(d) for d in raw_uces]

        # Diagnostic check to instantly reveal mismatches
        state_ids = set(raw.get("annotations_by_uce", {}).keys())
        uce_ids = {u.id for u in self.uces}
        overlap = state_ids & uce_ids
        print(
            f"State has {len(state_ids)} IDs, UCEs have {len(uce_ids)}, overlap: {len(overlap)}"
        )
        if state_ids - uce_ids:
            print(f"  ⚠️ IDs in state NOT in UCEs: {list(state_ids - uce_ids)[:5]}")
        for uce in self.uces:
            if not hasattr(uce, "discourse_annotations"):
                uce.discourse_annotations = []

        self._uces_by_doc: Dict[str, List[UCE]] = defaultdict(list)
        for uce in self.uces:
            _val = getattr(uce, "doc_id", None)
            doc_id = str(_val) if _val is not None else uce.id.split("_")[0]
            self._uces_by_doc[doc_id].append(uce)
        self.multi_agent = DiscourseMultiAgent(discourse_config_path)
        self.discourse_agents = self.multi_agent.agents

        state = _load_state(self.state_path)

        if "grammar_summary" in state and "structured_grammar" in state:
            print("📂 Resumen gramatical cargado desde estado previo.")
            self.global_grammar_summary = state["grammar_summary"]
            raw_sg = state.get("structured_grammar", {})
            # Rehydrate Counters from serialised dicts for the sections that need them
            self.global_structured_grammar = self._rehydrate_structured_grammar(raw_sg)
        else:
            print("🔬 Calculando resumen gramatical (primera vez)…")
            self.global_grammar_summary, self.global_structured_grammar = (
                self.grammar_summarizer.summarize(self.uces, by_cluster=False)
            )
            self._save_current_state()

        if "annotations_by_uce" in state:
            loaded = 0
            for uce in self.uces:
                saved = state["annotations_by_uce"].get(uce.id, [])
                uce.discourse_annotations = saved
                loaded += len(saved)
            print(f"📂 {loaded} anotaciones previas cargadas desde {self.state_path}")

    # ── State rehydration ─────────────────────────────────────────────────
    @staticmethod
    def _rehydrate_structured_grammar(raw: Dict) -> Dict:
        """
        Convert plain dicts back into Counter objects for the sections that
        build_agent_grammar_summary expects to have .most_common().
        Sections that have already been reduced to scalars (complejidad,
        metricas_lexicas, subtlex) are left as-is.
        """
        counter_sections = {
            "verbos": [
                "modo",
                "tiempo",
                "aspecto",
                "voz",
                "persona",
                "numero",
                "genero",
                "aux_tipo",
                "perifrasis",
                "subordinacion_tipo",
                "subordinacion_conjuncion",
                "concordancia",
            ],
            "pronombres": [
                "tipo",
                "subtipo",
                "persona",
                "numero",
                "genero",
                "es_referencial",
            ],
            "negaciones": ["tipo", "con_npi", "es_constituyente"],
            "cuantificadores": ["tipo", "pos"],
            "adverbios": ["categoria", "multipalabra"],
            "marcadores": ["categoria", "posicion"],
            "insubordinaciones": ["funcion", "tipo"],
            "rarezas": ["tipo"],
            "predicados": [
                "cluster_dist",
                "thematic_roles",
                "voice_dist",
                "top_verbs",
                "top_objects",
            ],
            "nouns": ["top_lemmas"],
            "adjectives": ["top_lemmas"],
            "prepositions": ["top_lemmas"],
            "verbos_lexicos": ["top_lemmas"],
        }
        result = dict(raw)
        for section, keys in counter_sections.items():
            if section not in result or not isinstance(result[section], dict):
                continue
            sec = dict(result[section])
            for k in keys:
                if k in sec and isinstance(sec[k], dict):
                    sec[k] = Counter(sec[k])
            result[section] = sec
            for k in keys:
                if k in sec and isinstance(sec[k], dict):
                    sec[k] = Counter(sec[k])
            result[section] = sec
        if "registros" in result and isinstance(result["registros"], dict):
            result["registros"] = Counter(result["registros"])

        # Rehydrate nested pos_by_cluster
        pbc = result.get("pos_by_cluster", {})
        if pbc:
            rehydrated_pbc = {}
            pos_bucket_keys = ("nouns", "adjectives", "prepositions", "verbos_lexicos")
            for cid, cdata in pbc.items():
                cdata = dict(cdata)
                for bkey in pos_bucket_keys:
                    if bkey in cdata:
                        bucket = dict(cdata[bkey])
                        if "top_lemmas" in bucket and isinstance(
                            bucket["top_lemmas"], dict
                        ):
                            bucket["top_lemmas"] = Counter(bucket["top_lemmas"])
                        cdata[bkey] = bucket
                rehydrated_pbc[cid] = cdata
            result["pos_by_cluster"] = rehydrated_pbc

        return result

    # ── doc-level helpers ─────────────────────────────────────────────────
    def _get_doc_ids(self) -> List[str]:
        doc_ids = list(self._uces_by_doc.keys())
        return sorted(doc_ids, key=lambda x: int(x) if str(x).isdigit() else x)

    def _load_uces_for_doc(self, doc_id: str) -> List[UCE]:
        return self._uces_by_doc.get(doc_id, [])

    # ── tagged-text builders ──────────────────────────────────────────────
    def _build_tagged_text_and_map(self) -> Tuple[str, List[Tuple[int, int, str]]]:
        return self._build_tagged_text_and_map_for(self.uces)

    def _build_tagged_text_and_map_for(
        self, uces: List[UCE]
    ) -> Tuple[str, List[Tuple[int, int, str]]]:
        parts, boundaries, cursor = [], [], 0
        for uce in uces:
            prefix = f"[{uce.id}]: "
            segment = prefix + uce.texto + "\n"
            start = cursor + len(prefix)
            boundaries.append((start, cursor + len(segment), uce.id))
            parts.append(segment)
            cursor += len(segment)
        return "".join(parts), boundaries

    def _map_annotations_to_uces(
        self,
        annotations: List[DiscourseAnnotation],
        full_tagged_text: str,
        boundaries: List[Tuple[int, int, str]],
        store: bool = True,
    ) -> int:
        uce_text_by_id: Dict[str, str] = {u.id: u.texto for u in self.uces}
        total_spans_mapped = 0
        if not annotations:
            return 0

        for anno in annotations:
            for span in anno.spans:
                if not span.quote or not span.quote.strip():
                    print(f"⚠️  Span sin quote válido, ignorando.")
                    continue

                cleaned_quote = _clean_quote_for_matching(span.quote)

                # ── STEP 1: local resolution ──────────────────────────────
                if span.uce_id and span.uce_id in uce_text_by_id:
                    local_start, local_end = _resolve_quote_in_text(
                        cleaned_quote, uce_text_by_id[span.uce_id]
                    )
                    if local_start != -1:
                        span.start_char = local_start
                        span.end_char = local_end
                        total_spans_mapped += 1
                        print(
                            f"    ✅ [LOCAL] '{cleaned_quote[:40]}' → [{span.uce_id}] {local_start}:{local_end}"
                        )
                        continue
                    else:
                        print(
                            f"    ⚠️ [LOCAL MISS] uce_id='{span.uce_id}' | quote='{cleaned_quote[:60]}'"
                        )
                        print(
                            f"        UCE text preview: '{uce_text_by_id[span.uce_id][:80]}'"
                        )
                else:
                    if span.uce_id:
                        print(
                            f"    ⚠️ [UCE_ID NOT FOUND] '{span.uce_id}' not in uce_text_by_id"
                        )
                        print(
                            f"        Known IDs sample: {list(uce_text_by_id.keys())[:5]}"
                        )
                    else:
                        print(f"    ⚠️ [NO UCE_ID] quote='{cleaned_quote[:60]}'")

                # ── STEP 2: global resolution ─────────────────────────────
                global_start, global_end = _resolve_quote_in_text(
                    cleaned_quote, full_tagged_text
                )

                if global_start != -1:
                    matched = False
                    for seg_start, seg_end, uce_id in boundaries:
                        if seg_start <= global_start < seg_end:
                            span.uce_id = uce_id
                            span.start_char = global_start - seg_start
                            span.end_char = global_end - seg_start
                            total_spans_mapped += 1
                            matched = True
                            print(
                                f"    ✅ [GLOBAL] '{cleaned_quote[:40]}' → [{uce_id}] {span.start_char}:{span.end_char}"
                            )
                            break
                    if not matched:
                        print(
                            f"    ⚠️ [GLOBAL FOUND BUT NO BOUNDARY MATCH] global={global_start}:{global_end}"
                        )
                    continue

                # ── STEP 3: fallback ──────────────────────────────────────
                fallback_applied = False
                if span.uce_id:
                    posibles_ids = [
                        i.strip()
                        for i in re.split(r"[,; y\n]+", span.uce_id)
                        if i.strip() in uce_text_by_id
                    ]
                    print(
                        f"    ⚠️ [FALLBACK] Trying UCE range. posibles_ids={posibles_ids}"
                    )
                    if posibles_ids:
                        first_id = posibles_ids[0]
                        last_id = posibles_ids[-1]
                        first_seg = next(
                            (b for b in boundaries if b[2] == first_id), None
                        )
                        last_seg = next(
                            (b for b in boundaries if b[2] == last_id), None
                        )
                        if first_seg and last_seg:
                            span.uce_id = first_id
                            span.start_char = 0
                            span.end_char = (last_seg[1] - 1) - first_seg[0]
                            total_spans_mapped += 1
                            fallback_applied = True
                            print(
                                f"    🔄 [FALLBACK OK] '{first_id}'→'{last_id}' 0:{span.end_char}"
                            )
                        else:
                            print(
                                f"    ❌ [FALLBACK FAIL] first_seg={first_seg} last_seg={last_seg}"
                            )

                if not fallback_applied:
                    print(f"    ❌ [ALL STRATEGIES FAILED]")
                    print(f"        uce_id  : '{span.uce_id}'")
                    print(f"        quote   : '{span.quote[:100]}'")
                    print(f"        cleaned : '{cleaned_quote[:100]}'")

            if store:
                for span in anno.spans:
                    if span.start_char != -1:
                        target = next(
                            (u for u in self.uces if u.id == span.uce_id), None
                        )
                        if target:
                            target.discourse_annotations.append(anno.to_dict())

        return total_spans_mapped

    # ── persistence ───────────────────────────────────────────────────────
    def _save_current_state(self, workflow: Optional[str] = None):
        """
        Persist grammar summary + all per-UCE annotations.
        workflow is optional; when given, used only as a label in the log.
        """
        state: Dict[str, Any] = {
            "grammar_summary": self.global_grammar_summary,
            "structured_grammar": _safe_truncate_dict(
                _counter_to_serialisable(self.global_structured_grammar),
                max_list_items=20,
            ),
            "annotations_by_uce": {u.id: u.discourse_annotations for u in self.uces},
        }
        if workflow:
            state["last_workflow"] = workflow
        _save_state(self.state_path, state)

    # ── public API ────────────────────────────────────────────────────────
    def run_discourse_agent(
        self,
        agent_name: Optional[str] = None,
        store: bool = True,
        workflow: Optional[str] = None,
    ) -> List[DiscourseAnnotation]:
        """
        Run a single named agent per-document, or all agents if agent_name
        is None.  Each agent receives a gram_cats-filtered grammar summary
        computed freshly for the document being processed.
        """
        if agent_name is None:
            return self.run_all_discourse_agents(store, workflow)

        if agent_name not in self.discourse_agents:
            print(f"❌ Agente '{agent_name}' no existe.")
            return []

        self._ensure_dependencies(agent_name, store, workflow)

        agent = self.discourse_agents[agent_name]
        print(f"\n🚀 EJECUTANDO (por documento): {agent_name}")
        if agent.gram_cats:
            print(f"   gram_cats activas: {agent.gram_cats}")

        all_annos: List[DiscourseAnnotation] = []

        for doc_id in self._get_doc_ids():
            doc_uces = self._load_uces_for_doc(doc_id)
            print(f"  📄 {doc_id} ({len(doc_uces)} UCEs)")

            # Per-document structured grammar
            _, doc_structured = self.grammar_summarizer.summarize(
                doc_uces, by_cluster=False
            )

            # ── KEY CHANGE: build agent-specific summary ──────────────────
            active_cluster_ids: Optional[Set] = {
                getattr(u, "cluster_id", None) for u in doc_uces
            } - {None}

            if agent.gram_cats:
                agent_grammar_summary = build_agent_grammar_summary(
                    agent.gram_cats,
                    doc_structured,
                    active_cluster_ids=active_cluster_ids or None,
                )
            else:
                # Agent declared no gram_cats → fall back to full doc summary
                agent_grammar_summary = self.grammar_summarizer._build_summary_text(
                    doc_structured
                )

            tagged_text, boundaries = self._build_tagged_text_and_map_for(doc_uces)
            if len(tagged_text) > self.MAX_TAGGED_CHARS:
                tagged_text = tagged_text[: self.MAX_TAGGED_CHARS]

            previous = [
                DiscourseAnnotation.from_dict(a)
                for u in doc_uces
                for a in u.discourse_annotations
                if a.get("agent") in agent.disc_cats
            ]

            annos = agent.analyze_corpus(tagged_text, agent_grammar_summary, previous)
            mapped = self._map_annotations_to_uces(
                annos, tagged_text, boundaries, store
            )
            print(f"    ✅ {mapped}/{len(annos)} anotaciones mapeadas")
            all_annos.extend(annos)

        if store:
            self._save_current_state(workflow)

        return all_annos

    def run_all_discourse_agents(
        self, store: bool = True, workflow: Optional[str] = None
    ) -> List[DiscourseAnnotation]:
        """Run every agent in topological order, each per-document."""
        print("\n🚀 ANÁLISIS GLOBAL (todos los agentes, por documento)")
        all_annos: List[DiscourseAnnotation] = []
        for agent_name in self.multi_agent.disc_execution_order:
            annos = self.run_discourse_agent(agent_name, store, workflow)
            all_annos.extend(annos)
        return all_annos

    def _ensure_dependencies(
        self, agent_name: str, store: bool, workflow: Optional[str] = None
    ):
        agent = self.discourse_agents[agent_name]
        for dep_name in agent.disc_cats:
            has = any(
                any(a.get("agent") == dep_name for a in u.discourse_annotations)
                for u in self.uces
            )
            if not has:
                print(f"🔁 Dependencia faltante '{dep_name}'. Ejecutando primero…")
                self.run_discourse_agent(dep_name, store, workflow)

    def list_agents(self):
        print("\n📋 AGENTES DISPONIBLES:")
        for name, agent in self.discourse_agents.items():
            deps = agent.disc_cats or ["ninguna"]
            gcats = agent.gram_cats or ["(todas)"]
            print(f"  • {name}")
            print(f"      deps disc : {', '.join(deps)}")
            print(f"      gram_cats : {', '.join(gcats)}")

    def get_annotation_stats(self) -> Dict:
        total_spans = 0
        by_agent: Counter = Counter()
        for u in self.uces:
            for a_dict in u.discourse_annotations:
                spans = a_dict.get("spans", [])
                total_spans += len(spans)
                by_agent[a_dict.get("agent", "?")] += len(spans)
        return {"total_spans": total_spans, "by_agent": dict(by_agent)}


# ─────────────────────────────────────────────
# ClassStatsReporter  (unchanged from original)
# ─────────────────────────────────────────────
class ClassStatsReporter:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with open(db_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.uces = [UCE.from_dict(u) for u in self.data.get("uces", [])]
        self.uces = [u for u in self.uces if u.cluster_id is not None]
        if not self.uces:
            raise ValueError("No se encontraron UCEs con cluster_id.")
        self.uces_by_cluster = defaultdict(list)
        for u in self.uces:
            self.uces_by_cluster[u.cluster_id].append(u)
        self.clusters = sorted(self.uces_by_cluster.keys())
        print(f"Cargadas {len(self.uces)} UCEs en {len(self.clusters)} clases.")

    def _safe_mean(self, v):
        return float(np.mean(v)) if v else 0.0

    def aggregate_all(self, cluster_id: int) -> Dict:
        uces = self.uces_by_cluster[cluster_id]
        verb_modo, verb_tiempo, verb_voz = Counter(), Counter(), Counter()
        for u in uces:
            for v in u.verbos:
                verb_modo[v.get("modo", "?")] += 1
                verb_tiempo[v.get("tiempo", "?")] += 1
                verb_voz[v.get("voz", "?")] += 1
        pron_tipo = Counter(p.get("tipo", "?") for u in uces for p in u.pronombres)
        adv_cat = Counter(a.get("categoria", "?") for u in uces for a in u.adverbios)
        marc_cat = Counter(
            m.get("categoria", "?") for u in uces for m in u.marcadores_discursivos
        )
        neg_tipo = Counter(n.get("tipo", "?") for u in uces for n in u.negaciones)
        prof = [u.complejidad_sintactica.get("profundidad_maxima", 0) for u in uces]
        ttrs = [u.metricas_lexicas.get("ttr", 0.0) for u in uces]
        return {
            "cluster_id": cluster_id,
            "n_uces": len(uces),
            "verbos": {
                "modo": dict(verb_modo),
                "tiempo": dict(verb_tiempo),
                "voz": dict(verb_voz),
            },
            "pronombres": dict(pron_tipo),
            "adverbios": dict(adv_cat),
            "marcadores_discursivos": dict(marc_cat),
            "negaciones": dict(neg_tipo),
            "complejidad_sintactica": {
                "profundidad_maxima_media": self._safe_mean(prof)
            },
            "metricas_lexicas": {"ttr_medio": self._safe_mean(ttrs)},
        }

    def export_json(self, output_path: str):
        all_stats = {str(cid): self.aggregate_all(cid) for cid in self.clusters}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print(f"JSON exportado a {output_path}")


# ─────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────
if __name__ == "__main__":
    mon = DebugOrchestrator(
        grammar_config_path="ia/0.json",
        discourse_config_path="ia/1.json",
        workflow_data_path="data/workflow_data.json",
        state_path="data/discourse_state.json",
    )

    mon.list_agents()
    print("\n📊 Stats actuales:", mon.get_annotation_stats())

    mon.run_discourse_agent(
        workflow="Ontológico-Cognitivo"
    )  # "Ontológico-Cognitivo" "Diversidades_epistémicas" "Performativo-Narrativo"
    # Por qué piensan, qué piensan y cómo lo expresan
    print("\n📊 Stats post-análisis:", mon.get_annotation_stats())
