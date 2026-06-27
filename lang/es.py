#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lang/es.py — Spanish language pack for PipelineGramatical.

Contains all language‑specific lexical data and syntactic rules.
Switching to another language means creating a new file in this directory
(e.g., lang/en.py, lang/fr.py) with the same interface.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Clases Español
# ------------------------------------------------------------
@dataclass
class PredicateFrame:
    """
    One resolved predicate-argument structure.
    All char offsets are GLOBAL (relative to the original full document).
    Local offsets (relative to uce.texto) are computed on demand via
    to_local(uce) to avoid storing both and risking desync.
    """

    # ── Entity (the coreferenced subject) ───────────────────────
    entity_text: str
    entity_head_lemma: str
    entity_start_char: int
    entity_end_char: int
    chain_representative: str

    # ── Predicate ───────────────────────────────────────────────
    verb_lemma: str
    verb_text: str
    verb_start_char: int
    verb_end_char: int
    voice: str  # Act|Pass|PassRefl|Impersonal|Media
    tense: str
    mood: str
    negated: bool

    # ── Internal arguments ──────────────────────────────────────
    direct_object: Optional[str] = None
    direct_object_lemma: Optional[str] = None
    direct_object_start: Optional[int] = None
    direct_object_end: Optional[int] = None
    indirect_object: Optional[str] = None
    oblique: Optional[str] = None
    oblique_lemma: Optional[str] = None

    # ── Thematic role of entity in this frame ───────────────────
    thematic_role: str = "UNSPECIFIED"

    # ── Clustering unit (computed once at construction) ─────────
    frame_fingerprint: str = ""
    doc_id: str = ""  # which interview/document this frame came from
    frame_idx: int = -1  # position in the per-doc extraction list (for write-back)

    # ── Assigned after clustering ────────────────────────────────
    cluster_id: int = -1
    cluster_label: str = ""

    # ── Provenance ───────────────────────────────────────────────
    uce_id: str = ""
    is_expansion: bool = False
    original_entity: str = ""  # filled for expansions

    # ── Convenience ──────────────────────────────────────────────
    def to_local(self, uce_start_char: int) -> Dict:
        """Returns a copy of all char fields shifted to UCE-local coords."""

        def loc(g):
            return (g - uce_start_char) if g is not None else None

        return {
            "entity_start": loc(self.entity_start_char),
            "entity_end": loc(self.entity_end_char),
            "verb_start": loc(self.verb_start_char),
            "verb_end": loc(self.verb_end_char),
            "obj_start": loc(self.direct_object_start),
            "obj_end": loc(self.direct_object_end),
        }

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> "PredicateFrame":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SpanAnnotation:
    """
    A single annotated span ready for HTML injection or export.
    char_start / char_end are LOCAL to the UCE.
    """

    char_start: int
    char_end: int
    span_type: str  # ENTITY | VERB | OBJECT | OBLIQUE
    cluster_id: int
    cluster_label: str
    thematic_role: str
    frame_fingerprint: str
    chain: str
    negated: bool
    voice: str

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


class MorphDeriver:
    """Deriva Aspecto, Voz, Número y Género para verbos del español."""

    def derive(self, token: Any) -> Dict[str, Optional[str]]:
        return {
            "asp": self._aspect(token),
            "voz": self._voice(token),
            "numero": self._number(token),
            "genero": self._gender(token),
        }

    # --- Aspecto ---
    def _aspect(self, t: Any) -> Optional[str]:
        asp = t.morph.get("Aspect")
        if asp:
            return asp[0]

        vf = (t.morph.get("VerbForm") or [None])[0]
        tense = (t.morph.get("Tense") or [None])[0]

        if vf == "Inf":
            return None
        if vf == "Ger":
            return "Imp"
        if vf == "Part":
            return "Perf"

        # Perífrasis progresiva o perfectiva
        if self._is_progressive(t):
            return "Prog"
        if self._is_perfective(t):
            return "Perf"

        if tense:
            return TENSE_TO_ASPECT.get(tense)
        return None

    def _is_progressive(self, t: Any) -> bool:
        # Gerundio gobernado por auxiliar progresivo
        if t.morph.get("VerbForm") == ["Ger"]:
            head = t.head
            if head.pos_ == "AUX" and head.lemma_ in PROGRESSIVE_AUX_LEMMAS:
                return True
        # Auxiliar progresivo con hijo gerundio
        if t.lemma_ in PROGRESSIVE_AUX_LEMMAS:
            for child in t.children:
                if child.morph.get("VerbForm") == ["Ger"]:
                    return True
        return False

    def _is_perfective(self, t: Any) -> bool:
        # Participio gobernado por haber
        if t.morph.get("VerbForm") == ["Part"]:
            head = t.head
            if head.pos_ == "AUX" and head.lemma_ in PERFECTIVE_AUX_LEMMAS:
                return True
        # 'haber' con participio hijo
        if t.lemma_ in PERFECTIVE_AUX_LEMMAS:
            for child in t.children:
                if child.morph.get("VerbForm") == ["Part"]:
                    return True
        return False

    # --- Voz ---
    def _voice(self, t: Any) -> Optional[str]:
        v = t.morph.get("Voice")
        if v:
            return v[0]

        if self._is_passive(t):
            return "Pass"
        if es_pasiva_refleja(t):
            return "PassRefl"
        if es_impersonal_se(t):
            return "Impersonal"
        if es_media_se(t):
            return "Media"

        if t.pos_ in ("VERB", "AUX"):
            return "Act"
        return None

    def _is_passive(self, t: Any) -> bool:
        # nsubj:pass
        if any(c.dep_ == PASSIVE_SUBJECT_DEP for c in t.children):
            return True
        # ser/estar + participio
        if t.morph.get("VerbForm") == ["Part"]:
            head = t.head
            if head.pos_ == "AUX" and head.lemma_ in COPULAR_VERBS:
                return True
        return False

    # --- Número ---
    def _number(self, t: Any) -> Optional[str]:
        n = t.morph.get("Number")
        if n:
            return n[0]
        num = obtener_numero_por_concordancia(t)
        if num:
            return num
        if t.pos_ == "AUX" and t.head:
            hn = t.head.morph.get("Number")
            if hn:
                return hn[0]
        return None

    # --- Género (solo para participios) ---
    def _gender(self, t: Any) -> Optional[str]:
        vf = (t.morph.get("VerbForm") or [None])[0]
        if vf not in ("Part", "Ger", None):
            g = t.morph.get("Gender")
            return g[0] if g else None
        return obtener_genero_para_participio(t)


# ------------------------------------------------------------------------------
# NEGACIONES
# ------------------------------------------------------------------------------
PALABRAS_NEGATIVAS: FrozenSet[str] = frozenset(
    {
        "no",
        "nunca",
        "jamás",
        "tampoco",
        "ni",
    }
)

NPI_WORDS: FrozenSet[str] = frozenset(
    {
        "nadie",
        "nada",
        "ningún",
        "ninguna",
        "ninguno",
        "jamás",
        "tampoco",
        "ni",
    }
)

# ------------------------------------------------------------------------------
# PRONOMBRES
# ------------------------------------------------------------------------------
NON_REFERENTIAL: FrozenSet[str] = frozenset(
    {
        "que",
        "quien",
        "quienes",
        "cual",
        "cuales",
        "cuyo",
        "cuya",
        "qué",
        "quién",
        "cómo",
        "cuándo",
        "dónde",
        "cuánto",
        "todo",
        "nada",
        "algo",
        "nadie",
        "alguien",
    }
)

CLITICOS: List[str] = [
    "los",
    "las",
    "les",
    "nos",
    "me",
    "te",
    "se",
    "lo",
    "la",
    "le",
    "os",
]

ENCLITICO_VERBFORMS: FrozenSet[str] = frozenset({"Inf", "Ger"})

# ------------------------------------------------------------------------------
# VERBOS / PERÍFRASIS
# NOTE: 'venir' is intentionally excluded — "viene a la mente" is NOT a
# verbal periphrasis.
# ------------------------------------------------------------------------------
PERIFRASIS_VERBOS: FrozenSet[str] = frozenset(
    {
        "ir",
        "acabar",
        "deber",
        "poder",
        "querer",
        "soler",
        "haber",
        "estar",
        "seguir",
        "andar",
        "echar",
        "tener",
        "llevar",
        "dejar",
        "ponerse",
    }
)

NON_FINITE_FORMS: FrozenSet[str] = frozenset({"Inf", "Part", "Ger"})

# ----------------------------------------------------------------------------
# Datos para MorphDeriver (español)
# ----------------------------------------------------------------------------

# Mapeo de Tiempo → Aspecto (para verbos finitos sin contexto perifrástico)
TENSE_TO_ASPECT: Dict[str, str] = {
    "Pres": "Imp",  # presente → imperfectivo
    "Imp": "Imp",  # pretérito imperfecto → imperfectivo
    "Fut": "Imp",  # futuro → imperfectivo
    "Cnd": "Imp",  # condicional → imperfectivo
    "Past": "Perf",  # pretérito indefinido → perfectivo
    "Pqp": "Perf",  # pluscuamperfecto → perfectivo
}

# Verbos auxiliares que desencadenan aspecto progresivo cuando su hijo es gerundio
PROGRESSIVE_AUX_LEMMAS: FrozenSet[str] = frozenset(
    {"estar", "seguir", "andar", "ir", "venir"}
)

# Verbos auxiliares que desencadenan aspecto perfectivo cuando su hijo es participio
PERFECTIVE_AUX_LEMMAS: FrozenSet[str] = frozenset({"haber"})

# Lemas de verbos copulativos (para voz pasiva con ser/estar)
COPULAR_VERBS: FrozenSet[str] = frozenset({"ser", "estar"})

# Lemas de pronombres reflexivos que pueden indicar pasiva refleja
REFLEXIVE_PRONOUN_LEMMAS: FrozenSet[str] = frozenset({"se"})

# Dependencias que indican sujeto pasivo
PASSIVE_SUBJECT_DEP: str = "nsubj:pass"

# Dependencias que indican sujeto activo (para concordancia de número)
ACTIVE_SUBJECT_DEPS: FrozenSet[str] = frozenset({"nsubj", "nsubj:pass"})

EXPRESIONES_IDIOMATICAS_NPI = frozenset(
    [
        "nada más",
        "nada que ver",
        "nadie dice nada",
        "sin nada",
        "nada menos",
        "nada del otro mundo",
        "nada de eso",
        "nada más y nada menos",
    ]
)

# ------------------------------------------------------------------------------
# CUANTIFICADORES
# NOTE: 'nada' and 'nadie' belong to NEGATIVO, not PROPORCIONAL.
# ------------------------------------------------------------------------------
CUANT_TIPOS: Dict[str, Set[str]] = {
    "UNIVERSAL": {
        "todo",
        "toda",
        "todos",
        "todas",
        "cada",
        "ambos",
        "ambas",
        "sendos",
        "sendas",
    },
    "EXISTENCIAL": {
        "algún",
        "alguna",
        "algunos",
        "algunas",
        "cierto",
        "cierta",
        "ciertos",
        "ciertas",
    },
    "NEGATIVO": {
        "ningún",
        "ninguna",
        "ningunos",
        "ningunas",
        "ni un",
        "ni una",
        "nada",
        "nadie",
    },
    "PROPORCIONAL": {
        "mucho",
        "mucha",
        "muchos",
        "muchas",
        "poco",
        "poca",
        "pocos",
        "pocas",
        "bastante",
        "bastantes",
        "demasiado",
        "demasiada",
        "demasiados",
        "demasiadas",
        "algo",
        "harto",
        "hartos",
        "hartas",
    },
    "NUMERAL_CARDINAL": set(),
    "NUMERAL_ORDINAL": set(),
}

PARTITIVOS: FrozenSet[str] = frozenset(
    {
        "montón",
        "pila",
        "sinfín",
        "puñado",
        "chorro",
        "cantidad",
        "grupo",
        "mayoría",
        "minoría",
    }
)

# ------------------------------------------------------------------------------
# ADVERBIOS — simple and multi-word lexicons
# ------------------------------------------------------------------------------
LEXICON_ADVERBS: Dict[str, Set[str]] = {
    "tiempo": {
        "ayer",
        "hoy",
        "mañana",
        "pronto",
        "tarde",
        "temprano",
        "siempre",
        "nunca",
        "jamás",
        "ahora",
        "luego",
        "antes",
        "después",
        "ya",
        "todavía",
        "aún",
    },
    "lugar": {
        "aquí",
        "ahí",
        "allí",
        "allá",
        "acá",
        "cerca",
        "lejos",
        "arriba",
        "abajo",
        "delante",
        "detrás",
        "dentro",
        "fuera",
        "encima",
        "debajo",
    },
    "modo": {
        "bien",
        "mal",
        "así",
        "despacio",
        "deprisa",
        "rápido",
        "claro",
        "igual",
        "peor",
        "mejor",
    },
    "grado": {
        "muy",
        "mucho",
        "mucha",
        "poco",
        "bastante",
        "demasiado",
        "casi",
        "más",
        "menos",
        "tan",
        "tanto",
        "nada",
        "apenas",
    },
    "epistemico": {
        "quizá",
        "quizás",
        "acaso",
        "seguramente",
        "probablemente",
        "posiblemente",
        "supuestamente",
    },
    "comparativo": {
        "más",
        "menos",
        "mejor",
        "peor",
        "tan",
        "tanto",
        "igual",
    },
    "foco": {
        "solo",
        "solamente",
        "únicamente",
        "exclusivamente",
        "precisamente",
    },
    "conjuntivo": {
        "además",
        "asimismo",
        "sin embargo",
        "no obstante",
        "por consiguiente",
        "por lo tanto",
        "en cambio",
    },
    "orientado_hablante": {
        "francamente",
        "sinceramente",
        "lamentablemente",
        "afortunadamente",
        "honestamente",
        "claramente",
        "obviamente",
    },
    "dominio": {
        "técnicamente",
        "políticamente",
        "económicamente",
        "socialmente",
        "legalmente",
        "éticamente",
        "ambientalmente",
        "matemáticamente",
        "físicamente",
        "psicológicamente",
        "directamente",
        "indirectamente",
        "generalmente",
        "particularmente",
        "específicamente",
        "básicamente",
        "fundamentalmente",
        "principalmente",
        "esencialmente",
        "concretamente",
        "formalmente",
        "teóricamente",
        "prácticamente",
    },
}

MULTI_WORD_ADVERBS: Dict[str, Set[str]] = {
    "tiempo": {"de repente", "de inmediato", "de pronto", "en seguida"},
    "lugar": {"al lado", "en frente", "de cerca", "de lejos"},
    "modo": {
        "de buen grado",
        "de mala manera",
        "a propósito",
        "de tal manera",
        "por supuesto",
    },
    "frecuencia": {"a menudo", "de vez en cuando", "con frecuencia", "rara vez"},
    "grado": {"un poco", "un montón", "en absoluto", "por completo"},
    "dominio": {"en general", "en particular", "en teoría", "en la práctica"},
    "comparativo": {
        "cada vez más",
        "cuanto más",
        "más que",
        "menos que",
    },
    "foco": {"más que todo", "más que nada"},
    "conjuntivo": {
        "sin embargo",
        "no obstante",
        "por consiguiente",
        "así que",
        "por eso",
        "en cambio",
    },
    "orientado_hablante": {
        "a decir verdad",
        "en realidad",
        "por desgracia",
        "afortunadamente",
    },
    "orientado_sujeto": {
        "a sabiendas",
        "de forma deliberada",
        "con cuidado",
        "sin querer",
    },
}

# Merged flat list — sorted longest-first for greedy matching
ALL_KNOWN_ADVERBS: List[Tuple[str, str]] = sorted(
    [
        (word, cat)
        for cat, words in {
            **LEXICON_ADVERBS,
            **{k: v for k, v in MULTI_WORD_ADVERBS.items()},
        }.items()
        for word in words
    ],
    key=lambda x: len(x[0]),
    reverse=True,
)

# Adverb classifier category list (includes epistemico)
ADVERB_CATEGORIES: List[str] = [
    "modo",
    "tiempo",
    "lugar",
    "frecuencia",
    "grado",
    "dominio",
    "comparativo",
    "foco",
    "conjuntivo",
    "orientado_hablante",
    "orientado_sujeto",
    "epistemico",
]

# Adverbs that should never be classified (e.g., short function words)
ADVERB_BLOCKED: FrozenSet[str] = frozenset({"no", "sí", "si", "ya"})

# ------------------------------------------------------------------------------
# MARCADORES DISCURSIVOS
# ------------------------------------------------------------------------------
LOCUCIONES_DISCURSIVAS: Dict[str, List[str]] = {
    "REFORMULADORES": ["o sea", "es decir", "en otras palabras", "mejor dicho"],
    "ESTRUCTURADORES": ["por un lado", "por otro lado", "en fin", "por cierto"],
    "ARGUMENTATIVOS": [
        "sin embargo",
        "por lo tanto",
        "en el fondo",
        "a fin de cuentas",
        "por ejemplo",
        "en cuanto a",
        "más que todo",
        "sobre todo",
    ],
    "CONVERSACIONALES": ["lo que pasa es que", "la cosa es que", "pues nada", "ya ves"],
}

CONECTORES_DISC_ADV: FrozenSet[str] = frozenset(
    {
        "además",
        "asimismo",
        "entonces",
        "luego",
        "sin embargo",
        "no obstante",
        "en cambio",
        "por tanto",
        "por consiguiente",
        "así que",
        "finalmente",
        "en conclusión",
        "ahora bien",
        "eso sí",
        "encima",
        "incluso",
        "tampoco",
        "también",
        "así",
        "bueno",
        "pues",
    }
)

# ------------------------------------------------------------------------------
# INSUBORDINACIÓN — function labels
# ------------------------------------------------------------------------------
INSUBORDINACION_FUNCIONES: Dict[str, str] = {
    "que": "EXCLAMATIVA_JUSIVA",
    "si": "REFUTATIVA",
    "pero si": "REFUTATIVA",
    "como": "ADVERTENCIA",
}
INSUBORDINACION_DEFAULT_FUNCION: str = "DESCONOCIDA"

SUBORDINATING_CONJUNCTIONS: Dict[str, Tuple[str, Optional[str]]] = {
    # Completivas
    "que": ("completiva", None),
    "si": ("completiva", "interrogativa"),
    # Adverbiales temporales
    "cuando": ("adverbial", "temporal"),
    "mientras": ("adverbial", "temporal"),
    "antes": ("adverbial", "temporal"),
    "después": ("adverbial", "temporal"),
    "apenas": ("adverbial", "temporal"),
    # Adverbiales causales
    "porque": ("adverbial", "causal"),
    "pues": ("adverbial", "causal"),
    "ya que": ("adverbial", "causal"),
    # Adverbiales condicionales/concesivas
    "si": ("adverbial", "condicional"),
    "como": ("adverbial", "condicional"),
    "aunque": ("adverbial", "concesiva"),
    "a pesar de": ("adverbial", "concesiva"),
    # Adverbiales modales
    "como": ("adverbial", "modal"),
    "según": ("adverbial", "modal"),
    # Comparativas
    "como": ("comparativa", None),
    "cuanto": ("comparativa", None),
}

AMBIGUOUS_SUBORDINATORS = {
    "como",
    "si",
    "cuando",
}  # pueden ser subordinantes o adverbios interrogativos

# Dependencias que indican subordinación
SUBORDINATING_DEPS = {"ccomp", "xcomp", "advcl", "relcl", "advmod"}

# ------------------------------------------------------------------------------
# RAREZAS / BuscadorAnalogico — dependency patterns
# ------------------------------------------------------------------------------
RAREZAS_PATTERNS: List[Tuple[str, str, List[Dict]]] = [
    (
        "DEQUEISMO",
        "Dequeísmo (verbo + de que)",
        [
            {"RIGHT_ID": "verbo", "RIGHT_ATTRS": {"POS": "VERB"}},
            {
                "LEFT_ID": "verbo",
                "REL_OP": ">",
                "RIGHT_ID": "de",
                "RIGHT_ATTRS": {"LOWER": "de"},
            },
            {
                "LEFT_ID": "de",
                "REL_OP": ">",
                "RIGHT_ID": "que",
                "RIGHT_ATTRS": {"LOWER": "que"},
            },
        ],
    ),
    (
        "DOBLADO_CLITICO",
        "Doblado de clítico (acusativo duplicado)",
        [
            {"RIGHT_ID": "verbo", "RIGHT_ATTRS": {"POS": "VERB"}},
            {
                "LEFT_ID": "verbo",
                "REL_OP": ">",
                "RIGHT_ID": "clitico",
                "RIGHT_ATTRS": {
                    "POS": "PRON",
                    "LOWER": {"IN": ["lo", "la", "los", "las", "le", "les"]},
                },
            },
            {
                "LEFT_ID": "verbo",
                "REL_OP": ">",
                "RIGHT_ID": "prep_a",
                "RIGHT_ATTRS": {"LOWER": "a", "POS": "ADP"},
            },
            {
                "LEFT_ID": "prep_a",
                "REL_OP": ">",
                "RIGHT_ID": "nombre",
                "RIGHT_ATTRS": {"POS": "PROPN"},
            },
        ],
    ),
    (
        "POSESIVO_POSPUESTO",
        "Posesivo pospuesto (ej. detrás mío)",
        [
            {"RIGHT_ID": "prep", "RIGHT_ATTRS": {"POS": {"IN": ["ADP", "ADV"]}}},
            {
                "LEFT_ID": "prep",
                "REL_OP": ">",
                "RIGHT_ID": "posesivo",
                "RIGHT_ATTRS": {
                    "LOWER": {
                        "IN": [
                            "mío",
                            "mía",
                            "míos",
                            "mías",
                            "tuyo",
                            "tuya",
                            "tuyos",
                            "tuyas",
                            "suyo",
                            "suya",
                            "suyos",
                            "suyas",
                        ]
                    }
                },
            },
        ],
    ),
]

# ------------------------------------------------------------------------------
# TRAINING DATA FOR ADVERB CLASSIFIER
# ------------------------------------------------------------------------------
MANUAL_TRAINING_EXAMPLES: List[Tuple[str, str]] = [
    ("Ella canta muy bien.", "modo"),
    ("El niño corrió rápidamente hacia la meta.", "modo"),
    ("Habla claramente o no te entenderé.", "modo"),
    ("Actuó correctamente ante la emergencia.", "modo"),
    ("El equipo jugó mal en el primer tiempo.", "modo"),
    ("Responde lentamente a las preguntas.", "modo"),
    ("Viste elegantemente para la ocasión.", "modo"),
    ("Condujo peligrosamente en la niebla.", "modo"),
    ("Expresa sus ideas fácilmente.", "modo"),
    ("Trabaja duro para conseguir sus metas.", "modo"),
    ("Llegaremos mañana por la mañana.", "tiempo"),
    ("Nunca he visto algo igual.", "tiempo"),
    ("Ya terminé el informe.", "tiempo"),
    ("Antes vivía en Madrid.", "tiempo"),
    ("Después nos vemos.", "tiempo"),
    ("Siempre llega tarde.", "tiempo"),
    ("Aún no ha llegado.", "tiempo"),
    ("Pronto empezará el concierto.", "tiempo"),
    ("Hoy hace buen día.", "tiempo"),
    ("Todavía no lo sé.", "tiempo"),
    ("El libro está ahí sobre la mesa.", "lugar"),
    ("Vive cerca de la estación.", "lugar"),
    ("Camina hacia adelante.", "lugar"),
    ("Dejó las llaves dentro.", "lugar"),
    ("El perro está fuera.", "lugar"),
    ("Mira hacia arriba.", "lugar"),
    ("Nos vemos allá.", "lugar"),
    ("Coloca el florero encima.", "lugar"),
    ("El niño se escondió detrás.", "lugar"),
    ("Viajamos lejos.", "lugar"),
    ("Siempre vamos al cine los viernes.", "frecuencia"),
    ("A menudo salgo a correr.", "frecuencia"),
    ("Nunca como carne.", "frecuencia"),
    ("Rara vez veo televisión.", "frecuencia"),
    ("Usualmente desayuno a las 8.", "frecuencia"),
    ("Frecuentemente visitamos a nuestros abuelos.", "frecuencia"),
    ("Ocasionalmente juego al ajedrez.", "frecuencia"),
    ("Diariamente reviso mi correo.", "frecuencia"),
    ("De vez en cuando tomo un café con leche.", "frecuencia"),
    ("Casi nunca salgo de noche.", "frecuencia"),
    ("La comida estaba demasiado salada.", "grado"),
    ("Me gusta mucho el chocolate.", "grado"),
    ("Es bastante caro.", "grado"),
    ("Casi me caigo.", "grado"),
    ("Está muy lejos.", "grado"),
    ("Tiene poco dinero.", "grado"),
    ("Es más interesante de lo que pensaba.", "grado"),
    ("Corrió menos de lo esperado.", "grado"),
    ("Es tan inteligente como su hermana.", "grado"),
    ("No me importa nada.", "grado"),
    ("Técnicamente, el proyecto es viable.", "dominio"),
    ("Políticamente, es una decisión arriesgada.", "dominio"),
    ("Económicamente, no es rentable.", "dominio"),
    ("Socialmente, está muy bien visto.", "dominio"),
    ("Legalmente, no podemos hacer eso.", "dominio"),
    ("Éticamente, es cuestionable.", "dominio"),
    ("Ambientalmente, es sostenible.", "dominio"),
    ("Matemáticamente, es correcto.", "dominio"),
    ("Físicamente, es imposible.", "dominio"),
    ("Psicológicamente, afecta su autoestima.", "dominio"),
    ("Directamente no se menciona en la ley.", "dominio"),
    ("Indirectamente afecta a muchas personas.", "dominio"),
    ("Generalmente, los resultados son positivos.", "dominio"),
    ("Específicamente me refiero al artículo 3.", "dominio"),
    ("Básicamente es un problema de comunicación.", "dominio"),
    ("Fundamentalmente, el sistema es correcto.", "dominio"),
    ("Principalmente se usa en contextos académicos.", "dominio"),
    ("Teóricamente podría funcionar.", "dominio"),
    ("Prácticamente es imposible de implementar.", "dominio"),
    ("Juan es más alto que Pedro.", "comparativo"),
    ("María corre más rápido que él.", "comparativo"),
    ("Este libro es menos interesante.", "comparativo"),
    ("Hoy hace mejor tiempo que ayer.", "comparativo"),
    ("El examen fue peor de lo esperado.", "comparativo"),
    ("Corre tan rápido como su hermano.", "comparativo"),
    ("Come tanto como yo.", "comparativo"),
    ("Es igual de inteligente.", "comparativo"),
    ("Cada vez más difícil.", "comparativo"),
    ("Cuanto más estudio, más aprendo.", "comparativo"),
    ("Solo quiero un poco de paz.", "foco"),
    ("Únicamente asistió él.", "foco"),
    ("Exclusivamente hablaron de eso.", "foco"),
    ("Precisamente eso es lo que necesito.", "foco"),
    ("Solamente dime la verdad.", "foco"),
    ("Tan solo mira.", "foco"),
    ("No pienses más que en ti.", "foco"),
    ("Específicamente, te lo digo a ti.", "foco"),
    ("Particularmente, me gusta este color.", "foco"),
    ("Justamente ahora llegó.", "foco"),
    ("Además, tenemos que comprar leche.", "conjuntivo"),
    ("Sin embargo, no estoy de acuerdo.", "conjuntivo"),
    ("Por consiguiente, debemos actuar.", "conjuntivo"),
    ("En cambio, ella prefirió quedarse.", "conjuntivo"),
    ("No obstante, seguiremos adelante.", "conjuntivo"),
    ("Asimismo, queremos agradecer su apoyo.", "conjuntivo"),
    ("En conclusión, ha sido un éxito.", "conjuntivo"),
    ("Por lo tanto, es necesario esperar.", "conjuntivo"),
    ("De todas formas, intentémoslo.", "conjuntivo"),
    ("En cualquier caso, no te preocupes.", "conjuntivo"),
    ("Francamente, no me importa.", "orientado_hablante"),
    ("Sinceramente, creo que te equivocas.", "orientado_hablante"),
    ("Lamentablemente, no pudimos asistir.", "orientado_hablante"),
    ("Afortunadamente, todo salió bien.", "orientado_hablante"),
    ("Sinceramente, me alegro por ti.", "orientado_hablante"),
    ("Honestamente, no lo sé.", "orientado_hablante"),
    ("Claramente, es un error.", "orientado_hablante"),
    ("Obviamente, eso no funcionará.", "orientado_hablante"),
    ("Personalmente, prefiero el otro.", "orientado_hablante"),
    ("En realidad, no es así.", "orientado_hablante"),
    ("El niño corrió rápidamente hacia su madre.", "orientado_sujeto"),
    ("El político mintió deliberadamente.", "orientado_sujeto"),
    ("Ella aceptó voluntariamente.", "orientado_sujeto"),
    ("Condujo descuidadamente.", "orientado_sujeto"),
    ("Actuó valientemente.", "orientado_sujeto"),
    ("Respondió cortésmente.", "orientado_sujeto"),
    ("Trabajó eficientemente.", "orientado_sujeto"),
    ("Se retiró discretamente.", "orientado_sujeto"),
    ("Lo hizo a propósito.", "orientado_sujeto"),
    ("Se comportó infantilmente.", "orientado_sujeto"),
    ("Quizás venga mañana.", "epistemico"),
    ("Probablemente no asista.", "epistemico"),
    ("Acaso tenga razón.", "epistemico"),
    ("Seguramente llegará tarde.", "epistemico"),
    ("Posiblemente sea verdad.", "epistemico"),
]

TRAINING_TEMPLATES: Dict[str, List[str]] = {
    "modo": [
        "{adv} {verb} {subject}.",
        "{subject} {verb} {adv}.",
        "{adv}, {subject} {verb}.",
    ],
    "tiempo": ["{adv} {verb} {subject}.", "{subject} {verb} {adv}.", "{adv} {verb}."],
    "lugar": ["{adv} {verb} {subject}.", "{subject} {verb} {adv}.", "Está {adv}."],
    "frecuencia": ["{adv} {verb} {subject}.", "{subject} {verb} {adv}."],
    "grado": ["Es {adv} {adj}.", "Es {adv} {adj} para {subject}."],
    "dominio": ["{adv}, {subject} {verb} {obj}.", "{adv} el resultado es {adj}."],
    "comparativo": [
        "{subject} es {adv} {adj} que {subject2}.",
        "{subject} {verb} {adv} que {subject2}.",
        "Es {adv} {adj} de lo que pensaba.",
    ],
    "foco": ["{adv} {subject} {verb} {obj}.", "{subject} {verb} {adv} {obj}."],
    "conjuntivo": ["{adv}, {subject} {verb} {obj}."],
    "orientado_hablante": ["{adv}, {subject} {verb} {obj}."],
    "orientado_sujeto": ["{subject} {verb} {adv}.", "{subject} {verb} {adv} {obj}."],
    "epistemico": ["{adv} {verb} {subject}.", "{adv}, {subject} {verb} {obj}."],
}

TRAINING_SUBJECTS: List[str] = [
    "él",
    "ella",
    "ellos",
    "nosotros",
    "yo",
    "tú",
    "Juan",
    "María",
    "el equipo",
]
TRAINING_VERBS: List[str] = [
    "corre",
    "come",
    "habla",
    "trabaja",
    "llega",
    "estudia",
    "canta",
    "baila",
    "escribe",
    "lee",
]
TRAINING_ADJECTIVES: List[str] = [
    "grande",
    "pequeño",
    "caro",
    "barato",
    "rápido",
    "lento",
    "interesante",
    "aburrido",
]
TRAINING_OBJECTS: List[str] = [
    "un libro",
    "la tarea",
    "en la escuela",
    "a casa",
    "el problema",
    "rápidamente",
]

LEXICON_MULTIPLIER: int = 3
MULTI_WORD_MULTIPLIER: int = 8
DOMINIO_EXTRA_MULTIPLIER: int = 5

# ------------------------------------------------------------------------------
# FUNCIONES DE REGLAS SINTÁCTICAS ESPECÍFICAS DEL ESPAÑOL
# Estas funciones encapsulan la lógica que antes estaba dispersa en el pipeline.
# ------------------------------------------------------------------------------


def tipo_negacion(token: Any, alcance: Any) -> str:
    """
    Determina el tipo de negación según el token negativo y su alcance.
    Retorna uno de: NEGACION_VERBAL, NEGACION_DE_GRADO, NEGACION_ADVERBIAL,
    NEGACION_NOMINAL, NEGACION_CONJUNTIVA, NEGACION_OTRA, más sufijo _CONSTITUYENTE
    si corresponde.
    """
    dep = token.dep_
    pos_alcance = alcance.pos_
    if dep == "advmod":
        if pos_alcance == "VERB":
            tipo = "NEGACION_VERBAL"
        elif pos_alcance in ("ADJ", "ADV"):
            tipo = "NEGACION_DE_GRADO"
        else:
            tipo = "NEGACION_ADVERBIAL"
    elif dep == "det":
        tipo = "NEGACION_NOMINAL"
    elif dep == "cc" and token.lower_ == "ni":
        tipo = "NEGACION_CONJUNTIVA"
    else:
        tipo = "NEGACION_OTRA"

    # Si el alcance es nominal o adjetival y el token es hijo directo, es constituyente
    if pos_alcance in ("NOUN", "PROPN", "ADJ", "ADV") and token.head == alcance:
        tipo += "_CONSTITUYENTE"
    return tipo


def clasificar_pronombre_explicito(token: Any) -> str:
    """
    Determina el subtipo de un pronombre explícito (PRON) basado en morfología UD.
    Retorna strings como: PRONOMBRE_REFLEXIVO, PRONOMBRE_ACUSATIVO, etc.
    """
    morph = token.morph
    pron_type = morph.get("PronType")
    reflex = morph.get("Reflex")
    case = morph.get("Case")

    if pron_type == ["Prs"]:
        if reflex == ["Yes"]:
            return "PRONOMBRE_REFLEXIVO"
        elif case == ["Acc"]:
            return "PRONOMBRE_ACUSATIVO"
        elif case == ["Dat"]:
            return "PRONOMBRE_DATIVO"
        else:
            return "PRONOMBRE_PERSONAL"
    elif pron_type == ["Dem"]:
        return "PRONOMBRE_DEMOSTRATIVO"
    elif pron_type == ["Rel"]:
        return "PRONOMBRE_RELATIVO"
    elif pron_type == ["Int"]:
        return "PRONOMBRE_INTERROGATIVO"
    elif pron_type == ["Ind"]:
        return "PRONOMBRE_INDEFINIDO"
    else:
        return "PRONOMBRE_OTRO"


def extraer_enclitico(token: Any) -> Optional[Tuple[str, str]]:
    """
    Si el verbo termina en un clítico pronominal, retorna (base, clitico).
    Ejemplo: "llevarla" → ("llevar", "la")
    Caso contrario retorna None.
    """
    texto_lower = token.text.lower()
    verbform = token.morph.get("VerbForm")
    if not verbform or verbform[0] not in ENCLITICO_VERBFORMS:
        return None
    for cl in CLITICOS:
        if texto_lower.endswith(cl):
            base = texto_lower[: -len(cl)]
            if len(base) >= 2:
                return (base, cl)
    return None


def corregir_lema_para_clitico(token: Any) -> str:
    """
    Retorna el lema del verbo sin el clítico final, si lo tiene.
    Si no tiene clítico, retorna el lema original.
    """
    lema = token.lemma_
    encl = extraer_enclitico(token)
    if encl:
        return encl[0]  # la base
    return lema


def tipo_cuantificador(token: Any) -> Optional[str]:
    """
    Determina el tipo de cuantificador basado en la palabra y contexto.
    Retorna: "UNIVERSAL", "EXISTENCIAL", "NEGATIVO", "PROPORCIONAL",
            "NUMERAL_CARDINAL", "NUMERAL_ORDINAL", "CUANTIFICADOR_SEMANTICO".
    """
    lower = token.lower_
    for cat, palabras in CUANT_TIPOS.items():
        if lower in palabras:
            return cat
    if token.pos_ == "NUM":
        if token.morph.get("NumType") == ["Ord"] or lower.endswith(("º", "ª")):
            return "NUMERAL_ORDINAL"
        else:
            return "NUMERAL_CARDINAL"
    if token.pos_ == "NOUN" and lower in PARTITIVOS:
        if any(c.lower_ == "de" for c in token.children):
            return "CUANTIFICADOR_SEMANTICO"
    # WordNet se aplica en el extractor, no aquí (por eficiencia)
    return None


def get_negation_scope(neg_token: Any) -> Any:
    """
    Determina el alcance de una palabra negativa en español.
    Retorna el token que representa el alcance (normalmente el verbo o auxiliar).
    Espera un token spaCy con atributos .head, .ancestors, .pos_, .lemma_, .children.
    """
    alcance = neg_token.head
    # Buscar auxiliar 'haber' con participio (tiempos compuestos)
    aux_haber = None
    for anc in neg_token.ancestors:
        if anc.pos_ == "AUX" and anc.lemma_ == "haber":
            # Verificar que tenga un hijo participio
            if any(
                c.pos_ == "VERB" and "Part" in c.morph.get("VerbForm", [])
                for c in anc.children
            ):
                aux_haber = anc
                break
    if aux_haber:
        alcance = aux_haber
    else:
        # Subir por encadenamiento de auxiliares/participios
        while alcance.pos_ in ("AUX", "PART") and alcance != alcance.head:
            alcance = alcance.head
    return alcance


def correct_pronoun_dependency(dep: str, morph_case: Optional[List[str]]) -> str:
    """
    Corrige la dependencia de un pronombre según la morfología.
    En español, los pronombres dativos deben tener dep='iobj' aunque el parser
    asigne 'obj' o 'obl'.
    """
    if morph_case == ["Dat"] and dep in ("obj", "obl"):
        return "iobj"
    return dep


def is_prodrop_verb(token: Any) -> bool:
    """
    Determina si un verbo finito en español tiene sujeto nulo (pro-drop).
    Retorna True si el verbo es finito y no tiene ningún sujeto explícito
    en su subárbol.
    """
    if token.pos_ != "VERB":
        return False
    verbform = token.morph.get("VerbForm")
    if verbform and verbform[0] in NON_FINITE_FORMS:
        return False  # No finitos no tienen pro-drop
    # Buscar cualquier sujeto explícito
    has_subject = any("subj" in t.dep_ for t in token.subtree if t != token)
    return not has_subject


def is_periphrastic_construction(aux_token: Any) -> Optional[Dict[str, str]]:
    """
    Detecta si un token (verbo o auxiliar) es parte de una perífrasis verbal.
    Retorna un dict con 'verbo_principal' y 'tipo' si la encuentra, o None.
    """
    if (
        aux_token.pos_ not in ("VERB", "AUX")
        or aux_token.lemma_ not in PERIFRASIS_VERBOS
    ):
        return None
    for child in aux_token.children:
        if child.dep_ in ("xcomp", "advcl") and child.pos_ == "VERB":
            child_vform = child.morph.get("VerbForm")
            if child_vform and child_vform[0] in ("Inf", "Ger"):
                return {"verbo_principal": child.lemma_, "tipo": "perifrasis"}
    return None


def detect_contraction(token: Any) -> Optional[Dict[str, str]]:
    """
    Detecta contracciones del español ('al', 'del') y retorna información.
    """
    lower = token.text.lower()
    if lower == "al":
        return {"preposicion": "a", "articulo": "el"}
    elif lower == "del":
        return {"preposicion": "de", "articulo": "el"}
    return None


def get_subordination_type(
    conj_text: str, dep: str
) -> Optional[Tuple[str, Optional[str]]]:
    """Retorna (tipo, subtipo) según conjunción y dependencia, o None."""
    # Si la dependencia es relativa -> siempre relativa
    if dep == "relcl":
        return ("relativa", None)
    # Si la dependencia es ccomp o xcomp -> completiva (salvo que conjunción diga lo contrario)
    if dep in ("ccomp", "xcomp"):
        if conj_text in SUBORDINATING_CONJUNCTIONS:
            return SUBORDINATING_CONJUNCTIONS[conj_text]
        return ("completiva", None)
    # Si es advcl -> adverbial, buscar subtipo en léxico
    if dep == "advcl":
        if conj_text in SUBORDINATING_CONJUNCTIONS:
            return SUBORDINATING_CONJUNCTIONS[conj_text]
        return ("adverbial", "otra")
    # Si es advmod y conjunción comparativa -> comparativa
    if dep == "advmod" and conj_text in ("como", "cuanto"):
        return ("comparativa", None)
    return None


def get_insubordination_function(conjunction_text: str) -> str:
    """
    Retorna la función pragmática de una conjunción insubordinada en español.
    """
    return INSUBORDINACION_FUNCIONES.get(
        conjunction_text.lower(), INSUBORDINACION_DEFAULT_FUNCION
    )


def es_pasiva_refleja(verbo_token: Any) -> bool:
    """
    Detecta pasiva refleja: 'se' + verbo transitivo en 3ª persona,
    con sujeto paciente (explícito o implícito). Ej: 'Se venden casas'.
    """
    if verbo_token.pos_ != "VERB":
        return False
    # Debe tener un hijo 'se' con dep_ 'obj' o 'expl'
    tiene_se = any(
        c.lemma_ == "se" and c.pos_ == "PRON" and c.dep_ in ("obj", "expl")
        for c in verbo_token.children
    )
    if not tiene_se:
        return False
    # Verbo en 3ª persona (singular o plural)
    persona = verbo_token.morph.get("Person")
    if not persona or persona[0] != "3":
        return False
    # Debe ser transitivo (tener objeto directo) o tener sujeto paciente (nsubj:pass)
    tiene_objeto = any(c.dep_ == "obj" for c in verbo_token.children)
    tiene_sujeto_pasivo = any(c.dep_ == "nsubj:pass" for c in verbo_token.children)
    return tiene_objeto or tiene_sujeto_pasivo


def es_impersonal_se(verbo_token: Any) -> bool:
    """
    Detecta impersonal con 'se': verbo intransitivo en 3ª singular,
    sin sujeto explícito. Ej: 'Se vive bien aquí'.
    """
    if verbo_token.pos_ != "VERB":
        return False
    # Verbo en 3ª persona singular
    persona = verbo_token.morph.get("Person")
    numero = verbo_token.morph.get("Number")
    if not (persona == ["3"] and numero == ["Sing"]):
        return False
    # No debe tener sujeto (nsubj ni nsubj:pass)
    tiene_sujeto = any("subj" in c.dep_ for c in verbo_token.children)
    if tiene_sujeto:
        return False
    # Debe tener un hijo 'se' con dep_ 'expl' (expletivo)
    tiene_se = any(
        c.lemma_ == "se" and c.pos_ == "PRON" and c.dep_ == "expl"
        for c in verbo_token.children
    )
    return tiene_se


def es_media_se(verbo_token: Any) -> bool:
    """
    Detecta voz media con 'se': verbo (normalmente ergativo) con sujeto afectado,
    sin agente. Ej: 'La puerta se abrió'.
    """
    if verbo_token.pos_ != "VERB":
        return False
    # Debe tener un hijo 'se' con dep_ 'obj' o 'expl'
    tiene_se = any(
        c.lemma_ == "se" and c.pos_ == "PRON" and c.dep_ in ("obj", "expl")
        for c in verbo_token.children
    )
    if not tiene_se:
        return False
    # No debe tener objeto directo (si lo tiene, sería pasiva refleja)
    tiene_objeto = any(c.dep_ == "obj" for c in verbo_token.children)
    if tiene_objeto:
        return False
    # Debe tener un sujeto (nsubj) que sea afectado (no podemos distinguir agente fácilmente)
    tiene_sujeto = any("subj" in c.dep_ for c in verbo_token.children)
    if not tiene_sujeto:
        # Si no tiene sujeto, podría ser impersonal, pero ya lo cubrimos antes
        return False
    return True


def obtener_numero_por_concordancia(verbo_token: Any) -> Optional[str]:
    """
    Intenta inferir el número gramatical del verbo a partir de su sujeto explícito.
    Recorre los hijos del verbo buscando dependencias de sujeto (activo o pasivo).
    Retorna 'Sing', 'Plur' o None.
    """
    for child in verbo_token.children:
        if child.dep_ in ACTIVE_SUBJECT_DEPS:
            num = child.morph.get("Number")
            if num:
                return num[0]
    return None


def obtener_genero_para_participio(verbo_token: Any) -> Optional[str]:
    """
    Para participios, intenta obtener el género del sujeto u objeto con el que concuerdan.
    En español, los participios en construcciones pasivas o perfectivas pueden concordar
    en género con el sujeto (voz pasiva) o con el objeto (voz activa con 'haber').
    """
    # Buscar sujeto pasivo
    for child in verbo_token.children:
        if child.dep_ == PASSIVE_SUBJECT_DEP:
            gen = child.morph.get("Gender")
            if gen:
                return gen[0]
    # Buscar objeto directo
    for child in verbo_token.children:
        if child.dep_ == "obj":
            gen = child.morph.get("Gender")
            if gen:
                return gen[0]
    return None


def verificar_concordancia_participio(
    verbo_token: Any, genero: Optional[str], numero: Optional[str]
) -> Optional[str]:
    """
    Comprueba concordancia de género/número entre participio y su sujeto.
    Retorna mensaje de error o None.
    """
    if verbo_token.pos_ != "VERB":
        return None
    vf = verbo_token.morph.get("VerbForm")
    if not vf or vf[0] != "Part":
        return None

    # Buscar sujeto
    sujeto = None
    for child in verbo_token.children:
        if "subj" in child.dep_:
            sujeto = child
            break
    if sujeto is None:
        return None

    gen_suj = (sujeto.morph.get("Gender") or [None])[0]
    num_suj = (sujeto.morph.get("Number") or [None])[0]
    errores = []
    if genero and gen_suj and genero != gen_suj:
        errores.append(f"género ({genero} vs {gen_suj})")
    if numero and num_suj and numero != num_suj:
        errores.append(f"número ({numero} vs {num_suj})")
    if errores:
        return f"Participio '{verbo_token.text}' no concuerda con sujeto '{sujeto.text}' en {', '.join(errores)}"
    return None


# ------------------------------------------------------------------------------
# OPCIONAL: funciones para construir matchers, etc., si se necesita
# ------------------------------------------------------------------------------
def build_adverb_phrase_matcher(nlp: Any) -> Any:
    """
    Construye un PhraseMatcher de spaCy con los adverbios multi-palabra.
    """
    from spacy.matcher import PhraseMatcher

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    phrases = [p for cat in MULTI_WORD_ADVERBS.values() for p in cat]
    matcher.add("MULTI_ADV", [nlp.make_doc(p) for p in phrases])
    return matcher


def build_discourse_matcher(nlp: Any) -> Any:
    """
    Construye un PhraseMatcher para las locuciones discursivas.
    """
    from spacy.matcher import PhraseMatcher

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for cat, frases in LOCUCIONES_DISCURSIVAS.items():
        matcher.add(cat, [nlp.make_doc(f) for f in frases])
    return matcher


# Offset field map per annotation type — used by the exporter
OFFSET_FIELDS: Dict[str, List[str]] = {
    "negaciones": ["char_start", "char_end", "alcance_char_start", "alcance_char_end"],
    "pronombres": ["char_start", "char_end"],
    "verbos": ["char_start", "char_end"],
    "cuantificadores": [
        "char_start",
        "char_end",
        "cuantificado_char_start",
        "cuantificado_char_end",
    ],
    "adverbios": ["char_start", "char_end"],
    "marcadores_discursivos": ["char_start", "char_end"],
    "insubordinaciones": ["char_start", "char_end"],
    "rarezas": ["char_start", "char_end"],
}


def uce_to_global_annotations(uce: "UCE") -> Dict:
    """
    Returns a dict of ALL annotations for one UCE,
    with every char offset converted to GLOBAL (document-level).
    This is the canonical representation for JSON export and Streamlit.

    Shape:
    {
        "uce_id":    str,
        "texto":     str,
        "start_char": int,   # global
        "end_char":   int,   # global
        "annotations": [
            {
                "type":        str,   # "negacion"|"pronombre"|"verbo"| ...
                "subtype":     str,   # tipo, categoria, pos, ...
                "text":        str,   # surface text of the span
                "start_char":  int,   # GLOBAL
                "end_char":    int,   # GLOBAL
                "attributes":  dict,  # all other fields
                "css_class":   str,   # for Streamlit/HTML highlight
            },
            ...
        ],
        "coref_chains": [...],        # already global from CoreferenceResolver
        "predicate_frames": [...],    # already global from SpanAnnotationIndex
        "metrics": {...},
    }
    """
    base = uce.start_char
    annotations = []

    # ── Helper ───────────────────────────────────────────────────
    def emit(
        ann_type: str,
        item: Dict,
        subtype_key: str,
        extra_offset_fields: List[str] = None,
        css: str = None,
    ):
        s = item.get("char_start")
        e = item.get("char_end")
        if s is None or e is None:
            return
        g_s = s + base
        g_e = e + base
        attrs = {k: v for k, v in item.items() if k not in ("char_start", "char_end")}
        # Shift any secondary offset fields
        for f in extra_offset_fields or []:
            if f in attrs and attrs[f] is not None:
                attrs[f] = attrs[f] + base
        annotations.append(
            {
                "type": ann_type,
                "subtype": item.get(subtype_key, ""),
                "text": item.get("texto") or item.get("text") or "",
                "start_char": g_s,
                "end_char": g_e,
                "attributes": attrs,
                "css_class": css or ann_type,
            }
        )

    # ── Negaciones ───────────────────────────────────────────────
    for n in uce.negaciones:
        emit(
            "negacion",
            n,
            "tipo",
            extra_offset_fields=["alcance_char_start", "alcance_char_end"],
            css=f"neg_{n.get('tipo', '').lower()}",
        )
        # Scope span as separate annotation
        if n.get("alcance_char_start") is not None:
            annotations.append(
                {
                    "type": "alcance_negacion",
                    "subtype": n.get("tipo", ""),
                    "text": n.get("alcance", ""),
                    "start_char": n["alcance_char_start"] + base,
                    "end_char": n["alcance_char_end"] + base,
                    "attributes": {"negacion_texto": n["texto"]},
                    "css_class": "alcance_neg",
                }
            )

    # ── Pronombres ───────────────────────────────────────────────
    for p in uce.pronombres:
        emit("pronombre", p, "subtipo", css=f"pron_{p.get('tipo', '').lower()}")

    # ── Verbos ───────────────────────────────────────────────────
    for v in uce.verbos:
        emit("verbo", v, "modo", css=f"verbo_{v.get('voz', 'act').lower()}")

    # ── Cuantificadores ──────────────────────────────────────────
    for c in uce.cuantificadores:
        emit(
            "cuantificador",
            c,
            "tipo",
            extra_offset_fields=["cuantificado_char_start", "cuantificado_char_end"],
            css="cuant",
        )
        if c.get("cuantificado_char_start") is not None:
            annotations.append(
                {
                    "type": "cuantificado",
                    "subtype": c.get("tipo", ""),
                    "text": c.get("cuantifica_a", ""),
                    "start_char": c["cuantificado_char_start"] + base,
                    "end_char": c["cuantificado_char_end"] + base,
                    "attributes": {"cuantificador": c["texto"]},
                    "css_class": "cuantificado",
                }
            )

    # ── Adverbios ────────────────────────────────────────────────
    for a in uce.adverbios:
        emit("adverbio", a, "categoria", css=f"adv_{a.get('categoria', '').lower()}")

    # ── Marcadores discursivos ───────────────────────────────────
    for m in uce.marcadores_discursivos:
        emit(
            "marcador_discursivo",
            m,
            "categoria",
            css=f"marc_{m.get('categoria', '').lower()}",
        )

    # ── Insubordinaciones ────────────────────────────────────────
    for ins in uce.insubordinaciones:
        emit("insubordinacion", ins, "tipo", css="insub")

    # ── Rarezas ──────────────────────────────────────────────────
    for r in uce.rarezas:
        emit("rareza", r, "tipo", css="rareza")

    # ── Correferencias (already global) ─────────────────────────
    # Emit one annotation per mention (span in original doc)
    coref_anns = []
    for chain in uce.coref_chains:
        rep = chain.get("representative", "")
        cid = chain.get("cluster_id", -1)  # if assigned
        label = chain.get("cluster_label", "")
        for mention in chain.get("mentions", []):
            # Only emit if this mention falls in THIS uce
            m_s = mention.get("start_char")
            m_e = mention.get("end_char")
            if m_s is None or m_e is None:
                continue
            # mention offsets from CoreferenceResolver are ALREADY global
            in_uce = uce.start_char <= m_s < uce.end_char
            annotations.append(
                {
                    "type": "coref_mention",
                    "subtype": rep,
                    "text": mention.get("text", ""),
                    "start_char": m_s,
                    "end_char": m_e,
                    "attributes": {
                        "representative": rep,
                        "chain_cluster_id": cid,
                        "chain_cluster_label": label,
                        "in_current_uce": in_uce,
                    },
                    "css_class": "coref_mention",
                }
            )
            coref_anns.append(mention)

    # ── Predicate frames (SpanAnnotation, already global) ────────
    frame_anns = []
    for frame in getattr(uce, "predicate_frames", []):
        # Deserialize if stored as dict (happens after DB round-trip or ejecutar() serialization)
        if isinstance(frame, dict):
            frame = PredicateFrame.from_dict(frame)

        # entity span
        frame_anns.append(
            {
                "type": "frame_entity",
                "subtype": frame.thematic_role,
                "text": frame.entity_text,
                "start_char": frame.entity_start_char,  # global
                "end_char": frame.entity_end_char,
                "attributes": {
                    "chain": frame.chain_representative,
                    "cluster_id": frame.cluster_id,
                    "cluster_label": frame.cluster_label,
                    "frame_fingerprint": frame.frame_fingerprint,
                    "is_expansion": frame.is_expansion,
                },
                "css_class": f"frame_entity role_{frame.thematic_role.lower()}",
            }
        )
        # verb span
        frame_anns.append(
            {
                "type": "frame_verb",
                "subtype": frame.voice,
                "text": frame.verb_text,
                "start_char": frame.verb_start_char,
                "end_char": frame.verb_end_char,
                "attributes": {
                    "verb_lemma": frame.verb_lemma,
                    "tense": frame.tense,
                    "mood": frame.mood,
                    "negated": frame.negated,
                    "cluster_id": frame.cluster_id,
                    "cluster_label": frame.cluster_label,
                    "frame_fingerprint": frame.frame_fingerprint,
                },
                "css_class": f"frame_verb",
            }
        )
        # object span (if present)
        if frame.direct_object_start is not None:
            frame_anns.append(
                {
                    "type": "frame_object",
                    "subtype": "direct_object",
                    "text": frame.direct_object or "",
                    "start_char": frame.direct_object_start,
                    "end_char": frame.direct_object_end,
                    "attributes": {
                        "object_lemma": frame.direct_object_lemma,
                        "cluster_id": frame.cluster_id,
                        "cluster_label": frame.cluster_label,
                        "frame_fingerprint": frame.frame_fingerprint,
                    },
                    "css_class": "frame_object",
                }
            )

    annotations.extend(frame_anns)

    # ── Sort by start_char (stable for Streamlit highlight injection) ──
    annotations.sort(key=lambda a: (a["start_char"], a["end_char"]))

    # ── Token surprisals: keep as {global_char_start: surprisal} ────
    global_surprisals = {(k + base): v for k, v in uce.token_surprisals.items()}

    return {
        "uce_id": uce.id,
        "texto": uce.texto,
        "start_char": uce.start_char,
        "end_char": uce.end_char,
        "annotations": annotations,
        "coref_chains": uce.coref_chains,  # full chain dicts (global)
        "predicate_frames": [
            (f.to_dict() if hasattr(f, "to_dict") else f)
            for f in getattr(uce, "predicate_frames", [])
        ],
        "metrics": {
            **uce.metricas_lexicas,
            "diversidad_semantica": uce.diversidad_semantica,
            "topic_shift_prev": uce.topic_shift_prev,
            "complejidad_sintactica": uce.complejidad_sintactica,
            "registro": uce.registro,
        },
        "token_surprisals": global_surprisals,
    }
