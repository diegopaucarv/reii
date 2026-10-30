"""Política de formas activas y caché de tokens (fidelidad léxica ALCESTE).

Concentra dos piezas que antes vivían cableadas en ``SegmentadorALCESTE``:

1. ``retention_policy``: knob declarativo que decide **qué tokens son formas
   activas** y cuáles van a ``marcadores``. Sustituye al par implícito
   ``content_pos`` + ``is_stop`` de ``lematizar_uces``.
2. ``TokenTableWriter``: persistencia por UCE de ``{forma, lemma, pos,
   is_stop, is_alpha}`` de **todos** los tokens, invalidada por
   ``corpus_fingerprint``. Permite reevaluar políticas sin volver a correr
   Stanza (~13 s/UCE).

Análisis, regla pre-registrada y veredicto: ``docs/design_v5/RETENTION_POLICY.md``.

Precedencia del knob (documentada para NO repetir el footgun de
``Config.content_pos``, que ``load_gramatica`` pisa siempre):

    Config.retention_policy (si no es None)  >  ia/gramatica.json  >  "tool_adverb_filtered"

El default es la política conservadora (``tool_adverb_filtered``), adoptada por
la enmienda del 2026-10-28 (``RETENTION_POLICY.md`` §10). ``pos_only`` sigue
disponible como elección explícita.

``content_pos`` sigue viniendo de ``load_gramatica`` (JSON > default) y
describe el **núcleo** categorial; ``PROPN``/``X`` no se declaran allí: los
gobierna la política.
"""

from __future__ import annotations

import json
import os
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Set

RETENTION_POLICIES = ("pos_only", "pos_plus_propn", "tool_adverb_filtered")
DEFAULT_RETENTION_POLICY = "tool_adverb_filtered"

# POS del núcleo activo. PROPN/X se añaden por política, nunca por content_pos.
BASE_CONTENT_POS = ("NOUN", "VERB", "ADJ", "ADV")
PROPN_LIKE_POS = ("PROPN", "X")

# Clase cerrada de palabras funcionales: ALCESTE las trata como suplementarias.
CLOSED_MARKER_POS = frozenset({"PRON", "SCONJ", "CCONJ", "DET", "AUX", "ADP", "INTJ"})

# Lista cerrada de adverbios-herramienta (negación, grado escalar, tiempo y
# dícticos, foco). Sólo se aplica con retention_policy="tool_adverb_filtered"
# y sólo a tokens con POS == "ADV". Pre-registrada en RETENTION_POLICY.md §2.1.
# Se compara plegada (minúsculas, sin tildes): ver ``fold``.
TOOL_ADVERB_LEMMAS: Set[str] = frozenset(
    {
        # negación
        "no",
        "ni",
        "nada",
        "nadie",
        "nunca",
        "jamas",
        "tampoco",
        # grado escalar
        "muy",
        "mas",
        "menos",
        "tan",
        "tanto",
        "demasiado",
        "bastante",
        "poco",
        "casi",
        "apenas",
        # tiempo y dícticos
        "ya",
        "todavia",
        "aun",
        "antes",
        "despues",
        "ahora",
        "hoy",
        "ayer",
        "manana",
        "siempre",
        "luego",
        "entonces",
        "mientras",
        "temprano",
        "tarde",
        # foco / adición / comparación
        "solo",
        "solamente",
        "tambien",
        "incluso",
        "ademas",
        "mismo",
        "igual",
        "siquiera",
    }
)

TOKEN_TABLE_VERSION = 1


def fold(value: Any) -> str:
    """Minúsculas + sin diacríticos (el corpus se normaliza sin tildes)."""
    if not text_like(value):
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value).lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def text_like(value: Any) -> bool:
    """True si ``value`` es una cadena con contenido."""
    return isinstance(value, str) and bool(value)


def normalize_lemmas(values: Iterable[str]) -> frozenset:
    """Pliega una colección de lemas para comparaciones robustas."""
    return frozenset(fold(value) for value in values if value)


def validate_retention_policy(policy: Any) -> str:
    """Normaliza y valida una política; lanza ``ValueError`` si no existe."""
    value = str(policy).strip().lower() if policy is not None else ""
    if value not in RETENTION_POLICIES:
        raise ValueError(
            "retention_policy desconocida: {!r}. Válidas: {}".format(
                policy, ", ".join(RETENTION_POLICIES)
            )
        )
    return value


def resolve_retention_policy(
    explicit: Optional[str] = None, gramatica: Optional[str] = None
) -> str:
    """Resuelve la política efectiva: Config > ia/gramatica.json > default."""
    for value in (explicit, gramatica):
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return validate_retention_policy(value)
    return DEFAULT_RETENTION_POLICY


def policy_content_pos(policy: str, base: Optional[Iterable[str]] = None) -> List[str]:
    """POS activos según la política, a partir del núcleo ``base``.

    ``PROPN``/``X`` se ignoran en ``base``: su inclusión la decide la política,
    no la lista ``content_pos`` (única fuente de verdad, sin el footgun de
    tener dos sitios que se contradicen).
    """
    policy = validate_retention_policy(policy)
    core = [pos for pos in (base or BASE_CONTENT_POS) if pos not in PROPN_LIKE_POS]
    if not core:
        core = list(BASE_CONTENT_POS)
    if policy == "pos_only":
        return core
    return core + list(PROPN_LIKE_POS)


def policy_uses_is_stop(policy: str) -> bool:
    """``is_stop`` es criterio de marcador sólo en la política baseline."""
    return validate_retention_policy(policy) == "pos_only"


def policy_filters_tool_adverbs(policy: str) -> bool:
    """True si la política excluye la lista cerrada de adverbios-herramienta."""
    return validate_retention_policy(policy) == "tool_adverb_filtered"


def is_tool_adverb(
    policy: str,
    pos: str,
    lemma: str,
    tool_adverb_lemmas: Optional[Iterable[str]] = None,
) -> bool:
    """True si ``(pos, lemma)`` es un adverbio-herramienta de la lista cerrada.

    Requiere ``pos == "ADV"``: el mismo lema con otro POS es contenido legítimo
    (p.ej. "solo" adjetivo = 'alone', "mucho" cuantificador nominal).
    """
    if not policy_filters_tool_adverbs(policy):
        return False
    if pos != "ADV":
        return False
    lemmas = (
        tool_adverb_lemmas if tool_adverb_lemmas is not None else TOOL_ADVERB_LEMMAS
    )
    return fold(lemma) in lemmas


def token_decision(
    *,
    policy: str,
    pos: str,
    lemma: str,
    is_stop: bool,
    is_alpha: bool,
    content_pos: Optional[Iterable[str]] = None,
    is_forced_supp: bool = False,
    is_extra_stop: bool = False,
    tool_adverb_lemmas: Optional[Iterable[str]] = None,
) -> str:
    """Decide ``"active"``, ``"marker"`` o ``"drop"`` para un token.

    Reproduce exactamente la cascada de ``lematizar_uces``: ``palabras_excluir``
    (drop) y ``palabras_activas`` (POS forzado) se resuelven en el llamador.
    """
    policy = validate_retention_policy(policy)
    if is_forced_supp or is_extra_stop:
        return "marker"

    uses_stop = policy_uses_is_stop(policy)
    if is_alpha and ((uses_stop and is_stop) or pos in CLOSED_MARKER_POS):
        return "marker"
    if is_alpha and is_tool_adverb(policy, pos, lemma, tool_adverb_lemmas):
        return "marker"

    active_pos = (
        set(content_pos) if content_pos is not None else set(policy_content_pos(policy))
    )
    if pos in active_pos and is_alpha and (not uses_stop or not is_stop):
        return "active"
    return "drop"


def token_rows(doc: Iterable[Any]) -> List[Dict[str, Any]]:
    """Convierte un ``Doc`` (o cualquier iterable de tokens) en filas planas.

    La tabla incluye **todos** los tokens —también PUNCT y los que luego se
    excluyen—: es la materia prima para reevaluar políticas sin Stanza.
    """
    rows: List[Dict[str, Any]] = []
    for token in doc:
        rows.append(
            {
                "forma": token.text,
                "lemma": (token.lemma_ or "").lower(),
                "pos": token.pos_,
                "is_stop": bool(token.is_stop),
                "is_alpha": bool(token.is_alpha),
            }
        )
    return rows


class TokenTableWriter:
    """Escribe/actualiza un JSONL con una fila por UCE.

    Formato (LF, UTF-8)::

        {"type": "header", "version": 1, "corpus_fingerprint": "..."}
        {"type": "uce", "uce_id": "...", "words": 35,
         "tokens": [{forma, lemma, pos, is_stop, is_alpha}, ...]}

    Si la cabecera no coincide con ``corpus_fingerprint`` el fichero se
    reescribe (misma semántica de invalidación que ``best_params.json``).
    ``add`` es idempotente por ``uce_id``.

    Ningún fallo de I/O puede tumbar el pipeline: ante error se desactiva.
    """

    def __init__(self, path: Optional[str], fingerprint: str, enabled: bool = True):
        self.path = path
        self.fingerprint = fingerprint or ""
        self.enabled = bool(enabled and path)
        self._fh = None
        self._done: Set[str] = set()
        if not self.enabled:
            return
        try:
            self._open()
        except OSError:
            self.enabled = False
            self._fh = None

    def _open(self) -> None:
        assert self.path is not None  # garantizado por __init__
        header_ok = False
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                first = fh.readline()
                header = {}
                try:
                    header = json.loads(first)
                except (TypeError, ValueError):
                    header = {}
                if isinstance(header, dict):
                    header_ok = (
                        header.get("type") == "header"
                        and header.get("version") == TOKEN_TABLE_VERSION
                        and header.get("corpus_fingerprint") == self.fingerprint
                    )
                if header_ok:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(record, dict) and record.get("type") == "uce":
                            self._done.add(str(record.get("uce_id")))

        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        if header_ok:
            self._fh = open(self.path, "a", encoding="utf-8", newline="\n")
        else:
            self._fh = open(self.path, "w", encoding="utf-8", newline="\n")
            header_record = {
                "type": "header",
                "version": TOKEN_TABLE_VERSION,
                "corpus_fingerprint": self.fingerprint,
            }
            self._fh.write(json.dumps(header_record, ensure_ascii=False) + "\n")
            self._fh.flush()

    def add(
        self,
        uce_id: Optional[str],
        rows: Iterable[Dict[str, Any]],
        words: Optional[int] = None,
    ) -> None:
        if not self.enabled or self._fh is None:
            return
        key = str(uce_id) if uce_id is not None else ""
        if not key or key in self._done:
            return
        try:
            record = {
                "type": "uce",
                "uce_id": key,
                "words": words,
                "tokens": list(rows),
            }
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
            self._done.add(key)
        except (OSError, TypeError, ValueError):
            self.enabled = False

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __enter__(self) -> "TokenTableWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_token_table(
    path: str, fingerprint: str
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Lee la tabla de tokens; ``None`` si falta o la huella no coincide.

    Devuelve ``{uce_id: {"words": int | None, "tokens": [fila, ...]}}``.
    ``None`` (y no ``{}``) distingue "cache inválida" de "cache vacía".
    """
    if not path or not os.path.exists(path):
        return None
    table: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.readline()
        try:
            header = json.loads(first)
        except (TypeError, ValueError):
            return None
        if not isinstance(header, dict):
            return None
        if (
            header.get("type") != "header"
            or header.get("version") != TOKEN_TABLE_VERSION
            or header.get("corpus_fingerprint") != (fingerprint or "")
        ):
            return None
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict) or record.get("type") != "uce":
                continue
            table[str(record.get("uce_id"))] = {
                "words": record.get("words"),
                "tokens": list(record.get("tokens") or []),
            }
    return table
