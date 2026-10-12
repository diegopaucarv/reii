"""Pure JSON I/O + validation for the IA agent config editor.

Touches only ia/*.json. No heavy NLP imports.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Tuple

from reii.config import (
    DISCOURSE_CONFIG_PATH,
    DISCOURSE_DIR,
    ENTREVISTAS_DIR,
    GRAMMAR_CONFIG_PATH,
    GUIA_ENTREVISTA_PATH,
    IA_DIR,
)

AGENT_TEMPLATE: Dict[str, Any] = {
    "name": "",
    "role": "",
    "instructions": "",
    "json_schema": {"type": "object", "properties": {}},
    "few_shot_examples": [],
    "gram_cats": [],
    "disc_cats": [],
}


def _read_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_grammar_config() -> Dict[str, Any]:
    data = _read_json(GRAMMAR_CONFIG_PATH)
    return data if isinstance(data, dict) else {}


def load_discourse_config() -> Dict[str, Dict[str, Any]]:
    data = _read_json(DISCOURSE_CONFIG_PATH)
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def list_discourse_agent_names() -> List[str]:
    return sorted(load_discourse_config().keys())


def list_inactive_agent_files() -> List[str]:
    wired = {
        os.path.basename(GRAMMAR_CONFIG_PATH),
        os.path.basename(DISCOURSE_CONFIG_PATH),
    }
    out: List[str] = []
    if DISCOURSE_DIR.is_dir():
        for p in sorted(DISCOURSE_DIR.glob("*.json")):
            if p.name not in wired:
                out.append(p.name)
    return out


def _atomic_write_json(path: str, data: Any) -> None:
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_grammar_config(data: Dict[str, Any]) -> None:
    _atomic_write_json(GRAMMAR_CONFIG_PATH, data)


def save_discourse_config(data: Dict[str, Dict[str, Any]]) -> None:
    _atomic_write_json(DISCOURSE_CONFIG_PATH, data)


def load_entrevista_prompt(name: str) -> Dict[str, Any]:
    """name: 'segmentacion' | 'extraccion'. Lee ia/entrevistas/{name}.json; {} si falta/inválido."""
    data = _read_json(str(ENTREVISTAS_DIR / f"{name}.json"))
    return data if isinstance(data, dict) else {}


def save_entrevista_prompt(name: str, data: Dict[str, Any]) -> None:
    _atomic_write_json(str(ENTREVISTAS_DIR / f"{name}.json"), data)


def load_guia_entrevista() -> Dict[str, Any]:
    data = _read_json(GUIA_ENTREVISTA_PATH)
    return data if isinstance(data, dict) else {}


def save_guia_entrevista(data: Dict[str, Any]) -> None:
    _atomic_write_json(GUIA_ENTREVISTA_PATH, data)


# ─────────────────────────────────────────────
# Guía estructurada + sincronización del enum
# ─────────────────────────────────────────────
TEXTO_NO_ASIGNADO = "TEXTO_NO_ASIGNADO"

# Ruta al enum de ``seccion_entrevista`` dentro de segmentacion.json.
_SEGMENTACION_ENUM_PATH = (
    "json_schema",
    "properties",
    "segmentos",
    "items",
    "properties",
    "seccion_entrevista",
    "enum",
)


def _get_nested(data: Any, path: Tuple[str, ...]) -> Any:
    """Devuelve el valor anidado en ``data`` siguiendo ``path`` o ``None``."""
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _set_nested(data: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    """Asigna ``value`` en la ruta anidada de ``data`` (muta in-place)."""
    cur = data
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def sync_seccion_enum() -> Dict[str, Any]:
    """Sincroniza el enum ``seccion_entrevista`` de segmentacion.json con las
    secciones de guia.json.

    El enum resultante es ``secciones + ["TEXTO_NO_ASIGNADO"]`` (orden de la
    guía, sin duplicados). Devuelve un reporte ``{added, removed, unchanged}``;
    si falta o es inválido algún archivo, devuelve el reporte con ``error``.
    """
    report: Dict[str, Any] = {"added": [], "removed": [], "unchanged": []}

    guia = load_guia_entrevista()
    secciones = guia.get("secciones") if isinstance(guia, dict) else None
    if not isinstance(secciones, list) or not all(
        isinstance(s, str) and s.strip() for s in secciones
    ):
        report["error"] = "guia.json no tiene una lista válida de secciones."
        return report
    secciones = [s.strip() for s in secciones]

    seg = load_entrevista_prompt("segmentacion")
    enum = _get_nested(seg, _SEGMENTACION_ENUM_PATH)
    if not isinstance(enum, list):
        report["error"] = (
            "segmentacion.json no tiene el enum seccion_entrevista esperado."
        )
        return report

    # Dedupe preservando el orden de la guía.
    seen: set = set()
    secciones_dedup: List[str] = []
    for s in secciones:
        if s not in seen:
            seen.add(s)
            secciones_dedup.append(s)

    current = [v for v in enum if isinstance(v, str)]
    current_set = set(current)
    guia_set = set(secciones_dedup)

    report["added"] = [s for s in secciones_dedup if s not in current_set]
    report["removed"] = [
        v for v in current if v != TEXTO_NO_ASIGNADO and v not in guia_set
    ]
    report["unchanged"] = [s for s in secciones_dedup if s in current_set]

    nuevo_enum = secciones_dedup + [TEXTO_NO_ASIGNADO]
    _set_nested(seg, _SEGMENTACION_ENUM_PATH, nuevo_enum)
    save_entrevista_prompt("segmentacion", seg)
    return report


def validate_guia(data: Any) -> Tuple[bool, str]:
    """Valida la estructura de una guía de entrevista.

    Devuelve ``(True, "")`` si ``data`` es un dict con ``secciones`` (lista no
    vacía de strings no vacíos) y ``preguntas`` (lista de dicts con ``numero``
    int, ``seccion`` str y ``pregunta`` str); en caso contrario ``(False, msg)``.
    """
    if not isinstance(data, dict):
        return False, "La guía debe ser un objeto JSON."
    secciones = data.get("secciones")
    if not isinstance(secciones, list) or not secciones:
        return False, "La guía debe incluir una lista no vacía de secciones."
    if not all(isinstance(s, str) and s.strip() for s in secciones):
        return False, "Cada sección debe ser un texto no vacío."
    preguntas = data.get("preguntas")
    if not isinstance(preguntas, list):
        return False, "La guía debe incluir una lista de preguntas."
    for p in preguntas:
        if not isinstance(p, dict):
            return False, "Cada pregunta debe ser un objeto."
        if not isinstance(p.get("numero"), int):
            return False, "Cada pregunta debe tener un número entero."
        if not isinstance(p.get("seccion"), str):
            return False, "Cada pregunta debe tener una sección de texto."
        if not isinstance(p.get("pregunta"), str):
            return False, "Cada pregunta debe tener un texto de pregunta."
    return True, ""


def normalize_guia(data: Dict[str, Any]) -> Dict[str, Any]:
    """Devuelve una copia limpia de la guía sin escribir en disco.

    - Recorta espacios en nombres de sección y textos de pregunta.
    - Descarta preguntas vacías.
    - Renumera las preguntas 1..N en el orden dado.
    - Conserva ``pregunta.seccion`` aunque no esté en ``secciones``.
    """
    secciones = [
        s.strip()
        for s in (data.get("secciones") or [])
        if isinstance(s, str) and s.strip()
    ]
    preguntas: List[Dict[str, Any]] = []
    for p in data.get("preguntas") or []:
        if not isinstance(p, dict):
            continue
        texto = p.get("pregunta")
        if not isinstance(texto, str) or not texto.strip():
            continue
        seccion = p.get("seccion")
        if not isinstance(seccion, str):
            seccion = ""
        preguntas.append(
            {"numero": 0, "seccion": seccion.strip(), "pregunta": texto.strip()}
        )
    for i, p in enumerate(preguntas, start=1):
        p["numero"] = i
    return {"secciones": secciones, "preguntas": preguntas}


def parse_json_text(text: str) -> Tuple[bool, Any, str]:
    try:
        return True, json.loads(text), ""
    except json.JSONDecodeError as e:
        return False, None, f"JSON inválido: {e}"


def parse_list_text(text: str) -> List[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def to_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def to_list_text(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ", ".join(str(v) for v in values)


def load_discourse_state_summary(
    state_path: str = "",
) -> Dict[str, int]:
    """Count saved annotations per agent from the discourse state file.

    Returns ``{agent_name: count}``. Empty dict if the file is missing or
    unreadable. Used by the dashboard to show what is already processed.
    """
    if not state_path:
        from reii.config import DISCOURSE_STATE_PATH

        state_path = DISCOURSE_STATE_PATH
    data = _read_json(state_path)
    if not isinstance(data, dict):
        return {}
    counts: Dict[str, int] = {}
    for annos in data.get("annotations_by_uce", {}).values():
        if not isinstance(annos, list):
            continue
        for a in annos:
            if not isinstance(a, dict):
                continue
            agent = a.get("agent") or "?"
            counts[agent] = counts.get(agent, 0) + 1
    return counts


# ─────────────────────────────────────────────
# Workflows (one per discourse JSON in ia/)
# ─────────────────────────────────────────────
WORKFLOW_NAMES_PATH = str(DISCOURSE_DIR / "_workflow_names.json")


def list_workflow_files() -> List[str]:
    """Return sorted filenames of discourse workflow JSONs in ``analisis_discursivo/``.

    A workflow is a dict-of-agents JSON file. The grammar summarizer
    (``GRAMMAR_CONFIG_PATH``, e.g. ``resumidor_gramatical.json``) and any
    ``_``-prefixed metadata files are excluded.
    """
    grammar_basename = os.path.basename(GRAMMAR_CONFIG_PATH)
    out: List[str] = []
    if not DISCOURSE_DIR.is_dir():
        return out
    for p in sorted(DISCOURSE_DIR.glob("*.json")):
        if p.name.startswith("_") or p.name == grammar_basename:
            continue
        data = _read_json(str(p))
        if isinstance(data, dict):
            out.append(p.name)
    return out


def get_workflow_agents(filename: str) -> List[str]:
    """Return the agent names inside a workflow JSON file."""
    data = _read_json(str(DISCOURSE_DIR / filename))
    if not isinstance(data, dict):
        return []
    return sorted(data.keys())


def load_workflow_names() -> Dict[str, str]:
    """Return ``{filename: display_name}`` for workflows."""
    data = _read_json(WORKFLOW_NAMES_PATH)
    return data if isinstance(data, dict) else {}


def save_workflow_names(names: Dict[str, str]) -> None:
    _atomic_write_json(WORKFLOW_NAMES_PATH, names)


def load_discourse_state_progress(
    state_path: str = "",
) -> Tuple[int, Dict[str, int]]:
    """Return ``(total_uces, {agent: n_uces_annotated})`` from the state file.

    ``n_uces_annotated`` counts UCEs that have at least one annotation from
    that agent (not the number of spans). Used to compute per-workflow
    progress.
    """
    if not state_path:
        from reii.config import DISCOURSE_STATE_PATH

        state_path = DISCOURSE_STATE_PATH
    data = _read_json(state_path)
    if not isinstance(data, dict):
        return 0, {}
    ann_by_uce = data.get("annotations_by_uce", {})
    if not isinstance(ann_by_uce, dict):
        return 0, {}
    total_uces = len(ann_by_uce)
    by_agent: Dict[str, int] = {}
    for annos in ann_by_uce.values():
        if not isinstance(annos, list):
            continue
        agents = {
            a.get("agent") for a in annos if isinstance(a, dict) and a.get("agent")
        }
        for agent in agents:
            by_agent[agent] = by_agent.get(agent, 0) + 1
    return total_uces, by_agent
