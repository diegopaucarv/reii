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
    GRAMMAR_CONFIG_PATH,
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
    if IA_DIR.is_dir():
        for p in sorted(IA_DIR.glob("*.json")):
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
WORKFLOW_NAMES_PATH = str(IA_DIR / "_workflow_names.json")


def list_workflow_files() -> List[str]:
    """Return sorted filenames of discourse workflow JSONs in ``ia/``.

    A workflow is a dict-of-agents JSON file. The grammar summarizer
    (``GRAMMAR_CONFIG_PATH``, e.g. ``0.json``) and any ``_``-prefixed
    metadata files are excluded.
    """
    grammar_basename = os.path.basename(GRAMMAR_CONFIG_PATH)
    out: List[str] = []
    if not IA_DIR.is_dir():
        return out
    for p in sorted(IA_DIR.glob("*.json")):
        if p.name.startswith("_") or p.name == grammar_basename:
            continue
        data = _read_json(str(p))
        if isinstance(data, dict):
            out.append(p.name)
    return out


def get_workflow_agents(filename: str) -> List[str]:
    """Return the agent names inside a workflow JSON file."""
    data = _read_json(str(IA_DIR / filename))
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
