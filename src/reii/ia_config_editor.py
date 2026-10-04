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
