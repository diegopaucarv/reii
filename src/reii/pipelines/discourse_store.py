"""Persistencia del estado del análisis de discurso en PostgreSQL + checkpoints.

Reemplaza el JSON ``discourse_state.json`` por una fila única en la tabla
``discourse_state`` y añade un mecanismo de checkpoints en
``discourse_checkpoints``. La API es la que consume la UI:

    load_state() -> Dict
    save_state(state) -> None
    save_checkpoint(name, state) -> int
    list_checkpoints() -> List[Dict]
    restore_checkpoint(checkpoint_id) -> Dict
    delete_checkpoint(checkpoint_id) -> None
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List

from reii.backend.sql.db import get_conn

_STATE_ID = 1


def _to_serialisable(c: Any) -> Any:
    """Convierte Counter / set a estructuras planas para que sobrevivan a json.dumps."""
    if isinstance(c, Counter):
        return dict(c)
    if isinstance(c, set):
        return sorted(c)
    if isinstance(c, dict):
        return {k: _to_serialisable(v) for k, v in c.items()}
    if isinstance(c, list):
        return [_to_serialisable(i) for i in c]
    return c


def _dumps(state: Dict) -> str:
    return json.dumps(_to_serialisable(state), ensure_ascii=False)


def _loads(raw: str) -> Dict:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_state() -> Dict:
    """Devuelve el estado persistido (fila id=1) o ``{}`` si no existe."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT state_json FROM discourse_state WHERE id = %s", (_STATE_ID,)
        ).fetchone()
    if row is None:
        return {}
    return _loads(row["state_json"])


def save_state(state: Dict) -> None:
    """Upserta el estado serializado en la fila única ``discourse_state``."""
    payload = _dumps(state)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO discourse_state (id, state_json, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (id) DO UPDATE
                SET state_json = EXCLUDED.state_json,
                    updated_at = CURRENT_TIMESTAMP
            """,
            (_STATE_ID, payload),
        )


def save_checkpoint(name: str, state: Dict) -> int:
    """Guarda un checkpoint y devuelve su ``checkpoint_id``."""
    payload = _dumps(state)
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO discourse_checkpoints (name, state_json, created_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            RETURNING checkpoint_id
            """,
            (name, payload),
        ).fetchone()
    return int(row["checkpoint_id"])


def list_checkpoints() -> List[Dict]:
    """Lista los checkpoints ordenados por creación (más reciente primero)."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT checkpoint_id, name, created_at, state_json
            FROM discourse_checkpoints
            ORDER BY created_at DESC, checkpoint_id DESC
            """
        ).fetchall()
    out: List[Dict] = []
    for r in rows:
        state = _loads(r["state_json"])
        annos = state.get("annotations_by_uce", {})
        n_uces = len(annos) if isinstance(annos, dict) else 0
        out.append(
            {
                "checkpoint_id": int(r["checkpoint_id"]),
                "name": r["name"],
                "created_at": r["created_at"],
                "n_uces": n_uces,
            }
        )
    return out


def restore_checkpoint(checkpoint_id: int) -> Dict:
    """Devuelve el ``state_json`` del checkpoint indicado (``{}`` si no existe)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT state_json FROM discourse_checkpoints WHERE checkpoint_id = %s",
            (checkpoint_id,),
        ).fetchone()
    if row is None:
        return {}
    return _loads(row["state_json"])


def delete_checkpoint(checkpoint_id: int) -> None:
    """Elimina el checkpoint indicado."""
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM discourse_checkpoints WHERE checkpoint_id = %s",
            (checkpoint_id,),
        )
