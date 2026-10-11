"""Puente entre las anotaciones del discurso y la base de datos.

Problema que resuelve: el análisis de discurso lee ``workflow_data.json`` y escribe
``discourse_state.json``; la tabla ``annotations`` solo se reconstruye cuando corre la
clasificación y descarta las anotaciones cuyo id no coincide con ``uces.uce_id``.
Aquí:
  * ``build_id_resolver`` reconcilia los esquemas de id (``__mfN``, ``doc_sección_idx``).
  * ``sync_state_to_db`` escribe las anotaciones resueltas en ``annotations``.
  * ``load_annotations_by_uce`` las lee desde la BD (misma forma que el JSON de estado).
  * ``export_discourse_input`` genera desde la BD la entrada del análisis (mismos ids).

``conn`` es una conexión psycopg (dict_row) o el adaptador SQLite de reii.quality.store.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional

_MF = re.compile(r"__mf\d+$")


def strip_mf(raw_id: Any) -> str:
    return _MF.sub("", str(raw_id))


def build_id_resolver(uce_rows: Iterable[Dict[str, Any]]) -> Callable[[Any], Optional[str]]:
    """Devuelve ``resolve(raw_id) -> uce_id | None``.

    Acepta: el id tal cual, sin sufijo ``__mfN``, ``{doc}_{sección}_{idx}`` y
    ``{doc}_{idx}`` (este último solo si no es ambiguo entre secciones).
    """
    index: Dict[str, str] = {}
    short: Dict[str, set] = defaultdict(set)
    for r in uce_rows:
        uid = str(r["uce_id"])
        index[uid] = uid
        doc, loc = r.get("doc_id"), r.get("local_idx")
        sec = r.get("section_id", 0)
        if doc is not None and loc is not None:
            index.setdefault(f"{doc}_{sec}_{loc}", uid)
            short[f"{doc}_{loc}"].add(uid)
    for key, ids in short.items():
        if len(ids) == 1 and key not in index:
            index[key] = next(iter(ids))

    def resolve(raw_id: Any) -> Optional[str]:
        raw = str(raw_id)
        return index.get(raw) or index.get(strip_mf(raw))

    return resolve


def _fetch_uce_rows(conn) -> List[Dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT uce_id, doc_id, local_idx, section_id FROM uces").fetchall()]


def _flatten(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for raw_id, anns in (state.get("annotations_by_uce") or {}).items():
        for a in anns or []:
            if isinstance(a, dict):
                out.append({"_raw_id": str(raw_id), **a})
    return out


def _first_quote(a: Dict[str, Any]) -> Optional[str]:
    if a.get("quote"):
        return a["quote"]
    for s in a.get("spans") or []:
        if isinstance(s, dict) and s.get("quote"):
            return s["quote"]
    return None


def sync_state_to_db(conn, state: Dict[str, Any], replace: str = "agents", tx=None) -> Dict[str, Any]:
    """Escribe las anotaciones del estado en ``annotations`` resolviendo los ids.

    replace="agents": borra solo las filas de los agentes presentes en el estado.
    replace="all":    reconstrucción total (equivale a migrate_annotations).
    Devuelve un informe con conteos y ejemplos de ids sin resolver.
    """
    flat = _flatten(state)
    resolve = build_id_resolver(_fetch_uce_rows(conn))
    rows, unmatched, seen = [], Counter(), set()
    for a in flat:
        uid = resolve(a["_raw_id"])
        if uid is None:
            unmatched[a["_raw_id"]] += 1
            continue
        payload = {k: v for k, v in a.items() if k != "_raw_id"}
        payload["uce_id"] = uid
        payload["source_uce_id"] = a["_raw_id"]
        key = (uid, payload.get("agent"), payload.get("trait"), json.dumps(payload.get("spans"), sort_keys=True, default=str), payload.get("quote"))
        if key in seen:
            continue
        seen.add(key)
        rows.append((uid, payload.get("agent") or payload.get("agent_name") or "unknown", payload.get("trait"),
                     _first_quote(payload), payload.get("confidence"), payload.get("subtype"),
                     json.dumps(payload, ensure_ascii=False, default=str)))
    agents = sorted({r[1] for r in rows} | {(a.get("agent") or "unknown") for a in flat})

    def _write():
        if replace == "all":
            conn.execute("DELETE FROM annotations")
        elif agents:
            conn.execute(f"DELETE FROM annotations WHERE agent_name IN ({','.join(['%s'] * len(agents))})", agents)
        for r in rows:
            conn.execute(
                "INSERT INTO annotations (uce_id, agent_name, trait, quote, confidence, subtype, payload_json) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)", r)

    if hasattr(conn, "transaction"):
        with conn.transaction():
            _write()
    else:
        _write()
    return {
        "n_state_uces": len(state.get("annotations_by_uce") or {}),
        "n_annotations": len(flat),
        "n_written": len(rows),
        "n_unmatched": sum(unmatched.values()),
        "unmatched_examples": list(unmatched)[:5],
        "agents": agents,
        "replace": replace,
    }


def load_annotations_by_uce(conn) -> Dict[str, List[Dict[str, Any]]]:
    """Anotaciones desde la BD, con la misma forma que ``annotations_by_uce`` del JSON de estado."""
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in conn.execute(
        "SELECT uce_id, agent_name, trait, quote, confidence, subtype, payload_json FROM annotations ORDER BY uce_id"
    ).fetchall():
        r = dict(r)
        try:
            ann = json.loads(r.get("payload_json") or "{}")
        except (TypeError, ValueError):
            ann = {}
        ann.setdefault("agent", r["agent_name"])
        ann.setdefault("trait", r["trait"])
        ann.setdefault("subtype", r["subtype"])
        ann.setdefault("confidence", r["confidence"] or "baja")
        if not ann.get("spans") and r.get("quote"):
            ann["spans"] = [{"uce_id": r["uce_id"], "quote": r["quote"], "start_char": -1, "end_char": -1}]
        ann["uce_id"] = r["uce_id"]
        out[r["uce_id"]].append(ann)
    return dict(out)


def export_discourse_input(adapter, path: str) -> Dict[str, int]:
    """Escribe desde la BD el JSON que lee el análisis de discurso (``uces`` + anotaciones).

    Los ids de las UCEs son exactamente ``uces.uce_id``, de modo que lo que se anota es
    lo que se muestra. Los metadatos del documento se fusionan en ``metadata``.
    """
    uces = adapter.uces()
    docs = adapter._doc_metadata()
    for u in uces:
        meta = dict(docs.get(str(u.get("doc_id")), {}))
        meta.update(u.get("metadata") or {})
        u["metadata"] = meta
        u.setdefault("id", u.get("uce_id"))
    payload = {"uces": uces, "annotations_by_uce": adapter_annotations(adapter)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return {"n_uces": len(uces), "n_uces_con_anotaciones": len(payload["annotations_by_uce"])}


def adapter_annotations(adapter) -> Dict[str, List[Dict[str, Any]]]:
    try:
        return adapter.annotations()
    except Exception:
        return {}
