"""
Migración JSON → SQLite.

Diseño:
  · Idempotente: INSERT OR REPLACE sobre PKs.
  · Transaccional: cada tabla va en su propia transacción.
  · No destructivo: nunca toca el JSON de entrada.
  · Reporta counts antes/después y aborta si no cuadran.

Uso:
    python -m reii.backend.migrate --json data/workflow_data.json --db data/workflow.db
    python -m reii.backend.migrate --json ... --db ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from reii.backend.sql.db import connect, transaction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def _json_or_empty(x: Any) -> str:
    if x is None:
        return "{}"
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _safe_str(x: Any) -> str:
    """
    str() sin la trampa de `or`: 0 y False son valores válidos,
    solo None y el string vacío colapsan a "".
    """
    if x is None:
        return ""
    return str(x)


def _json_or_list(x: Any) -> str:
    if x is None:
        return "[]"
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except (TypeError, ValueError):
        return "[]"


def _flatten_texto_completo(uce: Dict) -> Optional[str]:
    txt = uce.get("texto_completo_doc")
    if txt:
        return txt
    return None


# ════════════════════════════════════════════════════════════════════════════
# Migradores por tabla
# ════════════════════════════════════════════════════════════════════════════
def migrate_documents(
    conn: sqlite3.Connection, data: Dict, dry_run: bool = False
) -> int:
    """
    Extrae documentos únicos de:
      · data["doc_metadata"] — {str(doc_id): {metadata...}}
      · data["texto_completo_por_doc_id"] — {str(doc_id): texto}
      · data["uces"] — [{doc_id, ...}] (para doc_idx / orden)
    """
    doc_meta = data.get("doc_metadata", {}) or {}
    doc_texto = data.get("texto_completo_por_doc_id", {}) or {}
    uces = data.get("uces", []) or []

    # Agregar doc_ids conocidos
    all_ids: Dict[str, Dict] = {}
    for did, meta in doc_meta.items():
        all_ids[str(did)] = {"metadata": meta or {}, "texto": doc_texto.get(str(did))}
    for u in uces:
        did = str(u.get("doc_id"))
        if did not in all_ids:
            all_ids[did] = {"metadata": {}, "texto": None}

    # doc_idx / orden del primer UCE de cada doc
    for u in uces:
        did = str(u.get("doc_id"))
        if did in all_ids and "doc_idx" not in all_ids[did]:
            all_ids[did]["doc_idx"] = u.get("metadata", {}).get("doc_idx")
            all_ids[did]["orden"] = u.get("metadata", {}).get("indice_orden")
            all_ids[did]["origen"] = u.get("metadata", {}).get("origen")

    n_uces_per_doc: Dict[str, int] = {}
    for u in uces:
        did = str(u.get("doc_id"))
        n_uces_per_doc[did] = n_uces_per_doc.get(did, 0) + 1

    if dry_run:
        logger.info("[dry-run] documents: %d filas", len(all_ids))
        return len(all_ids)

    with transaction(conn):
        for did, payload in all_ids.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                  (doc_id, doc_idx, orden, origen, metadata_json,
                   texto_completo, n_uces)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    did,
                    int(payload.get("doc_idx") or 0),
                    payload.get("orden"),
                    payload.get("origen"),
                    _json_or_empty(payload.get("metadata")),
                    payload.get("texto"),
                    n_uces_per_doc.get(did, 0),
                ),
            )
    logger.info("documents: %d filas", len(all_ids))
    return len(all_ids)


def migrate_uces(conn: sqlite3.Connection, data: Dict, dry_run: bool = False) -> int:
    uces = data.get("uces", []) or []
    if dry_run:
        logger.info("[dry-run] uces: %d filas", len(uces))
        return len(uces)

    # El pipeline guarda phi_score en data["uce_phi"] como lista de
    # {uce_id, phi_score}, no dentro de cada UCE. Construimos un lookup
    # para inyectarlo al vuelo.
    phi_lookup: Dict[str, float] = {}
    for entry in data.get("uce_phi") or []:
        uid = entry.get("uce_id")
        score = entry.get("phi_score")
        if uid is not None and score is not None:
            phi_lookup[str(uid)] = float(score)
    # Columnas dinámicas que van a linguistic_json
    LINGUISTIC_KEYS = (
        "verbos",
        "pronombres",
        "adverbios",
        "negaciones",
        "verbos_enriquecido",
        "insubordinaciones",
        "rarezas",
        "marcadores_discursivos",
        "predicate_frames",
        "cuantificadores",
        "formas_tokens",
        "marcadores",
        "tokens",
        "bigrams",
        "trigrams",
        "bigram_stems",
        "trigram_stems",
        "pos_tags",
        "coref_chains",
        "subjects",
    )
    # Columnas que van a metrics_json
    METRIC_KEYS = (
        "metricas_lexicas",
        "complejidad_sintactica",
        "diversidad_semantica",
        "topic_shift_prev",
        "registro",
    )

    with transaction(conn):
        for u in uces:
            uce_id = str(u.get("id") or u.get("uce_id") or "")
            if not uce_id:
                continue

            # OJO: doc_id puede ser 0 (int), y `0 or ""` devuelve "".
            # Hay que comprobar None explícitamente.
            _raw_doc_id = u.get("doc_id")
            if _raw_doc_id is None:
                _raw_doc_id = (u.get("metadata") or {}).get("doc_idx")
            doc_id = str(_raw_doc_id) if _raw_doc_id is not None else ""

            linguistic = {k: u[k] for k in LINGUISTIC_KEYS if k in u}
            metrics = {k: u[k] for k in METRIC_KEYS if k in u}

            conn.execute(
                """
                INSERT OR REPLACE INTO uces (
                    uce_id, doc_id, local_idx, section_id, seccion, texto,
                    n_tokens, cluster_id, is_stable, stability_method,
                    is_terminal_consolidated,
                    projected_cluster_id, projection_distance,
                    projection_margin, projection_ratio,
                    phi_score, linguistic_json, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uce_id,
                    doc_id,
                    int(u.get("local_idx") or 0),
                    int((u.get("metadata") or {}).get("section_id") or 0),
                    u.get("seccion") or (u.get("metadata") or {}).get("seccion"),
                    u.get("texto") or "",
                    len((u.get("tokens") or [])) or None,
                    u.get("cluster_id"),
                    1 if u.get("is_stable") else 0,
                    u.get("stability_method"),
                    1 if u.get("is_terminal_consolidated") else 0,
                    u.get("projected_cluster_id"),
                    u.get("projection_distance"),
                    u.get("projection_margin"),
                    u.get("projection_ratio"),
                    # phi_score puede venir como float directo o dentro de phi_coefficients
                    phi_lookup.get(uce_id),
                    _json_or_empty(linguistic),
                    _json_or_empty(metrics),
                ),
            )
    logger.info("uces: %d filas", len(uces))
    return len(uces)


def migrate_ucs(conn: sqlite3.Connection, data: Dict, dry_run: bool = False) -> int:
    ucs = data.get("ucs", []) or []
    if dry_run:
        logger.info("[dry-run] ucs: %d filas", len(ucs))
        return len(ucs)

    with transaction(conn):
        for uc in ucs:
            uc_id = str(uc.get("id") or "")
            if not uc_id:
                continue
            lemmas = uc.get("lemmas") or []
            conn.execute(
                """
                INSERT OR REPLACE INTO ucs
                  (uc_id, doc_id, texto, cluster_id,
                   uce_ids_json, lemmas_json, n_lemmas)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uc_id,
                    _safe_str(
                        uc.get("doc_id")
                        if uc.get("doc_id") is not None
                        else uc.get("metadata", {}).get("doc_id")
                    ),
                    uc.get("texto") or "",
                    uc.get("cluster_id"),
                    _json_or_list(uc.get("uce_ids")),
                    _json_or_list(lemmas),
                    len(lemmas),
                ),
            )
    logger.info("ucs: %d filas", len(ucs))
    return len(ucs)


def migrate_clusters(
    conn: sqlite3.Connection, data: Dict, dry_run: bool = False
) -> int:
    """
    Deriva clusters desde:
      · uces[].cluster_id    → member_count
      · terms (significativos) → top_terms_json
      · kv_store["centroid_export"] → centroid_json
    """
    uces = data.get("uces", []) or []
    terms = data.get("terminos", []) or []
    centroids = (data.get("centroid_export") or {}).get("classes", {}) or {}

    member_count: Dict[int, int] = {}
    for u in uces:
        cid = u.get("cluster_id")
        if cid is not None:
            member_count[int(cid)] = member_count.get(int(cid), 0) + 1

    top_terms: Dict[int, List[Dict]] = {}
    for t in terms:
        cid = t.get("cluster")
        if cid is None:
            continue
        cid = int(cid)
        if not t.get("significativo"):
            continue
        top_terms.setdefault(cid, []).append(
            {"term": t.get("termino"), "phi": t.get("phi"), "chi2": t.get("chi2_yates")}
        )
    for cid in top_terms:
        top_terms[cid].sort(key=lambda x: abs(x.get("phi") or 0), reverse=True)

    all_cids = (
        set(member_count.keys())
        | set(top_terms.keys())
        | {int(k) for k in centroids.keys()}
    )

    if dry_run:
        logger.info("[dry-run] clusters: %d filas", len(all_cids))
        return len(all_cids)

    with transaction(conn):
        for cid in sorted(all_cids):
            conn.execute(
                """
                INSERT OR REPLACE INTO clusters
                  (cluster_id, cluster_name, member_count,
                   top_terms_json, centroid_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cid,
                    f"{cid:02d}",
                    member_count.get(cid, 0),
                    _json_or_list(top_terms.get(cid, [])[:50]),
                    _json_or_empty((centroids.get(str(cid)) or {}).get("centroid")),
                ),
            )
    logger.info("clusters: %d filas", len(all_cids))
    return len(all_cids)


def migrate_terms(conn: sqlite3.Connection, data: Dict, dry_run: bool = False) -> int:
    terms = data.get("terminos", []) or []
    if dry_run:
        logger.info("[dry-run] terms: %d filas", len(terms))
        return len(terms)

    with transaction(conn):
        for t in terms:
            term = t.get("termino")
            cid = t.get("cluster")
            if term is None or cid is None:
                continue
            conn.execute(
                """
                INSERT OR REPLACE INTO terms (
                    term, cluster_id, frecuencia_global, frecuencia_cluster,
                    chi2_yates, p_valor, p_adj, phi, cramer_v, c_value,
                    significativo, asignado_estricto
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(term),
                    int(cid),
                    int(t.get("frecuencia_global") or 0),
                    int(t.get("frecuencia_cluster") or 0),
                    t.get("chi2_yates"),
                    t.get("p_valor"),
                    t.get("p_adj"),
                    t.get("phi"),
                    t.get("cramer_v"),
                    t.get("C"),
                    1 if t.get("significativo") else 0,
                    1 if t.get("asignado_estricto") else 0,
                ),
            )
    logger.info("terms: %d filas", len(terms))
    return len(terms)


def migrate_annotations(
    conn: sqlite3.Connection, data: Dict, dry_run: bool = False
) -> int:
    """
    Formato esperado en el JSON (según el plan original):
        data["annotations_by_uce"] = {uce_id: [{agent_name, trait, quote, ...}, ...]}
    """
    ann_by_uce = data.get("annotations_by_uce") or {}
    if not ann_by_uce:
        logger.info("annotations: no hay datos en el JSON")
        return 0

    # Aplanar para contar
    flat: List[Dict] = []
    for uce_id, items in ann_by_uce.items():
        for item in items or []:
            flat.append({"uce_id": str(uce_id), **item})

    if dry_run:
        logger.info("[dry-run] annotations: %d filas", len(flat))
        return len(flat)

    with transaction(conn):
        conn.execute("DELETE FROM annotations")  # reconstrucción total
        for a in flat:
            conn.execute(
                """
                INSERT INTO annotations
                  (uce_id, agent_name, trait, quote, confidence,
                   subtype, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    a["uce_id"],
                    a.get("agent_name") or a.get("agent") or "unknown",
                    a.get("trait"),
                    a.get("quote"),
                    a.get("confidence"),
                    a.get("subtype"),
                    _json_or_empty(a),
                ),
            )
    logger.info("annotations: %d filas", len(flat))
    return len(flat)


def migrate_network(conn: sqlite3.Connection, data: Dict, dry_run: bool = False) -> int:
    """
    Acepta todos los formatos de edge plausibles:

      · [{"source": u, "target": v, "weight": w}, ...]
      · [{"src": u, "dst": v, "weight": w}, ...]
      · [{"from": u, "to": v, "weight": w}, ...]
      · [(u, v, w), ...]
      · [(u, v, {"weight": w}), ...]
      · [(u, v), ...]                    → weight por defecto 1.0
      · {"u__v": w, ...}                 → dict de aristas
    """
    net = data.get("network") or {}
    edges = net.get("edges") if isinstance(net, dict) else net
    if not edges:
        logger.info("network_edges: no hay datos en el JSON")
        return 0

    norm: List[tuple] = []

    def _coerce_weight(w: Any) -> float:
        if isinstance(w, dict):
            w = w.get("weight", 1.0)
        try:
            return float(w)
        except (TypeError, ValueError):
            return 1.0

    # Caso dict de aristas: {"u__v": w}
    if isinstance(edges, dict):
        for k, v in edges.items():
            if "__" not in k:
                continue
            u, v_ = k.split("__", 1)
            norm.append((str(u), str(v_), _coerce_weight(v)))

    # Caso lista
    elif isinstance(edges, list):
        for e in edges:
            if isinstance(e, dict):
                u = e.get("source") or e.get("src") or e.get("from") or e.get("u")
                v = e.get("target") or e.get("dst") or e.get("to") or e.get("v")
                w = e.get("weight", e.get("w", 1.0))
                if u is None or v is None:
                    continue
                norm.append((str(u), str(v), _coerce_weight(w)))

            elif isinstance(e, (list, tuple)):
                if len(e) >= 3:
                    u, v, w = e[0], e[1], e[2]
                elif len(e) == 2:
                    u, v = e
                    w = 1.0
                else:
                    continue
                if u is None or v is None:
                    continue
                norm.append((str(u), str(v), _coerce_weight(w)))

    # Deduplicar por (src, dst) manteniendo el mayor peso
    dedup: Dict[tuple, float] = {}
    for u, v, w in norm:
        # Normalizar simetría: (min, max) para no duplicar (a,b) y (b,a)
        key = (u, v) if u <= v else (v, u)
        if key not in dedup or w > dedup[key]:
            dedup[key] = w
    norm = [(u, v, w) for (u, v), w in dedup.items()]

    if dry_run:
        logger.info("[dry-run] network_edges: %d filas", len(norm))
        return len(norm)

    with transaction(conn):
        conn.execute("DELETE FROM network_edges")
        conn.executemany(
            "INSERT OR REPLACE INTO network_edges (src, dst, weight) VALUES (?, ?, ?)",
            norm,
        )
    logger.info("network_edges: %d filas", len(norm))
    return len(norm)


def migrate_kv(conn: sqlite3.Connection, data: Dict, dry_run: bool = False) -> int:
    """
    Guarda en kv_store todo lo que no tiene tabla propia:
    config, sintesis_por_clase, multivariate, term_stability,
    forma_index, cah_terminos, afc_result, cdh_tree_*, shap_analysis,
    liminal_projection, pairwise_stability, etc.
    """
    KV_KEYS = (
        "config",
        "sintesis_por_clase",
        "multivariate",
        "term_stability",
        "forma_index",
        "cah_por_clase",
        "cah_terminos_global",
        "proyeccion",
        "cdh_tree_umbral1",
        "cdh_tree_umbral2",
        "shap_analysis",
        "liminal_projection",
        "pairwise_stability",
        "metadata_residuals",
        "stem_summary",
        "words_per_cluster",
        "pos_by_cluster",
        "grammatical_summary_by_class",
        "condensed_tree_plot_data",
        "centroid_export",
        "retrofitted_vectors_significant",
        "clustering_method",
        "metadata_analysis",
        "section_registry",
        "origen_index",
    )
    present = {k: data[k] for k in KV_KEYS if k in data}
    if dry_run:
        logger.info("[dry-run] kv_store: %d keys", len(present))
        return len(present)

    with transaction(conn):
        for k, v in present.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO kv_store (key, value_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (k, _json_or_empty(v) if not isinstance(v, str) else v),
            )
    logger.info("kv_store: %d keys", len(present))
    return len(present)


# ════════════════════════════════════════════════════════════════════════════
# Orquestador
# ════════════════════════════════════════════════════════════════════════════
def migrate(json_path: str, db_path: str, dry_run: bool = False) -> Dict[str, int]:
    t0 = time.time()
    logger.info("Cargando %s ...", json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(
        "JSON cargado en %.2fs (%d claves top-level)", time.time() - t0, len(data)
    )

    conn = connect(db_path)
    try:
        counts = {
            "documents": migrate_documents(conn, data, dry_run),
            "uces": migrate_uces(conn, data, dry_run),
            "ucs": migrate_ucs(conn, data, dry_run),
            "clusters": migrate_clusters(conn, data, dry_run),
            "terms": migrate_terms(conn, data, dry_run),
            "annotations": migrate_annotations(conn, data, dry_run),
            "network_edges": migrate_network(conn, data, dry_run),
            "kv_store": migrate_kv(conn, data, dry_run),
        }
    finally:
        conn.close()

    logger.info("Migración completa en %.2fs", time.time() - t0)
    for k, v in counts.items():
        logger.info("  %-14s %d", k, v)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    migrate(args.json, args.db, dry_run=args.dry_run)
