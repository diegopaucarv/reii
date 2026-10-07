"""Single dashboard interface to SQLite: corpus + config + derived, in parallel."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from reii.backend.config_store import ConfigStore
from reii.backend.sql.db import get_conn
from reii.config import DATABASE_URL

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


class DashboardAdapter:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DATABASE_URL

    def uces(self, cluster_id: Optional[int] = None) -> List[Dict]:
        sql = "SELECT * FROM uces"
        params: List[Any] = []
        if cluster_id is not None:
            sql += " WHERE cluster_id = %s"
            params.append(cluster_id)
        sql += " ORDER BY doc_id, local_idx"
        with get_conn(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._rehydrate_uce(dict(r)) for r in rows]

    def _rehydrate_uce(self, row: Dict) -> Dict[str, Any]:
        d = {
            "id": row["uce_id"],
            "uce_id": row["uce_id"],
            "doc_id": row["doc_id"],
            "local_idx": row["local_idx"],
            "seccion": row["seccion"],
            "texto": row["texto"],
            "cluster_id": row["cluster_id"],
            "is_stable": bool(row["is_stable"]),
            "phi_score": row["phi_score"],
            "projected_cluster_id": row["projected_cluster_id"],
            "projection_distance": row["projection_distance"],
            "projection_margin": row["projection_margin"],
            "projection_ratio": row["projection_ratio"],
            "metadata": {
                "uce_local_idx": row["local_idx"],
                "section_id": row["section_id"],
            },
        }
        ling = json.loads(row["linguistic_json"] or "{}")
        met = json.loads(row["metrics_json"] or "{}")
        d.update(ling)
        d.update(met)
        return d

    def terms(self, cluster_id: Optional[int] = None) -> List[Dict]:
        sql = (
            "SELECT term AS termino, cluster_id AS cluster, frecuencia_global, "
            "frecuencia_cluster, chi2_yates, p_valor, p_adj, phi, cramer_v, "
            "c_value AS C, significativo, asignado_estricto FROM terms"
        )
        params: List[Any] = []
        if cluster_id is not None:
            sql += " WHERE cluster_id = %s"
            params.append(cluster_id)
        sql += " ORDER BY cluster_id, term"
        with get_conn(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["significativo"] = bool(d["significativo"])
            out.append(d)
        return out

    def ucs(self) -> List[Dict]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT uc_id, doc_id, texto, cluster_id, uce_ids_json, lemmas_json, n_lemmas FROM ucs ORDER BY uc_id"
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["uc_id"],
                    "doc_id": r["doc_id"],
                    "texto": r["texto"],
                    "cluster_id": r["cluster_id"],
                    "cluster_label_double": r["cluster_id"],
                    "uce_ids": json.loads(r["uce_ids_json"] or "[]"),
                    "lemmas": json.loads(r["lemmas_json"] or "[]"),
                    "coordinates": {},
                }
            )
        return out

    def clusters(self) -> List[Dict]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT cluster_id, cluster_name, member_count, top_terms_json, centroid_json FROM clusters ORDER BY cluster_id"
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["top_terms"] = json.loads(d.pop("top_terms_json") or "[]")
            d["centroid"] = json.loads(d.pop("centroid_json") or "null")
            out.append(d)
        return out

    def annotations(self) -> Dict[str, List[Dict]]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT uce_id, agent_name, trait, quote, confidence, subtype, payload_json FROM annotations"
            ).fetchall()
        out: Dict[str, List[Dict]] = {}
        for r in rows:
            ann = json.loads(r["payload_json"] or "{}")
            ann["uce_id"] = r["uce_id"]
            if "spans" not in ann and ann.get("quote"):
                ann["spans"] = [
                    {
                        "uce_id": r["uce_id"],
                        "quote": ann["quote"],
                        "start_char": -1,
                        "end_char": -1,
                    }
                ]
            out.setdefault(r["uce_id"], []).append(ann)
        return out

    def network_edges(self, min_weight: float = 0.0) -> List[Dict]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT src, dst, weight FROM network_edges WHERE weight >= %s ORDER BY weight DESC",
                (min_weight,),
            ).fetchall()
        return [
            {"source": r["src"], "target": r["dst"], "weight": r["weight"]}
            for r in rows
        ]

    def get_config(self, key: str, default: Any = None) -> Any:
        return ConfigStore(self.db_path).get(key, default)

    def get_config_all(self) -> Dict[str, Any]:
        return ConfigStore(self.db_path).get_all()

    def set_config(self, key: str, value: Any, by: str = "dashboard") -> None:
        ConfigStore(self.db_path).set(key, value, by)

    def set_config_many(self, updates: Dict[str, Any], by: str = "dashboard") -> int:
        return ConfigStore(self.db_path).set_config_many(updates, by)

    def reset_config(self, key: str) -> None:
        ConfigStore(self.db_path).reset(key)

    def get_derived(self, key: str, default: Any = None) -> Any:
        with get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT value_json FROM kv_store WHERE key = %s", (key,)
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value_json"])
        except (json.JSONDecodeError, TypeError):
            return row["value_json"]

    def get_many_derived(self, keys: List[str]) -> Dict[str, Any]:
        if not keys:
            return {}
        placeholders = ",".join(["%s"] * len(keys))
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT key, value_json FROM kv_store WHERE key IN ({placeholders})",
                keys,
            ).fetchall()
        out: Dict[str, Any] = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value_json"])
            except (json.JSONDecodeError, TypeError):
                out[r["key"]] = r["value_json"]
        return out

    def _doc_metadata(self) -> Dict[str, Dict]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT doc_id, metadata_json FROM documents"
            ).fetchall()
        return {str(r["doc_id"]): json.loads(r["metadata_json"] or "{}") for r in rows}

    def version_hash(self) -> str:
        with get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE((SELECT MAX(updated_at) FROM kv_store), '') AS kv, "
                "COALESCE((SELECT MAX(updated_at) FROM workflow_config), '') AS cfg, "
                "(SELECT COUNT(*) FROM uces) AS n_uces, (SELECT COUNT(*) FROM terms) AS n_terms"
            ).fetchone()
        raw = f"{row['kv']}|{row['cfg']}|{row['n_uces']}|{row['n_terms']}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def snapshot(self, with_derived: Optional[List[str]] = None) -> Dict[str, Any]:
        if with_derived is None:
            with_derived = list(KV_KEYS)
        with ThreadPoolExecutor(max_workers=4) as ex:
            f_uces = ex.submit(self.uces)
            f_terms = ex.submit(self.terms)
            f_ucs = ex.submit(self.ucs)
            f_clusters = ex.submit(self.clusters)
            f_derived = ex.submit(self.get_many_derived, with_derived)
            f_cfg = ex.submit(self.get_config_all)
            f_docs = ex.submit(self._doc_metadata)
            uces = f_uces.result()
            terms = f_terms.result()
            ucs = f_ucs.result()
            clusters = f_clusters.result()
            derived = f_derived.result()
            cfg_overrides = f_cfg.result()
            docs = f_docs.result()
        # config: last run's full config (kv_store) + user overrides (workflow_config)
        base_config = (
            derived.get("config", {}) if isinstance(derived.get("config"), dict) else {}
        )
        config = {**base_config, **cfg_overrides}
        data: Dict[str, Any] = {
            "uces": uces,
            "terminos": terms,
            "ucs": ucs,
            "clusters": clusters,
            "config": config,
            "doc_metadata": docs,
            "uce_phi": [
                {"uce_id": u["uce_id"], "phi_score": u.get("phi_score")} for u in uces
            ],
        }
        for k, v in derived.items():
            if k != "config":
                data[k] = v
        return data
