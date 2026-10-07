"""Workflow config store backed by the workflow_config table (schema v2)."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Optional

from reii.backend.sql.db import get_conn, transaction
from reii.config import DATABASE_URL

CONFIG_CATEGORIES: Dict[str, List[str]] = {
    "segmentacion": [
        "spacy_model",
        "use_bigrams",
        "use_trigrams",
        "min_bigram_freq",
        "bigram_pos_patterns",
        "trigram_pos_patterns",
        "stem_backend",
        "min_uce_words",
        "uce_target_size",
        "min_forms_uc",
    ],
    "frecuencias_y_ctest": [
        "tsj",
        "use_poisson_tsj",
        "poisson_alpha",
        "min_term_abs_freq",
        "use_ctest",
        "ctest_threshold",
        "adaptive_ctest",
        "ctest_alpha",
        "n_permutations_ctest",
    ],
    "clasificacion": [
        "use_cdh",
        "pseudocount",
        "swap_iterations",
        "n_perm_cdh",
        "perm_min_uc_size",
        "min_r2_threshold",
        "chi2_threshold_small",
        "max_depth_cdh",
        "min_cluster_size_cdh",
        "clustering_method",
        "n_clusters",
        "linkage_method",
        "distance_metric",
        "bootstrap_n_iter",
        "min_cluster_overlap",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "bootstrap_sample_frac",
        "hdbscan_cluster_selection_epsilon",
        "cluster_selection_method",
        "gap_n_references",
        "fdr_alpha",
        "prune_small_clusters",
        "min_class_size",
        "classification_mode",
        "hdbscan_uce_min_cluster_size",
        "hdbscan_uce_min_cluster_size_loose",
        "hdbscan_uce_epsilon",
        "hdbscan_uce_min_samples",
        "hdbscan_uce_metric",
    ],
    "afc_y_proyeccion": [
        "use_projection",
        "projection_method",
        "use_liminal_projection",
        "liminal_projection_min_ratio",
        "use_cah_per_class",
        "cah_per_class_top_terms",
    ],
    "metadatos_y_multivariado": [
        "analyze_metadata",
        "glm_method",
        "glm_alpha",
        "n_permutations",
        "use_multivariate_analysis",
        "multivariate_metadata",
    ],
    "red_semantica": [
        "use_network_analysis",
        "network_cooccurrence_threshold",
        "network_weight_method",
        "network_npmi_positive_only",
        "network_significance_filter",
        "network_significance_alpha",
    ],
    "rf_shap": [
        "use_rf_shap",
        "rf_n_estimators",
        "rf_max_depth",
        "rf_scale_features",
        "rf_feature_selection_threshold",
        "rf_outlier_method",
        "rf_impute_strategy",
        "rf_cat_encoding",
        "rf_min_samples_for_tuning",
    ],
    "llm_y_embeddings": [
        "use_llm_synthesis",
        "together_api_key",
        "llm_model",
        "synthesis_similarity_threshold",
        "use_embeddings",
        "embedding_model_name",
    ],
    "estabilidad": ["use_term_stability", "term_stability_n_iter"],
    "optimizacion": [
        "optimize",
        "optimize_trials",
        "optimize_sampler",
        "optimize_pruner",
        "optimize_storage",
        "optimize_study_name",
        "optimize_n_startup_trials",
        "optimize_multivariate",
        "optimize_prune_n_startup",
        "optimize_directions",
        "optimize_preference",
        "optimize_coverage_gate",
    ],
    "general": [
        "db_local_path",
        "subtlex_df_path",
        "random_state",
        "ppmi_k",
        "analytic_temperature",
        "coref_context_units_tight",
        "coref_context_units_loose",
        "db_dsn",
        "dual_write_json",
    ],
}

ALL_CONFIG_KEYS: List[str] = [k for keys in CONFIG_CATEGORIES.values() for k in keys]

_DEFAULT_CONFIG: Dict[str, Any] = {}


def _ensure_defaults() -> None:
    if _DEFAULT_CONFIG:
        return
    from reii.main_workflow import Config

    _DEFAULT_CONFIG.update(
        {k: v for k, v in dataclasses.asdict(Config()).items() if k in ALL_CONFIG_KEYS}
    )


def _value_type(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (list, tuple)):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return "str"


def _decode(value_json: str, value_type: str) -> Any:
    v = json.loads(value_json)
    if value_type == "null":
        return None
    if value_type == "bool":
        return bool(v)
    if value_type == "int":
        return int(v)
    if value_type == "float":
        return float(v)
    return v


def _category_of(key: str) -> str:
    for cat, keys in CONFIG_CATEGORIES.items():
        if key in keys:
            return cat
    return "general"


class ConfigStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DATABASE_URL

    def seed_from_config(self, config) -> int:
        _ensure_defaults()
        d = dataclasses.asdict(config)
        n = 0
        with get_conn(self.db_path) as conn, transaction(conn):
            for key in ALL_CONFIG_KEYS:
                if key not in d:
                    continue
                cur = conn.execute(
                    "SELECT 1 FROM workflow_config WHERE key = %s", (key,)
                ).fetchone()
                if cur:
                    continue
                conn.execute(
                    "INSERT INTO workflow_config (key, value_json, value_type, category, description, updated_by, updated_at) VALUES (%s, %s, %s, %s, %s, 'seed', CURRENT_TIMESTAMP)",
                    (
                        key,
                        json.dumps(d[key], ensure_ascii=False),
                        _value_type(d[key]),
                        _category_of(key),
                        None,
                    ),
                )
                n += 1
        return n

    def get_all(self) -> Dict[str, Any]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT key, value_json, value_type FROM workflow_config"
            ).fetchall()
        return {r["key"]: _decode(r["value_json"], r["value_type"]) for r in rows}

    def get(self, key: str, default: Any = None) -> Any:
        with get_conn(self.db_path) as conn:
            r = conn.execute(
                "SELECT value_json, value_type FROM workflow_config WHERE key = %s",
                (key,),
            ).fetchone()
        return _decode(r["value_json"], r["value_type"]) if r else default

    def set(self, key: str, value: Any, by: str = "dashboard") -> None:
        with get_conn(self.db_path) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO workflow_config (key, value_json, value_type, category, description, updated_by, updated_at) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP) ON CONFLICT (key) DO UPDATE SET value_json = EXCLUDED.value_json, value_type = EXCLUDED.value_type, updated_by = EXCLUDED.updated_by, updated_at = CURRENT_TIMESTAMP",
                (
                    key,
                    json.dumps(value, ensure_ascii=False),
                    _value_type(value),
                    _category_of(key),
                    None,
                    by,
                ),
            )

    def set_config_many(self, updates: Dict[str, Any], by: str = "dashboard") -> int:
        valid = {k: v for k, v in updates.items() if k in ALL_CONFIG_KEYS}
        with get_conn(self.db_path) as conn, transaction(conn):
            for k, v in valid.items():
                conn.execute(
                    "INSERT INTO workflow_config (key, value_json, value_type, category, description, updated_by, updated_at) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP) ON CONFLICT (key) DO UPDATE SET value_json = EXCLUDED.value_json, value_type = EXCLUDED.value_type, updated_by = EXCLUDED.updated_by, updated_at = CURRENT_TIMESTAMP",
                    (
                        k,
                        json.dumps(v, ensure_ascii=False),
                        _value_type(v),
                        _category_of(k),
                        None,
                        by,
                    ),
                )
        return len(valid)

    def reset(self, key: str) -> None:
        with get_conn(self.db_path) as conn, transaction(conn):
            conn.execute("DELETE FROM workflow_config WHERE key = %s", (key,))
