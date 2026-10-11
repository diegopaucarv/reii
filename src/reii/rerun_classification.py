#!/usr/bin/env python
"""
REII re-run: full classification + analysis stack, SKIPPING the ~14h
grammatical enrichment pass.

Calls ``WorkflowOrchestrator.ejecutar(grammatical_pipeline=None)`` so that
``PipelineGramatical.procesar_desde_uces()`` is never invoked. Everything
else runs exactly as in a normal full pipeline:

    · ClasificadorDescendente (CDH, inside DoubleClassifier)
    · DoubleClassifier stability (Hungarian, ARI, pairwise matrices)
    · AFC + CAH (global & per-class)
    · MetaAnalyzer (chi²/ANOVA on metadata)
    · MCA (multivariate)
    · TermStability bootstrap
    · RandomForest + SHAP
    · Co-occurrence network
    · Class centroids + analytic products P1–P5
    · Dashboard keys + ``db._save()`` (PostgreSQL + workflow JSON)

Usage (inside the container):

    python src/reii/rerun_classification.py
    python src/reii/rerun_classification.py --classification-mode all
    python src/reii/rerun_classification.py --classification-mode wc_coref
    python src/reii/rerun_classification.py --reclassify-raw
    python src/reii/rerun_classification.py --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd

from reii.config import BEST_PARAMS_PATH
from reii.config import DATA_DIR as REII_DATA_DIR
from reii.gram.gramatical_analyzer import Config as GramConfig
from reii.gram.gramatical_analyzer import NLPProvider
from reii.main_workflow import (
    Config,
    RetrofittingConfig,
    UCBuilderConfig,
    WorkflowOrchestrator,
    _corpus_fingerprint,
    obtener_llave_maestra,
)

_COREF_MODES = {"coref_only", "wc_coref", "coref_emb", "all"}


# ─────────────────────────────────────────────────────────────────────────
# Corpus reconstruction (replicates main_workflow L186-257, NO skip logic)
# ─────────────────────────────────────────────────────────────────────────
def rebuild_corpus_raw() -> Dict[str, Any]:
    """Rebuild ``uwu`` (corpus_raw) from scratch, ignoring the DB skip logic."""
    metadata_csv = os.path.join(REII_DATA_DIR, "txt_outputs", "Refined_Database.csv")
    df_meta = (
        pd.read_csv(metadata_csv, sep=";")
        if os.path.exists(metadata_csv)
        else pd.DataFrame()
    )
    meta_dict: Dict[str, Dict] = {}
    for _, row in df_meta.iterrows():
        llave = obtener_llave_maestra(row["Documento Fuente"])
        meta_dict[llave] = row[
            ["Edad_Cat", "Sexo", "Dependientes_Cat", "Ocupacion_Cat", "Procedencia_Cat"]
        ].to_dict()

    json_input_dir = os.path.join(REII_DATA_DIR, "txt_outputs", "tmp")
    archivos_json = sorted(glob.glob(os.path.join(json_input_dir, "*.json")))
    dir_txt = os.path.join(REII_DATA_DIR, "txt_outputs")

    uwu: Dict[str, Any] = {}
    secciones_ignoradas = {
        "TEXTO_NO_ASIGNADO",
        "I. Datos sociodemográficos y perfil profesional",
    }

    for indice_orden, ruta_json in enumerate(archivos_json):
        nombre_json = os.path.basename(ruta_json)
        llave_maestra = obtener_llave_maestra(nombre_json)

        ruta_txt = os.path.join(dir_txt, f"{llave_maestra}.txt")
        texto_completo = ""
        if os.path.exists(ruta_txt):
            with open(ruta_txt, "r", encoding="utf-8") as f_txt:
                texto_completo = f_txt.read()

        uwu[nombre_json] = {
            "indice_orden": indice_orden,
            "metadata": meta_dict.get(llave_maestra, {}),
            "texto_completo_txt": texto_completo,
            "texto_segmentos": [],
        }

        with open(ruta_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        segmentos = data.get("output", {}).get("segmentos", [])
        indice_interno = 0
        for seg in segmentos:
            sec = seg.get("seccion_entrevista")
            txt = seg.get("texto_literal")
            if sec and txt and sec not in secciones_ignoradas:
                uwu[nombre_json]["texto_segmentos"].append(
                    {
                        "indice_interno": indice_interno,
                        "texto": txt,
                        "otros_datos": {"seccion": sec},
                    }
                )
                indice_interno += 1

    return uwu


def build_doc_metadata_map(uwu: Dict[str, Any]) -> Dict[int, Dict]:
    """Replicate ``segmentar_en_uces``'s ``doc_metadata_map`` construction."""
    out: Dict[int, Dict] = {}
    for origen_key, doc_data in uwu.items():
        doc_idx = int(doc_data.get("indice_orden", 0))
        out[doc_idx] = {
            **doc_data.get("metadata", {}),
            "origen": origen_key,
            "doc_idx": doc_idx,
            "indice_orden": doc_data.get("indice_orden"),
            "texto_completo_txt": doc_data.get("texto_completo_txt", ""),
        }
    return out


def group_uces_by_doc(uces: List) -> List[List]:
    """Group a flat list of UCEs into per-document lists, ordered by local_idx."""
    by_doc: Dict[int, List] = defaultdict(list)
    for uce in uces:
        by_doc[uce.doc_id if uce.doc_id is not None else -1].append(uce)
    result = []
    for doc_id in sorted(by_doc.keys()):
        result.append(
            sorted(
                by_doc[doc_id],
                key=lambda u: u.local_idx if u.local_idx is not None else 0,
            )
        )
    return result


def apply_best_params(orchestrator: WorkflowOrchestrator, bp: Dict[str, Any]) -> None:
    """Apply best_params.json to the orchestrator's config."""
    cfg = orchestrator.config
    mf1 = bp.get("min_forms_uc_1", cfg.min_forms_uc[0])
    gap = bp.get("forms_gap", cfg.min_forms_uc[1] - mf1)
    cfg.min_forms_uc = [mf1, mf1 + gap]
    cfg.tsj = bp.get("tsj", cfg.tsj)
    cfg.pseudocount = 0.0
    cfg.min_cluster_size_cdh = bp.get("min_cluster_size_cdh", cfg.min_cluster_size_cdh)
    cfg.swap_iterations = bp.get("swap_iterations", cfg.swap_iterations)
    cfg.min_r2_threshold = bp.get("min_r2_threshold", cfg.min_r2_threshold)
    if "similarity_threshold" in bp:
        orchestrator.uc_config.similarity_threshold = bp["similarity_threshold"]
    if "coref_weight" in bp:
        orchestrator.uc_config.coref_weight = bp["coref_weight"]
    if "uc_window_size" in bp:
        orchestrator.uc_config.window_size = bp["uc_window_size"]


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="REII re-run without grammatical enrichment."
    )
    parser.add_argument(
        "--classification-mode",
        default="wc_only",
        choices=[
            "wc_only",
            "coref_only",
            "emb_only",
            "wc_coref",
            "wc_emb",
            "coref_emb",
            "all",
        ],
        help="Classifiers to run. 'wc_only' = CDH + double-pass WC "
        "(fast, no Stanza). 'all' = WC + coref + EMB triple "
        "intersection (slow, needs Stanza).",
    )
    parser.add_argument(
        "--reclassify-raw",
        action="store_true",
        help="Ignore UCEs in DB and re-segment from corpus_raw. Slower; "
        "loses any prior per-UCE enrichment fields.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run everything but skip the final db._save().",
    )
    parser.add_argument(
        "--strict-fingerprint",
        action="store_true",
        help="Abort if best_params.json's fingerprint mismatches the corpus.",
    )
    args = parser.parse_args()

    # ── 1. Rebuild corpus_raw ────────────────────────────────────────────
    print("=== Rebuilding corpus_raw ===")
    uwu = rebuild_corpus_raw()
    print(f"   {len(uwu)} documentos.")

    # ── 2. Configs ───────────────────────────────────────────────────────
    alceste_config = Config(
        use_bigrams=True,
        stem_backend="snowball",
        min_uce_words=3,
        min_forms_uc=[10, 14],
        tsj=3,
        use_cdh=True,
        uce_target_size=18,
        pseudocount=0.0,
        use_cah_per_class=True,
        cah_per_class_top_terms=20,
        use_projection=True,
        analyze_metadata=True,
        use_network_analysis=True,
        use_term_stability=True,
        random_state=42,
        use_rf_shap=True,
        rf_n_estimators=100,
        rf_max_depth=5,
        rf_outlier_method="iqr",
        rf_cat_encoding="onehot",
        glm_method="chi2",
        use_multivariate_analysis=True,
        multivariate_metadata=[
            "Edad_Cat",
            "Sexo",
            "Ocupacion_Cat",
            "Procedencia_Cat",
        ],
        ppmi_k=1.0,
        analytic_temperature=0.1,
        use_llm_synthesis=False,
        classification_mode=args.classification_mode,
        coref_context_units_tight=2,
        coref_context_units_loose=4,
        hdbscan_uce_min_cluster_size=5,
        hdbscan_uce_min_cluster_size_loose=3,
        hdbscan_uce_metric="euclidean",
        optimize=False,  # best_params.json already exists
        optimize_trials=300,
        optimize_sampler="tpe",
        optimize_pruner="median",
        optimize_storage="sqlite:///reii_optuna.db",
        optimize_study_name="reii_search_v2",
        optimize_n_startup_trials=25,
        optimize_multivariate=True,
        optimize_prune_n_startup=15,
        optimize_directions=("maximize", "maximize", "minimize"),
        optimize_preference="knee",
        optimize_coverage_gate=0.65,
    )

    gram_config = GramConfig(min_tokens_por_uce=30, max_tokens_por_uce=200)
    retro_config = RetrofittingConfig(
        alpha=0.8,
        beta=0.5,
        gamma=0.3,
        n_iter=10,
        svd_components=100,
        chi2_threshold=3.84,
    )
    uc_cfg = UCBuilderConfig()

    # ── 3. Pipelines (subtlex + word vectors; needed for retrofitting) ───
    subtlex = NLPProvider.get_subtlex(gram_config)
    we_analyzer = NLPProvider.get_word_vectors(gram_config)

    # ProgressiveSegmenter only when the mode actually needs Method B
    progressive_seg = None
    if args.classification_mode in _COREF_MODES:
        from reii.seg.segmentador import ProgressiveSegmenter

        progressive_seg = ProgressiveSegmenter(
            model_name=alceste_config.embedding_model_name,
            stanza_lang="es",
            spacy_model=alceste_config.spacy_model,
            debug_coref=False,
            similarity_threshold=0.6,
            max_depth=3,
        )
        if progressive_seg.get_stanza() is None:
            print("   [WARN] Stanza unavailable — Method B will be skipped.")
            progressive_seg = None

    orchestrator = WorkflowOrchestrator(
        config=alceste_config,
        uc_config=uc_cfg,
        retro_config=retro_config,
        we_analyzer=we_analyzer,
        subtlex_analizer=subtlex,
        progressive_segmenter=progressive_seg,
    )

    # ── 4. Apply best_params.json ────────────────────────────────────────
    print("=== Applying best_params.json ===")
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
            best_data = json.load(f)
        _stored = best_data.get("corpus_fingerprint")
        _current = _corpus_fingerprint()
        if _stored and _stored != _current:
            print(
                f"   [WARN] fingerprint mismatch: stored={_stored} current={_current}"
            )
            if args.strict_fingerprint:
                print("   [FATAL] --strict-fingerprint: aborting.")
                return 1
        bp = best_data.get("best_params") or best_data.get("params")
        if bp:
            apply_best_params(orchestrator, bp)
            print(f"   Loaded params: {bp}")
        else:
            print("   [WARN] best_params.json has no 'params'/'best_params'.")
    else:
        print("   [WARN] No best_params.json — using config defaults.")

    # ── 5. Prepare UCE injection (or leave the segmenter untouched) ──────
    if args.reclassify_raw:
        print("=== --reclassify-raw: will re-segment from corpus_raw ===")
    else:
        print("=== Loading UCEs from the workflow DB ===")
        uces = orchestrator.db.load_uces()
        if not uces:
            print("   [FATAL] No UCEs in DB. Retry with --reclassify-raw.")
            return 1
        print(f"   {len(uces)} UCEs loaded.")
        uces_por_doc = group_uces_by_doc(uces)
        doc_metadata_map = build_doc_metadata_map(uwu)
        orchestrator._enrich_uces_with_doc_metadata(uces_por_doc, doc_metadata_map)

        # Inject into the segmenter so double_clf.run() reuses these objects
        # instead of re-segmenting / re-lemmatizing the corpus.
        seg = orchestrator.double_clf.segmentador
        seg.doc_metadata_map = doc_metadata_map
        seg.segmentar_en_uces = lambda _cr, _u=uces_por_doc, _m=doc_metadata_map: (
            _u,
            _m,
        )
        seg.lematizar_uces = lambda _u: _u

    # ── 6. --dry-run: intercept the final save ───────────────────────────
    if args.dry_run:
        orchestrator.db._save = lambda: print("[dry-run] db._save() skipped")

    # ── 7. Run the full pipeline minus enrichment ────────────────────────
    print("=== ejecutar(grammatical_pipeline=None) — enrichment skipped ===")
    orchestrator.ejecutar(
        corpus_raw=uwu,
        grammatical_pipeline=None,  # ← skips the 14h block
        global_corpus=None,
        lex_analyzer=None,
        grammatical_dashboard_path="",  # ← skip legacy grammar JSON
    )
    print("=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
