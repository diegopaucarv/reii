#!/usr/bin/env python
"""
Standalone strategic re-run for REII.

Fixes the remaining defects WITHOUT a full ``--force`` re-run (which would
re-enrich all 28 documents, ~14h). This script does two things:

  1. Rebuilds the ``ucs`` table (fixes ``ucs=0``) by re-running a single-pass
     word-count classification over the UCEs already persisted in the DB.
  2. Re-enriches ONLY the documents whose offsets desynced (DESYNC) — by
     default docs 2, 13, 15, 19 — using the fixed ``lock_global_offsets`` and
     ``classify_verb`` code.

It also re-attaches the sociodemographic metadata (the corrected
``Refined_Database.csv`` path) to every UCE, and re-links each UCE to its
parent UC (``uc_id``).

The final ``db._save()`` triggers the migration fixes that run automatically:
terms/clusters anti-accumulation DELETE, annotations merge, and the network
edge rebuild (matmul fix).

Usage (inside the container):

    python src/reii/rerun_classification.py
    python src/reii/rerun_classification.py --docs 2,13,15,19
    python src/reii/rerun_classification.py --skip-classify   # only re-enrich
    python src/reii/rerun_classification.py --skip-enrich     # only classify+save UCs
    python src/reii/rerun_classification.py --dry-run
    python src/reii/rerun_classification.py --force           # bypass anti-wipe guard
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from reii.config import BEST_PARAMS_PATH
from reii.config import DATA_DIR as REII_DATA_DIR
from reii.gram.gramatical_analyzer import Config as GramConfig
from reii.gram.gramatical_analyzer import (
    GlobalCorpus,
    GlobalLexicalAnalyzer,
    NLPProvider,
    PipelineGramatical,
    train_adverb_classifier,
)

# NOTE: importing main_workflow runs its module-level code (builds ``uwu`` and
# applies the skip logic). That is harmless here — we rebuild ``uwu`` ourselves
# and never read ``main_workflow.uwu``.
from reii.main_workflow import (
    _UCE_PROJECTION_FIELDS,
    Config,
    MatrizBuilder,
    RetrofittingConfig,
    UCBuilderConfig,
    WorkflowOrchestrator,
    _corpus_fingerprint,
    obtener_llave_maestra,
)

DEFAULT_RE_ENRICH_DOCS = {2, 13, 15, 19}


# ─────────────────────────────────────────────────────────────────────────────
# Corpus reconstruction (replicates main_workflow L186-257, NO skip logic)
# ─────────────────────────────────────────────────────────────────────────────
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
    doc_metadata_map: Dict[int, Dict] = {}
    for origen_key, doc_data in uwu.items():
        doc_idx = int(doc_data.get("indice_orden", 0))
        doc_metadata_map[doc_idx] = {
            **doc_data.get("metadata", {}),
            "origen": origen_key,
            "doc_idx": doc_idx,
            "indice_orden": doc_data.get("indice_orden"),
            "texto_completo_txt": doc_data.get("texto_completo_txt", ""),
        }
    return doc_metadata_map


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def apply_best_params(
    orchestrator: WorkflowOrchestrator, best_params: Dict[str, Any]
) -> None:
    """Apply ``best_params.json`` to the orchestrator's config (replicates
    ``ejecutar()``'s ``_apply_best_params``)."""
    cfg = orchestrator.config
    mf1 = best_params.get("min_forms_uc_1", cfg.min_forms_uc[0])
    gap = best_params.get("forms_gap", cfg.min_forms_uc[1] - mf1)
    cfg.min_forms_uc = [mf1, mf1 + gap]
    cfg.tsj = best_params.get("tsj", cfg.tsj)
    cfg.pseudocount = 0.0
    cfg.min_cluster_size_cdh = best_params.get(
        "min_cluster_size_cdh", cfg.min_cluster_size_cdh
    )
    cfg.swap_iterations = best_params.get("swap_iterations", cfg.swap_iterations)
    cfg.min_r2_threshold = best_params.get("min_r2_threshold", cfg.min_r2_threshold)
    if "similarity_threshold" in best_params:
        orchestrator.uc_config.similarity_threshold = best_params[
            "similarity_threshold"
        ]
    if "coref_weight" in best_params:
        orchestrator.uc_config.coref_weight = best_params["coref_weight"]
    if "uc_window_size" in best_params:
        orchestrator.uc_config.window_size = best_params["uc_window_size"]


def serialize_uces(db, all_uces: List) -> None:
    """Serialize all UCEs, preserving ``texto_completo_doc`` and projection
    fields (replicates ``_save_all_uces``'s serialization, without the
    stable/unstable dedup)."""
    texto_map = db.data.get("texto_completo_por_doc_id", {})
    # Preservar campos de proyección ya persistidos: load_uces()→from_dict()
    # los descarta (no son campos de la dataclass) y getattr() devolvería
    # None, borrándolos silenciosamente en un re-run.
    prev = {d.get("id"): d for d in db.data.get("uces", [])}
    serialized = []
    for uce in all_uces:
        d = uce.to_dict()
        d["texto_completo_doc"] = texto_map.get(str(uce.doc_id), "")
        for field_name in _UCE_PROJECTION_FIELDS:
            d[field_name] = getattr(uce, field_name, None)
            if d[field_name] is None:
                d[field_name] = prev.get(uce.id, {}).get(field_name)
        serialized.append(d)
    db.data["uces"] = serialized


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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strategic REII re-run (ucs=0 + DESYNC)."
    )
    parser.add_argument(
        "--docs",
        type=str,
        default=None,
        help="Comma-separated doc_ids (indice_orden) to re-enrich. "
        f"Default: {','.join(str(d) for d in sorted(DEFAULT_RE_ENRICH_DOCS))}.",
    )
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Skip the classification pass (only re-enrich).",
    )
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Skip the re-enrichment pass (only classify + save UCs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run everything but do not write to the DB.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the anti-wipe guard (ucs/network vacíos en JSON con datos "
        "en PostgreSQL). Úsalo solo si sabes que el JSON es la fuente correcta "
        "o ya limpiaste las tablas explícitamente.",
    )
    parser.add_argument(
        "--strict-fingerprint",
        action="store_true",
        help="Abort if best_params.json's corpus_fingerprint does not match "
        "the current corpus.",
    )
    args = parser.parse_args()

    if args.docs is None:
        re_enrich_docs = set(DEFAULT_RE_ENRICH_DOCS)
    else:
        try:
            re_enrich_docs = {int(x.strip()) for x in args.docs.split(",") if x.strip()}
        except ValueError:
            print(
                f"   [FATAL] --docs debe ser una lista de enteros separados "
                f"por coma, se recibió: '{args.docs}'"
            )
            return 1
        if not re_enrich_docs:
            print("   [FATAL] --docs vacío. Proporciona al menos un doc_id.")
            return 1

    if args.skip_enrich and args.docs is not None:
        print(
            "   [WARNING] --docs se ignora con --skip-enrich "
            "(no hay re-enriquecimiento)."
        )
    if args.skip_classify and args.skip_enrich:
        print(
            "   [WARNING] --skip-classify + --skip-enrich: no se clasifica ni se "
            "re-enriquece. Solo se re-serializan las UCEs y se guarda."
        )

    # ── 1. Rebuild corpus_raw ────────────────────────────────────────────────
    print("=== Rebuilding corpus_raw (uwu) ===")
    uwu = rebuild_corpus_raw()
    print(f"   {len(uwu)} documentos.")

    # Fail-fast: validar --docs contra los doc_ids reales del corpus antes de
    # cargar modelos pesados. indice_orden == doc_id de las UCEs en DB.
    valid_doc_ids = {
        int(d["indice_orden"])
        for d in uwu.values()
        if d.get("indice_orden") is not None
    }
    missing = sorted(re_enrich_docs - valid_doc_ids)
    if missing:
        print(f"   [WARNING] Docs solicitados que no existen en el corpus: {missing}")
        print(f"   [WARNING] Doc_ids válidos: {sorted(valid_doc_ids)}")
        if not re_enrich_docs & valid_doc_ids:
            print("   [FATAL] Ningún doc solicitado existe en el corpus. Abortando.")
            return 1

    # ── 2. Configs (replicate __main__ block, but wc_only + no optimizer) ────
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
        multivariate_metadata=["Edad_Cat", "Sexo", "Ocupacion_Cat", "Procedencia_Cat"],
        ppmi_k=1.0,
        analytic_temperature=0.1,
        use_llm_synthesis=False,
        classification_mode="wc_only",  # only need word-count UCs
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

    gram_config = GramConfig(
        min_tokens_por_uce=30,
        max_tokens_por_uce=200,
    )
    retro_config = RetrofittingConfig(
        alpha=0.8,
        beta=0.5,
        gamma=0.3,
        n_iter=10,
        svd_components=100,
        chi2_threshold=3.84,
    )
    cfg = UCBuilderConfig()

    # Train adverb classifier if missing
    clf_path = os.path.join(
        gram_config.adverb_classifier_dir, "logistic_classifier.joblib"
    )
    if not os.path.exists(clf_path):
        train_adverb_classifier(
            gram_config.adverb_classifier_dir,
            gram_config.sentence_embedder_model,
        )

    # ── 3. Pipelines ─────────────────────────────────────────────────────────
    gram_pipeline = PipelineGramatical(gram_config)
    subtlex = NLPProvider.get_subtlex(gram_config)
    we_analyzer = NLPProvider.get_word_vectors(gram_config)
    global_corpus = GlobalCorpus()
    lex_analyzer = GlobalLexicalAnalyzer(
        subtlex_analyzer=subtlex,
        cooc_window=4,
        cooc_min_count=3,
        we_analyzer=we_analyzer,
    )

    orchestrator = WorkflowOrchestrator(
        config=alceste_config,
        uc_config=cfg,
        retro_config=retro_config,
        we_analyzer=we_analyzer,
        subtlex_analizer=subtlex,
        progressive_segmenter=None,
    )

    # ── 4. Apply best_params ─────────────────────────────────────────────────
    print("=== Applying best_params.json ===")
    best_params: Optional[Dict[str, Any]] = None
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
            best_data = json.load(f)
        _stored_fp = best_data.get("corpus_fingerprint")
        _current_fp = _corpus_fingerprint()
        if _stored_fp and _stored_fp != _current_fp:
            print(
                "   [WARNING] El fingerprint del corpus NO coincide con el "
                "sellado en best_params.json"
            )
            print(
                f"   [WARNING] stored={_stored_fp} actual={_current_fp}. "
                "El corpus cambió desde la optimización (o los archivos se "
                "re-exportaron)."
            )
            if args.strict_fingerprint:
                print("   [FATAL] --strict-fingerprint: abortando.")
                return 1
            print(
                "   Aplicando los params cargados de todos modos "
                "(usa --strict-fingerprint para abortar)."
            )
        elif _stored_fp:
            print(f"   Fingerprint OK ({_current_fp}).")
        else:
            print(
                "   [WARNING] best_params.json sin fingerprint (legacy). "
                "No se puede verificar que pertenezca a este dataset."
            )
        best_params = best_data.get("best_params") or best_data.get("params")
    if best_params:
        apply_best_params(orchestrator, best_params)
        print(f"   Loaded params: {best_params}")
    else:
        print("   [WARNING] No best_params found. Using config defaults.")

    # ── 5. Load existing UCEs from DB ────────────────────────────────────────
    print("=== Loading existing UCEs from DB ===")
    uces = orchestrator.db.load_uces()
    print(f"   {len(uces)} UCEs loaded.")
    if not uces:
        print("   [FATAL] No UCEs in DB. Run the full workflow first.")
        return 1

    # Pre-flight: con --skip-classify no se regeneran ucs/network. Si el JSON
    # ya los perdió pero PostgreSQL aún los tiene, _save() borraría
    # network_edges (DELETE) y dejaría ucs divergente. Fallamos temprano.
    # --force desactiva el guard (el usuario declara que el JSON es la fuente
    # correcta o que ya limpió las tablas explícitamente).
    if args.force:
        print("   [FORCE] Bypass del guard anti-wipe (--force).")
        orchestrator.db._guard_against_ucs_network_wipe = lambda: None
    elif args.skip_classify:
        try:
            orchestrator.db._guard_against_ucs_network_wipe()
            print(
                "   [OK] Guard anti-wipe: ucs/network presentes en JSON "
                "o PostgreSQL vacío."
            )
        except RuntimeError as e:
            print(f"   [FATAL] {e}")
            return 1

    uces_por_doc = group_uces_by_doc(uces)
    print(f"   {len(uces_por_doc)} documentos agrupados.")

    # ── 6. Re-attach sociodemographic metadata (corrected CSV path) ──────────
    print("=== Re-attaching sociodemographic metadata ===")
    doc_metadata_map = build_doc_metadata_map(uwu)
    orchestrator._enrich_uces_with_doc_metadata(uces_por_doc, doc_metadata_map)
    print("   Metadata re-attached.")

    # ── 7. Re-run classification (single pass) to rebuild UCs ────────────────
    uce_to_uc: Dict[str, str] = {}
    primary_stable = []
    if not args.skip_classify:
        print("=== Re-running word-count classification (single pass) ===")
        (
            primary_stable,
            _,
            voc_uc,
            _uces_por_doc_out,
            all_res,
            _labels1,
            _labels2,
            _,
            _,
            _doc_metadata_map_out,
        ) = orchestrator.double_clf.run(
            uwu,
            ppmi_builder=orchestrator.ppmi_builder,
            cached_uces_por_doc=uces_por_doc,
            cached_doc_metadata_map=doc_metadata_map,
        )

        if all_res and all_res[0].get("ucs"):
            ucs = all_res[0]["ucs"]
            uce_to_uc = all_res[0].get("uce_to_uc", {})
            if not args.dry_run:
                orchestrator.db.save_ucs(ucs)
            print(f"   {'[dry-run] ' if args.dry_run else ''}Saved {len(ucs)} UCs.")
        else:
            print("   [WARNING] No UCs produced by classification.")
    else:
        print("   [SKIP] Classification skipped (--skip-classify).")

    # ── 8. Re-link UCEs to their parent UC (uc_id) ───────────────────────────
    if uce_to_uc:
        linked = 0
        for doc_uces in uces_por_doc:
            for uce in doc_uces:
                uc_id = uce_to_uc.get(uce.id)
                if uc_id:
                    uce.uc_id = uc_id
                    linked += 1
        print(f"   Re-linked {linked} UCEs to their parent UC.")

    # ── 8b. Rebuild the co-occurrence network (fixes empty network_edges) ────
    # The matmul fix (Fix 1) only takes effect when build_cooccurrence_graph is
    # re-run. We have primary_stable + the matrix builder available here, so we
    # rebuild the network from the stable UCEs and persist it for migration.
    if not args.skip_classify and primary_stable:
        try:
            print("=== Rebuilding co-occurrence network ===")
            builder = MatrizBuilder(orchestrator.config)
            voc = builder.construir_vocabulario(primary_stable)
            mat_uces = builder.construir_matriz(primary_stable, voc)
            G, partition = orchestrator.network.build_cooccurrence_graph(voc, mat_uces)
            if G is not None and G.number_of_nodes() > 0:
                edges_serial = [
                    (
                        u,
                        v,
                        {
                            k: float(val)
                            if isinstance(val, (np.floating, float))
                            else val
                            for k, val in d.items()
                        },
                    )
                    for u, v, d in G.edges(data=True)
                ]
                orchestrator.db.data["network"] = {
                    "nodes": list(G.nodes()),
                    "edges": edges_serial,
                    "partition": partition,
                }
                print(
                    f"   Network rebuilt: {G.number_of_nodes()} nodes, "
                    f"{G.number_of_edges()} edges."
                )
            else:
                print("   [WARNING] Network rebuild produced no edges.")
        except Exception as e:
            print(f"   [WARNING] Network rebuild failed: {e}")

    # ── 9. Re-enrich ONLY the affected docs ──────────────────────────────────
    if not args.skip_enrich:
        filtered = [
            doc_uces
            for doc_uces in uces_por_doc
            if doc_uces and doc_uces[0].doc_id in re_enrich_docs
        ]
        print(
            f"=== Re-enriching {len(filtered)} docs: "
            f"{[d[0].doc_id for d in filtered]} ==="
        )
        if filtered:
            gram_pipeline.procesar_desde_uces(
                uces_por_doc=filtered,
                global_corpus=global_corpus,
                lex_analyzer=lex_analyzer,
                corpus_raw=uwu,
            )
            print("   Re-enrichment complete.")
    else:
        print("   [SKIP] Re-enrichment skipped (--skip-enrich).")

    # ── 10. Save everything ──────────────────────────────────────────────────
    print("=== Saving ===")
    if args.dry_run:
        print("   [dry-run] Skipping DB writes.")
        print("   Would save:", len(uces), "UCEs.")
        return 0

    serialize_uces(
        orchestrator.db, [uce for doc_uces in uces_por_doc for uce in doc_uces]
    )
    orchestrator.db._save()
    print("=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
