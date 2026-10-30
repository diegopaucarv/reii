# Patchset final `v5` — consolidación de los 7 documentos de diseño

Estado: **consolidado, pendiente de aplicación manual**. No se ha tocado
`src/**` (scope de escritura prohibido), ni `data/`, ni la BD de Optuna, ni la
corrida `reii-workflow-run-94ac7f4a75bf` que sigue UP con el código viejo.

Fuentes consolidadas: `BACKEND_C.md`, `GATE_NOISE.md`, `METRICS_PAPER.md`,
`AUDIT_V4.md`, `SELECT_V4.md`, `CLASSIC_MODE.md`, `V5_SPACE.md`.
Fuentes en curso (no leídas, no escritas, reservadas): `STABILITY_NULL.md`,
`HARDENING.md`.

Verificación de anclas: se muestrearon **34 pares `archivo:línea`** con
`grep`/`read_file` sobre `src/`. 22 coinciden exactamente y 12 tienen desfase
(§3.5). Los números de línea de este documento son los **reales** del árbol de
trabajo en `main_workflow.py` = **10 728 líneas** (los docs dicen 10 697).

---

## §0 Cómo usar este documento

Este patchset se aplica **a mano**, sobre `src/**`, en el orden dado por §2 y
respaldando antes lo que dice §10. Los diffs viven en los siete documentos de
origen; aquí no se copia el código de producción, se inventaría, se ordena y se
resuelven los conflictos. Cada paso del §2 es un bloque con una frontera
verificable: aplícalo entero, ejecuta el comando de verificación indicado y no
pases al siguiente hasta que pase. El orden no es estilístico: la numeración de
los bloques es una cadena de dependencias reales (por ejemplo el bloque 3 no
compila sin el 2, y el 5 depende de campos que introduce el 3).

Antes de tocar nada, lee §7 y ejecuta la sección «verificación previa»: es
read-only y te dice si el árbol está en el estado que este documento asume.
Antes de aplicar cada bloque, lee su fila en §1 y su pregunta en §8: el §8 no
dice «revisar», dice qué comprobar y cuál es el valor correcto. Los documentos
de origen no son autoridad sobre el código: cuando §3 dice «se aplica Z», es
porque dos fuentes dicen cosas distintas y una tiene evidencia medida; cuando
dice **DECISIÓN DEL USUARIO**, las dos ramas son defendibles y eliges tú. Hay
dos fuentes todavía en escritura (`STABILITY_NULL.md`, `HARDENING.md`): sus
hunks no están aquí y no deben aplicarse de memoria; §4 y §5 reservan el punto
de inserción exacto.

---

## §1 Inventario de hunks

Tipos: `add` (inserción), `mod` (reemplazo/edición), `del` (borrado),
`rewrite` (fichero completo). «Hunk completo» = el documento de origen contiene
el texto literal aplicable (`sí`), o sólo la especificación (`no`).

| #   | archivo destino                         | símbolo / función                                                                                                                                                                       | doc fuente (§ exacta)                                 | tipo    | riesgo                      | depende de                      | completo        |
| --- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------- | --------------------------- | ------------------------------- | --------------- |
| 1   | `src/reii/quality/metrics.py`           | imports `typing` (L6: añadir `Any`)                                                                                                                                                     | METRICS_PAPER §5.1                                    | mod     | bajo                        | —                               | sí              |
| 2   | `src/reii/quality/metrics.py`           | append tras `external_coherence` (L612, EOF 650): `_row_normalize`, `_cos_to_centroid`, `internal_coherence`, `_entropy_stats`, `partition_diversity`, `_rmsd_core`, `rmsd_proportions` | METRICS_PAPER §5.2                                    | add     | medio                       | reusa `bootstrap_ci` (L335)     | sí              |
| 3   | `src/reii/quality/metrics.py`           | `bootstrap_stability` (motor: submuestra + reajuste + ARI medio)                                                                                                                        | STABILITY_NULL §7(a)                                  | add     | **alto**                    | #2                              | sí              |
| 4   | `src/reii/quality/metrics.py`           | `null_partition_test` (nulo de **datos**, no de etiquetas)                                                                                                                              | STABILITY_NULL §7(a)                                  | add     | **alto**                    | #3                              | sí              |
| 5   | `src/reii/method_selection.py`          | docstring de la regla (L17-40)                                                                                                                                                          | SELECT_V4 §7.1                                        | mod     | medio                       | —                               | sí              |
| 6   | `src/reii/method_selection.py`          | `import logging` (L78-79)                                                                                                                                                               | SELECT_V4 §7.2                                        | add     | bajo                        | #14                             | sí              |
| 7   | `src/reii/method_selection.py`          | `RULE_VERSION` (L86) `select-v3`→`select-v4`                                                                                                                                            | SELECT_V4 §7.3                                        | mod     | bajo                        | —                               | sí              |
| 8   | `src/reii/method_selection.py`          | `MethodMetrics` campos + propiedades + `as_row` (L121-137)                                                                                                                              | SELECT_V4 §7.4                                        | mod     | **alto**                    | —                               | sí              |
| 9   | `src/reii/method_selection.py`          | `Selection` campos (L165-168)                                                                                                                                                           | SELECT_V4 §7.5                                        | mod     | medio                       | #8                              | sí              |
| 10  | `src/reii/method_selection.py`          | `Selection.from_dict` lista de `NaN` (L191-200)                                                                                                                                         | SELECT_V4 §7.6                                        | mod     | bajo                        | #8 #9                           | sí              |
| 11  | `src/reii/method_selection.py`          | `Selection.explain_lines` (L207-265)                                                                                                                                                    | SELECT_V4 §7.7                                        | mod     | medio                       | #8 #9                           | sí              |
| 12  | `src/reii/method_selection.py`          | constantes `_DIAGNOSTIC_BACKENDS`/`_TAUTOLOGICAL_BACKENDS`/`_NULL_MODES`, `_fmt`, `_null_class`, `_feasibility` (L271-309)                                                              | SELECT_V4 §7.8                                        | mod     | **alto**                    | #8                              | sí              |
| 13  | `src/reii/method_selection.py`          | `select_method` (L312-423)                                                                                                                                                              | SELECT_V4 §7.9                                        | mod     | **alto**                    | #8 #9 #12                       | sí              |
| 14  | `src/reii/method_selection.py`          | `load_selection` + `last_load_rejection` (L476-500)                                                                                                                                     | SELECT_V4 §7.10                                       | mod     | **alto**                    | #6 #7                           | sí              |
| 15  | `src/reii/main_workflow.py`             | `Config.method_selection_max_class_share` 0.90→0.70 (L1003-1006)                                                                                                                        | GATE_NOISE §12.2                                      | mod     | bajo                        | —                               | sí              |
| 16  | `src/reii/main_workflow.py`             | `select_method(..., max_class_share_max=…)` default 0.90→0.70 (L5706-5707)                                                                                                              | GATE_NOISE §12.2                                      | mod     | bajo                        | #15                             | sí              |
| 17  | `src/reii/main_workflow.py`             | `trial.set_user_attr("dominance_corpus_winner", …)` (tras L8672)                                                                                                                        | GATE_NOISE §12.2                                      | add     | bajo                        | #8                              | sí              |
| 18  | `src/reii/main_workflow.py`             | campos `n_total`/`n_assigned`/`n_noise`/`max_class_size`/`backend`/`diagnostic_only` en `MethodMetrics`                                                                                 | AUDIT_V4 §8.1                                         | add     | —                           | **REFERENCIA: ya en #8**        | no              |
| 19  | `src/reii/main_workflow.py`             | `_raw_stats(_m)` + `**_raw_stats(_m)` en la construcción de `MethodMetrics` (L5665-5689)                                                                                                | AUDIT_V4 §8.1                                         | add/mod | medio                       | #8                              | sí              |
| 20  | `src/reii/main_workflow.py`             | `context.method_c_backend` + `context.spec_hash` en el payload (L5964-5966)                                                                                                             | AUDIT_V4 §8.1 / §3.2                                  | add     | medio                       | #19                             | sí              |
| 21  | `src/reii/main_workflow.py`             | `Config.method_c_backend`, `method_c_hdbscan_selectable`, `method_c_target_k`, `dense_cdh_min_r2_1d`/`_dense`, `dense_cdh_loose_factor` (L1038)                                         | BACKEND_C §12 (#21)                                   | add     | **alto**                    | #8                              | sí              |
| 22  | `src/reii/main_workflow.py`             | `#22a` imports (L80) · `#22b` extracción del corte 1-D (L3831) · `#22c` firma + dispatcher (L5089) · `#22d` sustitución del bloque HDBSCAN (L5165-5214) · `#22e` llamadas (L6124/L6131) | BACKEND_C §12 (#22a-#22e)                             | add/mod | **alto**                    | #21 #72 #72b                    | sí              |
| 23  | `src/reii/main_workflow.py`             | `run()`: poblar `bootstrap_ari`/`coverage_stable`/`null_p` desde `res_emb1` y del motor de estabilidad                                                                                  | SELECT_V4 §2.1 + STABILITY_NULL §7(c) + BACKEND_C §12 | mod     | **alto**                    | #3 #4 #8 #22                    | sí              |
| 24  | `src/reii/backend/config_store.py`      | `CONFIG_CATEGORIES["clasificacion"]` (L36): `cdh_enforce_min_leaf`, `hdbscan_uce_min_cluster_frac`, `…_frac_loose`, `…_floor`, `…_samples_ratio`                                        | CLASSIC_MODE §9.7 / §8.1                              | mod     | bajo                        | —                               | sí              |
| 25  | `src/reii/backend/config_store.py`      | `verdict_mode`, `primary_cluster_source` en `"clasificacion"`                                                                                                                           | CLASSIC_MODE §9.7                                     | mod     | bajo                        | —                               | sí              |
| 26  | `src/reii/backend/config_store.py`      | `auto_calibrate_uce_size` en `"afc_y_proyeccion"`                                                                                                                                       | CLASSIC_MODE §9.7                                     | mod     | bajo                        | —                               | sí              |
| 27  | `src/reii/backend/steps.py`             | `_STEP_CONFIG_EXTRA_KEYS`, `_STEP_EXTRA_DEFAULTS`, `_config_subset` (L166-194)                                                                                                          | CLASSIC_MODE §9.8                                     | mod     | medio                       | #32 (`run_mode`)                | sí              |
| 28  | `src/9_selection_audit.py`              | reemplazo completo del fichero (122→470 líneas, 23 funciones)                                                                                                                           | AUDIT_V4 §4                                           | rewrite | **alto**                    | #8 #9 #14 #19 #20               | sí              |
| 29  | `src/reii/main_workflow.py`             | `CLASSIC_OVERRIDES`, `_CLASSIC_SPEC_FIELDS`, `run_spec_id`, `apply_classic_mode`                                                                                                        | CLASSIC_MODE §9.1                                     | add     | medio                       | #50 #51                         | sí              |
| 30  | `src/reii/main_workflow.py`             | `_NON_CONFIG_PARAM_KEYS`, `resolve_best_params`                                                                                                                                         | CLASSIC_MODE §9.1                                     | add     | bajo                        | —                               | sí              |
| 31  | `src/reii/main_workflow.py`             | `_MAX_TB_CHARS` (=4096) + `guarded_trial_eval`                                                                                                                                          | CLASSIC_MODE §9.1                                     | add     | **alto** (colisión con #59) | —                               | sí              |
| 32  | `src/reii/main_workflow.py`             | `Config.run_mode: str = "modern"`                                                                                                                                                       | CLASSIC_MODE §9.2                                     | add     | bajo                        | —                               | sí              |
| 33  | `src/reii/main_workflow.py`             | `_CLI_MODES_VALIDOS` (L347): `"classic"`                                                                                                                                                | CLASSIC_MODE §9.2                                     | mod     | bajo                        | #32                             | sí              |
| 34  | `src/reii/main_workflow.py`             | `__main__`: `_cli_mode != "classic"` en el override                                                                                                                                     | CLASSIC_MODE §9.2                                     | mod     | bajo                        | #33                             | sí              |
| 35  | `src/reii/main_workflow.py`             | `__main__`: `apply_classic_mode(...)` tras `cfg = UCBuilderConfig()`                                                                                                                    | CLASSIC_MODE §9.2                                     | add     | medio                       | #29 #32                         | sí              |
| 36  | `src/reii/main_workflow.py`             | `__main__`: `grammatical_dashboard_path` y `export_network_gexf` aislados                                                                                                               | CLASSIC_MODE §9.2                                     | mod     | bajo                        | #51                             | sí              |
| 37  | `src/reii/main_workflow.py`             | `_evaluate_trial`: `pseudocount` condicional a `run_mode`                                                                                                                               | CLASSIC_MODE §9.3                                     | mod     | bajo                        | #32                             | sí              |
| 38  | `src/reii/main_workflow.py`             | `_evaluate_trial`: `try/except` con `exception_*`                                                                                                                                       | CLASSIC_MODE §9.3                                     | mod     | —                           | **DESCARTADO, ver §3.4**        | no              |
| 39  | `src/reii/main_workflow.py`             | rama `select`: `-K` = `k_ach` + attrs `k_ach`/`k_target`/`k_shortfall`/`k_stable` (L8683)                                                                                               | CLASSIC_MODE §9.3 / §7                                | mod     | medio                       | —                               | sí              |
| 40  | `src/reii/main_workflow.py`             | rama `wc_only`: `-K` = `k_ach` + los mismos attrs (L8702)                                                                                                                               | CLASSIC_MODE §9.3 / §7                                | mod     | medio                       | —                               | sí              |
| 41  | `src/reii/main_workflow.py`             | `study.optimize(lambda … guarded_trial_eval …)` (L8975)                                                                                                                                 | CLASSIC_MODE §9.4                                     | mod     | —                           | **DESCARTADO, ver §3.4**        | no              |
| 42  | `src/reii/main_workflow.py`             | resumen post-`optimize`: print de `_trial_failures`                                                                                                                                     | CLASSIC_MODE §9.4                                     | add     | —                           | **DESCARTADO (depende de #41)** | no              |
| 43  | `src/reii/main_workflow.py`             | `Optimizador.load_best_params`: `resolve_best_params` (L8336)                                                                                                                           | CLASSIC_MODE §9.4                                     | mod     | medio                       | #30                             | sí              |
| 44  | `src/reii/main_workflow.py`             | `ejecutar`: guarda `classic` al leer `best_params.json`                                                                                                                                 | CLASSIC_MODE §9.4                                     | add     | medio                       | #32                             | sí              |
| 45  | `src/reii/main_workflow.py`             | `_apply_best_params`: `resolve_best_params` + `pseudocount` condicional (L9633)                                                                                                         | CLASSIC_MODE §9.4                                     | mod     | medio                       | #30 #37                         | sí              |
| 46  | `src/reii/main_workflow.py`             | `WorkflowOrchestrator.__init__`: `seed_from_config` sólo en moderno                                                                                                                     | CLASSIC_MODE §9.5                                     | mod     | medio                       | #32                             | sí              |
| 47  | `src/reii/main_workflow.py`             | `ejecutar` pass 1 (L9925): retrofitting condicional a `use_embeddings`                                                                                                                  | CLASSIC_MODE §9.5                                     | mod     | medio                       | #32                             | sí              |
| 48  | `src/reii/main_workflow.py`             | `DoubleClassifier.persist_method_selection`: no-op en clásico (L5951)                                                                                                                   | CLASSIC_MODE §9.5                                     | add     | bajo                        | #32                             | sí              |
| 49  | `src/reii/main_workflow.py`             | `_classic_stability` (método) + `_classic_stability_text` (módulo) + llamada en `ejecutar` (L9959)                                                                                      | CLASSIC_MODE §9.5 / §5                                | add     | medio                       | #29 #32                         | sí              |
| 50  | `src/reii/config.py`                    | `CLASSIC_DATA_DIR` (junto a `BEST_PARAMS_PATH`, L101)                                                                                                                                   | CLASSIC_MODE §9.6                                     | add     | bajo                        | —                               | sí              |
| 51  | `src/reii/config.py`                    | `CLASSIC_OUTPUT_DIR` (antes de `OUTPUT_DASHBOARD`, L252)                                                                                                                                | CLASSIC_MODE §9.6                                     | add     | bajo                        | —                               | sí              |
| 52  | `src/reii/pipelines/transcripciones.py` | redirección de `main_workflow_clasico.py` a `main_workflow.py --mode classic` (L155)                                                                                                    | CLASSIC_MODE §9.10                                    | mod     | medio                       | **EXCLUIDO** (§3.6, §9)         | sí (no aplicar) |
| 53  | `src/reii/main_workflow_clasico.py`     | shim de redirección (9.9)                                                                                                                                                               | CLASSIC_MODE §9.9                                     | rewrite | **alto**                    | **DECISIÓN DEL USUARIO** (§3.6) | sí (no aplicar) |
| 54  | `src/reii/main_workflow.py`             | `Config`: `optimize_n_startup_trials` 25→12, `optimize_n_ei_candidates`, `optimize_warm_start_max`, `optimize_freeze_inert` (L923-931)                                                  | V5_SPACE §7.1                                         | mod/add | bajo                        | —                               | sí              |
| 55  | `src/reii/main_workflow.py`             | `_suggest_conditional` + `_expand_levels` tras `_cdh_frac_from_params` (fin L7883)                                                                                                      | V5_SPACE §7.2                                         | add     | medio                       | —                               | sí              |
| 56  | `src/reii/main_workflow.py`             | `_precompute_coref_chains` (L7988-8000)                                                                                                                                                 | V5_SPACE §7.3                                         | mod     | **alto**                    | #55 #57                         | sí              |
| 57  | `src/reii/main_workflow.py`             | `get_search_space` (L8409-8432)                                                                                                                                                         | V5_SPACE §7.4                                         | mod     | **alto**                    | #54 #58                         | sí              |
| 58  | `src/reii/main_workflow.py`             | `_suggest` dos pasadas (L8435-8448)                                                                                                                                                     | V5_SPACE §7.5                                         | mod     | **alto**                    | #55 #57                         | sí              |
| 59  | `src/reii/main_workflow.py`             | `_MAX_TB_CHARS` (=2000), `_sentinel`, `_trial_exc_attrs`, `_make_guarded_objective`, `_sentinel_breakdown` (tras L7855)                                                                 | V5_SPACE §7.6                                         | add     | **alto** (colisión con #31) | —                               | sí              |
| 60  | `src/reii/main_workflow.py`             | `_evaluate_trial`: quitar el `except` local (L8589-8593)                                                                                                                                | V5_SPACE §7.7                                         | del     | **alto**                    | #59                             | sí              |
| 61  | `src/reii/main_workflow.py`             | 12 retornos de gate → `_sentinel(trial, motivo)`                                                                                                                                        | V5_SPACE §7.8                                         | mod     | medio                       | #59                             | sí              |
| 62  | `src/reii/main_workflow.py`             | `_make_sampler`: `n_ei_candidates` + `warn_independent_sampling=False` (L8773-8778)                                                                                                     | V5_SPACE §7.9                                         | mod     | medio                       | #54                             | sí              |
| 63  | `src/reii/main_workflow.py`             | warm start acotado en `_run_single_stage` (L8949)                                                                                                                                       | V5_SPACE §7.10                                        | mod     | medio                       | #54                             | sí              |
| 64  | `src/reii/main_workflow.py`             | `study.optimize(_make_guarded_objective(...))` + resumen de centinelas (L8975-8994)                                                                                                     | V5_SPACE §7.11                                        | mod     | **alto**                    | #59                             | sí              |
| 65  | `src/reii/main_workflow.py`             | `_bp_params`: emitir los congelados (L8188)                                                                                                                                             | V5_SPACE §7.12                                        | mod     | medio                       | #57                             | sí              |
| 66  | `src/reii/method_selection.py`          | `Selection.null_mode` default `"gate"`->`"diagnostic"`                                                                                                                                  | STABILITY_NULL §7(b)                                  | mod     | medio                       | #9                              | sí              |
| 67  | `src/reii/method_selection.py`          | `_feasibility`: `null_mode="diagnostic"` + piso `bootstrap_ari mean >= 0.30`                                                                                                            | STABILITY_NULL §7(b)                                  | mod     | **alto**                    | #12                             | sí              |
| 68  | `src/reii/method_selection.py`          | textos de `explain_lines` para el modo `diagnostic`                                                                                                                                     | STABILITY_NULL §7(b)                                  | mod     | medio                       | #11                             | sí              |
| 69  | `src/reii/main_workflow.py`             | `ward_max_n` (cota de ward por n) — consumida por #72, no por `Config`                                                                                                                  | HARDENING §2.2                                        | add     | **alto**                    | #22 #72                         | sí              |
| 70  | `src/reii/main_workflow.py`             | `Kach vs target_k`: sólo el ORDEN del gate (K real ya refinado)                                                                                                                         | HARDENING §5, §8.3                                    | mod     | **alto**                    | #39 #40                         | sí              |
| 71  | `src/reii/main_workflow.py`             | `except` mudo de `_evaluacion_rapida` (el del FAIL #81)                                                                                                                                 | HARDENING §6, §8.2                                    | mod     | **alto**                    | — (distinto de #60)             | sí              |
| 72  | `src/reii/method_c_policy.py`           | MÓDULO NUEVO: `resolve_method_c_backend`, `Precondition`, `ward_guard`, `effective_target_k`, `preflight`, `format_target_k_notice`, `guarded_fit`                                      | HARDENING §8.1                                        | add     | **alto**                    | #22                             | sí              |
| 73  | `src/reii/method_c_policy.py`           | Ampliación del módulo: `corte_optimo_1d`, `_pc1_local`, `dense_refine`, `r2_dense`, los 3 cuerpos densos y `build_method_c_result`                                                      | BACKEND_C §12 (#72b)                                  | add     | **alto**                    | #72                             | sí              |

**Totales: 73 filas.** Las 73 son aplicables; **cero PENDING, cero huecos**.
Quedan tres excepciones explícitas, todas deliberadas: #18 es una **referencia
redundante** (sus 6 campos ya vienen en #8, no debe aplicarse dos veces); #38,
#41 y #42 están **descartadas por conflicto** (§3.4); #52 y #53 quedan **fuera de
alcance** porque `main_workflow_clasico.py` es la baseline del Método A (§3.6).

Corrección de rutas respecto del plan: los ficheros de allowlist están en
`src/reii/backend/config_store.py` y `src/reii/backend/steps.py`, **no** en
`src/reii/`. El plan los nombraba sin el subdirectorio `backend/`.

---

## §2 Orden de aplicación (7 bloques)

Cada bloque es una unidad de verificación. No reordenes: `#8` (campos de
`MethodMetrics`) es requisito de #9-#13, #17, #19, #23 y #28; y `#57`+`#58`
son requisito de #56.

### Bloque 1 — `src/reii/quality/metrics.py` (#1, #2)

Funciones puras nuevas: `C_in` (`internal_coherence`), `d`
(`partition_diversity`), `RMSD` (`rmsd_proportions`) y 4 helpers privados.
No cablea nada al pipeline (`METRICS_PAPER` §7.1 lo deja como tarea aparte), así
que es el bloque de menor riesgo y se puede aislar.

Verificación: `python -m pytest tests/test_quality_core.py -q` (debe quedar igual
que antes: el bloque sólo **añade** símbolos) y
`python scripts/playground/gate_calibration.py --metrics-selftest` → `[ok]`.

**Pendiente del bloque**: `bootstrap_stability` y `null_partition_test` (#3,
#4) no tienen diff en ningún documento. No los inventes aquí; van con
`STABILITY_NULL.md` (§4).

### Bloque 2 — `src/reii/method_selection.py` (#5-#14)

Aplica **sólo** la versión de `SELECT_V4.md` §7.1-§7.10. Los bloques de
`GATE_NOISE.md` §12.1 cubren la misma región y están superseded (§3.1); no los
apliques después ni en paralelo o tendrás `dominance_corpus` declarado dos veces
con tipos incompatibles.

Orden interno: #5-#7 (cabecera), #8 (campos, base de todo), #9, #10, #11, #12,
#13, #14. El fichero no debe importarse a medias: #12 usa `dominance_corpus_effective`
de #8 y #13 usa `_feasibility` de #12.

Verificación: `python -c "from reii import method_selection as m; print(m.RULE_VERSION)"`
→ `select-v4`. Luego
`python -m pytest tests/test_select_v4.py -q` → **debe dejar de saltarse** y
pasar (42 tests). Si sigue todo en `skipped`, el `RULE_VERSION` no se aplicó.
Finalmente `python -m pytest tests/test_method_selection.py -q` → **fallarán
exactamente 2** hasta que hagas #6.2 de este documento; cualquier otro fallo es
un error de aplicación.

### Bloque 3 — `src/reii/main_workflow.py` (productor) (#15-#23)

Suborden obligatorio: #15-#17 (umbral de dominancia y su `user_attr`), #19
(`_raw_stats` y kwargs en la construcción de `MethodMetrics`), #20 (payload
`context`), #72/#72b (crear `method_c_policy.py` **antes** de llamarlo), #21
(campos de `Config`), #22a-#22e (dispatcher y backends densos), #69 (tope de
ward, consumido por #72), #23 (integración de estabilidad en `run()`).
#18 **no se aplica**: sus 6 campos ya vienen en #8.

Verificación: `python -c "import ast,sys; ast.parse(open('src/reii/main_workflow.py',encoding='utf-8').read())"`
y `python scripts/playground/classic_profile_check.py` (parsea `Config` por AST:
debe seguir diciendo `[ok]`). Los backends densos se validan con
`python scripts/playground/c_backends_arena.py` -> determinismo `[ok]` en las dos
corridas por celda, y con `python -m pytest tests/test_method_c_backends.py -q`.

**Orden crítico**: #72/#72b crean el módulo **antes** de que #22c lo importe; y
#22b (extracción del corte) debe ir **antes** de #22c, o el dispatcher llama a
una función que aún no existe. `hdbscan` se sigue computando siempre y se marca
`diagnostic_only=True` si `method_c_hdbscan_selectable=False`: no se elimina,
porque el usuario lo quiere como diagnóstico.

### Bloque 4 — allowlists (#24-#27)

`src/reii/backend/config_store.py` (superficie del dashboard) y
`src/reii/backend/steps.py` (payload del `step_cache`). El diseño de #27 es
«extras condicionales»: si la clave vale su default, **no** entra en el hash, de
modo que los `input_hash` ya persistidos siguen válidos. No se bumpea `version`
del step (`CLASSIC_MODE.md` §8.2).

Verificación: `python -c "import reii.backend.config_store as m; print(len(m.ALL_CONFIG_KEYS))"`
(no debe lanzar; el número sube exactamente en las claves añadidas por #24-#26) y
`python -m pytest tests/test_pipelines_data.py tests/test_premerge_cache.py -q`
(no debe cambiar nada respecto de la línea base). Verificación fuerte:
`python -m pytest tests/test_mode_classic.py -q`, cuyos tests de perfil y de
claves de `Config` corren en este bloque.

### Bloque 5 — `src/9_selection_audit.py` (#28)

Va **después** del 3 porque el auditor nuevo valida invariantes 2, 3 y 4 leyendo
`all_metrics[i].n_assigned`, `max_class_size`, `backend`, `diagnostic_only`,
`dominance_corpus` y `context.spec_hash`: sin el bloque 3 el payload no las
tiene y el auditor reporta `[!]` no verificable.

Verificación: `python scripts/playground/audit_smoke.py` → `[ok]` en los
fixtures, y `python src/9_selection_audit.py --strict --json -` sobre el
historial actual (exit 0 o 1 según inestabilidad, **nunca 2** con historial).

### Bloque 6 — `--mode classic` (#29-#51, menos #38/#41/#42 y #52/#53)

`CLASSIC_MODE.md` §9.1-§9.6 + §9.7/§9.8 (que están en el bloque 4). Suborden:
#50/#51 (`config.py`, porque #29 los importa) → #29/#30/#32 (helpers y `run_mode`)
→ #33-#36 (CLI) → #37, #39, #40 → #43-#45 (lector de params) → #46-#49
(aislamiento y estabilidad clásica) → #31 sólo si decides mantenerlo (§3.4).

No apliques #38, #41 ni #42: el guard autoritativo es #59/#64 del bloque 7
(§3.4).

Verificación: `python scripts/playground/classic_mode_probe.py` → `[ok] todos los
asserts pasaron`, y `python -m pytest tests/test_mode_classic.py -q` → 14 passed
(ya no debe haber `skipped`). Después del bloque 7, además
`python src/reii/main_workflow.py --mode classic --optimize-only` debe imprimir
`[i] classic: sin method_selection.json`.

### Bloque 7 — espacio v5 (#54-#65)

`V5_SPACE.md` §7.1-§7.12. **#57 y #58 son inseparables de #56**: con el spec
`dict` y sin la segunda pasada de `_suggest`, `_precompute_coref_chains` cae en
`ctx_values = {int(c) for c in ctx_values if int(c) > 0}` (**L7999**, fuera del
`try` de L7988-8000) e intenta `int()` sobre un `dict` → `TypeError` no capturado.
Suborden seguro: #55 → #54 → #58 → #57 → #56 → #59/#60/#61 → #62 → #63 → #64
→ #65.

Antes de #59, resuelve la colisión de `_MAX_TB_CHARS` (§3.4).

Verificación: `python scripts/playground/v5_space_sim.py` → `[ok] simulacion
completa`, con `inviables 0.00%`; y
`python -m pytest tests/test_v5_space.py -q` → 8 passed (los 2 rojos actuales
desaparecen).

---

## §3 Conflictos entre documentos y su resolución

### 3.1 `dominance_corpus`: propiedad vs campo, y dos implementaciones de la misma regla

- **A** (`GATE_NOISE.md:§12.1`) hace de `dominance_corpus` una **`@property`**
  calculada `max_class_share * (1 - noise_frac)`, la añade a `as_row` y cambia
  `_feasibility` para usar `dom * (1 - noise)`.
- **B** (`SELECT_V4.md:§7.4`, `§7.8`) la declara como **campo** del dataclass
  (`dominance_corpus: float = nan`, medido por el productor) y añade la
  propiedad `dominance_corpus_effective`, con precedencia explícita
  1. campo 2) `max_class_size / n_total` 3) la identidad de A.

⇒ **Se aplica B.** Porque los dos no pueden coexistir: un campo de dataclass y
una propiedad con el mismo nombre se pisan (`asdict` intenta serializar el
`property` y `Selection.from_dict` espera un número), y porque la precedencia de
B detecta el caso que A no puede: «si `a` se calculó mal, `s_max / n_total` lo
delata» (`SELECT_V4.md` §7.4). La identidad de A sobrevive como rama 3 de B, así
que no se pierde calibración. Consecuencia práctica: **todos** los hunks de
`GATE_NOISE.md` §12.1 son referencia, no aplicación; de §12.2 sobreviven sólo
#15, #16 y #17, que B no cubre.
Riesgo remanente: `AUDIT_V4.md` §3.2 pide leer `all_metrics[i].dominance_corpus`;
con B ese campo es `nan`/`null` salvo que el productor lo mida. El auditor debe
leer **`dominance_corpus_eff`** (que `as_row` sí emite con valor). Anótalo al
aplicar #28.

### 3.2 `BACKEND_C.md` no contenía diffs — RESUELTO con §12

`BACKEND_C.md` define el contrato (§1), la decisión por backend (§2), el control
de K (§3) y los casos límite (§7), pero su cabecera declaraba: «No hay cambios
aplicados en `src/reii/**`; los diffs van en el mensaje de entrega». Esa era la
razón por la que #21/#22 no eran aplicables.

⇒ **Ya tienen diff.** Se añadió `## 12. Diffs exactos` (1363 líneas, 30 fences
balanceados, 10/10 cabeceras `@@` consistentes, bloques Python compilables). La
numeración es §12 y no §7 porque §7 estaba ocupado por «Casos límite» y §8-§11
ya existían; renumerar habría roto las referencias de este índice. Contiene:

| hunk | contenido                                                                                                                                | ancla real               |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| #21  | `Config.method_c_backend`, `method_c_hdbscan_selectable`, `method_c_target_k`, `dense_cdh_min_r2_1d/_dense`, `dense_cdh_loose_factor`    | `main_workflow.py` L1038 |
| #22a | imports del módulo de política                                                                                                           | L80                      |
| #22b | extracción del corte 1-D                                                                                                                 | L3831                    |
| #22c | firma pública + dispatcher                                                                                                               | L5089                    |
| #22d | sustitución del bloque HDBSCAN                                                                                                           | L5165-5214               |
| #22e | sitios de llamada                                                                                                                        | L6124, L6131             |
| #72b | ampliación de `method_c_policy.py`: `corte_optimo_1d`, `_pc1_local`, `dense_refine`, `r2_dense`, los 3 cuerpos y `build_method_c_result` | nuevo                    |

**Corrección de ancla importante**: `_corte_optimo` está en
`main_workflow.py:3831` (`ClasificadorDescendente`, usado en L4123), **no**
dentro de `_clasificar_umbral_embedding` (L5089), que hoy es un envoltorio
estático de HDBSCAN. Extraerlo de L5089 habría sido inventar código.

**Sin duplicación de guardas**: el dispatcher no contiene ninguna condición de
n/K/memoria/salida; una sola llamada a `guarded_fit(...)` y consume
`resolve_method_c_backend`, `format_target_k_notice` y `build_method_c_result`.
Todo el acotado vive en `method_c_policy.py` (#72/#72b).

Riesgo residual declarado: el gate `r2_dense = 0.08` está calibrado a **d=30**;
con `target_dims = min(30, n//10, d)` (L5150), un corpus con n≲200 queda con
d=10 y el gate se vuelve laxo (medido: ruido con n=500, d=10 da
`mean_r2_dense=0.142` y `dense_cdh` devuelve K=8). Es una limitación de
calibración, no del dispatcher.

### 3.3 `null_mode`: `"gate"` por defecto contradice la evidencia del banco (inercia)

- **A** (`SELECT_V4.md:§4`) fija `null_mode = "gate"` como default, con el
  argumento de que un gate vacuo es inofensivo y un ranking sesgado no.
- **B** (evidencia en `GATE_NOISE.md` y el banco `gate_interval_probe.py`): sobre
  ruido puro el nulo de **etiquetas permutadas** da `p = 0.005` (el mínimo
  alcanzable con `n_perm = 200`), es decir el gate **nunca** rechaza. Un gate
  que no rechaza no es «inofensivo»: es un criterio que se cree activo y no lo
  está, y `SELECT_V4.md` §4.4 admite que con nulo de etiquetas «no puede
  producir abstención sobre ruido».

=> **RESUELTO.** Se aplica `null_mode="diagnostic"` mas el piso
`bootstrap_ari mean >= 0.30`, y ya no es una hipotesis: la evidencia esta en el
repo. `STABILITY_NULL.md` §2 mide `p = 0.005` en las 12 filas del banco,
incluidas las tres de `synth_null`; el piso de bootstrap es lo que discrimina
(ruido 0.09-0.11, real 0.42-0.60, estructura 0.78-0.97).

Se materializa en #66/#67/#68, documentados en §4 de este patchset. `\"gate\"`
sigue existiendo como valor valido (util con un nulo de DATOS, hoy inexistente)
y se sigue testeando con `null_p` inyectado. El default queda fijado, no
diferido: un gate que nunca rechaza no puede ser el default de nada.

### 3.4 Dos guards de excepción y un `_MAX_TB_CHARS` duplicado

- **A** (`CLASSIC_MODE.md:§9.1`,`§9.3`,`§9.4`) define `_MAX_TB_CHARS = 4096` y
  `guarded_trial_eval`, enriquece el `try/except` **local** de
  `_evaluate_trial` con `exception_type`/`exception_msg`/`exception_tb`, y envuelve
  `study.optimize` con `lambda t: guarded_trial_eval(...)`.
- **B** (`V5_SPACE.md:§7.6`,`§7.7`,`§7.11`) define `_MAX_TB_CHARS = 2000` y
  `_make_guarded_objective`, **borra** el `try/except` local y envuelve
  `study.optimize` con `_make_guarded_objective(self._evaluate_trial)`.

⇒ **Se aplica B** y se descartan #38, #41 y #42. Porque (i) los dos escriben la
misma llamada `study.optimize(...)`: aplicar ambos deja un guard dentro de otro y
el `lambda` de A recibe como `evaluate` un objetivo ya envuelto por B — funciona
por accidente pero duplica `user_attrs` con **nombres distintos**
(`error_type` vs `exception_type`), y `tests/test_v5_space.py` exige
`error_type`/`error_msg`/`traceback`; (ii) A enriquece el `except` que B borra:
son mutuamente excluyentes, no complementarios; (iii) B separa
`outcome="gate"` de `outcome="exception"` y añade `_sentinel_breakdown`, que es
lo que permite distinguir el FAIL #81 de un gate, y A no. **Colisión de nombre**:
`_MAX_TB_CHARS` se define en ambos; aplica **una sola vez** con el valor de B
(2000) o conserva el de A (4096) y borra la línea de B — pero nunca las dos, o
tendrás un `NameError`/valor silenciosamente redefinido según el orden de
inserción. Recomendado: **2000** (B), porque el `traceback` vive en
`user_attrs` y se serializa en el JSON comparativo.
Coste asumido: se pierde el print de `_trial_failures[-3:]` de A; el resumen de
#64 imprime los 5 primeros con `[X]`, que cubre el diagnóstico.

### 3.5 Desfases de línea de los documentos (anclas verificadas)

Medidos sobre el árbol actual. Se citan sólo los que **no** cuadran:

| doc:§             | dice                          | real                                                                | delta                  |
| ----------------- | ----------------------------- | ------------------------------------------------------------------- | ---------------------- |
| V5_SPACE §7.6     | `_is_sentinel` en L7857       | **L7855**                                                           | −2                     |
| CLASSIC_MODE §9.1 | `_SENTINEL_TRIAL` en L7851    | **L7852**                                                           | +1                     |
| V5_SPACE §7.3     | rango L7988-7997              | el `except` cierra en **L8000**; el `int(c)` peligroso en **L7999** | +2                     |
| V5_SPACE §7.4     | `get_search_space` L8408-8432 | dict en **L8409-8432** (`def` en L8384)                             | +1                     |
| V5_SPACE §7.5     | `_suggest` L8434-8448         | **L8435-8448**                                                      | +1                     |
| V5_SPACE §7.7     | `try/except` L8590-8594       | **L8589-8593**                                                      | −1                     |
| V5_SPACE cabecera | archivo de 10 697 líneas      | **10 728**                                                          | +31                    |
| AUDIT_V4 §8.1     | `MethodMetrics(` en L5687     | **L5677**                                                           | −10                    |
| AUDIT_V4 §8.1     | `context` en L5964            | **L5965**                                                           | +1                     |
| AUDIT_V4 §4       | «122 → 998 líneas»            | el hunk es `@@ -1,122 +1,470 @@` ⇒ **470**                          | inconsistencia interna |
| CLASSIC_MODE §9.5 | bloque de selección en L9952  | **L9959-9964**                                                      | +7                     |
| CLASSIC_MODE §9.9 | `transcripciones.py:151`      | **L155**                                                            | +4                     |

Anclas que **sí** coinciden exactamente (muestra de 34): `metrics.py:6`,
`metrics.py:335` (`bootstrap_ci`), `metrics.py:612` (`external_coherence`),
`method_selection.py:86/103/141/274/312/476`, `main_workflow.py:789` (`Config`),
`L7852` (`_SENTINEL_TRIAL`), `L7883` (`return float(default)`), `L8188`
(`_bp_params`), `L8435`, `L8589`, `L8767` (`_make_sampler`, bloque 8773-8778),
`L8949` (warm start), `L8975` (`study.optimize`), `L1006`
(`method_selection_max_class_share`), `L5707`, `L5665` (`_max_share`), `L8672`,
`L8683` y `L8702` (los dos `-float(n_clusters)`), `L10505`
(`classification_mode="select"`), `L10642` (`_optimizer_will_run`), `config.py:101`
y `252-254`, `backend/config_store.py:12/36/63`, `backend/steps.py:166/186`,
`pipelines/transcripciones.py:155`, `main_workflow.py:542`
(`_partition_sizes_from_labels`). Los 12 retornos de gate de `V5_SPACE` §7.8
coinciden **todos** (8561, 8563, 8598, 8604, 8610, 8628, 8650, 8652, 8690, 8695,
8700, 8718).

### 3.6 `main_workflow_clasico.py`: shim o baseline

- **A** (`CLASSIC_MODE.md:§9.9`,`§9.10`) propone convertirlo en shim de 30 líneas
  y redirigir `transcripciones.py` a `main_workflow.py --mode classic`.
- **B** (alcance de este patchset, §9) dice que un fichero de 4.519 líneas sin el
  fix de piso del CDH **es la baseline del Método A** y no se toca.

⇒ **Se aplica B**: ni #52 ni #53. Porque el paper necesita una referencia A
reproducible con el código que produjo los números históricos, y un shim la
destruye; y porque redirigir `transcripciones.py` cambiaría el comportamiento del
pipeline batch **sin verificación de particiones** (`CLASSIC_MODE.md` §12.2
admite que `cdh_enforce_min_leaf=False` altera K de forma no monótona).
**DECISIÓN DEL USUARIO** si `CLASSIC_MODE.md` §9.9 es requisito del paper:
entonces aplica #53 (shim) **sólo después** de archivar la corrida A como
`main_workflow_clasico_baseline.py` y de correr A y B sobre el mismo corpus.

---

### 3.7 Resolución de las 5 incidencias de ancla (`ANCHOR_CHECK.md`)

Verificado contra `src/**` **hoy** (anclas reales, no las de §1). **Esta sección
manda sobre §1 y §2** para los hunks que toca. Recuento final: de los 73 hunks,
**58 son aplicables**; los 15 restantes son 6 fuera de alcance (#24-#27, #52,
#53), 4 no-op/descartados (#18, #38, #41, #42) y **5 anulados aquí** (#31, #66,
#67, #68, #71).

| hunk | decisión             | por qué                                                     |
| ---: | -------------------- | ----------------------------------------------------------- |
|  #71 | **ANULADO**          | su ancla es exactamente el `except` que #60 borra           |
|  #31 | **ANULADO**          | `_MAX_TB_CHARS` + guard los aportan #59 y #64; 4096 vs 2000 |
|  #66 | **ABSORBIDO en #9**  | el campo lo crea #9: debe nacer ya en `"diagnostic"`        |
|  #67 | **ABSORBIDO en #12** | el piso entra en el rewrite de `_feasibility`               |
|  #68 | **ABSORBIDO en #11** | la línea de control entra en el rewrite de `explain_lines`  |

#### 3.7.1 #71 — no aplicar (crítico)

Hechos comprobados:

- `_evaluate_trial` se invoca en **un solo sitio**, `main_workflow.py:8976`,
  dentro de `study.optimize(...)`, que #64 envuelve con
  `_make_guarded_objective`. No hay ninguna llamada fuera del guard: al borrar el
  `except` local (#60) **no se pierde ninguna ruta de excepción**.
- `logger.debug("Trial %d exception: %s", ...)` aparece **una única vez** en el
  fichero (L8593). La afirmación de §5.2 de que son «dos `except` distintos» es
  falsa: `_evaluacion_rapida` (L8453-8497) sólo tiene `try:`/`finally:`.
- El rastro que #71 quería conservar ya lo emite #59 vía `_trial_exc_attrs` con
  las claves que exige `tests/test_v5_space.py`: `error_type`, `error_msg`,
  `traceback` (truncado a `_MAX_TB_CHARS`).

⇒ **No aplicar #71.** Si en el futuro se quiere el log de excepción _además_ del
attr, va dentro de `_make_guarded_objective` (#59), nunca como `except` local.

#### 3.7.2 #31 — no aplicar (absorbido por #59 + #64)

`#31` introducía `_MAX_TB_CHARS = 4096` y `guarded_trial_eval`; `#59` introduce
`_MAX_TB_CHARS = 2000`, `_sentinel`, `_trial_exc_attrs` y
`_make_guarded_objective`, y `#64` lo cablea en `study.optimize`. Con #38, #41 y
#42 ya descartados, **#31 no aporta nada y definiría el símbolo dos veces**.

Verificado: `guarded_trial_eval` **no existe hoy** ni en `src/` ni en `tests/`
(0 coincidencias), así que anular #31 no rompe ningún test. Consecuencia: los
snippets de `CLASSIC_MODE.md` §10.1 que importan `guarded_trial_eval` **no deben
crearse**; la cobertura de guard ya está en `tests/test_v5_space.py`.

⇒ `_MAX_TB_CHARS = 2000` entra **una sola vez**, con #59.

#### 3.7.3 #54 — el knob tiene 3 sitios, no 1

Los tres hay que tocarlos; cambiar sólo el campo de `Config` deja el valor
efectivo en 25.

```diff
--- a/src/reii/main_workflow.py   (Config, L929)
     optimize_study_name: str = "reii_search"
-    optimize_n_startup_trials: int = 25
+    # v5: 25 startup con 40 trials dejaba a TPE con n_below=2 (Parzen
+    # degenerado). 12 siembra el modelo y deja presupuesto para aprender.
+    optimize_n_startup_trials: int = 12
     optimize_multivariate: bool = True
+    # v5: resolucion del argmax de EI por trial (default de Optuna: 24).
+    optimize_n_ei_candidates: int = 48
+    # v5: tope de trials del frente reinyectados como warm start.
+    optimize_warm_start_max: int = 10
+    # v5: congelar los parametros inertes al expandir niveles.
+    optimize_freeze_inert: bool = True
     optimize_prune_n_startup: int = 15
```

```diff
--- a/src/reii/main_workflow.py   (_make_sampler, L8770)
-        n_startup = int(getattr(self.config_base, "optimize_n_startup_trials", 25))
+        n_startup = int(getattr(self.config_base, "optimize_n_startup_trials", 12))
```

```diff
--- a/src/reii/main_workflow.py   (__main__, L10518)
-        optimize_n_startup_trials=25,
+        optimize_n_startup_trials=12,
```

Los tres campos nuevos **no** hace falta replicarlos en el constructor de
L10512-10524 (sus defaults ya son los v5); añadirlos sería ruido.

#### 3.7.4 #66 / #67 / #68 — absorber en #9 / #11 / #12

No se aplican como hunks posteriores (citan literales de la versión nueva). Se
escribe el texto final **directamente** al aplicar #9, #11 y #12:

1. **En #7** (`method_selection.py`, junto a `RULE_VERSION`, L86), tras fijar
   `RULE_VERSION = "select-v4"`, añadir:

```python
#: Piso de viabilidad sobre el ARI bootstrap. Medido: ruido 0.09-0.11,
#: corpus real 0.42-0.60, estructura 0.78-0.97 (STABILITY_NULL.md §2-§3).
#: Debe coincidir con ``quality.metrics.STABILITY_MEAN_MIN``.
STABILITY_MEAN_MIN = 0.30
```

**Corrección de nombre**: `STABILITY_NULL.md` §7(b) lo escribe en minúsculas
(`stability_mean_min`). Se usa **mayúsculas** (`STABILITY_MEAN_MIN`) por dos
razones: (i) el propio doc exige que coincida con
`quality.metrics.STABILITY_MEAN_MIN`, que va en mayúsculas; (ii) es una
constante de módulo. Actualizar las **dos** referencias del bloque de §7(b)
(`boot < float(STABILITY_MEAN_MIN)` y el `f"{float(STABILITY_MEAN_MIN):.2f}"`).

2. **En #9** (`Selection`): el campo nace ya
   `null_mode: str = "diagnostic"` con su comentario. Nunca se escribe `"gate"`.
3. **En #12**: `_NULL_MODES` incluye `"diagnostic"`; la firma de `_feasibility`
   lleva `null_mode: str = "diagnostic"`; **y** el bloque del piso insertado
   justo después del chequeo de tautología y **antes** de `if null_mode == "gate":`:

```python
    boot = float(getattr(metric, "bootstrap_ari", float("nan")))
    if math.isfinite(boot) and boot < float(STABILITY_MEAN_MIN):
        return (
            f"sin senal de estructura: estabilidad media {boot:.3f} < piso "
            f"{float(STABILITY_MEAN_MIN):.2f} (el remuestreo no recupera la"
            " particion)"
        )
```

4. **En #11**: la línea de control de `explain_lines` (la que imprime
   `bootstrap=… IC95=… nulo p=… (modo …)`) se añade en el rewrite, después del
   bloque de `clase mayor=…`.

#### 3.7.5 #23 — partido en dos: #23a (inerte, anclas reales) y #23b (fase 2)

El hunk no tenía ancla porque su texto vive en `STABILITY_NULL.md` §7(c), no en
el patchset. Anclas **reales verificadas hoy**:

| paso | fichero:línea               | qué                                                                                                                                                                                                                                                                                                                                                                                              |
| ---: | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|    1 | `main_workflow.py:998-1000` | entre `method_selection_coverage_floor` (L999) y `method_selection_k_bounds` (L1000): los 4 campos `method_selection_null_mode` / `_stability_boot` / `_stability_perm` / `_stability_min_n`. El ancla de §7(c) **es válida** (las tres líneas son contiguas en 998/999/1000). Alternativa equivalente: después de L1006 (`max_class_share`) y antes de `optimize_coverage_gate_select` (L1007). |
|    2 | `main_workflow.py:5541`     | insertar `def _stability_blocks(...)` inmediatamente antes de `def _triple_stability_verdicts(`                                                                                                                                                                                                                                                                                                  |
|    3 | `main_workflow.py:5566`     | parámetro `stability_blocks: Optional[Dict[str, Dict]] = None,` tras `partition_sizes: ... = None,`                                                                                                                                                                                                                                                                                              |
|    4 | `main_workflow.py:5593`     | `_stab = stability_blocks or {}` justo tras `n_total = len(all_uids)`                                                                                                                                                                                                                                                                                                                            |
|    5 | `main_workflow.py:5674`     | helper `_sb` después de `_max_share` y antes de `metrics = [` (L5676)                                                                                                                                                                                                                                                                                                                            |
|    6 | `main_workflow.py:5689`     | 7 kwargs nuevos (`bootstrap_ari`, `bootstrap_ci_lo`, `bootstrap_ci_hi`, `coverage_stable`, `null_p`, `null_z`) tras `max_class_share=_max_share(_m),`                                                                                                                                                                                                                                            |
|    7 | `main_workflow.py:5708`     | `null_mode=…` en `select_method(...)` tras `max_class_share_max=…`                                                                                                                                                                                                                                                                                                                               |
|    8 | `main_workflow.py:5947`     | `"stability_blocks": _stab,` en el `return {` de `_triple_stability_verdicts`                                                                                                                                                                                                                                                                                                                    |
|    9 | `main_workflow.py:6234`     | bloque `stability_blocks = self._stability_blocks(...)` antes de `report = self._triple_stability_verdicts(`                                                                                                                                                                                                                                                                                     |
|   10 | `main_workflow.py:6252`     | `stability_blocks=stability_blocks,` tras `partition_sizes=_raw_sizes,`                                                                                                                                                                                                                                                                                                                          |

**#23a = pasos 1-10.** Es **inerte por construcción**: `self._stability_spaces`
no existe ⇒ `getattr(..., None)` ⇒ `spaces=None` ⇒ `_stab = {}` ⇒ todos los
campos caen a `nan` ⇒ `stability = ari`, `coverage_effective = coverage`,
`null_mode="diagnostic"` no filtra. Comportamiento observable idéntico al
actual; sólo aparecen claves nuevas en el payload.

**#23b (NO en este patchset).** Rellenar `self._stability_spaces` en los tres
puntos donde existen a la vez la matriz y el `predict_fn` de cada método:
`L6052` (A), `L6103` (B), `L6139` (C), junto a `_hungarian_stability_direct`.
Requiere decidir **qué matriz** es «el espacio de cada método» (A no tiene una
densa) ⇒ es un cambio con consecuencia numérica sobre la selección y debe
validarse con `stability_engine_bench.py` antes de activarlo. Mientras no se
haga, #23a queda inerte y el veredicto es el de `ari` tight/loose, declarado
como `stability_kind="perturbation"`.

#### 3.7.6 #46 — texto corregido (import local intercalado)

```diff
--- a/src/reii/main_workflow.py   (L9105-9110)
         try:
             from reii.backend.config_store import ConfigStore
-
-            ConfigStore(config.db_dsn).seed_from_config(config)
-        except Exception as e:
-            print(f"   [WARNING] ConfigStore.seed_from_config falló: {e}")
+            # v5: en clásico el seed NO se aplica (el modo fija sus overrides
+            # después); en moderno se conserva el comportamiento previo.
+            if str(getattr(config, "run_mode", "modern")) != "classic":
+                ConfigStore(config.db_dsn).seed_from_config(config)
+        except Exception as e:
+            print(f"   [WARNING] ConfigStore.seed_from_config falló: {e}")
```

Se conserva el `import` local tal cual (el hunk original lo borraba por accidente
al no contemplarlo).

#### 3.7.7 Correcciones a §3.5 (desfases que quedaban mal)

| doc               | dice                                     |                  real |
| ----------------- | ---------------------------------------- | --------------------: |
| §3.5              | `_cdh_frac_from_params` acaba en `L7883` |              **7885** |
| HARDENING §8.3    | gate de K en `L8606`                     |              **8609** |
| CLASSIC_MODE §9.7 | `"afc_y_proyeccion"` en `L63`            |                **68** |
| §1 #54            | `Config` L923-931 (1 sitio)              | **3 sitios** (§3.7.3) |
| §1 #20            | `context` L5964                          |              **5965** |
| V5_SPACE §7.7     | `try/except` L8590-8594                  |         **8589-8593** |
| V5_SPACE §7.6     | `_is_sentinel` L7857                     |              **7855** |

---

## §4 STABILITY_NULL — fusionado

Estado: `docs/design_v5/STABILITY_NULL.md` (948 líneas) **en disco y verificado**.
Test: `tests/test_stability_null.py` (16 tests; hoy `16 skipped`, se activan al
aplicar el diff). El conflicto que este documento resuelve está en §3.3.

### §4.1 Mapa de fusión

| hunk | destino                                           | fuente exacta        | se inserta                                |
| ---- | ------------------------------------------------- | -------------------- | ----------------------------------------- |
| #3   | `quality/metrics.py` — `bootstrap_stability()`    | STABILITY_NULL §7(a) | fin de `metrics.py`, tras #2              |
| #4   | `quality/metrics.py` — `null_partition_test()`    | STABILITY_NULL §7(a) | junto a #3 (mismo bloque)                 |
| #66  | `Selection.null_mode` default                     | STABILITY_NULL §7(b) | tras #9 (`SELECT_V4` §7.5)                |
| #67  | piso `stability_mean_min = 0.30`                  | STABILITY_NULL §7(b) | tras #12 (`_feasibility`)                 |
| #68  | textos de `explain_lines`                         | STABILITY_NULL §7(b) | tras #11                                  |
| #23b | `main_workflow.py` — llamada + volcado al payload | STABILITY_NULL §7(c) | dentro del bloque 3 (parcial: ver nota)   |
| —    | banner §4 de `SELECT_V4.md`                       | STABILITY_NULL §7(d) | **YA APLICADO** (`SELECT_V4.md` L175-186) |

### §4.2 Tres puntos donde STABILITY_NULL manda sobre SELECT_V4

1. `null_mode` default es `"diagnostic"`, no `"gate"` (banner ya escrito en
   `SELECT_V4.md` §4; los literales de §7.5/§7.8/§7.9 se leen como
   `"diagnostic"`).
2. `_NULL_MODES` (hunk #12) debe incluir `"diagnostic"`: `SELECT_V4` §7.8 sólo
   lista tres valores y `select_method` lanzaría `ValueError` en la primera
   llamada.
3. `tests/test_select_v4.py::TestNulo` está escrito contra el default `"gate"`;
   al aplicar #66 hay que revisar los casos que dependen del default (el propio
   `select_v4_arena.py` reproduce ambos modos explícitamente, así que la tabla de
   `SELECT_V4.md` §3 no cambia).

### §4.3 Dos discrepancias de contrato a vigilar en el auditor

- **Nombres de IC**: el contrato del payload usa `bootstrap_ari_ci_low/_high`;
  `MethodMetrics` los expone como `bootstrap_ci_lo/_hi` (STABILITY_NULL §4.2).
  Mapeo 1:1, pero `9_selection_audit.py` (#28) lee las claves del **payload**:
  debe usar las largas, no las cortas.
- **`stability_verdict`**: es **propiedad derivada** de
  `(ci_low >= 0.50, mean < 0.30)` en `MethodMetrics`, y a la vez se **emite** en
  el payload para auditoría. Es el único campo nuevo con doble vida; el auditor
  debe recalcularlo y comparar (invariante de replay, AUDIT_V4 §2), no confiar en
  el valor emitido.

**Contrato de fusión** (sin cambios): aditivo sobre `select-v4`; la firma de
`_feasibility` es la de `SELECT_V4` §7.8 y el piso entra como keyword-only.

---

## §5 HARDENING — fusionado

Estado: `docs/design_v5/HARDENING.md` (614 líneas) **en disco y verificado**.
Test: `tests/test_backend_hardening.py` (402 líneas; `60 passed, 4 xfailed` hoy,
los `xfail` son la lista de trabajo que el diff convierte en `xpass`).
**El nombre del fichero de test no es el que reservaba §6(c):** es
`test_backend_hardening.py`, no `test_hardening.py`.

### §5.1 Módulo nuevo (no estaba en el inventario)

`src/reii/method_c_policy.py` entra como **#72**: copia verbatim de
`scripts/playground/hardening_bench.py:88-510` (`resolve_method_c_backend`,
`Precondition`, `ward_guard`, `effective_target_k`, `preflight`,
`format_target_k_notice`, `guarded_fit`) más los 5 hunks de HARDENING §8.1.
Sin esos 5 hunks los `xfail` quedan rojos al aplicar el módulo. Se crea **antes**
que #22c (que lo importa) y **antes** que #73 (que lo amplía con los cuerpos
densos): sin #73 el dispatcher conseguiría `NameError`, porque `guarded_fit`
llama a `dense_cdh`/`ward_backend`/`kmeans_spherical`/`hdbscan_backend` como
globales del módulo.

### §5.2 Los tres PENDING, resueltos

- **`ward_max_n`** (#69) — valor canónico **6000** (HARDENING §2.2): presupuesto
  320 MiB, peor camino cuadrático 275 MiB a n=6000, n_max=6476. Consumidor: #72
  (`preflight`), no `Config` directamente. **Inconsistencia a corregir**:
  `scripts/playground/hardening_bench.py:117` fija `DEFAULT_WARD_MAX_N = 4000` y
  su autocalibración imprime 7500 con presupuesto de 256 MiB. Son tres números
  para la misma política. Decisión: manda el doc (6000) y el banco se alinea con
  **una línea**; si no, el banco valida un umbral distinto del que se aplica.
- **`Kach vs target_k`** (#70) — **sin decisión de usuario pendiente**. Gana el
  contrato de `CLASSIC_MODE.md` §7/§9.3 (`k_target`/`k_ach`/`k_shortfall` como
  `user_attr`, `-k_ach` como tercer objetivo); HARDENING §8.3 **no** propone
  nombres nuevos, sólo cambia el **orden**: el gate duro `2..20` se aplica al K
  real ya refinado y no a `n_clusters`, para que un `winner_k=25` no pase el gate
  reportando `-25`. Orden: aplicar #39 y #40, después §8.3.
- **`except` mudo** (#71) — son **dos** `except` distintos y aplican ambos, sin
  colisión: #60 (`V5_SPACE` §7.7) **borra** el de `_evaluate_trial`
  (L8589-8593); HARDENING §8.2 **endurece** el de `_evaluacion_rapida`, que es el
  que se tragó el FAIL #81 y que #60 no toca.

### §5.3 Núcleo de precondiciones (lo que el test ya fija)

`target_k < 2` ⇒ `None` en **todos** los backends (`k_below_kmin`);
`n_clusters < 2` ⇒ `None`; NaN/Inf y filas de ceros autodetectadas en
`guarded_fit` (hoy `nan < 0.08` es `False` y colaba); `min_r2_dense` default
`0.08` (hoy `0.0`). La guarda de memoria **degrada, no aborta**: devuelve
`backend_effective="dense_cdh"`, `reason="ward_memory_guard"` y el `[!]`.
Se descarta hacerla dinámica por RAM: rompería el determinismo entre máquinas y
el umbral debe ser un dato del paper.

---

## §6 Tests

### (a) Los 3 ficheros nuevos ya en disco

| fichero                      | tests colectados | qué cubre                                                                                                                                                                                                                             | estado actual                                             |
| ---------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `tests/test_select_v4.py`    | **42**           | contrato `select-v4`: C denso no gana por construcción, C con `hdbscan` nunca gana, `0.837` con `nu=0` se descarta, abstención, `criteria_conflict` por kinds, snapshot con `rule_version` vieja ignorada, métricas mínimas no lanzan | 42 `skipped` (guardado por `RULE_VERSION == "select-v4"`) |
| `tests/test_mode_classic.py` | **14**           | perfil `CLASSIC_OVERRIDES`, tipos, `cdh_enforce_min_leaf=False`, `run_mode` default, `--mode classic` aplica y `select`/`wc_only` no, `classic_profile_check.py` pasa                                                                 | 1 passed / 13 skipped                                     |
| `tests/test_v5_space.py`     | **8**            | espacio condicional (5000 draws sin violar `tight<loose` / `frac>frac_loose`), inertes congelados, guard de excepciones, `KeyboardInterrupt` no capturado, `outcome` gate vs exception                                                | **2 failed** / 1 passed / 5 skipped                       |

Nota: el plan decía 24 tests en `test_select_v4.py`; en disco hay **42**
(colectados). El resto de cifras coincide.

Los 2 rojos de `test_v5_space.py`
(`test_conditional_space_never_violates_structural_gates` y
`test_inert_params_are_frozen_by_default`) son **rojos por diseño hasta aplicar
el bloque 7**: asertan `isinstance(space["coref_context_units_tight"], dict)` y
`uc_window_size` escalar, y el `get_search_space` actual devuelve listas planas.
El guard `_need()` no los salta porque los símbolos que consultan
(`Optimizador`, `Config`) ya existen. No son una regresión del árbol: son la
especificación ejecutable del bloque 7.

### (b) Los 2 tests que el usuario DEBE reemplazar

Codifican el comportamiento viejo (`max_class_share_max = 0.90` sobre lo
asignado) y **romperán** al aplicar el bloque 2. Reemplazo propuesto, compatible
con `SELECT_V4.md` §7.4/§7.8/§7.9 (`dominance_corpus_effective` y
`winner_dominance_corpus`) — coincide con `GATE_NOISE.md` §13 pero usando la
implementación de `SELECT_V4.md` (§3.1):

```python
    def test_dominancia_089_con_ruido_no_descarta(self):
        """0.89 de las asignadas con 30% de ruido = 0.62 del corpus: viable.

        Reemplaza a ``test_dominancia_089_no_descarta``: el umbral ya no es
        0.90 sobre lo asignado sino 0.70 sobre el corpus, asi que 0.89 sin
        ruido SI se descarta; con ruido, no.
        """
        c = MethodMetrics(
            "C", 3, 0.785, 0.371, noise_frac=0.30, max_class_share=0.89
        )
        a = _m("A", 4, 0.250, 0.574)
        sel = select_method([a, c])
        assert sel.winner == "C"
        assert not any("dominante" in d["why"] for d in sel.discarded)
        # c = 0.89 * (1 - 0.30) = 0.623 del corpus
        assert sel.winner_dominance_corpus == pytest.approx(0.623)

    def test_dominancia_gate_aparece_en_explain(self):
        """El umbral se imprime en unidades del corpus (0.70)."""
        c = MethodMetrics("C", 3, 0.785, 0.371, max_class_share=0.60)
        sel = select_method([c])
        text = sel.explain_text()
        assert "clase mayor=0.60" in text
        assert "= 0.60 del corpus" in text
        assert "gate: >0.70 del corpus" in text
```

Se recomienda además añadir el test de regresión que propone `GATE_NOISE.md` §13
(`test_dominancia_ignora_el_ruido_en_el_denominador`): los dos casos tienen
`c = 0.837` y deben compartir veredicto; es el que ancla el bug que motiva todo
`GATE_NOISE.md`. No sustituye a los dos anteriores, los acompaña.

`tests/test_method_selection.py::test_dominancia_092_descarta` **sigue pasando**
(0.92 > 0.70 y el motivo sigue conteniendo `"0.92"`): no lo toques.
`tests/test_select_mode_objective.py::TestDominanceGateWiring` también sigue
pasando; su docstring queda conceptualmente viejo, pero el `assert` es correcto.

### (c) Tests nuevos de los agentes 1 y 2 — en disco

| fichero                           | tests             | estado hoy             | se activa con                     |
| --------------------------------- | ----------------- | ---------------------- | --------------------------------- |
| `tests/test_stability_null.py`    | 16                | `16 skipped`           | #3, #4, #66, #67                  |
| `tests/test_backend_hardening.py` | 64 (60 + 4 xfail) | `60 passed, 4 xfailed` | #72 (los `xfail` pasan a `xpass`) |
| `tests/test_method_c_backends.py` | 16                | `16 skipped`           | #21, #22c-#22e, #72b              |

Verificado en esta máquina: `python -m pytest tests/test_stability_null.py
tests/test_backend_hardening.py tests/test_method_c_backends.py -q` ⇒
`60 passed, 32 skipped, 4 xfailed` (los tres ficheros colectan con el código
actual y ninguno rompe la suite).

Cobertura de `test_stability_null.py`: determinismo del bootstrap, IC contiene la
media, `coverage_stable < 1` en ruido, `mean < 0.30` en ruido puro,
`ci_low >= 0.50` en estructura, `null_p` no gatea por defecto,
`null_mode="gate"` con `null_p` inyectado sí descarta, `nan` nunca descarta,
precondiciones sin excepción.

Cobertura de `test_backend_hardening.py`: backend desconocido degrada sin abortar,
`ward` sobre `ward_max_n` cae a `dense_cdh`, `target_k` no se reporta como
`n_clusters`, precondiciones (`n<min`, `K>n`, `K=1`, 1-D, ceros, NaN) devuelven
`None`/`nan` y nunca `IndexError`, y el `except` endurecido deja rastro sin
enmascarar `KeyboardInterrupt`/`SystemExit`.

Los 4 `xfail` **no son deuda oculta**: son exactamente los 4 defectos que
`hardening_bench.py` exhibe (NaN ⇒ K=4, ceros ⇒ K=4, `kmeans` con K=1,
puntos idénticos ⇒ K=4) y que HARDENING §8.1 corrige. Se documentan como
`xfail` para que el banco no mienta mientras el módulo no exista.

---

## §7 Comandos (PowerShell, listos para copiar)

Ejecutar desde la raíz del repo. Todos los de (a) son **read-only**.

### (a) Verificación previa

```powershell
# 1. la corrida vieja sigue UP con el codigo viejo (no debe cambiar durante el patchset)
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"

# 2. estado del arbol (sin tocar nada)
git --no-pager status --short
git --no-pager log -1 --oneline

# 3. lineas reales de los ficheros que se van a editar
(Get-Content src\reii\main_workflow.py).Count
(Get-Content src\reii\method_selection.py).Count
(Get-Content src\reii\quality\metrics.py).Count

# 4. analisis estatico: que el perfil classic y el AST de Config cuadran
python scripts\playground\classic_profile_check.py

# 5. baseline de tests (debe dar: 42 skipped / 1 passed 13 skipped / 2 failed 1 passed 5 skipped)
python -m pytest tests\test_select_v4.py tests\test_mode_classic.py tests\test_v5_space.py -q

# 6. playgrounds de referencia (cada uno imprime su propio [ok])
python scripts\playground\v5_space_sim.py
python scripts\playground\select_v4_arena.py
python scripts\playground\c_backends_arena.py
python scripts\playground\gate_calibration.py
python scripts\playground\gate_calibration.py --reps 200
python scripts\playground\gate_calibration.py --metrics-selftest
python scripts\playground\gate_interval_probe.py
python scripts\playground\audit_smoke.py
python scripts\playground\stability_engine_bench.py
python scripts\playground\classic_mode_probe.py
python scripts\playground\select_v4_sim.py

# 7. confirmar que los 2 ficheros de agentes NO existen aun (deben faltar)
Test-Path docs\design_v5\STABILITY_NULL.md
Test-Path docs\design_v5\HARDENING.md
```

### (b) Aplicación

Los diffs se pegan a mano, bloque a bloque (§2). Lo único que este patchset
**obliga** a hacer por comando es invalidar el snapshot de selección y archivar
los artefactos v4:

```powershell
# OBLIGATORIO antes de cualquier --mode select posterior al bloque 2:
# load_selection ignora rule_version (bug #14). Sin esto, un snapshot select-v3
# se reutiliza en silencio y "parece valido".

# 1) archivar el snapshot actual como historico (si existe). Si el destino ya
#    existe, -Force lo sobrescribe.
if (Test-Path data\method_selection.json) {
    Move-Item data\method_selection.json data\method_selection_v3.json -Force
}

# 2) garantizar que NO queda snapshot que load_selection pueda reutilizar
Remove-Item data\method_selection.json -ErrorAction SilentlyContinue
if (Test-Path data\method_selection.json) {
    Write-Host "[X] el snapshot sigue presente: --mode select lo reutilizaria"
}

# best_params.json es de la corrida v4 (mtime 28-Oct 00:12). Sobrevive, pero
# sus claves son del espacio viejo. Guarda copia antes de re-optimizar:
Copy-Item data\best_params.json data\best_params_v4_backup.json -Force
```

`data\method_selection_log.jsonl` **no** se toca: es append-only y cada línea
lleva su `rule_version`.

### (c) Verificación posterior

```powershell
# contrato de la regla (debe pasar entero, ya sin skips)
python -m pytest tests\test_select_v4.py -q

# 2 fallos esperados ANTES de reemplazar los tests de §6(b); 0 despues
python -m pytest tests\test_method_selection.py -q

# allowlists + perfil classic
python -m pytest tests\test_mode_classic.py tests\test_pipelines_data.py -q

# espacio v5 y guard (tras el bloque 7)
python -m pytest tests\test_v5_space.py -q

# metricas del paper: no deben romper nada existente
python -m pytest tests\test_quality_core.py tests\test_quality_validity.py -q

# auditor nuevo contra el historial real (exit 0 ok, 1 inestable, 2 sin historial)
python src\9_selection_audit.py --strict --json data\diag\selection_audit.json
python src\9_selection_audit.py --strict --json -
```

### (d) Corrida del optimizador v5

**Nunca** reutilices `reii_probe_v4`: el estudio viejo arrastra el espacio
plano, los 13 centinelas de gate y el `n_startup_trials=25`. Nombre nuevo
obligatorio (`reii_probe_v5`); además el código le añade el fingerprint del
corpus como sufijo.

```powershell
# presupuesto recomendado por V5_SPACE.md §5.3: 30 etapa 1 + 90 etapa 2
docker compose --profile batch run --rm workflow python -u src/reii/main_workflow.py `
    --mode select --trials 120 --stage1-fraction 0.3 `
    --study-name reii_probe_v5 --force
```

**Advertencia explícita**: `--mode select` **sin** `data\best_params.json`
ejecuta el optimizador con `optimize_trials = 200` y dos etapas —
`max(20, 0.3*200) = 60` en etapa 1 más 140 en etapa 2. Con el pipeline real
(Stanza, >4 min de arranque, ~1 min/trial en el corpus del probe) eso son
**horas**. Si sólo quieres comprobar que el espacio muestrea, usa `--trials 12`
sobre un corpus de juguete, no sobre el corpus del paper. Y si quieres reanudar
desde los params v4, acuérdate de que sus claves (`min_cluster_size_cdh`) ya no
son del espacio v5: `_cdh_frac_from_params` las respeta por compatibilidad, pero
congelan el piso; no las dejes en el `best_params` de la corrida nueva.

---

## §8 Checklist de revisión (hunks de riesgo alto)

Una pregunta por hunk alto. Son verificables leyendo el código, no «revisando».

1. **#8 `MethodMetrics`** — ¿`dominance_corpus` es un **campo** (`float = nan`) y
   `dominance_corpus_effective` una **propiedad**, o quedó el nombre duplicado?
   Debe haber un campo y una propiedad con nombres distintos. Si hay dos
   definiciones de `dominance_corpus`, el `asdict` de `as_row` revienta.
2. **#8 `MethodMetrics`** — ¿`usable_coverage` sigue dividiendo por `n_total` y no
   por `n_assigned`? El docstring dice `/ n_total`; el denominador no cambia en
   v4.
3. **#12 `_feasibility`** — ¿el denominador de dominancia es `n_assigned` o
   `n_corpus`? Debe ser **corpus**, vía `dominance_corpus_effective`. Comprueba
   que la rama 2 usa `max_class_size / n_total` y **no** `max_class_share`.
4. **#12 `_feasibility`** — ¿los 5 primeros parámetros conservan **nombre y
   posición** (`metric, coverage_floor, k_bounds, noise_max, max_class_share_max`)
   y todo lo nuevo es keyword-only? `9_selection_audit.py` llama posicionalmente.
5. **#12 `_feasibility`** — ¿`reject_tautological` compara contra
   `{"ward","kmeans_spherical"}` y exige `stability_kind == "perturbation"`? Si
   falta la segunda condición, descartaría particiones con bootstrap válido.
6. **#12 `_feasibility`** — ¿`null_p = nan` **no** descarta? El `if` debe ser
   `math.isfinite(p) and p > alpha`. Si olvidas `isfinite`, `nan > alpha` es
   `False` por IEEE y funciona por accidente, pero rompe con `None`.
7. **#13 `select_method`** — ¿`winner_dominance_corpus` se asigna desde
   `dominance_corpus_effective` (no desde el campo `dominance_corpus`)? Si es el
   campo, el payload escribe `null` y la invariante 2 del auditor no verifica.
8. **#13 `select_method`** — ¿`delta_ari` sigue siendo `win - runner` sobre
   **estabilidad** (no sobre ARI)? El nombre se conserva por compatibilidad; el
   valor cambia de significado. Comprueba que `ambiguous` usa ese delta.
9. **#14 `load_selection`** — ¿el rechazo por `rule_version` ocurre **antes** de
   `Selection.from_dict`, y `expected_rule_version=""` lo desactiva? Con `""` se
   desactiva; si se invierte, `del data\method_selection.json` del §7(b) deja de
   ser suficiente y todo `--mode select` se abstiene.
10. **#14** — ¿`persist_selection` escribe `rule_version` en el **nivel
    superior** del payload? `load_selection` lee
    `payload.get("rule_version", sel.get("rule_version"))`; ya lo hace (L461).
    Verifícalo y no lo «arregles».
11. **#15/#16** — ¿los **dos** sitios valen 0.70? El `getattr(config, ...,
0.70)` de L5707 y el default del dataclass L1006. Si sólo cambias uno, el
    umbral efectivo depende de si `Config` se construyó por defecto.
12. **#19 `_raw_stats`** — ¿`n_assigned = sum(sizes)` con `sizes > 0` y
    `n_noise = n_total - assigned`? El denominador del gate es `n_assigned`, el
    del corpus es `n_total`; no los mezcles en la misma función.
13. **#20 payload** — ¿`spec_hash` hashea `dataclasses.asdict(self.config)` con
    `default=str` y `sort_keys=True`? Sin `default=str`, un campo no-JSON
    (tuple, `Path`) lanza `TypeError` **dentro** de la ruta de persistencia.
14. **#20** — ¿`method_c_backend` sale de `config` y no de un literal? Hoy
    `Config` **no tiene** ese campo (§3.2); hasta el diff de `BACKEND_C.md` el
    `getattr(..., "hdbscan")` devuelve el default y el payload miente con
    elegancia. Anótalo o no apliques #20 todavía.
15. **#22 (si llega el diff)** — ¿`partition_sizes` **excluye** `-1` y está
    **ordenado**, y `n_assigned == len(uce_ids)` en los tres backends densos?
    Si `n_assigned != len(uce_ids)`, hay un backend devolviendo `-1` y el gate de
    dominancia medirá sobre un denominador recortado.
16. **#28 auditor** — ¿las invariantes 2, 3 y 4 leen `dominance_corpus_eff` (no
    `dominance_corpus`) y `n_assigned` (no `usable`)? Ver §3.1.
17. **#56+#57+#58** — ¿`get_search_space` devuelve `coref_context_units_tight` y
    `hdbscan_uce_min_cluster_frac` como **`dict` con `kind`**, y `_suggest` tiene
    la **segunda pasada** antes de tocar `_precompute_coref_chains`? Si aplicas
    #57 sin #58, el primer trial muere en `int(c)` de **L7999**, fuera del `try`.
18. **#57** — ¿`hdbscan_uce_min_cluster_frac_loose` quedó como **lista**
    (`[0.005, 0.01, 0.015, 0.02, 0.03]`) y sólo `..._frac` es condicional? El
    `diff` lo muestra así porque `frac_loose` no se mueve (`V5_SPACE.md` §3.5).
19. **#57** — ¿el congelado (`optimize_freeze_inert`) se aplica **después** de
    construir el dict y antes del `return`? Debe sustituir `min_forms_uc_1` por
    un `int` y `uc_window_size` por `uc_window_size` de `uc_config`.
20. **#59 vs #31** — ¿`_MAX_TB_CHARS` está definido **una sola vez** en el
    módulo? Dos definiciones = gana la última y el `traceback` se recorta a un
    valor que no esperabas.
21. **#60** — ¿se borró **todo** el `try/except` de `_evaluate_trial`
    (L8589-8593) y la llamada quedó `result = self._evaluacion_rapida(cfg, uc_cfg=uc_cfg)`
    sin indentación sobrante? Un `try:` huérfano es `IndentationError`.
22. **#61** — ¿los **12** retornos usan `_sentinel(trial, ...)` y no queda
    ninguno con `_SENTINEL_TRIAL` crudo? El `except` borrado en #60 también era
    un `_SENTINEL_TRIAL`, pero ese ya no existe.
23. **#64** — ¿`study.optimize(_make_guarded_objective(self._evaluate_trial), ...)`
    es la **única** envoltura? No debe haber además un `lambda` con
    `guarded_trial_eval` (§3.4).
24. **#39/#40** — ¿el tercer objetivo usa `-float(_k_ach)` y **no**
    `-float(n_clusters)`? Y ¿`k_target` queda `-1` cuando no hay
    `cdh_target_k`, en vez de lanzar `TypeError`?
25. **#45 `_apply_best_params`** — ¿`resolve_best_params` se aplica **antes** de
    leer las claves, y el `pseudocount` condicional no se salta el invariante en
    moderno? En clásico el `pseudocount` se conserva (0.0) pero se **reporta**.
26. **#27 `steps.py`** — ¿`_config_subset` sólo añade `_extras` cuando el valor
    **difiere** del default? Si añade siempre, invalida todo el `step_cache` de
    clasificación (el bug que el diseño evita).

---

## §9 Lo que NO entra en este patchset y por qué

- **Encoder contextual (transformer) en lugar de fastText + retrofitting.**
  Todo el código vivo usa `fastText` + `_compute_retrofitted_vectors`; el
  `space mismatch` que `BACKEND_C.md` §11 documenta (`real_ctx` con PCA 384→30)
  es consecuencia de ese diseño. Cambiar el encoder invalidaría la comparación
  con la baseline A, no tiene diff en ningún documento y no es un «hunk»:
  es otro proyecto. Fuera.
- **Allowlist `hdbscan_uce_min_cluster_frac` en el `step_cache`.** Sí entra
  (#27) para `run_mode` y las `frac`, pero **no** se añade el
  `version` bump del step: cambiar `version` invalidaría **todo** el
  `step_cache` de clasificación de golpe (`CLASSIC_MODE.md` §8.2). El diseño de
  «extras condicionales» es precisamente para evitarlo. Si alguien propone
  bumpear `version`, es un cambio de comportamiento que este patchset no
  autoriza.
- **`src/reii/main_workflow_clasico.py` (4.519 líneas, sin el fix de piso del
  CDH, vivo vía `src/reii/pipelines/transcripciones.py:155`).** **Es la baseline
  del Método A** y se etiqueta como tal: no se toca, no se convierte en shim
  (§3.6). Sus resultados A son los que dan sentido a la comparación A/B/C del
  paper; sin él no hay escenario A reproducible. `CLASSIC_MODE.md` §9.9/§9.10
  proponen lo contrario y quedan explícitamente fuera (#52, #53).
- **Los 3 backends densos y `Config.method_c_backend`** (#21-#23) — **YA ENTRAN**.
  Estuvieron fuera mientras `BACKEND_C.md` no tenía texto aplicable; su §12
  (añadido después) trae #21, #22a-#22e y #72b. Ver §3.2.
- **`bootstrap_stability` y `null_partition_test`** (#3, #4) — **YA ENTRAN**, con
  el diff literal de `STABILITY_NULL.md` §7(a). Ver §4.

Lo que **sí** queda fuera, y por decisión explícita: el encoder contextual
(3.2 del paper: el código usa fastText estático + retrofitting), el bug de
allowlist de `hdbscan_uce_min_cluster_frac` (§5.3/`HARDENING` §8.4, cambiaría el
comportamiento de caché) y el shim de `main_workflow_clasico.py` (#52, #53).

---

## §10 Plan de rollback

### Archivos a respaldar antes de empezar

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$dst = "backup_patchset_$stamp"
New-Item -ItemType Directory -Path $dst | Out-Null
Copy-Item src\reii\main_workflow.py            $dst
Copy-Item src\reii\method_selection.py         $dst
Copy-Item src\reii\config.py                   $dst
Copy-Item src\reii\quality\metrics.py          $dst
Copy-Item src\reii\backend\config_store.py     $dst
Copy-Item src\reii\backend\steps.py            $dst
Copy-Item src\9_selection_audit.py             $dst
Copy-Item src\reii\pipelines\transcripciones.py $dst
Copy-Item data\best_params.json                $dst
Copy-Item data\method_selection.json           $dst -ErrorAction SilentlyContinue
Copy-Item tests\test_method_selection.py       $dst
```

`main_workflow_clasico.py` **no** se respalda porque no se modifica (§9).

### Volver a `select-v3`

1. Restaura `src/reii/method_selection.py` del backup. Es el único fichero que
   define `RULE_VERSION`; con él vuelve `select-v3`.
2. Restaura `src/reii/main_workflow.py` si aplicaste #15-#20 (el umbral 0.70 y
   `_raw_stats` no tienen sentido con la regla vieja).
3. **Borra** `data\method_selection.json`: es un snapshot `select-v4` y
   `load_selection` de v3 **no** comprueba `rule_version` (ese es el bug #14), así
   que lo reutilizaría. Sin este borrado convives con una decisión v4 aplicada
   por código v3.
4. El historial `data\method_selection_log.jsonl` **se queda**: cada línea lleva
   su `rule_version` y el auditor filtra por ella. Borrarlo destruye la
   trazabilidad del paper.
5. `src/9_selection_audit.py`: si restauraste el auditor nuevo (#28), vuelve al
   viejo, porque las invariantes 2/3/4 leen claves que el producto v3 no escribe
   y reportaría `[!]` en todas.

### `data/best_params.json`

Sobrevive al patchset pero **no** es válido para v5:

- Contiene `min_cluster_size_cdh` (fracción 0.05-0.2) del espacio v3.
  `_cdh_frac_from_params` la respeta **con prioridad** sobre `cdh_target_k`
  (`main_workflow.py` L7871-7876), así que si la dejas, **congela el piso** y el
  optimizador v5 no explora K. Para la corrida v5, o la quitas del archivo, o
  arrancas con `--optimize-only`.
- Con `resolve_best_params` (#30/#43/#45) sus claves desconocidas ya no abortan:
  se filtran con `[!]` y se registran en log. Antes de ese hunk, un
  `Config(**params)` habría reventado.
- Rollback: `Copy-Item $dst\best_params.json data\best_params.json -Force`.

### Base de datos de Optuna

- El estudio **`reii_probe_v4`** queda intacto y **no debe reutilizarse**: su
  sampler aprendió sobre el espacio plano (29.4 bits nominales pero 53 %
  factible) y sus 13 centinelas de gate sesgan el `TPE` posterior. Reanudarlo
  mezcla dos espacios en un mismo estudio.
- La corrida v5 escribe en `reii_probe_v5_<fingerprint>` (el código añade el
  sufijo del corpus, `main_workflow.py` L8817). Rollback = **abandonar** el
  nombre, no borrar filas: `--study-name reii_probe_v5b` en el siguiente intento.
  Si hace falta borrar de verdad, hazlo con la API de Optuna
  (`optuna.delete_study(study_name=..., storage=...)`) y **nunca** con `DELETE`
  sobre las tablas `trials`/`trial_values` a mano: rompes las FKs y el
  `study_directions`.
- El SQLite/PostgreSQL de Optuna **no** lo toca este patchset. `optimize_storage`
  sigue siendo el mismo salvo en `--mode classic`, que lo redirige a
  `data/classic/reii_optuna.db` (#29).

### Tabla de reversibilidad

| bloque       | reversión                                  | efecto residual                                                    |
| ------------ | ------------------------------------------ | ------------------------------------------------------------------ |
| 1 métricas   | borrar las 7 funciones                     | ninguno (nadie las consume aún)                                    |
| 2 regla      | restaurar `method_selection.py`            | `data\method_selection.json` es v4 ⇒ **borrar**                    |
| 3 productor  | restaurar `main_workflow.py`               | payload con claves nuevas ⇒ el auditor viejo las ignora            |
| 4 allowlists | restaurar los 2 ficheros                   | los hashes nuevos difieren ⇒ primer run **recalcula** `step_cache` |
| 5 auditor    | restaurar `9_selection_audit.py`           | ninguno                                                            |
| 6 clásico    | restaurar `main_workflow.py` y `config.py` | `data/classic/**` queda huérfano (borrable)                        |
| 7 v5         | restaurar `main_workflow.py`               | `best_params.json` puede traer congelados ⇒ limpiar                |

Última comprobación tras cualquier rollback:
`python -m pytest tests\test_method_selection.py -q` (debe volver a 0 fallos con
los tests **viejos**) y `python src\9_selection_audit.py --strict` (exit 0 o 1,
nunca 2 con historial).
