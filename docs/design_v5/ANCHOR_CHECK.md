# Verificación de anclas del `FINAL_PATCHSET.md` (§2, tramos 1-7)

Fecha: 2026-10-28. **Sólo lectura**: no se ha modificado ningún fichero de
`src/**`, no se ha ejecutado `git`, no se ha importado `reii` (Stanza >4 min).

Método: para cada fila de §1 se ha cruzado el símbolo/línea citados y el texto
literal de los bloques `OLD:` / `-` de los documentos fuente
(`SELECT_V4` §7.1-7.10 usa bloques `OLD/NEW`; `V5_SPACE` §7, `CLASSIC_MODE` §9 y
`BACKEND_C` §12 usan `@@` + líneas `-`; `AUDIT_V4` §8.1 usa `@@`) contra el
árbol real, con `grep -n` y lecturas puntuales.

Árbol medido: `main_workflow.py` = **10 728** líneas, `method_selection.py` = 663,
`config.py` = 255, `quality/metrics.py` = 650, `9_selection_audit.py` = **122**
(coincide con `@@ -1,122 +1,470 @@`). `src/reii/method_c_policy.py` **no existe**
(correcto: lo crean #72/#73).

---

## Resumen

| estado                                                                    |                      hunks |
| ------------------------------------------------------------------------- | -------------------------: |
| Total inventariados (§1)                                                  |                     **73** |
| OK (ancla literal presente hoy)                                           |                     **57** |
| OK con matiz (ancla presente pero el bloque no es literal)                |                **1** (#46) |
| PENDIENTE-DE-TRAMO-PREVIO (ancla aún no existe; la crea un hunk anterior) |      **3** (#66, #67, #68) |
| SIN ANCLA (el patchset no da texto ni línea de inserción)                 |                **1** (#23) |
| CONFLICTO DE ANCLA (dos hunks sobre el mismo texto, excluyentes)          |                **1** (#71) |
| Fuera de los ficheros diana del encargo                                   |  **6** (#24-#27, #52, #53) |
| No-op / descartados por §3.4                                              | **4** (#18, #38, #41, #42) |

**Anclas obsoletas en sentido estricto** (texto que el patchset pide y que hoy
no está en el fichero): **0**. Ningún hunk apunta a código que ya no exista.
Las 6 incidencias son de otra naturaleza: 3 dependen de un hunk previo
(#66-#68), 1 no trae texto de ancla (#23), 1 no es literal por un import
intercalado (#46) y 1 colisiona con otro hunk (#71 vs #60). Detalle en
«ANCLAS PROBLEMÁTICAS» más abajo.

> Nota de nomenclatura: los números de línea de la columna «línea actual» son
> los del fichero real HOY. Cuando el documento cita otra línea, se indica en la
> sección «Desfases».

---

## Tabla de anclas

| Hunk | Fichero                        | Ancla pedida (§1 / doc fuente)                           |         Línea real actual | Estado                       |
| ---: | ------------------------------ | -------------------------------------------------------- | ------------------------: | ---------------------------- |
|    1 | `quality/metrics.py`           | `from typing import ...` sin `Any` (L6)                  |                         6 | OK                           |
|    2 | `quality/metrics.py`           | append tras `external_coherence` (L612, EOF 650)         |             612 / EOF 650 | OK                           |
|    3 | `quality/metrics.py`           | append al final (EOF)                                    |                       650 | OK (ver nota A)              |
|    4 | `quality/metrics.py`           | append al final (EOF)                                    |                       650 | OK (ver nota A)              |
|    5 | `method_selection.py`          | docstring `Regla (... select-v3)` L17-40                 |                     17-40 | OK                           |
|    6 | `method_selection.py`          | `import json` / `import math` L78-79                     |                     78-79 | OK                           |
|    7 | `method_selection.py`          | `RULE_VERSION = "select-v3"` L86                         |                        86 | OK                           |
|    8 | `method_selection.py`          | `MethodMetrics` campos+`as_row` L121-137                 |                   121-137 | OK                           |
|    9 | `method_selection.py`          | `Selection` cola de campos L165-168                      |                   166-168 | OK                           |
|   10 | `method_selection.py`          | `from_dict` tupla de `NaN` L191-200                      |                   191-200 | OK                           |
|   11 | `method_selection.py`          | `explain_lines` L207-265                                 |                   207-266 | OK                           |
|   12 | `method_selection.py`          | banner `# Regla` + `_feasibility` L271-309               |                   271-310 | OK (ver nota B)              |
|   13 | `method_selection.py`          | `select_method` L312-423                                 |                   312-427 | OK                           |
|   14 | `method_selection.py`          | `load_selection` L476-500                                |                   476-504 | OK                           |
|   15 | `main_workflow.py`             | `method_selection_max_class_share: ... 0.90` L1003-1006  |                      1006 | OK                           |
|   16 | `main_workflow.py`             | `getattr(..., 0.90)` L5706-5707                          |                      5707 | OK                           |
|   17 | `main_workflow.py`             | `set_user_attr` tras L8672                               |                 8671-8674 | OK                           |
|   18 | `main_workflow.py`             | campos ya en #8                                          |                         — | NO-OP (referencia)           |
|   19 | `main_workflow.py`             | `_max_share` (L5665) + `MethodMetrics(` (doc 5687)       |          5665 / 5677-5692 | OK (ver nota C)              |
|   20 | `main_workflow.py`             | `context={` payload L5964-5966                           |                 5965-5972 | OK                           |
|   21 | `main_workflow.py`             | `hdbscan_uce_metric` L1038                               |                      1038 | OK                           |
|  22a | `main_workflow.py`             | imports L80-81                                           |                     80-81 | OK                           |
|  22b | `main_workflow.py`             | `_corte_optimo` L3831                                    |                      3831 | OK                           |
|  22c | `main_workflow.py`             | `_clasificar_umbral_embedding` L5089                     |                      5089 | OK                           |
|  22d | `main_workflow.py`             | bloque HDBSCAN L5165-5214                                |                 5165-5214 | OK                           |
|  22e | `main_workflow.py`             | llamadas L6124 / L6131                                   |               6124 / 6131 | OK                           |
|   23 | `main_workflow.py`             | «`run()`: poblar bootstrap_ari/coverage_stable/null_p`   |                         — | **SIN ANCLA**                |
|   24 | `backend/config_store.py`      | `"clasificacion": [` L36                                 |                        36 | OK (fuera de alcance)        |
|   25 | `backend/config_store.py`      | `"clasificacion"` L36                                    |                        36 | OK (fuera de alcance)        |
|   26 | `backend/config_store.py`      | `"afc_y_proyeccion"` (doc L63)                           |                        68 | OK (fuera; ver desfases)     |
|   27 | `backend/steps.py`             | `_STEP_CONFIG_KEYS` L166 / `_config_subset` L186         |                 166 / 186 | OK (fuera de alcance)        |
|   28 | `9_selection_audit.py`         | `@@ -1,122 +1,470 @@`                                    |        1-122 (122 líneas) | OK                           |
|   29 | `main_workflow.py`             | `CLASSIC_OVERRIDES`, `run_spec_id`, `apply_classic_mode` |    inserción tras imports | OK-INSERT                    |
|   30 | `main_workflow.py`             | `_NON_CONFIG_PARAM_KEYS`, `resolve_best_params`          |    módulo (tras `Config`) | OK-INSERT                    |
|   31 | `main_workflow.py`             | `_MAX_TB_CHARS = 4096` tras `_SENTINEL_TRIAL`            |                 7852-7857 | OK (colisión con #59)        |
|   32 | `main_workflow.py`             | `Config.run_mode`                                        | dentro de `Config` (L789) | OK-INSERT                    |
|   33 | `main_workflow.py`             | `_CLI_MODES_VALIDOS` L347                                |                       347 | OK                           |
|   34 | `main_workflow.py`             | `__main__`: `if _cli_mode is not None:`                  |                     10545 | OK                           |
|   35 | `main_workflow.py`             | `cfg = UCBuilderConfig()`                                |                     10598 | OK                           |
|   36 | `main_workflow.py`             | `grammatical_dashboard_path=` / `export_network_gexf`    |             10699 / 10719 | OK                           |
|   37 | `main_workflow.py`             | `cfg.pseudocount = 0.0  # BLOQUE 1`                      |                      8530 | OK                           |
|   38 | `main_workflow.py`             | `except` enriquecido `exception_*`                       |                 8590-8593 | DESCARTADO (§3.4)            |
|   39 | `main_workflow.py`             | `-float(n_clusters)` L8683                               |                      8683 | OK (ver nota D)              |
|   40 | `main_workflow.py`             | `-float(n_clusters)` L8702                               |                      8702 | OK (ver nota D)              |
|   41 | `main_workflow.py`             | `study.optimize(lambda ...)` L8975                       |                      8975 | DESCARTADO (§3.4)            |
|   42 | `main_workflow.py`             | resumen `_trial_failures`                                |                         — | DESCARTADO (§3.4)            |
|   43 | `main_workflow.py`             | `load_best_params` L8336                                 |                      8336 | OK                           |
|   44 | `main_workflow.py`             | `best_data = json.load(f)`                               |                      9701 | OK                           |
|   45 | `main_workflow.py`             | `_apply_best_params` L9633                               |               9633 / 9639 | OK                           |
|   46 | `main_workflow.py`             | `try:` + `seed_from_config` + `except`                   |                 9105-9110 | OK-MATIZ (ver #46)           |
|   47 | `main_workflow.py`             | `retro_vecs_p1: Optional[...] = (` L9925                 |                 9925-9927 | OK                           |
|   48 | `main_workflow.py`             | `persist_method_selection` L5951                         |                      5951 | OK                           |
|   49 | `main_workflow.py`             | bloque de selección L9959                                |                 9959-9965 | OK                           |
|   50 | `config.py`                    | `BEST_PARAMS_PATH` L101                                  |                       101 | OK                           |
|   51 | `config.py`                    | antes de `OUTPUT_DASHBOARD` L252                         |                       252 | OK                           |
|   52 | `pipelines/transcripciones.py` | `main_workflow_clasico.py` L155                          |                       155 | EXCLUIDO (§3.6)              |
|   53 | `main_workflow_clasico.py`     | shim completo                                            |                         — | EXCLUIDO (§3.6)              |
|   54 | `main_workflow.py`             | `optimize_n_startup_trials: int = 25` L923-931           |                       929 | OK (ver nota E)              |
|   55 | `main_workflow.py`             | tras `_cdh_frac_from_params` (doc 7883/7885)             |                      7885 | OK (ver desfases)            |
|   56 | `main_workflow.py`             | `_precompute_coref_chains` L7988-8000                    |                 7988-8000 | OK                           |
|   57 | `main_workflow.py`             | `get_search_space` dict L8409-8432                       |                 8409-8432 | OK                           |
|   58 | `main_workflow.py`             | `_suggest` L8435-8448                                    |                 8435-8448 | OK                           |
|   59 | `main_workflow.py`             | tras `_is_sentinel` L7855                                |                 7852-7857 | OK (colisión con #31)        |
|   60 | `main_workflow.py`             | `try:`/`except` L8589-8593                               |                 8589-8593 | OK                           |
|   61 | `main_workflow.py`             | 12 retornos de gate                                      |                 8561-8718 | OK                           |
|   62 | `main_workflow.py`             | `_make_sampler` L8773-8778                               |          8767 / 8773-8778 | OK                           |
|   63 | `main_workflow.py`             | warm start L8949                                         |                      8949 | OK                           |
|   64 | `main_workflow.py`             | `study.optimize` L8975-8994                              |                 8975-8976 | OK                           |
|   65 | `main_workflow.py`             | `_bp_params` L8188                                       |                 8188-8198 | OK                           |
|   66 | `method_selection.py`          | `Selection.null_mode` default `"gate"`                   |             **no existe** | PENDIENTE (#9)               |
|   67 | `method_selection.py`          | `_feasibility`: `null_mode="diagnostic"` + piso          |        firma nueva de #12 | PENDIENTE (#12)              |
|   68 | `method_selection.py`          | textos `explain_lines` modo diagnostic                   |       bloque nuevo de #11 | PENDIENTE (#11)              |
|   69 | `main_workflow.py`             | `return sorted(counts.values())` L558                    |                       558 | OK                           |
|   70 | `main_workflow.py`             | gate K real (doc 8606) + L8683/L8702                     |        8609 / 8683 / 8702 | OK (depende #39/#40)         |
|   71 | `main_workflow.py`             | `except` mudo + `_MAX_TB_CHARS` L8590-8594               |          8589-8593 / 7852 | **CONFLICTO (#60)**          |
|   72 | `method_c_policy.py`           | módulo nuevo (ausente)                                   |                         — | OK (crear; ausente esperado) |
|   73 | `method_c_policy.py`           | ampliación del módulo                                    |                         — | OK (crear tras #72)          |

Notas de la tabla:

- **A (#3/#4)**: §2 bloque 1 dice que «no tienen diff en ningún documento»,
  mientras §4 y §9 dicen que entran con el diff literal de `STABILITY_NULL`
  §7(a). Contradicción interna del patchset; el ancla (EOF) es válida en ambos
  casos.
- **B (#12)**: `_fmt` y `_null_class` **no existen** hoy; el bloque OLD sólo
  cubre banner + `_feasibility` (271-310). Las funciones se **añaden** con el
  hunk, no se reemplazan.
- **C (#19)**: doc dice `MethodMetrics(` en L5687; real **5677** (−10, ya
  documentado en §3.5).
- **D (#39/#40)**: `-float(n_clusters)` aparece **3 veces** (8683, 8702 y 8732).
  El bloque de contexto desambigua (rama `select` / `wc_only` / etapa 2). No
  tocar la de 8732 (etapa 2 `all`).
- **E (#54)**: el knob `optimize_n_startup_trials` tiene **3 sitios**:
  campo de `Config` en **929** (`= 25`), fallback `getattr(..., 25)` en **8770**
  (`_make_sampler`) y constructor explícito `optimize_n_startup_trials=25,` en
  **10518**. Cambiar sólo el campo deja el valor efectivo en 25 si `Config` se
  construye por el helper de 10518.

---

## ANCLAS PROBLEMÁTICAS

### #71 — ancla en conflicto con #60 (crítico)

`HARDENING.md` §8.2 (que es #71) pide endurecer:

```
        try:
            result = self._evaluacion_rapida(cfg, uc_cfg=uc_cfg)
        except Exception as e:
-            logger.debug("Trial %d exception: %s", trial.number, e)
+            ...
             return _SENTINEL_TRIAL
```

Texto real (`main_workflow.py:8589-8593`, dentro de `Optimizador._evaluate_trial`,
`def` en L8500) — **idéntico**:

```
        try:
            result = self._evaluacion_rapida(cfg, uc_cfg=uc_cfg)
        except Exception as e:
            logger.debug("Trial %d exception: %s", trial.number, e)
            return _SENTINEL_TRIAL
```

**Pero ese es exactamente el bloque que #60 (`V5_SPACE` §7.7) borra.** §5.2
afirma que son «dos `except` distintos y aplican ambos, sin colisión»: **es
falso**. El método `_evaluacion_rapida` (L8453-8497) **no tiene ningún `except`**
— sólo `try:` / `finally:` — y en todo el fichero
`logger.debug("Trial %d exception: ...")` aparece **una sola vez**, en L8593.

Consecuencia: si se aplica #60, el ancla de #71 desaparece (hunk sin destino); si
se aplica #71, sobrevive el `except` que #60 quería eliminar y #64 envuelve un
objetivo que ya no lanza nunca. **Son mutuamente excluyentes en ese sitio.**

Resolución recomendada (coherente con §3.4, que elige B): **no aplicar #71**; el
guard autoritativo es #59/#60/#64. Si se quiere el rastro de excepción, hay que
reinyectarlo **dentro** del `_make_guarded_objective` de #59, no aquí.

### #23 — hunk sin ancla

`run()` debe poblar `bootstrap_ari`/`coverage_stable`/`null_p` desde `res_emb1` y
del motor de estabilidad. El patchset **no da ni una línea de texto ni de
contexto** para este hunk (sólo la especificación, y §4 lo remite a
«`STABILITY_NULL` §7(c)»). No es aplicable a mano tal cual: hay que decidir el
punto de inserción. Candidatos reales:

- `DoubleClassifier.run()` (def en **5978**), donde ya se construye `selection`.
- `WorkflowOrchestrator.ejecutar`, bloque de selección **9959-9965** (justo
  después de `_sel = getattr(self.double_clf, "last_selection", None)`).

Marcado como **SIN ANCLA**, no como obsoleto: falta texto, no sobra.

### #66 / #67 / #68 — PENDIENTE-DE-TRAMO-PREVIO

Los tres vienen de `STABILITY_NULL` §7(b) y **sustituyen texto que introducen
hunks del mismo tramo 2/3**:

- **#66** cambia el default de `Selection.null_mode`, campo que **crea #9**
  (`Selection` actual, L140-168, no tiene `null_mode`).
- **#67** añade `null_mode="diagnostic"` y el piso `bootstrap_ari >= 0.30` a
  `_feasibility`, cuya firma y cuerpo **reescribe #12**.
- **#68** añade textos al modo `diagnostic` en `explain_lines`, bloque que
  **reescribe #11**.

No son anclas obsoletas: son anclas que **aún no existen**. Aplicar en el orden
#8 → #9 → #10 → #11 → #12 → (#66, #67, #68) y **verificar el texto de #9/#11/#12
ya aplicado** antes de pegar #66-#68 (el doc cita literales de la versión nueva,
no de la actual).

### #46 — bloque no contiguo

`CLASSIC_MODE` §9.5 pide reemplazar:

```
-        try:
-            ConfigStore(config.db_dsn).seed_from_config(config)
-        except Exception as e:
-            print(f"   [WARNING] ConfigStore.seed_from_config falló: {e}")
```

Real (`main_workflow.py:9105-9110`) hay **dos líneas intercaladas** que el hunk
no contempla:

```
        try:
            from reii.backend.config_store import ConfigStore     # L9106

            ConfigStore(config.db_dsn).seed_from_config(config)   # L9108
        except Exception as e:
            print(f"   [WARNING] ConfigStore.seed_from_config falló: {e}")
```

No se puede pegar el diff tal cual (fallaría el contexto). Hay que editar a mano
conservando el `import` local, o moverlo arriba.

### #31 vs #59 / #31 vs #64 — colisión de símbolo y de sitio

- `_MAX_TB_CHARS` **no existe aún** y ambos hunks lo definen en el mismo punto
  (`_SENTINEL_TRIAL` L7852 / `_is_sentinel` L7857): #31 con 4096, #59 con 2000.
  Aplicar **una sola vez** (recomendado 2000, §3.4). Dos definiciones = gana la
  última sin avisar.
- #31 envuelve `study.optimize` con `lambda ... guarded_trial_eval(...)`, que es
  **la misma línea** que #64 reescribe con `_make_guarded_objective`. Tras #64,
  el ancla de #31 ya no existe. Si se mantiene #31, debe ir **antes** de #64 y
  #64 la sustituye.

---

## Desfases de línea (correcciones a `§3.5` del patchset)

El §3.5 del `FINAL_PATCHSET.md` ya avisa de casi todos; estas son las
discrepancias que **quedan mal en ese §3.5** o que no están listadas:

| doc:§                 | dice                                                     |             real | delta                              |
| --------------------- | -------------------------------------------------------- | ---------------: | ---------------------------------- |
| §3.5                  | `L7883` = `return float(default)` «coincide exactamente» |         **7885** | +2                                 |
| V5_SPACE §7.2         | «fin en L7885»                                           |             7885 | **correcto** (manda el doc fuente) |
| §1 #54                | `Config` L923-931                                        | campo en **929** | dentro del rango                   |
| §3.5 / HARDENING §8.3 | `@@ -8606` gate de K                                     |         **8609** | +3                                 |
| BACKEND_C §12.1       | ancla L1038                                              |             1038 | exacto                             |
| BACKEND_C §12.2       | `return sorted(counts.values())` L558                    |              558 | exacto                             |
| CLASSIC_MODE §9.7     | `"afc_y_proyeccion"` L63                                 |           **68** | +5                                 |
| §1 #19                | `MethodMetrics(` L5687                                   |             5677 | −10 (ya en §3.5)                   |
| §1 #20                | `context` L5964                                          |             5965 | +1 (ya en §3.5)                    |
| V5_SPACE §7.7         | `try/except` L8590-8594                                  |        8589-8593 | −1 (ya en §3.5)                    |
| V5_SPACE §7.6         | `_is_sentinel` L7857                                     |             7855 | −2 (ya en §3.5)                    |

Anclas comprobadas **exactas** (delta 0): #1 (6), #5 (17), #7 (86), #9 (167),
#10 (191), #11 (207), #13 (312), #14 (476), #15 (1006), #16 (5707), #17 (8671),
#19a (5665), #21 (1038), #22a (80-81), #22b (3831), #22c (5089), #22e
(6124/6131), #28 (122 líneas), #33 (347), #37 (8530), #43 (8336), #48 (5951),
#50 (101), #51 (252), #56 (7988-8000, con `int(c)` en 7999), #57 (8409-8432),
#58 (8435-8448), #59 (7852/7855), #60 (8589-8593), #62 (8773-8778), #63 (8949),
#64 (8975), #65 (8188), #69 (558), #70 (8683/8702).

---

## Notas de aplicación (no bloquean, pero cambian el resultado)

1. **#15 + #16 son un par**: `0.90` en `Config` (L1006) **y** el `getattr`
   (L5707). Cambiar uno solo hace que el umbral efectivo dependa de cómo se
   construyó `Config`. `method_selection.py` tiene además **su propio** `0.90`
   en tres sitios (`Selection.max_class_share_max` L167, `_feasibility` L279,
   `select_method` L319): si no se tocan, el default de la librería sigue en
   `select-v3`. #12/#13 los cambian dentro del rewrite; asegurarse de que no
   quedan `0.90`.
2. **#61**: los 12 retornos existen y coinciden (8561, 8563, 8598, 8604, 8610,
   8628, 8650, 8652, 8690, 8695, 8700, 8718). **No** tocar el `return` de L8594:
   es el `except` que #60 borra.
3. **#70**: depende de #39/#40 (`-float(_k_ach)`); el `_k_ach` de la rama
   `select` debe refinarse con `winner_k` **antes** del `return` de L8683.
4. **#72/#73**: crear `src/reii/method_c_policy.py` **antes** de #22c (imports) y
   antes de #69 en la práctica (la constante `_METHOD_C_WARD_MAX_N` se pasa como
   `ward_max_n=` a `guarded_fit`). Fichero ausente hoy: correcto.
5. **Fuera de los ficheros diana** (#24-#27, #52, #53): anclas localizadas
   (`config_store.py:36`, `config_store.py:68`, `steps.py:166/186`,
   `transcripciones.py:155`) pero no se han validado contra el diff de
   `CLASSIC_MODE` §8/§9.7-§9.10 porque el encargo limita el alcance a los 6
   ficheros diana.

---

## Orden de aplicación corregido (sólo las discrepancias con §2)

```
Bloque 1:  #1, #2            (y #3/#4 SÓLO si se acepta el diff de STABILITY_NULL §7(a))
Bloque 2:  #5..#14           (los OLD coinciden literalmente)
Bloque 3:  #15, #16, #17, #19, #20, #72, #73, #21, #22a-e, #69, #23*
Bloque 5:  #28
Bloque 6:  #29..#51 (sin #38/#41/#42; #31 ANTES de #64 si se mantiene)
Bloque 7:  #55, #54, #58, #57, #56, #59, #60, #61, #62, #63, #64, #65
Tras #9/#11/#12: #66, #67, #68
#70 tras #39/#40.   #71: NO aplicar (colisión con #60).
```

`#23*` = requiere decisión de punto de inserción antes de tocar nada.

---

## ESTADO: RESUELTO (ver `FINAL_PATCHSET.md` §3.7)

Las 5 incidencias de este informe están cerradas con texto exacto y anclas
reales. Lee §3.7 antes de aplicar cualquier hunk:

| incidencia      | resolución                                                                                                                                                                                         |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **#71**         | **ANULADO**. `_evaluate_trial` sólo se llama en L8976 (dentro de `study.optimize`, que #64 envuelve) ⇒ borrar su `except` (#60) no pierde ninguna ruta. El rastro lo da `_trial_exc_attrs` de #59. |
| **#31**         | **ANULADO**. `_MAX_TB_CHARS = 2000` + `_make_guarded_objective` vienen de #59 y #64. `guarded_trial_eval` no existe hoy en `src/` ni en `tests/` (0 hits) ⇒ sin daño colateral.                    |
| **#54**         | **3 sitios**: L929, L8770 (fallback `getattr(..., 12)`) y L10518. Con parches literales en §3.7.3.                                                                                                 |
| **#66/#67/#68** | **ABSORBIDOS en #9/#11/#12**. Se escribe el texto final de una vez; constante `STABILITY_MEAN_MIN` en mayúsculas (el doc la traía en minúsculas) junto a `RULE_VERSION`.                           |
| **#23**         | **PARTIDO**: #23a (10 pasos, anclas verificadas, **inerte** mientras `_stability_spaces` sea `None`) + #23b (rellenar espacios, fuera de este patchset).                                           |

Dos anclas de documentos fuente que **corrige** §3.7 y que este informe no había
detectado:

1. `STABILITY_NULL.md` §7(c) inserta los 4 campos de `Config` antes de
   `method_selection_k_bounds`: el ancla **es válida** (en el fichero real las
   tres líneas son contiguas: `margin` 998, `coverage_floor` 999, `k_bounds`
   1000). Lo que faltaba era que el _patchset_ remitiera a ella: ahora está
   tabulada como paso 1 de #23a en §3.7.5.
2. `#46` requiere conservar el `import` local (`from reii.backend.config_store
import ConfigStore`, L9106), que el hunk original borraba. Texto corregido en
   §3.7.6.

Recuento final: **58 hunks aplicables** de los 73 inventariados.
