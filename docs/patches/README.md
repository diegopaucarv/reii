# Patchset v5 — ficheros `.patch`

Generado a partir de `docs/design_v5/FINAL_PATCHSET.md` (§2 tramos y §3.7, que **manda**
sobre §1/§2) y verificado contra el árbol real el 2026-10-28.

## Estado de verificación

El patchset de **22 patches** aplica sin conflictos en un único árbol acumulado
(`git apply -p1`, scratch en `%TEMP%`, orden canónico de abajo). Verificado el
2026-10-28: **21/21 de la cadena + `15` independiente (`git apply --check` rc=0) sin
rechazos**, incluidos los cuatro que tocan
`src/reii/main_workflow.py` en regiones distintas (`12` en `apply_classic_mode`, `13` en
`lematizar_uces` y el import de `gramatica`, `14` en `lematizar_uces` + `Config`, y
`02g`/`02h`, que solo tocan `tests/`). `15` es independiente: toca
`src/reii/backend/config_store.py` y **crea** `tests/test_config_allowlist.py`.

```
01 ok · 02 ok · 02b ok · 03 ok · 03b ok · 05 ok · 06 ok · 07 ok · 06b ok
02c ok · 02d ok · 02e ok · 02f ok · 02g ok · 02h ok · 09 ok · 10 ok
11 ok · 12 ok · 13 ok · 14 ok · 15 ok
```

**Base de aplicación.** El árbol de partida **no es `git HEAD`**: es el árbol de trabajo
del 2026-10-28 (`backup_patchset_20261028_135329`), que incluye ficheros **untracked**
que los patches _modifican_, no crean: `tests/test_mode_classic.py` (reescrito por
`02g`), `scripts/playground/classic_profile_check.py` (reescrito por `02h`) e
`ia/gramatica.json` (tocado por `14`). Desde un clon limpio de `HEAD` esos tres patches
fallan con `No such file or directory`; hay que restaurar el árbol de trabajo antes.

Suite completa sobre ese árbol (12 ficheros, `PYTHONPATH=src`, sin `-o pythonpath`):

```
203 passed, 4 xpassed in 94.21s        # cadena de 19
22 passed in 4.07s                     # 02h + 14 en los 2 ficheros que tocan
 3 passed in 5.80s                     # 15 (test_config_allowlist.py)
```

Los dos ficheros que tocan `02h`/`14` aportaban 19 tests (14 + 5) y ahora aportan 22
(15 = 14 de `02g` + 1 de `02h`; 7 de retención) ⇒ global esperado ≈ **206 passed**,
4 xpassed, 0 skipped (203 − 19 + 22). `15` añade 3 ⇒ **≈ 209 passed**. Re-medir tras
aplicar; no dar la cifra por buena.

0 failed · **0 skipped**. Los 4 `xpassed` son de `test_backend_hardening`, que esperaba
`method_c_policy.py` (lo crea `03b`): pasan, y el `xfail` es `strict=False`.
Los 13 `skipped` que había en `test_mode_classic` los cierra `02g` + `12` (14/14).

`13` crea `ia/gramatica.json` si no existe; `14` **lo modifica** para fijar
`retention_policy: "tool_adverb_filtered"`.

## Orden de aplicación (canónico y obligatorio)

El orden **no** es el de los tramos de §2: `06b` fue rebaseado para aplicarse al final,
porque colisiona con `01` en `quality/metrics.py` y con `03` en `main_workflow.py`.
`09` se generó contra el árbol **con `06b` aplicado** y `10` contra el árbol **con `03b`
aplicado**, así que van después de esos dos. `12` parchea `apply_classic_mode` (lo crea `06`)
y `13` necesita el `gramatica.py` y el `lematizar_uces` de `11`: ambos van al final.

|   # | fichero                                           | tramo | contenido                                                                                                                                                              |
| --: | ------------------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   1 | `01_quality_metrics.patch`                        | 1     | #1, #2 — `C_in`, `d`, `rmsd_proportions` en `quality/metrics.py`                                                                                                       |
|   2 | `02_method_selection.patch`                       | 2     | #5-#14 (+ #66/#67/#68 absorbidos)                                                                                                                                      |
|   3 | `02b_tests_method_selection.patch`                | 2     | reemplazo de 2 tests (§6(b))                                                                                                                                           |
|   4 | `03_main_workflow_gate_select.patch`              | 3     | #15-#17, #19-#21 (gate 0.70 sobre corpus, campos `method_c_*`)                                                                                                         |
|   5 | `03b_backends_c.patch`                            | 3     | #22a-e + **fichero nuevo** `src/reii/method_c_policy.py` (#72/#72b)                                                                                                    |
|   6 | `05_selection_audit.patch`                        | 5     | #28 — rewrite de `src/9_selection_audit.py`                                                                                                                            |
|   7 | `06_classic_mode.patch`                           | 6     | #29-#37 — `--mode classic`                                                                                                                                             |
|   8 | `07_v5_space.patch`                               | 7     | #54-#65, #69, **#39/#40**, #70, #73 (incluye `_METHOD_C_WARD_MAX_N = 6000`)                                                                                            |
|   9 | `06b_stability.patch`                             | 6     | #3, #4, #43-#51 — motor de estabilidad (**rebaseado, va último de los grandes**)                                                                                       |
|  10 | `02c_tests_select_v4.patch`                       | 2     | 7 hunks: asserts de `null_mode`, tests que asumian el gate del nulo activo, `bootstrap_ari < 0.30`                                                                     |
|  11 | `02d_tests_backend_hardening.patch`               | 3     | 2 tests que asumian la politica PRE-endurecida (`dense_cdh` con K=1, `ward` acotando K<2)                                                                              |
|  12 | `02e_tests_stability_null.patch`                  | 6     | `test_stability_null.py` lee `STABILITY_MEAN_MIN` (mayúsculas): los 16 tests dejan de saltarse y pasan                                                                 |
|  13 | `02f_tests_kach_objective.patch`                  | 3+7   | doble `FakeTrial.set_user_attr` + fixture `-K` coherente con `winner_k` (2 fallos reales)                                                                              |
|  14 | `02g_tests_mode_classic.patch`                    | 6     | realinea `test_mode_classic.py` a `CLASSIC_OVERRIDES` / `apply_classic_mode` (14/14)                                                                                   |
|  15 | `02h_playground_classic_profile_check.patch`      | 6     | realinea `classic_profile_check.py` al contrato real (`CLASSIC_OVERRIDES` por AST, `--pipeline`) y añade 1 test (15/15)                                                |
|  16 | `09_fix44_classic_guard.patch`                    | 6     | `except RuntimeError: raise` — hace efectivo el guard de #44                                                                                                           |
|  17 | `10_hardening_ward_max_n.patch`                   | 7     | `ward_max_n` 4000 → **6000** y 256 → 320 MiB en `hardening_bench.py` **y** `method_c_policy.py`                                                                        |
|  18 | `11_formas_activas.patch`                         | —     | `content_pos` + `PROPN`/`X` y sin filtro `is_stop` (ver §Riesgos)                                                                                                      |
|  19 | `12_guard_optimize_only_classic.patch`            | 6     | `--optimize-only` + `--mode classic` → `SystemExit(2)`                                                                                                                 |
|  20 | `13_retention_policy.patch`                       | —     | knob `retention_policy`, tabla de tokens cacheada, evaluador offline (ver §Política de retención)                                                                      |
|  21 | `14_retention_default_tool_adverb_filtered.patch` | —     | default → `tool_adverb_filtered` (`ia/gramatica.json`, `gramatica.py`, `retention_policy.py`, doc §10) ⇒ **invalida `best_params.json`**                               |
|  22 | `15_config_allowlist.patch`                       | —     | allowlist de `Config` 105 → **136** claves (`frac`, `floor`, `method_selection_*`, `retention_policy`…) + **fichero nuevo** `tests/test_config_allowlist.py` (3 tests) |

```powershell
# Comprobación previa (todas deben decir OK antes de aplicar nada).
# Ojo: 09, 10 y 12 NO validan contra el árbol de partida porque parchean código que
# crean 06b (main_workflow.py) y 03b (method_c_policy.py). Para esos tres se usa la
# copia acumulada `.pcbase/` (verificado el 2026-10-28: rc=0 en los tres).
# 02h, 13 y 14 tampoco validan así: 02h reescribe el test que reescribe 02g; 13
# necesita el `gramatica.py` y el `lematizar_uces` de 11; 14 necesita el
# `retention_policy.py` y el `ia/gramatica.json` que crea 13. Los tres solo se
# validan aplicando la cadena en orden (rc=0 los 22).
git apply --check docs/patches/01_quality_metrics.patch
git apply --check docs/patches/02_method_selection.patch
git apply --check docs/patches/02b_tests_method_selection.patch
git apply --check docs/patches/03_main_workflow_gate_select.patch
git apply --check docs/patches/03b_backends_c.patch
git apply --check docs/patches/05_selection_audit.patch
git apply --check docs/patches/06_classic_mode.patch
git apply --check docs/patches/07_v5_space.patch
git apply --check docs/patches/06b_stability.patch
git apply --check docs/patches/02c_tests_select_v4.patch
git apply --check docs/patches/02d_tests_backend_hardening.patch
git apply --check docs/patches/02e_tests_stability_null.patch
git apply --check docs/patches/02f_tests_kach_objective.patch
git apply --check docs/patches/02g_tests_mode_classic.patch
git apply --check docs/patches/11_formas_activas.patch
git apply --check docs/patches/15_config_allowlist.patch

git apply -p1 --directory=.pcbase --check docs/patches/09_fix44_classic_guard.patch
git apply -p1 --directory=.pcbase --check docs/patches/10_hardening_ward_max_n.patch
git apply -p1 --directory=.pcbase --check docs/patches/12_guard_optimize_only_classic.patch

# 13 sólo se comprueba reproduciendo la cadena en un scratch (ver §Estado de
# verificación); `.pcbase/` no trae `gramatica.py` ni el `lematizar_uces` de 11.

# Aplicación, en este orden exacto
git apply docs/patches/01_quality_metrics.patch
git apply docs/patches/02_method_selection.patch
git apply docs/patches/02b_tests_method_selection.patch
git apply docs/patches/03_main_workflow_gate_select.patch
git apply docs/patches/03b_backends_c.patch
git apply docs/patches/05_selection_audit.patch
git apply docs/patches/06_classic_mode.patch
git apply docs/patches/07_v5_space.patch
git apply docs/patches/06b_stability.patch
git apply docs/patches/02c_tests_select_v4.patch
git apply docs/patches/02d_tests_backend_hardening.patch
git apply docs/patches/02e_tests_stability_null.patch
git apply docs/patches/02f_tests_kach_objective.patch
git apply docs/patches/02g_tests_mode_classic.patch
git apply docs/patches/02h_playground_classic_profile_check.patch
git apply docs/patches/09_fix44_classic_guard.patch
git apply docs/patches/10_hardening_ward_max_n.patch
git apply docs/patches/11_formas_activas.patch
git apply docs/patches/12_guard_optimize_only_classic.patch
git apply docs/patches/13_retention_policy.patch
git apply docs/patches/14_retention_default_tool_adverb_filtered.patch
git apply docs/patches/15_config_allowlist.patch
```

Usa `git apply`, **no** `patch`: los `.patch` van en LF y el árbol está en CRLF
(`core.autocrlf=true`); `git apply` lo maneja, `patch` aplica con fuzz y falla en
hunks con contexto en un solo lado.

## Gate de dominancia: qué cambió de significado

El gate pasó de _"la clase mayor no puede ser el 90 % de lo **asignado**"_ a
_"no puede ser el 70 % del **corpus**"_. La diferencia importa porque el denominador
cambia de `n_assigned` a `n_total` (el ruido entra en la cuenta):

| backend                       | clase mayor | asignadas | `n_total` | share asignadas → gate viejo 0.90 | corpus eff → gate nuevo 0.70 |
| ----------------------------- | ----------- | --------: | --------: | --------------------------------- | ---------------------------- |
| `dense_cdh` (asigna todo)     | 442         |       528 |       528 | 0.837 → **acepta**                | 0.837 → rechaza              |
| `hdbscan` (49 de ruido)       | 442         |       479 |       528 | 0.923 → **rechaza**               | 0.837 → rechaza              |
| `ward`/`kmeans` (asigna todo) | ~0.3-0.5    |       528 |       528 | acepta                            | acepta                       |

La asimetría del gate viejo era real: **la misma estructura** (442 en la clase mayor)
se aceptaba con `dense_cdh` (0.837) y se rechazaba con HDBSCAN (0.923), sólo porque el
ruido desaparecía del denominador. El gate nuevo la elimina: ambos leen 0.837.
El precio es que 0.70 sobre corpus es un umbral **más duro** en términos absolutos
(rechaza un corpus legítimamente dominado por un tema, p1 = 0.72): es exactamente la
objeción `C0 vs noise_max` del doc de estabilidad, y la razón para no subirlo a 0.75.

Campos: `dominance_corpus_eff` (`as_row`, patch `02`) = clase mayor / `n_total`;
`max_class_share` = clase mayor / **asignadas**. `#17` lee el primero a propósito.

## Dependencias entre patches

- `03b` invoca `guarded_fit(ward_max_n=_METHOD_C_WARD_MAX_N)`, símbolo que define `07`.
  Ambos van después de `03`, y `07` antes de `06b`.
- `06` importa 17 símbolos de `reii.config` que introduce `06b`: por eso `06b`
  **debe** aplicarse, aunque vaya al final del orden físico.
- `03` (#19/#20/#17) depende de los campos de `Selection` que añade `02`.
- `09_selection_audit.py` (#28 en `05`) asume el esquema de `02`.
- `09_fix44_classic_guard` parchea código que **añade** `06b` ⇒ no aplica antes.
- `10_hardening_ward_max_n` parchea `method_c_policy.py`, que **crea** `03b` ⇒ no aplica
  antes de `03b`.
- `02c`, `02d`, `02e`, `02f` y `02g` sólo tocan `tests/`: independientes del resto del
  orden (van después de los grandes porque sus asserts describen el comportamiento final).
- `11` toca `gramatica.py` y `lematizar_uces`; va antes de `13`, que necesita ese
  `gramatica.py` y ese bloque para insertar `retention_policy`.
- `12` parchea `apply_classic_mode`, que **crea** `06` ⇒ no aplica antes de `06`.
- `02h` reescribe `tests/test_mode_classic.py`, que reescribe `02g` ⇒ va **después** de
  `02g`. Tampoco valida sobre `.pcbase/` (esa copia no tiene el fichero).
- `13` depende de `11` y **no** valida sobre `.pcbase/`.
- `14` es el último de los dependientes: depende de `13` (crea `retention_policy.py` e
  `ia/gramatica.json`) y de `11`/`13` en `main_workflow.py`. **No** valida sobre árbol de
  partida.
- `15` es independiente de la cadena: solo toca `config_store.py` y crea un test nuevo.
  No cambia el vocabulario, así que **no invalida** `best_params.json`.

## Cobertura del inventario

**151 hunks** repartidos en 22 ficheros (40 ficheros destino, contando el `+++` de cada
`.patch`; verificado con `grep -c '^@@' docs/patches/*.patch`). Exclusiones respetadas
tal cual (§3.7):

- **Anulados**: `#31`, `#71`.
- **Absorbidos** en otros: `#66`→#9, `#67`→#12, `#68`→#11.
- **No-op**: `#18`, `#38`, `#41`, `#42`.
- **Fuera de alcance**: `#23b`, `#24`-`#27` (allowlists), `#52`, `#53`.
- **#73 ≡ #72b**: va dentro del fichero nuevo de `03b`, no como hunk aparte.
- **#39/#40**: verificado **hunk a hunk** que están dentro de `07`. En el árbol
  acumulado aparecen en `main_workflow.py:9248-9266` (rama `select`) y
  `9286-9301` (rama `wc_only`), con `trial.set_user_attr("k_ach"/"k_shortfall"/
"k_stable")` y retorno `-float(_k_ach)`. No hay fichero propio porque separarlos
  exigiría recortar `07` y renumerar sus hunks posteriores.

## Riesgos del patch 11 (retención léxica)

Diagnóstico completo en `docs/design_v5/RETENTION_ROOTCAUSE.md`. Resumen:

- El `not is_stop` de `main_workflow.py:2531` era un **predicado muerto**: la rama de
  `marcadores` ya se llevaba todo `is_stop ∧ is_alpha`. Borrarlo solo = no-op; por eso
  el patch lo quita de la rama de marcadores.
- `PROPN` aporta 293 tokens (+5.2 %); `X` aporta 0 en este corpus (knob inerte).
- Línea base honesta de este corpus: **10.72 lemas/UCE, retención 0.395**
  (el `0.400 / 9.0` de cabecera mezcla corpus: 7 de 11 corridas son de otro corpus).
  Con el patch: **13.16 / 0.485**.
- **Decisión CERRADA por regla pre-registrada**: de lo que rescata la agresiva, 484 de
  1291 son palabras herramienta ⇒ cuota **0.375 > 0.30** ⇒ **excluida**; la conservadora
  (`tool_adverb_filtered`, lista cerrada de adverbios-herramienta) da **12.24 / 0.451**,
  cuota 0.000 ⇒ **gana**. Desde `14` es el **default** (antes `pos_only`, que queda como
  elección explícita). Detalle y sensibilidad en §Política de retención.
- **Contaminación de Optuna**: `corpus_fingerprint` no cambia con el vocabulario, así
  que los trials nuevos caerían en la **misma** study que los 21 ya existentes.
  Mitigación obligatoria: archivar `data/reii_optuna.db`, `best_params.json`,
  `quality_multiverse.json` antes de re-optimizar, y usar `--study-name` nuevo.
- **Colisión de `run_id`**: `code_version()` = `git rev-parse --short HEAD`. Sin
  commitear, el `run_id` no cambia y `quality_run` sobrescribe filas. Mitigación:
  commit o `REII_GROUP_VERSION`.

## Decisiones de los agentes que conviene revisar

1. **`#17` lee `dominance_corpus_eff`, no `dominance_corpus`.** `GATE_NOISE.md` §12.2
   emitía `NaN` porque el campo corto no lo mide el productor; `as_row` publica
   `dominance_corpus_eff` (§3.1). Es una desviación deliberada del texto fuente.
2. **`#22c`**: el fence de `BACKEND_C.md` §12.2 borraba `if not _HDBSCAN_AVAILABLE:`
   y dejaba el guard nuevo como contexto; se emitió como **adiciones**.
3. **`method_c_policy.py`** declara además `WARN = r"[!]"` y `SEED = 42`, que el bloque
   heredado del bench usa pero vivían fuera del rango copiado.
4. **`#54`** usa `optimize_warm_start_max=10` (no el 8 que decía `V5_SPACE.md` §7.1).
5. **`10`** sube el presupuesto a 320 MiB (no 384): es el número de `HARDENING.md` §2.2 y
   cubre el peor camino (274.7 MiB a n=6000 por el cuadrado completo; el condensado real
   gasta 137 MiB). Se aplica **también** a `src/reii/method_c_policy.py`: el bloque del
   banco está documentado como _COPIA CANÓNICA_ que debe viajar VERBATIM, y dejarlo
   divergente era un footgun (el banco mediría con 6000 y el módulo con 4000).
6. **`10` también reescribe el `print` de calibración.** `calibrate_ward_max_n` usa la
   curva **medida** (condensado, 4 B/par) y devuelve ~8500; el default es una cifra de
   **política** sobre el peor camino (8 B/par). Antes el banco imprimía
   "calibrado=8500 ... default propuesto=6000" sin explicar la diferencia y parecía
   contradecirse.

## Fallos previsibles tras aplicar (decisión pendiente)

1. ~~`tests/test_stability_null.py` busca `_ms.stability_mean_min` en minúsculas~~ →
   **resuelto** por `02e`. El módulo expone `STABILITY_MEAN_MIN` (mayúsculas) y el
   `skipif` se abría a `False`: 16 tests dormidos. Ahora corren y **pasan** (33 s).
2. **Docstring de `#5`** (literal de `SELECT_V4.md` §7.1) sigue diciendo `"gate"` como
   default; §3.7.4 no ordenaba reescribirlo.
3. **`hdbscan_uce_min_cluster_frac`** y `_frac_loose` siguen fuera de la allowlist de
   `config_store.py` (hunks #24-#27, fuera de alcance por §3.7).
4. ~~`tests/test_v5_space.py` y `tests/test_optimizer_gate.py` no se pudieron correr~~ →
   **corridos**. `test_v5_space` pasa entero. `test_optimizer_gate` destapó **2 fallos
   reales** (`FakeTrial` sin `set_user_attr`, que `07` exige en la rama `wc_only`) y
   `test_select_mode_objective` otro (`-K` pasó de `n_clusters` a `winner_k`): los tres
   los cierra `02f`. Tiempo real de esos 4 ficheros: **104 s** (no 4 min cada uno).

## ~~Gap conocido: `--mode classic` está sin testear (13 skipped)~~ → **CERRADO** por `02g` + `12`

> **Estado actual: cerrado.** `test_mode_classic.py` queda en **14 passed, 0 skipped**
> (ver §Cierre del gap). El texto siguiente se conserva como registro del diagnóstico.

`tests/test_mode_classic.py` estaba escrito contra un **borrador** del diseño, no contra lo
que implementa `06`. El fichero sondeaba un símbolo que no existe y por eso **13 de sus
tests se saltaban** (el resto son anclas estáticas que sí pasan):

| borrador (lo que busca el test)                                                                                   | final (lo que hay en `06`)                                                                                      |
| ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `CLASSIC_PROFILE` (dict)                                                                                          | `CLASSIC_OVERRIDES` (dict, 8 claves)                                                                            |
| `_apply_classic_profile(cfg, argv)`                                                                               | `apply_classic_mode(config, uc_config) -> (Config, UCBuilderConfig)`                                            |
| `--optimize-only` + classic → `SystemExit(2)`                                                                     | **no** se rechaza; `--optimize-only` sólo imprime y sigue                                                       |
| `use_ppmi`, `uce_target_size: 18`, `tsj: 3`, `min_cluster_size_cdh: 0.1`, `swap_iterations: 2`, `n_perm_cdh: 100` | **no** están en el perfil: son los defaults del pipeline (`CLASSIC_MODE.md` §2.1, "deliberadamente SIN cambio") |

No es un rename: hay que reescribir la mitad del fichero. Dos decisiones antes de
hacerlo, porque son contractuales:

1. **`--optimize-only --mode classic`**: ¿se rechaza (`exit 2`) o se documenta como
   combinación sin efecto (classic fija `optimize=False`)? Hoy el código hace lo segundo.
2. **Fidelidad de spec**: ¿los valores históricos (`tsj`, `n_perm_cdh`, …) se afirman
   como _defaults del pipeline_ o se añaden a `CLASSIC_OVERRIDES`? El doc §2.1 eligió
   defaults; el test antiguo exigía overrides explícitos.

~~Recomendación: **no usar `--mode classic` en las corridas del paper hasta cerrarlo**.~~
Ya cerrado: la suite lo cubre (14/14) y el guard de `12` hace ruidoso el caso peligroso
(`--optimize-only` + classic), que antes se ejecutaba en silencio con el objetivo moderno.

## Resuelto en esta iteración

- **`#44` inerte** → `09_fix44_classic_guard.patch` (`except RuntimeError: raise` antes
  del `except Exception` que imprimía `Could not load best_params.json`).
- **Tests que asumían el default viejo o el gate del nulo** → `02c_tests_select_v4.patch`
  (7 hunks). Dos causas distintas: (i) el default `null_mode` pasó de `"gate"` a
  `"diagnostic"`, así que los tests que verificaban el gate lo piden explícito;
  (ii) `STABILITY_MEAN_MIN = 0.30` rechaza un método con `bootstrap_ari = 0.25`, que
  esos escenarios usaban como runner-up. Se suben a 0.45 (siguen por debajo del
  ganador) para que el test siga midiendo lo que dice medir.
- **2 tests contra la política PRE-endurecida** → `02d_tests_backend_hardening.patch`.
  `method_c_policy` (03b) rechaza `target_k < 2` para **todos** los backends y devuelve
  `None` cuando `n_clusters < K_MIN` (BACKEND_C §7), conservando el diagnóstico;
  los tests esperaban que `ward` acotara a K=2 y que hubiera partición con K=1.
- **`hardening_bench.py` a 4000** → `10_hardening_ward_max_n.patch` (6000 / 320 MiB,
  en el banco **y** en la copia canónica `method_c_policy.py`).
- **Retención 0.400 vs 0.537** → `11_formas_activas.patch` + `RETENTION_ROOTCAUSE.md`.
- **#39/#40**: confirmado hunk a hunk que están dentro de `07` (no requieren fichero
  propio).
- **`classic_profile_check.py` validaba el vacío** → `02h_playground_classic_profile_check.patch`.
  Sondeaba `CLASSIC_PROFILE` (símbolo inexistente), caía a `CANONICAL_CLASSIC_PROFILE` y
  reportaba `PENDIENTE` con **exit 0**: el test pasaba sin validar nada. Ahora valida
  `CLASSIC_OVERRIDES` (8 claves, por `ast`, tipos contra `Config`), `apply_classic_mode`
  (2 params, `_OPTIMIZE_ONLY`/`SystemExit`) y que no fije `pseudocount`; añade
  `--pipeline RUTA` y `test_checker_fails_without_classic_overrides` (falla si falta el
  dict). Medido: **15 passed** en `test_mode_classic.py` (antes 14).
- **Default de retención conservador** → `14_retention_default_tool_adverb_filtered.patch`
  (ver §Política de retención §14). **Consecuencia**: `best_params.json` v4 queda
  inválido (cambia el vocabulario de formas activas ⇒ cambia el significado de
  `min_forms_uc_1`/`forms_gap`) ⇒ re-tuneo obligatorio (§Re-tuneo).

## Cierre del gap de `--mode classic` (02g + 12)

`tests/test_mode_classic.py` se escribió contra un borrador del diseño y 13 de sus
tests se saltaban (sondeaban `CLASSIC_PROFILE` y `_apply_classic_profile`).

| fichero                                | toca                         | aplica sobre             | orden                     |
| -------------------------------------- | ---------------------------- | ------------------------ | ------------------------- |
| `02g_tests_mode_classic.patch`         | `tests/test_mode_classic.py` | árbol **prístino**       | después de `02f`          |
| `12_guard_optimize_only_classic.patch` | `src/reii/main_workflow.py`  | **sólo** árbol acumulado | al final, después de `11` |

- `02g` realinea el test al API final: `CLASSIC_OVERRIDES` (8 claves),
  `apply_classic_mode(config, uc_config)`, `uce_target_size=26` (el borrador decía 18),
  y ejecuta la función real para el rechazo de `--optimize-only` y para comprobar el
  `Config` resultante. Los valores históricos se afirman como **valor efectivo**
  (default de `Config` u override), no como pertenencia al dict.
- `12` añade el guard ruidoso al inicio de `apply_classic_mode`: `--optimize-only` +
  `--mode classic` → `SystemExit(2)`. Hace falta porque `_optimizer_will_run` fuerza el
  optimizador aunque classic fije `optimize=False`, y la etapa 2 corre en `mode="select"`
  (optimizaría el objetivo moderno y escribiría `best_params` en `data/classic/`).
  Como `09` y `10`, `12` **no** aplica sobre el árbol prístino: depende del
  contexto que crea `06`.

Verificación (árbol scratch `b1` = 16 patches + `02g` + `12`, y cadena completa en `b4`):

```sh
cd /c/Users/Julia/AppData/Local/Temp/b1
PYTHONPATH=src python -m pytest -q tests/test_mode_classic.py
# 14 passed in 7.66s   (antes: 1 passed, 13 skipped)
PYTHONPATH=src python -m pytest -q tests/test_select_v4.py tests/test_method_selection.py tests/test_stability_null.py
# 85 passed in 28.90s
```

En la cadena completa de 19 (`b4`, §Estado de verificación) el recuento global es
**203 passed, 4 xpassed, 0 skipped**: `test_mode_classic` no aporta ningún skip.

Nota: `02g` solo (sobre prístino, sin `12`) deja el fichero en `1 passed, 13 skipped`:
los 13 tests siguen saltándose porque `CLASSIC_OVERRIDES` no existe todavía. El guard de
`12` es necesario para que `test_optimize_only_is_rejected_loudly` pase, así que `02g` y
`12` deben aplicarse **ambos** para ver 14/14.

## Política de retención (13)

| fichero                     | toca                                                                                                                                                  | aplica sobre             | orden           |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | --------------- |
| `13_retention_policy.patch` | `src/reii/{retention_policy,gramatica,main_workflow}.py`, `ia/gramatica.json`, `scripts/retention_policy_report.py`, `tests/test_retention_policy.py` | **sólo** árbol acumulado | después de `11` |

- Añade el knob declarativo `retention_policy` (`pos_only` | `pos_plus_propn` |
  `tool_adverb_filtered`), la tabla de tokens cacheada por `corpus_fingerprint` y el
  evaluador offline. Regla pre-registrada y veredicto: `docs/design_v5/RETENTION_POLICY.md`.
- **Default en `13`: sin cambios** (`pos_only`, 10.72). `11` solo → 13.16; `11 + 13`
  → 10.72. El cambio de default lo hace `14` (§14, más abajo).
- Precedencia: `Config.retention_policy` > `ia/gramatica.json` > default del código (no
  repite el footgun de `content_pos`, que `load_gramatica()` pisa siempre).

Verificación (árbol scratch `b3` = 16 patches + `13`):

```sh
PYTHONPATH=src python -m pytest -q tests/test_retention_policy.py
# 5 passed in 0.68s
PYTHONPATH=src python -m pytest -q tests/test_select_v4.py tests/test_stability_null.py
# 58 passed in 18.25s
```

### Medición: la regla la aplica el script, no el analista

Árbol `b4`, modo residuo, bootstrap 1000, seed 42:

```sh
PYTHONPATH=src python scripts/retention_policy_report.py \
  --input /c/Users/Julia/Downloads/reii/data/workflow_data.json --bootstrap 1000 --seed 42
```

| política                | lemas/UCE | retención | rescatados | herramienta | cuota | dist. a 0.537 | regla      |
| ----------------------- | --------: | --------: | ---------: | ----------: | ----: | ------------: | ---------- |
| `pos_only` (baseline)   |     10.72 |     0.395 |          0 |           0 | 0.000 |         0.142 | admitida   |
| `pos_plus_propn` (agr.) |     13.16 |     0.485 |       1291 |         484 | 0.375 |         0.052 | EXCLUIDA   |
| `tool_adverb_filtered`  |     12.24 |     0.451 |        807 |           0 | 0.000 |         0.086 | GANADORA   |
| ALCESTE 2012            |     13.95 |     0.537 |          — |           — |     — |             — | referencia |

La regla está **codificada**: `ALCESTE_RETENTION = 0.537` y `TOOL_QUOTA_MAX = 0.30`
(`scripts/retention_policy_report.py:63-66`), `apply_rule()` filtra por cuota dura y
minimiza la distancia (`:384-390`), `render_verdict()` imprime VEREDICTO y sensibilidad
(`:457-522`), y `--json` lo persiste. Sensibilidad de la restricción dura: 0.15, 0.25,
0.30 y 0.35 dan **la misma ganadora**.

> **Matiz medido (pendiente de fijar en el doc).** §3.1 dice «una forma se cuenta una vez
> por UCE», pero `formas_tokens` trae **72 duplicados intra-UCE** (5658 entradas vs 5586
> formas únicas; p. ej. `noticias` dos veces con el mismo lemma/POS). Deduplicando, las
> cifras pasan a 5586/6797/6380 formas y la cuota de la agresiva baja a 418/1211 = **0.345**.
> A 0.30 sigue excluida y la ganadora no cambia, pero **a 0.35 la agresiva entraría y
> ganaría**: la sensibilidad a 0.35 no es robusta al dedup; a 0.30 sí. Para blindarlo hay que
> declarar el denominador como «formas **rescatadas distintas** por UCE».

## 14 — Default adoptado: `tool_adverb_filtered`

| fichero                              | cambio                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------- |
| `ia/gramatica.json`                  | `"retention_policy": "pos_only"` → `"tool_adverb_filtered"`               |
| `src/reii/gramatica.py`              | `DEFAULT_GRAMATICA["retention_policy"]` + comentario                      |
| `src/reii/retention_policy.py`       | `DEFAULT_RETENTION_POLICY` + docstring de precedencia                     |
| `src/reii/main_workflow.py`          | comentarios de `Config.retention_policy` y del `__init__` del segmentador |
| `tests/test_retention_policy.py`     | fija el default nuevo y las tres vías                                     |
| `docs/design_v5/RETENTION_POLICY.md` | §10 (enmienda); §2 y §5 se conservan como histórico                       |

Estado nuevo, verificado en el árbol real:

```
DEFAULT_RETENTION_POLICY            = "tool_adverb_filtered"
DEFAULT_GRAMATICA[...]              = "tool_adverb_filtered"
ia/gramatica.json                   = "tool_adverb_filtered"
resolve(None, None)                 = "tool_adverb_filtered"
resolve(None, "pos_only")           = "pos_only"        # el JSON manda
resolve("pos_only", None)           = "pos_only"        # Config manda
```

Desviación declarada al adoptar la conservadora (no la agresiva): **−1.71 lemas/UCE
(−12.3 %)** y **−0.086 de retención (−16.0 %)** frente a ALCESTE 2012. `pos_only`
(10.72 / 0.395) y `pos_plus_propn` (13.16 / 0.485, excluida por cuota 0.375 > 0.30)
siguen disponibles como elección explícita.

Verificación: `pytest -q tests/test_mode_classic.py tests/test_retention_policy.py`
→ **22 passed in 4.07s** (15 + 7). El test que cruza las tres definiciones
(`test_default_is_the_same_in_code_gramatica_and_json`) impide que se desalineen.

**Estado:** aplicado. `docs/paper.md` L235, L277 y L332 ya declaran
`tool_adverb_filtered` como default y `pos_only` como opt-in. Antes/después exactos en
`docs/design_v5/PAPER_AMENDMENTS.md` §«Patch 14 — APLICADO».

## 15 — Allowlist de `Config` (105 → 136 claves)

El patch cierra una asimetría precisa: `set_config_many`
(`src/reii/backend/config_store.py:252`) filtraba por `ALL_CONFIG_KEYS` y descartaba en
silencio knobs que el pipeline **sí** consume; el apply
(`src/reii/main_workflow.py:11297`) filtra por `__dataclass_fields__`, así que una fila
de BD ya se aplicaba y solo el dashboard no podía escribirla.

Categorías añadidas:

- `clasificacion` (11): `hdbscan_uce_min_cluster_frac`, `_frac_loose`, `_floor`,
  `hdbscan_uce_min_samples_ratio`, `cdh_enforce_min_leaf`, `method_c_backend`,
  `method_c_hdbscan_selectable`, `method_c_target_k`, `dense_cdh_min_r2_1d`,
  `_dense`, `_loose_factor`.
- `optimizacion` (7): `optimize_coverage_gate_all`, `_select`, `optimize_stage1_fraction`,
  `optimize_two_stage`, `optimize_n_ei_candidates`, `optimize_warm_start_max`,
  `optimize_freeze_inert`.
- `seleccion_de_metodo` (10): los `method_selection_*`, incluido `max_class_share`.
- `lexico_y_gramatica`: `retention_policy`, `token_table_path`, `use_token_table_cache`.

Exclusiones deliberadas (documentadas en el patch): `content_pos` es un knob muerto
(`load_gramatica()` lo pisa con `ia/gramatica.json`); `run_mode` lo manda `--mode`;
`corpus_name`/`n_entrevistas`/`fecha_analisis` son procedencia;
`palabras_excluir`/`sinonimos`/`stopwords_extra`/`expresiones` son overrides de corpus
en `ia/*.json`.

Verificación: `pytest -q tests/test_config_allowlist.py` → **3 passed** (knobs
requeridos escribibles, exclusiones fuera, allowlist ⊆ campos de `Config` por `ast` y
sin duplicados). `15` no toca léxico ni params: no invalida `best_params.json`.

## Re-tuneo obligatorio tras `14`

El vocabulario de formas activas cambia (5658 → 6465 formas), así que
`min_forms_uc_1` y `forms_gap` cambian de significado y **`data/best_params.json` v4
queda inválido**. `retention_policy` **no** entra en `run_spec_id` ni en
`corpus_fingerprint`, de modo que una study vieja aceptaría trials nuevos sin avisar:
por eso se usa `--study-name` nuevo y `--no-optuna-warm-start`.

```powershell
# 0) preflight: sin contenedores de workflow vivos

docker ps --filter "name=reii-workflow-run" --format "{{.Names}}  {{.Status}}"

# 1) archivar los artefactos de v4 (backup, no borrado)
New-Item -ItemType Directory -Path data\archive_v4 -Force | Out-Null
Move-Item data\best_params.json       data\archive_v4\best_params.v4.json -Force
Copy-Item data\reii_optuna.db         data\archive_v4\reii_optuna.db      -Force
Move-Item data\diag\quality_multiverse.json data\archive_v4\ -Force -ErrorAction SilentlyContinue
Remove-Item data\method_selection.json, data\method_selection_log.jsonl -ErrorAction SilentlyContinue

# 2) re-optimizar en una study NUEVA (v5). ~horas: 40 trials stage1

docker compose --profile batch run --rm workflow python -u src/reii/main_workflow.py --force --optimize-only --trials 40 --stage1-fraction 0.5 --study-name reii_probe_v5 --no-optuna-warm-start
python 3_diagnose_optuna_db.py --study reii_probe_v5

# 3) select con los params nuevos, y auditoría

docker compose --profile batch run --rm workflow python -u src/reii/main_workflow.py --force --mode select
docker compose --profile batch run --rm workflow python -u src/9_selection_audit.py --strict

# 4) refrescar la tabla de provenance

docker compose --profile batch run --rm workflow python -u src/7_quality_multiverse.py --corpus-words 14331
```

Se **copia** `reii_optuna.db` en vez de moverlo: v2/v3/v4 son evidencia del paper y
`3_diagnose_optuna_db.py` las lee del mismo fichero. La study `reii_probe_v5` es
independiente, así que no hay contaminación con la base compartida.

No se invalidan `data/uce_calibration.json` ni `data/coref_chains_cache.json`: la
segmentación en UCEs no depende del léxico (clave `fingerprint:target:min_uce_words`).
La tabla de tokens (`token_table.jsonl`) guarda **todos** los tokens con su POS/is_stop,
no la decisión, así que también sobrevive al cambio de política.
