# Espacio de búsqueda condicional `v5` + guard de excepciones

> Estado: **propuesta**, sin aplicar. Todos los diffs de §7 son para aplicar a
> mano en `src/reii/main_workflow.py`. Nada de este documento se ejecutó contra
> el pipeline (hay una corrida `Up 2 hours` en Docker) ni escribió en `data/`,
> `output/` ni `models/`.
>
> Medición reproducible: `python scripts/playground/v5_space_sim.py` (5000
> draws, 2.7 s, exit 0, sin importar `reii.main_workflow`).

---

## 0. Resumen

El probe v4 dejó 20 centinelas en 40 trials de etapa 1: **13 por gates de
espacio** (`tight >= loose`, `frac <= frac_loose`) y 7 por gates sustantivos
(cobertura / K). El espacio se puede arreglar para que los 13 primeros sean
imposibles **por construcción**, sin tocar los otros 7.

| Métrica (simulación, 5 000 draws U(espacio))                  | v4 actual                          | v5 condicional                         |
| ------------------------------------------------------------- | ---------------------------------- | -------------------------------------- |
| Tasa de draws inviables (`tight>=loose` o `frac<=frac_loose`) | **46.96 %**                        | **0.00 %**                             |
| Pares `(tight, loose)`                                        | 9 combinaciones, **8** factibles   | **21**, todos factibles                |
| Pares `(frac, frac_loose)`                                    | 25 combinaciones, **15** factibles | **48** (malla 0.0025), todos factibles |
| Pares distintos tras resolver `min_cluster_size` (n=528)      | 8 + 15                             | 21 + **46**                            |
| Combinaciones únicas (producto completo)                      | 6.998e8 (29.4 bits)                | 2.351e10 (34.5 bits)                   |
| Idem con 2 parámetros inertes congelados                      | —                                  | **5.22e8 (29.0 bits)**                 |
| Centinelas de gate esperados en 40 trials con `startup=25`    | 11.7 (medido: 13)                  | **0**                                  |
| Draws factibles con `mcs_tight == mcs_loose` (degenerados)    | 6.07 %                             | **2.46 %**                             |
| Centinelas por excepción con traceback                        | 0 de 1 (se perdió)                 | **0 sin traceback posibles**           |

Los otros dos cambios son ortogonales y baratos: un guard que convierte
cualquier excepción en centinela **con traceback** (hoy `#81` se perdió porque
`_evaluate_trial` la descartaba con `logger.debug`), y un presupuesto que deja a
TPE con muestra suficiente en 11 dimensiones efectivas.

---

## 1. Hechos medidos (Optuna 3.6.2, el pin del repo)

Todo lo siguiente se midió con `python -c` contra el `optuna` instalado
(3.6.2, `requirements.txt` lo pinea `<4.0.0`), no se deduce de la documentación:

| Hecho                                                                                                                          | Evidencia                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Un rango **int dinámico** funciona con TPE multiobjetivo y con `enqueue_trial`.                                                | `suggest_int("tight", 1, loose-1)` con `loose` categórico: 25 trials, 0 FAIL, `best_trials` correcto. |
| Un rango **float dinámico** funciona, siempre que `step` y `log` no cambien entre trials (`check_distribution_compatibility`). | `suggest_float("frac", loose+eps, 0.04)`: 25 trials, 0 FAIL.                                          |
| Un **categórico dinámico NO** funciona: `ValueError: CategoricalDistribution does not support dynamic value space`.            | `suggest_categorical("floose", subset)` → trial FAIL con ese `ValueError`.                            |
| Con `multivariate=True`, un parámetro con distribución dinámica queda **fuera** del TPE multivariado (warning por trial).      | `ExperimentalWarning`/`The parameter 'tight' ... is sampled independently`.                           |
| Un valor encolado **fuera** del rango dinámico **no falla**: solo emite `UserWarning` y se usa tal cual.                       | `enqueue_trial({"frac": 0.011})` con `low=0.012` → trial COMPLETE con `0.011`.                        |
| `hdbscan_uce_min_cluster_frac` se resuelve a `mcs = max(5, round(frac * n_emb))` (`_resolve_hdbscan_uce_params`, L5055-5087).  | El piso `hdbscan_uce_min_cluster_floor=5` domina por debajo de `frac ≈ 0.0095` con `n_emb=528`.       |

Consecuencia de diseño: **el parámetro dependiente se deriva con un rango
dinámico int/float** (nunca categórico). Eso mantiene los nombres reales en
`trial.params` / `best_params.json` (no aparecen parámetros auxiliares tipo
`frac_gap`), a costa de que el par acoplado se muestree de forma univariada. El
acoplamiento por construcción _no se puede_ modelar con TPE multivariado de
todos modos (la región factible es un triángulo), así que la pérdida es nula y
se silencia el warning con `warn_independent_sampling=False`.

---

## 2. Espacio condicional final (tabla)

`get_search_space()` devuelve el mismo tipo de dict que hoy, más dos specs
condicionales (`dict`) que `_suggest` resuelve en una segunda pasada:

```python
{"kind": "lt", "other": "<driver>", "lo": 1}                 # entero en [lo, driver-1]
{"kind": "gt", "other": "<driver>", "hi": 0.04, "eps": 1/n}   # float en (driver, hi]
```

| Parámetro                            | Tipo / rango v5                                     | Condición           | Niveles                       | Estado                      |
| ------------------------------------ | --------------------------------------------------- | ------------------- | ----------------------------- | --------------------------- |
| `min_forms_uc_1`                     | `int` **congelado** = `Config.min_forms_uc[0]` (13) | —                   | 1 (era 9)                     | ρ ≈ −0.25/−0.05/0.00 (§5.4) |
| `forms_gap`                          | `int` (2, 7)                                        | —                   | 6                             | sin cambios                 |
| `tsj`                                | `int` (`tsj−2`, `tsj+2`) = (2, 5)                   | —                   | 4                             | sin cambios                 |
| `cdh_target_k`                       | `int` (3, `k_hi`)                                   | —                   | 18 (n=528)                    | sin cambios: ρ(K)=−0.89     |
| `swap_iterations`                    | `categorical` [0, 1, 2, 3, 5]                       | —                   | 5 (era 3)                     | **borde** §3.4              |
| `min_r2_threshold`                   | `categorical` [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]  | —                   | 6 (era 4)                     | **borde** §3.3              |
| `similarity_threshold`               | `categorical` [0.25 … 0.65]                         | —                   | 5                             | sin cambios                 |
| `coref_weight`                       | `categorical` [0.0 … 0.7]                           | —                   | 8 (era 5)                     | **borde** §3.1              |
| `uc_window_size`                     | `int` **congelado** = `uc_config.window_size` (3)   | —                   | 1 (era 5)                     | ρ ≈ 0.03/−0.10/−0.05 (§5.4) |
| `coref_context_units_loose`          | `categorical` [3, 4, 5, 6, 8]                       | driver              | 5 (era 3)                     | **borde** §3.2              |
| `coref_context_units_tight`          | `int` [1, `loose−1`]                                | `tight < loose`     | 21 pares (era 9, 8 factibles) | **condicional**             |
| `hdbscan_uce_min_cluster_frac_loose` | `categorical` [0.005, 0.01, 0.015, 0.02, 0.03]      | driver              | 5                             | **no se mueve** §3.5        |
| `hdbscan_uce_min_cluster_frac`       | `float` (`loose + eps`, 0.04]                       | `frac > frac_loose` | 48 pares (malla 0.0025)       | **condicional**             |

`eps = max(0.0005, round(1/n_emb, 5))` (0.0019 con `n_emb=528`). No es un
número arbitrario: como `mcs = round(frac·n)`, sumar exactamente **1 UCE**
garantiza `mcs_tight ≥ mcs_loose + 1` siempre que el piso absoluto no
intervenga, porque `round(x + 1) = round(x) + 1` para todo `x ≥ 0`. Con
`eps=0.002` (el valor que se sugería en el encargo) se obtiene lo mismo para
`n_emb ≤ 500`, pero deja de valer para corpus grandes; por eso se deriva de
`n_emb`. Si `retrofitted_vectors` es `None` se usa `total_uces` como cota.

Verificación de la invariante (`tests/test_v5_space.py`, 5 000 draws): nunca
`tight >= loose`, nunca `frac <= frac_loose`.

---

## 3. Los 6 bordes clavados

El criterio es el mismo en todos: **el ganador del probe está en el borde ⇒ el
gradiente empírico apunta hacia afuera**, así que el óptimo no puede estar
_antes_ del borde; la pregunta es cuánto extender sin diluir el espacio. La
columna "coste" es el factor de cardinalidad medido por el simulador.

### 3.1 `coref_weight`: 0.4 → 0.7 (+0.5, +0.6, +0.7) — coste ×1.60

`CohesionScorer.score` (L1360-1370) usa
`similarity_weight · cos + coref_weight · coref_overlap`, con
`similarity_weight = 1 − coref_weight` (`UCBuilderConfig.similarity_weight`,
L1124-1126). Es una **combinación convexa**: no hay ninguna cota estructural en
0.4, es solo el tope arbitrario de la grilla anterior. El ganador está en 0.4
con peso de similitud 0.6; el gradiente dice que el corpus del probe tolera más
peso de correferencia. Corto en **0.7** (similitud 0.3): a 0.8-1.0 el peso de la
similitud léxica deja de discriminar y la construcción de UCs degenera en
"une todo lo que comparta una cadena", que ya es el modo `coref_only` por otra
vía — no tiene sentido duplicar ese régimen dentro del espacio continuo.

### 3.2 `loose`: 5 → 8 (+6, +8) y `tight` derivado hasta 7 — coste ×2.63 en pares

El ganador está en `loose=5` y `tight=3`, **los dos bordes superiores**: la
ventana de contexto más ancha medida es la mejor. Pero `context_units` no es
una escala libre: la ventana efectiva satura en el largo del documento (el
slice `max(0, i-ctx)` devuelve el documento completo), y en entrevistas con
pocas UCEs por documento `ctx=8` ya es el documento entero. Se corta en 8
(duplicando el paso 5→6→8 en vez de 5,6,7,8) para cubrir el régimen saturado con
2 niveles en vez de 4.

Coste colateral: `_precompute_coref_chains` precarga **todos** los ctx
alcanzables; con el espacio v5 eso pasa de `{1..5}` a `{1..8}` (3 pasadas de
extracción de cadenas más, una sola vez, cacheadas en
`COREF_CHAINS_CACHE_PATH`). Se indica en el diff §7.3 para que no reviente: un
spec `dict` se convertiría en sus claves (`"kind"`, `"other"`, …) y el
`{int(c) for c in ctx_values}` de L7999 —que está **fuera** del `try` de
L7988— lanzaría `ValueError` desde dentro de `_evaluate_trial`, justo el tipo
de excepción que el guard de §4 existe para registrar.

### 3.3 `min_r2_threshold`: 0.10 → 0.20 (+0.15, +0.20) — coste ×1.50

`ClasificadorDescendente._es_significativo` (L4067-4089, gate en L4078): el
split se rechaza si `var_obs < min_r2_threshold`. Subir el umbral = aceptar solo
expliquen más varianza ⇒ árboles más someros ⇒ menos clusters. El ganador está
en el máximo (0.10), luego la dirección es "más estricto es mejor" para este
corpus (los splits de baja R² son ruido que fragmenta).

Riesgo explícito: si el umbral sube tanto que ningún split es significativo,
CDH devuelve K=1 y el gate `n_clusters < 2` (L8609) emite centinela. Es
decir: **este borde puede fabricar centinelas nuevos**, a diferencia de §3.1/3.2.
Por eso se añaden solo 2 niveles (no una grilla fina) y se recomienda aplicar
0.15 primero si el próximo probe muestra K=2 en el ganador. Cota del daño: ≤ ⅙
de la masa del parámetro puede caer en el régimen K=1.

### 3.4 `swap_iterations`: [1, 2, 5] → [0, 1, 2, 3, 5] — coste ×1.67

El ganador está en el **mínimo** (1). `_partition` hace
`if config.swap_iterations > 0: labels = self._intercambio(..., max_iter=...)`
(L4137-4143): 0 es una ablación legítima (sin refinamiento por intercambio), y
es el único punto que falta _por debajo_ del borde. Añadir 3 rellena el salto
2→5 de una grilla no uniforme. No se añade 4: el régimen útil es el bajo.

### 3.5 `hdbscan_uce_min_cluster_frac_loose`: **no se mueve** (0.005 se queda)

El encargo sugería bajarlo a 0.002. **No sirve**, y la razón es medible:
`mcs = max(floor, round(frac · n_emb))` con `floor = hdbscan_uce_min_cluster_floor = 5`.

| `frac` | `round(frac·528)` | `mcs` efectivo |
| ------ | ----------------- | -------------- |
| 0.002  | 1                 | **5** (piso)   |
| 0.005  | 3                 | **5** (piso)   |
| 0.010  | 5                 | **5** (piso)   |
| 0.015  | 8                 | 8              |
| 0.020  | 11                | 11             |
| 0.030  | 16                | 16             |
| 0.040  | 21                | 21             |

Con `n_emb=528`, `0.002` y `0.005` son **el mismo experimento**: el piso de 5
UCEs los colapsa. El borde real no es la fracción, es
`hdbscan_uce_min_cluster_floor`. Si el objetivo es explorar clusters de
HDBSCAN más chicos, hay que exponer **el piso** (`{3, 4, 5}`), no la fracción:

> Límite de este argumento: la equivalencia vale mientras `round(0.005·n) ≤ 5`,
> es decir `n_emb ≲ 1 100`. Por encima, `0.005` ya sale del piso y `0.002`
> pasaría a ser un nivel distinto (`mcs = 5`). Para el corpus del probe
> (`n_emb = 528`) el argumento es exacto; si el corpus crece, revisar este
> borde antes de reusar la conclusión.

```python
# opcional, NO incluido en §7 (requiere validar el resto del pipeline)
"hdbscan_uce_min_cluster_floor": [3, 4, 5],
```

Por lo mismo `frac` (tight) tampoco se extiende por arriba: 0.04 → `mcs=21` ya
está por encima de la banda 5-20 donde el comentario de `get_search_space`
(L8423-8429) documenta que aparece la estructura real de Method C. **El ganador
está en 0.005 (el mínimo de `loose`) precisamente porque el piso ya estaba
activo**: el borde no informa de un gradiente, informa de una saturación.

#### Degeneración silenciosa del gate (y cómo la reduce v5)

El gate compara **fracciones** (`frac > frac_loose`) pero el efecto real es
**mcs**. Hay pares factibles que resuelven al mismo `mcs` y por lo tanto corren
C1 y C2 con parámetros idénticos (el gate `emb_persistence_c1/c2`, L8715-8718,
mide entonces dos veces lo mismo):

|                      | fracción de draws factibles con `mcs_tight == mcs_loose` |
| -------------------- | -------------------------------------------------------- |
| v4 (grilla discreta) | **6.07 %**                                               |
| v5 (`eps = 1/n_emb`) | **2.46 %**                                               |

Lo que queda en v5 es el caso `loose=0.005` + `frac ∈ (0.0069, 0.0095)`, donde
el piso de 5 vuelve a colapsar. Se documenta como **degeneración residual
conocida** (no se convierte en gate: sería volver a fabricar centinelas). Si se
quiere eliminar del todo, hay que resolver el par sobre `mcs` y no sobre la
fracción:

```python
# alternativa (no propuesta): muestrear mcs_loose int y derivar
# frac = mcs / n_emb, frac_loose = mcs_loose / n_emb
```

Se descarta porque ata el espacio al tamaño del corpus (el espacio ya depende de
`total_uces` por `cdh_target_k`, pero solo en la cota superior, no en la
semántica de cada nivel).

---

## 4. Guard de excepciones (el FAIL #81)

### 4.1 Diagnóstico

`#81` terminó en `TrialState.FAIL` y **sin traceback**. Causa directa: hay dos
capas y ninguna registra el error.

1. L8590-8594: `try: result = self._evaluacion_rapida(...)` /
   `except Exception as e: logger.debug(...); return _SENTINEL_TRIAL`. Esto
   **descarta** el traceback y, además, convierte el fallo en un COMPLETE con
   centinela — indistinguible de un gate.
2. Si la excepción nace **fuera** de ese `try` (que es lo que implica que
   Optuna lo marque FAIL), nadie la mira: `study.optimize` la traga y el trial
   queda FAIL con `system_attrs["fail_reason"]` como único rastro.

Los candidatos fuera del `try` interior son: el bloque de gates (L8560-8563),
`copy.deepcopy(self.config_base)` (L8526), `_cdh_frac_from_params` (L8531), los
`trial.set_user_attr` del camino `select` (L8657-8679), `_bottleneck` /
`np.mean` (L8723-8730) y `_extract_uce_labels` / `adjusted_rand_score` del
camino `wc_only` (L8691-8702). No se puede saber cuál fue (no hay traceback):
ese es exactamente el problema que este diff cierra.

### 4.2 Diseño

- **Un solo choke point**: se elimina el `try/except` interior (L8590-8594) y se
  envuelve `_evaluate_trial` completo con `_make_guarded_objective`. Así el
  sentinela sigue siendo el mismo valor (`_SENTINEL_TRIAL`) — no cambia la
  semántica para Optuna/TPE — pero ahora lleva `error_type`, `error_msg` y
  `traceback` en `user_attrs`.
- `except Exception` **no** captura `KeyboardInterrupt`/`SystemExit` (son
  `BaseException`); se declaran explícitamente y se re-lanzan para que un
  Ctrl-C corte el `study.optimize` de verdad y no lo convierta en "trial
  fallido".
- La escritura de atributos va en su propio `try/except`: un fallo al registrar
  el error nunca puede enmascarar el centinela.
- `[!] trial #N fallo: <tipo>: <msg> (traceback en user_attr)` se imprime **una
  vez** por trial (es el único sitio donde se imprime).
- El reporte final separa los dos sentinelas con `_sentinel_breakdown`
  (`error_type` presente ⇒ excepción; ausente ⇒ gate). Además `_sentinel()`
  etiqueta los retornos de gate con `outcome="gate"` y `gate_reason`, así que
  `3_diagnose_optuna_db.py` puede leer `outcome` directamente en vez de
  inferirlo.

### 4.3 Contrato resultante

| Estado en la BD | `values`          | `user_attrs`                                                  | Significado                                        |
| --------------- | ----------------- | ------------------------------------------------------------- | -------------------------------------------------- |
| COMPLETE        | reales            | `outcome=gate`?                                               | válido                                             |
| COMPLETE        | `_SENTINEL_TRIAL` | `outcome="gate"`, `gate_reason`                               | inviable por gate (cobertura, K, espacio…)         |
| COMPLETE        | `_SENTINEL_TRIAL` | `outcome="exception"`, `error_type`, `error_msg`, `traceback` | **excepción** (antes: FAIL sin rastro)             |
| FAIL            | —                 | `fail_reason` de Optuna                                       | excepción **fuera** del objetivo (sampler/storage) |

Nota de compatibilidad: `scripts/playground/classic_mode_probe.py` (diseño
anterior) usaba `exception_type/exception_msg/exception_tb` y `trial_failed`.
Este diff usa **`error_type/error_msg/traceback`** (los nombres del encargo) y
escribe también `trial_failed=True`, así que ambos consumidores siguen
funcionando si el script viejo consulta `trial_failed`.

---

## 5. Presupuesto

### 5.1 Por qué 11 válidos de 13 parámetros no entrenan a TPE

Con `--trials 40` y `n_startup_trials=25` (el valor vigente), **62 % del
presupuesto se va en draws aleatorios**. A los 15 restantes, TPE multiobjetivo
les aplica `n_below = min(ceil(0.1·n), 25)` ⇒ `n_below = 2`: el Parzen
"bueno" tiene 2 puntos en 13 dimensiones y el "malo" 13. De ahí el ruido.

Si además el 47 % de los draws cae en centinelas de gate (que entran al estudio
como valores dominados), la fracción de información útil por trial baja a
~53 %, y los válidos bajan a 11 en el probe real.

### 5.2 Cifras medidas por el simulador

| `n_startup_trials` + trials TPE   | `n_below` | Veredicto         |
| --------------------------------- | --------- | ----------------- |
| 25 + 15 (hoy: 40 trials)          | **2**     | Parzen degenerado |
| 12 + 108 (120 trials, `frac=0.3`) | **11**    | ok                |
| 12 + 188 (200 trials, `frac=0.3`) | **19**    | ok                |

Niveles marginales sin visitar a `n` draws, y `n` para cubrirlos todos al 95 %
(`L·(1−1/L)^n`):

| Parámetro                                                           | niveles | n=40 | n=120 | n≥ (95 %) |
| ------------------------------------------------------------------- | ------- | ---- | ----- | --------- |
| `cdh_target_k`                                                      | 18      | 1.83 | 0.02  | **103**   |
| `min_forms_uc_1`                                                    | 9       | 0.08 | 0.00  | 45        |
| `coref_weight`                                                      | 8       | 0.04 | 0.00  | 39        |
| `forms_gap` / `min_r2_threshold`                                    | 6       | 0.00 | 0.00  | 27        |
| `swap_iterations` / `similarity_threshold` / `loose` / `frac_loose` | 5       | 0.00 | 0.00  | 21        |
| `tsj`                                                               | 4       | 0.00 | 0.00  | 16        |

`cdh_target_k` es el único que domina: 18 niveles exigen `n≈103` para cobertura
marginal al 95 %. Es también el parámetro con mejor señal
(ρ(cdh_target_k, −K) = −0.89), así que **no se toca la grilla**: es un coste
aceptado, no un desperdicio.

### 5.3 Recomendación

```
--trials 120 --stage1-fraction 0.3      # 30 etapa 1 + 90 etapa 2
optimize_n_startup_trials = 12          # (era 25)
optimize_n_ei_candidates  = 48          # (era el default 24)
```

- `--trials 120`: mínimo para que `n_below ≥ 10` **y** `cdh_target_k` tenga
  cobertura marginal. Por debajo de 80, usar `optimize_sampler="random"` de
  forma explícita y no fingir que TPE aprende.
- `n_startup_trials=12`: el startup solo siembra el modelo; con 11 dimensiones
  efectivas y `n_below` como criterio, 12 es el punto donde el resto del
  presupuesto rinde. En etapa 2 los trials encolados (warm start) cuentan como
  COMPLETE y consumen startup, así que no hace falta subirlo.
- `n_ei_candidates=48`: duplica la resolución del argmax sobre `EI` por trial.
  Cuenta solo CPU (no invoca el pipeline), así que es gratis en tiempo de
  corrida.

### 5.4 Qué congelar (lista exacta)

| Parámetro        | Valor                             | ρ(stage1 ARI, stage2 ARI, coverage) | Por qué                                                                          |
| ---------------- | --------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------- |
| `uc_window_size` | **3** (`uc_config.window_size`)   | 0.03 / −0.10 / −0.05                | Inerte en los 3 objetivos.                                                       |
| `min_forms_uc_1` | **13** (`Config.min_forms_uc[0]`) | −0.25 / −0.05 / 0.00                | Con `n=11`, el IC95 % de ρ es ≈ ±0.6: los tres valores son indistinguibles de 0. |

Cómo: el dict del espacio acepta **valores escalares** (la rama `else:
params[k] = v` de `_suggest` ya existe), así que congelar es poner el número en
vez de la lista. No rompe nada aguas abajo (`_perturb_params` los ignora,
`_params_feasible` no los mira, `_apply_best_params` los lee de `best_params`
si están) y se emiten explícitamente en `best_params.json` (§7.12) para que la
trazabilidad sea completa.

Aviso de acoplamiento: congelar `min_forms_uc_1=13` deja `mf2 = 13 + forms_gap
∈ [15, 20]`, mientras que antes se exploraba `[11, 24]`. Si en el próximo probe
la señal apunta a vocabulario chico, la alternativa es **congelar `forms_gap=4`
y dejar libre `min_forms_uc_1`**; no ambas cosas a la vez.
Escape hatch: `Config.optimize_freeze_inert = False` restaura el espacio v4 en
esos dos parámetros sin tocar código.

Efecto medido: 13 dims (34.5 bits) → 11 dims (**29.0 bits congelado**), que es
_menos_ que el v4 nominal (29.4 bits) pero **100 % factible** en vez de 53 %.

### 5.5 Fuga de presupuesto en el warm start (hallazgo adicional)

`_run_single_stage` encola **cada** trial del frente de Pareto de la etapa 1 más
2 copias perturbadas (L8949-8953). Un frente de 3 objetivos con 25-30 puntos
encola 75-90 trials: con `--trials 120` (90 en etapa 2) **el presupuesto entero
de la etapa 2 se consume en puntos encolados** y TPE no muestrea nunca. De ahí
el tope `optimize_warm_start_max=8` del §7.1 (≤ 24 encolados).

---

## 6. Verificación

Transcripción **literal** de la corrida usada para las cifras de §0
(`python scripts/playground/v5_space_sim.py`, 2.7 s, exit 0):

```
------------------------------------------------------------------------------
-> V5 space sim · 5000 draws · seed=20251028 · n_emb=528
------------------------------------------------------------------------------
-> 0. Chequeo del diff en main_workflow.py (solo lectura/AST)
[!] diff aun NO aplicado (se simula la implementacion de referencia): espacio condicional, guard, resumen
------------------------------------------------------------------------------
-> 1. Tasa de inviabilidad estructural (5000 draws uniformes)
   v4: inviables 46.96%  (tight>=loose 10.86% · frac<=frac_loose 36.10%)   degenerados 6.07%
   v5: inviables  0.00%  (tight>=loose 0.00% · frac<=frac_loose 0.00%)   degenerados 2.46%
[ok] el espacio v5 nunca genera una combinacion inviable por gate
=> probe v4: 13/40 centinelas de gate (medido) vs 11.7 esperados con 25/40 draws uniformes (62% del presupuesto; el resto lo propone TPE)
=> en v5 esos 13 desaparecen; los 7 restantes del probe son de cobertura/K (otro problema: el rediseno no los toca)
------------------------------------------------------------------------------
-> 2. Cardinalidad alcanzable (solo combinaciones factibles)
   v4: 6.998e+08 combos · 29.4 bits · pares coref 8 · pares frac 15
   v5: 2.351e+10 combos · 34.5 bits · pares coref 21 · pares frac 48 (malla 0.0025)
   resueltos a min_cluster_size (n=528): v4 6.532e+08 · v5 2.253e+10 combos
=> espacio nominal x33.6 · todo el espacio v5 es factible (en v4 solo 53.3% de sus combos pasaba el gate)
[ok] el espacio v5 no degrada la cobertura (cardinalidad >= v4)
   [borde] coref_weight: 5 -> 8 niveles (x1.60)
   [borde] min_r2_threshold: 4 -> 6 niveles (x1.50)
   [borde] swap_iterations: 3 -> 5 niveles (x1.67)
   [borde] coref_context_units_tight: 8 -> 21 combos (mcs: 21)
   [borde] hdbscan_uce_min_cluster_frac: 15 -> 48 combos (mcs: 46)
------------------------------------------------------------------------------
-> 3. Marginales (v5, antes de congelar) — masa en los extremos
   min_forms_uc_1                         niveles=  9  extremo=22.5% (x1.01 uniforme)  moda=17 (12.0%)
   forms_gap                              niveles=  6  extremo=32.4% (x0.97 uniforme)  moda=5 (17.4%)
   tsj                                    niveles=  4  extremo=51.5% (x1.03 uniforme)  moda=5 (26.1%)
   cdh_target_k                           niveles= 18  extremo=11.3% (x1.02 uniforme)  moda=8 (6.4%)
   swap_iterations                        niveles=  5  extremo=39.4% (x0.98 uniforme)  moda=2 (20.3%)
   min_r2_threshold                       niveles=  6  extremo=34.7% (x1.04 uniforme)  moda=0.03 (17.9%)
   similarity_threshold                   niveles=  5  extremo=40.3% (x1.01 uniforme)  moda=0.65 (20.8%)
   coref_weight                           niveles=  8  extremo=25.1% (x1.00 uniforme)  moda=0.4 (13.1%)
   uc_window_size                         niveles=  5  extremo=41.2% (x1.03 uniforme)  moda=1 (21.2%)
   coref_context_units_loose              niveles=  5  extremo=39.4% (x0.98 uniforme)  moda=6 (20.9%)
   hdbscan_uce_min_cluster_frac_loose     niveles=  5  extremo=40.0% (x1.00 uniforme)  moda=0.015 (20.3%)
[ok] ningun marginal colapsa en los extremos
------------------------------------------------------------------------------
-> 4. Efecto de congelar parametros inertes
   v5 completo  :  34.5 bits (13 parametros, 2 condicionales)
   v5 congelado :  29.0 bits (11 parametros: -5.5 bits por uc_window_size=3, min_forms_uc_1=13)
   v4 (referencia):  29.4 bits
[ok] con los 2 congelados el espacio queda igual o mas chico que v4, pero 100% factible
   [congelado] uc_window_size = 3 (constante en el dict del espacio)
   [congelado] min_forms_uc_1 = 13 (constante en el dict del espacio)
------------------------------------------------------------------------------
-> 5. Presupuesto: niveles marginales sin visitar
   parametro                     n=40   n=60   n=80   n=120
   min_forms_uc_1                  0.08  0.01  0.00  0.00
   forms_gap                       0.00  0.00  0.00  0.00
   tsj                             0.00  0.00  0.00  0.00
   cdh_target_k                    1.83  0.58  0.19  0.02
   swap_iterations                 0.00  0.00  0.00  0.00
   min_r2_threshold                0.00  0.00  0.00  0.00
   similarity_threshold            0.00  0.00  0.00  0.00
   coref_weight                    0.04  0.00  0.00  0.00
   uc_window_size                  0.00  0.00  0.00  0.00
   coref_context_units_loose       0.00  0.00  0.00  0.00
   hdbscan_uce_min_cluster_frac_l  0.00  0.00  0.00  0.00

   n para P(todos los niveles vistos) >= 95%:
     min_forms_uc_1                     niveles= 9  n>=45
     forms_gap                          niveles= 6  n>=27
     tsj                                niveles= 4  n>=16
     cdh_target_k                       niveles=18  n>=103
     swap_iterations                    niveles= 5  n>=21
     min_r2_threshold                   niveles= 6  n>=27
     similarity_threshold               niveles= 5  n>=21
     coref_weight                       niveles= 8  n>=39
     uc_window_size                     niveles= 5  n>=21
     coref_context_units_loose          niveles= 5  n>=21
     hdbscan_uce_min_cluster_frac_loose niveles= 5  n>=21

   n_startup=25 +  15 TPE -> n_below= 2 (Parzen degenerado)
   n_startup=12 + 108 TPE -> n_below=11 (ok)
   n_startup=12 + 188 TPE -> n_below=19 (ok)
=> n_startup=12 y --trials>=120 dejan n_below>=10; con 40 trials y startup=25 TPE nunca llega a modelar
------------------------------------------------------------------------------
=> resumen: inviables 47.0% -> 0.0%
=> resumen: dimensionalidad efectiva congelada 29.0 bits
[ok] simulacion completa
```

El simulador replica el muestreo con la misma semántica que Optuna
(categórico = uniforme sobre la lista, int = uniforme en `[lo, hi]`,
float = uniforme en `[lo, hi]`) y, si el diff ya está aplicado, lo detecta por
nombre de función (`[ok] main_workflow.py ya contiene el espacio condicional y
el guard`). No importa `reii.main_workflow`.

`tests/test_v5_space.py` **no se ejecutó en esta sesión**: importa
`reii.main_workflow` (prohibido en este encargo, >4 min). Es el mismo patrón que
`tests/test_optimizer_gate.py`, que ya importa el módulo en la suite normal.

---

## 7. Diffs exactos

Todos sobre `src/reii/main_workflow.py`. Las líneas son las del archivo actual
(10 697 líneas). Aplicar en este orden.

### 7.1 `Config` — L923-931

```diff
@@ class Config
     optimize_storage: Optional[str] = None  # ej. "sqlite:///reii_optuna.db"
     optimize_study_name: str = "reii_search"
-    optimize_n_startup_trials: int = 25
+    # v5: 25 startup con 40 trials dejaba a TPE con n_below=2 (Parzen
+    # degenerado). 12 siembra el modelo y deja presupuesto para aprender.
+    optimize_n_startup_trials: int = 12
     optimize_multivariate: bool = True
+    # v5: resolucion del argmax de EI por trial (default de Optuna: 24).
+    # Solo cuesta CPU, no invoca el pipeline.
+    optimize_n_ei_candidates: int = 48
+    # v5: tope de puntos del frente de etapa 1 que se encolan en etapa 2
+    # (x3 copias cada uno); sin tope, el warm start puede comerse el
+    # presupuesto entero de la etapa 2.
+    optimize_warm_start_max: int = 8
+    # v5: congelar los parametros inertes (rho~0) en get_search_space().
+    optimize_freeze_inert: bool = True
     optimize_prune_n_startup: int = 15
```

### 7.2 Helpers nuevos — insertar tras `_cdh_frac_from_params` (fin en L7885)

```diff
@@
     return float(default)


+# ── Espacio condicional v5 ─────────────────────────────────────────────
+def _suggest_conditional(
+    trial: "optuna.Trial", name: str, spec: Dict[str, Any], params: Dict[str, Any]
+):
+    """Muestrea un parametro cuyo rango depende de otro ya muestreado.
+
+    ``spec = {"kind": "lt"|"gt", "other": <driver>, ...}``
+
+      lt → entero en ``[lo, other-1]``   (p.ej. tight < loose)
+      gt → float  en ``(other, hi]``     (p.ej. frac > frac_loose)
+
+    Rangos dinamicos: Optuna 3.6.2 los soporta para int/float (solo exige
+    que ``step`` y ``log`` no cambien entre trials) pero NO para categoricos
+    (``ValueError: CategoricalDistribution does not support dynamic value
+    space``). Por eso el dependiente nunca es categorico.
+    """
+    driver = spec.get("other")
+    base = params.get(driver)
+    if base is None:
+        raise ValueError(f"{name}: falta el parametro driver {driver!r}")
+
+    kind = spec.get("kind")
+    if kind == "lt":
+        lo = int(spec.get("lo", 1))
+        hi = int(base) - 1
+        if hi < lo:
+            raise ValueError(f"{name}: rango vacio [{lo}, {hi}] con {driver}={base}")
+        return trial.suggest_int(name, lo, hi)
+
+    if kind == "gt":
+        lo = float(base) + float(spec.get("eps", 0.002))
+        hi = float(spec.get("hi", 0.06))
+        if hi < lo:
+            raise ValueError(f"{name}: rango vacio [{lo}, {hi}] con {driver}={base}")
+        return trial.suggest_float(name, lo, hi)
+
+    raise ValueError(f"{name}: tipo condicional desconocido {kind!r}")
+
+
+def _expand_levels(spec: Any) -> List[Any]:
+    """Valores alcanzables de un spec (para precargar caches y contar)."""
+    if isinstance(spec, dict):
+        return []  # dependiente: lo cubre el driver + _suggest_conditional
+    if isinstance(spec, tuple) and len(spec) == 2:
+        lo, hi = spec
+        if isinstance(lo, int) and isinstance(hi, int):
+            return list(range(lo, hi + 1))
+        return [lo, hi]
+    if isinstance(spec, list):
+        return list(spec)
+    return [spec]
+
+
 class Optimizador:
```

### 7.3 `_precompute_coref_chains` — L7988-7997

```diff
@@ class Optimizador._precompute_coref_chains
             try:
                 space = self.get_search_space()
-                tight_vals = space.get("coref_context_units_tight", [2])
-                loose_vals = space.get("coref_context_units_loose", [4])
-                ctx_values = set(tight_vals) | set(loose_vals)
+                loose_vals = _expand_levels(
+                    space.get("coref_context_units_loose", [4])
+                )
+                tight_spec = space.get("coref_context_units_tight", [2])
+                if isinstance(tight_spec, dict):
+                    # v5: el dependiente es un rango dinamico [lo, max(loose)-1]
+                    _lo = int(tight_spec.get("lo", 1))
+                    _hi = max(int(c) for c in loose_vals) - 1 if loose_vals else _lo
+                    tight_vals = list(range(_lo, max(_lo, _hi) + 1))
+                else:
+                    tight_vals = _expand_levels(tight_spec)
+                ctx_values = set(tight_vals) | set(loose_vals)
             except Exception:
```

Con el espacio v5 esto precarga `{1..8}` en vez de `{1..5}`: 3 pasadas más de
extracción de cadenas, una sola vez y cacheadas en disco.

### 7.4 `get_search_space` — L8408-8432

```diff
@@ class Optimizador.get_search_space
         n_total = int(getattr(self, "total_uces", 0) or 0)
         k_hi = 20 if n_total <= 0 else max(3, min(20, n_total // 10))
+
+        # ── v5: separacion minima del par de fracciones = 1 UCE ──────────
+        # mcs = max(floor, round(frac*n_emb)). Sumar exactamente 1/n_emb
+        # garantiza round(frac_t*n) >= round(frac_l*n) + 1 (round(x+1) =
+        # round(x)+1), es decir una UCE de separacion real entre C1 y C2.
+        _rv = getattr(self, "retrofitted_vectors", None)
+        n_emb = int(len(_rv)) if _rv else n_total
+        frac_eps = round(1.0 / n_emb, 5) if n_emb > 0 else 0.002
+        frac_eps = max(0.0005, frac_eps)
+
+        space: Dict[str, Any] = {
-            "min_forms_uc_1": (max(2, f1 - 4), f1 + 4),
+            "min_forms_uc_1": (max(2, f1 - 4), f1 + 4),
             "forms_gap": (2, 7),
             "tsj": (max(2, tsj - 2), tsj + 2),
             # pseudocount eliminado (BLOQUE 1: siempre 0.0)
             "cdh_target_k": (3, k_hi),
-            "swap_iterations": [1, 2, 5],
-            "min_r2_threshold": [0.03, 0.05, 0.08, 0.10],
+            # v5 (§3.4): el ganador del probe v4 esta en el minimo (1);
+            # 0 es la ablacion sin refinamiento (_partition: `if > 0`).
+            "swap_iterations": [0, 1, 2, 3, 5],
+            # v5 (§3.3): el ganador esta en el maximo (0.10).
+            "min_r2_threshold": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
             "similarity_threshold": [0.25, 0.35, 0.45, 0.55, 0.65],
-            "coref_weight": [0.0, 0.1, 0.2, 0.3, 0.4],
+            # v5 (§3.1): combinacion convexa con similarity_weight=1-w; no
+            # hay cota estructural en 0.4.
+            "coref_weight": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
             "uc_window_size": [1, 2, 3, 4, 5],
-            "coref_context_units_tight": [1, 2, 3],
-            "coref_context_units_loose": [3, 4, 5],
+            # v5 (§3.2): driver primero, dependiente derivado. El ganador del
+            # probe esta en (loose=5, tight=3), ambos bordes superiores.
+            # 6 y 8 duplican el paso para cubrir el regimen saturado
+            # (ctx >= len(doc) es un no-op).
+            "coref_context_units_loose": [3, 4, 5, 6, 8],
+            "coref_context_units_tight": {
+                "kind": "lt",
+                "other": "coref_context_units_loose",
+                "lo": 1,
+            },
             # Fraccion de n_embeddable, no absoluto — ver _resolve_hdbscan_uce_params.
             # ([hdbscan-scale-patch])
             # Grilla angostada: REII_EMB_DEBUG confirmó que con corpus chicos
             # (cientos de UCEs) la estructura real de Method C solo aparece
             # en min_cluster_size≈5-20 ...
-            "hdbscan_uce_min_cluster_frac": [0.01, 0.015, 0.02, 0.03, 0.04],
# v5: frac_loose NO se mueve (§3.5): con floor=5, 0.002 == 0.005
# para el corpus del probe (n_emb=528, ver §3.5 para el limite).
             "hdbscan_uce_min_cluster_frac_loose": [0.005, 0.01, 0.015, 0.02, 0.03],
+            "hdbscan_uce_min_cluster_frac": {
+                "kind": "gt",
+                "other": "hdbscan_uce_min_cluster_frac_loose",
+                "hi": 0.04,
+                "eps": frac_eps,
+            },
         }
+
+        # ── v5 (§5.4): congelar los inertes (rho~0, n=11). El dict admite
+        # escalares: _suggest los devuelve tal cual, sin suggest_*.
+        if bool(getattr(self.config_base, "optimize_freeze_inert", True)):
+            space["min_forms_uc_1"] = int(f1)
+            space["uc_window_size"] = int(getattr(self.uc_config, "window_size", 3))
+        return space
```

### 7.5 `_suggest` — L8434-8448 (dos pasadas)

```diff
@@ class Optimizador._suggest
     @staticmethod
     def _suggest(trial: "optuna.Trial", space: Dict[str, Any]) -> Dict[str, Any]:
         params: Dict[str, Any] = {}
+        pending: Dict[str, Any] = {}
         for k, v in space.items():
+            # v5: condicional -> segunda pasada, cuando el driver ya esta
+            # muestreado (no depende del orden de insercion del dict).
+            if isinstance(v, dict):
+                pending[k] = v
+                continue
             if isinstance(v, tuple) and len(v) == 2:
                 lo, hi = v
                 if isinstance(lo, int) and isinstance(hi, int):
                     params[k] = trial.suggest_int(k, lo, hi)
                 else:
                     params[k] = trial.suggest_float(k, float(lo), float(hi))
             elif isinstance(v, list):
                 params[k] = trial.suggest_categorical(k, v)
             else:
                 params[k] = v
+        for k, spec in pending.items():
+            params[k] = _suggest_conditional(trial, k, spec, params)
         return params
```

### 7.6 Centinelas con motivo — módulo, tras `_is_sentinel` (L7857)

```diff
@@
 def _is_sentinel(values) -> bool:
     """True si `values` es el centinela de trial inválido."""
     return tuple(values) == _SENTINEL_TRIAL


+_MAX_TB_CHARS = 2000
+_MAX_TB_LINES = 20
+
+
+def _sentinel(trial: "optuna.Trial", reason: str):
+    """Centinela dominado + motivo, para separar gates de excepciones."""
+    try:
+        trial.set_user_attr("outcome", "gate")
+        trial.set_user_attr("gate_reason", str(reason)[:120])
+    except Exception:  # nunca enmascarar el centinela por un fallo de atributos
+        logger.debug("No se pudo registrar gate_reason=%s", reason)
+    return _SENTINEL_TRIAL
+
+
+def _trial_exc_attrs(exc: BaseException) -> Dict[str, str]:
+    """user_attrs de un trial que explotó: tipo, mensaje y ultimas ~20 lineas."""
+    tb = traceback.format_exc().splitlines()[-_MAX_TB_LINES:]
+    return {
+        "error_type": type(exc).__name__,
+        "error_msg": str(exc)[:500],
+        "traceback": "\n".join(tb)[-_MAX_TB_CHARS:],
+    }
+
+
+def _make_guarded_objective(objective, printer=print):
+    """Envuelve el objetivo de Optuna para que NINGUNA excepcion se pierda.
+
+    - `Exception` -> `_SENTINEL_TRIAL` + error_type/error_msg/traceback en
+      user_attrs (el FAIL #81 del probe v4 se perdio justo por no tener esto);
+    - `KeyboardInterrupt`/`SystemExit` NO se capturan (BaseException): se
+      re-lanzan para que Ctrl-C corte de verdad;
+    - imprime una sola vez por trial.
+    """
+
+    def _guarded(trial):
+        try:
+            return objective(trial)
+        except (KeyboardInterrupt, SystemExit):
+            raise
+        except Exception as exc:  # noqa: BLE001 — ver docstring
+            attrs = _trial_exc_attrs(exc)
+            try:
+                trial.set_user_attr("outcome", "exception")
+                trial.set_user_attr("trial_failed", True)
+                for k, v in attrs.items():
+                    trial.set_user_attr(k, v)
+            except Exception:  # registrar el fallo no puede tumbar el trial
+                logger.warning(
+                    "No se pudieron guardar los attrs del trial %s",
+                    getattr(trial, "number", "?"),
+                )
+            printer(
+                f"[!] trial #{getattr(trial, 'number', '?')} fallo: "
+                f"{attrs['error_type']}: {attrs['error_msg']} "
+                f"(traceback en user_attr)"
+            )
+            return _SENTINEL_TRIAL
+
+    return _guarded
+
+
+def _sentinel_breakdown(trials) -> Tuple[int, int, List[str]]:
+    """(por_gate, por_excepcion, detalle) sobre los trials centinela."""
+    n_gate = n_exc = 0
+    detail: List[str] = []
+    for t in trials:
+        try:
+            values = t.values
+        except Exception:
+            continue
+        if values is None or not _is_sentinel(values):
+            continue
+        etype = (getattr(t, "user_attrs", {}) or {}).get("error_type")
+        if etype:
+            n_exc += 1
+            detail.append(
+                f"#{getattr(t, 'number', '?')} {etype}: "
+                f"{(getattr(t, 'user_attrs', {}) or {}).get('error_msg', '')}"
+            )
+        else:
+            n_gate += 1
+    return n_gate, n_exc, detail
+
+
 def _cdh_frac_from_params(params: Dict[str, Any], default: float = 0.10) -> float:
```

### 7.7 `_evaluate_trial`: quitar el `except` local — L8590-8594

```diff
@@ class Optimizador._evaluate_trial
-        try:
-            result = self._evaluacion_rapida(cfg, uc_cfg=uc_cfg)
-        except Exception as e:
-            logger.debug("Trial %d exception: %s", trial.number, e)
-            return _SENTINEL_TRIAL
+        # v5: sin try/except local. La excepcion sube al guard de
+        # study.optimize (_make_guarded_objective), que la convierte en
+        # centinela CON traceback. Antes se descartaba aqui y el trial
+        # quedaba indistinguible de un gate (o FAIL sin rastro, #81).
+        result = self._evaluacion_rapida(cfg, uc_cfg=uc_cfg)
```

### 7.8 Retornos de gate con motivo (12 sitios)

Todos son `return _SENTINEL_TRIAL` dentro de `_evaluate_trial`:

| Línea | Reemplazo                                                       |
| ----- | --------------------------------------------------------------- |
| 8561  | `return _sentinel(trial, "espacio: tight >= loose")`            |
| 8563  | `return _sentinel(trial, "espacio: frac <= frac_loose")`        |
| 8598  | `return _sentinel(trial, "sin primary_stable")`                 |
| 8604  | `return _sentinel(trial, "sin labels primarios")`               |
| 8610  | `return _sentinel(trial, "K fuera de [2, 20]")`                 |
| 8628  | `return _sentinel(trial, "coverage < gate")`                    |
| 8650  | `return _sentinel(trial, "abstencion (sin metodo viable)")`     |
| 8652  | `return _sentinel(trial, "ARI del ganador no finito")`          |
| 8690  | `return _sentinel(trial, "wc_only: faltan los dos pases")`      |
| 8695  | `return _sentinel(trial, "wc_only: <2 UCEs comunes")`           |
| 8700  | `return _sentinel(trial, "wc_only: <2 UCEs validas")`           |
| 8718  | `return _sentinel(trial, "Method C sin persistencia en C1/C2")` |

El de la línea 8563 pasa a ser **defensa en profundidad**: con el espacio v5 no
puede dispararse (la simulación da 0.00 %). Si dispara, es un bug del espacio.

### 7.9 `_make_sampler` — L8773-8778

```diff
@@ class Optimizador._make_sampler
         if name == "tpe":
             return TPESampler(
                 seed=seed,
                 n_startup_trials=n_startup,
                 multivariate=multivariate,
+                n_ei_candidates=int(
+                    getattr(self.config_base, "optimize_n_ei_candidates", 48)
+                ),
+                # v5: tight/frac_loose se muestrean con rango dinamico, asi
+                # que Optuna los excluye del TPE multivariado y avisa una vez
+                # por trial. Es esperado (el par es un triangulo, no una
+                # gaussiana); se silencia para no inundar el log.
+                warn_independent_sampling=False,
             )
```

### 7.10 Warm start acotado — L8949-8953

```diff
@@ class Optimizador._run_single_stage
-        for prev_trial in warm_start_trials:
+        # v5 (§5.5): el frente de etapa 1 puede traer decenas de puntos y cada
+        # uno se encola x3; sin tope, el warm start consume el presupuesto
+        # entero de la etapa 2 y TPE nunca muestrea.
+        _ws_max = int(getattr(self.config_base, "optimize_warm_start_max", 8))
+        for prev_trial in list(warm_start_trials)[:_ws_max]:
             _enqueue(dict(prev_trial.params))
```

### 7.11 `study.optimize` + resumen — L8975-8994

```diff
@@ class Optimizador._run_single_stage
         study.optimize(
-            self._evaluate_trial,
+            _make_guarded_objective(self._evaluate_trial),
             n_trials=n_trials,
             show_progress_bar=False,
         )

         # ── Resumen final ────────────────────────────────────────────────
         n_completed = sum(
             1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
         )
@@
         print(
             f"\n   >>> Estado: {n_completed} completados, "
             f"{n_pruned} pruned, {n_failed} failed"
         )
+        # v5: separar los dos centinelas (antes se confundian).
+        _n_gate, _n_exc, _exc_detail = _sentinel_breakdown(study.trials)
+        print(
+            f"=> Centinelas: {_n_gate} por gates de factibilidad · "
+            f"{_n_exc} por excepción"
+        )
+        if _n_exc:
+            print(
+                f"[!] {_n_exc} trial(s) con excepcion; el detalle esta en "
+                f"user_attr 'traceback' (comparative_results.json)"
+            )
+            for _line in _exc_detail[:5]:
+                print(f"[X] {_line[:160]}")
```

### 7.12 `best_params` con los congelados — L8188-8198

```diff
@@ class Optimizador.save_pareto_front › def _bp_params
         def _bp_params(t) -> Dict[str, Any]:
             """params del trial + la fracción CDH derivada.

             Se emite también ``min_cluster_size_cdh`` para que los lectores
             legacy (``rerun_classification``, sondas) no tengan que conocer
             la parametrización por K.
             """
             out = dict(t.params)
             if "cdh_target_k" in out:
                 out["min_cluster_size_cdh"] = round(_cdh_frac_from_params(out), 4)
+            # v5: los parametros congelados no pasan por suggest_*, asi que no
+            # estan en trial.params. Se emiten explicitos para que
+            # _apply_best_params replique exactamente lo que corrio el trial.
+            for _k, _v in self._search_space_cached.items():
+                if not isinstance(_v, (list, tuple, dict)):
+                    out.setdefault(_k, _v)
             return out
```

---

## 8. Riesgos y límites

1. **`frac <= frac_loose` deja de ser detectado por el gate** (nunca se viola),
   por lo que el gate L8563 queda como aserción. Si algún día el espacio cambia,
   el síntoma será un centinela `gate_reason="espacio: frac <= frac_loose"` en el
   resumen, no un crash.
2. **Este rediseño no reduce los 7 centinelas sustantivos** del probe v4
   (cobertura/K/abstención). Esos dependen de los umbrales de calidad
   (`optimize_coverage_gate*`) y de `cdh_target_k`, no del muestreo.
3. **`min_r2_threshold` hasta 0.20 puede fabricar centinelas K=1** (§3.3). Es el
   único borde con coste de factibilidad; aplicar 0.15 primero.
4. **La degeneración por piso (2.46 %)** no desaparece sin resolver el par sobre
   `mcs` (§3.5).
5. **`n_emb` desconocido**: si `retrofitted_vectors` es `None` (modo degradado),
   `eps` se calcula con `total_uces`, que es una cota superior; el `eps` real
   resulta más pequeño y la separación de 1 UCE no está garantizada, aunque
   `frac > frac_loose` sí (la cota es estricta en cualquier caso).
6. **No se midió el coste de pared** de las 3 pasadas extra de coref (§7.3) ni
   del `coref_context_units_loose=8`. Se sabe que es una sola vez y cacheado en
   `COREF_CHAINS_CACHE_PATH`, pero el número de minutos hay que medirlo en el
   próximo probe.
7. **El probe v5 debe llamarse `reii_probe_v5`**; relanzar `reii_probe_v4` anexa
   trials al estudio existente y contamina el frente.

## 9. Reproducibilidad

```
python scripts/playground/v5_space_sim.py            # 5000 draws, 2.7 s
python scripts/playground/v5_space_sim.py -n 20000   # margen
pytest tests/test_v5_space.py                        # importa main_workflow (~min)
```

Semilla fija (`--seed 20251028`), `n_emb=528`, sin escrituras, exit 0.
