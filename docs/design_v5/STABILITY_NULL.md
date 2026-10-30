# Estabilidad por remuestreo y nulo: que senal discrimina y que no

Estado: **especificacion cerrada, no aplicada**. `docs/design_v5/SELECT_V4.md`
§4 decide `null_mode="gate"` por defecto. El banco
`scripts/playground/stability_engine_bench.py` (34.3 s, exit 0, seed 42) mide
que **ese default es inerte**: el nulo por permutacion de etiquetas da
`p = 0.005` sobre ruido puro. Este documento supera §4, no lo borra.

Alcance de escritura: **solo documentacion y tests**. Los diffs de la seccion 7
los aplica el usuario a mano sobre `src/**` (scope de escritura prohibido para
este agente). No se toca `HARDENING.md` ni `FINAL_PATCHSET.md`.

---

## CAMBIO RESPECTO A SELECT_V4.md

`SELECT_V4.md` §4 concluye:

> **Decision: `gate` por defecto.** ... Con varios candidatos viables, `gate` y
> `rank` casi siempre coinciden.

Eso sigue siendo cierto **condicionalmente**: con un nulo de etiquetas el gate
solo puede quitar candidatos que ya caen por otra razon, y no quita ninguno
sobre ruido. El banco lo demuestra con 12/12 filas con `p = 0.005`, incluidas
las tres de `synth_null` (gaussiana isotropica sin estructura):

| fila del banco | backend     | ARI bootstrap | IC95         | `null_p` | veredicto |
| -------------- | ----------- | ------------- | ------------ | -------- | --------- |
| `synth_null`   | `ward`      | 0.093         | [0.08, 0.12] | 0.005    | SIN_SENAL |
| `synth_null`   | `kmeans`    | 0.112         | [0.05, 0.15] | 0.005    | SIN_SENAL |
| `synth_null`   | `dense_cdh` | 0.097         | [0.08, 0.13] | 0.005    | SIN_SENAL |

Un gate que no puede descartar nada en el unico escenario donde descartar
importa (ruido) no es un gate: es una columna. Por lo tanto:

- **default `null_mode = "diagnostic"`**: el nulo se calcula, se persiste y se
  imprime como **columna de control**; no descarta ni ordena.
- se conservan `"gate" | "rank" | "off"`; `"gate"` solo tiene sentido con un
  **nulo de datos** (romper los datos, no las etiquetas), que hoy no existe.
  Debe seguir testeandose con `null_p` inyectado.
- nuevo **piso de viabilidad** independiente del nulo:
  `bootstrap_ari` finito y `mean < 0.30` ⇒ **NO VIABLE**. `nan` nunca descarta.

### Changelog

| version | cambio                                                                                     |
| ------- | ------------------------------------------------------------------------------------------ |
| v5.0    | nulo de etiquetas medido como inerte (12/12 con `p=0.005`); `gate`->`diagnostic`           |
| v5.0    | piso `stability_mean_min = 0.30` sobre el ARI bootstrap                                    |
| v5.0    | contrato de datos con `nan` como "no disponible" (nunca descarta)                          |
| v5.1    | `coverage_stable` definido por alineacion Hungarian por replica                            |
| v5.1    | `bootstrap_ari_ci_low/high` (bloque) vs `bootstrap_ci_lo/hi` (`MethodMetrics`) documentado |

---

## 1. Nulo de etiquetas vs bootstrap: por que no son intercambiables

Los dos procedimientos parecen primos y miden cosas opuestas.

| procedimiento      | que se perturba  | que se reajusta | que responde                               |
| ------------------ | ---------------- | --------------- | ------------------------------------------ |
| `permutation_null` | las ETIQUETAS    | nada            | "la particion separa X mejor que el azar"  |
| `bootstrap_ari`    | las FILAS (UCEs) | el clusterer    | "la particion se reproduce al remuestrear" |

### 1.1 El sesgo de seleccion del nulo de etiquetas

El estadistico observado se calcula sobre una particion **ajustada a X**. El
algoritmo eligio esa particion, entre todas, maximizando (directa o
indirectamente) la separacion. Compararla contra etiquetas aleatorias no
contrasta "hay estructura" sino **"el algoritmo optimiza"**.

Medido (TABLA 1 del banco, `r2_dense`, `n_perm=200`):

- `synth_null` / `ward`: observado 0.089, nulo 0.013 +- 0.001, **`z = 57.1`**,
  `p = 0.005`. `p = 1/(1+n_perm)` es el minimo alcanzable.
- `synth_null` / `kmeans`: `z = 83.7`, `p = 0.005`.
- `synth_null` / `dense_cdh`: `z = 74.0`, `p = 0.005`.

Con silhouette precomputada el sesgo se repite, mas debil: `blobs_s26/ward`
`z = 44.0`, `p = 0.010`; `synth_null/ward` `z = 10.5`, `p = 0.010`. Sobre ruido
puro el estadistico **rechaza el nulo**.

El nulo correcto para "hay estructura" fija el algoritmo y rompe los datos
(misma media y covarianza, logica de _gap statistic_). Ese nulo no existe hoy
en el repo. Mientras no exista, `null_p` no puede filtrar.

### 1.2 Por que el bootstrap si discrimina

`bootstrap_ari` submuestrea UCEs y **reajusta** el clusterer con el mismo K.
El ARI se mide contra la particion de referencia del ajuste completo. La
pregunta es "si veo 80 % del corpus, recupero la misma particion?".

Un nulo de etiquetas no puede responder eso, y un ARI tight/loose tampoco:
el pase tight/loose reutiliza los mismos datos y, en `ward`/`kmeans`, no tiene
grados de libertad (`BACKEND_C.md` §3: `ARI = 1.0` por construccion). Por eso
`stability = bootstrap_ari` cuando es finito y `stability_kind` lo declara.

### 1.3 Consecuencia operativa

| senal             | uso permitido                                 | uso prohibido                            |
| ----------------- | --------------------------------------------- | ---------------------------------------- |
| `bootstrap_ari`   | viabilidad (piso 0.30) y orden (`-stability`) | nada                                     |
| `null_p`          | **columna de control**, auditoria, paper      | filtrar u ordenar por defecto            |
| `ari` tight/loose | `fallback` cuando no hay bootstrap            | comparar con bootstrap sin declarar kind |

---

## 2. Evidencia medida

Salida literal de `python scripts/playground/stability_engine_bench.py`
(34.3 s, exit 0; `seed=42`, `B=5` en la matriz, `n_perm=200`, `r2_dense`).
`obs` es el R2 denso de la particion; `IC95` es la banda empirica del ARI
bootstrap (`ci_low`, `ci_high`), no el IC de la media.

| dataset      | backend     | estadistico | obs   | nulo mean+-sd  | z     | p     | ARI mean | IC95        |
| ------------ | ----------- | ----------- | ----- | -------------- | ----- | ----- | -------- | ----------- |
| `blobs_s26`  | `ward`      | `r2_dense`  | 0.477 | 0.013 +- 0.002 | 245.3 | 0.005 | 0.835    | [0.77,0.89] |
| `blobs_s26`  | `kmeans`    | `r2_dense`  | 0.479 | 0.013 +- 0.002 | 246.8 | 0.005 | 0.839    | [0.81,0.86] |
| `blobs_s26`  | `dense_cdh` | `r2_dense`  | 0.501 | 0.013 +- 0.002 | 275.9 | 0.005 | 0.968    | [0.94,0.98] |
| `blobs_s12`  | `ward`      | `r2_dense`  | 0.305 | 0.013 +- 0.002 | 185.9 | 0.005 | 0.833    | [0.78,0.88] |
| `blobs_s12`  | `kmeans`    | `r2_dense`  | 0.328 | 0.013 +- 0.002 | 206.5 | 0.005 | 0.960    | [0.94,0.98] |
| `blobs_s12`  | `dense_cdh` | `r2_dense`  | 0.295 | 0.013 +- 0.001 | 193.2 | 0.005 | 0.781    | [0.74,0.81] |
| `synth_null` | `ward`      | `r2_dense`  | 0.089 | 0.013 +- 0.001 | 57.1  | 0.005 | 0.093    | [0.08,0.12] |
| `synth_null` | `kmeans`    | `r2_dense`  | 0.123 | 0.013 +- 0.001 | 83.7  | 0.005 | 0.112    | [0.05,0.15] |
| `synth_null` | `dense_cdh` | `r2_dense`  | 0.108 | 0.013 +- 0.001 | 74.0  | 0.005 | 0.097    | [0.08,0.13] |
| `real_ctx`   | `ward`      | `r2_dense`  | 0.243 | 0.026 +- 0.003 | 68.6  | 0.005 | 0.416    | [0.30,0.52] |
| `real_ctx`   | `kmeans`    | `r2_dense`  | 0.272 | 0.026 +- 0.003 | 84.3  | 0.005 | 0.595    | [0.46,0.74] |
| `real_ctx`   | `dense_cdh` | `r2_dense`  | 0.233 | 0.026 +- 0.003 | 62.3  | 0.005 | 0.559    | [0.43,0.71] |

Nulo con silhouette precomputada (`n_perm=100`, reutiliza el bootstrap):

| dataset      | backend | estadistico  | obs   | nulo mean+-sd   | z    | p     | ARI mean | IC95        |
| ------------ | ------- | ------------ | ----- | --------------- | ---- | ----- | -------- | ----------- |
| `blobs_s26`  | `ward`  | `silhouette` | 0.408 | -0.054 +- 0.010 | 44.0 | 0.010 | 0.835    | [0.77,0.89] |
| `synth_null` | `ward`  | `silhouette` | 0.008 | -0.020 +- 0.003 | 10.5 | 0.010 | 0.093    | [0.08,0.12] |

Lectura:

1. **`p` es constante en 0.005 (o 0.010) en las 14 filas.** No ordena nada y no
   distingue estructura de ruido. Como columna de control sirve; como gate, no.
2. **El ARI bootstrap separa las tres familias sin solapamiento**:
   ruido 0.093-0.112, corpus real 0.416-0.595, estructura sintetica 0.781-0.968.
3. El `IC95` tambien separa por abajo: ruido `lo <= 0.15`, real `lo 0.30-0.46`,
   estructura `lo >= 0.74`. Es la base del umbral 0.50 (§3).
4. La regla propuesta acierta **12/12** filas del banco.

### 2.1 Advertencia sobre `real_ctx`

`real_ctx` es **INDICATIVO**, no una validacion del pipeline:

- el bench lee `data/workflow_data.json` en solo lectura; en esta corrida hay
  **235/600 UCEs con embedding** y `n = 235`, `n_eff = 188`. El encargo cita
  un corpus de `n ~ 528`: son corridas distintas del mismo fichero regenerado.
  No mezclar cifras entre ambas.
- el `embedding` del JSON es **384-d contextual** (SentenceTransformer), NO el
  fastText-retrofit 300-d que usa Method C. La geometria no es la del pipeline.
- por lo tanto `real_ctx` solo dice "un corpus real cae entre ruido y
  estructura", que es exactamente `FRAGIL` (0.42-0.60, `ci_low` 0.30-0.46).
  **No** autoriza a mover el umbral 0.50 ni a declarar que el corpus tiene K.

---

## 3. Regla de veredicto y piso

Regla medida, identica a `table_verdict` del banco
(`stability_engine_bench.py`, L731-795):

```
ESTABLE    <=> ci_low >= 0.50
SIN_SENAL  <=> mean   <  0.30
FRAGIL     el resto
NO_DISPONIBLE  <=> mean no finito (nan)
```

Con `B=10` la banda es mas ancha que con `B=20` (IC95 de ancho 0.144 vs 0.176
en `blobs_s26/kmeans`), asi que `ci_low >= 0.50` es conservador con B=10: si
una particion es fragil, el IC ancho tiende a bajarla, no a subirla.

### 3.1 Justificacion de los umbrales

| umbral           | valor | por que ese valor                                                                                                      |
| ---------------- | ----- | ---------------------------------------------------------------------------------------------------------------------- |
| `ci_low` ESTABLE | 0.50  | el maximo del ruido es 0.15 y el minimo de estructura 0.74; 0.50 queda a 0.35 del ruido y a 0.24 de la estructura      |
| `mean` SIN_SENAL | 0.30  | ruido 0.09-0.11; el siguiente valor medido es 0.42 (real). 0.30 deja 0.19 de margen sobre el ruido y 0.12 bajo el real |
| `B`              | 10    | desvio 0.0022 frente a `B=20`; `B=5` ya desvia 0.0151                                                                  |
| `n_perm`         | 50    | `p=0.020` en vez de 0.005; irrelevante porque el nulo no filtra (§6)                                                   |

No son umbrales "magicos": son los puntos medios de los dos huecos medidos.
Si aparece un corpus entre 0.30 y 0.50, el veredicto es `FRAGIL` y la regla
**elige igual** (no se abstiene): el piso solo corta cuando el remuestreo no
recupera la particion.

### 3.2 El piso es independiente del nulo

```
NO VIABLE  <=> bootstrap_ari es finito Y bootstrap_ari < 0.30
```

- `nan` **nunca** descarta: significa "no se midio", no "no hay estructura".
  Es la misma politica que `null_p = nan` en `SELECT_V4.md` §4.4.
- el piso se evalua **despues** de dominancia y cobertura y **antes** del gate
  del nulo: si un metodo ya cae por un hecho duro, el motivo impreso debe ser
  el hecho duro (`SELECT_V4.md` §2.3).
- el piso **no** sube el 0.70 de dominancia ni sustituye a `stability` en el
  orden: `stability = bootstrap_ari` ya era el criterio principal en v4.

---

## 4. Contrato de datos

Bloque serializable que produce `bootstrap_stability()` +
`null_partition_test()` y que viaja al payload. `nan` = no disponible;
`nan` nunca descarta ni ordena.

| clave                   | tipo  | origen                           | si falta                 |
| ----------------------- | ----- | -------------------------------- | ------------------------ |
| `bootstrap_ari`         | float | media del ARI de replicas        | `nan`                    |
| `bootstrap_ari_ci_low`  | float | cuantil `alpha/2`                | `nan`                    |
| `bootstrap_ari_ci_high` | float | cuantil `1-alpha/2`              | `nan`                    |
| `bootstrap_n_boot`      | int   | replicas solicitadas             | `0`                      |
| `null_p`                | float | `(1 + #{nulo >= obs}) / (n+1)`   | `nan`                    |
| `null_stat`             | float | estadistico observado            | `nan`                    |
| `null_n_perm`           | int   | permutaciones solicitadas        | `0`                      |
| `coverage_stable`       | float | UCEs estables / `n_total`        | `nan` (cae a `coverage`) |
| `stability_kind`        | str   | `"bootstrap"` / `"perturbation"` | `"perturbation"`         |
| `stability_verdict`     | str   | §3                               | `"NO_DISPONIBLE"`        |

Claves auxiliares que se persisten para el auditor y el paper (no deciden):
`ok`, `reason`, `null_z`, `null_mean`, `null_std`, `bootstrap_std`.

### 4.1 `coverage_stable`

Fraccion de UCEs del corpus con etiqueta **estable en el remuestreo**:

1. en cada replica se ajusta `predict_fn` sobre la submuestra;
2. se alinea la prediccion con la referencia por asignacion Hungarian (K <= 20,
   coste despreciable) sobre las UCEs comunes con etiqueta `>= 0`;
3. una UCE es estable si, en al menos `stable_threshold = 0.75` de las replicas
   en que fue muestreada, su etiqueta alineada coincide con la de referencia;
4. `coverage_stable = #estables / n_total`.

Se divide por `n_total` (no por asignadas) para que sea comparable entre
metodos y para que un backend que asigna todo no obtenga `1.0` por
construccion (`BACKEND_C.md` §3).

### 4.2 Discrepancia de nombres (deliberada, documentada)

`SELECT_V4.md` §7.4 declara en `MethodMetrics` los campos
`bootstrap_ci_lo` / `bootstrap_ci_hi`. El bloque de este documento usa
`bootstrap_ari_ci_low` / `bootstrap_ari_ci_high`. **No es un error**: el
productor (`quality/metrics.py`) habla el contrato del bloque; el consumidor
(`MethodMetrics`) conserva los nombres cortos que ya fija el doc de v4. El
mapeo es 1:1 y ocurre en un solo sitio (`main_workflow.py`, §5). Renombrar los
campos de `MethodMetrics` rompería `9_selection_audit.py`, que los lee del
payload por nombre.

### 4.3 `stability_verdict` como propiedad derivada

En `MethodMetrics` no se guarda el veredicto: se **deriva** de
`bootstrap_ari` y `bootstrap_ci_lo` (propiedad, §7b). Un campo almacenado
podria quedar desincronizado de las cifras tras una edicion manual del JSON.

---

## 5. Integracion en `main_workflow.py`

Anclas reales verificadas en el archivo actual (no hay que buscarlas):

| linea(s)         | elemento                                                                                       |
| ---------------- | ---------------------------------------------------------------------------------------------- |
| 81-86            | `from reii.method_selection import MethodMetrics, Selection, persist_selection, select_method` |
| 998-1006         | campos `method_selection_*` de `Config`                                                        |
| 5541-5567        | `def _triple_stability_verdicts(self, ...)` firma y defaults                                   |
| 5659-5662        | `_stats`, `_noise`, `_ari`, `_pert` (dicts por metodo)                                         |
| 5665-5674        | `def _max_share(_m)`                                                                           |
| 5676-5692        | construccion de `MethodMetrics(...)` (una por metodo A/B/C)                                    |
| 5693-5709        | `selection = select_method(metrics, ...)`                                                      |
| 5943-5949        | `return {"primary_stable_uces": ..., "summary": ..., "selection": ...}`                        |
| 6234-6253        | llamada real a `_triple_stability_verdicts(...)` desde `run()`                                 |
| 6052, 6103, 6139 | `_hungarian_stability_direct` de A, B y C (donde existen X y labels)                           |

Punto unico que este documento **no puede cerrar sin `src/`**: construir
`spaces` (§7c) requiere la matriz densa y el `predict_fn` de cada metodo. Las
lineas 6052/6103/6139 son los tres sitios donde ambas cosas existen. Hasta que
el llamador rellene `self._stability_spaces`, el bloque es `{}` y todo cae a
`nan`: el comportamiento observable es **identico al actual**.

Ayudante de medicion (por metodo):

```
spaces[m] = (X, uce_ids, cluster_of_uce, predict_fn)
```

- `X` matriz densa `n x d` del espacio del metodo;
- `uce_ids` ids alineados con las filas de `X`;
- `cluster_of_uce` `uce_id -> etiqueta` de la particion de referencia (`-1` = sin asignar);
- `predict_fn` `predict_fn(X_sub) -> labels` con K y `random_state` fijos.

`bootstrap_stability` y `null_partition_test` se importan **dentro** del
ayudante: `main_workflow.py` no adquiere dependencia de import nueva y el
tiempo de arranque no cambia.

---

## 6. Presupuesto

Medido en esta corrida (TABLA 2 y PRESUPUESTO):

| backend     | B=20, `n_perm=200` (s) | nota                                  |
| ----------- | ---------------------- | ------------------------------------- |
| `ward`      | 0.96                   | el mas barato                         |
| `kmeans`    | 1.73                   | el unico que se acerca al presupuesto |
| `dense_cdh` | 0.73                   |                                       |

Curva de B (`blobs_s26/kmeans`): `B=5` 0.874 s, mean 0.839; `B=10` 1.240 s,
mean 0.856; `B=20` 1.839 s, mean 0.854. `|mean(B=10) - mean(B=20)| = 0.0022`;
`B=5` desvia 0.0151. **Presupuesto: `B=10`.**

Curva de `n_perm` (`blobs_s26/kmeans`, `r2_dense`): 50 -> `p=0.020`, 0.124 s;
100 -> `p=0.010`, 0.192 s; 200 -> `p=0.005`, 0.196 s. **Presupuesto:
`n_perm=50`**, solo como control: el p-valor no filtra, asi que su resolucion
da igual mas alla de que sea legible en el log.

Determinismo: 6/6 combinaciones (2 datasets x 3 backends) producen etiquetas
identicas con 1 y 4 hilos BLAS, `ARI = 1.0000`. Robustez: `n<30`, una sola
clase, `K>n`, vectores cero y guarda de memoria devuelven `nan`/`None` sin
excepcion (ROBUSTEZ del banco).

---

## 7. Diffs exactos

Se aplican a mano. Los bloques se numeran (a)-(d) y son independientes entre si
salvo que (b) y (c) asumen el v4 de `SELECT_V4.md` ya aplicado.

### (a) `src/reii/quality/metrics.py`: `bootstrap_stability()` y `null_partition_test()`

El fichero tiene 650 lineas; ambas funciones se anaden al final. Hacen falta
`Any` y `Callable` en el import de `typing`.

```diff
--- a/src/reii/quality/metrics.py
+++ b/src/reii/quality/metrics.py
@@
-from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
+from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
@@
     if n >= 2:
         with warnings.catch_warnings():
             warnings.simplefilter("ignore")
             out["nmi"] = float(normalized_mutual_info_score(x, y))
             out["ari"] = float(adjusted_rand_score(x, y))
     return out
+
+
+# ── Estabilidad por remuestreo y nulo de etiquetas (STABILITY_NULL.md) ──────
+
+#: `ci_low` minimo para declarar ESTABLE. Medido: ruido <= 0.15, estructura
+#: >= 0.74 (STABILITY_NULL.md §2). No subir sin nueva evidencia.
+STABILITY_CI_LOW_MIN = 0.50
+
+#: `bootstrap_ari` medio por debajo del cual no hay senal recuperable.
+#: Debe coincidir con `method_selection.stability_mean_min`.
+STABILITY_MEAN_MIN = 0.30
+
+
+def _r2_partition(X, labels) -> float:
+    """R2 denso `1 - SSE/TSS` de `labels` sobre `X`. NaN si no es calculable.
+
+    Es el estadistico por defecto del nulo porque es **el que el clusterer
+    optimiza**: por eso el nulo de etiquetas no puede rechazar ruido
+    (STABILITY_NULL.md §1.1). Se expone tambien como utilidad de auditoria.
+    """
+    try:
+        A = np.asarray(X, dtype=float)
+        y = np.asarray(labels).astype(int)
+    except (TypeError, ValueError):
+        return float("nan")
+    if A.ndim != 2 or y.ndim != 1 or A.shape[0] != y.shape[0]:
+        return float("nan")
+    keep = y >= 0
+    if int(keep.sum()) < 3 or np.unique(y[keep]).size < 2:
+        return float("nan")
+    A, y = A[keep], y[keep]
+    mu = A.mean(axis=0, keepdims=True)
+    tss = float(np.sum((A - mu) ** 2))
+    if not np.isfinite(tss) or tss <= 0.0:
+        return float("nan")
+    sse = 0.0
+    for c in np.unique(y):
+        Ac = A[y == c]
+        sse += float(np.sum((Ac - Ac.mean(axis=0, keepdims=True)) ** 2))
+    value = 1.0 - sse / tss
+    return float(value) if np.isfinite(value) else float("nan")
+
+
+def _stability_verdict(mean: float, ci_low: float) -> str:
+    """Veredicto de §3: ESTABLE | SIN_SENAL | FRAGIL | NO_DISPONIBLE."""
+    if not math.isfinite(float(mean)):
+        return "NO_DISPONIBLE"
+    if float(mean) < STABILITY_MEAN_MIN:
+        return "SIN_SENAL"
+    if math.isfinite(float(ci_low)) and float(ci_low) >= STABILITY_CI_LOW_MIN:
+        return "ESTABLE"
+    return "FRAGIL"
+
+
+def bootstrap_stability(
+    X,
+    labels,
+    *,
+    predict_fn: Optional[Callable] = None,
+    n_boot: int = 10,
+    frac: float = 0.8,
+    replace: bool = False,
+    seed: int = 42,
+    min_n: int = 30,
+    alpha: float = 0.05,
+    stable_threshold: float = 0.75,
+) -> Dict[str, Any]:
+    """Estabilidad por submuestreo de UCEs + reajuste con el mismo K.
+
+    Contrato de `predict_fn`: `predict_fn(X_sub) -> labels` de longitud
+    `len(X_sub)`. El K, el piso y el `random_state` los cierra el llamador, de
+    modo que la funcion no se acopla a ward, kmeans ni dense_cdh.
+
+    En cada replica se submuuestrea `ceil(frac*n)` filas (sin reemplazo por
+    defecto), se reajusta el clusterer y se mide `ARI(pred, labels)` sobre las
+    filas comunes con etiqueta `>= 0`. `labels` es la particion de REFERENCIA
+    (la del ajuste completo).
+
+    Devuelve el bloque de STABILITY_NULL.md §4:
+    `bootstrap_ari`, `bootstrap_ari_ci_low`, `bootstrap_ari_ci_high`,
+    `bootstrap_n_boot`, `coverage_stable`, `stability_kind`,
+    `stability_verdict` (+ `ok`, `reason`, `bootstrap_std`, `n`, `n_failed`).
+
+    Precondiciones (todas devuelven `ok=False` + `nan`, NUNCA excepcion):
+    `X` denso 2-D, `labels` 1-D de la misma longitud, `n >= min_n`,
+    `predict_fn` callable, `n_boot >= 1`, `0 < frac <= 1`.
+    """
+    out: Dict[str, Any] = {
+        "ok": False,
+        "reason": "",
+        "bootstrap_ari": float("nan"),
+        "bootstrap_ari_ci_low": float("nan"),
+        "bootstrap_ari_ci_high": float("nan"),
+        "bootstrap_n_boot": int(n_boot) if isinstance(n_boot, int) else 0,
+        "bootstrap_std": float("nan"),
+        "coverage_stable": float("nan"),
+        "stability_kind": "",
+        "stability_verdict": "NO_DISPONIBLE",
+        "n": 0,
+        "n_failed": 0,
+    }
+    try:
+        A = np.asarray(X, dtype=float)
+        y = np.asarray(labels).astype(int)
+    except (TypeError, ValueError):
+        out["reason"] = "X_o_labels_invalidos"
+        return out
+    if A.ndim != 2 or y.ndim != 1 or A.shape[0] != y.shape[0]:
+        out["reason"] = "forma_invalida_o_longitudes_distintas"
+        return out
+    n = int(A.shape[0])
+    out["n"] = n
+    if n < int(min_n):
+        out["reason"] = f"n={n}_menor_que_min_n={int(min_n)}"
+        return out
+    if not callable(predict_fn):
+        out["reason"] = "predict_fn_no_es_callable"
+        return out
+    if int(n_boot) < 1:
+        out["reason"] = "n_boot<1"
+        return out
+    frac = float(frac)
+    if not (0.0 < frac <= 1.0):
+        out["reason"] = "frac_fuera_de_(0,1]"
+        return out
+    if not np.isfinite(A).all():
+        out["reason"] = "X_con_NaN_o_inf"
+        return out
+
+    n_sub = int(min(n, max(3, math.ceil(frac * n))))
+    rng = np.random.default_rng(int(seed))
+    ari_vals: List[float] = []
+    hits = np.zeros(n, dtype=int)
+    seen = np.zeros(n, dtype=int)
+    fails = 0
+
+    with warnings.catch_warnings():
+        warnings.simplefilter("ignore")
+        for _ in range(int(n_boot)):
+            idx = rng.choice(n, size=n_sub, replace=bool(replace))
+            if replace:
+                idx = np.unique(idx)
+            if idx.size < 3:
+                fails += 1
+                continue
+            try:
+                pred = np.asarray(predict_fn(A[idx])).ravel().astype(int)
+            except Exception:
+                fails += 1
+                continue
+            if pred.size != idx.size:
+                fails += 1
+                continue
+            ys = y[idx]
+            keep = ys >= 0
+            if int(keep.sum()) < 2:
+                fails += 1
+                continue
+            if np.unique(ys[keep]).size < 2 or np.unique(pred[keep]).size < 2:
+                fails += 1
+                continue
+            value = float(adjusted_rand_score(ys[keep], pred[keep]))
+            if not np.isfinite(value):
+                fails += 1
+                continue
+            ari_vals.append(value)
+
+            # Alineacion Hungarian para poder contar UCEs estables.
+            ca, cb = np.unique(ys[keep]), np.unique(pred[keep])
+            ia = {int(v): i for i, v in enumerate(ca)}
+            ib = {int(v): j for j, v in enumerate(cb)}
+            ov = np.zeros((ca.size, cb.size))
+            for u, v in zip(ys[keep], pred[keep]):
+                ov[ia[int(u)], ib[int(v)]] += 1
+            rows, cols = linear_sum_assignment(-ov)
+            mapped = {int(cb[j]): int(ca[i]) for i, j in zip(rows, cols)}
+            for pos, uce in enumerate(idx):
+                if ys[pos] < 0:
+                    continue
+                seen[uce] += 1
+                if mapped.get(int(pred[pos])) == int(ys[pos]):
+                    hits[uce] += 1
+
+    out["n_failed"] = int(fails)
+    if not ari_vals:
+        out["reason"] = "todas_las_replicas_fallaron"
+        return out
+    arr = np.asarray(ari_vals, dtype=float)
+    out["bootstrap_ari"] = float(arr.mean())
+    out["bootstrap_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
+    out["bootstrap_ari_ci_low"] = float(np.quantile(arr, alpha / 2.0))
+    out["bootstrap_ari_ci_high"] = float(np.quantile(arr, 1.0 - alpha / 2.0))
+    assigned = y >= 0
+    if assigned.any():
+        need = np.ceil(float(stable_threshold) * np.maximum(seen, 1))
+        ok_rows = (seen > 0) & (hits >= need) & assigned
+        out["coverage_stable"] = float(int(ok_rows.sum()) / float(n))
+    out["stability_kind"] = "bootstrap"
+    out["stability_verdict"] = _stability_verdict(
+        out["bootstrap_ari"], out["bootstrap_ari_ci_low"]
+    )
+    out["ok"] = True
+    return out
+
+
+def null_partition_test(
+    X,
+    labels,
+    *,
+    stat_fn: Optional[Callable] = None,
+    n_perm: int = 50,
+    seed: int = 42,
+    min_n: int = 30,
+) -> Dict[str, Any]:
+    """Nulo por permutacion de ETIQUETAS. `stat_fn(X, labels) -> float`.
+
+    Por defecto `_r2_partition`. AVISO (STABILITY_NULL.md §1.1): el estadistico
+    observado procede de una particion AJUSTADA a X, asi que este nulo responde
+    "mejor que etiquetas aleatorias", no "hay estructura". Sobre ruido puro
+    devuelve `p = 1/(1+n_perm)` por sesgo de seleccion. **No es un gate.**
+
+    Devuelve `null_p` (una cola superior, correccion +1), `null_stat`,
+    `null_n_perm`, `null_z`, `null_mean`, `null_std`. Nunca lanza.
+    """
+    out: Dict[str, Any] = {
+        "ok": False,
+        "reason": "",
+        "null_p": float("nan"),
+        "null_stat": float("nan"),
+        "null_n_perm": int(n_perm) if isinstance(n_perm, int) else 0,
+        "null_z": float("nan"),
+        "null_mean": float("nan"),
+        "null_std": float("nan"),
+    }
+    try:
+        A = np.asarray(X, dtype=float)
+        y = np.asarray(labels).astype(int)
+    except (TypeError, ValueError):
+        out["reason"] = "X_o_labels_invalidos"
+        return out
+    if A.ndim != 2 or y.ndim != 1 or A.shape[0] != y.shape[0]:
+        out["reason"] = "forma_invalida_o_longitudes_distintas"
+        return out
+    if int(A.shape[0]) < int(min_n):
+        out["reason"] = f"n={int(A.shape[0])}_menor_que_min_n={int(min_n)}"
+        return out
+    if int(n_perm) < 1:
+        out["reason"] = "n_perm<1"
+        return out
+    stat = stat_fn if callable(stat_fn) else _r2_partition
+    try:
+        observed = float(stat(A, y))
+    except Exception:
+        observed = float("nan")
+    if not np.isfinite(observed):
+        out["reason"] = "estadistico_observado_NaN"
+        return out
+    out["null_stat"] = observed
+    rng = np.random.default_rng(int(seed))
+    nulls: List[float] = []
+    with warnings.catch_warnings():
+        warnings.simplefilter("ignore")
+        for _ in range(int(n_perm)):
+            try:
+                value = float(stat(A, rng.permutation(y)))
+            except Exception:
+                continue
+            if np.isfinite(value):
+                nulls.append(value)
+    if not nulls:
+        out["reason"] = "todos_los_nulos_NaN"
+        return out
+    null = np.asarray(nulls, dtype=float)
+    out["null_mean"] = float(null.mean())
+    out["null_std"] = float(null.std(ddof=1)) if null.size > 1 else 0.0
+    if out["null_std"] > 0.0:
+        out["null_z"] = float((observed - out["null_mean"]) / out["null_std"])
+    out["null_p"] = float((1.0 + float(np.sum(null >= observed))) / (null.size + 1.0))
+    out["ok"] = True
+    return out
```

### (b) `src/reii/method_selection.py`: default `null_mode`, piso y textos

Cuatro ediciones: constante, `Selection`, `_feasibility`, `select_method` y la
linea de control de `explain_lines`. Los nombres de `MethodMetrics`
(`bootstrap_ci_lo/hi`) se quedan como los fija `SELECT_V4.md` §7.4.

```diff
--- a/src/reii/method_selection.py
+++ b/src/reii/method_selection.py
@@
 RULE_VERSION = "select-v3"
+
+#: Piso de viabilidad sobre el ARI bootstrap. Medido: ruido 0.09-0.11, corpus
+#: real 0.45-0.57, estructura 0.83-0.96 (docs/design_v5/STABILITY_NULL.md §3).
+#: Debe coincidir con `quality.metrics.STABILITY_MEAN_MIN`.
+stability_mean_min = 0.30
@@
-    null_mode: str = "gate"
+    #: "diagnostic" (default) = columna de control; "gate" solo con nulo de
+    #: datos; "rank" y "off" se conservan. Ver STABILITY_NULL.md §7b.
+    null_mode: str = "diagnostic"
     null_alpha: float = 0.05
@@
-#: Modos de uso del contraste contra el nulo.
-_NULL_MODES: Tuple[str, ...] = ("gate", "rank", "off")
+#: Modos de uso del contraste contra el nulo. "diagnostic" persiste e imprime
+#: el p-valor pero no descarta ni ordena (STABILITY_NULL.md §1.3).
+_NULL_MODES: Tuple[str, ...] = ("gate", "rank", "off", "diagnostic")
@@
-    null_mode: str = "gate",
+    null_mode: str = "diagnostic",
     null_alpha: float = 0.05,
     reject_tautological: bool = True,
     diagnostic_backends: Sequence[str] = _DIAGNOSTIC_BACKENDS,
@@
     if (
         reject_tautological
         and metric.stability_kind == "perturbation"
         and str(metric.backend) in _TAUTOLOGICAL_BACKENDS
     ):
         return (
             f"estabilidad tautologica: backend {metric.backend} no tiene grados"
             " de libertad en el pase tight/loose (ARI=1.0 por construccion);"
             " falta bootstrap_ari"
         )
+    boot = float(getattr(metric, "bootstrap_ari", float("nan")))
+    if math.isfinite(boot) and boot < float(stability_mean_min):
+        return (
+            f"sin senal de estructura: estabilidad media {boot:.3f} < piso "
+            f"{float(stability_mean_min):.2f} (el remuestreo no recupera la"
+            " particion)"
+        )
     if null_mode == "gate":
         p = float(metric.null_p)
         if math.isfinite(p) and p > float(null_alpha):
             return (
-                f"sin contraste contra el nulo: p={p:.3f} > alpha="
+                f"sin contraste contra el nulo de DATOS: p={p:.3f} > alpha="
                 f"{float(null_alpha):.2f} (las clases no separan mas que"
                 " etiquetas permutadas)"
             )
     return None
@@
-    null_mode: str = "gate",
+    null_mode: str = "diagnostic",
     null_alpha: float = 0.05,
     reject_tautological: bool = True,
     diagnostic_backends: Sequence[str] = _DIAGNOSTIC_BACKENDS,
@@
         out.append(
             f"     clase mayor={float(_win_row.get('max_class_share') or 0.0):.2f} "
             f"de las asignadas (gate: >{self.max_class_share_max:.2f} = "
             "particion degenerada)"
         )
+        out.append(
+            f"     control: bootstrap={_fmt(_win_row.get('bootstrap_ari'))}"
+            f" IC95=[{_fmt(_win_row.get('bootstrap_ci_lo'))},"
+            f"{_fmt(_win_row.get('bootstrap_ci_hi'))}]"
+            f" nulo p={_fmt(_win_row.get('null_p'), '.3f')}"
+            f" (modo {self.null_mode}"
+            f"{' - filtra' if self.null_mode == 'gate' else ' - no filtra'})"
+        )
         if self.ambiguous and self.runner_up:
```

### (c) `src/reii/main_workflow.py`: llamada y volcado al payload

Tres puntos: campos de `Config`, el ayudante y la firma de
`_triple_stability_verdicts`, y el consumo en `MethodMetrics` + `select_method`

- el `return`. Todo es **inerte** mientras `self._stability_spaces` sea `None`.

```diff
--- a/src/reii/main_workflow.py
+++ b/src/reii/main_workflow.py
@@ -998,6 +998,14 @@
     method_selection_margin: float = 0.02
     method_selection_coverage_floor: float = 0.15
+    #: Modo del contraste contra el nulo: "diagnostic" (default) lo imprime
+    #: como control y no filtra. Ver docs/design_v5/STABILITY_NULL.md.
+    method_selection_null_mode: str = "diagnostic"
+    #: Presupuesto del motor de estabilidad (STABILITY_NULL.md §6).
+    method_selection_stability_boot: int = 10
+    method_selection_stability_perm: int = 50
+    method_selection_stability_min_n: int = 30
     method_selection_k_bounds: Tuple[int, int] = (3, 20)
@@
-    def _triple_stability_verdicts(
+    def _stability_blocks(
+        self,
+        spaces: Optional[Dict[str, Tuple[Any, List[str], Dict[str, int], Any]]],
+        *,
+        n_boot: int = 10,
+        n_perm: int = 50,
+        seed: int = 42,
+        min_n: int = 30,
+    ) -> Dict[str, Dict]:
+        """``{metodo: bloque}`` con el ARI bootstrap y el nulo de cada metodo.
+
+        ``spaces[m] = (X, uce_ids, cluster_of_uce, predict_fn)``. Un metodo sin
+        espacio produce ``{}``: sus claves caen a ``nan`` y la regla no lo
+        descarta por falta de dato (STABILITY_NULL.md §4). Nunca lanza.
+        """
+        from reii.quality.metrics import bootstrap_stability, null_partition_test
+
+        out: Dict[str, Dict] = {}
+        for method, spec in (spaces or {}).items():
+            try:
+                X, uce_ids, cluster_of, predict_fn = spec
+                A = np.asarray(X, dtype=float)
+                if A.ndim != 2 or A.shape[0] != len(uce_ids):
+                    out[method] = {}
+                    continue
+                y = np.asarray(
+                    [int(cluster_of.get(u, -1)) for u in uce_ids], dtype=int
+                )
+                if int((y >= 0).sum()) < int(min_n) or not callable(predict_fn):
+                    out[method] = {}
+                    continue
+                boot = bootstrap_stability(
+                    A,
+                    y,
+                    predict_fn=predict_fn,
+                    n_boot=int(n_boot),
+                    seed=int(seed),
+                    min_n=int(min_n),
+                )
+                nul = null_partition_test(
+                    A, y, n_perm=int(n_perm), seed=int(seed), min_n=int(min_n)
+                )
+                block = dict(boot)
+                block.update(
+                    null_p=nul.get("null_p", float("nan")),
+                    null_stat=nul.get("null_stat", float("nan")),
+                    null_n_perm=int(nul.get("null_n_perm", n_perm)),
+                    null_z=nul.get("null_z", float("nan")),
+                )
+                out[method] = block
+            except Exception:
+                out[method] = {}
+        return out
+
+    def _triple_stability_verdicts(
         self,
         uces_por_doc: List[List[UCE]],
@@
         # tamanos de clase de la particion CRUDA de cada metodo (sin -1).
         # Solo los usa el modo 'select' para el gate de no dominancia.
         partition_sizes: Optional[Dict[str, List[int]]] = None,
+        # Salida de ``_stability_blocks`` (STABILITY_NULL.md §5). ``None`` o
+        # ``{}`` => todo cae a nan y la seleccion se comporta como antes.
+        stability_blocks: Optional[Dict[str, Dict]] = None,
     ) -> Dict:
@@
         all_uids = {uce.id for doc in uces_por_doc for uce in doc}
         n_total = len(all_uids)
+        _stab = stability_blocks or {}
         selection: Optional[Selection] = None
@@
             def _max_share(_m: str) -> float:
@@
                 sizes = [int(s) for s in (_raw_sizes.get(_m) or []) if int(s) > 0]
                 tot = sum(sizes)
                 return (max(sizes) / tot) if tot > 0 else 0.0
+
+            def _sb(_m: str) -> Dict:
+                """Bloque de estabilidad del metodo, o {} si no se midio."""
+                return _stab.get(_m) or {}

             metrics = [
                 MethodMetrics(
                     method=_m,
@@
                     median_class_size=float(_stats[_m][1]),
                     max_class_share=_max_share(_m),
+                    bootstrap_ari=float(_sb(_m).get("bootstrap_ari", float("nan"))),
+                    bootstrap_ci_lo=float(
+                        _sb(_m).get("bootstrap_ari_ci_low", float("nan"))
+                    ),
+                    bootstrap_ci_hi=float(
+                        _sb(_m).get("bootstrap_ari_ci_high", float("nan"))
+                    ),
+                    coverage_stable=float(_sb(_m).get("coverage_stable", float("nan"))),
+                    null_p=float(_sb(_m).get("null_p", float("nan"))),
+                    null_z=float(_sb(_m).get("null_z", float("nan"))),
                 )
                 for _m in ("A", "B", "C")
             ]
             selection = select_method(
                 metrics,
@@
                 max_class_share_max=float(
                     getattr(self.config, "method_selection_max_class_share", 0.90)
                 ),
+                null_mode=str(
+                    getattr(self.config, "method_selection_null_mode", "diagnostic")
+                ),
             )
@@
             "primary_stable_uces": primary_stable_uces,
             "summary": summary,
             "effective_mode": _mode,
             "selection": selection,
+            "stability_blocks": _stab,
             "selected_method": selection.winner if selection is not None else None,
@@
             "C": _partition_sizes(res_emb1["uce_to_label"] if res_emb1 else None),
         }

+        stability_blocks = self._stability_blocks(
+            getattr(self, "_stability_spaces", None),
+            n_boot=int(
+                getattr(self.config, "method_selection_stability_boot", 10)
+            ),
+            n_perm=int(
+                getattr(self.config, "method_selection_stability_perm", 50)
+            ),
+            min_n=int(
+                getattr(self.config, "method_selection_stability_min_n", 30)
+            ),
+            seed=int(getattr(self.config, "random_state", 42) or 42),
+        )
         report = self._triple_stability_verdicts(
@@
             partition_sizes=_raw_sizes,
+            stability_blocks=stability_blocks,
         )
```

### (d) `docs/design_v5/SELECT_V4.md` §4: banner de enmienda

```diff
--- a/docs/design_v5/SELECT_V4.md
+++ b/docs/design_v5/SELECT_V4.md
@@
-**Decision: `gate` por defecto.** No es una preferencia estetica; sale de que
-`null_p` (en el contrato) es un p-valor **contra etiquetas permutadas**, y ese
-nulo tiene un sesgo conocido y del signo que nos perjudica si se usa para
-ordenar.
+> **AMENDMENT (STABILITY_NULL.md v5): el default es `diagnostic`, no `gate`.**
+> `STABILITY_NULL.md` §2 mide `p = 0.005` sobre ruido puro en 12/12 filas del
+> banco: con un nulo de ETIQUETAS el gate es inerte y no puede producir
+> abstencion sobre ruido. `null_mode="diagnostic"` calcula, persiste e imprime
+> el p-valor como **columna de control** sin descartar ni ordenar. `"gate"`
+> queda disponible y solo tiene sentido con un **nulo de datos** (romper los
+> datos, no las etiquetas), que hoy no existe; se sigue testeando con `null_p`
+> inyectado. El razonamiento de abajo se conserva porque sigue siendo correcto
+> sobre POR QUE `rank` esta prohibido: un ranking sesgado hacia C es peor que
+> un gate vacuo, y `diagnostic` es el gate vacuo sin el coste de abstencion.
+> Leer los literales `null_mode: str = "gate"` de §7.5 y §7.8/§7.9 como
+> `"diagnostic"`, y anadir `"diagnostic"` a `_NULL_MODES`.
```

---

## 8. Que NO se toca

1. **El nulo no se vuelve gate.** `diagnostic` es el default; `gate` sigue
   existiendo solo para un nulo de datos inexistente hoy.
2. **No se sube el 0.50.** El `ci_low >= 0.50` es el umbral medido; `real_ctx`
   no es evidencia suficiente para moverlo (§2.1) y moverlo hacia abajo
   convertiria `FRAGIL` en `ESTABLE` sin justificacion.
3. **No se toca el 0.70 de dominancia** (`c = a(1-nu) > 0.70`), ni su escala
   sobre el corpus (`GATE_NOISE.md`). El piso de 0.30 es una condicion
   **adicional** de viabilidad, no un sustituto.
4. **No se toca `HARDENING.md` ni `FINAL_PATCHSET.md`** (otros agentes).
5. **No se toca `src/**`** desde este documento: los diffs son texto.
6. **No se toca `ari` tight/loose** ni `stability_kind`: el fallback a
   perturbacion sigue siendo el mismo y sigue marcando `criteria_conflict`
   cuando los kinds difieren.
7. **No se toca el contrato de `Selection`** salvo el default de `null_mode`:
   `winner_stability`, `winner_null_p`, etc. se quedan.
8. `coverage_stable` **no** sustituye a `coverage` en `usable_coverage` (nombre
   publico que consume `9_selection_audit.py`); solo alimenta
   `coverage_effective`, como ya fijaba `SELECT_V4.md` §2.1.

---

## Referencias

- `scripts/playground/stability_engine_bench.py`: `bootstrap_ari` L267,
  `permutation_null` L429, `stability_report` L540, `make_predict` L644,
  `table_verdict` L731, `build_datasets` L825, `run_blas` L1079.
- `docs/design_v5/SELECT_V4.md` §0, §2, §4, §7.1, §7.4, §7.5, §7.8, §7.9.
- `docs/design_v5/BACKEND_C.md` §2 (diagnostico), §3 (ARI tautologico).
- `docs/design_v5/GATE_NOISE.md` §1 (`c = a(1-nu)`).
- `docs/design_v5/METRICS_PAPER.md` §1.4 (nulo de etiquetas y `C_in`).
- `src/reii/quality/metrics.py` (650 lineas; se anade al final).
- `src/reii/method_selection.py` L86, L102-137, L140-168, L207, L274-309,
  L312-423.
- `src/reii/main_workflow.py` L81-86, L998-1006, L5541-5567, L5659-5674,
  L5676-5692, L5693-5709, L5943-5949, L6234-6253.
