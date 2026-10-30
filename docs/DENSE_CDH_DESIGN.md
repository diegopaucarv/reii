# DenseCDH: CDH dicotomica recursiva en el espacio denso

Estado: diseno previo a implementacion. No se modifica codigo de produccion.
Alcance: definir un backend alternativo para Method C que reemplace el HDBSCAN
plano por una CDH dicotomica recursiva sobre embeddings, con K controlable y
sin ruido masivo (`-1`).
Restriccion temporal: se activa despues de la corrida "probe v4" (rejilla-K,
Method B y gate de dominancia). El default no cambia el comportamiento actual.

El paper anuncia en `docs/paper.md:65` un "algoritmo de division/agrupamiento
jerarquico denso (reemplazo de la CDH clasica)". Hoy ese lugar lo ocupa un
HDBSCAN plano. Este documento especifica el reemplazo real.

---

## 1. Anclajes verificados en el codigo actual

Todo el diseno se apoya en estos puntos exactos. No se propone reescribirlos.

| Pieza                 | Ubicacion                                           | Que hace y por que importa                                                                                                                                                                                                              |
| --------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_primer_factor`      | `src/reii/main_workflow.py:3747-3829`               | Eje 1-D por promediado reciproco sobre la matriz sparse (loop `3814-3827`). El warm-start por PCA de centroides esta en `3785-3804`. Devuelve `x` en `3829`.                                                                            |
| `_corte_optimo`       | `src/reii/main_workflow.py:3831-3871`               | Ordena el eje, calcula `SS_between` (`3848-3855`) y elige el corte que maximiza R2 (`3865-3867`). Restringe `k` a `[min_side, n-min_side]` en `3859-3863`; etiqueta en `3869-3870`. Es el optimo exacto de 2-means en 1-D.              |
| `_intercambio`        | `src/reii/main_workflow.py:3873-4014`               | Refinamiento por swap chi2; banda de ambiguedad y desempate por centroide en `3913-3927`; actualizacion incremental `3994-4009`.                                                                                                        |
| `_permutation_pvalue` | `src/reii/main_workflow.py:4020-4065`               | Test empirico de permutacion con `n1` fijo (`4037-4038`), semilla `default_rng(random_state)` (`4052`), p-valor `4064-4065`.                                                                                                            |
| `_es_significativo`   | `src/reii/main_workflow.py:4067-4089`               | Piso de R2 (`4078-4079`), `n_perm_cdh` (`4081`), fallback determinista (`4083-4086`) y permutacion (`4088-4089`).                                                                                                                       |
| `_partition`          | `src/reii/main_workflow.py:4095-4172`               | Recursion. Piso duro `n_min_split=2*floor` (`4103-4107`), corte (`4122-4123`), gate (`4125-4134`), swap y **reversion** si el swap rompe el piso (`4136-4147`), colapso si queda una sola clase (`4149-4158`), recursion (`4160-4164`). |
| `clasificar`          | `src/reii/main_workflow.py:3703-3745`               | Resuelve `floor` (`3710`), imprime `[CDH]` (`3712-3715`), inicializa `-1` (`3727`), rellena hojas (`3728`) e imprime tamanos (`3732-3744`).                                                                                             |
| `_fill_labels`        | `src/reii/main_workflow.py:3695-3701`               | Asigna el label de cada hoja a todos sus indices.                                                                                                                                                                                       |
| Method C (entrada)    | `src/reii/main_workflow.py:5089-5214`               | Centroides de UCE (`5122`), rechazo de OOV `>50%` (`5124-5131`), PCA a 30 dims (`5149-5163`), HDBSCAN (`5176-5184`), ruido (`5186-5194`), `uce_to_label` solo no-ruido (`5200-5202`) y contrato de salida (`5204-5214`).                |
| Method C (dos pases)  | `src/reii/main_workflow.py:6108-6142`               | Resuelve floors tight/loose (`6117-6119`), construye `res_emb1`/`res_emb2` (`6124-6133`) y calcula ARI C1-C2 (`6136-6140`).                                                                                                             |
| Seleccion             | `src/reii/method_selection.py:274-309` y `312-423`  | Gates de viabilidad y ranking. Ruido `290-294`, dominancia `295-301`, coverage util `302-308`, orden por ARI y coverage `372-379`.                                                                                                      |
| Contrato de metricas  | `src/reii/method_selection.py:102-137`              | `MethodMetrics`: `k`, `ari`, `coverage`, `noise_frac`, `tiny_frac`, `max_class_share`.                                                                                                                                                  |
| Persistencia          | `src/reii/main_workflow.py:5949-5973` y `9959-9965` | Snapshot + JSONL de la seleccion; el orquestador la guarda en `db.data["method_selection"]`.                                                                                                                                            |

### 1.1 Como se obtiene hoy el eje 1-D sparse

`_primer_factor` construye una matriz de masa y ejecuta promediado reciproco
(`3814-3827`): `y = M.T x`, `z = y / col_sums`, `w = M z`, `x_new = w / row_sums`,
centrado por masas de fila y normalizado. El warm-start (`3785-3804`) siembra
`x` con la primera componente principal de los centroides retrofitted, pero el
punto fijo final es el mismo. El eje no es PCA: es la direccion de maxima
asociacion entre filas y columnas de la tabla de contingencia.

### 1.2 Como se corta y que invariante se conserva

`_corte_optimo` recibe solo un vector `coord` (`3831-3832`). Esto es clave: es
agnostico a la representacion. El corte factible se restringe a
`k in [min_side, n-min_side]` (`3859-3863`), de modo que nunca produce un
1-vs-(n-1). En 1-D, maximizar `SS_between` equivale a minimizar la suma de
cuadrados intra-clase: es el optimo exacto de k-means con k=2. `var_inter` es
`SS_between / SS_total` (`3865-3867`), es decir un R2 del corte.

### 1.3 Cuando se detiene la recursion

`_partition` declara hoja terminal si `n < 2*floor` o si se alcanza
`max_depth_cdh` (`4103-4107`). Si el corte no pasa `_es_significativo`
(`4125-4134`), tambien es hoja. `_es_significativo` exige
`var_obs >= min_r2_threshold` y un p-valor de permutacion `< alpha`
(`4078-4089`). Con `n_perm_cdh <= 0` cae a `n * var_obs >= chi2_threshold_small`
(`4083-4086`).

### 1.4 Recursion, piso y reversion del swap

El piso duro se fija antes de recursar: `n_min_split = 2*floor` (`4106`). Tras
el swap chi2, si algun hijo queda por debajo de `floor`, se revierte a la
particion del promediado reciproco (`4144-4147`). Si el swap colapsa a una sola
clase, el nodo es hoja (`4149-4158`). Los hijos se construyen con
`indices[labels==0]` e `indices[labels==1]` (`4160-4164`).

### 1.5 Etiquetado de hojas

`clasificar` inicializa `labels` a `-1` (`3727`), pero `_fill_labels` recorre el
arbol y asigna el label de cada hoja a todos sus indices (`3695-3701`). Como
todo indice termina en una hoja, la CDH nunca emite `-1`; el `-1` es solo el
valor centinela previo al relleno. La impresion `[CDH]` reporta `n`, floor,
`K_max~`, y tamanos min/max/mediana (`3712-3715`, `3732-3744`).

### 1.6 Ruta actual de Method C

`_clasificar_umbral_embedding` (`5089-5214`) hace:

1. Aplana todas las UCEs y normaliza IDs (`5115-5120`).
2. Construye centroides densos con `_compute_uce_centroids_from_retro`
   (`3572-3622`, llamado en `5122`). Filas todo-cero significan UCE sin
   cobertura (`3585`, `3612-3613`).
3. Rechaza si mas del 50% son cero (`5124-5131`).
4. PCA global a `target_dims = max(2, min(30, n//10, d))` (`5149-5163`), con
   `random_state=42` fijo (`5154`). Para el corpus real: `528//10 = 52`, luego
   `target_dims = 30`.
5. HDBSCAN con `min_cluster_size` y `min_samples` (`5176-5184`).
6. Descarta si hay menos de 2 clusters reales (`5196-5197`).
7. Devuelve `uce_to_label` solo para `labels >= 0` (`5200-5202`) y el dict con
   `labels_array`, `uce_ids`, `embedding_matrix`, `cluster_persistence`,
   `n_clusters`, `n_noise` (`5204-5214`).

Los floors tight/loose salen de `_resolve_hdbscan_uce_params` (`5055-5087`):
`frac * n` con piso absoluto 5 (`5073-5079`). Para `n=528`, tight `0.08 -> 42`,
loose `0.05 -> 26`; `min_samples` se deriva con razon `0.25` (`5081-5086`),
aproximadamente 10 y 6.

### 1.7 Regla de seleccion y gates

`_feasibility` (`274-309`) exige `K in [3,20]` (`287-289`), ruido
`<= 0.50` (`290-294`), clase mayor `<= 0.90` de las **asignadas**
(`295-301`) y coverage util `>= 0.15` (`302-308`). El ranking es
`(-ARI, -coverage_util, orden A<B<C)` (`372-379`). La abstitencion ocurre si
ningun metodo es viable (`354-370`).

---

## 2. Decision de corte denso

### 2.1 Opciones evaluadas

**Opcion (a): biseccion por PC1 local del bloque + `_corte_optimo`.**

- Se parte de la matriz ya reducida por la PCA global a 30 dims, la misma que
  consume HDBSCAN (`5149-5163`). Esto preserva la comparabilidad de entrada
  entre backends y acota el coste.
- En cada nodo: centrar el bloque, calcular la direccion de mayor varianza
  (PC1 local) por covarianza + `eigh`, proyectar y llamar a `_corte_optimo`
  con `min_side=floor`. En 1-D ese corte es el optimo exacto de 2-means
  restringido (`3859-3867`).
- Refinamiento: un "swap denso" determinista (asignacion al centroide mas
  cercano en 30-d, iterado como Lloyd) con reversion si rompe el piso,
  espejo de `4136-4147`. No usa RNG ni reinicios.

**Opcion (b): k-means k=2 con reinicios + criterio de significancia.**

- Optimiza SSE en el espacio completo, es invariante a rotaciones y no depende
  de que PC1 alinee con los clusters.
- Pero introduce semillas, numero de reinicios y un nuevo hiperparametro;
  k-means puede violar `min_side` y hay que repararlo; y su resultado no es
  reutilizable por `_corte_optimo`.

### 2.2 Numeros del corpus real (528 UCEs, 300-d, 29 docs)

La PCA global deja 30 dims. El piso de DenseCDH se deriva del mismo
`min_cluster_size_cdh` que usa Method A, y ese valor viene de
`cdh_target_k -> frac = 1/(2K)` (`7860-7885`, en particular `7879-7883`).
Formula de A: `floor = max(5, int(n * frac))` (`3710`).

| K objetivo | frac = 1/(2K) | floor (n=528) | max hojas = n//floor | K cabe en gate [3,20] |
| ---------: | ------------: | ------------: | -------------------: | :-------------------: |
|          3 |        0.1667 |            88 |                    6 |          si           |
|          4 |        0.1250 |            66 |                    8 |          si           |
|          5 |        0.1000 |            52 |                   10 |          si           |
|          6 |        0.0833 |            44 |                   12 |          si           |
|          8 |        0.0625 |            33 |                   16 |          si           |
|         10 |        0.0500 |            26 |                   20 |          si           |
|         12 |        0.0417 |            22 |                   24 |   no, requiere cap    |
|         15 |        0.0333 |            17 |                   31 |   no, requiere cap    |
|         20 |        0.0250 |            13 |                   40 |   no, requiere cap    |

Conclusion numerica: para `cdh_target_k` en `[3,10]`, el piso solo ya garantiza
`K <= 20` en este corpus. Para `[11,20]` hace falta el tope opcional
`dense_cdh_max_leaves=20` (o el gate de seleccion descartara C por K). Esto no
es un defecto del diseno: es exactamente el mismo compromiso que ya tiene A.

### 2.3 Coste por nodo y por arbol

Con `n=528`, `d=30`:

| Etapa                                      | Complejidad        |  Orden de flops | Cuando              |
| ------------------------------------------ | ------------------ | --------------: | ------------------- |
| PCA global 300 -> 30                       | `O(n * d_in^2)`    |          ~4.8e7 | una vez por corrida |
| PC1 local (cov + eigh)                     | `O(n*d^2 + d^3)`   |   ~5e5 por nodo | cada nodo           |
| Proyeccion + sort + corte                  | `O(n*d + n log n)` |   ~2e4 por nodo | cada nodo           |
| Swap denso (2 iteraciones)                 | `O(iters*n*d*k)`   |   ~6e4 por nodo | cada nodo           |
| Gate de permutacion                        | `O(n_perm*n*d)`    | ~1.6e6 por nodo | cada nodo           |
| Arbol completo (<=16 hojas)                | suma por nivel     |           < 3e7 | por pase            |
| Alternativa k-means (5 reinicios, 20 iter) | `O(r*it*n*d*k)`    | ~3.2e6 por nodo | por nodo            |

La opcion (a) es entre 5x y 20x mas barata que (b) y no depende de semillas.
Con `n=528` ambas son triviales en tiempo absoluto; la diferencia importa
porque Method C se ejecuta dentro de cada trial del optimizador (ver
`8529-8533` y `8408-8418`), no solo en la corrida final.

### 2.4 Determinismo y sensibilidad a semillas

- **Opcion (a)**: la PC1 se obtiene por `eigh` de una matriz simetrica; no hay
  RNG. El unico grado de libertad es el signo del autovector. Se canoniza
  exigiendo que la entrada de mayor valor absoluto sea positiva. La particion
  es invariante al signo; el canon solo hace reproducible la etiqueta 0/1.
- **Opcion (b)**: cada reinicio parte de una semilla distinta; dos corridas con
  la misma semilla pueden coincidir, pero el resultado depende de la
  inicializacion y del numero de reinicios. Para un corte binario, el
  beneficio de multiples reinicios es marginal frente al coste de un nuevo
  hiperparametro.
- **Degeneracion de PC1**: si `lambda1` y `lambda2` son casi iguales, PC1 no
  esta bien definida (subespacio degenerado). Fallback determinista: usar la
  primera coordenada de la PCA global y registrar
  `[DENSE-CDH] PC1 degenerada`. No se rompe el determinismo.
- **Semilla del gate**: las permutaciones usan
  `default_rng(random_state + clave_estable(indices))`, donde la clave es un
  CRC32 de los indices ordenados. Misma semilla y mismo bloque -> mismo
  p-valor. Cambiar `random_state` solo puede cambiar el resultado si un corte
  esta exactamente en el borde de `alpha`; los labels no dependen de la semilla.

### 2.5 Decision

Se elige **(a) PC1 local + `_corte_optimo` + swap denso determinista**.

Razones:

1. **Reutiliza el invariante exacto de A**: `_corte_optimo` solo necesita un
   vector 1-D (`3831-3832`), y su restriccion `min_side` (`3859-3863`) es la
   misma que protege el piso. No se reimplementa el corte.
2. **Determinismo sin semillas**: no hay reinicios; el unico ajuste es el signo,
   canonizado.
3. **Coste acotado**: PC1 local en 30-d es barato y reutiliza la PCA global que
   ya existe (`5149-5163`).
4. **El refinamiento denso recupera la ventaja de k-means**: asignar al
   centroide mas cercano desde el corte PC1 es exactamente una iteracion de
   Lloyd; iterarlo con reversion al piso reproduce la logica de
   `_intercambio` + `4144-4147` sin chi2 ni semillas.
5. **La debilidad de PC1 se mitiga y se hace observable**: si PC1 no separa,
   el gate de significancia rechaza el corte y el nodo queda hoja; no se fuerza
   una particion espuria. La alternativa (b) no elimina esa debilidad, solo la
   cambia por sensibilidad a inicializacion.

No se implementa k-means con reinicios en v1. Queda como extension experimental
(`dense_cdh_axis="pc1_pc2"` podria evaluar PC1 y PC2 y quedarse con el corte de
mayor R2 denso), pero no es el default.

### 2.6 Metrica de significancia que reemplaza a `_es_significativo`

Se reemplaza la permutacion sobre el eje 1-D por una **permutacion sobre el R2
denso del corte**, manteniendo el mismo espiritu:

- `r2_1d = var_obs` devuelto por `_corte_optimo` (`3865-3867`).
- `r2_denso = 1 - SSE/TSS` en las 30 dims, con los labels del corte.
- Gate: `r2_1d >= min_r2_threshold` **y** p-valor de permutacion de `r2_denso`
  `< dense_cdh_alpha` (default 0.05).
- Si `n_perm_cdh <= 0`, fallback `n * r2_denso >= chi2_threshold_small`,
  espejo de `4083-4086`.

La permutacion usa `n1` fijo y reasigna membresia, igual que
`_permutation_pvalue` (`4037-4038`, `4052-4064`). Limitacion conocida: el corte
fue elegido sobre los datos, asi que el null condicional es algo optimista; la
implementacion de A tiene la misma limitacion. Para tests de calibracion se
puede activar `dense_cdh_perm_mode="full"`, que re-ejecuta PC1 + corte en cada
permutacion; no sera el default por coste.

Por que **no** se usa:

- **Silhouette como gate**: cuesta `O(n^2 d)` por nodo (en el root, ~8.4e6
  operaciones), su umbral no es invariante a escala ni a dimension, y en
  embeddings L2-normalizados de 30-d suele quedar comprimida (0.05-0.20). Se
  calcula solo como diagnostico en root y hojas, no decide.
- **Varianza explicada por PC1 sola**: mide el eje, no el corte. Un eje de alta
  varianza puede no tener gap. Por eso se exige ademas `r2_1d` del corte.
- **Gap statistic**: requiere un modelo nulo (uniforme o gaussiano rotado),
  cuesta `O(B*n*d)` con `B~10` y anade supuestos. El test de permutacion ya
  calibra contra el corpus real y es mas barato.

### 2.7 Pseudocodigo de diseno (no ejecutable)

```
dense_partition(indices, X, depth):
    n = len(indices)
    if n < 2*floor or depth >= max_depth:
        return leaf(indices)
    B = X[indices]
    nz = rows_nonzero(B)
    v = pc1(B[nz])                     # eigh, signo canonico
    if pc1_degenerada: v = e1_global
    coord = (B - mean(B[nz])) @ v
    r2_1d, labels = _corte_optimo(coord, min_side=floor)
    r2_d = dense_r2(B, labels)
    if not (r2_1d >= min_r2 and perm_p(r2_d) < alpha):
        return leaf(indices)
    labels2 = dense_swap(B, labels, swap_iterations)
    if min(counts(labels2)) < floor:
        labels2 = labels               # reversion, espejo de 4144-4147
    if len(unique(labels2)) < 2:
        return leaf(indices)
    return node(dense_partition(indices[labels2==0], X, depth+1),
                dense_partition(indices[labels2==1], X, depth+1))
```

---

## 3. Integracion como Method C

### 3.1 Flag de activacion

Propuesta en `Config`, junto al bloque de Method C (`src/reii/main_workflow.py:1009-1038`):

| Campo                    | Tipo              | Default         | Efecto                                                                 |
| ------------------------ | ----------------- | --------------- | ---------------------------------------------------------------------- |
| `method_c_backend`       | `str`             | `"hdbscan"`     | `"hdbscan"` conserva v4; `"dense_cdh"` activa DenseCDH.                |
| `dense_cdh_loose_factor` | `float`           | `0.75`          | Floor del pase C2 = `floor_tight * factor`.                            |
| `dense_cdh_alpha`        | `float`           | `0.05`          | Alpha del pase C1; C2 usa `min(0.10, 2*alpha)`.                        |
| `dense_cdh_min_r2`       | `Optional[float]` | `None`          | `None` -> usa `min_r2_threshold`.                                      |
| `dense_cdh_max_leaves`   | `Optional[int]`   | `None`          | `None` -> sin tope (comparable con A). Un entero activa consolidacion. |
| `dense_cdh_axis`         | `str`             | `"pc1"`         | Reservado; v1 solo `"pc1"`.                                            |
| `dense_cdh_perm_mode`    | `str`             | `"conditional"` | Reservado; `"full"` solo para tests.                                   |

Validacion: si `method_c_backend` no esta en `{"hdbscan","dense_cdh"}`, lanzar
`ValueError` al inicio del pase C, igual que `_resolve_hdbscan_uce_params`
valida `variant` (`5064-5065`).

Default `"hdbscan"`: con el flag ausente o en su default, `run()` ejecuta
exactamente el camino de `6108-6142` y no se toca ningun resultado de v4.

### 3.2 Punto de despacho

En `run()`, dentro de `if _do_emb:` (`6109-6113`), se ramifica antes de resolver
los parametros de HDBSCAN:

- `method_c_backend == "hdbscan"`: camino actual `6117-6133`, sin cambios.
- `method_c_backend == "dense_cdh"`: un helper nuevo
  `_run_dense_cdh_passes(...)` que:
  1. construye la matriz con `_compute_uce_centroids_from_retro` (`5122`);
  2. aplica la misma PCA global (`5149-5163`), una sola vez;
  3. llama dos veces a `DenseCDH` con floors tight/loose;
  4. devuelve dos dicts con el mismo contrato que `_clasificar_umbral_embedding`
     (`5204-5214`).

Para no duplicar la preparacion, se extrae un helper
`_prepare_method_c_matrix(uces_por_doc, retrofitted_vectors)` que devuelve
`(mat, uce_ids, zero_rows)`. HDBSCAN y DenseCDH lo comparten. Esto es un
refactor pequeno y localizado; no altera el resultado de HDBSCAN.

`DenseCDH` propuesto en un modulo nuevo `src/reii/dense_cdh.py`. Reutiliza
`CDHNode`, `_next_leaf`, `_fill_labels` y, sobre todo, `_corte_optimo`. Como
`_corte_optimo` solo depende de `coord` (`3831-3832`), puede heredarse o
extraerse a una funcion libre sin tocar el camino sparse.

### 3.3 Contrato de salida (identico a Method C actual)

DenseCDH debe devolver las mismas claves que `5204-5214`:

| Clave                 | Valor en DenseCDH                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------------------- |
| `method`              | `"embedding"`                                                                                                  |
| `backend`             | `"dense_cdh"` (clave nueva, aditiva)                                                                           |
| `min_cluster_size`    | floor usado en ese pase                                                                                        |
| `uce_to_label`        | **todas** las UCEs, `{id: label}` con `label >= 0`                                                             |
| `labels_array`        | `np.ndarray` int, longitud `n`, `min >= 0`                                                                     |
| `uce_ids`             | mismo orden que `all_uces` (`5115-5120`)                                                                       |
| `embedding_matrix`    | matriz 30-d usada                                                                                              |
| `cluster_persistence` | `[]` (HDBSCAN tiene persistencia; DenseCDH no)                                                                 |
| `n_clusters`          | numero de hojas                                                                                                |
| `n_noise`             | `0`                                                                                                            |
| `dense_cdh`           | diagnostico aditivo: `r2_1d`, `r2_denso`, `p_value`, `floor`, `alpha`, `n_zero_rows`, `perturbation_effective` |

`cluster_persistence` debe existir aunque sea lista vacia porque el orquestador
la persiste en `6277-6278`. Las claves nuevas son aditivas y no rompen a los
consumidores de `uce_to_label` (`6136`, `6169`, `6179`, `6213`, `6231`).

### 3.4 Pases C1 (tight) y C2 (loose) y ARI intra-metodo

Para comparabilidad con A, el floor tight es exactamente el de A:

- `floor_t = max(5, int(n_emb * config.min_cluster_size_cdh))`, igual que
  `3710`. Como `min_cluster_size_cdh` se deriva de `cdh_target_k` con
  `frac = 1/(2K)` (`7860-7885`), "mismo `cdh_target_k` -> mismo `min_leaf`".
- `floor_l = max(5, round(floor_t * config.dense_cdh_loose_factor))`.
- C1: `min_r2 = config.dense_cdh_min_r2 or config.min_r2_threshold`,
  `alpha = config.dense_cdh_alpha`.
- C2: `min_r2 = 0.75 * min_r2_t`, `alpha = min(0.10, 2*alpha_t)`.
- Ambos: `max_depth = config.max_depth_cdh`, `swap_iterations` de `Config`,
  `n_perm = config.n_perm_cdh`, misma matriz PCA.

La pareja C1/C2 se pasa a `_hungarian_stability_direct` exactamente donde hoy
se pasan `res_emb1`/`res_emb2` (`6136-6140`). El ARI y el conjunto estable
salen del mismo codigo (`5273-5313`), sin cambios en `method_selection.py`.

Riesgo de ARI trivial: si el corpus tiene estructura muy fuerte, C1 y C2 pueden
producir la misma particion y ARI=1. Se registra
`dense_cdh.perturbation_effective=False` y se imprime un aviso. No se altera la
regla `select-v3` en v4; el diagnostico queda para la auditoria.

### 3.5 Control de K

Hay tres niveles, en orden:

1. **Piso**: `floor` derivado de `cdh_target_k` limita `K <= n/floor = 2K`.
2. **Profundidad**: `max_depth_cdh` (`4107`) limita el arbol a `2^10` hojas.
3. **Tope opcional**: `dense_cdh_max_leaves`. Si no es `None`, tras construir
   el arbol se consolida de forma determinista: mientras `K > tope`, se colapsa
   el padre de la hoja mas pequena (la hoja y su hermano se convierten en una
   sola hoja). La consolidacion solo aumenta tamanos, asi que preserva el piso
   y el invariante "sin `-1`". Elige el padre de la hoja mas pequena para
   minimizar la perdida de granularidad; los empates se rompen por el menor
   label de hoja, de modo que es reproducible.

Default `None` para no divergir de A. En sondas que quieran K exacto, se pasa
`dense_cdh_max_leaves = cdh_target_k`.

DenseCDH **no fabrica splits** para alcanzar K: si los datos solo sostienen 2
clases, devuelve 2 hojas y el gate de seleccion lo descarta por `K < 3`
(`287-289`), igual que A.

### 3.6 UCEs sin cobertura (filas cero)

`_compute_uce_centroids_from_retro` deja filas todo-cero para UCEs OOV o sin
lemas (`3585`, `3612-3613`). Regla:

- La covarianza local para PC1 se calcula **solo con filas no-cero**; las cero
  no deben orientar el eje.
- Las filas cero se proyectan igual y se asignan al corte.
- Tras el arbol, se reasignan a la hoja cuyo centroide en 30-d sea el mas
  cercano; si hay empate exacto (todas las cero son identicas), se elige la
  hoja de menor label. Se registra `n_zero_rows` y
  `n_zero_rows_reassigned`.
- Se conserva el rechazo si `zero_rows > 50%` (`5124-5131`): sin cobertura
  suficiente, DenseCDH no puede afirmar estructura.

Esto cumple "sin `-1`" sin fingir que una UCE OOV tiene senal semantica: queda
asignada para cobertura, pero marcada en el diagnostico.

### 3.7 Determinismo

- PCA global: se mantiene `random_state=42` (`5154`) para que HDBSCAN y
  DenseCDH vean **la misma** matriz. No se cambia a `config.random_state` en
  v1, porque alteraria la entrada de Method C y romperia la comparacion con
  corridas previas.
- PC1, corte y swap denso: sin RNG.
- Permutacion: `config.random_state` + clave estable del bloque.
- El resultado debe ser bit-a-bit igual con la misma semilla. El test lo exige.

### 3.8 Impresion y observabilidad

Se propone una linea paralela a `[CDH]` (`3712-3715`) y a `[EMB ...]` (`5190-5194`):

```
[DENSE-CDH] n=528 floor=33 K=9 min_leaf=34 max_leaf=78
            r2_1d=0.41 r2_denso=0.22 p=0.009 zero_rows=12
```

Y en cada pase:

```
--- Classification C1: DENSE-CDH tight (floor=33, alpha=0.05) ---
--- Classification C2: DENSE-CDH loose (floor=25, alpha=0.10) ---
```

---

## 4. Riesgos de implementacion

Los cinco primeros son los criticos. Cada uno indica el punto exacto donde se
tocaria y el invariante aguas abajo que se rompe si se ignora.

### R1. El flag no se despacha y DenseCDH nunca corre

- Punto exacto: `run()` construye C1/C2 solo en `src/reii/main_workflow.py:6108-6142`.
  `_do_emb` esta en `6109-6112`, los parametros HDBSCAN en `6117-6119` y las
  llamadas en `6124-6133`.
- Si se agrega el campo a `Config` pero no se ramifica aqui, `res_emb1` y
  `res_emb2` siguen saliendo de HDBSCAN y el flag es decorativo.
- Invariante afectado: todo el bloque de comparaciones B-C y A-C (`6164-6182`),
  la seleccion (`6234-6253`) y la persistencia (`9959-9965`).
- Mitigacion: ramificar en `6113`; test `test_method_c_backend_default_hdbscan`
  y `test_dense_cdh_is_called`.

### R2. Se rompe el contrato de `res_emb1`/`res_emb2` y sus consumidores

- Punto exacto: dict de retorno en `5204-5214`; clave `uce_to_label` en
  `5200-5202`. Consumidores: `6136`, `6169`, `6179`, `6213`, `6231` y
  `6277-6278` (`cluster_persistence`).
- Si DenseCDH omite `cluster_persistence`, `n_noise` o `labels_array`, o si
  `labels_array` no esta alineado con `uce_ids`, aparecen `KeyError`, ARI
  incorrecto o matrices de overlap mal indexadas.
- Invariante afectado: `_hungarian_stability_direct` (`5235-5322`) y el
  `pairwise_stability` que consume el dashboard.
- Mitigacion: devolver exactamente las mismas claves; `n_noise=0`;
  `cluster_persistence=[]`; assert `len(labels_array) == len(uce_ids) == n`.

### R3. La comparabilidad de la seleccion se sesga a favor de C

- Punto exacto: `_hungarian_stability_direct` usa `common = set(A) & set(B)`
  (`5273`) y calcula ARI sobre los comunes con label valido (`5305-5313`).
  HDBSCAN solo aporta UCEs no-ruido (`5200-5202`); DenseCDH aporta todas.
  El ranking usa `-ARI` y `-coverage_util` (`372-379`), y el gate de ruido
  (`290-294`) pasa trivialmente con `noise=0`.
- Efecto concreto en el corpus: una particion con clase mayor de 442 sobre 528
  asignadas da `442/528 = 0.837`, que **pasa** el gate de dominancia `0.90`
  (`295-301`), mientras que la misma clase sobre 479 asignadas daba
  `442/479 = 0.923` y era rechazada. El denominador "asignadas" cambia de
  semantica al desaparecer el ruido.
- Invariante afectado: `MethodMetrics.max_class_share` y la regla `select-v3`.
- Mitigacion: reportar `ari_full` y `ari_assigned` en el diagnostico; anadir
  un aviso `dense_cdh_warn_class_share=0.85` (solo diagnostico, no gate) y
  auditar con `9_selection_audit.py`. No cambiar los defaults de `select-v3`
  en v4.

### R4. Cambian vocabulario, phi, AFC, CAH y centroides aguas abajo

- Punto exacto: `primary_stable` (`6253`) -> `uces_est_list` (`9954-9965`) ->
  `labels_uces` (`10163`) -> vocabulario (`10170`), matriz (`10171-10173`),
  AFC (`10184`), CAH global (`10219-10221`), CAH por clase (`10223-10235`).
  Phase 8 y centroides en `10007-10026`; proyeccion liminal en `10094-10117`.
- Al asignar mas UCEs, el corpus analitico crece: cambian las frecuencias del
  vocabulario, los denominadores de phi, los perfiles de fila del AFC y los
  dendrogramas CAH. No es un bug, pero rompe la comparacion "manzanas con
  manzanas" contra A y contra corridas HDBSCAN previas.
- Invariante afectado: `db.data["vocabulario"]`, `df_terms`, `proyeccion`,
  `cah_por_clase`, y los extras de quality (`quality/hooks.py:149-196`).
- Mitigacion: persistir `backend` en `db.data["method_selection"]` y en el
  resultado; comparar solo corridas con el mismo backend; documentar que el
  denominador de coverage cambia.

### R5. Plomeria de configuracion y `run_id` de quality

- Punto exacto: `Config` en `src/reii/main_workflow.py:1009-1038`; allowlist
  `src/reii/backend/config_store.py:36-46`; hash por step
  `src/reii/backend/steps.py:166-183` (bloque `classify` en `170-178`);
  optimizador en `8408-8418`, `8529-8533` y `9634-9644`.
- Si el campo no se agrega a la allowlist, no se puede fijar desde el backend
  ni sobrevive a la UI. Si no se agrega a `_STEP_CONFIG_KEYS["classify"]`, el
  cache de steps no invalida al cambiar de backend.
- Pero agregarlo a `Config` cambia el hash canonico que genera `run_id` en
  `quality/hooks.py:265-277`, incluso cuando el valor es el default. Eso
  rompe la continuidad del historial de quality aunque el comportamiento sea
  identico. `9_selection_audit.py:2-18` agrupa por fingerprint de corpus, no
  por config, asi que no se rompe; `7_quality_multiverse.py:166-176` y
  `209-216` solo leen extras de seleccion.
- Mitigacion: excluir el campo del hash canonico cuando vale `"hdbscan"`, o
  versionar el esquema de config; agregarlo a la allowlist del backend y a
  `classify`; no incluirlo en el espacio de Optuna en v4.

### R6. Filas cero y determinismo (riesgo secundario)

- Punto exacto: filas cero en `3572-3622` (`3585`, `3612-3613`); rechazo
  `>50%` en `5124-5131`; PCA fija en `5154`; RNG del gate en `4052`.
- Si las filas cero entran en la covarianza, PC1 se orienta hacia el origen y
  el corte puede separar "UCEs con cobertura" de "UCEs OOV" en lugar de
  separar temas. Si se siembra la permutacion con un RNG global, el arbol deja
  de ser reproducible.
- Mitigacion: excluir cero de la covarianza, reasignar al centroide mas
  cercano, contar y registrar; semilla estable por bloque; no usar RNG en
  PC1/corte/swap. Mantener `random_state=42` en la PCA compartida.

---

## 5. Plan de tests

Solo nombres y afirmaciones. No se implementa aqui.

| Caso                                      | Fixture                                                | Afirma                                                                                    |
| ----------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `test_dense_cdh_k_in_gate_range`          | 3, 6 y 10 blobs; `n=528`; `cdh_target_k in {3,5,8,10}` | `3 <= K <= 20` y `K <= 2*cdh_target_k`.                                                   |
| `test_dense_cdh_leaf_floor`               | blobs desbalanceados; `floor=33`                       | Toda hoja tiene `size >= floor`, salvo el caso raiz `n < 2*floor`.                        |
| `test_dense_cdh_never_minus_one`          | cualquier fixture, ambos pases                         | `labels_array.min() >= 0`, `len(uce_to_label) == n`, `n_noise == 0`.                      |
| `test_dense_cdh_deterministic_same_seed`  | mismo corpus, misma config, dos corridas               | `np.array_equal(labels_a, labels_b)` y mismo `uce_to_label`.                              |
| `test_dense_cdh_small_n_single_leaf`      | `n=7`, `floor=5`                                       | `K=1`, no lanza excepcion, imprime aviso de rama sin `2*floor`.                           |
| `test_dense_cdh_complete_and_disjoint`    | `n=200`                                                | Cada indice cae en exactamente una hoja; `len(unique(labels)) == K`.                      |
| `test_dense_cdh_tight_loose_ari`          | 4 clases plantadas, floors 33 y 25                     | `0.4 <= ARI < 1.0` y las particiones no son identicas; `perturbation_effective=True`.     |
| `test_dense_cdh_zero_vector_assignment`   | 10% filas cero                                         | Todas asignadas, sin NaN, `n_zero_rows` correcto.                                         |
| `test_dense_cdh_zero_vector_rejection`    | 60% filas cero                                         | Devuelve `None`, igual que `5124-5131`.                                                   |
| `test_dense_cdh_degenerate_pc1`           | datos isotropicos (autovalores casi iguales)           | Fallback determinista; dos corridas dan el mismo resultado.                               |
| `test_dense_cdh_max_leaves_consolidation` | `dense_cdh_max_leaves=4`                               | `K <= 4`, piso preservado, sin `-1`, consolidacion reproducible.                          |
| `test_method_c_backend_default_hdbscan`   | `Config()` sin argumentos                              | `method_c_backend == "hdbscan"`; labels golden de C no cambian.                           |
| `test_dense_cdh_sparse_path_untouched`    | flag off y on                                          | Los labels de Method A son identicos en ambas corridas.                                   |
| `test_selection_dense_cdh_metrics`        | `MethodMetrics(K=5, ari=0.7, noise=0, coverage=0.9)`   | `_feasibility` devuelve `None`; con `max_class_share=0.92` devuelve motivo de dominancia. |
| `test_dense_cdh_pc1_sign_canonical`       | bloque con signo invertido y filas permutadas          | La particion es igual salvo permutacion de etiquetas 0/1.                                 |

Criterio de aceptacion de la implementacion: todos los tests verdes, y una
corrida A/B/C con `method_c_backend="dense_cdh"` sobre el corpus de 528 UCEs
que produzca `K` en rango, `noise=0`, y un ARI C1-C2 que no sea 1.0 trivial.

---

## 6. Rollout y rollback

| Fase | Contenido                                                                  | Criterio de salida                             |
| ---- | -------------------------------------------------------------------------- | ---------------------------------------------- |
| 0    | Este documento. Sin cambios de codigo.                                     | Revision del diseno.                           |
| 1    | `DenseCDH` + helper de matriz + tests unitarios. Flag default `"hdbscan"`. | Tests verdes; golden de v4 intacto.            |
| 2    | Corrida A/B/C en `select` con `dense_cdh`, comparar contra HDBSCAN.        | `K` en [3,20], `noise=0`, seleccion auditable. |
| 3    | Repetir con `uce_target_size` vecinos y correr `9_selection_audit.py`.     | El ganador no cambia por segmentacion.         |
| 4    | Evaluar si entra al espacio de Optuna (fuera de v4).                       | Decision separada.                             |

Rollback: `method_c_backend="hdbscan"` restaura el comportamiento actual sin
migracion de datos. La clave `backend` en el dict de resultado es aditiva; los
consumidores que no la conocen la ignoran.

---

## 7. Invariantes que DenseCDH debe cumplir (checklist)

1. Toda UCE termina en exactamente una hoja; `labels >= 0` siempre.
2. Ninguna hoja por debajo de `floor`, salvo raiz `n < 2*floor`.
3. `n_min_split = 2*floor` antes de intentar un corte.
4. El corte usa `_corte_optimo` con `min_side=floor`.
5. Si el swap denso rompe el piso, se revierte a la particion pre-swap.
6. Si queda una sola clase, el nodo es hoja.
7. La significancia exige `r2_1d >= min_r2` y permutacion densa `< alpha`.
8. Misma semilla -> mismos labels.
9. La matriz de entrada es la misma PCA de 30 dims que usa HDBSCAN.
10. El dict de salida conserva las claves de `5204-5214`.
