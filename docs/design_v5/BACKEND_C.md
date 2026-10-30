# Backend de clasificación densa para Method C

Estado: **propuesta de diseño**, validada con mediciones en
`scripts/playground/c_backends_arena.py`. No hay cambios aplicados en
`src/reii/**`; los diffs van en el mensaje de entrega.

Alcance: **solo** el paso "clasificación densa" de Method C
(`_clasificar_umbral_embedding`). No cubre el gate de selección de método
(otro agente), ni A/B, ni retrofitting.

---

## 1. Contrato de resultado

Un solo paso, cuatro backends intercambiables vía
`Config.method_c_backend in {"hdbscan","dense_cdh","ward","kmeans_spherical"}`.
Todos devuelven el **mismo dict**, o `None` si no hay partición utilizable.

### Claves heredadas (intactas, ya consumidas por `run()`)

| clave                 | tipo            | notas                                        |
| --------------------- | --------------- | -------------------------------------------- |
| `method`              | `str`           | siempre `"embedding"` (lo espera `run()`)    |
| `uce_to_label`        | `Dict[str,int]` | **solo** etiquetas `>= 0`                    |
| `labels_array`        | `np.ndarray`    | alineado 1:1 con `uce_ids`; puede tener `-1` |
| `uce_ids`             | `List[str]`     | ids normalizados                             |
| `embedding_matrix`    | `np.ndarray`    | matriz **post-PCA** usada por el backend     |
| `cluster_persistence` | `List[float]`   | HDBSCAN; `[]` en los densos                  |
| `n_clusters`          | `int`           | clases efectivas                             |
| `n_noise`             | `int`           | `sum(labels < 0)`; `0` en los densos         |
| `min_cluster_size`    | `int`           | piso aplicado                                |

### Claves nuevas (aditivas)

| clave             | tipo        | por qué                                                          |
| ----------------- | ----------- | ---------------------------------------------------------------- |
| `backend`         | `str`       | trazabilidad; el gate y el paper deben saber qué se corrió       |
| `target_k`        | `int`       | K **pedido**, no K logrado (en HDBSCAN: derivado, no vinculante) |
| `partition_sizes` | `List[int]` | tamaños crudos, **excluyendo** `-1`, ordenados                   |
| `n_assigned`      | `int`       | denominador explícito del gate de dominancia                     |
| `diagnostics`     | `Dict`      | métricas propias del backend (ver abajo)                         |

`diagnostics` por backend:

| backend            | claves                                                                                                                                                     |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dense_cdh`        | `bisection_depths`, `r2_1d`, `r2_dense`, `mean_r2_dense`, `degenerate_pc1`, `reverted_floor`, `rejected_min_r2_dense`, `n_splits`, `floor`, `min_r2_dense` |
| `ward`             | `merge_heights`, `cut_height`, `mean_merge_height`, `n_merges`                                                                                             |
| `kmeans_spherical` | `inertia`, `n_iter`, `n_init`, `random_state`, `normalized_input`                                                                                          |
| `hdbscan`          | `cluster_persistence`, `min_cluster_size`, `min_samples`, `epsilon`                                                                                        |

**Regla del contrato:** `n_noise == 0` y `n_assigned == len(uce_ids)` en los
tres backends densos, por construcción. Si un denso devolviera `-1` sería un
bug del backend, no un caso previsto.

### Por qué `partition_sizes` y `n_assigned` son obligatorios

`_partition_sizes_from_labels` (L542) **excluye** `label < 0` del denominador.
Con HDBSCAN eso es correcto (el ruido no debe diluir la dominancia). Con
backends densos no hay ruido, así que el denominador deja de estar diluido y
**la misma partición cambia de veredicto** (ver §5). El backend debe exponer
el numerador y el denominador por separado para que el gate sea explícito y
no dependa de que alguien reconstruya los tamaños desde `labels_array`.

---

## 2. Decisión por backend

| backend            | K exacto                             | `-1` | determinista           | coste        | rol                         |
| ------------------ | ------------------------------------ | ---- | ---------------------- | ------------ | --------------------------- |
| `dense_cdh`        | **sí** (si no lo frena el gate/piso) | no   | sí                     | bajo         | **candidato por defecto**   |
| `ward`             | **sí**                               | no   | sí                     | alto (O(n²)) | alternancia / control       |
| `kmeans_spherical` | **sí**                               | no   | sí (`random_state=42`) | medio        | alternancia / control       |
| `hdbscan`          | **no**                               | sí   | sí                     | bajo         | **diagnóstico, nunca gana** |

### `dense_cdh` — bisección divisiva _best-first_

En cada paso se parte la hoja con **mayor `r2_dense`** entre todas las hojas
evaluables, no la siguiente en un orden fijo. Consecuencias:

1. **K controlable**: se para al llegar a `target_k`. La recursión pura no
   puede hacerlo (ver §9).
2. **Los splits débiles quedan para el final**: si el gate corta, corta por
   el borde, no por el centro del árbol.
3. **Cada hoja se evalúa una sola vez** (`done` flag): no hay re-evaluación
   ni barrido O(n²).

Eje de corte: **PC1 local** del bloque, calculada con `eigh` sobre la
covarianza de las filas no-cero (determinista), con canon de signo (el
componente de mayor valor absoluto se fuerza positivo) y _fallback_ a un eje
canónico si `λ1 ≈ λ2`. Corte por `_corte_optimo_1d` con `min_side = floor`:
**el mismo mecanismo de piso que el CDH de A/B**, no uno nuevo. Refinamiento
posterior por centroide (Lloyd determinista, sin RNG) con **reversión** a la
partición del corte si el refinamiento rompe el piso — espejo exacto de
`main_workflow.py:4144-4147`. Etiquetas finales ordenadas por `idx[0]`.

### `ward`

`AgglomerativeClustering(n_clusters=K, linkage="ward", compute_distances=True)`
sobre PCA(30). Determinista, dendrograma completo disponible
(`merge_heights`) como diagnóstico. Es el único backend O(n²) en memoria;
con n=528 es irrelevante, con n≈5800 (corpus grande) son ~270 MB de matriz
de distancias. **No usar como default si el corpus crece.**

### `kmeans_spherical`

L2-normalizar las filas y `KMeans(n_clusters=K, n_init=10, random_state=42)`.
El prefijo "spherical" es honesto: la normalización L2 hace que la distancia
euclídea sea monótona con la similitud coseno. Determinista porque
`n_init=10` recorre 10 inicializaciones y se queda con la de menor inercia;
el RNG está fijado, no oculto. Es el backend más lento medido (hasta 218 ms
con n=528, K=20) porque paga 10 arranques.

### Por qué HDBSCAN baja a diagnóstico

No es una opinión estética; son tres artefactos medidos:

1. **No controla K.** El "K" del trial solo fija `min_cluster_size`; el K
   real emerge. En `synth_blobs` (K_true=8) HDBSCAN da 2, 5, 8, 8, 8, 8, 8
   para target 3…20: monótono y saturado, no controlable.
2. **Colapsa a K=2 o a ruido total.** En `real_ctx` da `Kach=0` (todo ruido)
   para `min_cluster_size` 39/23/14 y solo 2–4 clases con `mcs ≤ 10`. En
   `synth_null` da `Kach=0` en los 7 valores de K.
3. **Su ARI parece bueno solo cuando K colapsa.** El probe v4 ya lo mostró:
   ARI C1↔C2 = 0.785 con K=2 vs 3, y 0.092 con K=6 vs 12. La persistencia
   es 0.02–0.10 (clusters superficiales). Un ARI alto con K=2 no es
   estabilidad, es el artefacto de monotonía con la coarseza.

Por eso se mantiene disponible (es el comportamiento histórico y sirve para
comparar), pero **no puede ganar** la regla de selección: se marca
`diagnostic_only=True` y `_feasibility` lo descarta con motivo explícito.
Alternativa descartada: `available=False` — mentiría ("no produjo partición")
cuando sí produjo una; el motivo del descarte tiene que ser legible en el
paper.

Interruptor de escape: `method_c_hdbscan_selectable: bool = False`. Si un
experimento quiere volver al comportamiento previo, se pone en `True` sin
tocar código.

---

## 3. Control de K y determinismo

### De dónde sale `target_k`

`Config.method_c_target_k: Optional[int] = None`. Si es `None`, se deriva:

```
frac  = min_cluster_size_cdh            # ya viene de cdh_target_k vía
                                        # _cdh_frac_from_params: frac = 1/(2K)
K     = int(round(1.0 / (2.0 * frac)))
K     = clamp(K, 2, n // floor)
```

Es decir, **K se recupera del plumbing que ya existe** (`cdh_target_k` está
en el espacio de búsqueda, L8413) sin tocar el optimizador ni añadir
parámetros al sampler. Advertencia: `round(2.5) == 2` en Python (banker's
rounding), así que `frac = 1/5` da K=2, no 3; está acotado a `>= 2` y no
rompe nada, pero conviene saberlo si alguien compara con el valor crudo de
`cdh_target_k`.

El pase _loose_ multiplica el piso por `dense_cdh_loose_factor = 0.75`
(nunca por debajo de 5 UCEs). En los densos el pase loose **no cambia K**,
solo el tamaño mínimo de hoja.

### Determinismo (verificado, no afirmado)

El playground corre **dos veces cada celda (backend × K)** y compara
`labels_array` y `uce_to_label`. Resultado: **56/56 celdas `[ok]`** en los
tres datasets (168 pares de corridas). Sin RNG oculto:

- `dense_cdh`: `eigh` + `argsort` + canon de signo. Cero aleatoriedad.
- `ward`: sin RNG.
- `kmeans_spherical`: `random_state=42` explícito, `n_init=10` fijo.
- `hdbscan`: determinista para estos datos.

### Consecuencia crítica: el pase tight/loose es **vacuo** para ward y kmeans

`ward_backend` y `kmeans_spherical` **no reciben el piso**; su partición
depende solo de `(X, K)`. Por lo tanto C1 y C2 son **idénticas** y
`ari(C1,C2) == 1.0` **por construcción**, sobre estructura real y sobre ruido
puro. Medido: ARI=1.0 en `synth_null` para los 7 valores de K.

`dense_cdh` sí cambia (el piso entra en `min_side` y en la condición de
parada). Medido en `real_ctx`, tight vs loose: ARI 1.000 (K=3), 0.531 (K=5),
0.982 (K=8), 0.383 (K=11), 0.406 (K=14), 0.995 (K=18), 1.000 (K=20). En
`synth_null`: 1.000, 1.000, 0.472, 0.508, 0.476, 0.486, 0.428.

**Implicación para el paper:** `ARI(C1,C2)` **no puede ser el criterio
principal** cuando el backend es denso. Con `ward`/`kmeans` da 1.0 siempre, y
con `dense_cdh` da valores altos también sobre ruido (0.43–0.49 en null). No
es estabilidad, es una propiedad algebraica del backend. La regla de
selección debe: (a) no premiar `ari == 1.0` cuando el backend no tiene
grados de libertad en el pase, o (b) sustituir la perturbación por una que
exista de verdad (submuestreo bootstrap + proyección por centroide). Esto
está **fuera** de mi alcance (es el gate), pero el backend lo expone:
`diagnostics` incluye lo necesario para detectarlo, y §6 da la señal
alternativa medida.

---

## 4. Coste: complejidad y runtime medido

n=528, d=30, hilos OpenMP/BLAS fijados a 1.

| backend            | complejidad                                   | memoria | runtime medido (ms) |
| ------------------ | --------------------------------------------- | ------- | ------------------- |
| `dense_cdh`        | O(n·d² + K·d³ + n log n) ≈ O(n) con d=30 fijo | O(n·d)  | 6.3–39.1            |
| `ward`             | O(n²·d) tiempo, **O(n²) memoria**             | O(n²)   | 4.1–20.9            |
| `kmeans_spherical` | O(n·K·d·i·n_init)                             | O(n·K)  | 29.6–218.4          |
| `hdbscan`          | O(n log n) típico, O(n²) peor caso            | O(n)    | 15.4–89.5           |

Rango = mínimo y máximo sobre los tres datasets y los 7 valores de K
(detalle por celda en la tabla de §11).

`dense_cdh` **no es O(n²)**: cada hoja se evalúa una sola vez; el trabajo por
evaluación es `eigh` sobre una covarianza 30×30 (constante) más un
`argsort` de las filas de la hoja. Sumando sobre el árbol, el total es
proporcional a `n · d²` más un término `K · d³` que con d=30 es despreciable.
Medido: 26 ms para K=20 en n=235 y 39 ms para K=20 en n=528 — escala con
`n`, no con `n²`.

Corrida completa del playground: **15.7 s** con la caché de imports caliente,
**~33 s** en frío (el import de `reii.quality.metrics` varía entre 5 y 32 s
según Windows Defender). Presupuesto: <60 s. ✔

---

## 5. El gate de dominancia cuando no hay `-1`

El gate actual (`method_selection_max_class_share = 0.90`, aplicado en
`_feasibility` L295-301) divide la clase mayor por las UCEs **asignadas**.

Caso concreto del corpus real (528 UCEs):

| escenario                | clase mayor | denominador   | share     | gate 0.90  |
| ------------------------ | ----------- | ------------- | --------- | ---------- |
| HDBSCAN con 49 de ruido  | 442         | 479 asignadas | **0.923** | ✗ descarta |
| backend denso, sin ruido | 442         | **528**       | **0.837** | ✓ **pasa** |

La **misma** estructura (una clase de 442 + satélites) se descarta o se
acepta según si el backend emite `-1`. Eso es un artefacto del denominador,
no una propiedad del corpus. Un backend denso no puede "arreglar" una
partición degenerada simplemente porque asigna todo.

**Qué debe exponer el backend para que el gate funcione** (esto sí es mi
responsabilidad):

1. `partition_sizes` crudos, **sin** `-1`, sin recortes por `tiny_frac`, sin
   filtrar por la intersección tight/loose.
2. `n_assigned` explícito, para que el gate no tenga que inferir el
   denominador.
3. `n_noise`, para que el gate pueda distinguir "no hay ruido" de "el ruido
   se contó mal".

**Recomendación (no implementada aquí, es del gate):** si el objetivo es que
0.90 signifique lo mismo en los cuatro backends, el denominador debe ser
`n_total` (528), no `n_assigned`. Con `n_total`, los dos escenarios dan 0.837
y 0.923 — siguen discrepando, porque el ruido de HDBSCAN _sí_ cambia la
partición. La alternativa honesta es **elegir un denominador y declararlo en
el paper**, y no comparar `max_class_share` entre backends con distinto
tratamiento del ruido. Lo que no se puede es dejar que el denominador lo
elija el backend implícitamente.

---

## 6. ¿Qué señal de validez sirve? (medido)

Se probaron tres candidatas. Solo una discrimina.

| señal                        | ruido (`synth_null`) | estructura real (`real_ctx`) | veredicto         |
| ---------------------------- | -------------------- | ---------------------------- | ----------------- |
| `r2_1d` del corte óptimo     | **0.626–0.654**      | 0.699–0.723                  | **inútil** (≈2/π) |
| `r2_dense` medio             | **0.036–0.064**      | 0.103–0.142                  | **discrimina**    |
| `eta²` de la partición final | 0.056–0.205          | 0.196–0.412                  | confundido con K  |

1. **`r2_1d` es vacuo.** El corte 1-D óptimo sobre una proyección PCA de una
   gaussiana isotrópica explica ≈ **2/π = 0.637** de la varianza del eje, por
   pura geometría del corte central. Un umbral `r2_1d >= 0.05` **nunca
   rechaza nada**. Medido: 0.626–0.654 en null. Es un error del diseño previo
   (§9). El test de permutación sobre ese mismo eje es igualmente vacuo: se
   permuta el eje cuyo corte ya se optimizó.
2. **`r2_dense` discrimina** y no está confundido con K, porque es el R² del
   _split_ (1 − SSE/TSS en las 30 dims), no de la partición final. En
   `synth_blobs` sube a 0.146–0.240. Gate propuesto:
   `dense_cdh_min_r2_dense = 0.08`.
3. **`eta²` (varianza explicada por la partición final) NO sirve como gate
   universal.** Crece monótonamente con K en los dos regímenes (null:
   0.056@K=5 → 0.205@K=20; real: 0.196@K=5 → 0.412@K=20), así que un umbral
   fijo está confundido por el K elegido. Se expone como diagnóstico, no como
   gate.

**Barrido del gate `r2_dense`** (K=8, `dense_cdh`):

| umbral | `real_ctx` Kach / silh | `synth_null` Kach | `synth_blobs` Kach |
| ------ | ---------------------- | ----------------- | ------------------ |
| 0.00   | 8 / 0.042              | 8                 | 8                  |
| 0.05   | 8 / 0.042              | **1**             | 8                  |
| 0.08   | 8 / 0.042              | **1**             | 8                  |
| 0.12   | **1**                  | 1                 | **1**              |
| 0.20   | 1                      | 1                 | 1                  |

0.08 es el punto que deja pasar estructura real y corta ruido; 0.12 ya mata
`real_ctx` y `synth_blobs`. `Kach=1` ⇒ el backend devuelve `None` ⇒ Method C
queda no disponible, que es el comportamiento correcto sobre ruido.

---

## 7. Casos límite

| caso                           | comportamiento                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **n pequeño** (n < 2·floor)    | `dense_cdh`: una sola hoja ⇒ `n_clusters=1` ⇒ `None`. `ward`/`kmeans`: `KMeans`/`Agglomerative` exigen `n >= K`; se acota `K` a `n // floor` antes de llamar y, si aun así `K < 2`, se devuelve `None`.                                                                                                                                                                                                                                                                                                  |
| **K > clases reales**          | `dense_cdh`: el gate `r2_dense` frena y `Kach < K` (nunca fabrica splits). `ward`/`kmeans`: **parten igual** y producen K clases, algunas artificiales (clases "satélite" que son trozos de una real). Medido en `synth_blobs` K=20: silh de ward 0.499 vs 0.408 en K=8 ⇒ el exceso de K no se detecta con silhouette. Consecuencia: `K` debe venir acotado del optimizador, no del backend.                                                                                                             |
| **UCEs con vector cero** (OOV) | Se detectan antes: si `zero_rows > 50%` se aborta (comportamiento heredado). Si no, se conservan **con** su fila cero: `dense_cdh` los excluye del cálculo de PC1 local (`nz` mask) pero **no** de las hojas (se asignan por el corte). `kmeans` los normaliza con `norms<1e-12 → 1.0` para no dividir por cero. Consecuencia: un vector cero cae en la clase del centroide más cercano (habitualmente la de mayor masa) — es una UCE no informativa asignada, no un error. Se reporta en `diagnostics`. |
| **`λ1 ≈ λ2` en PC1 local**     | _Fallback_ a eje canónico `e1`; se cuenta en `diagnostics.degenerate_pc1`. No aborta.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Todos los puntos iguales**   | `r2_dense = 0` ⇒ gate corta ⇒ `None`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **`n_clusters < 2`**           | `None` (mismo contrato que HDBSCAN en L5196).                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

---

## 8. Qué NO tocar

- Tipos `UC` / `UCE` y su contrato. Method C **no** construye `UC`; asigna
  etiqueta por UCE (`uce_to_label`). Cambiar eso rompe `_extract_uce_labels`
  y `_hungarian_stability_direct`.
- `MatrizBuilder` y todo el pipeline de A (chi², promediado recíproco,
  `_intercambio`). El backend denso **no** reutiliza la matriz dispersa de
  A/B; trabaja sobre los centroides fastText en `np.ndarray`.
- El **retrofitting** (`_compute_uce_centroids_from_retro`,
  `retrofitted_vectors`, PPMI/`ppmi_k`). El backend consume lo que ya está
  calculado; no lo recalcula, no lo cambia de espacio, no re-normaliza la
  matriz de entrada (salvo el L2 explícito de `kmeans_spherical`, que es
  local a ese backend).
- `ClasificadorDescendente.clasificar` y el CDH de A. Solo se **extrae**
  `_corte_optimo` a `_corte_optimo_1d` (misma matemática, verificado por
  comparación de etiquetas en el playground); el comportamiento de A no
  cambia.
- `main_workflow_clasico.py`. Tiene su propia copia de
  `ClasificadorDescendente`; no se toca.

---

## 9. Crítica a `docs/DENSE_CDH_DESIGN.md`

**Qué se adopta:** bisección por PC1 local en PCA(30), corte 1-D por
`_corte_optimo` con piso, refinamiento por centroide determinista, sin `-1`.
El núcleo es correcto y se implementa tal cual.

**Qué se descarta o corrige:**

1. **Recursión pura — descartada.** El diseño parte siempre las dos ramas
   hasta que el criterio de parada corta. Eso **no controla K**: el K emerge
   del umbral, que es exactamente el problema que tiene HDBSCAN hoy. La
   versión _best-first_ (partir la hoja de mayor ganancia, parar en
   `target_k`) sí controla K y además deja los cortes débiles al final.
2. **`dense_cdh_max_leaves` — innecesario.** Era el parche para que la
   recursión pura no explotara en número de hojas. Con _best-first_ y
   `target_k`, el tope es el propio K. Dos parámetros para lo mismo es una
   fuente de configuraciones incoherentes.
3. **Gate `r2_1d >= min_r2_threshold` — vacuo.** Ver §6: en ruido isotrópico
   el corte 1-D óptimo explica ≈2/π y nunca cae por debajo de 0.05. El gate
   tal como está propuesto **aprueba todo**, incluido ruido puro. Se
   reemplaza por `r2_dense` (calibrado, 0.08) y `r2_1d` queda como
   diagnóstico con default 0.0 (apagado).
4. **Test de permutación sobre el eje 1-D — vacuo.** Permutar el eje cuyo
   corte ya se optimizó no puede rechazar el corte: el estadístico es
   invariante. No aporta información sobre la existencia de estructura.
5. **`dense_cdh_axis` / `perm_mode` — reservados, no implementados.** Añaden
   superficie de configuración sin evidencia de que otra opción ayude. Mejor
   una sola política determinista y documentada.

---

## 10. Cómo reproducir

```
python scripts/playground/c_backends_arena.py
```

Sin red, sin docker, solo lectura de `data/workflow_data.json`, exit 0,
~16 s en caliente.

Limitaciones del playground (declararlas si se citan sus números):

- `embedding` es **384-d SentenceTransformer**, no el fastText-retrofit
  300-d de Method C: _space mismatch_. Los valores son indicativos.
- Solo **235/600** UCEs del archivo tienen `embedding` (las `is_stable`); el
  resto es `None`. `retrofitted_vectors_significant` está vacío, así que el
  espacio real **no se puede reconstruir**.
- El playground fuerza PCA(30). Producción usa
  `target_dims = max(2, min(30, n//10, d))`: con n=235 serían **23** dims, con
  n=528 son 30. La tabla de `real_ctx` sobreestima ligeramente la dimensión
  respecto de una corrida de producción con 235 UCEs.
- El archivo corresponde a un bench con `uce_target=26` (600 UCEs), no a la
  corrida `select` en curso.

---

## 11. Resultados medidos (arena, dos corridas por celda, `[ok]` = determinismo)

### `real_ctx` (n=235, PCA 384→30) — indicativo por el _space mismatch_

| backend   | K   | Kach | silh  | ARI K/K+2 | cv_sz | dom   | noise | ms   |
| --------- | --- | ---- | ----- | --------- | ----- | ----- | ----- | ---- |
| hdbscan   | 3   | 0    | n/a   | n/a       | n/a   | n/a   | 235   | 20.1 |
| hdbscan   | 5   | 0    | n/a   | n/a       | n/a   | n/a   | 235   | 16.8 |
| hdbscan   | 8   | 0    | n/a   | n/a       | n/a   | n/a   | 235   | 15.4 |
| hdbscan   | 11  | 2    | 0.239 | n/a       | 0.154 | 0.577 | 183   | 18.3 |
| hdbscan   | 14  | 2    | 0.239 | n/a       | 0.154 | 0.577 | 183   | 16.8 |
| hdbscan   | 18  | 3    | 0.193 | 1.000     | 0.637 | 0.625 | 187   | 16.8 |
| hdbscan   | 20  | 4    | 0.191 | n/a       | 0.757 | 0.566 | 182   | 18.3 |
| dense_cdh | 3   | 3    | 0.077 | 0.558     | 0.593 | 0.613 | 0     | 6.3  |
| dense_cdh | 5   | 5    | 0.051 | n/a       | 0.462 | 0.323 | 0     | 6.8  |
| dense_cdh | 8   | 8    | 0.042 | n/a       | 0.842 | 0.323 | 0     | 13.0 |
| dense_cdh | 11  | 11   | 0.038 | n/a       | 0.814 | 0.323 | 0     | 23.7 |
| dense_cdh | 14  | 14   | 0.026 | n/a       | 1.063 | 0.323 | 0     | 16.3 |
| dense_cdh | 18  | 18   | 0.019 | 0.979     | 1.287 | 0.323 | 0     | 22.5 |
| dense_cdh | 20  | 20   | 0.010 | n/a       | 1.371 | 0.323 | 0     | 26.0 |
| ward      | 8   | 8    | 0.056 | n/a       | 0.286 | 0.170 | 0     | 4.4  |
| ward      | 20  | 20   | 0.064 | n/a       | 0.383 | 0.089 | 0     | 4.8  |
| kmeans    | 8   | 8    | 0.075 | n/a       | 0.214 | 0.162 | 0     | 42.9 |
| kmeans    | 20  | 20   | 0.059 | n/a       | 0.362 | 0.072 | 0     | 78.0 |

### `synth_blobs` (n=528, K_true=8, 12 % uniforme)

| backend   | K   | Kach | silh  | ARI K/K+2 | cv_sz | dom   | noise | ms    |
| --------- | --- | ---- | ----- | --------- | ----- | ----- | ----- | ----- |
| hdbscan   | 8   | 8    | 0.594 | n/a       | 0.249 | 0.172 | 63    | 66.4  |
| hdbscan   | 20  | 8    | 0.594 | n/a       | 0.249 | 0.172 | 63    | 57.6  |
| dense_cdh | 8   | 8    | 0.415 | n/a       | 0.223 | 0.165 | 0     | 16.7  |
| dense_cdh | 20  | 20   | 0.210 | n/a       | 0.803 | 0.140 | 0     | 31.2  |
| ward      | 8   | 8    | 0.408 | n/a       | 0.266 | 0.169 | 0     | 16.7  |
| ward      | 20  | 20   | 0.499 | n/a       | 1.048 | 0.152 | 0     | 15.3  |
| kmeans    | 8   | 8    | 0.428 | n/a       | 0.208 | 0.167 | 0     | 41.7  |
| kmeans    | 20  | 20   | 0.333 | n/a       | 0.847 | 0.129 | 0     | 100.1 |

HDBSCAN acierta K=8 en blobs (silh 0.594, el mejor), lo que confirma que el
problema no es "HDBSCAN es malo" sino "HDBSCAN no es controlable".

### `synth_null` (n=528, gaussiana isotrópica, control negativo)

| backend   | K    | Kach  | silh   | ARI K/K+2 | cv_sz | dom   | noise | ms        |
| --------- | ---- | ----- | ------ | --------- | ----- | ----- | ----- | --------- |
| hdbscan   | 3…20 | **0** | n/a    | n/a       | n/a   | n/a   | 528   | 41.2–55.6 |
| dense_cdh | 8    | 8     | 0.014  | n/a       | 0.535 | 0.280 | 0     | 21.4      |
| dense_cdh | 20   | 20    | −0.003 | n/a       | 1.117 | 0.280 | 0     | 39.1      |
| ward      | 8    | 8     | 0.008  | n/a       | 0.319 | 0.193 | 0     | 20.9      |
| ward      | 18   | 18    | 0.010  | **0.935** | 0.261 | 0.083 | 0     | 15.2      |
| kmeans    | 8    | 8     | 0.026  | n/a       | 0.088 | 0.140 | 0     | 44.4      |
| kmeans    | 18   | 18    | 0.028  | **0.130** | 0.181 | 0.076 | 0     | 218.4     |

Lectura: `ARI K/K+2` = 0.935 para ward sobre **ruido puro** → mide
anidamiento, no validez. `max_class_share` de ward cae a 0.083 en K=18 →
una partición de ruido pasa el gate de dominancia sin problema. La única
barrera efectiva sobre ruido es `r2_dense` (`dense_cdh` con 0.08 ⇒ `Kach=1`),
y para ward/kmeans **no hay barrera interna**: su validez tiene que venir de
fuera.

---

## 12. Diffs exactos

**Numeración:** este documento ya usa §1-§11 y §7 está ocupado por «Casos
límite». La sección se añade al final como **§12**, no como §7 ni §9: insertar
un §7 duplicado o renumerar §8-§11 rompería todas las referencias cruzadas de
`FINAL_PATCHSET.md` (que cita «BACKEND_C §1-§4, §7» para los hunks #22/#23).
La cabecera del documento (L3-5) dice que «los diffs van en el mensaje de
entrega»: eso queda **superado** por esta sección, pero no se edita la cabecera.

Cierra los hunks **#21** y **#22** de `FINAL_PATCHSET.md` §1 (las dos filas
«**sin diff**»), más **#69** (constante consumida) y un hunk **nuevo #72b**
que es _consecuencia mecánica_ de #72, no un capricho: `guarded_fit` de
`HARDENING.md` §8.1 llama a `dense_cdh`, `ward_backend`, `kmeans_spherical` y
`hdbscan_backend` como **globales del módulo**, y `method_c_policy.py` es un
módulo nuevo. Si esos nombres no existen en el módulo, `guarded_fit` lanza
`NameError` en la primera llamada. Ver §12.3.

### 12.1 Hunk #21 — `Config.method_c_backend`

Ancla real: `src/reii/main_workflow.py:1038` (`hdbscan_uce_metric`), final del
bloque Method C del dataclass `Config` (definido en L788).

```diff
--- a/src/reii/main_workflow.py
+++ b/src/reii/main_workflow.py
@@ -1038,1 +1038,35 @@ class Config:
     hdbscan_uce_metric: str = "euclidean"  # "euclidean" | "cosine"
+
+    # ── Method C — backend de clasificación densa (BACKEND_C.md §2, hunk #21)
+    # Valores y rol:
+    #   "dense_cdh"        bisección best-first por PC1 local con gate r2_dense.
+    #                      Controla K, no emite -1, O(n·d²) con d fijo. DEFAULT.
+    #   "ward"             AgglomerativeClustering(n_clusters=K, linkage=ward).
+    #                      K exacto, sin RNG, O(n²) en memoria: 144 MiB en el
+    #                      camino condensado y 275 MiB en el cuadrado a n=6000
+    #                      (HARDENING.md §2.2). Guarda: _METHOD_C_WARD_MAX_N.
+    #   "kmeans_spherical" L2 + KMeans(n_init=10, random_state=42). K exacto.
+    #   "hdbscan"          comportamiento histórico. NO controla K (target_k sólo
+    #                      fija min_cluster_size) y emite -1. Diagnostic only
+    #                      salvo que method_c_hdbscan_selectable sea True.
+    # Valor desconocido: NO aborta la corrida. Se resuelve con
+    # method_c_policy.resolve_method_c_backend(source="persisted"), que degrada
+    # al DEFAULT_METHOD_C_BACKEND de la política y emite un aviso "[!]".
+    # La procedencia (persistida vs interna) y el resto de la regla viven en
+    # CLASSIC_MODE.md §6; aquí sólo se declara el campo, no se duplica.
+    method_c_backend: str = "dense_cdh"
+    # Escape hatch: True devuelve a hdbscan el derecho a competir en `select`.
+    # False => diagnostic_only=True y `select-v4` lo descarta por
+    # backend == "hdbscan" (SELECT_V4.md §2.1).
+    method_c_hdbscan_selectable: bool = False
+    # K objetivo de dense_cdh. None => se deriva del plumbing existente:
+    # min_cluster_size_cdh = frac = 1/(2K) => K = round(1/(2·frac)), acotado a
+    # [2, n//floor] por method_c_policy.effective_target_k. Ver §3.
+    method_c_target_k: Optional[int] = None
+    # Gates medidos en §6: r2_1d ≈ 2/π sobre ruido => inútil (0.0, apagado);
+    # r2_dense discrimina (null 0.036-0.064, real 0.103-0.142) => 0.08.
+    dense_cdh_min_r2_1d: float = 0.0
+    dense_cdh_min_r2_dense: float = 0.08
+    # Pase loose: piso × 0.75, nunca por debajo de
+    # hdbscan_uce_min_cluster_floor (§3). En los densos no cambia K.
+    dense_cdh_loose_factor: float = 0.75
```

Nota de default: el valor de `Config` es `"dense_cdh"` (candidato por defecto,
§2), mientras que `DEFAULT_METHOD_C_BACKEND` de la política sigue siendo
`"hdbscan"` (`hardening_bench.py:94`) y se usa **sólo como destino de
degradación** de un valor inválido. Discrepancia declarada en §12.6.

### 12.2 Hunk #69 — constante `_METHOD_C_WARD_MAX_N`

Ancla real: `src/reii/main_workflow.py:558`, justo después de
`_partition_sizes_from_labels` (L542-558). HARDENING §2.2 fija 6000; el
`DEFAULT_WARD_MAX_N` del módulo de política es 4000 (§12.6). #69 dice
explícitamente que **no** va en `Config`.

```diff
@@ -558,1 +558,8 @@ def _partition_sizes_from_labels(labels: Optional[Dict[str, int]]) -> List[int]:
     return sorted(counts.values())
+
+
+# Tope de n para `ward` (memoria O(n²) de la matriz de distancias). A n=6000
+# son ~144 MiB condensados / ~275 MiB cuadrados. Por encima, la política
+# degrada a `dense_cdh` con aviso "[!]" en vez de abortar (HARDENING.md §2,
+# hunk #69). Se pasa a `guarded_fit(ward_max_n=...)`; NO es un campo de Config.
+_METHOD_C_WARD_MAX_N = 6000
```

### 12.3 Hunk #72b — ampliación de `src/reii/method_c_policy.py`

Se **añade al final** del módulo de #72, sin tocar ninguna de las funciones
copiadas de `hardening_bench.py:88-510`. Usa los imports que el módulo ya
tiene (`Any`, `Dict`, `List`, `Optional`, `Sequence`, `Tuple`, `np`).

Por qué aquí y no en `main_workflow.py`: (a) `guarded_fit` invoca los cuatro
backends como globales del módulo (§12.3.1 los define); (b) el módulo es
numpy-only, sin Stanza, así que `tests/test_method_c_backends.py` puede
ejercer el contrato §1 en <5 s; importar `reii.main_workflow` tarda minutos.

#### 12.3.1 Corte 1-D, PC1 local, refinamiento y R² denso

Copia literal de `c_backends_arena.py:240-341` (verificado línea a línea). Es
el «1-D cut» que `ClasificadorDescendente._corte_optimo` hace hoy por
bisección sobre la 1ª componente (`main_workflow.py:3831-3871`, invocado en
L4123): una sola implementación, no dos.

```python
SEED = 42  # mismo valor que c_backends_arena.py y Config.random_state
```

`SEED` no viene en el bloque copiado (en `hardening_bench.py` está en L79, antes
del rango 88-510): si se copia el fichero completo en vez del bloque, **no**
duplicar la constante.

```python
def corte_optimo_1d(coord: np.ndarray, min_side: int = 1) -> Tuple[float, np.ndarray]:
    """Corte que maximiza SS_between restringido a k en [min_side, n-min_side].

    COPIA LITERAL de ClasificadorDescendente._corte_optimo (main_workflow).
    En 1-D es el optimo exacto de 2-means con esa restriccion. Devuelve
    (R2 del corte, labels 0/1 ordenadas por `coord`).
    """
    n = len(coord)
    if n < 2:
        return 0.0, np.zeros(n, dtype=int)

    order = np.argsort(coord)
    sorted_c = coord[order]
    total = sorted_c.sum()
    mean = total / n

    cum = np.cumsum(sorted_c)
    left_sum = cum[:-1]
    right_sum = total - left_sum
    left_n = np.arange(1, n)
    right_n = n - left_n

    ss_between = left_sum**2 / left_n + right_sum**2 / right_n - total**2 / n

    lo = max(1, int(min_side))
    hi = n - lo
    if hi < lo:
        return 0.0, np.zeros(n, dtype=int)
    k_best = lo + int(np.argmax(ss_between[lo - 1 : hi]))

    ss_total = float(np.sum((sorted_c - mean) ** 2))
    var_inter = float(ss_between[k_best - 1]) / (ss_total + 1e-12)

    labels = np.zeros(n, dtype=int)
    labels[order[k_best:]] = 1
    return var_inter, labels


def _pc1_local(B: np.ndarray, nz: np.ndarray, fallback: np.ndarray):
    """PC1 local del bloque, calculada SOLO con filas no-cero.

    Devuelve (v, degenerada). Sin RNG: `eigh` de una matriz simetrica es
    determinista; el signo se canoniza exigiendo positivo el componente de
    mayor valor absoluto (la particion es invariante al signo).
    """
    if int(nz.sum()) < 3:
        return fallback, True
    Bn = B[nz]
    Bc = Bn - Bn.mean(axis=0, keepdims=True)
    cov = (Bc.T @ Bc) / max(1, Bc.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    l1 = float(vals[order[0]])
    l2 = float(vals[order[1]]) if len(order) > 1 else 0.0
    if not np.isfinite(l1) or l1 <= 1e-12 or (l1 - l2) <= 1e-9 * max(l1, 1e-12):
        return fallback, True  # subespacio degenerado: eje global canonico
    v = np.asarray(vecs[:, order[0]], dtype=np.float64)
    k = int(np.argmax(np.abs(v)))
    if v[k] < 0:
        v = -v
    return v, False


def dense_refine(B: np.ndarray, labels: np.ndarray, floor: int, max_iter: int = 5):
    """Refinamiento por centroide: Lloyd determinista con reversion al piso.

    Espejo de `_intercambio` + reversion de main_workflow: si una iteracion
    rompe el piso (o vacia una clase) se descarta y se conserva la particion
    anterior. Sin RNG, sin semillas.
    """
    cur = labels.copy()
    reverted = 0
    for _ in range(int(max_iter)):
        m0 = cur == 0
        m1 = cur == 1
        if not m0.any() or not m1.any():
            break
        c0 = B[m0].mean(axis=0)
        c1 = B[m1].mean(axis=0)
        d0 = np.einsum("ij,ij->i", B - c0, B - c0)
        d1 = np.einsum("ij,ij->i", B - c1, B - c1)
        new = (d1 < d0).astype(int)
        if np.array_equal(new, cur):
            break
        if int(np.bincount(new, minlength=2).min()) < int(floor):
            reverted += 1
            break
        cur = new
    return cur, reverted


def r2_dense(B: np.ndarray, labels: np.ndarray) -> float:
    """1 - SSE/TSS en las d dims de la matriz reducida (R2 del corte denso)."""
    mu = B.mean(axis=0, keepdims=True)
    tss = float(np.sum((B - mu) ** 2))
    if tss <= 0:
        return 0.0
    sse = 0.0
    for cls in np.unique(labels):
        Bc = B[labels == cls]
        sse += float(np.sum((Bc - Bc.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - sse / tss
```

#### 12.3.2 `dense_cdh`, `ward_backend`, `kmeans_spherical`

Copia literal de `c_backends_arena.py:344-520`. Firmas idénticas a las que
`guarded_fit` ya invoca (posicionales), así que el hunk no toca #72.

```python
def dense_cdh(
    X: np.ndarray,
    target_k: int,
    floor: int,
    min_r2_1d: float = 0.0,
    min_r2_dense: float = 0.0,
    refine_iter: int = 5,
    max_depth: int = 10,
) -> Tuple[np.ndarray, Dict]:
    """Biseccion divisiva best-first por PC1 local + corte 1-D (sin -1).

    A diferencia de la recursion pura de DENSE_CDH_DESIGN.md, este esquema
    elige en cada paso la hoja con mayor ganancia (r2_denso) entre TODAS las
    hojas: eso da control real de K (se para al llegar a target_k) y deja los
    splits mas debiles para el final. Cada hoja se evalua una sola vez, asi que
    el coste total es O(n*d^2*profundidad) y no O(n^2).

    `min_r2_1d` es el gate del eje (por defecto 0 = apagado): se midio que en
    ruido isotropico el corte 1-D optimo explica ~2/pi = 0.637, asi que un
    umbral tipo min_r2_threshold=0.05 NO discrimina. `min_r2_dense` es el gate
    calibrado sobre el R2 denso (0.08 en produccion; ver BACKEND_C.md).
    """
    n, d = X.shape
    nz = np.abs(X).sum(axis=1) > 1e-12
    fallback = np.zeros(d)
    fallback[0] = 1.0
    stats = {
        "r2_1d": [],
        "r2_dense": [],
        "depths": [],
        "degenerate_pc1": 0,
        "reverted_floor": 0,
        "rejected_min_r2": 0,
        "rejected_min_r2_dense": 0,
        "splits": 0,
    }

    def evaluate(leaf: Dict) -> Optional[Dict]:
        idx = leaf["idx"]
        if len(idx) < 2 * int(floor) or leaf["depth"] >= int(max_depth):
            return None
        B = X[idx]
        m = nz[idx]
        v, degen = _pc1_local(B, m, fallback)
        center = B[m].mean(axis=0) if int(m.sum()) > 0 else np.zeros(B.shape[1])
        coord = (B - center) @ v
        r2_1d, lab = corte_optimo_1d(coord, min_side=int(floor))
        if len(np.unique(lab)) < 2:
            return None
        if r2_1d < float(min_r2_1d):
            return {"rejected": True, "r2_1d": float(r2_1d)}
        lab_ref, reverted = dense_refine(B, lab, int(floor), refine_iter)
        if int(np.bincount(lab_ref, minlength=2).min()) < int(floor):
            lab_ref = lab  # reversion al piso (espejo de main_workflow:4144-4147)
            reverted += 1
        return {
            "rejected": False,
            "labels": lab_ref,
            "r2_1d": float(r2_1d),
            "r2_dense": float(r2_dense(B, lab_ref)),
            "degenerate_pc1": bool(degen),
            "reverted": int(reverted),
        }

    leaves: List[Dict] = [
        {"idx": np.arange(n), "depth": 0, "cand": None, "done": False}
    ]
    n_zero_rows = int((~nz).sum())

    if n >= 2 * int(floor):
        while len(leaves) < int(target_k):
            best_key, best_i, best_c = None, None, None
            for i, leaf in enumerate(leaves):
                if not leaf["done"]:
                    leaf["cand"] = evaluate(leaf)
                    leaf["done"] = True
                    c = leaf["cand"]
                    if c is not None and c.get("rejected"):
                        stats["rejected_min_r2"] += 1
                c = leaf["cand"]
                if c is None or c.get("rejected"):
                    continue
                key = (float(c["r2_dense"]), -int(leaf["idx"][0]))
                if best_key is None or key > best_key:
                    best_key, best_i, best_c = key, i, c
            if best_c is None:
                break
            # Gate calibrado sobre el R2 denso. Best-first ya eligio la hoja de
            # MAYOR ganancia: si esa no pasa el umbral, ninguna otra pasa.
            if float(best_c["r2_dense"]) < float(min_r2_dense):
                stats["rejected_min_r2_dense"] += 1
                break
            leaf = leaves[best_i]
            idx = leaf["idx"]
            lab = best_c["labels"]
            left, right = idx[lab == 0], idx[lab == 1]
            stats["r2_1d"].append(best_c["r2_1d"])
            stats["r2_dense"].append(best_c["r2_dense"])
            stats["depths"].append(int(leaf["depth"]))
            stats["degenerate_pc1"] += int(best_c["degenerate_pc1"])
            stats["reverted_floor"] += int(best_c["reverted"])
            stats["splits"] += 1
            leaves[best_i] = {
                "idx": left,
                "depth": leaf["depth"] + 1,
                "cand": None,
                "done": False,
            }
            leaves.insert(
                best_i + 1,
                {"idx": right, "depth": leaf["depth"] + 1, "cand": None, "done": False},
            )

    leaves.sort(key=lambda lf: int(lf["idx"][0]))
    labels = np.full(n, -1, dtype=int)
    for label, leaf in enumerate(leaves):
        labels[leaf["idx"]] = label
    diag = {
        "bisection_depths": [int(x) for x in stats["depths"]],
        "r2_1d": [round(float(x), 4) for x in stats["r2_1d"]],
        "r2_dense": [round(float(x), 4) for x in stats["r2_dense"]],
        "mean_r2_1d": float(np.mean(stats["r2_1d"])) if stats["r2_1d"] else 0.0,
        "mean_r2_dense": float(np.mean(stats["r2_dense"]))
        if stats["r2_dense"]
        else 0.0,
        "degenerate_pc1": int(stats["degenerate_pc1"]),
        "reverted_floor": int(stats["reverted_floor"]),
        "rejected_min_r2": int(stats["rejected_min_r2"]),
        "rejected_min_r2_dense": int(stats["rejected_min_r2_dense"]),
        "n_splits": int(stats["splits"]),
        "n_zero_rows": n_zero_rows,
        "floor": int(floor),
        "min_r2_1d": float(min_r2_1d),
        "min_r2_dense": float(min_r2_dense),
    }
    return labels, diag


def ward_backend(X: np.ndarray, k: int) -> Tuple[np.ndarray, Dict]:
    """Ward sobre PCA(30). Determinista; O(n^2) en memoria (full tree)."""
    from sklearn.cluster import AgglomerativeClustering

    ac = AgglomerativeClustering(
        n_clusters=int(k), linkage="ward", compute_distances=True
    )
    labels = ac.fit_predict(X).astype(int)
    heights = np.asarray(ac.distances_, dtype=float)
    n = X.shape[0]
    used = heights[: max(0, int(n) - int(k))]
    diag = {
        "merge_heights": [round(float(h), 4) for h in used.tolist()],
        "cut_height": float(used[-1]) if len(used) else 0.0,
        "mean_merge_height": float(used.mean()) if len(used) else 0.0,
        "n_merges": int(len(used)),
        "linkage": "ward",
        "metric": "euclidean",
    }
    return labels, diag


def kmeans_spherical(X: np.ndarray, k: int) -> Tuple[np.ndarray, Dict]:
    """L2-normalizar + KMeans(n_init=10, random_state=42). K exacto."""
    from sklearn.cluster import KMeans

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    Xn = X / norms
    km = KMeans(n_clusters=int(k), n_init=10, random_state=SEED)
    labels = km.fit_predict(Xn).astype(int)
    diag = {
        "inertia": round(float(km.inertia_), 4),
        "n_iter": int(km.n_iter_),
        "n_init": 10,
        "random_state": SEED,
        "normalized_input": True,
    }
    return labels, diag
```

#### 12.3.3 `hdbscan_backend` — firma extendida (único cambio de cuerpo)

Único cuerpo que **no** es verbatim. La arena apaga `gen_min_span_tree` y
`prediction_data` sólo por velocidad (`c_backends_arena.py:526-538`);
producción los deja en `True` (`main_workflow.py:5181-5182`). Los defaults se
fijan al valor de producción y los cuatro parámetros quedan inyectables por
kwarg, para no perder `hdbscan_uce_epsilon` / `hdbscan_uce_metric`.

```diff
@@ -523,11 +523,20 @@
-def hdbscan_backend(X: np.ndarray, min_cluster_size: int, min_samples: int):
+def hdbscan_backend(
+    X: np.ndarray,
+    min_cluster_size: int,
+    min_samples: int,
+    *,
+    epsilon: float = 0.0,
+    metric: str = "euclidean",
+    gen_min_span_tree: bool = True,
+    prediction_data: bool = True,
+):
     """HDBSCAN plano: el Method C actual. No controla K; puede emitir -1.

     Se usa gen_min_span_tree=False y prediction_data=False solo por velocidad
     (no cambian las etiquetas); produccion los deja en True.
     """
     import hdbscan

     clusterer = hdbscan.HDBSCAN(
         min_cluster_size=int(min_cluster_size),
         min_samples=int(min_samples),
@@ -534,5 +543,5 @@
-        cluster_selection_epsilon=0.0,
-        metric="euclidean",
-        gen_min_span_tree=False,
-        prediction_data=False,
+        cluster_selection_epsilon=float(epsilon),
+        metric=str(metric),
+        gen_min_span_tree=bool(gen_min_span_tree),
+        prediction_data=bool(prediction_data),
     )
```

`guarded_fit` lo llama posicionalmente (`hdbscan_backend(X, mcs, ms)`), así que
en la práctica usa `epsilon=0.0` / `metric="euclidean"`. Ver §12.6 punto 4.

#### 12.3.4 `build_method_c_result` — ensamblador puro del contrato §1

El payload de §1 se construye en una función pura y no en el método, para que
`tests/test_method_c_backends.py` pueda verificar el **mismo conjunto de
claves** de los cuatro backends sin importar `reii.main_workflow` (Stanza).
El dispatcher (§12.4) sólo la consume.

```python
def build_method_c_result(
    labels: np.ndarray,
    diag: Dict[str, Any],
    uce_ids: Sequence[str],
    matrix: np.ndarray,
    target_k: int,
    floor: int,
    *,
    requested: Optional[str] = None,
    diagnostic_only: bool = False,
    resolution_notice: str = "",
) -> Optional[Dict[str, Any]]:
    """Ensambla el dict de resultado de Method C (BACKEND_C.md §1).

    Puro: no imprime ni toca el pipeline. Devuelve ``None`` cuando no hay
    particion (``n_clusters < 2``), mismo contrato que ``guarded_fit``.
    ``partition_sizes`` excluye ``-1``; ``n_assigned`` es el denominador
    explicito del gate de dominancia (§5).
    """
    labels = np.asarray(labels, dtype=int).ravel()
    assigned = labels >= 0
    n_assigned = int(assigned.sum())
    n_noise = int(labels.shape[0] - n_assigned)
    n_clusters = int(np.unique(labels[assigned]).size)
    if n_clusters < K_MIN:
        return None

    be = str(diag.get("backend_used", diag.get("backend_requested", "")))
    bdiag = diag.get("backend_diagnostics") or {}
    sizes = sorted(int(c) for c in np.bincount(labels[assigned]))
    pre = diag.get("precondition") or {}
    return {
        "method": "embedding",
        "min_cluster_size": int(floor),
        "uce_to_label": {
            uce_ids[i]: int(labels[i])
            for i in range(min(len(uce_ids), int(labels.shape[0])))
            if labels[i] >= 0
        },
        "labels_array": labels,
        "uce_ids": list(uce_ids),
        "embedding_matrix": matrix,
        "cluster_persistence": [float(p) for p in bdiag.get("cluster_persistence", [])],
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "backend": be,
        "target_k": int(target_k),
        "partition_sizes": sizes,
        "n_assigned": n_assigned,
        "diagnostic_only": bool(diagnostic_only),
        "diagnostics": {
            "method_c": {
                "backend_resolution": {
                    "requested": str(
                        diag.get("backend_requested", "") if requested is None else requested
                    ),
                    "effective": be,
                    "reason": str(pre.get("reason", "")),
                    "notice": str(resolution_notice),
                },
                "precondition": pre,
                "notices": list(diag.get("notices", [])),
            },
            "backend_diagnostics": bdiag,
        },
    }
```

### 12.4 Hunk #22.1-#22.5 — `src/reii/main_workflow.py`

#### #22.0 imports

Ancla: L80-86 (`from reii.gramatica import ...` / `from reii.method_selection
import (...)`).

```diff
@@ -80,2 +80,9 @@
 from reii.gramatica import load_gramatica
+from reii.method_c_policy import (
+    build_method_c_result,
+    corte_optimo_1d as _corte_optimo_1d,
+    format_target_k_notice,
+    guarded_fit,
+    resolve_method_c_backend,
+)
 from reii.method_selection import (
```

#### #22.1 Extracción del corte 1-D (sin cambio de comportamiento)

El cuerpo de `_corte_optimo` (L3831-3871) se mueve a
`method_c_policy.corte_optimo_1d` (§12.3.1, copia literal) y el método queda
como delegado. Hay **una** implementación: el CDH de A y `dense_cdh` usan la
misma. `_partition` (L4123) y todo lo demás no cambia.

```diff
--- a/src/reii/main_workflow.py
+++ b/src/reii/main_workflow.py
@@ -3831,41 +3831,11 @@ class ClasificadorDescendente:
     def _corte_optimo(
         self, coord: np.ndarray, min_side: int = 1
     ) -> Tuple[float, np.ndarray]:
-        """Corte que maximiza SS_between restringido a k en [min_side, n-min_side].
-
-        min_side > 1 evita cortes degenerados (1 vs n-1), que son la causa de
-        clases de 1-2 UCs y de K inestable entre corridas.
-        """
-        n = len(coord)
-        if n < 2:
-            return 0.0, np.zeros(n, dtype=int)
-
-        order = np.argsort(coord)
-        sorted_c = coord[order]
-        total = sorted_c.sum()
-        mean = total / n
-
-        # Between-class sum of squares (unnormalized)
-        cum = np.cumsum(sorted_c)
-        left_sum = cum[:-1]
-        right_sum = total - left_sum
-        left_n = np.arange(1, n)
-        right_n = n - left_n
-
-        ss_between = left_sum**2 / left_n + right_sum**2 / right_n - total**2 / n
-
-        # Cortes factibles: k UCs a la izquierda, n-k a la derecha, ambos
-        # lados con al menos min_side UCs.
-        lo = max(1, int(min_side))
-        hi = n - lo
-        if hi < lo:
-            return 0.0, np.zeros(n, dtype=int)
-        k_best = lo + int(np.argmax(ss_between[lo - 1 : hi]))
-
-        # Normalize: var_inter = SS_between / SS_total (= R²)
-        ss_total = float(np.sum((sorted_c - mean) ** 2))
-        var_inter = float(ss_between[k_best - 1]) / (ss_total + 1e-12)
-
-        labels = np.zeros(n, dtype=int)
-        labels[order[k_best:]] = 1
-        return var_inter, labels
+        """Corte optimo 1-D con piso. Delegado a la funcion pura.
+
+        El cuerpo se extrajo a `method_c_policy.corte_optimo_1d` (misma
+        matematica, misma reversion, mismo orden de desempate) para que el
+        backend `dense_cdh` use EXACTAMENTE el mismo corte que el CDH de A.
+        `min_side > 1` evita cortes 1-vs-(n-1).
+        """
+        return _corte_optimo_1d(coord, min_side=min_side)
```

#### #22.2 Dispatcher (firma pública intacta)

Llamadores reales (verificado con `grep`): **sólo** L6124 y L6131, ambos con
`uces_por_doc, mcs_*, retrofitted_vectors, min_samples=ms_*` posicionales +
kwarg. La firma conserva esos cuatro parámetros; `variant` es keyword-only con
default, así que ningún llamador se rompe si no se actualiza.

```diff
@@ -5089,25 +5089,32 @@ class DoubleClassifier:
     def _clasificar_umbral_embedding(
         self,
         uces_por_doc: List[List[UCE]],
         min_cluster_size: int,
         retrofitted_vectors: Dict[str, np.ndarray],
         min_samples: Optional[
             int
         ] = None,  # ([hdbscan-scale-patch]) resuelto por el caller
+        *,
+        variant: str = "tight",  # "tight" | "loose": solo cambia el piso
     ) -> Optional[Dict]:
         """
-        Method C: direct HDBSCAN clustering of all UCEs in retrofitted space.
+        Method C: clasificacion densa de las UCEs en espacio retrofitted.

-        No UC objects are created here; cluster labels are assigned per-UCE.
-        The result dict carries 'uce_to_label' instead of 'ucs'/'labels',
-        making it structurally distinct from Methods A and B — handled
-        explicitly in _extract_uce_labels and _hungarian_stability_direct.
+        El backend lo elige `Config.method_c_backend` (hunk #21) y lo resuelve
+        `method_c_policy` (hunk #72): PREFLIGHT, guarda de memoria de ward,
+        acotado de K y validacion de salida NO se reimplementan aqui.

-        Quality signal: HDBSCAN cluster_persistence_ (higher = more stable).
+        No UC objects are created here; cluster labels are assigned per-UCE.
+        The result dict carries 'uce_to_label' instead of 'ucs'/'labels'
+        (BACKEND_C.md §1).
         """
-        if not _HDBSCAN_AVAILABLE:
-            print("   [EMB] hdbscan not installed — skipping Method C.")
-            return None
+        requested = getattr(self.config, "method_c_backend", "dense_cdh")
+        backend, _notice = resolve_method_c_backend(requested, source="persisted")
+        if _notice:
+            print(f"   {_notice['message']}")
+        if backend == "hdbscan" and not _HDBSCAN_AVAILABLE:
+            print("   [EMB] hdbscan not installed - skipping Method C.")
+            return None

         we = self.uc_builder.vectorizer.we
         sub = self.uc_builder.vectorizer.sublex
```

El bloque PCA (L5149-5163) y el rechazo de `zero_rows` (L5124-5131) quedan
**intactos**: el backend consume la matriz post-PCA, no la reconstruye (§8).

#### #22.3 Cuerpos: sustitución del bloque HDBSCAN por el dispatcher

Se reemplaza desde `# Si el caller no resolvió min_samples` (L5165) hasta el
`return {...}` final (L5204-5214).

```diff
@@ -5165,50 +5172,83 @@ class DoubleClassifier:
-        # Si el caller no resolvió min_samples explícitamente (pasos nuevos
-        # SIEMPRE lo resuelven con _resolve_hdbscan_uce_params), cae al
-        # mismo criterio escalado — nunca al 1 fijo de antes. ([hdbscan-scale-patch])
-        if min_samples is None:
-            ratio = float(getattr(self.config, "hdbscan_uce_min_samples_ratio", 0.25))
-            explicit = getattr(self.config, "hdbscan_uce_min_samples", None)
-            min_samples = (
-                max(1, int(explicit))
-                if explicit is not None
-                else max(2, round(ratio * min_cluster_size))
-            )
-        clusterer = hdbscan.HDBSCAN(
-            min_cluster_size=min_cluster_size,
-            min_samples=min_samples,
-            cluster_selection_epsilon=getattr(self.config, "hdbscan_uce_epsilon", 0.0),
-            metric=getattr(self.config, "hdbscan_uce_metric", "euclidean"),
-            gen_min_span_tree=True,
-            prediction_data=True,
-        )
-        labels = clusterer.fit_predict(mat)
-
-        real = np.unique(labels[labels >= 0])
-        n_noise = int(np.sum(labels == -1))
-        persist = list(getattr(clusterer, "cluster_persistence_", []))
-
-        print(
-            f"   [EMB min_cluster_size={min_cluster_size}] "
-            f"UCEs={len(all_uces)}, clusters={len(real)}, "
-            f"noise={n_noise}, persistence={[round(p, 3) for p in persist]}"
-        )
-
-        if len(real) < 2:
-            return None
-
-        # uce_to_label: only non-noise UCEs
-        uce_to_label: Dict[str, int] = {
-            uce_ids[i]: int(labels[i]) for i in range(len(uce_ids)) if labels[i] >= 0
-        }
-
-        return {
-            "method": "embedding",
-            "min_cluster_size": min_cluster_size,
-            "uce_to_label": uce_to_label,  # ← per-UCE, no UC wrapper
-            "labels_array": labels,  # full array aligned with uce_ids
-            "uce_ids": uce_ids,
-            "embedding_matrix": mat,
-            "cluster_persistence": persist,
-            "n_clusters": int(len(real)),
-            "n_noise": n_noise,
-        }
+        cfg = self.config
+
+        # Piso de hoja: el mcs del pase (ya escalado al corpus) es el piso de
+        # los backends densos; el pase loose lo reduce por
+        # dense_cdh_loose_factor, nunca por debajo del floor absoluto (§3).
+        _floor_min = int(getattr(cfg, "hdbscan_uce_min_cluster_floor", 5))
+        floor = max(_floor_min, int(round(float(min_cluster_size))))
+        if variant == "loose":
+            _factor = float(getattr(cfg, "dense_cdh_loose_factor", 0.75))
+            floor = max(_floor_min, int(round(floor * _factor)))
+
+        # K pedido: explícito o recuperado del plumbing existente
+        # (min_cluster_size_cdh = 1/(2K) => K = round(1/(2*frac))).
+        target_k = getattr(cfg, "method_c_target_k", None)
+        if target_k is None:
+            _frac = getattr(cfg, "min_cluster_size_cdh", None)
+            _frac = 0.10 if _frac is None else float(_frac)
+            target_k = int(round(1.0 / (2.0 * max(1e-6, _frac))))
+
+        # Unica llamada a los backends: las guardas viven en la politica.
+        labels, _diag = guarded_fit(
+            backend,
+            mat,
+            int(target_k),
+            floor,
+            n_zero_rows=zero_rows,
+            ward_max_n=_METHOD_C_WARD_MAX_N,
+            min_r2_1d=float(getattr(cfg, "dense_cdh_min_r2_1d", 0.0)),
+            min_r2_dense=float(getattr(cfg, "dense_cdh_min_r2_dense", 0.08)),
+            # mcs_frac reproduce el min_cluster_size del pase para hdbscan.
+            mcs_frac=float(min_cluster_size) / float(max(1, len(all_uces))),
+        )
+        for _msg in _diag.get("notices", []):
+            print(f"   {_msg}")
+
+        if labels is None:
+            _be = str(_diag.get("backend_used", backend))
+            # I3 (HARDENING §9): target_k -> n_clusters tambien cuando no hay
+            # particion. guarded_fit ya dejo n_clusters en diag.
+            _ncls = _diag.get("n_clusters")
+            if _be == "dense_cdh" and _ncls is not None:
+                _rej = int(
+                    (_diag.get("backend_diagnostics") or {}).get(
+                        "rejected_min_r2_dense", 0
+                    )
+                )
+                _k_notice = format_target_k_notice(int(target_k), int(_ncls), _rej)
+                if _k_notice:
+                    print(f"   {_k_notice}")
+            print(
+                f"   [!] Method C backend={_be} sin particion "
+                f"({_diag['precondition']['reason']})"
+            )
+            return None
+
+        # hdbscan no controla K y emite -1: se computa, no compite, salvo
+        # escape hatch (BACKEND_C.md §2, hunk #22.5).
+        _be = str(_diag["backend_used"])
+        diagnostic_only = bool(
+            _be == "hdbscan"
+            and not bool(getattr(cfg, "method_c_hdbscan_selectable", False))
+        )
+
+        result = build_method_c_result(
+            labels,
+            _diag,
+            uce_ids,
+            mat,
+            int(target_k),
+            int(floor),
+            requested=str(requested),
+            diagnostic_only=diagnostic_only,
+            resolution_notice=(_notice["message"] if _notice else ""),
+        )
+        if result is None:
+            return None
+        print(
+            f"   [EMB backend={result['backend']}] UCEs={len(all_uces)}, "
+            f"clusters={result['n_clusters']}, noise={result['n_noise']}, "
+            f"K_pedido={result['target_k']}"
+            + (", diagnostic_only" if diagnostic_only else "")
+        )
+        return result
```

`min_samples` deja de consumirse en la ruta densa (los densos no lo usan) y en
la ruta hdbscan pasa a derivarse del ratio 0.25 de la política: ver §12.6.4.

#### #22.4 Payload (§1) — qué claves quedan disponibles

No es un hunk aparte: lo produce `build_method_c_result` (§12.3.4) y lo
consumen #19 (`_raw_stats`), #20 (`context`), #23 (`bootstrap_ari`,
`coverage_stable`, `null_p`) y el auditor:

| clave                                        | consumidor                                                  |
| -------------------------------------------- | ----------------------------------------------------------- |
| `backend`                                    | `select-v4` (`backend == "hdbscan"`), auditor, paper        |
| `diagnostic_only`                            | #11/#23: excluye a C del concurso sin perder la partición   |
| `partition_sizes` / `n_assigned` / `n_noise` | #15-#17, `_max_share` (L5665-5674), gate §5                 |
| `target_k`                                   | #19 (`k_target` vs `k_ach`), aviso `format_target_k_notice` |
| `diagnostics.method_c.backend_resolution`    | auditor: `requested` / `effective` / `reason` / `notice`    |
| `diagnostics.backend_diagnostics`            | tabla §1 (`r2_dense`, `merge_heights`, `inertia`, ...)      |

Ojo: `run()` sigue leyendo `res_emb1["cluster_persistence"]` (L6277) y
`res_emb1["uce_to_label"]` (L6136): ambas se conservan, con `[]` en los densos.

#### #22.5 Sitios de llamada (`variant`)

```diff
@@ -6124,3 +6124,4 @@ class WorkflowOrchestrator:
             res_emb1 = self._clasificar_umbral_embedding(
-                uces_por_doc, mcs_tight, retrofitted_vectors, min_samples=ms_tight
+                uces_por_doc, mcs_tight, retrofitted_vectors,
+                min_samples=ms_tight, variant="tight"
             )
@@ -6131,3 +6131,4 @@ class WorkflowOrchestrator:
             res_emb2 = self._clasificar_umbral_embedding(
-                uces_por_doc, mcs_loose, retrofitted_vectors, min_samples=ms_loose
+                uces_por_doc, mcs_loose, retrofitted_vectors,
+                min_samples=ms_loose, variant="loose"
             )
```

### 12.5 Orden de aplicación y verificación

1. #72 (`method_c_policy.py`, HARDENING §8.1) → 2. **#72b** (§12.3) → 3. #69
   (§12.2) → 4. #21 (§12.1) → 5. #22.0 imports → 6. #22.1 corte → 7. #22.2
   firma → 8. #22.3 dispatcher → 9. #22.5 llamadas. Aplicar #22 antes de #72 deja
   `ImportError` en el import de L80 (módulo inexistente), no un fallo silencioso.

```
python scripts/playground/c_backends_arena.py          # ~16-19 s, exit 0
python -m pytest tests/test_method_c_backends.py -q    # skipped (hoy) / verde
python -m pytest tests/test_backend_hardening.py -q    # HARDENING §9
```

### 12.6 Discrepancias doc ↔ código (declaradas, no corregidas en otro sitio)

1. **Premisa de la extracción.** No hay corte 1-D dentro de
   `_clasificar_umbral_embedding` (L5089): ese método es hoy un envoltorio
   estático de HDBSCAN (L5176-5214) y no contiene ninguna bisección. El «1-D
   cut» que se extrae es `ClasificadorDescendente._corte_optimo`
   (**L3831-3871**, usado en L4123). Extraerlo de L5089 habría sido inventar
   código.
2. **Default de backend.** `Config.method_c_backend = "dense_cdh"` (§12.1) vs
   `DEFAULT_METHOD_C_BACKEND = "hdbscan"` (`hardening_bench.py:94`). No es
   contradicción: el segundo es el destino de degradación de un valor
   inválido, no el default de producto.
3. **Tope de ward.** HARDENING §2.2 propone `6000`; el código de referencia
   trae `DEFAULT_WARD_MAX_N = 4000` (`hardening_bench.py:117`). Se aplica 6000
   en `_METHOD_C_WARD_MAX_N` y `guarded_fit(ward_max_n=...)` lo sobreescribe en
   cada llamada, así que la constante del módulo no interviene.
4. **`guarded_fit` acota la inyección de hdbscan.** Llama
   `hdbscan_backend(X, mcs, max(2, round(0.25*mcs)))`
   (`hardening_bench.py:481-482`), así que `hdbscan_uce_min_samples`
   explícito, `hdbscan_uce_epsilon` y `hdbscan_uce_metric` **no** llegan al
   clusterer: se usan 0.25/0.0/"euclidean". Con los defaults de `Config`
   (`min_samples=None` ⇒ ratio 0.25, epsilon 0.0, metric euclidean,
   L1034-1038) el resultado es idéntico; sólo cambia si alguien activó esos
   overrides. Arreglo correcto (fuera de alcance): añadir `**hdbscan_kwargs` a
   `guarded_fit`.
5. **El aviso de backend inválido se pierde en `preflight`.**
   `preflight` hace `backend, _notice = resolve_method_c_backend(...)` y
   **descarta** `_notice` (`hardening_bench.py:252-253`), así que
   `guarded_fit("banana", ...)` corre hdbscan en silencio. El dispatcher llama
   `resolve_method_c_backend` antes (§12.4, #22.2) y emite el `[!]`; si alguien
   usa `guarded_fit` sin pasar por ahí, no hay aviso.
6. **I3 vs orden de los hunks de #72.** El hunk de «K=1 deja de ser partición»
   (HARDENING §8.1, `hardening_bench.py:502`) retorna **antes** del
   `format_target_k_notice` de L507-509, de modo que el aviso
   `target_k=8 -> 1 clases` se pierde justo en el caso que I3 exige. El
   dispatcher lo re-emite en la rama `labels is None` (§12.4, #22.3).
7. **Calibración de `r2_dense` dependiente de d.** El gate 0.08 está medido a
   **d=30**. Con d menor el R² de un corte se dispara: medido aquí con
   `n=500, d=10`, gaussiana isótropa, `rng(42)` ⇒ `mean_r2_dense=0.142` y
   `dense_cdh` devuelve **K=8 sobre ruido puro** (no lo corta el gate). Como
   producción usa `target_dims = max(2, min(30, n//10, d))` (L5150), corpus con
   n≲200 corren con <20 dims y el gate queda laxo. El playground fuerza 30
   (`BACKEND_C` §10) y por eso no lo vio. Mitigación disponible sin cambiar el
   backend: `dense_cdh_min_r2_dense` es configurable.
8. `BACKEND_C.md` §1 declara `diagnostics` «por backend»; aquí se anida como
   `diagnostics.backend_diagnostics` porque `diagnostics.method_c` está ocupado
   por `backend_resolution` (pedido por #20/#23). Ambas claves conviven.
9. El guard `_HDBSCAN_AVAILABLE` (L204-206, comprobado en L5108) se mueve de la
   cabecera del método a la rama `hdbscan`: con `dense_cdh`/`ward`/`kmeans`
   Method C funciona sin el paquete `hdbscan` instalado (antes devolvía `None`
   incluso para un backend que no lo necesita).
