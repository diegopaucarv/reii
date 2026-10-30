# PAPER_AMENDMENTS — registro de cambios de `docs/paper.md` (v5)

Fecha: 2026-10-28. Ámbito: **solo redacción**. No se ha modificado `src/**`, `tests/**`,
`scripts/**`, `docs/patches/**` ni `data/**`.

Los números de línea de cada entrada se refieren a `docs/paper.md` **antes** de aplicar
estas enmiendas (estado base). Al insertar o sustituir texto, las líneas posteriores se
desplazan; por eso cada entrada identifica también el texto anterior de forma literal.

Criterio: solo se modifican afirmaciones que contradicen el código o la implementación
efectivamente verificada. Lo que no pudo verificarse se conserva y se marca con
`TODO` en el propio texto del artículo (ver §Notas).

> **Nota de vigencia (2026-10-28).** Las entradas anteriores a §«Patch `14` — APLICADO»
> registran estados previos del artículo. En particular, las cifras de retención
> (0,395 / 0,451) y la etiqueta «opt-in» de `tool_adverb_filtered` quedaron **superadas**
> por el patch 14: la redacción vigente es la de esa sección final (default =
> `tool_adverb_filtered`, 0,451; `pos_only` = opt-in, 0,395).

---

## §3.2 — Método de representación vectorial (L65)

**Ubicación:** §3.2, línea 65 (título de la subsección dentro del bloque de índice).

**Antes:**

```text
3.2\. Proyección vectorial densa mediante contextual embeddings
```

**Después:**

```text
3.2\. Proyección vectorial densa mediante embeddings estáticos fastText con retroadaptación (retrofitting; 300 d, hasta 30 componentes tras PCA)
```

**Motivo y fuente:** el pipeline no usa embeddings contextuales. El método C construye
cada vector de UCE como media ponderada de los vectores fastText **retroadaptados**
(_retrofitted_) de 300 dimensiones, con retroceso a subpalabras fastText
(`_compute_uce_centroids_from_retro`, `src/reii/main_workflow.py`). Los vectores de 384 d
proceden de un banco de pruebas con `SentenceTransformer` sobre `data/workflow_data.json`;
no son el espacio de clasificación del método C. Fuente: `docs/design_v5/PAPER_GAPS.md`
(D02), `docs/design_v5/BACKEND_C.md` (`embedding` es 384-d, no el fastText-retrofit).

---

## §3.3 — Backend de división densa (L65)

**Ubicación:** §3.3, línea 65 (título de la subsección dentro del bloque de índice).

**Antes:**

```text
3.3\. Algoritmo de división/agrupamiento jerárquico denso (Reemplazo de la CDH clásica)
```

**Después:**

```text
3.3\. Backend de división densa del método C (dense_cdh por defecto, con gate r2_dense; alternativas ward y k-means esférico; HDBSCAN solo diagnóstico; elección pre-registrada y validada). La CDH clásica se conserva para los métodos A y B y para el modo classic.
```

**Motivo y fuente:** `src/reii/method_c_policy.py` declara cuatro backends
(`dense_cdh`, `ward`, `kmeans_spherical`, `hdbscan`) y `guarded_fit` como único punto de
despacho. `dense_cdh` es una bisección _best-first_ sobre la PC1 local con poda por
`r2_dense`, no un agrupamiento aglomerativo; `hdbscan` queda excluido de la selección por
`diagnostic_only`/`backend="hdbscan"` (no controla K y su ARI alto en blobs es artefacto
del colapso). El backend es una elección pre-registrada y validada, no "el algoritmo" del
sistema. Fuente: `docs/design_v5/BACKEND_C.md` §2, `docs/design_v5/PAPER_GAPS.md` (D03),
`src/reii/method_selection.py` (regla de viabilidad).

---

## §4.2 — Coherencia interna y diversidad (L67)

**Ubicación:** §4.2, línea 67 (título de la subsección dentro del bloque de índice).

**Antes:**

```text
4.2\. Métricas de Calidad Semántica (Coherencia Externa C\_ex, Coherencia Interna C\_in, Diversidad d)
```

**Después:**

```text
4.2\. Coherencia interna `C_in` (coseno UCE-centroide, macro, clip [0,1], con nulo de etiquetas permutadas) y diversidad de la partición `d` (H/ln K, Simpson, `K_eff` de Hill y `max_share`; mide tamaños de clase, no contenido semántico). Medidas en el espacio de embeddings del banco de pruebas (384 d, SentenceTransformer), no en el de 300 d del método C: `C_in` = 0,8472, IC95 [0,8374, 0,8593], nulo 0,8211 (exceso +0,0261, z = 16,36, p = 0,000); `d` = 0,9704, nulo Dirichlet(1) = 0,8590 (q05 0,7753, q95 0,9228), p = 0,000, Simpson 0,9314, `K_eff` 15,63, `max_share` 0,1104; n = 154 de 528 UCEs, K = 17. `C_ext` solo si existe referencia humana.
```

**Motivo y fuente:** `C_in` es el coseno medio (macro, truncado a [0,1]) de cada UCE a su
centroide de clase, no una homogeneidad de términos; su valor absoluto no es
interpretable porque el nulo de etiquetas permutadas ya vale 0,8211. `d` depende solo del
vector de tamaños de clase (H/ln K), no del contenido semántico. Cifras medidas sobre
`data/workflow_data.json` (528 UCEs; 154 con embedding; K = 17) con
`quality/metrics.py::internal_coherence` y `partition_diversity`. Fuente:
`docs/design_v5/METRICS_PAPER.md` §1-§2, `docs/design_v5/PAPER_GAPS.md` (D12-D13).

---

## §4.3 — RMSD y validez externa (L69)

**Ubicación:** §4.3, línea 69 (título de la subsección dentro del bloque de índice).

**Antes:**

```text
4.3\. Métricas de Validez Externa y Proporción Discursiva (RMSD, Cohen's Kappa, Inter-Coder Reliability)
```

**Después:**

```text
4.3\. Concordancia de perfiles de proporción (RMSD) y validez externa solo con estándar humano (`C_ext`, Cohen's Kappa, inter-coder). RMSD = 0,0300, IC95 [0,0242, 0,0451], `k_ref` = `k_obs` = 17; nulo de permutación 0,0364 (sd 0,0047, q95 0,0426), p = 0,910; partición aleatoria con el mismo perfil 0,0327. No distingue acuerdo de asignación: se reporta como descriptor junto a ARI/NMI, nunca como validez externa.
```

**Motivo y fuente:** `RMSD` es una distancia de perfiles de proporción con
emparejamiento húngaro; el nulo de permutación da p = 0,910 y una partición aleatoria con
el mismo perfil puntúa 0,0327 frente a 0,0300. No mide acuerdo de asignación. Cifras
medidas en el mismo banco de pruebas (`rmsd_proportions`). Fuente:
`docs/design_v5/METRICS_PAPER.md` §3.2-§3.4, `docs/design_v5/PAPER_GAPS.md` (D14).

---

## §4.1 — Piso empírico del ARI (L97)

**Ubicación:** §4.1 (tabla de fundamento de métricas), fila **ARI**, línea 97.

**Antes:**

```text
| **ARI** (Adjusted Rand Index)           | Concordancia entre dos particiones (ej. dos corridas independientes) | `ARI ∈ [0,1]`                                                         | Qué bien se alinean dos clasificaciones. Si ARI = 1: misma clasificación. Si ARI ≈ 0: sin relación.                                                |
```

**Después:**

```text
| **ARI** (Adjusted Rand Index)           | Concordancia entre dos particiones (ej. dos corridas independientes) | `ARI ∈ [0,1]`                                                         | Qué bien se alinean dos clasificaciones. Si ARI = 1: misma clasificación. Si ARI ≈ 0: sin relación. Con la clase mayor al 0,837 del corpus, un predictor trivial que copia esa clase obtiene ARI ≈ 0,72 (K = 3) y 0,903 (K = 8): no se reporta sin contraste contra el nulo. |
```

**Motivo y fuente:** con una clase mayor que ocupa 0,837 del corpus, un predictor trivial
que copia la clase mayor obtiene ARI ≈ 0,72 (K = 3) y 0,903 (K = 8); el ARI no es
interpretable sin un nulo. Cifras del contraste trivial medido sobre la misma estructura
de 442/528 UCEs. Fuente: `src/reii/method_selection.py` (viabilidad y criterio de
estabilidad), `docs/design_v5/STABILITY_NULL.md`.

---

## §4.2 — Definición de `C_in` (L101)

**Ubicación:** §4.1/§4.2 (tabla de fundamento de métricas), fila **Coherencia Interna**, línea 101.

**Antes:**

```text
| **Coherencia Interna (C_in)**           | ¿Qué tan homogéneos están los términos dentro de cada clase?         | `C_in ∈ [0,1]`                                                        | ¿Tienen los términos de una misma clase palabras en común?                                                                                         |
```

**Después:**

```text
| **Coherencia Interna (C_in)**           | Coseno medio (macro, clip [0,1]) de cada UCE al centroide de su clase, en un espacio declarado | `C_in ∈ [0,1]`                                                        | Coherencia geométrica UCE-centroide. Se reporta con su nulo de etiquetas, su n y su espacio; el valor absoluto no es interpretable por sí solo.       |
```

**Motivo y fuente:** la métrica implementada en `src/reii/quality/metrics.py`
(`internal_coherence`) opera sobre UCEs y centroides, no sobre términos. Fuente:
`docs/design_v5/METRICS_PAPER.md` §1.1-§1.4, `docs/design_v5/PAPER_GAPS.md` (D12).

---

## §3 — Paso 2: número de métodos (L171-175)

**Ubicación:** §3, algoritmo paso a paso, bloque **Paso 2**, líneas 171-175.

**Antes:**

```text
    ├── Paso 2: Dualidad de Construcción (4 métodos)
    │         ├── Método A: Word Count (Reinert clásico)
    │         ├── Método B: Similaridad (UCEVectorizer)
    │         ├── Método C: Coref (ProgressiveSegmenter)
    │         └── Método D: Embedding (progressiveSegmenter)
```

**Después:**

```text
    ├── Paso 2: Construcción de UCEs con tres métodos
    │         ├── Método A: Word Count (Reinert clásico; ALCESTE)
    │         ├── Método B: Similitud/coref (ProgressiveSegmenter)
    │         └── Método C: Embeddings (centroides fastText retroadaptados)
```

**Motivo y fuente:** existen tres métodos, no cuatro: A = word-count/ALCESTE, B =
similitud/coref (`ProgressiveSegmenter`), C = embeddings. No existe un método D.
`--mode classic` es un escenario de fidelidad, no un cuarto método. Fuente:
`src/reii/main_workflow.py` (modos `wc_only`, `coref_only`, `emb_only`, `all`, `select`;
`src/reii/method_selection.py`), `docs/design_v5/CLASSIC_MODE.md`, `PAPER_GAPS.md` (D04).

---

## §3 — Paso 3: clasificación y backends (L177-179)

**Ubicación:** §3, algoritmo paso a paso, bloque **Paso 3**, líneas 177-179.

**Antes:**

```text
    ├── Paso 3: Clasificación Jerárquica
    │         └─ CLASIFICADOR 1: Clasificador Descendente (Reinert)
    │         └─ CLASIFICADOR 2: KMeans / Agglomerative
```

**Después:**

```text
    ├── Paso 3: Clasificación
    │         ├─ Métodos A y B: CDH clásica (Clasificador Descendente)
    │         └─ Método C: backend denso pre-registrado y validado
    │               ├─ dense_cdh (bisección best-first por PC1 local, gate r2_dense; por defecto)
    │               ├─ ward | k-means esférico
    │               └─ HDBSCAN (solo diagnóstico: no controla K; su "K" es min_cluster_size)
    │         └─ min_cluster_size_cdh se deriva de K (1/(2K)); no es un parámetro libre
```

**Motivo y fuente:** el método C despacha por `method_c_policy.guarded_fit` a
`dense_cdh`, `ward`, `kmeans_spherical` o `hdbscan`; `dense_cdh` es el backend por
defecto en la configuración del pipeline y usa `dense_cdh_min_r2_dense = 0.08`. `hdbscan`
es diagnóstico: no controla K. `min_cluster_size_cdh` se interpreta como `1/(2K)` y se
deriva de K, no es un parámetro libre independiente. Los métodos A y B siguen usando la
CDH clásica. Fuente: `src/reii/method_c_policy.py`, `src/reii/main_workflow.py`
(comentario de `min_cluster_size_cdh`), `docs/design_v5/BACKEND_C.md` §2-§3,
`docs/design_v5/PAPER_GAPS.md` (D03).

---

## §3 — Paso 5 y 6: veredicto triple y niveles (L185-194)

**Ubicación:** §3, algoritmo paso a paso, bloques **Paso 5** y **Paso 6**, líneas 185-194.

**Antes:**

```text
    ├── Paso 5: Triple Verdict de Estabilidad
    │         └── stable_wc: ¿estable por conteo de palabras?
    │         └── stable_sim: ¿estable por similitud?
    │         └── stable_coref: ¿estable por coreferencia?
    │         └── UCE puede ser estable en 1, 2 o 3 métodos
    │
    ├── Paso 6: Niveles de Calidad (2–3 niveles)
    │         ├── Núcleo: stable en todas las 4 formas de construcción
    │         ├── Extendido: estable en ≥ 2 formas de construcción
    │         └─ Proyectado: UCEs liminales (no estables, pero coherentes)
```

**Después:**

```text
    ├── Paso 5: Estabilidad Intra-Método y Selección
    │         └── Cada método se estabiliza con dos pases (tight/loose): A1↔A2, B1↔B2, C1↔C2
    │         └── select-v4 elige UN método por viabilidad y estabilidad (bootstrap ARI; nulo diagnóstico)
    │         └── primary_stable: estable en el método elegido
    │
    ├── Paso 6: Proyección Liminal (post-hoc)
    │         └─ UCEs no estables pero coherentes; quedan fuera del corpus analítico
```

**Motivo y fuente:** en el modo por defecto (`select`), `select-v4` elige **un** método y
su `primary_stable`; cada método se estabiliza con dos pases tight/loose (A1↔A2, B1↔B2,
C1↔C2). No existe un veredicto triple por UCE ni los niveles núcleo/extendido/proyectado;
la proyección liminal es post-hoc y no entra al corpus analítico. El modo `all` es la
intersección de los tres conjuntos estables, no una escala de niveles. Fuente:
`src/reii/method_selection.py` (`RULE_VERSION = "select-v4"`), `docs/design_v5/
STABILITY_NULL.md` §4.1, `docs/design_v5/PAPER_GAPS.md` (D05).

---

## §3 — Paso 7: métricas de calidad (L196-204)

**Ubicación:** §3, algoritmo paso a paso, bloque **Paso 7**, líneas 196-204.

**Antes:**

```text
    └── Paso 7: Métricas de Calidad
            └─ Tasa de Cobertura: UCEs Estables / UCEs Totales
            └─ Cobertura por Tokens: Términos en UCEs Estables / Términos Totales
            └─ Silhouette Score: qué bien definidas están las clases
            └─ ARI: concordancia entre el núcleo y la extensión
            └─ FMI: concordancia entre el núcleo y la extensión
            └─ C_ext: si hay referencia humana
            └─ Coherencia Interna: homogeneidad intra-clase
```

**Después:**

```text
    └── Paso 7: Métricas de Calidad
            └─ Tasa de Cobertura: UCEs Estables / UCEs Totales
            └─ Cobertura por Tokens: Términos en UCEs Estables / Términos Totales
            └─ Silhouette Score: qué bien definidas están las clases
            └─ ARI/FMI/NMI: concordancia entre particiones (tight/loose; bootstrap)
            └─ C_in: coherencia UCE-centroide, con nulo de etiquetas
            └─ d: equilibrio de tamaños de la partición (no mide contenido)
            └─ RMSD: distancia de perfiles de proporción (no mide acuerdo)
            └─ C_ext: si hay referencia humana
```

**Motivo y fuente:** no existen los niveles "núcleo/extensión"; ARI/FMI/NMI se calculan
entre particiones/corridas, no entre niveles. `C_in`, `d` y `RMSD` existen desde el patch
`01_quality_metrics.patch`. Fuente: `src/reii/quality/metrics.py`
(`cross_partition_agreement`, `internal_coherence`, `partition_diversity`,
`rmsd_proportions`), `docs/design_v5/PAPER_GAPS.md` (D16).

---

## §3 — Regla de estabilidad en la tabla de parámetros (L215)

**Ubicación:** §3, tabla «Parámetros clave de REII (propuesta)», fila **Regla de estabilidad**, línea 215.

**Antes:**

```text
| **Regla de estabilidad** | **No binaria**: tiene niveles          | Permite medir la gradación de estabilidad          |
```

**Después:**

```text
| **Regla de estabilidad** | Un método elegido + `primary_stable`   | `select-v4` elige un método; sin niveles ni consenso entre métodos |
```

**Motivo y fuente:** la regla implementada no tiene niveles: elige un método por gates de
viabilidad y estabilidad intra-método. Fuente: `src/reii/method_selection.py`,
`docs/design_v5/SELECT_V4.md`, `docs/design_v5/PAPER_GAPS.md` (D05).

---

## §3 — Tabla «Qué MEJORA REII» (L223, L226)

**Ubicación:** §3, tabla **Qué MEJORA REII**, filas de _Niveles de calidad_ y _Triple
verdict_, líneas 223 y 226.

**Antes:**

```text
| Niveles de calidad (núcleo, extendido, proyectado) | La cobertura mejora (incluye más UCEs), pero se define qué nivel es "núcleo" y cuál es "extendido"  |
```

```text
| Triple verdict de estabilidad                      | UCEs que son estables en 2 de 4 métodos son más informativos que las que son inestables en todos    |
```

**Después:**

```text
| Selección pre-registrada de un método (select-v4)  | Elige un método por gates de viabilidad y estabilidad intra-método; evita publicar una partición degenerada |
```

```text
| Estabilidad intra-método (bootstrap + tight/loose) | El ARI se compara solo dentro de su método; el nulo de permutación se reporta como diagnóstico      |
```

**Motivo y fuente:** no existen niveles ni veredicto triple; la mejora real es la
selección pre-registrada de un método y la estabilidad intra-método. Fuente:
`src/reii/method_selection.py`, `docs/design_v5/STABILITY_NULL.md`, `PAPER_GAPS.md` (D05).

---

## §3 — Tabla «Qué COMPROMETE REII» (L233-236)

**Ubicación:** §3, tabla **Qué COMPROMETE REII**, filas de costes, líneas 233-236.

**Antes:**

```text
| Incluir más UCEs (nivel extendido) | Mayor cobertura pero mayor ruido                                          |
| Consenso blando                    | Clases más grandes pero menos precisas                                    |
| Mapeo muchos a uno                 | Si fusionas clases genuinamente distintas, bajas la validez de las clases |
| Proyección liminal                 | Los UCEs proyectados tienen menor error, pero no son tão representativos  |
```

**Después:**

```text
| Gate de dominancia sobre el corpus (≤ 0,70)    | Un corpus legítimamente dominado por un tema (p₁ = 0,72) se rechaza; el umbral no se sube a 0,75.                                              |
| Selección de un solo método (select-v4)        | La selección no es un consenso entre métodos: la estabilidad es intra-método y K no es estable (K ∈ {3, 7, 11, 16} con segmentación idéntica). |
| Backend del método C pre-registrado            | HDBSCAN queda como diagnóstico: no controla K (su "K" es `min_cluster_size`) y su ARI alto en blobs es artefacto del colapso.                  |
| Filtro de formas activas (`pos_only`, default) | Retiene menos léxico que ALCESTE 2012: 10,72 lemas/UCE y 0,395 frente a 13,95 y 0,537 (brecha −3,23 lemas/UCE, −23,2 %).                       |
```

**Motivo y fuente:** los mecanismos anteriores (niveles, consenso blando, mapeo
muchos-a-uno) no existen. Los costes declarados y medidos son: (i) el gate de dominancia
sobre el corpus rechaza un corpus legítimamente dominado (p₁ = 0,72) y por eso no se sube
a 0,75; (ii) K es inestable con segmentación idéntica (target 30, `lem_total = 5658`;
K ∈ {3, 7, 11, 16}); (iii) HDBSCAN es diagnóstico; (iv) la retención léxica por defecto es
0,395 frente a 0,537 de ALCESTE 2012. Fuente: `docs/design_v5/GATE_NOISE.md` §5.3/§6,
`docs/design_v5/BACKEND_C.md` §2, `docs/design_v5/RETENTION_POLICY.md` §1.1/§3.3,
`src/reii/method_policy.py` (backends), `docs/design_v5/PAPER_GAPS.md` (D11, D21).

---

## §4 — Explicación simplificada, punto 3 (L276)

**Ubicación:** §4, «Explicación simplificada», punto 3, línea 276.

**Antes:**

```text
3. La ventaja de REII es que **no sacrifica nada**: las correcciones al mapeo y las métricas son mejores, y el inclusion de niveles extended es una opción additional que puedes usar si quieres más cobertura.
```

**Después:**

```text
3. REII no mejora todas las métricas a la vez: gana cobertura de UCEs estables y métricas con nulo de etiquetas, pero retiene menos léxico que ALCESTE 2012 (0,395 frente a 0,537; 0,451 con la política conservadora `tool_adverb_filtered`, opt-in). El default sigue siendo `pos_only`; la variante agresiva `pos_plus_propn` (0,485) queda excluida por su cuota de palabras herramienta (0,375 > 0,30).
```

**Motivo y fuente:** la retención léxica por defecto (10,72 lemas/UCE; 0,395) es inferior
a la referencia ALCESTE (13,95; 0,537); la política conservadora opt-in sube a 12,24 y
0,451, y la agresiva (13,16; 0,485) está excluida por la regla pre-registrada de cuota de
palabras herramienta. Fuente: `docs/design_v5/RETENTION_POLICY.md` §1.1/§3.3/§4.2,
`docs/design_v5/PAPER_GAPS.md` (D22).

---

## §4 — Título de la descomposición estadística (L278)

**Ubicación:** título del bloque «Por qué REII da 30% de riqueza y 51% de retención…», línea 278.

**Antes:**

```text
# Por qué REII da 30% de riqueza y 51% de retención mientras ALCESTE da 97% y 90%: descomposición estadística
```

**Después:**

```text
# Por qué la retención léxica de REII (39,5 %; 45,1 % con la política conservadora opt-in) difiere de la referencia ALCESTE 2012 (53,7 %): descomposición estadística
```

**Motivo y fuente:** la retención léxica verificada es 0,395 por defecto (0,451 con
`tool_adverb_filtered`) frente a 0,537 de ALCESTE 2012; las cifras 30 %/97 %/51 %/90 % no
tienen artefacto en el repositorio. Las de riqueza (30/97) se conservan en el cuerpo
marcadas con `TODO(verificar)`. Fuente: `docs/design_v5/RETENTION_POLICY.md` §1.1/§3.3,
`docs/design_v5/PAPER_GAPS.md` §5.

---

## §4 — Tesis central: número de efectos (L283)

**Ubicación:** §4, «Tesis central», línea 283.

**Antes:**

```text
La brecha no indica que un algoritmo sea "mejor" que el otro: es la consecuencia predecible de cinco efectos estadísticos que se acumulan.
```

**Después:**

```text
La brecha no indica que un algoritmo sea "mejor" que el otro: es la consecuencia predecible de varios efectos estadísticos y decisiones de definición que se acumulan.
```

**Motivo y fuente:** tras reescribir el Efecto 1, este ya no describe una conjunción de
seis condiciones que explique la retención, sino la regla real de selección; el recuento
fijo de «cinco efectos» deja de corresponder a la estructura del argumento. Fuente:
`src/reii/method_selection.py`, `docs/design_v5/PAPER_GAPS.md` (D10).

---

## §4 — Encabezado del Efecto 1 (L286)

**Ubicación:** subtítulo «Efecto 1», línea 286.

**Antes:**

```text
## Efecto 1 — El criterio conjuntivo (explica el 51% vs 90%)
```

**Después:**

```text
## Efecto 1 — La regla de selección no es conjuntiva
```

**Motivo y fuente:** el pipeline no aplica una conjunción de condiciones; `select-v4`
elige un método. Fuente: `src/reii/method_selection.py`, `PAPER_GAPS.md` (D10).

---

## §4 — Cuerpo del Efecto 1: eliminación de `p⁶` (L288-297)

**Ubicación:** §4, «Efecto 1», párrafos y tabla de `p^k`, líneas 288-297.

**Antes:**

```text
ALCESTE declara estable una UCE si sobrevive a **2 corridas** con distinto tamaño de ventana (p. ej. 10 vs 14 palabras). REII exige que la UCE sea estable en **6 condiciones** (wordcount, similitud, coref, embedding y sus cruces).

Si cada condición tiene probabilidad _p_ de aprobarse, la probabilidad conjunta es _p^k_:

| Sistema | Condiciones (k) | P(estable) con p = 0.90 |
| ------- | --------------- | ----------------------- |
| ALCESTE | 2               | 0.90² = **0.81**        |
| REII    | 6               | 0.90⁶ = **0.53**        |

Tu 51% es casi exactamente lo que predice un criterio de 6 condiciones con ~90% de aprobación individual. **REII no "pierde" UCEs: exige más evidencia por UCE.** Y hay un agravante: las 6 condiciones de REII no son 6 repeticiones de la misma prueba (como las 2 ventanas de ALCESTE), sino 6 pruebas _cualitativamente distintas_ (frecuencia, similitud, coreferencia, embeddings). Cada una tiene un modo de fallo diferente, así que la intersección se encoge aún más de lo que sugiere el cálculo simple.
```

**Después:**

```text
ALCESTE declara estable una UCE si sobrevive a **2 corridas** con distinto tamaño de ventana (p. ej. 10 vs 14 palabras). REII no aplica un veredicto conjunto de condiciones heterogéneas: en el modo por defecto (`select`), la regla pre-registrada `select-v4` elige **un** método entre tres (A word-count, B similitud/coref, C embeddings). La elección aplica, en orden, gates de viabilidad —el método produjo partición, K ∈ [3, 20], ruido ≤ 0,50, cobertura efectiva ≥ 0,15 y no dominancia sobre el corpus (c ≤ 0,70)— y un criterio de estabilidad intra-método: ARI de bootstrap (submuestra y reajuste) o, en su defecto, ARI entre los pases tight/loose. El contraste contra un nulo de etiquetas permutadas se reporta como diagnóstico (`null_mode="diagnostic"`) y no filtra. Las estabilidades no son comparables entre métodos: cada uno tiene su propia perturbación (A mueve `min_forms_uc`, B la ventana de coref, C el backend).

El gate de dominancia se mide sobre el **corpus**, no sobre las UCE asignadas: c = max_class_share × (1 − noise_frac) ≤ 0,70. La razón es medida: una misma estructura (442 de 528 UCEs) daba 0,837 con el backend denso —y se aceptaba— y 0,923 con HDBSCAN —y se rechazaba— solo porque el ruido salía del denominador; con el umbral sobre el corpus ambas lecturas dan 0,837. El coste declarado es que un corpus legítimamente dominado por un tema (p₁ = 0,72) se rechaza; por eso el umbral no se sube a 0,75.

El modo `all` sí exige la intersección de los tres conjuntos estables, cada uno definido por dos pases dependientes, y no es el modo por defecto. Ninguna de estas rutas equivale a seis condiciones independientes con probabilidad p ≈ 0,90: la intersección no se calcula como una potencia de probabilidades independientes.
```

**Motivo y fuente:** la aritmética `p⁶ ≈ 0,53` modela seis condiciones independientes que
el pipeline no ejecuta. `select-v4` elige un método por gates de viabilidad (K ∈ [3, 20],
ruido ≤ 0,50, cobertura ≥ 0,15, dominancia sobre corpus ≤ 0,70) y estabilidad
intra-método; el nulo de permutación es diagnóstico por defecto. La motivación del gate
0,70 es la discrepancia 0,837/0,923 causada por el denominador con ruido. Fuente:
`src/reii/method_selection.py` (docstring y `_feasibility`), `src/reii/config.py` /
`src/reii/main_workflow.py` (`method_selection_null_mode = "diagnostic"`,
`method_selection_max_class_share = 0.70`), `docs/design_v5/GATE_NOISE.md` §0/§5/§10,
`docs/design_v5/SELECT_V4.md`, `PAPER_GAPS.md` (D10).

---

## §4 — Efecto 2: riqueza y `TODO` (L303-306)

**Ubicación:** §4, «Efecto 2», viñetas y párrafo final, líneas 303-306.

**Antes:**

```text
- **ALCESTE** cuenta la riqueza sobre las UCEs _clasificadas_ (~90% del corpus). Como las hapax están repartidas por todo el texto, casi todos los types aparecen en ese 90% → **97%**.
- **REII** cuenta la riqueza sobre el _núcleo estable_ (51% de las UCEs). Por sesgo de selección, ese núcleo contiene el vocabulario prototípico y frecuente, no las palabras raras → **30%**.

Las hapax viven en las UCEs descartadas. Si las UCEs estables son las "prototípicas" (las que sobreviven a 6 condiciones), su vocabulario es _necesariamente_ restringido. La riqueza baja no es un defecto: es la consecuencia estadística de filtrar por estabilidad.
```

**Después:**

```text
- **ALCESTE** cuenta la riqueza sobre las UCEs _clasificadas_ (~90% del corpus). Como las hapax están repartidas por todo el texto, casi todos los types aparecen en ese 90% → **97%**.
- **REII** cuenta la riqueza sobre las UCEs estables del método elegido, una fracción menor del corpus. Por sesgo de selección, ese conjunto contiene el vocabulario prototípico y frecuente, no las palabras raras → **30%**.

Las hapax viven en las UCEs descartadas. Si las UCEs estables son las "prototípicas" (las que sobreviven al filtro de estabilidad de su método), su vocabulario es _necesariamente_ restringido. La riqueza baja no es un defecto: es la consecuencia estadística de filtrar por estabilidad.

<!-- TODO(verificar): las cifras 97 % (riqueza de ALCESTE), 30 % (riqueza de REII) y ~90 % (UCEs clasificadas por ALCESTE) no tienen artefacto en el repositorio; deben re-derivarse con el mismo corpus, denominador y unidad (types vs tokens) antes de publicarse. -->
```

**Motivo y fuente:** no existen los niveles «núcleo» ni la conjunción de seis condiciones;
la riqueza se mide sobre las UCEs estables del método elegido. Las cifras 30/97/90 no
tienen artefacto en el repositorio: se conservan y se marcan con `TODO(verificar)`.
Fuente: `docs/design_v5/PAPER_GAPS.md` §5, `src/reii/quality/metrics.py`
(`corpus_coverage`).

---

## §4 — Efecto 3: denominador de la cobertura (L308)

**Ubicación:** §4, «Efecto 3», párrafo, línea 308.

**Antes:**

```text
REII calcula la retención sobre el **total** de UCEs del corpus, incluidas las que nunca entraron a clasificación.
```

**Después:**

```text
REII calcula la cobertura de UCEs sobre el **total** de UCEs del corpus, incluidas las que nunca entraron a clasificación.
```

**Motivo y fuente:** el denominador de UCEs corresponde a la **cobertura de UCEs estables**,
no a la retención léxica de formas activas (cuyo denominador son palabras, no UCEs).
Mantener «retención» aquí conflacionaría ambas cantidades. Fuente:
`src/reii/quality/metrics.py` (`coverage`), `docs/design_v5/RETENTION_POLICY.md` §1.1,
`docs/design_v5/PAPER_GAPS.md` (D11).

---

## §4 — Efecto 4: estabilidad de K (L312-314)

**Ubicación:** §4, «Efecto 4», título y párrafo, líneas 312-314.

**Antes:**

```text
## Efecto 4 — La granularidad de la partición (el "3 clases estables")

ALCESTE con 3 clases es una partición **gruesa**: cada clase cubre un tercio del corpus, así que la cobertura es alta por construcción. Además, con 3 clases el mapeo Húngaro es trivial (pocas clases que alinear). REII produce más clases, o clases definidas por embeddings que capturan similitud _semántica_ y no solapamiento _léxico_. A más clases, menos vocabulario por clase y menor cobertura léxica aparente.
```

**Después:**

```text
## Efecto 4 — La granularidad de la partición no es una constante

ALCESTE con 3 clases es una partición **gruesa**: cada clase cubre un tercio del corpus, así que la cobertura es alta por construcción. En REII, en cambio, K no es una propiedad del método ni una curva de especificación: con una segmentación **idéntica** (target 30, `lem_total = 5658`) el clasificador produce K ∈ {3, 7, 11, 16}. La inestabilidad es del clasificador, no del tamaño de UCE; por eso K se reporta por método, junto con ARI y cobertura, y no se presenta como un único valor del sistema.
```

**Motivo y fuente:** con segmentación idéntica (target 30, `lem_total = 5658`) el
clasificador produce K ∈ {3, 7, 11, 16}; debe declararse como inestabilidad del
clustering, no como curva de especificación. Fuente: `docs/design_v5/PAPER_GAPS.md`
(D21/§5), `src/reii/method_selection.py` (docstring: «K salta entre 3, 7 y 16»).

---

## §4 — Conclusión: eliminación de `p⁶` y retención verificada (L334)

**Ubicación:** §4, «Conclusión (versión mejorada para el paper)», línea 334.

**Antes:**

```text
> La discrepancia entre la riqueza de vocabulario (30% vs 97%) y la retención (51% vs 90%) no refleja una diferencia de calidad entre REII y ALCESTE, sino una diferencia de definición operacional. La retención de REII es el resultado de un criterio conjuntivo de seis condiciones de estabilidad cualitativamente heterogéneas: si cada condición se aprueba con probabilidad p ≈ 0.90, la intersección esperada es p⁶ ≈ 0.53, consistente con el 51% observado. La riqueza de vocabulario, por su parte, está dominada por la ley de Zipf: las hapax legomena constituyen la mayoría de los types del corpus y se concentran en las UCEs descartadas por el filtro de estabilidad, de modo que el núcleo estable retiene necesariamente un vocabulario restringido. ALCESTE, al clasificar el 90% de las UCEs con una partición de solo tres clases, captura casi todos los types por construcción. La comparación solo es válida si se igualan el denominador (UCEs clasificadas vs totales), la unidad de conteo (types vs tokens) y la granularidad de la partición.
```

**Después:**

```text
> La diferencia entre la retención léxica de REII (10,72 lemas por UCE, retención 0,395; 12,24 y 0,451 con la política conservadora `tool_adverb_filtered`, opt-in) y la de la referencia ALCESTE 2012 (13,95 y 0,537) no refleja una diferencia de calidad entre ambos sistemas, sino una diferencia de definición operacional: REII filtra las formas activas con una lista de palabras funcionales que ALCESTE trata como suplementarias, de modo que la retención por defecto queda 0,142 por debajo de la referencia (−26,4 %). La retención no es una probabilidad de conjunción de seis condiciones: el pipeline elige un solo método con la regla pre-registrada `select-v4`, y cada método se estabiliza con dos pases dependientes. La riqueza de vocabulario, por su parte, está dominada por la ley de Zipf: las hapax legomena constituyen la mayoría de los types del corpus y se concentran en las UCEs descartadas por el filtro de estabilidad, de modo que el conjunto estable retiene necesariamente un vocabulario restringido. ALCESTE, al clasificar el 90% de las UCEs con una partición de solo tres clases, captura casi todos los types por construcción. La comparación solo es válida si se igualan el denominador (UCEs clasificadas vs totales), la unidad de conteo (types vs tokens) y la granularidad de la partición.
```

**Motivo y fuente:** `p⁶` no modela el pipeline; la retención verificada es 0,395 por
defecto (0,451 con la política conservadora) frente a 0,537 de ALCESTE 2012, y la brecha
por defecto es −0,142 (−26,4 %). Las cifras de riqueza quedan cubiertas por el
`TODO(verificar)` del Efecto 2. Fuente: `docs/design_v5/RETENTION_POLICY.md` §1.1/§3.3,
`src/reii/method_selection.py`, `docs/design_v5/PAPER_GAPS.md` (D10-D11).

---

## §4 — Lista «Qué verificar antes de concluir», punto 3 (L324)

**Ubicación:** §4, «Qué verificar antes de concluir», punto 3, línea 324.

**Antes:**

```text
3. **¿REII mide sobre el núcleo estable o sobre todas las UCEs?** Si mide solo el núcleo, el 30% es esperable.
```

**Después:**

```text
3. **¿REII mide sobre las UCEs estables o sobre todas las UCEs?** Si mide solo las estables, el 30% es esperable.
```

**Motivo y fuente:** no existe el nivel «núcleo»; las UCEs estables son el `primary_stable`
del método elegido. Fuente: `src/reii/method_selection.py`, `docs/design_v5/PAPER_GAPS.md`
(D05).

---

## Notas de implementación (no son cambios del artículo)

1. **`min_cluster_size_cdh` es derivado.** Se interpreta como `1/(2K)` y se calcula a
   partir de K; no es un parámetro libre independiente. Se declara en el Paso 3.
2. **Bug abierto conocido.** `hdbscan_uce_min_cluster_frac` y
   `hdbscan_uce_min_cluster_frac_loose` existen en `Config` y se consumen en
   `main_workflow.py`, pero no figuran en la allowlist de
   `src/reii/backend/config_store.py` (que solo incluye `hdbscan_min_cluster_size`,
   `hdbscan_min_samples` y `hdbscan_cluster_selection_epsilon`). No afecta a la redacción
   de este artículo, pero conviene registrarlo antes de publicar la reproducibilidad.
3. **Artefacto persistido obsoleto.** `data/method_selection.json` conserva
   `rule_version: "select-v3"` y `max_class_share_max: 0.9`; no se usa como fuente de
   cifras en estas enmiendas.
4. **Origen de los 384 d.** El artículo declara explícitamente que `C_in`, `d` y `RMSD`
   se midieron en el espacio de embeddings de 384 d del banco de pruebas (528 UCEs; 154
   con embedding), no en el espacio de 300 d del método C. No se atribuyen al pipeline de
   clasificación.

---

## Patch `14` — default de retención adoptado (APLICADO a `docs/paper.md`)

**Estado: APLICADO el 2026-10-28.** El patch
`14_retention_default_tool_adverb_filtered` cambia el default de `retention_policy` de
`pos_only` a `tool_adverb_filtered`. Las tres frases de abajo se aplicaron tal cual
(con una fila adicional para conservar la baseline `pos_only` como opt-in). Este
registro se conserva como historial: no volver a aplicar.

### 1. §3, tabla «Qué COMPROMETE REII» (L235)

**Antes:**

```text
| Filtro de formas activas (`pos_only`, default) | Retiene menos léxico que ALCESTE 2012: 10,72 lemas/UCE y 0,395 frente a 13,95 y 0,537 (brecha −3,23 lemas/UCE, −23,2 %). |
```

**Después:**

```text
| Filtro de formas activas (`tool_adverb_filtered`, default) | Retiene menos léxico que ALCESTE 2012: 12,24 lemas/UCE y 0,451 frente a 13,95 y 0,537 (brecha −1,71 lemas/UCE, −12,3 %; retención −0,086, −16,0 %). |
| Política `pos_only` (opt-in)                               | Baseline del repositorio hasta v5: 10,72 lemas/UCE y 0,395 (brecha −3,23 y −0,142, −23,2 % / −26,4 %). Se conserva como elección explícita.        |
```

### 2. §4, explicación simplificada, punto 3 (L277)

**Antes:**

```text
3. REII no mejora todas las métricas a la vez: gana cobertura de UCEs estables y métricas con nulo de etiquetas, pero retiene menos léxico que ALCESTE 2012 (0,395 frente a 0,537; 0,451 con la política conservadora `tool_adverb_filtered`, opt-in). El default sigue siendo `pos_only`; la variante agresiva `pos_plus_propn` (0,485) queda excluida por su cuota de palabras herramienta (0,375 > 0,30).
```

**Después:**

```text
3. REII no mejora todas las métricas a la vez: gana cobertura de UCEs estables y métricas con nulo de etiquetas, pero retiene menos léxico que ALCESTE 2012 (0,451 frente a 0,537 con la política conservadora `tool_adverb_filtered`, que es el default). La variante agresiva `pos_plus_propn` (0,485) queda excluida por su cuota de palabras herramienta (0,375 > 0,30), y la baseline `pos_only` (0,395) queda disponible como elección explícita.
```

### 3. §4, conclusión (L332)

**Antes:**

```text
...REII (10,72 lemas por UCE, retención 0,395; 12,24 y 0,451 con la política conservadora `tool_adverb_filtered`, opt-in)... de modo que la retención por defecto queda 0,142 por debajo de la referencia (−26,4 %).
```

**Después:**

```text
...REII (12,24 lemas por UCE, retención 0,451 con la política conservadora `tool_adverb_filtered`, que es el default)... de modo que la retención por defecto queda 0,086 por debajo de la referencia (−16,0 %).
```

**Motivo y fuente:** `docs/design_v5/RETENTION_POLICY.md` §10;
`src/reii/retention_policy.py` (`DEFAULT_RETENTION_POLICY`); `ia/gramatica.json`.
Cifras de §3.3 de `RETENTION_POLICY.md` (12,24 / 0,451). La desviación se declara en
la tabla de compromisos porque el default ya no es la baseline del repo.

**Efecto colateral resuelto:** se conserva la baseline `pos_only` como fila propia
(opt-in) en vez de reescribir el texto centrado en ella. La alternativa de declarar el
cambio como _configuración de las corridas_ era inviable: `retention_policy.py`
resuelve `tool_adverb_filtered` sin argumentos, así que el paper y el artefacto se
habrían contradicho.
