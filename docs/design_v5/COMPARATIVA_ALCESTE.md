# Comparación ALCESTE 2012 ↔ REII: qué debe contener la sección del paper

Estado: **propuesta basada en artefactos**. Cada cifra de aquí está anclada a un
artefacto del repositorio o a una página del informe original; lo que no tiene
artefacto está marcado como `[sin artefacto]` y **no debe publicarse**.

Fuentes:

- `docs/alceste_baseline/relatorio detalhado.pdf` (ALCESTE, 16-03-2012, 51 pág.).
  Tablas reconstruidas por coordenadas (`page.get_text("words")`), no por texto
  plano: el orden plano intercala etiquetas y valores.
- `docs/alceste_baseline/CARTOGRAFIA NATURAL DO CORPUS.rtf` (corpus con marcas `*var`).
- `docs/alceste_baseline/proceso clasico.svg` (diagrama del método clásico).
- REII: `data/method_selection.json`, `data/best_params.json`,
  `data/uce_calibration.json`, `data/token_table.jsonl`,
  `data/diag/uce_size_bench.json`, y la salida de
  `scripts/retention_policy_report.py --token-table data/token_table.jsonl`.

---

## 0. Lo que hay hoy en `paper.md` y no puede sostenerse

`paper.md` §4 (líneas 242–278) compara con una tabla que **no tiene artefacto
detrás**:

| Afirmación actual                             | Problema                                                                                                                                |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Tasa de retención ALCESTE «65%–85%»           | El baseline da **un** valor: 13.95/26 = **0.537**. El rango no existe en ningún artefacto.                                              |
| «REII 70–85%»                                 | Medido: **0.451** (default) / 0.395 (`pos_only`). El rango tampoco existe.                                                              |
| C_ext 0.80 vs 0.85, Silhouette >0.5, FMI >0.7 | `[sin artefacto]`. No hay referencia humana ni cálculo guardado.                                                                        |
| «Cobertura por tokens: baja / alta»           | Cualitativo. Hoy es **medible**: 61% vs 94% (§2).                                                                                       |
| Riqueza 97% vs 30%                            | `[sin artefacto]`. Además el 97.84% de ALCESTE **no declara denominador** en el informe: no es comparable.                              |
| Efecto 3: «el denominador no es el mismo»     | **Falso para la retención** (§4): 26 vs 27.1 palabras crudas/UCE, 4% de diferencia. Verdadero para la _cobertura_, que es otra métrica. |

`CONTRADICTIONS_OPTIONS.md` §4 ya lo registró como brecha abierta. La sección se
reescribe, no se parchea.

---

## 1. Baseline verificado (lo que ALCESTE reporta de verdad)

Corpus: 29 entrevistas, español, cambio climático y educación ambiental
(Ayacucho). Tratamiento `Classification double (121)`.

| Magnitud                                     | Valor                                              | Dónde    |
| -------------------------------------------- | -------------------------------------------------- | -------- |
| UCI (entrevistas)                            | 29                                                 | p.1      |
| Ocurrencias totales                          | 14 337                                             | p.1      |
| Formas distintas                             | 2 590                                              | p.1      |
| Frecuencia media / máxima                    | 6 / 676                                            | p.1      |
| **Hapax**                                    | **1 427** (55.1% de los types)                     | p.1      |
| Formas tras reducción                        | 428                                                | p.1, p.6 |
| Formas suplementarias                        | 110                                                | p.1      |
| Modalidades de variables / palabras estrella | 36                                                 | p.1      |
| **UCE**                                      | **376**                                            | p.1      |
| **Ocurrencias por UCE**                      | **26**                                             | p.1      |
| **Palabras analizadas por UCE**              | **13.95**                                          | p.1      |
| Frecuencia mínima de análisis                | 4                                                  | p.1      |
| Pares de palabras                            | 11 168                                             | p.1      |
| Riqueza de vocabulario                       | 97.84% (denominador no documentado)                | p.1      |
| Clases pedidas por clasificación             | **10** (valor por defecto)                         | p.1      |
| **Clases estables retenidas**                | **3**                                              | p.1, p.7 |
| UCE mínimas para retener una clase           | **19** (= 5.05% de 376)                            | p.7      |
| **% de UCE clasificadas**                    | **90%**                                            | p.7      |
| Clasificación 1                              | mín. 12 palabras analizadas/UC; 296 UC; 428 formas | p.6      |
| Clasificación 2                              | mín. 14 palabras analizadas/UC; 260 UC; 428 formas | p.6      |

Variables de diseño (`CARTOGRAFIA...rtf`): `ind` (29 niveles), `edad` (2),
`sexo` (2); `esci`, `gi`, `proc` son **constantes** ⇒ no permiten χ². El cruce
clases×variables solo es posible con `edad` y `sexo` (2×2).

**Perfiles de clase** (φ, de las tablas «Présences significatives», p.11/27/39):

| Clase | Términos (φ)                                                                                                                                                                     | Tema                                   |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| 1     | para .38 · curso .30 · tema .29 · deber .28 · enseñ .24 · docente .22 · solo .20 · acción .18 · charla .18 · proyecto .18 · concientiz .18 · universidad .17 · plan .16          | escuela, currículo y docencia          |
| 2     | televisión .43 · refier .39 · calor .30 · debido .28 · a través .28 · cambios .28 · tala_de_árboles .27 · información .26 · calentamiento .26 · contaminación .25 · noticias .24 | medios, información y cambio climático |
| 3     | afect .43 · frío .34 · lluvia .30 · piel .29 · helad .29 · antes .26 · fuerte .26 · precio .25 · sequía .26 · provoc .25 · tos .25 · agricultura .25 · económico .24             | impactos: salud, economía, agro        |

---

## 2. La unidad es la misma; la cobertura no

Este es el eje central de la sección y **hoy está sin escribir**.

|                                  | ALCESTE 2012               | REII (`uce_target_size=30`)               |
| -------------------------------- | -------------------------- | ----------------------------------------- |
| Palabras crudas del corpus       | 14 337                     | **14 333** (`texto.split`, `token_table`) |
| UCE                              | 376                        | **528**                                   |
| Palabras crudas / UCE            | 26                         | **27.15**                                 |
| Palabras analizadas / UCE        | 13.95                      | **12.23** (default); 10.71–13.15 (§4)     |
| Ocurrencias dentro de UCE        | 376×26 = **9 776 (68.2%)** | 528×27.15 ≈ **14 335 (≈100%)**            |
| UCE clasificadas                 | 90% → 338                  | 95.3% («usable coverage») → 503           |
| **Texto que acaba en una clase** | 338×26 = 8 788 → **61.3%** | 503×27.15 ≈ 13 654 → **95.3%**            |

Lectura: **la UCE no es la diferencia** (26 vs 27.15 palabras crudas: 4%). La
diferencia es **cuánto texto entra en una UCE**: ALCESTE deja fuera el 32% del
corpus (preguntas del entrevistador, ficha sociodemográfica y restos de
segmentación — ver el SVG del proceso clásico), y sobre ese 68% clasifica el 90%.
El «90% de UCE clasificadas» de ALCESTE es, sobre el corpus, **61%**.

Aritmética que lo cierra: `n_UCE × palabras_crudas/UCE ≈ total del corpus`. Para
llegar a 376 UCE con cobertura total harían falta ≈38.1 palabras crudas/UCE, es
decir `target≈42`, **fuera** del rango de especificación `[14, 30]`
(`data/diag/uce_size_bench.json`). No es un defecto de calibración: es una
elección de cobertura.

> Cuidado al escribirlo: los dos estimadores (`n_UCE ~ target` y `pal_cr/UCE ~
target`) **no son independientes**; son la misma identidad porque el corpus
> tiene un número fijo de palabras. No presentarlos como confirmación doble.

---

## 3. Cómo llega cada sistema a la partición

|                            | ALCESTE 2012                                                                                                         | REII (`select-v4`)                                          |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Estrategia                 | **2 clasificaciones descendentes** con umbral distinto (12 y 14 palabras analizadas/UC) y se retiene la intersección | **1 método elegido** entre A/B/C + estabilidad intra-método |
| Clases pedidas → retenidas | 10 → **3**                                                                                                           | K no es un parámetro del sistema: es del método             |
| Umbral de permanencia      | ≥19 UCE (5.05%)                                                                                                      | `min_class_share = 0.01`; medido: mín. 69 UCE (13.1%)       |
| Estabilidad                | congruencia entre los dos árboles, «evaluación visual del investigador» (SVG)                                        | `bootstrap_ari` (submuestra + reajuste), IC95               |
| Nulo                       | —                                                                                                                    | etiquetas permutadas (diagnóstico)                          |
| K final                    | 3                                                                                                                    | C: **3** (247 / 212 / 69 UCE)                               |
| Se descarta un método por  | —                                                                                                                    | A: K=4; B: cobertura útil 0.118 < 0.15                      |

**Correspondencia de parámetros** (la tabla que hace legítima la comparación):

| ALCESTE                             | REII                                             | Valor usado                                                                          |
| ----------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| 12 y 14 palabras analizadas/UC      | `min_forms_uc = [mf1, mf1+forms_gap]`            | `min_forms_uc_1=10`, `forms_gap=2` → [10, 12]                                        |
| (espaciado 2 entre clasificaciones) | `forms_gap`                                      | **2** (el optimizador lo eligió; coincide con ALCESTE)                               |
| 26 ocurrencias/UCE                  | `uce_target_size` (calibración bayesiana previa) | entrada 26 → efectivo **30**, medido 26.86 en calibración / 27.15 en corpus completo |
| 3 clases                            | `cdh_target_k`                                   | **3**                                                                                |
| ≥19 UCE/clase (5%)                  | `min_cluster_size_cdh`                           | 0.1667 (borde superior del espacio)                                                  |
| frecuencia mínima 4                 | `min_term_abs_freq` / `tsj`                      | `tsj=2`                                                                              |

> **No presentar K=3 como validación externa**: `cdh_target_k=3` es una meta del
> optimizador y `-K` es un objetivo. Que coincida con ALCESTE es concordancia de
> granularidad, no evidencia independiente. Se dice así, explícitamente.

---

## 4. Retención léxica: sí son comparables (hallazgo nuevo)

El informe de ALCESTE permite **cerrar** la discusión del paper: su «retención»
no es una métrica exótica. Es exactamente

    retención = palabras analizadas por UCE / ocurrencias por UCE = 13.95 / 26 = 0.5365 ≈ 0.537

Y REII usa la misma definición con su propio denominador medido (27.1):
12.23/27.1 = 0.451; 10.71/27.1 = 0.395. **Los denominadores están a 4% de
distancia**, así que la brecha es real y atribuible a la política de filtrado, no
a un artefacto de denominador (esto corrige el «Efecto 3» del paper).

Artefacto: `python scripts/retention_policy_report.py --input data/workflow_data.json
--token-table data/token_table.jsonl --fingerprint 93fabdc4d0ff6590 --bootstrap 1000 --seed 42`

| Política                              | lemas/UCE | IC95           | retención | IC95           | cuota herramienta |
| ------------------------------------- | --------- | -------------- | --------- | -------------- | ----------------- |
| `pos_only` (baseline histórica)       | 10.71     | [10.40, 10.99] | 0.395     | [0.388, 0.402] | 0.000             |
| **`tool_adverb_filtered`** (default)  | **12.23** | [11.92, 12.54] | **0.451** | [0.445, 0.457] | 0.000             |
| `pos_plus_propn` (agresiva, excluida) | 13.15     | [12.82, 13.47] | 0.485     | [0.479, 0.491] | 0.375 > 0.30      |
| **ALCESTE 2012** (referencia)         | **13.95** | —              | **0.537** | —              | —                 |

Corpus de la medición: 528 UCE, 14 333 palabras crudas, 13 824 tokens alfa.
Regla pre-registrada: cuota de palabras herramienta ≤ 0.30 y luego min
|retención − 0.537| ⇒ veredicto `tool_adverb_filtered`, robusto a cuotas 0.15/0.25/0.35.

Lo que se puede afirmar: REII retiene **1.7 lemas/UCE menos** que ALCESTE con el
default (−12.3%), y **3.2 menos** con `pos_only` (−23.2%), a cambio de clasificar
el 95% del texto frente al 61%. Es un _trade-off_ medido, no una inferioridad.

---

## 5. Métricas que ALCESTE no reporta (aquí REII aporta, no iguala)

| Métrica                         | A: word-count        | B: similitud          | C: embeddings (dense_cdh) |
| ------------------------------- | -------------------- | --------------------- | ------------------------- |
| K                               | 4                    | 5                     | **3**                     |
| ARI entre pases                 | 0.489                | 0.288                 | **1.000**                 |
| `bootstrap_ari` [IC95]          | 0.496 [0.405, 0.601] | 0.101 [−0.005, 0.222] | **0.926 [0.893, 0.958]**  |
| Cobertura (UCE asignadas)       | 0.763                | 0.525                 | **1.000**                 |
| Cobertura útil                  | 0.612                | 0.118                 | **0.953**                 |
| `null_z` (etiquetas permutadas) | 73.4                 | 16.5                  | **212.5**                 |
| Clase mín./máx. (UCE)           | 93 / 150             | 8 / 270               | 69 / 247                  |
| Dominancia del corpus           | 0.284                | 0.511                 | 0.468                     |

Fuente: `data/method_selection.json` (regla `select-v4`, huella `93fabdc4d0ff6590`,
`n_uces=528`, `seed=42`).

Tres hechos que el paper **sí** puede reclamar como aportación:

1. **B (similitud) no es viable en este corpus**: cobertura útil 0.118, bootstrap
   0.101 con IC que cruza 0. Se descarta por gate, no por gusto. La sonda B
   (parámetros permisivos) dio `B=0` y `B↔C=0` ⇒ es estructural, no tuning.
2. **La estabilidad se puede medir**: ALCESTE la delega a «evaluación visual del
   investigador»; REII da ARI con IC y un nulo.
3. **El nulo importa**: B tiene ARI 0.288 (parece «razonable») pero su bootstrap
   es 0.101 con IC que contiene 0 y `null_z` 16.5, frente a 212.5 de C. Sin el
   nulo, B parecería aceptable.

---

## 6. Figuras sugeridas

Prioridad = valor para la comparación, no vistosidad. Todas se pueden generar con
datos ya presentes; ninguna necesita volver a correr el pipeline.

| #      | Figura                                            | Tipo                                                                                                   | Datos                                                        | Por qué                                                                              |
| ------ | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| **F1** | **Escalera de cobertura** (ALCESTE vs REII)       | barras apiladas horizontales en 3 tramos: _corpus → dentro de UCE → dentro de clase_                   | §2                                                           | Es la figura que reemplaza la tabla sin artefacto. Muestra de un vistazo 61% vs 95%. |
| **F2** | **Calibración del tamaño de UCE**                 | línea + ajuste hiperbólico; puntos (18,866) (22,715) (26,600) (30,528); marcas en 376 y en `target=42` | `uce_size_bench.json`                                        | Justifica por qué no se igualan las 376 UCE. R²=0.976 (n_UCE) y 0.9987 (pal/UCE).    |
| **F3** | **Retención léxica con IC**                       | barras con bigotes + línea de referencia 0.537 + línea de cuota 0.30                                   | §4                                                           | Convierte una discusión de definiciones en una medida.                               |
| **F4** | **Selección de método**                           | dumbbell: A/B/C en (ARI bootstrap, cobertura útil) con IC y el gate 0.15                               | §5                                                           | Explica `select-v4` y el descarte de B.                                              |
| **F5** | **Perfiles de clase lado a lado**                 | heatmap 3 filas (clases) × ~12 términos, dos columnas (ALCESTE φ / REII score)                         | §1 + `reii.diag multiverse`                                  | Única validación **cualitativa**: ¿las 3 clases hablan de lo mismo?                  |
| **F6** | **K no es una constante**                         | panel: K por método (3/4/5) y el artefacto de dominancia 0.837 vs 0.923                                | `method_selection.json` + `METRICS_PAPER.md`                 | Sostiene la crítica a presentar K como propiedad del sistema.                        |
| F7     | _Matriz de acuerdo ALCESTE×REII_                  | matriz de confusión 3×3 + ARI                                                                          | 60 UCE características (§8, opción b)                        | La figura que un revisor de ALCESTE pedirá primero; cota superior (optimista).       |
| F8     | _Distribución bootstrap vs pase único_            | ECDF/violin del ARI bootstrap + punto del ARI entre pases                                              | `method_selection.json` (solo IC) o re-correr con `--trials` | Muestra por qué se adopta bootstrap y no perturbación.                               |
| F9     | _Plano factorial (AFC) de las clases REII_        | biplot 2D de UCE coloreadas por clase                                                                  | `workflow_data.json`                                         | Espeja la AFC que ALCESTE imprime (p.8-9); habla el idioma del lector de ALCESTE.    |
| F10    | _Apéndice: presupuesto de centinelas y sign-flip_ | barras por `gate_reason` + ρ por parámetro y etapa                                                     | `reii.diag optuna-db`                                        | Robustez metodológica; no va en la comparación.                                      |

Notas de ejecución:

- Las figuras de `docs/alceste_baseline/` son capturas (`gráfico global.bmp` 2.1 MB,
  `procesos.png` 1.0 MB). Para el paper hay que **re-plotear en vectorial**; el BMP
  no es publicable.
- `proceso clasico.svg` **sí** es reutilizable como figura de método (es vectorial):
  es el diagrama Reinert/ALCESTE con segmentación, AFC, CJD, χ² y repetición con
  segundo umbral. Conviene contrastarlo con un diagrama equivalente de REII.
- F2 y F3 son las únicas que un revisor puede recalcular desde el PDF; por eso son
  las que más conviene que estén bien.

---

## 7. Estructura propuesta de la sección

1. **Qué se compara y qué no**: misma UCE (26 vs 27.1 palabras crudas), distinta
   cobertura. Aquí va F1.
2. **Cómo se llega a la partición**: doble clasificación vs `select-v4`; 10→3 vs
   K por método. Aquí F4 y F6, y la tabla de correspondencia de parámetros.
3. **Retención léxica**: la definición es la misma; la brecha es de política.
   Aquí F3.
4. **Perfiles de clase**: comparación temática. Aquí F5 y, si se puede, F7.
5. **Lo que ALCESTE no mide**: bootstrap, IC, nulo, ARI entre métodos. Aquí F8.
6. **Límites**: `[sin artefacto]` fuera; `min_cluster_size_cdh` en el borde del
   espacio; K no independiente; sin referencia humana ⇒ no hay C_ext.

---

## 8. Lo que falta medir antes de cerrar (3 tareas)

1. **Asignación UCE→clase de ALCESTE** (la que falta para F7). El informe solo
   lista las 20 UCE más características por clase (p.24/36/49): 60 UCE, muestra
   sesgada al prototipo. Opciones: (a) exportar la tabla completa desde ALCESTE
   (preferible), (b) usar esas 60 y declararlo como cota **superior** (optimista)
   con su sesgo — **elegida**. Sin esto, el paper no puede afirmar _acuerdo_
   entre particiones, solo comparar perfiles.
2. **Tamaños de clase de ALCESTE**: no están en texto, solo en el dendrograma
   (p.7) / histograma. Sin ellos no se puede comparar el balance de clases
   (REII: 247/212/69).
3. **Verificar la semántica de `min_cluster_size_cdh`**: el valor elegido
   (0.1667, borde superior) no se refleja como piso de la clase menor (69/528 =
   13.1%). Antes de escribir «REII exige clases ≥16.7%» hay que leer el código;
   si no, no se afirma.

---

## 9. Frase de cierre propuesta (sustituye a la actual)

> Sobre el mismo corpus (14 333 palabras, 29 entrevistas), ALCESTE 2012 y REII
> segmentan con la misma unidad —26 y 27.1 palabras crudas por UCE
> respectivamente— pero no cubren lo mismo: ALCESTE clasifica 338 de sus 376 UCE,
> que contienen el 61% del texto; REII clasifica el 95% del texto con 528 UCE.
> ALCESTE retiene 13.95 palabras analizadas por UCE (0.537 de las ocurrencias) y
> REII 12.23 (0.451, IC95 [0.445, 0.457]) con su política por defecto: la
> definición de retención es idéntica y los denominadores difieren un 4%, de modo
> que la brecha es de filtrado léxico y no de aritmética. Ambos convergen en tres
> clases, aunque en REII K es una meta del optimizador y no una propiedad del
> sistema, y la estabilidad se reporta con un intervalo de confianza (ARI
> bootstrap 0.926 [0.893, 0.958]) en lugar de una inspección visual.
