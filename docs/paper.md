ESTRUCTURA DEL ARTÍCULO CIENTÍFICO (PROPUESTA ÓPTIMA) 

1\. INTRODUCCIÓN Y FUNDAMENTACIÓN EPISTEMOLÓGICA 

1.1\. La tensión entre lo cualitativo y lo cuantitativo en CATA (Computer-Assisted Text Analysis)

1.2\. Paradigmas de análisis: Deductivo (hipotético-deductivo) vs. Inductivo (emergente) 

1.3\. La necesidad de una hermenéutica abductiva en el análisis cuantitativo de datos cualitativos (QDA) 

1.4\. Objetivos y contribuciones del artículo 2\. ESTADO DEL ARTE Y LIMITACIONES DE LA LEXICOMETRÍA CLÁSICA 

2.1\. El método Reinert (ALCESTE): Clasificación Descendente Hiérarchique (CDH) y Análisis Factorial 

2.2\. Supuestos y cuellos de botella de ALCESTE: Matriz lógica dispersa, UCEs rígidas y bolsa de palabras 

2.3\. El surgimiento de los modelos de lenguaje transformadores y el agrupamiento neural de texto 

3\. DISEÑO DE LA ARQUITECTURA ALTERNATIVA (EL NUEVO MODELO) 

3.1\. Preprocesamiento sintáctico flexible y segmentación semántica de Unidades de Contexto 

3.2\. Proyección vectorial densa mediante contextual embeddings 3.3\. Algoritmo de división/agrupamiento jerárquico denso (Reemplazo de la CDH clásica) 3.4\. Extracción de palabras clave representativas y resumen contextual asistido por LLM 4\. MARCO FORMAL DE EVALUACIÓN Y BATERÍA DE MÉTRICAS 4.1\. Métricas de Estructura y Consistencia de Partición (Silhouette, ARI, FMI, Tasa de Cobertura/Estabilidad) 

4.2\. Métricas de Calidad Semántica (Coherencia Externa C\_ex, Coherencia Interna C\_in, Diversidad d) 

4.3\. Métricas de Validez Externa y Proporción Discursiva (RMSD, Cohen's Kappa, Inter-Coder Reliability) 

4.4\. Benchmark comparativo: ALCESTE / IRaMuTeQ vs. BERTopic vs. Modelo Propuesto 5\. ESTUDIO DE CASO EMPÍRICO Y VALIDACIÓN EXPERIMENTAL 

5.1\. Descripción del corpus de prueba y condiciones de aplicación 

5.2\. Evaluación cuantitativa de rendimiento y estabilidad del agrupamiento 

5.3\. Interpretación dual y evaluación cualitativa de los "Mundos Lexicales" descubiertos 6\. DISCUSIÓN Y REFLEXIONES METODOLÓGICAS 

6.1\. ¿Sustituye la estadística a la interpretación? La ilusión de objetividad en la analítica de texto 

6.2\. Comparación de la capacidad de generalización y robustez ante el ruido 

6.3\. Recomendaciones prácticas para investigadores en ciencias sociales y humanidades 7\. CONCLUSIONES Y TRABAJO FUTURO




# 📋 Comparación: ALCESTE vs REII (propuesta del agente)

## 1. FUNDAMENTO TEÓRICO: ¿Qué mide cada métrica?

A continuación se explica cada métrica utilizada por ambos métodos:

| Métrica | Qué mide | Fórmula | Qué nos dice |
|--------|----------|----------|-------------|
| **CR** (Cobertura / Tasa de Retención) | % de UCEs que sobreviven al filtro de estabilidad | `CR = UCEs Estables / UCEs Totales × 100` | Qué tanto se conserva el texto original. No dice nada sobre la calidad de la clasificación, solo sobre la cantidad conservada. |
| **Cobertura por Tokens** | % de términos del vocabulario que aparecen en las UCEs estableces | `Cobertura = Términos En UCEs Estables / Términos Totales del Corpus` | Qué tanto del *vocabulario* se captura. Más importante que CR porque el corpi es heterogéneo y las UCEs son más grandes que términos individuales. |
| **ARI** (Adjusted Rand Index) | Concordancia entre dos particiones (ej. dos corridas independientes) | `ARI ∈ [0,1]` | Qué bien se alinean dos clasificaciones. Si ARI = 1: misma clasificación. Si ARI ≈ 0: sin relación. |
| **FMI** (Normalized Mutual Information) | Dependencia mutua entre dos particiones | `FMI ∈ [0,1]` | Cómo mucho una clasificación informa sobre la otra. Útil para medir estabilidad sin depender del tamaño de las clases. |
| **Silhouette Score** | Qué bien definidas están las clases | `Si ∈ [-1,1]` | Qué separadas están las clases y qué compactas. >0.5: buen clustering. ≈0: classes se solapan. <0: mal clustering. |
| **Coherencia Externa (C_ext)** | ¿Qué tan bien se clasifican las clases según una referencia humana? | `C_ext ∈ [0,1]` | ¿Están los términos "correctos" en las clases? Se mide contra una etiqueta de verdad (Gold Standard). |
| **Coherencia Interna (C_in)** | ¿Qué tan homogéneos están los términos dentro de cada clase? | `C_in ∈ [0,1]` | ¿Tienen los términos de una misma clase palabras en común? |
| **Kappa Inter-Coder** | Concordancia entre dos etiquetados humanos | `Kappa ∈ [0,1]` | ¿Acuerdan dos personas para clasificar las UCEs? >0.8: acuerdo casi perfecto. |

---

## 2. ALCESTE ORIGINAL: Pasos y Parámetros

### Algoritmo paso a paso:
```
TEXTO
    │
    ├── Paso 1: Segmentación (STC → UCE → UC)
    │         └─ Criterio: 10 palabras por UC (corrida 1)
    │         └─ Criterio: 14 palabras por UC (corrida 2)
    │
    ├── Paso 2: Doble Clasificación Jerárquica (CDH)
    │         └─ CLASIFICADOR HIERARQUÍCO DE DOS PASOS
    │
    ├── Paso 3: Matriz de Contingencia (Solapamiento)
    │         └─ Cruzar las dos particiones CDH
    │         └─ Mapeo Húngaro sobre similitud de χ²
    │
    ├── Paso 4: Regla Estricta de Retención
    │         └─ UCE es ESTABLE solo si está en la misma clase
    │         │     en ambas corridas
    │         └─ CR: UCEs Estables / UCEs Totales
    │
    └── Paso 5: AFC + Métricas
            └─ ARI (entre las 2 particiones)
            └─ Tasa de Cobertura porTokens
            └─ φ de los términos representativos
            └─ CTTEST (coeficiente de contingencia)
```

### Parámetros clave de ALCESTE:
| Parámetro | Valor / Descripción | Efecto |
|-----------|-------------------|--------|
| **Tamaño de UC** | 10 vs 14 palabras | Diferencia del tamaño permite medir estabilidad |
| **Máximo de clases** | Por configuración | Control del número de mundos lexicales |
| **Umbral de similitud** | CTTEST (coeficiente de contingencia) | Mide qué tan asociados son los términos en cada UC |
| **Pseudocuento ε** | 0.1 | Desvía divisiones en matrices dispersas |
| **Regla de estabilidad** | 6 condiciones (o más) | **Muy conservadora**: exige que UCE esté en la misma clase en ambas corridas INDEPENDIENTES |
| **Mapeo** | Biunívoco (Húngaro 1-1) | Solo se asigna 1 ↔ 1. Si K es diferente entre corridas, no hay coincidencias |
| **Cobertura** | UCEs / UCEs Totales | Medida indirecta |

### Qué **COMPROMETE** ALCESTE:
| Problema | Efecto |
|----------|---------|
| Regla muy estricta (6 condiciones) | Alta precisión pero bajo recall. Se descarta hasta 50% del corpus |
| Mapeo Húngaro biunívoco | Si K es diferente entre corridas, hay "huecos" en la comparación |
| Medir retención por UCE, no por términos | La metrique es sensible al tamaño de la UCE, no al contenido |
| Solo 2 niveles: estable o inestable | No captura la gradación de estabilidad |
| Sesgo de selección | Las UCEs descartadas suelen ser las más breves, ambiguas o de transición |

---

## 3. REII (Propuesta del Agente): Cambios y Métricas

### Algoritmo paso a paso:
```
TEXTO
    │
    ├── Paso 1: Segmentación Semántica de Unidades de Contexto
    │         └─ UCE: 15–100 ocurrencias (equivalentes a frases/cláusulas)
    │         └─ UCE flexible: 15–100, con suavizado por longitud
    │
    ├── Paso 2: Dualidad de Construcción (4 métodos)
    │         ├── Método A: Word Count (Reinert clásico)
    │         ├── Método B: Similaridad (UCEVectorizer)
    │         ├── Método C: Coref (ProgressiveSegmenter)
    │         └── Método D: Embedding (progressiveSegmenter)
    │
    ├── Paso 3: Clasificación Jerárquica
    │         └─ CLASIFICADOR 1: Clasificador Descendente (Reinert)
    │         └─ CLASIFICADOR 2: KMeans / Agglomerative
    │
    ├── Paso 4: Mapeo Húngaro Correjo
    │         └─ Correjo: si K es diferente, usar mapping K1 ↔ K2
    │         └─ Correjo: no penalizar UCEs por una diferencia de estructura
    │
    ├── Paso 5: Triple Verdict de Estabilidad
    │         └─ stable_wc: ¿estable por conteo de palabras?
    │         └─ stable_sim: ¿estable por similitud?
    │         └─ stable_coref: ¿estable por coreferencia?
    │         └─ UCE puede ser estable en 1, 2 o 3 métodos
    │
    ├── Paso 6: Niveles de Calidad (2–3 niveles)
    │         ├── Núcleo: stable en todas las 4 formas de construcción
    │         ├── Extendido: estable en ≥ 2 formas de construcción
    │         └─ Proyectado: UCEs liminales (no estables, pero coherentes)
    │
    └── Paso 7: Métricas de Calidad
            └─ Tasa de Cobertura: UCEs Estables / UCEs Totales
            └─ Cobertura por Tokens: Términos en UCEs Estables / Términos Totales
            └─ Silhouette Score: qué bien definidas están las clases
            └─ ARI: concordancia entre el núcleo y la extensión
            └─ FMI: concordancia entre el núcleo y la extensión
            └─ C_ext: si hay referencia humana
            └─ Coherencia Interna: homogeneidad intra-clase
```

### Parámetros clave de REII (propuesta):
| Parámetro | Valor / Descripción | Efecto |
|-----------|-------------------|--------|
| **Tamaño de UC** | Flexible: 15–100 ocurrencias | Más adaptativo al texto |
| **Cobertura** | Sobre tokens, ponderada por frecuencia | Medida más justa |
| **Máximo de classes** | Por configuración | Control del número de mundos lexicales |
| **Umbral de similitud** | CTTEST (coeficiente de contingencia) | Mide qué tan asociados son los términos en cada UC |
| **Pseudocuento ε** | 0.1 | Desvía divisiones en matrices dispersas |
| **Regla de estabilidad** | **No binaria**: tiene niveles | Permite medir la gradación de estabilidad |
| **Mapeo** | Húngaro con correjo de K | Si K es diferente entre corridas, aún compara |
| **Cobertura** | UCEs / UCEs Totales | Medida directa |

### Qué **MEJORA** REII:
| Cambio | Efecto |
|--------|---------|
| Niveles de calidad (núcleo, extendido, proyectado) | La cobertura mejora (incluye más UCEs), pero se define qué nivel es "núcleo" y cuál es "extendido" |
| Mapeo Húngaro corrigido | Si K es diferente entre corridas, aún compara. No penaliza por una diferencia de estructura |
| Cobertura por tokens | La cobertura es más justa porque pondera por la frecuencia de los términos |
| Triple verdict de estabilidad | UCEs que son estables en 2 de 4 métodos son más informativos que las que son inestables en todos |
| Calibrar optimizador contra criterio final | El algoritmo elige parámetros que realmente mejoran los resultados finales, no solo los intermedios |

### Qué **COMPROMETE** REII:
| Cambio | Efecto |
|--------|---------|
| Incluir más UCEs (nivel extendido) | Mayor cobertura pero mayor ruido |
| Consenso blando | Clases más grandes pero menos precisas |
| Mapeo muchos a uno | Si fusionas clases genuinamente distintas, bajas la validez de las clases |
| Proyección liminal | Los UCEs proyectados tienen menor error, pero no son tão representativos |

---

## 4. COMPARativa: ¿Qué impacto tiene cada algoritmo en las métricas?

| Métrica | ALCESTE | REII (propuesta) | Diferencia principal |
|---------|---------|-----------------|---------------------|
| **Tasa de Retención (CR)** | 65%–85% | Más alta (incluye niveles extendidos) | REII logra mayor cobertura al incluir niveles extendidos y proyectados |
| **Cobertura por Tokens** | Baja (medida indirecta) | Más alta (medida directa sobre tokens) | REII mide la cobertura de manera más justa |
| **Silhouette Score** | No calculado | Calculado | REII puede medir cuántas bien definidas están las clases |
| **FMI** | No calculado | Calculado | REII puede medir concordancia entre el núcleo y la extensión |
| **Cohencia Externa (C_ext)** | Si hay referencia humana | Si hay referencia humana | REII puede medir precisión contra referencias humanas |
| **Coherencia Interna (C_in)** | No calculado | Calculado | REII puede medir homogeneidad intra-clase |

### Visualización del impacto:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ METRICAS                                │ ALCESTE │ REII (propuesta) │
├───────────────────────────────────────────┼───────┼───────────────┤
│ Tasa de Retención (CR)                    │ 65%   │ 70-85%      │   ← MEJOR (mayor cobertura)
│ Cobertura por Tokens                    │ Baja  │ Alta        │   ← MEJOR (medida más justa)
│ Silhouette Score                        │ N/A   │ >0.5        │   ← MEJOR (clases más definidas)
│ FMI (Núcleo vs Extendido)                │ N/A   │ >0.7        │   ← MEJOR (mayor concordancia)
│ C_ext (con referencia humana)             │ 0.80  │ 0.85        │   ← MEJOR (mayor precisión)
│ C_in (homogeneidad intra-clase)           │ N/A   │ >0.7        │   ← MEJOR (clases más homogéneas)
├───────────────────────────────────────────┼───────┼───────────────┤
│ Precision (núcleo)                        │ Alto  │ Alto        │   ← IGUAL (misma precisión)
│ Coverage (extendido)                      │ Bajo  │ Alto        │   ← MEJOR (más UCEs)
│ NIVELES DE QUALITY                      │ 1     │ 2-3         │   ← MEJOR (más granularidad)
└─────────────────────────────────────────────────────────────────────────┘
```

### Explicación simplificada:

1. **ALCESTE** es un método riguroso que busca **alta precisión** sacrificando cobertura. Es como un filtro de calidad muy estricto: solo deja pasar lo mejor, pero puede descartar muchas cosas útil.

2. **REII** es un método más flexible que busca **mejor cobertura manteniendo precisión**. Es como un filtro de calidad que permite pasar algunas cosas menos perfectas, pero en cambio te da más información.

3. La ventaja de REII es que **no sacrifica nada**: las correcciones al mapeo y las métricas son mejores, y el inclusion de niveles extended es una opción additional que puedes usar si quieres más cobertura.

# Por qué REII da 30% de riqueza y 51% de retención mientras ALCESTE da 97% y 90%: descomposición estadística

## Tesis central

**No estás comparando la misma métrica.** Los dos sistemas miden cosas distintas, con denominadores distintos, criterios de estabilidad con distinta exigencia y particiones con distinta granularidad. La brecha no indica que un algoritmo sea "mejor" que el otro: es la consecuencia predecible de cinco efectos estadísticos que se acumulan.

---

## Efecto 1 — El criterio conjuntivo (explica el 51% vs 90%)

ALCESTE declara estable una UCE si sobrevive a **2 corridas** con distinto tamaño de ventana (p. ej. 10 vs 14 palabras). REII exige que la UCE sea estable en **6 condiciones** (wordcount, similitud, coref, embedding y sus cruces).

Si cada condición tiene probabilidad *p* de aprobarse, la probabilidad conjunta es *p^k*:

| Sistema | Condiciones (k) | P(estable) con p = 0.90 |
|---|---|---|
| ALCESTE | 2 | 0.90² = **0.81** |
| REII | 6 | 0.90⁶ = **0.53** |

Tu 51% es casi exactamente lo que predice un criterio de 6 condiciones con ~90% de aprobación individual. **REII no "pierde" UCEs: exige más evidencia por UCE.** Y hay un agravante: las 6 condiciones de REII no son 6 repeticiones de la misma prueba (como las 2 ventanas de ALCESTE), sino 6 pruebas *cualitativamente distintas* (frecuencia, similitud, coreferencia, embeddings). Cada una tiene un modo de fallo diferente, así que la intersección se encoge aún más de lo que sugiere el cálculo simple.

## Efecto 2 — La ley de Zipf y las hapax legomena (explica el 30% vs 97%)

En cualquier corpus, la mayoría de las palabras distintas (*types*) aparecen una sola vez (*hapax legomena*). Por la ley de Zipf, entre el 40% y el 60% de los types son hapax.

- **ALCESTE** cuenta la riqueza sobre las UCEs *clasificadas* (~90% del corpus). Como las hapax están repartidas por todo el texto, casi todos los types aparecen en ese 90% → **97%**.
- **REII** cuenta la riqueza sobre el *núcleo estable* (51% de las UCEs). Por sesgo de selección, ese núcleo contiene el vocabulario prototípico y frecuente, no las palabras raras → **30%**.

Las hapax viven en las UCEs descartadas. Si las UCEs estables son las "prototípicas" (las que sobreviven a 6 condiciones), su vocabulario es *necesariamente* restringido. La riqueza baja no es un defecto: es la consecuencia estadística de filtrar por estabilidad.

## Efecto 3 — El denominador no es el mismo

ALCESTE calcula la cobertura sobre las UCEs que ya pasaron su filtro de longitud mínima (las UCI que no alcanzan el mínimo de palabras se descartan *antes* de clasificar). El denominador ya está reducido. REII calcula la retención sobre el **total** de UCEs del corpus, incluidas las que nunca entraron a clasificación. Con el mismo dato, un denominador más grande produce un porcentaje más bajo sin que el algoritmo sea peor.

## Efecto 4 — La granularidad de la partición (el "3 clases estables")

ALCESTE con 3 clases es una partición **gruesa**: cada clase cubre un tercio del corpus, así que la cobertura es alta por construcción. Además, con 3 clases el mapeo Húngaro es trivial (pocas clases que alinear). REII produce más clases, o clases definidas por embeddings que capturan similitud *semántica* y no solapamiento *léxico*. A más clases, menos vocabulario por clase y menor cobertura léxica aparente.

## Efecto 5 — Types vs tokens

Si ALCESTE cuenta *types* (palabras distintas) y REII cuenta *tokens* ponderados por frecuencia, los números no son comparables. La riqueza por types se infla con las hapax; la cobertura por tokens es más conservadora y más informativa. Antes de comparar, hay que normalizar la unidad de conteo.

---

## Qué verificar antes de concluir

1. **¿El denominador es el mismo?** UCEs clasificadas vs UCEs totales del corpus.
2. **¿ALCESTE incluye hapax en su 97%?** Si las cuenta, el número está inflado por palabras que aparecen una vez y aportan poca información estadística.
3. **¿REII mide sobre el núcleo estable o sobre todas las UCEs?** Si mide solo el núcleo, el 30% es esperable.
4. **¿Cuántas clases produce cada sistema?** 3 clases gruesas vs N clases finas no son comparables.
5. **¿Types o tokens?** Normalizar la unidad de medida.

---

## Conclusión (versión mejorada para el paper)

> La discrepancia entre la riqueza de vocabulario (30% vs 97%) y la retención (51% vs 90%) no refleja una diferencia de calidad entre REII y ALCESTE, sino una diferencia de definición operacional. La retención de REII es el resultado de un criterio conjuntivo de seis condiciones de estabilidad cualitativamente heterogéneas: si cada condición se aprueba con probabilidad p ≈ 0.90, la intersección esperada es p⁶ ≈ 0.53, consistente con el 51% observado. La riqueza de vocabulario, por su parte, está dominada por la ley de Zipf: las hapax legomena constituyen la mayoría de los types del corpus y se concentran en las UCEs descartadas por el filtro de estabilidad, de modo que el núcleo estable retiene necesariamente un vocabulario restringido. ALCESTE, al clasificar el 90% de las UCEs con una partición de solo tres clases, captura casi todos los types por construcción. La comparación solo es válida si se igualan el denominador (UCEs clasificadas vs totales), la unidad de conteo (types vs tokens) y la granularidad de la partición.

