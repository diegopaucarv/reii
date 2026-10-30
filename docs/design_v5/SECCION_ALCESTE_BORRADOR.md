# Borrador de la sección «Comparación con ALCESTE 2012»

Estado: **borrador de trabajo**. Sustituye al bloque `📋 Comparación: ALCESTE vs REII
(propuesta del agente)` de `docs/paper.md` (líneas 87-333). El inventario de qué debe
contener la sección, con su justificación, está en `COMPARATIVA_ALCESTE.md`; este
documento es la redacción.

Va destinado a **§4.4** (_Benchmark comparativo_) con un reenvío desde **§5.3**
(_Interpretación de los mundos lexicales_). Las cifras de REII son las de la corrida
`select-v4` persistida (`data/method_selection.json`, huella `93fabdc4d0ff6590`); las de
ALCESTE, las del relatorio de 2012 reconstruido por coordenadas.

Todas las cifras citadas tienen artefacto. La tabla de procedencia está en el §C.

---

## A. Prosa

### 4.4.1 Material y condiciones de la comparación

La comparación se establece sobre el mismo material empírico: las 29 entrevistas que
ALCESTE documenta como unidades de contexto iniciales (UCI) y que aquí constituyen los
29 documentos del corpus (huella `93fabdc4d0ff6590`). La coincidencia no es solo de
tamaño; los perfiles léxicos de ambas soluciones comparten el vocabulario temático
(`tema`, `curso`, `estudiante`, `universidad`, `docente`, `lluvia`, `calor`, `frío`,
`afect-`, `provoc-`), lo que permite sostener que se trata del mismo conjunto textual y
no de dos corpus paralelos. Esta condición es la que hace legítima la comparación
directa y conviene declararla explícitamente, porque la mayor parte de los estudios que
contrastan herramientas de análisis lexicométrico comparan salidas sobre corpus
distintos y, por tanto, no pueden separar el efecto del método del efecto del material.

Las dos soluciones no operan, en cambio, sobre el mismo número de unidades de contexto
elementales (UCE). ALCESTE segmenta el corpus en 376 UCE, con una media de 26
ocurrencias por unidad y 13,95 palabras analizadas por unidad. REII no declara ese
promedio: declara un tamaño objetivo de segmentación y lo calibra contra el corpus. El
valor de entrada es `uce_target_size = 26`, pero con `auto_calibrate_uce_size = True` se
sustituye antes de segmentar por el que reproduce la convención de ALCESTE para un corpus
de este volumen (26 ocurrencias/UCE, tolerancia ±1). La calibración mide los dos extremos
del rango de diseño `[14, 30]` y se queda con el superior, **`uce_target_size = 30`**: con
él la muestra de calibración da 26,86 palabras/UCE, a 0,86 del objetivo y dentro de
tolerancia, y sobre el corpus completo el valor efectivo es 27,15 palabras crudas por
unidad. Con ese ajuste, REII produce 528 UCE con 12,23 lemas analizados por unidad. La
diferencia de tamaño de la unidad es, por tanto, marginal (4 % en el denominador crudo),
y es la primera razón por la que las tasas de retención léxica de ambos sistemas
resultan comparables.

### 4.4.2 Correspondencia de parámetros y divergencias de diseño

La tabla 4.4.1 resume la correspondencia entre los parámetros explícitos de ALCESTE y
los de REII. Cuatro decisiones de diseño separan a los dos sistemas y explican la mayor
parte de las diferencias que se reportan después.

**Tabla 4.4.1.** Correspondencia de parámetros entre ALCESTE 2012 y REII (`select-v4`).

| Función           | ALCESTE 2012                                                    | REII                                                                            | Efecto observable                             |
| ----------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------- |
| Reducción léxica  | REDUC; 428 formas tras reducción, 110 suplementarias            | spaCy + stemming; `tool_adverb_filtered` (NOUN/VERB/ADJ/ADV, sin clase cerrada) | REII retiene 0,451 frente a 0,537             |
| Frecuencia mínima | 4 ocurrencias                                                   | `min_forms_uc_1 = 10`, `forms_gap = 2` (banda [10, 12])                         | REII más restrictivo                          |
| Tamaño de UCE     | doble pasada: 12 y 14 palabras analizadas por UC (296 y 260 UC) | `uce_target_size = 30` efectivo (entrada 26, calibrado)                         | 26 occ/UCE frente a 27,15 palabras crudas/UCE |
| Partición         | CDH + doble clasificación con cruce por chi-cuadrado            | `select-v4`: elige entre A/B/C por ARI de bootstrap, gate de cobertura 0,15     | 0,926 [0,893; 0,958] frente a «no reportado»  |
| Número de clases  | 10 solicitadas, 3 estables                                      | `cdh_target_k = 3`; espacio `[3, 12]`, gate `[3, 20]`                           | ambos K = 3, por mecanismos distintos         |
| Clase mínima      | 19 UCE (5,05 %)                                                 | piso de hoja 21 UCE (4,0 %) vía `hdbscan_uce_min_cluster_frac`                  | 69 UCE (13,1 %) > 21: el piso no muerde       |
| Cobertura         | 68,2 % segmentado, 61,3 % clasificado                           | 100 % segmentado, 95,3 % clasificado                                            | la divergencia estructural                    |

Una precisión sobre la fila del tamaño de la UCE: **no es un hiperparámetro del
optimizador**. `uce_target_size` queda fuera del espacio de Optuna —que muestrea
`forms_gap`, `tsj`, `cdh_target_k`, `swap_iterations`, `min_r2_threshold`,
`similarity_threshold`, `coref_weight`, `uc_window_size`, `coref_context_units_loose`,
`coref_context_units_tight`, `hdbscan_uce_min_cluster_frac` y
`hdbscan_uce_min_cluster_frac_loose`— y lo fija una calibración bayesiana previa a la
segmentación (`auto_calibrate_uce_size`), cacheada por huella del corpus en
`data/uce_calibration.json`. Por eso el valor se reporta como una constante de la
corrida, no como una elección del optimizador: entrada 26, efectivo 30, promedio medido
26,86 en la muestra de calibración y 27,15 sobre el corpus completo. La diferencia entre
las dos mediciones es de muestreo: la calibración estima sobre una muestra de secciones
parseadas y el barrido de `uce-size` mide la segmentación completa.

Primera, la reducción léxica. ALCESTE lematiza con REDUC y distingue formas activas
(sustantivos, verbos, adjetivos) de formas suplementarias (artículos, preposiciones),
que se conservan pero no se analizan; el resultado es de 428 formas tras reducción y 110
formas suplementarias sobre 2 590 formas distintas. REII lematiza con spaCy y aplica un
filtro de categorías gramaticales configurable (`pos_only`, `tool_adverb_filtered`,
`pos_plus_propn`), con `tool_adverb_filtered` como valor por defecto. Su núcleo activo
son sustantivos, verbos, adjetivos y adverbios, y la clase cerrada —pronombres,
conjunciones, determinantes, auxiliares, adposiciones e interjecciones— queda excluida
siempre, con el mismo criterio con el que ALCESTE las declara suplementarias. La
consecuencia inmediata es que REII retiene menos léxico por unidad, y ese es el hecho
que la subsección siguiente cuantifica en lugar de postular. Hay una consecuencia
secundaria que conviene anticipar: los perfiles léxicos de ALCESTE incluyen formas
suplementarias en su tabla de φ (el término de mayor φ de su clase 1 es `para`, una
preposición), mientras que los de REII no pueden incluirlas por construcción. Las dos
tablas de φ son, por tanto, comparables en el método de asociación y no en la clase de
vocabulario sobre la que se calculan.

Segunda, la frecuencia mínima de análisis. ALCESTE fija el umbral en 4 ocurrencias;
REII lo fija en `min_forms_uc_1 = 10` con `forms_gap = 2`, es decir, una banda [10, 12].
El umbral de REII es, por tanto, más exigente, y actúa en la misma dirección que el
filtro de categorías: reduce el vocabulario analizado y, con él, la retención.

Tercera, la construcción de la partición. ALCESTE ejecuta una **doble clasificación**
(protocolo 121) variando ligeramente la ventana de contexto, cruza ambas particiones por
chi-cuadrado y retiene únicamente las UCE y clases estables en el cruce. REII sustituye
esa doble clasificación por una regla de selección pre-registrada (`select-v4`) que
enfrenta tres implementaciones —recuento de palabras (A), similitud (B) y agrupamiento
denso sobre embeddings (C)— y elige ganador por estabilidad intra-método estimada por
bootstrap, con un piso de cobertura útil de 0,15. La lógica es distinta: ALCESTE filtra
por acuerdo entre dos ejecuciones del mismo algoritmo; REII elige entre algoritmos
distintos y cuantifica la incertidumbre de la elección.

Cuarta, el número de clases. ALCESTE solicita 10 clases y retiene 3 estables, con un
mínimo de 19 UCE por clase (5,05 % de las 376) y un 90 % de UCE clasificadas. REII no
tiene un valor de K fijado por el sistema: `cdh_target_k` es un hiperparámetro del
optimizador, `−K` es uno de sus tres objetivos y el espacio realmente muestreado es
`cdh_target_k ∈ [3, 12]`, mientras que el gate de admisión de la regla acepta
`k_bounds = [3, 20]`. El valor elegido fue 3, el extremo inferior. Conviene además no
leer `min_cluster_size_cdh = 0,1667` como un parámetro independiente: en el método C se
deriva como `1 / (2 · cdh_target_k)`, de modo que 0,1667 y K = 3 son la misma decisión
expresada en dos unidades, y el piso de hoja efectivo lo fija
`hdbscan_uce_min_cluster_frac` (0,03885 × 528 ≈ **21 UCE, 4,0 %**). Esta diferencia es
decisiva para leer la coincidencia de K que se discute en 4.4.5.

### 4.4.3 Retención léxica: misma definición, denominadores comparables

El indicador de retención es, en ambos sistemas, el cociente entre las palabras
efectivamente analizadas por unidad de contexto y las palabras de esa unidad antes del
filtro. En ALCESTE, 13,95 / 26 = **0,537**. En REII, con la política por defecto
(`tool_adverb_filtered`), 12,23 / 27,15 = **0,451**, con intervalo de confianza al 95 %
de [0,445; 0,457] estimado por bootstrap sobre las 528 UCE (1 000 réplicas, semilla 42).
Una medición independiente sobre la corrida persistida en PostgreSQL devuelve 0,452,
lo que confirma el valor dentro del error de redondeo del denominador.

La lectura correcta es que **REII retiene entre 15 y 16 puntos porcentuales menos léxico
que ALCESTE**, y que esa brecha es atribuible al filtro morfosintáctico y al umbral de
frecuencia, no a una definición distinta del indicador. Es un resultado empírico y no
una diferencia de unidad de medida. La alternativa más permisiva, `pos_plus_propn`,
alcanza 0,485 (13,15 lemas por UCE) y se acerca a ALCESTE, pero queda excluida por
defecto porque su cuota de palabras herramienta asciende a 0,375, por encima del límite
de 0,30 que el propio diseño se impone; la política más estricta, `pos_only`, se sitúa
en 0,395. El rango completo [0,395; 0,485] delimita, por tanto, el margen de maniobra
del analista y hace visible que la retención no es una propiedad del método sino de una
decisión declarada.

### 4.4.4 Cobertura: la diferencia estructural

Si la retención es comparable, la cobertura no lo es, y esta es la divergencia que debe
ocupar el centro de la comparación. ALCESTE analiza 376 UCE de 26 ocurrencias, esto es,
9 776 ocurrencias sobre las 14 337 del corpus: el **68,2 %** del material. Como además
clasifica el 90 % de esas UCE, la proporción de texto efectivamente adscrita a una clase
es del **61,3 %**. REII segmenta todo el corpus (528 UCE × 27,15 palabras crudas
≈ 14 335, frente a las 14 333 medidas) y su cobertura útil —UCE que sobreviven a la
perturbación de estabilidad— es del **95,3 %**.

La diferencia no es un artefacto de configuración: para producir 376 UCE con cobertura
total, REII necesitaría `uce_target_size ≈ 42`, un valor fuera del espacio explorado
`[14, 30]` (la identidad `n_UCE × target ≈ 15 690` lo fija con un ajuste de
R² = 0,976). Dicho de otro modo, ALCESTE y REII no discrepan en cuánto léxico conservan
por unidad, sino en **cuánto corpus deciden mirar**. ALCESTE concentra el análisis en el
61 % del texto que su segmentación y su filtro consideran analizable; REII analiza el
corpus completo y traslada la pérdida al interior de cada unidad. Esta es la afirmación
que la sección debe sostener, y conviene formularla como un intercambio explícito
(cobertura frente a densidad léxica) y no como una ventaja unilateral.

### 4.4.5 Partición: coincidencia de K sin validación independiente

Ambos sistemas producen tres clases. La clase menor de ALCESTE tiene 19 UCE (5,05 % de
376). En REII hay que distinguir dos particiones y elegir una de forma consistente: la
**partición evaluada** en `method_selection.json`, de 69 / 212 / 247 UCE sobre las 528
(13,1 %, 40,2 % y 46,8 %), y la **partición persistida** junto a los perfiles léxicos, de
52 / 207 / 244 sobre las 503 UCE bootstrap-estables (10,3 %, 41,2 % y 48,5 %). No son dos
particiones distintas: la segunda es la primera una vez retiradas las 25 UCE (4,7 %) que
no superan la máscara de estabilidad, y cada clase sólo pierde unidades. Si se citan
perfiles, debe citarse también la base de 503.

La coincidencia en K es notable pero **no constituye una validación cruzada**, y el texto
debe decirlo con claridad. La razón es que 3 es el límite inferior del espacio realmente
muestreado (`cdh_target_k ∈ [3, 12]`) y además el valor que el optimizador fijó como meta,
siendo `−K` uno de los tres objetivos. Un K = 3 en estas condiciones no es un hallazgo
emergente del corpus: es parcialmente una consecuencia de cómo se parameterizó la
búsqueda. Hay un segundo motivo, y conviene formularlo con precisión para no exagerarlo:
la configuración ganadora se sitúa en el extremo más grueso de esa parameterización
—`min_cluster_size_cdh = 1 / (2 · cdh_target_k) = 0,1667`, que es el valor máximo que el
cociente puede tomar—, pero **no son dos bordes independientes**: es el mismo borde visto
en dos unidades. Un óptimo en la frontera indica que el óptimo verdadero puede estar fuera
del espacio explorado, y debe reportarse como limitación, no como convergencia. Conviene
añadir que el piso de hoja que realmente operó fue `max(5, round(0,03885 × 528)) = 21 UCE
(4,0 %)`, muy por debajo de la clase menor obtenida: el piso no fue vinculante.

Lo que sí es un resultado propio de REII es la estimación de la incertidumbre de la
partición, que ALCESTE no reporta. La regla `select-v4` no elige entre dos pasadas del
mismo algoritmo, sino entre tres implementaciones que compiten sobre la misma
segmentación: **A**, recuento de palabras (la doble clasificación descendente que
reproduce el protocolo de ALCESTE); **B**, similitud con correferencia; y **C**,
agrupamiento denso sobre embeddings de coocurrencia. Las tres se evalúan con el mismo
protocolo —ARI entre sus dos pases, ARI de bootstrap sobre las 528 UCE y cobertura
útil— y la tabla 4.4.2 recoge el resultado.

**Tabla 4.4.2.** Las tres implementaciones de `select-v4` sobre las 528 UCE (huella
`93fabdc4d0ff6590`, semilla 42).

| Método                            | K     | ARI entre pases | ARI bootstrap [IC95]     | Cobertura | Cobertura útil | Clase menor / mayor | Veredicto                 |
| --------------------------------- | ----- | --------------- | ------------------------ | --------- | -------------- | ------------------- | ------------------------- |
| **A** — recuento de palabras      | 4     | 0,489           | 0,496 [0,405; 0,601]     | 0,763     | 0,612          | 93 / 150            | segundo                   |
| **B** — similitud / correferencia | 5     | 0,288           | 0,101 [−0,005; 0,222]    | 0,525     | 0,118          | 8 / 270             | descartado (0,118 < 0,15) |
| **C** — embeddings, agrup. denso  | **3** | 1,000           | **0,926 [0,893; 0,958]** | **1,000** | **0,953**      | 69 / 247            | **ganador**               |

Dos lecturas se siguen de la tabla, y ambas refuerzan la cautela sobre la coincidencia de
K. La primera es que el valor nulo por permutación de etiquetas separa lo que el ARI
entre pases, por sí solo, confundiría: B presenta un ARI de 0,288, aparentemente
aceptable, pero su bootstrap es 0,101 con un intervalo que contiene el cero (z = 16,5),
mientras que C alcanza 0,926 con un intervalo estrictamente positivo (z = 212,5). B no
es una partición ruidosa por muestreo: es una partición que no se reproduce, y su
cobertura útil de 0,118 la deja por debajo del piso de 0,15 que la regla exige antes de
comparar estabilidad. La segunda es que **las tres implementaciones no producen el mismo
número de clases**: A devuelve 4, B devuelve 5 y solo C devuelve 3. La coincidencia con
ALCESTE es, por tanto, una propiedad de la implementación ganadora y no de REII como
sistema; un desempate favorable a A habría reportado cuatro clases y uno favorable a B,
cinco. Esto no invalida la comparación, pero impide presentar el 3 como un resultado
robusto a la elección de método, y es la razón por la que la coincidencia se reporta como
concordancia de granularidad y no como acuerdo.

Una salvedad afecta a la lectura de la primera columna de estabilidad. El ARI entre pases
de C es 1,000, y no debe leerse como evidencia de calidad: el backend denso fija K por
construcción y no emite unidades sin asignar, de modo que sus dos pases coinciden por
diseño cuando el piso de hoja no es vinculante —y aquí no lo fue, porque la clase menor
(69 UCE) está muy por encima del piso (21 UCE, 4,0 %)—. La cifra informativa es el
bootstrap (0,926), no el acuerdo bruto entre pases; es también la razón por la que la
regla selecciona por estabilidad de bootstrap y no por ARI entre pases.

### 4.4.6 Contenido de las clases: donde la coincidencia se rompe

La comparación de perfiles léxicos —estadístico φ en ALCESTE, φ de la tabla de términos
en REII— muestra que la igualdad de K oculta una estructura distinta. Las tres clases de
REII se ordenan así: un eje de **universidad y currículo** (`tema` 0,31; `curso` 0,30;
`estudiante` 0,23; `universidad` 0,23; `import-` 0,21; `enseñ-` 0,20; 97 términos con
asociación nominal), un eje de **impacto climático y corporal** (`lluvia` 0,30; `calor`
0,26; `provoc-` 0,26; `frío` 0,25; `intens-` 0,24; `afect-` 0,23; `extrem-` 0,20; 102
con asociación nominal) y un eje de **registro enunciativo** (`ver` 0,74; `ir` 0,40;
`usar` 0,22; `realidad` 0,20; 15 con asociación nominal). Los valores citados son φ
**sin** corrección de continuidad, que es la columna comparable con ALCESTE.

Los perfiles se calculan sobre las 503 UCE con clase asignada (las 25 inestables quedan
fuera, como en la retención), y φ es la asociación 2×2 con signo entre la presencia de la
forma en la UCE y la pertenencia a la clase, es decir la misma familia de medida que usa
ALCESTE. Tres diferencias impiden leer los valores de forma literal. Primera, la
corrección de continuidad que el pipeline aplica por defecto (Yates) deflacta φ de forma
dependiente de la fuerza de la asociación: el cociente φ_Yates/φ_sin tiene mediana 0,89
en los términos con |φ| ≥ 0,10, 0,95 en los de |φ| ≥ 0,15 y 0,98 en los más fuertes,
pero sobre el conjunto completo baja a 0,60 y llega a **anular** el estadístico en 705 de
los 2 115 pares término-clase. Por eso la tabla publicable usa φ sin Yates
(`phi_sin_yates`, recalculado desde los recuentos 2×2 con `reii.diag term-metrics`); la
columna de Yates se conserva como control. Segunda, los recuentos de 102 / 97 / 15
términos son asociaciones **nominalmente** significativas: la corrección de
Benjamini–Hochberg se calcula por clase en `p_adj`, pero el flag `significativo` compara
el p-valor crudo, de modo que solo 74 de los 214 términos marcados sobreviven al FDR
(33 / 37 / 4); o se usa el flag `significativo_fdr` o se presentan como tamaño de la
tabla de términos, nunca como hallazgos confirmados. Tercera, la clase pequeña de REII
(52 de 503) eleva el techo alcanzable de φ: el máximo observado es 0,74, mientras que
ALCESTE, con clases de ~58/13/26 %, no pasa de 0,43. Los máximos no son comparables
entre sistemas aunque el significado de cada valor sí lo sea.

La correspondencia con ALCESTE es parcial y asimétrica. La clase 1 de REII reproduce el
perfil de la clase 1 de ALCESTE (`para` 0,38; `curso` 0,30; `tema` 0,29; `deber` 0,28;
`enseñ-` 0,24; `docente` 0,22), aunque con una salvedad que el propio relatorio permite
verificar: el término de mayor φ de ALCESTE es una preposición —forma suplementaria—, de
modo que la coincidencia se sostiene sobre los términos de contenido (`curso`, `tema`,
`deber`, `enseñ-`, `docente`) y no sobre la cima de la tabla. La clase 0 de REII **funde** las clases 2 y
3 de ALCESTE, que el sistema francés separa en un eje mediático-global (`televisión`
0,43; `referir` 0,39; `calor` 0,30; `cambio` 0,28; `tala` 0,27; `contaminación` 0,25;
`noticias` 0,24) y un eje de daño local y económico (`afect-` 0,43; `frío` 0,34;
`lluvia` 0,30; `piel` 0,29; `helada` 0,29; `precio` 0,25; `sequía` 0,26; `tos` 0,25;
`agricultura` 0,25). Donde ALCESTE separa dos modulaciones del mismo tema, REII conserva
una sola. Y a la inversa, REII aísla un eje que ALCESTE no retiene entre sus tres clases
estables: el de la enunciación de la entrevista misma (`ver`, `ir`, `usar`), con un φ de
0,73 que lo convierte en la asociación más fuerte de toda la tabla.

Tres lecturas son posibles y el texto debe presentarlas sin elegir por el lector. La
primera es que ALCESTE tenga mayor resolución temática y REII la pierda por agregación.
La segunda es que la escisión de ALCESTE entre lo mediático y lo corporal sea un efecto
de su doble clasificación y de su filtro de chi-cuadrado, y que la clase fusionada de
REII sea la unidad temática real. La tercera es que ambas soluciones estén recortando
dimensiones distintas de un mismo espacio: ALCESTE, ejes de contenido; REII, un eje de
contenido y un eje de género discursivo. Decidir entre ellas exige un estándar humano de
referencia que hoy no existe, y así debe declararse.

### 4.4.7 Alcance y límites de la comparación

Seis límites acotan lo que esta sección puede afirmar. Primero, la comparación de
perfiles se apoya en los 20 términos característicos por clase que el relatorio publica
(60 de 376 UCE), no en la matriz completa de asignación UCE-clase, que no es un
artefacto disponible; cualquier conclusión sobre solapamiento fino de contenido es, por
tanto, indicativa. Segundo, no existe partición de referencia humana, de modo que ni
ALCESTE ni REII pueden ser evaluados aquí con ARI contra un estándar externo: la
estabilidad se mide contra perturbaciones del propio método, no contra la verdad.
Tercero, el índice de riqueza de vocabulario que reporta ALCESTE (97,84 %) tiene un
denominador no documentado y no debe usarse en la comparación. Cuarto, la coincidencia
de K, por las razones expuestas en 4.4.5, no puede presentarse como evidencia
independiente de acuerdo entre sistemas.

Quinto, los dos φ no son el mismo número. Coinciden en fórmula (asociación 2×2 con signo
sobre presencia en UCE), pero difieren en el n (503 frente a 376) y sobre todo en el
universo léxico: ALCESTE incluye formas suplementarias y REII no, de modo que los
rankings no son equiparables término a término aunque las magnitudes sí sean de escala
comparable. La corrección de continuidad ya no es una diferencia: `phi_sin_yates` la
elimina, y es la columna que debe usarse en cualquier tabla comparativa. Sexto, la lectura
del propio relatorio exige cautela: la leyenda de su tabla de vocabulario etiqueta
equivocadamente `Effectif` y `Total` —el valor de 150 ocurrencias de `para` no puede ser
un recuento de UCE cuando el total de UCE es 128—, por lo que las columnas se han
interpretado por consistencia aritmética sobre las 348 filas publicadas. Conviene citar
el relatorio por sus valores de φ y no por sus encabezados.

### 4.4.8 Cierre

En conjunto, la comparación sostiene una conclusión más estrecha —y más defendible— que
la de superioridad: ALCESTE y REII operan sobre la misma unidad de contexto y conservan
fracciones comparables del léxico de cada unidad (0,537 frente a 0,451, con
denominadores a un 4 % de distancia), pero difieren en la extensión del material
analizado (61,3 % del texto frente a 95,3 %), en el criterio con que eligen la partición
(doble clasificación frente a selección pre-registrada entre algoritmos) y en el grado
en que hacen explícita su propia incertidumbre (no cuantificada frente a ARI de
bootstrap 0,926 con IC95 [0,893; 0,958]). La coincidencia en el número de clases no debe
leerse como validación mutua, sino como el punto donde dos heurísticas distintas se
cruzan sobre un mismo corpus.

---

## B. Figuras

Las diez figuras están justificadas en `COMPARATIVA_ALCESTE.md` §6. Los pies están
redactados para insertarse tal cual. Marca de estado: **[listo]** = datos ya en disco;
**[1 comando]** = falta una ejecución corta; **[pendiente]** = falta un artefacto que
hoy no existe.

**F1 — Escalera de cobertura. [listo]**
Barras apiladas horizontales, dos filas (ALCESTE, REII), tres segmentos: corpus total,
texto segmentado en UCE, texto efectivamente clasificado. ALCESTE 100 % → 68,2 % →
61,3 %; REII 100 % → 100 % → 95,3 %. _Pie:_ «Del corpus a la clase: ALCESTE analiza el
68,2 % del texto y clasifica el 61,3 %; REII segmenta el corpus completo y clasifica el
95,3 %. La diferencia no está en la unidad de contexto sino en la extensión analizada.»

**F2 — Calibración de la segmentación. [listo]**
Dos paneles: `n_UCE` frente a `uce_target_size` (puntos 18/22/26/30 → 866/715/600/528,
R² = 0,976) y palabras crudas por UCE frente al mismo eje (16,6/20,0/23,9/27,1,
R² = 0,9987). Línea vertical marcando `target = 26` (equivalencia con ALCESTE) y banda
sombreada marcando `target ≈ 42`, fuera del espacio. _Pie:_ «La segmentación es
gobernada por una identidad estable: fijar el tamaño de la UCE fija su número. Reproducir
las 376 UCE de ALCESTE con cobertura completa exigiría un tamaño de 42, fuera del rango
explorado.»

**F3 — Retención léxica con incertidumbre. [listo]**
Puntos con IC95 para `pos_only` (0,395), `tool_adverb_filtered` (0,451) y
`pos_plus_propn` (0,485); línea horizontal en 0,537 (ALCESTE); línea discontinua en 0,30
(cuota máxima de palabras herramienta) anotando la exclusión de la variante permisiva.
_Pie:_ «Retención léxica por política de filtrado. La definición es la misma en ambos
sistemas; la brecha con ALCESTE se explica por el filtro morfosintáctico y el umbral de
frecuencia, no por un cambio de denominador.»

**F4 — Selección de método. [listo]**
Gráfico de halterios (dumbbell) por método A/B/C con dos ejes: ARI de bootstrap (con
IC95) y cobertura útil. Línea de gate en 0,15. C: 0,926 [0,893; 0,958] y 0,953; A: 0,496
[0,405; 0,601] y 0,612; B: 0,101 [−0,005; 0,222] y 0,118 (descartado). _Pie:_ «La regla
`select-v4` elige C por estabilidad intra-método y descarta B por cobertura útil
insuficiente. Los intervalos hacen visible que la ventaja de C no depende del azar de una
sola partición.»

**F5 — Perfiles de clase, ALCESTE frente a REII. [listo]**
Mapa de calor con los términos de mayor φ por clase, tres columnas por sistema, escala
de color común, celdas vacías donde el término no aparece. Para REII se usa la columna
**φ sin Yates** (`phi_sin_yates`, calculada por `reii.diag term-metrics`), que es la
comparable con ALCESTE; la de Yates queda como columna de control. Dos notas al pie:
los lemas de REII están truncados por stemming (`lluvi`, `afect`) y su base es de 503
UCE, frente a las 376 de ALCESTE. _Pie:_ «Perfiles léxicos de las tres
clases. La clase 1 coincide (universidad y currículo); la clase 0 de REII fusiona las
clases 2 y 3 de ALCESTE (clima global y daño local); la clase 2 de REII es un eje
enunciativo que ALCESTE no retiene. Las magnitudes de φ son comparables en escala, no en
valor: difieren el universo léxico y la base (503 frente a 376).»

**F6 — Estabilidad de K bajo la especificación. [listo]**
Panel izquierdo: K frente a `uce_target_size` dentro de la especificación constante
(18/22/26 → K = 7), mostrando que el tamaño de la UCE no altera el número de clases.
Panel derecho: K para un mismo `lem_total` y target con especificaciones distintas
(5658 → K ∈ {3, 7, 11, 16, 17}; 6465 → K ∈ {3, 4, 4, 4, 6}). _Pie:_ «La segmentación está
fija y el número de clases lo determina el clasificador, no el tamaño de la unidad. La
inestabilidad de K es un problema del agrupamiento, no de la segmentación.»

**F7 — Matriz de acuerdo ALCESTE × REII. [1 comando]**
Matriz de contingencia 3 × 3 (UCE por clase en ambos sistemas) con ARI y NMI en el
margen. Se construye sobre las 60 UCE que el relatorio publica como más características
(20 por clase, págs. 24/36/49), no sobre las 376 completas: es la opción (b) de
`COMPARATIVA_ALCESTE.md` §8. Falta un paso de emparejado (casar cada una de las 60 UCE
de ALCESTE con su UCE de REII por texto), que es un script corto. _Pie:_ «Acuerdo de
asignación UCE→clase entre ALCESTE y REII sobre las 60 UCE más características de
ALCESTE (20 por clase). Es una cota **superior** (optimista) del acuerdo real: al
seleccionar los casos más prototípicos se sobreestima la concordancia que se obtendría
sobre las 376 UCE completas, y la muestra excluye las UCE ambiguas. Se publica como
apéndice con esta advertencia, no como medida del acuerdo global.»

**F8 — Distribución de la estabilidad. [listo]**
Curvas ECDF del ARI de bootstrap por método, con la partición de referencia marcada en
1,0. _Pie:_ «Distribución completa —no solo la media— del acuerdo entre remuestreos. El
solapamiento entre la curva de C y la de A es mínimo; la de B cruza el cero.»

**F9 — Plano factorial. [listo]**
Generado por `reii.diag term-metrics --figure`; fichero
`docs/figs/F9_plano_factorial.svg` (la versión `.png` a 200 ppp usada para revisión
visual vive en `data/diag/F9_preview.png`). Biplot de las 503
UCE con clase (color por clase) y de los términos, al estilo del plano factorial de
ALCESTE. Tres correcciones respecto del artefacto guardado: (a) los ejes declaran la
inercia real por eje (0,99 % y 0,79 %), no el reparto relativo entre los dos ejes
retenidos que guarda `explained_inertia` (55,6 % / 44,4 %); (b) para los términos se usa
el nivel de **presencia** de cada par de categorías —`col_coords` trae dos filas por
forma y el render tomaba la de ausencia, colapsando los términos en el origen—; (c) las
coordenadas de columna se reescalan a principales (×√λ) para que sean comparables con
las de fila, y el pie declara la base. _Pie:_ «Plano factorial de la partición de REII
sobre las 503 UCE bootstrap-estables (95,3 %); las 25 restantes quedan sin clase. La
partición es la del ganador de `select-v4` (método C, `dense_cdh`), restringida a las
UCE estables.»

**F10 — Apéndice de diagnóstico. [listo]**
Dos paneles: centinelas del optimizador agrupados por `gate_reason`, y ρ de Spearman
parámetro-objetivo por etapa con los casos de inversión de signo marcados. _Pie:_
«Auditoría del optimizador: de dónde sale el presupuesto perdido y qué parámetros cambian
de signo entre etapas. Material de apéndice, no de cuerpo.»

---

## C. Tabla de procedencia de las cifras

| Cifra                         | Valor                     | Artefacto                                               | Campo                               |
| ----------------------------- | ------------------------- | ------------------------------------------------------- | ----------------------------------- |
| UCI / documentos              | 29 / 29                   | relatorio p. 1 · `data/workflow_data.json`              | `doc_metadata`                      |
| UCE ALCESTE                   | 376                       | relatorio p. 1                                          | —                                   |
| Ocurrencias / UCE ALCESTE     | 26 / 13,95                | relatorio p. 1                                          | —                                   |
| UCE REII                      | 528                       | `data/method_selection.json`                            | `context.n_uces`                    |
| UCE estables (bootstrap)      | 503                       | `data/workflow_data.json`                               | `uces[].is_stable` (recuento)       |
| `uce_target_size` efectivo    | 30 (entrada 26)           | `data/uce_calibration.json` · `data/workflow_data.json` | `target` · `config.uce_target_size` |
| Palabras crudas / UCE         | **27,15** (corpus entero) | `data/workflow_data.json` (528 UCE, 14 333 palabras)    | derivado: 14 333 / 528 — ver C.1    |
| Promedio de calibración       | 26,86 (muestra, tol. ±1)  | `data/uce_calibration.json`                             | `measured_avg` — ver C.1            |
| Lemas / UCE (default)         | 12,23 [11,92; 12,54]      | `retention_policy_report.py`                            | `tool_adverb_filtered`              |
| Retención REII                | 0,451 [0,445; 0,457]      | idem                                                    | idem                                |
| Retención ALCESTE             | 0,537                     | 13,95 / 26                                              | —                                   |
| Métricas A/B/C (tabla 4.4.2)  | K, ARI, cobertura         | `data/method_selection.json`                            | `selection.all_metrics[]`           |
| Cobertura útil (C)            | 0,9527                    | `data/method_selection.json`                            | `all_metrics[2].usable_eff`         |
| K ganador                     | 3                         | idem                                                    | `selection.winner_k`                |
| Tamaños de clase (528 UCE)    | 69 / 212 / 247            | idem                                                    | `min/median/max_class_size`         |
| Tamaños de clase (503 est.)   | 52 / 207 / 244            | `data/workflow_data.json`                               | `multivariate.cluster_labels`       |
| Lemas por clase (503 est.)    | 6 197 = 617/2 563/3 017   | idem                                                    | `words_per_cluster`                 |
| ARI bootstrap (C)             | 0,926 [0,893; 0,958]      | idem                                                    | `bootstrap_ari`, `bootstrap_ci_*`   |
| z nulo (C)                    | 212,5                     | idem                                                    | `null_z`                            |
| `k_bounds` (gate de la regla) | [3, 20]                   | idem                                                    | `selection.k_bounds`                |
| `cdh_target_k` (muestreado)   | [3, 12]                   | `src/reii/main_workflow.py`                             | `_CDH_TARGET_K_HI`                  |
| `min_cluster_size_cdh`        | 0,1667 = 1/(2·K)          | `data/best_params.json`                                 | `best_params`                       |
| n de φ (REII)                 | 503                       | `data/diag/paper_terms.json`                            | `n_uces_con_clase`                  |
| φ sin Yates (REII)            | ver 4.4.6                 | idem                                                    | `terms[].phi_sin_yates`             |
| Deflación de Yates            | 0,89 / 0,95 / 0,98 ×      | idem                                                    | `yates_deflation`, por \|φ\|        |
| Términos con FDR < 0,05       | 74 de 214 (33/37/4)       | idem                                                    | `terms[].significativo_fdr`         |
| Perfiles φ ALCESTE            | ver 4.4.6                 | relatorio págs. 24 / 36 / 49                            | —                                   |
| Inercia MCA por eje           | 0,99 % / 0,79 %           | idem                                                    | `mca.explained_inertia_corrected`   |
| Estabilidad de K por target   | K = 7 en 18/22/26         | `reii.diag multiverse`                                  | salida en consola                   |

---

### C.1 Las tres mediciones del tamaño de UCE no tienen la misma base

Las cifras de tamaño de UCE proceden de tres medidas distintas y **solo la segunda es
una propiedad del corpus**. Confundirlas es el modo más fácil de publicar un número que
no se sostiene.

| Medición                                         | Valor     | Base de la medición                                                                               | Artefacto                       | Campo                  |
| ------------------------------------------------ | --------- | ------------------------------------------------------------------------------------------------- | ------------------------------- | ---------------------- |
| Promedio de calibración (el que elige el target) | 26,86     | Muestra de secciones parseadas con semilla fija (`muestra_para_calibracion`), no el corpus entero | `data/uce_calibration.json`     | `measured_avg`         |
| **Promedio del corpus completo**                 | **27,15** | 528 UCE sobre 14 333 palabras crudas, corpus entero                                               | `data/workflow_data.json`       | derivado: 14 333 / 528 |
| Barrido de `uce-size` en `target = 30`           | 27,1      | Log de la corrida de barrido, marcada `incomplete`                                                | `data/diag/uce_size_bench.json` | `mean_words_per_uce`   |

- La cifra que debe aparecer en el texto es **27,15**: es la única medida sobre el corpus
  completo y la que cierra la identidad 528 × 27,15 ≈ 14 333.
- 26,86 es una **estimación de calibración**: sirve para decidir el `target` (cae a 0,86
  del objetivo 26,0, dentro de la tolerancia ±1) y no para describir el material.
- 27,1 es el mismo parámetro medido en una corrida de barrido que no terminó; coincide
  con 27,15 dentro del redondeo a un decimal y no se usa como fuente.

---

## D. Comprobaciones antes de publicar

Las cuatro discrepancias que frenaban la publicación están cerradas, y las cuatro
acciones derivadas de ellas están ejecutadas o decididas. Estado:

- **(a) Partición única y denominadores explícitos — HECHO (regla de redacción).** Se
  publica 69 / 212 / 247 sobre las 528 UCE (salida del select, la que citan
  `method_selection.json` y la tabla 4.4.1) y 52 / 207 / 244 sobre las 503 estables solo
  en los pies de perfiles, estabilidad o plano factorial. Toda cifra de clase lleva su
  denominador en la misma frase.
- **(b) φ sin Yates — HECHO (código + artefacto).** `TermAnalyzer.compute` persiste ahora
  `chi2_sin_yates`, `p_valor_sin_yates` y `phi_sin_yates` junto a los de Yates, y
  `python diag.py term-metrics` los deriva del artefacto actual sin relanzar el pipeline.
  Deflación medida: ratio φ_Yates/φ_sin de 0,597 global, 0,685 (|φ| ≥ 0,05), 0,893
  (≥ 0,10), 0,953 (≥ 0,15) y 0,978 en los términos con |φ| ≥ 0,30. Los ratios por umbral
  se recalculan de `terms[].phi_sin_yates` y `terms[].phi` en
  `data/diag/paper_terms.json`. Yates anula por completo 705 de las 2 115 filas
  término-clase. La tabla del paper usa φ sin Yates.
- **(c) Flag de significación.** El pipeline escribe ahora `significativo_fdr`
  (`p_adj < 0,05`) junto al histórico `significativo` (p crudo). Resultado sobre la
  corrida persistida: **74 de 214** términos sobreviven al FDR (33 / 37 / 4). El texto del
  paper debe citar 74, no 214, cuando hable de hallazgos confirmados.
- **(d) F9 regenerable.** Corregidos los dos defectos de origen en `run_mca`
  (`explained_inertia = eigen / total_inertia`, no `eigen / eigen.sum()`, y
  `col_coords` al nivel de presencia `__1` en vez de `[::2]`) y en el render del
  dashboard. La figura se regenera con un comando:

```powershell
python diag.py term-metrics --figure docs/figs/F9_plano_factorial.svg --top 8
```

Comprobado y sin acción pendiente: la partición de `multivariate` **no** es una segunda
partición (es el `cluster_id` del ganador C restringido a las 503 UCE estables, 69−17 /
212−5 / 247−3); los 6 465 lemas son los de las 528 UCE y los 6 197 los de las 503 con
clase, de modo que 6 197 + 268 = 6 465 cierra exacto; y `min_cluster_size_cdh = 0,1667`
**no** es un piso de clase: en el método C se convierte en el selector de K
(`target_k = round(1/(2·frac)) = 3`) y el piso de hoja efectivo es
`max(5, round(0,0388547 · 528)) = 21` UCE, que 69 no roza.

Validación de la derivación: `term-metrics` reconstruye la tabla 2×2 desde los recuentos
y comprueba que χ² de Yates y φ coinciden con los persistidos (Δmax = 5·10⁻⁵, el redondeo
a cuatro decimales). Si un artefacto futuro no cuadra, el comando falla con rc=1 en vez
de publicar cifras silenciosamente distintas.

---

## E. Estado de las referencias a `docs/patches/`

Tras mover el patchset a `tools/archive/patchset_v5_v6/`, quedaban referencias que ya no
resolvían. Estado tras la revisión:

| Fichero                                                | Línea             | Contenido                                                 | Estado                                                  |
| ------------------------------------------------------ | ----------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| `docs/design_v5/RETENTION_POLICY.md`                   | 10, 431, 437, 438 | ruta del patch 13                                         | **corregido** (apunta a la ruta nueva)                  |
| `docs/design_v5/RETENTION_ROOTCAUSE.md`                | 513, 521          | `docs/patches/11_formas_activas.patch`                    | **corregido**                                           |
| `docs/design_v5/PAPER_AMENDMENTS.md`                   | 4                 | registro fechado: «no se ha modificado `docs/patches/**`» | dejar (es un registro histórico) o anotar la ruta nueva |
| `tools/archive/patchset_v5_v6/PATCHSET_V5_RUNBOOK.ps1` | 53-142            | ~30 literales `docs/patches/...`                          | archivo: marcar como histórico o reescribir             |
| `tools/archive/patchset_v5_v6/PATCHSET_V6_RUNBOOK.ps1` | 112, 149          | `Join-Path $Root "docs/patches/$p"`                       | idem                                                    |
| `tools/archive/patchset_v5_v6/patches/README.md`       | 92-140            | orden de aplicación con rutas viejas                      | idem                                                    |

Si se quiere que los runbooks archivados vuelvan a ser ejecutables (no recomendado: son
historia), basta reescribir las rutas dentro de `tools/archive/patchset_v5_v6/`:

```powershell
$dir = 'tools/archive/patchset_v5_v6'
Get-ChildItem $dir -Filter *.ps1 -Recurse | ForEach-Object {
    (Get-Content $_.FullName -Raw) -replace 'docs/patches/', 'tools/archive/patchset_v5_v6/patches/' |
        Set-Content $_.FullName -NoNewline
}
```

Ojo: `PATCHSET_V6_RUNBOOK.ps1` calcula `$Root = Split-Path -Parent $PSScriptRoot`, que ahora
es `tools/archive/` y no la raíz del repo. Aunque se arregle la ruta de los patches, el
runbook seguiría ejecutándose desde el directorio equivocado; para reactivarlo haría falta
fijar `$Root` a mano.
