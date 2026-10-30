# Causa raíz de la baja retención de formas activas

**Síntoma reportado** (`src/7_quality_multiverse.py`, 11 corridas): `lemas/UCE = 9.0`
frente a `13.95` de ALCESTE; retención `0.400` frente a `0.537`; retención de UCEs
estables `40-52 %`.

**Fecha:** 2026-10-28 · **Ámbito:** sólo lectura de `src/`, `data/`, `docs/`.
No se ha ejecutado el pipeline, ni pytest, ni contenedores.

---

## 0. Veredicto

La hipótesis del usuario es **parcialmente correcta**:

| Punto                                                           | Veredicto                                                   | Evidencia decisiva                                                                                                                                                                                                                        |
| --------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) `not is_stop` elimina adverbios/verbos/adjetivos frecuentes | **Correcta en el fondo, errónea en el punto de aplicación** | El `not is_stop` de `main_workflow.py:2531` es un **predicado muerto**: eliminarlo no cambia nada. El filtro real está en `main_workflow.py:2520-2526`, que manda _todo_ token `is_stop` a `marcadores` antes de llegar a la rama activa. |
| (b) `content_pos` excluye `PROPN` y `X`                         | **Parcial**                                                 | `PROPN` sí aporta (**293** tokens, +5.2 %). `X` aporta **0** tokens en este corpus: es un knob inerte. La inclusión de `PROPN` **no está documentada** en el repo (es inferencia).                                                        |
| (c) Cascada lemas→UCs→fronteras→estabilidad                     | **Correcta en dirección, y grande**                         | Con el criterio propuesto, el **32.8 %** de las UCEs (tier `min_forms=11`) cambia de UC; en clásico (`14`), el **43.2 %**.                                                                                                                |
| Corrección de datos vs código                                   | **Código**                                                  | `content_pos` sí es configurable por datos, pero el 77 % de los tokens recuperables está en el filtro `is_stop`, que **ningún JSON puede tocar**.                                                                                         |

Además, el `0.400` citado es una **media sobre corpus distintos** (ver §4.6); el valor
homogéneo de este corpus es `0.395`.

---

## 1. Hechos verificados en el código

### 1.1 El criterio vigente

`src/reii/main_workflow.py`, `SegmentadorALCESTE.lematizar_uces`:

```python
2516                elif (
2517                    is_forced_supp
2518                    or is_extra_stop
2519                    or (
2520                        is_alpha
2521                        and (
2522                            is_stop
2523                            or pos
2524                            in {"PRON", "SCONJ", "CCONJ", "DET", "AUX", "ADP", "INTJ"}
2525                        )
2526                    )
2527                ):
2528                    marcadores.append(...)
2531                elif pos in self.content_pos and not is_stop and is_alpha:
```

`tokens_raw` se construye en `main_workflow.py:2416-2419` con
`(i, t.text, t.lemma_.lower(), t.pos_, t.is_stop, t.is_alpha)`: es decir, `is_stop`
es el **flag léxico de spaCy** (`es_core_news_lg`), no una lista cerrada de
palabras herramienta.

Consecuencia operativa: una forma es **activa** si y sólo si
`pos in {NOUN,VERB,ADJ,ADV}  AND  not is_stop  AND  is_alpha`.
El `is_stop` **manda sobre el POS**: un `VERB`/`ADV`/`ADJ`/`NOUN` que spaCy marque
como stopword nunca entra al vocabulario, por muy "de contenido" que sea su categoría.

### 1.2 `not is_stop` en L2531 es un predicado muerto (prueba)

Sea un token con `is_alpha = True` (si `is_alpha` fuera `False`, L2531 tampoco lo
tomaría). Si `is_stop = True`, el `elif` de L2516 se satisface por el tercer
disyunto (`is_alpha and (is_stop or ...)`) y el token va a `marcadores`; **nunca
evalúa L2531**. Por tanto, en L2531 `is_stop` es siempre `False` y el término
`not is_stop` es lógicamente redundante: **borrarlo solo no cambia una sola forma**.

Esto es decisivo: la propuesta literal ("eliminar `not is_stop`") es un **no-op**.
Para que el cambio tenga efecto hay que quitar `is_stop` de la **rama de marcadores**
(L2520-2526).

### 1.3 La rama de marcadores es una lista híbrida

Combina dos criterios distintos:

- **clase cerrada por POS**: `PRON, SCONJ, CCONJ, DET, AUX, ADP, INTJ` → correcto y
  equivalente al criterio de ALCESTE;
- **lista abierta por frecuencia léxica**: `is_stop` de spaCy → **instrumento
  equivocado**, porque mezcla palabras funcionales con ítems léxicos muy frecuentes.

`is_stop` cubre, en `es_core_news_lg`, adverbios como `no`, `más`, `muy`, `ya`,
`también` (ADV) y verbos plenos como `hacer`, `tener`, `dar`, `decir` (VERB), además
de sustantivos léxicos como `uso`, `parte`, `cuenta`, `vez` (NOUN) y adjetivos como
`grande`, `mayor`, `primero` (ADJ). Todo eso cae hoy en `marcadores`.

### 1.4 Cadena de override de `content_pos` (trampa de implementación)

1. `Config.content_pos` (`main_workflow.py:1047-1049`) se declara con
   `["NOUN","VERB","ADJ","ADV"]`.
2. `SegmentadorALCESTE.__init__` (`main_workflow.py:2177-2189`) hace:
   ```python
   gramatica = load_gramatica()
   self.config.content_pos = list(gramatica.get("content_pos", config.content_pos))
   self.content_pos = set(self.config.content_pos)
   ```
3. `load_gramatica()` (`gramatica.py:32-52`) **siempre** devuelve
   `DEFAULT_GRAMATICA` fusionado con el JSON, y `DEFAULT_GRAMATICA`
   (`gramatica.py:21-29`) **siempre** tiene la clave `"content_pos"`.

**Consecuencia:** `Config.content_pos` se pisa **siempre**; editarlo solo es inerte.
La fuente efectiva es `gramatica.py:22` o bien `ia/gramatica.json`.

### 1.5 ¿Es configurable por datos? (sí, pero insuficiente)

- `GRAMATICA_PATH` = `ia/gramatica.json` (`config.py:132-134`).
- Ese fichero **no existe**: `ia/` sólo contiene `localismos.json`,
  `analisis_discursivo/` y `entrevistas/`.
- `ia/analisis_discursivo/resumidor_gramatical.json` **no** es esta configuración:
  es el _prompt + json_schema_ de un agente LLM (`config.py:126-131`,
  consumido en `ia_discursiva.py:1691-1701` vía `GRAMMAR_CONFIG_PATH`). No tiene
  `content_pos` ni nada parecido.
- `.gitignore` no ignora `ia/`, así que un `ia/gramatica.json` sería versionable
  (a diferencia de `data/`, ignorado entero en `.gitignore:141`).

Por tanto la "variante de datos" existe, pero **sólo puede tocar `content_pos`**:
no puede desactivar el `is_stop` de L2522. Ver §5.1 para el cálculo de su efecto.

### 1.6 Clásico vs actual

`src/reii/main_workflow_clasico.py` (`SegmentadorALCESTE` propio):

- `content_pos` **hardcodeado** y sin capa de gramática: L563.
- Orden de ramas invertido respecto del actual: **primero** activa
  (`L728: if pos in self.content_pos and not is_stop and is_alpha`), **después**
  marcadores (`L742-745`).

El conjunto efectivo de formas activas es **idéntico** al de `main_workflow.py`:
la intersección `content_pos ∧ ¬is_stop ∧ is_alpha` es la misma, sólo cambia el
orden de evaluación. **Ninguna de las dos se parece a ALCESTE 2012 en este punto.**
Diferencias reales entre ambos ficheros (no relacionadas con la fidelidad léxica):
el clásico no tiene `expresiones`, `palabras_activas`, `sinonimos` ni
`stopwords_extra` (`main_workflow.py:2432-2494` vs `main_workflow_clasico.py:725-748`).

Además, el fork está retirado: `main_workflow.py:8060-8066` documenta
_"el fork main_workflow_clasico.py queda retirado (shim, §9.5)"_ y
`--mode classic` se implementa con **el mismo** `SegmentadorALCESTE` de
`main_workflow.py` más `CLASSIC_OVERRIDES` (`main_workflow.py:8064-8073`).
**Implicación importante: el parche de §5.2 corrige también el brazo clásico.**
El fichero `main_workflow_clasico.py` quedaría desalineado, pero no es el que corre.

### 1.7 Qué dice la especificación del repo

- `docs/paper.md:31-32`: ALCESTE clasifica el vocabulario en formas **activas**
  (sustantivos, verbos, adjetivos) y **suplementarias** (artículos, preposiciones),
  y _"estas clases son analizadas de forma separada, pero no excluidas del análisis"_.
- `docs/fixes to the algorithm.md:31-36`: activas = _"sustantivos, verbos (excluyendo
  auxiliares y modales), adjetivos, adverbios de modo y **palabras no reconocidas de
  alta frecuencia**"_; suplementarias = _"preposiciones, pronombres, artículos,
  números"_. Añade el umbral **TSJ** y el truncamiento de hapax.
- `docs/design_v5/CLASSIC_MODE.md:56-66`: fija `tsj=3`, `min_uce_words=2`… pero
  **no menciona `content_pos`**. `_CLASSIC_SPEC_FIELDS`
  (`CLASSIC_MODE.md:136-141`, `main_workflow.py:8084-8090`) tampoco lo incluye.

Ningún documento del repo pide filtrar por `is_stop`; el criterio es **categorial +
frecuencia**, no una lista de stopwords de un tagger.

### 1.8 TSJ no se salta al ampliar `content_pos`

`MatrizBuilder.construir_vocabulario` aplica `doc_freq >= tsj` y
`freq >= min_term_abs_freq` **después** de recolectar las formas activas
(`main_workflow.py:3438-3458`). Por tanto ampliar `content_pos` no relaja el umbral:
solo añade _candidatos_ que luego TSJ poda. Esto es coherente con la formulación
"activas **si** superan el TSJ".

---

## 2. Cuantificación (exacta, sin ejecutar el pipeline)

**Método.** Sobre el artefacto ya segmentado `data/diag/../workflow_data.json`
(528 UCEs, 14 333 palabras crudas, corpus `dd14a8c17a7fa593`) re-etiqueté cada UCE
con spaCy (`es_core_news_lg`, sin parser ni NER) y apliqué dos criterios de forma
independiente. No se modificó ningún fichero.

**Control de validez.** El criterio actual reproduce **exactamente** el artefacto:

|                  | mi replay | `quality_multiverse.json` run-ffd93a5f2d5d3659 |
| ---------------- | --------- | ---------------------------------------------- |
| `lemmas_total`   | 5 658     | 5 658                                          |
| `lemmas_per_uce` | 10.7159   | 10.7159                                        |
| `retention`      | 0.395     | 0.3954                                         |

Es decir, la medición no es una estimación: reproduce el número publicado.

**Embudo de tokens del corpus (hecho, sobre el artefacto):** 14 333 palabras crudas
(separadas por espacios) → 13 824 tokens alfa → **5 658 activos (41.0 %)** +
7 877 `marcadores` (57.0 %) + 289 descartes (`PROPN` 278, `NUM` 11). Los 509 tokens
restantes por diferencia son no alfa (puntuación y cifras). De los 8 675 tokens que
hoy no se analizan, el flag `is_stop` es responsable directo de **998** (11.5 % del
crudo) y `PROPN` descartado de 278 más.

### 2.1 Efecto sobre palabras analizadas

| Criterio                                              | Formas activas | Activas/UCE | Retención (raw 26-27) |
| ----------------------------------------------------- | -------------: | ----------: | --------------------: |
| Actual: `pos∈{N,V,A,ADV} ∧ not is_stop`               |          5 658 |       10.72 |             **0.395** |
| Propuesto: POS-only + `PROPN`,`X`                     |          6 949 |       13.16 |             **0.485** |
| ALCESTE (referencia, `7_quality_multiverse.py:61-66`) |              — |       13.95 |                 0.537 |

Delta: **+1 291 tokens activos (+22.8 %)**, +2.44 palabras analizadas por UCE.
Faltarían ~0.8 palabras/UCE (−9.7 %) para igualar a ALCESTE.

### 2.2 De dónde sale cada token recuperado

| Origen                                                       |    Tokens | % del total recuperado |
| ------------------------------------------------------------ | --------: | ---------------------: |
| `is_stop` con POS de contenido — ADV                         |       543 |                 42.1 % |
| `is_stop` con POS de contenido — VERB                        |       252 |                 19.5 % |
| `is_stop` con POS de contenido — NOUN                        |       105 |                  8.1 % |
| `is_stop` con POS de contenido — ADJ                         |        98 |                  7.6 % |
| `PROPN` (278 que hoy se descartan + 15 que hoy son marcador) |       293 |                 22.7 % |
| `X`                                                          |     **0** |                    0 % |
| **Total**                                                    | **1 291** |                  100 % |

Dos hechos que corrigen la hipótesis:

1. **X es irrelevante en este corpus.** spaCy no emite ningún token `X` alfa en las
   528 UCEs. Incluirlo es correcto por especificación ("palabras no reconocidas"),
   pero no mueve la aguja. En cambio los numerales (`NUM`, 11 tokens) quedan fuera,
   y es correcto que lo estén.

2. **El 77 % de la ganancia viene de `is_stop`, no de `PROPN`.** La variante
   "solo datos" (que solo puede tocar `content_pos`) captura como máximo el 23 %.

### 2.3 Qué palabras exactamente entrarían

Top de formas rescatadas por el cambio (selección; lema, ocurrencias):
`no` 243, `más` 102, `hacer` 63, `tener` 52, `mucho` 38, `dar` 35, `ya` 33,
`antes` 27, `ahora` 21, `uso` 18, `también` 14, `solo` 13, `poner` 9, `día` 7,
`primero` 7, `demasiado` 5, `bien` 5.

Esto revela un matiz importante que el diagnóstico no puede ignorar: **parte de lo
rescatado no son formas activas de ALCESTE, sino palabras herramienta**. Del total
de 998 tokens rescatados por quitar `is_stop`, alrededor de **484 (48.5 %)** son
formas de negación/grado/dícticos (`no`, `ni`, `tampoco`, `nunca`, `muy`, `más`,
`tan`, `casi`, `ya`, `también`, `sólo`, `antes`, `ahora`...) que ALCESTE trataría
como _supplémentaires_. El resto (≈514) sí son formas activas indiscutibles:
verbos plenos (`hacer`, `tener`, `dar`, `poner`, `ir`, `existir`, `decir`),
sustantivos (`uso`, `manera`, `veces`, `parte`, `cuenta`, `día`, `estado`) y
adjetivos (`grande`, `mayor`, `bueno`, `primero`, `debido`).

Por eso el documento distingue dos cotas:

| Política                                               | Activos | Analizadas/UCE | Retención |
| ------------------------------------------------------ | ------: | -------------: | --------: |
| sin `is_stop` (patch de este entregable)               |   6 949 |          13.16 | **0.485** |
| sin `is_stop` + lista cerrada de adverbios-herramienta |   6 465 |          12.24 | **0.451** |

Ambas mejoran sustancialmente el 0.395 y quedan por debajo o cerca de ALCESTE. La
segunda es más fiel a la definición documentada ("adverbios **de modo**"; el resto
de adverbios no está listado como activo), pero exige fijar la lista **antes** de
volver a correr, no después de ver el número.

---

## 3. La cascada (hipótesis c): cuantificada

`construir_ucs` (`main_workflow.py:2585-2607`) agrupa UCEs consecutivas hasta reunir
`min_forms` **lemas distintos**, delegando el corte en
`agrupar_secuencial_por_umbral` (`reii/seg/forms_merging.py:34-44`).
Con más lemas por UCE se necesitan **menos** UCEs para
alcanzar el umbral, luego las UCs se **acortan** y sus fronteras se mueven.

Recomputado sobre las 528 UCEs (UCEs que cambian de UC = mismo conjunto de miembros):

| `min_forms` | régimen                         |           UCs |     UCEs/UC | UCEs con UC recompuesta | UCs de 1 UCE |
| ----------- | ------------------------------- | ------------: | ----------: | ----------------------: | -----------: |
| 8           | moderno, tier 1 (`best_params`) | 459 → **497** | 1.15 → 1.06 |              **14.4 %** |    390 → 466 |
| 11          | moderno, tier 2                 | 367 → **428** | 1.44 → 1.23 |              **32.8 %** |    208 → 328 |
| 10          | clásico, tier 1                 | 396 → **452** | 1.33 → 1.17 |              **28.2 %** |    266 → 376 |
| 14          | clásico, tier 2                 | 292 → **349** | 1.81 → 1.51 |              **43.2 %** |     68 → 173 |

Lectura: **no es un ajuste léxico cosmético**. En el tier 2 moderno, una de cada tres
UCEs cambia de contexto de entrenamiento; en clásico, casi la mitad. Eso rehace los
dos pases de la CDH, la matriz de coocurrencia y, por tanto, el emparejamiento
húngaro. La dirección que predice el usuario (UCs más gruesas hoy → fronteras
distintas → más UCEs inestables) es coherente con estos números, pero **el efecto
neto sobre `stable%` no se puede deducir sólo del vocabulario** (ver §6).

---

## 4. Riesgos de invalidación (y corrección de dos supuestos del usuario)

### 4.1 `corpus_fingerprint` **no** cambia — el riesgo es el contrario

`_corpus_fingerprint()` (`main_workflow.py:507-526`) hashea **sólo ficheros de entrada**
(nombre + `mtime` + tamaño de `data/txt_outputs/**`). El criterio de formas activas no
interviene. Por tanto:

- No cambia el `corpus_fingerprint` de `best_params.json` (`"42c630f03946b0cd"`, L130)
  ni el `cache_key` de `uce_calibration.json` (`"42c630f03946b0cd:26.0:14-30"`).
- `study_name = f"{study_name}_{_corpus_fingerprint()}"` (`main_workflow.py:8815-8819`):
  los trials nuevos caen en la **misma** study de Optuna que los 21 antiguos, cuyas
  funciones objetivo se midieron con el vocabulario viejo. No hay aviso, no hay
  separación. **Es peor que invalidar: es contaminación silenciosa del espacio de
  búsqueda y del frente de Pareto.**
- Además `run_id = sha1(corpus_hash|code_version|seed|mode|config)`
  (`7_quality_multiverse.py:13-14`) usa `code_version()` = `git rev-parse --short HEAD`
  (`quality/hooks.py:45-62`). Con el cambio **sin commitear**, `code_version` no
  cambia; y si solo se toca el `is_stop` de L2522, `config` tampoco (no hay campo que
  lo refleje) → **mismo `run_id`** que una corrida previa, con sobrescritura de filas.

**Mitigación obligatoria:** commitear el cambio (o exportar `REII_CODE_VERSION`) y
**archivar/renombrar la study de Optuna y `best_params.json`** antes de re-optimizar.

### 4.2 `uce_calibration.json`: sigue siendo válido, pero se queda sin propósito

`cache_key = f"{fp}:{desired_avg}:{bounds}"` (`main_workflow.py:727-728`) no cambia.
Y la medición es de **palabras crudas** (`medir_promedio_real` usa
`len(span.text.split())`, `main_workflow.py:2303-2314`), que no dependen del criterio
léxico: 26.857 palabras crudas/UCE sigue siendo cierto.

Lo que caduca es la **justificación**: `target=30` se eligió para que 26-27 palabras
crudas dieran 13.95 analizadas. Con el criterio nuevo, ese mismo target ya da 13.16.
No hay que recalibrar (recalibrar movería las fronteras y confundiría dos cambios),
pero hay que **reescribir la nota metodológica** y no "afinar" `target` para clavar
13.95: eso sería ajustar el diseño al resultado.

### 4.3 `spec` / aislamiento de corridas

- En `7_quality_multiverse.py`, `spec_signature` hashea el **dict de config** entero
  menos target/seed (L104-121), y `redact_config` volca `asdict(Config)`
  (`quality/hooks.py:40-42`): como `content_pos` es campo de `Config` y se muta en
  `__init__`, **la parte `content_pos` sí cambia la firma**. Las corridas viejas y
  nuevas se separan en la tabla de script 7.
- Pero **la parte `is_stop` no es visible en ningún sitio**: es una decisión de código
  sin contrapartida en config. Sin commit (o `REII_CODE_VERSION`), dos
  especificaciones distintas comparten firma.
- En `--mode classic`, `run_spec_id` usa una lista **fija** de campos
  (`CLASSIC_MODE.md:136-141`, `main_workflow.py:8084-8090`) que **no incluye
  `content_pos`** → dos corridas clásicas con vocabularios distintos obtendrían el
  **mismo** `spec_id`. Si el paper va a reportar el brazo clásico, esto hay que
  arreglarlo en el mismo movimiento (añadir `content_pos` a `_CLASSIC_SPEC_FIELDS`).

### 4.4 `min_forms_uc` cambia de significado

Los umbrales son **lemas distintos**. Al subir un 22.8 % los lemas distintos por UCE,
el mismo `min_forms` se alcanza con UCEs más cortas: UCs más numerosas y más
singletons (§3). Los valores vigentes —`[8, 11]` desde `best_params.json`
(`min_forms_uc_1=8`, `forms_gap=3`), `[10, 14]` en `CLASSIC_OVERRIDES`
(`main_workflow.py:8064-8073`) y `[13, 17]` por defecto (`Config.min_forms_uc`, L804) —
se optimizaron sobre la distribución vieja. Hay que **re-optimizar, no reinterpretar**.

### 4.5 Invariantes calibrados sobre la distribución vieja

- Gate de dominancia / `max_class_share` (denominador = UCEs asignadas,
  `main_workflow.py:543-549`, `method_selection.py:33-42`).
- Nivel "amplio" `p_cal >= 0.70` (`quality/calibration.py:1-8, 23-30`).
- `tsj=4` (actual) y `min_term_abs_freq=2` (`main_workflow.py:825-829`,
  `MatrizBuilder` L3438-3458): con +22.8 % de tokens, más tipos cruzan el umbral, la
  matriz crece y cambia el truncamiento de hapax.

Todos ellos son funciones de la partición; no se "recalibran solos".

### 4.6 El `0.400` de cabecera mezcla corpus

En `data/diag/quality_multiverse.json` las 11 corridas incluyen **siete** con
`n_uces_total` entre 600 y 5 809 y `corpus_hash` distinto del actual
(`f04715c9f001de12`, `c02f84e1abb9c256`, ...), algunas con `lemmas_per_uce = 4.56`
y retención `0.2744`. La media `0.400` sale de mezclarlas con las **cuatro** corridas
del corpus actual (todas en `0.3954`). Script 7 agrupa por `(spec, mode)` y no por
corpus (docstring L23-26), y avisa de un corpus ajeno **sólo** para la cobertura
(L267-273), no para `lemmas_per_uce` ni para la retención. **Para el paper: la línea
base honesta de este corpus es 10.72 / 0.395, no 9.0 / 0.400.**

### 4.7 Riesgos de coherencia y de contenido

1. **Definiciones paralelas de "contenido" que no se actualizan**:
   `gramatical_analyzer.py:1136`, `:1200`, `:1303-1316`, `:1236-1240` y
   `build_doc_record` `:3079` mantienen `{NOUN,VERB,ADJ,ADV}` hardcodeado. Tras el
   cambio, el clustering usaría el criterio nuevo y **las métricas léxicas
   (mean_zipf, sofisticación, registro, `n_content`) seguirían usando el viejo**.
   Hay que decidir explícitamente si se alinean (y decirlo en el paper).
2. **`marcadores` se reduce** (deja de contener los `is_stop` de contenido):
   afecta a `densidad_discursiva`, `top_marcadores` y `pos_by_cluster`.
3. **Inflación por palabras herramienta** (§2.3): el 37.5 % del +22.8 % son
   `no/más/ya/también`. Reportar 0.485 como "más contenido analítico" sería
   defendible sólo si se documenta que incluye negación/grado. Riesgo de que el
   revisor lo lea como un ajuste al objetivo.
4. **`PROPN` en un corpus de entrevistas**: meter nombres propios en el vocabulario
   activo los lleva a la matriz, a los perfiles de clase y a las tablas publicadas.
   En un corpus con personas identificables esto es un problema de **anonimización**,
   no sólo de fidelidad. Debe decidirse (y probablemente excluirse por lista).

---

## 5. Corrección propuesta: dos variantes y recomendación

### 5.1 Variante A — sólo datos (`ia/gramatica.json` nuevo)

Crear `ia/gramatica.json`:

```json
{ "content_pos": ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "X"] }
```

- Técnicamente funciona: `load_gramatica` fusiona sobre `DEFAULT_GRAMATICA`
  (`gramatica.py:40-48`), así que un JSON parcial es seguro (claves ausentes toman
  el default). `ia/` no está ignorado, luego es versionable.
- Efecto medido: **+293 tokens (+5.2 %)**, 11.27 activas/UCE, retención ≈ **0.415**.
- **No toca el 77 % del problema** (los 998 `is_stop`), porque la rama de marcadores
  se evalúa **antes** y no consulta `content_pos`.
- Trampa: cambiar sólo `Config.content_pos` (L1047-1049) **no hace nada** (§1.4).

Veredicto: **insuficiente como corrección**, pero es el sitio natural para el paso 2
(lista cerrada de adverbios-herramienta vía `stopwords_extra`, que sí desvía a
`marcadores` en `main_workflow.py:2491-2494`).

_Nota:_ existe un tercer camino "sólo datos" —enumerar cada forma `is_stop` de
contenido en `palabras_activas`—. Es un parche invertido e inmantenible (forzar a
activo lo que debería serlo por categoría) y depende del vocabulario del corpus.
**No se recomienda.**

### 5.2 Variante B — código (recomendada, es el `.patch` de este entregable)

Tres hunks en dos ficheros:

1. `src/reii/gramatica.py:22` — `DEFAULT_GRAMATICA["content_pos"]` añade `PROPN`, `X`
   (la fuente **efectiva**, §1.4).
2. `src/reii/main_workflow.py:1048` — mismo `Config.content_pos`, para que la
   declaración del dataclass no contradiga la realidad (es inerte hoy, pero deja de
   serlo si algún día se lee sin pasar por `load_gramatica`).
3. `src/reii/main_workflow.py:2520-2531` — la rama de marcadores pasa a ser
   **estrictamente categorial** (`PRON/SCONJ/CCONJ/DET/AUX/ADP/INTJ`), se elimina
   `is_stop`, y la condición activa queda `pos in content_pos and is_alpha`.

Efecto: 13.16 activas/UCE, retención **0.485** (cota superior; ver §2.3 para la cota
0.451 con lista cerrada).

**Principio de diseño:** el filtro de palabras herramienta debe ser la **clase
cerrada por POS** (`PRON, DET, ADP, CCONJ, SCONJ, AUX, INTJ`), no una lista de
frecuencia de un tagger. Con el filtro POS ya no queda nada que `is_stop` pueda
aportar en la rama de contenido: `is_stop` **sobra**, y mantenerlo "sólo para
categorías funcionales" sería redundante con el POS (las funcionales ya están
excluidas por POS antes de mirar `is_stop`).

Por eso **no** se propone la variante "mantener `not is_stop` para funcionales":
no es una decisión semántica, es una duplicación.

### 5.3 Recomendación

**Aplicar la variante B**, en dos pasos separados y **pre-registrados**:

1. **Paso 1 (este patch):** criterio categorial + `PROPN`/`X`. Es la corrección del
   defecto de instrumento. Re-baselinizar todo (§4).
2. **Paso 2 (decisión de datos, NO incluida en el patch):** `ia/gramatica.json` con
   `stopwords_extra` = lista **cerrada** de adverbios-herramienta (negación, grado,
   dícticos temporales) y, si el equipo lo decide, `palabras_excluir` para nombres
   propios por anonimización. Debe fijarse **antes** de correr y reportarse como
   decisión metodológica, no ajustarse para llegar a 13.95.

Si el equipo no puede re-baselinizar ahora, la alternativa honesta es **no aplicar
nada y publicar la discrepancia como limitación** (0.395 vs 0.537, con la explicación
de §1.2-1.3). Lo que no es aceptable es aplicar el cambio y seguir mezclando corridas.

---

## 6. Qué NO se ha verificado

Hechos frente a inferencias:

**Verificado (hecho, reproducible):**

- El criterio efectivo y la redundancia de `not is_stop` (§1.2), por lectura y prueba
  lógica.
- La cadena de override y la ausencia de `ia/gramatica.json` (§1.4-1.5).
- El conjunto de formas activas actual (5 658) y propuesto (6 949) sobre las 528 UCEs,
  con reproducción exacta del artefacto (§2).
- La recomposición de UCs (§3).
- Que `_corpus_fingerprint` no depende del léxico (§4.1).

**Inferencia (no verificada aquí):**

- Que ALCESTE 2012 trate `PROPN` como forma activa. El repo **no** lo documenta:
  `docs/paper.md:31-32` dice "sustantivos, verbos, adjetivos" y
  `docs/fixes to the algorithm.md:31-36` añade "adverbios de modo" y "palabras no
  reconocidas". `PROPN` no aparece. Es una decisión del equipo, no un dato del repo.
- Que el `13.95` sea comparable: es una constante del script
  (`7_quality_multiverse.py:61-66`) sin procedencia en el repo (no hay informe de
  ALCESTE, ni nº de UCEs que lo respalde más allá de `n_uce=376`).
- Que el efecto neto sobre `stable%` sea positivo. **No se ha medido**: requiere CDH +
  emparejamiento húngaro.

**No ejecutado (explícitamente fuera de alcance):**

- Pipeline, pytest, contenedores, Stanza.
- Re-segmentación: el criterio léxico **no** interviene en
  `_segmentar_reinert_spacy`, por lo que las 528 UCEs y sus textos no cambian; los
  offsets y `corpus_hash` de `quality` (`service.py:351-358`, hash de `uce_id|texto`)
  tampoco. Esto es lectura de código, no ejecución.
- Las corridas de 5 809 UCEs (otro corpus/segmentación) no se han replicado.
- No se ha comprobado si `reii/seg/segmentador.py:203-206` y `:592-596` (que también
  hardcodean `content_pos`) están en alguna ruta viva: por lectura, `main_workflow`
  usa su propio `_segmentar_reinert_spacy`. **Sin verificar en runtime.**

### Plan de medición para cerrar lo que falta (cuando se autorice ejecutar)

1. `REII_CODE_VERSION=activas-v1 git ...` (o commit) → evita colisión de `run_id`.
2. Archivar `data/best_params.json`, `data/reii_optuna.db` y
   `data/diag/quality_multiverse.json` actuales.
3. Correr el corpus **con los mismos parámetros fijos** (sin optimizar) antes/después:
   comparar `lemmas_total`, `stable_pct`, y el ARI entre los dos pases.
4. Correr `src/7_quality_multiverse.py` y verificar que el `spec` cambia (si no cambia,
   el aislamiento falló).
5. Sólo después, re-optimizar `min_forms_uc`/`tsj` y recalibrar el gate 0.70 sobre la
   distribución nueva.
6. Reportar la retención con las dos cotas de §2.3 (con y sin lista cerrada).

---

## 7. Resumen del parche

`tools/archive/patchset_v5_v6/patches/11_formas_activas.patch` — 3 hunks, 2 ficheros, LF.

| Fichero                     |     Línea | Cambio                                             |
| --------------------------- | --------: | -------------------------------------------------- |
| `src/reii/gramatica.py`     |        22 | `content_pos` + `PROPN`, `X` (fuente efectiva)     |
| `src/reii/main_workflow.py` |      1048 | ídem en `Config` (coherencia declarativa)          |
| `src/reii/main_workflow.py` | 2520-2531 | marcadores sólo por POS; activa = `pos ∧ is_alpha` |

Verificación: `git apply --check tools/archive/patchset_v5_v6/patches/11_formas_activas.patch` → **OK** sobre el
árbol actual. No modifica los 9 patches existentes ni toca `data/`.
