# Política de retención de formas activas (patch `13_retention_policy`)

> **Veredicto (una línea):** se adopta `tool_adverb_filtered` (conservadora,
> **12.24** lemas/UCE, retención **0.451**) y se **excluye** `pos_plus_propn`
> (agresiva, **13.16** / **0.485**) por incumplir la restricción dura de
> palabras herramienta (cuota **0.375** > **0.30**). La desviación frente a
> ALCESTE 2012 (**13.95** / **0.537**) se declara cuantificada en §4.2; no se
> corrige ajustando la métrica.

Documento asociado al patch `tools/archive/patchset_v5_v6/patches/13_retention_policy.patch`, que se
aplica **después** de `11_formas_activas.patch` y cuyo **default no cambia la
retención actual** (`pos_only`).

> **Enmienda 2026-10-28:** el default adoptado ya **no** es `pos_only`, sino
> `tool_adverb_filtered` (patch
> `14_retention_default_tool_adverb_filtered`). El texto de arriba y el de §5.1
> se conservan como histórico; ver §10.

---

## 1. El problema

`SegmentadorALCESTE.lematizar_uces` (`src/reii/main_workflow.py`) selecciona
formas activas con una cascada que, en su núcleo, es:

```python
elif pos in self.content_pos and not is_stop and is_alpha:
    # -> forma activa
```

ALCESTE 2012 no usa lista de stopwords ni excluye _nombres propios_: trata las
palabras funcionales como **supplémentaires** (se conservan y se usan en la
descripción del mundo, pero no como formas analizadas del vocabulario). La
consecuencia es una retención estructuralmente menor que la referencia.

### 1.1 Estado medido del corpus

`data/workflow_data.json`: **528 UCEs**, `uce_target_size=30`,
**14 333** palabras crudas (`texto.split()`), **13 824** tokens alfa
(`uce.tokens`, excluye PUNCT y `palabras_excluir`).

| variante                                | lemas/UCE | retención | nota                             |
| --------------------------------------- | --------- | --------- | -------------------------------- |
| `pos_only` (baseline del repo)          | **10.72** | 0.395     | filtro estricto                  |
| `pos_plus_propn` ("agresiva", patch 11) | **13.16** | 0.485     | +PROPN/X y sin `is_stop`         |
| `tool_adverb_filtered` ("conservadora") | **12.24** | 0.451     | agresiva − adverbios-herramienta |
| ALCESTE 2012 (referencia)               | **13.95** | **0.537** | "palabras analizadas/UCE"        |

La tentación incorrecta es elegir la agresiva porque 13.16 se acerca a 13.95.
Eso es **ajustar la métrica**, no fidelidad: el 48.5 % de lo que rescata la
agresiva son palabras herramienta (`no`, `más`, `ya`, …), que ALCESTE no
cuenta como formas analizadas. Este documento fija la regla **antes** de mirar
la tabla de §3 y la aplica sin retoques.

---

## 2. Regla pre-registrada

> **Estado de pre-registro.** El contenido de §2 (definiciones, umbral, criterio
> y sensibilidades) se fijó **antes** de la primera ejecución del evaluador
> `scripts/retention_policy_report.py`. La tabla de §3 es salida literal de ese
> script, que implementa esta regla carácter a carácter (`apply_rule`). Cambiar
> el umbral _a posteriori_ para que gane otra política invalida el pre-registro
> y debe hacerse en un PR aparte, declarándolo como enmienda.

### 2.0 Quién manda: precedencia del knob

El footgun que **no** se repite: en el código actual `Config.content_pos` es
**inerte**, porque `Workflow.__init__` hace

```python
self.config.content_pos = list(gramatica.get("content_pos", config.content_pos))
```

es decir, `ia/gramatica.json` pisa siempre el valor de `Config`. El knob nuevo
tiene precedencia explícita y documentada:

```
Config.retention_policy  (si no es None)   >   ia/gramatica.json   >   "pos_only"
```

- `resolve_retention_policy(explicit, gramatica)` implementa exactamente eso.
- `policy_content_pos(policy, base)` es la **única** fuente de verdad de los
  POS activos: `PROPN`/`X` **no** se declaran en `content_pos` (si estuvieran,
  habría dos sitios contradictorios). `content_pos` queda como **núcleo**
  categorial (`NOUN, VERB, ADJ, ADV`).
- Un valor desconocido lanza `ValueError`; **no** se degrada en silencio.

### 2.1 Definición operativa de "palabra herramienta" y denominadores

"Palabra herramienta" = **adverbio de una lista cerrada pre-registrada**
(negación, grado escalar, tiempo/dícticos, foco/adición/comparación), evaluado
como `pos == "ADV"` **y** `fold(lemma) in TOOL_ADVERB_LEMMAS` (`fold` =
minúsculas + sin diacríticos, porque el corpus está normalizado sin tildes).

La condición `pos == "ADV"` **no es cosmética**: el mismo lema con otro POS es
contenido legítimo (`solo` adjetivo = _alone_; `mucho` cuantificador nominal).
La lista (`reii/retention_policy.py::TOOL_ADVERB_LEMMAS`):

```text
no ni nada nadie nunca jamas tampoco
muy mas menos tan tanto demasiado bastante poco casi apenas
ya todavia aun antes despues ahora hoy ayer manana siempre luego
entonces mientras temprano tarde
solo solamente tambien incluso ademas mismo igual siquiera
```

Conteo que reproduce **exactamente** los 484 rescates marcados (sólo ADV):

```text
no 243 · mas 102 · ya 33 · antes 27 · ahora 21 · solo 14 · tanto 10 ·
demasiado 5 · siempre 4 · hoy 3 · poco 3 · solamente 3 · menos 3 · luego 3 ·
bastante 3 · aun 2 · casi 1 · apenas 1 · nunca 1 · igual 1 · tambien 1
```

**Cuota de herramienta** = `herramienta / rescatadas`, donde _rescatadas_ son
las formas que la política añade respecto al baseline `pos_only`. Se reportan
**dos denominadores**, porque el enunciado y el código no coinciden (§6):

- **(a)** `is_stop + PROPN/X` rescatados → denominador **1291**;
- **(b)** sólo `is_stop` rescatados → denominador **998**.

La restricción se evalúa con **(a)**, que es el denominador **más favorable a
la agresiva** (0.375 < 0.485): incluso con el más benigno, incumple. Con (b)
incumple con más margen (0.485).

### 2.2 Restricción dura

```text
cuota de palabras herramienta sobre lo rescatado  <=  0.30
```

Justificación del número:

- la agresiva mide **0.375** con (a) y **0.485** con (b) → incumple con ambos;
- ALCESTE trata estas palabras como _supplémentaires_, no como formas
  analizadas: la cuota fiel sería ~0, no 0.30;
- **0.30 ya es tolerante** (casi un tercio de lo rescatado podría ser
  herramienta) y deja margen para adverbios lexicalizados que la lista cerrada
  no capture (p. ej. `realmente`, `absolutamente`), que sí son contenido.

### 2.3 Criterio de selección

Entre las políticas **admitidas**:

```text
argmin  | retención - 0.537 |
```

`0.537` es la retención de ALCESTE 2012 sobre este corpus. El empate (a dos
decimales) se resuelve por menor cuota de herramienta y, si persiste, por menor
número de formas rescatadas (política más conservadora).

### 2.4 Regla de no-ajuste

Si **ninguna** política cumple la restricción, **no se toca el número**: se
declara la desviación cuantificada en el paper, se documenta aquí y se queda
la política admitida con menor `|retención − 0.537|`. Si no hay ninguna
admitida, se queda el **baseline** (`pos_only`). En ningún caso se sube el
umbral para "hacer pasar" a la agresiva.

### 2.5 Sensibilidad pre-registrada

Se reporta el veredicto con umbrales **{0.15, 0.25, 0.30, 0.35}**. Se
considera robusto si el veredicto no cambia en ese rango. (Para información: la
agresiva sólo ganaría con un umbral **≥ 0.375**, o **≥ 0.485** con el
denominador (b).)

### 2.6 Aplicación

El umbral, el criterio y la lista de sensibilidades están codificados en
`apply_rule()` del evaluador; el veredicto de §4 es su salida, no una lectura
manual de la tabla.

---

## 3. Medición offline (modo residuo)

### 3.1 Cómo se reconstruye sin Stanza

`data/workflow_data.json` **no persiste** PROPN/X en ningún sitio:

- `tokens`: formas crudas normalizadas (sin POS);
- `pos_tags`: lista **filtrada** de POS activos (5658 = `lem_total`);
- `formas_tokens`: formas activas `{forma, lemma, stem, pos, pos_idx}`;
- `marcadores`: palabras función `{forma, lemma, pos}` (7877).

Por tanto la agresiva se reconstruye por **residuo de multiconjunto**:

1. activas del baseline ← `formas_tokens`;
2. rescatadas al quitar `is_stop` ← `marcadores` con POS de contenido (998);
3. rescatadas por PROPN/X ← `marcadores` con POS `PROPN`/`X` (15) **más** el
   residuo `tokens − (formas ∪ marcadores)` (**289** = 278 PROPN + 11 NUM,
   confirmado por replay offline del etiquetador sobre el mismo corpus);
4. los numerales (`mil`, `veinte`, `trece`, `media`, `quince`, `ochenta`,
   `cien`) se excluyen del rescate: `NUM` no es contenido ni nombre propio;
5. `lemma ≈ forma` y `stem` con Snowball para los candidatos del residuo.

Una forma se cuenta **una vez por UCE** (formas distintas), que es la métrica
de "formas analizadas/UCE" que reporta ALCESTE.

### 3.2 Validación de la reconstrucción (condición para creerla)

El script compara los totales con las cifras ya publicadas y, si no coinciden,
imprime `FALLA` en vez de la tabla. Salida:

```text
VALIDACIÓN de la reconstrucción: VALIDADA | pos_only=5658, pos_plus_propn=6949, tool_adverb_filtered=6465
```

- `pos_only` = 5658 (lemas activos totales; coincide con `lem_total`);
- `pos_plus_propn` = 6949 → 6949 / 528 = **13.16** lemas/UCE;
- `tool_adverb_filtered` = 6465 → **12.24** lemas/UCE.

**La reconstrucción reproduce 13.16 y 12.24** (y 10.72). No hay maquillaje: la
conservadora exige además el filtro `pos == "ADV"` (§2.1); con la lista aplicada
sólo por lema saldrían 517 rescates → 12.17, que **no** reproduce la cifra
medida. Esa es una discrepancia real y se declara en §6.

### 3.3 Tabla comparativa

Bootstrap de 1000 remuestreos por UCE, `seed=42`, percentiles 2.5/97.5:

| política                              | lemas/UCE | IC95 %         | retención | IC95 %         | formas/UCE | rescatadas | herramienta | cuota     |
| ------------------------------------- | --------- | -------------- | --------- | -------------- | ---------- | ---------- | ----------- | --------- |
| `pos_only` (baseline)                 | **10.72** | [10.46, 11.00] | **0.395** | [0.388, 0.402] | 10.58      | 0          | 0           | 0.000     |
| `pos_plus_propn` (agresiva)           | **13.16** | [12.86, 13.46] | **0.485** | [0.479, 0.491] | 12.87      | 1291       | 484         | **0.375** |
| `tool_adverb_filtered` (conservadora) | **12.24** | [11.96, 12.53] | **0.451** | [0.445, 0.457] | 12.08      | 807        | 0           | **0.000** |
| ALCESTE 2012 (referencia)             | **13.95** | –              | **0.537** | –              | –          | –          | –           | –         |

Cuota con denominador (b): agresiva `484/998 = 0.485`; conservadora
`0/514 = 0.000`.

Lectura: los IC del baseline y de la conservadora **no se solapan** (10.72 vs
12.24), así que la diferencia es real, no ruido de muestreo. El IC de la
conservadora tampoco alcanza 13.95 (máximo 12.53): la desviación es
sistemática, no un artefacto del bootstrap.

### 3.4 Artefacto que refuerza el rechazo de la agresiva

Del residuo PROPN (278 tokens), **166 (59.7 %)** son palabras terminadas en
`-ción`/`-sión` (**51 formas distintas**: `informacion`, `educacion`,
`poblacion`, …). No son nombres propios: el corpus está normalizado **sin
tildes** y el etiquetador las marca `PROPN` al no reconocerlas como sustantivos
comunes. Es decir, buena parte del "rescate" de la agresiva no sólo son
palabras herramienta: también es **ruido de etiquetado por normalización**. La
conservadora elimina ambas fuentes de contaminación a la vez que sube la
retención de 0.395 a 0.451.

---

## 4. Veredicto

### 4.1 Aplicación literal de la regla

```text
  pos_only                             cuota=0.000  ADMITIDA   |ret-ALCESTE|=0.142
  pos_plus_propn                       cuota=0.375  EXCLUIDA   |ret-ALCESTE|=0.052
  tool_adverb_filtered                 cuota=0.000  ADMITIDA   |ret-ALCESTE|=0.086

  VEREDICTO: tool_adverb_filtered (conservadora)
```

La agresiva tiene la desviación más pequeña (0.052) y aun así **queda fuera**:
la restricción dura se evalúa primero y no es negociable. Ese es precisamente el
punto del pre-registro.

### 4.2 Desviación declarada (para el paper)

| métrica   | conservadora | ALCESTE 2012 | desviación absoluta | relativa    |
| --------- | ------------ | ------------ | ------------------- | ----------- |
| lemas/UCE | 12.24        | 13.95        | **−1.71**           | **−12.3 %** |
| retención | 0.451        | 0.537        | **−0.086**          | **−16.0 %** |

Redacción sugerida: _"La política seleccionada por la regla pre-registrada
alcanza 12.24 formas analizadas por UCE (retención 0.451), un 12.3 % por debajo
de las 13.95 de ALCESTE 2012. La diferencia se atribuye a (i) las palabras
funcionales que ALCESTE trata como suplementarias pero que aquí no se rescatan,
y (ii) el filtro de adverbios-herramienta exigido por la restricción de
fidelidad; no se compensa inflando el vocabulario con palabras herramienta."_

### 4.3 Sensibilidad (pre-registrada)

```text
    cuota <= 0.15 -> tool_adverb_filtered (conservadora)
    cuota <= 0.25 -> tool_adverb_filtered (conservadora)
    cuota <= 0.30 -> tool_adverb_filtered (conservadora)
    cuota <= 0.35 -> tool_adverb_filtered (conservadora)
```

Veredicto **robusto** en todo el rango. Sólo un umbral ≥ 0.375 (≥ 0.485 con el
denominador (b)) elegiría la agresiva, y ese umbral no está justificado por
ALCESTE.

### 4.4 Por qué no se elige la agresiva pese a estar más cerca

- 48.5 % de lo rescatado (denominador b) son palabras herramienta: `no`,
  `más`, `ya` como "formas analizadas" no es ALCESTE, es inflar el vocabulario.
- 59.7 % del rescate PROPN es un artefacto de normalización sin tildes.
- El objetivo del paper es **fidelidad metodológica**, no igualar un escalar.

---

## 5. Qué hace el patch `13_retention_policy`

| pieza                                | contenido                                                                                                                                                                                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/reii/retention_policy.py`       | knob (`RETENTION_POLICIES`, `resolve_retention_policy`, `validate_retention_policy`), decisión por token (`token_decision`), lista cerrada (`TOOL_ADVERB_LEMMAS`), tabla de tokens (`token_rows`, `TokenTableWriter`, `read_token_table`) |
| `src/reii/gramatica.py`              | default `retention_policy="pos_only"`; `content_pos` pasa a **núcleo** (`NOUN, VERB, ADJ, ADV`); admite `tool_adverb_lemmas`                                                                                                              |
| `ia/gramatica.json`                  | **nuevo** (no existía): `retention_policy`, `content_pos`, `tool_adverb_lemmas`                                                                                                                                                           |
| `src/reii/main_workflow.py`          | `Config.retention_policy=None`, `token_table_path`, `use_token_table_cache`; resolución con precedencia; `lematizar_uces` usa `token_decision` y **escribe** la tabla de tokens                                                           |
| `scripts/retention_policy_report.py` | evaluador offline (residuo y modo exacto), bootstrap, regla y sensibilidad                                                                                                                                                                |
| `tests/test_retention_policy.py`     | 5 tests; **no** importa `main_workflow`                                                                                                                                                                                                   |

### 5.1 Default: sin cambios de retención

`pos_only` es el default efectivo. `11_formas_activas` **solo**, deja la
retención en 13.16; `11 + 13` la devuelve a **10.72**. Esto es deliberado:
`best_params.json` se tuneó con el baseline y cambiarlo en silencio lo
invalidaría. La agresiva y la conservadora quedan **opt-in**:

```python
Config(retention_policy="tool_adverb_filtered")   # o
Config(retention_policy="pos_plus_propn")
```

### 5.2 Caché de tokens (la clave de la automatización)

Hoy no existe y por eso §3 es aproximado. `TokenTableWriter` persiste un JSONL
(LF, UTF-8) con **un registro por UCE**, y dentro de cada registro **un objeto
por token** (todos los tokens, sin filtrar):

```json
{"type":"header","version":1,"corpus_fingerprint":"..."}
{"type":"uce","uce_id":"...","words":35,"tokens":[
  {"forma":"...","lemma":"...","pos":"...","is_stop":false,"is_alpha":true}, ...]}
```

- Incluye **todos** los tokens (PUNCT y excluidos incluidos): es la materia
  prima para reevaluar cualquier política futura.
- **Invalidación** por `corpus_fingerprint` en la cabecera: si no coincide, el
  fichero se reescribe (misma semántica que `best_params.json`).
- `add()` es **idempotente** por `uce_id` (reanudable tras un corte).
- Un fallo de I/O **desactiva** la caché, nunca tumba el pipeline.
- `read_token_table()` devuelve `None` (no `{}`) si la caché no es válida.

Con la tabla, `python scripts/retention_policy_report.py --token-table
data/token_table.jsonl` mide en modo **exacto**, sin aproximación por residuo y
sin volver a pagar los ~13 s/UCE.

### 5.3 Test (dentro del patch)

`tests/test_retention_policy.py` verifica:

1. el knob mapea a las tres políticas (POS activos y `is_stop` correctos);
2. el default no cambia el resultado actual (`resolve_retention_policy(None,
None) == "pos_only"`, y `policy_content_pos("pos_only")` no incluye PROPN);
3. la lista de herramienta **sólo** se aplica a `ADV`;
4. la tabla es **completa** (`len(rows) == len(doc)`, incluidos PUNCT);
5. la caché **invalida** por fingerprint y es reanudable.

Salida literal (árbol scratch `b3`):

```text
5 passed in 0.68s
```

---

## 6. Discrepancias encontradas (enunciado vs. código/datos)

1. **Denominador de la retención.** El enunciado define retención como
   `lemas/UCE ÷ palabras crudas/UCE`. Con `sum(tokens)=13824` salen
   0.409 / 0.503 / 0.468, que **no** reproducen 0.395 / 0.485 / 0.451. El
   denominador que reproduce las cifras publicadas es
   `sum(len(texto.split())) = 14333`. Se usa 14333 y se documenta.
2. **`pos_tags` no es una lista de POS por token.** Es la lista **filtrada** de
   POS activos (5658 = `lem_total`); sólo contiene NOUN/VERB/ADJ/ADV
   (3134/1660/813/51). No sirve para reconstruir el residuo.
3. **PROPN/X no se persisten.** Ni en `formas_tokens`, ni en `marcadores`, ni en
   `pos_tags`. La reconstrucción por residuo es la única vía offline, y por eso
   se valida contra 13.16/12.24 antes de usarla.
4. **La lista de herramienta necesita POS.** Con la lista aplicada sólo por
   lema se obtienen 517 rescates → 12.17, no 12.24. Para reproducir 484 hacen
   falta `pos == "ADV"`. El patch implementa la condición POS.
5. **`ia/gramatica.json` no existía.** El enunciado lo da por hecho; el patch lo
   **crea**. El default del knob está también en `gramatica.py`, para que la
   ausencia del JSON no cambie nada.
6. **`Config.content_pos` es inerte** (lo pisa `load_gramatica()`). El knob nuevo
   no repite el patrón: precedencia explícita y `content_pos` degradado a
   "núcleo" (§2.0).
7. **Artefacto PROPN `-ción`/`-sión`.** 166/278 del residuo PROPN (59.7 %) son
   palabras sin tilde mal etiquetadas. No es un hecho del enunciado y cambia la
   interpretación del "rescate".
8. **El etiquetado léxico es spaCy (`es_core_news_lg`), no Stanza.** Stanza
   interviene en la segmentación (de ahí los ~13 s/UCE); los POS/lemas de la
   tabla de tokens salen de `self.nlp(uce.texto)`.

---

## 7. Pendiente

- **Medición exacta con tabla de tokens**: requiere **1 pasada** del pipeline
  (~13 s/UCE × 528 ≈ **1.9 h**). No se ha ejecutado (prohibido correr el
  pipeline en este encargo). Es la única forma de eliminar la aproximación
  `lemma ≈ forma` del residuo y de contar PUNCT/excluidos con POS real.
- **Decisión de anonimización de PROPN**: si en el futuro se rescatan nombres
  propios, hay que decidir si se enmascaran antes del vocabulario (ALCESTE los
  conserva; el paper puede requerirlo por privacidad).
- **Re-baselinizar `best_params.json`** si se adopta la conservadora como
  default: hoy el default se mantiene en `pos_only` a propósito (§5.1). Adoptar
  `tool_adverb_filtered` por defecto exige re-tunear y volver a validar.
- **Enmienda del pre-registro** si el paper decide otro umbral: debe hacerse en
  PR aparte y declararse como enmienda, nunca editando §2 en silencio.

---

## 8. Reproducción

```sh
# 1. cadena de 16 patches en árbol scratch, luego el 13
rm -rf /c/Users/Julia/AppData/Local/Temp/b2 && mkdir -p /c/Users/Julia/AppData/Local/Temp/b2
cp -r src tests scripts pyproject.toml /c/Users/Julia/AppData/Local/Temp/b2/
cd /c/Users/Julia/AppData/Local/Temp/b2
for p in 01_quality_metrics 02_method_selection 02b_tests_method_selection \
         03_main_workflow_gate_select 03b_backends_c 05_selection_audit \
         06_classic_mode 07_v5_space 06b_stability 02c_tests_select_v4 \
         02d_tests_backend_hardening 02e_tests_stability_null \
         02f_tests_kach_objective 09_fix44_classic_guard \
         10_hardening_ward_max_n 11_formas_activas; do
  git apply -p1 /c/Users/Julia/Downloads/reii/tools/archive/patchset_v5_v6/patches/$p.patch || echo "FAIL $p"
done

# 2. aplicar el 13 y testear (PITFALL: -o pythonpath= no funciona)
cp -r /c/Users/Julia/AppData/Local/Temp/b2 /c/Users/Julia/AppData/Local/Temp/b3
cd /c/Users/Julia/AppData/Local/Temp/b3
git apply --check -p1 /c/Users/Julia/Downloads/reii/tools/archive/patchset_v5_v6/patches/13_retention_policy.patch
git apply      -p1 /c/Users/Julia/Downloads/reii/tools/archive/patchset_v5_v6/patches/13_retention_policy.patch
PYTHONPATH=src python -m pytest -q tests/test_retention_policy.py

# 3. medir (modo residuo; determinista, sin red)
PYTHONPATH=src python scripts/retention_policy_report.py \
  --input /c/Users/Julia/Downloads/reii/data/workflow_data.json

# 4. regresión
PYTHONPATH=src python -m pytest -q tests/test_select_v4.py tests/test_stability_null.py

# 5. medición EXACTA (requiere 1 pasada previa del pipeline, ~1.9 h)
PYTHONPATH=src python scripts/retention_policy_report.py \
  --input data/workflow_data.json --token-table data/token_table.jsonl \
  --bootstrap 1000 --seed 42
```

---

## 9. Anexo: cifras usadas

| magnitud                                  | valor                         |
| ----------------------------------------- | ----------------------------- |
| UCEs                                      | 528                           |
| palabras crudas (`texto.split`)           | 14 333                        |
| tokens alfa (`uce.tokens`)                | 13 824                        |
| formas activas baseline (`formas_tokens`) | 5 658                         |
| marcadores                                | 7 877                         |
| residuo `tokens − (formas ∪ marcadores)`  | 289 (278 PROPN + 11 NUM)      |
| rescate `is_stop` con POS de contenido    | 998                           |
| PROPN/X en `marcadores`                   | 15                            |
| herramienta rescatada (agresiva)          | 484                           |
| PROPN `-ción/-sión` en el residuo         | 166 / 278 (59.7 %), 51 formas |

---

## 10. Enmienda 2026-10-28: default adoptado

**Decisión.** El default efectivo de `retention_policy` pasa de `pos_only` a
**`tool_adverb_filtered`**. No es una política nueva: es aplicar el veredicto ya
registrado en §4.1 (única candidata no trivial admitida por la restricción dura
y, entre las admitidas, la de menor `|retención − 0.537|`). `pos_only` y
`pos_plus_propn` siguen **disponibles** como elección explícita vía
`Config(retention_policy=...)` o `ia/gramatica.json`.

Esta enmienda **supersede** §5.1, el punto 2 de §5.3 y el bullet de §7 sobre
re-baselinizar `best_params.json`. **No** toca §2: la restricción de cuota de
palabras herramienta sigue en **0.30**, la regla de no-ajuste sigue vigente y
los valores de referencia de ALCESTE siguen siendo **13.95** lemas/UCE y
**0.537** de retención.

### 10.1 Texto anterior (histórico, §5.1 literal)

Se conserva tal cual, como registro de lo pre-registrado antes de esta
enmienda:

````markdown
### 5.1 Default: sin cambios de retención

`pos_only` es el default efectivo. `11_formas_activas` **solo**, deja la
retención en 13.16; `11 + 13` la devuelve a **10.72**. Esto es deliberado:
`best_params.json` se tuneó con el baseline y cambiarlo en silencio lo
invalidaría. La agresiva y la conservadora quedan **opt-in**:

```python
Config(retention_policy="tool_adverb_filtered")   # o
Config(retention_policy="pos_plus_propn")
```
````

### 10.2 Nuevo estado

| punto de definición                               | antes      | ahora                  |
| ------------------------------------------------- | ---------- | ---------------------- |
| `retention_policy.DEFAULT_RETENTION_POLICY`       | `pos_only` | `tool_adverb_filtered` |
| `gramatica.DEFAULT_GRAMATICA["retention_policy"]` | `pos_only` | `tool_adverb_filtered` |
| `ia/gramatica.json` → `retention_policy`          | `pos_only` | `tool_adverb_filtered` |
| `Config.retention_policy`                         | `None`     | `None` (sin cambio)    |

Precedencia resultante (sin cambios en su semántica):

```text
Config.retention_policy (si no es None)  >  ia/gramatica.json  >  "tool_adverb_filtered"
```

El diagrama equivalente de §2.0 se conserva **literal**: §2 es texto
pre-registrado y no se edita en silencio (ver §7). Su tercer término
(`"pos_only"`) queda sustituido por el de arriba; esta sección es la enmienda
declarada.

Vocabulario: pasan a activas **6465** formas frente a las **5658** del baseline
(+807, las rescatadas de §3.3). Los tests de `tests/test_retention_policy.py`
fijan las tres vías (default, JSON y `Config`) y comprueban que
`DEFAULT_RETENTION_POLICY`, `DEFAULT_GRAMATICA` y `ia/gramatica.json` coinciden.

### 10.3 Consecuencia operativa: invalidación de `best_params.json`

`min_forms_uc_1` y `forms_gap` cuentan **formas distintas por UC**, así que
cambiar el vocabulario de formas activas mueve las fronteras de UC y cambia el
significado del espacio de parámetros. Los artefactos tuneados con el baseline
quedan **inválidos** y el procedimiento es:

1. archivar `data/reii_optuna.db`, `data/best_params.json`,
   `data/quality_multiverse.json` y `data/method_selection.*`;
2. re-optimizar con `--study-name reii_probe_v5`. El `corpus_fingerprint` **no**
   cambia con el vocabulario, de modo que reutilizar `reii_probe_v4`
   contaminaría el estudio con trials medidos bajo otro vocabulario.

**No** se invalidan `data/uce_calibration.json` ni `data/coref_chains_cache.json`:
la segmentación en UCEs no depende del léxico y la clave del cache es
`fingerprint:target:min_uce_words`.

Esta enmienda **declara** el procedimiento; **no** lo ejecuta. El archivado y la
re-optimización son pasos manuales posteriores (y quedan fuera de este patch).

### 10.4 Desviación declarada frente a ALCESTE

Con `tool_adverb_filtered` el corpus queda en **12.24** lemas/UCE (retención
**0.451**), frente a **13.95** / **0.537** de ALCESTE 2012:

| magnitud  | `tool_adverb_filtered` | ALCESTE 2012 | diferencia | relativa    |
| --------- | ---------------------- | ------------ | ---------- | ----------- |
| lemas/UCE | 12.24                  | 13.95        | **−1.71**  | **−12.3 %** |
| retención | 0.451                  | 0.537        | **−0.086** | **−16.0 %** |

La desviación se declara cuantificada (como ya hacía §4.2); no se corrige
ajustando la métrica, el umbral (0.30) ni los valores de referencia.

### 10.5 Enmienda 2026-10-30: segunda causa de invalidación de `best_params.json`

§10.3 invalidó `best_params.json` por el vocabulario. Hay una segunda causa,
independiente: el tercer objetivo del optimizador pasa de `-K` a `-K_util`. El
detalle está pre-registrado en `docs/design_v5/AHORRO_K_Y_PERSISTENCIA.md`.

Este documento no reabre §2 ni cambia la política de retención: solo anota que
el archivado de §10.3 se repite, con estudio nuevo, porque los `best_values`
anteriores ya no son comparables en su tercer slot.
