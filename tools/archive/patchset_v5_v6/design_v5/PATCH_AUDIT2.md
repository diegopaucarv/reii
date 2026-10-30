# PATCH_AUDIT2 — auditoría hunk a hunk de 4 patches v5

Fecha: 2026-10-28. Alcance: `02c`, `09`, `10`, `11`.
Base de contraste: árbol de trabajo (árbol limpio), `.pcfull/` (todo aplicado),
`.pcbase/` (hasta `06b`) para `09`. No se modificó nada fuera de este informe.

## 1. `02c_tests_select_v4.patch` → `tests/test_select_v4.py`

| hunk                   | veredicto | razón                                                                                                                                                                         |
| ---------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `@@ -495,7 +495,7 @@`  | OK        | Contexto exacto (495-501); `null_mode` default del módulo es `"diagnostic"` (`.pcfull/src/reii/method_selection.py:550`), así que el assert viejo fallaba.                    |
| `@@ -525,7 +525,10 @@` | OK        | `test_gate_abstiene` pasa `null_mode="gate"` explícito; conserva el propósito (abstención por `null_p=0.610`). +3 líneas, offsets coherentes con el hunk 3 (`-630` → `+633`). |
| `@@ -630,7 +633,7 @@`  | OK        | Roundtrip: `to_dict()` persiste el modo efectivo, que con el default nuevo es `"diagnostic"`.                                                                                 |

Residuales buscados: 0 asserts que asuman el default `"gate"` implícito.
`tests/test_stability_null.py:244` ya pasa `null_mode="gate"` explícito (no afectado).
Producción no usa el default del módulo: `main_workflow.py:5893-5895` pasa
`null_mode=getattr(config, "method_selection_null_mode", "diagnostic")`.

## 2. `09_fix44_classic_guard.patch` → `src/reii/main_workflow.py`

| hunk                       | veredicto | razón                                                                                                                                                                                                                                                                                                  |
| -------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `@@ -10371,6 +10371,10 @@` | OK        | Contra `.pcbase`: contexto 6 líneas exacto (10371-10376). `except RuntimeError: raise` a 12 espacios, mismo nivel que el `except Exception as e:` de 10374 ⇒ se cuelga del `try` de 10327, que es el que contiene el guard de 10330-10334. Orden correcto (`RuntimeError ⊂ Exception`, debe ir antes). |

Efecto verificado en `.pcfull:10371`: el guard de `#44` (línea 10328) ya no cae en
"Could not load best_params.json … Running optimizer if enabled" y sale como
traceback por `orchestrator.ejecutar(...)` (10694), que **no** está envuelto en un
`try/except` ancho. Aborta ruidosamente: correcto para un error de contrato.

Otros `RuntimeError` del fichero (`.pcfull` 677, 2112, 5599) **no** son víctimas del
mismo patrón: el de 677 es flujo de control deliberado capturado localmente
(`except Exception` → `continue`, no cachea el doc); 2112 (`_save`) y 5599
(invariante de proyección) no están dentro de un handler ancho. Ver Hallazgo H4.

## 3. `10_hardening_ward_max_n.patch` → `scripts/playground/hardening_bench.py`

| hunk                    | veredicto | razón                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `@@ -112,10 +112,14 @@` | DUDOSO    | Aplica bien y los contadores cuadran (6 contexto + 4 quitadas = 10; 6 + 8 nuevas = 14), pero (a) el bloque parcheado es la **"COPIA CANONICA (destino: `src/reii/method_c_policy.py`)"** (líneas 84-118), no el destino: copy 6000/320 vs. destino 4000/256; (b) el comentario afirma que el presupuesto cubre el peor camino, pero `calibrate_ward_max_n` no conoce ese camino y con 320 MiB devuelve **8500** (ver §Números). |

## 4. `11_formas_activas.patch` → `src/reii/gramatica.py`, `src/reii/main_workflow.py`

| hunk                                       | veredicto | razón                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------ | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gramatica.py @@ -19,7 +19,7 @@`           | OK        | `DEFAULT_GRAMATICA` en 21-29; 3 contexto + 1 quitada + 3 contexto = 7. `ia/gramatica.json` **no existe** ⇒ `load_gramatica()` devuelve los defaults y el cambio sí es activo en runtime.                                                                                                                                                             |
| `main_workflow.py @@ -1045,7 +1045,7 @@`   | OK        | `Config.content_pos` en 1046-1051; contadores 7/7. Coherente con el hunk 1, pero es la fuente **no autoritativa** (ver H2).                                                                                                                                                                                                                          |
| `main_workflow.py @@ -2513,22 +2513,19 @@` | DUDOSO    | Contexto exacto y único (22/19, primera línea `"pos_idx": i,` en 2513). Hace lo declarado, pero **quita dos predicados, no uno**: `is_stop` de la rama `marcadores` (2522) _y_ `not is_stop` de la rama activa (2531). Es lo único que hace el cambio efectivo, pero el efecto neto es "stopwords con POS de contenido pasan a activas" (ver H1/H2). |

## Hallazgos

**H1 — decisión del usuario (alto). `11` contamina métricas derivadas de `uce.lemmas`.**
No hay ningún bloqueante mecánico, pero el blast radius es mayor de lo que dice el
motivo del patch. `uce.lemmas` alimenta funciones que filtran **solo por POS**, sin
`is_stop`:
`gram/gramatical_analyzer.py:1289-1290` → `sofisticacion_lexica` (1136) y
`perfil_registro` (1200); `clasificar_registro` (1303-1316);
`build_doc_record` (`content`, 3071-3095). Con `11`, `no`, `más`, `ya`, `muy`,
`también`… (ADV con `zipf` altísima) entran en `mean_zipf`, `pct_low_freq`,
`oral_count` y en los términos de `doc_record`. Es decir: además de la retención
(+13.16 lemas/UCE), cambian métricas de calidad y el TF-IDF/registro a nivel de doc.
Si el objetivo era solo vocabulario ALCESTE, conviene decidir si `uce.lemmas` debe
seguir siendo "todas" o si esas 4 funciones deben filtrar `is_stop` explícitamente.

**H2 — decisión del usuario (medio). Dos fuentes de verdad para `content_pos`.**
`gramatica.py:22` (DEFAULT_GRAMATICA) gana siempre en runtime: `load_gramatica()`
fusiona sobre los defaults y por tanto **siempre** trae la clave, así que
`gramatica.get("content_pos", config.content_pos)` (`main_workflow.py:2178`) nunca
usa el fallback; `self.content_pos` (2189) se construye desde ahí. El default de
`Config` (1048) es código muerto salvo lectura directa de `Config`.
`11` actualiza ambos ⇒ coherentes, pero si alguien pone un `ia/gramatica.json`
(propio o vía `REII_GRAMATICA_PATH`) con la lista vieja, `11` queda inerte.
Sugerencia: una sola fuente y que `Config` derive de `gramatica.py`.

**H3 — cosmético/consistencia (medio). `10` alinea la copia, no el destino.**
`src/reii/method_c_policy.py` (creado por `03b`) sigue con
`DEFAULT_WARD_MAX_N = 4000` / `DEFAULT_WARD_MAX_BYTES = 256 MiB`. En producción es
**inerte** (único call site: `main_workflow.py:5235-5242` pasa
`ward_max_n=_METHOD_C_WARD_MAX_N` = 6000). No es un bug de hoy, pero:
(a) el comentario de `hardening_bench.py:84-86` dice que ese bloque es la copia
canónica del destino, y ya no coinciden; (b) `guarded_fit`/`preflight`/`ward_guard`
tienen `ward_max_n=DEFAULT_WARD_MAX_N` como **default de firma**: cualquier llamada
futura sin explícito se degrada a 4000 y el banco diría 6000; (c) los tests leen la
constante del módulo, así que no lo detectarían. **Alinear es seguro y verificado**:
`tests/test_backend_hardening.py:199-201` exige `8·n² ≤ 320 MiB` y `n ≤ 8000`; con
6000 da 274.7 MiB y 6000 ⇒ pasa.

**H4 — decisión del usuario (bajo). `09` amplía el radio de `RuntimeError`.**
`except RuntimeError: raise` vale para todo el `try` de 10327, no solo para el guard
de `#44`. Hoy no hay otro `RuntimeError` en ese bloque (`json.load`,
`_apply_best_params`, `_corpus_fingerprint`, sellado), pero cualquier guarda futura
que lance `RuntimeError` ahí abortará una corrida de ~2 h en vez de degradar. Es el
comportamiento deseado para contratos, pero conviene un tipo de excepción propio
(p. ej. `ContractError`) para no capturar `RuntimeError` ajenos.
Aparte: el `except Exception` ancho del evaluador de trials (HARDENING §8.2) sigue
igual y seguiría convirtiendo un `RuntimeError` de contrato lanzado **dentro** de un
trial en `_SENTINEL_TRIAL`; `09` no lo cubre (y probablemente no deba, por el uso de
`RuntimeError` como control de flujo en la precomputación de coref, `.pcfull:677`).

**H5 — cosmético. El banco se contradice al imprimir.**
Con `10`, `main()` imprime `calibrado=8500` y, dos líneas después,
`default propuesto=6000`; y `ward_guard(DEFAULT_WARD_MAX_N + 1000)` = `ward_guard(7000)`
imprime el aviso de degradación. Un lector concluirá que 8500 cabe (según el
calibrador) y que 7000 no (según la guarda). La causa es que `calibrate_ward_max_n`
usa la curva **medida** (condensada, ~4.3 B/par) y `DEFAULT_WARD_MAX_N` se justifica
con el **peor** camino (8 B/par). Antes pasaba lo mismo (7500 vs 4000), así que no es
regresión; pero si el comentario nuevo invoca el peor camino, el calibrador debería
recibir/declarar `bytes_per_pair=8` o imprimir ambos caminos.

**H6 — cosmético. Unidad.** El comentario dice "el condensado 4 B/par (144 MiB)";
son 144 **MB decimales** = 137.3 MiB (el banco mide en MiB, `peak_mb = peak/2**20`).
Heredado de HARDENING §2.2.

**H7 — cosmético (heredado, fuera de estos 4).** El docstring de `#5` sigue
diciendo `"gate"` como default (README §"Fallos previsibles" 2). `02c` no lo toca.

**H8 — decisión del usuario (bajo). `11` no toca `main_workflow_clasico.py`.**
Verificado: la clase `SegmentadorALCESTE` clásica tiene su propio `content_pos`
hardcodeado a 4 POS (`.pcfull:560`) y su propio `lematizar_uces` con `not is_stop`
(721-745). Consecuencia: moderno y clásico dejan de reproducir el mismo vocabulario
(clásico se queda en la línea base ~10.72). Si `--mode classic` debe reproducir
ALCESTE 2012, es lo correcto; si debe compararse con el moderno, hay que decidir.

## Números verificados

Memoria de ward (`MiB = B/2**20`, presupuesto 320 MiB = 335.544.320 B):

| camino                   | B/par |                        n=6000 | cabe en 320 MiB              | n máx con 320 MiB |
| ------------------------ | ----: | ----------------------------: | ---------------------------- | ----------------: |
| cuadrado completo (peor) |     8 | 288.000.000 B = **274.7 MiB** | sí (margen 45.3 MiB, 16.5 %) |          **6476** |
| condensado (medido)      |     4 |     144.000.000 B = 137.3 MiB | sí                           |              9157 |

- Mejor múltiplo de 500 bajo el peor camino: `floor(6476/500)·500` = **6000** ✓
  (también el mejor múltiplo de 1000, como dice HARDENING §2.2).
- `calibrate_ward_max_n` con la curva medida (`peak_mb ≈ 4.3e-6·n²`, de 17.2 MB a
  n=2000): `n ≤ sqrt(320/4.3e-6) = 8627` → **8500** (step 500). Con el presupuesto
  viejo (256 MiB) daba **7500**. Es decir: el calibrador **no** devuelve 6000.
- `ward_guard(DEFAULT_WARD_MAX_N + 1000)` = `ward_guard(7000)` → `False`, aviso
  `"ward: n=7000 > method_c_ward_max_n=6000 -> fallback dense_cdh"`. Sigue teniendo
  sentido como control negativo, y además está justificado por memoria
  (8·7000² = 392 MB = 373.8 MiB > 320 MiB).
- `method_c_policy`: 8·4000² = 122.1 MiB (hoy); 8·6000² = 274.7 MiB (si se alinea).

Conteos y validaciones:

- 4 patches, **8 hunks** (3 + 1 + 1 + 3). Contadores `@@`: todos correctos
  (7/7, 7/10, 7/7; 6/10; 10/14; 7/7, 7/7, 22/19).
- `git apply --check`: `02c` rc=0, `10` rc=0, `11` rc=0 (árbol limpio);
  `09` rc=0 con `-p1 --directory=.pcbase`.
- `ast.parse` OK en `.pcfull`: `main_workflow.py`, `gramatica.py`,
  `test_select_v4.py`, `hardening_bench.py`.
- `02c`: 2 asserts cambiados (498, 633) + 1 test con `null_mode="gate"` explícito
  (527-529); 0 asserts restantes que asuman el default viejo.
- `11`: `content_pos` pasa de 4 a **6** POS (`+PROPN`, `+X`) en las dos fuentes;
  se eliminan 2 predicados `is_stop` (2522 y 2531 de `main_workflow.py`).
- `.pcfull/tests/test_select_v4.py` refleja `02c`; `.pcfull/src/reii/main_workflow.py:10371`
  refleja `09` ⇒ la copia acumulada es consistente con los 4 patches.

## Veredicto

Los 4 patches **aplican** (`git apply --check` OK) y el resultado **parsea**
(`ast.parse` OK). Ningún hallazgo es bloqueante: los problemas son de alcance
(H1), de duplicación de constantes (H3) y de mensajes del banco (H5).
