# HARDENING — guardas, precondiciones y traza de fallos de Method C

Documento de cierre de los riesgos residuales **5–9** de `BACKEND_C.md`. No
redefine backends ni el gate de selección:

| documento                 | decide                                                          |
| ------------------------- | --------------------------------------------------------------- |
| `BACKEND_C.md`            | qué backend, contrato de salida, coste, señal de validez        |
| `SELECT_V4.md`            | qué método gana (bootstrap, nulo, dominancia)                   |
| `CLASSIC_MODE.md`         | política de `best_params.json`, `Kach`, allowlists              |
| **`HARDENING.md`** (este) | guardas de precondición, memoria, determinismo, traza de fallos |

Todo número de este documento está medido con
`scripts/playground/hardening_bench.py` (py 3.13.13, numpy 2.5.4, sklearn 1.8.0,
threadpoolctl 3.6.0, 4 CPUs; corrida completa 144.3 s). El banco contiene la
**copia canónica** de la política (`hardening_bench.py:88-510`), que es el
destino literal de `src/reii/method_c_policy.py` (§8.1). Los tests de
`tests/test_backend_hardening.py` importan esa copia mientras el módulo de
producción no exista.

Convención de salida: `[ok]`, `[!]`, `[X]`, `->`, `=>` (ASCII, consola cp1252).
Sin emojis.

---

## 1. Invariantes verificables

Cuatro invariantes. Cada una nombra el test que la pinza; si el test no existe o
no corre, la invariante es una opinión.

| id     | invariante                                                                                                                                                                                                                                         | donde se verifica                                                                                                                                                                                  |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **I1** | Ninguna precondición produce `IndexError`/`AxisError`/`TypeError` hacia arriba. Un corpus vacío, 1 UCE, `K > n`, `K < 2`, matriz 1-D, todo ceros, NaN o todo iguales devuelven `None` (o etiquetas con `n_clusters` real) **nunca** una excepción. | `test_preflight_nunca_lanza` (4 backends), `test_precondiciones_nunca_indexerror` (fit), `test_matriz_1d_devuelve_none`, `test_n_bajo_el_minimo_devuelve_none`, `test_k_mayor_que_n_devuelve_none` |
| **I2** | Un `method_c_backend` o una clave de `best_params.json` desconocidos **no abortan la corrida**. La fuente persistida degrada con `[!]`; sólo una constante interna (`source="internal"`) lanza `ValueError`.                                       | `test_backend_desconocido_degrada_con_aviso`, `test_backend_desconocido_no_aborta_el_preflight`, `test_backend_desconocido_interno_lanza`                                                          |
| **I3** | El `K` reportado es el `K` **real** de la partición, no `cdh_target_k`. `target_k` es objetivo, no promesa.                                                                                                                                        | `test_target_k_no_se_reporta_como_n_clusters`                                                                                                                                                      |
| **I4** | Ninguna excepción se pierde sin traza legible: tipo + mensaje + parámetros del trial, y traceback ≤ 4 KB en `user_attrs`.                                                                                                                          | `test_excepcion_del_backend_deja_rastro`                                                                                                                                                           |

Punto de entrada único y orden de capas:

```
_preflight(backend, n, target_k, ...)   # política pura: no toca datos, no lanza
    -> Precondition(ok, backend_effective, k_effective, reason, message)
_guarded_fit(backend, X, target_k, ...) # ÚNICO sitio que llama a un backend
    -> (labels|None, diag)              # diag SIEMPRE serializable
```

Las funciones de backend (`dense_cdh`, `ward_backend`, `kmeans_spherical`,
`hdbscan_backend`) son **privadas**: reciben matrices ya validadas y no
implementan precondiciones. Un `AxisError` de una llamada directa a
`dense_cdh(1-D)` no viola I1 porque ese camino no es API (ver §4, fila 1-D).

---

## 2. Guarda `ward_max_n` (memoria O(n²))

`AgglomerativeClustering(linkage="ward")` materializa una estructura de
distancias de tamaño cuadrático antes de fusionar. El coste **no depende de la
hipótesis científica** del trial: sólo de `n`. Por eso la guarda se evalúa en
`preflight`, antes de cualquier reserva:

```python
# hardening_bench.py:202-216
def ward_guard(n, max_n=DEFAULT_WARD_MAX_N) -> Tuple[bool, str]:
    if int(n) <= int(max_n):
        return True, ""
    return False, (
        f"[!] ward: n={int(n)} > method_c_ward_max_n={int(max_n)} "
        f'-> fallback "dense_cdh" (guard de memoria O(n^2))'
    )
```

### 2.1 Cuánto pesa de verdad

Medido con `tracemalloc` (`measure_ward_memory`, `hardening_bench.py:607-643`):

|    n |   d |   K | pico MB | segundos | B/n² |
| ---: | --: | --: | ------: | -------: | ---: |
|  528 |  30 |   8 |     1.8 |    0.322 |  6.8 |
| 1000 |  30 |   8 |     4.3 |    0.175 |  4.5 |
| 2000 |  30 |   8 |    17.2 |    0.539 |  4.5 |

Ajuste `peak ~ n^2.00` (exponente 2.00, `extrapolate_ward`,
`hardening_bench.py:646-660`) => **144 MiB a n=5800**. `B/n² ≈ 4.5` es el camino
_condensado_ (triángulo superior, 4 B por par). El otro camino posible de
sklearn (matriz cuadrada completa, 8 B por par) da `8 n²` = **257 MiB a n=5800**
(270 MB decimales, la cifra de la medición previa). Ambos son O(n²); el factor 2
es la elección de representación, no una discrepancia de crecimiento.

Unidades: MiB = `bytes / 2**20`, igual que `peak_mb` del banco.

### 2.2 Valor propuesto: `method_c_ward_max_n = 6000`

Presupuesto declarado: **320 MiB** para la estructura de distancias (16 % de un
contenedor de 2 GB; deja sitio para X, PCA(30) y el intérprete).

| camino                   | bytes/par | n máximo con 320 MiB | valor en 6000 |
| ------------------------ | --------: | -------------------: | ------------: |
| cuadrado completo (peor) |         8 |                 6476 |   **275 MiB** |
| condensado (medido)      |         4 |                 9157 |       144 MiB |

6000 es el mayor múltiplo de 1000 que respeta el **peor** camino bajo
presupuesto, y es >10× el corpus validado (528 UCEs; 600 en el artefacto de
`data/workflow_data.json`). Por encima de 6000 el corpus ya no es el que se
midió: la guarda es una alarma de política, no un predictor de OOM.

**Degradación, no aborto.** `preflight` devuelve
`backend_effective="dense_cdh"`, `reason="ward_memory_guard"` y el aviso `[!]`
de arriba. El trial sigue corriendo con otro backend y la diferencia queda
registrada en `diagnostics.method_c.backend_resolution`. Abortar por un knob de
alternancia costaría la corrida completa (2 h) para ahorrar 300 MB.

### 2.3 Por qué NO se hace dinámico por RAM

1. **Rompe la reproducibilidad.** La partición dependería de la RAM libre del
   momento: mismo `best_params.json`, dos máquinas, dos backends. Eso invalida
   §3 y la trazabilidad del paper.
2. **La RAM libre no es la restricción.** En Windows manda el _commit charge_ y
   el límite del contenedor, no `psutil.virtual_memory().available`; una guarda
   dinámica adivinaría el número equivocado y aun así no sería auditable.
3. **Un umbral fijo es un dato del paper.** "ward se usó hasta n=6000" se
   escribe en la tabla de métodos; "se usó hasta donde cupo" no.

`dense_cdh` es O(n·d) en memoria (`BACKEND_C.md` §4), así que el fallback nunca
hereda el problema.

---

## 3. Determinismo y BLAS

### 3.1 Qué se fija

- `random_state=42` explícito en `KMeans` y `n_init=10` fijo (el default de
  sklearn 1.8 ya es `"auto"`, que puede cambiar entre versiones: se fija).
- `dense_cdh`: `np.linalg.eigh` (determinista), `argsort` estable por contrato,
  canon de signo por el componente de mayor valor absoluto. Cero RNG.
- `ward`: sin RNG.
- `threadpool_limits(limits=1)` alrededor del `fit` (`hardening_bench.py:534-544`;
  producción: `method_c_policy`), **no** global.

### 3.2 Qué se midió

`measure_thread_determinism` (`hardening_bench.py:547-595`): 20 repeticiones por
celda, 1 hilo vs 4 hilos, sobre `synth_blobs`, `synth_null` y el corpus real
(235 UCEs × 30 dims, sólo lectura). Resultado de la corrida de validación:

```
-> inestables con 1 hilo: 0/9 celdas; 1 vs N con ARI<1: 0/9 celdas
```

9/9 celdas con `distinct=1` en ambos modos, `ARI=1.000`, `pts!= 0`. Las 6 celdas
sintéticas son el "6/6" del informe previo; las 3 restantes son `real_ctx`.

### 3.3 Qué NO se garantiza

- **Bitwise entre máquinas.** Distinto vendor de BLAS (OpenBLAS vs MKL), ISA
  (AVX2/AVX-512) o `sklearn` pueden elegir otra ruta de `KMeans`/`ward`. Lo
  garantizado es _misma máquina + mismo entorno_ => mismas etiquetas.
- **Identidad de etiquetas entre corridas de distinto backend**: la partición es
  la misma pero los ids pueden permutarse; se compara con ARI, no con `==`.
- **`hdbscan`**: determinista para estos datos, no por construcción
  (`BACKEND_C.md` §3).
- **El resto del pipeline.** `threadpool_limits(1)` cubre la región envuelta; no
  convierte el proceso en determinista (Stanza, red, orden de ficheros).
- **`n_init`.** Si alguien lo sube, cambia el óptimo local. Está fijado a 10 y
  no es un knob.

`threadpool_limits(1)` se mantiene como **seguro barato**: si alguien olvida
`OMP_NUM_THREADS=1` (producción no lo exporta), el fit sigue confinado. Cuesta
~0 ms y evita el modo "800 ms con 4 hilos vs 128 ms con 1" ya medido en la arena
para n≈600.

---

## 4. Tabla de precondiciones

`floor` = `hdbscan_uce_min_cluster_floor` (5 por defecto); `mcs` =
`max(floor, round(frac*n))`. "Hoy" = comportamiento del banco
(`hardening_bench.py:236-404`, `419-510`) medido en §11 del propio banco.

| precondición                                     | detección     | salida esperada              | mensaje exacto                                                         | hoy                                                   |
| ------------------------------------------------ | ------------- | ---------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------- |
| `n = 0`                                          | `preflight`   | `None`                       | `[!] method_c: corpus vacio (n=0) -> sin particion`                    | cumple                                                |
| `n < 2*floor` (`dense_cdh`)                      | `preflight`   | `None`                       | `[!] dense_cdh: n=%d < 2*floor=%d -> sin particion (K=1 no es viable)` | cumple                                                |
| `n < 2` (`ward`, `kmeans`)                       | `preflight`   | `None`                       | `[!] ward: n=%d < n_clusters minimo (2) -> sin particion`              | cumple                                                |
| `n < mcs` (`hdbscan`)                            | `preflight`   | `None`                       | `[!] hdbscan: n=%d < min_cluster_size=%d -> sin particion (K=0)`       | cumple                                                |
| `K > n` (`ward`, `kmeans`)                       | `preflight`   | `None`                       | `[!] ward: n=%d < target_k=%d -> sin particion`                        | cumple                                                |
| `target_k < 2` (cualquier backend)               | `preflight`   | `None`                       | `[!] method_c: target_k=%d < 2 -> sin particion`                       | **incumple** sólo en `kmeans_spherical` (xfail)       |
| `target_k < 2` (`dense_cdh`)                     | `preflight`   | `None`                       | idem                                                                   | cumple (`reason=k_below_kmin`)                        |
| `K` efectivo `< 2` (`dense_cdh`, `n//floor = 1`) | `preflight`   | `None`                       | `[!] dense_cdh: K efectivo=%d < 2 -> sin particion`                    | cumple                                                |
| `K > 20` (`dense_cdh`)                           | `preflight`   | cap a 20 + aviso             | `[!] dense_cdh: target_k=%d > n//floor=%d -> K efectivo=%d`            | cumple el cap                                         |
| `K > 20` (`ward`, `kmeans`)                      | —             | lo censura `_evaluate_trial` | —                                                                      | aceptado (`main_workflow.py:8609`)                    |
| ceros > 50 % (con `n_zero_rows`)                 | `preflight`   | `None`                       | `[!] method_c: %d/%d zero-vectors (%.0f%%) > 50%% -> sin particion`    | cumple                                                |
| ceros > 50 % (autodetectado)                     | `guarded_fit` | `None`                       | idem                                                                   | **incumple**: `K=4` basura (xfail)                    |
| NaN / Inf                                        | `guarded_fit` | `None`                       | `[!] method_c: %d valores no finitos -> sin particion`                 | **incumple**: `K=4` basura (xfail)                    |
| clase única / todo iguales (`gate=0.08`)         | `dense_cdh`   | `None` (K=1)                 | `[!] dense_cdh: n_clusters=%d < 2 -> sin particion`                    | **incumple**: devuelve etiquetas K=1 (xfail)          |
| clase única / todo iguales (`gate=0.0`)          | —             | `None`                       | idem                                                                   | **incumple**: `K=4` fabricado; default peligroso (§7) |
| matriz 1-D                                       | `guarded_fit` | `None`                       | `[!] method_c: corpus vacio (n=0) -> sin particion`                    | cumple (no `AxisError`)                               |
| matriz no convertible                            | `guarded_fit` | `None`                       | `[!] method_c: excepcion controlada TypeError: ...`                    | cumple (degradación explícita)                        |

Notas que el contrato debe leer junto con `BACKEND_C.md` §7:

- **`K=1` no es partición.** `main_workflow.py:5196`
  (`if len(real) < 2: return None`) es el contrato de HDBSCAN y se extiende a
  los cuatro backends.
  Donde "hoy" dice _incumple_, `guarded_fit` devuelve etiquetas de una sola
  clase; aguas arriba `_evaluate_trial` (`main_workflow.py:8609`) las censura,
  pero el backend no debe emitirlas: el wrapper es API.
- **`n_zero_rows` no se confía del llamador.** Hoy es un parámetro con default
  `0` (`hardening_bench.py:426,444`); el banco ya lo calcula dentro de
  `dense_cdh` (`nz`, `hardening_bench.py:367`). Debe calcularse en
  `guarded_fit` una sola vez (misma definición que `main_workflow.py:5125`,
  `np.all(mat == 0, axis=1)`).
- **El gate `min_r2_dense` es 0.08 en producción**, no el default `0.0` del
  banco (`hardening_bench.py:428`). Con `0.0`, "todos iguales" produce 4 clases
  de puntos idénticos (medido).

---

## 5. Contrato `Kach` vs `target_k`

`cdh_target_k` es **objetivo, no promesa**. Dos mecanismos legítimos lo
incumplen:

1. el gate `dense_cdh_min_r2_dense = 0.08` corta el primer split cuyo R² denso
   no llega (ruido isótropo: 0.036–0.064; estructura real: 0.103–0.142), y
2. el piso `floor` impide partir hojas por debajo de `2*floor` UCEs.

Medido en el banco (528 UCEs, `floor = n/(2K)`):

| dataset     | `target_k` | `K` real | splits | rechazos r2_dense |
| ----------- | ---------: | -------: | -----: | ----------------: |
| synth_blobs |          3 |        3 |      2 |                 0 |
| synth_blobs |          5 |        5 |      4 |                 0 |
| synth_blobs |          8 |        8 |      7 |                 0 |
| synth_blobs |         14 |        9 |      8 |                 1 |
| synth_null  |          3 |        1 |      0 |                 1 |
| synth_null  |          5 |        1 |      0 |                 1 |
| synth_null  |          8 |        1 |      0 |                 1 |
| synth_null  |         14 |        1 |      0 |                 1 |

`synth_null` da `K=1` en las cuatro celdas: el gate funciona y `Kach = 1` debe
leerse como "sin partición", no como "una clase". `synth_blobs` coincide en 3/4
celdas: `target_k=14 -> 9` es correcto y **silencioso** sin el aviso
`format_target_k_notice` (`hardening_bench.py:407-416`):

```
[!] dense_cdh: target_k=14 -> 9 clases (1 splits rechazados por r2_dense)
```

Correlación medida en v4 entre el objetivo y el K alcanzado:
`rho(cdh_target_k, -K) = -0.89` (a más objetivo, menos K penalizado: el sampler
sí ve el efecto, pero sobre una etiqueta que no es la que creía).

**Regla para el optimizador:** leer `n_clusters` **REAL** (el de la partición
del ganador y, en `wc_only`, el de la partición cruda de A1) y registrar
`k_target`/`k_ach`/`k_shortfall` como `user_attrs`. Sustituir `Kach` por
`target_k` sería falsear el objetivo; suprimirlo perdería la señal. Esto es
exactamente `CLASSIC_MODE.md` §7, y el diff de `_evaluate_trial` es
`CLASSIC_MODE.md` §9.3 — **no se duplica aquí**. La única diferencia de este
documento (§8.3) es el **orden**: el gate duro de rango `2..20` debe aplicarse
al `K` real _después_ de refinarlo, porque §9.3 lo aplica a `n_clusters`
(estable) y un ganador con `winner_k=25` pasaría el gate reportando `-25`.

---

## 6. Traza de fallos

Causa raíz del trial #81 de v4: `_evaluate_trial` envuelve
`_evaluacion_rapida` en un `try/except` que sólo hace `logger.debug`. Con el
logger a nivel INFO, el mensaje no llega a ningún sitio: el trial se convierte
en `_SENTINEL_TRIAL` (`-2,-1,0`) y Optuna lo cuenta como FAIL sin causa.

Hoy (`src/reii/main_workflow.py:8590-8594`):

```python
try:
    result = self._evaluacion_rapida(cfg, uc_cfg=uc_cfg)
except Exception as e:
    logger.debug("Trial %d exception: %s", trial.number, e)
    return _SENTINEL_TRIAL
```

Formato exacto del aviso (`[!]` = tipo + mensaje + params del trial):

```
[!] trial 81 fallo en _evaluacion_rapida: ValueError: <mensaje> | params={'cdh_target_k': 8, 'hdbscan_uce_min_cluster_frac': 0.04, ...}
```

- `tipo` = `type(e).__name__`, `mensaje` = `str(e)[:200]`, `params` = subconjunto
  de las claves que gobiernan la partición y la estabilidad (no los 20 parámetros
  del sampler), truncado a 300 caracteres.
- Traza completa: `traceback.format_exc()[-4096:]` en
  `user_attr["exception_tb"]`. 4 KB porque un traceback de sklearn/HDBSCAN ronda
  1–3 KB y no infla el payload del estudio.
- El trial **sigue siendo centinela**: no se cambia la semántica del objetivo, se
  añade auditoría.
- Resumen post-optimize: `[!] N trial(s) con excepción -> centinela
(user_attrs exception_type/exception_tb)` y las 3 últimas causas.

El diff está en §8.2. Es el mismo de `CLASSIC_MODE.md` §9.3 salvo que el mensaje
incluye `params`; los nombres de `user_attrs` (`trial_failed`,
`exception_type`, `exception_msg`, `exception_tb`) son idénticos para que §9.4
(`guarded_trial_eval`, el contador `_trial_failures`) siga funcionando sin
cambios.

---

## 7. Riesgos conocidos NO resueltos

1. **Allowlist de `config_store.py` (bug preexistente).**
   `CONFIG_CATEGORIES["clasificacion"]` (`src/reii/backend/config_store.py:36-67`)
   contiene `hdbscan_uce_min_cluster_size` y `_loose` (L62-63) pero **no**
   `hdbscan_uce_min_cluster_frac` ni `_frac_loose`, que sí son campos de
   `Config` (`main_workflow.py:1023-1024`) y sí están en el espacio de búsqueda
   (`main_workflow.py:8430-8431`). Consecuencia doble:
   `set_config_many` las descarta en silencio (`config_store.py:252`) y
   `_STEP_CONFIG_KEYS["classify"]` (`src/reii/backend/steps.py:170-178`) tampoco
   las hashea, así que **ni siquiera añadiéndolas a la allowlist se invalidaría
   la caché de clasificación**. La primera mitad es el diff de 2 líneas de §8.4;
   la segunda es `CLASSIC_MODE.md` §8.2/§9.4 y no se toca aquí.
   **§8.4 NO entra en este patchset**: `FINAL_PATCHSET.md` decide. Se documenta
   porque sin él, mover `hdbscan_uce_min_cluster_frac` desde el dashboard no
   tiene efecto observable, y eso es indistinguible de "la guarda no funciona".
2. **`ward` sigue siendo O(n²) si el corpus crece.** La guarda evita el OOM, no
   el coste. A n=6000 el fit tarda ~5 s (0.539 s a n=2000, exponente 2.0 en
   tiempo) por trial; con 300 trials sería inaceptable. Si el corpus pasa de
   ~1000 UCEs, `ward` debe salir del espacio de búsqueda, no sólo de la guarda.
3. **`dense_cdh` hereda la monotonía divisiva de A.** El CV de tamaños de clase
   empeora con K (`cv_sz` 0.286 en K=8 -> 1.371 en K=20) y `max_class_share` se
   queda estancado en 0.323: la bisección best-first reparte masa, no la
   equilibra. Un `max_class_share` bajo en `dense_cdh` **no** es evidencia de
   estructura equilibrada; es el estilo del árbol. El gate de dominancia
   (`BACKEND_C.md` §5) debe leerse con esa advertencia.
4. **`min_r2_dense=0.0` por defecto en el banco** (`hardening_bench.py:428`)
   fabrica splits sobre puntos idénticos. El valor de producción (0.08) debe
   fijarse en la política, no depender de que el llamador lo pase.
5. **Sin chequeo de finitud.** NaN/Inf entran en `eigh` y producen una partición
   de 4 clases con `r2_dense` NaN. El gate no corta porque `nan < 0.08` es
   `False`.
6. **`kmeans_spherical` acepta `target_k=1`** (`hardening_bench.py:381`,
   `k_effective=int(target_k)`) y emite una partición de una clase, contra
   `BACKEND_C.md` §7.
7. **Asimetría de `K_MAX`**: `effective_target_k` acota `dense_cdh` a 20, pero
   `ward`/`kmeans` reciben el `target_k` crudo (medido: `K=21` con n=50 se
   ejecuta). Hoy lo censura `_evaluate_trial` (`main_workflow.py:8609`); el
   backend no debe confiar en ello. §8.1 añade el cap por rama **sin** tocar
   `k_requested`, para que el aviso y el `K` real sigan siendo auditables.
8. **Deduplicación de avisos**: el banco emite el `[!]` de `ward` en cada
   llamada a `preflight`; en producción debe imprimirse una vez por
   `(backend, n)` distinto para no llenar el log de 300 líneas iguales.

---

## 8. Diffs exactos

Rutas y líneas reales del árbol actual. Nada de esto está aplicado: `src/**` es
sólo lectura para este documento.

### 8.1 `src/reii/method_c_policy.py` — módulo nuevo

No es un diff de 400 líneas. El módulo es **copia verbatim** de
`scripts/playground/hardening_bench.py:88-510` (constantes, `resolve_method_c_backend`,
`Precondition`, `ward_guard`, `effective_target_k`, `preflight`,
`format_target_k_notice`, `guarded_fit`) más los cinco hunks siguientes. Sin
ellos, los tests marcados `xfail` en §9 quedarían rojos al aplicar el módulo.

```diff
@@ hardening_bench.py:272  (preflight, tras el chequeo de zero-rows, antes de la rama dense_cdh)
+    # K<2 no es una particion en NINGUN backend: rechazar en la politica, no
+    # dejar que ward lo acote a 2 (fabrica el corte minimo) o que kmeans lo
+    # emita tal cual. BACKEND_C.md §7.
+    if target_k < K_MIN:
+        return _fail(
+            "k_below_kmin",
+            f"{WARN} method_c: target_k={target_k} < {K_MIN} -> sin particion",
+        )
+    # target_k > K_MAX NO se rebana aqui: `k_requested` debe seguir siendo el
+    # valor pedido (auditoria, I3 y §5). Cada rama lo acota y lo anuncia.
@@ hardening_bench.py:351  (rama ward: cap explicito, sin tocar k_requested)
-        k_eff = max(K_MIN, min(target_k, n))
+        k_eff = max(K_MIN, min(target_k, n, K_MAX))
+        if k_eff != int(target_k):
+            notices.append(
+                f"{WARN} ward: target_k={target_k} > K_MAX={K_MAX} "
+                f"-> K efectivo={k_eff}"
+            )
@@ hardening_bench.py:370-385  (rama kmeans_spherical)
+        k_eff = min(int(target_k), K_MAX)
+        if k_eff != int(target_k):
+            notices.append(
+                f"{WARN} kmeans_spherical: target_k={target_k} > K_MAX={K_MAX} "
+                f"-> K efectivo={k_eff}"
+            )
         return Precondition(
             ok=True,
             backend=backend,
             backend_effective=backend,
             k_requested=target_k,
-            k_effective=int(target_k),
+            k_effective=k_eff,
             reason="ok",
             message="",
             notices=tuple(notices),
         )
@@ hardening_bench.py:236  (preflight: dense_cdh ya acota con effective_target_k)
@@ hardening_bench.py:436  (guarded_fit, sustitucion de las dos primeras lineas)
-    X = np.asarray(X)
-    n = int(X.shape[0]) if X.ndim == 2 else 0
+    try:
+        X = np.asarray(X, dtype=np.float64)
+    except (TypeError, ValueError) as exc:
+        return None, {
+            "backend_requested": backend,
+            "backend_used": backend,
+            "precondition": {
+                "ok": False,
+                "reason": "not_convertible",
+                "message": f"{WARN} method_c: matriz no convertible a float64 "
+                           f"({type(exc).__name__}) -> sin particion",
+                "k_requested": int(target_k),
+                "k_effective": None,
+            },
+            "notices": [],
+        }
+    n = int(X.shape[0]) if X.ndim == 2 else 0
+    if n and not np.isfinite(X).all():
+        n_nonfinite = int((~np.isfinite(X)).any(axis=1).sum())
+        return None, {
+            "backend_requested": backend,
+            "backend_used": backend,
+            "precondition": {
+                "ok": False,
+                "reason": "nonfinite_values",
+                "message": f"{WARN} method_c: {n_nonfinite} valores no finitos "
+                           f"(NaN/Inf) -> sin particion",
+                "k_requested": int(target_k),
+                "k_effective": None,
+            },
+            "notices": [],
+        }
+    if n and not n_zero_rows:
+        # Misma definicion que main_workflow.py:5125. No se confia del llamador.
+        n_zero_rows = int(np.all(X == 0, axis=1).sum())
@@ hardening_bench.py:502  (validacion de salida; K=1 deja de ser particion)
     real = np.unique(labels[labels >= 0])
     diag["n_clusters"] = int(len(real))
     diag["n_noise"] = int(np.sum(labels == -1))
+    if int(len(real)) < K_MIN:
+        diag["precondition"]["ok"] = False
+        diag["precondition"]["reason"] = "below_kmin"
+        diag["precondition"]["message"] = (
+            f"{WARN} {be}: n_clusters={int(len(real))} < {K_MIN} -> sin particion"
+        )
+        return None, diag
@@ hardening_bench.py:428  (default del gate, firma de guarded_fit)
-    min_r2_dense: float = 0.0,
+    min_r2_dense: float = 0.08,   # valor de produccion (BACKEND_C.md §6)
```

### 8.2 `src/reii/main_workflow.py` — traza del `except` mudo

```diff
--- a/src/reii/main_workflow.py
+++ b/src/reii/main_workflow.py
@@ -7852,3 +7852,8 @@
 _SENTINEL_TRIAL = (-2.0, -1.0, 0.0)
+
+# Tope del traceback persistido en user_attrs. 4 KB: un traceback de
+# sklearn/HDBSCAN ronda 1-3 KB y no infla el payload de Optuna.
+_MAX_TB_CHARS = 4096
+
@@ -8590,5 +8595,26 @@
         try:
             result = self._evaluacion_rapida(cfg, uc_cfg=uc_cfg)
         except Exception as e:
-            logger.debug("Trial %d exception: %s", trial.number, e)
+            # Probe v4, trial #81: FAIL con params factibles y SIN traceback.
+            # Un logger.debug no llega al log (nivel INFO) ni al estudio.
+            _tb = traceback.format_exc()
+            _params_brief = {
+                k: params.get(k)
+                for k in (
+                    "cdh_target_k",
+                    "hdbscan_uce_min_cluster_frac",
+                    "hdbscan_uce_min_cluster_frac_loose",
+                    "coref_context_units_tight",
+                    "coref_context_units_loose",
+                )
+                if k in params
+            }
+            trial.set_user_attr("trial_failed", True)
+            trial.set_user_attr("exception_type", type(e).__name__)
+            trial.set_user_attr("exception_msg", str(e)[:500])
+            trial.set_user_attr("exception_tb", _tb[-_MAX_TB_CHARS:])
+            logger.warning(
+                "[!] trial %d fallo en _evaluacion_rapida: %s: %s | params=%s",
+                trial.number, type(e).__name__, str(e)[:200], _params_brief,
+            )
             return _SENTINEL_TRIAL
```

### 8.3 `src/reii/main_workflow.py` — gate de K sobre el K real

Reproduce el cambio de `CLASSIC_MODE.md` §9.3 (`-n_clusters` -> `-_k_ach`) y
anade **cuando** se aplica el gate. No lo contradice: los `user_attrs` son los
mismos y el valor devuelto tambien. La unica diferencia es que aqui el gate se
revalida sobre el `K` ya refinado, y alli se aplica solo a `n_clusters`.

```diff
--- a/src/reii/main_workflow.py
+++ b/src/reii/main_workflow.py
@@ -8606,5 +8611,15 @@
         n_clusters = int(len(np.unique(labels_primary)))

-        # ── Gate duro: K fuera de rango útil ─────────────────────────────
-        if n_clusters < 2 or n_clusters > 20:
-            return _SENTINEL_TRIAL
+        # ── K real vs K pedido (CLASSIC_MODE.md §7) ──────────────────────
+        # `n_clusters` cuenta solo UCEs ESTABLES. El K que el sampler cree
+        # haber evaluado es `cdh_target_k`. Se separan los tres numeros.
+        _k_target = params.get("cdh_target_k")
+        _k_stable = int(n_clusters)
+        _k_ach = _k_stable  # se refina por rama (winner_k / A1 crudo)
+
+        # ── Gate duro: K fuera de rango util ─────────────────────────────
+        # Se aplica al K REAL. Un objetivo fuera de rango ya lo censura el
+        # sampler (espacio [3, k_hi], main_workflow.py:8413).
+        if _k_ach < 2 or _k_ach > 20:
+            trial.set_user_attr("k_rejected", int(_k_ach))
+            return _SENTINEL_TRIAL
@@ -8680,4 +8695,4 @@
             return (
                 float(coverage_primary),
                 float(sel.winner_ari),
-                -float(n_clusters),
+                -float(_k_ach),
             )
@@ -8702,1 +8717,10 @@
-            return ari, float(coverage_primary), -float(n_clusters)
+            _k_ach = int(len(np.unique(arr_a[valid])))
+            if _k_ach < 2 or _k_ach > 20:
+                trial.set_user_attr("k_rejected", int(_k_ach))
+                return _SENTINEL_TRIAL
+            trial.set_user_attr("k_ach", _k_ach)
+            trial.set_user_attr(
+                "k_target", int(_k_target) if _k_target is not None else -1
+            )
+            trial.set_user_attr("k_stable", _k_stable)
+            return ari, float(coverage_primary), -float(_k_ach)
```

En la rama `select`, `_k_ach` se refina con
`int(getattr(sel, "winner_k", 0) or n_clusters)` **antes** del `return`, y se
registran `k_ach`/`k_target`/`k_shortfall`/`k_stable` como en §9.3. La
revalidación `2..20` se repite tras el refinamiento (una línea), porque
`winner_k=25` no debe llegar al frente de Pareto como `-25`.

### 8.4 `src/reii/backend/config_store.py` — allowlist (opcional, NO en el patchset)

```diff
--- a/src/reii/backend/config_store.py
+++ b/src/reii/backend/config_store.py
@@ -62,6 +62,8 @@
         "hdbscan_uce_min_cluster_size",
         "hdbscan_uce_min_cluster_size_loose",
+        "hdbscan_uce_min_cluster_frac",
+        "hdbscan_uce_min_cluster_frac_loose",
         "hdbscan_uce_epsilon",
         "hdbscan_uce_min_samples",
         "hdbscan_uce_metric",
```

Cierra la mitad "config_store" del riesgo §7.1. No invalida la caché de
clasificación por sí solo (`steps.py:170-178`); esa mitad es
`CLASSIC_MODE.md` §8.2 y **queda fuera de este documento**.

---

## 9. Checklist de aceptación

Ejecutable sin pipeline (>4 min prohibido). Cada casilla nombra un test de
`tests/test_backend_hardening.py`.

- [ ] `python -m pytest tests/test_backend_hardening.py -q` termina en < 5 s,
      verde, con 0 fallos. Si la API no está aplicada, **skipped**, no error de
      colección.
- [ ] I1: la rejilla `n ∈ {0,1,2,7,50} × K ∈ {1,2,3,8,21}` no lanza excepción
      en los **cuatro** backends en `preflight` (`test_preflight_nunca_lanza`) ni
      con `fit` real sobre `dense_cdh` (`test_precondiciones_nunca_indexerror`).
      `ward`/`kmeans` no se fitean: importar `sklearn.cluster` cuesta 14.87 s
      (medido con `-X importtime`) y rompería el presupuesto de 5 s; su
      contrato se cubre en `preflight` y en la arena.
- [ ] I1: 1-D, `n<2*floor`, `K>n` devuelven `None` con `reason` conocido.
- [ ] I2: `method_c_backend="banana"` desde fuente persistida => `hdbscan` +
      `[!]`, sin abortar; desde fuente interna => `ValueError`.
- [ ] I2: clave desconocida en `best_params.json` => `[!]` y corrida viva
      (`CLASSIC_MODE.md` §6/§9.4; cubierto allí, no se duplica).
- [ ] I3: `target_k=8` sobre ruido puro reporta `n_clusters=1` y aviso
      `target_k=8 -> 1 clases`, nunca `n_clusters=8`.
- [ ] I4: un backend que lanza deja `precondition.reason="exception"` y
      `[!] <backend>: excepcion controlada <Tipo>: <msg>`; en producción,
      `_evaluate_trial` deja 4 `user_attrs` y `logger.warning` (§8.2).
- [ ] §2: `ward_guard(max_n)` pasa, `ward_guard(max_n+1)` degrada a `dense_cdh`
      y el fit de fallback devuelve etiquetas de longitud `n`.
- [ ] §3: `python scripts/playground/hardening_bench.py` (sección 1) reporta
      `0/9 celdas` inestables y `ARI=1.000`.
- [ ] §4: los 4 `xfail` (NaN, ceros autodetectados, `kmeans` con `K=1`, `K=1`
      post-validación) son la lista de trabajo: cuando §8.1 esté aplicado deben
      pasar a `xpass`, y entonces se quita el marcador.
- [ ] §5: el frente de Pareto lleva `k_target`/`k_ach`/`k_shortfall` y el
      objetivo `-K` usa el K real.
- [ ] §7.1 documentado y **no** aplicado en este patchset (decisión de
      `FINAL_PATCHSET.md`).
