# Modo clásico (`--mode classic`): ALCESTE 2012 como configuración del mismo pipeline

Estado: **diseño + diffs pendientes de aplicar**. No se toca `data/`, `output/`, `models/`
ni `src/**` en esta entrega (los diffs van aquí; se aplican a mano).

Validación ejecutable: `scripts/playground/classic_mode_probe.py` (sin docker, sin
`reii.main_workflow`, <60 s).

---

## 1. Mapeo del paper a los cuatro escenarios

| Esc.  | Qué es                             | En este código                                                                                 |
| ----- | ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| **A** | ALCESTE clásico (Reinert 2012)     | `--mode classic` → `run_mode="classic"`, `classification_mode="wc_only"`, sin piso duro, sin C |
| **B** | ALCESTE + innovaciones (el actual) | default: `classification_mode="select"`, piso duro, liminal, coref, C                          |
| **C** | Embeddings densos                  | `classification_mode="emb_only"` + backend denso (`docs/design_v5/BACKEND_C.md`)               |
| **D** | HDBSCAN actual (**diagnóstico**)   | `hdbscan_uce_*`; se conserva sólo como diagnóstico, no como criterio                           |

El objetivo de A es **fidelidad**, no rendimiento. Cada override de §2 existe porque el
paper lo fija o porque el código lo contradice; todo lo demás queda intacto (y el probe lo
verifica con asserts: la diferencia es _exactamente_ la tabla).

---

## 2. Tabla de overrides: clásico vs moderno

Sólo 8 campos cambian respecto de la configuración **operativa** (la que `__main__`
construye). `Config` = dataclass de `main_workflow.py`; `uc` = `UCBuilderConfig`.

> **Nuance verificada por el probe**: el default _del dataclass_ es
> `classification_mode="wc_only"`, pero el default _de la corrida_ es `"select"`
> (`__main__` lo fija a mano, L10505). El modo clásico fuerza `wc_only` en ambos
> niveles, de modo que ni `ConfigStore` ni el default de `__main__` lo desvían.
> Por eso, comparado contra `Config()`, `classification_mode` es un **no-op**; la
> diferencia real es 7 campos + el campo nuevo `run_mode`.

| Campo                     | Moderno (B)            | Clásico (A) | Por qué                                                                                                                                                                                                    | Paper                                                                  |
| ------------------------- | ---------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `run_mode`                | `"modern"`             | `"classic"` | Trazabilidad + aislamiento de caché (§4). No existía; se añade.                                                                                                                                            | —                                                                      |
| `classification_mode`     | `"select"` (operativo) | `"wc_only"` | ALCESTE tiene **un** clasificador (CDH descendente). B y C son invenciones REII.                                                                                                                           | §2 Paso 2; §3 Paso 2                                                   |
| `min_forms_uc`            | `[13, 17]`             | `[10, 14]`  | Las dos corridas independientes son **10 vs 14 palabras por UC**; de ahí sale la estabilidad.                                                                                                              | §2 Paso 1; §2 tabla "Tamaño de UC 10 vs 14"                            |
| `cdh_enforce_min_leaf`    | `True`                 | `False`     | El clásico **no tiene piso de hoja**: el piso es sólo regla de parada de la recursión, así que una clase puede quedar con 1 UC y los cortes 1-vs-(n−1) son legítimos. El piso duro es un _fix_ propio (B). | §2 Paso 2 (no menciona piso); §2 "Máximo de clases: por configuración" |
| `optimize`                | `True`                 | `False`     | ALCESTE 2012 no busca hiperparámetros: los fija el paper. Optimizar A lo convertiría en B.                                                                                                                 | §2 tabla completa (valores fijos)                                      |
| `use_embeddings`          | `True`                 | `False`     | Sin embeddings no hay retrofitting ni Method C.                                                                                                                                                            | §2 (no existe); §3 Método C                                            |
| `use_liminal_projection`  | `True`                 | `False`     | La proyección liminal es explícitamente **abductiva** y de REII; el propio comentario del código dice `Default False: ALCESTE puro`.                                                                       | §3 Paso 6 ("Proyectado")                                               |
| `auto_calibrate_uce_size` | `True`                 | `False`     | La calibración bayesiana del tamaño de UCE es una capa propia. El clásico usa el valor publicado y fijo.                                                                                                   | §2 (tamaños fijos)                                                     |

`db_local_path`, `optimize_storage`, rutas de selección y de salida **no** son "fidelidad":
son aislamiento y van en §4.

### 2.1 Deliberadamente SIN cambio (y por qué)

Esta tabla es tan importante como la anterior: dice qué **no** se toca para no inventar.

| Campo                                                    | Valor           | Decisión                         | Motivo                                                                                                                                                                                                                                                          |
| -------------------------------------------------------- | --------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pseudocount` (ε)                                        | `0.00`          | **Sin cambio.** Ver §3.          | El paper dice 0.1, pero el código **nunca lee** `pseudocount`: es un knob muerto. Poner 0.1 no cambiaría nada y sugestionaría una fidelidad inexistente.                                                                                                        |
| `tsj`                                                    | `3`             | Sin cambio                       | §2 no da un valor numérico distinto; el `alceste_config` histórico ya usaba 3. Cambiarlo sería invención.                                                                                                                                                       |
| `similarity_threshold` (`uc`)                            | `0.45`          | Sin cambio                       | El "umbral de similitud" de ALCESTE es el **CTTEST**, un coeficiente de asociación (§2 tabla), no un corte de cohesión entre UCEs. No hay escalar publicado; 0.45 es un proxy nuestro. Se documenta como discrepancia, no se "arregla" con un número inventado. |
| `use_ctest` / `ctest_threshold`                          | `False` / `0.3` | Sin cambio                       | El CTTEST en este código es un **filtro de términos**, no el criterio de partición del paper. Activarlo cambiaría el vocabulario, no la fidelidad estructural. Se reporta como métrica.                                                                         |
| `min_uce_words`                                          | `2`             | Sin cambio                       | 2 vs 3 es un mínimo de segmentación, no el "tamaño de UC" del paper (que es `min_forms_uc`). `alceste_config` usaba 3; no hay cita que lo fije. Se deja y se anota.                                                                                             |
| `min_cluster_size_cdh`                                   | `0.10`          | Sin cambio                       | El piso como **parada** sigue existiendo en ALCESTE ("máximo de clases por configuración"). Lo que se apaga es su enforcement como hoja mínima (`cdh_enforce_min_leaf`).                                                                                        |
| `use_cdh`                                                | `True`          | Sin cambio                       | Es el núcleo clásico.                                                                                                                                                                                                                                           |
| `use_term_stability`, `use_rf_shap`, `use_cah_per_class` | —               | Sin cambio (**deuda declarada**) | Son capas REII. Apagarlas sería más fiel, pero el encargo no las incluye y afectan a consumidores aguas abajo (φ, AFC, CAH, P1–P5). Se listan como trabajo futuro, no se tocan.                                                                                 |

---

## 3. Hallazgo: ε = 0.1 del paper no está implementado (y el código lo fuerza a 0.0)

**Localizado y confirmado con grep** (`pseudocount` en `src/reii/main_workflow.py`):

1. `Config.pseudocount: float = 0.00` — declaración (L837).
2. `Optimizador._evaluate_trial`: `cfg.pseudocount = 0.0  # BLOQUE 1: nunca sumar ε` (L8530).
3. `WorkflowOrchestrator._apply_best_params`: `self.config.pseudocount = 0.0  # invariante` (L9640).
4. `get_search_space`: `# pseudocount eliminado (BLOQUE 1: siempre 0.0)` (L8412).
5. `steps.py`: `"afc": ["pseudocount"]` — única referencia a un consumidor.
6. El fork `main_workflow_clasico.py` **declaraba** `pseudocount=0.1` en su `__main__`
   (L4363-4373) pero sus propios `_evaluate_trial` (L3312) y `_apply_best_params` (L4042)
   lo volvían a clavar en `0.0`. **El fork nunca usó 0.1.**

Además, `grep 'config\.pseudocount'` no devuelve **ninguna lectura** en
`MatrizBuilder`/`ClasificadorDescendente`/`_primer_factor` (que dice literalmente
"Sin pseudoconteo: preservar masas marginales reales"). Es decir:

> **ε no está implementado en la CDH.** No es que esté en 0.0 por una decisión de
> calibración: no existe el punto de código donde se sumaría. `pseudocount` sólo
> llegaría al paso AFC como clave de caché.

**Decisión:**

- El modo clásico **usa `pseudocount=0.0`** y lo reporta como `pseudocount=0.0` con la
  bandera `epsilon_status="not_implemented"` en el payload (§4.4). Así el auditor no lee
  "clásico fiel" donde hubo una omisión.
- **El modo moderno no se toca.**
- **El paper debe corregirse** (texto propuesto): _"Pseudocuento ε = 0.1 en la
  especificación original; en esta implementación la CDH usa promediado recíproco sin
  pseudocuento (`ε = 0`), de modo que las masas marginales se preservan exactamente. La
  diferencia se declara como desviación de la réplica A."_
- El diff que **sí** usaría ε queda **condicionado** y **no se incluye activo**: sin un
  punto de consumo en la CDH, `pseudocount=0.1` es un no-op decorativo. No puedo verificar
  ε=0.1 sin ejecutar el pipeline, y no lo he ejecutado.

**Diff condicionado (NO aplicar hasta implementar ε en `_primer_factor`):**

```python
# main_workflow.py — Config
    pseudocount: float = 0.00
    # ε del paper (§2: "Pseudocuento ε 0.1"). Estado real: NO consumido por la CDH
    # (ver docs/design_v5/CLASSIC_MODE.md §3). Sólo el paso AFC lo lee como clave.
    # Se expone para que el modo clásico pueda declararlo sin mentir sobre su efecto.
    classic_epsilon: float = 0.0          # 0.0 = comportamiento observado hoy
    classic_epsilon_status: str = "not_implemented"
```

```python
# main_workflow.py — apply_classic_mode(), si algún día se implementa ε en _primer_factor
    if float(cfg.classic_epsilon) > 0.0:
        cfg.pseudocount = float(cfg.classic_epsilon)
        cfg.classic_epsilon_status = "active"
    else:
        cfg.pseudocount = 0.0
        cfg.classic_epsilon_status = "not_implemented"
```

---

## 4. Aislamiento de caché, config y artefactos

Requisito crítico: si A y B comparten caché, los números del paper se mezclan. Hay **seis**
superficies de contaminación y cada una tiene su mecanismo.

### 4.1 Clave de especificación (`spec_id`)

```python
_CLASSIC_SPEC_FIELDS = (
    "classification_mode", "min_forms_uc", "tsj", "pseudocount",
    "cdh_enforce_min_leaf", "min_cluster_size_cdh", "min_r2_threshold",
    "uce_target_size", "min_uce_words", "stem_backend",
    "use_bigrams", "use_trigrams", "use_embeddings", "random_state",
)

def run_spec_id(config) -> str:
    """Identidad de la especificación.

    Moderno -> "modern" (constante): NO invalida ninguna caché existente.
    Clásico -> "classic:<12 hex>" del subconjunto que define la fidelidad.
    """
    if str(getattr(config, "run_mode", "modern")) != "classic":
        return "modern"
    payload = {k: getattr(config, k, None) for k in _CLASSIC_SPEC_FIELDS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "classic:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
```

Por qué así: el moderno mantiene la clave `"modern"` literal, de modo que **todas** las
cachés actuales (`best_params.json`, `step_cache`, estudio Optuna) siguen siendo válidas. El
clásico obtiene una clave que cambia si alguien toca un parámetro de fidelidad: dos "clásicos"
distintos no se pisan.

### 4.2 Rutas (una sola palanca: `db_local_path` + directorio)

`Optimizador._best_params_path()` ya deriva de `os.path.dirname(config_base.db_local_path)`;
por eso basta mover `db_local_path` para que `best_params.json`, `comparative_results.json`
y el SQLite de Optuna caigan en otro sitio.

| Artefacto                  | Moderno                           | Clásico                                 |
| -------------------------- | --------------------------------- | --------------------------------------- |
| `db_local_path`            | `data/workflow_data.json`         | `data/classic/workflow_data.json`       |
| `best_params.json`         | `data/best_params.json`           | `data/classic/best_params.json`         |
| `comparative_results.json` | `data/comparative_results.json`   | `data/classic/comparative_results.json` |
| Optuna SQLite              | `data/reii_optuna.db`             | `data/classic/reii_optuna.db`           |
| study name                 | `reii_search_v2:<fp>`             | `reii_search_v2:<fp>:<spec_id>`         |
| snapshot selección         | `data/method_selection.json`      | **no se escribe**                       |
| historial selección        | `data/method_selection_log.jsonl` | **no se escribe**                       |
| calibración UCE            | `data/uce_calibration.json`       | no se usa (`auto_calibrate=False`)      |
| checkpoint coref           | `data/coref_seg_checkpoint.json`  | no se usa (sin `coref`)                 |
| dashboard / red / léxico   | `output/*`                        | `output/classic/*`                      |

`data/classic/` y `output/classic/` se crean en `apply_classic_mode()` (o en
`__main__`), nunca en import.

### 4.3 Caché de clasificación (`step_cache`, PostgreSQL)

`steps.py::_STEP_CONFIG_KEYS["classify"]` **no incluía** `hdbscan_uce_min_cluster_frac`,
`_frac_loose`, `cdh_enforce_min_leaf` ni `run_mode`. Consecuencia doble:

- cambiar la fracción de HDBSCAN **no invalidaba** el caché (bug del "cohuete");
- un run clásico podía leer el caché de clasificación del moderno.

Arreglo mínimo en §8: añadir esas claves **como extras condicionales**, de modo que el hash
del moderno no cambie (extras en su valor por defecto ⇒ payload idéntico ⇒ caché válida) y
el clásico (extras ≠ default) obtenga hash distinto.

### 4.4 Trazabilidad en payload y log

`best_params.json` clásico:

```json
{
  "run_mode": "classic",
  "run_spec_id": "classic:9f2c1ab34de5",
  "paper_scenario": "A",
  "classic_overrides": {
    "min_forms_uc": [10, 14],
    "cdh_enforce_min_leaf": false,
    "...": "..."
  },
  "epsilon_status": "not_implemented",
  "pseudocount": 0.0,
  "corpus_fingerprint": "…",
  "params": { "...": "..." }
}
```

Log al arrancar (una vez, no por trial):

```
   [CLI] --mode classic -> ALCESTE 2012 (escenario A)
         spec=classic:9f2c1ab34de5  data_dir=data/classic
         overrides: classification_mode=wc_only min_forms_uc=[10,14]
                    cdh_enforce_min_leaf=False optimize=False
                    use_embeddings=False use_liminal_projection=False
                    auto_calibrate_uce_size=False
         [!] pseudocount=0.0 (paper pide 0.1; epsilon no implementado en la CDH)
```

`ConfigStore.seed_from_config` **se salta** en clásico: sembraría el catálogo de
`workflow_config` con valores clásicos y el dashboard los serviría como "defaults" del
moderno. Un run clásico no debe escribir config compartida.

---

## 5. Estabilidad clásica (CR + ARI 10↔14), sin `select-v4`

ALCESTE no elige método: **mide** si una UCE cae en la misma clase en las dos corridas
(UC = 10 y UC = 14 palabras). Eso ya está calculado en el pipeline como Método A
(`res_wc1`, `res_wc2`, `ari_wc`, `stable_wc`) — no hay que añadir cómputo.

- `primary_stable` en `wc_only` = `stable_wc` (UCEs en la misma clase en A1 y A2).
- **CR** = `|stable_wc| / n_total` (= `coverage_primary`, ya se calcula).
- **ARI** = `pairwise_stability["ari_wc"]` (ya se persiste en `db.data`).
- **K** = nº de clases distintas de `stable_wc`; se reporta junto con
  `n_wc1`, `n_wc2` y el shortfall contra el K objetivo (§7).
- **No** se aplica `select_method`, **no** se calcula coverage útil, **no** se escribe
  `method_selection.json`, **no** hay "ganador".

Salida (consola y `db.data["classic_stability"]`):

```
  ALCESTE 2012 — ESTABILIDAD (UC 10 vs 14)
    UCEs totales      : 5809
    A1 (10) clases    : 7     A2 (14) clases : 8
    Estables (misma cl. en A1 y A2) : 4213  CR = 0.725
    ARI(A1,A2)        : 0.612
    K (partición A1)  : 7     K objetivo = 8  (Kach < target)
```

---

## 6. Encargo 2 — política de `best_params.json` con claves desconocidas

Hoy: si `best_params` trae una clave de un diseño futuro, `_apply_best_params` la ignora en
silencio (usa `.get`) pero `dataclasses.replace` **no** se usa; en cambio un backend que
haga `Config(**params)` o un `ValueError` de un backend inválido aborta la corrida. Política:

1. **Filtrar**: se conserva toda clave aplicable.
2. **Avisar** con `[!]` y registrar las ignoradas en log (`logger.warning`) — nunca silencio.
3. **Fallar sólo si no queda ninguna válida** (archivo de otra versión / corrupto): ahí sí
   `ValueError`, porque arrancar sin parámetros es peor que fallar.
4. Archivo vacío (`{}`) → `{}`, sin error (equivale a "no hay params").

Conjunto "conocido" = campos de `Config` ∪ claves del espacio de búsqueda que no son campos
(`cdh_target_k`, `min_forms_uc_1`, `forms_gap`) ∪ claves que se aplican a `UCBuilderConfig`
(`similarity_threshold`, `coref_weight`, `uc_window_size`). **Esto es crítico**: filtrar sólo
por `Config.__dataclass_fields__` borraría `similarity_threshold`, `coref_weight`,
`uc_window_size`, `min_forms_uc_1`, `forms_gap` y `cdh_target_k`, que son parámetros
aprendidos reales.

---

## 7. Encargo 2 — `Kach < target_k`

**Dónde se calcula `-K`**: `Optimizador._evaluate_trial`, tercer objetivo `-float(n_clusters)`
(L8683 en `select`, L8702 en `wc_only`). `n_clusters` sale de
`len(np.unique([u.cluster_id for u in primary_stable ...]))`.

**El defecto**: en `select` eso es el K de la **partición estable**, no el de la partición
cruda; y como `Kach` puede ser legítimamente menor que el `cdh_target_k` muestreado (el gate
`r2_dense` no fabrica splits, y el piso puede frenar antes), el objetivo podía estar
premiando un K que el trial **no pidió**, sin registrar la diferencia. El sampler cree haber
evaluado K=8 y en realidad evaluó K=5; el frente de Pareto se construye sobre una etiqueta
falsa.

**Corrección** (diff en §9.3): el objetivo usa el **K real de la partición del ganador**
(`sel.winner_k`, o el K crudo de A1 en `wc_only`) y se registran
`k_target`/`k_ach`/`k_shortfall` como `user_attr`. `Kach < target_k` deja de ser un
misterio y pasa a ser un dato auditable. **No** se sustituye por `target_k`: eso sería
falsear el objetivo.

---

## 8. Encargo 2 — allowlists (`config_store.py` y `steps.py`)

### 8.1 `config_store.py` (superficie del dashboard)

`hdbscan_uce_min_cluster_frac` y `_frac_loose` no están en `CONFIG_CATEGORIES`; el dashboard
no puede fijarlas y `set_config_many` las descarta en silencio. Añadir a
`"clasificacion"`: `cdh_enforce_min_leaf`, `hdbscan_uce_min_cluster_frac`,
`hdbscan_uce_min_cluster_frac_loose`, `hdbscan_uce_min_cluster_floor`,
`hdbscan_uce_min_samples_ratio`, `verdict_mode`, `primary_cluster_source`,
`auto_calibrate_uce_size`, `use_liminal_projection`, `min_r2_threshold`(ya está).

**No** añadir `run_mode`: que el dashboard pueda convertir el pipeline en clásico es
exactamente el tipo de accidente que este documento evita.

### 8.2 `steps.py` (caché de clasificación)

Diff en §9.4. **¿Invalida cachés existentes?** Con el diseño de _extras condicionales_, **no**
para el moderno: si los extras valen su default, el payload es byte-idéntico al actual y el
`input_hash` no cambia. Sí cambia (y debe cambiar) cuando alguien mueve una `frac` o corre en
clásico — que es el bug que se arregla. No se bumpea `version` del step: bumpearla invalidaría
**todo** el `step_cache` de clasificación de golpe.

---

## 9. Diffs completos

> Convención: `...` = código existente sin cambios. Todo lo demás es literal.

### 9.1 `src/reii/main_workflow.py` — imports + constantes + helpers

```diff
@@ imports (traceback, hashlib, dataclasses y json YA existen: L9-L18)
 import json
 import os
 import re
 import uuid
@@ from reii.config import (
     BEST_PARAMS_PATH,
+    CLASSIC_DATA_DIR,
+    CLASSIC_OUTPUT_DIR,
     COREF_CHAINS_CACHE_PATH,
@@

+# ══════════════════════════════════════════════════════════════════════
+# MODO CLÁSICO (ALCESTE 2012) — preset + identidad de especificación
+# ══════════════════════════════════════════════════════════════════════
+# Escenario A del paper (docs/design_v5/CLASSIC_MODE.md). Un override, no un
+# fork: el mismo pipeline con la fidelidad de 2012. El fork
+# main_workflow_clasico.py queda retirado (shim, §9.5).
+
+CLASSIC_OVERRIDES: Dict[str, Any] = {
+    "run_mode": "classic",
+    "classification_mode": "wc_only",     # §2 Paso 2: un solo CDH
+    "min_forms_uc": [10, 14],             # §2 Paso 1: UC 10 vs 14 palabras
+    "cdh_enforce_min_leaf": False,        # §2: el clásico no tiene piso de hoja
+    "optimize": False,                    # §2: parámetros fijos, sin búsqueda
+    "use_embeddings": False,              # §2/§3: sin retrofitting ni Method C
+    "use_liminal_projection": False,      # §3 Paso 6: proyección es de REII
+    "auto_calibrate_uce_size": False,     # §2: tamaño de UCE fijo
+}
+
+_CLASSIC_SPEC_FIELDS = (
+    "classification_mode", "min_forms_uc", "tsj", "pseudocount",
+    "cdh_enforce_min_leaf", "min_cluster_size_cdh", "min_r2_threshold",
+    "uce_target_size", "min_uce_words", "stem_backend",
+    "use_bigrams", "use_trigrams", "use_embeddings", "random_state",
+)
+
+
+def run_spec_id(config) -> str:
+    """Identidad de la especificación (§4.1). Moderno => "modern" literal,
+    para no invalidar ninguna caché previa."""
+    if str(getattr(config, "run_mode", "modern")) != "classic":
+        return "modern"
+    payload = {k: getattr(config, k, None) for k in _CLASSIC_SPEC_FIELDS}
+    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
+    return "classic:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
+
+
+def apply_classic_mode(config: Config, uc_config: "UCBuilderConfig"):
+    """Devuelve (Config, UCBuilderConfig) del escenario A. Idempotente.
+
+    No toca `pseudocount`: el paper pide 0.1 pero la CDH no lo consume
+    (docs/design_v5/CLASSIC_MODE.md §3). Queda 0.0 y se declara.
+    """
+    base = os.path.join(CLASSIC_DATA_DIR)
+    os.makedirs(base, exist_ok=True)
+    os.makedirs(CLASSIC_OUTPUT_DIR, exist_ok=True)
+    classic = dataclasses.replace(
+        config,
+        **CLASSIC_OVERRIDES,
+        # ── Aislamiento (§4.2): una sola palanca arrastra best_params,
+        #    comparative_results y el SQLite de Optuna.
+        db_local_path=os.path.join(base, "workflow_data.json"),
+        optimize_storage=(
+            f"sqlite:///{os.path.join(base, 'reii_optuna.db').replace(os.sep, '/')}"
+            if getattr(config, "optimize_storage", None)
+            else None
+        ),
+        optimize_study_name=f"{getattr(config, 'optimize_study_name', 'reii_search')}"
+        f":{run_spec_id(dataclasses.replace(config, **CLASSIC_OVERRIDES))}",
+    )
+    spec = run_spec_id(classic)
+    print("   [CLI] --mode classic -> ALCESTE 2012 (escenario A)")
+    print(f"         spec={spec}  data_dir={CLASSIC_DATA_DIR}")
+    print(
+        "         overrides: classification_mode=wc_only min_forms_uc=[10, 14]\n"
+        "                    cdh_enforce_min_leaf=False optimize=False\n"
+        "                    use_embeddings=False use_liminal_projection=False\n"
+        "                    auto_calibrate_uce_size=False"
+    )
+    print(
+        "         [!] pseudocount=0.0 (paper pide 0.1; epsilon no implementado "
+        "en la CDH)"
+    )
+    return classic, uc_config
+
+
+# ══════════════════════════════════════════════════════════════════════
+# LECTOR DE PARAMS ROBUSTO (claves desconocidas) — §6
+# ══════════════════════════════════════════════════════════════════════
+_NON_CONFIG_PARAM_KEYS = {
+    # espacio de búsqueda / parámetros que se aplican fuera de Config
+    "cdh_target_k",
+    "min_forms_uc_1",
+    "forms_gap",
+    "similarity_threshold",   # UCBuilderConfig
+    "coref_weight",           # UCBuilderConfig
+    "uc_window_size",         # UCBuilderConfig
+}
+
+
+def resolve_best_params(
+    params: Dict[str, Any],
+    *,
+    where: str = "best_params.json",
+) -> Dict[str, Any]:
+    """Filtra claves no aplicables; falla SÓLO si no queda ninguna válida.
+
+    Un archivo con una clave de un diseño futuro debe degradar, no abortar:
+    el resto de los parámetros sigue siendo el resultado del optimizador.
+    """
+    if not params:
+        return {}
+    known = set(Config.__dataclass_fields__) | _NON_CONFIG_PARAM_KEYS
+    valid: Dict[str, Any] = {}
+    ignored: List[str] = []
+    for key, value in params.items():
+        if key in known:
+            valid[key] = value
+        else:
+            ignored.append(str(key))
+    if ignored:
+        print(
+            f"   [!] {where}: {len(ignored)} clave(s) desconocida(s) ignorada(s): "
+            f"{', '.join(sorted(ignored))}"
+        )
+        logger.warning(
+            "%s: claves ignoradas (no aplicables a esta versión de Config): %s",
+            where,
+            sorted(ignored),
+        )
+        for _k in sorted(ignored):
+            print(f"       - {_k} = {params[_k]!r}")
+    if not valid:
+        raise ValueError(
+            f"{where}: ninguna de las {len(params)} claves es aplicable "
+            f"(ignoradas: {', '.join(sorted(ignored))}). "
+            "No se puede continuar sin parámetros."
+        )
+    return valid
+
+
+# ══════════════════════════════════════════════════════════════════════
+# GUARD DE EXCEPCIONES DEL OPTIMIZADOR — §9.2
+# ══════════════════════════════════════════════════════════════════════
+_MAX_TB_CHARS = 4096
+
+
+def guarded_trial_eval(evaluate, trial, failures: List[str]):
+    """Convierte CUALQUIER excepción de un trial en centinela + user_attrs.
+
+    Motivo: en el probe v4 se perdió el trial #81 (FAIL) sin traceback. Con
+    esto el trial queda COMPLETE con valores centinela y el diagnóstico vive
+    en el propio trial (auditable desde Optuna/Postgres).
+    """
+    try:
+        return evaluate(trial)
+    except Exception as exc:  # noqa: BLE001 — el contrato es no propagar
+        tb = traceback.format_exc()
+        failures.append(f"#{trial.number} {type(exc).__name__}: {exc}")
+        del failures[:-50]
+        trial.set_user_attr("trial_failed", True)
+        trial.set_user_attr("exception_type", type(exc).__name__)
+        trial.set_user_attr("exception_msg", str(exc)[:500])
+        trial.set_user_attr("exception_tb", tb[-_MAX_TB_CHARS:])
+        logger.warning(
+            "Trial %d exception (%s): %s", trial.number, type(exc).__name__, exc
+        )
+        return _SENTINEL_TRIAL
```

> `hashlib`, `dataclasses`, `traceback`, `logger` y `_SENTINEL_TRIAL` ya existen en el
> módulo (L9, L6, L18, L11 y L7851).

### 9.2 `main_workflow.py` — Config, CLI y `__main__`

```diff
@@ class Config
     stem_backend: str = "snowball"  # 'snowball' | 'none'
+
+    # Escenario del paper (docs/design_v5/CLASSIC_MODE.md, §1):
+    #   "modern"  = B (ALCESTE + innovaciones)
+    #   "classic" = A (ALCESTE 2012, fidelidad; se activa con --mode classic)
+    # NO está en ConfigStore a propósito: el dashboard no debe poder convertir
+    # una corrida en clásica por accidente.
+    run_mode: str = "modern"

@@ _CLI_MODES_VALIDOS
     "select",
+    "classic",  # preset ALCESTE 2012 (NO es un classification_mode)
     # aliases legacy
@@ __main__ — override CLI de --mode
     _cli_mode = _parse_cli_mode()
-    if _cli_mode is not None:
+    if _cli_mode is not None and _cli_mode != "classic":
         alceste_config = dataclasses.replace(
             alceste_config, classification_mode=_cli_mode
         )
         print(f"   [CLI] classification_mode={_cli_mode} (--mode override)")
@@ __main__ — tras cfg = UCBuilderConfig()
     cfg = UCBuilderConfig()
+    if _cli_mode == "classic":
+        alceste_config, cfg = apply_classic_mode(alceste_config, cfg)

@@ __main__ — rutas de salida aisladas
-        grammatical_dashboard_path=OUTPUT_DASHBOARD,
+        grammatical_dashboard_path=(
+            os.path.join(CLASSIC_OUTPUT_DIR, "global_dashboard.json")
+            if alceste_config.run_mode == "classic"
+            else OUTPUT_DASHBOARD
+        ),
         checkpoint_path=_coref_ckpt_path,
@@
-    lex_analyzer.export_network_gexf(OUTPUT_NETWORK_GEXF)
+    lex_analyzer.export_network_gexf(
+        os.path.join(CLASSIC_OUTPUT_DIR, "red_semantica.gexf")
+        if alceste_config.run_mode == "classic"
+        else OUTPUT_NETWORK_GEXF
+    )
```

> **El bloque de `ConfigStore` en `__main__` no cambia.** El preset clásico se aplica
> _después_ de `ConfigStore` y de `_cli_overrides`, así que ahí sólo se leen valores
> modernos y el override clásico gana en memoria. La guarda que sí hace falta está en
> `WorkflowOrchestrator.__init__`, porque _ese_ `seed_from_config` corre después del
> preset (§9.5).

Además, `_optimizer_will_run` (L10642) ya queda `False` por `optimize=False`, así que
`ProgressiveSegmenter` **no se construye** en clásico y no se paga Stanza.

### 9.3 `main_workflow.py` — `_evaluate_trial`: `Kach`, guard interno, ε

```diff
@@ Optimizador._evaluate_trial
         cfg.min_forms_uc = [mf1, mf2]
         cfg.tsj = params["tsj"]
-        cfg.pseudocount = 0.0  # BLOQUE 1: nunca sumar ε
+        # BLOQUE 1: nunca sumar ε. En clásico se conserva el valor declarado
+        # para poder REPORTARLO (sigue siendo 0.0 mientras ε no exista en la
+        # CDH — ver CLASSIC_MODE.md §3). En moderno, invariante 0.0.
+        if str(getattr(cfg, "run_mode", "modern")) != "classic":
+            cfg.pseudocount = 0.0
         cfg.min_cluster_size_cdh = _cdh_frac_from_params(
@@
         try:
             result = self._evaluacion_rapida(cfg, uc_cfg=uc_cfg)
         except Exception as e:
-            logger.debug("Trial %d exception: %s", trial.number, e)
+            # No perder el trial en silencio (probe v4, trial #81).
+            _tb = traceback.format_exc()
+            trial.set_user_attr("trial_failed", True)
+            trial.set_user_attr("exception_type", type(e).__name__)
+            trial.set_user_attr("exception_msg", str(e)[:500])
+            trial.set_user_attr("exception_tb", _tb[-_MAX_TB_CHARS:])
+            logger.warning(
+                "Trial %d exception en _evaluacion_rapida (%s): %s",
+                trial.number, type(e).__name__, e,
+            )
             return _SENTINEL_TRIAL
@@ rama select: K real, no target_k
         if mode == "select":
             sel = self._last_selection
@@
-            return (
-                float(coverage_primary),
-                float(sel.winner_ari),
-                -float(n_clusters),
-            )
+            # ── Kach vs target_k (CLASSIC_MODE.md §7) ────────────────────
+            # El objetivo −K debe ser el K REAL de la partición del ganador.
+            # `n_clusters` cuenta sólo UCEs estables: un trial puede haber
+            # pedido K=8 y producir 5 hojas. Se reporta la brecha en vez de
+            # fingir que se alcanzó el objetivo del sampler.
+            _k_ach = int(getattr(sel, "winner_k", 0) or n_clusters)
+            _k_target = params.get("cdh_target_k")
+            trial.set_user_attr("k_ach", _k_ach)
+            trial.set_user_attr(
+                "k_target", int(_k_target) if _k_target is not None else -1
+            )
+            trial.set_user_attr(
+                "k_shortfall",
+                int(_k_target) - _k_ach if _k_target is not None else 0,
+            )
+            trial.set_user_attr("k_stable", int(n_clusters))
+            return (
+                float(coverage_primary),
+                float(sel.winner_ari),
+                -float(_k_ach),
+            )
@@ rama wc_only
             ari = float(adjusted_rand_score(arr_a[valid], arr_b[valid]))
-            return ari, float(coverage_primary), -float(n_clusters)
+            # K de la partición CRUDA de A1 (no el de la estable).
+            _k_ach = int(len(np.unique(arr_a[valid])))
+            _k_target = params.get("cdh_target_k")
+            trial.set_user_attr("k_ach", _k_ach)
+            trial.set_user_attr(
+                "k_target", int(_k_target) if _k_target is not None else -1
+            )
+            trial.set_user_attr(
+                "k_shortfall",
+                int(_k_target) - _k_ach if _k_target is not None else 0,
+            )
+            trial.set_user_attr("k_stable", int(n_clusters))
+            return ari, float(coverage_primary), -float(_k_ach)
```

### 9.4 `main_workflow.py` — `study.optimize` (guard) y lector de params

```diff
@@ _run_single_stage, antes de study.optimize
+        _trial_failures: List[str] = []
         study.optimize(
-            self._evaluate_trial,
+            lambda t: guarded_trial_eval(
+                self._evaluate_trial, trial=t, failures=_trial_failures
+            ),
             n_trials=n_trials,
             show_progress_bar=False,
         )
@@ resumen post-optimize
         print(
             f"\n   >>> Estado: {n_completed} completados, "
             f"{n_pruned} pruned, {n_failed} failed"
         )
+        if _trial_failures:
+            print(
+                f"   [!] {len(_trial_failures)} trial(s) con excepción -> centinela "
+                "(user_attrs exception_type/exception_tb)"
+            )
+            for _f in _trial_failures[-3:]:
+                print(f"       {_f}")
@@ Optimizador.load_best_params
         params = data.get("best_params") or data.get("params")
         if params:
+            params = resolve_best_params(params, where=os.path.basename(path))
             print(f"   [Optimizador] Loaded params from {path}: {params}")
         return params
@@ WorkflowOrchestrator.ejecutar — lectura directa del JSON
             best_data = json.load(f)
+            if str(getattr(self.config, "run_mode", "modern")) == "classic":
+                raise RuntimeError(
+                    "classic: no se lee best_params.json del modo moderno "
+                    "(aislamiento de artefactos, CLASSIC_MODE.md §4)"
+                )
             _stored_fp = best_data.get("corpus_fingerprint")
@@ _apply_best_params (defensivo: también si alguien llama el método directo)
             """Aplica los hiperparámetros ganadores al Config activo."""
+            bp = resolve_best_params(bp, where="best_params (apply)")
             mf1 = bp.get("min_forms_uc_1", self.config.min_forms_uc[0])
             gap = bp.get("forms_gap", self.config.min_forms_uc[1] - mf1)
             self.config.min_forms_uc = [mf1, mf1 + gap]
             self.config.tsj = bp.get("tsj", self.config.tsj)
-            self.config.pseudocount = 0.0  # BLOQUE 1: invariante
+            if str(getattr(self.config, "run_mode", "modern")) != "classic":
+                self.config.pseudocount = 0.0  # BLOQUE 1: invariante
```

### 9.5 `main_workflow.py` — seed de ConfigStore, retrofitting y estabilidad clásica

```diff
@@ WorkflowOrchestrator.__init__ — seed_from_config
-        try:
-            ConfigStore(config.db_dsn).seed_from_config(config)
-        except Exception as e:
-            print(f"   [WARNING] ConfigStore.seed_from_config falló: {e}")
+        if str(getattr(config, "run_mode", "modern")) != "classic":
+            try:
+                ConfigStore(config.db_dsn).seed_from_config(config)
+            except Exception as e:
+                print(f"   [WARNING] ConfigStore.seed_from_config falló: {e}")
+        else:
+            # Sembraría workflow_config con valores clásicos, que el dashboard
+            # serviría como "defaults" del moderno (CLASSIC_MODE.md §4.4).
+            print("   [i] classic: no se siembra workflow_config (aislamiento)")
```

```diff
@@ ejecutar — pass 1 (L9925)
-        retro_vecs_p1: Optional[Dict[str, np.ndarray]] = (
-            self._compute_retrofitted_vectors(resultados_p1, voc_p1)
-        )
+        if self.config.use_embeddings:
+            retro_vecs_p1: Optional[Dict[str, np.ndarray]] = (
+                self._compute_retrofitted_vectors(resultados_p1, voc_p1)
+            )
+        else:
+            # Escenario A: sin retrofitting (no hay espacio denso que poblar).
+            retro_vecs_p1 = {}
+            print("   [i] use_embeddings=False: sin retrofitting (modo clasico).")
```

> Verificar la indentación y el `try/except` reales en L9925 al aplicar: el diff asume
> la forma citada en §9 del documento.

```diff
@@ DoubleClassifier.persist_method_selection
     def persist_method_selection(self, selection: Selection, n_total: int) -> None:
+        if str(getattr(self.config, "run_mode", "modern")) == "classic":
+            print("   [i] classic: sin method_selection.json (no hay elección de método).")
+            return
         target = int(getattr(self.config, "uce_target_size", 0) or 0)
@@ WorkflowOrchestrator.ejecutar — junto al bloque de selección (L9952)
             _sel = getattr(self.double_clf, "last_selection", None)
             if _sel is not None:
                 self.db.data["method_selection"] = _sel.to_dict()
                 self.double_clf.persist_method_selection(_sel, len(uces_est_list))
+
+            # ── Estabilidad ALCESTE clásica (CR + ARI 10↔14) — §5 ────────
+            if str(getattr(self.config, "run_mode", "modern")) == "classic":
+                _pairwise = getattr(
+                    self.double_clf, "last_pairwise_stability", {}
+                ) or {}
+                _cr = self._classic_stability(uces_est_list, _pairwise)
+                self.db.data["classic_stability"] = _cr
+                print(_classic_stability_text(_cr))
@@ WorkflowOrchestrator — métodos nuevos
+    def _classic_stability(self, uces_por_doc, pairwise) -> Dict[str, Any]:
+        """CR y ARI de las dos corridas (UC 10 vs UC 14). Sin select-v4."""
+        total = sum(len(d) for d in uces_por_doc)
+        labels = [
+            u.cluster_id
+            for d in uces_por_doc
+            for u in d
+            if getattr(u, "cluster_id", None) is not None
+        ]
+        k = len(set(labels))
+        sizes = Counter(labels)
+        return {
+            "scenario": "A",
+            "run_mode": "classic",
+            "spec_id": run_spec_id(self.config),
+            "n_uces": int(total),
+            "n_stable": int(len(labels)),
+            "cr": float(len(labels) / max(1, total)),
+            "ari_a1_a2": float(pairwise.get("ari_wc", float("nan"))),
+            "k_partition": int(k),
+            "min_class_size": int(min(sizes.values())) if sizes else 0,
+            "max_class_share": float(max(sizes.values()) / max(1, len(labels)))
+            if sizes else 0.0,
+            "pseudocount": float(getattr(self.config, "pseudocount", 0.0)),
+            "epsilon_status": "not_implemented",
+        }
+
+
+def _classic_stability_text(rep: Dict[str, Any]) -> str:
+    return (
+        "\n  ALCESTE 2012 - ESTABILIDAD (UC 10 vs 14)\n"
+        f"    UCEs totales       : {rep['n_uces']}\n"
+        f"    Estables (misma clase en A1 y A2) : {rep['n_stable']}"
+        f"  CR = {rep['cr']:.3f}\n"
+        f"    ARI(A1,A2)         : {rep['ari_a1_a2']:.3f}\n"
+        f"    K (particion)      : {rep['k_partition']}"
+        f"  clase min={rep['min_class_size']}"
+        f"  mayor={rep['max_class_share']:.2f}\n"
+        "    [!] pseudocount=0.0 (paper: 0.1; epsilon no implementado en la CDH)\n"
+    )
```

> `Counter`, `Dict`, `Any` ya están importados. `ari_wc` existe en
> `last_pairwise_stability` (L6270) y ya se persiste en `db.data["pairwise_stability"]`.

### 9.6 `src/reii/config.py`

```diff
@@ DATABASE / persistence
 BEST_PARAMS_PATH: str = str(
     ROOT_DIR / os.environ.get("REII_BEST_PARAMS_PATH", "data/best_params.json")
 )
+# Modo clásico (ALCESTE 2012): artefactos en un directorio propio para que
+# A y B no compartan caché ni se pisen los resultados del paper.
+# Ver docs/design_v5/CLASSIC_MODE.md §4.
+CLASSIC_DATA_DIR: str = str(
+    ROOT_DIR / os.environ.get("REII_CLASSIC_DATA_DIR", "data/classic")
+)
```

```diff
@@ OUTPUT file names
+CLASSIC_OUTPUT_DIR: str = str(
+    OUTPUT_DIR / os.environ.get("REII_CLASSIC_OUTPUT_DIR", "classic")
+)
 OUTPUT_DASHBOARD: str = str(OUTPUT_DIR / "global_dashboard.json")
```

### 9.7 `src/reii/backend/config_store.py`

```diff
@@ "clasificacion"
         "min_cluster_size_cdh",
+        # Fidelidad ALCESTE 2012 (CLASSIC_MODE.md §2/§8.1). `run_mode` NO se
+        # expone: el dashboard no debe convertir una corrida en clásica.
+        "cdh_enforce_min_leaf",
         "clustering_method",
@@
         "hdbscan_uce_min_cluster_size",
         "hdbscan_uce_min_cluster_size_loose",
+        "hdbscan_uce_min_cluster_frac",
+        "hdbscan_uce_min_cluster_frac_loose",
+        "hdbscan_uce_min_cluster_floor",
+        "hdbscan_uce_min_samples_ratio",
         "hdbscan_uce_epsilon",
         "hdbscan_uce_min_samples",
         "hdbscan_uce_metric",
+        "verdict_mode",
+        "primary_cluster_source",
@@ "afc_y_proyeccion"
         "use_liminal_projection",
         "liminal_projection_min_ratio",
+        "auto_calibrate_uce_size",
```

### 9.8 `src/reii/backend/steps.py`

```diff
@@ _STEP_CONFIG_KEYS
     "classify": [
         "min_forms_uc",
         "tsj",
         "min_cluster_size_cdh",
         "swap_iterations",
         "min_r2_threshold",
         "n_perm_cdh",
         "classification_mode",
     ],
@@
 def _config_subset(config: Any, step_name: str) -> Dict[str, Any]:
     keys = _STEP_CONFIG_KEYS.get(step_name)
     if keys is None:
         ...
     return {k: getattr(config, k, None) for k in keys}
```

Reemplazo del bloque final por (extras condicionales):

```python
# Claves que deben invalidar el caché pero que NO se hashean cuando valen su
# default: así los hashes ya persistidos (que se calcularon sin ellas) siguen
# siendo válidos y no se invalida el step_cache entero. `run_mode` va aquí para
# que un run clásico nunca lea el caché del moderno.
_STEP_CONFIG_EXTRA_KEYS: Dict[str, List[str]] = {
    "classify": [
        "hdbscan_uce_min_cluster_frac",
        "hdbscan_uce_min_cluster_frac_loose",
        "hdbscan_uce_epsilon",
        "hdbscan_uce_min_samples",
        "hdbscan_uce_metric",
        "cdh_enforce_min_leaf",
        "coref_context_units_tight",
        "coref_context_units_loose",
        "verdict_mode",
        "run_mode",
    ],
}

# Valores de referencia = defaults de Config en el momento de introducir el
# extra. Si coinciden, el extra no entra en el payload (compatibilidad).
_STEP_EXTRA_DEFAULTS: Dict[str, Any] = {
    "hdbscan_uce_min_cluster_frac": 0.08,
    "hdbscan_uce_min_cluster_frac_loose": 0.05,
    "hdbscan_uce_epsilon": 0.0,
    "hdbscan_uce_min_samples": None,
    "hdbscan_uce_metric": "euclidean",
    "cdh_enforce_min_leaf": True,
    "coref_context_units_tight": 2,
    "coref_context_units_loose": 4,
    "verdict_mode": "all",
    "run_mode": "modern",
}


def _config_subset(config: Any, step_name: str) -> Dict[str, Any]:
    keys = _STEP_CONFIG_KEYS.get(step_name)
    if keys is None:
        try:
            return {k: getattr(config, k) for k in vars(config)}
        except TypeError:
            return {"_repr": repr(config)}
    subset = {k: getattr(config, k, None) for k in keys}
    extras = {
        k: getattr(config, k, None)
        for k in _STEP_CONFIG_EXTRA_KEYS.get(step_name, [])
        if getattr(config, k, None) != _STEP_EXTRA_DEFAULTS.get(k)
    }
    if extras:
        subset["_extras"] = extras
    return subset
```

**Invalidación**: ninguna para el moderno con defaults (payload idéntico). Sí invalida,
correctamente, para quien haya movido `hdbscan_uce_min_cluster_frac` — hasta hoy ese cambio
reutilizaba caché con resultados equivocados, así que ese "fallo de caché" es el bug fix.

### 9.9 Retiro del fork: `src/reii/main_workflow_clasico.py` (shim completo)

**Opción elegida: shim que redirige**, no borrar. Justificación: (a) `pipelines/transcripciones.py:151`
lo invoca por ruta de archivo y otros scripts/notebooks pueden hacer lo mismo — borrarlo rompe
con un `FileNotFoundError` opaco; (b) un shim de 30 líneas no puede divergir del pipeline,
que es justo el fallo del fork de 4.519 líneas; (c) el shim avisa `[!]` y arrastra el argv,
así que `--trials`, `--force`, etc. siguen funcionando.

```python
#!/usr/bin/env python3
"""DEPRECADO — shim de compatibilidad de ``main_workflow_clasico.py``.

Este archivo era un fork de 4.519 líneas que quedó sin el fix de piso del CDH
y sin los backends densos: sus resultados ya no son comparables con los del
pipeline. ALCESTE 2012 es ahora una CONFIGURACIÓN del pipeline único:

    python src/reii/main_workflow.py --mode classic

Ver docs/design_v5/CLASSIC_MODE.md. Este shim sólo redirige; no contiene
lógica propia. Se eliminará cuando ningún llamador externo use la ruta vieja.
"""

from __future__ import annotations

import os
import subprocess
import sys

_TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_workflow.py")


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    print(
        "[!] main_workflow_clasico.py esta RETIRADO (fork sin mantenimiento).\n"
        "    Redirigiendo a: main_workflow.py --mode classic\n"
        "    Ver docs/design_v5/CLASSIC_MODE.md"
    )
    # El --mode del llamador viejo se descarta: aquí el modo ES classic.
    if "--mode" in args:
        _i = args.index("--mode")
        del args[_i : _i + 2]
    cmd = [sys.executable, _TARGET, "--mode", "classic", *args]
    print(f"    -> {' '.join(cmd)}")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
```

### 9.10 `src/reii/pipelines/transcripciones.py`

```diff
     mode = params.get("workflow_mode", "classic")
-    if mode == "transformer":
-        script = os.path.join(_REII_DIR, "main_workflow.py")
-    else:
-        script = os.path.join(_REII_DIR, "main_workflow_clasico.py")
+    # Escenario A = configuración del pipeline único (CLASSIC_MODE.md §9.9).
+    script = os.path.join(_REII_DIR, "main_workflow.py")
+    extra_args = ["--mode", "classic"] if mode != "transformer" else []
     log.append(f"🚀 Ejecutando workflow: python {script}")
     try:
         proc = subprocess.Popen(
-            [sys.executable, script],
+            [sys.executable, script, *extra_args],
```

---

## 10. Tests a añadir

### 10.1 `tests/test_classic_mode.py`

```python
"""Modo clásico: preset exacto + política de params + guard + spec id."""
import dataclasses

import pytest

from reii.main_workflow import (
    CLASSIC_OVERRIDES,
    Config,
    _SENTINEL_TRIAL,
    apply_classic_mode,
    guarded_trial_eval,
    resolve_best_params,
    run_spec_id,
)
from reii.main_workflow import UCBuilderConfig


def test_classic_diff_is_exactly_the_table():
    modern = Config()
    uc = UCBuilderConfig()
    classic, uc2 = apply_classic_mode(modern, uc)
    changed = {
        f.name
        for f in dataclasses.fields(Config)
        if getattr(modern, f.name) != getattr(classic, f.name)
    }
    assert changed == set(CLASSIC_OVERRIDES) | {
        "db_local_path",
        "optimize_storage",
        "optimize_study_name",
    }
    # el UCBuilderConfig no se toca
    assert uc2 == uc


def test_classic_keeps_epsilon_declared_not_invented():
    classic, _ = apply_classic_mode(Config(), UCBuilderConfig())
    assert classic.pseudocount == 0.0


def test_spec_ids_differ():
    classic, _ = apply_classic_mode(Config(), UCBuilderConfig())
    assert run_spec_id(Config()) == "modern"
    assert run_spec_id(classic).startswith("classic:")
    assert run_spec_id(classic) != run_spec_id(Config())


def test_spec_id_changes_with_fidelity_param():
    classic, _ = apply_classic_mode(Config(), UCBuilderConfig())
    other = dataclasses.replace(classic, min_forms_uc=[11, 15])
    assert run_spec_id(classic) != run_spec_id(other)


@pytest.mark.parametrize(
    "params, expected, raises",
    [
        ({"tsj": 4, "cdh_target_k": 8}, {"tsj": 4, "cdh_target_k": 8}, False),
        ({"tsj": 4, "bogus": 1}, {"tsj": 4}, False),
        ({"bogus": 1, "otra": 2}, None, True),
        ({"min_cluster_size_cdh": 0.06, "futuro": 3}, {"min_cluster_size_cdh": 0.06}, False),
    ],
)
def test_resolve_best_params(params, expected, raises, capsys):
    if raises:
        with pytest.raises(ValueError):
            resolve_best_params(params, where="test")
    else:
        assert resolve_best_params(params, where="test") == expected
    out = capsys.readouterr().out
    if any(k not in Config.__dataclass_fields__ for k in params if k != "cdh_target_k"):
        assert "[!]" in out


def test_resolve_keeps_uc_builder_params():
    # These are real learned params, not Config fields: must NOT be filtered.
    p = {"similarity_threshold": 0.35, "coref_weight": 0.1, "uc_window_size": 3}
    assert resolve_best_params(p) == p


class _Trial:
    number = 81

    def __init__(self):
        self.attrs = {}

    def set_user_attr(self, k, v):
        self.attrs[k] = v


def test_guard_returns_sentinel_and_records_traceback():
    def evaluate(_trial):
        raise RuntimeError("svd no converge")

    trial, failures = _Trial(), []
    out = guarded_trial_eval(evaluate, trial=trial, failures=failures)

    assert out == _SENTINEL_TRIAL
    assert trial.attrs["trial_failed"] is True
    assert trial.attrs["exception_type"] == "RuntimeError"
    assert "RuntimeError" in trial.attrs["exception_tb"]
    assert len(trial.attrs["exception_tb"]) <= 4096
    assert failures == ["#81 RuntimeError: svd no converge"]


def test_guard_passes_happy_path_through():
    trial = _Trial()
    assert guarded_trial_eval(lambda t: (0.5, 0.6, -7.0), trial, []) == (0.5, 0.6, -7.0)
    assert trial.attrs == {}
```

---

## 11. Resultado de la validación ejecutada

`python scripts/playground/classic_mode_probe.py` → **`[ok] todos los asserts pasaron`**
(exit 0, ~12 s incluyendo arranque de Python y parseo AST). Qué se comprobó de verdad:

| Comprobación                                                                                       | Resultado                                                 |
| -------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `Config` real extraída por AST (136 campos, sin importar `main_workflow`)                          | `[ok]`                                                    |
| Defaults modernos = tabla del doc                                                                  | `[ok]`                                                    |
| Diferencia clásico↔moderno **exactamente** la tabla (6 campos de valor)                            | `[ok]`                                                    |
| `classification_mode` no-op en el dataclass + default operativo `select` en `__main__`             | `[ok]`                                                    |
| `run_mode`, `pseudocount=0.0`                                                                      | `[ok]`                                                    |
| 4+1 fixtures de params: todo válido / una basura / todas basura / válida+basura / params UCBuilder | `[ok]`                                                    |
| `spec_id` clásico ≠ moderno, y cambia con un parámetro de fidelidad                                | `[ok]` (moderno="modern", clásico="classic:71d26b4a3540") |
| Guard: excepción → centinela + 4 user_attrs + traceback ≤4 KB + contador; camino feliz intacto     | `[ok]`                                                    |

Lo que **no** valida (y no puede): el comportamiento de las particiones. Corre A y B sobre el
mismo corpus antes de escribir cifras en el paper.

---

## 12. Riesgos abiertos (los tres que quedaron)

1. **ε = 0.1 no verificable sin ejecutar el pipeline.** No hay punto de consumo en la CDH:
   poner 0.1 no cambia ningún número y afirmar fidelidad sería falso. Decisión: declarar la
   desviación en el paper (§3) y dejar el diff condicionado.
2. **`cdh_enforce_min_leaf=False` cambia K y la estabilidad de forma no monótona.** Sin piso
   de hoja, los cortes 1-vs-(n−1) pueden disparar K; el `Kach` clásico puede superar el
   `target_k` (el caso inverso al de §7). El probe valida la _configuración_, no el
   comportamiento: hay que correr A y B sobre el mismo corpus antes de escribir la tabla
   comparativa del paper.
3. **Aislamiento de `output/` es parcial si algún consumidor importa `OUTPUT_DASHBOARD`
   directamente** (constante de módulo). En este diseño sólo se redirigen las rutas que
   `__main__` pasa explícitamente; `OUTPUT_LEXICAL_JSON/XLSX` siguen apuntando al directorio
   moderno. Añadirlos al mismo `if` antes de publicar cifras del escenario A.
