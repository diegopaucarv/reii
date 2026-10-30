# Experimento de sensibilidad: `forms_gap` 2 vs 4

Estado: **pre-registrado, no ejecutado**. Lo cita el patch 19c
(`_BOUNDARY_DECLINED["forms_gap"]`), que declara el piso `gap=2` en vez de
ampliarlo. Este documento fija cómo se mide esa decisión para que no se decida
"a ojo" después.

## 1. Qué controla `forms_gap` (y qué NO)

```
mf1 = min_forms_uc_1        # piso de la pasada 1
mf2 = mf1 + forms_gap       # piso de la pasada 2   (_evaluate_trial)
```

`min_forms_uc` es el mínimo de **formas analizadas distintas** para agrupar
UCEs consecutivas en una UC (`construir_ucs`). Por tanto `gap` mueve:

- las fronteras de las UC de la pasada 2 (y con ellas la partición),
- la disimilitud entre las dos pasadas (y con ella `ARI(2 pasadas)`).

`gap` **no** mueve:

- la segmentación en UCEs (`uce_target_size`, anterior e independiente),
- **la selección de formas activas**, que la decide `retention_policy`
  (patch 11/14) a nivel de token.

> Consecuencia que hay que declarar en el paper: **la retención es invariante a
> `forms_gap`**. Las cifras 12,24 lemas/UCE · 0,451 de retención
> (`tool_adverb_filtered`) frente a 13,95 · 0,537 de ALCESTE no se arreglan
> subiendo `gap` a 4. Ese desvío pertenece a `RETENTION_POLICY.md`, no a este
> experimento.

## 2. La desviación concreta

| | pasada 1 | pasada 2 |
| --- | --- | --- |
| ALCESTE (referencia) | 10 | 14 |
| default del código | 13 | 17 |
| óptimo v6 | 10 | 12 |

`gap=2` es el mínimo que mantiene las dos pasadas distintas; con `gap<2` las
particiones se acercan y el `ARI(2 pasadas)` sube **por construcción** (misma
familia de artefacto que `swap_iterations=0`, patch 19). Por eso el piso se
declara en `_BOUNDARY_DECLINED` y no se amplía a 1.

Lo que este experimento decide es distinto: si **10/12** (óptimo) y **10/14**
(ALCESTE) producen o no la misma partición. Si la producen, el óptimo de v6 no
es una desviación sustantiva y se adopta la paridad con ALCESTE. Si no la
producen, la desviación existe, se mide y se declara.

## 3. Diseño (una sola corrida)

- **Condición A (baseline, ya medida):** `best_params.json` de v6,
  `forms_gap=2`, `min_forms_uc_1=10` → `score=0.6212121212121212`,
  K=4, coverage 0,6212, ARI 0,3435.
- **Condición B:** el mismo `best_params.json` con `forms_gap=4` y todo lo
  demás congelado (`min_forms_uc_1=10`, `cdh_target_k=3`, `swap_iterations=5`,
  `min_r2_threshold=0.05`, `similarity_threshold=0.45`, `coref_weight=0.1`,
  `coref_context_units_tight=2`, `coref_context_units_loose=6`, `tsj=3`,
  `hdbscan_uce_min_cluster_frac=0.017283630049070842`,
  `hdbscan_uce_min_cluster_frac_loose=0.01`, `min_cluster_size_cdh=0.1667`).

Arnés: el probe existente, que muta `best_params.json`, corre el pipeline,
restaura el archivo en `finally` y deja el log. **Ojo:** el probe aplica un
`PRESET` (pensado para la sonda B) que hay que neutralizar explícitamente, o la
comparación queda confundida con otras 4 variables:

```powershell
python 8_probe_b_permissive.py `
  --param forms_gap=4 `
  --param swap_iterations=5 `
  --param min_cluster_size_cdh=0.1667 `
  --param coref_weight=0.1 `
  --param similarity_threshold=0.45 `
  --log data/diag/gap4.log --out data/diag/gap4.json
```

Los cuatro `--param` de "reset" repiten el valor persistido (el `PRESET` pone
`0.05 / 0.4 / 0.25`), así que la única variable es `forms_gap`.

Antes de gastar las ~2 h, verificar con `--dry-run` que los 5 overrides impresos
son los esperados.

## 4. Endpoints y regla de decisión (fijados antes de medir)

**Primario — acuerdo de partición a nivel UCE.**
`ARI(clase modal de cada UCE en A, clase modal de cada UCE en B)`, con la clase
tomada de `predicate_frames[].cluster_id` en `output/global_dashboard.json`
(una fila por frame; la clase de la UCE es la moda). Es comparable entre
condiciones porque las UCEs son **las mismas** en A y B (solo cambian las UC).

**Secundarios (se reportan, no deciden):**

- `K`, tamaño de clase mínima/mediana/mayor,
- `coverage` / `stable%` del método elegido por `select`,
- `ARI(2 pasadas)` por método (A/B/C) — **se reporta pero no se usa para
  decidir**: depende de `gap` por construcción (§1),
- retención y lemas/UCE — **constante por diseño** (§1); se incluye solo para
  demostrar que no cambió.

**Regla pre-registrada:**

> Se adopta `gap=4` (paridad ALCESTE) si y solo si
> (a) `ARI_UCE(A,B) ≥ 0.90`, y
> (b) `|K_B − K_A| ≤ 2`.
> Si no se cumple (a) o (b), se mantiene `gap=2` y el paper declara la
> desviación 10/12 frente a 10/14 **con los números medidos** (ARI_UCE y K).

Notas sobre los umbrales, que son convenciones y así se declaran en el paper:

- `0.90` no sale de la literatura: se adopta como "prácticamente la misma
  partición" (≈10 % de UCEs cambiando de clase). No es un umbral de validez.
- `±2` en K: el corpus produce K entre 3 y 20; dos clases de diferencia sobre un
  K del orden de 4–16 es el orden de magnitud de la variación entre semillas.
- No se usa un test de significancia: hay **una** corrida por condición. La
  regla es de equivalencia práctica, no inferencial, y se reporta como tal.

## 5. Qué NO prueba este experimento

- No prueba que `gap=2` sea mejor o peor que `gap=4` en general: mide **una**
  configuración del espacio (la del ganador v6).
- No sustituye a la comparación con ALCESTE sobre retención
  (`RETENTION_POLICY.md`), que es donde vive el desvío 0,451 vs 0,537.
- No valida el `ARI(2 pasadas)` como métrica de calidad: solo lo reporta.

## 6. Salida esperada

`data/diag/gap4.json` (del probe) + el bloque `STABILITY REPORT` de
`data/diag/gap4.log` + `output/global_dashboard.json` de la corrida B (copiarlo
antes de la siguiente corrida, porque `output/` se sobrescribe).

Con eso se rellena:

| magnitud | A (gap=2) | B (gap=4) |
| --- | --- | --- |
| K | 4 | |
| coverage | 0,6212 | |
| ARI ganador | 0,3435 | |
| `ARI_UCE(A,B)` | — | |
| retención | 0,451 | (debe salir 0,451) |
