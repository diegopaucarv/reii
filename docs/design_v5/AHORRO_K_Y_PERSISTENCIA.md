# Ahorro de K y persistencia A/B/C

Pre-registro. Escrito y fechado **antes** de relanzar el optimizador.
Corpus de referencia de todas las cifras: **n = 528 UCEs**.

Este documento declara tres cambios. No los ejecuta. Ningún resultado se
promete aquí: se fija el significado de `best_params.json` que se espera
obtener después.

## 1. Resumen ejecutivo

Hoy el tercer objetivo es `-K`, y eso paga por partir en clases diminutas. Se
cambia a `-K_util`, que solo cuenta clases con masa mínima. Además, el óptimo
de `cdh_target_k` cae en el piso del espacio (3) en v5 y v6; se pre-registra
una caída única al piso 2, con estudio nuevo y adopción estricta. Por último,
las particiones de A y B se calculan y se tiran: se persisten las tres y el
dashboard gana un selector. Los tres cambios invalidan los estudios previos.

## 2. Estado medido antes del cambio

Todo lo de esta tabla viene de la última corrida sobre el corpus actual
(n = 528). No son estimaciones.

| artefacto                    | dato                                            | valor                                                   |
| ---------------------------- | ----------------------------------------------- | ------------------------------------------------------- |
| corpus                       | UCEs totales                                    | 528                                                     |
| `data/best_params.json`      | `cdh_target_k` vigente                          | 3                                                       |
| `data/best_params.json`      | `min_cluster_size_cdh` vigente                  | 0.1667                                                  |
| `data/best_params.json`      | `best_values`                                   | `[0.7879, 0.5315, -4.0]`                                |
| `data/best_params.json`      | K real del ganador (`-best_values[2]`)          | 4, método A                                             |
| `data/best_params.json`      | `objectives` / `directions`                     | `[coverage_primary, ari_winner, neg_k]` / max, max, min |
| `data/method_selection.json` | ganador                                         | C, K = 3                                                |
| `data/method_selection.json` | C: ARI / bootstrap ARI [IC] / coverage          | 1.000 / 0.926 [0.893; 0.958] / 1.000                    |
| `data/method_selection.json` | A: K / ARI / bootstrap ARI [IC] / coverage      | 4 / 0.489 / 0.496 [0.405; 0.601] / 0.763                |
| `data/method_selection.json` | B: K / ARI / bootstrap ARI [IC] / coverage      | 5 / 0.288 / 0.101 [-0.005; 0.222] / 0.525               |
| `data/workflow_data.json`    | partición persistida (`cluster_id`), K = 3      | 244, 207, 52 UCEs; 25 sin clase                         |
| espacio de búsqueda          | `cdh_target_k`                                  | `[3, 12]`                                               |
| espacio de búsqueda          | techo `_CDH_TARGET_K_HI`                        | 12                                                      |
| espacio de búsqueda          | recorte adicional                               | `n // 10` (con n = 528, 52)                             |
| espacio de búsqueda          | `min_cluster_size_cdh`                          | no es libre: `1 / (2K)` en `_cdh_frac_from_params`      |
| regla select                 | `k_bounds`                                      | `(3, 20)`                                               |
| optimizador                  | gate duro sobre K real                          | `K` fuera de `[2, 20]` produce trial centinela          |
| regla select                 | `min_class_share`                               | 0.01                                                    |
| regla select                 | `min_class_size = max(2, int(n_total * share))` | 5 UCEs con n = 528                                      |

Nota sobre los dos artefactos. `data/best_params.json` es el resultado del
optimizador y `data/method_selection.json` es el veredicto de la regla select.
Son objetos distintos y no se afirma que describan la misma partición. La
partición persistida en `data/workflow_data.json` es la del último pipeline
ejecutado, y es la que hoy sirve `cluster_id`.

Nota sobre el umbral de "clase útil". La definición pre-registrada en §3 usa
`ceil`, que da 6 con n = 528. El campo existente `min_class_size` de la regla
usa `int()`, que da 5. La diferencia es de una UCE y **no afecta a ninguna
partición medida**, porque la clase menor de A, B y C supera ambas cotas. La
implementación debe usar un solo valor para `K_util` y para `tiny_frac`, y este
pre-registro fija `ceil`. Se deja escrito para que nadie lo descubra después.

## 3. Cambio 1: tercer objetivo de `-K` a `-K_util`

### 3.1 Definición operativa

```
umbral_util = max(2, ceil(min_class_share * n_total))

K_util = número de clases de la partición del ganador
         cuyo tamaño es >= umbral_util
```

Las clases por debajo de ese piso no cuentan. Son polvo. Con `min_class_share`
por defecto (0.01) y n = 528, `umbral_util = 6`.

El tercer objetivo pasa a ser `-K_util`. En `data/best_params.json`, el campo
`objectives` pasa de `neg_k` a `neg_k_util`. Ese nombre es la marca legible por
máquina que distingue un artefacto nuevo de uno viejo.

### 3.2 Motivación

Minimizar `-K` es premiar más clases. Sin cota inferior, el optimizador puede
comprar una mejora del tercer objetivo partiendo en clases que no son
interpretables: la ganancia es aritmética y el coste lo paga la lectura del
paper. `-K_util` corta la recompensa en cuanto las clases nuevas no alcanzan la
masa mínima. Es ahorro: deja de premiarse el incremento de clases cuando no
aporta estructura.

El cambio no oculta el K real. Solo cambia lo que se paga.

### 3.3 Ejemplo numérico: partición real

Partición que hoy sirve `cluster_id` en `data/workflow_data.json` (K = 3):

```
clases  = {244, 207, 52}             25 UCEs sin clase
umbral  = max(2, ceil(0.01 * 528)) = 6
K       = 3
K_util  = 3      (244 >= 6, 207 >= 6, 52 >= 6)
```

Con la partición que se sirve hoy el cambio es neutro: `K_util = K = 3` y
`-K_util = -3`. Lo que cambia es el paisaje de la búsqueda, no este artefacto.

### 3.4 Ejemplo numérico: partición hipotética

Ejemplo construido, no medido. Sirve para ver el mecanismo:

```
clases  = {244, 207, 60, 9, 4, 4}     suma 528
K       = 6
K_util  = 4      (244, 207, 60, 9 cuentan; 4 y 4 no)
```

Antes: `-K = -6`. Después: `-K_util = -4`. Las dos clases de 4 UCEs siguen
existiendo en la partición, pero ya no pagan. Con `int()` en vez de `ceil` el
resultado es el mismo 4, así que el ejemplo no depende de esa decisión.

### 3.5 Efecto esperado

El efecto es sobre el espacio de búsqueda, no sobre un artefacto concreto. A
partir de ahora, subir K deja de pagar en cuanto las clases nuevas caen por
debajo del piso. Se espera que el frente de Pareto deje de premiar particiones
con cola de clases diminutas y que la frontera entre soluciones con K distinto
se desplace hacia particiones con masa repartida.

No se pre-registra ningún resultado. No se afirma que el ARI suba, que K baje
ni que el ganador cambie.

### 3.6 Consecuencias que quedan escritas

1. `-best_values[2]` pasa a ser `-K_util`, **no** `-K`. Todo script de
   diagnóstico que imprima `K = -best_values[2]` cambia de significado. Hay que
   leer `k_ach` en los `user_attrs` del trial para el K real.
2. El gate de viabilidad sigue aplicándose sobre el K real (`k_ach`), no sobre
   `K_util`. `K_util <= k_ach` siempre, así que usar `K_util` en el gate
   relajaría la admisión sin decirlo.
3. Se persisten ambos: `k_ach` y `k_util` en los `user_attrs` del trial, y
   `n_useful_classes` como campo de `MethodMetrics` en la tabla de la regla.
   Hoy `k_ach` ya se escribe; `k_target`, `k_stable` y `k_shortfall` también.
4. Los estudios de Optuna anteriores no son comparables en el tercer objetivo.
   **Study nuevo obligatorio.** Mezclarlos contamina el frente de Pareto.

### 3.7 Invariantes que NO cambian

- Direcciones: `(maximize, maximize, minimize)`. El tercer objetivo se sigue
  minimizando; solo cambia su magnitud.
- Los objetivos 1 y 2 (`coverage_primary`, `ari_winner`) y su cálculo.
- El gate duro `K` real en `[2, 20]` y el trial centinela.
- `k_bounds = (3, 20)` de la regla select.
- El mapeo `cdh_target_k -> min_cluster_size_cdh = 1 / (2K)`.
- Segmentación, vocabulario, retención y `corpus_fingerprint`.
- La versión de la regla select (`select-v4`). `K_util` es del optimizador y de
  la tabla de métricas, no de la regla de decisión.

### 3.8 Reversión

Volver a `-K` es cambiar el tercer slot de vuelta a `-k_ach`, retirar
`n_useful_classes` de la tabla y relanzar con estudio nuevo. No se puede
"deshacer" reutilizando el estudio de `-K_util`.

## 4. Cambio de rango: piso 3 a 2 con fallback automático

### 4.1 Situación

El óptimo de `cdh_target_k` cae en el piso del espacio (3) en las campañas v5 y
v6; el `best_params.json` vigente tiene `cdh_target_k = 3`. Un óptimo pegado al
borde inferior admite dos lecturas: el piso no estorba, o el optimizador
querría menos clases y no puede. Hoy no se distingue.

### 4.2 Regla pre-registrada

Se aplica una sola vez, al final de la etapa 2.

1. Termina la etapa 2 y hay ganador `g`.
2. Si `g.cdh_target_k > 3`, el piso no era vinculante. Se adopta `g`. Fin.
3. Si `g.cdh_target_k == 3`, se relanza **una sola vez** la etapa 2 con el piso
   en 2, en un estudio nuevo y con el mismo presupuesto.
4. El resultado de la caída se **adopta** solo si su ganador elige
   `cdh_target_k == 2`.
5. Si la caída vuelve a elegir 3 o más, el piso 3 no era vinculante. Se
   **descarta la caída** y se conserva el resultado original.
6. Nunca se baja de 2. El gate duro es `K >= 2`; por debajo no hay partición
   admisible.

La caída no es recursiva: no existe una segunda caída a 1. Si el ganador de la
caída elige `cdh_target_k == 2`, se adopta y se termina.

La adopción exige que exista un ganador con `cdh_target_k == 2`. Si la caída
solo produce trials centinela, no hay ganador que adoptar y se conserva el
resultado original. La regla no se relaja para "salvar" la caída.

### 4.3 Coste

Una etapa 2 extra. Y solo cuando el óptimo queda pegado al piso. No hay coste
cuando `cdh_target_k > 3`.

### 4.4 Efecto esperado

Si el piso 3 era vinculante, el ganador adoptado tendrá `cdh_target_k = 2` y
sus particiones podrán tener menos clases que el ganador original. Si no lo
era, la caída vuelve a elegir 3 o más y no cambia nada: el resultado original
queda vigente y la caída solo deja constancia escrita de que el piso no
estorba.

No se pre-registra ningún resultado. La regla decide qué artefacto se adopta,
no qué valor debe salir.

### 4.5 Política de bordes: categoría nueva

`cdh_target_k` deja de ser "estructural: no se amplía" en el piso. Pasa a
tener categoría `fallback`: se ensancha **solo por abajo**, de forma automática,
con estudio nuevo. El techo sigue siendo estructural.

Esto hay que decirlo de forma explícita porque hoy
`_BOUNDARY_STRUCTURAL_WHY["cdh_target_k"]` justifica únicamente el techo: "por
encima de `_CDH_TARGET_K_HI` el CDH sale del gate `k_bounds` y el trial es
centinela por construcción". Esa frase no dice nada del piso, y el auditor de
bordes la usa para declinar cualquier ampliación. Con `fallback` el auditor
puede recomendar la caída por abajo sin abrir el techo.

Categorías resultantes:

| categoría   | parámetros                                                                                                           |
| ----------- | -------------------------------------------------------------------------------------------------------------------- |
| expandible  | `min_forms_uc_1`, `similarity_threshold`, `min_r2_threshold`, `coref_weight`                                         |
| estructural | `swap_iterations`, `coref_context_units_tight`, `hdbscan_uce_min_cluster_frac`, `hdbscan_uce_min_cluster_frac_loose` |
| fallback    | `cdh_target_k` (solo piso)                                                                                           |

### 4.6 Qué NO cambia

- El techo: `_CDH_TARGET_K_HI = 12`. Por encima de 12 el CDH sale de
  `[3, 20]` y el trial es centinela por construcción. Sigue siendo estructural.
- El gate `k_bounds = (3, 20)` de la regla select.
- El gate duro `[2, 20]` del optimizador.
- El mapeo `1 / (2K)`: con K = 2 el piso vale 0.25, dentro del recorte
  `[0.01, 0.5]` de `_cdh_frac_from_params`.
- El recorte por `n // 10`.

### 4.7 Reversión

Retirar la caída es dejar de dispararla: la categoría `fallback` vuelve a
`estructural` y el piso vuelve a 3. Un resultado ya adoptado no se puede
"desadoptar" dentro del mismo estudio; hay que relanzar.

## 5. Persistencia A/B/C y toggle en dashboard

### 5.1 Antes

Solo la partición del ganador llegaba a `cluster_id`. Las particiones de los
otros dos métodos se calculaban para el veredicto y se tiraban. Para
reconstruir A y B había que releer el log.

### 5.2 Contrato de datos

Se añade una clave de nivel superior en `data/workflow_data.json`:

```json
"method_labels": {
  "A": {"0_0_0": 1, "0_0_3": 2},
  "B": {"0_0_0": 0, "0_0_3": 0},
  "C": {"0_0_0": 2, "0_0_3": 1}
}
```

Reglas del contrato:

- Tres mapas, uno por método (`A` word-count, `B` similitud/coref, `C`
  embeddings + HDBSCAN).
- Solo UCEs **estables** de cada método. La cobertura de cada mapa es la suya:
  A 0.763, B 0.525, C 1.000 en la última corrida.
- Claves `str`. Es el campo `id` de la UCE (p. ej. `"0_0_3"`), no `uc_id` ni el
  índice de lista.
- Valores: entero de clase del método. Sin `-1` y sin `null`: si una UCE no es
  estable en un método, no aparece en ese mapa.

`cluster_id` **no cambia**. Sigue siendo la partición del ganador persistido.
`method_labels` es información adicional, no un reemplazo.

### 5.3 Dashboard

El dashboard (`src/reii/dashboard.py`, servicio `dashboard` de
`docker-compose.yml`, que lee `data/workflow_data.json` por
`REII_WORKFLOW_DB_PATH`) gana un selector de método con tres opciones y el
ganador como default. La vista por defecto es numéricamente idéntica a la
actual: mismo `cluster_id`, mismas cifras, mismos gráficos.

### 5.4 Efecto esperado

El paper puede comparar los tres métodos sobre el mismo corpus sin releer el
log. El selector no recalcula nada: cambia qué etiquetas se leen. Las cifras de
la vista por defecto no se mueven.

### 5.5 Limitaciones declaradas

1. Las tablas de términos por clase siguen siendo las del ganador persistido.
   Cambiar de método cambia la asignación de cada UCE, no las tablas de
   términos. Se avisa en la interfaz.
2. Los mapas tienen coberturas distintas. Una vista de A no cubre el corpus
   completo; no se renormaliza ninguna estadística.
3. Los productos derivados (φ, AFC, CAH, inercia) siguen calculados sobre
   `cluster_id` del ganador. No se recalculan por método.
4. No se recalculan métricas de estabilidad ni bootstrap por método durante la
   lectura: se muestran las del `method_selection.json` vigente.

### 5.6 Reversión

Quitar la clave `method_labels` y el selector devuelve el dashboard al
comportamiento actual. `cluster_id` no se toca en ningún caso, así que la
reversión no requiere relanzar el pipeline.

## 6. Protocolo de re-lanzamiento

### 6.1 Reglas

- **Study nuevo obligatorio.** Los tres cambios mueven el tercer objetivo o el
  espacio muestreado. Un estudio nuevo es la única forma de que el frente de
  Pareto sea comparable consigo mismo.
- Sin warm start desde estudios viejos. `--no-optuna-warm-start` es explícito.
- Nombre sugerido: `reii_v7`. El código añade sufijos por etapa, de modo que
  los estudios reales son `reii_v7_stage1_wc_<fp>` y
  `reii_v7_stage2_select_<fp>`.
- La caída de §4 usa prefijo propio: `reii_v7_k2`, que produce
  `reii_v7_k2_stage2_select_<fp>`. El marcador `k2` tiene que ir en el prefijo,
  porque el sufijo lo pone el código.
- Archivar antes: `data/reii_optuna.db`, `data/best_params.json`,
  `data/method_selection.*`.

### 6.2 Comando

```bash
docker compose --profile batch run --rm workflow python -u src/reii/main_workflow.py \
  --force --optimize-only --trials 300 --stage1-fraction 0.5 \
  --study-name reii_v7 --no-optuna-warm-start
```

Notas: `--profile batch` es obligatorio (el servicio `workflow` está declarado
bajo ese perfil; sin él `docker compose run workflow` falla). `--trials` tiene
default operativo 300; `--stage1-fraction` tiene default 0.3 y aquí se fija en
0.5 de forma explícita. `--force` sobrescribe artefactos. `--optimize-only` ya
ignora `best_params.json`, así que no hace falta moverlo antes.

La caída de §4 es **automática**: no se lanza a mano. Si el ganador de la
etapa 2 cae en el piso, el optimizador relanza la etapa 2 una sola vez con el
sufijo `_stage2_select_k2` (estudio `reii_v7_stage2_select_k2_<fp>`) y adopta
el resultado solo si su ganador elige ese piso. Para medirla aislada en otro
momento, el equivalente manual es:

```bash
docker compose --profile batch run --rm workflow python -u src/reii/main_workflow.py \
  --force --optimize-only --single-stage --trials 300 \
  --study-name reii_v7_k2 --no-optuna-warm-start
```

`--single-stage` restringe el run al espacio select, que es la etapa que se
relanza. Mismo presupuesto, estudio nuevo, única diferencia real: el piso del
espacio.

### 6.3 Verificación

Antes de lanzar, comprobar que artefactos y estudio están listos:

```bash
python diag.py preflight --study reii_v7
```

Antes del primer run el preflight falla G1 (todavía no existe ningún estudio
`reii_v7` para esta huella): es el comportamiento esperado y no bloquea
`--optimize-only`. Sirve para confirmar que `best_params` y la selección
sellados son del corpus en disco.

Después de lanzar:

```bash
python diag.py optuna-db --study reii_v7
python diag.py selection-audit
```

Qué mirar:

- `optuna-db`: existencia y tamaño de `reii_v7_stage2_select_<fp>`, número de
  trials, y que los `user_attrs` del ganador traigan `k_ach` y `k_util`.
- `optuna-db`: que el estudio declare `neg_k_util` y no `neg_k`. Si dice
  `neg_k`, se lanzó con el código viejo.
- `optuna-db`: proporción de trials centinela. Un exceso apunta al piso o al
  techo, no a la muestra.
- `selection-audit`: que la regla siga dando `select-v4` y que el ganador y su
  `n_useful_classes` sean coherentes con el `K_util` del optimizador.
- `preflight` de nuevo: confirma que estudio y artefactos quedaron
  consistentes.

Ninguna de estas comprobaciones autoriza a cambiar el objetivo ni el rango a
mitad de campaña. Si algo no cuadra, se corrige y se relanza con estudio nuevo.

## 7. Qué invalida qué

| cambio                         | artefactos invalidados                                    | cómo se regeneran                                                |
| ------------------------------ | --------------------------------------------------------- | ---------------------------------------------------------------- |
| `-K` a `-K_util`               | `data/reii_optuna.db` (estudios), `data/best_params.json` | archivar y lanzar §6.2 con `reii_v7`                             |
| piso 3 a 2, solo si se dispara | `data/best_params.json`, si se adopta la caída            | `reii_v7_k2`; el ganador adoptado reescribe `best_params.json`   |
| particiones A/B/C persistidas  | `data/workflow_data.json`                                 | corrida completa posterior al optimizador (no `--optimize-only`) |
| nuevo `best_params.json`       | `data/method_selection.json` y `method_selection_log.*`   | se recalculan en la corrida completa con las particiones nuevas  |
| objetivo o piso                | nada de segmentación ni de léxico                         | no se invalidan                                                  |

**No** se invalidan `data/uce_calibration.json` ni
`data/coref_chains_cache.json`: la segmentación y el cache no dependen del
objetivo ni del rango. Tampoco cambia la política de retención.

## 8. Riesgos y mitigaciones

| riesgo                                                                                                    | por qué                                                        | mitigación                                                                                                                      |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `K_util` favorece muchas clases pequeñas-pero-legales                                                     | el umbral es un piso, no una penalización por cercanía al piso | `min_class_share` ya acota; una clase de 6 UCEs cuenta igual que una de 200, y eso se declara como límite conocido del objetivo |
| el fallback duplica el coste de la etapa 2                                                                | es una etapa 2 completa                                        | solo se dispara si el óptimo está exactamente en el piso 3; con `cdh_target_k > 3` no hay coste                                 |
| con piso 2, los trials que aterrizan en K = 2 son infeasibles para la regla select (`k_bounds = (3, 20)`) | el gate duro del optimizador admite K >= 2, la regla no        | si la caída solo produce centinelas no hay ganador con `cdh_target_k == 2`, y se conserva el resultado original (§4.2)          |
| el toggle muestra las tablas de términos del ganador                                                      | las tablas no se recalculan por método                         | aviso visible en la interfaz; declarado en §5.5                                                                                 |
| `-best_values[2]` se lee como K en scripts viejos                                                         | el campo no cambia de nombre, solo de significado              | `objectives` pasa a `neg_k_util` como marca; leer `k_ach` de los `user_attrs`                                                   |
| umbral de clase útil con `ceil` (6) frente a `int` (5) del código                                         | dos expresiones para la misma idea                             | una sola fuente de verdad; este pre-registro fija `ceil` y ninguna partición medida cambia (§2)                                 |

## 9. Referencias internas

- `docs/design_v5/SELECT_V4.md` — regla de decisión, `k_bounds`, viabilidad y
  los campos de `MethodMetrics` (§2, §2.3, §7.4).
- `docs/design_v5/GATE_NOISE.md` — gates de ruido y dominancia que condicionan
  qué cuenta como clase.
- `docs/design_v5/SECCION_ALCESTE_BORRADOR.md` — sección 4.4.5 sobre K y su
  coincidencia sin validación independiente; contexto de por qué K se reporta
  con cuidado en el paper.
- `docs/DIAG_CLI.md` — comandos `preflight`, `optuna-db`, `selection-audit`.
- `docs/design_v5/RETENTION_POLICY.md` — §10.3, invalidación previa de
  `best_params.json` por vocabulario; §10.5 enlaza este pre-registro.
- Código de referencia, no modificado aquí: `src/reii/main_workflow.py`
  (`_CDH_TARGET_K_HI`, `_cdh_frac_from_params`, `get_search_space`,
  `_evaluate_trial`, `_BOUNDARY_STRUCTURAL*`) y `src/reii/method_selection.py`
  (`MethodMetrics`).
