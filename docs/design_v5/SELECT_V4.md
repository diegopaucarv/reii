# `select-v4`: regla de decision con estabilidad, nulo, backend y dominancia sobre el corpus

Estado: **especificacion cerrada, no aplicada**. `src/reii/method_selection.py`
sigue en `select-v3`; el diff de la seccion 7 es el texto exacto que hay que
aplicar (anclas verificadas contra el archivo actual: cada bloque `OLD`
aparece **una sola vez**).

Alcance: `src/reii/method_selection.py` (solo). No toca `main_workflow.py`,
`9_selection_audit.py`, `data/`, `output/` ni `models/`.

Documentos con los que se articula:

| documento               | que aporta a esta regla                                   |
| ----------------------- | --------------------------------------------------------- |
| `GATE_NOISE.md`         | la escala de dominancia (`c = a(1-nu)`, `C0 = 0.70`)      |
| `BACKEND_C.md` §2, §3   | `hdbscan` como diagnostico; `ari` tautologico en densos   |
| `AUDIT_V4.md` §2, §3    | que campos debe persistir el pipeline para auditar        |
| `METRICS_PAPER.md` §1.4 | que es (y que no es) un nulo por permutacion de etiquetas |
| `select_v4_sim.py`      | calibracion medida del nulo: etiquetas vs datos           |

`STABILITY_NULL.md` **no existe** en el repo a la fecha. Por eso el nulo entra
con un flag (`null_mode`) y un default conservador, en vez de con una decision
cableada: la seccion 4 deja escrito que cambia el veredicto en cada modo, de
modo que cuando llegue la evidencia solo hay que mover el default, no la regla.

---

## 0. Resumen en cinco lineas

1. **Viabilidad**: existe, no es diagnostico, `K in [3,20]`, `nu <= 0.50`,
   `c = a(1-nu) <= 0.70`, cobertura efectiva `>= 0.15`, estabilidad finita,
   sin `ari` tautologico, y `null_p <= 0.05` si el nulo esta disponible.
2. **Ranking**: `stability = bootstrap_ari` si existe; si no `ari`. Se declara
   el kind. Nunca se compara C contra A/B sin declararlo.
3. **Desempate**: coverage efectiva, y despues orden canonico `A < B < C`.
4. **Banderas**: `|dStab| <= margin` → `ambiguous`; kinds distintos o runner-up
   con mas coverage → `criteria_conflict`.
5. **Abstencion**: sin viables, `winner=None`. Nunca una excepcion.

---

## 1. Por que hay que cambiar la regla (y no solo el umbral)

`select-v3` decide con tres senales: ARI intra-metodo, coverage usable y un gate
de dominancia sobre lo **asignado**. Cada una falla por un motivo distinto
cuando el metodo C deja de ser HDBSCAN:

| fallo de v3                            | evidencia medida                                               | consecuencia                                    |
| -------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------- |
| premia `ari == 1.0` de `ward`/`kmeans` | `BACKEND_C.md` §3: ARI=1.0 en `synth_null` para 7 valores de K | C gana sobre ruido puro, por construccion       |
| no tiene contraste contra el nulo      | un corpus sin estructura elige metodo igual que uno con ella   | publica como hallazgo lo que es azar            |
| dominancia sobre lo asignado           | `{442,45,30,11}` → 0.837 pasa; `{442,23}`+63 ruido → 0.951 no  | el ruido decide el veredicto (bug del probe v4) |
| `coverage` de un denso es 1.0          | un backend denso asigna el 100 % del corpus                    | el desempate favorece a C siempre               |
| snapshot sin validar `rule_version`    | `load_selection` solo mira fingerprint y target                | se reutiliza una decision `select-v3` como v4   |

La regla nueva anade cuatro senales (`backend`/`diagnostic_only`,
`bootstrap_ari`, `null_p`, `coverage_stable`) y **cambia el denominador** de la
dominancia. No cambia ningun nombre de campo existente (`coverage`, `ari`,
`usable_coverage`, `max_class_share` se quedan).

---

## 2. La regla, paso a paso

Pseudocodigo (el detalle exacto esta en el diff de la seccion 7):

```
para cada m en metrics:
    descartar si  not m.available
                  o m.diagnostic_only  o m.backend in {"hdbscan"}
                  o not isfinite(m.stability)
                  o m.k fuera de [3, 20]
                  o m.noise_frac > 0.50
                  o m.dominance_corpus_effective > 0.70
                  o m.usable_effective < 0.15
                  o (m.backend in {"ward","kmeans_spherical"} and kind == "perturbation")
                  o (null_mode == "gate" and isfinite(m.null_p) and m.null_p > 0.05)

if no hay viables:                                   -> abstención
orden = (-stability, -usable_effective, A<B<C)        (null_mode == "rank" antepone
                                                      la clase de evidencia)
ganador, runner_up = orden[0], orden[1] (si existe)
ambiguous         = |dStab| <= margin
criteria_conflict = (kind_ganador != kind_runner) or (usable_runner > usable_ganador)
```

### 2.1 Las tres cantidades nuevas

| cantidad                     | definicion                                       | si falta               |
| ---------------------------- | ------------------------------------------------ | ---------------------- |
| `coverage_effective`         | `coverage_stable` si es finita, si no `coverage` | `coverage` (compat v3) |
| `usable_effective`           | `max(0, coverage_effective - tiny_frac)`         | identico a v3          |
| `dominance_corpus_effective` | ver 2.2                                          | derivado               |
| `stability`                  | `bootstrap_ari` si es finita, si no `ari`        | `ari` (compat v3)      |
| `stability_kind`             | `"bootstrap"` / `"perturbation"`                 | `"perturbation"`       |

`usable_coverage` **no cambia** (sigue usando `coverage`, no la efectiva): es un
nombre publico que el auditor y los tests ya consumen. La cobertura efectiva se
expone aparte, en `usable_effective` y en `as_row()["usable_eff"]`.

### 2.2 Precedencia de `dominance_corpus`

```
1. m.dominance_corpus  (si el productor lo midio y es finito)   <- campo, no propiedad
2. m.max_class_size / m.n_total  (si ambos > 0: es el cociente exacto)
3. m.max_class_share * (1 - m.noise_frac)  (identidad, GATE_NOISE.md §1)
```

Por que un campo **y** una propiedad efectiva: `AUDIT_V4.md` §3 necesita
distinguir "el productor midio `c`" de "el modulo lo derivo" para poder
comprobar la identidad `c = a(1-nu)`. Si `dominance_corpus` fuera una propiedad
(como proponia `GATE_NOISE.md` §12.1) el auditor no podria distinguir un valor
mal calculado de uno derivado. El campo guarda lo que llego; la propiedad
`dominance_corpus_effective` es lo que decide.

La precedencia da prioridad a los **conteos** sobre la identidad porque
`n_assigned + n_noise == n_total` y `s_max / n_total` son medibles
directamente: si alguien calcula mal `a`, los conteos lo delatan.

### 2.3 Orden de evaluacion de la viabilidad (importa)

1. `available`, 2. `diagnostic_only`/`backend == "hdbscan"`, 3. estabilidad
   finita, 4. `K`, 5. ruido, 6. dominancia, 7. cobertura, 8. tautologia, 9. nulo.

El nulo va **ultimo** a proposito: es la senal mas discutible y la unica que
puede no existir. Si un metodo ya cae por dominancia, el motivo del descarte
debe ser la dominancia (es el hecho duro, no la inferencia). El motivo se
imprime y el auditor lo compara con su propio replay (`AUDIT_V4.md` invariante
2), asi que el orden forma parte del contrato.

---

## 3. Tabla de decision (salida real de `select_v4_arena.py`)

17 escenarios, cada uno aislando una decision. `v3` es el selector actual;
`v4` la regla nueva. `run` es el runner-up.

| escenario        | v3  | v4   | run | flags    | por que (medido)                                                      |
| ---------------- | --- | ---- | --- | -------- | --------------------------------------------------------------------- |
| `base_lex`       | A   | A    | B   | -        | estabilidad 0.350 vs 0.212: la regla nueva no cambia el caso facil    |
| `conflict_cov`   | A   | A    | B   | amb+conf | d=+0.037 <= margin 0.05 y B cubre mas (0.375 vs 0.267)                |
| `c_hdbscan`      | C   | A    | -   | -        | C con `backend=hdbscan`: descartado aunque ARI=0.99 y cov=0.90        |
| `c_ward_taut`    | C   | A    | -   | -        | C ward sin bootstrap: `ari=1.0` tautologico, no es estabilidad        |
| `c_dense_dom`    | C   | A    | -   | -        | `a=0.837, nu=0` → `c=0.84 > 0.70`: degenerada (bug del probe v4)      |
| `c_bad_null`     | C   | A    | -   | -        | bootstrap 0.70 pero `p=0.610`: sin evidencia contra el nulo           |
| `c_dense_real`   | C   | C    | A   | -        | bootstrap 0.72 vs 0.30 y `p=0.004`: **C leguimo gana** (no es anti-C) |
| `c_low_boot`     | A   | A    | C   | conf     | mismo kind: 0.550 > 0.250, pero C cubre mas (0.85 vs 0.60)            |
| `null_noise`     | C   | None | -   | abst     | los tres con `p > 0.05`: no se clasifica                              |
| `k_floor`        | A   | A    | -   | -        | C con `K=2`: ARI=1.0 trivial descartado                               |
| `noise`          | A   | A    | -   | -        | C con `nu=0.60 > 0.50`: mas de medio corpus sin clasificar            |
| `cov_stable_low` | C   | A    | -   | -        | C con ARI 0.70 pero solo 10 % de UCEs estables                        |
| `kinds_differ`   | A   | A    | C   | amb+conf | A mide perturbacion y C bootstrap, `                                  | d   | =0.010`: escalas distintas |
| `only_taut`      | C   | None | -   | abst     | el unico candidato es tautologico: se abstiene, no gana               |
| `minimal`        | -   | None | -   | abst     | `MethodMetrics("A",1,0.9,0.9)`: `K=1`, sin excepcion                  |
| `null_only_gate` | C   | None | -   | abst     | unico candidato con `p=0.61`: el modo `gate` abstiene (default)       |
| `null_only_rank` | C   | C    | -   | -        | **las mismas metricas** con `null_mode="rank"`: C gana                |

Reproducir (2.5 s, exit 0):

```
python scripts/playground/select_v4_arena.py
python scripts/playground/select_v4_arena.py --use-module   # tras aplicar el diff
```

`--use-module` compara la copia canonica contra el modulo real y **falla** si
divergen: es la garantia de que lo de arriba no es una maqueta.

Las dos ultimas filas son el corazon de la seccion 4: mismas metricas, un flag
distinto, veredicto distinto.

---

## 4. ¿El nulo como gate o como ranking?

> **AMENDMENT (STABILITY_NULL.md v5): el default es `diagnostic`, no `gate`.**
> `STABILITY_NULL.md` §2 mide `p = 0.005` sobre ruido puro en 12/12 filas del
> banco: con un nulo de ETIQUETAS el gate es inerte y no puede producir
> abstencion sobre ruido. `null_mode="diagnostic"` calcula, persiste e imprime
> el p-valor como **columna de control** sin descartar ni ordenar. `"gate"`
> queda disponible y solo tiene sentido con un **nulo de datos** (romper los
> datos, no las etiquetas), que hoy no existe; se sigue testeando con `null_p`
> inyectado. El razonamiento de abajo se conserva porque sigue siendo correcto
> sobre POR QUE `rank` esta prohibido: un ranking sesgado hacia C es peor que
> un gate vacuo, y `diagnostic` es el gate vacuo sin el coste de abstencion.
> Leer los literales `null_mode: str = "gate"` de §7.5 y §7.8/§7.9 como
> `"diagnostic"`, y anadir `"diagnostic"` a `_NULL_MODES`.

### 4.1 La evidencia disponible

`scripts/playground/select_v4_sim.py::data_null_evidence` documenta el
resultado medido, y es el unico dato de calibracion del nulo que existe hoy
(`STABILITY_NULL.md` no esta):

> `ward` y `kmeans` MINIMIZAN la dispersion intra-clase, asi que su silhouette
> es positiva por construccion frente a etiquetas aleatorias. Medido: sobre
> ruido puro, la permutacion de etiquetas da `z ~ 8` y `p = 1/(1+n_perm)`
> (falso positivo). Es el mismo defecto que `r2_1d`: el estadistico se
> optimiza.

Es decir: el algoritmo **elige** la particion que maximiza el estadistico, de
modo que compararla contra etiquetas aleatorias no contrasta "hay estructura"
sino "el algoritmo optimiza". El nulo correcto fija el algoritmo y rompe los
datos (reajustar el mismo backend sobre una gaussiana con la misma media y
covarianza: logica de _gap statistic_).

Nota de alcance: `METRICS_PAPER.md` §1.4 si propone la permutacion de etiquetas,
pero para `C_in`, que **no** es el estadistico que el clustering optimiza. Un
nulo no es bueno o malo en abstracto: depende del estadistico.

### 4.2 Por que eso descarta `rank`

| uso del nulo | que hace con el sesgo medido                                              |
| ------------ | ------------------------------------------------------------------------- |
| `gate`       | puede **quitar** candidatos. Con un nulo debil no quita nada (vacuo).     |
| `rank`       | **premia** al que mas optimiza el estadistico → sesgo sistematico hacia C |

Un gate vacuo es inofensivo: se comporta como `off` cuando el nulo no informa.
Un ranking sesgado no: ordena por "cuanto optimiza el algoritmo", que es
exactamente el artefacto que `BACKEND_C.md` §3 documento y que v4 viene a
eliminar. **Por eso `rank` no es el default, aunque exista.**

### 4.3 Que cambia el veredicto en cada modo

| `null_mode` | viabilidad              | ranking                                | caso `null_only_*` (unico candidato, `p=0.61`) |
| ----------- | ----------------------- | -------------------------------------- | ---------------------------------------------- |
| `gate`      | descarta si `p > alpha` | `-stability`                           | **abstencion** (`winner=None`)                 |
| `rank`      | no descarta por nulo    | clase de evidencia, luego `-stability` | **gana C** (bootstrap 0.70)                    |
| `off`       | ignora                  | `-stability`                           | gana C                                         |

Con varios candidatos viables, `gate` y `rank` casi siempre coinciden: `rank`
solo reordena cuando alguno tiene `p > alpha` y otro no. La diferencia
practica es el caso "el unico candidato no supera el nulo": `gate` se abstiene
(visible, recuperable), `rank` clasifica igual. Dado que el costo de una
particion falsa es publicarla como hallazgo, el default es `gate`.

### 4.4 Como se degrada honestamente

- `null_p` es `nan` ⇒ el gate **no interviene** (nunca descarta por falta de
  dato). Es lo que hace que la regla sea compatible con los productores que aun
  no emiten el campo: todos los tests de v3 siguen pasando.
- Con un nulo debil (etiquetas), el gate **no puede producir abstención sobre
  ruido**; solo un nulo de datos lo hace. Esta declarado como riesgo (§9.1).
  `null_z` se persiste para poder auditar la calibracion sin cambiar la regla.
- `alpha = 0.05` exige que el minimo alcanzable sea menor: con `n_perm = 20`,
  `p_min = 1/21 = 0.048` (hay que superar **todos** los nulos); con
  `n_perm = 200`, `p_min = 0.005`. El productor no debe entregar menos de 20
  permutaciones o el gate queda sin resolucion.
- Si mañana `STABILITY_NULL.md` mide que el nulo de datos discrimina, el cambio
  es `null_mode="gate"` (ya es el default): no hay que tocar codigo.

---

## 5. Familias de estabilidad y `criteria_conflict`

`bootstrap_ari` y `ari` no miden lo mismo:

| kind           | que perturba                          | escala                                                |
| -------------- | ------------------------------------- | ----------------------------------------------------- |
| `bootstrap`    | submuestra 80 % de UCEs y reajusta    | depende de `n`, `subsample_frac` y `n_boot`           |
| `perturbation` | el pase tight/loose del propio metodo | depende del hiperparametro perturbado (piso, ventana) |

Ademas, para `backend in {ward, kmeans_spherical}` el pase tight/loose es
**identico** (no reciben el piso): `ari == 1.0` por construccion, medido en
`synth_null` para los 7 valores de K (`BACKEND_C.md` §3). Ese 1.0 no es
estabilidad: es una propiedad algebraica del backend.

La regla hace tres cosas, en este orden de honestidad:

1. **No lo usa**: si `backend` esta en la lista tautologica y no hay
   `bootstrap_ari`, el metodo no compite (`reject_tautological`, activable).
   Esto es lo que impide que C gane "por construccion" en el caso `c_ward_taut`.
2. **Lo declara**: `winner_stability_kind` viaja en la `Selection` y se imprime.
3. **Lo marca**: si ganador y runner-up usan kinds distintos, `criteria_conflict`
   es `True` aunque el delta sea grande, y `explain_lines()` dice **por que**:

```
[!] conflicto de criterios: A mide 'perturbation' y C mide 'bootstrap'; no estan
    en la misma escala (bootstrap mide remuestreo, perturbacion mide el pase
    tight/loose)
```

No se reordena por kind porque no hay una escala comun disponible: preferir
bootstrap siempre castigaria a A/B (que no lo tienen), y preferir perturbacion
siempre premiaria el tautologico. Ordenar y **declarar** es lo unico
defendible; el flag es la senal para revision humana, igual que `ambiguous`.

Corolario para el paper: los valores de `stability` **no** son comparables entre
corridas con distinto `n`, `subsample_frac` o `n_boot`. Se reporta siempre con
su kind.

---

## 6. Dominancia: `c`, `C0 = 0.70`, y el umbral efectivo

La regla de `GATE_NOISE.md` se aplica tal cual:

```
c_m = max_class_share_m * (1 - noise_frac_m)          (fraccion del corpus)
degenerada(m)  <=>  c_m > C0,   C0 = 0.70
```

`C0` **no se sube a 0.75** (decision del usuario): la simulacion de
`GATE_NOISE.md` §6.4 da `fv = 0.008` a 0.70 y `0.024` a 0.72, con el peor caso
por regimen disparandose (`0.051 → 0.127`). 0.75 queda fuera del conjunto
admisible (`fv <= 1 %`).

Lectura dual (`a` es la variable que hoy se imprime):

| `nu` | umbral sobre `c` | umbral sobre `a = c/(1-nu)` | efecto          |
| ---- | ---------------- | --------------------------- | --------------- |
| 0.00 | 0.70             | 0.700                       | activo          |
| 0.10 | 0.70             | 0.778                       | activo          |
| 0.20 | 0.70             | 0.875                       | activo          |
| 0.30 | 0.70             | 1.000                       | frontera        |
| 0.35 | 0.70             | 1.077                       | sin restriccion |
| 0.50 | 0.70             | 1.400                       | sin restriccion |

`explain_lines()` imprime el umbral **efectivo** (`self.max_class_share_max`),
nunca el 0.90 cableado, y muestra `domC` por metodo:

```
     clase mayor=0.60 de las asignadas = 0.60 del corpus (gate: >0.70 del corpus = particion degenerada)
```

A partir de `nu = 0.30` el gate es vacuo por construccion (no puede disparar).
Eso **no** es un hueco de esta regla: en ese regimen el que descarta es
`k_bounds` (`GATE_NOISE.md` §8). Si se quisiera proteccion ahi, hay que bajar
`noise_max`, no `C0`.

---

## 7. Diffs exactos de `src/reii/method_selection.py`

Cada par `OLD`/`NEW` es un reemplazo literal; los bloques son disjuntos y
aplicables en cualquier orden. Los `OLD` incluyen el texto actual **tal cual**
(incluidos los separadores de linea de los comentarios).

### 7.1 Cabecera de la regla (docstring)

<!-- OLD:DOCSTRING -->

```python
Regla (lexicográfica, versión ``select-v3``)
────────────────────────────────────────────
1. **Viabilidad**: el método existe (produjo partición), su ``K`` cae en
   ``k_bounds`` (por defecto **3..20**: ``K=2`` es trivial y su ARI=1.0 no
   significa nada), su **ruido** no supera ``noise_max`` (0.50: más de medio
   corpus sin clasificar no es una partición) y su **coverage usable** alcanza
   ``coverage_floor`` (por defecto **0.15**).
2. **Maximizar ARI intra-método** (A1↔A2, B1↔B2, C1↔C2). Es la única señal de
   reproducibilidad que no depende de comparar particiones distintas.
3. **Desempate por coverage usable** (más material analizable gana).
4. **Ambigüedad declarada**: si ``|ARI_1 - ARI_2| <= margin`` la elección se
   marca ``ambiguous``. Si además el segundo tiene **más coverage usable** que
   el ganador, se marca ``criteria_conflict`` (los dos criterios apuntan a
   métodos distintos). En ambos casos el ganador se mantiene —la regla es
   determinista— pero el run queda marcado para revisión humana.
5. **Abstención**: si ningún método es viable, ``winner=None`` y
   ``abstained=True``. El pipeline **no clasifica** pero continúa: las UCEs
   quedan sin ``cluster_id`` y las etapas posteriores corren sobre vacío.
6. **No dominancia** (select-v3): si la clase mayor se come más del
   ``max_class_share_max`` (90 %) de las UCEs **asignadas**, la partición es
   degenerada (una clase + satélites) y el método se descarta. El ARI de una
   partición así es alto por construcción: replicar una clase gigante es
   fácil. Se mide sobre lo asignado, no sobre el corpus, para que el ruido de
   HDBSCAN no maquille la dominancia.
```

<!-- NEW:DOCSTRING -->

```python
Regla (lexicográfica, versión ``select-v4``)
────────────────────────────────────────────
1. **Viabilidad**: el método existe (produjo partición), **no es diagnóstico**
   (``diagnostic_only`` o ``backend="hdbscan"``: no controla K y su ARI alto es
   artefacto del colapso, ``BACKEND_C.md`` §2), su ``K`` cae en ``k_bounds``
   (3..20: ``K=2`` es trivial y su ARI=1.0 no significa nada), su **ruido** no
   supera ``noise_max`` (0.50) y su **cobertura efectiva**
   (``coverage_stable`` si el motor la produjo; si no ``coverage``) alcanza
   ``coverage_floor`` (0.15).
2. **No dominancia sobre el corpus**: ``c = a * (1 - nu)``, con
   ``a = max_class_share`` y ``nu = noise_frac``. Si ``c > C0`` (0.70) la
   partición es degenerada (una clase + satélites) y el método se descarta.
   Medir sobre lo asignado hacía depender el veredicto del ruido, y con el
   signo equivocado: misma ``c = 0.837``, aceptada sin ruido y descartada con
   ruido. Derivación y calibración en ``docs/design_v5/GATE_NOISE.md``.
3. **Contraste contra el nulo** (``null_p``): ``null_mode`` decide qué se hace
   con él. ``"gate"`` (default) descarta si ``null_p > null_alpha``; ``"rank"``
   no descarta y antepone la clase de evidencia al orden; ``"off"`` lo ignora.
   ``nan`` = no disponible y **nunca** descarta. Con un nulo de etiquetas
   permutadas el gate es débil (no puede rechazar ruido); con un nulo de datos
   es el que produce abstención sobre ruido. Ver ``SELECT_V4.md`` §4.
4. **Criterio principal: estabilidad**. ``stability = bootstrap_ari`` si existe
   (remuestreo real: submuestra + reajuste), si no ``ari`` (pase tight/loose).
   El criterio usado se declara en ``winner_stability_kind``.
5. **Las familias no son la misma escala**: un ARI de bootstrap y un ARI de
   perturbación no son comparables. La regla ordena igual (no hay escala común
   disponible) pero si ganador y runner-up usan kinds distintos marca
   ``criteria_conflict`` y lo explica. Además, ``backend in
   {"ward","kmeans_spherical"}`` no tiene grados de libertad en el pase
   tight/loose (``ARI == 1.0`` por construcción, ``BACKEND_C.md`` §3): ese
   ``ari`` es **tautológico** y, sin ``bootstrap_ari``, no compite
   (``reject_tautological``).
6. **Desempate**: coverage efectiva (``usable_effective``) y, por último, orden
   canónico ``A < B < C``.
7. **Ambigüedad declarada**: si ``|dStab| <= margin`` la elección se marca
   ``ambiguous``. Si además el runner-up cubre más, o los kinds difieren, se
   marca ``criteria_conflict``. El ganador se mantiene —la regla es
   determinista— pero el run queda marcado para revisión humana.
8. **Abstención**: si ningún método es viable, ``winner=None`` y
   ``abstained=True``. El pipeline **no clasifica** pero continúa: las UCEs
   quedan sin ``cluster_id`` y las etapas posteriores corren sobre vacío.
   Nunca se lanza una excepción por métricas incompletas: todo campo nuevo cae
   a ``nan`` = "no disponible" y su señal no interviene.
```

### 7.2 `import logging` (para registrar el rechazo del snapshot)

<!-- OLD:IMPORTS -->

```python
import json
import math
```

<!-- NEW:IMPORTS -->

```python
import json
import logging
import math
```

### 7.3 Version de la regla

<!-- OLD:RULE_VERSION -->

```python
RULE_VERSION = "select-v3"
```

<!-- NEW:RULE_VERSION -->

```python
RULE_VERSION = "select-v4"
```

### 7.4 `MethodMetrics`: campos nuevos, propiedades y `as_row`

<!-- OLD:METRIC_FIELDS -->

```python
    # ── diagnósticos (no entran en la decisión) ──────────────────────────
    assigned_frac: Optional[float] = None
    min_class_size: int = 0
    median_class_size: float = 0.0
    #: Tamaño de la clase mayor / UCEs asignadas (partición cruda del método).
    #: Alimenta el gate de no dominancia; también se imprime.
    max_class_share: float = 0.0

    @property
    def usable_coverage(self) -> float:
        """UCEs estables y en una clase con masa suficiente / ``n_total``."""
        return max(0.0, float(self.coverage) - float(self.tiny_frac))

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["usable"] = self.usable_coverage
        return row
```

<!-- NEW:METRIC_FIELDS -->

```python
    # ── diagnósticos (no entran en la decisión) ──────────────────────────
    assigned_frac: Optional[float] = None
    min_class_size: int = 0
    median_class_size: float = 0.0
    #: Tamaño de la clase mayor / UCEs asignadas (partición cruda del método).
    #: Alimenta el gate de no dominancia; también se imprime.
    max_class_share: float = 0.0
    # ── select-v4: señales nuevas (default = "no disponible") ────────────
    #: "" (A/B) | "hdbscan" | "dense_cdh" | "ward" | "kmeans_spherical".
    backend: str = ""
    #: True => el método es diagnóstico y no compite (C con hdbscan).
    diagnostic_only: bool = False
    #: Estabilidad por remuestreo (ARI medio contra la partición completa).
    bootstrap_ari: float = float("nan")
    bootstrap_ci_lo: float = float("nan")
    bootstrap_ci_hi: float = float("nan")
    #: Cobertura con estabilidad por UCE (nan => se usa ``coverage``).
    coverage_stable: float = float("nan")
    #: p-valor contra etiquetas permutadas (nan => no disponible).
    null_p: float = float("nan")
    null_z: float = float("nan")
    n_total: int = 0
    n_assigned: int = 0
    n_noise: int = 0
    max_class_size: int = 0
    #: c = a * (1 - nu) medido por el productor; nan => se deriva.
    dominance_corpus: float = float("nan")

    @property
    def usable_coverage(self) -> float:
        """UCEs estables y en una clase con masa suficiente / ``n_total``."""
        return max(0.0, float(self.coverage) - float(self.tiny_frac))

    @property
    def coverage_effective(self) -> float:
        """``coverage_stable`` si el motor la produjo; si no ``coverage``.

        En un backend denso ``coverage`` es 1.0 por construcción (asigna todo),
        así que premiarla premiaría a C siempre. ``coverage_stable`` mide la
        fracción de UCEs con etiqueta estable en el remuestreo.
        """
        stable = float(self.coverage_stable)
        return stable if math.isfinite(stable) else float(self.coverage)

    @property
    def usable_effective(self) -> float:
        """Cobertura efectiva descontando las clases diminutas."""
        return max(0.0, self.coverage_effective - float(self.tiny_frac))

    @property
    def dominance_corpus_effective(self) -> float:
        """Clase mayor / ``n_total``, con precedencia explícita.

        ``1`` si el productor midió ``dominance_corpus``; ``2`` el cociente
        exacto ``max_class_size / n_total``; ``3`` la identidad
        ``max_class_share * (1 - noise_frac)`` (``GATE_NOISE.md`` §1). Los
        conteos van antes que la identidad porque si ``a`` se calculó mal,
        ``s_max / n_total`` lo delata.
        """
        measured = float(self.dominance_corpus)
        if math.isfinite(measured):
            return measured
        if int(self.n_total) > 0 and int(self.max_class_size) > 0:
            return float(self.max_class_size) / float(self.n_total)
        share = float(self.max_class_share or 0.0)
        noise = min(1.0, max(0.0, float(self.noise_frac or 0.0)))
        return share * (1.0 - noise)

    @property
    def stability(self) -> float:
        """Criterio principal: bootstrap si existe, perturbación si no."""
        boot = float(self.bootstrap_ari)
        return boot if math.isfinite(boot) else float(self.ari)

    @property
    def stability_kind(self) -> str:
        """``"bootstrap"`` | ``"perturbation"`` (lo que mide ``stability``)."""
        return "bootstrap" if math.isfinite(float(self.bootstrap_ari)) else "perturbation"

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["usable"] = self.usable_coverage
        row["usable_eff"] = self.usable_effective
        row["coverage_effective"] = self.coverage_effective
        row["dominance_corpus_eff"] = self.dominance_corpus_effective
        row["stability"] = self.stability
        row["stability_kind"] = self.stability_kind
        return row
```

### 7.5 `Selection`: umbral, escala y las senales del ganador

<!-- OLD:SELECTION_FIELDS -->

```python
    noise_max: float = 0.50
    max_class_share_max: float = 0.90
    k_bounds: Tuple[int, int] = (3, 20)
```

<!-- NEW:SELECTION_FIELDS -->

```python
    noise_max: float = 0.50
    #: Gate de dominancia sobre el CORPUS: c = max_class_share*(1-noise_frac).
    max_class_share_max: float = 0.70
    k_bounds: Tuple[int, int] = (3, 20)
    # ── select-v4 ────────────────────────────────────────────────────────
    #: Escala del umbral de dominancia: la clase mayor se mide sobre el corpus.
    dominance_scale: str = "corpus"
    #: Modo del contraste contra el nulo: "gate" | "rank" | "off".
    null_mode: str = "gate"
    #: Nivel del contraste contra el nulo (p de una cola).
    null_alpha: float = 0.05
    #: Estabilidad del ganador y el criterio con el que se midió.
    winner_stability: float = float("nan")
    winner_stability_kind: str = ""
    winner_null_p: float = float("nan")
    #: c del ganador (clase mayor / n_total). Para el auditor, no para decidir.
    winner_dominance_corpus: float = float("nan")
    runner_up_stability: float = float("nan")
    runner_up_stability_kind: str = ""
```

### 7.6 `from_dict`: restaurar los nuevos `NaN`

<!-- OLD:FROM_DICT -->

```python
        for key in (
            "winner_ari",
            "winner_coverage",
            "winner_usable_coverage",
            "runner_up_ari",
            "runner_up_coverage",
            "runner_up_usable_coverage",
            "delta_ari",
            "delta_coverage",
        ):
```

<!-- NEW:FROM_DICT -->

```python
        for key in (
            "winner_ari",
            "winner_coverage",
            "winner_usable_coverage",
            "runner_up_ari",
            "runner_up_coverage",
            "runner_up_usable_coverage",
            "delta_ari",
            "delta_coverage",
            "winner_stability",
            "winner_null_p",
            "winner_dominance_corpus",
            "runner_up_stability",
        ):
```

### 7.7 `explain_lines`: kind, `domC`, `nullp` y umbral efectivo

<!-- OLD:EXPLAIN -->

```python
    def explain_lines(self) -> List[str]:
        out = [
            f"METHOD SELECTION (rule {self.rule_version})",
            f"  {'met':4}{'K':>4}{'ARI':>8}{'cov':>7}{'usable':>8}{'noise':>7}"
            f"{'tiny':>7}{'dom':>7}  {'perturbacion':14}  veredicto",
        ]
        for row in self.all_metrics:
            tag = ""
            if row["method"] == self.winner:
                tag = "<- winner"
            elif row["method"] == self.runner_up:
                tag = "<- runner-up"
            if not row.get("available", True):
                tag = "no disponible"
            out.append(
                f"  {row['method']:4}{int(row.get('k') or 0):>4}"
                f"{float(row.get('ari') or 0.0):>8.3f}"
                f"{float(row.get('coverage') or 0.0):>7.3f}"
                f"{float(row.get('usable') or 0.0):>8.3f}"
                f"{float(row.get('noise_frac') or 0.0):>7.2f}"
                f"{float(row.get('tiny_frac') or 0.0):>7.2f}  "
                f"{float(row.get('max_class_share') or 0.0):>7.2f}  "
                f"{str(row.get('perturbation') or '-'):14}  {tag}"
            )
        for item in self.discarded:
            out.append(f"  [!] {item['method']} descartado: {item['why']}")
        if self.winner is None:
            out.append(
                "  [X] SIN CLASIFICACION (abstencion): "
                f"{self.reason}. El pipeline continua sin clases."
            )
            return out
        out.append(
            f"  => metodo elegido: {self.winner} "
            f"({METHOD_LABELS.get(self.winner, '?')})"
        )
        out.append(
            f"     ARI={self.winner_ari:.3f}  coverage={self.winner_coverage:.3f}  "
            f"usable={self.winner_usable_coverage:.3f}  K={self.winner_k}"
        )
        _win_row = next((r for r in self.all_metrics if r["method"] == self.winner), {})
        out.append(
            f"     clase mayor={float(_win_row.get('max_class_share') or 0.0):.2f} "
            f"de las asignadas (gate: >{self.max_class_share_max:.2f} = "
            "particion degenerada)"
        )
        if self.ambiguous and self.runner_up:
            out.append(
                f"  [!] AMBIGUO: {self.winner} vs {self.runner_up}, "
                f"|dARI|={abs(self.delta_ari):.3f} <= margin={self.margin:.3f}"
            )
        if self.criteria_conflict and self.runner_up:
            out.append(
                f"  [!] conflicto de criterios: {self.runner_up} tiene mas coverage "
                f"usable ({self.runner_up_usable_coverage:.3f} vs "
                f"{self.winner_usable_coverage:.3f}); el ARI manda pero la eleccion "
                "no es holgada"
            )
        return out
```

<!-- NEW:EXPLAIN -->

```python
    def explain_lines(self) -> List[str]:
        out = [
            f"METHOD SELECTION (rule {self.rule_version})",
            f"  dominancia sobre '{self.dominance_scale}': c=a*(1-nu) > "
            f"{self.max_class_share_max:.2f} | nulo={self.null_mode}"
            f"@{self.null_alpha:.2f} | margen {self.margin:.2f}",
            f"  {'met':4}{'K':>4}{'stab':>7}{'knd':>6}{'cov':>7}{'covS':>7}"
            f"{'usable':>8}{'noise':>7}{'dom':>6}{'domC':>7}{'nullp':>7}  "
            f"{'perturbacion':14}  veredicto",
        ]
        for row in self.all_metrics:
            tag = ""
            if row["method"] == self.winner:
                tag = "<- winner"
            elif row["method"] == self.runner_up:
                tag = "<- runner-up"
            if not row.get("available", True):
                tag = "no disponible"
            out.append(
                f"  {row['method']:4}{int(row.get('k') or 0):>4}"
                f"{_fmt(row.get('stability')):>7}"
                f"{str(row.get('stability_kind') or '-'):>6}"
                f"{float(row.get('coverage') or 0.0):>7.3f}"
                f"{_fmt(row.get('coverage_stable')):>7}"
                f"{float(row.get('usable') or 0.0):>8.3f}"
                f"{float(row.get('noise_frac') or 0.0):>7.2f}"
                f"{float(row.get('max_class_share') or 0.0):>6.2f}"
                f"{_fmt(row.get('dominance_corpus_eff')):>7}"
                f"{_fmt(row.get('null_p')):>7}  "
                f"{str(row.get('perturbation') or '-'):14}  {tag}"
            )
        for item in self.discarded:
            out.append(f"  [!] {item['method']} descartado: {item['why']}")
        if self.winner is None:
            out.append(
                "  [X] SIN CLASIFICACION (abstencion): "
                f"{self.reason}. El pipeline continua sin clases."
            )
            return out
        out.append(
            f"  => metodo elegido: {self.winner} "
            f"({METHOD_LABELS.get(self.winner, '?')})"
        )
        out.append(
            f"     estabilidad={_fmt(self.winner_stability, '.3f')} "
            f"({self.winner_stability_kind or '-'}; ARI={self.winner_ari:.3f})  "
            f"coverage={self.winner_coverage:.3f}  "
            f"usable={self.winner_usable_coverage:.3f}  K={self.winner_k}"
        )
        out.append(
            f"     nulo: p={_fmt(self.winner_null_p, '.3f')}"
        )
        _win_row = next((r for r in self.all_metrics if r["method"] == self.winner), {})
        out.append(
            f"     clase mayor={float(_win_row.get('max_class_share') or 0.0):.2f} "
            f"de las asignadas = {_fmt(_win_row.get('dominance_corpus_eff'))} del corpus "
            f"(gate: >{self.max_class_share_max:.2f} del corpus = "
            "particion degenerada)"
        )
        if self.ambiguous and self.runner_up:
            out.append(
                f"  [!] AMBIGUO: {self.winner} vs {self.runner_up}, "
                f"|dStab|={abs(self.delta_ari):.3f} <= margin={self.margin:.3f}"
            )
        if self.criteria_conflict and self.runner_up:
            if self.winner_stability_kind != self.runner_up_stability_kind:
                out.append(
                    "  [!] conflicto de criterios: "
                    f"{self.winner} mide '{self.winner_stability_kind}' y "
                    f"{self.runner_up} mide '{self.runner_up_stability_kind}'; "
                    "no estan en la misma escala (bootstrap = remuestreo, "
                    "perturbacion = pase tight/loose)"
                )
            if (
                math.isfinite(float(self.runner_up_usable_coverage))
                and self.runner_up_usable_coverage
                > self.winner_usable_coverage + 1e-12
            ):
                out.append(
                    f"  [!] conflicto de criterios: {self.runner_up} tiene mas "
                    f"coverage util ({self.runner_up_usable_coverage:.3f} vs "
                    f"{self.winner_usable_coverage:.3f}); la estabilidad manda "
                    "pero la eleccion no es holgada"
                )
        return out
```

### 7.8 Constantes, `_fmt`, `_null_class` y `_feasibility`

<!-- OLD:FEASIBILITY -->

```python
# ══════════════════════════════════════════════════════════════════════════
# Regla
# ══════════════════════════════════════════════════════════════════════════
def _feasibility(
    metric: MethodMetrics,
    coverage_floor: float,
    k_bounds: Tuple[int, int],
    noise_max: float = 0.50,
    max_class_share_max: float = 0.90,
) -> Optional[str]:
    """``None`` si el método es viable; si no, el motivo del descarte."""
    if not metric.available:
        return "no produjo particion"
    ari = float(metric.ari)
    if not math.isfinite(ari):
        return "ARI no finito"
    lo, hi = int(k_bounds[0]), int(k_bounds[1])
    if not (lo <= int(metric.k) <= hi):
        return f"K={int(metric.k)} fuera de [{lo}, {hi}]"
    if float(metric.noise_frac) > float(noise_max):
        return (
            f"ruido {float(metric.noise_frac):.3f} > max {float(noise_max):.2f} "
            "(mas de la mitad del corpus sin clasificar)"
        )
    dom = float(getattr(metric, "max_class_share", 0.0) or 0.0)
    if dom > float(max_class_share_max):
        return (
            f"clase dominante {dom:.2f} de las asignadas > max "
            f"{float(max_class_share_max):.2f} (particion degenerada: una clase"
            " y satelites)"
        )
    usable = metric.usable_coverage
    if usable < float(coverage_floor):
        detail = f"coverage util {usable:.3f} < umbral {coverage_floor:.2f}"
        tiny = float(getattr(metric, "tiny_frac", 0.0) or 0.0)
        if tiny > 0.0:
            detail += f" (coverage {float(metric.coverage):.3f} menos tiny {tiny:.3f})"
        return detail
    return None
```

<!-- NEW:FEASIBILITY -->

```python
# ══════════════════════════════════════════════════════════════════════════
# Regla
# ══════════════════════════════════════════════════════════════════════════
#: Backends que son solo diagnóstico: no compiten (``BACKEND_C.md`` §2).
_DIAGNOSTIC_BACKENDS: Tuple[str, ...] = ("hdbscan",)

#: Backends sin grados de libertad en el pase tight/loose: ``ari == 1.0`` por
#: construcción (``BACKEND_C.md`` §3), así que ese ARI no mide estabilidad.
_TAUTOLOGICAL_BACKENDS: Tuple[str, ...] = ("ward", "kmeans_spherical")

#: Modos de uso del contraste contra el nulo.
_NULL_MODES: Tuple[str, ...] = ("gate", "rank", "off")


def _fmt(value: Any, spec: str = ".2f") -> str:
    """Formatea un float que puede ser ``NaN`` sin romper la tabla."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    return "nan" if not math.isfinite(number) else format(number, spec)


def _null_class(metric: MethodMetrics, null_alpha: float) -> int:
    """``0`` = evidencia contra el nulo, ``1`` = sin medir, ``2`` = sin evidencia.

    Se usa solo con ``null_mode="rank"``; el gate compara el p-valor directo.
    """
    p = float(metric.null_p)
    if not math.isfinite(p):
        return 1
    return 0 if p <= float(null_alpha) else 2


def _feasibility(
    metric: MethodMetrics,
    coverage_floor: float,
    k_bounds: Tuple[int, int],
    noise_max: float = 0.50,
    max_class_share_max: float = 0.70,
    *,
    null_mode: str = "gate",
    null_alpha: float = 0.05,
    reject_tautological: bool = True,
    diagnostic_backends: Sequence[str] = _DIAGNOSTIC_BACKENDS,
) -> Optional[str]:
    """``None`` si el método es viable; si no, el motivo del descarte.

    Los cinco primeros parámetros conservan nombre y posición: ``9_selection_audit``
    los pasa posicionalmente y lee el umbral del payload. Todo lo de v4 es
    keyword-only, de modo que una llamada vieja sigue funcionando.
    """
    if not metric.available:
        return "no produjo particion"
    if metric.diagnostic_only or str(metric.backend) in diagnostic_backends:
        backend = str(metric.backend or "?")
        return (
            f"backend de diagnostico {backend} (no seleccionable: no controla K"
            " y su ARI alto es artefacto del colapso)"
        )
    if not math.isfinite(float(metric.stability)):
        return "estabilidad no finita"
    lo, hi = int(k_bounds[0]), int(k_bounds[1])
    if not (lo <= int(metric.k) <= hi):
        return f"K={int(metric.k)} fuera de [{lo}, {hi}]"
    noise = float(metric.noise_frac)
    if noise > float(noise_max):
        return (
            f"ruido {noise:.3f} > max {float(noise_max):.2f} "
            "(mas de la mitad del corpus sin clasificar)"
        )
    share = float(metric.max_class_share or 0.0)
    dom = float(metric.dominance_corpus_effective)
    if dom > float(max_class_share_max):
        return (
            f"clase dominante {share:.2f} de las asignadas = {dom:.2f} del "
            f"corpus > max {float(max_class_share_max):.2f} (particion "
            f"degenerada: una clase y satelites; ruido {noise:.2f} descontado)"
        )
    usable = float(metric.usable_effective)
    if usable < float(coverage_floor):
        detail = f"coverage util {usable:.3f} < umbral {coverage_floor:.2f}"
        stable = float(metric.coverage_stable)
        if math.isfinite(stable):
            detail += f" (coverage estable {stable:.3f}"
        else:
            detail += f" (coverage {float(metric.coverage):.3f}"
        tiny = float(metric.tiny_frac or 0.0)
        if tiny > 0.0:
            detail += f" menos tiny {tiny:.3f}"
        return detail + ")"
    if (
        reject_tautological
        and metric.stability_kind == "perturbation"
        and str(metric.backend) in _TAUTOLOGICAL_BACKENDS
    ):
        return (
            f"estabilidad tautologica: backend {metric.backend} no tiene grados"
            " de libertad en el pase tight/loose (ARI=1.0 por construccion);"
            " falta bootstrap_ari"
        )
    if null_mode == "gate":
        p = float(metric.null_p)
        if math.isfinite(p) and p > float(null_alpha):
            return (
                f"sin contraste contra el nulo: p={p:.3f} > alpha="
                f"{float(null_alpha):.2f} (las clases no separan mas que"
                " etiquetas permutadas)"
            )
    return None
```

### 7.9 `select_method`

<!-- OLD:SELECT -->

```python
def select_method(
    metrics: Sequence[MethodMetrics],
    margin: float = 0.02,
    coverage_floor: float = 0.15,
    k_bounds: Tuple[int, int] = (3, 20),
    min_class_share: float = 0.01,
    noise_max: float = 0.50,
    max_class_share_max: float = 0.90,
) -> Selection:
    """Aplica la regla ``select-v3`` y devuelve la ``Selection``.

    Determinista: con los mismos ``metrics`` devuelve siempre lo mismo,
    incluso ante ARI y coverage idénticos (desempate por orden A<B<C).
    """
    rows = list(metrics)
    if not rows:
        return Selection(
            winner=None,
            winner_ari=float("nan"),
            winner_coverage=float("nan"),
            winner_k=0,
            abstained=True,
            reason="no se recibieron metricas",
            margin=margin,
            coverage_floor=coverage_floor,
            min_class_share=min_class_share,
            noise_max=noise_max,
            max_class_share_max=max_class_share_max,
            k_bounds=tuple(k_bounds),
        )

    discarded: List[Dict[str, Any]] = []
    viable: List[MethodMetrics] = []
    for m in rows:
        why = _feasibility(m, coverage_floor, k_bounds, noise_max, max_class_share_max)
        if why is None:
            viable.append(m)
        else:
            discarded.append({"method": m.method, "why": why})

    all_rows = [m.as_row() for m in rows]

    if not viable:
        return Selection(
            winner=None,
            winner_ari=float("nan"),
            winner_coverage=float("nan"),
            winner_k=0,
            abstained=True,
            reason="ningun metodo cumple viabilidad",
            discarded=discarded,
            all_metrics=all_rows,
            margin=margin,
            coverage_floor=coverage_floor,
            min_class_share=min_class_share,
            noise_max=noise_max,
            max_class_share_max=max_class_share_max,
            k_bounds=tuple(k_bounds),
        )

    ranked = sorted(
        viable,
        key=lambda m: (
            -float(m.ari),
            -m.usable_coverage,
            _METHOD_ORDER.get(m.method, 99),
        ),
    )
    win = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None

    delta_ari = float(win.ari - runner.ari) if runner else float("nan")
    delta_cov = (
        float(win.usable_coverage - runner.usable_coverage) if runner else float("nan")
    )
    ambiguous = bool(runner is not None and abs(delta_ari) <= float(margin))
    conflict = bool(
        runner is not None and runner.usable_coverage > win.usable_coverage + 1e-12
    )

    reason = "max ARI intra-metodo"
    if runner is not None and abs(delta_ari) <= 1e-12:
        reason = "empate de ARI; desempate por coverage util"

    return Selection(
        winner=win.method,
        winner_ari=float(win.ari),
        winner_coverage=float(win.coverage),
        winner_k=int(win.k),
        winner_usable_coverage=float(win.usable_coverage),
        runner_up=runner.method if runner else None,
        runner_up_ari=float(runner.ari) if runner else float("nan"),
        runner_up_coverage=float(runner.coverage) if runner else float("nan"),
        runner_up_k=int(runner.k) if runner else 0,
        runner_up_usable_coverage=(
            float(runner.usable_coverage) if runner else float("nan")
        ),
        delta_ari=delta_ari,
        delta_coverage=delta_cov,
        ambiguous=ambiguous,
        criteria_conflict=conflict,
        abstained=False,
        reason=reason,
        discarded=discarded,
        all_metrics=all_rows,
        margin=float(margin),
        coverage_floor=float(coverage_floor),
        min_class_share=float(min_class_share),
        noise_max=float(noise_max),
        max_class_share_max=float(max_class_share_max),
        k_bounds=tuple(k_bounds),
    )
```

<!-- NEW:SELECT -->

```python
def select_method(
    metrics: Sequence[MethodMetrics],
    margin: float = 0.02,
    coverage_floor: float = 0.15,
    k_bounds: Tuple[int, int] = (3, 20),
    min_class_share: float = 0.01,
    noise_max: float = 0.50,
    max_class_share_max: float = 0.70,
    null_mode: str = "gate",
    null_alpha: float = 0.05,
    reject_tautological: bool = True,
    diagnostic_backends: Sequence[str] = _DIAGNOSTIC_BACKENDS,
) -> Selection:
    """Aplica la regla ``select-v4`` y devuelve la ``Selection``.

    Determinista: con los mismos ``metrics`` devuelve siempre lo mismo, incluso
    ante estabilidad y coverage idénticas (desempate por orden A<B<C). No lanza
    excepción con métricas incompletas: todo campo nuevo cae a ``nan`` = "no
    disponible" y su señal no interviene.
    """
    if null_mode not in _NULL_MODES:
        raise ValueError(
            f"null_mode={null_mode!r} no valido; usar uno de {_NULL_MODES}"
        )
    rows = list(metrics)
    common: Dict[str, Any] = dict(
        margin=float(margin),
        coverage_floor=float(coverage_floor),
        min_class_share=float(min_class_share),
        noise_max=float(noise_max),
        max_class_share_max=float(max_class_share_max),
        k_bounds=tuple(k_bounds),
        null_mode=str(null_mode),
        null_alpha=float(null_alpha),
    )
    if not rows:
        return Selection(
            winner=None,
            winner_ari=float("nan"),
            winner_coverage=float("nan"),
            winner_k=0,
            abstained=True,
            reason="no se recibieron metricas",
            **common,
        )

    discarded: List[Dict[str, Any]] = []
    viable: List[MethodMetrics] = []
    for m in rows:
        why = _feasibility(
            m,
            coverage_floor,
            k_bounds,
            noise_max,
            max_class_share_max,
            null_mode=null_mode,
            null_alpha=null_alpha,
            reject_tautological=reject_tautological,
            diagnostic_backends=diagnostic_backends,
        )
        if why is None:
            viable.append(m)
        else:
            discarded.append({"method": m.method, "why": why})

    all_rows = [m.as_row() for m in rows]

    if not viable:
        return Selection(
            winner=None,
            winner_ari=float("nan"),
            winner_coverage=float("nan"),
            winner_k=0,
            abstained=True,
            reason="ningun metodo cumple viabilidad",
            discarded=discarded,
            all_metrics=all_rows,
            **common,
        )

    def _rank_key(m: MethodMetrics) -> Tuple[Any, ...]:
        head: Tuple[Any, ...] = ()
        if null_mode == "rank":
            head = (_null_class(m, null_alpha),)
        return head + (
            -float(m.stability),
            -float(m.usable_effective),
            _METHOD_ORDER.get(m.method, 99),
        )

    ranked = sorted(viable, key=_rank_key)
    win = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None

    # ``delta_ari`` conserva el nombre por compatibilidad; con bootstrap_ari
    # disponible mide la diferencia de estabilidad, no la de ARI.
    delta_ari = float(win.stability - runner.stability) if runner else float("nan")
    delta_cov = (
        float(win.usable_effective - runner.usable_effective)
        if runner
        else float("nan")
    )
    ambiguous = bool(runner is not None and abs(delta_ari) <= float(margin))
    kinds_differ = bool(
        runner is not None and win.stability_kind != runner.stability_kind
    )
    coverage_conflict = bool(
        runner is not None
        and runner.usable_effective > win.usable_effective + 1e-12
    )
    conflict = bool(runner is not None and (kinds_differ or coverage_conflict))

    if runner is None:
        reason = "unico metodo viable"
    elif abs(delta_ari) <= 1e-12:
        if abs(delta_cov) > 1e-12:
            reason = "empate de estabilidad; desempate por coverage util"
        else:
            reason = "empate de estabilidad; desempate por orden canonico"
    else:
        reason = f"max estabilidad intra-metodo ({win.stability_kind})"
    if null_mode == "rank" and runner is not None:
        if _null_class(win, null_alpha) != _null_class(runner, null_alpha):
            reason = (
                "evidencia contra el nulo (0=si, 1=sin medir, 2=no); despues "
                f"estabilidad intra-metodo ({win.stability_kind})"
            )

    return Selection(
        winner=win.method,
        winner_ari=float(win.ari),
        winner_coverage=float(win.coverage),
        winner_k=int(win.k),
        # ``usable`` del ganador = cobertura efectiva (coincide con v3 cuando
        # ``coverage_stable`` no existe).
        winner_usable_coverage=float(win.usable_effective),
        runner_up=runner.method if runner else None,
        runner_up_ari=float(runner.ari) if runner else float("nan"),
        runner_up_coverage=float(runner.coverage) if runner else float("nan"),
        runner_up_k=int(runner.k) if runner else 0,
        runner_up_usable_coverage=(
            float(runner.usable_effective) if runner else float("nan")
        ),
        delta_ari=delta_ari,
        delta_coverage=delta_cov,
        ambiguous=ambiguous,
        criteria_conflict=conflict,
        abstained=False,
        reason=reason,
        discarded=discarded,
        all_metrics=all_rows,
        winner_stability=float(win.stability),
        winner_stability_kind=win.stability_kind,
        winner_null_p=float(win.null_p),
        winner_dominance_corpus=float(win.dominance_corpus_effective),
        runner_up_stability=float(runner.stability) if runner else float("nan"),
        runner_up_stability_kind=runner.stability_kind if runner else "",
        **common,
    )
```

### 7.10 `load_selection`: validar `rule_version` (bug obligatorio)

<!-- OLD:LOAD -->

```python
def load_selection(
    path: str,
    corpus_fingerprint: Optional[str] = None,
    uce_target_size: Optional[int] = None,
) -> Optional[Selection]:
    """Carga el snapshot. Devuelve ``None`` si falta o si no corresponde al
    corpus/segmentación pedidos (invalidación explícita)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    ctx = payload.get("context") or {}
    if corpus_fingerprint is not None:
        stored = ctx.get("corpus_fingerprint")
        if stored is not None and stored != corpus_fingerprint:
            return None
    if uce_target_size is not None:
        stored = ctx.get("uce_target_size")
        if stored is not None and int(stored) != int(uce_target_size):
            return None
    sel = payload.get("selection")
    if not isinstance(sel, dict):
        return None
    return Selection.from_dict(sel)
```

<!-- NEW:LOAD -->

```python
#: Motivo del último snapshot ignorado por ``load_selection`` (o ``None``).
_LAST_LOAD_REJECTION: Optional[Dict[str, Any]] = None


def last_load_rejection() -> Optional[Dict[str, Any]]:
    """Motivo por el que ``load_selection`` ignoró el último snapshot."""
    return dict(_LAST_LOAD_REJECTION) if _LAST_LOAD_REJECTION else None


def load_selection(
    path: str,
    corpus_fingerprint: Optional[str] = None,
    uce_target_size: Optional[int] = None,
    expected_rule_version: Optional[str] = None,
    explain: Optional[List[str]] = None,
) -> Optional[Selection]:
    """Carga el snapshot. Devuelve ``None`` si no corresponde (invalidación).

    Tres invalidaciones explícitas, en este orden:

    1. ``rule_version`` distinta de ``expected_rule_version`` (por defecto la
       del módulo). Reutilizar un snapshot ``select-v3`` bajo ``select-v4``
       aplicaría una decisión tomada con otra regla **y parecería válida**,
       porque el corpus y el target coinciden: es exactamente el fallo que costó
       tiempo. Con ``expected_rule_version=""`` se desactiva el chequeo (solo
       para leer historia).
    2. ``corpus_fingerprint`` distinto.
    3. ``uce_target_size`` distinto.

    Cada rechazo se registra en ``last_load_rejection()`` y en el log del
    módulo; si se pasa ``explain``, se añade una línea legible.
    """
    global _LAST_LOAD_REJECTION
    _LAST_LOAD_REJECTION = None

    def _reject(why: str, declared: Any = None, expected: Any = None) -> None:
        global _LAST_LOAD_REJECTION
        _LAST_LOAD_REJECTION = {
            "path": str(path),
            "why": why,
            "declared": "" if declared is None else str(declared),
            "expected": "" if expected is None else str(expected),
        }
        logging.getLogger(__name__).info("snapshot ignorado: %s", why)
        if explain is not None:
            explain.append(f"[!] snapshot ignorado ({path}): {why}")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        _reject(f"no se pudo leer ({type(exc).__name__})")
        return None

    sel = payload.get("selection")
    if not isinstance(sel, dict):
        _reject("payload sin 'selection'")
        return None

    expected = (
        RULE_VERSION if expected_rule_version is None else str(expected_rule_version)
    )
    if expected:
        declared = payload.get("rule_version", sel.get("rule_version"))
        if declared != expected:
            _reject(
                f"rule_version {declared!r} != {expected!r} (snapshot de otra regla)",
                declared,
                expected,
            )
            return None

    ctx = payload.get("context") or {}
    if corpus_fingerprint is not None:
        stored = ctx.get("corpus_fingerprint")
        if stored is not None and stored != corpus_fingerprint:
            _reject(f"corpus_fingerprint {stored!r} != {corpus_fingerprint!r}")
            return None
    if uce_target_size is not None:
        stored = ctx.get("uce_target_size")
        try:
            mismatch = int(stored) != int(uce_target_size)
        except (TypeError, ValueError):
            mismatch = True
        if mismatch:
            _reject(f"uce_target_size {stored!r} != {uce_target_size!r}")
            return None
    return Selection.from_dict(sel)
```

### 7.11 Resumen de aplicacion

| #   | bloque             | lineas actuales | tipo      |
| --- | ------------------ | --------------- | --------- |
| 1   | `DOCSTRING`        | 17-40           | reemplazo |
| 2   | `IMPORTS`          | 78-79           | inserción |
| 3   | `RULE_VERSION`     | 86              | reemplazo |
| 4   | `METRIC_FIELDS`    | 121-137         | reemplazo |
| 5   | `SELECTION_FIELDS` | 165-168         | reemplazo |
| 6   | `FROM_DICT`        | 191-200         | inserción |
| 7   | `EXPLAIN`          | 207-265         | reemplazo |
| 8   | `FEASIBILITY`      | 271-309         | reemplazo |
| 9   | `SELECT`           | 312-423         | reemplazo |
| 10  | `LOAD`             | 476-500         | reemplazo |

Ningún bloque se solapa. Los `OLD` se verificaron con un script contra el
archivo actual: cada uno aparece exactamente una vez.

---

## 8. Compatibilidad

### 8.1 `9_selection_audit.py` (otro agente lo rediseña)

| lo que usa el auditor                                    | estado en v4                                                                                                          |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `_feasibility(metric, floor, k_bounds, noise_max, gate)` | **igual**: 5 posicionales, mismos nombres, misma posición                                                             |
| `Selection.max_class_share_max`                          | **igual nombre**; default 0.90 → 0.70                                                                                 |
| `hasattr(MethodMetrics, "dominance_corpus")`             | `True` (campo con default)                                                                                            |
| umbral efectivo leído del payload                        | funciona: `explain_lines` y `_feasibility` lo usan                                                                    |
| `all_metrics[i].maximums`                                | se conservan; se **añaden** `usable_eff`, `coverage_effective`, `dominance_corpus_eff`, `stability`, `stability_kind` |

Cambios de semántica que el auditor debe conocer (no rompen firmas):

- `winner_usable_coverage` / `runner_up_usable_coverage` / `delta_coverage`
  pasan a ser **efectivos** (`usable_effective`). Coinciden con v3 cuando
  `coverage_stable` es `nan`, que es el caso de todo payload actual.
- `delta_ari` conserva el nombre pero mide `stability` (bootstrap si existe).
- `discarded[*].why` cambia de redacción para dominancia: ahora incluye `a`,
  `c` y `nu`. `AUDIT_V4.md` clasifica por familia de motivo, no por substring
  exacto, así que sigue funcionando.

Para desactivar el chequeo nuevo en el replay (por ejemplo, al auditar un
payload de otra regla), el auditor puede pasar `expected_rule_version=""` a
`load_selection`; hoy no lo llama, así que no requiere cambios.

### 8.2 Tests existentes que rompen

`tests/test_method_selection.py` (27 tests, 2.4 s). Rompen **dos**:

| test                                      | por que rompe                                                       |
| ----------------------------------------- | ------------------------------------------------------------------- |
| `test_dominancia_089_no_descarta`         | `0.89` sin ruido ahora es `c = 0.89 > 0.70`: degenerada             |
| `test_dominancia_gate_aparece_en_explain` | busca `"gate: >0.90"`; ahora el gate es sobre el corpus y vale 0.70 |

Siguen pasando (verificado a mano contra el diff): `test_dominancia_092_descarta`
(`0.92 > 0.70` y `"0.92"` sigue apareciendo), `test_noise_bajo_no_descarta`,
`test_noise_alto_descarta`, `test_tiny_reduce_coverage_util`,
`test_k_bounds_discard`, `test_desempate_por_coverage_util`,
`test_exact_tie_is_deterministic`, `test_tie_broken_by_coverage`,
`test_criteria_conflict_and_ambiguity`, `test_clear_winner_is_not_ambiguous`,
`TestSerialization.*`, `TestPersistence.*`, `TestAudit.*`. Motivo: sin campos
nuevos, `stability == ari`, `usable_effective == usable_coverage` y el gate del
nulo no interviene (`null_p is nan`).

Reemplazos exactos (mismo espiritu, nueva escala). Reemplazan a los dos que
rompen y añaden el test de regresión del bug del denominador:

```python
    def test_dominancia_089_con_ruido_no_descarta(self):
        """0.89 de las asignadas con 30% de ruido = 0.62 del corpus: valida."""
        c = MethodMetrics(
            "C", 3, 0.785, 0.371, noise_frac=0.30, max_class_share=0.89
        )
        a = _m("A", 4, 0.250, 0.574)
        sel = select_method([a, c])
        assert sel.winner == "C"
        assert not any("dominante" in d["why"] for d in sel.discarded)
        assert sel.winner_dominance_corpus == pytest.approx(0.623)

    def test_dominancia_gate_aparece_en_explain(self):
        """El umbral se imprime en unidades del corpus (0.70), no cableado."""
        c = MethodMetrics("C", 3, 0.785, 0.371, max_class_share=0.60)
        sel = select_method([c])
        text = sel.explain_text()
        assert "clase mayor=0.60" in text
        assert "= 0.60 del corpus" in text
        assert "gate: >0.70 del corpus" in text
        assert "0.90" not in text

    def test_dominancia_ignora_el_ruido_en_el_denominador(self):
        """Misma particion, misma c: el ruido mueve ``a`` pero no el veredicto."""
        limpia = MethodMetrics("C", 3, 0.785, 0.371, max_class_share=442 / 528)
        ruidosa = MethodMetrics(
            "C",
            3,
            0.785,
            0.371,
            noise_frac=63 / 528,
            max_class_share=442 / 465,
        )
        a = _m("A", 4, 0.250, 0.574)
        for m in (limpia, ruidosa):
            sel = select_method([a, m])
            assert sel.winner == "A", m.max_class_share
            assert any("dominante" in d["why"] for d in sel.discarded)
```

`tests/test_select_mode_objective.py::TestDominanceGateWiring` sigue pasando
(`0.52` con `nu=0` → `c = 0.52`); su docstring queda conceptualmente
desactualizado ("diluir con `-1` no debe valer"): ahora se diluye de forma
explícita sobre el corpus. Conviene reescribir el comentario, no el assert.

### 8.3 `tests/test_select_v4.py` (nuevo, 24 tests)

Se activa solo cuando `RULE_VERSION == "select-v4"`; hasta entonces **se salta**
(la suite queda verde y el archivo documenta el contrato). Cubre:

| requisito del encargo                            | test                                              |
| ------------------------------------------------ | ------------------------------------------------- |
| (a) C denso ya no gana por construccion          | `test_denso_no_gana_si_su_bootstrap_es_peor`      |
| (a) y con nulo malo                              | `test_denso_no_gana_sin_evidencia_contra_el_nulo` |
| (ii) C con hdbscan nunca gana                    | `TestDiagnosticoC.*` (3 tests)                    |
| (iii) 0.837 con `nu=0` descarta por dominancia   | `test_dominancia_corpus_0837_descarta`            |
| (iv) abstención                                  | `TestAbstencion.*` (3 tests)                      |
| (v) conflicto cuando los kinds difieren          | `test_kinds_distintos_marcan_conflicto`           |
| (vi) snapshot con `rule_version` vieja se ignora | `TestSnapshotVersion.*` (3 tests)                 |
| (vii) métricas mínimas no lanzan                 | `TestMetricasMinimas.*` (3 tests)                 |

---

## 9. Riesgos y limites (declarados, no ocultos)

1. **`null_p` puede ser un nulo debil.** Si viene de permutación de etiquetas,
   el gate es casi vacuo y **no** produce abstención sobre ruido
   (§4.1). La regla esta preparada (`null_mode`), pero el comportamiento real
   depende del productor. Mientras no exista `STABILITY_NULL.md`, no se puede
   afirmar que el gate discrimine.
2. **La lista de backends tautologicos es conocimiento del código, no del
   corpus.** Si `ward` o `kmeans_spherical` ganan un pase con grados de libertad
   real, hay que sacarlos de `_TAUTOLOGICAL_BACKENDS`; si no, se descartarían
   particiones validas (`fd`).
3. **Escalas no comparables.** `stability` no es comparable entre corridas con
   distinto `n`, `subsample_frac`, `n_boot` ni entre kinds. El flag
   `criteria_conflict` avisa, no corrige.
4. **`dominance_corpus` como campo puede enmascarar un error** del productor si
   lo llena con un denominador equivocado; los conteos (`max_class_size`,
   `n_total`) tienen prioridad por eso, y el auditor comprueba la identidad.
5. **Cobertura efectiva cambia el significado de dos campos publicos**
   (`winner_usable_coverage`, `delta_coverage`). Coinciden con v3 cuando
   `coverage_stable` es `nan`.
6. **`hdbscan` se bloquea por defecto aunque el flag venga mal puesto**
   (fail-closed). El interruptor de escape de `BACKEND_C.md` §2 exige pasar
   `diagnostic_backends=()` además de `diagnostic_only=False`.
7. **`null_mode` invalido lanza `ValueError`.** Es un error de programación, no
   de datos: la garantía "nunca excepcion con métricas incompletas" se refiere
   a los campos, no a los parámetros de la regla.
8. **Sin cambios en `main_workflow.py`.** Este diff no hace que el pipeline
   _produzca_ las señales nuevas; sólo define cómo se consumen. Mientras no
   lleguen, la regla es equivalente a v3 salvo el gate de dominancia (que sí
   cambia de escala y ya corrige el bug del probe) y el bloqueo de `hdbscan`.
9. **`data/method_selection.json` de la corrida en curso.** Tras aplicar el
   diff, `load_selection` lo rechazará por `rule_version` (motivo registrado),
   así que no hay que borrarlo. Si se quiere conservar como registro, moverlo a
   `data/method_selection_v3.json`.

---

## 10. Reproducibilidad

```
python scripts/playground/select_v4_arena.py                 # 2.5 s, exit 0
python scripts/playground/select_v4_arena.py --use-module    # tras el diff
python -m pytest tests/test_select_v4.py -q                  # 24 tests
python -m pytest tests/test_method_selection.py -q           # 27 tests (2 roturas)
```

- `select_v4_arena.py` es autocontenido: stdlib puro, sin numpy, sin red, sin
  Docker, sin BD. No escribe en `data/`, `output/`, `models/` ni `src/`.
- Semillas no aplican: los escenarios son tablas de métricas fijas, así que el
  resultado es determinista bit a bit.
- La columna `v3` se calcula con el módulo real mientras siga siendo
  `select-v3`; tras aplicar el diff desaparece (`-`) y el chequeo se traslada a
  `--use-module`, que falla si la copia canónica y el módulo divergen.
- Salida ASCII puro (`->`, `[ok]`, `[!]`, `[X]`, `=>`), sin box-drawing ni emojis.
