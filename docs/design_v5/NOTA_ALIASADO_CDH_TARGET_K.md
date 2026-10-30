# `cdh_target_k` está aliasado: el plateau de K

Estado: hallazgo de la corrida v6 (`reii_probe_v6_*_42c630f03946b0cd`).
No es un defecto del optimizador de Optuna ni del gate de selección: es una
propiedad del mapeo `cdh_target_k → K realizado` que hay que declarar antes de
interpretar correlaciones o bordes sobre ese parámetro.

## 1. Qué es `cdh_target_k`

No es "el número de clases". Es un **piso de tamaño de hoja** expresado como K
objetivo:

```
frac = 1 / (2 * K)          # _cdh_frac_from_params, main_workflow.py
min_cluster_size_cdh = frac # piso duro de cada clase terminal
```

El árbol CDH parte equilibrado (K ≈ 2^profundidad), así que pedir K hojas exige
un piso de `n/(2K)`. Lo que emerge después depende de dos cosas más: el gate
`r2_dense` (que corta la recursión cuando un split no explica varianza
suficiente) y la partición cruda de A1 en la etapa `wc_only`.

Consecuencia: **`cdh_target_k` y `K` son variables distintas.** El código ya lo
sabe y lo registra en `user_attrs` de cada trial:

| attr | significado |
| --- | --- |
| `k_target` | `cdh_target_k` muestreado por Optuna |
| `k_ach` | K realmente logrado (etapa 2: `winner_k`; etapa 1: K de la partición cruda de A1) |
| `k_shortfall` | `k_target - k_ach` (negativo = el árbol pasó del objetivo) |
| `k_stable` | nº de clusters con UCEs estables |

## 2. Evidencia v6: el parámetro deja de identificar arriba de cierto punto

Etapa 1 (`wc_only`), 20 trials. `K` logrado contra `kT` pedido:

| kT | K logrado (trials) |
| --- | --- |
| 3 | 4 (×2) |
| 4 | 5 |
| 6 | 5, 8, 8, 8 |
| 8 | 10, 12, 12, 13 |
| 9 | 15 |
| 10 | 16, 16, 16, 16 |
| 11 | 16 |
| 12 | 16, 16 |

`k_shortfall` es negativo en 19 de 20 trials (el árbol **pasa** del objetivo) y
`K` se satura en **16** para `kT ∈ [10, 12]`. En la etapa 2 la saturación es más
baja (**11–13** para `kT ≥ 8`) porque `k_ach` usa `winner_k`, no la partición
cruda.

Lectura (mecanismo, no hecho medido): por debajo de `kT ≈ 8` manda el piso
`1/(2K)` y `K` crece con `kT`; por encima manda el gate `r2_dense`, que corta
antes, y `kT` deja de mover `K`.

## 3. El síntoma que delata el alias: objetivos idénticos con `kT` distinto

En la etapa 2 hay tres pares de trials con **triple objetivo idéntico** y
`cdh_target_k` distinto:

| trials | objetivo (coverage, ARI, −K) | `kT` | `K` |
| --- | --- | --- | --- |
| 3, 8 | (0.456439, 0.257283, −11.0) | 11 vs 8 | 11, 11 |
| 6, 18 | (0.376894, 0.242804, −11.0) | 10 vs 8 | 11, 11 |
| 10, 19 | (0.412879, 0.311955, −12.0) | 9 vs 8 | 12, 12 |

Son 6 de 20 trials (30 % del presupuesto de la etapa 2) gastados en 3 desenlaces
distintos. El diagnóstico ya lo detecta y lo imprime como
`Puntos de objetivo repetidos por ≥2 trials con parámetros distintos: 3`
(`3_diagnose_optuna_db.py`).

## 4. Por qué esto invalida leer ρ(`cdh_target_k`, ·) sin estratificar

La correlación de Spearman que imprime el diagnóstico mezcla dos regímenes: la
zona donde el piso manda (`kT ≤ 8`, `K` responde) y el plateau (`kT ≥ 10`, `K`
constante). En el plateau el parámetro no aporta información, solo ruido de
orden. Por eso el mismo ρ puede cambiar de signo entre etapas sin que cambie
nada del corpus — es lo que muestra el bloque `SIGN FLIP ENTRE ETAPAS`:

```
cdh_target_k   obj[0]  rho_etapa1=-0.61  rho_etapa2=-0.14
cdh_target_k   obj[1]  rho_etapa1=-0.70  rho_etapa2=+0.41   <<< INVIERTE
tsj            obj[1]  rho_etapa1=-0.35  rho_etapa2=+0.42   <<< INVIERTE
```

Parte de esa inversión es artefacto del plateau, no señal sobre el objetivo.

Además el tercer objetivo (`-K`) se calcula sobre `k_ach`, así que dos trials que
solo difieren en `kT` dentro del plateau puntúan igual: el frente de Pareto los
trata como duplicados y el knee puede caer en cualquiera de ellos.

## 5. Regla de lectura pre-registrada (para el próximo re-tuneo)

Se declara **antes** de mirar el próximo estudio:

1. **`kT` no se interpreta como "número de clases"** en el texto ni en las
   tablas del paper. Se reporta `k_target` junto a `k_ach` y `k_shortfall`.
2. **ρ(`cdh_target_k`, ·) solo se interpreta si el régimen es identificable.**
   Criterio: el plateau se declara si (a) existe un valor de `K` que aparece en
   ≥3 trials con `kT` distintos, o (b) `|k_shortfall| ≥ 4` en más del 30 % de
   los trials. Si se cumple, el ρ de ese parámetro se marca `no identificable`
   y no se usa para justificar bordes ni para elegir etapa.
3. **El alias no se "arregla" cambiando el espacio** en esta iteración. Se
   declara y se instrumenta (abajo). Cambiar el rango de `kT` sin medir
   rompería la comparabilidad con v4/v5/v6.

## 6. Instrumentación propuesta (barata, sin tocar el pipeline)

- `3_diagnose_optuna_db.py`: añadir al bloque de alias la columna `kT` de cada
  trial del par (ya se tienen los `trial.params`), y un contador
  `plateau: K=<v>, kT=<lista>` cuando se cumple el criterio 2.
- Tabla de régimen en el informe: `kT → K` en una línea por etapa, como la §2.
- Nada de esto cambia el exit code ni filtra trials: es diagnóstico, igual que
  el bloque de sign flip.

## 7. Qué queda pendiente

- Confirmar el mecanismo del plateau con una ablación de una sola variable:
  fijar `gap=2, tsj=3, coref_weight=0.1` (los del ganador) y barrer
  `kT ∈ {3, 6, 8, 10, 12}` sin optimizador. Son 5 corridas cortas y separa
  "piso" de "gate r2_dense". No está hecho: la nota lo deja como el experimento
  que sostendría la lectura de §2.
