# `reii.diag` — CLI de diagnóstico

Los análisis de diagnóstico que antes eran scripts sueltos (`2_…` … `9_…`) viven ahora en un
paquete con una única puerta de entrada. Nada de esto se importa desde `main_workflow`, así que
no añade coste al pipeline (ni carga Stanza salvo que un comando lo necesite de verdad).

## Cómo se ejecuta

```bash
python -m reii.diag                          # lista los comandos  (o: python -m reii.diag.cli)
python -m reii.diag optuna-db --help         # flags de un comando
python diag.py optuna-db --help              # host SIN `pip install -e .` (shim de la raíz)
reii-diag optuna-db --help                   # tras `pip install -e .` (entry point del pyproject)
```

Dentro del contenedor (donde `reii` está instalado con `pip install -e .` y `./src` está montado):

```bash
docker compose --profile batch run --rm workflow python -u -m reii.diag <comando> ...
```

`python -m reii.diag` funciona porque `src/reii/diag/__main__.py` delega en `cli.main`. En el host,
sin instalar el paquete, usa `python diag.py`: ese shim añade `src/` a `sys.path`.

## Comandos

| comando           | qué hace                                                                                                | lee                                             | escribe                                                                       | riesgo                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `optuna-db`       | radiografía de un estudio: centinelas, `gate_reason`, frente de Pareto, ρ parámetro↔objetivo, sign flip | `data/reii_optuna.db` (read-only)               | nada (stdout)                                                                 | ninguno                                                        |
| `uce-size`        | forma y coste de la segmentación por `uce_target_size`; extrapola coste                                 | `data/diag/bench_uce*.log`                      | `data/diag/uce_size_bench.json`                                               | lanza el pipeline **solo** con `--force-run` (~2 h por tamaño) |
| `trials`          | estabilidad del optimizador entre semillas TPE; ¿los ganadores tocan bordes?                            | `data/reii_optuna.db` + `data/best_params.json` | mueve/restaura `best_params.json`, escribe `data/diag/best_params_seed*.json` | **alto**: lanza `--optimize-only` por semilla (horas)          |
| `preflight`       | ¿`best_params`, estudio y selección son del corpus en disco? Gates G1/G2                                | artefactos en `data/`                           | nada sin `--exec`                                                             | **alto** solo con `--exec` (lanza el plan)                     |
| `db-hygiene`      | inventario / backup / borrado / vacuum de la BD de Optuna                                               | `data/reii_optuna.db`                           | copias `.bak-*`; **borra** con `--delete`                                     | **alto** solo con `--delete`/`--vacuum`                        |
| `multiverse`      | una fila por especificación × target: retención y niveles de calidad                                    | PostgreSQL (`--dsn`)                            | `--out` (JSON)                                                                | ninguno (solo lee)                                             |
| `selection-audit` | ¿la elección de método (A/B/C) es estable entre targets vecinos?                                        | `data/method_selection*.json(l)`                | `--json` (opcional)                                                           | ninguno                                                        |
| `term-metrics`    | φ sin Yates, flag FDR y plano factorial F9 desde el artefacto persistido                                | `data/workflow_data.json`                       | `data/diag/paper_terms.json`, `--figure`                                      | ninguno (solo lee)                                             |

Todos los comandos imprimen en ASCII puro (`[ok]`, `[!]`, `[X]`, `->`): la consola `cp1252` de
Windows no tiene los glifos de check ni de flecha.

## De dónde viene cada uno

| antes (raíz / `src/`)          | ahora                                           |
| ------------------------------ | ----------------------------------------------- |
| `3_diagnose_optuna_db.py`      | `python -m reii.diag optuna-db`                 |
| `4_measure_uce_target_size.py` | `python -m reii.diag uce-size`                  |
| `2_run_300_trials.py`          | `python -m reii.diag trials`                    |
| `6_preflight.py`               | `python -m reii.diag preflight`                 |
| `src/6_db_hygiene.py`          | `python -m reii.diag db-hygiene`                |
| `src/7_quality_multiverse.py`  | `python -m reii.diag multiverse`                |
| `src/9_selection_audit.py`     | `python -m reii.diag selection-audit`           |
| (nuevo, sin predecesor)        | `python -m reii.diag term-metrics`              |
| `diag_common.py`               | `reii/diag/common.py` (eliminado el de la raíz) |
| `5_s2_power_analysis.py`       | archivado en `tools/archive/diag/` (ver abajo)  |
| `8_probe_b_permissive.py`      | archivado en `tools/archive/diag/` (ver abajo)  |

Los dos archivados **no** son herramientas: fueron experimentos de una sola vez cuya conclusión ya
está tomada y registrada. Se conservan por trazabilidad, no para volver a ejecutarlos.

## Resultados ya fijados (no repetir)

| pregunta                                                                         | cómo se midió                                                              | conclusión                                                                                                                                                                                                                                                                                                               |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ¿Qué `uce_target_size` iguala a ALCESTE en ocurrencias/UCE?                      | `uce-size` sobre `bench_uce*.log`                                          | `26` cae dentro del rango de diseño `[14, 30]` (`pal_cr/UCE ≈ 0.885·target + 0.66`). Igualar las **376 UCE** de ALCESTE exigiría `target ≈ 42`, **fuera de especificación** y con UCEs un 45 % más largas: se descarta, porque destruye el objeto de análisis. La diferencia 600 vs 376 UCE es **cobertura**, no tamaño. |
| ¿S2 (factibilidad exacta por pares) supera a S1 en el benchmark del optimizador? | `5_s2_power_analysis.py` (archivado), 8 paisajes × 3 escenarios, `min_n=8` | **Sin evidencia**: en los tres paneles los IC95 cruzan 0 y ningún escenario queda significativo. A igualdad de tiempo (paneles `T=*`) no gana ⇒ **se queda S1**.                                                                                                                                                         |
| ¿Method B (similitud) está muerto o mal parametrizado?                           | `8_probe_b_permissive.py` (archivado)                                      | Con los parámetros más permisivos del espacio seguía dando `B = 0` y `B↔C = 0` ⇒ no era tuning. El objetivo de la etapa 2 ya no exige la triple intersección `A∩B∩C`, así que el acoplamiento dejó de ser bloqueante; la sonda no se vuelve a correr.                                                                    |
| ¿Bootstrap real o perturbación para la estabilidad?                              | ronda 2c del benchmark de selección                                        | **Bootstrap** (ganador C, K=3, `bootstrap_ari = 0.926 [0.893, 0.958]`, `coverage_stable = 0.9527`) ⇒ `Config.stability_bootstrap = True` por defecto.                                                                                                                                                                    |

El opt-out del bootstrap está en el pipeline, no aquí:

```bash
python src/reii/main_workflow.py --mode select --no-stability-bootstrap
```

`--optimize-only` fuerza `stability_bootstrap=False` siempre (1-10 s × cientos de trials, y movería
el objetivo de la etapa 2); `--stability-bootstrap` y `--no-stability-bootstrap` juntos abortan
con exit 2.

## `term-metrics`: cifras comparables con ALCESTE

Deriva del artefacto ya persistido (no relanza el pipeline) las tres columnas que faltaban
para la comparación del paper:

- `phi_sin_yates`: φ sin corrección de continuidad, la columna comparable con ALCESTE.
  Se recalcula desde la tabla 2×2 (a = `frecuencia_cluster`, b = `frecuencia_global` − a,
  c = tamaño de clase − a, d = n − resto), y **verifica** que el χ² de Yates y el φ
  reconstruidos coinciden con los guardados (Δmax ~5·10⁻⁵). Si no cuadra, sale con rc=1.
- `significativo_fdr`: `p_adj < alpha`, el Benjamini-Hochberg que el pipeline ya calcula por
  clase. El flag histórico `significativo` usa el p crudo y está **mal** para publicar.
- `--figure`: plano factorial F9 con el nivel de presencia de cada término y la inercia por
  eje sobre la inercia total del MCA.

```bash
python diag.py term-metrics
python diag.py term-metrics --top 8 --fdr-alpha 0.05
python diag.py term-metrics --figure docs/figs/F9_plano_factorial.svg
```

Resultado sobre la corrida `93fabdc4d0ff6590` (528 UCE, 503 con clase): 214 términos con
p crudo < 0,05 pasan a **74** con FDR < 0,05 (33 / 37 / 4). La deflación de Yates depende de
la fuerza: mediana 0,89 en |φ| ≥ 0,10, 0,95 en |φ| ≥ 0,15, 0,98 en los más fuertes, y anula
705 de las 2 115 filas término-clase.

## Notas de mantenimiento

- **`REPO`** (`reii/diag/common.py`) se resuelve subiendo hasta el directorio que contiene
  `pyproject.toml` y `src/reii`; se puede forzar con `REII_REPO`. No se usa `parents[N]`.
- Los comandos que lanzan el pipeline lo hacen con `--force` (si no, `main_workflow` salta los
  documentos ya presentes en PostgreSQL).
- `trials`, `preflight --exec` y `uce-size --force-run` mutan `data/`; el resto son de solo lectura.
  Antes de lanzar nada, comprueba que no queda un contenedor vivo:
  `docker ps --format "{{.Names}}" | grep -i workflow`.
