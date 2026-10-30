# `tools/archive/` — scripts retirados

Nada de aquí se importa ni se ejecuta en el flujo normal. Se conserva por trazabilidad: son los
scripts exactos que produjeron los números citados en `docs/DIAG_CLI.md` y en el paper.

## `diag/5_s2_power_analysis.py`

Potencia estadística **S2 vs S1** sobre el benchmark _mock_ del optimizador
(`tests/reii_optimizer_report.py`, paisajes sintéticos). **No mide el corpus.**

Decisión cerrada: con 8 semillas y 3 escenarios, los IC95 del delta pareado cruzan 0 en todos los
paneles y ningún escenario es significativo; en los paneles de igualdad de tiempo `S2` no gana.
⇒ **se queda S1**. El script solo tendría sentido si se cambiara la estrategia de búsqueda del
optimizador, cosa que hoy no está en el plan.

Reemplazo para la reanudación del análisis: los datos que produjo están aquí al lado
(`benchmark.md`, `benchmark.json`, `s2_results.json`).

## `8_probe_b_permissive.py`

Sonda de una sola corrida: mutaba `best_params.json` hacia valores permisivos y lanzaba el pipeline
completo (~2 h) para distinguir «Method B mal parametrizado» de «Method B estructuralmente muerto».

Decisión cerrada: con los parámetros más permisivos del espacio, `B` seguía en 0 (y `B�C` en 0) ⇒
no era tuning. Además el objetivo de la etapa 2 ya no es la triple intersección `A∩B∩C`, así que el
hallazgo dejó de ser bloqueante. No se vuelve a ejecutar.

Para que el fichero siga siendo legible/importable se portó su import a `reii.diag.common` (antes
`diag_common`, que ya no existe). Eso obliga a tener el paquete importable (`pip install -e .` o
`PYTHONPATH=src`). Aviso si se revive: `run_streaming()` usa `cwd=Path(__file__).parent`, que aquí es
`tools/archive/diag/`; desde ahí `docker compose` no encuentra `docker-compose.yml`. Lánzalo desde la
raíz o cambia esa línea a `cwd=REPO`.

## Reglas

- No añadir aquí código que se use: si algo se usa, va a `src/reii/diag/`.
- Si un script archivado se resucita, primero pórtalo al CLI (`python -m reii.diag …`).
