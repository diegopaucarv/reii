# reii.quality — paquete de niveles de calidad

Descomprimir en la raíz del repo (conserva las rutas). No requiere dependencias nuevas
(numpy, pandas, scipy, scikit-learn, plotly y streamlit ya están en requirements.txt).

## Orden de ejecución
1. `python -m reii.quality.cli selftest`                       # prueba extremo a extremo (SQLite en memoria)
2. `python -m pytest tests/test_quality_core.py tests/test_quality_store_flow.py`
3. `python scripts/apply_quality_patches.py --dry-run`          # revisa; luego sin --dry-run (crea .quality.bak)
4. `python -m reii.quality.cli init`                            # tablas quality_* en el mismo PostgreSQL
5. `python -m reii.quality.cli migrate-legacy --dry-run`        # lee PostgreSQL (REII_DATABASE_URL) o `--json RUTA`
   `python -m reii.quality.cli migrate-legacy`                  # migra la corrida actual como "legacy"
6. `streamlit run src/reii/dashboard.py`                        # pestaña "F · Calidad y niveles"
   (o `python -m reii.quality.cli serve` para la página independiente)
7. Próximas corridas del pipeline guardan su corrida de calidad automáticamente (hook).
   Opcional: `REII_STABILITY_MAPPING=many_to_one` activa el mapeo muchos-a-uno (por defecto: hungarian).

## Archivos (src/reii/quality/)
store · mapping · consensus · metrics · calibration · service · hooks · migrate_legacy ·
dashboard_panel · app · cli      (+ scripts/apply_quality_patches.py, tests/test_quality_*.py)

## Límites conocidos
- La migración de la corrida previa solo conserva `is_stable`: las banderas por comparación
  (stable_wc, stable_sim, stable_emb, stable_cross_*) no se persisten en el pipeline actual.
- El camino PostgreSQL (psycopg) no se probó contra una base real; el mismo código se probó en SQLite.
- El hook se probó con objetos simulados, no con una corrida real del pipeline.
