# reii.pipelines — pestaña de configuración, figuras del discurso en C y vínculo con la BD

Descomprimir en la raíz del repo. Requiere el paquete `reii_quality_pack` (solo sus pruebas usan
`reii.quality.store`). Dependencias: las ya instaladas (streamlit, pandas, psycopg).

## Orden
1. `python -m pytest tests/test_pipelines_data.py tests/test_pipelines_ui.py tests/test_pipelines_patches.py tests/test_diagnose_rf.py`
2. `python scripts/apply_pipeline_patches.py --dry-run`, y luego sin `--dry-run` (copias `.pipelines.bak`; `--revert` restaura)
3. Alinear lo que ya está: `python -m reii.pipelines.runtime sync` (anotaciones del estado -> tabla `annotations`)
4. `streamlit run src/reii/dashboard.py`: pestaña E «Pipelines y configuración»; figuras en la última sección de la C
5. `python scripts/diagnose_rf.py` para saber por qué la pestaña D no muestra RF

## Qué cambia
- E: mapa de los 3 pipelines con estado, chequeos de coherencia con botón de arreglo, y una pestaña por pipeline
  (qué hace / lee / escribe, configuración, ejecutar, registro). Ya no muestra resultados.
- C (final): figuras del discurso por clase, leídas de la BD: Evidencia y KWIC por figura, filtros por agente y confianza.
- Ejecución del discurso: entrada generada desde la BD (ids = `uces.uce_id`) y sincronización automática al terminar.
- Configuración: se escribe en la BD (ConfigStore) y en `workflow_config.json` a la vez.
- `migrate_annotations` reconcilia ids en vez de descartar anotaciones.

## No incluido / sin probar
- Las trayectorias y otras vistas de resultados de la antigua pestaña E no se portaron a C.
- El código pegado en `dashboard.py` (funciones `_pipeline_*`) y el camino PostgreSQL real no se probaron con tu base.
- Los parches se probaron sobre una copia del repo (compilan, son idempotentes, reversibles y conservan CRLF).
