# Manifiesto de limpieza — REII v5

Generado: 2026-10-28. Raíz: `C:\Users\Julia\Downloads\reii`.
Tamaños medidos con `PowerShell` (`Measure-Object Length -Sum`) y `du -sh`, no estimados.

Objetivo: (1) borrar solo basura regenerable y segura, (2) dejar preparado el
borrado del resto **después** de que termine el pipeline y el re-tuneo.

> Regla de oro: `data/` está **entero** en `.gitignore` (línea 141: `data/`).
> Git **no** puede recuperar nada de `data/` salvo los pocos ficheros
> re-añadidos a mano con `git add -f`. `output/` también está ignorado
> (línea 50). Borrar ahí **no** es reversible.

---

## 1. Tabla de candidatos

`AHORA` = seguro en este momento. `tras paso 8` = después de `--mode select`
(runbook §8). `tras bloque 1` = después de runbook §9 y del re-tuneo (§7),
cuando el árbol ya esté validado.

| ruta                                        | tamaño          | por qué es basura / regenerable                                                                                 | cuándo borrar        | comando PowerShell exacto                                                 | riesgo |
| ------------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------- | ------ |
| `.pytest_cache/`                            | 763 B           | caché de pytest; `.gitignore:30`                                                                                | AHORA (ya ejecutado) | `Remove-Item -Recurse -Force '.pytest_cache'`                             | nulo   |
| `src/reii/__pycache__/`                     | 12 224 B        | bytecode Python; `.gitignore:4`                                                                                 | AHORA (ya ejecutado) | `Remove-Item -Recurse -Force 'src\reii\__pycache__'`                      | nulo   |
| `src/reii/backend/__pycache__/`             | 11 695 B        | bytecode Python                                                                                                 | AHORA (ya ejecutado) | `Remove-Item -Recurse -Force 'src\reii\backend\__pycache__'`              | nulo   |
| `src/reii/backend/sql/__pycache__/`         | 3 186 B         | bytecode Python                                                                                                 | AHORA (ya ejecutado) | `Remove-Item -Recurse -Force 'src\reii\backend\sql\__pycache__'`          | nulo   |
| `tests/__pycache__/`                        | 7 879 B         | bytecode Python                                                                                                 | AHORA (ya ejecutado) | `Remove-Item -Recurse -Force 'tests\__pycache__'`                         | nulo   |
| `.pcbase/scripts/playground/__pycache__/`   | 416 700 B       | bytecode dentro de la base de backup; no se necesita para restaurar                                             | AHORA (ya ejecutado) | `Remove-Item -Recurse -Force '.pcbase\scripts\playground\__pycache__'`    | nulo   |
| `data/diag/seed0.log`                       | 1 180 754 B     | log de tuneo v4 (27-oct, corpus anterior); nombre sin `_<stamp>`, formato legado                                | AHORA                | `Remove-Item -Recurse -Force 'data\diag\seed0.log'`                       | bajo   |
| `data/diag/seed1.log`                       | 1 182 118 B     | idem                                                                                                            | AHORA                | `Remove-Item -Recurse -Force 'data\diag\seed1.log'`                       | bajo   |
| `data/diag/seed2.log`                       | 1 272 021 B     | idem                                                                                                            | AHORA                | `Remove-Item -Recurse -Force 'data\diag\seed2.log'`                       | bajo   |
| `output/global_dashboard.json`              | 45 680 952 B    | artefacto de `main_workflow`; `output/` ignorado y se reescribe en cada corrida                                 | tras paso 8          | `Remove-Item -Recurse -Force 'output\global_dashboard.json'`              | bajo   |
| `output/red_semantica.gexf`                 | 567 660 B       | idem                                                                                                            | tras paso 8          | `Remove-Item -Recurse -Force 'output\red_semantica.gexf'`                 | bajo   |
| `docs/BLOQUE2_COMANDOS.ps1`                 | 5 456 B         | superseded: el runbook lo dice literalmente («SUPERSEDE docs\BLOQUE2_COMANDOS.ps1 y docs\BLOQUE3_COMANDOS.ps1») | tras bloque 1        | `Remove-Item -Recurse -Force 'docs\BLOQUE2_COMANDOS.ps1'`                 | bajo   |
| `docs/BLOQUE3_COMANDOS.ps1`                 | 9 007 B         | idem                                                                                                            | tras bloque 1        | `Remove-Item -Recurse -Force 'docs\BLOQUE3_COMANDOS.ps1'`                 | bajo   |
| `src/reii/backend/migrate.py.pipelines.bak` | 29 694 B        | backup de `migrate.py` (10-oct, pre-`pipelines`); `*.bak` ignorado, **no trackeado**                            | tras bloque 1        | `Remove-Item -Recurse -Force 'src\reii\backend\migrate.py.pipelines.bak'` | bajo   |
| `data/archive_logs/output.log`              | 4 817 904 B     | log de corrida del 24-oct (corpus viejo); no es el único rastro de nada                                         | tras bloque 1        | `Remove-Item -Recurse -Force 'data\archive_logs\output.log'`              | medio  |
| `data/archive_logs/output_2.log`            | 70 548 B        | idem                                                                                                            | tras bloque 1        | `Remove-Item -Recurse -Force 'data\archive_logs\output_2.log'`            | medio  |
| `data/entrevistas_old.json`                 | 902 948 B       | corpus viejo (04-oct); `entrevistas.json` actual es el trackeado                                                | tras bloque 1        | `Remove-Item -Recurse -Force 'data\entrevistas_old.json'`                 | medio  |
| `data/txt_outputs/uci_*.txt` (29)           | 273 566 B*      | intermedios de export de texto; se regeneran                                                                    | tras bloque 1        | `Remove-Item -Recurse -Force 'data\txt_outputs\uci_*.txt'`                | medio  |
| `data/txt_outputs/tmp/` (29 `.json`)        | (incluido en *) | intermedios temporales                                                                                          | tras bloque 1        | `Remove-Item -Recurse -Force 'data\txt_outputs\tmp'`                      | bajo   |
| `data/best_params.json.bak-20261027_210140` | 10 938 B        | backup de `best_params.json` pre-patchset; queda obsoleto tras el re-tuneo                                      | tras bloque 1        | `Remove-Item -Recurse -Force 'data\best_params.json.bak-20261027_210140'` | medio  |
| `data/best_params.json.bak-20261027_210920` | 10 938 B        | idem                                                                                                            | tras bloque 1        | `Remove-Item -Recurse -Force 'data\best_params.json.bak-20261027_210920'` | medio  |
| `data/best_params.pre-fix.json`             | 3 687 B         | idem                                                                                                            | tras bloque 1        | `Remove-Item -Recurse -Force 'data\best_params.pre-fix.json'`             | medio  |
| `data/best_params.pre_probe3.json`          | 3 112 B         | idem                                                                                                            | tras bloque 1        | `Remove-Item -Recurse -Force 'data\best_params.pre_probe3.json'`          | medio  |
| `data/best_params.pre_v4.json`              | 3 126 B         | idem                                                                                                            | tras bloque 1        | `Remove-Item -Recurse -Force 'data\best_params.pre_v4.json'`              | medio  |
| `data/diag.bak-20261027_210140.zip`         | 271 102 B       | zip de `data/diag/` pre-patchset                                                                                | tras bloque 1        | `Remove-Item -Recurse -Force 'data\diag.bak-20261027_210140.zip'`         | medio  |
| `data/diag.bak-20261027_210920.zip`         | 271 102 B       | idem                                                                                                            | tras bloque 1        | `Remove-Item -Recurse -Force 'data\diag.bak-20261027_210920.zip'`         | medio  |
| `data/reii_optuna.db.bak-20261027`          | 196 608 B       | backup del estudio Optuna                                                                                       | tras bloque 1        | `Remove-Item -Recurse -Force 'data\reii_optuna.db.bak-20261027'`          | medio  |
| `data/reii_optuna.db.bak-20261027_210140`   | 3 559 424 B     | idem (el más grande)                                                                                            | tras bloque 1        | `Remove-Item -Recurse -Force 'data\reii_optuna.db.bak-20261027_210140'`   | medio  |
| `data/reii_optuna.db.bak-20261027_210920`   | 114 688 B       | idem                                                                                                            | tras bloque 1        | `Remove-Item -Recurse -Force 'data\reii_optuna.db.bak-20261027_210920'`   | medio  |
| `data/reii_optuna.db.pre-fix`               | 196 608 B       | idem                                                                                                            | tras bloque 1        | `Remove-Item -Recurse -Force 'data\reii_optuna.db.pre-fix'`               | medio  |

\* `273 566 B` = 29 `.txt` + 29 `.json` de `tmp/`. **No** incluye
`data/txt_outputs/Refined_Database.csv`, que está **trackeado** y se queda.

**Totales**

| bloque                  | bytes          | ≈              |
| ----------------------- | -------------- | -------------- |
| ya borrado (cachés)     | 452 447        | 0,43 MiB       |
| AHORA (seed logs)       | 3 634 893      | 3,5 MiB        |
| tras paso 8 (`output/`) | 46 248 612     | 44,1 MiB       |
| tras bloque 1           | 11 653 404     | 11,1 MiB       |
| **total reclamable**    | **61 989 356** | **≈ 59,1 MiB** |

---

## 2. Verificación previa (PowerShell, solo lectura)

```powershell
Set-Location C:\Users\Julia\Downloads\reii

# ¿Siguen existiendo los objetivos?
Test-Path .pytest_cache, 'src\reii\__pycache__', 'src\reii\backend\__pycache__', 'src\reii\backend\sql\__pycache__', 'tests\__pycache__', '.pcbase\scripts\playground\__pycache__'
Test-Path 'data\diag\seed0.log', 'data\diag\seed1.log', 'data\diag\seed2.log'
Test-Path 'output\global_dashboard.json', 'output\red_semantica.gexf'
Test-Path 'docs\BLOQUE2_COMANDOS.ps1', 'docs\BLOQUE3_COMANDOS.ps1'
Test-Path 'src\reii\backend\migrate.py.pipelines.bak'
Test-Path 'data\archive_logs\output.log', 'data\archive_logs\output_2.log'
Test-Path 'data\entrevistas_old.json'
Test-Path 'data\txt_outputs\uci_01.txt', 'data\txt_outputs\tmp'

# Tamaño total de lo que se va a borrar (bytes)
$t = @(
  '.pytest_cache','src\reii\__pycache__','src\reii\backend\__pycache__',
  'src\reii\backend\sql\__pycache__','tests\__pycache__',
  '.pcbase\scripts\playground\__pycache__',
  'data\diag\seed0.log','data\diag\seed1.log','data\diag\seed2.log',
  'output\global_dashboard.json','output\red_semantica.gexf',
  'docs\BLOQUE2_COMANDOS.ps1','docs\BLOQUE3_COMANDOS.ps1',
  'src\reii\backend\migrate.py.pipelines.bak',
  'data\archive_logs\output.log','data\archive_logs\output_2.log',
  'data\entrevistas_old.json','data\txt_outputs\tmp'
)
$sum = 0
foreach ($p in $t) {
  if (Test-Path -LiteralPath $p) {
    $i = Get-Item -Force -LiteralPath $p
    if ($i.PSIsContainer) { $s = (Get-ChildItem -Force -Recurse -File $i.FullName | Measure-Object Length -Sum).Sum }
    else { $s = $i.Length }
    $sum += $s
    '{0} = {1} bytes' -f $p, $s
  } else { '{0} = MISSING' -f $p }
}
'total sin txt_outputs uci_* = {0} bytes' -f $sum

# Comprobación de que NO hay cachés colgando en ningún sitio
Get-ChildItem -Force -Recurse -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -in @('__pycache__','.pytest_cache','.mypy_cache','.ruff_cache') } |
  ForEach-Object { $_.FullName }
```

---

## 3. Borrado (uno por línea)

### AHORA — seed logs stale (los cachés ya se borraron)

```powershell
Remove-Item -Recurse -Force 'data\diag\seed0.log'
Remove-Item -Recurse -Force 'data\diag\seed1.log'
Remove-Item -Recurse -Force 'data\diag\seed2.log'
```

### tras paso 8 — `output/`

```powershell
Remove-Item -Recurse -Force 'output\global_dashboard.json'
Remove-Item -Recurse -Force 'output\red_semantica.gexf'
```

### tras bloque 1

```powershell
Remove-Item -Recurse -Force 'docs\BLOQUE2_COMANDOS.ps1'
Remove-Item -Recurse -Force 'docs\BLOQUE3_COMANDOS.ps1'
Remove-Item -Recurse -Force 'src\reii\backend\migrate.py.pipelines.bak'
Remove-Item -Recurse -Force 'data\archive_logs\output.log'
Remove-Item -Recurse -Force 'data\archive_logs\output_2.log'
Remove-Item -Recurse -Force 'data\entrevistas_old.json'
Remove-Item -Recurse -Force 'data\txt_outputs\uci_*.txt'
Remove-Item -Recurse -Force 'data\txt_outputs\tmp'
Remove-Item -Recurse -Force 'data\best_params.json.bak-20261027_210140'
Remove-Item -Recurse -Force 'data\best_params.json.bak-20261027_210920'
Remove-Item -Recurse -Force 'data\best_params.pre-fix.json'
Remove-Item -Recurse -Force 'data\best_params.pre_probe3.json'
Remove-Item -Recurse -Force 'data\best_params.pre_v4.json'
Remove-Item -Recurse -Force 'data\diag.bak-20261027_210140.zip'
Remove-Item -Recurse -Force 'data\diag.bak-20261027_210920.zip'
Remove-Item -Recurse -Force 'data\reii_optuna.db.bak-20261027'
Remove-Item -Recurse -Force 'data\reii_optuna.db.bak-20261027_210140'
Remove-Item -Recurse -Force 'data\reii_optuna.db.bak-20261027_210920'
Remove-Item -Recurse -Force 'data\reii_optuna.db.pre-fix'
```

> **No** borrar los backups de `data/` _antes_ de que el re-tuneo (§7) haya
> producido un `best_params.json` válido: son la única vuelta atrás de un
> `reii_optuna.db` que `data/` no trackea.

---

## 4. NO BORRAR

| ruta                                                                                      | motivo                                                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/diag/bench_uce18.log`, `bench_uce22.log`, `bench_uce26.log`, `bench_uce30.log`      | **INPUT** de `4_measure_uce_target_size.py`: hace glob de _todos_ los `bench_uce*.log`. Sin ellos, el ajuste lineal del paso 9 (bloque 1) falla por falta de puntos. **Hallazgo que corrige la premisa inicial.**                                          |
| `data/diag/uce_size_bench.json`                                                           | **INPUT** de `src/7_quality_multiverse.py` (`--bench-json`).                                                                                                                                                                                               |
| `data/archive_logs/p.txt` (9 728 B)                                                       | Única traza de salida de `scripts/playground/gate_interval_probe.py`. No hay `.log` equivalente.                                                                                                                                                           |
| `8_probe_b_permissive.py` (raíz, 15 722 B)                                                | Citado por nombre en `src/reii/main_workflow.py:8371` (docstring de `_cdh_frac_from_params`). **Ojo: no está en `src/`**, está en la raíz y no está trackeado (`?? 8_probe_b_permissive.py`): si se borra, no hay vuelta atrás en git. No mover ni borrar. |
| `data/workflow_data.json` (27 253 726 B)                                                  | Corpus procesado; entrada de §10 y §11.                                                                                                                                                                                                                    |
| `data/workflow_data.db` (38 203 392 B)                                                    | **Trackeado** (`git add -f` pese a `data/` ignorado). Fuente de provenance.                                                                                                                                                                                |
| `data/best_params.json`                                                                   | Lo regenera §7; hasta entonces es el único válido.                                                                                                                                                                                                         |
| `data/reii_optuna.db`                                                                     | Estudio Optuna vivo.                                                                                                                                                                                                                                       |
| `data/coref_chains_cache.json` (2 248 433 B), `data/uce_calibration.json`, `*.lock`       | Caché/calibración esenciales.                                                                                                                                                                                                                              |
| `data/method_selection.json`, `data/method_selection_log.jsonl`                           | **No existen ahora** (los movió §5), pero §8 los regenera: protegerlos en cuanto aparezcan.                                                                                                                                                                |
| `data/token_table.jsonl`                                                                  | **No existe** hoy. Si el pipeline lo genera, protegerlo.                                                                                                                                                                                                   |
| `data/txt_outputs/Refined_Database.csv` (2 120 B)                                         | **Trackeado**. No borrar `data/txt_outputs/` completo, solo `uci_*` y `tmp/`.                                                                                                                                                                              |
| `data/query_db.py`, `data/entrevistas.json`, `data/entrevistas/**`                        | **Trackeados** (32 ficheros en `git ls-files data`).                                                                                                                                                                                                       |
| `ia/gramatica.json`, `ia/localismos.json`, `ia/analisis_discursivo/`, `ia/entrevistas/`   | Datos de entrada de la IA.                                                                                                                                                                                                                                 |
| `tests/`, `src/` (fuente), `scripts/`, `docs/patches/*.patch`                             | Código y parches.                                                                                                                                                                                                                                          |
| `srt_analysis.json` (3 137 001 B), `timing.tsv`, `clean_manifest.json`, `_ast_extract.py` | **Trackeados en la raíz**. Borrarlos sería un cambio de git.                                                                                                                                                                                               |
| `docs/PATCHSET_V5_RUNBOOK.ps1`                                                            | Runbook vigente (no superseded).                                                                                                                                                                                                                           |

---

## 5. DUDOSOS

| ruta                                  | tamaño      | recomendación                                                                                                                                                                                                                                                              | coste de regenerar                                                                                                                                                |
| ------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.pcbase/`                            | 1 059 566 B | **Mantener hasta cerrar el patchset.** No es duplicado puro: 4 ficheros difieren del árbol vivo (`src/reii/main_workflow.py`, `src/reii/method_c_policy.py`, `scripts/playground/classic_profile_check.py`, `scripts/playground/hardening_bench.py`). Es la base _pre-v5_. | Medio: `git apply` de los 22 patches sobre el árbol actual (verificado 21/21 de la cadena + 15 independiente, 151 hunks), pero se pierde el estado previo exacto. |
| `backup_patchset_20261028_135329/`    | 549 380 B   | **Mantener.** Es el _revert base_ que cita el runbook («repetir la cadena desde el backup si hubiera que revertir»). Contiene la única copia de `data/method_selection.json` y `method_selection_log.jsonl`.                                                               | Alto: esos dos JSON no existen en `data/` hoy; solo se recuperan re-corriendo §8 (~1,8 h).                                                                        |
| `data/diag/best_params_seed0.json`    | 17 512 B    | Borrar tras bloque 1 si no se re-lanza `2_run_300_trials.py` (son el análisis de semillas v4, consumido por `analyze()` vía glob).                                                                                                                                         | Alto: re-correr el tuneo multi-semilla.                                                                                                                           |
| `data/diag/best_params_seed1.json`    | 24 760 B    | idem                                                                                                                                                                                                                                                                       | idem                                                                                                                                                              |
| `data/diag/best_params_seed2.json`    | 30 782 B    | idem                                                                                                                                                                                                                                                                       | idem                                                                                                                                                              |
| `data/diag/control_wc_only.log`       | 16 662 B    | Borrar tras bloque 1. Salida de `--mode wc_only` (comando de `BLOQUE2_COMANDOS.ps1`, ya superseded); ningún script lo lee.                                                                                                                                                 | ~2 h (`--mode wc_only`).                                                                                                                                          |
| `data/diag/quality_multiverse.json`   | 8 647 B     | Borrar tras bloque 1: §5 lo mueve a `data/archive_v4/` y §10/`7_quality_multiverse.py` lo re-emite.                                                                                                                                                                        | Bajo: `7_quality_multiverse.py`.                                                                                                                                  |
| `data/archive_v4/`                    | —           | **No existe todavía.** Lo crea el runbook §5. Revisar cuando aparezca.                                                                                                                                                                                                     | n/a                                                                                                                                                               |
| `data/txt_outputs/uci_*.txt` + `tmp/` | 273 566 B   | Regenerable, pero no lo toca el pipeline: solo si se re-exporta texto.                                                                                                                                                                                                     | Bajo/medio.                                                                                                                                                       |

---

## 6. Hallazgos que cambian la recomendación

1. **`data/diag/bench_uce*.log` y `uce_size_bench.json` NO son basura** — son
   entradas del paso 9 (`4_measure_uce_target_size.py --parse-only`) y de
   `7_quality_multiverse.py`. Borrarlos rompe el bloque 1.
2. **`data/method_selection*.json` y `data/token_table.jsonl` no existen.** La
   lista de protegidos los menciona, pero hoy no están en `data/`. Solo hay
   copia en `backup_patchset_20261028_135329/data/`.
3. **`data/archive_v4/` no existe** (lo crea el runbook §5, hoy no ejecutado).
4. **`data/` está completamente en `.gitignore`**: no hay red de seguridad de
   git para los borrados de esa carpeta.
5. Solo hay **un** `.bak` en todo el repo: `src/reii/backend/migrate.py.pipelines.bak`.
   No hay `*.tmp` ni `*.orig` en ninguna parte, ni en la raíz.
6. No existe `.mypy_cache` ni `.ruff_cache` (sí están en `.gitignore`, pero no
   en disco). No se inventan.
7. `.pcbase/scripts/playground/` es casi un duplicado de `scripts/playground/`
   (9 de 11 ficheros idénticos byte a byte), pero `.pcbase/src/` sí difiere:
   no es un duplicado puro.
8. **`src/8_probe_b_permissive.py` no existe**: el fichero está en la raíz
   (`8_probe_b_permissive.py`) y `main_workflow.py` lo cita por nombre, sin
   prefijo `src/`. No hay ninguna ruta `src/8_*` en el repo.
9. El runbook §5 hace `Move-Item data\quality_multiverse.json`, pero el fichero
   está en `data\diag\quality_multiverse.json`. El `Move-Item` fallará
   silenciosamente (`-ErrorAction SilentlyContinue`): discrepancia a corregir
   en el runbook, no aquí.
