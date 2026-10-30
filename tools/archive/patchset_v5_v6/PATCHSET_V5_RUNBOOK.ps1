# ======================================================================
#  REII - RUNBOOK PATCHSET v5 (22 patches)
#  NOTA (posterior a este runbook): los scripts 2_/3_/4_/5_/6_preflight y
#  src/6_/7_/9_ que se invocan mas abajo viven ahora en el CLI
#  `python -m reii.diag` (ver docs/DIAG_CLI.md). 5_ y 8_ estan archivados
#  en tools/archive/diag/.
# ======================================================================
#  SUPERSEDE docs\BLOQUE2_COMANDOS.ps1 y docs\BLOQUE3_COMANDOS.ps1
#  Raiz: C:\Users\Julia\Downloads\reii   (PowerShell)
#
#  Estado al escribirlo (2026-10-28): los 22 patches YA estan aplicados
#  en el arbol de trabajo. Este fichero sirve para (a) re-verificar el
#  estado, (b) re-tunear (el patch 14 invalidó best_params.json) y
#  (c) repetir la cadena desde el backup si hubiera que revertir.
#
#  Verificado 2026-10-28 en arbol scratch (%TEMP%\b4):
#    21/21 de la cadena + 15 independiente (rc=0) = 22 patches (151 hunks)
#    203 passed, 4 xpassed, 0 skipped, 0 failed  (94 s, 11 ficheros base)
#    +3 netos de 02h/14 => 206; +3 de 15 => esperado 209 passed (12 ficheros)
#
#  NUMERACION: "bloque 1/2/3" = milestones del proyecto.
#  "tramos 1-7" = docs\design_v5\FINAL_PATCHSET.md seccion 2. No son lo mismo.
# ======================================================================


# ----------------------------------------------------------------------
# 0) PREFLIGHT
#    Si aparece un reii-workflow-run-*, espera: el select y el
#    --force-run escriben el MISMO data\workflow_data.json y el sqlite
#    se bloquea.
# ----------------------------------------------------------------------
docker ps --filter "name=reii-workflow-run" --format "{{.Names}}  {{.Status}}"
docker ps --filter "name=reii-db" --format "{{.Names}}  {{.Status}}"

# Estado actual de los params (v4 = INVALIDO desde el patch 14; ver paso 7)
Test-Path data\best_params.json
python -c "import json,os;d=json.load(open('data/best_params.json'));print('score',d.get('score'));print('fp?','corpus_fingerprint' in d);print(d.get('best_params'))"


# ----------------------------------------------------------------------
# 1) COMPROBACIONES PREVIAS  (solo si vas a aplicar en un arbol limpio)
#    Ojo: el arbol de partida NO es `git HEAD`; es el arbol de trabajo
#    del 2026-10-28 (backup_patchset_20261028_135329), que incluye
#    untracked que los patches MODIFICAN:
#       tests/test_mode_classic.py               (02g, 02h)
#       scripts/playground/classic_profile_check.py (02h)
#       ia/gramatica.json                        (14)
#    09, 10 y 12 validan contra la copia acumulada `.pcbase/`.
#    02h, 13 y 14 no validan ahi: 02h necesita el test que reescribe
#    02g; 13 necesita el gramatica.py y el lematizar_uces de 11;
#    14 necesita retention_policy.py e ia/gramatica.json de 13.
#    Los tres solo se validan por cadena (rc=0 los 22).
#    15 si valida sobre arbol limpio: toca config_store.py y crea un test.
# ----------------------------------------------------------------------
git apply --check docs/patches/01_quality_metrics.patch
git apply --check docs/patches/02_method_selection.patch
git apply --check docs/patches/02b_tests_method_selection.patch
git apply --check docs/patches/03_main_workflow_gate_select.patch
git apply --check docs/patches/03b_backends_c.patch
git apply --check docs/patches/05_selection_audit.patch
git apply --check docs/patches/06_classic_mode.patch
git apply --check docs/patches/07_v5_space.patch
git apply --check docs/patches/06b_stability.patch
git apply --check docs/patches/02c_tests_select_v4.patch
git apply --check docs/patches/02d_tests_backend_hardening.patch
git apply --check docs/patches/02e_tests_stability_null.patch
git apply --check docs/patches/02f_tests_kach_objective.patch
git apply --check docs/patches/02g_tests_mode_classic.patch
git apply --check docs/patches/11_formas_activas.patch
git apply --check docs/patches/15_config_allowlist.patch

git apply -p1 --directory=.pcbase --check docs/patches/09_fix44_classic_guard.patch
git apply -p1 --directory=.pcbase --check docs/patches/10_hardening_ward_max_n.patch
git apply -p1 --directory=.pcbase --check docs/patches/12_guard_optimize_only_classic.patch

# 02h / 13 / 14: sin comprobacion previa posible (ver nota de arriba).


# ----------------------------------------------------------------------
# 2) BACKUP  (no hay commit en juego: copia manual)
# ----------------------------------------------------------------------
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$dst = "backup_patchset_$stamp"
New-Item -ItemType Directory -Path $dst | Out-Null
Copy-Item src\reii\main_workflow.py     $dst
Copy-Item src\reii\method_selection.py  $dst
Copy-Item src\reii\config.py            $dst
Copy-Item src\reii\gramatica.py         $dst
Copy-Item src\reii\retention_policy.py  $dst
Copy-Item src\reii\quality\metrics.py   $dst
Copy-Item src\9_selection_audit.py      $dst
Copy-Item ia\gramatica.json             $dst
Copy-Item data\best_params.json         $dst
Copy-Item data\method_selection.json    $dst -ErrorAction SilentlyContinue
Copy-Item data\method_selection_log.jsonl $dst -ErrorAction SilentlyContinue


# ----------------------------------------------------------------------
# 2b) REVERTIR (solo si hace falta): desaplicar en orden INVERSO
#     `git apply -R`, NUNCA `patch` (los .patch van en LF y el arbol en CRLF).
# ----------------------------------------------------------------------
# git apply -R docs/patches/15_config_allowlist.patch
# git apply -R docs/patches/14_retention_default_tool_adverb_filtered.patch
# git apply -R docs/patches/13_retention_policy.patch
# git apply -R docs/patches/12_guard_optimize_only_classic.patch
# git apply -R docs/patches/11_formas_activas.patch
# git apply -R docs/patches/10_hardening_ward_max_n.patch
# git apply -R docs/patches/09_fix44_classic_guard.patch
# git apply -R docs/patches/02h_playground_classic_profile_check.patch
# git apply -R docs/patches/02g_tests_mode_classic.patch
# git apply -R docs/patches/06b_stability.patch
# ... (resto en orden inverso al paso 3)


# ----------------------------------------------------------------------
# 3) APLICAR EN ESTE ORDEN EXACTO  (22 patches)
#    `git apply`, NUNCA `patch` (los .patch van en LF y el arbol en CRLF).
# ----------------------------------------------------------------------
git apply docs/patches/01_quality_metrics.patch
git apply docs/patches/02_method_selection.patch
git apply docs/patches/02b_tests_method_selection.patch
git apply docs/patches/03_main_workflow_gate_select.patch
git apply docs/patches/03b_backends_c.patch
git apply docs/patches/05_selection_audit.patch
git apply docs/patches/06_classic_mode.patch
git apply docs/patches/07_v5_space.patch
git apply docs/patches/06b_stability.patch
git apply docs/patches/02c_tests_select_v4.patch
git apply docs/patches/02d_tests_backend_hardening.patch
git apply docs/patches/02e_tests_stability_null.patch
git apply docs/patches/02f_tests_kach_objective.patch
git apply docs/patches/02g_tests_mode_classic.patch
git apply docs/patches/02h_playground_classic_profile_check.patch
git apply docs/patches/09_fix44_classic_guard.patch
git apply docs/patches/10_hardening_ward_max_n.patch
git apply docs/patches/11_formas_activas.patch
git apply docs/patches/12_guard_optimize_only_classic.patch
git apply docs/patches/13_retention_policy.patch
git apply docs/patches/14_retention_default_tool_adverb_filtered.patch
git apply docs/patches/15_config_allowlist.patch

# Sintaxis de los ficheros tocados (falla ruidoso antes de gastar docker)
python -c "import ast;fs=['src/reii/main_workflow.py','src/reii/method_selection.py','src/reii/quality/metrics.py','src/reii/method_c_policy.py','src/reii/retention_policy.py','src/reii/gramatica.py','src/9_selection_audit.py'];[ast.parse(open(f,encoding='utf-8').read()) for f in fs];print('ast ok:', len(fs), 'ficheros')"


# ----------------------------------------------------------------------
# 3b) CHECK DEL DEFAULT DE RETENCION (patch 14)
#     Las tres definiciones tienen que coincidir y el resolutor sin
#     argumentos tiene que devolver la conservadora. Si falla, NO sigas:
#     el pipeline estaria midiendo otra politica que la del paper.
# ----------------------------------------------------------------------
python -c "import json,sys;sys.path.insert(0,'src');from reii.retention_policy import DEFAULT_RETENTION_POLICY as d,resolve_retention_policy as r;g=json.load(open('ia/gramatica.json',encoding='utf-8')).get('retention_policy');print('code=',d,'| json=',g,'| resuelto=',r(None,None));assert d==g=='tool_adverb_filtered' and r(None,None)=='tool_adverb_filtered', 'default de retencion desalineado';print('=> default OK: tool_adverb_filtered')"

# La precedencia tambien (Config > JSON > default):
python -c "import sys;sys.path.insert(0,'src');from reii.retention_policy import resolve_retention_policy as r;print('json gana:', r(None,'pos_only'));print('config gana:', r('pos_only',None));assert r(None,'pos_only')=='pos_only' and r('pos_only',None)=='pos_only'"


# ----------------------------------------------------------------------
# 4) TESTS  (dos tandas; ~2 min la primera, ~2 min la segunda)
#    NUNCA uses `-o pythonpath=`: importa el src del repo y mide codigo viejo.
#    Esperado: 209 passed, 4 xpassed, 0 failed, 0 skipped
#    (203 de la cadena de 19, −19 de los 2 ficheros que tocan 02h/14,
#     +22 de ahora: 15 en test_mode_classic + 7 en retencion; +3 de 15)
# ----------------------------------------------------------------------
python -m pytest -q tests/test_select_v4.py tests/test_method_selection.py tests/test_backend_hardening.py tests/test_method_c_backends.py tests/test_stability_null.py
python -m pytest -q tests/test_optimizer_gate.py tests/test_select_mode_objective.py tests/test_cdh_min_leaf_floor.py tests/test_v5_space.py tests/test_mode_classic.py tests/test_retention_policy.py tests/test_config_allowlist.py


# ----------------------------------------------------------------------
# 4b) CHECKS ESTATICOS (baratos, sin docker, no tocan nada)
# ----------------------------------------------------------------------
python scripts/playground/classic_profile_check.py
python scripts/playground/audit_smoke.py
python scripts/playground/select_v4_arena.py


# ----------------------------------------------------------------------
# 5) LIMPIAR LOS SNAPSHOTS STALE  (son de OTRO corpus: 866/600 UCEs, no 528)
#    Sin esto, el audit compara peras con manzanas.
# ----------------------------------------------------------------------
Remove-Item data\method_selection.json       -ErrorAction SilentlyContinue
Remove-Item data\method_selection_log.jsonl  -ErrorAction SilentlyContinue


# ----------------------------------------------------------------------
# 6) METRICAS NUEVAS (C_in, d, RMSD)  -- ANTES de --mode select
#    Lee data\workflow_data.json, que `--mode select` va a sobrescribir.
#    NO existe scripts/metricas_paper.py: el auto-test es este modulo.
#    (bench uce=26: 528 UCEs; 154 con embedding, 374 sin)
# ----------------------------------------------------------------------
python -c "import sys; sys.path.insert(0,'scripts/playground'); import gate_calibration as g; sys.exit(g.run_metrics_selftest())"


# ======================================================================
# 7) RE-TUNEO OBLIGATORIO  (patch 14: el default de retencion cambio)
# ----------------------------------------------------------------------
#    El vocabulario de formas activas pasa de 5658 a 6465 formas, asi que
#    min_forms_uc_1 y forms_gap cambian de significado y el
#    data\best_params.json de v4 (score 0.4943, knee trial #3,
#    coverage=0.4943 agreement=0.3967 K=11) queda INVALIDO.
#    `retention_policy` NO entra en run_spec_id ni en corpus_fingerprint:
#    una study vieja aceptaria trials nuevos sin avisar. Por eso:
#    --study-name nuevo + --no-optuna-warm-start.
#    NO correr --mode select con los params viejos.
# ----------------------------------------------------------------------
New-Item -ItemType Directory -Path data\archive_v4 -Force | Out-Null
Move-Item data\best_params.json          data\archive_v4\best_params.v4.json -Force
Copy-Item data\reii_optuna.db            data\archive_v4\reii_optuna.db      -Force
Move-Item data\diag\quality_multiverse.json data\archive_v4\ -Force -ErrorAction SilentlyContinue
Remove-Item data\method_selection.json, data\method_selection_log.jsonl -ErrorAction SilentlyContinue

# re-optimizar (~horas; v4 uso 40 trials stage1 + 20 stage2)
docker compose --profile batch run --rm workflow python -u src/reii/main_workflow.py --force --optimize-only --trials 40 --stage1-fraction 0.5 --study-name reii_probe_v5 --no-optuna-warm-start
python 3_diagnose_optuna_db.py --study reii_probe_v5

# comprobar que el best_params nuevo apunta al vocabulario nuevo
python -c "import json;d=json.load(open('data/best_params.json'));print('score',d['score']);print(d['best_params'])"


# ----------------------------------------------------------------------
# 8) SELECT-v4  (CON best_params NUEVOS: nunca con los de v4)
#    ~1.8 h. Regla select-v4, gate de dominancia 0.70 sobre corpus.
# ----------------------------------------------------------------------
docker compose --profile batch run --rm workflow python -u src/reii/main_workflow.py --force --mode select

# Debe salir 'winner' != None, K en [3,20] y 'abstained' False
python -c "import json;d=json.load(open('data/method_selection.json'));s=d['selection'];print('rule',d['rule_version'],'winner',s.get('winner'),'K',s.get('winner_k'),'abstained',s.get('abstained'));print('ctx',d.get('context'))"

docker compose --profile batch run --rm workflow python -u src/9_selection_audit.py --strict
docker compose --profile batch run --rm workflow python -u src/9_selection_audit.py --strict --json data\diag\selection_audit_v4.json


# ----------------------------------------------------------------------
# 9) BLOQUE 1  (independiente de docker; seguro en paralelo con 10)
#    --force-run (1.8 h) OMITIDO a proposito: solo rellena wall_s de una
#    fila que ya aporta n_UCE y pal_cr/UCE, y re-segmenta pisando
#    workflow_data.json (invalida el select del paso 8).
# ----------------------------------------------------------------------
python 4_measure_uce_target_size.py --parse-only --extrapolate 5809
python 5_s2_power_analysis.py --seeds 8 --min-n 8 --out s2_results.json --bench-out bench_output.md


# ----------------------------------------------------------------------
# 10) RETENCION: reproducir la regla pre-registrada (veredicto automatico)
#     Modo residuo (sin Stanza, determinista). Esperado: GANADORA
#     tool_adverb_filtered (cuota 0.000); pos_plus_propn EXCLUIDA (0.375).
#     OJO: el evaluador calcula las 3 politicas, no lee el default; corre
#     igual antes y despues del patch 14.
# ----------------------------------------------------------------------
docker compose --profile batch run --rm workflow python -u scripts/retention_policy_report.py --input data/workflow_data.json --bootstrap 1000 --seed 42


# ----------------------------------------------------------------------
# 11) PROVENANCE: refrescar la tabla de quality_run con la corrida nueva
#     (run_id = code_version() = git rev-parse --short HEAD: sin commit,
#      sobrescribe filas. Usa REII_GROUP_VERSION si quieres separarlas.)
# ----------------------------------------------------------------------
docker compose --profile batch run --rm workflow python -u src/7_quality_multiverse.py --corpus-words 14331


# ----------------------------------------------------------------------
# 12) LO QUE NO SE HACE (y por que)
#    * 4_measure --sizes 30 --force-run : 1.8 h para un solo campo (paso 9).
#    * --mode classic en el paper        : ya testeado (15/15), pero no es
#                                          el modo de las corridas.
#    * reusar --study-name reii_probe_v4: appendaria trials medidos con el
#                                          vocabulario viejo (pos_only).
#    * borrar data\reii_optuna.db       : v2/v3/v4 son evidencia del paper;
#                                          se COPIA a archive_v4, no se mueve.
#    * borrar data\coref_chains_cache.json ni data\uce_calibration.json:
#                                          recalcular = horas de Stanza, y la
#                                          segmentacion no depende del lexico.
#    * data\token_table.jsonl           : guarda TODOS los tokens con su POS;
#                                          la decision es de la politica, no de
#                                          la tabla -> sobrevive al cambio.
# ======================================================================
