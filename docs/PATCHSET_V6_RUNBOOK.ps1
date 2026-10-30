#Requires -Version 5.1
<#
  REII - RUNBOOK PATCHSET v6 (patches 16..20)
  Raiz: C:\Users\Julia\Downloads\reii     (PowerShell, Windows)
  Idempotente: cada patch se comprueba en ambos sentidos antes de aplicarse,
  y la limpieza usa -ErrorAction SilentlyContinue.

  ESTADO VERIFICADO 2026-10-28 sobre el arbol de trabajo (solo lectura):
    git apply --check docs/patches/16_select_noncommensurable.patch
      -> OK, sin offsets (method_selection.py + tests/test_select_v4.py)
    git apply --check docs/patches/17_cdh_target_k_ceiling.patch
      -> OK, con offsets INTERNOS al propio patch:
           main_workflow.py                hunk #2 @8982 (+10 lineas)
           tests/test_cdh_min_leaf_floor.py hunk #2 @99  (+6 lineas)
         El +10 y el +6 son exactamente las lineas que inserta el hunk #1
         del MISMO fichero (10 = 7 comentario + constante + 2 blancos;
         6 = 1 import -> 7 imports). No hay drift del arbol: los dos
         anclajes (8358 y 8972; 10 y 93) estan donde dice el patch.
    git apply --check docs/patches/18_diag_sentinel_signflip.patch
      -> OK, sin offsets (3_diagnose_optuna_db.py + test nuevo)
    git apply --check docs/patches/19_search_space_precommit.patch
      -> OK, sin offsets (regenerado 2026-10-28 17:02; la version previa
         fallaba el hunk #1 por 1 vs 2 lineas en blanco). 4 hunks:
           main_workflow.py       @@8436  (+202: precommit del espacio)
                                  @@8988  (tsj+3; swap_iterations sin 0)
                                  @@9702  (auditoria de fronteras)
           tests/test_v5_space.py @@291   (+147)
         Verificado tambien EN SERIE 16+17+18+19 (git apply --check con
         los cuatro ficheros): rc=0.
    docs/patches/20_stability_spaces_bootstrap.patch -> NO existe todavia.
    data\best_params.json            -> NO existe hoy.
    data\method_selection*.json      -> NO existen hoy (el paso 6 es no-op).

  NO se hace commit/add/stash/checkout/reset/clean/push en ningun punto.
  Los .patch van en LF y el arbol en CRLF: usar `git apply`, NUNCA `patch`.
#>

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Assert-LastExit {
    param([Parameter(Mandatory)][string]$What)
    if ($LASTEXITCODE -ne 0) {
        throw "FALLO (rc=$LASTEXITCODE): $What"
    }
}

# Orden oficial. 16/17/18 deben EXISTIR (si faltan, aborta). 19/20 pueden
# no existir todavia (se avisa y se omiten); pero si existen, son
# obligatorios: un --check fallido en ellos aborta el runbook.
$PatchOrder = @(
    '16_select_noncommensurable.patch',
    '17_cdh_target_k_ceiling.patch',
    '18_diag_sentinel_signflip.patch',
    '19_search_space_precommit.patch',
    '20_stability_spaces_bootstrap.patch'
)
$Required = @(
    '16_select_noncommensurable.patch',
    '17_cdh_target_k_ceiling.patch',
    '18_diag_sentinel_signflip.patch'
)

# ======================================================================
# 0) PREFLIGHT
#    Un `reii-workflow-run-*` vivo = corrida batch en curso. El select y
#    el --force-run escriben el MISMO data\workflow_data.json y el sqlite
#    se bloquea: hay que esperar. `reii-db` arriba es NORMAL (servicio
#    Postgres, restart: unless-stopped); solo se informa.
#    Fallo = aparece cualquier reii-workflow-run-* o `docker ps` falla.
# ======================================================================
$wfRun = @(docker ps --filter "name=reii-workflow-run" --format "{{.Names}}")
Assert-LastExit 'docker ps --filter name=reii-workflow-run'
if ($wfRun.Count -gt 0) {
    throw "Corrida en curso (reii-workflow-run): $($wfRun -join ', '). Espera a que termine."
}

$dbRun = @(docker ps --filter "name=reii-db" --format "{{.Names}}")
Assert-LastExit 'docker ps --filter name=reii-db'
if ($dbRun.Count -eq 0) {
    Write-Warning 'reii-db no esta arriba; `docker compose run workflow` lo arrancara (depends_on: service_healthy).'
} else {
    Write-Host "[i] reii-db: $($dbRun -join ', ')"
}

# Estado de los params. Hoy el fichero NO existe: se informa y se sigue
# (el re-tuneo del paso 8 los regenera con --optimize-only).
# Fallo = el fichero existe pero no es JSON legible, o le falta best_params.
if (Test-Path -LiteralPath 'data\best_params.json') {
    python -c "import json;d=json.load(open('data/best_params.json',encoding='utf-8'));print('score',d.get('score'));print('fp?',d.get('corpus_fingerprint'));print('has_best_params',bool(d.get('best_params') or d.get('params')))"
    Assert-LastExit 'lectura de data\best_params.json'
} else {
    Write-Warning 'data\best_params.json NO existe: no hay params previos que validar.'
}

# ======================================================================
# 1) BACKUP  (solo los ficheros DESTINO de los patches; ni arbol ni data/)
#    Los destinos se EXTRAEN de las cabeceras `--- a/...` de cada patch,
#    asi el paso sigue siendo correcto cuando aparezca el 20.
#    Fallo = un destino citado por un patch no existe en el arbol.
# ======================================================================
$stamp  = Get-Date -Format 'yyyyMMdd_HHmmss'
$backup = Join-Path $Root "backup_patchset_$stamp"
New-Item -ItemType Directory -Path $backup -Force | Out-Null

$targets = New-Object System.Collections.Generic.List[string]
foreach ($p in $PatchOrder) {
    $path = Join-Path $Root "docs/patches/$p"
    if (-not (Test-Path -LiteralPath $path)) { continue }
    foreach ($ln in (Get-Content -LiteralPath $path -Encoding UTF8)) {
        if ($ln -match '^--- a/(.+)$') {
            $rel = $Matches[1].Trim()
            if ($rel -eq '/dev/null') { continue }
            if (-not $targets.Contains($rel)) { $targets.Add($rel) }
        }
    }
}
if ($targets.Count -eq 0) { throw "No se extrajo ningun destino de docs/patches." }

foreach ($rel in $targets) {
    $src = Join-Path $Root $rel
    if (-not (Test-Path -LiteralPath $src)) {
        throw "El patch cita '$rel' pero no existe en el arbol."
    }
    $dst = Join-Path $backup ($rel -replace '/', '\')
    New-Item -ItemType Directory -Path (Split-Path -Parent $dst) -Force | Out-Null
    Copy-Item -LiteralPath $src -Destination $dst -Force
}
Write-Host "[i] backup en $backup ($($targets.Count) ficheros)"

# ======================================================================
# 4) APPLY  (16 -> 17 -> 18 -> 19 -> 20)
#    Dos fases, como se pide: (A) --check de cada patch en orden y
#    (B) aplicacion real uno a uno abortando al primer fallo.
#    Se comprueba patch a patch (no en una sola invocacion) porque una
#    invocacion combinada se detiene en el primer fallo y no dice CUAL de
#    los condicionales esta stale.
#    Idempotencia: si `--check --reverse` pasa, el patch ya estaba
#    aplicado y no se reaplica.
#    Fallo = --check falla en 16/17/18, o `git apply` falla en cualquiera
#    de los que pasaron el check.
# ======================================================================
$toApply = @()
foreach ($p in $PatchOrder) {
    $path = Join-Path $Root "docs/patches/$p"
    if (-not (Test-Path -LiteralPath $path)) {
        if ($Required -contains $p) { throw "Falta patch obligatorio: $p" }
        Write-Warning "CONDICIONAL: $p no existe todavia; se omite."
        continue
    }

    & git apply --check --reverse $path 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[=] $p ya aplicado (check inverso OK); no se reaplica."
        continue
    }

    & git apply --check $path 2>&1 | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -eq 0) {
        $toApply += $path
        continue
    }

    # Un patch PRESENTE que no aplica NO se omite en silencio: se aborta.
    # La fase A corre antes de cualquier `git apply`, asi que abortar aqui
    # deja el arbol intacto. Regenera el patch y vuelve a lanzar.
    throw "Preflight --check FALLO en $p (existe pero no aplica). Abortado sin tocar el arbol. Ver cabecera."
}

foreach ($path in $toApply) {
    & git apply $path
    if ($LASTEXITCODE -ne 0) { throw "git apply fallo: $path" }
    Write-Host "[+] aplicado: $path"
}
if ($toApply.Count -eq 0) { Write-Host '[i] Nada que aplicar: todos los patches presentes ya estaban.' }

# ======================================================================
# 5) TESTS  (dos tandas, orden exacto)
#    Los ficheros de test que introduce un patch condicional pueden no
#    existir; pytest falla con "file not found", asi que se filtran y se
#    avisa. test_v5_space.py EXISTE siempre (19 solo lo amplia).
#    Fallo = pytest rc != 0 (rojo o error de coleccion).
# ======================================================================
$batch1 = @(
    'tests/test_select_v4.py',
    'tests/test_method_selection.py',
    'tests/test_backend_hardening.py',
    'tests/test_method_c_backends.py',
    'tests/test_stability_null.py',
    'tests/test_diag_signflip.py',          # lo crea 18 (obligatorio)
    'tests/test_stability_bootstrap.py'      # lo crearia 20 (condicional)
)
$run1 = @()
foreach ($f in $batch1) {
    if (Test-Path -LiteralPath $f) { $run1 += $f }
    else { Write-Warning "CONDICIONAL ausente: $f; se omite de la tanda 1." }
}
if ($run1.Count -eq 0) { throw 'Tanda 1 vacia.' }
python -m pytest -q @run1
Assert-LastExit 'pytest tanda 1'

$batch2 = @(
    'tests/test_optimizer_gate.py',
    'tests/test_select_mode_objective.py',
    'tests/test_cdh_min_leaf_floor.py',
    'tests/test_v5_space.py',
    'tests/test_mode_classic.py',
    'tests/test_retention_policy.py',
    'tests/test_config_allowlist.py'
)
$run2 = @()
foreach ($f in $batch2) {
    if (Test-Path -LiteralPath $f) { $run2 += $f }
    else { Write-Warning "CONDICIONAL ausente: $f; se omite de la tanda 2." }
}
if ($run2.Count -eq 0) { throw 'Tanda 2 vacia.' }
python -m pytest -q @run2
Assert-LastExit 'pytest tanda 2'

# ======================================================================
# 6) SNAPSHOTS STALE
#    method_selection.json / _log.jsonl son de OTRO corpus o de otra
#    regla. Sin borrarlos, `9_selection_audit --strict` compara peras con
#    manzanas (replay vigente vs payload viejo).
#    Fallo: ninguno; hoy no existen (Remove-Item SilentlyContinue).
# ======================================================================
Remove-Item data\method_selection.json, data\method_selection_log.jsonl -ErrorAction SilentlyContinue

# ======================================================================
# 7) AUTO-TEST DE METRICAS, ANTES de --mode select
#    Lee data\workflow_data.json, que el select va a sobrescribir.
#    rc=1 solo si falta el fichero o hay <10 UCEs con embedding (hoy hay).
#    Fallo = rc != 0.
# ======================================================================
python -c "import sys; sys.path.insert(0,'scripts/playground'); import gate_calibration as g; sys.exit(g.run_metrics_selftest())"
Assert-LastExit 'auto-test de metricas (gate_calibration.run_metrics_selftest)'

# ======================================================================
# 8) RE-TUNEO  (STUDY NUEVO OBLIGATORIO: reii_probe_v6)
#    Por que NO se puede reutilizar reii_probe_v5 (verificado en codigo):
#      * save/restore del estudio: `load_if_exists=bool(storage)` y el
#        nombre se sella con _corpus_fingerprint(); con el mismo corpus,
#        reii_probe_v5 ES el mismo estudio y se REANUDA.
#      * `--no-optuna-warm-start` solo vacia la lista de enqueue
#        (warm_start_trials); NO impide que Optuna cargue el historial del
#        estudio. TPE modelaria con trials del espacio VIEJO
#        (tsj<=5, swap_iterations incluia 0, cdh_target_k<=20) mezclados
#        con los del espacio v6.
#      * `--trials 40` se SUMARIA a los trials ya existentes: no seria una
#        campana de 40 sino "viejos + 40", y la particion de etapa 1 solo
#        aplica a los nuevos.
#      * `3_diagnose_optuna_db.py --study reii_probe_v5` agrega TODOS los
#        trials del estudio: la tabla de rho y el detector de sign-flip
#        quedarian contaminados por dos espacios distintos.
#    Guarda: si ya hay un estudio `reii_probe_v6*`, ABORTA (no re-gasta
#    presupuesto). Para repetir de verdad, usa otro prefijo.
#    Fallo = ya existe el estudio, o rc != 0 del docker compose run.
# ======================================================================
python -c "import os,sqlite3; p='data/reii_optuna.db'; n=[r[0] for r in sqlite3.connect(p).execute('SELECT study_name FROM studies')] if os.path.exists(p) else []; h=[x for x in n if x.startswith('reii_probe_v6')]; print('reii_probe_v6 existentes:',h); raise SystemExit(1 if h else 0)"
if ($LASTEXITCODE -ne 0) {
    throw 'Ya existe un estudio reii_probe_v6*: el re-tuneo debe usar un study NUEVO (otro prefijo).'
}

# 40 trials, mitad etapa 1 (--stage1-fraction 0.5), sin warm start.
# --force + --optimize-only: reprocesa y SALE tras el optimizador (no corre
# select con params a medias).
docker compose --profile batch run --rm workflow python -u src/reii/main_workflow.py --force --optimize-only --trials 40 --stage1-fraction 0.5 --study-name reii_probe_v6 --no-optuna-warm-start
Assert-LastExit 're-tuneo reii_probe_v6 (40 trials, stage1 0.5)'

# ======================================================================
# 9) DIAGNOSTICO DEL ESTUDIO
#    PYTHONIOENCODING=utf-8 es OBLIGATORIO: el script imprime U+25C4
#    (BLACK LEFT-POINTING TRIANGLE, en "... BORDE" / "... inerte") y con
#    stdout a un pipe en Windows
#    revienta con UnicodeEncodeError (cp1252).
#    Se restaura la variable al salir.
#    Fallo = rc != 0 o no aparece ningun estudio con reii_probe_v6.
# ======================================================================
$oldEnc = $env:PYTHONIOENCODING
$env:PYTHONIOENCODING = 'utf-8'
try {
    python 3_diagnose_optuna_db.py --study reii_probe_v6
    Assert-LastExit 'diagnostico 3_diagnose_optuna_db.py --study reii_probe_v6'
}
finally {
    $env:PYTHONIOENCODING = $oldEnc
}

# ======================================================================
# 10) SELECT + AUDIT  (docker compose --profile batch, como en v5)
#     El select persiste data\method_selection.json y
#     data\method_selection_log.jsonl; el audit los necesita.
#     Fallo:
#       * select rc != 0;
#       * audit rc == 2 -> no hay historial: el select NO persistio;
#       * audit rc != 0 -> invariantes rotas / ganador inestable (--strict).
# ======================================================================
docker compose --profile batch run --rm workflow python -u src/reii/main_workflow.py --force --mode select
Assert-LastExit 'select-v4 (--mode select)'

docker compose --profile batch run --rm workflow python -u src/9_selection_audit.py --strict
if ($LASTEXITCODE -eq 2) {
    throw 'audit rc=2: sin historial. El select no escribio method_selection_log.jsonl.'
}
if ($LASTEXITCODE -ne 0) {
    throw "audit --strict rc=${LASTEXITCODE}: invariantes rotas o eleccion inestable."
}

# ======================================================================
# 11) REVERSION DOCUMENTADA  (descomentar si hay que deshacer)
#     Orden inverso 20 -> 19 -> 18 -> 17 -> 16. `git apply -R` deshace
#     exactamente lo aplicado (incluye borrar el test nuevo de 18).
#     Alternativa: restaurar los ficheros desde $backup (paso 1).
#     NO usar `patch -R` (LF vs CRLF).
# ======================================================================
# git apply -R docs/patches/20_stability_spaces_bootstrap.patch   # si existe
# git apply -R docs/patches/19_search_space_precommit.patch       # si aplicado
# git apply -R docs/patches/18_diag_sentinel_signflip.patch
# git apply -R docs/patches/17_cdh_target_k_ceiling.patch
# git apply -R docs/patches/16_select_noncommensurable.patch
#
# Restaurar desde el backup del paso 1 (solo los destinos de los patches):
# $restore = Get-ChildItem -LiteralPath $backup -Recurse -File
# foreach ($f in $restore) {
#     $rel = $f.FullName.Substring($backup.Length + 1)
#     Copy-Item -LiteralPath $f.FullName -Destination (Join-Path $Root $rel) -Force
# }

Write-Host '[OK] Runbook v6 completado.'
