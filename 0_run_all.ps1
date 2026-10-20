<#
.SYNOPSIS
    Script maestro: Ejecuta los 5 scripts en orden en Windows.
.EXAMPLE
    .\0_run_all.ps1 -ReiiDir "C:\ruta\a\reii" -Corpus "C:\ruta\a\corpus.txt"
#>

param(
    $ReiiDir,
    $Corpus
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhitespace($ReiiDir) -or [string]::IsNullOrWhitespace($Corpus)) {
    Write-Host "Uso: .\0_run_all.ps1 -ReiiDir 'C:\ruta\a\reii' -Corpus 'C:\ruta\a\corpus.txt'" -ForegroundColor Red
    exit 1
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $ScriptDir

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "PIPELINE DE DIAGNOSTICO DE REII" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "REII: $ReiiDir"
Write-Host "Corpus: $Corpus`n"

# Script 1: Verificar cache
Write-Host "PASO 1: Verificar fix de caché de coref" -ForegroundColor Yellow
Write-Host "  -> Esto corre DOS veces. Espera 5-10 min"
try {
    python 1_check_coref_cache_hit.py --reii-dir "$ReiiDir"
} catch {
    Write-Host "x Script 1 fallo. Comprueba:" -ForegroundColor Red
    Write-Host "  - Ua REII es accesible en $ReiiDir?" -ForegroundColor Red
    Write-Host "  - Ua Los patches están aplicados?" -ForegroundColor Red
    exit 1
}
Read-Host "Presiona ENTER para continuar con paso 2..."

# Script 2: 300 trials
Write-Host "`nPASO 2: Correr optimizador con 300 trials (3 semillas)" -ForegroundColor Yellow
Write-Host "  -> Esto toma 30-60 minutos"
try {
    python 2_run_300_trials.py `
      --reii-dir "$ReiiDir" `
      --corpus "$Corpus" `
      --seeds 3 `
      --max-trials 300 `
      --out results_300trials.tsv
} catch {
    Write-Host "x Script 2 fallo" -ForegroundColor Red
    exit 1
}
Read-Host "Presiona ENTER para continuar con paso 3..."

# Script 3: Diagnosticar DB
Write-Host "`nPASO 3: Diagnosticar base de Optuna" -ForegroundColor Yellow
Write-Host "  -> Esto es rápido (~1 min)"
$DbPath = Join-Path -Path $ReiiDir -ChildPath "reii_optuna.db"
if (Test-Path -Path $DbPath) {
    python 3_diagnose_optuna_db.py --db "$DbPath" --out optuna_diagnosis.txt
} else {
    Write-Host "U Base de Optuna no encontrada en $DbPath" -ForegroundColor Yellow
    Write-Host "  Ua esta en otro sitio? Corre manualmente:" -ForegroundColor Yellow
    Write-Host "  python 3_diagnose_optuna_db.py --db C:\ruta\a\reii_optuna.db" -ForegroundColor Yellow
}
Read-Host "Presiona ENTER para continuar con paso 4..."

# Script 4: Medir uce_target_size
Write-Host "`nPASO 4: Medir coste de uce_target_size" -ForegroundColor Yellow
Write-Host "  -> Esto toma 15-30 minutos (7 valores x 2 repeticiones)"
try {
    python 4_measure_uce_target_size.py `
      --reii-dir "$ReiiDir" `
      --corpus "$Corpus" `
      --uce-values "18,22,26,30,34,38" `
      --repeats 2 `
      --out timing_uce.tsv
} catch {
    Write-Host "x Script 4 fallo" -ForegroundColor Red
    exit 1
}
Read-Host "Presiona ENTER para ir al paso 5 (opcional)..."

# Script 5: Analisis de S2 (opcional)
Write-Host "`nPASO 5: Analisis de potencia para S2 (OPCIONAL)" -ForegroundColor Yellow
Write-Host "  -> Esto toma 1+ hora. Solo si quieres decidir sobre S2.`n"

$response = Read-Host "Quieres ejecutar el analysis de S2? (s/n)"
if ($response -match "^[Ss]$") {
    try {
        python 5_s2_power_analysis.py `
          --bench-script "C:\mnt\user-data\outputs\reii_optimizer_report.py" `
          --seeds 32 `
          --out s2_analysis_final.json
    } catch {
        Write-Host "x Script 5 fallo" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "FIN DEL PIPELINE" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "`nFicheros generados:"
Get-ChildItem -Path "results_300trials.tsv", "optuna_diagnosis.txt", "timing_uce.tsv" -ErrorAction SilentlyContinue | Select-Object Name, Length, LastWriteTime

Write-Host "`nProximos pasos:"
Write-Host "  1. Lee los resultados arriba"
Write-Host "  2. Si cache = 0 en paso 2, el fix funciona (checkmark)"
Write-Host "  3. Si 300 trials converge, el presupuesto era el problema"
Write-Host "  4. Usa los datos del paso 4 para modelar uce_target_size"
Write-Host "  5. Paso 5 es solo si quieres decidir S2"
