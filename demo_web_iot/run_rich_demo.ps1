param(
  [string]$BaseUrl = $env:SLEEP_PORTAL_API_BASE,
  [int]$PatientsPerClass = 3,
  [int]$MixedPatients = 3,
  [int]$EpochsPerPatient = 48,
  [int]$MaxFiles = 0,
  [switch]$Regenerate
)

if (-not $BaseUrl) {
  $BaseUrl = "http://127.0.0.1:8000"
}

$ErrorActionPreference = "Stop"
$BaseUrl = $BaseUrl.TrimEnd("/")
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $Here
$GeneratedDir = Join-Path $Here "generated"

function Show-Step {
  param([string]$Title)
  Write-Host ""
  Write-Host "== $Title ==" -ForegroundColor Cyan
}

function Resolve-Python {
  if ($env:PYTHON) {
    return $env:PYTHON
  }
  $venvPython = Join-Path $RepoRoot "venv\Scripts\python.exe"
  if (Test-Path $venvPython) {
    return $venvPython
  }
  return "python"
}

function Invoke-DemoPost {
  param(
    [string]$Path,
    [string]$JsonBody,
    [int]$TimeoutSec = 120
  )
  Invoke-RestMethod -Method Post -Uri "$BaseUrl$Path" -ContentType "application/json" -Body $JsonBody -TimeoutSec $TimeoutSec
}

if ($Regenerate -or -not (Test-Path (Join-Path $GeneratedDir "manifest.json"))) {
  Show-Step "Generate rich IoT demo files"
  $python = Resolve-Python
  & $python "$Here\generate_rich_iot_demo.py" `
    --patients-per-class $PatientsPerClass `
    --mixed-patients $MixedPatients `
    --epochs-per-patient $EpochsPerPatient `
    --output-dir "demo_web_iot/generated"
}

$manifest = Get-Content -Encoding UTF8 -Raw (Join-Path $GeneratedDir "manifest.json") | ConvertFrom-Json

Show-Step "1. Health check"
Invoke-RestMethod -Uri "$BaseUrl/api/v1/health/" -TimeoutSec 30 | Format-List

Show-Step "2. Model info"
Invoke-RestMethod -Uri "$BaseUrl/api/v1/model-info/" -TimeoutSec 90 |
  Select-Object ready,model_name,model_stage,model_type,feature_count,tracking_uri |
  Format-List

Show-Step "3. Predict rich CSV sample"
$rows = Import-Csv (Join-Path $GeneratedDir "predict_batch_rich.csv")
$featureColumns = @($rows[0].PSObject.Properties.Name)
$sampleRows = @($rows | Select-Object -First 32)
$features = @()
foreach ($row in $sampleRows) {
  $values = @()
  foreach ($col in $featureColumns) {
    $values += [double]$row.$col
  }
  $features += ,$values
}
($predictionResult = Invoke-DemoPost -Path "/api/v1/predict/" -JsonBody (@{ features = $features } | ConvertTo-Json -Depth 10)) | Format-List

Show-Step "4. Ingest rich IoT patient sessions"
$files = Get-ChildItem -Path $GeneratedDir -Filter "ingest_*.json" | Sort-Object Name
if ($MaxFiles -gt 0) {
  $files = @($files | Select-Object -First $MaxFiles)
}

$totalEpochs = 0
$posted = 0
foreach ($file in $files) {
  $payload = Get-Content -Encoding UTF8 -Raw $file.FullName
  $result = Invoke-DemoPost -Path "/api/v1/ingest/" -JsonBody $payload
  $posted += 1
  $totalEpochs += [int]$result.epochs_saved
  "{0,2}. {1,-34} saved={2,3} feature_rows={3,3} diagnosis={4}" -f `
    $posted, $result.patient_id, $result.epochs_saved, $result.feature_rows_saved, $result.diagnosis
}

Show-Step "5. Summary"
Write-Host "Generated manifest: $GeneratedDir\manifest.json"
Write-Host "Patients in manifest: $($manifest.patients)"
Write-Host "Epochs in manifest: $($manifest.total_epochs)"
Write-Host "Posted patient files: $posted"
Write-Host "Posted epochs: $totalEpochs"
Write-Host ""
Write-Host "Open these pages:"
Write-Host "$BaseUrl/"
Write-Host "$BaseUrl/patients/"
Write-Host "$BaseUrl/patients/demo-rich-mixed-01/"
Write-Host "$BaseUrl/predict/"
Write-Host "$BaseUrl/pipeline/"
