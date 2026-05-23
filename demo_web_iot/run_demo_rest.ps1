param(
  [string]$BaseUrl = $env:SLEEP_PORTAL_API_BASE
)

if (-not $BaseUrl) {
  $BaseUrl = "http://127.0.0.1:8000"
}

$ErrorActionPreference = "Stop"
$BaseUrl = $BaseUrl.TrimEnd("/")
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Show-Step {
  param([string]$Title)
  Write-Host ""
  Write-Host "== $Title ==" -ForegroundColor Cyan
}

function Invoke-DemoPost {
  param(
    [string]$Path,
    [object]$Body
  )
  $json = $Body
  if ($Body -isnot [string]) {
    $json = $Body | ConvertTo-Json -Depth 20
  }
  Invoke-RestMethod -Method Post -Uri "$BaseUrl$Path" -ContentType "application/json" -Body $json -TimeoutSec 90
}

Show-Step "1. Health check"
Invoke-RestMethod -Uri "$BaseUrl/api/v1/health/" -TimeoutSec 30 | Format-List

Show-Step "2. Model info"
Invoke-RestMethod -Uri "$BaseUrl/api/v1/model-info/" -TimeoutSec 90 |
  Select-Object ready,model_name,model_stage,model_type,feature_count,tracking_uri |
  Format-List

Show-Step "3. Predict single vector"
$single = Get-Content -Encoding UTF8 -Raw "$Here\predict_single_healthy.json" | ConvertFrom-Json
Invoke-DemoPost -Path "/api/v1/predict/" -Body $single | Format-List

Show-Step "4. Predict batch CSV"
$rows = Import-Csv "$Here\predict_batch.csv"
$featureColumns = @($rows[0].PSObject.Properties.Name)
$features = @()
foreach ($row in $rows) {
  $values = @()
  foreach ($col in $featureColumns) {
    $values += [double]$row.$col
  }
  $features += ,$values
}
Invoke-DemoPost -Path "/api/v1/predict/" -Body @{ features = $features } | Format-List

Show-Step "5. Ingest IoT demo sessions"
Get-ChildItem -Path $Here -Filter "ingest_*.json" |
  Sort-Object Name |
  ForEach-Object {
    Write-Host "Posting $($_.Name)"
    $payload = Get-Content -Encoding UTF8 -Raw $_.FullName
    Invoke-DemoPost -Path "/api/v1/ingest/" -Body $payload |
      Select-Object patient_id,diagnosis,epochs_saved,feature_rows_saved |
      Format-Table -AutoSize
  }

Show-Step "6. Open these pages for the live demo"
Write-Host "$BaseUrl/"
Write-Host "$BaseUrl/patients/"
Write-Host "$BaseUrl/patients/demo-iot-mixed-001/"
Write-Host "$BaseUrl/predict/"
Write-Host "$BaseUrl/pipeline/"
