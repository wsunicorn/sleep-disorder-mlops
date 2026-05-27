param(
  [string]$BaseUrl = $env:SLEEP_PORTAL_API_BASE,
  [string]$SessionId = "live-demo",
  [int]$PatientsPerClass = 1,
  [int]$MixedPatients = 1,
  [int]$Cycles = 6,
  [int]$EpochsPerCycle = 4,
  [double]$Interval = 1.5,
  [int]$Workers = 2,
  [int]$Seed = 20260527,
  [double]$DriftStrength = 0.12,
  [int]$TimeoutSec = 45,
  [int]$Retries = 5,
  [switch]$CheckApi,
  [switch]$ResetSession,
  [switch]$DryRun
)

if (-not $BaseUrl) {
  $BaseUrl = "http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com"
}

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $Here

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

$python = Resolve-Python
$argsList = @(
  (Join-Path $Here "realtime_iot_stream.py"),
  "--base-url", $BaseUrl,
  "--session-id", $SessionId,
  "--patients-per-class", $PatientsPerClass,
  "--mixed-patients", $MixedPatients,
  "--cycles", $Cycles,
  "--epochs-per-cycle", $EpochsPerCycle,
  "--interval", $Interval,
  "--workers", $Workers,
  "--seed", $Seed,
  "--drift-strength", $DriftStrength,
  "--timeout", $TimeoutSec,
  "--retries", $Retries
)

if ($ResetSession) {
  $argsList += "--reset-session"
}
if ($DryRun) {
  $argsList += "--dry-run"
}
if ($CheckApi) {
  $argsList += "--check-api"
}

Write-Host ""
Write-Host "Starting realtime IoT demo" -ForegroundColor Cyan
Write-Host "Python   : $python"
Write-Host "Base URL : $BaseUrl"
Write-Host "Session  : $SessionId"
Write-Host ""

& $python @argsList
