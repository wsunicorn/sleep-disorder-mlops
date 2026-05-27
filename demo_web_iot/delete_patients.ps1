param(
  [string]$BaseUrl = "http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com",
  [string[]]$PatientId = @(),
  [string[]]$IdPrefix = @(),
  [string[]]$Diagnosis = @(),
  [switch]$UnknownDiagnosis,
  [switch]$MixedDemo,
  [switch]$DemoRich,
  [switch]$RealtimeDemo,
  [switch]$QuickDemo,
  [switch]$AllDemo,
  [switch]$List,
  [switch]$Yes,
  [int]$TimeoutSec = 30
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$ArgsList = @(
  (Join-Path $ScriptDir "delete_patients.py"),
  "--base-url", $BaseUrl,
  "--timeout", $TimeoutSec
)

foreach ($Item in $PatientId) {
  $ArgsList += @("--patient-id", $Item)
}
foreach ($Item in $IdPrefix) {
  $ArgsList += @("--id-prefix", $Item)
}
foreach ($Item in $Diagnosis) {
  $ArgsList += @("--diagnosis", $Item)
}
if ($UnknownDiagnosis) { $ArgsList += "--unknown-diagnosis" }
if ($MixedDemo) { $ArgsList += "--mixed-demo" }
if ($DemoRich) { $ArgsList += "--demo-rich" }
if ($RealtimeDemo) { $ArgsList += "--realtime-demo" }
if ($QuickDemo) { $ArgsList += "--quick-demo" }
if ($AllDemo) { $ArgsList += "--all-demo" }
if ($List) { $ArgsList += "--list" }
if ($Yes) { $ArgsList += "--yes" }

& $PythonExe @ArgsList
