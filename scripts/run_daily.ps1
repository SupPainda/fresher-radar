# What Task Scheduler runs every day (see install_schedule.ps1): one scheduled run of fetch_jobs.py.
# The run writes its own log to <data dir>\logs\; anything printed before logging starts (e.g. a config
# error) is appended to <repo>\data\logs\scheduler.log.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$logDir = Join-Path $repo "data\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "scheduler.log"
"=== $(Get-Date -Format 'yyyy-MM-dd hh:mm tt') scheduled run" | Out-File -Append -Encoding utf8 $log
& (Join-Path $repo ".venv\Scripts\python.exe") fetch_jobs.py --scheduled 2>&1 | Out-File -Append -Encoding utf8 $log
"=== exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
exit $LASTEXITCODE
