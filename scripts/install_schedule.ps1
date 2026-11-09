# Registers (or updates) a Windows Task Scheduler task that runs scripts\run_daily.ps1 every day.
#   powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1            # 7:00 PM
#   powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -At 8:30PM
#   powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -Remove
# Runs as you, only while you're signed in (no password stored). If the laptop was off or asleep at that
# time, the run starts as soon as it's back on; it also runs on battery.
param([string]$At = "7:00PM", [string]$Name = "JSearch daily", [switch]$Remove)
$ErrorActionPreference = "Stop"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $Name -Confirm:$false
    "Removed the task '$Name'."
    exit 0
}

$script = Join-Path $PSScriptRoot "run_daily.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`"" `
    -WorkingDirectory (Split-Path -Parent $PSScriptRoot)
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 3) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings `
    -Description "JSearch: scrape jobs, update data\jobs.csv and email the top jobs" -Force | Out-Null
"Task '$Name' registered: daily at $At, next run $((Get-ScheduledTaskInfo -TaskName $Name).NextRunTime)."
