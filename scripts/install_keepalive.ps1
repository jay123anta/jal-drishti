# Register (or remove) the JalDrishti keep-alive Scheduled Task on this machine.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_keepalive.ps1            # every 3 h
#   powershell -ExecutionPolicy Bypass -File scripts\install_keepalive.ps1 -EveryHours 6
#   powershell -ExecutionPolicy Bypass -File scripts\install_keepalive.ps1 -Uninstall
#   powershell -ExecutionPolicy Bypass -File scripts\install_keepalive.ps1 -Status
#
# The task: runs scripts\keepalive.ps1 as the current user, every N hours,
# starts as soon as possible if a run was missed (laptop asleep / off),
# wakes the machine to run, runs on battery, never overlaps itself,
# 30-minute time limit. No admin rights needed.
param(
    [int]$EveryHours = 3,
    [switch]$Uninstall,
    [switch]$Status
)

$task = "JalDrishtiPipeline"
$repo = Split-Path -Parent $PSScriptRoot

if ($Status) {
    $t = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
    if (-not $t) { Write-Host "Task '$task' is NOT installed."; exit 1 }
    $i = Get-ScheduledTaskInfo -TaskName $task
    Write-Host "Task '$task': state=$($t.State) last=$($i.LastRunTime) result=$($i.LastTaskResult) next=$($i.NextRunTime)"
    Get-ChildItem "$repo\logs\pipeline_*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1 |
        ForEach-Object { Write-Host "Latest log: $($_.FullName)"; Get-Content $_.FullName -Tail 3 }
    exit 0
}

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Task '$task' removed."
    exit 0
}

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { Write-Error "python not found on PATH"; exit 1 }

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$repo\scripts\keepalive.ps1`"" `
    -WorkingDirectory $repo
# No -RepetitionDuration: on Windows 10/11 that means "repeat indefinitely"
# ([TimeSpan]::MaxValue serialises to an XML duration the scheduler rejects).
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Hours $EveryHours)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

try {
    Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger `
        -Settings $settings -Force -ErrorAction Stop | Out-Null
} catch {
    Write-Error "Task registration FAILED: $($_.Exception.Message)"
    exit 1
}
$t = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
if (-not $t) { Write-Error "Task '$task' not found after registration"; exit 1 }
$i = Get-ScheduledTaskInfo -TaskName $task
Write-Host "Task '$task' installed: every $EveryHours h, state=$($t.State), next run $($i.NextRunTime), python=$py"
Write-Host "Logs: $repo\logs\pipeline_<date>.log   Check: -Status   Remove: -Uninstall"
